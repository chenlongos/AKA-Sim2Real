"""AKA-Sim backend ACT inference runtime."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import torch
from PIL import Image

from backend.config import config
from backend.services.inference.checkpoint import (
    ACTNormalizationStats,
    get_default_device,
    instantiate_model,
    load_checkpoint_bundle,
    load_stats,
)
from backend.services.inference.execution import TemporalEnsemblingPolicy
from backend.services.inference.preprocess import ACTPreprocessor
from policies.models.act.defaults import build_act_config

if TYPE_CHECKING:
    from policies.models.act.modeling_act import ACTConfig, ACTModel

logger = logging.getLogger(__name__)


class ACTInferenceRuntime:
    def __init__(self):
        self.model: Optional["ACTModel"] = None
        self.device = get_default_device()
        self.stats = ACTNormalizationStats()
        self.preprocessor = ACTPreprocessor()
        self.execution_policy = TemporalEnsemblingPolicy()

    def create_config(self, config_dict: Optional[dict] = None) -> "ACTConfig":
        return build_act_config(**(config_dict or {}))

    def reset_inference_context(self):
        self.execution_policy.reset()
        logger.info("已重置 ACT 推理时序上下文")

    def _temporal_decay(self) -> float:
        if self.model is None:
            return 0.5
        decay = float(getattr(self.model.config, "temporal_ensembling_weight", 0.5))
        return min(max(decay, 1e-3), 1.0)

    def blend_current_action(self, action_chunk: torch.Tensor) -> torch.Tensor:
        self.execution_policy.update_decay(self._temporal_decay())
        return self.execution_policy.blend(action_chunk)

    def _load_stats(self, stats_dir: Optional[str]):
        self.stats = load_stats(stats_dir)
        logger.info("状态归一化: mean=%s, std=%s", self.stats.state_mean, self.stats.state_std)
        logger.info("动作归一化: mean=%s, std=%s", self.stats.action_mean, self.stats.action_std)

    def load_model(self, model_path: str = None, stats_dir: str = None) -> "ACTModel":
        logger.info("加载 ACT 模型...")
        self.device = get_default_device()
        bundle = load_checkpoint_bundle(model_path, self.device)
        self.model = instantiate_model(bundle, self.device)
        self.reset_inference_context()
        self._load_stats(stats_dir)
        logger.info("ACT 模型加载完成，使用设备: %s", self.device)
        return self.model

    def process_image(self, image_input: Union[str, Image.Image, None]) -> torch.Tensor:
        return self.preprocessor.process_image(image_input, self.device)

    def infer(self, state: list, image: Optional[Union[str, Image.Image]] = None) -> list:
        if self.model is None:
            logger.warning("ACT 模型未加载，返回随机动作")
            return [[0.0, 0.0]]

        with torch.no_grad():
            state_tensor = self.preprocessor.normalize_state(state, self.stats, self.device)
            image_tensor = self.process_image(image)
            action = self.model.get_action(
                image_tensor,
                state_tensor,
                use_temporal_ensembling=False,
            )
            action = self.blend_current_action(action)
            action = self.preprocessor.denormalize_action(action, self.stats, self.device)
            return action.cpu().numpy().tolist()

    def is_model_loaded(self) -> bool:
        return self.model is not None


_runtime = ACTInferenceRuntime()


def get_act_runtime() -> ACTInferenceRuntime:
    return _runtime


def create_act_config(config_dict: Optional[dict] = None) -> "ACTConfig":
    return _runtime.create_config(config_dict)


def reset_inference_context():
    _runtime.reset_inference_context()


def load_act_model(model_path: str = None, stats_dir: str = None) -> "ACTModel":
    return _runtime.load_model(model_path=model_path, stats_dir=stats_dir)


def act_inference(state: list, image: Optional[Union[str, Image.Image]] = None) -> list:
    return _runtime.infer(state, image)


def is_model_loaded() -> bool:
    return _runtime.is_model_loaded()


def get_model_device() -> str:
    return _runtime.device
