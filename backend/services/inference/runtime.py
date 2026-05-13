"""AKA-Sim backend ACT inference runtime."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import torch
from PIL import Image

from backend.services.inference.checkpoint import (
    ACTNormalizationStats,
    get_default_device,
    instantiate_model,
    load_checkpoint_bundle,
    load_stats,
)
from backend.services.inference.preprocess import ACTPreprocessor
from policies.models.act.defaults import build_act_config
from policies.models.act.modeling_act import ACTTemporalEnsembler

if TYPE_CHECKING:
    from policies.models.act.modeling_act import ACTConfig, ACTModel

logger = logging.getLogger(__name__)


class ACTInferenceRuntime:
    def __init__(self):
        self.model: Optional["ACTModel"] = None
        self.device = get_default_device()
        self.stats = ACTNormalizationStats()
        self.preprocessor = ACTPreprocessor()
        self.temporal_ensembler: Optional[ACTTemporalEnsembler] = None
        self.action_chunk_size = 50  # 默认值，会在加载模型时更新

    def create_config(self, config_dict: Optional[dict] = None) -> "ACTConfig":
        return build_act_config(**(config_dict or {}))

    def reset_inference_context(self):
        """重置推理上下文（每个 episode 开始时调用）"""
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    def should_use_temporal_ensembling(self) -> bool:
        """是否启用 temporal ensembling"""
        if self.model is None:
            return False
        return bool(getattr(self.model.config, "use_temporal_ensembling", False))

    def _load_stats(self, stats_dir: Optional[str]):
        self.stats = load_stats(stats_dir)
        logger.info("状态归一化 (QUANTILES): q01=%s, q99=%s", self.stats.state_q01, self.stats.state_q99)
        logger.info("动作归一化 (QUANTILES): q01=%s, q99=%s", self.stats.action_q01, self.stats.action_q99)

    def load_model(self, model_path: str = None, stats_dir: str = None) -> "ACTModel":
        logger.info("加载 ACT 模型...")
        self.device = get_default_device()
        bundle = load_checkpoint_bundle(model_path, self.device)
        self.model = instantiate_model(bundle, self.device)
        self.reset_inference_context()
        self._load_stats(stats_dir)

        # 初始化 temporal ensembler（如果启用）
        if self.should_use_temporal_ensembling():
            self.action_chunk_size = getattr(self.model.config, "action_chunk_size", 8)
            coeff = float(getattr(self.model.config, "temporal_ensembling_coeff", 0.01))
            self.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=coeff,
                chunk_size=self.action_chunk_size,
            )
            logger.info(f"已初始化 Temporal Ensembler: coeff={coeff}, chunk_size={self.action_chunk_size}")
        else:
            self.temporal_ensembler = None

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
            logger.info(f"[ACT推理] 输入state: {state}, 归一化后: {state_tensor.tolist()}")
            image_tensor = self.process_image(image)

            use_temporal = self.should_use_temporal_ensembling()
            action = self.model.get_action(
                image_tensor,
                state_tensor,
                use_temporal_ensembling=use_temporal,
                temporal_ensembler=self.temporal_ensembler,
            )

            logger.info(f"[ACT推理] 模型原始输出: {action[0].tolist()}")
            action = self.preprocessor.denormalize_action(action, self.stats, self.device)
            logger.info(f"[ACT推理] 归一化后输出: {action[0].tolist()}")
            # 返回单步 [left, right] 而非 [[left, right]]
            return action[0].cpu().tolist()

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
