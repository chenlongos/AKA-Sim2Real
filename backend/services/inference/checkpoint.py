"""Checkpoint and stats helpers for ACT runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Optional, TYPE_CHECKING

import torch

from policies.models.act.defaults import build_act_config

if TYPE_CHECKING:
    from policies.models.act.modeling_act import ACTConfig, ACTModel


@dataclass
class ACTNormalizationStats:
    state_min: torch.Tensor = field(default_factory=lambda: torch.zeros(2))
    state_max: torch.Tensor = field(default_factory=lambda: torch.ones(2))
    action_min: torch.Tensor = field(default_factory=lambda: torch.zeros(2))
    action_max: torch.Tensor = field(default_factory=lambda: torch.ones(2))


@dataclass
class ACTCheckpointBundle:
    model_config: "ACTConfig"
    state_dict: dict
    inference_latent_mu: Optional[torch.Tensor] = None
    inference_latent_log_sigma: Optional[torch.Tensor] = None


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_default_model_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "output" / "train" / "final_model.pt"
    if not model_path.exists():
        model_path = project_root / "output" / "train" / "model.pt"
    return model_path


def load_checkpoint_bundle(model_path: Optional[str], device: str) -> ACTCheckpointBundle:
    project_root = Path(__file__).resolve().parents[3]
    if model_path is not None:
        path = Path(model_path) if Path(model_path).is_absolute() else project_root / model_path
    else:
        path = resolve_default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_config = checkpoint.get("config", {})
        inference_latent_mu = checkpoint.get("inference_latent_mu")
        inference_latent_log_sigma = checkpoint.get("inference_latent_log_sigma")
    else:
        state_dict = checkpoint
        checkpoint_config = {}
        inference_latent_mu = None
        inference_latent_log_sigma = None

    return ACTCheckpointBundle(
        model_config=build_act_config(**checkpoint_config),
        state_dict=state_dict,
        inference_latent_mu=inference_latent_mu,
        inference_latent_log_sigma=inference_latent_log_sigma,
    )


def load_stats(stats_dir: Optional[str]) -> ACTNormalizationStats:
    project_root = Path(__file__).resolve().parents[3]
    if stats_dir is not None:
        # 如果是绝对路径直接用，否则相对于项目根目录
        data_dir = Path(stats_dir) if Path(stats_dir).is_absolute() else project_root / stats_dir
    else:
        data_dir = project_root / "output" / "dataset"
    stats_path = data_dir / "meta" / "stats.json"
    logger.info(f"[load_stats] stats_dir={stats_dir}, resolved data_dir={data_dir}, stats_path={stats_path}, exists={stats_path.exists()}")
    if not stats_path.exists():
        return ACTNormalizationStats()

    with open(stats_path, "r") as f:
        stats = json.load(f)

    return ACTNormalizationStats(
        state_min=torch.tensor(stats.get("observation.state", {}).get("min", [0.0, 0.0]), dtype=torch.float32),
        state_max=torch.tensor(stats.get("observation.state", {}).get("max", [1.0, 1.0]), dtype=torch.float32),
        action_min=torch.tensor(stats.get("action", {}).get("min", [0.0, 0.0]), dtype=torch.float32),
        action_max=torch.tensor(stats.get("action", {}).get("max", [1.0, 1.0]), dtype=torch.float32),
    )


def instantiate_model(bundle: ACTCheckpointBundle, device: str) -> "ACTModel":
    from policies.models.act.modeling_act import ACTModel as PyACTModel

    model = PyACTModel(bundle.model_config)
    model.load_state_dict(bundle.state_dict)
    if bundle.inference_latent_mu is not None and bundle.inference_latent_log_sigma is not None:
        model.set_inference_latent(bundle.inference_latent_mu, bundle.inference_latent_log_sigma)
    model = model.to(device)
    model.eval()
    return model
