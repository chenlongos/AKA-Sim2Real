"""Shared ACT default configuration helpers."""

from __future__ import annotations

from typing import Any

from .configuration_act import ACTConfig


DEFAULT_ACT_CONFIG: dict[str, Any] = {
    "state_dim": 2,  # [vel_left, vel_right] - 后续可扩展 gripper 状态
    "action_dim": 3,  # [left_vel, right_vel, gripper_target]
    "action_chunk_size": 8,  # 预测8步未来动作
    "n_action_steps": 1,     # Temporal Ensembling 模式下必须为 1
    "hidden_dim": 512,
    "num_attention_heads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "dim_feedforward": 3200,
    "use_cvae": True,
    "kl_weight": 0.1,
    "use_temporal_ensembling": True,
    "temporal_ensembling_coeff": 0.01,  # LeRobot ACT 原版值，建议不要改
    "use_spatial_softmax": True,
    "latent_dim": 32,
    "num_cameras": 1,
}


def build_act_config(**overrides: Any) -> ACTConfig:
    config_dict = DEFAULT_ACT_CONFIG.copy()
    config_dict.update(overrides)
    return ACTConfig(**config_dict)


def act_config_to_dict(config: ACTConfig) -> dict[str, Any]:
    return {
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "action_chunk_size": config.action_chunk_size,
        "n_action_steps": config.n_action_steps,
        "hidden_dim": config.hidden_dim,
        "num_attention_heads": config.num_attention_heads,
        "num_encoder_layers": config.num_encoder_layers,
        "num_decoder_layers": config.num_decoder_layers,
        "dim_feedforward": config.dim_feedforward,
        "latent_dim": config.latent_dim,
        "use_cvae": config.use_cvae,
        "kl_weight": config.kl_weight,
        "use_temporal_ensembling": config.use_temporal_ensembling,
        "temporal_ensembling_coeff": config.temporal_ensembling_coeff,
        "use_spatial_softmax": config.use_spatial_softmax,
        "spatial_softmax_temperature": config.spatial_softmax_temperature,
        "num_cameras": config.num_cameras,
        "image_size": config.image_size,
        "in_channels": config.in_channels,
        "dropout": config.dropout,
    }
