"""
AKA-Sim 后端 - ACT 模型模块
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import config

if TYPE_CHECKING:
    from policies.models.act.modeling_act import ACTModel, ACTConfig

logger = logging.getLogger(__name__)

# 模块级变量
_act_model: Optional["ACTModel"] = None
_model_device = "cpu"
_state_mean: Optional[torch.Tensor] = None
_state_std: Optional[torch.Tensor] = None
_action_mean: Optional[torch.Tensor] = None
_action_std: Optional[torch.Tensor] = None


def create_act_config() -> "ACTConfig":
    """创建 ACT 模型配置 - 与训练时一致"""
    from policies.models.act.modeling_act import ACTConfig as PyACTConfig

    return PyACTConfig(
        state_dim=4,  # 只用 x, y, angle, speed
        action_dim=5,
        action_chunk_size=16,
        hidden_dim=512,
        num_attention_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=3200,  # LeRobot 使用 3200
        use_cvae=False,
        use_temporal_ensembling=False,
        use_spatial_softmax=True,
        latent_dim=32,
    )


def load_act_model(model_path: str = None) -> "ACTModel":
    """加载 ACT 模型"""
    global _act_model, _state_mean, _state_std, _action_mean, _action_std

    from policies.models.act.modeling_act import ACTModel as PyACTModel

    logger.info("加载 ACT 模型...")

    # 尝试从output目录加载训练好的模型
    if model_path is None:
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "output" / "train" / "model.pt"

    model_path = Path(model_path)

    if not model_path.exists():
        logger.warning(f"模型文件不存在: {model_path}")
        _act_model = None
        return None

    logger.info(f"从 {model_path} 加载模型权重")

    # 加载 state_dict
    state_dict = torch.load(model_path, map_location=_model_device, weights_only=True)

    # 创建配置（必须与训练时一致！）
    model_config = create_act_config()
    _act_model = PyACTModel(model_config)
    _act_model.load_state_dict(state_dict)

    _act_model = _act_model.to(_model_device)
    _act_model.eval()

    # 加载归一化统计信息
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "output" / "dataset"
    stats_path = data_dir / "meta" / "stats.json"

    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)

        # 状态归一化统计
        if "observation.state" in stats:
            _state_mean = torch.tensor(stats["observation.state"]["mean"], dtype=torch.float32)
            _state_std = torch.tensor(stats["observation.state"]["std"], dtype=torch.float32)
            logger.info(f"状态归一化: mean={_state_mean}, std={_state_std}")
        else:
            _state_mean = torch.zeros(7)
            _state_std = torch.ones(7)

        # 动作归一化统计
        if "action" in stats:
            _action_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32)
            _action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32)
            logger.info(f"动作归一化: mean={_action_mean}, std={_action_std}")
        else:
            _action_mean = torch.zeros(5)
            _action_std = torch.ones(5)
    else:
        logger.warning(f"未找到归一化统计文件: {stats_path}，使用默认归一化")
        _state_mean = torch.zeros(7)
        _state_std = torch.ones(7)
        _action_mean = torch.zeros(5)
        _action_std = torch.ones(5)

    logger.info(f"ACT 模型加载完成，使用设备: {_model_device}")
    return _act_model


def _process_image(image_input: Union[str, Image.Image, None]) -> torch.Tensor:
    """
    处理图像输入并转换为 tensor

    Args:
        image_input: base64 编码的图像字符串或 PIL Image 或 None

    Returns:
        图像 tensor [1, 1, 3, 224, 224]
    """
    if image_input is None:
        logger.warning("未提供图像，使用随机噪声")
        return torch.randn(1, 1, 3, 224, 224).to(_model_device)

    try:
        if isinstance(image_input, str):
            # 处理 data URL 格式 (data:image/jpeg;base64,...)
            if "," in image_input:
                image_input = image_input.split(",")[1]
            # base64 解码
            image_data = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            logger.info(f"图像处理成功，尺寸: {image.size}")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
            logger.info(f"PIL 图像处理成功，尺寸: {image.size}")
        else:
            logger.warning(f"不支持的图像类型: {type(image_input)}，使用随机噪声")
            return torch.randn(1, 1, 3, 224, 224).to(_model_device)

        # 图像预处理 - 使用 ImageNet 归一化（与训练时一致）
        transform_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_tensor = transform_pipeline(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 224, 224]
        logger.info(f"图像 tensor shape: {image_tensor.shape}, mean: {image_tensor.mean().item():.4f}")
        return image_tensor.to(_model_device)

    except Exception as e:
        logger.warning(f"图像处理失败: {e}，使用随机噪声")
        return torch.randn(1, 1, 3, 224, 224).to(_model_device)


def act_inference(state: list, image: Optional[Union[str, Image.Image]] = None) -> list:
    """
    ACT 模型推理

    Args:
        state: 状态向量 [state_dim]
        image: base64 编码的图像字符串或 PIL Image

    Returns:
        预测的动作序列 [action_chunk_size, action_dim]
    """
    if _act_model is None:
        logger.warning("ACT 模型未加载，返回随机动作")
        action_dim = 5
        action_chunk_size = 16
        return [[0.0] * action_dim for _ in range(action_chunk_size)]

    with torch.no_grad():
        # 1. 准备状态输入并进行归一化
        # 只用前 4 维：x, y, angle, speed
        state = state[:4]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(_model_device)
        logger.info(f"原始 state: {state}")

        if _state_mean is not None and _state_std is not None:
            # 只用前 4 维的 mean/std
            state_mean = _state_mean[:4].to(_model_device)
            state_std = _state_std[:4].to(_model_device)
            # 确保 std 不为 0
            state_std = torch.where(state_std > 1e-6, state_std, torch.ones_like(state_std))
            state_tensor = (state_tensor - state_mean) / state_std
            logger.info(f"归一化后 state: {state_tensor}")

        # 2. 处理图像输入
        image_tensor = _process_image(image)
        logger.info(f"图像 tensor shape: {image_tensor.shape}")

        # 3. 推理
        action = _act_model.get_action(
            image_tensor,
            state_tensor,
            use_temporal_ensembling=False,
        )

        logger.info(f"原始推理结果: {action[0][0].tolist()}")

        # 4. 动作反归一化
        if _action_mean is not None and _action_std is not None:
            action = action * _action_std.to(_model_device) + _action_mean.to(_model_device)
            logger.info(f"反归一化后: {action[0][0].tolist()}")

        # 5. 转换为 Python 列表
        action_list = action.cpu().numpy()[0].tolist()

    return action_list


def is_model_loaded() -> bool:
    """检查模型是否已加载"""
    return _act_model is not None


def get_model_device() -> str:
    """获取模型设备"""
    return _model_device
