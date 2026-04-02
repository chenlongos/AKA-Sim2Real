"""Dataset loading for ACT training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_dataset(data_dir: str = "output/dataset") -> Dict[str, torch.Tensor]:
    """
    加载数据集

    Args:
        data_dir: 数据集目录
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"训练集目录不存在: {data_path}")
        return {}

    with open(data_path / "meta" / "stats.json", "r") as f:
        stats = json.load(f)

    with open(data_path / "meta" / "info.json", "r") as f:
        info = json.load(f)

    action_dim = info.get("action_dim", 2)

    parquet_files = sorted(data_path.glob("data/chunk-*/file-*.parquet"))
    logger.info(f"加载 {len(parquet_files)} 个数据文件")

    images = []
    states = []
    actions = []

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)

        for img_path in df["observation.image"]:
            full_path = data_path / img_path
            if full_path.exists():
                try:
                    img = Image.open(full_path).convert("RGB")
                    img = img.resize((224, 224))
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    images.append(img_tensor)
                except Exception as exc:
                    logger.warning(f"读取图像失败，使用零图像替代: {full_path} ({exc})")
                    images.append(torch.zeros(3, 224, 224))
            else:
                images.append(torch.zeros(3, 224, 224))

        for state_list in df["observation.state"]:
            state_arr = np.array(state_list, dtype=np.float32)
            states.append(torch.from_numpy(state_arr).float())

        for action_data in df["action"]:
            if isinstance(action_data, np.ndarray):
                action_arr = np.vstack(action_data).astype(np.float32)
            elif isinstance(action_data, list):
                if len(action_data) > 0 and isinstance(action_data[0], (list, np.ndarray)):
                    action_arr = np.array(action_data, dtype=np.float32)
                else:
                    action_arr = np.array([action_data], dtype=np.float32)
            else:
                action_arr = np.array([[action_data]], dtype=np.float32)
            actions.append(torch.from_numpy(action_arr).float())

    images = torch.stack(images)
    states = torch.stack(states)
    actions = torch.stack(actions)

    logger.info(f"加载完成: {len(images)} 个样本, action_dim={action_dim}")

    return {
        "observation.image": images,
        "observation.state": states,
        "action": actions,
        "stats": stats,
    }
