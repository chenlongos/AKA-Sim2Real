"""
AKA-Sim 训练模块
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import pandas as pd

from policies.models.act.modeling_act import ACTModel, ACTConfig

logger = logging.getLogger(__name__)

# 全局训练状态
training_state = {
    "is_running": False,
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "progress": 0.0,
}


class TrainingCallbacks:
    """训练回调 - 用于推送进度"""

    def __init__(self, sio_server=None):
        self.sio_server = sio_server

    def on_epoch_start(self, epoch: int, total_epochs: int):
        """每个epoch开始时调用"""
        training_state["epoch"] = epoch + 1
        training_state["total_epochs"] = total_epochs
        self._emit_progress()

    def on_batch_end(self, loss: float, batch_idx: int, total_batches: int):
        """每个batch结束时调用"""
        training_state["loss"] = loss
        training_state["progress"] = (training_state["epoch"] - 1 + batch_idx / total_batches) / training_state["total_epochs"]
        self._emit_progress()

    def on_epoch_end(self, epoch: int, avg_loss: float):
        """每个epoch结束时调用"""
        training_state["loss"] = avg_loss
        self._emit_progress()

    def on_train_end(self, model_path: str):
        """训练结束时调用"""
        training_state["is_running"] = False
        training_state["progress"] = 1.0
        self._emit_progress()

    def _emit_progress(self):
        """发送进度到前端"""
        if self.sio_server:
            try:
                asyncio.create_task(self.sio_server.emit("training_progress", training_state))
            except Exception:
                pass


def load_dataset(data_dir: str = "output/dataset") -> Dict[str, torch.Tensor]:
    """
    加载数据集

    Args:
        data_dir: 数据集目录
    """
    # 使用项目根目录
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / data_dir

    # 如果 dataset 目录存在，也尝试从那里加载（兼容旧格式）
    legacy_path = project_root / "dataset"
    if not data_path.exists() and legacy_path.exists():
        data_path = legacy_path

    # 加载统计信息
    with open(data_path / "meta" / "stats.json", "r") as f:
        stats = json.load(f)

    # 加载info获取维度信息
    with open(data_path / "meta" / "info.json", "r") as f:
        info = json.load(f)

    action_dim = info.get("action_dim", 2)  # [steering, throttle]

    # 加载所有parquet文件
    parquet_files = sorted(data_path.glob("data/chunk-*/file-*.parquet"))
    logger.info(f"加载 {len(parquet_files)} 个数据文件")

    images = []
    states = []
    actions = []

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)

        # 加载图像
        for img_path in df["observation.image"]:
            full_path = data_path / img_path
            if full_path.exists():
                img = Image.open(full_path).convert("RGB")
                img = img.resize((224, 224))
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                images.append(img_tensor)
            else:
                images.append(torch.zeros(3, 224, 224))

        # 加载状态
        for state_list in df["observation.state"]:
            state_arr = np.array(state_list, dtype=np.float32)
            state = torch.from_numpy(state_arr).float()
            states.append(state)

        # 加载动作（[chunk_size, 3]）
        for action_data in df["action"]:
            # action_data 可能是 numpy.ndarray 或 list
            if isinstance(action_data, np.ndarray):
                # 直接堆叠
                action_arr = np.vstack(action_data).astype(np.float32)
            elif isinstance(action_data, list):
                if len(action_data) > 0 and isinstance(action_data[0], (list, np.ndarray)):
                    action_arr = np.array(action_data, dtype=np.float32)
                else:
                    action_arr = np.array([action_data], dtype=np.float32)
            else:
                action_arr = np.array([[action_data]], dtype=np.float32)
            action = torch.from_numpy(action_arr).float()
            actions.append(action)

    images = torch.stack(images)
    states = torch.stack(states)
    actions = torch.stack(actions)

    logger.info(f"加载完成: {len(images)} 个样本")

    # 返回数据和统计信息
    return {
        "observation.image": images,
        "observation.state": states,
        "action": actions,
        "stats": stats,  # 返回统计信息用于归一化
    }


class SimpleDataset(torch.utils.data.Dataset):
    """简化数据集 - 支持归一化"""

    def __init__(self, data: Dict[str, torch.Tensor], stats: Optional[Dict] = None):
        self.data = data
        self.stats = stats
        # 每条记录已经是一个完整样本
        self.num_samples = data["action"].shape[0]
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # 状态和动作归一化参数
        if stats:
            self.state_mean = torch.tensor(stats.get("observation.state", {}).get("mean", [0, 0, 0]), dtype=torch.float32)
            self.state_std = torch.tensor(stats.get("observation.state", {}).get("std", [1, 1, 1]), dtype=torch.float32)
            self.action_mean = torch.tensor(stats.get("action", {}).get("mean", [0, 0, 0]), dtype=torch.float32)
            self.action_std = torch.tensor(stats.get("action", {}).get("std", [1, 1, 1]), dtype=torch.float32)
            # 避免除零
            self.state_std = torch.where(self.state_std > 1e-6, self.state_std, torch.ones_like(self.state_std))
            self.action_std = torch.where(self.action_std > 1e-6, self.action_std, torch.ones_like(self.action_std))
        else:
            self.state_mean = torch.zeros(3)
            self.state_std = torch.ones(3)
            self.action_mean = torch.zeros(3)
            self.action_std = torch.ones(3)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 每条记录已经是一个完整的 (image, state, action_chunk) 样本
        images = self.data["observation.image"][idx]
        state = self.data["observation.state"][idx]
        action = self.data["action"][idx]

        # 归一化图像
        images = (images.unsqueeze(0) - self.image_mean) / self.image_std

        # 归一化状态 (只取前3维)
        if self.stats:
            state = (state[:3] - self.state_mean) / self.state_std

        # 归一化动作
        if self.stats:
            action = (action - self.action_mean) / self.action_std

        return {
            "observation": {
                "image": images,
                "state": state,
            },
            "action": action,
        }


async def train_model(
    sio_server,
    data_dir: str = "output/dataset",
    output_dir: str = None,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    resume_from: str = None,  # 从已有模型继续训练
) -> Optional[ACTModel]:
    """
    训练ACT模型

    Args:
        sio_server: Socket.IO服务器实例
        data_dir: 数据集目录
        output_dir: 模型输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        resume_from: 从已有模型文件继续训练，None表示从头训练

    Returns:
        训练好的模型
    """
    global training_state

    training_state["is_running"] = True
    training_state["epoch"] = 0
    training_state["total_epochs"] = epochs
    training_state["loss"] = 0.0
    training_state["progress"] = 0.0

    callbacks = TrainingCallbacks(sio_server)

    try:
        logger.info("=" * 50)
        logger.info("开始训练ACT模型")
        if resume_from:
            logger.info(f"从已有模型继续训练: {resume_from}")
        logger.info("=" * 50)

        # 加载数据
        data = load_dataset(data_dir)

        # 获取动作维度
        action_dim = data["action"].shape[-1]  # 现在是 2
        raw_state_dim = data["observation.state"].shape[-1]

        # 使用 3 维速度：x_vel, y_vel, theta_vel
        state_dim = 3

        logger.info(f"action_dim: {action_dim}, state_dim: {state_dim} (原始: {raw_state_dim})")

        # 创建配置 - 与推理配置一致
        config = ACTConfig(
            state_dim=3,  # [x_vel, y_vel, theta_vel]
            action_dim=action_dim,
            action_chunk_size=8,  # 预测未来8帧
            hidden_dim=512,
            num_encoder_layers=4,
            num_decoder_layers=4,
            num_attention_heads=8,
            dim_feedforward=3200,
            use_cvae=True,  # 启用 CVAE
            kl_weight=0.1,
            use_temporal_ensembling=False,
            use_spatial_softmax=True,
            latent_dim=32,
        )

        # 创建模型
        model = ACTModel(config)

        # 如果指定了从已有模型继续训练
        if resume_from:
            resume_path = Path(resume_from)
            if resume_path.exists():
                model.load_state_dict(torch.load(resume_path, map_location='cpu'))
                logger.info(f"已加载已有模型: {resume_path}")
            else:
                logger.warning(f"指定的可模型文件不存在: {resume_path}，从头开始训练")

        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 数据集 - 传入统计信息用于归一化
        stats = data.get("stats")
        dataset = SimpleDataset(data, stats=stats)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # 设备
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        device = torch.device(device)
        model = model.to(device)
        logger.info(f"使用设备: {device}")

        # 训练循环
        model.train()
        total_batches = len(dataloader)

        for epoch in range(epochs):
            callbacks.on_epoch_start(epoch, epochs)

            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                images = batch["observation"]["image"].to(device)
                states = batch["observation"]["state"].to(device)
                actions = batch["action"].to(device)

                optimizer.zero_grad()

                # actions 已经是 [batch, chunk_size, action_dim] = [batch, 8, 3]
                # 使用forward进行训练
                output = model(images, states, action_target=actions, infer_cvae=False)
                predicted_actions = output["action"]
                kl_loss = output.get("kl_loss")

                # 计算 L1 损失
                l1_loss = criterion(predicted_actions, actions)

                # 计算总损失
                if kl_loss is not None:
                    loss = l1_loss + kl_loss * config.kl_weight
                else:
                    loss = l1_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                callbacks.on_batch_end(loss.item(), batch_idx, total_batches)

            avg_loss = total_loss / total_batches
            callbacks.on_epoch_end(epoch, avg_loss)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # 保存模型到output目录
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_path = project_root / "output" / "train"
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / "model.pt"
        torch.save(model.state_dict(), final_path)
        logger.info(f"模型已保存到: {final_path}")

        callbacks.on_train_end(str(final_path))

        return model

    except Exception as e:
        logger.error(f"训练失败: {e}")
        training_state["is_running"] = False
        raise


def get_training_state():
    """获取训练状态"""
    return training_state
