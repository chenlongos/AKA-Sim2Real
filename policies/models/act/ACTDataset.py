from typing import Dict, Tuple, Optional

import torch


class ACTDataset(torch.utils.data.Dataset):
    """
    ACT 数据集 - 用于行为克隆训练
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        action_chunk_size: int = 16,
        normalize_images: bool = True,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        state_mean: Optional[torch.Tensor] = None,
        state_std: Optional[torch.Tensor] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            data: 包含 'observation.image', 'observation.state', 'action' 的字典
            action_chunk_size: 动作分块大小
            state_mean/std: 状态归一化参数
            action_mean/std: 动作归一化参数
        """
        self.data = data
        self.action_chunk_size = action_chunk_size

        # 图像归一化参数
        self.normalize_images = normalize_images
        self.image_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std).view(1, 3, 1, 1)

        # 状态和动作归一化参数
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std

        action_tensor = data["action"]
        if action_tensor.ndim not in (2, 3):
            raise ValueError(
                f"Unsupported action tensor shape {tuple(action_tensor.shape)}; "
                "expected [T, action_dim] or [N, chunk_size, action_dim]."
            )

        # 支持两种动作数据格式：
        # 1. [T, action_dim]：按时间滑窗构造未来 action chunk
        # 2. [N, chunk_size, action_dim]：每条记录已经是完整 chunk
        self.actions_are_chunked = action_tensor.ndim == 3
        if self.actions_are_chunked:
            self.num_samples = action_tensor.shape[0]
            if action_tensor.shape[1] != action_chunk_size:
                raise ValueError(
                    f"Configured action_chunk_size={action_chunk_size}, "
                    f"but action data has chunk size {action_tensor.shape[1]}."
                )
        else:
            self.num_samples = action_tensor.shape[0] - action_chunk_size + 1
            if self.num_samples <= 0:
                raise ValueError(
                    f"Not enough timesteps ({action_tensor.shape[0]}) for "
                    f"action_chunk_size={action_chunk_size}."
                )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Returns:
            sample: 包含 'observation' 和 'action' 的字典

        注意: 根据 ACT 论文，模型根据当前观测预测未来 chunk_size 步的动作
        所以观测和动作应该是对齐的：观测是时刻 t，动作是 t 到 t+chunk_size-1
        """
        # 获取当前时间步的观察（不是未来时刻！）
        current_idx = idx

        # 支持两种数据格式
        if "observation.image" in self.data:
            images = self.data["observation.image"][current_idx]
            state = self.data["observation.state"][current_idx]
        else:
            images = self.data["observation"]["image"][current_idx]
            state = self.data["observation"]["state"][current_idx]

        if self.actions_are_chunked:
            action = self.data["action"][idx]
        else:
            # 动作是从当前时刻开始的未来 chunk_size 步
            action = self.data["action"][idx:idx + self.action_chunk_size]

        # 归一化图像
        if self.normalize_images:
            images = (images - self.image_mean.to(images.device)) / self.image_std.to(images.device)

        # 归一化状态
        if self.state_mean is not None and self.state_std is not None:
            state = (state - self.state_mean.to(state.device)) / (self.state_std.to(state.device) + 1e-8)

        # 归一化动作
        if self.action_mean is not None and self.action_std is not None:
            action = (action - self.action_mean.to(action.device)) / (self.action_std.to(action.device) + 1e-8)

        return {
            "observation": {
                "image": images,
                "state": state,
            },
            "action": action,
        }
