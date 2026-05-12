from typing import Dict, Tuple, Optional

import torch


class ACTDataset(torch.utils.data.Dataset):
    """
    ACT 数据集 - 用于行为克隆训练
    使用 QUANTILES 归一化 (1%/99% 百分位数)
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        action_chunk_size: int = 16,
        normalize_images: bool = True,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        state_q01: Optional[torch.Tensor] = None,
        state_q99: Optional[torch.Tensor] = None,
        action_q01: Optional[torch.Tensor] = None,
        action_q99: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            data: 包含 'observation.image', 'observation.state', 'action' 的字典
            action_chunk_size: 动作分块大小
            state_q01/q99: 状态百分位数归一化参数
            action_q01/q99: 动作百分位数归一化参数
        """
        self.data = data
        self.action_chunk_size = action_chunk_size

        # 图像归一化参数
        self.normalize_images = normalize_images
        self.image_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std).view(1, 3, 1, 1)

        # QUANTILES 归一化参数
        self.state_q01 = state_q01
        self.state_q99 = state_q99
        self.action_q01 = action_q01
        self.action_q99 = action_q99

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

        # 使用 QUANTILES 归一化状态: 2 * (x - q01) / (q99 - q01) - 1
        if self.state_q01 is not None and self.state_q99 is not None:
            q01 = self.state_q01.to(state.device)
            q99 = self.state_q99.to(state.device)
            denom = q99 - q01
            denom = torch.where(denom == 0, torch.tensor(1e-8, device=state.device), denom)
            state = 2 * (state - q01) / denom - 1

        # 使用 QUANTILES 归一化动作: 2 * (x - q01) / (q99 - q01) - 1
        if self.action_q01 is not None and self.action_q99 is not None:
            q01 = self.action_q01.to(action.device)
            q99 = self.action_q99.to(action.device)
            denom = q99 - q01
            denom = torch.where(denom == 0, torch.tensor(1e-8, device=action.device), denom)
            action = 2 * (action - q01) / denom - 1

        return {
            "observation": {
                "image": images,
                "state": state,
            },
            "action": action,
        }
