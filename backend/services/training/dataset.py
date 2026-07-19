"""Torch dataset adapters for ACT training - QUANTILES 归一化版本"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class SimpleDataset(torch.utils.data.Dataset):
    """简化数据集 - 使用 QUANTILES 归一化 (1%/99% 百分位数)

    通过滑动窗口构造 action chunk，与 ACTDataset 行为一致。
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        stats: Optional[Dict] = None,
        action_chunk_size: int = 8,
    ):
        self.data = data
        self.stats = stats
        self.action_chunk_size = action_chunk_size

        # 滑动窗口：num_samples = N - chunk_size + 1
        self.num_samples = max(1, data["action"].shape[0] - action_chunk_size + 1)

        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if stats:
            state_entry = stats.get("observation.state", {})
            action_entry = stats.get("action", {})
            self.state_q01 = torch.tensor(state_entry.get("q01", [0, 0]), dtype=torch.float32)
            self.state_q99 = torch.tensor(state_entry.get("q99", [1, 1]), dtype=torch.float32)
            self.action_q01 = torch.tensor(action_entry.get("q01", [0, 0]), dtype=torch.float32)
            self.action_q99 = torch.tensor(action_entry.get("q99", [1, 1]), dtype=torch.float32)
        else:
            self.state_q01 = torch.zeros(2)
            self.state_q99 = torch.ones(2)
            self.action_q01 = torch.zeros(2)
            self.action_q99 = torch.ones(2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 观测是当前时刻 t
        images = self.data["observation.image"][idx]
        state = self.data["observation.state"][idx]

        # 动作是从 t 开始的未来 chunk_size 步 (滑动窗口)
        action = self.data["action"][idx:idx + self.action_chunk_size]

        images = (images.unsqueeze(0) - self.image_mean) / self.image_std

        if self.stats:
            # QUANTILES 归一化: 2 * (x - q01) / (q99 - q01) - 1
            denom_state = self.state_q99 - self.state_q01
            denom_state = torch.where(denom_state == 0, torch.tensor(1e-8), denom_state)
            state = 2 * (state - self.state_q01) / denom_state - 1

        if self.stats:
            denom_action = self.action_q99 - self.action_q01
            denom_action = torch.where(denom_action == 0, torch.tensor(1e-8), denom_action)
            action = 2 * (action - self.action_q01) / denom_action - 1

        return {
            "observation": {
                "image": images,
                "state": state,
            },
            "action": action,
        }
