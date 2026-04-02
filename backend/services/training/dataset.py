"""Torch dataset adapters for ACT training."""

from __future__ import annotations

from typing import Dict, Optional

import torch


class SimpleDataset(torch.utils.data.Dataset):
    """简化数据集 - 支持归一化"""

    def __init__(self, data: Dict[str, torch.Tensor], stats: Optional[Dict] = None):
        self.data = data
        self.stats = stats
        self.num_samples = data["action"].shape[0]
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if stats:
            self.state_mean = torch.tensor(stats.get("observation.state", {}).get("mean", [0, 0]), dtype=torch.float32)
            self.state_std = torch.tensor(stats.get("observation.state", {}).get("std", [1, 1]), dtype=torch.float32)
            self.action_mean = torch.tensor(stats.get("action", {}).get("mean", [0, 0]), dtype=torch.float32)
            self.action_std = torch.tensor(stats.get("action", {}).get("std", [1, 1]), dtype=torch.float32)
            self.state_std = torch.where(self.state_std > 1e-6, self.state_std, torch.ones_like(self.state_std))
            self.action_std = torch.where(self.action_std > 1e-6, self.action_std, torch.ones_like(self.action_std))
        else:
            self.state_mean = torch.zeros(2)
            self.state_std = torch.ones(2)
            self.action_mean = torch.zeros(2)
            self.action_std = torch.ones(2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        images = self.data["observation.image"][idx]
        state = self.data["observation.state"][idx]
        action = self.data["action"][idx]

        images = (images.unsqueeze(0) - self.image_mean) / self.image_std

        if self.stats:
            state = (state - self.state_mean) / self.state_std

        if self.stats:
            action = (action - self.action_mean) / self.action_std

        return {
            "observation": {
                "image": images,
                "state": state,
            },
            "action": action,
        }
