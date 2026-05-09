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
            state_min = torch.tensor(stats.get("observation.state", {}).get("min", [0, 0]), dtype=torch.float32)
            state_max = torch.tensor(stats.get("observation.state", {}).get("max", [1, 1]), dtype=torch.float32)
            action_min = torch.tensor(stats.get("action", {}).get("min", [0, 0]), dtype=torch.float32)
            action_max = torch.tensor(stats.get("action", {}).get("max", [1, 1]), dtype=torch.float32)
            self.state_min = state_min
            self.state_max = state_max
            self.action_min = action_min
            self.action_max = action_max
        else:
            self.state_min = torch.zeros(2)
            self.state_max = torch.ones(2)
            self.action_min = torch.zeros(2)
            self.action_max = torch.ones(2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        images = self.data["observation.image"][idx]
        state = self.data["observation.state"][idx]
        action = self.data["action"][idx]

        images = (images.unsqueeze(0) - self.image_mean) / self.image_std

        if self.stats:
            state = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)

        if self.stats:
            action = (action - self.action_min) / (self.action_max - self.action_min + 1e-8)

        return {
            "observation": {
                "image": images,
                "state": state,
            },
            "action": action,
        }
