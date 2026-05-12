"""Preprocess helpers for ACT runtime."""

from __future__ import annotations

import base64
import io
from typing import Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from backend.services.inference.checkpoint import ACTNormalizationStats


class ACTPreprocessor:
    """Pure preprocessing helpers for image/state/action runtime tensors."""

    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_image(self, image_input: Union[str, Image.Image, None], device: str) -> torch.Tensor:
        if image_input is None:
            return torch.randn(1, 1, 3, 224, 224).to(device)

        try:
            if isinstance(image_input, str):
                if "," in image_input:
                    image_input = image_input.split(",")[1]
                image_data = base64.b64decode(image_input)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                return torch.randn(1, 1, 3, 224, 224).to(device)
            return self.image_transform(image).unsqueeze(0).unsqueeze(0).to(device)
        except Exception:
            return torch.randn(1, 1, 3, 224, 224).to(device)

    def normalize_state(self, state: list, stats: ACTNormalizationStats, device: str) -> torch.Tensor:
        """使用 QUANTILES 归一化状态：2 * (x - q01) / (q99 - q01) - 1"""
        state_tensor = torch.tensor(state[:2], dtype=torch.float32).unsqueeze(0).to(device)
        q01 = stats.state_q01.unsqueeze(0).to(device)
        q99 = stats.state_q99.unsqueeze(0).to(device)
        denom = q99 - q01
        denom = torch.where(denom == 0, torch.tensor(1e-8, device=device), denom)
        return 2 * (state_tensor - q01) / denom - 1

    def denormalize_action(self, action: torch.Tensor, stats: ACTNormalizationStats, device: str) -> torch.Tensor:
        """使用 QUANTILES 反归一化动作：(x + 1) / 2 * (q99 - q01) + q01"""
        q01 = stats.action_q01.unsqueeze(0).to(device)
        q99 = stats.action_q99.unsqueeze(0).to(device)
        denom = q99 - q01
        denom = torch.where(denom == 0, torch.tensor(1e-8, device=device), denom)
        return (action + 1) / 2 * denom + q01
