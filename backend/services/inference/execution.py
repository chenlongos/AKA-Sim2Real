"""Execution-time policies for ACT inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import torch


@dataclass
class TemporalEnsemblingPolicy:
    """Online action-chunk fusion policy for ACT inference."""

    decay: float = 0.5
    step: int = 0
    predictions: Deque[dict] = field(default_factory=deque)

    def reset(self):
        self.step = 0
        self.predictions.clear()

    def update_decay(self, decay: float):
        self.decay = min(max(float(decay), 1e-3), 1.0)

    def blend(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
            return action_chunk

        current_step = self.step
        chunk_size = action_chunk.shape[1]
        prediction = action_chunk[0].detach().clone().cpu()
        self.predictions.append({
            "start_step": current_step,
            "actions": prediction,
        })

        min_valid_start = current_step - chunk_size + 1
        while self.predictions and self.predictions[0]["start_step"] < min_valid_start:
            self.predictions.popleft()

        candidates = []
        weights = []
        for pred in self.predictions:
            rel_idx = current_step - pred["start_step"]
            if 0 <= rel_idx < pred["actions"].shape[0]:
                weight = self.decay ** rel_idx
                candidates.append(pred["actions"][rel_idx])
                weights.append(weight)

        if candidates:
            weight_tensor = torch.tensor(weights, dtype=prediction.dtype).unsqueeze(1)
            candidate_tensor = torch.stack(candidates, dim=0)
            blended = (candidate_tensor * weight_tensor).sum(dim=0) / weight_tensor.sum()
            action_chunk = action_chunk.clone()
            action_chunk[0, 0] = blended.to(action_chunk.device)

        self.step += 1
        return action_chunk
