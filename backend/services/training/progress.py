"""Training progress callbacks."""

from __future__ import annotations

import asyncio

from backend.services.training.state import training_state


class TrainingCallbacks:
    """训练回调 - 用于推送进度"""

    def __init__(self, sio_server=None):
        self.sio_server = sio_server

    def on_epoch_start(self, epoch: int, total_epochs: int):
        training_state["epoch"] = epoch + 1
        training_state["total_epochs"] = total_epochs
        self._emit_progress()

    def on_batch_end(self, loss: float, batch_idx: int, total_batches: int):
        training_state["loss"] = loss
        training_state["progress"] = (training_state["epoch"] - 1 + batch_idx / total_batches) / training_state["total_epochs"]
        self._emit_progress()

    def on_epoch_end(self, epoch: int, avg_loss: float):
        training_state["loss"] = avg_loss
        self._emit_progress()

    def on_train_end(self, model_path: str):
        training_state["is_running"] = False
        training_state["progress"] = 1.0
        self._emit_progress()

    def _emit_progress(self):
        if self.sio_server:
            try:
                asyncio.create_task(self.sio_server.emit("training_progress", training_state))
            except Exception:
                pass
