"""Training progress callbacks."""

from __future__ import annotations

import asyncio

from backend.services.training.state import training_state
from backend.utils import log_broadcast


class TrainingCallbacks:
    """训练回调 - 用于推送进度"""

    def __init__(self, sio_server=None, loop=None, namespace=None):
        self.sio_server = sio_server
        self.loop = loop
        self.namespace = namespace or "/"

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
                if self.loop and self.loop.is_running():
                    # 在线程池中运行时，通过call_soon_threadsafe回到主线程
                    async def emit():
                        # 给每个连接单独发送
                        for sid in log_broadcast._connected_sids:
                            try:
                                await self.sio_server.emit(
                                    "training_progress",
                                    training_state,
                                    room=sid,
                                    namespace=self.namespace
                                )
                            except Exception:
                                pass
                    asyncio.run_coroutine_threadsafe(emit(), self.loop)
                else:
                    # 在主线程运行时直接create_task
                    async def emit():
                        # 给每个连接单独发送
                        for sid in log_broadcast._connected_sids:
                            try:
                                await self.sio_server.emit(
                                    "training_progress",
                                    training_state,
                                    room=sid,
                                    namespace=self.namespace
                                )
                            except Exception:
                                pass
                    asyncio.create_task(emit())
            except Exception:
                pass
