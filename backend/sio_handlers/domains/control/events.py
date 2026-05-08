from __future__ import annotations

import logging

from backend.utils import log_broadcast

logger = logging.getLogger(__name__)


class ControlEventsMixin:
    async def on_connect(self, sid: str, environ: dict, auth: dict | None = None):
        self.runtime.connected_clients.add(sid)
        log_broadcast.add_connected_sid(sid, namespace=self.namespace)
        logger.info(f"客户端连接: {sid}, namespace={self.namespace}, auth={auth}")
        await self.emit("connected", {"sid": sid})

    async def on_disconnect(self, sid: str):
        self.runtime.connected_clients.discard(sid)
        log_broadcast.remove_connected_sid(sid, namespace=self.namespace)
        if not self.runtime.connected_clients:
            self.runtime.current_action_vector = None
        logger.info(f"客户端断开: {sid}, namespace={self.namespace}")

    async def on_action(self, sid: str, action: list[float]):
        logger.info(f"收到控制动作: action={action}")
        self.sim_controller.set_action(action)
