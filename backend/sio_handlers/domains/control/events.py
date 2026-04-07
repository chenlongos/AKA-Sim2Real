from __future__ import annotations

import logging

from backend.models import state
from backend.utils import log_broadcast

logger = logging.getLogger(__name__)


class ControlEventsMixin:
    async def on_connect(self, sid: str, environ: dict, auth: dict | None = None):
        self.runtime.connected_clients.add(sid)
        log_broadcast.add_connected_sid(sid, namespace=self.namespace)
        logger.info(f"客户端连接: {sid}, namespace={self.namespace}, auth={auth}")
        await self.emit("connected", {"sid": sid})
        await self.emit("car_state_update", self.sim_controller.get_car_state())
        if state.camera_image:
            await self.emit("camera_image", {"image": state.camera_image})

    async def on_disconnect(self, sid: str):
        self.runtime.connected_clients.discard(sid)
        log_broadcast.remove_connected_sid(sid, namespace=self.namespace)
        if not self.runtime.connected_clients:
            self.runtime.current_action_vector = None
        logger.info(f"客户端断开: {sid}, namespace={self.namespace}")

    async def on_action(self, sid: str, action: list[float]):
        logger.info(f"收到控制动作: action={action}")
        self.sim_controller.set_action(action)

    async def on_reset_car_state(self, sid: str):
        logger.info("收到复位场景请求")
        await self.emit("car_state_update", self.sim_controller.reset_car_state())

    async def on_get_car_state(self, sid: str):
        await self.emit("car_state_update", self.sim_controller.get_car_state())
