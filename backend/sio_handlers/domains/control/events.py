from __future__ import annotations

import logging

from backend.models import state

logger = logging.getLogger(__name__)


class ControlEventsMixin:
    async def on_connect(self, sid: str, environ: dict):
        self.runtime.connected_clients.add(sid)
        logger.info(f"客户端连接: {sid}")
        await self.emit("connected", {"sid": sid})
        await self.emit("car_state_update", self.sim_controller.get_car_state())
        if state.camera_image:
            await self.emit("camera_image", {"image": state.camera_image})

    async def on_disconnect(self, sid: str):
        self.runtime.connected_clients.discard(sid)
        if not self.runtime.connected_clients:
            self.runtime.current_actions.clear()
        logger.info(f"客户端断开: {sid}")

    async def on_action(self, sid: str, actions: list):
        logger.info(f"收到控制动作: actions={actions}")
        self.sim_controller.set_actions(actions)

    async def on_reset_car_state(self, sid: str):
        logger.info("收到复位场景请求")
        await self.emit("car_state_update", self.sim_controller.reset_car_state())

    async def on_get_car_state(self, sid: str):
        await self.emit("car_state_update", self.sim_controller.get_car_state())
