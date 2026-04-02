from __future__ import annotations

import logging

from backend.config import config as backend_config

logger = logging.getLogger(__name__)


class InferenceEventsMixin:
    async def on_act_infer(self, sid: str, payload: dict):
        logger.info("收到 ACT 推理请求")

        try:
            inference_state = payload.get("state", [0.0] * backend_config.STATE_DIM)
            image = payload.get("image")
            logger.info(f"state: {inference_state}, image provided: {image is not None}")
            if image:
                logger.info(f"image length: {len(image)}")

            action = self.sim_controller.infer(inference_state, image)
            await self.emit("car_state_update", self.sim_controller.get_car_state())
            await self.emit("act_infer_result", {"success": True, "action": action})
        except Exception as exc:
            logger.error(f"ACT 推理失败: {exc}")
            await self.emit("act_infer_result", {"success": False, "error": str(exc)})
