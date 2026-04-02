from __future__ import annotations

import asyncio
import logging

from backend.models import state
from backend.services.simulator.controller import SimController
from backend.sio_handlers.core.runtime import SioRuntimeState

logger = logging.getLogger(__name__)


async def game_loop_task(sio_server, runtime: SioRuntimeState, sim_controller: SimController):
    """游戏循环 - 处理物理更新和状态广播"""
    logger.info("[游戏循环] 任务已启动")
    frame_count = 0
    last_emitted_state = dict(state.car_state)

    while True:
        try:
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(
                    f"[游戏循环] frame={frame_count}, inference_mode={runtime.inference_mode}, current_actions={runtime.current_actions}"
                )

            sim_controller.tick()

            if runtime.connected_clients and state.car_state != last_emitted_state:
                await sio_server.emit("car_state_update", state.car_state)
                last_emitted_state = dict(state.car_state)
        except Exception as exc:
            logger.error(f"游戏循环错误: {exc}")

        if not runtime.connected_clients:
            sleep_interval = 0.2
        elif runtime.current_actions or runtime.inference_mode or sim_controller.is_moving():
            sleep_interval = 1 / 30
        else:
            sleep_interval = 0.1

        await asyncio.sleep(sleep_interval)
