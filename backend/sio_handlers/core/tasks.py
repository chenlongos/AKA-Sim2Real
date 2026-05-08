from __future__ import annotations

import asyncio
import logging

from backend.services.simulator.controller import SimController
from backend.sio_handlers.core.runtime import SioRuntimeState

logger = logging.getLogger(__name__)


async def game_loop_task(sio_server, runtime: SioRuntimeState, sim_controller: SimController, namespace: str = "/"):
    """仿真循环 - 处理推理模式和客户端连接状态"""
    logger.info(f"[仿真循环] 任务已启动, namespace={namespace}")
    frame_count = 0

    while True:
        try:
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(
                    f"[仿真循环] frame={frame_count}, inference_mode={runtime.inference_mode}, current_action_vector={runtime.current_action_vector}, connected_clients={len(runtime.connected_clients)}, namespace={namespace}"
                )

        except Exception as exc:
            logger.error(f"仿真循环错误: {exc}")

        if not runtime.connected_clients:
            sleep_interval = 0.2
        elif runtime.current_action_vector is not None or runtime.inference_mode:
            sleep_interval = 1 / 30
        else:
            sleep_interval = 0.1

        await asyncio.sleep(sleep_interval)
