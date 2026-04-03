from __future__ import annotations

import asyncio
import logging

from backend.models import state
from backend.services.simulator.controller import SimController
from backend.sio_handlers.core.runtime import SioRuntimeState

logger = logging.getLogger(__name__)


async def game_loop_task(sio_server, runtime: SioRuntimeState, sim_controller: SimController, namespace: str = "/"):
    """游戏循环 - 处理物理更新，给每个连接单独发送状态"""
    logger.info(f"[游戏循环] 任务已启动, namespace={namespace}")
    frame_count = 0
    # 注意：这里依然使用全局 state.car_state，因为 car_state 是全局的
    # 如果需要完全隔离，需要把 car_state 移到 SioRuntimeState 中
    last_emitted_state = dict(state.car_state)

    while True:
        try:
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(
                    f"[游戏循环] frame={frame_count}, inference_mode={runtime.inference_mode}, current_actions={runtime.current_actions}, connected_clients={len(runtime.connected_clients)}, namespace={namespace}"
                )

            sim_controller.tick()

            if runtime.connected_clients and state.car_state != last_emitted_state:
                # 给每个连接单独发送状态，而不是广播
                for sid in runtime.connected_clients:
                    try:
                        await sio_server.emit("car_state_update", state.car_state, room=sid, namespace=namespace)
                    except Exception:
                        pass
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
