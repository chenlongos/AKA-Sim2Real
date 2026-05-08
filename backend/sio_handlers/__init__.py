"""
AKA-Sim 后端 - Socket.IO 事件处理
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from backend.sio_handlers.core.namespace import SimNamespace as _SimNamespace
from backend.sio_handlers.core.runtime import SioRuntimeState
from backend.sio_handlers.core.tasks import game_loop_task
from backend.sio_handlers.domains.mujoco import MujocoEventsMixin
from backend.sio_handlers.core.base import BaseSimNamespace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from backend.services.episode import EpisodeService
    from backend.services.simulator import SimController

# 全局共享的 act_runtime
_act_runtime = None

# Sim 命名空间的独立状态
_sim_runtime_state = SioRuntimeState()
_sim_controller = None
_sim_episode_service = None

# Real 命名空间的独立状态
_real_runtime_state = SioRuntimeState()
_real_controller = None
_real_episode_service = None


def _get_sim_controller():
    global _sim_controller
    if _sim_controller is None:
        from backend.services.simulator import SimController

        _sim_controller = SimController(_sim_runtime_state)
    return _sim_controller


def _get_sim_episode_service():
    global _sim_episode_service
    if _sim_episode_service is None:
        from backend.services.episode import EpisodeService

        _sim_episode_service = EpisodeService()
    return _sim_episode_service


def _get_real_controller():
    global _real_controller
    if _real_controller is None:
        from backend.services.simulator import SimController

        _real_controller = SimController(_real_runtime_state)
    return _real_controller


def _get_real_episode_service():
    global _real_episode_service
    if _real_episode_service is None:
        from backend.services.episode import EpisodeService

        _real_episode_service = EpisodeService()
    return _real_episode_service


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime
    _sim_runtime_state.set_act_runtime(runtime)
    _real_runtime_state.set_act_runtime(runtime)


class MujocoNamespace(
    MujocoEventsMixin,
    BaseSimNamespace,
):
    """MuJoCo 仿真页面专用命名空间 - /mujoco"""

    def __init__(
        self,
        namespace: str | None = "/mujoco",
        runtime: SioRuntimeState | None = None,
        sim_controller: SimController | None = None,
        episode_service: EpisodeService | None = None,
    ):
        super().__init__(
            namespace=namespace,
            runtime=runtime or _sim_runtime_state,
            sim_controller=sim_controller or _get_sim_controller(),
            episode_service=episode_service or _get_sim_episode_service(),
        )


class SimNamespace(_SimNamespace):
    """Sim 页面专用命名空间 - /sim"""
    def __init__(
        self,
        namespace: str | None = "/sim",
        runtime: SioRuntimeState | None = None,
        sim_controller: SimController | None = None,
        episode_service: EpisodeService | None = None,
    ):
        super().__init__(
            namespace=namespace,
            runtime=runtime or _sim_runtime_state,
            sim_controller=sim_controller or _get_sim_controller(),
            episode_service=episode_service or _get_sim_episode_service(),
        )


class RealNamespace(_SimNamespace):
    """Real 页面专用命名空间 - /real"""
    def __init__(
        self,
        namespace: str | None = "/real",
        runtime: SioRuntimeState | None = None,
        sim_controller: SimController | None = None,
        episode_service: EpisodeService | None = None,
    ):
        super().__init__(
            namespace=namespace,
            runtime=runtime or _real_runtime_state,
            sim_controller=sim_controller or _get_real_controller(),
            episode_service=episode_service or _get_real_episode_service(),
        )


def start_game_loop(
    sio_server,
    runtime: SioRuntimeState | None = None,
    sim_controller: SimController | None = None,
    namespace: str = "/",
):
    """启动游戏循环 - 支持指定命名空间"""
    if namespace == "/sim":
        runtime_state = runtime or _sim_runtime_state
        controller = sim_controller or _get_sim_controller()
    elif namespace == "/real":
        runtime_state = runtime or _real_runtime_state
        controller = sim_controller or _get_real_controller()
    else:
        runtime_state = runtime or _sim_runtime_state
        controller = sim_controller or _get_sim_controller()

    asyncio.create_task(game_loop_task(sio_server, runtime_state, controller, namespace=namespace))


async def mujoco_game_loop_task(sio_server, runtime: SioRuntimeState, namespace: str = "/mujoco"):
    """MuJoCo 渲染循环 - 持续渲染并推送状态给连接的客户端"""
    from backend.services.mujoco_renderer.service import get_mujoco_service

    logger.info(f"[mujoco_game_loop] 任务已启动, namespace={namespace}")
    service = get_mujoco_service()
    frame_count = 0

    while True:
        try:
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"[mujoco_game_loop] frame={frame_count}, clients={len(runtime.connected_clients)}")

            if runtime.connected_clients:
                # 每帧都渲染并推送（~50ms 间隔）
                topdown, firstperson, state = service.render()
                # Convert ndarray values in state to lists
                serializable_state = {
                    k: v.tolist() if hasattr(v, "tolist") else v
                    for k, v in state.items()
                }
                payload = {
                    "topdown": topdown,
                    "firstperson": firstperson,
                    "state": serializable_state,
                }
                # 广播给所有连接的客户端
                await sio_server.emit("mujoco_state_update", payload, namespace=namespace)

            # 物理步进
            service.step()

        except Exception as exc:
            logger.error(f"[mujoco_game_loop] 错误: {exc}")

        await asyncio.sleep(0.05)


def start_mujoco_game_loop(sio_server, namespace: str = "/mujoco"):
    """启动 MuJoCo 游戏循环"""
    # 使用 _sim_runtime_state 作为 runtime（MujocoNamespace 也用它）
    asyncio.create_task(mujoco_game_loop_task(sio_server, _sim_runtime_state, namespace=namespace))
