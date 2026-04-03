"""
AKA-Sim 后端 - Socket.IO 事件处理
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from backend.sio_handlers.core.namespace import SimNamespace as _SimNamespace
from backend.sio_handlers.core.runtime import SioRuntimeState
from backend.sio_handlers.core.tasks import game_loop_task

if TYPE_CHECKING:
    from backend.services.episode import EpisodeService
    from backend.services.simulator import CameraService, SimController

# 全局共享的 act_runtime
_act_runtime = None

# Sim 命名空间的独立状态
_sim_runtime_state = SioRuntimeState()
_sim_controller = None
_sim_episode_service = None
_sim_camera_service = None

# Real 命名空间的独立状态
_real_runtime_state = SioRuntimeState()
_real_controller = None
_real_episode_service = None
_real_camera_service = None


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


def _get_sim_camera_service():
    global _sim_camera_service
    if _sim_camera_service is None:
        from backend.services.simulator import CameraService

        _sim_camera_service = CameraService(_sim_runtime_state)
    return _sim_camera_service


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


def _get_real_camera_service():
    global _real_camera_service
    if _real_camera_service is None:
        from backend.services.simulator import CameraService

        _real_camera_service = CameraService(_real_runtime_state)
    return _real_camera_service


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime
    _sim_runtime_state.set_act_runtime(runtime)
    _real_runtime_state.set_act_runtime(runtime)


class SimNamespace(_SimNamespace):
    """Sim 页面专用命名空间 - /sim"""
    def __init__(
        self,
        namespace: str | None = "/sim",
        runtime: SioRuntimeState | None = None,
        sim_controller: SimController | None = None,
        episode_service: EpisodeService | None = None,
        camera_service: CameraService | None = None,
    ):
        super().__init__(
            namespace=namespace,
            runtime=runtime or _sim_runtime_state,
            sim_controller=sim_controller or _get_sim_controller(),
            episode_service=episode_service or _get_sim_episode_service(),
            camera_service=camera_service or _get_sim_camera_service(),
        )


class RealNamespace(_SimNamespace):
    """Real 页面专用命名空间 - /real"""
    def __init__(
        self,
        namespace: str | None = "/real",
        runtime: SioRuntimeState | None = None,
        sim_controller: SimController | None = None,
        episode_service: EpisodeService | None = None,
        camera_service: CameraService | None = None,
    ):
        super().__init__(
            namespace=namespace,
            runtime=runtime or _real_runtime_state,
            sim_controller=sim_controller or _get_real_controller(),
            episode_service=episode_service or _get_real_episode_service(),
            camera_service=camera_service or _get_real_camera_service(),
        )


def start_game_loop(
    sio_server,
    runtime: SioRuntimeState | None = None,
    sim_controller: SimController | None = None,
    camera_service: CameraService | None = None,
    namespace: str = "/",
):
    """启动游戏循环 - 支持指定命名空间"""
    if namespace == "/sim":
        runtime_state = runtime or _sim_runtime_state
        controller = sim_controller or _get_sim_controller()
        camera = camera_service or _get_sim_camera_service()
    elif namespace == "/real":
        runtime_state = runtime or _real_runtime_state
        controller = sim_controller or _get_real_controller()
        camera = camera_service or _get_real_camera_service()
    else:
        # 默认使用 sim 状态（保持向后兼容）
        runtime_state = runtime or _sim_runtime_state
        controller = sim_controller or _get_sim_controller()
        camera = camera_service or _get_sim_camera_service()

    asyncio.create_task(game_loop_task(sio_server, runtime_state, controller, namespace=namespace))
    asyncio.create_task(camera.run_loop(sio_server, namespace=namespace))
