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
    from backend.services.camera_service import CameraService
    from backend.services.episode_service import EpisodeService
    from backend.services.sim_controller import SimController

_runtime_state = SioRuntimeState()
_sim_controller = None
_episode_service = None
_camera_service = None


def _get_sim_controller():
    global _sim_controller
    if _sim_controller is None:
        from backend.services.sim_controller import SimController

        _sim_controller = SimController(_runtime_state)
    return _sim_controller


def _get_episode_service():
    global _episode_service
    if _episode_service is None:
        from backend.services.episode_service import EpisodeService

        _episode_service = EpisodeService()
    return _episode_service


def _get_camera_service():
    global _camera_service
    if _camera_service is None:
        from backend.services.camera_service import CameraService

        _camera_service = CameraService(_runtime_state)
    return _camera_service


def set_act_runtime(runtime):
    _runtime_state.set_act_runtime(runtime)


class SimNamespace(_SimNamespace):
    def __init__(
        self,
        namespace: str | None = None,
        runtime: SioRuntimeState | None = None,
        sim_controller: SimController | None = None,
        episode_service: EpisodeService | None = None,
        camera_service: CameraService | None = None,
    ):
        super().__init__(
            namespace=namespace,
            runtime=runtime or _runtime_state,
            sim_controller=sim_controller or _get_sim_controller(),
            episode_service=episode_service or _get_episode_service(),
            camera_service=camera_service or _get_camera_service(),
        )


def start_game_loop(
    sio_server,
    runtime: SioRuntimeState | None = None,
    sim_controller: SimController | None = None,
    camera_service: CameraService | None = None,
):
    runtime_state = runtime or _runtime_state
    controller = sim_controller or _get_sim_controller()
    camera = camera_service or _get_camera_service()
    asyncio.create_task(game_loop_task(sio_server, runtime_state, controller))
    asyncio.create_task(camera.run_loop(sio_server))
