from __future__ import annotations

from typing import TYPE_CHECKING

from backend.sio_handlers.core.base import BaseSimNamespace
from backend.sio_handlers.core.runtime import SioRuntimeState
from backend.sio_handlers.domains.camera import CameraEventsMixin
from backend.sio_handlers.domains.control import ControlEventsMixin
from backend.sio_handlers.domains.episode import EpisodeEventsMixin
from backend.sio_handlers.domains.inference import InferenceEventsMixin

if TYPE_CHECKING:
    from backend.services.camera_service import CameraService
    from backend.services.episode_service import EpisodeService
    from backend.services.sim_controller import SimController


class SimNamespace(
    ControlEventsMixin,
    InferenceEventsMixin,
    EpisodeEventsMixin,
    CameraEventsMixin,
    BaseSimNamespace,
):
    """模拟器 Socket.IO 命名空间"""

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
            runtime=runtime,
            sim_controller=sim_controller,
            episode_service=episode_service,
            camera_service=camera_service,
        )
