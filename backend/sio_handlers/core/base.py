from __future__ import annotations

from typing import TYPE_CHECKING

from socketio import AsyncNamespace

from backend.sio_handlers.core.runtime import SioRuntimeState

if TYPE_CHECKING:
    from backend.services.episode import EpisodeService
    from backend.services.simulator import SimController


class BaseSimNamespace(AsyncNamespace):
    def __init__(
        self,
        namespace: str | None = None,
        runtime: SioRuntimeState | None = None,
        sim_controller: SimController | None = None,
        episode_service: EpisodeService | None = None,
    ):
        super().__init__(namespace)
        self.runtime = runtime
        self.sim_controller = sim_controller
        self.episode_service = episode_service
