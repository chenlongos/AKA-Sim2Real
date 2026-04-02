from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.services.act_model import get_act_runtime


@dataclass
class SioRuntimeState:
    current_actions: set[str] = field(default_factory=set)
    connected_clients: set[str] = field(default_factory=set)
    inference_mode: bool = False
    camera_active: bool = False
    act_runtime: Any = field(default_factory=get_act_runtime)

    def set_act_runtime(self, runtime: Any) -> None:
        self.act_runtime = runtime
