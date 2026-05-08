from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from backend.services.mujoco_renderer.renderer import MujocoRenderer


class MujocoState:
    """Manages MuJoCo scene state and car+arm control."""

    def __init__(self, renderer: MujocoRenderer):
        self._renderer = renderer

    def set_arm_position(self, qpos: list[float]) -> None:
        """Set arm joint positions [yaw, pitch, roll, wrist]."""
        if len(qpos) >= 3:
            self._renderer._data.qpos[7] = qpos[0]
        if len(qpos) >= 3:
            self._renderer._data.qpos[8] = qpos[1]
        if len(qpos) >= 4:
            self._renderer._data.qpos[9] = qpos[2]

    def get_state(self) -> dict:
        """Get current state dict."""
        return self._renderer.get_state()

    def reset(self) -> None:
        """Reset to initial state."""
        mujoco.mj_resetData(self._renderer._model, self._renderer._data)