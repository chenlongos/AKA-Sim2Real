from __future__ import annotations

import logging
import os
from typing import Tuple

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


class MujocoRenderer:
    """Dual-view MuJoCo renderer for top-down and first-person views."""

    def __init__(self, xml_path: str | None = None):
        if xml_path is None:
            xml_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "mujoco", "car_arm.xml"
            )
        self._xml_path = os.path.abspath(xml_path)

        self._model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._data = mujoco.MjData(self._model)

        self._renderer_topdown = mujoco.Renderer(self._model, width=640, height=480)
        self._renderer_firstperson = mujoco.Renderer(self._model, width=640, height=480)

        self._cam_topdown = mujoco.MjvCamera()
        self._cam_firstperson = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._cam_topdown)
        mujoco.mjv_defaultCamera(self._cam_firstperson)

        self._setup_cameras()
        logger.info(f"MujocoRenderer initialized with {self._xml_path}")

    def _camera_name_to_id(self, name: str) -> int:
        """Convert camera name to ID using mujoco.mj_name2id."""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    def _actuator_name_to_id(self, name: str) -> int:
        """Convert actuator name to ID using mujoco.mj_name2id."""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _setup_cameras(self) -> None:
        """Configure top-down and first-person cameras."""
        self._cam_topdown.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._cam_topdown.fixedcamid = self._camera_name_to_id("topdown")
        self._cam_topdown.lookat = np.array([0, 0, 0])

        self._cam_firstperson.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._cam_firstperson.fixedcamid = self._camera_name_to_id("firstperson")

    def get_topdown_image(self) -> np.ndarray:
        """Render top-down view."""
        self._renderer_topdown.update_scene(self._data, self._cam_topdown)
        return self._renderer_topdown.render()[::-1]

    def get_firstperson_image(self) -> np.ndarray:
        """Render first-person view from camera on car."""
        self._renderer_firstperson.update_scene(self._data, self._cam_firstperson)
        return self._renderer_firstperson.render()[::-1]

    def step(self, arm_action: dict | None = None) -> None:
        """Step the simulation."""
        if arm_action is not None:
            for actuator_name, torque in arm_action.items():
                try:
                    actuator_id = self._actuator_name_to_id(actuator_name)
                    if actuator_id >= 0:
                        self._data.actuator_force[actuator_id] = torque
                except Exception:
                    pass

        mujoco.mj_step(self._model, self._data)

    def set_wheel_torques(self, left_torque: float, right_torque: float) -> None:
        """Apply torques to wheels for differential drive.

        Args:
            left_torque: torque for left wheels (fl, rl)
            right_torque: torque for right wheels (fr, rr)
        """
        wheel_torques = {
            "motor_wheel_fl": left_torque,
            "motor_wheel_rl": left_torque,
            "motor_wheel_fr": right_torque,
            "motor_wheel_rr": right_torque,
        }
        for actuator_name, torque in wheel_torques.items():
            try:
                actuator_id = self._actuator_name_to_id(actuator_name)
                if actuator_id >= 0:
                    self._data.actuator_force[actuator_id] = torque
            except Exception:
                pass

    def get_state(self) -> dict:
        """Get current state."""
        return {
            "car_pos": self._data.body("car").xpos.copy(),
            "car_quat": self._data.body("car").xquat.copy(),
            "arm_qpos": self._data.qpos[7:].copy(),
            "arm_qvel": self._data.qvel[6:].copy(),
        }

    def close(self) -> None:
        """Clean up renderers."""
        self._renderer_topdown.close()
        self._renderer_firstperson.close()