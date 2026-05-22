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

        self._cam_state = {"azimuth": 0.0, "elevation": -70.0, "distance": 6.0}
        self._setup_cameras()
        logger.info(f"MujocoRenderer initialized with {self._xml_path}")
    def _camera_name_to_id(self, name: str) -> int:
        """Convert camera name to ID using mujoco.mj_name2id."""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    def _actuator_name_to_id(self, name: str) -> int:
        """Convert actuator name to ID using mujoco.mj_name2id."""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _setup_cameras(self) -> None:
        """Configure top-down (free orbit) and first-person (fixed) cameras."""
        self._cam_topdown.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam_topdown.lookat = np.array([0, 0, 0.3])
        self._cam_topdown.distance = self._cam_state["distance"]
        self._cam_topdown.elevation = self._cam_state["elevation"]
        self._cam_topdown.azimuth = self._cam_state["azimuth"]

        self._cam_firstperson.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._cam_firstperson.fixedcamid = self._camera_name_to_id("firstperson")

    def update_topdown_camera(self, delta_azimuth: float, delta_elevation: float) -> None:
        """Rotate the top-down camera by mouse drag deltas."""
        sensitivity = 0.3
        self._cam_state["azimuth"] += delta_azimuth * sensitivity
        self._cam_state["elevation"] += delta_elevation * sensitivity
        self._cam_state["elevation"] = max(-89.0, min(89.0, self._cam_state["elevation"]))

        self._cam_topdown.azimuth = self._cam_state["azimuth"]
        self._cam_topdown.elevation = self._cam_state["elevation"]

    def update_topdown_distance(self, delta: float) -> None:
        """Zoom the top-down camera by scroll delta."""
        self._cam_state["distance"] += delta * 0.5
        self._cam_state["distance"] = max(0.5, min(50.0, self._cam_state["distance"]))
        self._cam_topdown.distance = self._cam_state["distance"]

    def get_topdown_image(self) -> np.ndarray:
        """Render top-down view."""
        self._renderer_topdown.update_scene(self._data, self._cam_topdown)
        return self._renderer_topdown.render()

    def get_firstperson_image(self) -> np.ndarray:
        """Render first-person view from camera on car."""
        self._renderer_firstperson.update_scene(self._data, self._cam_firstperson)
        return self._renderer_firstperson.render()

    def step(self, arm_action: dict | None = None) -> None:
        """Step the simulation."""
        if arm_action is not None:
            for actuator_name, torque in arm_action.items():
                try:
                    actuator_id = self._actuator_name_to_id(actuator_name)
                    if actuator_id >= 0:
                        self._data.ctrl[actuator_id] = torque
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
                    self._data.ctrl[actuator_id] = torque
            except Exception:
                pass

    def get_state(self) -> dict:
        """Get current state."""
        car_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "car"
        )
        return {
            "car_pos": self._data.xpos[car_body_id].copy(),
            "car_quat": self._data.xquat[car_body_id].copy(),
            "arm_qpos": self._data.qpos[11:].copy(),
            "arm_qvel": self._data.qvel[10:].copy(),
        }

    def close(self) -> None:
        """Clean up renderers."""
        self._renderer_topdown.close()
        self._renderer_firstperson.close()