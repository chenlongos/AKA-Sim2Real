"""
MuJoCo 渲染服务 - 提供双视角渲染能力
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Optional

import mujoco
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_renderer: Optional["MujocoRenderer"] = None
_mujoco_service: Optional["MujocoService"] = None


def get_renderer() -> "MujocoRenderer":
    """获取全局 MuJoCo 渲染器单例"""
    global _renderer
    if _renderer is None:
        from backend.services.mujoco_renderer.renderer import MujocoRenderer
        _renderer = MujocoRenderer()
    return _renderer


def get_mujoco_service() -> "MujocoService":
    """获取全局 MuJoCo 服务单例"""
    global _mujoco_service
    if _mujoco_service is None:
        _mujoco_service = MujocoService()
    return _mujoco_service


def close_mujoco_service() -> None:
    """关闭 MuJoCo 服务"""
    global _mujoco_service
    if _mujoco_service:
        _mujoco_service.close()
        _mujoco_service = None


class MujocoService:
    """MuJoCo 服务 - 管理渲染器和状态"""

    def __init__(self):
        self._renderer = get_renderer()
        self._interval_ms = 50

    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    def set_arm_action(self, yaw: float, pitch: float, roll: float) -> None:
        """设置机械臂动作（仅设定力矩，由 game loop 统一步进）"""
        action = {
            "motor_yaw": yaw * 50,
            "motor_pitch": pitch * 50,
            "motor_roll": roll * 30,
        }
        for actuator_name, torque in action.items():
            try:
                actuator_id = mujoco.mj_name2id(
                    self._renderer._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                if actuator_id >= 0:
                    self._renderer._data.actuator_force[actuator_id] = torque
            except Exception:
                pass

    def set_car_action(self, vel_left: float, vel_right: float) -> None:
        """设置小车差速动作"""
        self._renderer.set_wheel_torques(vel_left, vel_right)

    def step(self) -> None:
        """单步模拟"""
        self._renderer.step()

    def update_topdown_camera(self, delta_azimuth: float, delta_elevation: float) -> None:
        """旋转俯视相机"""
        self._renderer.update_topdown_camera(delta_azimuth, delta_elevation)

    def update_topdown_distance(self, delta: float) -> None:
        """缩放俯视相机"""
        self._renderer.update_topdown_distance(delta)

    def render(self) -> tuple[str, str, dict]:
        """
        渲染双视角图像
        Returns: (topdown_b64, firstperson_b64, state)
        """
        topdown_img = self._renderer.get_topdown_image()
        firstperson_img = self._renderer.get_firstperson_image()

        topdown_b64 = self._image_to_b64(topdown_img)
        firstperson_b64 = self._image_to_b64(firstperson_img)
        state = self._renderer.get_state()

        return topdown_b64, firstperson_b64, state

    def _image_to_b64(self, img: np.ndarray) -> str:
        """numpy 图像转 base64"""
        if img.shape[2] == 3:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(img[:, :, :3])
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def close(self) -> None:
        """关闭渲染器"""
        global _renderer
        if _renderer:
            _renderer.close()
            _renderer = None