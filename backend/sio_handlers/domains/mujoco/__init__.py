from __future__ import annotations

import logging

from backend.services.mujoco_renderer.service import get_mujoco_service

logger = logging.getLogger(__name__)


class MujocoEventsMixin:
    """MuJoCo 相关 Socket.IO 事件处理"""

    async def on_connect(self, sid: str, environ: dict, auth: dict | None = None):
        """客户端连接"""
        self.runtime.connected_clients.add(sid)
        logger.info(f"[mujoco] 客户端连接: {sid}")
        await self.emit("connected", {"sid": sid})

    async def on_disconnect(self, sid: str):
        """客户端断开"""
        self.runtime.connected_clients.discard(sid)
        logger.info(f"[mujoco] 客户端断开: {sid}")

    async def on_mujoco_action(self, sid: str, data: dict):
        """
        处理 MuJoCo 机械臂控制动作
        data: { yaw: float, pitch: float, roll: float }
        """
        logger.info(f"[mujoco_action] sid={sid}, data={data}")
        service = get_mujoco_service()
        yaw = data.get("yaw", 0)
        pitch = data.get("pitch", 0)
        roll = data.get("roll", 0)
        service.set_arm_action(yaw, pitch, roll)

    async def on_mujoco_car_action(self, sid: str, data: dict):
        """
        处理 MuJoCo 小车差速控制动作
        data: { vel_left: float, vel_right: float }
        """
        logger.info(f"[mujoco_car_action] sid={sid}, data={data}")
        service = get_mujoco_service()
        vel_left = data.get("vel_left", 0)
        vel_right = data.get("vel_right", 0)
        service.set_car_action(vel_left, vel_right)

    async def on_mujoco_camera_move(self, sid: str, data: dict):
        """
        处理鼠标拖拽旋转俯视相机
        data: { delta_azimuth: float, delta_elevation: float }
        """
        service = get_mujoco_service()
        delta_azimuth = data.get("delta_azimuth", 0)
        delta_elevation = data.get("delta_elevation", 0)
        service.update_topdown_camera(delta_azimuth, delta_elevation)

    async def on_mujoco_camera_zoom(self, sid: str, data: dict):
        """
        处理鼠标滚轮缩放
        data: { delta: float }
        """
        service = get_mujoco_service()
        service.update_topdown_distance(data.get("delta", 0))

    async def on_get_mujoco_state(self, sid: str):
        """请求当前 MuJoCo 状态和图像"""
        service = get_mujoco_service()
        topdown, firstperson, state = service.render()
        serializable_state = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in state.items()
        }
        await self.emit("mujoco_state_update", {
            "topdown": topdown,
            "firstperson": firstperson,
            "state": serializable_state,
        })