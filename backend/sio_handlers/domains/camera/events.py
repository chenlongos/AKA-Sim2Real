from __future__ import annotations


class CameraEventsMixin:
    async def on_start_camera(self, sid: str):
        self.camera_service.start(sid)

    async def on_stop_camera(self, sid: str):
        self.camera_service.stop(sid)
