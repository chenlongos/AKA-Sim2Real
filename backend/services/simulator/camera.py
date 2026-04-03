from __future__ import annotations

import asyncio
import logging

from backend.sio_handlers.core.runtime import SioRuntimeState

logger = logging.getLogger(__name__)


class CameraService:
    def __init__(self, runtime: SioRuntimeState):
        self.runtime = runtime

    def start(self, sid: str) -> None:
        self.runtime.camera_active = True
        logger.info(f"[摄像头] 已启动, sid={sid}")

    def stop(self, sid: str) -> None:
        self.runtime.camera_active = False
        logger.info(f"[摄像头] 已停止, sid={sid}")

    async def run_loop(self, sio_server, namespace: str = "/") -> None:
        import cv2

        logger.info(f"[摄像头] 任务已启动, namespace={namespace}")
        last_frame_time = 0.0
        capture = None

        while True:
            try:
                if not self.runtime.camera_active:
                    if capture is not None:
                        capture.release()
                        capture = None
                        logger.info("[摄像头] 资源已释放")
                    await asyncio.sleep(0.2)
                    continue

                current_time = asyncio.get_event_loop().time()
                if current_time - last_frame_time < 0.01:
                    await asyncio.sleep(0.01)
                    continue
                last_frame_time = current_time

                if capture is None or not capture.isOpened():
                    capture = cv2.VideoCapture(0)
                    if not capture.isOpened():
                        logger.warning("[摄像头] 无法打开摄像头")
                        await asyncio.sleep(1)
                        continue

                ok, frame = capture.read()
                if not ok:
                    logger.warning("[摄像头] 无法读取画面")
                    capture.release()
                    capture = None
                    await asyncio.sleep(0.5)
                    continue

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, img_encoded = cv2.imencode(".jpg", frame, encode_param)
                image_bytes = img_encoded.tobytes()

                # 给每个连接单独发送，而不是广播
                for sid in self.runtime.connected_clients:
                    try:
                        await sio_server.emit("camera_image", image_bytes, room=sid, namespace=namespace)
                    except Exception:
                        pass
            except Exception as exc:
                logger.error(f"[摄像头] 捕获失败: {exc}")
                if capture is not None:
                    capture.release()
                    capture = None
                await asyncio.sleep(1)
