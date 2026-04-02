"""
AKA-Sim 后端 - 小车控制 API
"""

import httpx
from fastapi import APIRouter

router = APIRouter(prefix="/api/car", tags=["car"])


@router.post("/heartbeat")
async def car_heartbeat(car_ip: str):
    """代理心跳请求到小车"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"http://{car_ip}/heartbeat")
            if res.status_code != 200:
                return {"ok": False, "status": res.status_code}
            else:
                return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/control")
async def car_control(car_ip: str, action: str, speed: int = 50):
    """代理控制请求到小车"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"http://{car_ip}/api/control?action={action}&speed={speed}")
            return {"ok": True, "status": res.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/motor_status")
async def car_snapshot(car_ip: str, timestamp: int):
    """从真实小车请求指定时间戳的图像数据

    转发请求到小车，返回其响应的图像数据
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                f"http://{car_ip}/api/motor_status",
                params={"timestamp": timestamp}
            )
            if res.status_code == 200:
                return res.json()
            else:
                return {"ok": False, "status": res.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}
