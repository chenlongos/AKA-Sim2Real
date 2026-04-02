"""
AKA-Sim 后端 - 控制域 API
"""

from fastapi import APIRouter

from backend.services.gateways import car_gateway

router = APIRouter(prefix="/api/car", tags=["control"])


@router.post("/heartbeat")
async def car_heartbeat(car_ip: str):
    """代理心跳请求到小车"""
    try:
        return await car_gateway.heartbeat(car_ip)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/control")
async def car_control(car_ip: str, action: str, speed: int = 50):
    """代理控制请求到小车"""
    try:
        return await car_gateway.control(car_ip, action, speed)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/motor_status")
async def car_snapshot(car_ip: str, timestamp: int):
    """从真实小车请求指定时间戳的电机状态。"""
    try:
        return await car_gateway.get_motor_status(car_ip, timestamp)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
