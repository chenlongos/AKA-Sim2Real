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
    """从真实小车直接请求一次电机状态。"""
    try:
        payload = await car_gateway.get_motor_status(car_ip, timestamp)
        if not isinstance(payload, dict) or payload.get("ok") is False:
            return {"ok": False, "error": "无法获取小车状态", "payload": payload}

        matched_timestamp_ms = car_gateway.extract_timestamp(payload) or timestamp
        wheel_velocity = car_gateway.extract_wheel_velocity(payload)
        if wheel_velocity is None:
            return {"ok": False, "error": "无法从小车状态中解析左右轮速度"}

        vel_left, vel_right = wheel_velocity

        return {
            "ok": True,
            "query_timestamp_ms": timestamp,
            "matched_timestamp_ms": matched_timestamp_ms,
            "delta_ms": matched_timestamp_ms - timestamp,
            "source": "direct_request",
            "state": {
                "vel_left": vel_left,
                "vel_right": vel_right,
            },
            "payload": payload,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/motor_direct")
async def motor_direct(car_ip: str, left: int, right: int, duration: float = 0):
    """直接设置左右轮速度（带持续时间）

    Query params:
        car_ip: str  小车 IP 地址
        left: int    左轮速度 (-255 ~ 255)
        right: int   右轮速度 (-255 ~ 255)
        duration: float 持续时间（秒），0 表示无限
    """
    try:
        return await car_gateway.motor_direct(car_ip, left, right, duration)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
