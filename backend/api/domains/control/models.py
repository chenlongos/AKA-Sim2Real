"""
AKA-Sim 后端 - 控制域 Pydantic 模型
"""

from pydantic import BaseModel


class ActionRequest(BaseModel):
    """动作请求"""
    action: str


class CarState(BaseModel):
    """车辆状态"""
    x: float = 400
    y: float = 300
    angle: float = -1.5708
    speed: float = 0
    maxSpeed: float = 5
    acceleration: float = 0.2
    friction: float = 0.95
    rotationSpeed: float = 0.05
