"""
AKA-Sim 后端 - Pydantic 模型
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class ActionRequest(BaseModel):
    """动作请求"""
    action: str


class DatasetPayload(BaseModel):
    """数据集载荷"""
    observation: Dict[str, Any]
    action: Dict[str, Any]


class ACTInferenceRequest(BaseModel):
    """ACT 推理请求"""
    image: Optional[str] = None  # base64 编码的图像
    state: list  # 状态向量


class CarState(BaseModel):
    """车辆状态"""
    x: float = 400
    y: float = 300
    angle: float = -1.5708  # -90度
    speed: float = 0
    maxSpeed: float = 5
    acceleration: float = 0.2
    friction: float = 0.95
    rotationSpeed: float = 0.05


class ACTInferenceResponse(BaseModel):
    """ACT 推理响应"""
    success: bool
    action: Optional[list] = None
    error: Optional[str] = None
