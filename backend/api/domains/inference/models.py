"""
AKA-Sim 后端 - 推理域 Pydantic 模型
"""

from typing import Optional

from pydantic import BaseModel


class ACTInferenceRequest(BaseModel):
    """ACT 推理请求"""
    image: Optional[str] = None
    state: list


class ACTInferenceResponse(BaseModel):
    """ACT 推理响应"""
    success: bool
    action: Optional[list] = None
    error: Optional[str] = None
