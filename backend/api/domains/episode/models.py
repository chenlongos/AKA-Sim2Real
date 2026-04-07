"""
AKA-Sim 后端 - Episode/数据采集域 Pydantic 模型
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class DatasetPayload(BaseModel):
    """数据集载荷"""
    observation: Dict[str, Any]
    action: Dict[str, Any]


class CollectImagePayload(BaseModel):
    """前端直接采集图像到当前 episode。"""

    image: str = Field(..., min_length=1)
    actions: list[str] = Field(default_factory=list)
    timestamp: int | None = None
    state: Dict[str, Any] | None = None
