"""
AKA-Sim 后端 - Episode/数据采集域 Pydantic 模型
"""

from typing import Any, Dict

from pydantic import BaseModel


class DatasetPayload(BaseModel):
    """数据集载荷"""
    observation: Dict[str, Any]
    action: Dict[str, Any]
