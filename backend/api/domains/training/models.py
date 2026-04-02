"""
AKA-Sim 后端 - 训练域 Pydantic 模型
"""

from typing import List, Optional

from pydantic import BaseModel


class TrainRequest(BaseModel):
    """训练请求"""
    data_dir: str = "output/dataset"
    output_dir: Optional[str] = None
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    resume_from: Optional[str] = None
    episode_ids: Optional[List[int]] = None
