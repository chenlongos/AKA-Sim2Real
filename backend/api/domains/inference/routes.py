"""
AKA-Sim 后端 - 推理域 API
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.services import inference as inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/act", tags=["inference"])

_act_runtime = inference_service.get_act_runtime()


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime


@router.post("/load_trained")
async def load_trained_model(model_path: str = None):
    """加载训练好的 ACT 模型"""
    try:
        logger.info(f"收到加载模型请求: model_path={model_path}")
        _act_runtime.load_model(model_path)
        return {
            "success": True,
            "message": "模型加载成功",
        }
    except Exception as exc:
        logger.error(f"加载模型失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
