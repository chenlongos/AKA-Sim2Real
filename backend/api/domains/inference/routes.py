"""
AKA-Sim 后端 - 推理域 API
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from backend.services import inference as inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/act", tags=["inference"])

_act_runtime = inference_service.get_act_runtime()


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime


@router.post("/load_trained")
async def load_trained_model(
    data_dir: str = Query(default=None),
    model_path: str = Query(default=None),
):
    """加载训练好的 ACT 模型"""
    try:
        logger.info(f"收到加载模型请求: model_path={model_path}, data_dir={data_dir}")
        _act_runtime.load_model(model_path, stats_dir=data_dir)
        stats = _act_runtime.get_stats()
        return {
            "success": True,
            "message": "模型加载成功",
            "stats": {
                "state_min": stats.state_min.tolist() if hasattr(stats.state_min, 'tolist') else list(stats.state_min),
                "state_max": stats.state_max.tolist() if hasattr(stats.state_max, 'tolist') else list(stats.state_max),
            } if stats else None,
        }
    except Exception as exc:
        logger.error(f"加载模型失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
