"""
AKA-Sim 后端 - 推理域 API
"""

import logging
from pathlib import Path

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
    user_id: str = Query(...),
    data_dir: str = Query(default=None),
    model_path: str = Query(default=None),
):
    """加载训练好的 ACT 模型"""
    try:
        project_root = Path(__file__).resolve().parents[4]

        # 如果没有指定data_dir，使用用户的数据目录
        if data_dir is None:
            data_dir = str(project_root / "output" / "dataset" / user_id)

        # 如果没有指定model_path，尝试使用用户训练的模型
        if model_path is None:
            user_model_path = project_root / "output" / "train" / user_id
            if (user_model_path / "final_model.pt").exists():
                model_path = str(user_model_path / "final_model.pt")
            elif (user_model_path / "model.pt").exists():
                model_path = str(user_model_path / "model.pt")

        logger.info(f"收到加载模型请求: user={user_id}, model_path={model_path}, data_dir={data_dir}")
        _act_runtime.load_model(model_path, stats_dir=data_dir)
        return {
            "success": True,
            "message": "模型加载成功",
        }
    except Exception as exc:
        logger.error(f"加载模型失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
