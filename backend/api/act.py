"""
AKA-Sim 后端 - ACT 推理 API
"""

from fastapi import APIRouter, HTTPException

from backend.api.models import ACTInferenceRequest
from backend.services import act_model as act_model_module

router = APIRouter(prefix="/api/act", tags=["act"])

_act_runtime = act_model_module.get_act_runtime()


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime


@router.post("/infer")
async def infer_act(request: ACTInferenceRequest):
    """ACT 模型推理 API"""
    try:
        action = _act_runtime.infer(request.state, request.image)
        return {
            "success": True,
            "action": action,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load_trained")
async def load_trained_model(model_path: str = None):
    """加载训练好的ACT模型"""
    try:
        _act_runtime.load_model(model_path)
        return {
            "success": True,
            "message": "模型加载成功",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
