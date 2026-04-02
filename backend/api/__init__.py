"""
AKA-Sim 后端 - REST API 端点
"""

import logging

from fastapi import APIRouter

from backend.services import act_model as act_model_module
from backend.api import car, dataset, train
from backend.api.act import router as act_router, set_act_runtime as act_set_runtime

logger = logging.getLogger(__name__)

router = APIRouter()

# ACT runtime 需要在启动时传递进来
_act_runtime = act_model_module.get_act_runtime()


def get_act_runtime():
    return _act_runtime


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime
    act_set_runtime(runtime)


def set_sio_server(sio):
    """设置 Socket.IO 服务器实例，供 train.py 使用"""
    from backend.api.train import set_sio_server as train_set_sio
    train_set_sio(sio)


# 健康检查
@router.get("/")
async def root():
    """根路径"""
    return {
        "name": "AKA-Sim Backend",
        "version": "1.0.0",
        "status": "running",
    }


@router.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": _act_runtime.is_model_loaded(),
    }


# 注册子路由
router.include_router(car.router)
router.include_router(dataset.router)
router.include_router(act_router)
router.include_router(train.router)
