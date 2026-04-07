"""
AKA-Sim 后端 - REST API 端点
"""

import asyncio
import logging

from fastapi import APIRouter

from backend.services import inference as inference_service
from backend.api.domains import episode, inference, training

logger = logging.getLogger(__name__)

router = APIRouter()

# ACT runtime 需要在启动时传递进来
_act_runtime = inference_service.get_act_runtime()


def get_act_runtime():
    return _act_runtime


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime
    inference.set_act_runtime(runtime)


def set_sio_server(sio):
    """设置 Socket.IO 服务器实例，供 training API 使用"""
    training.set_sio_server(sio)


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
router.include_router(episode.router)
router.include_router(inference.router)
router.include_router(training.router)
