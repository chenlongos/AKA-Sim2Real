"""
AKA-Sim 后端 - 训练域 API
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.domains.training.models import TrainRequest
from backend.services import training

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/train", tags=["training"])

_sio_server = None


def set_sio_server(sio):
    global _sio_server
    _sio_server = sio


@router.post("")
async def start_training(request: TrainRequest):
    """启动训练。"""
    try:
        if training.training_state["is_running"]:
            logger.warning("训练启动失败: 训练正在进行中")
            return {
                "success": False,
                "message": "训练正在进行中",
            }

        project_root = Path(__file__).resolve().parents[4]
        data_path = project_root / request.data_dir
        output_path = project_root / request.output_dir
        if not data_path.exists():
            logger.error(f"训练启动失败: 数据集目录不存在: {data_path}")
            return {
                "success": False,
                "message": f"数据集目录不存在: {data_path}",
            }

        resume_path = None
        if request.resume_from:
            resume_path = str(project_root / request.resume_from)

        logger.info(f"收到开始训练请求: epochs={request.epochs}, batch_size={request.batch_size}, lr={request.lr}, resume_from={resume_path}")

        asyncio.create_task(
            training.train_model(
                _sio_server,
                data_dir=str(data_path),
                output_dir=str(output_path),
                epochs=request.epochs,
                batch_size=request.batch_size,
                lr=request.lr,
                resume_from=resume_path,
            )
        )

        resume_msg = f"，从模型继续: {request.resume_from}" if request.resume_from else ""
        return {
            "success": True,
            "message": f"训练已启动{resume_msg}",
        }
    except Exception as exc:
        logger.error(f"启动训练失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def get_training_status():
    """获取训练状态"""
    return training.get_training_state()


@router.post("/stop")
async def stop_training():
    """停止训练"""
    logger.info("收到停止训练请求")
    training.training_state["is_running"] = False
    logger.info("训练已停止")
    return {
        "success": True,
        "message": "训练已停止",
    }
