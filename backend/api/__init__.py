"""
AKA-Sim 后端 - REST API 端点
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Body

from backend.services import act_model as act_model_module, training
from backend.api.models import ACTInferenceRequest, DatasetPayload, TrainRequest
from backend.models import state

logger = logging.getLogger(__name__)

router = APIRouter()
_act_runtime = act_model_module.get_act_runtime()

# 存储sio_server实例
_sio_server = None


def set_sio_server(sio):
    """设置Socket.IO服务器实例"""
    global _sio_server
    _sio_server = sio


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime


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


@router.post("/api/dataset")
async def save_dataset(payload: DatasetPayload):
    """保存数据集样本"""
    try:
        state.dataset_samples.append({
            "observation": payload.observation,
            "action": payload.action,
        })

        logger.info(f"保存数据集样本，当前共 {len(state.dataset_samples)} 个样本")

        return {
            "success": True,
            "samples_count": len(state.dataset_samples),
        }
    except Exception as e:
        logger.error(f"保存数据集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/dataset")
async def get_dataset():
    """获取数据集"""
    return {
        "samples": state.dataset_samples,
        "count": len(state.dataset_samples),
    }


@router.delete("/api/dataset")
async def clear_dataset():
    """清空数据集"""
    state.dataset_samples = []
    return {
        "success": True,
        "message": "数据集已清空",
    }


@router.post("/api/act/infer")
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


@router.post("/api/act/load_trained")
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

@router.post("/api/train")
async def start_training(request: TrainRequest):
    """启动训练

    Args:
        data_dir: 数据集目录
        output_dir: 模型输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        resume_from: 从已有模型文件继续训练，None表示从头训练
    """
    try:
        data_dir = request.data_dir
        output_dir = request.output_dir
        epochs = request.epochs
        batch_size = request.batch_size
        lr = request.lr
        resume_from = request.resume_from

        if training.training_state["is_running"]:
            return {
                "success": False,
                "message": "训练正在进行中",
            }

        # 检查数据集是否存在 - 使用项目根目录
        import os
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / data_dir
        output_dir = project_root / output_dir

        if not data_path.exists():
            return {
                "success": False,
                "message": f"数据集目录不存在: {data_path}",
            }

        # 转换为绝对路径
        data_dir = str(data_dir)
        output_dir = str(output_dir)

        # 处理 resume_from 路径
        resume_path = None
        if resume_from:
            resume_path = str(project_root / resume_from)

        # 异步启动训练
        asyncio.create_task(
            training.train_model(
                _sio_server,
                data_dir=data_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                resume_from=resume_path,
            )
        )

        resume_msg = f"，从模型继续: {resume_from}" if resume_from else ""
        return {
            "success": True,
            "message": f"训练已启动{resume_msg}",
        }
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/train/status")
async def get_training_status():
    """获取训练状态"""
    return training.get_training_state()


@router.post("/api/train/stop")
async def stop_training():
    """停止训练"""
    training.training_state["is_running"] = False
    return {
        "success": True,
        "message": "训练已停止",
    }


@router.post("/api/car/heartbeat")
async def car_heartbeat(car_ip: str):
    """代理心跳请求到小车"""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"http://{car_ip}/heartbeat")
            if res.status_code != 200:
                return {"ok": False, "status": res.status_code}
            else:
                return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/car/control")
async def car_control(car_ip: str, action: str, speed: int = 50):
    """代理控制请求到小车"""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"http://{car_ip}/api/control?action={action}&speed={speed}")
            return {"ok": True, "status": res.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}
