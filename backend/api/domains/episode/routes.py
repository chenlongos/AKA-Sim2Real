"""
AKA-Sim 后端 - Episode/数据采集域 API
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.api.domains.episode.models import CollectImagePayload, DatasetPayload
from backend.models import state
from backend.services.episode import EpisodeService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["episode"])
episode_service = EpisodeService()


@router.get("")
async def get_dataset():
    """获取数据集"""
    return {
        "samples": state.dataset_samples,
        "count": len(state.dataset_samples),
    }


@router.post("")
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
    except Exception as exc:
        logger.error(f"保存数据集失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("")
async def clear_dataset():
    """清空数据集"""
    state.dataset_samples = []
    return {
        "success": True,
        "message": "数据集已清空",
    }


@router.post("/collect")
async def collect_image(payload: CollectImagePayload):
    """将前端直接采集到的图像写入当前 episode。"""
    try:
        count = await episode_service.collect_data(
            payload.image,
            payload.actions,
            timestamp=payload.timestamp,
            state_payload=payload.state,
        )
        if count is None:
            raise HTTPException(status_code=409, detail="当前未处于录制状态")

        return {
            "success": True,
            "episode_id": state.current_episode_id,
            "count": count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"前端图像采集失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
