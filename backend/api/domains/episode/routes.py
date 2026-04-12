"""
AKA-Sim 后端 - Episode/数据采集域 API
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.api.domains.episode.models import CollectImagePayload
from backend.models import state
from backend.services.episode import EpisodeService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["episode"])
episode_service = EpisodeService()


@router.post("/collect")
async def collect_image(payload: CollectImagePayload):
    """将前端直接采集到的图像写入当前 episode。"""
    try:
        count = await episode_service.collect_data(
            payload.image,
            timestamp=payload.timestamp,
            state_payload=payload.state,
            action_payload=payload.action,
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
