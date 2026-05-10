"""
AKA-Sim 后端 - Episode/数据采集域 API
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.domains.episode.models import CollectImagePayload
from backend.models import state
from backend.services.episode import EpisodeService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["episode"])
episode_service = EpisodeService()


@router.get("/dirs")
async def list_dataset_dirs(user_id: str = "default"):
    """列出用户下的所有数据集目录"""
    project_root = Path(__file__).resolve().parents[4]
    user_dataset_path = project_root / "output" / "dataset" / user_id

    if not user_dataset_path.exists():
        return {"datasets": []}

    datasets = []
    for item in user_dataset_path.iterdir():
        if item.is_dir() and (item / "data").exists() or (item / "meta").exists():
            datasets.append(item.name)

    return {"datasets": sorted(datasets)}


@router.get("/models")
async def list_models(user_id: str = "default", dataset_name: str = "default"):
    """列出用户下的所有训练模型（返回文件夹名）"""
    project_root = Path(__file__).resolve().parents[4]
    # 训练输出在 output/train/{user_id}/ 下
    train_path = project_root / "output" / "train" / user_id

    if not train_path.exists():
        return {"models": []}

    # 返回子文件夹名称
    models = []
    for item in train_path.iterdir():
        if item.is_dir():
            models.append(item.name)

    return {"models": sorted(models)}


@router.post("/collect")
async def collect_image(payload: CollectImagePayload):
    """将前端直接采集到的图像写入当前 episode。"""
    try:
        count = await episode_service.collect_data(
            payload.image,
            user_id=payload.user_id,
            dataset_name=payload.dataset_name,
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
