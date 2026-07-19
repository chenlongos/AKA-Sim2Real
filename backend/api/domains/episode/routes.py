"""
AKA-Sim 后端 - Episode/数据采集域 API
"""

import json
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
async def list_dataset_dirs(user_id: str):
    """列出用户下的所有数据集目录"""
    project_root = Path(__file__).resolve().parents[4]
    user_dataset_path = project_root / "output" / "dataset" / user_id

    if not user_dataset_path.exists():
        return {"datasets": []}

    datasets = []
    for item in user_dataset_path.iterdir():
        if item.is_dir() and ((item / "data").exists() or (item / "meta").exists()):
            info_path = item / "meta" / "info.json"
            sort_time = info_path.stat().st_mtime if info_path.exists() else item.stat().st_mtime
            datasets.append((item.name, sort_time))

    datasets.sort(key=lambda entry: entry[1], reverse=True)
    return {"datasets": [name for name, _ in datasets]}


@router.get("/models")
async def list_models(user_id: str, dataset_name: str = "default"):
    """列出用户下的所有训练模型（返回文件夹名）"""
    project_root = Path(__file__).resolve().parents[4]
    # 训练输出在 output/train/{user_id}/ 下
    train_path = project_root / "output" / "train" / user_id

    if not train_path.exists():
        return {"models": []}

    if dataset_name:
        dataset_model_path = train_path / dataset_name
        has_model = (dataset_model_path / "model.pt").exists() or (dataset_model_path / "final_model.pt").exists()
        if dataset_model_path.is_dir() and has_model:
            return {"models": [dataset_name]}
        return {"models": []}

    # 返回子文件夹名称
    models = []
    for item in train_path.iterdir():
        has_model = (item / "model.pt").exists() or (item / "final_model.pt").exists()
        if item.is_dir() and has_model:
            models.append(item.name)

    return {"models": sorted(models)}


@router.get("/info")
async def get_dataset_info(user_id: str, dataset_name: str):
    """读取数据集元信息。"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not dataset_name or ".." in dataset_name or "/" in dataset_name or "\\" in dataset_name:
        raise HTTPException(status_code=400, detail="invalid dataset_name")

    project_root = Path(__file__).resolve().parents[4]
    info_path = project_root / "output" / "dataset" / user_id / dataset_name / "meta" / "info.json"
    if not info_path.exists():
        return {
            "dataset_name": dataset_name,
            "total_frames": 0,
            "total_episodes": 0,
            "exists": False,
        }

    with open(info_path, "r") as f:
        info = json.load(f)

    return {
        "dataset_name": dataset_name,
        "total_frames": int(info.get("total_frames", 0) or 0),
        "total_episodes": int(info.get("total_episodes", 0) or 0),
        "exists": True,
    }


@router.post("/collect")
async def collect_image(payload: CollectImagePayload):
    """将前端直接采集到的图像写入当前 episode。"""
    user_id = payload.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        count = await episode_service.collect_data(
            payload.image,
            user_id=user_id,
            dataset_name=payload.dataset_name,
            timestamp=payload.timestamp,
            state_payload=payload.state,
            action_payload=payload.action,
        )
        if count is None:
            raise HTTPException(status_code=409, detail="当前未处于录制状态")

        return {
            "success": True,
            "episode_id": state.user_current_episode_id.get(user_id, 1),
            "count": count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"前端图像采集失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
