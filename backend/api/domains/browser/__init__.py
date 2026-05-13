"""
AKA-Sim 后端 - 数据浏览器 API
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/browser", tags=["browser"])


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[4]


# ============ 数据集相关 ============

class DatasetItem:
    """数据集项（文件夹）"""
    def __init__(self, name: str, path: str, episode_count: int = 0):
        self.name = name
        self.path = path
        self.episode_count = episode_count


class ModelItem:
    """模型项（文件夹）"""
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


@router.get("/dataset")
async def browse_datasets(user_id: str) -> List[dict]:
    """浏览用户的所有数据集（以文件夹形式）"""
    project_root = _get_project_root()
    user_dataset_path = project_root / "output" / "dataset" / user_id

    if not user_dataset_path.exists():
        return []

    result = []
    for item in sorted(user_dataset_path.iterdir()):
        if not item.is_dir():
            continue
        meta_path = item / "meta"
        episode_count = 0
        if meta_path.exists():
            # 尝试读取 meta/info.json 获取 episode 数
            info_path = meta_path / "info.json"
            if info_path.exists():
                import json
                try:
                    with open(info_path) as f:
                        info = json.load(f)
                        episode_count = info.get("total_episodes", 0)
                except Exception:
                    pass
        result.append({
            "name": item.name,
            "path": str(item.relative_to(project_root)),
            "episode_count": episode_count,
        })

    return result


@router.get("/dataset/{dataset_name}/model")
async def browse_models(user_id: str, dataset_name: str) -> List[dict]:
    """浏览指定数据集下的所有模型（以文件夹形式）"""
    project_root = _get_project_root()
    train_path = project_root / "output" / "train" / user_id

    if not train_path.exists():
        return []

    # 按 dataset_name 过滤
    result = []
    for item in sorted(train_path.iterdir()):
        if not item.is_dir():
            continue
        # 检查是否属于指定数据集
        # 模型文件夹名格式: {dataset_name}___{model_name}
        if "___" in item.name:
            parts = item.name.split("___", 1)
            if len(parts) == 2 and parts[0] == dataset_name:
                result.append({
                    "name": parts[1],
                    "path": str(item.relative_to(project_root)),
                })
        else:
            # 兼容旧格式：直接以模型名为文件夹名
            result.append({
                "name": item.name,
                "path": str(item.relative_to(project_root)),
            })

    return result


@router.delete("/model")
async def delete_model_folder(user_id: str, model_path: str):
    """删除指定模型文件夹"""
    project_root = _get_project_root()
    full_path = project_root / model_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="模型不存在")

    # 安全检查：确保在 output/train 目录下
    train_root = project_root / "output" / "train"
    if not str(full_path.resolve()).startswith(str(train_root.resolve())):
        raise HTTPException(status_code=403, detail="无效路径")

    shutil.rmtree(full_path)
    logger.info(f"已删除模型文件夹: {full_path}")
    return {"success": True}