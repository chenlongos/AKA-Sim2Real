"""
AKA-Sim 后端 - 数据浏览器 API
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/browser", tags=["browser"])


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[4]


# ============ 数据集相关 ============

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
            info_path = meta_path / "info.json"
            if info_path.exists():
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


@router.get("/dataset/{dataset_name}/content")
async def browse_dataset_content(user_id: str, dataset_name: str, path: str = "") -> dict:
    """浏览指定数据集下的内容（子文件夹）"""
    project_root = _get_project_root()
    dataset_path = project_root / "output" / "dataset" / user_id / dataset_name

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    # 安全检查：防止目录遍历
    if ".." in path:
        raise HTTPException(status_code=403, detail="无效路径")

    current_path = Path(path) if path else Path(".")
    full_path = dataset_path / current_path

    if not full_path.exists() or not full_path.is_dir():
        raise HTTPException(status_code=404, detail="路径不存在")

    items = []
    for item in sorted(full_path.iterdir()):
        items.append({
            "name": item.name,
            "path": str(item.relative_to(dataset_path)),
            "is_dir": item.is_dir(),
        })

    return {
        "dataset_name": dataset_name,
        "path": str(current_path),
        "children": items,
    }


# ============ 模型相关 ============

@router.get("/model")
async def browse_models(user_id: str, dataset_name: str = "") -> List[dict]:
    """浏览用户的所有模型（按数据集分组）"""
    project_root = _get_project_root()
    train_path = project_root / "output" / "train" / user_id

    if not train_path.exists():
        return []

    # 按 dataset_name 过滤或全部返回
    result = []
    for item in sorted(train_path.iterdir()):
        if not item.is_dir():
            continue
        # 模型文件夹名格式: {dataset_name}___{model_name} 或直接是 {dataset_name}
        if "___" in item.name:
            parts = item.name.split("___", 1)
            ds_name = parts[0]
            model_name = parts[1]
            if dataset_name and ds_name != dataset_name:
                continue
            result.append({
                "name": model_name,
                "dataset": ds_name,
                "path": str(item.relative_to(project_root)),
            })
        else:
            # 兼容旧格式：直接以数据集名为文件夹名
            if dataset_name and item.name != dataset_name:
                continue
            result.append({
                "name": item.name,
                "dataset": item.name,
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

    train_root = project_root / "output" / "train"
    if not str(full_path.resolve()).startswith(str(train_root.resolve())):
        raise HTTPException(status_code=403, detail="无效路径")

    shutil.rmtree(full_path)
    logger.info(f"已删除模型文件夹: {full_path}")
    return {"success": True}


@router.delete("/dataset")
async def delete_dataset_folder(user_id: str, dataset_name: str):
    """删除指定数据集文件夹"""
    project_root = _get_project_root()
    full_path = project_root / "output" / "dataset" / user_id / dataset_name

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    dataset_root = project_root / "output" / "dataset" / user_id
    if not str(full_path.resolve()).startswith(str(dataset_root.resolve())):
        raise HTTPException(status_code=403, detail="无效路径")

    shutil.rmtree(full_path)
    logger.info(f"已删除数据集文件夹: {full_path}")
    return {"success": True}