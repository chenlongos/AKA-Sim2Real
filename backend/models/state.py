"""
AKA-Sim 后端 - 全局状态

Episode 管理：支持多轮采集，按用户隔离
"""

from typing import TYPE_CHECKING, Dict, Any, List, Optional

if TYPE_CHECKING:
    from policies.models.act.modeling_act import ACTModel

# ACT 模型 - 实际存储在 act_model 模块中，这里仅作为便捷引用
# act_model: Optional["ACTModel"] = None
# model_device = "cuda"

# ============ 用户隔离的 Episode 状态 ============
# 格式: { user_id: { episode_id: {"samples": [], "start_idx": 0, "task_name": "default"} } }
user_episode_buffers: Dict[str, Dict[int, Dict[str, Any]]] = {}

# 用户 episode 的帧索引
# 格式: { user_id: { episode_id: frame_index } }
user_episode_frame_index: Dict[str, Dict[int, int]] = {}

# 用户 episode 元数据缓存
# 格式: { user_id: { episode_id: metadata } }
user_episode_metadata: Dict[str, Dict[int, Dict[str, Any]]] = {}

# 当前用户的当前 episode ID
# 格式: { user_id: episode_id }
user_current_episode_id: Dict[str, int] = {}

# 全局数据集存储 - 保留用于向后兼容
# 格式: { episode_id: [samples] }
episode_samples: dict = {}

# 数据集样本（用于 /api/dataset 端点）
dataset_samples: list = []

# 当前是否正在录制（按用户）
user_is_recording: Dict[str, bool] = {}

# 当前 episode ID（保留用于向后兼容）
current_episode_id: int = 1


def _ensure_user_exists(user_id: str):
    """确保用户状态存在"""
    if user_id not in user_episode_buffers:
        user_episode_buffers[user_id] = {}
    if user_id not in user_episode_frame_index:
        user_episode_frame_index[user_id] = {}
    if user_id not in user_episode_metadata:
        user_episode_metadata[user_id] = {}
    if user_id not in user_current_episode_id:
        user_current_episode_id[user_id] = 1
    if user_id not in user_is_recording:
        user_is_recording[user_id] = False


def start_episode(user_id: str, episode_id: int, task_name: str = "default") -> Dict[str, Any]:
    """
    开始新的 episode（用户隔离）

    Args:
        user_id: 用户ID
        episode_id: episode ID
        task_name: 任务名称

    Returns:
        episode 元数据
    """
    global current_episode_id

    _ensure_user_exists(user_id)

    user_episode_buffers[user_id][episode_id] = {
        "samples": [],
        "start_idx": 0,
        "task_name": task_name,
    }
    user_episode_frame_index[user_id][episode_id] = 0
    user_current_episode_id[user_id] = episode_id
    user_is_recording[user_id] = True

    # 向后兼容：同时更新全局状态
    current_episode_id = episode_id

    return user_episode_buffers[user_id][episode_id]


def add_frame_to_episode(user_id: str, episode_id: int, sample: Dict[str, Any]) -> int:
    """
    添加帧到 episode buffer（用户隔离）

    Args:
        user_id: 用户ID
        episode_id: episode ID
        sample: 样本数据 (包含 image, state, actions)

    Returns:
        当前帧索引
    """
    _ensure_user_exists(user_id)

    if episode_id not in user_episode_buffers[user_id]:
        start_episode(user_id, episode_id)

    frame_idx = user_episode_frame_index[user_id][episode_id]
    user_episode_buffers[user_id][episode_id]["samples"].append(sample)
    user_episode_frame_index[user_id][episode_id] += 1

    return frame_idx


def end_episode(user_id: str, episode_id: int) -> List[Dict[str, Any]]:
    """
    结束 episode（用户隔离）

    Args:
        user_id: 用户ID
        episode_id: episode ID

    Returns:
        episode 的所有样本
    """
    _ensure_user_exists(user_id)

    samples = []
    if episode_id in user_episode_buffers[user_id]:
        samples = user_episode_buffers[user_id][episode_id]["samples"]
        user_episode_metadata[user_id][episode_id] = {
            "episode_index": episode_id,
            "task_name": user_episode_buffers[user_id][episode_id]["task_name"],
            "num_frames": len(samples),
        }

    user_is_recording[user_id] = False
    return samples


def get_episode_samples(user_id: str, episode_id: int) -> List[Dict[str, Any]]:
    """获取指定用户指定 episode 的所有样本"""
    if user_id in user_episode_buffers and episode_id in user_episode_buffers[user_id]:
        return user_episode_buffers[user_id][episode_id]["samples"]
    return episode_samples.get(episode_id, [])


def clear_episode_buffer(user_id: str, episode_id: Optional[int] = None):
    """清除用户 episode buffer"""
    _ensure_user_exists(user_id)

    if episode_id is not None:
        if episode_id in user_episode_buffers.get(user_id, {}):
            del user_episode_buffers[user_id][episode_id]
        if episode_id in user_episode_frame_index.get(user_id, {}):
            del user_episode_frame_index[user_id][episode_id]
    else:
        user_episode_buffers[user_id].clear()
        user_episode_frame_index[user_id].clear()
        user_episode_metadata[user_id].clear()


def get_current_buffer_size(user_id: str, episode_id: Optional[int] = None) -> int:
    """获取用户当前 episode buffer 的大小"""
    if episode_id is None:
        episode_id = user_current_episode_id.get(user_id, 1)
    return user_episode_frame_index.get(user_id, {}).get(episode_id, 0)


def is_user_recording(user_id: str) -> bool:
    """检查用户是否正在录制"""
    return user_is_recording.get(user_id, False)


def get_user_current_episode(user_id: str) -> int:
    """获取用户当前 episode ID"""
    return user_current_episode_id.get(user_id, 1)


# ============ 向后兼容的全局状态函数（不推荐使用） ============
# 这些函数保留用于兼容旧代码，新代码应该使用带 user_id 的版本

def start_episode_global(episode_id: int, task_name: str = "default") -> Dict[str, Any]:
    """向后兼容：使用 default 用户创建 episode"""
    return start_episode("default", episode_id, task_name)


def add_frame_to_episode_global(episode_id: int, sample: Dict[str, Any]) -> int:
    """向后兼容：使用 default 用户添加帧"""
    return add_frame_to_episode("default", episode_id, sample)


def end_episode_global(episode_id: int) -> List[Dict[str, Any]]:
    """向后兼容：使用 default 用户结束 episode"""
    return end_episode("default", episode_id)


def get_episode_samples_global(episode_id: int) -> List[Dict[str, Any]]:
    """向后兼容：获取 default 用户的 episode 样本"""
    return get_episode_samples("default", episode_id)


def clear_episode_buffer_global(episode_id: Optional[int] = None):
    """向后兼容：清除 default 用户的 buffer"""
    return clear_episode_buffer("default", episode_id)


def get_current_buffer_size_global(episode_id: int) -> int:
    """向后兼容：获取 default 用户的 buffer 大小"""
    return get_current_buffer_size("default", episode_id)


# 向后兼容的全局 episode_buffer 和 episode_frame_index（只用于读取）
@property
def episode_buffer() -> Dict[int, Dict[str, Any]]:
    """向后兼容：返回 default 用户的 buffer"""
    return user_episode_buffers.get("default", {})


@property
def episode_frame_index() -> Dict[int, int]:
    """向后兼容：返回 default 用户的 frame index"""
    return user_episode_frame_index.get("default", {})


@property
def episode_metadata() -> Dict[int, Dict[str, Any]]:
    """向后兼容：返回 default 用户的 metadata"""
    return user_episode_metadata.get("default", {})
