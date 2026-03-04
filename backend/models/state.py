"""
AKA-Sim 后端 - 全局状态
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from config import config

if TYPE_CHECKING:
    from policies.models.act.modeling_act import ACTModel

# 车辆状态
car_state = {
    "x": 400,
    "y": 300,
    "angle": -3.14159 / 2,  # -90度
    "speed": 0,
    "maxSpeed": 5,
    "acceleration": 0.2,
    "friction": 0.98,
    "rotationSpeed": 0.05,
}

# ACT 模型 - 实际存储在 act_model 模块中，这里仅作为便捷引用
# act_model: Optional["ACTModel"] = None
# model_device = "cuda"

# 数据集存储 - 支持多轮采集
# 格式: { episode_id: [samples] }
episode_samples: dict = {}
current_episode_id: int = 1  # 当前采集的轮次

# Episode Buffer - 用于实时保存帧到磁盘
# 格式: { episode_id: {"samples": [], "start_idx": 0, "task_name": "default"}}
episode_buffer: Dict[int, Dict[str, Any]] = {}

# 当前 episode 的帧索引（用于实时保存）
episode_frame_index: Dict[int, int] = {}

# Episode 元数据缓存
episode_metadata: Dict[int, Dict[str, Any]] = {}

# 当前是否正在录制
is_recording: bool = False


def reset_car_state():
    """重置车辆状态"""
    car_state["x"] = 400
    car_state["y"] = 300
    car_state["angle"] = -3.14159 / 2
    car_state["speed"] = 0


def update_car_state(action: str):
    """更新车辆状态"""
    if action == "forward":
        car_state["speed"] = min(
            car_state["speed"] + car_state["acceleration"],
            car_state["maxSpeed"]
        )
    elif action == "backward":
        car_state["speed"] = max(
            car_state["speed"] - car_state["acceleration"],
            -car_state["maxSpeed"] / 2
        )
    elif action == "left":
        car_state["angle"] -= car_state["rotationSpeed"]
    elif action == "right":
        car_state["angle"] += car_state["rotationSpeed"]
    elif action == "stop":
        car_state["speed"] = 0

    # 更新位置
    import math
    car_state["x"] += math.cos(car_state["angle"]) * car_state["speed"]
    car_state["y"] += math.sin(car_state["angle"]) * car_state["speed"]

    # 边界检测
    car_state["x"] = max(0, min(config.MAP_WIDTH, car_state["x"]))
    car_state["y"] = max(0, min(config.MAP_HEIGHT, car_state["y"]))


def apply_friction():
    """应用摩擦力减速"""
    if car_state["speed"] > 0:
        car_state["speed"] = max(0, car_state["speed"] * car_state["friction"])
    elif car_state["speed"] < 0:
        car_state["speed"] = min(0, car_state["speed"] * car_state["friction"])

    # 摩擦力时也要更新位置（小车滑行）
    import math
    car_state["x"] += math.cos(car_state["angle"]) * car_state["speed"]
    car_state["y"] += math.sin(car_state["angle"]) * car_state["speed"]

    # 边界检测
    car_state["x"] = max(0, min(config.MAP_WIDTH, car_state["x"]))
    car_state["y"] = max(0, min(config.MAP_HEIGHT, car_state["y"]))


# Episode Buffer 管理函数

def start_episode(episode_id: int, task_name: str = "default") -> Dict[str, Any]:
    """
    开始新的 episode

    Args:
        episode_id: episode ID
        task_name: 任务名称

    Returns:
        episode 元数据
    """
    global is_recording, current_episode_id

    # 初始化 episode buffer
    episode_buffer[episode_id] = {
        "samples": [],
        "start_idx": 0,
        "task_name": task_name,
    }
    episode_frame_index[episode_id] = 0
    current_episode_id = episode_id
    is_recording = True

    return episode_buffer[episode_id]


def add_frame_to_episode(episode_id: int, sample: Dict[str, Any]) -> int:
    """
    添加帧到 episode buffer

    Args:
        episode_id: episode ID
        sample: 样本数据 (包含 image, state, actions)

    Returns:
        当前帧索引
    """
    if episode_id not in episode_buffer:
        start_episode(episode_id)

    frame_idx = episode_frame_index[episode_id]
    episode_buffer[episode_id]["samples"].append(sample)
    episode_frame_index[episode_id] += 1

    return frame_idx


def end_episode(episode_id: int) -> List[Dict[str, Any]]:
    """
    结束 episode

    Args:
        episode_id: episode ID

    Returns:
        episode 的所有样本
    """
    global is_recording

    samples = []
    if episode_id in episode_buffer:
        samples = episode_buffer[episode_id]["samples"]
        episode_metadata[episode_id] = {
            "episode_index": episode_id,
            "task_name": episode_buffer[episode_id]["task_name"],
            "num_frames": len(samples),
        }

    is_recording = False
    return samples


def get_episode_samples(episode_id: int) -> List[Dict[str, Any]]:
    """获取指定 episode 的所有样本"""
    if episode_id in episode_buffer:
        return episode_buffer[episode_id]["samples"]
    return episode_samples.get(episode_id, [])


def clear_episode_buffer(episode_id: int = None):
    """清除 episode buffer"""
    global episode_buffer, episode_frame_index, episode_metadata

    if episode_id is not None:
        if episode_id in episode_buffer:
            del episode_buffer[episode_id]
        if episode_id in episode_frame_index:
            del episode_frame_index[episode_id]
    else:
        episode_buffer.clear()
        episode_frame_index.clear()
        episode_metadata.clear()


def get_current_buffer_size(episode_id: int) -> int:
    """获取当前 episode buffer 的大小"""
    return episode_frame_index.get(episode_id, 0)
