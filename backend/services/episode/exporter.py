"""
AKA-Sim 数据导出模块 - LeRobot 风格数据采集
支持完整的 episode 元数据管理、任务管理、自动分块
"""

import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading

import numpy as np
import torch
import pandas as pd
from PIL import Image


# 默认配置
DEFAULT_CHUNK_SIZE = 1000  # 每个 parquet 文件的最大样本数
DEFAULT_CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100MB 基于大小的分块
ACTION_CHUNK_SIZE = 8  # 预测未来8帧的动作序列
ACTION_DIM = 2  # [pwm_left, pwm_right] 每帧2维
DEFAULT_FPS = 10

# 状态编码 (2维) - 左右轮速度（从 PWM 估计）
STATE_KEYS = ["vel_left", "vel_right"]  # 当前轮子速度
STATE_DIM = len(STATE_KEYS)

# 动作到连续值的映射（离散动作 -> 轮子速度）
# 速度单位：m/s，与 STATE 保持一致
ACTION_TO_CONTINUOUS = {
    "forward": [0.3, 0.3],      # 前进：左右轮同速
    "backward": [-0.3, -0.3],   # 后退：左右轮同速反转
    "left": [-0.2, 0.2],        # 左转：左轮后退，右轮前进
    "right": [0.2, -0.2],      # 右转：左轮前进，右轮后退
    "stop": [0.0, 0.0],         # 停止
}
# 确保所有动作都是 2 维 (ACTION_DIM=2)
assert all(len(v) == 2 for v in ACTION_TO_CONTINUOUS.values()), "动作必须是2维"

# 动作统计范围 (速度 m/s)
ACTION_MIN = [-0.5, -0.5]
ACTION_MAX = [0.5, 0.5]


class LeRobotDatasetMetadata:
    """
    LeRobot 风格数据集元数据管理类

    目录结构:
    output/dataset/
    ├── data/
    │   └── chunk-XXX/
    │       └── file-XXX.parquet
    ├── meta/
    │   ├── episodes/
    │   │   └── chunk-XXX/
    │   │       └── file-XXX.parquet    # 每个 episode 的索引信息
    │   ├── info.json                    # 数据集元信息
    │   ├── stats.json                   # 统计信息
    │   └── tasks.parquet                # 任务列表
    └── videos/
        └── observation.images.fpv/
            └── chunk-XXX/
                └── frame_XXXXXX.jpg
    """

    def __init__(
        self,
        output_dir: str = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        fps: int = DEFAULT_FPS,
        features: dict = None,
    ):
        if output_dir is None:
            output_dir = Path(__file__).resolve().parents[3] / "output" / "dataset"
        self.output_dir = Path(output_dir)

        self.data_dir = self.output_dir / "data"
        self.videos_dir = self.output_dir / "videos" / "observation.images.fpv"
        self.meta_dir = self.output_dir / "meta"
        self.episodes_dir = self.meta_dir / "episodes"

        self.chunk_size = chunk_size
        self.fps = fps

        # 默认特征定义
        self.features = features or {
            "observation": {
                "image": {"dtype": "jpeg", "shape": [240, 320, 3]},
                "state": {"dtype": "float32", "shape": [STATE_DIM]},
            },
            "action": {"dtype": "float32", "shape": [ACTION_CHUNK_SIZE, ACTION_DIM]},
        }

        # 元数据缓存
        self._info = None
        self._stats = None
        self._tasks = []
        self._episodes = []
        self._metadata_buffer = []
        self._buffer_lock = threading.Lock()
        self._stats_accumulators = {}

        # 当前写入状态
        self._total_samples = 0
        self._total_episodes = 0

        # 加载已有元数据
        self._load_existing_metadata()

    def _load_existing_metadata(self):
        """加载已有元数据（如果存在）"""
        info_path = self.meta_dir / "info.json"
        stats_path = self.meta_dir / "stats.json"

        if info_path.exists():
            with open(info_path, "r") as f:
                self._info = json.load(f)
            self._total_samples = self._info.get("total_frames", 0)
            self._total_episodes = self._info.get("total_episodes", 0)

        if stats_path.exists():
            with open(stats_path, "r") as f:
                self._stats = json.load(f)
            self._load_stats_accumulators()

        # 加载已有 episodes
        if self.episodes_dir.exists():
            for parquet_file in sorted(self.episodes_dir.glob("**/*.parquet")):
                df = pd.read_parquet(parquet_file)
                for _, row in df.iterrows():
                    self._episodes.append(row.to_dict())

    @property
    def info(self):
        if self._info is None:
            self._info = {
                "version": "1.0",
                "dataset_name": "aka-sim",
                "fps": self.fps,
                "total_frames": self._total_samples,
                "total_episodes": self._total_episodes,
                "chunk_size": self.chunk_size,
                "features": self.features,
            }
        return self._info

    @property
    def stats(self):
        if self._stats is None:
            self._stats = {
                "observation.state": {
                    "min": [],
                    "max": [],
                    "mean": [],
                    "std": [],
                }
            }
        return self._stats

    def _load_stats_accumulators(self):
        """从已有 stats.json 反序列化统计累积量。"""
        self._stats_accumulators = {}
        for key in ("observation.state", "action"):
            entry = self._stats.get(key)
            if not entry:
                continue
            if all(acc_key in entry for acc_key in ("count", "sum", "sumsq", "min", "max")):
                self._stats_accumulators[key] = {
                    "count": int(entry["count"]),
                    "sum": np.array(entry["sum"], dtype=np.float64),
                    "sumsq": np.array(entry["sumsq"], dtype=np.float64),
                    "min": np.array(entry["min"], dtype=np.float64),
                    "max": np.array(entry["max"], dtype=np.float64),
                }

    def _build_stats_entry(self, key: str, values: np.ndarray) -> Dict[str, Any]:
        """根据 batch 数据更新统计累积量并生成 stats.json 条目。"""
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(-1, values.shape[-1])

        batch_count = int(values.shape[0])
        batch_sum = values.sum(axis=0, dtype=np.float64)
        batch_sumsq = np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        batch_min = values.min(axis=0).astype(np.float64)
        batch_max = values.max(axis=0).astype(np.float64)

        existing = self._stats_accumulators.get(key)
        if existing is None:
            accum = {
                "count": batch_count,
                "sum": batch_sum,
                "sumsq": batch_sumsq,
                "min": batch_min,
                "max": batch_max,
            }
        else:
            accum = {
                "count": existing["count"] + batch_count,
                "sum": existing["sum"] + batch_sum,
                "sumsq": existing["sumsq"] + batch_sumsq,
                "min": np.minimum(existing["min"], batch_min),
                "max": np.maximum(existing["max"], batch_max),
            }

        self._stats_accumulators[key] = accum

        mean = accum["sum"] / max(accum["count"], 1)
        variance = np.maximum(accum["sumsq"] / max(accum["count"], 1) - np.square(mean), 0.0)
        std = np.sqrt(variance) + 1e-6

        return {
            "count": accum["count"],
            "sum": accum["sum"].tolist(),
            "sumsq": accum["sumsq"].tolist(),
            "min": accum["min"].tolist(),
            "max": accum["max"].tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    def _ensure_dirs(self):
        """确保目录结构存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def save_episode(
        self,
        samples: List[Dict[str, Any]],
        episode_id: int,
        task_name: str = "default",
    ) -> Dict[str, Any]:
        """
        保存 episode 到磁盘

        Args:
            samples: 该 episode 的样本列表
            episode_id: episode ID
            task_name: 任务名称

        Returns:
            episode 元数据
        """
        if not samples:
            return {}

        self._ensure_dirs()

        episode_start_idx = self._total_samples
        num_frames = len(samples)

        # 收集状态统计
        all_states = []
        all_actions = []

        # 处理每个样本
        for i, sample in enumerate(samples):
            # 保存图像
            image_data = sample.get("image", "")
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)

            # 根据总索引计算 chunk
            # 使用全局递增的帧索引
            global_frame_idx = episode_start_idx + i
            chunk_idx = global_frame_idx // self.chunk_size
            file_idx = global_frame_idx % self.chunk_size

            image_filename = f"frame_{file_idx:06d}.jpg"
            chunk_dir = self.videos_dir / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(exist_ok=True)
            image_path = chunk_dir / image_filename

            # 如果图像已存在则跳过（避免覆盖）
            if not image_path.exists():
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

            # 收集状态
            state_values = [sample.get("state", {}).get(k, 0) for k in STATE_KEYS]
            all_states.append(state_values)

            # 收集动作（取最后一个动作转为连续值）
            sample_actions = sample.get("actions", [])
            if sample_actions:
                last_action = sample_actions[-1]
                action_values = ACTION_TO_CONTINUOUS.get(last_action, [0.0, 0.0])
            else:
                action_values = [0.0, 0.0]  # 默认停止
            all_actions.append(action_values)

        states_array = np.array(all_states, dtype=np.float32)

        # 将动作扩展为未来 chunk 帧的序列
        # 每帧的动作重复 ACTION_CHUNK_SIZE 次，形成序列
        actions_expanded = []
        for i in range(len(all_actions)):
            chunk_actions = []
            for j in range(ACTION_CHUNK_SIZE):
                future_idx = min(i + j, len(all_actions) - 1)
                chunk_actions.append(all_actions[future_idx])
            actions_expanded.append(chunk_actions)

        # 动作数据：[num_samples, chunk_size, action_dim]
        actions_array = np.array(actions_expanded, dtype=np.float32)

        # 写入数据 parquet
        for chunk_idx in range((episode_start_idx // self.chunk_size), ((self._total_samples + num_frames) // self.chunk_size) + 1):
            start_idx = max(episode_start_idx, chunk_idx * self.chunk_size)
            end_idx = min(self._total_samples + num_frames, (chunk_idx + 1) * self.chunk_size)

            if end_idx <= episode_start_idx or start_idx >= episode_start_idx + num_frames:
                continue

            chunk_start = max(start_idx, episode_start_idx)
            chunk_end = min(end_idx, episode_start_idx + num_frames)

            # 构建该 chunk 的数据
            local_start = chunk_start - episode_start_idx
            local_end = chunk_end - episode_start_idx

            # 动作数据已经是 2 维连续值
            actions_list = actions_array[local_start:local_end].tolist()

            chunk_data = {
                "observation.image": [
                    f"videos/observation.images.fpv/chunk-{chunk_idx:03d}/frame_{i - (chunk_idx * self.chunk_size):06d}.jpg"
                    for i in range(chunk_start, chunk_end)
                ],
                "observation.state": states_array[local_start:local_end].tolist(),
                "action": actions_list,
            }

            df = pd.DataFrame(chunk_data)
            chunk_file_dir = self.data_dir / f"chunk-{chunk_idx:03d}"
            chunk_file_dir.mkdir(exist_ok=True)

            # 每 100 个样本一个文件
            # 使用全局帧索引计算文件编号，避免不同 episode 覆盖
            global_start = start_idx  # 全局起始帧索引
            for local_idx in range(0, end_idx - start_idx, 100):
                sub_end = min(local_idx + 100, end_idx - start_idx)
                sub_df = df.iloc[local_idx:sub_end]
                # 使用全局索引：(global_start + local_idx) // 100
                global_file_idx = (global_start + local_idx) // 100
                sub_file = chunk_file_dir / f"file-{global_file_idx:03d}.parquet"

                # 追加模式：如果文件已存在，则读取并合并
                if sub_file.exists():
                    existing_df = pd.read_parquet(sub_file)
                    combined_df = pd.concat([existing_df, sub_df], ignore_index=True)
                    combined_df.to_parquet(sub_file, index=False)
                else:
                    sub_df.to_parquet(sub_file, index=False)

        # 更新统计
        self._update_stats(states_array, actions_array)

        # 创建 episode 元数据
        episode_meta = {
            "episode_index": episode_id,
            "task_name": task_name,
            "start_frame_index": episode_start_idx,
            "num_frames": num_frames,
            "chunk_index": episode_start_idx // self.chunk_size,
            "start_timestamp": time.time() - (num_frames / self.fps),
            "duration": num_frames / self.fps,
        }

        # 保存 episode 元数据
        self._save_episode_metadata(episode_meta)

        # 更新 info
        self._total_samples += num_frames
        self._total_episodes += 1
        self._info = None  # 标记需要重新计算

        # 写入元数据
        self._save_all_metadata()

        return episode_meta

    def _update_stats(self, states_array: np.ndarray, actions_array: np.ndarray):
        """更新统计信息"""
        if self._stats is None:
            self._stats = {}

        self._stats["observation.state"] = self._build_stats_entry("observation.state", states_array)
        self._stats["action"] = self._build_stats_entry("action", actions_array)

    def _save_episode_metadata(self, episode_meta: Dict[str, Any]):
        """保存单个 episode 元数据到 parquet"""
        self._episodes.append(episode_meta)

        # 写入 episodes parquet
        df = pd.DataFrame(self._episodes)
        chunk_idx = episode_meta["start_frame_index"] // self.chunk_size

        chunk_dir = self.episodes_dir / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(exist_ok=True)

        # 只保存该 chunk 的 episodes
        chunk_start = chunk_idx * self.chunk_size
        chunk_end = (chunk_idx + 1) * self.chunk_size

        chunk_episodes = [
            ep for ep in self._episodes
            if chunk_start <= ep["start_frame_index"] < chunk_end
        ]

        if chunk_episodes:
            df_chunk = pd.DataFrame(chunk_episodes)
            parquet_file = chunk_dir / "episodes.parquet"
            df_chunk.to_parquet(parquet_file, index=False)

    def _save_all_metadata(self):
        """保存所有元数据文件"""
        # 保存 info.json
        self._ensure_dirs()
        info = self.info
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # 保存 stats.json
        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        # 保存 tasks.parquet
        if self._tasks:
            df_tasks = pd.DataFrame(self._tasks)
            df_tasks.to_parquet(self.meta_dir / "tasks.parquet", index=False)

    def add_task(self, task_name: str, description: str = ""):
        """添加任务"""
        task = {
            "task_name": task_name,
            "description": description,
            "created_at": time.time(),
        }
        self._tasks.append(task)

    def get_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return self.info

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats

    def get_episodes(self) -> List[Dict[str, Any]]:
        """获取所有 episodes"""
        return self._episodes

    def finalize(self):
        """完成数据导出，保存所有元数据"""
        self._save_all_metadata()


# 全局元数据实例
_global_metadata: Optional[LeRobotDatasetMetadata] = None
_metadata_lock = threading.Lock()


def get_metadata(
    output_dir: str = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    fps: int = DEFAULT_FPS,
) -> LeRobotDatasetMetadata:
    """获取全局元数据实例（单例模式）"""
    global _global_metadata

    with _metadata_lock:
        if _global_metadata is None:
            _global_metadata = LeRobotDatasetMetadata(
                output_dir=output_dir,
                chunk_size=chunk_size,
                fps=fps,
            )
    return _global_metadata


def reset_metadata():
    """重置全局元数据实例"""
    global _global_metadata
    with _metadata_lock:
        if _global_metadata is not None:
            _global_metadata.finalize()
        _global_metadata = None


def export_episode(
    samples: List[Dict[str, Any]],
    episode_id: int = 1,
    task_name: str = "default",
    output_dir: str = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    导出单个 episode

    Args:
        samples: 样本列表
        episode_id: episode ID
        task_name: 任务名称
        output_dir: 输出目录
        chunk_size: 分块大小

    Returns:
        导出的目录路径
    """
    if not samples:
        return ""

    metadata = get_metadata(output_dir, chunk_size)
    metadata.save_episode(samples, episode_id, task_name)

    return str(metadata.output_dir)


def export_all_episodes(
    episode_samples: Dict[int, List[Dict[str, Any]]],
    output_dir: str = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    导出所有 episodes

    Args:
        episode_samples: 所有 episodes 的样本 {episode_id: samples}
        output_dir: 输出目录
        chunk_size: 分块大小

    Returns:
        导出的目录路径
    """
    if not episode_samples:
        return ""

    metadata = get_metadata(output_dir, chunk_size)

    # 添加默认任务
    metadata.add_task("default", "Default driving task")

    for episode_id in sorted(episode_samples.keys()):
        samples = episode_samples[episode_id]
        if samples:
            metadata.save_episode(samples, episode_id, "default")

    metadata.finalize()

    return str(metadata.output_dir)


def load_dataset(data_dir: str = "output/dataset") -> Dict[str, torch.Tensor]:
    """
    加载数据集

    Args:
        data_dir: 数据集目录
    """
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / data_dir

    # 加载统计信息
    stats_path = data_path / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)

    # 加载info获取维度信息
    info_path = data_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)

    action_dim = ACTION_DIM

    # 加载所有parquet文件
    parquet_files = sorted(data_path.glob("data/chunk-*/file-*.parquet"))

    images = []
    states = []
    actions = []

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)

        # 加载图像
        for img_path in df["observation.image"]:
            full_path = data_path / img_path
            if full_path.exists():
                img = Image.open(full_path).convert("RGB")
                img = img.resize((224, 224))
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                images.append(img_tensor)
            else:
                images.append(torch.zeros(3, 224, 224))

        # 加载状态
        for state_list in df["observation.state"]:
            state_arr = np.array(state_list, dtype=np.float32)
            state = torch.from_numpy(state_arr).float()
            states.append(state)

        # 加载动作（[chunk_size, 3]）
        for action_data in df["action"]:
            # action_data 可能是嵌套列表
            action_arr = np.array(action_data, dtype=np.float32)
            # 确保是正确形状 [chunk_size, 3]
            if action_arr.ndim == 1:
                action_arr = action_arr.reshape(1, -1)
            action = torch.from_numpy(action_arr).float()
            actions.append(action)

    images = torch.stack(images)
    states = torch.stack(states)
    actions = torch.stack(actions)

    return {
        "observation.image": images,
        "observation.state": states,
        "action": actions,
    }


def create_demo_samples(num_samples: int = 100) -> List[Dict[str, Any]]:
    """创建演示样本用于测试导出功能"""
    samples = []
    for i in range(num_samples):
        sample = {
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAB//9k=",
            "state": {
                "x": 400 + i * 0.5,
                "y": 300,
                "angle": -np.pi / 2,
                "speed": 1.0,
                "maxSpeed": 5,
                "acceleration": 0.2,
                "rotationSpeed": 0.05,
            },
            "actions": ["forward"] if i % 2 == 0 else [],
        }
        samples.append(sample)
    return samples


if __name__ == "__main__":
    # 测试导出功能
    from backend.models import state as sim_state

    if sim_state.dataset_samples:
        print(f"使用已采集的 {len(sim_state.dataset_samples)} 个样本导出")
    else:
        print("没有采集数据，使用演示样本测试导出")
        demo_samples = create_demo_samples(100)
