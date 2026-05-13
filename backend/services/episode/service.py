from __future__ import annotations

import logging
import time
from typing import Any

from backend.models import state

logger = logging.getLogger(__name__)


class EpisodeService:
    def start_episode(self, user_id: str, episode_id: int, task_name: str) -> dict:
        logger.info(f"开始新 episode: user={user_id}, episode={episode_id}, task={task_name}")
        state.start_episode(user_id, episode_id, task_name)
        return {
            "user_id": user_id,
            "episode_id": episode_id,
            "task_name": task_name,
            "frame_count": 0,
        }

    def end_episode(self, user_id: str, episode_id: int) -> dict:
        logger.info(f"结束 episode: user={user_id}, episode={episode_id}")
        samples = state.end_episode(user_id, episode_id)

        if not samples:
            state.clear_episode_buffer(user_id, episode_id)
            return {
                "user_id": user_id,
                "episode_id": episode_id,
                "frame_count": 0,
            }

        # 使用传入的 user_id 和 dataset_name 用于创建子目录
        dataset_name = samples[0].get("dataset_name", "default") if samples else "default"
        subdir = f"{user_id}/{dataset_name}"

        logger.info(f"导出 episode {episode_id} 的 {len(samples)} 个样本 (user={user_id}, dataset={dataset_name})...")
        try:
            from backend.services.episode.exporter import export_episode

            task = state.user_episode_metadata.get(user_id, {}).get(episode_id, {}).get("task_name", "default")
            output_path = export_episode(
                samples,
                episode_id=episode_id,
                task_name=task,
                subdir=subdir,
            )
            logger.info(f"数据已导出到: {output_path}")
            result = {
                "user_id": user_id,
                "episode_id": episode_id,
                "frame_count": len(samples),
                "exported": True,
                "output_path": output_path,
            }
        except Exception as exc:
            logger.error(f"导出失败: {exc}")
            result = {
                "user_id": user_id,
                "episode_id": episode_id,
                "frame_count": len(samples),
                "exported": False,
                "error": str(exc),
            }

        state.clear_episode_buffer(user_id, episode_id)
        return result

    def finalize_episode(self, user_id: str, episode_id: int) -> dict | None:
        logger.info(f"完成 episode: user={user_id}, episode={episode_id}")
        samples = state.get_episode_samples(user_id, episode_id)
        if not samples:
            return None

        # 使用传入的 user_id 和 dataset_name 用于创建子目录
        dataset_name = samples[0].get("dataset_name", "default") if samples else "default"
        subdir = f"{user_id}/{dataset_name}"

        try:
            from backend.services.episode.exporter import export_episode

            task = state.user_episode_metadata.get(user_id, {}).get(episode_id, {}).get("task_name", "default")
            output_path = export_episode(
                samples,
                episode_id=episode_id,
                task_name=task,
                subdir=subdir,
            )
            logger.info(f"数据已导出到: {output_path}")
            return {
                "user_id": user_id,
                "episode_id": episode_id,
                "frame_count": len(samples),
                "output_path": output_path,
            }
        except Exception as exc:
            logger.error(f"导出失败: {exc}")
            return {
                "user_id": user_id,
                "episode_id": episode_id,
                "frame_count": len(samples),
                "error": str(exc),
            }

    def set_episode(self, user_id: str, episode_id: int) -> dict:
        state._ensure_user_exists(user_id)
        if episode_id in state.episode_samples:
            state.episode_samples[episode_id] = []
        state.clear_episode_buffer(user_id, episode_id)
        state.user_current_episode_id[user_id] = episode_id

        state.current_episode_id = episode_id
        logger.info(f"设置当前采集轮次: user={user_id}, episode={episode_id}, samples长度: {len(state.episode_samples.get(episode_id, []))}")
        return self.get_episodes_info(user_id)

    def get_episodes_info(self, user_id: str) -> dict:
        state._ensure_user_exists(user_id)
        return {
            "user_id": user_id,
            "current_episode": state.user_current_episode_id.get(user_id, 1),
            "episodes": {k: len(v) for k, v in state.user_episode_buffers.get(user_id, {}).items()},
            "buffer_size": state.get_current_buffer_size(user_id),
        }

    def delete_episode(self, user_id: str, episode_id: int | None) -> dict | None:
        if episode_id is None:
            return None

        state._ensure_user_exists(user_id)
        if episode_id in state.episode_samples:
            del state.episode_samples[episode_id]
        if episode_id in state.user_episode_buffers.get(user_id, {}):
            del state.user_episode_buffers[user_id][episode_id]
        if episode_id in state.user_episode_metadata.get(user_id, {}):
            del state.user_episode_metadata[user_id][episode_id]
        if episode_id in state.user_episode_frame_index.get(user_id, {}):
            del state.user_episode_frame_index[user_id][episode_id]

        return self.get_episodes_info(user_id)

    def get_episode_status(self, user_id: str) -> dict:
        state._ensure_user_exists(user_id)
        episode_id = state.user_current_episode_id.get(user_id, 1)
        return {
            "user_id": user_id,
            "episode_id": episode_id,
            "is_recording": state.user_is_recording.get(user_id, False),
            "frame_count": state.get_current_buffer_size(user_id, episode_id),
            "task_name": state.user_episode_buffers.get(user_id, {}).get(episode_id, {}).get("task_name", "default"),
        }

    async def collect_data(
        self,
        image_data: str,
        user_id: str,
        dataset_name: str = "default",
        *,
        timestamp: int | None = None,
        state_payload: dict[str, Any] | None = None,
        action_payload: list[float] | None = None,
    ) -> int | None:
        if not state.is_user_recording(user_id):
            return None

        capture_timestamp = timestamp if timestamp is not None else int(time.time() * 1000)

        # 从前端传来的状态
        vel_left = 0
        vel_right = 0
        if isinstance(state_payload, dict):
            if isinstance(state_payload.get("vel_left"), (int, float)):
                vel_left = state_payload["vel_left"]
            if isinstance(state_payload.get("vel_right"), (int, float)):
                vel_right = state_payload["vel_right"]

        sample = {
            "image": image_data,
            "state": {
                "vel_left": vel_left,
                "vel_right": vel_right,
            },
            "capture_timestamp_ms": capture_timestamp,
            "user_id": user_id,
            "dataset_name": dataset_name,
        }
        if isinstance(action_payload, list) and len(action_payload) >= 2:
            left_target = action_payload[0]
            right_target = action_payload[1]
            if isinstance(left_target, (int, float)) and isinstance(right_target, (int, float)):
                action = [float(left_target), float(right_target)]
                # 第三个维度 gripper_target（0=释放, 1=夹取）
                if len(action_payload) >= 3 and isinstance(action_payload[2], (int, float)):
                    action.append(float(action_payload[2]))
                else:
                    action.append(0.0)  # 默认释放
                sample["action"] = action

        current_episode_id = state.user_current_episode_id.get(user_id, 1)

        state.add_frame_to_episode(user_id, current_episode_id, sample)

        # 从用户隔离的 buffer 获取 count
        user_buffer = state.user_episode_buffers.get(user_id, {}).get(current_episode_id, {})
        count = len(user_buffer.get("samples", []))
        # 每50帧输出一次采集日志，避免太频繁
        if count % 50 == 0:
            logger.info(f"数据采集中: user={user_id}, episode={current_episode_id}, frames={count}")

        return count
