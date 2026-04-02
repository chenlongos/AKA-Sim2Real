from __future__ import annotations

import logging

from backend.models import state

logger = logging.getLogger(__name__)


class EpisodeService:
    def start_episode(self, episode_id: int, task_name: str) -> dict:
        logger.info(f"开始新 episode: {episode_id}, task: {task_name}")
        state.start_episode(episode_id, task_name)
        return {
            "episode_id": episode_id,
            "task_name": task_name,
            "frame_count": 0,
        }

    def end_episode(self, episode_id: int) -> dict:
        logger.info(f"结束 episode: {episode_id}")
        samples = state.end_episode(episode_id)

        if not samples:
            state.clear_episode_buffer(episode_id)
            return {
                "episode_id": episode_id,
                "frame_count": 0,
            }

        logger.info(f"导出 episode {episode_id} 的 {len(samples)} 个样本...")
        try:
            from backend.services.episode.exporter import export_episode

            output_path = export_episode(
                samples,
                episode_id=episode_id,
                task_name=state.episode_buffer.get(episode_id, {}).get("task_name", "default"),
            )
            logger.info(f"数据已导出到: {output_path}")
            result = {
                "episode_id": episode_id,
                "frame_count": len(samples),
                "exported": True,
                "output_path": output_path,
            }
        except Exception as exc:
            logger.error(f"导出失败: {exc}")
            result = {
                "episode_id": episode_id,
                "frame_count": len(samples),
                "exported": False,
                "error": str(exc),
            }

        state.clear_episode_buffer(episode_id)
        return result

    def finalize_episode(self, episode_id: int) -> dict | None:
        logger.info(f"完成 episode: {episode_id}")
        samples = state.get_episode_samples(episode_id)
        if not samples:
            return None

        try:
            from backend.services.episode.exporter import export_episode

            output_path = export_episode(
                samples,
                episode_id=episode_id,
                task_name=state.episode_metadata.get(episode_id, {}).get("task_name", "default"),
            )
            logger.info(f"数据已导出到: {output_path}")
            return {
                "episode_id": episode_id,
                "frame_count": len(samples),
                "output_path": output_path,
            }
        except Exception as exc:
            logger.error(f"导出失败: {exc}")
            return {
                "episode_id": episode_id,
                "frame_count": len(samples),
                "error": str(exc),
            }

    def set_episode(self, episode_id: int) -> dict:
        if episode_id in state.episode_samples:
            state.episode_samples[episode_id] = []
        state.clear_episode_buffer(episode_id)
        if episode_id in state.episode_frame_index:
            state.episode_frame_index[episode_id] = []

        state.current_episode_id = episode_id
        logger.info(f"设置当前采集轮次: {episode_id}, samples长度: {len(state.episode_samples.get(episode_id, []))}")
        return self.get_episodes_info()

    def get_episodes_info(self) -> dict:
        return {
            "current_episode": state.current_episode_id,
            "episodes": {k: len(v) for k, v in state.episode_samples.items()},
            "buffer_size": state.get_current_buffer_size(state.current_episode_id),
        }

    def delete_episode(self, episode_id: int | None) -> dict | None:
        if episode_id is None:
            return None

        if episode_id in state.episode_samples:
            del state.episode_samples[episode_id]
        if episode_id in state.episode_buffer:
            del state.episode_buffer[episode_id]
        if episode_id in state.episode_metadata:
            del state.episode_metadata[episode_id]
        if episode_id in state.episode_frame_index:
            del state.episode_frame_index[episode_id]

        return self.get_episodes_info()

    def get_episode_status(self) -> dict:
        episode_id = state.current_episode_id
        return {
            "episode_id": episode_id,
            "is_recording": state.is_recording,
            "frame_count": state.get_current_buffer_size(episode_id),
            "task_name": state.episode_buffer.get(episode_id, {}).get("task_name", "default"),
        }

    def collect_data(self, image_data: str, actions: list[str]) -> int | None:
        if not state.is_recording:
            return None

        car_state_copy = state.car_state.copy()
        sample = {
            "image": image_data,
            "state": {
                "vel_left": car_state_copy.get("vel_left", 0),
                "vel_right": car_state_copy.get("vel_right", 0),
            },
            "actions": actions,
        }

        if state.current_episode_id not in state.episode_samples:
            state.episode_samples[state.current_episode_id] = []

        state.episode_samples[state.current_episode_id].append(sample)
        state.add_frame_to_episode(state.current_episode_id, sample)
        return len(state.episode_samples[state.current_episode_id])
