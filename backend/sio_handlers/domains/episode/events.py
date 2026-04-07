from __future__ import annotations

import logging

from backend.models import state

logger = logging.getLogger(__name__)


class EpisodeEventsMixin:
    async def on_start_episode(self, sid: str, payload: dict):
        episode_id = payload.get("episode_id", state.current_episode_id)
        task_name = payload.get("task_name", "default")
        logger.info(f"收到开始采集请求: episode_id={episode_id}, task_name={task_name}")

        result = self.episode_service.start_episode(episode_id, task_name)
        await self.emit("car_state_update", self.sim_controller.get_car_state())
        await self.emit("episode_started", result)

    async def on_end_episode(self, sid: str, payload: dict):
        episode_id = payload.get("episode_id", state.current_episode_id)
        logger.info(f"收到结束采集请求: episode_id={episode_id}")
        await self.emit("episode_ended", self.episode_service.end_episode(episode_id))

    async def on_finalize_episode(self, sid: str, payload: dict):
        episode_id = payload.get("episode_id", state.current_episode_id)
        logger.info(f"收到完成episode请求: episode_id={episode_id}")
        result = self.episode_service.finalize_episode(episode_id)
        if result is not None:
            await self.emit("episode_finalized", result)

    async def on_set_episode(self, sid: str, episode_id: int):
        logger.info(f"收到设置轮次请求: episode_id={episode_id}")
        info = self.episode_service.set_episode(episode_id)
        info["current_episode"] = episode_id
        info["buffer_size"] = state.get_current_buffer_size(episode_id)
        await self.emit("episode_info", info)

    async def on_get_episodes(self, sid: str):
        await self.emit("episode_info", self.episode_service.get_episodes_info())

    async def on_delete_episode(self, sid: str, payload: dict):
        episode_id = payload.get("episode_id")
        logger.info(f"收到删除轮次请求: episode_id={episode_id}")
        info = self.episode_service.delete_episode(episode_id)
        if info is not None:
            await self.emit("episode_info", info)

    async def on_get_episode_status(self, sid: str):
        await self.emit("episode_status", self.episode_service.get_episode_status())

    async def on_collect_data(self, sid: str, payload: dict):
        try:
            count = await self.episode_service.collect_data(
                payload.get("image", ""),
                timestamp=payload.get("timestamp"),
                state_payload=payload.get("state"),
                action_payload=payload.get("action"),
            )
            if count is not None and count % 10 == 0:
                await self.emit("collection_count", {"count": count, "episode_id": state.current_episode_id})
        except Exception as exc:
            logger.error(f"数据采集失败: {exc}")
