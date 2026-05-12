from __future__ import annotations

import logging

from backend.models import state

logger = logging.getLogger(__name__)


class EpisodeEventsMixin:
    async def on_start_episode(self, sid: str, payload: dict):
        user_id = payload.get("user_id")
        if not user_id:
            logger.error("收到开始采集请求但缺少user_id")
            await self.emit("error", {"message": "user_id is required"}, room=sid)
            return

        episode_id = payload.get("episode_id", state.user_current_episode_id.get(user_id, 1))
        task_name = payload.get("task_name", "default")
        logger.info(f"收到开始采集请求: user={user_id}, episode_id={episode_id}, task_name={task_name}")

        result = self.episode_service.start_episode(user_id, episode_id, task_name)

        # 重置 ACT 推理上下文（temporal ensembling 需要）
        self.sim_controller.reset_act_inference_context()

        await self.emit("episode_started", result, room=sid)

    async def on_end_episode(self, sid: str, payload: dict):
        user_id = payload.get("user_id")
        if not user_id:
            logger.error("收到结束采集请求但缺少user_id")
            await self.emit("error", {"message": "user_id is required"}, room=sid)
            return

        episode_id = payload.get("episode_id", state.user_current_episode_id.get(user_id, 1))
        logger.info(f"收到结束采集请求: user={user_id}, episode_id={episode_id}")
        await self.emit("episode_ended", self.episode_service.end_episode(user_id, episode_id), room=sid)

    async def on_finalize_episode(self, sid: str, payload: dict):
        user_id = payload.get("user_id")
        if not user_id:
            logger.error("收到完成episode请求但缺少user_id")
            await self.emit("error", {"message": "user_id is required"}, room=sid)
            return

        episode_id = payload.get("episode_id", state.user_current_episode_id.get(user_id, 1))
        logger.info(f"收到完成episode请求: user={user_id}, episode_id={episode_id}")
        result = self.episode_service.finalize_episode(user_id, episode_id)
        if result is not None:
            await self.emit("episode_finalized", result, room=sid)

    async def on_set_episode(self, sid: str, payload: dict):
        user_id = payload.get("user_id")
        episode_id = payload.get("episode_id")
        if not user_id or episode_id is None:
            logger.error("收到设置轮次请求但缺少user_id或episode_id")
            await self.emit("error", {"message": "user_id and episode_id are required"}, room=sid)
            return

        logger.info(f"收到设置轮次请求: user={user_id}, episode_id={episode_id}")
        info = self.episode_service.set_episode(user_id, episode_id)
        info["current_episode"] = episode_id
        info["buffer_size"] = state.get_current_buffer_size(user_id, episode_id)
        await self.emit("episode_info", info, room=sid)

    async def on_get_episodes(self, sid: str, payload: dict = None):
        user_id = payload.get("user_id") if payload else None
        if not user_id:
            logger.error("收到获取episodes请求但缺少user_id")
            await self.emit("error", {"message": "user_id is required"}, room=sid)
            return
        await self.emit("episode_info", self.episode_service.get_episodes_info(user_id), room=sid)

    async def on_delete_episode(self, sid: str, payload: dict):
        user_id = payload.get("user_id")
        episode_id = payload.get("episode_id")
        if not user_id or episode_id is None:
            logger.error("收到删除轮次请求但缺少user_id或episode_id")
            await self.emit("error", {"message": "user_id and episode_id are required"}, room=sid)
            return

        logger.info(f"收到删除轮次请求: user={user_id}, episode_id={episode_id}")
        info = self.episode_service.delete_episode(user_id, episode_id)
        if info is not None:
            await self.emit("episode_info", info, room=sid)

    async def on_get_episode_status(self, sid: str, payload: dict = None):
        user_id = payload.get("user_id") if payload else None
        if not user_id:
            logger.error("收到获取episode状态请求但缺少user_id")
            await self.emit("error", {"message": "user_id is required"}, room=sid)
            return
        await self.emit("episode_status", self.episode_service.get_episode_status(user_id), room=sid)

    async def on_collect_data(self, sid: str, payload: dict):
        user_id = payload.get("user_id")
        if not user_id:
            logger.error("收到数据采集请求但缺少user_id")
            return

        try:
            count = await self.episode_service.collect_data(
                payload.get("image", ""),
                user_id=user_id,
                dataset_name=payload.get("dataset_name", "default"),
                timestamp=payload.get("timestamp"),
                state_payload=payload.get("state"),
                action_payload=payload.get("action"),
            )
            if count is not None and count % 10 == 0:
                current_episode_id = state.user_current_episode_id.get(user_id, 1)
                await self.emit("collection_count", {"count": count, "episode_id": current_episode_id}, room=sid)
        except Exception as exc:
            logger.error(f"数据采集失败: {exc}")
