"""
AKA-Sim 后端 - Socket.IO 事件处理
"""

import asyncio
import logging
import math

from socketio import AsyncNamespace

from models import state

logger = logging.getLogger(__name__)

# 当前按下的按键集合
current_actions = set()

class SimNamespace(AsyncNamespace):
    """模拟器 Socket.IO 命名空间"""

    async def on_connect(self, sid: str, environ: dict):
        """客户端连接"""
        logger.info(f"客户端连接: {sid}")
        await self.emit("connected", {"sid": sid})
        # 发送当前车辆状态
        await self.emit("car_state_update", state.car_state)

    async def on_disconnect(self, sid: str):
        """客户端断开"""
        logger.info(f"客户端断开: {sid}")

    async def on_action(self, sid: str, actions: list):
        """处理动作命令 - 接收前端发送的当前按键列表"""
        global current_actions
        current_actions = set(actions)

        # 如果没有按键，立刻减速到0
        if not actions:
            state.car_state["speed"] = 0

    async def on_reset_car_state(self, sid: str):
        """重置车辆状态"""
        logger.info("重置车辆状态")
        state.reset_car_state()
        await self.emit("car_state_update", state.car_state)

    async def on_get_car_state(self, sid: str):
        """获取车辆状态"""
        await self.emit("car_state_update", state.car_state)

    async def on_act_infer(self, sid: str, payload: dict):
        """ACT 模型推理"""
        logger.info(f"收到 ACT 推理请求")

        try:
            import config as cfg
            inference_state = payload.get("state", [0.0] * cfg.config.STATE_DIM)
            image = payload.get("image", None)
            logger.info(f"state: {inference_state}, image provided: {image is not None}")
            if image:
                logger.info(f"image length: {len(image)}")

            from services.act_model import act_inference
            action = act_inference(inference_state, image)

            logger.info(f"推理结果: {action[0]}")

            await self.emit("act_infer_result", {
                "success": True,
                "action": action,
            })
        except Exception as e:
            logger.error(f"ACT 推理失败: {e}")
            await self.emit("act_infer_result", {
                "success": False,
                "error": str(e),
            })

    async def on_start_episode(self, sid: str, payload: dict):
        """
        开始新的 episode

        Args:
            payload: {"episode_id": int, "task_name": str}
        """
        episode_id = payload.get("episode_id", state.current_episode_id)
        task_name = payload.get("task_name", "default")

        logger.info(f"开始新 episode: {episode_id}, task: {task_name}")

        # 使用新的 episode buffer 机制
        state.start_episode(episode_id, task_name)

        await self.emit("car_state_update", state.car_state)

        await self.emit("episode_started", {
            "episode_id": episode_id,
            "task_name": task_name,
            "frame_count": 0,
        })

    async def on_end_episode(self, sid: str, payload: dict):
        """
        结束当前 episode

        Args:
            payload: {"episode_id": int}
        """
        episode_id = payload.get("episode_id", state.current_episode_id)

        logger.info(f"结束 episode: {episode_id}")

        # 获取 episode 的所有样本
        samples = state.end_episode(episode_id)

        # 导出数据
        if samples:
            logger.info(f"导出 episode {episode_id} 的 {len(samples)} 个样本...")
            try:
                from services.data_export import export_episode
                output_path = export_episode(
                    samples,
                    episode_id=episode_id,
                    task_name=state.episode_buffer.get(episode_id, {}).get("task_name", "default"),
                )
                logger.info(f"数据已导出到: {output_path}")
                await self.emit("episode_ended", {
                    "episode_id": episode_id,
                    "frame_count": len(samples),
                    "exported": True,
                    "output_path": output_path,
                })
            except Exception as e:
                logger.error(f"导出失败: {e}")
                await self.emit("episode_ended", {
                    "episode_id": episode_id,
                    "frame_count": len(samples),
                    "exported": False,
                    "error": str(e),
                })
        else:
            await self.emit("episode_ended", {
                "episode_id": episode_id,
                "frame_count": 0,
            })

        # 清除 buffer
        state.clear_episode_buffer(episode_id)

    async def on_finalize_episode(self, sid: str, payload: dict):
        """
        完成 episode 并保存到磁盘（实时保存模式）

        Args:
            payload: {"episode_id": int}
        """
        episode_id = payload.get("episode_id", state.current_episode_id)

        logger.info(f"完成 episode: {episode_id}")

        # 获取 episode 的所有样本
        samples = state.get_episode_samples(episode_id)

        if samples:
            try:
                from services.data_export import export_episode
                output_path = export_episode(
                    samples,
                    episode_id=episode_id,
                    task_name=state.episode_metadata.get(episode_id, {}).get("task_name", "default"),
                )
                logger.info(f"数据已导出到: {output_path}")
                await self.emit("episode_finalized", {
                    "episode_id": episode_id,
                    "frame_count": len(samples),
                    "output_path": output_path,
                })
            except Exception as e:
                logger.error(f"导出失败: {e}")
                await self.emit("episode_finalized", {
                    "episode_id": episode_id,
                    "frame_count": len(samples),
                    "error": str(e),
                })

    async def on_set_episode(self, sid: str, episode_id: int):
        """设置当前采集的轮次"""
        state.current_episode_id = episode_id
        # 如果这个轮次不存在，初始化为空列表
        if episode_id not in state.episode_samples:
            state.episode_samples[episode_id] = []
        logger.info(f"设置当前采集轮次: {episode_id}")
        # 发送当前各轮次的样本数
        await self.emit("episode_info", {
            "current_episode": episode_id,
            "episodes": {k: len(v) for k, v in state.episode_samples.items()},
            "buffer_size": state.get_current_buffer_size(episode_id),
        })

    async def on_get_episodes(self, sid: str):
        """获取所有轮次信息"""
        await self.emit("episode_info", {
            "current_episode": state.current_episode_id,
            "episodes": {k: len(v) for k, v in state.episode_samples.items()},
            "buffer_size": state.get_current_buffer_size(state.current_episode_id),
        })

    async def on_delete_episode(self, sid: str, payload: dict):
        """删除指定轮次的数据"""
        episode_id = payload.get("episode_id")
        if episode_id is None:
            return

        # 从内存中删除
        if episode_id in state.episode_samples:
            del state.episode_samples[episode_id]
        if episode_id in state.episode_buffer:
            del state.episode_buffer[episode_id]
        if episode_id in state.episode_metadata:
            del state.episode_metadata[episode_id]
        if episode_id in state.episode_frame_index:
            del state.episode_frame_index[episode_id]

        # 重新广播轮次信息
        await self.on_get_episodes(sid)

    async def on_get_episode_status(self, sid: str):
        """获取当前 episode 状态"""
        episode_id = state.current_episode_id
        await self.emit("episode_status", {
            "episode_id": episode_id,
            "is_recording": state.is_recording,
            "frame_count": state.get_current_buffer_size(episode_id),
            "task_name": state.episode_buffer.get(episode_id, {}).get("task_name", "default"),
        })

    async def on_collect_data(self, sid: str, payload: dict):
        """接收前端发送的图像数据进行保存"""
        # 检查是否正在录制 episode
        if not state.is_recording:
            return

        try:
            # 解析图像数据 (base64)
            image_data = payload.get("image", "")

            # 获取当前车辆状态
            car_state_copy = state.car_state.copy()

            # 计算速度（基于当前 speed 和 angle）
            speed = car_state_copy.get("speed", 0)
            angle = car_state_copy.get("angle", 0)
            rotation_speed = car_state_copy.get("rotationSpeed", 0.05)

            # 车体坐标系下的速度
            # x.vel: 前向速度 (cos方向)
            # y.vel: 横向速度 (sin方向)
            # 当前实际速度（从 car_state 计算）
            x_vel = speed * math.cos(angle)
            y_vel = speed * math.sin(angle)
            # 当前实际转向速度（没有按键时为0）
            actions = list(current_actions)
            if "left" in actions:
                current_theta_vel = rotation_speed  # 当前正在左转
            elif "right" in actions:
                current_theta_vel = -rotation_speed  # 当前正在右转
            else:
                current_theta_vel = 0

            # 状态: 当前实际速度
            state_3d = {
                "x_vel": x_vel,
                "y_vel": y_vel,
                "theta_vel": current_theta_vel,
            }

            # 保存样本到当前轮次
            sample = {
                "image": image_data,  # base64编码的JPEG图像
                "state": state_3d,
                "actions": actions,  # 离散动作，导出时会转成连续值
            }

            # 确保当前轮次的列表存在
            if state.current_episode_id not in state.episode_samples:
                state.episode_samples[state.current_episode_id] = []

            state.episode_samples[state.current_episode_id].append(sample)

            # 同时添加到 episode buffer（用于新的导出机制）
            state.add_frame_to_episode(state.current_episode_id, sample)

            # 定期广播计数 (每10个样本)
            current_count = len(state.episode_samples[state.current_episode_id])
            if current_count % 10 == 0:
                await self.emit("collection_count", {
                    "count": current_count,
                    "episode_id": state.current_episode_id
                })

        except Exception as e:
            logger.error(f"数据采集失败: {e}")


# 游戏循环任务
async def game_loop_task(sio_server):
    """游戏循环 - 处理物理更新和状态广播"""
    global current_actions

    while True:
        try:
            # 根据当前按键更新状态
            if current_actions:
                # logger.info(f"游戏循环处理动作: {current_actions}")
                for action in current_actions:
                    state.update_car_state(action)
                # logger.info(f"更新后状态: x={state.car_state['x']}, y={state.car_state['y']}, speed={state.car_state['speed']}")
            else:
                # 没有按键时应用摩擦力减速
                state.apply_friction()

            # 广播状态
            await sio_server.emit("car_state_update", state.car_state)
        except Exception as e:
            logger.error(f"游戏循环错误: {e}")

        # 30 FPS
        await asyncio.sleep(1 / 30)


def start_game_loop(sio_server):
    """启动游戏循环"""
    asyncio.create_task(game_loop_task(sio_server))
