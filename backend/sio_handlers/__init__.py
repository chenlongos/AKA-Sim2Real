"""
AKA-Sim 后端 - Socket.IO 事件处理
"""

import asyncio
import logging

from socketio import AsyncNamespace

from backend.models import state
from backend.services.act_model import get_act_runtime

logger = logging.getLogger(__name__)

# 当前按下的按键集合
current_actions = set()
connected_clients = set()

# 推理模式标志 - 推理时直接使用设置的速度，不应用摩擦力
inference_mode = False

# 摄像头是否激活
camera_active = False

# 调试计数器
_debug_inference_set_count = 0
_debug_game_loop_inference_count = 0
_act_runtime = get_act_runtime()


def set_act_runtime(runtime):
    global _act_runtime
    _act_runtime = runtime


def _extract_velocity_from_action(action) -> tuple[float, float]:
    """从模型输出中提取第一帧左右轮速度，兼容不同嵌套层级。"""
    if not isinstance(action, (list, tuple)) or len(action) == 0:
        raise ValueError(f"无效 action 格式: {action}")

    first = action[0]
    if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
        first = first[0]

    if not isinstance(first, (list, tuple)) or len(first) < 2:
        raise ValueError(f"无法从 action 提取速度: {action}")

    vel_left = float(first[0])
    vel_right = float(first[1])
    return vel_left, vel_right


def _reset_act_inference_context():
    try:
        _act_runtime.reset_inference_context()
    except Exception as exc:
        logger.debug(f"重置 ACT 推理上下文失败: {exc}")

class SimNamespace(AsyncNamespace):
    """模拟器 Socket.IO 命名空间"""

    async def on_connect(self, sid: str, environ: dict):
        """客户端连接"""
        global connected_clients
        connected_clients.add(sid)
        logger.info(f"客户端连接: {sid}")
        await self.emit("connected", {"sid": sid})
        # 发送当前车辆状态
        await self.emit("car_state_update", state.car_state)
        # 发送当前摄像头图像（如果有）
        if state.camera_image:
            await self.emit("camera_image", {"image": state.camera_image})

    async def on_disconnect(self, sid: str):
        """客户端断开"""
        global connected_clients, current_actions
        connected_clients.discard(sid)
        if not connected_clients:
            current_actions = set()
        logger.info(f"客户端断开: {sid}")

    async def on_action(self, sid: str, actions: list):
        """处理动作命令 - 接收前端发送的当前按键列表"""
        global current_actions, inference_mode
        current_actions = set(actions)

        # 只有在有实际控制动作时才退出推理模式（stop 和空动作不算用户主动控制）
        if actions and actions != ['stop']:
            inference_mode = False
            _reset_act_inference_context()
            logger.info(f"[on_action] 用户控制，退出推理模式: {actions}")
        elif not actions or actions == ['stop']:
            # 空动作或 stop 时保持推理模式
            pass

        # 如果没有按键，立刻停止
        if not actions:
            state.car_state["vel_left"] = 0
            state.car_state["vel_right"] = 0

    async def on_reset_car_state(self, sid: str):
        """重置车辆状态"""
        logger.info("重置车辆状态")
        _reset_act_inference_context()
        state.reset_car_state()
        await self.emit("car_state_update", state.car_state)

    async def on_get_car_state(self, sid: str):
        """获取车辆状态"""
        await self.emit("car_state_update", state.car_state)

    async def on_act_infer(self, sid: str, payload: dict):
        """ACT 模型推理"""
        global inference_mode, current_actions
        logger.info(f"收到 ACT 推理请求")

        try:
            from backend.config import config as backend_config
            inference_state = payload.get("state", [0.0] * backend_config.STATE_DIM)
            image = payload.get("image", None)
            logger.info(f"state: {inference_state}, image provided: {image is not None}")
            if image:
                logger.info(f"image length: {len(image)}")

            action = _act_runtime.infer(inference_state, image)
            vel_left, vel_right = _extract_velocity_from_action(action)

            current_actions = set()
            inference_mode = True

            # 推理结果直接在后端落到小车状态，前端只负责触发推理。
            state.update_car_state([vel_left, vel_right])

            logger.info(f"推理结果: {action[0]}")
            logger.info(f"[on_act_infer] 已应用推理速度: vel_left={vel_left:.4f}, vel_right={vel_right:.4f}")

            await self.emit("car_state_update", state.car_state)

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
                from backend.services.data_export import export_episode
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
                from backend.services.data_export import export_episode
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
        # 确保目标轮次的数据被清空（从内存和buffer）
        if episode_id in state.episode_samples:
            state.episode_samples[episode_id] = []
        state.clear_episode_buffer(episode_id)
        if episode_id in state.episode_frame_index:
            state.episode_frame_index[episode_id] = []

        state.current_episode_id = episode_id
        logger.info(f"设置当前采集轮次: {episode_id}, samples长度: {len(state.episode_samples.get(episode_id, []))}")
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

    async def on_start_camera(self, sid: str):
        """启动摄像头捕获"""
        global camera_active
        camera_active = True
        logger.info(f"[摄像头] 已启动, sid={sid}")

    async def on_stop_camera(self, sid: str):
        """停止摄像头捕获"""
        global camera_active
        camera_active = False
        logger.info(f"[摄像头] 已停止, sid={sid}")


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

            # 获取左右轮速度
            vel_left = car_state_copy.get("vel_left", 0)
            vel_right = car_state_copy.get("vel_right", 0)

            # 状态: 左右轮速度
            state_2d = {
                "vel_left": vel_left,
                "vel_right": vel_right,
            }

            # 当前动作
            actions = list(current_actions)

            # 保存样本到当前轮次
            sample = {
                "image": image_data,  # base64编码的JPEG图像
                "state": state_2d,
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
    global current_actions, inference_mode, connected_clients
    logger.info("[游戏循环] 任务已启动")
    _frame_count = 0
    last_emitted_state = dict(state.car_state)

    while True:
        try:
            _frame_count += 1
            if _frame_count % 100 == 0:
                logger.info(f"[游戏循环] frame={_frame_count}, inference_mode={inference_mode}, current_actions={current_actions}")
            # 根据当前按键更新状态
            if current_actions:
                # 合并所有动作的速度（使用 state.py 中的统一配置）
                combined_vel_left = 0
                combined_vel_right = 0

                for action in current_actions:
                    if action == "forward":
                        combined_vel_left += state.SPEED_FORWARD
                        combined_vel_right += state.SPEED_FORWARD
                    elif action == "backward":
                        combined_vel_left -= state.SPEED_FORWARD
                        combined_vel_right -= state.SPEED_FORWARD
                    elif action == "left":
                        combined_vel_left -= state.SPEED_TURN
                        combined_vel_right += state.SPEED_TURN
                    elif action == "right":
                        combined_vel_left += state.SPEED_TURN
                        combined_vel_right -= state.SPEED_TURN

                # 应用合并后的速度
                state.update_car_state([combined_vel_left, combined_vel_right])
                inference_mode = False  # 手动控制时退出推理模式
            elif inference_mode:
                # 推理模式：使用之前设置的速度，只更新位置（不应用摩擦力）
                vel = [state.car_state["vel_left"], state.car_state["vel_right"]]
                state.update_car_state(vel)
                if _frame_count % 30 == 0:
                    global _debug_game_loop_inference_count
                    _debug_game_loop_inference_count += 1
                    logger.info(
                        f"[游戏循环#{_debug_game_loop_inference_count}] inference_mode=True, vel={vel}, current_actions={current_actions}"
                    )
            else:
                # 没有按键且非推理模式时应用摩擦力减速
                state.apply_friction()

            # 仅在有客户端且状态变化时广播，避免空闲时持续占用 CPU/网络。
            if connected_clients and state.car_state != last_emitted_state:
                await sio_server.emit("car_state_update", state.car_state)
                last_emitted_state = dict(state.car_state)
        except Exception as e:
            logger.error(f"游戏循环错误: {e}")

        is_moving = abs(state.car_state["vel_left"]) > 1e-3 or abs(state.car_state["vel_right"]) > 1e-3
        if not connected_clients:
            sleep_interval = 0.2
        elif current_actions or inference_mode or is_moving:
            sleep_interval = 1 / 30
        else:
            sleep_interval = 0.1

        await asyncio.sleep(sleep_interval)


# 摄像头捕获任务
async def camera_loop_task(sio_server):
    """摄像头循环 - 定期捕获摄像头画面并广播"""
    global camera_active
    import cv2
    import base64

    logger.info("[摄像头] 任务已启动")
    _last_frame_time = 0
    _cap = None

    while True:
        try:
            if not camera_active:
                # 摄像头未激活时释放资源并等待
                if _cap is not None:
                    _cap.release()
                    _cap = None
                    logger.info("[摄像头] 资源已释放")
                await asyncio.sleep(0.2)
                continue

            # 每隔约10帧捕获一次（约10fps）
            current_time = asyncio.get_event_loop().time()
            if current_time - _last_frame_time < 0.01:  # 10ms
                await asyncio.sleep(0.01)
                continue
            _last_frame_time = current_time

            # 打开摄像头（如果尚未打开）
            if _cap is None or not _cap.isOpened():
                _cap = cv2.VideoCapture(0)
                if not _cap.isOpened():
                    logger.warning("[摄像头] 无法打开摄像头")
                    await asyncio.sleep(1)
                    continue

            ret, frame = _cap.read()
            if not ret:
                logger.warning("[摄像头] 无法读取画面")
                _cap.release()
                _cap = None
                await asyncio.sleep(0.5)
                continue

            # 压缩为 JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
            image_bytes = img_encoded.tobytes()

            await sio_server.emit("camera_image", image_bytes)

        except Exception as e:
            logger.error(f"[摄像头] 捕获失败: {e}")
            if _cap is not None:
                _cap.release()
                _cap = None
            await asyncio.sleep(1)

def start_game_loop(sio_server):
    """启动游戏循环"""
    asyncio.create_task(game_loop_task(sio_server))
    asyncio.create_task(camera_loop_task(sio_server))
