from __future__ import annotations

import logging
from typing import Any

from backend.models import state
from backend.sio_handlers.core.runtime import SioRuntimeState

logger = logging.getLogger(__name__)


def extract_velocity_from_action(action: Any) -> tuple[float, float]:
    """Extract the first frame wheel velocities from nested ACT outputs."""
    if not isinstance(action, (list, tuple)) or len(action) == 0:
        raise ValueError(f"无效 action 格式: {action}")

    first = action[0]
    if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
        first = first[0]

    if not isinstance(first, (list, tuple)) or len(first) < 2:
        raise ValueError(f"无法从 action 提取速度: {action}")

    return float(first[0]), float(first[1])


class SimController:
    def __init__(self, runtime: SioRuntimeState):
        self.runtime = runtime

    def reset_act_inference_context(self) -> None:
        try:
            self.runtime.act_runtime.reset_inference_context()
        except Exception as exc:
            logger.debug(f"重置 ACT 推理上下文失败: {exc}")

    def set_actions(self, actions: list[str]) -> None:
        self.runtime.current_actions = set(actions)

        if actions and actions != ["stop"]:
            self.runtime.inference_mode = False
            self.reset_act_inference_context()
            logger.info(f"[on_action] 用户控制，退出推理模式: {actions}")

        if not actions:
            state.car_state["vel_left"] = 0
            state.car_state["vel_right"] = 0

    def reset_car_state(self) -> dict:
        logger.info("重置车辆状态")
        self.reset_act_inference_context()
        state.reset_car_state()
        return state.car_state

    def get_car_state(self) -> dict:
        return state.car_state

    def infer(self, inference_state: list[float], image: Any) -> Any:
        action = self.runtime.act_runtime.infer(inference_state, image)
        vel_left, vel_right = extract_velocity_from_action(action)

        self.runtime.current_actions.clear()
        self.runtime.inference_mode = True
        state.update_car_state([vel_left, vel_right])

        logger.info(f"推理结果: {action[0]}")
        logger.info(f"[on_act_infer] 已应用推理速度: vel_left={vel_left:.4f}, vel_right={vel_right:.4f}")
        return action

    def tick(self) -> bool:
        car_state_before = dict(state.car_state)

        if self.runtime.current_actions:
            combined_vel_left = 0.0
            combined_vel_right = 0.0

            for action in self.runtime.current_actions:
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

            state.update_car_state([combined_vel_left, combined_vel_right])
            self.runtime.inference_mode = False
        elif self.runtime.inference_mode:
            vel = [state.car_state["vel_left"], state.car_state["vel_right"]]
            state.update_car_state(vel)
        else:
            state.apply_friction()

        return state.car_state != car_state_before

    def is_moving(self) -> bool:
        return abs(state.car_state["vel_left"]) > 1e-3 or abs(state.car_state["vel_right"]) > 1e-3
