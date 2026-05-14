from __future__ import annotations

import logging
from typing import Any

from backend.sio_handlers.core.runtime import SioRuntimeState

logger = logging.getLogger(__name__)


def extract_velocity_from_action(action: Any) -> tuple[float, float]:
    """Extract wheel velocities from ACT action output.

    action 格式是单步 [left_vel, right_vel]
    """
    if not isinstance(action, (list, tuple)) or len(action) < 2:
        raise ValueError(f"无效 action 格式: {action}")

    return float(action[0]), float(action[1])


class SimController:
    def __init__(self, runtime: SioRuntimeState):
        self.runtime = runtime

    def reset_act_inference_context(self) -> None:
        try:
            self.runtime.act_runtime.reset_inference_context()
        except Exception as exc:
            logger.debug(f"重置 ACT 推理上下文失败: {exc}")

    def set_action(self, action: list[float]) -> None:
        if len(action) >= 2 and all(isinstance(value, (int, float)) for value in action[:2]):
            self.runtime.current_action_vector = (float(action[0]), float(action[1]))
        else:
            self.runtime.current_action_vector = None

        if action and action != [0, 0]:
            self.runtime.inference_mode = False
            self.reset_act_inference_context()
            logger.info(f"[on_action] 用户控制，退出推理模式: {action}")

        if not action:
            self.runtime.current_action_vector = None

    def infer(self, inference_state: list[float], image: Any, user_id: str = None) -> Any:
        action = self.runtime.act_runtime.infer(inference_state, image, user_id)

        self.runtime.current_action_vector = None
        self.runtime.inference_mode = True

        logger.info(f"推理结果: {action[0]}")
        logger.info(f"[on_act_infer] 进入推理模式")
        return action
