from __future__ import annotations

import logging
from typing import Any

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

    def infer(self, inference_state: list[float], image: Any) -> Any:
        action = self.runtime.act_runtime.infer(inference_state, image)

        self.runtime.current_action_vector = None
        self.runtime.inference_mode = True

        logger.info(f"推理结果: {action[0]}")
        logger.info(f"[on_act_infer] 进入推理模式")
        return action
