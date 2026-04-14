import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.sio_handlers.domains.inference import InferenceEventsMixin
from backend.sio_handlers.core.runtime import SioRuntimeState
from backend.models import state


class MockInferenceHandler(InferenceEventsMixin):
    """Testable mock of InferenceEventsMixin."""

    def __init__(self):
        self.runtime = SioRuntimeState()
        self.sim_controller = MagicMock()
        self.emit = AsyncMock()
        self.namespace = "/test"


def make_handler():
    state.reset_car_state()
    return MockInferenceHandler()


def test_on_act_infer_success_emits_state_and_result():
    handler = make_handler()
    handler.sim_controller.infer.return_value = [[0.5, 0.9]]
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.5, "vel_right": 0.9}

    asyncio.run(handler.on_act_infer("sid123", {"state": [1.0, 2.0], "image": "data:image/jpeg;base64,abc"}))

    handler.sim_controller.infer.assert_called_once_with([1.0, 2.0], "data:image/jpeg;base64,abc")
    assert handler.emit.call_count == 2
    assert handler.emit.call_args_list[0][0][0] == "car_state_update"
    assert handler.emit.call_args_list[1][0][0] == "act_infer_result"
    assert handler.emit.call_args_list[1][0][1]["success"] is True


def test_on_act_infer_uses_default_state_when_not_provided():
    handler = make_handler()
    handler.sim_controller.infer.return_value = [[0.0, 0.0]]
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.0, "vel_right": 0.0}

    asyncio.run(handler.on_act_infer("sid123", {"image": "data:image/jpeg;base64,abc"}))

    handler.sim_controller.infer.assert_called_once_with([0.0] * 2, "data:image/jpeg;base64,abc")


def test_on_act_infer_handles_exception():
    handler = make_handler()
    handler.sim_controller.infer.side_effect = RuntimeError("model error")

    asyncio.run(handler.on_act_infer("sid123", {"state": [1.0, 2.0], "image": "data:image/jpeg;base64,abc"}))

    assert handler.emit.call_count == 1
    assert handler.emit.call_args[0][0] == "act_infer_result"
    assert handler.emit.call_args[0][1]["success"] is False
    assert "model error" in handler.emit.call_args[0][1]["error"]


def test_on_act_infer_with_none_image():
    handler = make_handler()
    handler.sim_controller.infer.return_value = [[0.1, 0.2]]
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.1, "vel_right": 0.2}

    asyncio.run(handler.on_act_infer("sid123", {"state": [0.5, 0.5], "image": None}))

    handler.sim_controller.infer.assert_called_once_with([0.5, 0.5], None)


def test_on_act_infer_with_empty_image():
    handler = make_handler()
    handler.sim_controller.infer.return_value = [[0.1, 0.2]]
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.1, "vel_right": 0.2}

    asyncio.run(handler.on_act_infer("sid123", {"state": [0.5, 0.5], "image": ""}))

    handler.sim_controller.infer.assert_called_once_with([0.5, 0.5], "")