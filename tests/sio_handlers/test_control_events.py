from unittest.mock import AsyncMock, MagicMock

import asyncio

from backend.sio_handlers.domains.control import ControlEventsMixin
from backend.sio_handlers.core.runtime import SioRuntimeState
from backend.models import state


class MockControlHandler(ControlEventsMixin):
    """Testable mock of ControlEventsMixin."""

    def __init__(self):
        self.runtime = SioRuntimeState()
        self.sim_controller = MagicMock()
        self.emit = AsyncMock()
        self.namespace = "/test"


def make_handler():
    state.reset_car_state()
    return MockControlHandler()


def test_on_connect_adds_client_and_emits_state():
    handler = make_handler()
    assert len(handler.runtime.connected_clients) == 0

    asyncio.run(handler.on_connect("sid123", {}, auth={"token": "abc"}))

    assert "sid123" in handler.runtime.connected_clients
    assert handler.emit.call_count == 2
    first_call_event = handler.emit.call_args_list[0][0][0]
    second_call_event = handler.emit.call_args_list[1][0][0]
    assert first_call_event == "connected"
    assert second_call_event == "car_state_update"


def test_on_disconnect_removes_client():
    handler = make_handler()
    handler.runtime.connected_clients.add("sid123")

    asyncio.run(handler.on_disconnect("sid123"))

    assert "sid123" not in handler.runtime.connected_clients


def test_on_disconnect_clears_action_vector_when_last_client():
    handler = make_handler()
    handler.runtime.connected_clients.add("sid123")
    handler.runtime.current_action_vector = (0.5, 0.5)

    asyncio.run(handler.on_disconnect("sid123"))

    assert handler.runtime.current_action_vector is None


def test_on_disconnect_preserves_action_vector_with_other_clients():
    handler = make_handler()
    handler.runtime.connected_clients.add("sid1")
    handler.runtime.connected_clients.add("sid2")
    handler.runtime.current_action_vector = (0.5, 0.5)

    asyncio.run(handler.on_disconnect("sid1"))

    assert handler.runtime.current_action_vector == (0.5, 0.5)


def test_on_action_sets_action_on_sim_controller():
    handler = make_handler()
    handler.sim_controller.set_action = MagicMock()

    asyncio.run(handler.on_action("sid123", [0.8, 0.9]))

    handler.sim_controller.set_action.assert_called_once_with([0.8, 0.9])


def test_on_reset_car_state_resets_and_emits():
    handler = make_handler()
    expected_state = {"vel_left": 0.0, "vel_right": 0.0}
    handler.sim_controller.reset_car_state.return_value = expected_state

    asyncio.run(handler.on_reset_car_state("sid123"))

    handler.sim_controller.reset_car_state.assert_called_once()
    handler.emit.assert_called_once_with("car_state_update", expected_state)


def test_on_get_car_state_emits_current_state():
    handler = make_handler()
    expected_state = {"vel_left": 1.0, "vel_right": 2.0}
    handler.sim_controller.get_car_state.return_value = expected_state

    asyncio.run(handler.on_get_car_state("sid123"))

    handler.sim_controller.get_car_state.assert_called_once()
    handler.emit.assert_called_once_with("car_state_update", expected_state)