import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.sio_handlers.domains.episode import EpisodeEventsMixin
from backend.sio_handlers.core.runtime import SioRuntimeState
from backend.models import state


class MockEpisodeHandler(EpisodeEventsMixin):
    """Testable mock of EpisodeEventsMixin."""

    def __init__(self):
        self.runtime = SioRuntimeState()
        self.episode_service = MagicMock()
        self.sim_controller = MagicMock()
        self.emit = AsyncMock()
        self.namespace = "/test"


def make_handler():
    # Reset global state between tests
    state.reset_car_state()
    state.current_episode_id = 1
    return MockEpisodeHandler()


def test_on_start_episode_emits_started_and_car_state():
    handler = make_handler()
    handler.episode_service.start_episode.return_value = {"episode_id": 5, "task_name": "test", "frame_count": 0}
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.0, "vel_right": 0.0}

    asyncio.run(handler.on_start_episode("sid123", {"episode_id": 5, "task_name": "test"}))

    handler.episode_service.start_episode.assert_called_once_with(5, "test")
    assert handler.emit.call_count == 2
    assert handler.emit.call_args_list[0][0][0] == "car_state_update"
    assert handler.emit.call_args_list[1][0][0] == "episode_started"
    assert handler.emit.call_args_list[1][0][1]["episode_id"] == 5


def test_on_start_episode_uses_default_episode_id_and_task_name():
    handler = make_handler()
    state.current_episode_id = 7
    handler.episode_service.start_episode.return_value = {"episode_id": 7, "task_name": "default", "frame_count": 0}
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.0, "vel_right": 0.0}

    asyncio.run(handler.on_start_episode("sid123", {}))

    handler.episode_service.start_episode.assert_called_once_with(7, "default")


def test_on_end_episode_emits_episode_ended():
    handler = make_handler()
    handler.episode_service.end_episode.return_value = {"episode_id": 5, "frame_count": 10, "exported": True}

    asyncio.run(handler.on_end_episode("sid123", {"episode_id": 5}))

    handler.episode_service.end_episode.assert_called_once_with(5)
    handler.emit.assert_called_once_with("episode_ended", {"episode_id": 5, "frame_count": 10, "exported": True})


def test_on_finalize_episode_emits_when_result_not_none():
    handler = make_handler()
    handler.episode_service.finalize_episode.return_value = {"episode_id": 5, "frame_count": 20, "output_path": "/tmp/ep5"}

    asyncio.run(handler.on_finalize_episode("sid123", {"episode_id": 5}))

    handler.episode_service.finalize_episode.assert_called_once_with(5)
    handler.emit.assert_called_once_with("episode_finalized", {"episode_id": 5, "frame_count": 20, "output_path": "/tmp/ep5"})


def test_on_finalize_episode_does_not_emit_when_result_is_none():
    handler = make_handler()
    handler.episode_service.finalize_episode.return_value = None

    asyncio.run(handler.on_finalize_episode("sid123", {"episode_id": 5}))

    handler.emit.assert_not_called()


def test_on_set_episode_emits_episode_info():
    handler = make_handler()
    handler.episode_service.set_episode.return_value = {"current_episode": 7, "episodes": {7: 0}, "buffer_size": 0}

    asyncio.run(handler.on_set_episode("sid123", 7))

    handler.episode_service.set_episode.assert_called_once_with(7)
    handler.emit.assert_called_once()
    call_args = handler.emit.call_args[0]
    assert call_args[0] == "episode_info"
    assert call_args[1]["current_episode"] == 7


def test_on_get_episodes_emits_episode_info():
    handler = make_handler()
    handler.episode_service.get_episodes_info.return_value = {"current_episode": 1, "episodes": {}, "buffer_size": 0}

    asyncio.run(handler.on_get_episodes("sid123"))

    handler.episode_service.get_episodes_info.assert_called_once()
    handler.emit.assert_called_once_with("episode_info", {"current_episode": 1, "episodes": {}, "buffer_size": 0})


def test_on_delete_episode_emits_when_info_not_none():
    handler = make_handler()
    handler.episode_service.delete_episode.return_value = {"current_episode": 1, "episodes": {}, "buffer_size": 0}

    asyncio.run(handler.on_delete_episode("sid123", {"episode_id": 5}))

    handler.episode_service.delete_episode.assert_called_once_with(5)
    handler.emit.assert_called_once()


def test_on_delete_episode_does_not_emit_when_info_is_none():
    handler = make_handler()
    handler.episode_service.delete_episode.return_value = None

    asyncio.run(handler.on_delete_episode("sid123", {"episode_id": None}))

    handler.emit.assert_not_called()


def test_on_get_episode_status_emits_status():
    handler = make_handler()
    handler.episode_service.get_episode_status.return_value = {"episode_id": 1, "is_recording": False, "frame_count": 0, "task_name": "default"}

    asyncio.run(handler.on_get_episode_status("sid123"))

    handler.episode_service.get_episode_status.assert_called_once()
    handler.emit.assert_called_once_with("episode_status", {"episode_id": 1, "is_recording": False, "frame_count": 0, "task_name": "default"})


def test_on_collect_data_emits_count_every_10_samples():
    handler = make_handler()
    state.start_episode(1, "test")
    handler.episode_service.collect_data = AsyncMock(return_value=10)
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.0, "vel_right": 0.0}

    asyncio.run(handler.on_collect_data("sid123", {"image": "data:image/jpeg;base64,abc", "timestamp": 1000}))

    handler.emit.assert_called_once()
    call_args = handler.emit.call_args[0]
    assert call_args[0] == "collection_count"
    assert call_args[1]["count"] == 10
    state.end_episode(1)
    state.clear_episode_buffer(1)


def test_on_collect_data_does_not_emit_count_for_non_multiples_of_10():
    handler = make_handler()
    state.start_episode(1, "test")
    handler.episode_service.collect_data = AsyncMock(return_value=5)
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.0, "vel_right": 0.0}

    asyncio.run(handler.on_collect_data("sid123", {"image": "data:image/jpeg;base64,abc", "timestamp": 1000}))

    handler.emit.assert_not_called()
    state.end_episode(1)
    state.clear_episode_buffer(1)


def test_on_collect_data_handles_exception():
    handler = make_handler()
    handler.episode_service.collect_data = AsyncMock(side_effect=Exception("db error"))
    handler.sim_controller.get_car_state.return_value = {"vel_left": 0.0, "vel_right": 0.0}

    # Should not raise
    asyncio.run(handler.on_collect_data("sid123", {"image": "data:image/jpeg;base64,abc"}))