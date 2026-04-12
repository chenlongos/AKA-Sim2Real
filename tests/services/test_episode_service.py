from backend.models import state
from backend.services.episode.service import EpisodeService


def test_episode_service_collect_data_uses_frontend_state_payload():
    service = EpisodeService()

    state.start_episode(7, "default")
    state.car_state = {"vel_left": 0.0, "vel_right": 0.0}

    count = __import__("asyncio").run(
        service.collect_data(
            "data:image/jpeg;base64,abc123",
            timestamp=2000,
            state_payload={"vel_left": 0.8, "vel_right": 0.9},
        )
    )

    sample = state.episode_samples[7][-1]

    assert count == 1
    assert sample["state"] == {"vel_left": 0.8, "vel_right": 0.9}
    assert sample["capture_timestamp_ms"] == 2000

    state.end_episode(7)
    state.clear_episode_buffer(7)


def test_episode_service_collect_data_preserves_numeric_action_payload():
    service = EpisodeService()

    state.start_episode(8, "default")
    state.car_state = {"vel_left": 0.0, "vel_right": 0.0}

    count = __import__("asyncio").run(
        service.collect_data(
            "data:image/jpeg;base64,abc123",
            timestamp=3000,
            action_payload=[32, -18],
        )
    )

    sample = state.episode_samples[8][-1]

    assert count == 1
    assert sample["action"] == [32.0, -18.0]

    state.end_episode(8)
    state.clear_episode_buffer(8)


def test_episode_service_collect_data_preserves_sim_numeric_action_payload():
    service = EpisodeService()

    state.start_episode(9, "default")
    state.car_state = {"vel_left": 0.0, "vel_right": 0.0}

    count = __import__("asyncio").run(
        service.collect_data(
            "data:image/jpeg;base64,abc123",
            timestamp=4000,
            action_payload=[0.0, 0.2],
        )
    )

    sample = state.episode_samples[9][-1]

    assert count == 1
    assert sample["action"] == [0.0, 0.2]

    state.end_episode(9)
    state.clear_episode_buffer(9)
