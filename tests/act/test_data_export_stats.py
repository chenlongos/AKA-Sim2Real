"""Minimal tests for data export statistics aggregation."""

import numpy as np

from backend.services.episode import LeRobotDatasetMetadata


SAMPLE_IMAGE = (
    "data:image/jpeg;base64,"
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
    "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/"
    "8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/"
    "9oADAMBEQCEAwEPwAB//9k="
)


def _sample(vel_left: float, vel_right: float, action=None):
    return {
        "image": SAMPLE_IMAGE,
        "state": {"vel_left": vel_left, "vel_right": vel_right},
        "action": action,
    }


def test_stats_aggregation(tmp_path):
    metadata = LeRobotDatasetMetadata(output_dir=tmp_path, chunk_size=10)
    metadata.save_episode(
        [
            _sample(1.0, 2.0, [0.2, 0.2]),
            _sample(3.0, 4.0, [-0.2, 0.2]),
        ],
        episode_id=1,
    )
    metadata.save_episode(
        [
            _sample(5.0, 6.0, [-0.2, -0.2]),
            _sample(7.0, 8.0, [0.2, -0.2]),
        ],
        episode_id=2,
    )

    stats = metadata.get_stats()
    state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float64)
    state_std = np.array(stats["observation.state"]["std"], dtype=np.float64)

    expected_states = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=np.float64,
    )
    expected_mean = expected_states.mean(axis=0)
    expected_std = expected_states.std(axis=0) + 1e-6

    assert np.allclose(state_mean, expected_mean)
    assert np.allclose(state_std, expected_std)
    assert stats["observation.state"]["count"] == 4
    assert "count" in stats["action"]
    print("data_export stats 聚合测试通过!")


def test_export_defaults_to_zero_when_numeric_action_missing(tmp_path):
    metadata = LeRobotDatasetMetadata(output_dir=tmp_path, chunk_size=10)
    metadata.save_episode(
        [
            _sample(1.0, 2.0),
        ],
        episode_id=1,
    )

    parquet_file = next((tmp_path / "data").glob("chunk-*/file-*.parquet"))
    df = __import__("pandas").read_parquet(parquet_file)
    action = np.asarray(df.iloc[0]["action"].tolist(), dtype=np.float32)

    assert action.shape[-1] == 2
    assert np.allclose(action[0], np.array([0.0, 0.0], dtype=np.float32))
