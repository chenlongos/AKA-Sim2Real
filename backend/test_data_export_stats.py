"""Minimal tests for data export statistics aggregation."""

from pathlib import Path
import sys
import tempfile

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.services.data_export import LeRobotDatasetMetadata


SAMPLE_IMAGE = (
    "data:image/jpeg;base64,"
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
    "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/"
    "8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/"
    "9oADAMBEQCEAwEPwAB//9k="
)


def _sample(vel_left: float, vel_right: float, actions):
    return {
        "image": SAMPLE_IMAGE,
        "state": {"vel_left": vel_left, "vel_right": vel_right},
        "actions": actions,
    }


def test_stats_aggregation():
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = LeRobotDatasetMetadata(output_dir=tmpdir, chunk_size=10)
        metadata.save_episode(
            [
                _sample(1.0, 2.0, ["forward"]),
                _sample(3.0, 4.0, ["left"]),
            ],
            episode_id=1,
        )
        metadata.save_episode(
            [
                _sample(5.0, 6.0, ["backward"]),
                _sample(7.0, 8.0, ["right"]),
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


if __name__ == "__main__":
    test_stats_aggregation()
