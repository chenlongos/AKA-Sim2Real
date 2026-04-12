"""Episode API tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import router
from backend.models import state


def setup_app():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_get_dataset_empty():
    """GET /api/dataset returns empty list when no samples."""
    client = setup_app()

    response = client.get("/api/dataset")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["samples"] == []


def test_get_dataset_with_samples():
    """GET /api/dataset returns stored samples."""
    client = setup_app()

    state.dataset_samples = [
        {"observation": {"image": "abc"}, "action": [1.0, 2.0]},
        {"observation": {"image": "def"}, "action": [3.0, 4.0]},
    ]

    response = client.get("/api/dataset")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["samples"]) == 2

    state.dataset_samples = []


def test_post_dataset_saves_sample():
    """POST /api/dataset saves a sample."""
    client = setup_app()
    state.dataset_samples = []

    response = client.post(
        "/api/dataset",
        json={
            "observation": {"image": "data:image/jpeg;base64,xyz"},
            "action": {"left": 0.5, "right": 0.6},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["samples_count"] == 1
    assert len(state.dataset_samples) == 1
    assert state.dataset_samples[0]["observation"]["image"] == "data:image/jpeg;base64,xyz"

    state.dataset_samples = []


def test_post_dataset_increments_count():
    """POST /api/dataset increments sample count."""
    client = setup_app()
    state.dataset_samples = []

    for i in range(3):
        response = client.post(
            "/api/dataset",
            json={
                "observation": {"idx": i},
                "action": {"left": float(i), "right": float(i)},
            },
        )
        assert response.status_code == 200
        assert response.json()["samples_count"] == i + 1

    state.dataset_samples = []


def test_delete_dataset_clears_all():
    """DELETE /api/dataset clears all samples."""
    client = setup_app()
    state.dataset_samples = [
        {"observation": {}, "action": {}},
        {"observation": {}, "action": {}},
    ]

    response = client.delete("/api/dataset")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert len(state.dataset_samples) == 0


def test_delete_dataset_empty_is_ok():
    """DELETE /api/dataset works when already empty."""
    client = setup_app()
    state.dataset_samples = []

    response = client.delete("/api/dataset")

    assert response.status_code == 200
    assert response.json()["success"] is True
