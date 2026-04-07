from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import router
from backend.api.domains.episode import routes as episode_routes


def test_dataset_collect_api_accepts_frontend_image(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    captured = {}

    async def fake_collect_data(image, *, timestamp=None, state_payload=None, action_payload=None):
        captured["image"] = image
        captured["timestamp"] = timestamp
        captured["state_payload"] = state_payload
        captured["action_payload"] = action_payload
        return 12

    monkeypatch.setattr(episode_routes.episode_service, "collect_data", fake_collect_data)

    response = client.post(
        "/api/dataset/collect",
        json={
            "image": "data:image/jpeg;base64,abc123",
            "timestamp": 1234567890,
            "state": {"vel_left": 0.8, "vel_right": 0.9},
            "action": [25, -25],
        },
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["count"] == 12
    assert captured == {
        "image": "data:image/jpeg;base64,abc123",
        "timestamp": 1234567890,
        "state_payload": {"vel_left": 0.8, "vel_right": 0.9},
        "action_payload": [25.0, -25.0],
    }


def test_dataset_collect_api_requires_recording(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    async def fake_collect_data(image, **kwargs):
        return None

    monkeypatch.setattr(episode_routes.episode_service, "collect_data", fake_collect_data)

    response = client.post(
        "/api/dataset/collect",
        json={"image": "data:image/jpeg;base64,abc123"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "当前未处于录制状态"
