"""Inference API tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import router
from backend.services import inference as inference_service


def setup_app():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_infer_returns_random_action_when_no_model(monkeypatch):
    """POST /api/act/infer returns fallback action when no model loaded."""
    client = setup_app()

    runtime = inference_service.get_act_runtime()
    runtime.model = None

    response = client.post(
        "/api/act/infer",
        json={"state": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["action"] == [[0.0, 0.0]]


def test_infer_accepts_state_only(monkeypatch):
    """POST /api/act/infer works with state only, no image."""
    client = setup_app()

    runtime = inference_service.get_act_runtime()
    runtime.model = None

    response = client.post(
        "/api/act/infer",
        json={"state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "action" in data


def test_infer_accepts_state_and_image(monkeypatch):
    """POST /api/act/infer accepts both state and image."""
    client = setup_app()

    runtime = inference_service.get_act_runtime()
    runtime.model = None

    response = client.post(
        "/api/act/infer",
        json={
            "state": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "image": "data:image/jpeg;base64,abc123",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_infer_returns_500_on_exception(monkeypatch):
    """POST /api/act/infer returns 500 on inference error."""
    client = setup_app()

    def fake_infer(*args, **kwargs):
        raise RuntimeError("Fake inference error")

    runtime = inference_service.get_act_runtime()
    monkeypatch.setattr(runtime, "infer", fake_infer)

    response = client.post(
        "/api/act/infer",
        json={"state": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
    )

    assert response.status_code == 500
    assert "Fake inference error" in response.json()["detail"]


def test_load_trained_model_success(monkeypatch):
    """POST /api/act/load_trained loads model successfully."""
    client = setup_app()

    runtime = inference_service.get_act_runtime()

    def fake_load_model(model_path=None, stats_dir=None):
        return runtime.model

    monkeypatch.setattr(runtime, "load_model", fake_load_model)

    response = client.post(
        "/api/act/load_trained?model_path=/fake/path.pt",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "模型加载成功"


def test_load_trained_model_returns_500_on_error(monkeypatch):
    """POST /api/act/load_trained returns 500 on load error."""
    client = setup_app()

    runtime = inference_service.get_act_runtime()

    def fake_load_model_fail(*args, **kwargs):
        raise FileNotFoundError("Model file not found")

    monkeypatch.setattr(runtime, "load_model", fake_load_model_fail)

    response = client.post(
        "/api/act/load_trained?model_path=/nonexistent/model.pt",
    )

    assert response.status_code == 500
    assert "Model file not found" in response.json()["detail"]


def test_health_endpoint(monkeypatch):
    """GET /health returns health status."""
    client = setup_app()

    runtime = inference_service.get_act_runtime()
    monkeypatch.setattr(runtime, "model", None)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False


def test_health_endpoint_model_loaded(monkeypatch):
    """GET /health reflects model loaded state."""
    client = setup_app()

    runtime = inference_service.get_act_runtime()
    monkeypatch.setattr(runtime, "model", object())

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
