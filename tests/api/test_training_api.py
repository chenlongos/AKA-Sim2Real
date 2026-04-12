"""Training API tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import router
from backend.services import training


def setup_app():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_get_training_status_default():
    """GET /api/train/status returns default state."""
    client = setup_app()
    training.training_state["is_running"] = False
    training.training_state["epoch"] = 0
    training.training_state["total_epochs"] = 50
    training.training_state["loss"] = 0.0
    training.training_state["progress"] = 0.0

    response = client.get("/api/train/status")

    assert response.status_code == 200
    data = response.json()
    assert data["is_running"] is False
    assert data["epoch"] == 0
    assert data["total_epochs"] == 50
    assert data["loss"] == 0.0
    assert data["progress"] == 0.0


def test_get_training_status_running():
    """GET /api/train/status reflects running state."""
    client = setup_app()
    training.training_state["is_running"] = True
    training.training_state["epoch"] = 25
    training.training_state["total_epochs"] = 50
    training.training_state["loss"] = 0.123
    training.training_state["progress"] = 50.0

    response = client.get("/api/train/status")

    assert response.status_code == 200
    data = response.json()
    assert data["is_running"] is True
    assert data["epoch"] == 25
    assert data["loss"] == 0.123
    assert data["progress"] == 50.0


def test_post_stop_training():
    """POST /api/train/stop stops training."""
    client = setup_app()
    training.training_state["is_running"] = True

    response = client.post("/api/train/stop")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert training.training_state["is_running"] is False


def test_post_stop_when_not_running():
    """POST /api/train/stop works when not running."""
    client = setup_app()
    training.training_state["is_running"] = False

    response = client.post("/api/train/stop")

    assert response.status_code == 200
    assert response.json()["success"] is True
