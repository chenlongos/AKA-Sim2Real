from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import router
from backend.api.domains.control import routes as control_routes
from backend.models import state
from backend.services.episode.service import EpisodeService
from backend.services.gateways import car_gateway


def test_car_motor_status_route_requests_once_directly(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    async def fake_get_motor_status(car_ip: str, timestamp: int):
        assert car_ip == "192.168.1.8"
        assert timestamp == 1234
        return {"ok": True, "timestamp": 1250, "vel_left": 1.5, "vel_right": 2.5}

    monkeypatch.setattr(control_routes.car_gateway, "get_motor_status", fake_get_motor_status)

    response = client.get("/api/car/motor_status", params={"car_ip": "192.168.1.8", "timestamp": 1234})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "query_timestamp_ms": 1234,
        "matched_timestamp_ms": 1250,
        "delta_ms": 16,
        "source": "direct_request",
        "state": {
            "vel_left": 1.5,
            "vel_right": 2.5,
        },
        "payload": {"ok": True, "timestamp": 1250, "vel_left": 1.5, "vel_right": 2.5},
    }


def test_car_motor_direct_route_passes_direct_values(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    captured = {}

    async def fake_motor_direct(car_ip: str, left: int, right: int):
        captured["car_ip"] = car_ip
        captured["left"] = left
        captured["right"] = right
        return {"ok": True, "status": 200, "left": left, "right": right, "payload": {"applied": True}}

    monkeypatch.setattr(control_routes.car_gateway, "motor_direct", fake_motor_direct)

    response = client.post(
        "/api/car/motor_direct",
        params={"car_ip": "192.168.1.8", "left": 6, "right": -12},
    )

    assert response.status_code == 200
    assert captured == {"car_ip": "192.168.1.8", "left": 6, "right": -12}
    assert response.json()["ok"] is True


async def _fake_get_motor_status(car_ip: str, timestamp: int):
    return {
        "ok": True,
        "data": {
            "timestamp_ms": timestamp + 5,
            "left_speed": 0.8,
            "right_speed": 0.9,
        },
    }


def test_episode_service_collect_data_fetches_one_state_per_frame(monkeypatch):
    service = EpisodeService()

    state.start_episode(7, "default")
    state.car_state = {"vel_left": 0.0, "vel_right": 0.0}

    calls: list[tuple[str, int]] = []

    async def fake_get_motor_status(car_ip: str, timestamp: int):
        calls.append((car_ip, timestamp))
        return await _fake_get_motor_status(car_ip, timestamp)

    monkeypatch.setattr(car_gateway, "get_motor_status", fake_get_motor_status)

    count = __import__("asyncio").run(
        service.collect_data(
            "data:image/jpeg;base64,abc123",
            ["forward"],
            car_ip="192.168.1.9",
            timestamp=2000,
        )
    )

    sample = state.episode_samples[7][-1]

    assert count == 1
    assert calls == [("192.168.1.9", 2000)]
    assert sample["state"] == {"vel_left": 0.8, "vel_right": 0.9}
    assert sample["motor_status"]["data"]["left_speed"] == 0.8
    assert sample["motor_status"]["data"]["right_speed"] == 0.9

    state.end_episode(7)
    state.clear_episode_buffer(7)
