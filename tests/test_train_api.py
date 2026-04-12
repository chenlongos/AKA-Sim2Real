from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api as api_module
from backend.api import router
from backend.services import training


def test_train_api_reads_json_body(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    captured = {}

    async def fake_train_model(*args, **kwargs):
        captured.update(kwargs)
        return None

    def fake_create_task(coro):
        if coro.cr_frame and "kwargs" in coro.cr_frame.f_locals:
            captured.update(coro.cr_frame.f_locals["kwargs"])
        coro.close()
        return None

    monkeypatch.setattr(training, "train_model", fake_train_model)
    monkeypatch.setattr(api_module.asyncio, "create_task", fake_create_task)
    monkeypatch.setitem(training.training_state, "is_running", False)

    response = client.post(
        "/api/train",
        json={
            "data_dir": str(dataset_dir),
            "output_dir": "checkpoints",
            "epochs": 7,
            "batch_size": 4,
            "lr": 0.002,
            "resume_from": "output/train/model.pt",
        },
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert captured["data_dir"] == str(dataset_dir)
    assert "checkpoints" in captured["output_dir"]
    assert captured["epochs"] == 7
    assert captured["batch_size"] == 4
    assert captured["lr"] == 0.002
    assert captured["resume_from"].endswith("output/train/model.pt")
