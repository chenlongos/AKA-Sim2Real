"""End-to-end smoke test for ACT export/train/load/infer pipeline."""

import base64
from pathlib import Path
import asyncio
import io
import sys
import tempfile

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from PIL import Image

from backend.services import act_model, data_export, training


def _sample_image() -> str:
    image = Image.new("RGB", (8, 8), color=(64, 128, 192))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _sample(vel_left: float, vel_right: float, actions):
    return {
        "image": _sample_image(),
        "state": {"vel_left": vel_left, "vel_right": vel_right},
        "actions": actions,
    }


async def _run_smoke_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_dir = tmp_path / "dataset"
        train_dir = tmp_path / "train"

        data_export.reset_metadata()
        samples = [
            _sample(0.10, 0.10, ["forward"]),
            _sample(0.12, 0.11, ["forward"]),
            _sample(0.08, 0.20, ["left"]),
            _sample(0.20, 0.09, ["right"]),
        ]
        data_export.export_episode(samples, episode_id=1, output_dir=str(dataset_dir), chunk_size=10)

        model = await training.train_model(
            sio_server=None,
            data_dir=str(dataset_dir),
            output_dir=str(train_dir),
            epochs=1,
            batch_size=2,
            lr=1e-4,
        )
        assert model is not None

        checkpoint_path = train_dir / "final_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = train_dir / "model.pt"
        assert checkpoint_path.exists(), checkpoint_path

        act_model.load_act_model(str(checkpoint_path), stats_dir=str(dataset_dir))
        action = act_model.act_inference([0.1, 0.1], _sample_image())

        assert isinstance(action, list) and len(action) > 0
        first = action[0]
        if isinstance(first[0], list):
            first = first[0]
        assert len(first) == 2
        print("ACT end-to-end smoke test 通过!")


def main():
    asyncio.run(_run_smoke_test())


if __name__ == "__main__":
    main()
