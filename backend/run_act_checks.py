"""Unified runner for the current ACT regression/smoke checks."""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    ["python3", "-m", "policies.models.act.test_act_pytorch"],
    ["python3", "backend/test_act_model_backend.py"],
    ["python3", "backend/test_data_export_stats.py"],
    ["python3", "backend/test_act_end_to_end_smoke.py"],
]


def main() -> int:
    for cmd in CHECKS:
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            return result.returncode
    print("All ACT checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
