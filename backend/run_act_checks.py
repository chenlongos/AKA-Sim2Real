"""Unified runner for the current ACT regression/smoke checks."""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTEST_CMD = ["python3", "-m", "pytest", "tests/act", "-q"]


def main() -> int:
    print(f"$ {' '.join(PYTEST_CMD)}")
    result = subprocess.run(PYTEST_CMD, cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
