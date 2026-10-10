"""Validate the Python environment and required SkyGuard project resources."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.dependency_check import environment_report, required_dependencies_missing


def main() -> int:
    print(environment_report())
    required_dirs = [
        "data/sample", "data/unknown_faults", "models", "logs",
    ]
    print("\nProject resources")
    for relative in required_dirs:
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] folder: {relative}")
    config = ROOT / "config.yaml"
    print(f"[{'OK' if config.exists() else 'MISSING'}] configuration: {config.name}")
    model = ROOT / "models/root_cause_classifier.joblib"
    print(f"[{'OK' if model.exists() else 'INFO'}] model: {model.name} ({'available' if model.exists() else 'train before dashboard use'})")
    missing = required_dependencies_missing()
    if missing:
        print("\nRequired dependencies are missing. Install the commands shown above.")
        return 1
    print("\nEnvironment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())