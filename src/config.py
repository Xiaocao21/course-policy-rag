"""Load a reproduction/training configuration from YAML and validate it."""
from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED = ("name", "split", "out")


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise SystemExit(f"REFUSING: {path} is missing keys: {', '.join(missing)}")
    if data["split"] not in ("dev", "test", "test2"):
        raise SystemExit(f"REFUSING: split {data['split']!r} is not dev/test/test2")
    return data
