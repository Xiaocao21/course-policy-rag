"""Shared helpers re-exported for the entry points.

The implementation lives in `scripts/common.py`; this module keeps `src/` self-documenting
without duplicating code.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import (PROJECT_ROOT, project_paths, read_jsonl, sha256_file,  # noqa: E402
                    write_json)

__all__ = ["PROJECT_ROOT", "project_paths", "read_jsonl", "write_json", "sha256_file"]
