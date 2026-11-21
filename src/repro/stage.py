"""Build a temporary root that isolates a reproduction run from the official tree.

`19_rerank_eval.py` reads its inputs and appends to `experiments/log.jsonl` under `--root`.
Running it against the repository would therefore touch official artefacts. Instead we build a
throw-away root whose large/read-only directories are junctions (Windows) or symlinks (POSIX):
the run writes its results and log inside the temporary root, and the repository is untouched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

READ_ONLY = ("scripts", "config", "annotations", "models", "indexes", "chunks",
             "cleaned", "corpus_safe_512", "reports")


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"REFUSING: cannot create junction {dst} -> {src}: "
                             f"{result.stderr or result.stdout}")
    else:
        os.symlink(src, dst, target_is_directory=True)


def build(root: Path, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    for name in READ_ONLY:
        src = root / name
        if src.exists():
            link(src, work / name)
    (work / "results").mkdir(exist_ok=True)
    test2 = root / "results" / "test2"
    if test2.exists():
        link(test2, work / "results" / "test2")
    (work / "experiments").mkdir(exist_ok=True)
    # goldset.load_corpus selects documents from root/sources.csv (include=TRUE); without this
    # file the trajectory would load an empty corpus and all metrics would be zero.
    for name in ("sources.csv", "MANIFEST.json"):
        src = root / name
        if src.exists():
            shutil.copy2(src, work / name)
    return work


def _remove_link(path: Path) -> None:
    try:
        if path.is_symlink():
            path.unlink()
        elif getattr(path, "is_junction", lambda: False)():
            os.rmdir(path)
    except OSError:
        pass


def cleanup(work: Path) -> None:
    """Remove junctions/symlinks first so rmtree can never follow into the repository."""
    for name in READ_ONLY:
        _remove_link(work / name)
    _remove_link(work / "results" / "test2")
    shutil.rmtree(work, ignore_errors=True)
