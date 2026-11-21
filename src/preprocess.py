"""preprocess.py - rebuild chunks and FAISS indexes from the corpus snapshot (R2 pre-step).

A clean clone has no `chunks/` or `indexes/` (they are derived and git-ignored). Retrieval
replay needs both. This entry point runs the two existing steps in order:

    11_chunking.py     corpus_safe_512/ -> chunks/<strategy>/<config>.jsonl
    12_build_index.py  chunks/ + bge-small encoder -> indexes/<strategy>/<config>.faiss

The encoder is `BAAI/bge-small-en-v1.5`, downloaded by `tools/fetch_models.py --base`.

Usage
    python src/preprocess.py --config configs/proposed_model.yaml --dry-run
    python src/preprocess.py --config configs/proposed_model.yaml
    python src/preprocess.py --stage chunks
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg  # noqa: E402


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "proposed_model.yaml"))
    ap.add_argument("--stage", choices=["all", "chunks", "index"], default="all")
    ap.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    conf = cfg.load(Path(args.config))
    device = conf.get("device", "cuda")
    py = sys.executable

    chunks = ROOT / "chunks"
    indexes = ROOT / "indexes"
    do_chunks = args.stage in ("all", "chunks") and (args.force or not any(chunks.glob("*/*.jsonl")))
    do_index = args.stage in ("all", "index") and (args.force or not any(indexes.glob("*/*.faiss")))

    if args.stage in ("all", "chunks") and not do_chunks:
        print("chunks/ already built; use --force to rebuild")
    if args.stage in ("all", "index") and not do_index:
        print("indexes/ already built; use --force to rebuild")

    if do_chunks:
        cmd = [py, str(ROOT / "scripts" / "11_chunking.py"), "--root", str(ROOT)]
        if args.dry_run:
            print("$ " + " ".join(cmd))
        else:
            run(cmd)
    if do_index:
        if not (ROOT / "models" / "reranker").exists():
            print("NOTE: the encoder is downloaded on first use (tools/fetch_models.py --base)")
        cmd = [py, str(ROOT / "scripts" / "12_build_index.py"), "--root", str(ROOT),
               "--device", device]
        if args.dry_run:
            print("$ " + " ".join(cmd))
        else:
            run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
