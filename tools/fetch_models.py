"""fetch_models.py - install and verify the frozen reranker and the base encoders.

The checkpoint used for every reported result is `547060da…` (partial fine-tune, 10/12 encoder
layers frozen). It is published as release assets because it is 1.06 GB. This tool downloads
the assets, verifies every SHA-256, and lays them out exactly where `19_rerank_eval.py` expects
them:

    models/reranker/FROZEN.json
    models/reranker/runs/rk-1e-05-n4-s20261027-partial-3060-FROZEN/
        model.safetensors  config.json  tokenizer.json  tokenizer_config.json  train_log.json

`19` loads the tokenizer from the checkpoint directory (not from the base model), so these files
must be the checkpoint's own copies.

Usage
    python tools/fetch_models.py --check
    python tools/fetch_models.py --download
    python tools/fetch_models.py --download --base        # also fetch the two HF base models
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ("https://github.com/Xiaocao21/course-policy-rag/releases/download/v0.1.5-repro")
RUN = "models/reranker/runs/rk-1e-05-n4-s20261027-partial-3060-FROZEN"
ASSETS: dict[str, tuple[str, str]] = {
    f"{RUN}/model.safetensors": (
        "reranker-FROZEN-model.safetensors",
        "547060da6310cfd6cb31b75521f75c6af4161aa916196330ef1492ee91719ba2"),
    f"{RUN}/config.json": (
        "reranker-FROZEN-config.json",
        "f8921fc136966127e56b76518da385bade346820a18178a3637b6bb891341739"),
    f"{RUN}/tokenizer.json": (
        "reranker-FROZEN-tokenizer.json",
        "7132ed20a040fba715eb868be52c6a7278ce29f0985dae642cd91e573b8d4f3f"),
    f"{RUN}/tokenizer_config.json": (
        "reranker-FROZEN-tokenizer_config.json",
        "7f8e591fccaa155b910473a53702be26990095e8904323a4241f7750ecc0273e"),
    f"{RUN}/train_log.json": (
        "reranker-FROZEN-train_log.json",
        "8d7ef1d4671c41c4094e63a8de3609a7a1b27e6e34ffe982470ac0b9235b4851"),
    "models/reranker/FROZEN.json": (
        "FROZEN.json",
        "68e06ad79e7ab3c49ad4bcb0687c394a2eddad754872cdb9f0e34b7aeb21911d"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root: Path) -> int:
    bad = []
    for rel, (_, want) in ASSETS.items():
        path = root / rel
        if not path.exists():
            bad.append(f"missing {rel}")
        elif sha256(path) != want:
            bad.append(f"hash mismatch {rel}")
    for line in bad:
        print(f"FAIL: {line}")
    print(f"assets: {len(ASSETS) - len(bad)}/{len(ASSETS)} ok")
    return 0 if not bad else 1


def download(root: Path) -> int:
    for rel, (asset, want) in ASSETS.items():
        target = root / rel
        if target.exists() and sha256(target) == want:
            print(f"ok      {rel}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / asset
            url = f"{RELEASE}/{asset}"
            print(f"download {asset} ({url})")
            with urllib.request.urlopen(url, timeout=600) as response, \
                    tmp_path.open("wb") as f:
                shutil.copyfileobj(response, f)
            got = sha256(tmp_path)
            if got != want:
                print(f"FAIL: {asset} sha256 {got} != {want}", file=sys.stderr)
                return 1
            shutil.move(str(tmp_path), target)
            print(f"wrote   {rel}")
    return check(root)


def download_base() -> int:
    import os
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download
    for model in ("BAAI/bge-small-en-v1.5", "BAAI/bge-reranker-base"):
        path = snapshot_download(model)
        print(f"cached  {model} -> {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--base", action="store_true", help="also download the HF base models")
    args = ap.parse_args()
    root = Path(args.root)
    if args.download:
        code = download(root)
        if code:
            return code
    if args.base:
        return download_base()
    return check(root)


if __name__ == "__main__":
    raise SystemExit(main())
