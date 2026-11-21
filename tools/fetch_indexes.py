"""fetch_indexes.py - install and verify the official FAISS indexes.

Exact retrieval replay cannot rebuild the indexes: FAISS files embed float32 vectors that vary
with the BLAS/GPU stack, and the test2 pre-registration pins their hashes. This tool downloads
the published archive, verifies every file against `PREREGISTRATION.json`, and installs them.

Usage
    python tools/fetch_indexes.py --url https://github.com/Xiaocao21/course-policy-rag/releases/download/v0.1.5-repro/indexes_v0.1.5.tar.gz
    python tools/fetch_indexes.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "data" / "INDEX_STATUS.json"
DEFAULT_URL = ("https://github.com/Xiaocao21/course-policy-rag/releases/download/"
               "v0.1.5-repro/indexes_v0.1.5.tar.gz")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected(root: Path) -> dict[str, str]:
    prereg = json.loads((root / "results" / "test2" / "PREREGISTRATION.json")
                        .read_text(encoding="utf-8"))
    return {k: v for k, v in prereg["artifacts"].items() if k.startswith("indexes/")}


def check(root: Path) -> tuple[bool, list[str]]:
    mismatched = []
    for rel, want in expected(root).items():
        path = root / rel
        if not path.exists() or sha256(path) != want:
            mismatched.append(rel)
    return not mismatched, mismatched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if args.check:
        ok, bad = check(root)
        print(f"indexes: {'ok' if ok else f'{len(bad)} mismatch'}")
        for rel in bad[:5]:
            print(f"  {rel}")
        return 0 if ok else 1
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        archive = staging / "indexes.tar.gz"
        print(f"downloading {args.url}")
        with urllib.request.urlopen(args.url, timeout=600) as response, \
                archive.open("wb") as f:
            shutil.copyfileobj(response, f)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target = (staging / member.name).resolve()
                if staging.resolve() not in target.parents and target != staging.resolve():
                    print(f"REFUSING: unsafe path {member.name}", file=sys.stderr)
                    return 2
            tar.extractall(staging, filter="data")
        manifest_path = staging / "INDEX_MANIFEST.json"
        if not manifest_path.exists():
            print("REFUSING: INDEX_MANIFEST.json missing", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatched = []
        for entry in manifest["files"]:
            rel = entry["path"]
            path = staging / rel
            if not path.exists() or sha256(path) != entry["sha256"]:
                mismatched.append(rel)
        if mismatched:
            print(f"REFUSING: {len(mismatched)} files in the archive do not match its manifest",
                  file=sys.stderr)
            return 2
        for entry in manifest["files"]:
            rel = entry["path"]
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging / rel, dst)
    ok, bad = check(root)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"verified": ok, "mismatched": bad}, indent=1),
                      encoding="utf-8")
    print(f"installed indexes; verification: {'ok' if ok else f'{len(bad)} mismatch'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
