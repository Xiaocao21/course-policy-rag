"""fetch_data.py - install and verify the author-provided corpus snapshot.

The live rebuild of v0.1.5 was measured and failed (data/REBUILD_REPORT.md): 18 of the 36
required pages changed and 17 could not be fetched. Retrieval replay therefore uses the
snapshot produced by `make_corpus_snapshot.py`, checked file by file against MANIFEST.

Usage
    python tools/fetch_data.py --snapshot dist/corpus_v0.1.5.tar.gz --verify
    python tools/fetch_data.py --snapshot https://.../corpus_v0.1.5.tar.gz --verify
    python tools/fetch_data.py --status
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
STATUS = ROOT / "data" / "SNAPSHOT_STATUS.json"
CLEANED_LAYERS = ("evidence", "retrieval", "charmap")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(source: str, dest: Path) -> Path:
    if source.startswith(("http://", "https://")):
        print(f"downloading {source}")
        with urllib.request.urlopen(source, timeout=120) as response, dest.open("wb") as f:
            shutil.copyfileobj(response, f)
    else:
        src = Path(source)
        if not src.exists():
            raise SystemExit(f"REFUSING: {src} does not exist")
        shutil.copyfile(src, dest)
    return dest


def verify(root: Path, archive: Path, staging: Path) -> tuple[bool, dict]:
    expected_archive = archive.with_suffix(archive.suffix + ".sha256")
    if expected_archive.exists():
        want = expected_archive.read_text(encoding="utf-8").strip()
        if sha256(archive) != want:
            return False, {"reason": "archive sha256 does not match its .sha256 file"}
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            target = (staging / member.name).resolve()
            if staging.resolve() not in target.parents and target != staging.resolve():
                return False, {"reason": f"unsafe path in archive: {member.name}"}
        tar.extractall(staging, filter="data")
    manifest_path = staging / "SNAPSHOT_MANIFEST.json"
    if not manifest_path.exists():
        return False, {"reason": "SNAPSHOT_MANIFEST.json missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))["artefact_hashes"]
    missing, mismatched = [], []
    for entry in manifest["files"]:
        rel = entry["path"]
        path = staging / rel
        if not path.exists():
            missing.append(rel)
            continue
        got = sha256(path)
        layer = entry["layer"]
        want = frozen[layer].get(rel, "")
        if got != entry["sha256"] or (want and got != want):
            mismatched.append(rel)
    ok = not missing and not mismatched
    return ok, {"files": len(manifest["files"]), "missing": missing,
                "mismatched": mismatched, "documents": manifest.get("documents")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(STATUS.read_text(encoding="utf-8") if STATUS.exists() else "no snapshot status")
        return 0
    if not args.snapshot:
        print("REFUSING: --snapshot PATH|URL is required", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        archive = staging / "snapshot.tar.gz"
        fetch(args.snapshot, archive)
        ok, report = verify(ROOT, archive, staging / "unpacked")
        if ok:
            for rel in ("cleaned", "corpus_safe_512"):
                src = staging / "unpacked" / rel
                if src.exists():
                    dst = ROOT / rel
                    dst.mkdir(parents=True, exist_ok=True)
                    for item in src.iterdir():
                        shutil.copy2(item, dst / item.name)
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps({
            "basis": "author_snapshot",
            "verified": bool(ok),
            "archive_sha256": sha256(archive) if archive.exists() else "",
            "report": report,
        }, indent=1), encoding="utf-8")
    print(f"snapshot verified: {ok}  files: {report.get('files')}  "
          f"missing: {len(report.get('missing', []))}  "
          f"mismatched: {len(report.get('mismatched', []))}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
