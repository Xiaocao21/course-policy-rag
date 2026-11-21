"""make_index_snapshot.py - package the official FAISS indexes for exact replay.

Why this exists: FAISS index files embed float32 vectors that depend on the BLAS/GPU stack, so
rebuilding them on another machine produces different bytes. The test2 pre-registration pins
the official index hashes, and `19_rerank_eval.py` correctly refuses anything else. Exact
retrieval replay therefore installs these files rather than regenerating them.

Outputs
    dist/indexes_v0.1.5.tar.gz
    dist/indexes_v0.1.5.tar.gz.sha256
    dist/INDEX_MANIFEST.json   (per-file hashes from PREREGISTRATION.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="dist/indexes_v0.1.5.tar.gz")
    args = ap.parse_args()
    root = Path(args.root)
    prereg = json.loads((root / "results" / "test2" / "PREREGISTRATION.json")
                        .read_text(encoding="utf-8"))
    artifacts = {k: v for k, v in prereg["artifacts"].items() if k.startswith("indexes/")}
    if not artifacts:
        raise SystemExit("REFUSING: the pre-registration lists no index artefacts")
    files, entries, mismatched = [], [], []
    for rel in sorted(artifacts):
        path = root / rel
        if not path.exists():
            raise SystemExit(f"REFUSING: {rel} is missing from this checkout")
        got = sha256(path)
        if got != artifacts[rel]:
            mismatched.append(rel)
        files.append(path)
        entries.append({"path": rel, "sha256": got, "prereg_sha256": artifacts[rel]})
    if mismatched:
        raise SystemExit(f"REFUSING: {len(mismatched)} indexes do not match the "
                         f"pre-registration: {mismatched[:5]}")
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = out.parent / "INDEX_MANIFEST.json"
    manifest.write_text(json.dumps({
        "corpus_version": "v0.1.5",
        "source": "results/test2/PREREGISTRATION.json#artifacts[indexes/]",
        "note": ("Official FAISS indexes. They are byte-pinned by the test2 pre-registration; "
                 "rebuilding them on another BLAS/GPU stack changes the bytes."),
        "files": entries,
    }, indent=1), encoding="utf-8")
    with tarfile.open(out, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(root).as_posix())
        tar.add(manifest, arcname="INDEX_MANIFEST.json")
    digest = sha256(out)
    out.with_suffix(out.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(root)} ({out.stat().st_size/1e6:.1f} MB, {len(files)} files)")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
