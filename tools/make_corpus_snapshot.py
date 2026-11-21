"""make_corpus_snapshot.py - package the 36-document corpus for the reproduction release.

The snapshot is the *only* way to replay retrieval without depending on live web pages: the
live rebuild was measured and failed (see data/REBUILD_REPORT.md). It contains only the 36
documents actually used by the questions, not the 265 collected documents, and every file is
listed with its SHA-256 so `fetch_data.py` can prove it matches the frozen MANIFEST.

Outputs
    dist/corpus_v0.1.5.tar.gz          cleaned/ (36 docs × 3 layers) + corpus_safe_512/
    dist/corpus_v0.1.5.tar.gz.sha256   sha256 of the archive
    dist/SNAPSHOT_MANIFEST.json        per-file hashes + license note + provenance statement
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEANED_LAYERS = ("evidence", "retrieval", "charmap")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_docs(root: Path) -> list[dict]:
    with (root / "data" / "REQUIRED_SOURCES.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect(root: Path) -> tuple[list[Path], list[dict]]:
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    hashes = manifest["artefact_hashes"]
    docs = required_docs(root)
    files: list[Path] = []
    entries: list[dict] = []
    for row in docs:
        doc = row["doc_id"]
        for layer in CLEANED_LAYERS:
            rel = f"cleaned/{doc}.{layer}.jsonl"
            path = root / rel
            if not path.exists():
                raise SystemExit(f"REFUSING: {rel} is missing from this checkout")
            files.append(path)
            entries.append({"path": rel, "sha256": sha256(path),
                            "manifest_sha256": hashes["cleaned"].get(rel, ""),
                            "doc_id": doc, "layer": "cleaned",
                            "license_note": row["license_note"], "url": row["final_url"]})
    for rel in sorted(hashes["corpus_safe_512"]):
        path = root / rel
        if not path.exists():
            raise SystemExit(f"REFUSING: {rel} is missing from this checkout")
        files.append(path)
        entries.append({"path": rel, "sha256": sha256(path),
                        "manifest_sha256": hashes["corpus_safe_512"][rel],
                        "doc_id": "", "layer": "corpus_safe_512",
                        "license_note": "", "url": ""})
    mismatched = [e["path"] for e in entries
                  if e["manifest_sha256"] and e["sha256"] != e["manifest_sha256"]]
    if mismatched:
        raise SystemExit(f"REFUSING: {len(mismatched)} files do not match MANIFEST: "
                         f"{mismatched[:5]}")
    return files, entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="dist/corpus_v0.1.5.tar.gz")
    args = ap.parse_args()
    root = Path(args.root)
    files, entries = collect(root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = {
        "corpus_version": "v0.1.5",
        "documents": 36,
        "note": ("Only the 36 documents used by the question sets are included. Files are "
                 "byte-identical to the frozen MANIFEST; the live sites have since changed, "
                 "which is why a snapshot is required for retrieval replay."),
        "files": entries,
    }
    manifest_path = out.parent / "SNAPSHOT_MANIFEST.json"
    manifest_path.write_text(json.dumps(staging, indent=1), encoding="utf-8")
    with tarfile.open(out, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(root).as_posix())
        tar.add(manifest_path, arcname="SNAPSHOT_MANIFEST.json")
    digest = sha256(out)
    out.with_suffix(out.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(root)} ({out.stat().st_size/1e6:.1f} MB, {len(files)} files)")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
