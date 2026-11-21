"""make_code_manifest.py - prove which scripts changed between the original run and the release.

`19_rerank_eval.py` hashes `scripts/*.py` as a set, so `FINAL_RUN.code_sha256` changes if any
script changes. This tool records, per file, the hash before the public-release edits
(`data/CODE_MANIFEST.original.csv`) and the hash now, and writes a separate list for the eight
files in `EVALUATION_CODE` that must be byte-identical.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data" / "CODE_MANIFEST.original.csv"
OUT = ROOT / "CODE_MANIFEST.csv"
CORE_OUT = ROOT / "RESULTS_CODE_SHA256.json"
EVALUATION_CODE = ("scripts/13_evaluate.py", "scripts/18_train_reranker.py",
                   "scripts/19_rerank_eval.py", "scripts/16_stats.py",
                   "scripts/goldset.py", "scripts/common.py",
                   "scripts/11_chunking.py", "scripts/12_build_index.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    rows = []
    with ORIGINAL.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rel = row["path"]
            current = sha256(ROOT / rel) if (ROOT / rel).exists() else ""
            rows.append({
                "path": rel,
                "original_sha256": row["original_sha256"],
                "public_sha256": current,
                "changed": str(current != row["original_sha256"]).lower(),
                "role": "evaluation_code" if rel in EVALUATION_CODE else "supporting",
            })
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    core = {r["path"]: {"original": r["original_sha256"], "public": r["public_sha256"],
                        "identical": r["changed"] == "false"}
            for r in rows if r["role"] == "evaluation_code"}
    CORE_OUT.write_text(json.dumps(core, indent=1), encoding="utf-8")
    changed = [r["path"] for r in rows if r["changed"] == "true"]
    print(f"scripts: {len(rows)}  changed: {len(changed)}  core identical: "
          f"{all(v['identical'] for v in core.values())}")
    for rel in changed:
        print(f"  changed {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
