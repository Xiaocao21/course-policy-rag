"""make_required_sources.py - freeze the 36 required documents and verify the freeze.

The reproduction gate must not be decided by whatever a fresh crawl happens to return: a
source that disappears would silently drop out of `sources.csv` and the corpus would look
complete. This tool writes the required list once, from the frozen v0.1.5 sources table, and
`--check` proves the committed list still matches that baseline.

Usage
    python tools/make_required_sources.py            # write data/REQUIRED_SOURCES.csv
    python tools/make_required_sources.py --check    # verify the committed freeze
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "REQUIRED_SOURCES.csv"
SHA = ROOT / "data" / "REQUIRED_SOURCES.sha256"
COLUMNS = ["doc_id", "tier", "institution", "seed_id", "final_url", "bytes",
           "sha256_raw", "etag", "last_modified", "robots_allowed", "license_note",
           "clause_count"]
EXPECTED_TIERS = {"course": 14, "faculty": 9, "institution": 13}
EXPECTED_DOCS = 36
EXPECTED_INSTITUTIONS = 12


def read_included(root: Path) -> list[dict]:
    rows = []
    with (root / "sources.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("include", "")).strip().upper() == "TRUE":
                rows.append({c: row.get(c, "") for c in COLUMNS})
    return rows


def write_freeze(root: Path) -> int:
    rows = read_included(root)
    if len(rows) != EXPECTED_DOCS:
        print(f"REFUSING: expected {EXPECTED_DOCS} included documents, found {len(rows)}",
              file=sys.stderr)
        return 2
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    SHA.write_text(hashlib.sha256(OUT.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(root)} ({len(rows)} rows)")
    return 0


def check(root: Path) -> int:
    problems = []
    if not OUT.exists() or not SHA.exists():
        print("REFUSING: data/REQUIRED_SOURCES.csv(.sha256) is missing", file=sys.stderr)
        return 2
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if SHA.read_text(encoding="utf-8").strip() != digest:
        problems.append("REQUIRED_SOURCES.csv does not match its recorded sha256")
    with OUT.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_DOCS:
        problems.append(f"expected {EXPECTED_DOCS} rows, found {len(rows)}")
    tiers = {t: sum(1 for r in rows if r["tier"] == t) for t in {r["tier"] for r in rows}}
    if tiers != EXPECTED_TIERS:
        problems.append(f"tier counts {tiers} != {EXPECTED_TIERS}")
    institutions = len({r["institution"] for r in rows})
    if institutions != EXPECTED_INSTITUTIONS:
        problems.append(f"institutions {institutions} != {EXPECTED_INSTITUTIONS}")
    for r in rows:
        if not r["final_url"] or not r["sha256_raw"] or not r["seed_id"]:
            problems.append(f"{r['doc_id']}: missing url/sha256/seed_id")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("documents_included") != EXPECTED_DOCS:
        problems.append("MANIFEST.documents_included disagrees with the freeze")
    if manifest.get("tier_counts") != EXPECTED_TIERS:
        problems.append(f"MANIFEST.tier_counts {manifest.get('tier_counts')} disagrees")
    included_now = {r["doc_id"] for r in read_included(root)}
    frozen = {r["doc_id"] for r in rows}
    if included_now != frozen:
        problems.append(f"current sources.csv include set differs "
                        f"(missing {sorted(frozen - included_now)[:5]}, "
                        f"extra {sorted(included_now - frozen)[:5]})")
    for p in problems:
        print(f"FAIL: {p}")
    print(f"required sources: {len(rows)} docs, tiers {tiers}, {institutions} institutions")
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    return check(root) if args.check else write_freeze(root)


if __name__ == "__main__":
    raise SystemExit(main())
