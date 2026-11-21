"""select_course_seeds.py - pick the densest course-tier seeds from the OCW pool.

Why selection is needed: the harvest keeps every syllabus page above a low keyword
floor, which is 1083 candidates. Loading all of them would swamp the tier balance
required by the plan (no tier above 40%). This ranks by policy keyword density and
keeps a configurable number, so the course tier is deliberately sized rather than
accidentally over-represented.

Usage:
    .venv\\Scripts\\python.exe scripts\\select_course_seeds.py --top 18
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT  # noqa: E402

REGISTRY_FIELDS = [
    "seed_id", "tier", "institution", "country", "doc_type", "expected_format",
    "start_urls", "allowed_domains", "url_allow_regex", "url_deny_regex",
    "license_note", "robots_decision", "status", "notes",
]


def parse_notes(notes: str) -> dict:
    out = {}
    for part in (notes or "").split():
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="config/registry_ocw_candidates.csv")
    ap.add_argument("--out", default="config/registry_course_selected.csv")
    ap.add_argument("--top", type=int, default=18,
                    help="how many course-tier seeds to keep in the registry")
    ap.add_argument("--min-late", type=int, default=0,
                    help="require at least this many 'late' keyword hits")
    args = ap.parse_args()

    pool = Path(args.pool)
    if not pool.is_absolute():
        pool = PROJECT_ROOT / pool
    if not pool.exists():
        print(f"no OCW pool at {pool}; run: python run.py ocw", file=sys.stderr)
        return 2

    with pool.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    scored = []
    for r in rows:
        n = parse_notes(r.get("notes", ""))
        try:
            kw = int(n.get("kw", 0))
            domains = int(n.get("domains", 0))
            late = int(n.get("late", 0))
        except ValueError:
            continue
        if late < args.min_late:
            continue
        scored.append((kw, domains, late, r))

    scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
    keep = scored[: args.top]

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        for kw, domains, late, r in keep:
            sid = r.get("seed_id", "SEED")
            w.writerow({
                "seed_id": sid,
                "tier": "course",
                "institution": "MIT OpenCourseWare",
                "country": "US",
                "doc_type": "course_syllabus",
                "expected_format": "html",
                "start_urls": r["start_urls"],
                "allowed_domains": "ocw.mit.edu",
                "url_allow_regex": "",
                "url_deny_regex": "",
                "license_note": "CC BY-NC-SA 4.0",
                "robots_decision": "allow",
                "status": "candidate",
                "notes": f"selected kw={kw} domains={domains} late={late}",
            })

    print("=== course-tier selection ===")
    print(f"  pool size      : {len(rows)}")
    print(f"  selected       : {len(keep)}  (top {args.top} by keyword density)")
    if keep:
        kws = [k[0] for k in keep]
        print(f"  keyword range  : {min(kws)}..{max(kws)}  median {sorted(kws)[len(kws)//2]}")
        print("\n  selected pages:")
        for kw, domains, late, r in keep[:20]:
            print(f"    kw={kw:>3} dom={domains} late={late:>2}  {r['start_urls'][:92]}")
    print(f"\nwrote {out}")
    print("\nThese are syllabus pages, which are denser in late-work and grading rules")
    print("than the corpus median; they supply the course tier without dominating it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
