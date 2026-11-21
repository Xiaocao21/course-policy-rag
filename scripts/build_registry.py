"""build_registry.py - turn probe/discovery results into collector-ready seeds.

Measured lesson: guessed deep URLs mostly 404 (or return JavaScript shells), while
site index pages return real content. So the registry is built from pages that were
actually fetched and scored, and each one becomes a seed whose *links* the collector
then follows to reach the individual policy documents.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_registry.py
    .venv\\Scripts\\python.exe scripts\\build_registry.py --include-discovery
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, write_json  # noqa: E402

REGISTRY_FIELDS = [
    "seed_id", "tier", "institution", "country", "doc_type", "expected_format",
    "start_urls", "allowed_domains", "crawl_depth", "url_allow_regex", "url_deny_regex",
    "license_note", "robots_decision", "status", "notes",
]

# Course-tier pages are a single dense syllabus page each; following links from them
# would pull in lecture notes and problem sets that are not course policy.
DEFAULT_DEPTH = {"course": 0, "faculty": 2, "institution": 2}
LINK_POLICY_RE = r"(?i)(polic\w*|regulation\w*|integrity|conduct|plagiar\w*|misconduct|grading|grade\w*|assessment|attendance|absence|exam\w*|deferral|extension|appeal\w*|academic-standing|handbook|rules|requirements)"

# Verdicts that justify spending crawl budget.
KEEP_VERDICTS = {"STRONG", "usable"}

# Manual-only sources documented for the reader; never crawled by the tool.
MANUAL_ONLY = [
    {"institution": "University of Alberta", "host": "apps.ualberta.ca",
     "reason": "robots.txt: Disallow: /catalogue/syllabus",
     "note": "syllabus PDFs must be downloaded by hand and dropped into raw/"},
    {"institution": "University of Michigan", "host": "ro.umich.edu",
     "reason": "robots.txt: Disallow: /",
     "note": "blanket disallow on the registrar host"},
    {"institution": "University of Michigan", "host": "cse.engin.umich.edu",
     "reason": "robots.txt disallows the policies path"},
    {"institution": "QAA", "host": "www.qaa.ac.uk",
     "reason": "robots.txt disallows /the-quality-code*"},
    {"institution": "University College London", "host": "www.ucl.ac.uk",
     "reason": "robots.txt disallows the academic-manual and examinations paths"},
    {"institution": "Monash University", "host": "www.monash.edu",
     "reason": "robots.txt disallows /students/policies"},
    {"institution": "University of Melbourne", "host": "policy.unimelb.edu.au",
     "reason": "JavaScript-rendered shell; no server-side content"},
    {"institution": "University of Edinburgh", "host": "www.ed.ac.uk",
     "reason": "policy pages return ~80-100 char JavaScript shells to non-browser clients"},
]

TIER_ORDER = {"course": 0, "faculty": 1, "institution": 2}


def make_seed_id(tier: str, institution: str, url: str) -> str:
    host = urlsplit(url).netloc.replace("www.", "")
    tail = url.rstrip("/").split("/")[-1] or "index"
    if tail.lower() in ("policies", "policy", "academic", "index.html", "index"):
        tail = ""
    raw = f"{institution}-{host}-{tail}".strip("-")
    sid = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").upper()
    return sid[:60] or "SEED"


def rows_from_probe(path: Path, tier_override: str = "") -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("verdict") not in KEEP_VERDICTS:
                continue
            url = r.get("url", "")
            tier = tier_override or r.get("tier", "")
            institution = r.get("institution", "")
            host = urlsplit(url).netloc
            out.append({
                "seed_id": make_seed_id(tier, institution, url),
                "tier": tier,
                "institution": institution,
                "country": r.get("country", ""),
                "doc_type": r.get("doc_type", "") or f"{tier}_policy_index",
                "expected_format": "pdf" if url.lower().endswith(".pdf") else "html",
                "start_urls": url,
                "allowed_domains": host,
                "crawl_depth": DEFAULT_DEPTH.get(tier, 1),
                "url_allow_regex": LINK_POLICY_RE if DEFAULT_DEPTH.get(tier, 1) else "",
                "url_deny_regex": "",
                "license_note": "public institutional policy text; academic research use",
                "robots_decision": "allow",
                "status": "candidate",
                "notes": (f"probe verdict={r.get('verdict')} chars={r.get('chars')} "
                          f"kw={r.get('kw_total')} domains={r.get('domains_hit')} "
                          f"exc={r.get('exception_count')} xr={r.get('crossref_count')}"),
            })
    return out


def rows_from_discovery(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "candidate" and r.get("start_urls"):
                out.append({k: r.get(k, "") for k in REGISTRY_FIELDS})
    return out


def rows_from_manual(path: Path) -> list[dict]:
    """Read a hand-curated seed CSV (tier,institution,country,doc_type,url,crawl_depth)."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            url = (r.get("url") or r.get("start_urls") or "").strip()
            if not url:
                continue
            tier = (r.get("tier") or "").strip()
            institution = (r.get("institution") or "").strip()
            depth = str(r.get("crawl_depth") or "").strip() or str(DEFAULT_DEPTH.get(tier, 1))
            # An explicit url_allow_regex in the seed CSV wins, so a targeted collection
            # round can crawl only the link types it needs (e.g. accommodation /
            # attendance pages) instead of every policy link on the site.
            allow = (r.get("url_allow_regex") or "").strip()
            if not allow and int(depth) > 0:
                allow = LINK_POLICY_RE
            out.append({
                "seed_id": make_seed_id(tier, institution, url),
                "tier": tier,
                "institution": institution,
                "country": (r.get("country") or "").strip(),
                "doc_type": (r.get("doc_type") or f"{tier}_policy").strip(),
                "expected_format": "pdf" if url.lower().endswith(".pdf") else "html",
                "start_urls": url,
                "allowed_domains": urlsplit(url).netloc,
                "crawl_depth": depth,
                "url_allow_regex": allow,
                "url_deny_regex": (r.get("url_deny_regex") or "").strip(),
                "license_note": "public institutional policy text; academic research use",
                "robots_decision": "allow",
                "status": "candidate",
                "notes": (r.get("notes") or "hand-curated seed").strip(),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="config/source_registry.csv")
    ap.add_argument("--include-discovery", action="store_true",
                    help="also merge config/discovered_candidates.csv")
    ap.add_argument("--include-ocw", action="store_true",
                    help="also merge config/registry_ocw_candidates.csv (all course candidates)")
    ap.add_argument("--include-course-selected", action="store_true",
                    help="merge config/registry_course_selected.csv (densest course seeds; preferred)")
    ap.add_argument("--include-manual", action="append", default=[],
                    help="merge a hand-curated seed CSV (repeatable)")
    ap.add_argument("--institution-only", action="store_true",
                    help="emit only institution-tier seeds (used to rebalance the corpus)")
    args = ap.parse_args()

    reports = PROJECT_ROOT / "reports"
    config = PROJECT_ROOT / "config"

    # _priority decides which entry wins when two sources target the same start_url.
    # 0 = automated probe/harvest, 1 = curated selection, 2 = hand-curated seed.
    seeds: list[dict] = []
    for name, p in (("seed_probe_round3.csv", 1), ("seed_probe_round2.csv", 1),
                    ("seed_probe.csv", 1)):
        for s in rows_from_probe(reports / name):
            s["_priority"] = 0
            seeds.append(s)

    for extra in args.include_manual:
        p = Path(extra)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        for s in rows_from_manual(p):
            s["_priority"] = 2
            seeds.append(s)

    if args.include_discovery:
        for s in rows_from_discovery(config / "discovered_candidates.csv"):
            s["_priority"] = 0
            seeds.append(s)
    if args.include_ocw:
        ocw = config / "registry_ocw_candidates.csv"
        if ocw.exists():
            with ocw.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("start_urls"):
                        s = {k: r.get(k, "") for k in REGISTRY_FIELDS}
                        s["_priority"] = 0
                        seeds.append(s)
    if args.include_course_selected:
        sel = config / "registry_course_selected.csv"
        if sel.exists():
            with sel.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("start_urls"):
                        s = {k: r.get(k, "") for k in REGISTRY_FIELDS}
                        s["_priority"] = 1
                        seeds.append(s)

    # normalise depth and fill tier-appropriate link policy
    for s in seeds:
        tier = s.get("tier", "")
        if not str(s.get("crawl_depth", "")).strip():
            s["crawl_depth"] = DEFAULT_DEPTH.get(tier, 1)
        if not str(s.get("url_allow_regex", "")).strip() and int(s["crawl_depth"] or 0) > 0:
            s["url_allow_regex"] = LINK_POLICY_RE

    if args.institution_only:
        seeds = [s for s in seeds if s.get("tier") == "institution"]

    # de-dup by start_url. A hand-curated seed may deliberately re-target the same
    # landing page with a narrower allow regex (a gap-fill round), so the entry with
    # the highest priority wins instead of "first seen wins" — the latter silently
    # discarded the targeted seeds and kept the broad probe versions.
    best: dict[str, dict] = {}
    for s in seeds:
        key = s.get("start_urls", "")
        if not key:
            continue
        prev = best.get(key)
        if prev is None or s.get("_priority", 0) >= prev.get("_priority", 0):
            best[key] = s
    unique = list(best.values())
    for s in unique:
        s.pop("_priority", None)

    unique.sort(key=lambda s: (TIER_ORDER.get(s.get("tier", ""), 9), s.get("institution", "")))

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        for s in unique:
            w.writerow({k: s.get(k, "") for k in REGISTRY_FIELDS})

    write_json(reports / "manual_only_sources.json", MANUAL_ONLY)

    by_tier: dict[str, int] = {}
    for s in unique:
        by_tier[s["tier"]] = by_tier.get(s["tier"], 0) + 1

    print("=== registry built ===")
    print(f"  seeds written   : {len(unique)}")
    for t in ("course", "faculty", "institution"):
        print(f"    {t:12s}: {by_tier.get(t, 0)}")
    print(f"  institutions    : {len({s['institution'] for s in unique})}")
    print(f"  manual-only     : {len(MANUAL_ONLY)} documented (never crawled)")
    print(f"\nwrote {out}")
    print(f"wrote {reports / 'manual_only_sources.json'}")
    print("\nNext: python run.py collect -- --max-pages 400")
    return 0


if __name__ == "__main__":
    sys.exit(main())
