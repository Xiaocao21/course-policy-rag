"""harvest_ocw.py - discover real MIT OCW syllabus/policy pages for the course tier.

Why this exists: guessed syllabus URLs 404, and OCW syllabus pages vary a lot in
policy density (probed range 4k-12k chars, 5-13 policy keywords). This script walks
the OCW sitemap index, collects each course's sitemap, keeps syllabus/policy-ish
pages, scores them for policy density, and emits a source_registry.csv fragment.

Robots: ocw.mit.edu/robots.txt is `User-Agent: * / Allow: /` (verified 2026-10-13),
so sitemap-driven discovery is permitted. The crawler still re-checks robots.

Usage:
    .venv\\Scripts\\python.exe scripts\\harvest_ocw.py --out config\\registry_ocw_candidates.csv
    .venv\\Scripts\\python.exe scripts\\harvest_ocw.py --max-courses 800 --min-keywords 8
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, domain_hits, load_taxonomy  # noqa: E402

UA = "CPS5801-research-bot/0.1 (+academic research; contact: Xiaocao21@users.noreply.github.com)"
ROBOTS_OK_NOTE = "robots.txt Allow: / (verified)"

# pages worth harvesting inside a course
PAGE_ALLOW = re.compile(r"(?i)/(pages/(syllabus|policies|grading|assignments|exams|calendar|overview))/?$")
SITEMAP_INDEX = "https://ocw.mit.edu/sitemap.xml"


def get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return body


def extract_locs(xml: bytes) -> list[str]:
    return re.findall(r"<loc>\s*(.*?)\s*</loc>", xml.decode("utf-8", "ignore"))


def plain_text(html: str) -> str:
    """Cheap text extraction for the density pre-check (no deps beyond stdlib)."""
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    from html import unescape
    text = unescape(html)
    return re.sub(r"[ \t]+", " ", text)


def score_page(url: str, taxonomy: dict, timeout: int) -> dict | None:
    try:
        body = get(url, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    html = body.decode("utf-8", "ignore")
    text = plain_text(html)
    if len(text) < 1200:
        return None
    hits = domain_hits(text, taxonomy)
    domains_hit = sum(1 for v in hits.values() if v)
    return {
        "url": url,
        "chars": len(text),
        "domains_hit": domains_hit,
        "kw_total": sum(hits.values()),
        "domain_list": ";".join(f"{k}:{v}" for k, v in hits.items() if v),
        "late": hits.get("late", 0),
        "integrity": hits.get("integrity", 0),
        "attendance": hits.get("attendance", 0),
        "exam": hits.get("exam", 0),
        "grading": hits.get("grading", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="config/registry_ocw_candidates.csv")
    ap.add_argument("--max-courses", type=int, default=400,
                    help="cap on course sitemaps fetched (each yields its pages)")
    ap.add_argument("--min-keywords", type=int, default=8,
                    help="keep pages with at least this many policy keyword hits")
    ap.add_argument("--min-domains", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    taxonomy = load_taxonomy(PROJECT_ROOT)

    print("fetching sitemap index ...", flush=True)
    index = extract_locs(get(SITEMAP_INDEX))
    course_sitemaps = [u for u in index if "/courses/" in u and u.endswith("sitemap.xml")]
    print(f"  course sitemaps available: {len(course_sitemaps)}", flush=True)
    course_sitemaps = course_sitemaps[: args.max_courses]

    # step 1: collect candidate page URLs from course sitemaps
    page_urls: list[str] = []
    for i, sm in enumerate(course_sitemaps, 1):
        try:
            locs = extract_locs(get(sm, timeout=args.timeout))
        except Exception:  # noqa: BLE001
            continue
        for loc in locs:
            if PAGE_ALLOW.search(loc):
                page_urls.append(loc)
        if i % 50 == 0:
            print(f"  sitemaps {i}/{len(course_sitemaps)} -> {len(page_urls)} candidate pages",
                  flush=True)
        time.sleep(0.1)

    # de-dup, keep deterministic order
    page_urls = sorted(set(page_urls))
    print(f"\ncandidate pages: {len(page_urls)}", flush=True)

    # step 2: score pages concurrently
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_page, u, taxonomy, args.timeout): u for u in page_urls}
        done = 0
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            if res:
                rows.append(res)
            if done % 100 == 0:
                print(f"  scored {done}/{len(page_urls)}  kept={len(rows)}", flush=True)

    rows.sort(key=lambda r: (-r["kw_total"], -r["domains_hit"]))
    keep = [r for r in rows
            if r["kw_total"] >= args.min_keywords and r["domains_hit"] >= args.min_domains]

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = ["seed_id", "tier", "institution", "country", "doc_type", "expected_format",
              "start_urls", "allowed_domains", "url_allow_regex", "url_deny_regex",
              "license_note", "robots_decision", "status", "notes"]

    def course_code(url: str) -> str:
        m = re.search(r"/courses/([^/]+)/", url)
        return m.group(1) if m else "course"

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep:
            code = course_code(r["url"])
            sid = f"MIT-OCW-{re.sub(r'[^A-Za-z0-9]+', '-', code).upper()[:40]}"
            w.writerow({
                "seed_id": sid,
                "tier": "course",
                "institution": "MIT OpenCourseWare",
                "country": "US",
                "doc_type": f"course_{r['url'].rstrip('/').split('/')[-1]}",
                "expected_format": "html",
                "start_urls": r["url"],
                "allowed_domains": "ocw.mit.edu",
                "url_allow_regex": "",
                "url_deny_regex": "",
                "license_note": "CC BY-NC-SA 4.0",
                "robots_decision": "allow",
                "status": "candidate",
                "notes": (f"chars={r['chars']} kw={r['kw_total']} domains={r['domains_hit']} "
                          f"late={r['late']} integrity={r['integrity']} "
                          f"attendance={r['attendance']} exam={r['exam']} grading={r['grading']}"),
            })

    print(f"\n=== OCW harvest summary ===")
    print(f"  courses scanned      : {len(course_sitemaps)}")
    print(f"  pages scored         : {len(rows)}")
    print(f"  pages kept (>= {args.min_keywords} kw, >= {args.min_domains} domains): {len(keep)}")
    if keep:
        print(f"  best page            : {keep[0]['url']} (kw={keep[0]['kw_total']})")
        print(f"  kw distribution      : " +
              ", ".join(f"p{q}={sorted(r['kw_total'] for r in keep)[int(len(keep)*q/100)] if keep else 0}"
                        for q in (50, 75, 90)))
    print(f"\nwrote {out}")
    print("\nNOTE: OCW syllabus pages are thin on average. Treat this file as the "
          "course-tier candidate pool; the content gate in 01_collect.py will "
          "decide which ones actually enter the corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
