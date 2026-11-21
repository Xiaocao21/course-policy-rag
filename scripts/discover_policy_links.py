"""discover_policy_links.py - homepage-driven seed discovery.

Guessed deep URLs mostly 404 (measured), while site homepages return real
content (e.g. regulations.manchester.ac.uk -> 26k chars). This script fetches a
set of *verified-good* landing pages, follows links whose anchor text or href
looks policy-bearing, scores each target, and writes a ranked candidate registry.

This is the honest replacement for guessing paths, and it reuses the same
robots-respecting fetcher as the collector.

Usage:
    .venv\\Scripts\\python.exe scripts\\discover_policy_links.py --landings config\\landings.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.robotparser as robotparser
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_taxonomy, minhash_signature, minhash_similarity,
)

UA = "CPS5801-research-bot/0.1 (+academic research; contact: student@wku.edu.cn)"

LINK_TEXT_RE = re.compile(
    r"(?i)\b(polic|regulation|integrity|conduct|plagiar|misconduct|grading|grade|"
    r"assessment|attendance|absence|exam|examination|deferral|extension|appeal|"
    r"academic|handbook|calendar|syllab|procedure|code of|rules)\b"
)
URL_RE = re.compile(
    r"(?i)(polic|regulation|integrity|conduct|plagiar|misconduct|grading|grade|"
    r"assessment|attendance|absence|exam|deferral|extension|appeal|academic|handbook|rules)"
)
DENY_RE = re.compile(
    r"(?i)(login|signin|sign-in|logout|search|user/|wp-admin|feed|print|calendar/event|"
    r"news|donate|give|shop|apply|admission|job|career|alumni|facebook|twitter|linkedin|"
    r"instagram|youtube|\.jpg|\.png|\.gif|\.svg|\.mp4|mailto:|tel:)"
)


def robots_allowed(url: str) -> bool:
    parts = urlsplit(url)
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{parts.scheme}://{parts.netloc}/robots.txt")
    try:
        rp.read()
        return rp.can_fetch(UA, url)
    except Exception:  # noqa: BLE001
        return True   # treat an unfetchable robots.txt as unrestricted; audit records it


def fetch(url: str, timeout: int = 15, max_bytes: int = 4_000_000):
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(max_bytes), r.headers.get("Content-Type", ""), None
    except urllib.error.HTTPError as e:
        return e.code, b"", "", f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, b"", "", f"{type(e).__name__}"


def page_text_and_links(body: bytes) -> tuple[str, list[tuple[str, str]]]:
    try:
        from selectolax.lexbor import LexborHTMLParser as HTMLParser
    except ImportError:
        from selectolax.parser import HTMLParser  # legacy selectolax
    tree = HTMLParser(body.decode("utf-8", "ignore"))
    for tag in ("script", "style", "noscript", "nav", "footer", "aside", "form"):
        for node in tree.css(tag):
            node.decompose()
    links = []
    for a in tree.css("a[href]"):
        href = a.attributes.get("href", "")
        if href:
            links.append((href, (a.text() or "").strip()))
    root = None
    for sel in ("main", "article", "[role=main]", "#main-content", "#content"):
        nodes = tree.css(sel)
        if nodes:
            root = max(nodes, key=lambda n: len(n.text() or ""))
            break
    root = root or tree.body or tree.root
    text = " ".join((root.text() or "").split()) if root is not None else ""
    return text, links


def score(url: str, taxonomy: dict, timeout: int) -> dict | None:
    status, body, ctype, err = fetch(url, timeout=timeout)
    if not body:
        return None
    text = ""
    if body[:5] == b"%PDF-":
        try:
            import io
            import pdfplumber
            with pdfplumber.open(io.BytesIO(body)) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages[:40])
        except Exception:  # noqa: BLE001
            return None
    else:
        text, _links = page_text_and_links(body)
    if len(text) < 1200:
        return None
    low = text.lower()
    hits = {d: sum(low.count(k.lower()) for k in spec.get("keywords", []))
            for d, spec in taxonomy.get("domains", {}).items()}
    return {
        "url": url,
        "chars": len(text),
        "domains_hit": sum(1 for v in hits.values() if v),
        "kw_total": sum(hits.values()),
        "domain_list": ";".join(f"{k}:{v}" for k, v in hits.items() if v),
        "fingerprint": minhash_signature(text[:20000]),
        "text_head": text[:400],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landings", default="config/landings.csv")
    ap.add_argument("--out", default="config/discovered_candidates.csv")
    ap.add_argument("--max-per-landing", type=int, default=25)
    ap.add_argument("--min-keywords", type=int, default=12)
    ap.add_argument("--min-domains", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--delay", type=float, default=1.2)
    args = ap.parse_args()

    taxonomy = load_taxonomy(PROJECT_ROOT)
    land_path = Path(args.landings)
    if not land_path.is_absolute():
        land_path = PROJECT_ROOT / land_path
    with land_path.open(encoding="utf-8", newline="") as f:
        landings = [r for r in csv.DictReader(f) if r.get("url")]

    # ---------------------------------------------------- step 1: collect links
    per_host_last: dict[str, float] = defaultdict(float)
    candidates: dict[str, dict] = {}
    for i, land in enumerate(landings, 1):
        url = land["url"]
        host = urlsplit(url).netloc
        wait = args.delay - (time.time() - per_host_last[host])
        if wait > 0:
            time.sleep(wait)
        status, body, ctype, err = fetch(url, timeout=args.timeout)
        per_host_last[host] = time.time()
        if not body:
            print(f"[{i:2d}/{len(landings)}] LANDING FAIL {err or status}  {url[:70]}", flush=True)
            continue
        if body[:5] == b"%PDF-":
            print(f"[{i:2d}/{len(landings)}] landing is a PDF, scoring directly  {url[:60]}", flush=True)
            candidates[url] = {"source_url": url, "landing": url, "anchor": "(direct pdf)",
                               "tier": land.get("tier", ""), "institution": land.get("institution", "")}
            continue
        text, links = page_text_and_links(body)
        kept = 0
        seen_local = set()
        for href, anchor in links:
            if DENY_RE.search(href):
                continue
            if not (LINK_TEXT_RE.search(anchor) or URL_RE.search(href)):
                continue
            nxt = urljoin(url, href).split("#")[0]
            if nxt in seen_local or not nxt.startswith("http"):
                continue
            seen_local.add(nxt)
            candidates.setdefault(nxt, {
                "source_url": nxt, "landing": url, "anchor": anchor[:90],
                "tier": land.get("tier", ""), "institution": land.get("institution", ""),
            })
            kept += 1
            if kept >= args.max_per_landing:
                break
        print(f"[{i:2d}/{len(landings)}] {host:32s} chars={len(text):>6} links_kept={kept:>3} "
              f"total_candidates={len(candidates)}", flush=True)

    print(f"\nscoring {len(candidates)} candidate URLs ...", flush=True)

    # ---------------------------------------------------- step 2: score targets
    rows: list[dict] = []
    for j, (url, meta) in enumerate(sorted(candidates.items()), 1):
        host = urlsplit(url).netloc
        if not robots_allowed(url):
            rows.append({**meta, "url": url, "verdict": "blocked_by_robots", "chars": 0,
                         "domains_hit": 0, "kw_total": 0, "domain_list": "", "fingerprint": ()})
            continue
        wait = args.delay - (time.time() - per_host_last[host])
        if wait > 0:
            time.sleep(wait)
        res = score(url, taxonomy, args.timeout)
        per_host_last[host] = time.time()
        if res is None:
            continue
        rows.append({**meta, "url": url, "verdict": "candidate", **res})

    rows.sort(key=lambda r: (-r["kw_total"], -r["domains_hit"]))

    # drop near-duplicate pages (the same policy mirrored on several URLs)
    unique: list[dict] = []
    for r in rows:
        fp = r.get("fingerprint") or ()
        if fp and any(minhash_similarity(fp, u["fingerprint"]) >= 0.9 for u in unique if u.get("fingerprint")):
            continue
        unique.append(r)

    keep = [r for r in unique
            if r["kw_total"] >= args.min_keywords and r["domains_hit"] >= args.min_domains]

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    fields = ["seed_id", "tier", "institution", "country", "doc_type", "expected_format",
              "start_urls", "allowed_domains", "url_allow_regex", "url_deny_regex",
              "license_note", "robots_decision", "status", "notes"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep:
            host = urlsplit(r["url"]).netloc
            sid = re.sub(r"[^A-Za-z0-9]+", "-", f"{r.get('institution','')}-{host}-{r['url'].rstrip('/').split('/')[-1]}").strip("-").upper()[:60]
            w.writerow({
                "seed_id": sid or "SEED",
                "tier": r.get("tier", ""),
                "institution": r.get("institution", ""),
                "country": "",
                "doc_type": "policy_page",
                "expected_format": "pdf" if r["url"].lower().endswith(".pdf") else "html",
                "start_urls": r["url"],
                "allowed_domains": host,
                "url_allow_regex": "",
                "url_deny_regex": "",
                "license_note": "public institutional policy; research use",
                "robots_decision": "allow",
                "status": "candidate",
                "notes": f"chars={r['chars']} kw={r['kw_total']} domains={r['domains_hit']} anchor={r.get('anchor','')[:40]}",
            })

    print("\n=== discovery summary ===")
    print(f"  landings fetched   : {len(landings)}")
    print(f"  candidate urls     : {len(candidates)}")
    print(f"  scored pages       : {len(rows)}")
    print(f"  after dedup        : {len(unique)}")
    print(f"  kept (>= {args.min_keywords} kw, >= {args.min_domains} domains): {len(keep)}")
    for r in keep[:15]:
        print(f"    kw={r['kw_total']:>4} dom={r['domains_hit']} chars={r['chars']:>6}  {r['url'][:88]}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
