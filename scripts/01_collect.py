"""01_collect.py - polite, auditable crawler for the course-policy corpus.

Reading order for a new contributor:
  1. robots preflight   -> reports/robots_audit.json (audit evidence, never bypassed)
  2. rate-limited fetch -> raw/<tier>/<doc_id>/raw.{html,pdf} + meta.json (immutable)
  3. content gate       -> reports/pending_review.csv for pages that are too thin
  4. wayback snapshot   -> snapshots/wayback_index.json (version evidence)
  5. capability metrics -> reports/corpus_capability.csv (feeds plan section B gate)

Registries:
  config/source_registry.csv  human-maintained seed list (never overwritten)
  sources.csv                 pipeline-owned provenance + decisions (regenerated)

Usage:
    .venv\\Scripts\\python.exe scripts\\01_collect.py --limit 5
    .venv\\Scripts\\python.exe scripts\\01_collect.py --seed-id BU-INSTITUTION-ACADEMIC-CONDUCT-CODE
    .venv\\Scripts\\python.exe scripts\\01_collect.py --force
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, RobotsCache, RateLimitedFetcher, build_text_layers,
    compile_detectors, domain_hits, load_settings, load_taxonomy, now_iso,
    project_paths, sha256_bytes, slugify, write_json,
)

try:
    # selectolax >= 1.0 removed the deprecated "modest" backend from
    # selectolax.parser; LexborHTMLParser is the supported entry point.
    from selectolax.lexbor import LexborHTMLParser as HTMLParser
except ImportError:  # pragma: no cover
    try:
        from selectolax.parser import HTMLParser  # type: ignore
    except ImportError:
        HTMLParser = None

REGISTRY_FIELDS = [
    "seed_id", "tier", "institution", "country", "doc_type", "expected_format",
    "start_urls", "allowed_domains", "crawl_depth", "url_allow_regex", "url_deny_regex",
    "license_note", "robots_decision", "status", "notes",
]

SOURCE_FIELDS = [
    "doc_id", "seed_id", "tier", "institution", "country", "doc_type",
    "url", "final_url", "url_snapshot", "retrieved_at", "last_modified", "etag",
    "http_status", "content_type", "bytes", "sha256_raw",
    "pub_date", "term", "version", "policy_effective_date", "version_group",
    "license_note", "robots_allowed", "robots_checked_at",
    "chars_evidence", "chars_retrieval", "domains_hit", "domain_list", "kw_total",
    "clause_count", "exception_count", "crossref_count", "alt_evidence_count",
    "academic_year", "version_marker", "verdict", "include", "exclude_reason",
]


def read_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_done(fetch_log: Path) -> dict[str, dict]:
    if not fetch_log.exists():
        return {}
    done: dict[str, dict] = {}
    with fetch_log.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            done[row["url"]] = row
    return done


def append_fetch_log(fetch_log: Path, row: dict, header: bool) -> None:
    fetch_log.parent.mkdir(parents=True, exist_ok=True)
    with fetch_log.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header:
            w.writeheader()
        w.writerow(row)


VERDICT_RANK = {"STRONG": 2, "usable": 1, "PENDING_REVIEW": 0}


def flush_sources(root: Path, collected: list[dict], sources_csv: Path) -> None:
    """Merge this run's documents into sources.csv, preserving earlier rows.

    O(n) in the size of the existing file: previously collected rows are streamed
    once and only this run's doc_ids are replaced. An earlier version rebuilt the
    whole table per seed, which is quadratic once a corpus grows past a few hundred
    documents.
    """
    fresh: dict[str, dict] = {}
    for s in collected:
        cur = fresh.get(s["doc_id"])
        if cur is None or VERDICT_RANK.get(s.get("verdict", ""), 0) > VERDICT_RANK.get(cur.get("verdict", ""), 0):
            fresh[s["doc_id"]] = s

    merged: list[dict] = []
    if sources_csv.exists():
        with sources_csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                doc_id = row.get("doc_id")
                if doc_id and doc_id in fresh:
                    continue                      # superseded by this run
                merged.append(row)
    merged.extend(fresh.values())
    merged.sort(key=lambda r: (r.get("tier", ""), r.get("institution", ""), r.get("doc_id", "")))

    tmp = sources_csv.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SOURCE_FIELDS)
        w.writeheader()
        for s in merged:
            w.writerow({k: s.get(k, "") for k in SOURCE_FIELDS})
    tmp.replace(sources_csv)                      # atomic swap; never leave a partial table


def flush_pending(paths: dict, pending: list[dict]) -> None:
    """Rewrite pending_review.csv without duplicating URLs across runs."""
    out = paths["reports"] / "pending_review.csv"
    seen: dict[str, dict] = {}
    if out.exists():
        with out.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("url"):
                    seen[row["url"]] = row
    for row in pending:
        seen[row["url"]] = row
    if not seen:
        return
    fields = list(next(iter(seen.values())).keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in seen.values():
            w.writerow(row)


def html_main_text(html: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (main_text_with_linebreaks, links) preferring main/article containers.

    Raises rather than returning "" when no parser is available: an empty string
    used to be indistinguishable from a genuinely empty page and silently pushed
    real documents into pending_review.
    """
    if HTMLParser is None:
        raise RuntimeError(
            "no HTML parser available; install stage-A deps: "
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )
    tree = HTMLParser(html)
    for tag in ("script", "style", "noscript", "nav", "header", "footer", "aside", "form"):
        for node in tree.css(tag):
            node.decompose()
    root = None
    for sel in ("main", "article", "[role=main]", "#main-content", "#content", ".content"):
        nodes = tree.css(sel)
        if nodes:
            root = max(nodes, key=lambda n: len(n.text() or ""))
            break
    if root is None:
        root = tree.body or tree.root
    if root is None:
        return "", []
    # keep block structure: insert newlines at block boundaries
    parts: list[str] = []
    block_tags = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br",
                  "section", "article", "table", "ul", "ol", "blockquote", "td", "th"}
    for node in root.traverse(include_text=True):
        if node.tag == "-text":
            parts.append(node.text_content or "")
        elif node.tag in block_tags:
            parts.append("\n")
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    links = [(a.attributes.get("href", ""), (a.text() or "").strip())
             for a in root.css("a[href]")]
    return text, links


def pdf_text(path: Path) -> tuple[str, int, str | None]:
    """Extract PDF text. Returns (text, page_count, error)."""
    try:
        import pdfplumber
    except ImportError:
        return "", 0, "pdfplumber_missing"
    try:
        chunks: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                chunks.append(page.extract_text() or "")
        return "\n\n".join(chunks), len(chunks), None
    except Exception as e:  # noqa: BLE001
        return "", 0, f"{type(e).__name__}: {e}"


def guess_version(text: str, detectors: dict) -> tuple[str, str, str]:
    """Return (academic_year, version_marker, version_group_hint)."""
    ym = detectors["academic_year"].search(text)
    year = f"{ym.group(1)}-{ym.group(2)}" if ym else ""
    ver = "yes" if detectors["version"].search(text) else ""
    return year, ver, year


def make_doc_id(institution: str, doc_type: str, url: str, year: str) -> str:
    """Build a page-unique doc_id from the URL path.

    NOTE: an earlier version keyed the id off the seed's doc_type, so every page
    discovered under one seed collapsed onto the same id and overwrote each other.
    The URL path is the only thing guaranteed unique per page, so it drives the id;
    doc_type stays in meta.json as provenance.
    """
    parts = urlsplit(url)
    segments = [s for s in parts.path.split("/") if s]
    # drop boilerplate leading segments that add no information
    drop = {"academics", "policies", "policy", "en", "www", "index.html", "current",
            "students", "student", "about", "info", "pages"}
    tail = [s for s in segments if s.lower() not in drop]
    if not tail:
        tail = [parts.netloc.split(".")[0]]
    # keep the last three segments: enough to stay unique, short enough to read
    label = "-".join(tail[-3:])
    label = re.sub(r"\.(html?|php|aspx|pdf)$", "", label, flags=re.I)
    base = f"{slugify(institution, 18).upper()}-{slugify(label, 52).upper()}"
    if year:
        base = f"{base}-{year}"
    return base.strip("-")


def crawl_seed(seed: dict, cfg: dict, taxonomy: dict, detectors: dict,
               robots: RobotsCache, fetcher: RateLimitedFetcher, paths: dict,
               done: dict[str, dict], force: bool, budget: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Crawl one seed. Returns (sources, fetch_log_rows, pending_rows)."""
    sources: list[dict] = []
    log_rows: list[dict] = []
    pending: list[dict] = []

    allow = re.compile(seed.get("url_allow_regex") or cfg["url_filters"]["allow_regex"])
    deny = re.compile(seed.get("url_deny_regex") or cfg["url_filters"]["deny_regex"])
    hard_deny_ext = tuple(cfg["url_filters"]["hard_deny_ext"])
    gate = cfg["content_gate"]

    allowed_domains = {d.strip().lower() for d in (seed.get("allowed_domains") or "").split(";") if d.strip()}
    starts = [u.strip() for u in (seed.get("start_urls") or "").split(";") if u.strip()]
    try:
        max_depth = int(seed.get("crawl_depth") or 2)
    except ValueError:
        max_depth = 2

    queue: list[tuple[str, int]] = [(u, 0) for u in starts]
    seen: set[str] = set()

    while queue:
        if budget["pages"] <= 0:
            break
        url, depth = queue.pop(0)
        url = urldefrag(url)[0].rstrip()
        if url in seen:
            continue
        seen.add(url)

        if any(url.lower().endswith(ext) for ext in hard_deny_ext):
            continue
        host = urlsplit(url).netloc.lower()
        if allowed_domains and not any(host == d or host.endswith("." + d) for d in allowed_domains):
            continue
        if depth > 0 and not allow.search(url):
            continue
        if depth > 0 and deny.search(url):
            continue

        decision = robots.decision(url)
        if not decision["allowed"]:
            log_rows.append({
                "url": url, "seed_id": seed["seed_id"], "status": "blocked_by_robots",
                "http_status": "", "bytes": 0, "sha256": "", "retrieved_at": now_iso(),
                "note": decision["robots_url"],
            })
            continue

        cached = done.get(url)
        if cached and cached.get("status") == "ok" and not force:
            budget["pages"] -= 1
            continue

        budget["pages"] -= 1
        res = fetcher.fetch(url, crawl_delay=decision.get("crawl_delay"))
        log_rows.append({
            "url": url, "seed_id": seed["seed_id"],
            "status": "ok" if res.ok else "error",
            "http_status": res.status or "", "bytes": len(res.content),
            "sha256": sha256_bytes(res.content) if res.content else "",
            "retrieved_at": now_iso(), "note": res.error or "",
        })
        if not res.ok:
            continue

        is_pdf = res.content[:5] == b"%PDF-" or "pdf" in res.content_type.lower()
        year = ""
        version_group = ""
        raw_text = ""
        links: list[tuple[str, str]] = []
        page_count = 0
        extract_error = None

        if is_pdf:
            doc_id = make_doc_id(seed["institution"], seed["doc_type"], url, "")
            dest_dir = paths["raw"] / seed["tier"] / doc_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            raw_path = dest_dir / "raw.pdf"
            raw_path.write_bytes(res.content)
            raw_text, page_count, extract_error = pdf_text(raw_path)
            if extract_error or len(raw_text) < gate["min_chars"]:
                pending.append({
                    "url": url, "seed_id": seed["seed_id"], "tier": seed["tier"],
                    "reason": f"pdf_extract:{extract_error or 'too_short'}",
                    "chars": len(raw_text), "pages": page_count,
                    "suggest": "ocr_review_or_exclude",
                })
        else:
            html = res.content.decode("utf-8", "ignore")
            raw_text, links = html_main_text(html)
            doc_id = make_doc_id(seed["institution"], seed["doc_type"], url, "")

        if len(raw_text) < gate["min_chars"]:
            pending.append({
                "url": url, "seed_id": seed["seed_id"], "tier": seed["tier"],
                "reason": "below_min_chars", "chars": len(raw_text), "pages": page_count,
                "suggest": "skip_or_manual",
            })
            continue

        hits = domain_hits(raw_text, taxonomy)
        domains_hit = sum(1 for v in hits.values() if v)
        if domains_hit < gate["min_domains_hit"]:
            pending.append({
                "url": url, "seed_id": seed["seed_id"], "tier": seed["tier"],
                "reason": "below_min_domains", "chars": len(raw_text), "pages": page_count,
                "suggest": "skip_navigation_page",
            })
            continue

        year, version_marker, version_group = guess_version(raw_text, detectors)
        if year:
            doc_id = make_doc_id(seed["institution"], seed["doc_type"], url, year)

        layers = build_text_layers(raw_text, cfg["text_layers"])
        clause_count = len(detectors["clause"].findall(raw_text))
        exc = len(detectors["exception"].findall(raw_text))
        xr = len(detectors["crossref"].findall(raw_text))
        alt = len(detectors["alt_evidence"].findall(raw_text))

        strong = (
            len(raw_text) >= gate["min_chars"]
            and domains_hit >= gate["min_domains_hit"]
            and exc >= gate["strong_exception_min"]
            and (clause_count >= gate["strong_clause_or_crossref"]["clause_min"]
                 or xr >= gate["strong_clause_or_crossref"]["crossref_min"])
        )
        verdict = "STRONG" if strong else "usable"

        if not is_pdf:
            dest_dir = paths["raw"] / seed["tier"] / doc_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "raw.html").write_bytes(res.content)

        meta = {
            "doc_id": doc_id,
            "seed_id": seed["seed_id"],
            "tier": seed["tier"],
            "institution": seed["institution"],
            "country": seed.get("country", ""),
            "doc_type": seed["doc_type"],
            "url": url,
            "final_url": res.final_url,
            "retrieved_at": now_iso(),
            "last_modified": res.last_modified,
            "etag": res.etag,
            "content_type": res.content_type,
            "bytes": len(res.content),
            "sha256_raw": sha256_bytes(res.content),
            "robots": decision,
            "license_note": seed.get("license_note", ""),
            "format": "pdf" if is_pdf else "html",
            "pages": page_count,
            "academic_year": year,
            "version_marker": version_marker,
            "version_group": version_group,
            "capability": {
                "chars_evidence": len(layers.evidence),
                "chars_retrieval": len(layers.retrieval),
                "charmap_len": len(layers.charmap),
                "domains_hit": hits,
                "clause_count": clause_count,
                "exception_count": exc,
                "crossref_count": xr,
                "alt_evidence_count": alt,
            },
            "verdict": verdict,
        }
        write_json(dest_dir / "meta.json", meta)
        (dest_dir / "raw.txt").write_text(raw_text, encoding="utf-8")

        sources.append({
            "doc_id": doc_id, "seed_id": seed["seed_id"], "tier": seed["tier"],
            "institution": seed["institution"], "country": seed.get("country", ""),
            "doc_type": seed["doc_type"], "url": url, "final_url": res.final_url,
            "url_snapshot": "", "retrieved_at": meta["retrieved_at"],
            "last_modified": res.last_modified or "", "etag": res.etag or "",
            "http_status": res.status, "content_type": res.content_type,
            "bytes": len(res.content), "sha256_raw": meta["sha256_raw"],
            "pub_date": "", "term": year, "version": version_marker,
            "policy_effective_date": "", "version_group": version_group,
            "license_note": seed.get("license_note", ""),
            "robots_allowed": decision["allowed"], "robots_checked_at": decision["checked_at"],
            "chars_evidence": len(layers.evidence), "chars_retrieval": len(layers.retrieval),
            "domains_hit": domains_hit,
            "domain_list": ";".join(f"{k}:{v}" for k, v in hits.items() if v),
            "kw_total": sum(hits.values()), "clause_count": clause_count,
            "exception_count": exc, "crossref_count": xr, "alt_evidence_count": alt,
            "academic_year": year, "version_marker": version_marker,
            "verdict": verdict, "include": "PENDING_REVIEW", "exclude_reason": "",
        })

        # controlled same-domain link following
        if not is_pdf and depth < max_depth:
            for href, _label in links:
                if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                    continue
                nxt = urldefrag(urljoin(url, href))[0]
                if nxt not in seen:
                    queue.append((nxt, depth + 1))

    return sources, log_rows, pending


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--limit", type=int, default=0, help="max seeds to process (0 = all)")
    ap.add_argument("--seed-id", action="append", default=[], help="only these seed_ids")
    ap.add_argument("--tier", action="append", default=[], help="only these tiers")
    ap.add_argument("--force", action="store_true", help="refetch even if cached ok")
    ap.add_argument("--max-pages", type=int, default=0, help="override crawl max_pages")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)
    detectors = compile_detectors(taxonomy)

    seeds = read_registry(paths["config"] / "source_registry.csv")
    if not seeds:
        print(f"ERROR: no seeds in {paths['config'] / 'source_registry.csv'}", file=sys.stderr)
        return 2
    if args.seed_id:
        want = set(args.seed_id)
        seeds = [s for s in seeds if s["seed_id"] in want]
    if args.tier:
        want = set(args.tier)
        seeds = [s for s in seeds if s["tier"] in want]
    if args.limit:
        seeds = seeds[: args.limit]

    robots = RobotsCache(user_agent=cfg["crawler"]["user_agent"])
    fetcher = RateLimitedFetcher(
        user_agent=cfg["crawler"]["user_agent"],
        delay=cfg["crawler"]["delay_seconds"],
        timeout=cfg["crawler"]["timeout_seconds"],
        max_retries=cfg["crawler"]["max_retries"],
        backoff_base=cfg["crawler"]["backoff_base_seconds"],
        backoff_max=cfg["crawler"]["backoff_max_seconds"],
        respect_crawl_delay=cfg["crawler"]["respect_crawl_delay"],
    )

    fetch_log = paths["reports"] / "fetch_log.csv"
    done = load_done(fetch_log)
    log_header_needed = not fetch_log.exists()

    pages = args.max_pages or cfg["crawler"]["max_pages"]
    budget = {"pages": pages}

    sources_csv = root / "sources.csv"
    all_sources: list[dict] = []
    all_pending: list[dict] = []

    for i, seed in enumerate(seeds, 1):
        if seed.get("status", "").lower() in ("manual_only", "excluded", "skip"):
            print(f"[{i}/{len(seeds)}] SKIP(manual_only) {seed['seed_id']}")
            continue
        # write the fetch log incrementally per seed so a long crawl stays
        # resumable and auditable even if it is interrupted
        print(f"[{i}/{len(seeds)}] crawl {seed['seed_id']} ({seed['tier']}) "
              f"pages_left={budget['pages']}", flush=True)
        src, logs, pend = crawl_seed(seed, cfg, taxonomy, detectors, robots, fetcher,
                                     paths, done, args.force, budget)
        for row in logs:
            append_fetch_log(fetch_log, row, log_header_needed)
            log_header_needed = False
        all_sources.extend(src)
        all_pending.extend(pend)
        print(f"           -> {len(src)} ingested, {len(pend)} pending", flush=True)

        # flush partial results so an interrupted run keeps its work
        flush_sources(root, all_sources, sources_csv)
        if all_pending:
            flush_pending(paths, all_pending)

    # dedupe by doc_id (keep strongest verdict) and merge with any earlier sources.csv
    flush_sources(root, all_sources, sources_csv)
    if all_pending:
        flush_pending(paths, all_pending)
    with sources_csv.open(encoding="utf-8", newline="") as f:
        sources = list(csv.DictReader(f))

    write_json(paths["reports"] / "robots_audit.json", robots.audit())

    # NOTE: this script deliberately no longer writes a capability report.
    # The per-document counts captured at crawl time are page-level regex hits, not
    # clause counts, and treating them as clause counts produced misleading totals.
    # `reports/clause_stats.csv`, written by 02_extract_clean.py from the extracted
    # clause records, is the authoritative source for every downstream gate.
    acc = cfg["acceptance"]
    strong = [s for s in sources if s.get("verdict") == "STRONG"]
    by_tier: dict[str, int] = {}
    for s in sources:
        by_tier[s["tier"]] = by_tier.get(s["tier"], 0) + 1

    clause_stats = paths["reports"] / "clause_stats.csv"
    print("\n=== collection summary ===")
    print(f"  documents collected       : {len(sources)}")
    for t in ("course", "faculty", "institution"):
        print(f"    {t:12s}           : {by_tier.get(t, 0)}")
    print(f"  STRONG (crawl-time)       : {len(strong)}")
    print(f"  institutions              : {len({s['institution'] for s in sources})}  "
          f"(target >= {acc['min_institutions']})")
    print(f"  pages budget left         : {budget['pages']}")
    print("\n  clause-level capability is NOT reported here; run 02_extract_clean.py and")
    print(f"  read {clause_stats.name} (it does not exist yet)" if not clause_stats.exists()
          else f"\n  clause-level capability: {clause_stats}")
    print(f"\nwrote {sources_csv}")
    print(f"wrote {paths['reports'] / 'robots_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
