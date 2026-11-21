"""report_content.py - single source of truth for every number in the progress report.

Why this module exists: the report must never disagree with the repository. Rather
than letting the document generator read files ad hoc, all figures are collected here
once, cross-checked against each other, and returned as a plain dict. If a check fails
the report is refused instead of being generated with wrong numbers.

Two classes of number, deliberately kept apart:

* **collected** - what the crawler brought in (251 documents, 13 institutions)
* **corpus**    - what actually entered the frozen corpus, decided by curate_corpus.py

The authoritative clause counts come from `reports/clause_stats.csv`, which is measured
on the cleaned evidence layer. The collector's `*_raw_hits` columns are page-level regex
hits and are never used as clause counts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# --------------------------------------------------------------------------- helpers


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def collect(root: Path) -> dict:
    """Gather every figure the report needs, with consistency assertions."""
    root = Path(root)
    reports = root / "reports"

    manifest = _read_json(root / "MANIFEST.json")
    clause_stats = _read_csv(reports / "clause_stats.csv")
    curate = _read_csv(reports / "curation_decisions.csv")
    sources = _read_csv(root / "sources.csv")
    validation = _read_json(reports / "validation.json")
    split_audit = _read_json(root / "annotations" / "split_audit.json")

    # ---- corpus membership: read from the corpus file, which is what experiments use
    # Discover the file instead of hardcoding the "_approx" name: once the real
    # bge-small tokenizer was installed, 03_build_safe_corpus.py started writing
    # corpus_safe_512.jsonl and the hardcoded path silently yielded an empty corpus.
    corpus_dir = root / "corpus_safe_512"
    candidates = sorted(corpus_dir.glob("corpus_safe_512*.jsonl"))
    # prefer the exact-tokenizer file when both exist
    candidates.sort(key=lambda p: (p.name.endswith("_approx.jsonl"), p.name))
    corpus_path = candidates[0] if candidates else corpus_dir / "corpus_safe_512.jsonl"
    chunks: list[dict] = []
    if corpus_path.exists():
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

    corpus_doc_ids: set[str] = {c["doc_id"] for c in chunks}
    tokens_total = sum(_to_int(c.get("token_count")) for c in chunks)
    token_counts = sorted(_to_int(c.get("token_count")) for c in chunks)
    split_chunks = sum(1 for c in chunks if (c.get("split_reason") or "").strip())

    # ---- clause-level stats, authoritative
    stats_by_doc = {r["doc_id"]: r for r in clause_stats}
    corpus_stats = [stats_by_doc[d] for d in corpus_doc_ids if d in stats_by_doc]

    # corpus composition by tier: prefer curation (it holds the effective tier)
    curate_by_doc = {r["doc_id"]: r for r in curate}
    tier_counts: dict[str, int] = {}
    institutions: dict[str, int] = {}
    for doc_id in sorted(corpus_doc_ids):
        row = curate_by_doc.get(doc_id, {})
        tier = row.get("tier") or stats_by_doc.get(doc_id, {}).get("tier") or "unknown"
        inst = row.get("institution") or stats_by_doc.get(doc_id, {}).get("institution") or "unknown"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        institutions[inst] = institutions.get(inst, 0) + 1

    n_corpus = len(corpus_doc_ids)
    tier_shares = {t: round(n / n_corpus, 4) for t, n in tier_counts.items()} if n_corpus else {}

    # ---- capability counts over the corpus (all from clause_stats, never raw hits)
    def _sum(field: str) -> int:
        return sum(_to_int(r.get(field)) for r in corpus_stats)

    capability = {
        "clauses_total": _sum("clauses_total"),
        "clauses_usable": _sum("clauses_usable"),
        "docs_with_exception": sum(1 for r in corpus_stats if _to_int(r.get("clauses_exception"))),
        "docs_with_crossref": sum(1 for r in corpus_stats if _to_int(r.get("clauses_crossref"))),
        "docs_with_alt": sum(1 for r in corpus_stats if _to_int(r.get("clauses_alt"))),
        "clauses_exception": _sum("clauses_exception"),
        "clauses_crossref": _sum("clauses_crossref"),
        "clauses_alt": _sum("clauses_alt"),
        "clauses_boilerplate": _sum("clauses_boilerplate"),
    }

    # ---- collected totals
    collected = {
        "documents": len(sources) or _to_int(manifest.get("sources_count")),
        "institutions": len({r.get("institution", "") for r in sources if r.get("institution")}),
        "clauses_total": sum(_to_int(r.get("clauses_total")) for r in clause_stats),
        "cleaned_documents": len(clause_stats),
    }

    # ---- robots
    robots_hosts = _to_int(manifest.get("robots", {}).get("hosts"))
    if not robots_hosts:
        audit = _read_json(reports / "robots_audit.json")
        robots_hosts = len(audit.get("hosts", {}))

    # ---- acceptance gates, evaluated against the plan's thresholds
    cfg_acceptance = {
        "min_documents": 20, "max_documents": 40,
        "min_institutions": 6,
        "min_docs_with_exceptions": 10,
        "min_docs_with_crossrefs": 8,
        "min_docs_with_alt_evidence": 6,
        "tier_share_max": 0.40,
    }
    gates = [
        ("语料规模 20-40 篇", n_corpus, "20–40",
         cfg_acceptance["min_documents"] <= n_corpus <= cfg_acceptance["max_documents"]),
        ("机构数 ≥6", len(institutions), "≥6",
         len(institutions) >= cfg_acceptance["min_institutions"]),
        ("含例外条款的文档 ≥10", capability["docs_with_exception"], "≥10",
         capability["docs_with_exception"] >= cfg_acceptance["min_docs_with_exceptions"]),
        ("含交叉引用的文档 ≥8", capability["docs_with_crossref"], "≥8",
         capability["docs_with_crossref"] >= cfg_acceptance["min_docs_with_crossrefs"]),
        ("含替代证据的文档 ≥6", capability["docs_with_alt"], "≥6",
         capability["docs_with_alt"] >= cfg_acceptance["min_docs_with_alt_evidence"]),
        ("单层占比 ≤40%", f"{max(tier_shares.values()):.0%}" if tier_shares else "n/a", "≤40%",
         bool(tier_shares) and max(tier_shares.values()) <= cfg_acceptance["tier_share_max"]),
    ]

    # ---- quality control
    quality_report = _read_csv(reports / "quality_report.csv")
    quality = {
        "span_errors": sum(_to_int(r.get("span_errors")) for r in quality_report),
        "chunks_over_max_input": _to_int(validation.get("checks", {}).get("chunks_over_max_input")),
        "blockers": len(validation.get("blockers", [])),
        "warnings": len(validation.get("warnings", [])),
        "validation_status": validation.get("status", "not_run"),
    }

    # test count is read from the test suite itself when available
    test_files = sorted((root / "tests").glob("test_*.py"))
    n_tests = 0
    for tf in test_files:
        text = tf.read_text(encoding="utf-8", errors="ignore")
        n_tests += text.count("def test_")

    consistency = consistency_check(
        corpus_doc_ids, stats_by_doc, tier_shares, quality, corpus_path, chunk_count=len(chunks))

    return {
        "project": manifest.get("project", "course-policy-rag"),
        "corpus_version": manifest.get("corpus_version", "unknown"),
        "frozen_at": manifest.get("frozen_at", ""),
        "random_seed": manifest.get("random_seed"),
        "approximate_tokenizer": bool(manifest.get("chunk_stats", {}).get("approximate_tokenizer",
                                                                        corpus_path.name.endswith("_approx.jsonl"))),
        "corpus_file": corpus_path.name,
        "collected": collected,
        "corpus": {
            "documents": n_corpus,
            "institutions": len(institutions),
            "institution_counts": dict(sorted(institutions.items(), key=lambda kv: (-kv[1], kv[0]))),
            "tier_counts": tier_counts,
            "tier_shares": tier_shares,
            "chunks": len(chunks),
            "tokens_total": tokens_total,
            "token_min": token_counts[0] if token_counts else 0,
            "token_max": token_counts[-1] if token_counts else 0,
            "token_median": token_counts[len(token_counts) // 2] if token_counts else 0,
            "token_mean": round(tokens_total / len(token_counts), 1) if token_counts else 0,
            "chunks_from_split": split_chunks,
        },
        "capability": capability,
        "quality": quality,
        "n_tests": n_tests,
        "robots_hosts": robots_hosts,
        "gates": gates,
        "validation_status": validation.get("status", "not_run"),
        "split_audit": {
            "status": split_audit.get("status", "not_run"),
            "blockers": [b.get("kind") for b in split_audit.get("blockers", [])],
            "counts": split_audit.get("counts", {}),
        },
        "consistency": consistency,
        "documents": _document_rows(corpus_doc_ids, curate_by_doc, stats_by_doc),
    }


def _document_rows(corpus_doc_ids: set[str], curate_by_doc: dict, stats_by_doc: dict) -> list[dict]:
    """Per-document appendix rows, ordered by tier then institution then id."""
    order = {"course": 0, "faculty": 1, "institution": 2}
    rows = []
    for doc_id in corpus_doc_ids:
        cur = curate_by_doc.get(doc_id, {})
        st = stats_by_doc.get(doc_id, {})
        rows.append({
            "doc_id": doc_id,
            "tier": cur.get("tier") or st.get("tier", ""),
            "institution": cur.get("institution") or st.get("institution", ""),
            "clauses": _to_int(st.get("clauses_total")),
            "clauses_usable": _to_int(st.get("clauses_usable")),
            "exception": _to_int(st.get("clauses_exception")),
            "crossref": _to_int(st.get("clauses_crossref")),
            "alt": _to_int(st.get("clauses_alt")),
        })
    rows.sort(key=lambda r: (order.get(r["tier"], 9), r["institution"], r["doc_id"]))
    return rows


def consistency_check(corpus_doc_ids: set[str], stats_by_doc: dict, tier_shares: dict,
                      quality: dict, corpus_path: Path, chunk_count: int) -> dict:
    """Fail loudly when the numbers the report would print disagree with each other."""
    problems: list[str] = []

    if not corpus_path.exists():
        problems.append(f"corpus file missing: {corpus_path.name}")
    if chunk_count == 0:
        problems.append("corpus contains no chunks")
    if not corpus_doc_ids:
        problems.append("corpus contains no documents")
    missing_stats = corpus_doc_ids - set(stats_by_doc)
    if missing_stats:
        problems.append(f"{len(missing_stats)} corpus documents have no clause_stats row")
    if quality.get("span_errors"):
        problems.append(f"clause span errors present: {quality['span_errors']}")
    for tier, share in tier_shares.items():
        if share > 0.4001:
            problems.append(f"tier '{tier}' exceeds the 40% cap ({share:.1%})")

    return {"ok": not problems, "problems": problems}
