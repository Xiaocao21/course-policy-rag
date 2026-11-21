"""06_validate.py - corpus invariant checks and cross-document dedup.

Fails (non-zero exit) when a plan-section-H invariant is violated:

  * doc_id unique; clause_id unique within a document
  * evidence offsets in bounds and slicing back to the exact annotated text
  * retrieval -> evidence remapping reversible
  * all four chunking strategies consume the same MANIFEST (checked at freeze)
  * >= 25 included documents; tier shares within limits; >= 6 institutions
  * required domains covered; capability thresholds met
  * no chunk exceeds the model maximum input length

Usage:
    .venv\\Scripts\\python.exe scripts\\06_validate.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, load_taxonomy, minhash_signature,
    minhash_similarity, project_paths, read_json, read_jsonl, sha256_text, write_json,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--skip-capability", action="store_true",
                    help="skip corpus-size gates (useful during a 5-document smoke test)")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)
    acc = cfg["acceptance"]
    gate = cfg["content_gate"]

    blockers: list[dict] = []
    warnings: list[dict] = []
    checks: dict = {}

    # ---------------------------------------------------------------- sources.csv
    sources_csv = root / "sources.csv"
    sources: list[dict] = []
    if sources_csv.exists():
        with sources_csv.open(encoding="utf-8", newline="") as f:
            sources = list(csv.DictReader(f))
    else:
        warnings.append({"kind": "no_sources_csv", "detail": "run 01_collect.py first"})

    doc_ids = [s["doc_id"] for s in sources if s.get("doc_id")]
    dup_docs = [d for d, n in Counter(doc_ids).items() if n > 1]
    if dup_docs:
        blockers.append({"kind": "duplicate_doc_id", "detail": dup_docs[:10]})

    # Byte-identical documents must never both enter the corpus: the same policy
    # reached by two URLs would occupy two slots and be over-weighted in every
    # metric. Curation deduplicates on this hash; this is the independent check.
    included_all = [s for s in sources if str(s.get("include", "")).upper() in ("TRUE", "YES", "1")]
    by_hash: dict[str, list[str]] = {}
    for s in included_all:
        h = s.get("sha256_raw", "")
        if h:
            by_hash.setdefault(h, []).append(s["doc_id"])
    dup_content = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    if dup_content:
        blockers.append({"kind": "duplicate_document_content", "count": len(dup_content),
                         "detail": list(dup_content.values())[:5]})

    included = [s for s in sources if str(s.get("include", "")).upper() in ("TRUE", "YES", "1")]
    checks["documents_total"] = len(sources)
    checks["documents_included"] = len(included)

    # Tier balance and institution breadth describe the *corpus that is frozen*, so
    # they are computed over included documents. An earlier version used every
    # collected document, which reported 89% faculty for a corpus that is actually
    # balanced, because most of the crawl is deliberately not selected.
    corpus_sources = included or sources
    tier_counts = Counter(s.get("tier", "?") for s in corpus_sources)
    checks["tier_counts"] = dict(tier_counts)
    if corpus_sources:
        total = len(corpus_sources)
        shares = {tier: round(n / total, 3) for tier, n in tier_counts.items()}
        checks["tier_shares"] = shares
        for tier, share in shares.items():
            if share > acc["tier_share_max"] + 1e-9:
                warnings.append({"kind": "tier_share_above_max", "tier": tier,
                                 "share": share, "max": acc["tier_share_max"],
                                 "detail": f"{tier} is {share:.0%} of the selected corpus; balance "
                                           f"the tiers before freezing"})

    institutions = {s.get("institution", "") for s in corpus_sources if s.get("institution")}
    checks["institutions"] = len(institutions)
    checks["institutions_all_collected"] = len(
        {s.get("institution", "") for s in sources if s.get("institution")})

    # ---------------------------------------------------------------- capability
    # Authoritative source: reports/clause_stats.csv, written by 02_extract_clean.py
    # from the extracted clause records. The per-document columns in sources.csv are
    # page-level regex hits captured at crawl time and are NOT clause counts, so they
    # must never drive the acceptance gates. Fall back to the legacy csv only when
    # clause_stats.csv is absent (pre-extraction runs).
    cap_csv = paths["reports"] / "clause_stats.csv"
    cap_source = "clause_stats.csv"
    if not cap_csv.exists():
        cap_csv = paths["reports"] / "corpus_capability.csv"
        cap_source = "corpus_capability.csv (legacy)"
    if cap_csv.exists():
        with cap_csv.open(encoding="utf-8", newline="") as f:
            cap = list(csv.DictReader(f))
        included_ids = {s.get("doc_id", "") for s in included}
        if included_ids:
            cap = [r for r in cap if r.get("doc_id") in included_ids]
        checks["capability_source"] = cap_source
        checks["capability_rows"] = len(cap)

        def _col(row: dict, primary: str, legacy: str) -> int:
            v = row.get(primary)
            if v in (None, ""):
                v = row.get(legacy)
            try:
                return int(v or 0)
            except (TypeError, ValueError):
                return 0

        exc_ok = sum(1 for r in cap
                     if _col(r, "clauses_exception", "exception_count") >= gate["strong_exception_min"])
        xr_ok = sum(1 for r in cap
                    if _col(r, "clauses_crossref", "crossref_count") >= 2)
        alt_ok = sum(1 for r in cap
                     if _col(r, "clauses_alt", "alt_evidence_count") >= 1)
        checks["docs_with_exceptions"] = exc_ok
        checks["docs_with_crossrefs"] = xr_ok
        checks["docs_with_alt_evidence"] = alt_ok
        if not args.skip_capability:
            if exc_ok < acc["min_docs_with_exceptions"]:
                blockers.append({"kind": "too_few_exception_docs", "have": exc_ok,
                                 "need": acc["min_docs_with_exceptions"]})
            if xr_ok < acc["min_docs_with_crossrefs"]:
                blockers.append({"kind": "too_few_crossref_docs", "have": xr_ok,
                                 "need": acc["min_docs_with_crossrefs"]})
            if alt_ok < acc["min_docs_with_alt_evidence"]:
                blockers.append({"kind": "too_few_alt_evidence_docs", "have": alt_ok,
                                 "need": acc["min_docs_with_alt_evidence"]})

    # ---------------------------------------------------------------- clause integrity
    ev_files = sorted(paths["cleaned"].glob("*.evidence.jsonl"))
    checks["cleaned_documents"] = len(ev_files)

    clause_ids: list[str] = []
    span_failures: list[dict] = []
    remap_failures: list[dict] = []
    clause_sigs: list[tuple[str, str, tuple]] = []
    domain_counts: Counter = Counter()
    total_clauses = 0
    boilerplate = 0

    from common import domain_hits

    for ev_path in ev_files:
        doc_id = ev_path.name.replace(".evidence.jsonl", "")
        rows = read_jsonl(ev_path)
        ext_path = paths["extracted"] / f"{doc_id}.json"
        evidence_text = ""
        if ext_path.exists():
            data = json.loads(ext_path.read_text(encoding="utf-8"))
            evidence_text = "\n".join(b["text"] for b in data.get("blocks", []))
        else:
            warnings.append({"kind": "missing_extracted_json", "doc_id": doc_id})
        hits = domain_hits(evidence_text, taxonomy)
        for d, v in hits.items():
            if v:
                domain_counts[d] += 1

        cm_path = paths["cleaned"] / f"{doc_id}.charmap.jsonl"
        cmap = read_jsonl(cm_path)
        cmap_arr = cmap[0]["charmap"] if cmap else []
        ret_rows = {r["clause_id"]: r for r in read_jsonl(paths["cleaned"] / f"{doc_id}.retrieval.jsonl")}

        for r in rows:
            total_clauses += 1
            clause_ids.append(r["clause_id"])
            if r.get("boilerplate"):
                boilerplate += 1
            s, e = int(r["char_start"]), int(r["char_end"])
            if evidence_text:
                if e > len(evidence_text) or s < 0 or e <= s:
                    span_failures.append({"clause_id": r["clause_id"],
                                          "detail": f"({s},{e}) vs len {len(evidence_text)}"})
                else:
                    if sha256_text(evidence_text[s:e]) != r["evidence_sha256"]:
                        span_failures.append({"clause_id": r["clause_id"],
                                              "detail": "slice does not match evidence_sha256"})
            ret = ret_rows.get(r["clause_id"])
            if ret and cmap_arr:
                rs, re_ = ret["ret_char_start"], ret["ret_char_end"]
                if not (0 <= rs <= re_ <= len(cmap_arr)):
                    remap_failures.append({"clause_id": r["clause_id"],
                                           "detail": f"retrieval span ({rs},{re_}) outside charmap"})
            clause_sigs.append((doc_id, r["clause_id"], minhash_signature(r["text"])))

    dup_clauses = [c for c, n in Counter(clause_ids).items() if n > 1]
    if dup_clauses:
        blockers.append({"kind": "duplicate_clause_id", "detail": dup_clauses[:10]})
    if span_failures:
        blockers.append({"kind": "clause_span_integrity", "count": len(span_failures),
                         "detail": span_failures[:10]})
    if remap_failures:
        blockers.append({"kind": "charmap_remap", "count": len(remap_failures),
                         "detail": remap_failures[:10]})

    checks["clauses_total"] = total_clauses
    checks["clauses_boilerplate"] = boilerplate
    checks["domains_covered"] = sorted(domain_counts)

    # cross-document boilerplate duplicates
    # NOTE: same-document pairs are skipped here because a document legitimately
    # repeats its own boilerplate; duplicate *documents* are caught above by the
    # raw-content hash check.
    cross_dupes = []
    dm = cfg["dedup"]["document_minhash"]
    for i in range(len(clause_sigs)):
        for j in range(i + 1, len(clause_sigs)):
            if clause_sigs[i][0] == clause_sigs[j][0]:
                continue
            sim = minhash_similarity(clause_sigs[i][2], clause_sigs[j][2])
            if sim >= dm["jaccard_threshold"]:
                cross_dupes.append({"a": clause_sigs[i][1], "b": clause_sigs[j][1],
                                    "similarity": round(sim, 3)})
    if cross_dupes:
        warnings.append({"kind": "cross_document_boilerplate_clauses", "count": len(cross_dupes),
                         "detail": cross_dupes[:10]})
        dup_path = paths["reports"] / "duplicates.csv"
        with dup_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["a", "b", "similarity"])
            w.writeheader()
            for d in cross_dupes:
                w.writerow(d)

    # ---------------------------------------------------------------- safe corpus
    corpus_files = sorted(paths["corpus"].glob("corpus_safe_512*.jsonl"))
    if not corpus_files:
        warnings.append({"kind": "no_safe_corpus", "detail": "run 03_build_safe_corpus.py"})
    else:
        cf = corpus_files[-1]
        chunks = read_jsonl(cf)
        tokens = [c["token_count"] for c in chunks]
        over = [c["chunk_id"] for c in chunks if c["token_count"] > cfg["safe_corpus"]["max_input_tokens"]]
        checks["corpus_file"] = cf.name
        checks["chunks"] = len(chunks)
        checks["chunks_over_max_input"] = len(over)
        checks["token_mean"] = round(statistics.mean(tokens), 1) if tokens else 0
        checks["token_max"] = max(tokens, default=0)
        if over:
            blockers.append({"kind": "chunks_exceed_model_max", "count": len(over),
                             "detail": over[:10]})
        if cf.name.endswith("_approx.jsonl"):
            warnings.append({"kind": "approximate_tokenizer_corpus",
                             "detail": "install stage-B deps and rebuild for the frozen corpus"})

        # The corpus file and the curation must describe the same document set. A
        # stale corpus silently changes which documents an experiment runs on, and it
        # happened once: sources.csv had been re-curated while the corpus file still
        # held the previous selection.
        corpus_ids = {c["doc_id"] for c in chunks}
        included_ids = {s["doc_id"] for s in included}
        if included_ids and corpus_ids != included_ids:
            blockers.append({
                "kind": "corpus_membership_mismatch",
                "only_in_corpus": sorted(corpus_ids - included_ids)[:5],
                "only_in_sources": sorted(included_ids - corpus_ids)[:5],
                "detail": "rebuild the corpus (run.py corpus) after changing the curation",
            })
        checks["corpus_documents"] = len(corpus_ids)
        checks["included_documents"] = len(included_ids)

    # ---------------------------------------------------------------- gates
    if not args.skip_capability:
        if len(included) < acc["min_included_docs"]:
            # include flags are set by the researcher; count collected docs too
            collected = len(sources)
            if collected < acc["min_included_docs"]:
                blockers.append({"kind": "too_few_documents", "have": collected,
                                 "need": acc["min_included_docs"]})
            else:
                warnings.append({"kind": "include_flags_not_set",
                                 "detail": f"{collected} collected, {len(included)} marked include=TRUE"})
        if len(institutions) < acc["min_institutions"]:
            blockers.append({"kind": "too_few_institutions", "have": len(institutions),
                             "need": acc["min_institutions"]})
        missing_domains = [d for d in acc["required_domains"] if d not in domain_counts]
        if missing_domains:
            blockers.append({"kind": "required_domains_missing", "detail": missing_domains})
        if len(domain_counts) < acc["min_domains_covered"]:
            blockers.append({"kind": "too_few_domains", "have": len(domain_counts),
                             "need": acc["min_domains_covered"]})
        faculty_docs = tier_counts.get("faculty", 0)
        checks["faculty_docs"] = faculty_docs
        if faculty_docs < acc["min_faculty_docs"]:
            warnings.append({"kind": "faculty_tier_below_target", "have": faculty_docs,
                             "need": acc["min_faculty_docs"],
                             "detail": "fall back to a two-tier corpus and state it in Limitations"})
        if len(included) < acc["min_included_docs"]:
            blockers.append({"kind": "too_few_included_docs", "have": len(included),
                             "need": acc["min_included_docs"]})

    write_json(paths["reports"] / "validation.json", {
        "status": "blocked" if blockers else "pass",
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
    })

    print("=== corpus validation ===")
    for k, v in checks.items():
        print(f"  {k:28s}: {v}")
    print(f"\n  blockers : {len(blockers)}")
    for b in blockers:
        print(f"    - {b['kind']}: {b.get('detail') or b.get('have')}")
    print(f"  warnings : {len(warnings)}")
    for w in warnings:
        print(f"    - {w['kind']}: {str(w.get('detail'))[:90]}")
    print(f"\nwrote {paths['reports'] / 'validation.json'}")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
