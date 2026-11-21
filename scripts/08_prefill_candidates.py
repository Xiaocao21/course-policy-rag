"""08_prefill_candidates.py - propose annotation candidates for the researcher to judge.

IMPORTANT: this script does NOT produce gold annotation. It produces *candidates*:
clauses, rule+exception pairs and alternative-evidence groups that are likely
answerable, with exact character offsets and verbatim quotes so the researcher can
accept, edit or reject them quickly.

The proposal's AI disclosure states that gold source spans and reference answers are
annotated by the student and that AI did not create source annotations. Every artefact
written here therefore carries:
    prefilled = true
    manual_reviewed = false
and `goldset.validate_question(..., require_manual_review_for_test=True)` refuses to
let such a question enter the held-out test split until the researcher sets
manual_reviewed = true. The judgement, and the signature, stay with the student.

What is generated
    candidates/clause_candidates.csv     usable clauses labelled by domain, with flags
    candidates/exception_pairs.csv       a clause with exception language + a candidate
                                         partner rule clause from the same document
    candidates/alt_evidence_groups.csv   clauses that could form alternative sufficient
                                         sets (same domain, different document)
    annotations/question_skeletons.jsonl question skeletons with candidate refs

Known false-positive modes (documented because they are the first thing to check):
    * keyword homonyms: "mark" in "marking scheme" vs "mark the calendar"
    * navigation residue that survived the content gate
    * boilerplate clauses shared across institutions
    * a clause that is on-topic but not *sufficient* to answer anything alone
    * the same policy appearing in two documents (already deduplicated by hash, but
      paraphrased versions can still resemble each other)

Usage:
    .venv\\Scripts\\python.exe scripts/08_prefill_candidates.py
    .venv\\Scripts\\python.exe scripts/08_prefill_candidates.py --per-cell 6 --out-dir candidates
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, load_taxonomy, minhash_signature, minhash_similarity,
    project_paths, read_jsonl, write_json, write_jsonl,
)
import goldset  # noqa: E402

# Planned marginals (kept in step with 07_annotation_feasibility.py)
DOMAIN_QUOTA = {"grading": 35, "integrity": 30, "late": 25,
                "exam": 25, "attendance": 20, "resources": 15}
TYPE_QUOTA = {"single-clause": 50, "rule-and-exception": 55, "cross-clause": 45}

SKELETONS = {
    "grading": "Which policy governs how {topic} is graded, and what does it require?",
    "late": "What happens if I submit work late, and are there any exceptions?",
    "attendance": "What are the attendance requirements, and what happens if I miss a class?",
    "exam": "What are the rules for {topic}, including make-up or deferred arrangements?",
    "integrity": "What counts as a breach of academic integrity, and what process applies?",
    "resources": "What support or resources are available, and who is eligible?",
}


def usable_clauses(corpus: goldset.Corpus, taxonomy: dict) -> list[dict]:
    out = []
    for doc_id, clauses in corpus.by_doc.items():
        for c in clauses:
            if c.boilerplate or c.char_len < 80:
                continue
            domain, hits = goldset.classify_domain(c.text, taxonomy, context=c.section_path)
            out.append({"clause": c, "doc_id": doc_id, "domain": domain,
                        "domain_hits": hits,
                        "tier": corpus.doc_tier.get(doc_id, ""),
                        "institution": corpus.doc_institution.get(doc_id, "")})
    return out


def topic_of(text: str) -> str:
    words = [w.strip(".,;:()[]").lower() for w in text.split()[:12]]
    keep = [w for w in words if len(w) > 3 and w.isalpha()]
    return " ".join(keep[:3]) if keep else "this policy"


def find_exception_pairs(pool: list[dict]) -> list[dict]:
    """Pair a clause carrying exception language with a plausible rule clause.

    The partner is chosen from the same document, same domain, without exception
    language, preferring a shared section prefix and then the nearest clause.
    """
    by_doc_domain: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in pool:
        if item["domain"]:
            by_doc_domain[(item["doc_id"], item["domain"])].append(item)

    pairs = []
    for key, items in by_doc_domain.items():
        exc = [i for i in items if i["clause"].has_exception]
        plain = [i for i in items if not i["clause"].has_exception]
        if not exc or not plain:
            continue
        for e in exc:
            esec = (e["clause"].section_path or "").split(">")[0].strip().lower()

            def rank(p: dict) -> tuple:
                psec = (p["clause"].section_path or "").split(">")[0].strip().lower()
                same_section = 0 if (esec and psec and esec == psec) else 1
                distance = abs(p["clause"].char_start - e["clause"].char_start)
                return (same_section, distance)

            partner = sorted(plain, key=rank)[0]
            pairs.append({
                "doc_id": e["doc_id"],
                "domain": e["domain"],
                "tier": e["tier"],
                "institution": e["institution"],
                "exception_clause_id": e["clause"].clause_id,
                "exception_char_start": e["clause"].char_start,
                "exception_char_end": e["clause"].char_end,
                "exception_quote": e["clause"].text,
                "rule_clause_id": partner["clause"].clause_id,
                "rule_char_start": partner["clause"].char_start,
                "rule_char_end": partner["clause"].char_end,
                "rule_quote": partner["clause"].text,
                "pair_basis": "same document, same domain, nearest / same section",
                "confidence": "medium",
            })
    return pairs


def find_alt_evidence(pool: list[dict], threshold: float = 0.55) -> list[dict]:
    """Find clause pairs from *different* documents that could be alternative evidence.

    Only used to suggest alternative sufficient sets. Similar wording across
    institutions is a hint, not proof that both answer the same question.
    """
    sigs = []
    for item in pool:
        if item["domain"] and item["clause"].char_len >= 80:
            sigs.append((item, minhash_signature(item["clause"].text)))
    groups = []
    used: set[str] = set()
    for i, (a, sa) in enumerate(sigs):
        if a["clause"].clause_id in used:
            continue
        partners = []
        for b, sb in sigs[i + 1:]:
            if a["doc_id"] == b["doc_id"] or a["domain"] != b["domain"]:
                continue
            sim = minhash_similarity(sa, sb)
            if sim >= threshold:
                partners.append((sim, b))
        if partners:
            partners.sort(key=lambda t: -t[0])
            best = partners[0][1]
            used.add(a["clause"].clause_id)
            used.add(best["clause"].clause_id)
            groups.append({
                "domain": a["domain"],
                "clause_a": a["clause"].clause_id,
                "clause_a_doc": a["doc_id"],
                "clause_a_quote": a["clause"].text,
                "clause_b": best["clause"].clause_id,
                "clause_b_doc": best["doc_id"],
                "clause_b_quote": best["clause"].text,
                "similarity": round(partners[0][0], 3),
                "confidence": "low",
                "note": "similarity is a hint only; confirm both clauses truly answer one question",
            })
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--out-dir", default="candidates")
    ap.add_argument("--per-cell", type=int, default=6,
                    help="max skeletons per (domain, evidence_type) cell")
    ap.add_argument("--alt-threshold", type=float, default=0.55)
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)

    corpus = goldset.load_corpus(root)
    if not corpus.clauses:
        print("no corpus loaded; run 02_extract_clean.py and curate first", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = usable_clauses(corpus, taxonomy)
    unlabelled = sum(1 for p in pool if not p["domain"])

    # ---------------------------------------------------------------- clause candidates
    rows = []
    for item in sorted(pool, key=lambda p: (p["domain"], p["doc_id"], p["clause"].char_start)):
        c = item["clause"]
        rows.append({
            "clause_id": c.clause_id,
            "doc_id": c.doc_id,
            "tier": item["tier"],
            "institution": item["institution"],
            "domain": item["domain"],
            "domain_hits": item["domain_hits"],
            "section_path": c.section_path,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "char_len": c.char_len,
            "has_exception": c.has_exception,
            "has_crossref": c.has_crossref,
            "has_alt_evidence": c.has_alt_evidence,
            "quote_evidence": c.text,
            "confidence": "high" if (item["domain"] and item["domain_hits"] >= 2) else "medium",
            "prefilled": "true",
            "manual_reviewed": "false",
        })
    clause_csv = out_dir / "clause_candidates.csv"
    with clause_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["clause_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    pairs = find_exception_pairs(pool)
    pairs_csv = out_dir / "exception_pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()) if pairs else
                           ["doc_id", "exception_clause_id", "rule_clause_id"])
        w.writeheader()
        for r in pairs:
            w.writerow(r)

    alts = find_alt_evidence(pool, args.alt_threshold)
    alt_csv = out_dir / "alt_evidence_groups.csv"
    with alt_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(alts[0].keys()) if alts else
                           ["domain", "clause_a", "clause_b"])
        w.writeheader()
        for r in alts:
            w.writerow(r)

    # ---------------------------------------------------------------- skeletons
    def ref(c: goldset.Clause, corpus_obj: goldset.Corpus) -> dict:
        return {
            "doc_id": c.doc_id,
            "version": corpus_obj.corpus_version,
            "clause_id": c.clause_id,
            "section_path": c.section_path,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "quote_evidence": goldset.evidence_text(corpus_obj, c.doc_id)[c.char_start:c.char_end],
            "token_start": None,
            "token_end": None,
        }

    # bucket clause candidates by (domain, type)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in pool:
        if not item["domain"]:
            continue
        c = item["clause"]
        buckets[(item["domain"], "single-clause")].append(item)
        if c.has_exception:
            buckets[(item["domain"], "rule-and-exception")].append(item)
        if c.has_crossref or c.has_alt_evidence:
            buckets[(item["domain"], "cross-clause")].append(item)

    # enrich rule-and-exception buckets with the paired rule clause
    pair_lookup: dict[tuple[str, str], dict] = {}
    for p in pairs:
        pair_lookup[(p["doc_id"], p["exception_clause_id"])] = p

    skeletons: list[dict] = []
    counter = 0
    for domain, dq in DOMAIN_QUOTA.items():
        for etype, tq in TYPE_QUOTA.items():
            cell = buckets.get((domain, etype), [])
            # spread across documents so one giant document does not dominate
            by_doc: dict[str, list[dict]] = defaultdict(list)
            for item in cell:
                by_doc[item["doc_id"]].append(item)
            picked: list[dict] = []
            docs_sorted = sorted(by_doc.items(), key=lambda kv: -len(kv[1]))
            idx = 0
            while len(picked) < args.per_cell and any(by_doc.values()):
                for doc_id, items in docs_sorted:
                    if idx < len(items) and len(picked) < args.per_cell:
                        picked.append(items[idx])
                idx += 1
                if idx > 50:
                    break

            for item in picked:
                counter += 1
                c = item["clause"]
                qid = f"S{counter:04d}"
                clauses = [ref(c, corpus)]
                confidence = "high"
                condition = "baseline"

                if etype == "rule-and-exception":
                    p = pair_lookup.get((c.doc_id, c.clause_id))
                    if p:
                        rule_clause = corpus.clauses.get(p["rule_clause_id"])
                        if rule_clause is not None:
                            clauses.append(ref(rule_clause, corpus))
                        confidence = "medium"
                        condition = "exception"
                    else:
                        confidence = "low"

                if etype == "cross-clause":
                    confidence = "low" if len(clauses) == 1 else "medium"

                skeletons.append({
                    "skeleton_id": qid,
                    "domain": domain,
                    "evidence_type": etype,
                    "source_level": item["tier"],
                    "institution": item["institution"],
                    "group_id": f"G{counter:04d}",
                    "split": "UNASSIGNED",
                    "condition": condition,
                    "skeleton_question": SKELETONS.get(domain, "What does the policy require?"),
                    "candidate_clause_refs": clauses,
                    "confidence": confidence,
                    "prefilled": True,
                    "manual_reviewed": False,
                    "generated_by": "scripts/08_prefill_candidates.py",
                    "review_notes": (
                        "Rewrite the skeleton into a natural student question, write the "
                        "reference_answer, confirm or replace the clause refs, set split and "
                        "group_id, then set manual_reviewed=true."),
                })

    skeletons_path = paths["annotations"] / "question_skeletons.jsonl"
    write_jsonl(skeletons_path, skeletons)

    summary = {
        "corpus_version": corpus.corpus_version,
        "documents": len(corpus.by_doc),
        "clauses_in_corpus": len(corpus.clauses),
        "usable_clauses": len(pool),
        "clauses_without_domain": unlabelled,
        "clause_candidates": len(rows),
        "exception_pairs": len(pairs),
        "alt_evidence_groups": len(alts),
        "skeletons": len(skeletons),
        "skeletons_by_type": dict(__import__("collections").Counter(s["evidence_type"] for s in skeletons)),
        "skeletons_by_domain": dict(__import__("collections").Counter(s["domain"] for s in skeletons)),
        "disclaimer": (
            "Candidates only. prefilled=true and manual_reviewed=false on every item; "
            "gold annotation remains the researcher's own work per the AI disclosure."),
    }
    write_json(out_dir / "candidates_summary.json", summary)

    print("=== prefill candidates ===")
    print(f"  corpus                : {corpus.corpus_version} "
          f"({len(corpus.by_doc)} docs, {len(corpus.clauses)} clauses)")
    print(f"  usable clauses        : {len(pool)}  (no domain label: {unlabelled})")
    print(f"  clause candidates     : {len(rows)} -> {clause_csv.name}")
    print(f"  exception pairs       : {len(pairs)} -> {pairs_csv.name}")
    print(f"  alt-evidence groups   : {len(alts)} -> {alt_csv.name}")
    print(f"  skeletons             : {len(skeletons)} -> {skeletons_path.name}")
    print(f"    by type   : {summary['skeletons_by_type']}")
    print(f"    by domain : {summary['skeletons_by_domain']}")
    print("\n  Every item is prefilled=true, manual_reviewed=false.")
    print("  Rewrite each skeleton into a question, write the reference answer, confirm the")
    print("  clause refs, then set manual_reviewed=true. Prefilled questions cannot enter")
    print("  the test split until that flag is set.")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
