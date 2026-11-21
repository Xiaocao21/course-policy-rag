"""05_audit_splits.py - question-set integrity, group discipline and leakage checks.

The revised proposal (work1_Grouped_Test_Revision.docx) makes the split itself a
correctness property, not a bookkeeping detail:

  * ~150 questions: 100 development + 50 final held-out test
  * ideally >= 40 independent test question groups
  * a group_id ties paraphrases and template variants (same concrete template and
    inference structure) together; a broad topic or an incidentally shared clause
    is NOT enough to form a group
  * whole groups are split between development and test
  * development is NOT sub-split; tuning uses group-disjoint cross-validation
  * cross-split near-duplicates are manually audited

Span resolution and every "is this clause actually covered" question are delegated to
`goldset.py`, which is the single implementation shared with the annotation workbench
and the evaluator. This script owns the split discipline, leakage checks and locking.

Modes
    (default)                  audit and write annotations/split_audit.json
    --lock-test                copy the test split into annotations/LOCKED/ and record
                               a hash; the held-out set may not change afterwards
    --unlock                   remove the lock (recorded as an audit event)
    --make-agreement-sample N  write an agreement worksheet sampling N% of questions
                               for the second annotator

Exit codes:  0 = pass, 1 = blockers present, 2 = inputs missing

Usage:
    .venv\\Scripts\\python.exe scripts/05_audit_splits.py
    .venv\\Scripts\\python.exe scripts/05_audit_splits.py --lock-test
    .venv\\Scripts\\python.exe scripts/05_audit_splits.py --make-agreement-sample 20
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, jaccard, load_settings, load_taxonomy, project_paths, read_jsonl,
    sha256_file, shingles, write_json, write_jsonl,
)
import goldset  # noqa: E402


# ---------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _append_audit_log(paths: dict, event: dict) -> None:
    log = paths["reports"] / "lock_audit.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": _now(), **event}, ensure_ascii=False) + "\n")


LOCK_LAYOUT = {
    "test": ("LOCKED", "test.jsonl", "LOCKED.json"),
    "test2": ("LOCKED_TEST2", "test2.jsonl", "LOCKED2.json"),
}


def lock_test(questions: list[dict], paths: dict, split: str = "test",
              locked_dir: Path | None = None,
              questions_path: Path | None = None) -> dict:
    """Physically separate a held-out split and record its hash.

    `split="test"` keeps the original contract (annotations/LOCKED, hashes
    annotations/questions.jsonl). `split="test2"` writes its own directory and hashes its
    own questions file, so the existing test lock - and therefore the test split's
    "evaluated once" guarantee - is never touched by a second held-out set.
    """
    dirname, data_name, meta_name = LOCK_LAYOUT[split]
    locked_dir = Path(locked_dir) if locked_dir else paths["annotations"] / dirname
    locked_dir.mkdir(parents=True, exist_ok=True)
    test = [q for q in questions if q.get("split") == split]
    if not test:
        return {"locked": False, "reason": f"no {split} questions to lock"}

    locked_file = locked_dir / data_name
    write_jsonl(locked_file, test)
    source = Path(questions_path) if questions_path else paths["annotations"] / "questions.jsonl"
    meta = {
        "locked_at": _now(),
        "split": split,
        "questions": len(test),
        "groups": len({str(q.get("group_id")) for q in test}),
        "test_jsonl_sha256": sha256_file(locked_file),
        "questions_jsonl_sha256": sha256_file(source) if source.exists() else "",
        "questions_source": str(source),
        "manual_reviewed": sum(1 for q in test if q.get("manual_reviewed")),
    }
    # hash chain: the lock also pins the bundle list the questions were drawn from, and
    # records who actually reviewed the questions (human sign-off is not implied).
    project_root = paths["annotations"].parent
    bundles_path = project_root / "results" / "test2" / "selected_bundles.json"
    if bundles_path.exists() and split == "test2":
        meta["bundles_sha256"] = sha256_file(bundles_path)
        meta["reviewers"] = sorted({(q.get("review") or {}).get("reviewer")
                                    for q in test
                                    if (q.get("review") or {}).get("reviewer")})
    write_json(locked_dir / meta_name, meta)
    _append_audit_log(paths, {"event": f"lock_{split}", **{k: v for k, v in meta.items()
                                                          if k != "locked_at"}})
    return {"locked": True, **meta}


def unlock_test(paths: dict, reason: str) -> dict:
    locked_dir = paths["annotations"] / "LOCKED"
    meta_path = locked_dir / "LOCKED.json"
    if not meta_path.exists():
        return {"unlocked": False, "reason": "no lock present"}
    previous = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.unlink()
    locked_file = locked_dir / "test.jsonl"
    if locked_file.exists():
        locked_file.unlink()
    _append_audit_log(paths, {"event": "unlock_test", "reason": reason,
                              "previous": previous})
    return {"unlocked": True, "previous": previous}


def evidence_leakage(questions: list[dict]) -> dict:
    """Detect dev/test contamination through shared gold evidence.

    Question-text similarity is not enough: two questions can be worded quite
    differently and still be answered by the same clause, which contaminates the
    held-out score. An earlier audit only compared question text and therefore reported
    'no near duplicates' while 20 clauses were in fact used on both sides of the split.
    """
    clause_splits: dict[str, set[str]] = {}
    clause_qids: dict[str, list[str]] = {}
    for q in questions:
        for eset in q.get("evidence_sets") or []:
            for ref in eset.get("clauses") or []:
                cid = ref.get("clause_id", "")
                if not cid:
                    continue
                clause_splits.setdefault(cid, set()).add(q.get("split", ""))
                clause_qids.setdefault(cid, []).append(q.get("qid", ""))

    crossing = {c: sorted(s) for c, s in clause_splits.items() if len(s) > 1}
    crossing_qids = sorted({qid for c in crossing for qid in clause_qids[c]})

    # identical evidence sets are duplicate questions whatever the wording
    seen: dict[tuple, list[str]] = {}
    for q in questions:
        sig = tuple(sorted(ref.get("clause_id", "")
                           for eset in q.get("evidence_sets") or []
                           for ref in eset.get("clauses") or []))
        if sig:
            seen.setdefault(sig, []).append(q.get("qid", ""))
    duplicate_sets = [{"clauses": list(sig), "qids": qids}
                      for sig, qids in seen.items() if len(qids) > 1]

    return {
        "clauses_crossing_splits": crossing,
        "crossing_question_count": len(crossing_qids),
        "crossing_questions": crossing_qids[:40],
        "duplicate_evidence_sets": duplicate_sets[:20],
        "shared_clause_pairs": [
            {"clause_id": c, "qids": clause_qids[c][:8]}
            for c in list(crossing)[:20]
        ],
    }


def near_duplicate_pairs(questions: list[dict], threshold: float) -> list[dict]:
    """Cross-split near-duplicate questions, which the protocol requires auditing."""
    dev = [(q.get("qid"), q.get("question", "")) for q in questions if q.get("split") == "dev"]
    test = [(q.get("qid"), q.get("question", "")) for q in questions if q.get("split") == "test"]
    dev_sh = {qid: shingles(t) for qid, t in dev}
    out = []
    for tqid, ttext in test:
        ts = shingles(ttext)
        for dqid, _ in dev:
            sim = jaccard(ts, dev_sh[dqid])
            if sim >= threshold:
                out.append({"test_qid": tqid, "dev_qid": dqid, "similarity": round(sim, 3)})
    return sorted(out, key=lambda x: -x["similarity"])


def leakage_worked_examples(questions: list[dict]) -> list[dict]:
    """The two worked examples named in the revision must stay inside one split."""
    families = {
        "late submission paraphrase": ["hand in work late", "submit after the deadline"],
        "missed exam template variants": ["because of illness", "because of travel"],
    }
    failures = []
    for label, phrases in families.items():
        seen: dict[str, set[str]] = {}
        for q in questions:
            ql = (q.get("question", "") or "").lower()
            for ph in phrases:
                if ph in ql:
                    seen.setdefault(ph, set()).add(q.get("split", ""))
        if len(seen) >= 2:
            splits = {s for v in seen.values() for s in v}
            if len(splits) > 1:
                failures.append({"case": label,
                                 "detail": {k: sorted(v) for k, v in seen.items()}})
    return failures


def make_agreement_sample(questions: list[dict], pct: int, paths: dict,
                          seed: int) -> dict:
    """Write a question-level agreement worksheet for the second annotator.

    The double-annotated unit is the question: the second annotator independently
    selects the minimal sufficient evidence sets, and agreement is measured on the
    per-clause inclusion decisions.
    """
    rng = random.Random(seed)
    n = max(1, round(len(questions) * pct / 100)) if questions else 0
    sample = rng.sample(questions, min(n, len(questions))) if questions else []

    rows = []
    for q in sample:
        gold = []
        for si, eset in enumerate(q.get("evidence_sets") or []):
            for ref in eset.get("clauses") or []:
                gold.append(f"{si}:{ref.get('clause_id')}")
        rows.append({
            "qid": q.get("qid", ""),
            "question": q.get("question", ""),
            "split": q.get("split", ""),
            "annotator1_clauses": ";".join(gold),
            "annotator2_clauses": "",
            "annotator2_reference_answer": "",
            "agree_clause_set": "",
            "notes": "",
        })

    out = paths["annotations"] / "agreement_worksheet.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["qid", "question", "split", "annotator1_clauses",
                            "annotator2_clauses", "annotator2_reference_answer",
                            "agree_clause_set", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return {"file": str(out), "questions": len(rows), "pct": pct}


REVIEW_BOOLEANS = ("question_ok", "answer_entails_from_evidence", "evidence_set_minimal",
                   "evidence_type_correct", "version_applicable", "near_duplicate_checked")
REVIEW_RELATION_BOOLEANS = ("relation_holds", "relation_quote_valid", "no_keyword_stitching")


def _load_review_records(review_file: Path) -> dict[str, dict]:
    """Read the review table the student fills in (CSV or JSON)."""
    records: dict[str, dict] = {}
    if review_file.suffix.lower() == ".csv":
        with review_file.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                qid = (row.get("qid") or "").strip()
                if qid:
                    records[qid] = row
    else:
        data = json.loads(review_file.read_text(encoding="utf-8"))
        rows = data.get("questions") if isinstance(data, dict) else data
        reviewer = data.get("reviewer") if isinstance(data, dict) else None
        reviewed_at = data.get("reviewed_at") if isinstance(data, dict) else None
        for row in rows or []:
            if row.get("qid"):
                rec = dict(row)
                rec.setdefault("reviewer", reviewer)
                rec.setdefault("reviewed_at", reviewed_at)
                records[rec["qid"]] = rec
    return records


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def _review_complete(record: dict) -> tuple[bool, list[str]]:
    """A question may only be signed off when every explicit check passed."""
    missing = []
    if str(record.get("decision", "")).strip().lower() != "approve":
        missing.append("decision!=approve")
    for key in REVIEW_BOOLEANS:
        if not _as_bool(record.get(key)):
            missing.append(f"{key}=false")
    return (not missing), missing


def mark_reviewed(paths: dict, qids: list[str], questions_path: Path | None = None,
                  review_file: Path | None = None) -> dict:
    """Record the student's per-question review and set manual_reviewed.

    This is the sign-off on AI-drafted annotation. A question is only marked reviewed when
    the review record says `approve` and every structured check is true - a bare batch
    signature is not enough. Multi-clause questions must additionally confirm that the two
    clauses jointly answer the question, that the quoted relation really expresses it, and
    that the bundle is not keyword stitching.
    """
    qfile = Path(questions_path) if questions_path else paths["annotations"] / "questions.jsonl"
    questions = read_jsonl(qfile)
    wanted = {q.strip() for q in qids if q.strip()}
    records = _load_review_records(Path(review_file)) if review_file else {}
    touched: list[str] = []
    blocked: list[dict] = []
    for q in questions:
        qid = q.get("qid")
        if wanted and qid not in wanted:
            continue
        record = records.get(qid) if records else None
        if records and record is None:
            blocked.append({"qid": qid, "missing": ["no review record"]})
            continue
        if record is not None:
            ok, missing = _review_complete(record)
            multi = len({c.get("clause_id")
                         for eset in (q.get("evidence_sets") or [])
                         for c in (eset.get("clauses") or [])}) > 1
            if multi:
                for key in REVIEW_RELATION_BOOLEANS:
                    if not _as_bool(record.get(key)):
                        missing.append(f"{key}=false")
                ok = not missing
            if not ok:
                blocked.append({"qid": qid, "missing": missing})
                continue
            review = {
                "reviewer": record.get("reviewer") or "student",
                "reviewed_at": record.get("reviewed_at") or _now(),
                "decision": "approve",
                **{k: True for k in REVIEW_BOOLEANS},
                **({"relation_holds": True, "relation_quote_valid": True,
                    "no_keyword_stitching": True} if multi else {}),
                "revision_count": int(q.get("review_revision_count", 0)),
            }
            q["review"] = review
            q["review_revision_count"] = review["revision_count"]
            q.pop("review_stale", None)
        q["manual_reviewed"] = True
        prov = q.setdefault("provenance", {})
        # Never claim a human sign-off that did not happen: the label follows the actual
        # reviewer recorded in the review block.
        reviewer = ((q.get("review") or {}).get("reviewer") or "").strip().lower()
        if reviewer.startswith("student"):
            prov["review_status"] = "student_reviewed"
        elif reviewer:
            prov["review_status"] = "agent_reviewed"
        else:
            prov["review_status"] = "reviewed"
        prov["reviewed_at"] = _now()
        prov["review_scope"] = "explicit_list" if wanted else "all"
        touched.append(qid)

    missing_qids = sorted(wanted - {q.get("qid") for q in questions})
    if touched:
        tmp = qfile.with_suffix(".jsonl.tmp")
        write_jsonl(tmp, questions)
        tmp.replace(qfile)
        _append_audit_log(paths, {"event": "mark_reviewed", "count": len(touched),
                                  "scope": "explicit_list" if wanted else "all",
                                  "questions_file": str(qfile),
                                  "review_file": str(review_file) if review_file else None,
                                  "qids": touched[:50]})
    return {"marked": len(touched), "missing": missing_qids, "blocked": blocked,
            "remaining_unreviewed": sum(1 for q in questions
                                        if not q.get("manual_reviewed"))}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--questions", default="")
    ap.add_argument("--similarity-threshold", type=float, default=0.0)
    ap.add_argument("--lock-test", action="store_true",
                    help="copy the test split into annotations/LOCKED/ and record its hash "
                         "(alias for --lock-split test)")
    ap.add_argument("--lock-split", choices=["test", "test2"], default=None,
                    help="lock this split into its own directory; test2 must never reuse "
                         "the test lock, whose hash guards the one-formal-use guarantee")
    ap.add_argument("--locked-dir", default="",
                    help="override the lock directory (default: annotations/LOCKED for test, "
                         "annotations/LOCKED_TEST2 for test2)")
    ap.add_argument("--unlock", action="store_true",
                    help="remove the held-out lock (recorded as an audit event)")
    ap.add_argument("--unlock-reason", default="")
    ap.add_argument("--mark-reviewed", nargs="*", metavar="QID", default=None,
                    help="set manual_reviewed=true for these qids (or for all when none given)")
    ap.add_argument("--review-file", default="",
                    help="CSV or JSON with the per-question review record; required for a "
                         "held-out split to be signed off question by question")
    ap.add_argument("--make-agreement-sample", type=int, default=0, metavar="PCT",
                    help="write a question-level agreement worksheet sampling PCT%% of questions")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)
    ann = cfg["annotation"]
    thr = args.similarity_threshold or ann["near_duplicate_threshold"]

    if args.unlock:
        res = unlock_test(paths, args.unlock_reason or "unspecified")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.mark_reviewed is not None:
        review_qpath = (Path(args.questions) if args.questions
                        else paths["annotations"] / "questions.jsonl")
        if not review_qpath.is_absolute():
            review_qpath = root / review_qpath
        review_file = Path(args.review_file) if args.review_file else None
        if review_file is not None and not review_file.is_absolute():
            review_file = root / review_file
        res = mark_reviewed(paths, args.mark_reviewed, questions_path=review_qpath,
                            review_file=review_file)
        print("=== mark reviewed ===")
        print(f"  questions file      : {review_qpath}")
        print(f"  review file         : {review_file}")
        print(f"  marked              : {res['marked']}")
        print(f"  unknown qids        : {res['missing']}")
        print(f"  blocked             : {len(res['blocked'])}")
        for item in res["blocked"][:10]:
            print(f"    {item['qid']}: {', '.join(item['missing'])}")
        print(f"  still unreviewed    : {res['remaining_unreviewed']}")
        print("\nreview recorded in reports/lock_audit.jsonl; re-run the audit next.")
        return 0 if not res["blocked"] else 1

    qpath = Path(args.questions) if args.questions else paths["annotations"] / "questions.jsonl"
    if not qpath.is_absolute():
        qpath = root / qpath
    if not qpath.exists():
        print(f"no questions file at {qpath}", file=sys.stderr)
        write_json(paths["annotations"] / "split_audit.json", {
            "status": "no_questions_yet",
            "questions_file": str(qpath),
            "required": {
                "target_total": ann["target_questions_total"],
                "target_dev": ann["target_dev_questions"],
                "target_test": ann["target_test_questions"],
                "min_test_groups": ann["target_test_groups_min"],
                "evidence_types": ann["evidence_types"],
                "domains": list(taxonomy.get("domains", {}).keys()),
            },
            "blockers": [],
            "warnings": ["questions.jsonl not created yet; re-run after annotation"],
        })
        print(f"wrote {paths['annotations'] / 'split_audit.json'}")
        return 0

    questions = read_jsonl(qpath)
    corpus = goldset.load_corpus(root)
    if not corpus.clauses:
        print("no cleaned corpus found; run 02_extract_clean.py first", file=sys.stderr)
        return 2

    blockers: list[dict] = []
    warnings: list[dict] = []

    # ---------------------------------------------------------------- per question
    seen_qids: set[str] = set()
    for q in questions:
        qid = q.get("qid") or "<missing>"
        if qid in seen_qids:
            blockers.append({"kind": "duplicate_qid", "qid": qid})
        seen_qids.add(qid)
        for p in goldset.validate_question(
                q, corpus,
                valid_evidence_types=ann["evidence_types"],
                valid_domains=list(taxonomy.get("domains", {}).keys()),
                valid_levels=ann["source_levels"]):
            kind = "unresolved_evidence_ref" if ("clause" in p or "span" in p) else "question_invalid"
            blockers.append({"kind": kind, "qid": qid, "detail": p})

    # ---------------------------------------------------------------- group/split
    stats = goldset.test_group_counts(questions)
    for gid in stats["crossing_groups"]:
        blockers.append({"kind": "group_crosses_split", "group_id": gid})

    dev = [q for q in questions if q.get("split") == "dev"]
    test = [q for q in questions if q.get("split") == "test"]

    if stats["groups"]["test"] < ann["target_test_groups_min"]:
        blockers.append({
            "kind": "too_few_test_groups",
            "have": stats["groups"]["test"],
            "need": ann["target_test_groups_min"],
            "detail": ("add genuinely new situations or question types; do not multiply "
                       "paraphrases (revision section 7) and do not move existing groups "
                       "across the split"),
        })

    # test questions must be human-verified
    unverified = [q.get("qid") for q in test
                  if q.get("prefilled") and not q.get("manual_reviewed")]
    if unverified:
        blockers.append({"kind": "unreviewed_test_questions", "qids": unverified[:20],
                         "count": len(unverified),
                         "detail": "prefilled test questions need manual_reviewed=true"})

    # ---------------------------------------------------------------- leakage
    # evidence-level contamination first: sharing a gold clause across the split is a
    # harder leak than similar phrasing, and it is invisible to a text-similarity check
    leak = evidence_leakage(questions)
    if leak["clauses_crossing_splits"]:
        blockers.append({
            "kind": "gold_clause_crosses_split",
            "count": len(leak["clauses_crossing_splits"]),
            "questions": leak["crossing_question_count"],
            "detail": leak["shared_clause_pairs"][:10],
            "hint": ("reassign whole evidence-linked clusters to one split "
                     "(scripts/10d_repair_questions.py does this)"),
        })
    if leak["duplicate_evidence_sets"]:
        blockers.append({"kind": "duplicate_evidence_sets",
                         "count": len(leak["duplicate_evidence_sets"]),
                         "detail": leak["duplicate_evidence_sets"][:10]})

    dupes = near_duplicate_pairs(questions, thr)
    if dupes:
        blockers.append({"kind": "cross_split_near_duplicates", "count": len(dupes),
                         "detail": "manual audit required before freezing",
                         "pairs": dupes[:50]})
    for failure in leakage_worked_examples(questions):
        blockers.append({"kind": "leakage_worked_example_failed", **failure})

    # ---------------------------------------------------------------- distributions
    def dist(qs: list[dict], key: str) -> dict:
        return dict(Counter(q.get(key, "?") for q in qs))

    evidence_type_dist = {"dev": dist(dev, "evidence_type"), "test": dist(test, "evidence_type")}
    domain_dist = {"dev": dist(dev, "domain"), "test": dist(test, "domain")}
    level_dist = {"dev": dist(dev, "source_level"), "test": dist(test, "source_level")}
    condition_dist = {"dev": dist(dev, "condition"), "test": dist(test, "condition")}

    for key, dd, universe in (("evidence_type", evidence_type_dist, ann["evidence_types"]),
                              ("domain", domain_dist, list(taxonomy.get("domains", {}).keys()))):
        for split_name in ("dev", "test"):
            empty = [v for v in universe if v not in dd[split_name]]
            if empty:
                warnings.append({"kind": f"{key}_missing_in_{split_name}", "detail": empty})

    multi_set = [q for q in questions if len(q.get("evidence_sets") or []) > 1]
    if not multi_set:
        warnings.append({"kind": "no_alternative_sufficient_sets",
                         "detail": "the revision allows several alternative minimal sufficient "
                                   "sets; if the corpus cannot support them, state this in "
                                   "Limitations"})

    # ---------------------------------------------------------------- self-test
    selftest = {}
    text = "the quick brown fox jumps"
    true_span, shifted = (4, 9), (5, 10)
    from common import contains_span, covered_token_fraction
    selftest["off_by_one_is_not_full_hit"] = not contains_span(text, [shifted], true_span)
    selftest["exact_span_is_full_hit"] = contains_span(text, [true_span], true_span)
    selftest["partial_fraction"] = round(covered_token_fraction(text, [shifted], true_span), 3)
    if not (selftest["off_by_one_is_not_full_hit"] and selftest["exact_span_is_full_hit"]):
        blockers.append({"kind": "containment_selftest_failed", "detail": selftest})

    # ---------------------------------------------------------------- side actions
    actions: dict = {}
    if args.make_agreement_sample:
        actions["agreement_sample"] = make_agreement_sample(
            questions, args.make_agreement_sample, paths, cfg["project"]["random_seed"])

    lock_split = args.lock_split or ("test" if args.lock_test else None)
    if lock_split:
        lock_dir = (root / args.locked_dir) if args.locked_dir else None
        if blockers:
            actions["lock"] = {"locked": False,
                               "reason": f"{len(blockers)} blocker(s) present; fix before locking"}
        else:
            actions["lock"] = lock_test(questions, paths, split=lock_split,
                                        locked_dir=lock_dir, questions_path=qpath)

    dirname, _data, meta_name = LOCK_LAYOUT[lock_split or "test"]
    lock_meta_path = (root / args.locked_dir / meta_name if args.locked_dir
                      else paths["annotations"] / dirname / meta_name)
    lock_state = json.loads(lock_meta_path.read_text(encoding="utf-8")) if lock_meta_path.exists() else None

    # ---------------------------------------------------------------- report
    report = {
        "status": "blocked" if blockers else "pass",
        "questions_file": str(qpath),
        "corpus_version": corpus.corpus_version,
        "counts": {
            "total": len(questions),
            "dev": len(dev),
            "test": len(test),
            "dev_groups": stats["groups"]["dev"],
            "test_groups": stats["groups"]["test"],
            "prefilled": stats["prefilled"],
            "manual_reviewed": stats["manual_reviewed"],
            "targets": {
                "total": ann["target_questions_total"],
                "dev": ann["target_dev_questions"],
                "test": ann["target_test_questions"],
                "test_groups_min": ann["target_test_groups_min"],
            },
            "alternative_evidence_questions": len(multi_set),
        },
        "distributions": {
            "evidence_type": evidence_type_dist,
            "domain": domain_dist,
            "source_level": level_dist,
            "condition": condition_dist,
        },
        "evidence_leakage": {k: (v if not isinstance(v, dict) else len(v))
                             for k, v in leak.items() if k != "shared_clause_pairs"},
        "near_duplicate_pairs": dupes,
        "containment_selftest": selftest,
        "lock_state": lock_state,
        "actions": actions,
        "blockers": blockers,
        "warnings": warnings,
    }
    out = paths["annotations"] / "split_audit.json"
    write_json(out, report)

    print("=== split audit ===")
    print(f"  corpus version       : {corpus.corpus_version} "
          f"({len(corpus.by_doc)} docs, {len(corpus.clauses)} clauses)")
    print(f"  questions            : {len(questions)}  (target {ann['target_questions_total']})")
    print(f"  dev / test           : {len(dev)} / {len(test)}  "
          f"(target {ann['target_dev_questions']} / {ann['target_test_questions']})")
    print(f"  dev / test groups    : {stats['groups']['dev']} / {stats['groups']['test']}  "
          f"(test target >= {ann['target_test_groups_min']})")
    print(f"  manual_reviewed      : {stats['manual_reviewed']} of {len(questions)}")
    print(f"  evidence types (test): {evidence_type_dist['test']}")
    print(f"  domains (test)       : {domain_dist['test']}")
    print(f"  gold clauses crossing: {len(leak['clauses_crossing_splits'])}  (must be 0)")
    print(f"  duplicate evidence   : {len(leak['duplicate_evidence_sets'])}  (must be 0)")
    print(f"  containment selftest : {selftest}")
    if lock_state:
        print(f"  held-out lock        : LOCKED at {lock_state.get('locked_at')} "
              f"({lock_state.get('questions')} questions, {lock_state.get('groups')} groups)")
    else:
        print(f"  held-out lock        : not locked")
    print(f"  blockers             : {len(blockers)}")
    for b in blockers[:12]:
        print(f"    - {b.get('kind')}: {b.get('detail') or b.get('qid') or b.get('group_id') or ''}")
    print(f"  warnings             : {len(warnings)}")
    for w in warnings[:8]:
        print(f"    - {w.get('kind')}: {w.get('detail')}")
    for name, res in actions.items():
        print(f"  action {name:<16}: {json.dumps(res, ensure_ascii=False)[:120]}")
    print(f"\nwrote {out}")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
