"""27_check_test2.py - validate, group, review-packet and lock the second held-out set.

This script never touches retrieval or any metric: the questions must be fixed before anyone
sees how they rank. Its job is the four mechanical gates that a human review cannot replace:

    schema + verbatim evidence      every ref resolves; every quote matches the evidence layer
    leakage                         (doc_id, version, clause_id) AND character-interval overlap
                                    against everything already used by dev and test
    redundancy                      near-duplicate questions, duplicate evidence sets
    composition                     length quartiles, domain quota, institution cap, groups

It also computes the *real* group structure. A unique group id per question is not evidence of
independence: questions that share a concrete template and an inference shape, an evidence
set, or near-identical wording must share a group. Pairs that merely share a broad signature
are only flagged for the reviewer.

Usage
    .venv\\Scripts\\python.exe scripts/27_check_test2.py --check-only
    .venv\\Scripts\\python.exe scripts/27_check_test2.py --assign-groups
    .venv\\Scripts\\python.exe scripts/27_check_test2.py --review-packet
    .venv\\Scripts\\python.exe scripts/27_check_test2.py --lock
    .venv\\Scripts\\python.exe scripts/27_check_test2.py --verify-hashes
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, load_settings, project_paths, read_jsonl, write_json  # noqa: E402
import goldset  # noqa: E402

JACCARD = 0.85
TEMPLATE_JACCARD = 0.80
MIN_GROUPS = 40
QUARTILES = (728.0, 1009.5, 1121.8)
DOMAIN_TARGETS = {"exam": 13, "grading": 12, "integrity": 12, "late": 8,
                  "attendance": 8, "resources": 7}
INSTITUTION_CAP = 11


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(rel: str, name: str, root: Path):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ loads
def load_all(root: Path, questions_path: Path):
    paths = project_paths(root)
    bundles_path = root / "results" / "test2" / "selected_bundles.json"
    amendments_path = root / "results" / "test2" / "bundle_amendments.jsonl"
    if not bundles_path.exists():
        raise SystemExit(f"REFUSING: {bundles_path} missing; freeze the bundles first")
    frozen = json.loads(bundles_path.read_text(encoding="utf-8"))
    amendments = read_jsonl(amendments_path) if amendments_path.exists() else []
    bundles = {b["bundle_id"]: b for b in frozen["bundles"]}

    def clauses_of(bundle: dict) -> list[dict]:
        return [bundle["primary"]] + list(bundle.get("chosen_companions") or [])

    # replay amendments: replace_clause / change_type / drop
    replaced: dict[str, str] = {}
    dropped: set[str] = set()
    for a in amendments:
        bid = a.get("bundle_id")
        if a.get("action") == "drop":
            dropped.add(bid)
        elif a.get("action") == "replace_clause" and bid in bundles:
            replaced[a.get("old", "")] = a.get("new", "")
        elif a.get("action") == "change_type" and bid in bundles:
            bundles[bid]["type"] = a.get("new")

    questions = read_jsonl(questions_path) if questions_path.exists() else []
    corpus = goldset.load_corpus(root)
    existing = read_jsonl(paths["annotations"] / "questions.jsonl")
    return paths, frozen, bundles, amendments, replaced, dropped, questions, corpus, existing


def question_clause_ids(q: dict) -> list[str]:
    return [c.get("clause_id")
            for eset in (q.get("evidence_sets") or [])
            for c in (eset.get("clauses") or [])]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def question_lengths(q: dict, corpus) -> int:
    return max((corpus.clauses[c].char_len for c in question_clause_ids(q)
                if c in corpus.clauses), default=0)


# ------------------------------------------------------------------ checks
def check_questions(questions, corpus, bundles, replaced, dropped, existing,
                    config: dict) -> tuple[list[dict], dict]:
    problems: list[dict] = []
    valid_types = set(config["annotation"]["evidence_types"])
    valid_levels = set(config["annotation"]["source_levels"]) if "source_levels" in config["annotation"] else None
    domains = set(config.get("domains") or []) or None
    if valid_levels is None:
        valid_levels = {"course", "faculty", "institution"}

    used_existing = {c for q in existing for c in question_clause_ids(q)}
    # interval map for the interval-level leakage check
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for q in existing:
        for eset in (q.get("evidence_sets") or []):
            for ref in (eset.get("clauses") or []):
                cid = ref.get("clause_id", "")
                clause = corpus.clauses.get(cid)
                if clause:
                    intervals[clause.doc_id].append((clause.char_start, clause.char_end))

    seen_qids: set[str] = set()
    seen_sets: dict[tuple, str] = {}
    for q in questions:
        qid = q.get("qid", "<missing>")
        if qid in seen_qids:
            problems.append({"qid": qid, "problem": "duplicate qid"})
        seen_qids.add(qid)
        for p in goldset.validate_question(q, corpus,
                                           valid_evidence_types=valid_types,
                                           valid_levels=valid_levels,
                                           require_manual_review_for_test=False):
            problems.append({"qid": qid, "problem": p})

        cids = question_clause_ids(q)
        key = tuple(sorted(cids))
        if key in seen_sets:
            problems.append({"qid": qid, "problem": f"duplicate evidence set with {seen_sets[key]}"})
        seen_sets[key] = qid

        for cid in cids:
            if cid in used_existing:
                problems.append({"qid": qid, "problem": f"clause {cid} is used by dev/test"})
            clause = corpus.clauses.get(cid)
            if clause:
                for (s, e) in intervals.get(clause.doc_id, []):
                    if clause.char_start < e and s < clause.char_end:
                        problems.append({"qid": qid,
                                         "problem": f"character interval overlaps dev/test "
                                                    f"gold span in {clause.doc_id}"})

        bundle_id = (q.get("provenance") or {}).get("bundle_id")
        if not bundle_id or bundle_id not in bundles:
            problems.append({"qid": qid, "problem": f"unknown bundle_id {bundle_id!r}"})
        else:
            bundle = bundles[bundle_id]
            expected = [bundle["primary"]["clause_id"]] + [
                r["clause_id"] for r in (bundle.get("chosen_companions") or [])]
            expected = [replaced.get(c, c) for c in expected]
            if sorted(cids) != sorted(expected):
                problems.append({"qid": qid,
                                 "problem": f"evidence {sorted(cids)} != bundle {sorted(expected)}"})
            if q.get("evidence_type") != bundle.get("type"):
                problems.append({"qid": qid, "problem": "evidence_type differs from bundle type"})
            if bundle_id in dropped:
                problems.append({"qid": qid, "problem": "bundle was dropped by amendment"})

        text = (q.get("question") or "").strip()
        if len(text.split()) < 6:
            problems.append({"qid": qid, "problem": "question is too short"})
        if not any(t in text.lower() for cid in cids
                   for t in tokens(corpus.clauses[cid].text) if len(t) >= 5):
            problems.append({"qid": qid,
                             "problem": "no content word from the evidence appears in the question"})

    # near duplicates: within test2 and against the existing 172
    all_q = [(q.get("qid"), tokens(q.get("question", ""))) for q in questions]
    for i, (qa, ta) in enumerate(all_q):
        for qb, tb in all_q[i + 1:]:
            if jaccard(ta, tb) >= JACCARD:
                problems.append({"qid": qa, "problem": f"near-duplicate wording with {qb}"})
        for old in existing:
            if jaccard(ta, tokens(old.get("question", ""))) >= JACCARD:
                problems.append({"qid": qa,
                                 "problem": f"near-duplicate of existing question {old.get('qid')}"})

    lengths = sorted(question_lengths(q, corpus) for q in questions)
    quart = [float(__import__("numpy").percentile(lengths, p)) for p in (25, 50, 75)] \
        if lengths else [0.0, 0.0, 0.0]
    composition = {
        "questions": len(questions),
        "length_quartiles": [round(v, 1) for v in quart],
        "length_target": list(QUARTILES),
        "length_ok": len(quart) == 3 and all(
            abs(quart[i] - QUARTILES[i]) <= 0.10 * QUARTILES[i] for i in range(3)),
        "domains": dict(Counter(q.get("domain") for q in questions)),
        "institutions": dict(Counter((q.get("provenance") or {}).get("institution")
                                     for q in questions)),
        "evidence_types": dict(Counter(q.get("evidence_type") for q in questions)),
    }
    if composition["length_ok"] is False:
        problems.append({"qid": "-", "problem": f"length quartiles {composition['length_quartiles']} "
                                                f"outside 10% of {list(QUARTILES)}"})
    for d, target in DOMAIN_TARGETS.items():
        have = composition["domains"].get(d, 0)
        if abs(have - target) > 1 and d != "resources":
            problems.append({"qid": "-", "problem": f"domain {d}: {have} vs target {target}"})
    for inst, n in composition["institutions"].items():
        if n > INSTITUTION_CAP:
            problems.append({"qid": "-", "problem": f"institution {inst} has {n} > {INSTITUTION_CAP}"})
    return problems, composition


# ------------------------------------------------------------------ grouping
def assign_groups(questions: list[dict]) -> tuple[dict[str, str], dict]:
    parent = {q["qid"]: q["qid"] for q in questions}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    prov = {q["qid"]: (q.get("provenance") or {}) for q in questions}
    by_signature = defaultdict(list)
    by_evidence = defaultdict(list)
    for q in questions:
        p = prov[q["qid"]]
        by_signature[(p.get("concrete_template"), p.get("inference_shape"))].append(q["qid"])
        by_evidence[tuple(sorted(question_clause_ids(q)))].append(q["qid"])
    for qids in by_signature.values():
        for other in qids[1:]:
            union(qids[0], other)
    for qids in by_evidence.values():
        for other in qids[1:]:
            union(qids[0], other)
    for i, qa in enumerate(questions):
        for qb in questions[i + 1:]:
            if jaccard(tokens(qa.get("question", "")), tokens(qb.get("question", ""))) >= JACCARD:
                union(qa["qid"], qb["qid"])

    clusters = defaultdict(list)
    for q in questions:
        clusters[find(q["qid"])].append(q["qid"])
    ordered = sorted(clusters.values(), key=lambda qs: min(qs))
    mapping = {}
    for i, qs in enumerate(ordered, start=1):
        for qid in qs:
            mapping[qid] = f"T2G{i:02d}"

    # review flags: never auto-merged
    merge_candidates = []
    similar_candidates = []
    for i, qa in enumerate(questions):
        for qb in questions[i + 1:]:
            pa, pb = prov[qa["qid"]], prov[qb["qid"]]
            same_shape = pa.get("inference_shape") == pb.get("inference_shape")
            if not same_shape:
                continue
            ja = jaccard(set(question_clause_ids(qa)), set(question_clause_ids(qb)))
            if ja >= TEMPLATE_JACCARD and mapping[qa["qid"]] != mapping[qb["qid"]]:
                merge_candidates.append({"a": qa["qid"], "b": qb["qid"],
                                         "clause_jaccard": round(ja, 3),
                                         "reason": "same clause set overlap + same inference shape"})
            elif (pa.get("template_family") == pb.get("template_family")
                  and pa.get("concrete_template") != pb.get("concrete_template")
                  and mapping[qa["qid"]] != mapping[qb["qid"]]):
                similar_candidates.append({"a": qa["qid"], "b": qb["qid"],
                                           "reason": "same template family and inference shape"})
    report = {"groups": len(ordered),
              "group_members": {k: v for k, v in sorted(mapping.items())},
              "merge_candidates": merge_candidates,
              "similar_candidates": similar_candidates}
    return mapping, report


def apply_groups(questions_path: Path, mapping: dict[str, str]) -> None:
    rows = read_jsonl(questions_path)
    for q in rows:
        if q.get("qid") in mapping:
            q["group_id"] = mapping[q["qid"]]
    tmp = questions_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows),
                   encoding="utf-8")
    tmp.replace(questions_path)


# ------------------------------------------------------------------ review packet
REVIEW_COLUMNS = ["qid", "bundle_id", "evidence_type", "domain", "condition",
                  "template_family", "concrete_template", "inference_shape",
                  "length_chars", "question", "reference_answer", "evidence_quotes",
                  "relation_basis", "relation_quote",
                  "decision", "question_ok", "answer_entails_from_evidence",
                  "evidence_set_minimal", "evidence_type_correct", "version_applicable",
                  "near_duplicate_checked", "relation_holds", "relation_quote_valid",
                  "no_keyword_stitching", "note"]


def write_review_packet(root: Path, questions, corpus, bundles) -> None:
    out_csv = root / "annotations" / "TEST2_REVIEW.csv"
    out_md = root / "annotations" / "TEST2_REVIEW.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    md = ["# test2 review packet", "",
          f"Generated {now()} - {len(questions)} questions", "",
          "For each question: decision = approve / revise / replace_clause, and mark",
          "question_ok, answer_entails_from_evidence, evidence_set_minimal,",
          "evidence_type_correct, version_applicable, near_duplicate_checked.",
          "Multi-clause questions additionally need relation_holds, relation_quote_valid",
          "and no_keyword_stitching.", ""]
    for q in questions:
        bid = (q.get("provenance") or {}).get("bundle_id")
        bundle = bundles.get(bid, {})
        cids = question_clause_ids(q)
        multi = len(set(cids)) > 1
        rel = bundle.get("relation_evidence") or {}
        length = question_lengths(q, corpus)
        rows.append({
            "qid": q.get("qid"), "bundle_id": bid,
            "evidence_type": q.get("evidence_type"), "domain": q.get("domain"),
            "condition": q.get("condition"),
            "template_family": (q.get("provenance") or {}).get("template_family"),
            "concrete_template": (q.get("provenance") or {}).get("concrete_template"),
            "inference_shape": (q.get("provenance") or {}).get("inference_shape"),
            "length_chars": length,
            "question": q.get("question"), "reference_answer": q.get("reference_answer"),
            "evidence_quotes": " | ".join(cids),
            "relation_basis": rel.get("basis", "") if multi else "",
            "relation_quote": rel.get("quote", "") if multi else "",
            "decision": "", "question_ok": "", "answer_entails_from_evidence": "",
            "evidence_set_minimal": "", "evidence_type_correct": "",
            "version_applicable": "", "near_duplicate_checked": "",
            "relation_holds": "" if multi else "n/a",
            "relation_quote_valid": "" if multi else "n/a",
            "no_keyword_stitching": "" if multi else "n/a", "note": "",
        })
        md += [f"## {q.get('qid')} - {q.get('evidence_type')} - {q.get('domain')}"
               f" - {length} chars", "",
               f"**Q.** {q.get('question')}", "",
               f"**A.** {q.get('reference_answer')}", "",
               f"**Bundle** {bid} - group {q.get('group_id')}" + (" - **multi-clause**"
                                                                   if multi else ""), ""]
        for cid in cids:
            clause = corpus.clauses.get(cid)
            if clause:
                md += [f"- {cid} ({clause.char_len} chars, {clause.doc_id})",
                       f"  > {clause.text[:400]}", ""]
        if multi:
            md += [f"Relation evidence: **{rel.get('basis')}** — `{rel.get('quote')}` "
                   f"(in {rel.get('where')})", ""]
        md += ["Review: [] approve [] revise [] replace_clause", ""]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {out_csv}")
    print(f"  wrote {out_md}")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--questions", default="annotations/questions_test2.jsonl")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--assign-groups", action="store_true")
    ap.add_argument("--review-packet", action="store_true")
    ap.add_argument("--verify-hashes", action="store_true")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--accept-composition", action="store_true",
                    help="record domain/institution deviations as warnings, requiring "
                         "results/test2/DESIGN_AMENDMENTS.json")
    args = ap.parse_args()

    root = Path(args.root)
    qpath = Path(args.questions)
    if not qpath.is_absolute():
        qpath = root / qpath
    paths, frozen, bundles, amendments, replaced, dropped, questions, corpus, existing = \
        load_all(root, qpath)
    settings = load_settings(root)
    config = {"annotation": settings["annotation"],
              "domains": settings["annotation"].get("domains")}

    if not questions:
        print(f"REFUSING: no questions in {qpath}", file=sys.stderr)
        return 2

    if args.assign_groups:
        mapping, report = assign_groups(questions)
        apply_groups(qpath, mapping)
        write_json(root / "results" / "test2" / "grouping.json", report)
        print(f"=== grouping === {report['groups']} groups for {len(questions)} questions")
        print(f"  merge candidates (review): {len(report['merge_candidates'])}")
        for c in report["merge_candidates"][:10]:
            print(f"    {c['a']} ~ {c['b']}  clause-jaccard {c['clause_jaccard']}")
        print(f"  similar candidates (review): {len(report['similar_candidates'])}")
        if report["groups"] < MIN_GROUPS:
            print(f"  WARNING: {report['groups']} < {MIN_GROUPS} groups; use reserve bundles")
        return 0

    problems, composition = check_questions(questions, corpus, bundles, replaced, dropped,
                                            existing, config)
    # Composition deviations (domain mix, institution share) are recorded pre-evaluation in
    # results/test2/DESIGN_AMENDMENTS.json. When that file exists and --accept-composition is
    # passed they are reported as warnings, never silently dropped; every question-level check
    # remains a hard blocker.
    warnings: list[dict] = []
    if args.accept_composition:
        amend_path = root / "results" / "test2" / "DESIGN_AMENDMENTS.json"
        if not amend_path.exists():
            print(f"REFUSING: --accept-composition needs {amend_path}", file=sys.stderr)
            return 2
        warnings = [p for p in problems if p["qid"] == "-"]
        problems = [p for p in problems if p["qid"] != "-"]
        print(f"  composition amendments: {amend_path.name} "
              f"({len(json.loads(amend_path.read_text(encoding='utf-8'))['amendments'])} items)")
    print(f"=== test2 check ({len(questions)} questions) ===")
    print(f"  composition: {json.dumps(composition, ensure_ascii=False)}")
    print(f"  problems   : {len(problems)}  warnings: {len(warnings)}")
    for p in problems[:25]:
        print(f"    {p['qid']}: {p['problem']}")
    for w in warnings[:10]:
        print(f"    warning: {w['problem']}")

    if args.review_packet:
        write_review_packet(root, questions, corpus, bundles)
        return 0 if not problems else 1

    if args.verify_hashes:
        lock_path = root / "annotations" / "LOCKED_TEST2" / "LOCKED2.json"
        pre_path = root / "results" / "test2" / "PREREGISTRATION.json"
        ok = True
        checks = []
        if lock_path.exists():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            checks.append(("questions vs lock",
                           sha256_file(qpath) == lock.get("questions_jsonl_sha256")))
            checks.append(("lock" if False else "bundle file vs lock",
                           sha256_file(root / "results/test2/selected_bundles.json")
                           == (lock.get("bundles_sha256") or "")))
        else:
            checks.append(("lock exists", False))
        if pre_path.exists():
            pre = json.loads(pre_path.read_text(encoding="utf-8"))
            checks.append(("questions vs prereg",
                           sha256_file(qpath) == pre.get("questions_sha256")))
        for name, passed in checks:
            print(f"  {name:<28} {'PASS' if passed else 'FAIL'}")
            ok = ok and passed
        return 0 if ok else 1

    if args.lock:
        if problems:
            print("REFUSING to lock: fix the problems above", file=sys.stderr)
            return 3
        mapping, _ = assign_groups(questions)
        if len({mapping[q["qid"]] for q in questions}) < MIN_GROUPS:
            print(f"REFUSING to lock: fewer than {MIN_GROUPS} independent groups", file=sys.stderr)
            return 3
        unreviewed = [q["qid"] for q in questions if not q.get("manual_reviewed")]
        if unreviewed:
            print(f"REFUSING to lock: {len(unreviewed)} questions not signed off "
                  f"({', '.join(unreviewed[:8])})", file=sys.stderr)
            return 3
        audit = load_module("scripts/05_audit_splits.py", "audit05", root)
        res = audit.lock_test(questions, paths, split="test2", questions_path=qpath)
        print(f"=== locked === {json.dumps({k: v for k, v in res.items() if k != 'locked_at'})}")
        return 0

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
