"""24_preregister_test2.py - freeze the design of the second held-out set before it is used.

Why this exists
---------------
The test split has already had its one formal use, so a new held-out set is the only way to
produce a confirmatory number. That only works if the design is fixed *before* the questions
are drafted and evaluated, and if the evaluation cannot silently drift from it. This script
writes that record and nothing else:

    results/test2/PREREGISTRATION.json

It records the endpoint definitions and thresholds, the sample size and its power, the
sampling rules (length first, then domain, then institution), the clause pool it will draw
from, and a hash of every artefact the evaluation depends on. `19_rerank_eval.py
--preregistration` refuses to run when any of those hashes has moved.

Two phases
----------
    design only   run before the questions exist; the questions hash is recorded as null
    finalised     re-run with --questions/--lock; the question file and its lock are hashed

Both phases are additive: the design fields never change, only the question hash appears.

The primary endpoint is the structural-vs-semantic gap, not C3 − C1. On the test split the
reranking gain is +0.047 and 40 groups can only resolve effects above ~0.08, while the
family gap is ~0.68 — an order of magnitude larger. Pre-registering the small contrast as
primary would guarantee an underpowered null.

Usage
    .venv\\Scripts\\python.exe scripts/24_preregister_test2.py
    .venv\\Scripts\\python.exe scripts/24_preregister_test2.py \\
        --questions annotations/questions_test2.jsonl --lock annotations/LOCKED_TEST2/LOCKED2.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import PROJECT_ROOT, project_paths, read_jsonl, write_json  # noqa: E402
import goldset  # noqa: E402

EVALUATION_CODE = ("scripts/13_evaluate.py", "scripts/18_train_reranker.py",
                   "scripts/19_rerank_eval.py", "scripts/16_stats.py",
                   "scripts/goldset.py", "scripts/common.py",
                   "scripts/11_chunking.py", "scripts/12_build_index.py")
STRUCTURAL = ("heading", "paragraph")
SEMANTIC = ("thr0.70", "thr0.75", "thr0.80")
MIN_CLAUSE_CHARS = 150


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def power(effect: float, sd: float, n: int) -> float:
    return phi(abs(effect) / (sd / math.sqrt(n)) - 1.959964)


def mde(sd: float, n: int) -> float:
    return (1.959964 + 0.841621) * sd / math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--questions", default="", help="test2 questions file, once it exists")
    ap.add_argument("--lock", default="", help="LOCKED2.json, once the split is locked")
    ap.add_argument("--questions-target", type=int, default=45)
    ap.add_argument("--groups-target", type=int, default=40)
    ap.add_argument("--out", default="results/test2/PREREGISTRATION.json")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    corpus = goldset.load_corpus(root)
    questions = read_jsonl(paths["annotations"] / "questions.jsonl")

    used_ids = {ref.get("clause_id")
                for q in questions
                for eset in (q.get("evidence_sets") or [])
                for ref in (eset.get("clauses") or [])}
    clause_len = {c.clause_id: c.char_end - c.char_start
                  for c in corpus.clauses.values()}
    pool = [c for c in corpus.clauses.values()
            if c.clause_id not in used_ids and not c.boilerplate
            and len(c.text) >= MIN_CLAUSE_CHARS]
    pool.sort(key=lambda c: c.clause_id)

    # --- the difficulty target: how long the clauses behind the existing test questions are
    def question_length(q: dict) -> int:
        lens = [clause_len.get(ref.get("clause_id"), 0)
                for eset in (q.get("evidence_sets") or [])
                for ref in (eset.get("clauses") or [])]
        return max(lens) if lens else 0

    test_len = sorted(question_length(q) for q in questions if q.get("split") == "test")
    quartiles = [float(np.percentile(test_len, p)) for p in (25, 50, 75)]

    # availability in the pool, by the same quartile bins
    bins = [0.0] + quartiles + [float("inf")]
    availability = Counter()
    for c in pool:
        length = clause_len[c.clause_id]
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                availability[f"bin{i+1}"] += 1
                break

    # --- power, measured on the existing splits rather than assumed
    def group_means(perq_path: Path, structural: bool) -> dict[str, float]:
        rows = read_jsonl(perq_path)
        acc: dict[str, dict[str, list[float]]] = {}
        for r in rows:
            # structural families are heading/paragraph; every semantic threshold shares
            # strategy == "semantic" and differs only in config
            wanted = (r["strategy"] in STRUCTURAL) if structural else (r["strategy"] == "semantic")
            if not wanted:
                continue
            acc.setdefault(r.get("group_id") or r["qid"], {}).setdefault(
                r["config"], []).append(float(r["complete@10"]))
        return {g: float(np.mean([np.mean(v) for v in cfg_map.values()]))
                for g, cfg_map in acc.items()}

    # standard deviation of the per-group structural-minus-semantic delta, aligned by group
    try:
        struct = group_means(root / "results/final/perq_C1.jsonl", True)
        seman = group_means(root / "results/final/perq_C1.jsonl", False)
        shared = sorted(set(struct) & set(seman))
        sd_gap = float(np.std([struct[g] - seman[g] for g in shared], ddof=1)) \
            if len(shared) > 1 else None
    except Exception:  # noqa: BLE001 - the record is still useful without it
        sd_gap = None

    # reranking-gain dispersion, for the secondary endpoint
    sd_gain = None
    try:
        c1 = read_jsonl(root / "results/final/perq_C1.jsonl")
        c3 = read_jsonl(root / "results/final/perq_C3.jsonl")
        k1 = {(r["qid"], r["config"]): float(r["complete@10"]) for r in c1}
        by_group: dict[str, list[float]] = {}
        for r in c3:
            key = (r["qid"], r["config"])
            if key in k1:
                by_group.setdefault(r.get("group_id") or r["qid"], []).append(
                    float(r["complete@10"]) - k1[key])
        deltas = [float(np.mean(v)) for v in by_group.values()]
        sd_gain = float(np.std(deltas, ddof=1))
    except Exception:  # noqa: BLE001
        sd_gain = None

    artefacts = {}
    for rel in ("corpus_safe_512/corpus_safe_512.jsonl",):
        p = root / rel
        if p.exists():
            artefacts[rel] = sha256_file(p)
    for pattern in ("chunks/*/*.jsonl", "indexes/*/*.faiss", "indexes/*/*.meta.json"):
        for p in sorted(root.glob(pattern)):
            artefacts[p.relative_to(root).as_posix()] = sha256_file(p)
    for rel in EVALUATION_CODE:
        p = root / rel
        if p.exists():
            artefacts[rel] = sha256_file(p)
    frozen_path = root / "models" / "reranker" / "FROZEN.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        artefacts["models/reranker/FROZEN.json"] = sha256_file(frozen_path)
        run_sha = frozen["model"]["checkpoint_sha256"]
    else:
        run_sha = None

    spec26 = importlib.util.spec_from_file_location("s26", root / "scripts"
                                                     / "26_select_test2_clauses.py")
    s26 = importlib.util.module_from_spec(spec26)
    spec26.loader.exec_module(s26)
    readable_pool = [c for c in pool if s26.is_readable(c)]

    questions_path = Path(args.questions) if args.questions else None
    if questions_path and not questions_path.is_absolute():
        questions_path = root / questions_path
    lock_path = Path(args.lock) if args.lock else None
    if lock_path and not lock_path.is_absolute():
        lock_path = root / lock_path

    # test2-specific artefacts: the bundle list, its amendments, the audit trail and the
    # questions file all have to be hashed, or the "evaluate once against frozen design"
    # claim cannot be checked afterwards.
    for rel in ("results/test2/selected_bundles.json",
                "results/test2/bundle_amendments.jsonl",
                "results/test2/DESIGN_AMENDMENTS.json",
                "results/test2/grouping.json",
                "annotations/TEST2_REVIEW.csv",
                "annotations/TEST2_SELF_AUDIT.md"):
        p = root / rel
        if p.exists():
            artefacts[rel] = sha256_file(p)
    questions_rows = read_jsonl(questions_path) if questions_path and questions_path.exists() else []
    n_groups_actual = len({q.get("group_id") for q in questions_rows}) or args.groups_target
    if questions_rows:
        artefacts[str(questions_path.relative_to(root)).replace("\\", "/")] = \
            sha256_file(questions_path)

    record = {
        "kind": "test2_preregistration",
        "created_at": now(),
        "status": "finalised" if questions_path and questions_path.exists() else "design_only",
        "env_role": "canonical_model",
        "evaluation_env": ("one run on the A5000 (fp16); the primary endpoint is a contrast "
                           "between two configuration families computed inside that same run, "
                           "so it is not compared with the RTX 3060 numbers"),
        "questions_sha256": (sha256_file(questions_path)
                             if questions_path and questions_path.exists() else None),
        "lock_sha256": (sha256_file(lock_path)
                        if lock_path and lock_path.exists() else None),
        "artifacts": artefacts,
        "reranker_checkpoint_sha256": run_sha,
        "design": {
            "split_discipline": ("whole evidence groups go to one split; test2 clauses must be "
                                 "unused by any existing question, so no gold evidence is "
                                 "shared with dev or test"),
            "questions_target": args.questions_target,
            "groups_target": args.groups_target,
            "sampling_priority": ["clause length", "domain", "institution", "document novelty"],
            "clause_pool": {
                "definition": (f"unused by any question, >= {MIN_CLAUSE_CHARS} characters, "
                               "not boilerplate, and readable as prose (no OCR/citation "
                               "fragments, no URL lines, no table rows, no bullet continuations)"),
                "clauses": len(pool),
                "readable_clauses": len(readable_pool),
                "documents": len({c.doc_id for c in pool}),
                "institutions": len({c.institution for c in pool}),
                "note": ("the length/boilerplate definition alone gives 1,239 clauses; the "
                         "readability screen is what makes them question-usable"),
            },
            "recorded_amendments": {
                "file": "results/test2/DESIGN_AMENDMENTS.json",
                "items": ["A1 domain composition", "A2 institution cap",
                          "A3 evidence types"],
                "note": ("recorded before any evaluation; they describe what the clean pool "
                         "could support, and are disclosed in the report"),
            },
            "difficulty_target": {
                "statistic": "max annotated clause span per test question, characters",
                "test_quartiles": [round(v, 1) for v in quartiles],
                "tolerance": ("each quartile median within +/-10% and the overall median "
                              "within +/-100 characters of the target"),
                "availability_by_quartile_bin": dict(availability),
                "note": ("matching is a composition safeguard, not a correction for an "
                         "established effect: the short-vs-long gradient is not significant "
                         "in any split x track combination"),
            },
            "domain_labelling": ("manual; the keyword classifier returns an empty label for "
                                 "about 42% of the pool, so the domain of each chosen clause "
                                 "must be assigned by hand and recorded"),
        },
        "endpoints": {
            "primary": {
                "name": "structural-minus-semantic Complete@10 (C1)",
                "definition": ("mean over the 2 structural configurations (heading/heading, "
                               "paragraph/paragraph) minus mean over the 3 semantic "
                               "configurations (semantic/thr0.70, thr0.75, thr0.80), "
                               "group-level paired bootstrap"),
                "judged_on": "C1 (dense top-50), replicated for direction on C3",
                "confirm_if": "95% CI lower bound > 0.30 (strict: equal does not count)",
                "sign_note": ("16_stats reports semantic-minus-structural (negative); 28 "
                              "negates it and asserts the two agree"),
            },
            "secondary": {
                "name": "C3 - C1 Complete@10",
                "status": "underpowered, descriptive only",
                "support_semantics": {
                    "positive_supported": "CI lower bound > 0 (this alone is the positive claim)",
                    "exceeds_delta": "CI lower bound > delta, e.g. 0.06 (an additional claim, never the replication criterion)",
                    "smaller_than_0.047": "CI upper bound < 0.047",
                    "indeterminate": ("CI includes 0: report that effects below the MDE cannot "
                                      "be excluded; never report 'no effect'"),
                    "boundary": "strict inequalities; equality never supports a claim",
                },
            },
            "robustness": ["partial@10 (union character coverage)",
                           "tok512 recall (equal token budget)"],
            "statistics": {"unit": "group_id", "bootstrap": 2000, "seed": 20261013,
                           "ci_method": "percentile bootstrap",
                           "collapse": "macro over the family's configurations",
                           "multiplicity": ("fixed-sequence gatekeeping: the secondary is "
                                            "interpreted only if the primary confirms; no "
                                            "post-hoc correction"),
                           "holm": "only within the secondary family, never for the endpoint"},
        },
        "power": {
            "n_questions": len(questions_rows) or None,
            "n_groups": n_groups_actual,
            "structural_vs_semantic": {
                "sd_per_group": round(sd_gap, 4) if sd_gap else None,
                "mde_at_n": (round(mde(sd_gap, n_groups_actual), 3)
                             if sd_gap and n_groups_actual else None),
                "observed_test_delta": 0.682,
                "note": ("the gap is an order of magnitude above the MDE, so the primary "
                         "endpoint is well powered at this n"),
            },
            "reranking_gain": {
                "sd_per_group": round(sd_gain, 4) if sd_gain else None,
                "mde_at_n": round(mde(sd_gain, n_groups_actual), 4)
                if sd_gain and n_groups_actual else None,
                "power_at_test_effect_0.047": (round(power(0.047, sd_gain, n_groups_actual), 3)
                                               if sd_gain and n_groups_actual else None),
                "power_at_dev_effect_0.107": (round(power(0.107, sd_gain, n_groups_actual), 3)
                                              if sd_gain and n_groups_actual else None),
            },
            "statement": ("test2 can resolve a development-sized effect; it cannot reliably "
                          "exclude a +0.047 effect. A CI that includes zero is not evidence of "
                          "no effect and not a failed replication."),
        },
        "decision_matrix": [
            {"primary": "CI lower > 0.30", "secondary": "CI lower > 0.06",
             "claim": "chunking confirmed at >=0.30 and reranking shows an effect above 0.06"},
            {"primary": "CI lower > 0.30", "secondary": "0 < CI lower <= 0.06",
             "claim": "chunking confirmed; reranking positive, magnitude unresolved"},
            {"primary": "CI lower > 0.30", "secondary": "CI includes 0",
             "claim": ("chunking confirmed; reranking not confirmed, effects below the MDE "
                       "not excluded")},
            {"primary": "0 < CI lower <= 0.30", "secondary": "any",
             "claim": ("direction positive but the 0.30 threshold is not confirmed; report the "
                       "interval, make no 'significantly better' claim")},
            {"primary": "CI upper < 0.30", "secondary": "any",
             "claim": "the gap is confirmed to be below 0.30; the ordering conclusion is weaker"},
            {"primary": "CI includes 0", "secondary": "any",
             "claim": ("no directional difference confirmed at this sample size; compatible "
                       "with zero and with small positive effects; no re-run with different "
                       "questions")},
        ],
        "one_shot": ("19_rerank_eval.py writes FINAL_RUN.json after a completed run and then "
                     "refuses to run again; there is no --force"),
        "existing_artifacts_unchanged": {
            "results/final/FINAL_RUN.json": (sha256_file(root / "results/final/FINAL_RUN.json")
                                             if (root / "results/final/FINAL_RUN.json").exists()
                                             else None),
            "models/reranker/FROZEN.json": (sha256_file(frozen_path)
                                            if frozen_path.exists() else None),
        },
    }

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, record)

    print("=== test2 pre-registration ===")
    print(f"  status        : {record['status']}")
    print(f"  clause pool   : {len(pool)} clauses / {len({c.doc_id for c in pool})} documents")
    print(f"  length target : test quartiles {[round(v,1) for v in quartiles]}")
    print(f"  availability  : {dict(availability)}")
    if sd_gain:
        print(f"  secondary MDE : {mde(sd_gain, args.groups_target):.3f} at "
              f"{args.groups_target} groups (sd {sd_gain:.3f})")
    print(f"  hashed files  : {len(artefacts)}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
