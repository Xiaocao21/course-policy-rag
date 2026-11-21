"""Makefile-style task runner. Windows-friendly; no external dependencies.

Usage:
    .venv\\Scripts\\python.exe run.py <task> [-- extra args]

Collection and preprocessing
    probe        probe candidate seeds (robots, reachability, policy density)
    discover     follow policy links from verified landing pages
    ocw          harvest MIT OpenCourseWare syllabus candidates
    registry     rebuild config/source_registry.csv from probe/discovery output
    collect      crawl the registry into raw/ + sources.csv
    extract      structure + dual text layers + dedup
    curate       select the documented corpus and write include=TRUE back
    corpus       build the shared 512-token corpus
    validate     run corpus invariant checks
    freeze       write MANIFEST.json + COMPLIANCE.md (needs --corpus-version)

Annotation
    feasibility  can the corpus support the planned question set?
    candidates   propose clause / exception-pair / alt-evidence candidates
    template     regenerate the annotation scaffolding (schema, worksheet)
    annotate     start the local annotation workbench (127.0.0.1 only)
    audit        split, group and leakage checks
    lock         lock the held-out test split (audit must pass first)
    unlock       remove the lock (recorded in reports/lock_audit.jsonl)
    agreement    question-level agreement worksheet for the second annotator
    kappa        Cohen's kappa over the double-annotated questions

Experiment
    prefill      (re)generate annotation worklists from the corpus
    gold         AI-draft annotation ingest / status / restamp
    balance      rebalance evidence types to the protocol quota
    review       build the human review packet (REVIEW.md / REVIEW.csv)
    reviewed     record that you reviewed the questions (manual_reviewed=true)
    chunk        run the four chunking strategies
    index        embed every configuration and build FAISS indexes
    evaluate     retrieval metrics on the development split
    analyze      tune, tabulate, plot and write results/REPORT.md
    env          capture the CPU and CUDA training environments into experiments/env.json
    rerank-data  build development reranker pairs (labels, dedup, group split)
    rerank-train train the single shared reranker and freeze it (FROZEN.json)
    final-eval   four-track evaluation (dev by default; test needs --acknowledge-test-use)
    stats        group-level bootstrap, paired contrasts, per-document macro
    brief        build docs/TEACHER_BRIEF.docx from the final numbers
    project-report  build docs/PROJECT_REPORT.docx in the recommended 10-section order

Utilities
    test         run the offline unit tests
    report       print the current state of every report
    all          extract + curate + corpus + validate + freeze

Typical order after a fresh collection:
    registry -> collect -> extract -> curate -> corpus -> validate -> freeze
    -> feasibility -> template -> candidates -> annotate
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

TASKS = {
    "probe": ["scripts/probe_seeds.py", "--out", "reports/seed_probe_round2.csv",
              "--urls-file", "config/seed_candidates_round2.csv"],
    "discover": ["scripts/discover_policy_links.py", "--landings", "config/landings.csv",
                 "--out", "config/discovered_candidates.csv"],
    "ocw": ["scripts/harvest_ocw.py", "--out", "config/registry_ocw_candidates.csv",
            "--max-courses", "600"],
    "registry": ["scripts/build_registry.py", "--include-course-selected",
                 "--include-manual", "config/seed_institution_round3.csv",
                 "--include-manual", "config/seed_round4_gapfill.csv"],
    "collect": ["scripts/01_collect.py"],
    "extract": ["scripts/02_extract_clean.py"],
    "curate": ["scripts/curate_corpus.py", "--total", "36", "--min-clauses", "3"],
    "corpus": ["scripts/03_build_safe_corpus.py"],
    "validate": ["scripts/06_validate.py"],
    "freeze": ["scripts/04_freeze_manifest.py"],

    "feasibility": ["scripts/07_annotation_feasibility.py"],
    "candidates": ["scripts/08_prefill_candidates.py", "--per-cell", "9"],
    "gold": ["scripts/10_ai_annotate.py"],
    "template": ["scripts/make_annotation_template.py"],
    "annotate": ["scripts/09_annotate_server.py"],
    "audit": ["scripts/05_audit_splits.py"],
    "lock": ["scripts/05_audit_splits.py", "--lock-test"],
    "unlock": ["scripts/05_audit_splits.py", "--unlock",
               "--unlock-reason", "manual", "--questions", "annotations/questions.jsonl"],
    "agreement": ["scripts/05_audit_splits.py", "--make-agreement-sample", "20"],
    "kappa": ["scripts/kappa_agreement.py"],

    "prefill": ["scripts/10_ai_annotate.py"],
    "balance": ["scripts/10b_balance_questions.py", "--drop-type", "single-clause",
                "--target-count", "55", "--min-per-domain", "5"],
    "review": ["scripts/10c_make_review.py"],
    "reviewed": ["scripts/05_audit_splits.py", "--mark-reviewed"],
    "chunk": ["scripts/11_chunking.py"],
    "index": ["scripts/12_build_index.py"],
    "evaluate": ["scripts/13_evaluate.py", "--split", "dev"],
    "analyze": ["scripts/14_analyze.py"],

    "env": ["scripts/15_experiment_log.py", "--env"],
    "rerank-data": ["scripts/17_build_rerank_data.py"],
    "rerank-train": ["scripts/18_train_reranker.py"],

    "final-eval": ["scripts/19_rerank_eval.py", "--split", "dev"],
    "stats": ["scripts/16_stats.py", "--split", "dev"],
    "brief": ["scripts/make_teacher_brief.py"],
    "project-report": ["scripts/make_project_report.py"],

    "test": ["-m", "unittest", "discover", "-s", "tests", "-v"],
}

REPORT_FILES = [
    "sources.csv", "MANIFEST.json", "COMPLIANCE.md", "README.md",
    "config/source_registry.csv",
    "reports/seed_probe_round2.csv", "reports/seed_probe_round3.csv",
    "reports/fetch_log.csv", "reports/pending_review.csv",
    "reports/quality_report.csv", "reports/clause_stats.csv",
    "reports/corpus_membership.csv", "reports/curation_decisions.csv",
    "reports/validation.json", "reports/robots_audit.json",
    "reports/manual_only_sources.json", "reports/duplicates.csv",
    "reports/annotation_feasibility.json", "reports/annotation_feasibility.png",
    "reports/token_distribution_approx.png",
    "candidates/clause_candidates.csv", "candidates/exception_pairs.csv",
    "candidates/alt_evidence_groups.csv", "candidates/candidates_summary.json",
    "annotations/SCHEMA.md", "annotations/questions.jsonl",
    "annotations/question_skeletons.jsonl", "annotations/gold_worksheet.csv",
    "annotations/agreement_sample.csv", "annotations/agreement_worksheet.csv",
    "annotations/agreement_report.json", "annotations/split_audit.json",
    "annotations/LOCKED/LOCKED.json",
]


def run(args: list[str]) -> int:
    cmd = [str(PY)] + args
    print("> " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def report() -> int:
    print("=== report state ===")
    missing = 0
    for rel in REPORT_FILES:
        p = ROOT / rel
        if p.exists():
            n = ""
            if p.suffix in (".jsonl", ".csv"):
                try:
                    n = f"  ({sum(1 for _ in p.open(encoding='utf-8'))} lines)"
                except OSError:
                    n = ""
            print(f"  [x] {rel}{n}  {p.stat().st_size:,} bytes")
        else:
            print(f"  [ ] {rel}")
            missing += 1
    print(f"\n  {len(REPORT_FILES) - missing}/{len(REPORT_FILES)} present")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    task = sys.argv[1]
    extra = sys.argv[2:]
    if extra and extra[0] == "--":
        extra = extra[1:]

    if task == "report":
        return report()
    if task == "all":
        # audit and freeze are expected to fail before annotation exists; stop only on
        # the preprocessing stages, whose failure means the corpus is not trustworthy.
        for t in ("extract", "curate", "corpus", "validate"):
            rc = run(TASKS[t])
            if rc:
                return rc
        return 0
    if task not in TASKS:
        print(f"unknown task '{task}'\n")
        print(__doc__)
        return 2
    return run(TASKS[task] + extra)


if __name__ == "__main__":
    sys.exit(main())
