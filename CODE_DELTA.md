# Code delta: original experiment code vs public release

`19_rerank_eval.py` hashes `scripts/*.py` as a set, so any edit changes `FINAL_RUN.code_sha256`.
This file records exactly what changed so "no experiment logic changed" is checkable rather than
asserted.

## Files changed (3 of 47; none added or deleted under `scripts/`)

| file | change | reason |
|---|---|---|
| `scripts/make_progress_report.py` | replaced the author's absolute Python path and desktop output path with `python` / `reports/…` | remove machine-specific paths |
| `scripts/make_status_report.py` | same desktop output path | remove machine-specific path |
| `scripts/make_teacher_brief.py` | replaced the absolute Python path in the usage docstring with `python` | remove machine-specific path |

Per-file hashes: `CODE_MANIFEST.csv` (`original_sha256`, `public_sha256`, `changed`, `role`).
The original values were taken before the edits (`data/CODE_MANIFEST.original.csv`).

## Result-generating code is byte-identical

`RESULTS_CODE_SHA256.json` records original and public hashes for the eight files in
`EVALUATION_CODE`; all eight are identical:

`scripts/11_chunking.py`, `scripts/12_build_index.py`, `scripts/13_evaluate.py`,
`scripts/16_stats.py`, `scripts/18_train_reranker.py`, `scripts/19_rerank_eval.py`,
`scripts/common.py`, `scripts/goldset.py`.

## Overall hashes (expected to differ)

| context | `code_sha256` |
|---|---|
| training run (FROZEN, 2026-10-25) | `cefb847e50c273a8…` |
| test run (2026-10-28) | `6f0bcff6a4d8ce57…` |
| test2 run (2026-11-13) | `da6a039b984a310a…` |
| public tree (this release) | `0a9e80ebf4d5f9d1…` |

The three official values already differ from one another: the evaluation code advanced between
the runs. The public value differs because of the path-only edits and the crawler-contact change above. The historical
`FINAL_RUN.json` / `FROZEN.json` are **not** modified to match; this document is the mapping
between the original run and the public release.

## Not changed

No experiment logic, metric, threshold, seed, evaluation protocol, lock, pre-registration or
result file was edited. `FROZEN.json` still contains the author's original paths; it is
hash-locked by `FINAL_RUN.frozen_sha256`, so it is left as-is and explained in `PROVENANCE.md`.
