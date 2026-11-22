# Paper draft notes (2026-11-22)

## State

* Source: `paper/main.tex`, bibliography `paper/refs.bib`, IEEEtran conference, English.
* Compiled: `paper/main.pdf` and `docs/PAPER.pdf`; **7 pages main content + 1 page references**
  (target 6–8 main).
* 17 references (15 scholarly + 2 model cards); 7 tables (corpus, 17 configurations, training, four-track test, per-family,
  test2, token budget) and 1 figure (per-family C1→C3).
* All numbers were transcribed from `results/`; the next step is the automated
  `NUMBERS.json` binding and checker described in `docs/PAPER_PLAN.md`.

## Must-do before submission

1. **References verified (2026-11-22).** All 17 entries in docs/REFERENCES_VERIFIED.md are marked verified; no pending rows remain.

## Professor-feedback revision (2026-11-22)

* Methodology: added configuration-selection rationale (families, not tuning; semantic thresholds as a
  sensitivity range) and a limitation that the distributions are not equalized.
* Experimental Setup: added comparability controls (same corpus/evidence/encoder/index procedure/
  pools/reranker/metrics; matched token budgets; no claim of identical chunk distributions).
* Results: added a "Main findings" paragraph; strengthened the non-significant first-test reranking
  statement and the test2 single-clause/integrity/run-history caveat.
* Analysis: replaced the unsupported qualitative claim with a "why semantic is weaker" discussion
  separating observations from possible mechanisms.
* Added a compact two-case table (Q0001, Q0088) from `results/analysis/case_studies.json`, produced by
  `tools/case_studies.py` (diagnostic replay of frozen indexes; no official result changed). Note: the
  real Q0001 case retrieved c6 and missed c8, not the rule/exception direction assumed in the draft.
* Compressed T06/T07 numbers out of the main text; they remain in `models/reranker/COMPAT.json`.
* Added Nogueira et al. 2020 (Findings of EMNLP) as reference 18; still pending student verification.
* Current PDF: 8 pages = 7 main + 1 references.
