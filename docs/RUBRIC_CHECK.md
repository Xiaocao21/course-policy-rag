# Rubric self-check — `docs/PROJECT_REPORT.docx`

Checked on 2026-10-29 (revision 2), again on 2026-11-12 (revision 3, after supervisor
feedback and the A5000 compatibility experiments), and again on 2026-11-13 (revision 4, after
the second held-out set) against three sources: (A) the five requirements given in class, (B)
the recommended report structure, (C) the design and metric requirements in
`problem_statement.docx`. Revision 3 does report A5000 numbers, but only as compatibility
measurements with an explicit time/memory budget; every number that carries a result claim
still comes from the canonical RTX 3060 run, except the single test2 run, whose endpoints are
computed inside that run itself.

---

## A. The five required components

| # | Requirement | Where | Evidence | Status |
|---|---|---|---|---|
| A1 | Project status and completed work | §2 Work Completed | corpus 36 docs / 1,988 clauses / 2,007 units; dual-layer preprocessing; 17 configs; one fine-tuned reranker; 2×2 evaluation | PASS |
| A2 | Experimental log with model, dataset/version, hyperparameters, seed, training configuration, date/run ID, results | §4 Experiment Log (11 rows) + seed-derivation note | E01–E06 and T01–T05; each row has a date, settings including seed, metric, result, observation; the paragraph below explains why training seeds are 20261027 / 20261037 (derived from the base seed and hyperparameters) | PASS |
| A3 | Preliminary quantitative results with ≥1 table or figure and interpretation | §5 | track table, matched-budget table, Figure 1 (CI), Figure 2 (rank), four interpretation paragraphs | PASS |
| A4 | Problems, changes, decisions, and why | §6 (7 items) | offsets, over-claimed coverage, leakage, pilot exposure, broken selection metric, **why 0.396 was not selected**, hardware | PASS |
| A5 | Remaining work and feasibility within the semester | §7, §8 | three remaining experiments with cost; W1–W4 schedule; feasibility argued from a completed run on a 6 GB card | PASS |

---

## B. Recommended structure (10 parts, in order)

| # | Section | Present | Status |
|---|---|---|---|
| B1 | Project Title | title + subtitle (date, corpus version) | PASS |
| B2 | Research Question | §1, with **primary** (chunking only) and **secondary** (pool × reranking) experiments stated separately | PASS |
| B3 | Work Completed | §2 | PASS |
| B4 | Experimental Setup | §3 table: encoder, index, reranker, reranker input, training data, model selection, data, metrics, tracks, statistics, hardware | PASS |
| B5 | Experiment Log | §4, 7 columns as in the example | PASS |
| B6 | Preliminary Results | §5 | PASS |
| B7 | Problems and Changes | §6 | PASS |
| B8 | Remaining Experiments | §7 | PASS |
| B9 | Updated Timeline | §8: the proposal's weeks 1–14 (planned) beside what actually happened (dates taken from artefact timestamps), plus a small remaining-work table | PASS |
| B10 | AI Disclosure | §9 | PASS |

Failed runs are retained: T01–T03 are aborted with their settings and cause, T05 is a weaker
completed run kept in the log.

---

## C. Requirements from `problem_statement.docx`

| # | Requirement | Where | Status |
|---|---|---|---|
| C1 | Answer the stated research question | §1 + §5 | PASS |
| C2 | Only chunking varies in the controlled comparison | §1 separates the primary comparison (chunking only) from the secondary 2×2 check; §3 lists the fixed factors | PASS |
| C3 | Recall@k, MRR, Hit Rate, evidence completeness, retrieval latency | §3 defines all five; §5 reports all five, including a matched-budget table and an efficiency paragraph | PASS |
| C4 | Feasible on the hardware the assignment assumes | canonical results come from a 6 GB RTX 3060 with fp16 and partial fine-tuning; the A5000 compatibility runs (T06 full fine-tune, T07 matched-seed lr grid) and the single test2 run are reported separately with time and memory budgets in §7 | PASS (both budgets documented; a 24 GB card only added the optional experiments) |
| C5 | Unsuccessful experiments documented | §4 T01–T03; `experiments/aborted_runs.jsonl` | PASS |
| C6 | Single formal evaluation with disclosure | §5 opens with the exposure caveat; E02–E05 are labelled "re-evaluation"; `FINAL_RUN.json` records attempt 1 and `test_previously_used` | PASS |
| C7 | 3–5 pages | PDF export | PASS (5 pages) |

---

## D. Corrections applied in revision 2 (from the review)

| # | Review point | Change |
|---|---|---|
| 1 | Research question implied chunking was the only variable, then added pool and reranker | §1 now states a **primary** experiment (chunking only) and a separate **secondary** 2×2 check, plus a paragraph on how they relate |
| 2a | T05 seed 20261037 looked like a typo | verified: seeds are derived as base + 10·(lr×10⁵) + negatives; a note under the log explains the derivation, and `FROZEN.json` records it |
| 2b | T01 reached 0.396 but 0.388 was frozen | §6 adds "Why the higher number was not selected": T01 never completed and its epoch-1 checkpoint was discarded, so selection ran over completed runs only |
| 2c | Setup table lacked reranker input limits, truncation, split sizes, selection pool | §3 adds four rows: reranker input (query 128 / pair 512 / tail truncation / 25.3%), training data (85/21 groups), model selection (top-50 macro Complete@10) |
| 3 | test positioning was inconsistent | §5 opens with the exposure caveat; E02–E05 now read "test (re-evaluation)" and "locked test re-evaluation on a previously exposed split"; a matched-budget table was added |
| 4 | Problems section lacked the T01/T04 selection logic | added as its own bullet |
| 5 | A5000 unavailable vs A5000 remaining experiment | all A5000 mentions removed; remaining experiments list a full fine-tune that needs more than the available 6 GB, with no machine named |
| 6 | AI disclosure wording | now reads "AI assistance was used in accordance with the course policy." |
| 7 | "Updated Timeline" was a future schedule | rebuilt as the *updated* version of the proposal plan: phase / planned (weeks 1–14) / actual (artefact dates) / status, plus a short remaining-work table. The added reranker phase is flagged as the one change against the proposal. |

---

## E. Known limitations (research, not format)

1. The reranking gain is **not significant on test** (+0.047, 95% CI [−0.005, +0.095], Holm
   p = 0.15) against +0.110 (p = 0.001) on development.
2. **Partial fine-tuning**: 10 of 12 encoder layers frozen because of the 6 GB limit.
3. **The test split has been seen** (pilot); all test numbers are re-evaluations.
4. `resources` has 7 test questions; domain-level CIs are wide.
5. Pairs longer than 512 tokens are tail-truncated; the sensitivity analysis shows gains are
   larger, not smaller, where truncation occurs.

## F. Optional work that would remove a limitation

| item | effort | limitation removed |
|---|---|---|
| second held-out set | **done**: 45 questions / 45 groups, evaluated once on 2026-11-13 | #3 |
| full fine-tune with more than 6 GB VRAM | done on the A5000 (T06): no measurable gain | #2 |
| expand `resources` to ≥15 test questions | 1 day, but the pool offers only 19 clauses → ~5–8 questions | #4 |

---

## G. Supervisor feedback → revision 3 (2026-11-12)

| # | Comment | Change | Evidence |
|---|---|---|---|
| 1 | Highest priority is evaluation on a new untouched set, with configurations frozen first | `results/test2/PREREGISTRATION.json` fixes endpoints, thresholds, sampling and artefact hashes before drafting; `19_rerank_eval.py --preregistration` refuses on drift and keeps the one-shot rule | `24_preregister_test2.py`, `tests/test_prereg.py` |
| 2 | Keep additional full fine-tuning optional | T06 records the A5000 full fine-tune as done and null (val 0.393 vs 0.390 re-measured; dev Δ +0.004, CI includes zero); §7 lists it as completed, not pending | `models/reranker/COMPAT.json`, `docs/COMPAT_FULL_FINETUNE.md` |
| 3 | Clarify whether completeness can combine several chunks; a per-chunk rule would handicap sentence chunking | §3 gains an explicit union rule; §5 states the semantic failure is retrieval, not the metric, and cites the decomposition (76% of `fixed/128` completions need ≥2 chunks vs 16% for `fixed/512`; semantic completes 0.033 at k=50) | `results/analysis/union_audit_test.json`, `tests/test_goldset.py::test_split_retrieval_union_covers_when_adjacent` |
| 4 | Present individual chunking-strategy results prominently beside the aggregate | §5 gains a per-strategy table (7 families with CIs, union coverage, equal-token recall) immediately after the macro table; `final_strategy.png` added | `scripts/16_stats.py::by_strategy`, `scripts/23_make_report_figures.py` |
| 5 | If comparing learning rates, use matching seed sets | T07 re-runs both rates on three matched seeds: +0.021 paired, positive for every seed; the report discloses that T04/T05 confounded the two factors and does not use them as an lr claim | `results/compat_lr/lr_grid.json`, D16 |
| 6 | Specify an A5000 time and memory budget and protect writing/reproducibility time | §7 table now carries A5000 time and memory per experiment (full FT 9 min / 9.7 GB; test2 eval <2 min / 2–3 GB; lr grid ~25 min); §8 timeline adds the A5000 phase and the revision date | `docs/COMPAT_FULL_FINETUNE.md`, §7 |


---

## H. Second held-out set (test2), revision 4 (2026-11-13)

| item | value |
|---|---|
| questions / groups | 45 / 45, one question per group |
| design | pre-registered before drafting (`results/test2/PREREGISTRATION.json`), frozen bundles (`selected_bundles.json`, `selection_sha256`), one evaluation |
| primary endpoint | structural minus semantic Complete@10 (C1): **+0.800 [0.666, 0.911]** → confirmed at least 0.30 |
| secondary endpoint | C3 − C1: **+0.052 [0.020, 0.089]** → positive, magnitude unresolved (n = 45, MDE 0.079) |
| length match | quartiles 750 / 1022 / 1122 against target 728 / 1010 / 1122 (pass) |
| composition | amended before evaluation (D23): integrity 25/45, Glasgow 33%, all questions single-clause |
| provenance | drafted **and reviewed by the agent** (`agent_self_audit`), no human sign-off (D24); the report says so in 5.1 and §9 |
| artefacts | `results/final_test2/` (four tracks, FINAL_RUN with input hashes), `results/test2/OUTCOME.json`, `DESIGN_AMENDMENTS.json`, `annotations/TEST2_SELF_AUDIT.md` |

Interpretation limits stated in the report: the absolute level is not comparable with the first
test split (single-clause evidence makes completeness easier), the domain mix is
integrity-heavy, and the reranking endpoint cannot exclude a +0.047 effect at this n.
