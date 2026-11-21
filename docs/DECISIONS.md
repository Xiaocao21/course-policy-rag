# Research decisions log

Each entry records a problem, the decision taken, the evidence behind it, and the effect on
the study. Decisions that changed the protocol after data had been seen are marked
**[post-hoc]** and are disclosed in the report.

---

## D1. Two text layers instead of one normalised corpus
**Problem.** Normalising whitespace and quotes destroys character offsets, so annotated
spans no longer point at the source text.
**Decision.** Keep an evidence layer (character-faithful) plus a retrieval layer and a
`charmap` back to evidence offsets; score only on `ev_ranges` mapped back through the map.
**Evidence.** `03_build_safe_corpus.py` asserts that split pieces tile their clause span.
**Effect.** Span containment can be checked exactly; normalisation cannot silently shift a
boundary.

## D2. Split pieces carry their own evidence ranges
**Problem.** When a long clause was split, every piece claimed the whole clause span, so
retrieving piece 0 scored as full coverage and inflated Recall / Complete.
**Decision.** Each piece records its own `ev_ranges`, and an assertion requires the pieces
of a clause to tile its span with no gap larger than 4 characters.
**Evidence.** 15 clauses were affected; after the fix `split-span errors: 0`.
**Effect.** Coverage is no longer overstated for split chunks.

## D3. Corpus tiers come from `sources.csv`, not from clause records
**Problem.** Nine documents were labelled `faculty` while `sources.csv` said `institution`.
**Decision.** Take tier from the curated source table, with a hard assertion on mismatch.
**Effect.** Tier shares are correct and the 40% cap is enforced on real values.

## D4. Real tokenizer instead of an approximate one
**Problem.** The first corpus used an approximate token count, so "512-token" chunks were
not actually 512 tokens and the model limit was not guaranteed.
**Decision.** Rebuild with the `bge-small-en-v1.5` tokenizer, shrink each chunk until it fits,
and bump the corpus to v0.1.5.
**Effect.** Nominal sizes are exact, `over-limit 0`, maximum unit 509 tokens. All 172
questions were restamped to v0.1.5.

## D5. Question groups are defined by shared gold evidence, splits by groups
**Problem.** The first audit compared only question *text*, reported "no near duplicates",
and missed 20 clauses used on both sides of the split.
**Decision.** Union-find over shared clause ids defines an evidence cluster; a cluster is
never divided across splits. Group ids are then clustered by wording inside the cluster, so
a group means a paraphrase family rather than an evidence tangle.
**Evidence.** Leakage went from 20 clauses to 0; test groups 59.
**Effect.** Tuning and test are evidence-disjoint; the earlier "no leakage" claim was wrong
and is retracted.

## D6. AI-drafted questions, human sign-off
**Problem.** 150+ questions had to be written; the supervisor approved AI assistance for a
course assignment.
**Decision.** Draft with AI from corpus clauses, record provenance on every question, refuse
to let any prefilled question enter the test split until `manual_reviewed` is true.
**Evidence.** 23 content defects were found by the reviewer and fixed; all 172 questions were
then accepted, and the sign-off is logged in `reports/lock_audit.jsonl`.
**Effect.** Provenance is honest and auditable; no fake inter-annotator agreement is claimed.

## D7. Test split demoted to pilot after it was used
**Problem.** The test split was evaluated once before the reranker work; naming that run
"pilot" afterwards does not make the split unseen again.
**Decision.** Keep the pilot numbers, archive them under `results/pilot/`, and describe the
later run as a **formal re-evaluation of a seen split**, never as a first use.
**Effect.** Every report carries the disclosure; no claim of a pristine holdout.

## D8. Invariance is asserted, not tested
**Problem.** The first selection plan used Complete@10 on a reranked top-10 pool, which is a
constant: the set never changes, so the metric cannot move.
**Decision.** Selection uses the top-50 pool; and two contrasts (C2−main, C1−main) are
recorded as arithmetic identities with an assertion of exact equality, never as hypothesis
tests.
**Evidence.** Maximum absolute difference 0.0 across 17 configurations and 60 questions.
**Effect.** No spurious p-values; the informative comparisons are C3−C1 and C3−C2.

## D9. Statistics resample groups, not questions or cells
**Problem.** 60 test questions come from 59 groups and each question is evaluated under 17
configurations, so treating questions or (question, configuration) cells as independent
would shrink intervals and inflate significance.
**Decision.** One shared group-level bootstrap (B = 2000, seed 20261013); configurations are
collapsed by the pre-specified macro-average before resampling; Holm correction within
declared families.
**Effect.** Intervals reflect the actual number of independent units (59).

## D10. One reranker for all 17 configurations
**Problem.** Per-configuration rerankers would confound the chunking effect with the model
effect.
**Decision.** Train a single cross-encoder on development data only, apply it unchanged to
every configuration.
**Effect.** Differences between configurations remain attributable to chunking.

## D11. Partial fine-tuning on a 6 GB card
**Problem.** Full fine-tuning of 278 M parameters in fp32, plus AdamW moments and gradients,
exceeds 6 GB VRAM; the Windows driver paged to system memory, throughput fell to ~8.7
pairs/s and an epoch took ~20 minutes.
**Decision.** Adafactor (factorised state) plus freezing the first 10 of 12 encoder layers
(85.6 M trainable), which brought VRAM to 4.1 GB and an epoch to ~197 s.
**Evidence.** Aborted attempts are recorded in `experiments/aborted_runs.jsonl`.
**Effect.** Training is feasible on the available hardware, at the cost of a weaker
fine-tune; this is recorded in `FROZEN.json` (`frozen_encoder_layers: 10`) and in the
limitations.

## D12. Canonical training environment
**Problem.** The lab A5000 and the local 3060 would produce slightly different fp16 results,
and mixing them would silently break reproducibility.
**Decision.** The 3060 is canonical; other devices may only write `COMPAT.json`, never
`FROZEN.json`, and their numbers are not comparable. The A5000 was ultimately unavailable,
so all reported numbers come from the canonical machine.
**Effect.** One environment, one checkpoint, one set of numbers.

## D13. No `--force` on the formal test run
**Problem.** A `--force` flag would make "evaluate once" unenforceable.
**Decision.** Replace it with an attempt protocol: `IN_PROGRESS.json` before, `FINAL_RUN.json`
after, `--resume` archives a failed attempt under `attempts/attempt-N/` and increments the
counter. A completed run cannot be repeated at all.
**Effect.** Failures stay on the record and the number of attempts is reportable.

## D14. Report at 3–5 pages in Word
**Problem.** The deliverable is a short PDF; a long Markdown file does not meet the brief.
**Decision.** Write `TEACHER_BRIEF.md`, generate `.docx`, convert to PDF for a page-count
check, and iterate until 3–5 pages with readable tables and figures.
**Effect.** The submitted document matches the required format without hand-editing tables.

---

## D15. Completeness is a union rule, MRR is the single-chunk exception
**Problem.** A reviewer read "a clause counts only when the retrieved intervals contain its
entire span" as "one whole clause must sit inside one chunk", which would handicap
sentence-level chunking by construction.
**Decision.** Keep the union rule and say so explicitly. `coverage_for_set` accumulates the
`ev_ranges` of every retrieved chunk in a document and asks whether their union covers the
span; adjacent intervals join, gaps do not. MRR is the one metric that needs a single chunk,
so the report leads with Complete@10 / Recall@10 and uses MRR only for ordering. The report
also carries the empirical decomposition (`22_union_coverage_audit.py`): 76% of the complete
answers produced by `fixed/128` on the test split needed two or more chunks, against 16% for
`fixed/512`, and even at k=50 the semantic family completes 0.033 of questions.
**Effect.** The sentence-level result is explained as a retrieval failure, not as a metric
artefact, and the explanation is checkable in `results/analysis/union_audit_test.json`.

## D16. Learning rates may only be compared with matched seeds
**Problem.** Training seeds were derived as `base + 10·(lr×10⁵) + negatives`, so the two
learning rates in the frozen grid also differed in the seed (20261027 vs 20261037). The
T04/T05 gap of 0.046 therefore mixed the two factors.
**Decision.** Keep the derivation for the frozen run (it makes a re-run exact), but add
`--seed-mode paired`, which uses the requested seed verbatim for every grid point, and use it
for the comparison (`25_lr_grid.py`). The frozen selection is unaffected: it needed one
model, not a learning-rate ranking.
**Evidence.** Three matched seeds: lr 1e-5 scores 0.3877 / 0.3797 / 0.3797, lr 2e-5 scores
0.3636 / 0.3529 / 0.3663; the paired difference is +0.021 and positive for all three seeds.
`results/compat_lr/lr_grid.json`.
**Effect.** The learning-rate claim survives, at roughly half its previously reported size.

## D17. The second held-out set is pre-registered
**Problem.** The test split has had its one formal use (D7), so a new set is the only route
to a confirmatory number, and an unregistered new set invites post-hoc endpoint shopping.
**Decision.** Write `results/test2/PREREGISTRATION.json` before drafting: endpoint
definitions, thresholds, decision matrix, power, sampling rules and a hash of every artefact
the evaluation depends on. `19_rerank_eval.py --preregistration` refuses to run when a hash
has moved, and the existing one-shot `FINAL_RUN.json` rule still applies.
**Effect.** The second set can be evaluated once, against settings that provably did not
change between registration and evaluation.

## D18. The test split is only re-analysed on the dense tracks
**Problem.** The union audit needs retrieved lists, but recomputing metrics on the test split
looks like a second evaluation.
**Decision.** On the test split the audit is restricted to `main` and `C1`, which are CPU
deterministic and reproduce the canonical per-cell values exactly (0/1904 differing cells).
The reranked decomposition runs on development only. Outputs go to `results/analysis/`,
never `results/final*`.
**Effect.** The clarification is supported without a second test use, and the restriction is
visible in the code (`22_union_coverage_audit.py` refuses `--split test` with C3).

## D19. A new question set is length-matched to the existing difficulty
**Problem.** The usable unused pool has a median clause of 439 characters against 758 for the
clauses already used, so a new set drawn naively would be systematically easier and its
absolute numbers would not be comparable with the test split.
**Decision.** Sample by clause length first (target: the test split's per-question maximum
clause span, quartiles [728, 1010, 1122]), then domain, then institution, with document
novelty only as a tie-break.
**Effect.** The comparison is within-regime. The length *gradient* itself is not invoked as
justification: direction is positive in all four split×track combinations but not significant
in any (D21), so matching is a composition safeguard rather than a correction.

## D20. Derived statistics are produced by code
**Problem.** A per-strategy table and a length-stratified contrast were first drafted as
hand-written numbers, which drift from the data on the next re-run.
**Decision.** `16_stats.py` emits `by_strategy`, `family_contrasts` and `by_length_tercile`;
the report generator reads them and a test asserts the printed values equal the stored ones.
**Effect.** Re-running the statistics cannot leave the report quoting stale figures.

## D21. Small-sample contrasts report t intervals, bootstrap secondarily
**Problem.** The length contrast rests on buckets of about 19-20 groups. A percentile
bootstrap interval there moves its lower bound between -0.013 and +0.009 across seeds, so it
can accidentally appear to exclude zero while the corresponding t interval does not.
**Decision.** Report the Welch t 95% interval as the precision estimate, keep the bootstrap
interval as a secondary column with a `stable` flag, and publish no p-value for that contrast.
**Effect.** The statement "positive in all four combinations, significant in none" is
defensible without quoting a number that flips sign with the random seed.

---

## D22. The second held-out set is built from frozen evidence bundles
**Problem.** A rule or cross-clause question needs more than one clause, so freezing 45 primary
clauses would not have frozen the evidence of 45 questions. It also became clear that the
"usable" pool of 1,239 clauses is not 1,239 *question-usable* clauses: a readability screen
(no OCR/citation fragments, no URL lines, no table rows, no bullet continuations) leaves 768.
**Decision.** `26_select_test2_clauses.py` draws 45+8 bundles deterministically (seed
20261013, length quartiles first, then domain, then a 10-per-institution cap and 5-per-document
cap), records every eligible companion candidate with its exclusion reasons, and freezes the
list with `selection_sha256`. After freezing, only a choice from the frozen candidates, a
downgrade to single-clause, or an append-only amendment is allowed.
**Effect.** The evidence of each question is fixed before any question text exists, and no
companion can be found by re-searching the corpus after the fact.

## D23. test2 composition was amended before evaluation, not forced
**Problem.** The clean pool could not supply the planned composition: it is dominated by
student-conduct codes.
**Decision.** Three deviations were recorded in `results/test2/DESIGN_AMENDMENTS.json` before
any retrieval ran: the domain mix is integrity-heavy (25 of 45), Glasgow supplies 33% (above
the 25% design cap), and no defensible two-clause relation survived review, so all 45
questions are single-clause. Forcing the quotas would have required clauses that cannot carry
a question.
**Effect.** test2 answers the structural-versus-semantic question, but domain- and
type-stratified findings for this split are indicative only. The report says so in 5.1.

## D24. test2 was agent-reviewed, and the record says so
**Problem.** The second set was drafted and reviewed by the agent, not by the student. Setting
`manual_reviewed=true` without saying who reviewed would misrepresent the provenance, and the
earlier `review_status` label said `student_reviewed` regardless.
**Decision.** `05_audit_splits.py` now derives the status from the recorded reviewer
(`agent_reviewed` vs `student_reviewed`). Every test2 question carries
`review.reviewer = agent_self_audit`, its six checks, and a `review_history` that keeps the
pre-revision answer. The lock records the reviewers, and `reports/lock_audit.jsonl` has an
explicit `review_provenance` event. The report and this log state that test2 has no human
sign-off, unlike the first set.
**Effect.** The provenance is accurate; the limitation (an agent can check entailment but not
supply independent human judgement) is disclosed rather than hidden behind a flag.
