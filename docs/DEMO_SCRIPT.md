# Live demo script (≈3 minutes)

## What to open, in this order

1. **`results/dashboard.html`** — the results page. Open by double-clicking, or paste
   `file:///<repo>/results/dashboard.html` into the browser.
2. **`docs/PROJECT_REPORT.docx`** — the 5-page report. Hand it over / show on screen.
3. Only if asked: **`models/reranker/FROZEN.json`** (what was frozen and when) and
   **`docs/RUBRIC_CHECK.md`** (requirement-by-requirement self-check).

Do **not** show the chat window: it is the tool used to build the project, not a result.

## 60-second opening

> University course-policy documents are rule-dense, and a policy answer is only correct if
> the *whole* clause is retrieved — including its exception. I built a small frozen corpus of
> 36 documents and 172 questions with annotated evidence spans, then held the encoder, the
> index, the questions and top-k fixed and varied only how text is cut: fixed-length at four
> sizes and three overlaps, paragraph, heading, and semantic splitting — 17 configurations.
> The main finding is that structure-aware chunking wins: heading 0.833 and paragraph 0.750
> on complete-evidence recall at 10, against 0.717 for the best fixed-length setting, while
> sentence-level semantic splitting collapses to near zero because a sentence cannot contain a
> multi-sentence clause. I also added one cross-encoder reranker as a sensitivity check: it
> lifts every configuration by about five points but does **not** change which chunking is
> best.

## 3-minute walkthrough

| step | point at | say |
|---|---|---|
| 1 | four cards, top of the dashboard | the headline: 0.405 complete-evidence recall vs 0.358 baseline, and that the +0.047 gain is **not** significant |
| 2 | four-track table | the 2×2 design; note that two of the contrasts are arithmetic identities and are asserted, not tested |
| 3 | matched-budget table | at 512 tokens reranking closes part of the gap: 0.217 → 0.258 |
| 4 | per-configuration table | the ordering of strategies; heading and paragraph on top, semantic at the bottom |
| 5 | second figure | the rank correlation 0.963 — the chunking conclusion survives reranking |
| 6 | the Word report | methodology, experiment log including **three failed training runs**, and the AI disclosure |

## Questions to expect

**"Is the reranking gain real?"**
> On test, no — +0.047 with a 95% CI of [−0.005, +0.095], Holm p = 0.15. On development it was
> +0.110 with p = 0.001, so the test result did not replicate at the same size. I report it as
> suggestive, not established.

**"Why is the test set not independent?"**
> It was used once as a pilot before the reranker stage; the formal run repeats the measurement
> on the same questions with every setting frozen first. I disclose that in the report — the
> numbers are re-evaluations, not a pristine holdout.

**"Why is the reranker only partially fine-tuned?"**
> A 6 GB card cannot hold a full AdamW update of 278 M parameters; it paged to system memory
> and an epoch took 20 minutes. Freezing 10 of 12 encoder layers and using Adafactor brought it
> to 197 s per epoch. The three aborted attempts are in the experiment log.

**"Why not report the higher 0.396?"**
> That was epoch 1 of a run that never finished; its checkpoint was discarded when aborted, so
> it could never be frozen or evaluated. Selection ran over completed runs only.

**"So which chunking should we use?"**
> Heading-based, or paragraph-based where headings are unreliable. Fixed-length only if the
> document has no structure. Do not use sentence-level semantic splitting for policy text.

**"What would you do with more time?"**
> That experiment is done. A second held-out set was built from clauses no question used,
> pre-registered, locked and evaluated once: the structural-versus-semantic gap replicated
> (+0.800, CI [0.666, 0.911]) and the reranking gain was +0.052, CI [0.020, 0.089] — positive,
> but 45 groups cannot settle its size. What is left is optional: a broader `resources`
> domain, and more questions in domains the clean corpus under-supplies.

## Phrases to avoid

- "state of the art" — this is a controlled comparison on a small corpus, not a benchmark win.
- "the reranker improves retrieval" without a number and a split: +0.047 [−0.005, +0.095] on the
  first test split (not significant), +0.052 [0.020, 0.089] on test2 (positive, size unresolved).
- "the test set shows" — say "the re-evaluation on the test split shows".
- "test2 proves the reranker works": it cannot exclude a +0.047 effect, so a positive interval
  is not a size claim.
- saying A5000 numbers are comparable with the canonical ones: the canonical results are the
  RTX 3060; the A5000 ran compatibility experiments (T06, T07) and the single test2 run.

## If the professor wants to see data, not slides

- `results/final/stats.json` — every number behind the page, with CIs and p-values.
- `results/final_test2/stats.json` + `results/test2/OUTCOME.json` — the second held-out set and
  its pre-registered verdict.
- `results/final/perq_*.jsonl` — per-question scores for all 17 configurations.
- `models/reranker/FROZEN.json` — checkpoint, data and environment hashes.
- `annotations/REVIEW.md` — the 172 questions with the clause text each one cites.
- `annotations/TEST2_REVIEW.md` — the 45 test2 questions, flagged as agent-reviewed.
