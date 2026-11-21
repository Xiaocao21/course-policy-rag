# Four-track evaluation: chunking × reranking on course-policy retrieval

> **Scope.** This document analyses the **first** test split. The second held-out set (test2,
> 45 questions, evaluated once on 2026-11-13) is reported in §5.1 of
> `docs/PROJECT_REPORT.docx` and in `results/test2/OUTCOME.json`; it is not merged into the
> tables below, because its questions are single-clause and its absolute level is not
> comparable.

Corpus **v0.1.5** · 172 questions (112 development / 60 test, 59 test groups) · 17 chunking
configurations · frozen bi-encoder `BAAI/bge-small-en-v1.5` (384-d, CLS pooling) · one
frozen cross-encoder `BAAI/bge-reranker-base` (`FROZEN.json`, run `8c0e9c786bb6`).

**Disclosure.** The test split was evaluated once before this run, as an exploratory pilot
(`results/pilot/`). This report is therefore a **formal re-evaluation of a split that has
already been seen**, with every setting frozen beforehand; it is not a first, unseen test.
Pilot and formal numbers are reported side by side in §7.

---

## 1. Design

A 2×2 factorial over candidate pool size and reranking:

| track | dense pool | reranker | what it isolates |
|---|---|---|---|
| main | top-10 | no | baseline; same protocol as the pilot |
| C1 | top-50 | no | pool effect alone |
| C2 | top-10 | yes | ordering effect at a fixed pool |
| C3 | top-50 | yes | the proposed configuration |

Two contrasts are arithmetic identities, not hypotheses, and are **asserted** rather than
tested:

* **C2 = main** on Recall@10 / Hit@10 / Complete@10 — reranking the same ten chunks cannot
  change the set;
* **C1 = main** on the same three — the first ten of a dense top-50 *are* the dense top-10.

Both assertions passed with a maximum absolute difference of **0.0** across all 17
configurations and 60 questions.

The informative set-level comparisons are therefore **C3 − C1** (fixed pool; reranking may
pull chunks 11–50 into the top ten; pre-specified primary) and **C3 − C2** (pool effect with
reranking active; secondary), with Holm correction over the two.

---

## 2. Main results

Group-level bootstrap (B = 2000, 59 groups, seed 20261013), configurations collapsed by a
macro-average before resampling:

| track | Complete@10 | 95% CI | Recall@10 | MRR |
|---|---|---|---|---|
| main | 0.358 | [0.299, 0.418] | 0.412 | 0.276 |
| C1 | 0.358 | [0.299, 0.418] | 0.412 | 0.282 |
| C2 | 0.358 | [0.299, 0.418] | 0.412 | 0.312 |
| **C3** | **0.405** | **[0.347, 0.464]** | **0.467** | **0.322** |

| contrast | metric | Δ | 95% CI | p (Holm) |
|---|---|---|---|---|
| C3 − C1 (primary) | Complete@10 | +0.047 | [−0.005, +0.095] | 0.150 |
| C3 − C2 (secondary) | Complete@10 | +0.047 | [−0.005, +0.095] | 0.150 |

**The gain is directionally consistent but not statistically significant on the test
split.** The confidence interval includes zero. On the development split the same comparison
gave **+0.110 [+0.079, +0.141], p_holm = 0.001**, so the effect did not replicate at the
same size; the honest reading is *suggestive, not established*.

---

## 3. Per configuration

Complete@10 on the test split; ranking of configurations is nearly unchanged by reranking
(Spearman ρ = 0.963 between C1 and C3).

| configuration | main | C3 | Δ | main MRR | C3 MRR |
|---|---|---|---|---|---|
| heading | 0.733 | **0.833** | +0.100 | 0.613 | 0.622 |
| paragraph | 0.667 | 0.750 | +0.083 | 0.541 | 0.606 |
| fixed/512_ov64 | 0.617 | 0.717 | +0.100 | 0.483 | 0.530 |
| fixed/384_ov64 | 0.633 | 0.650 | +0.017 | 0.385 | 0.423 |
| fixed/512_ov32 | 0.533 | 0.533 | 0.000 | 0.394 | 0.392 |
| fixed/256_ov64 | 0.500 | 0.533 | +0.033 | 0.240 | 0.284 |
| fixed/256_ov32 | 0.483 | 0.517 | +0.033 | 0.282 | 0.344 |
| fixed/512_ov0 | 0.433 | 0.483 | +0.050 | 0.402 | 0.473 |
| fixed/384_ov32 | 0.433 | 0.567 | +0.133 | 0.359 | 0.455 |
| fixed/384_ov0 | 0.350 | 0.367 | +0.017 | 0.275 | 0.352 |
| fixed/256_ov0 | 0.283 | 0.350 | +0.067 | 0.297 | 0.355 |
| fixed/128_ov32 | 0.167 | 0.217 | +0.050 | 0.052 | 0.086 |
| fixed/128_ov64 | 0.133 | 0.183 | +0.050 | 0.050 | 0.116 |
| fixed/128_ov0 | 0.050 | 0.050 | 0.000 | 0.047 | 0.082 |
| semantic/0.70 | 0.033 | 0.067 | +0.033 | 0.086 | 0.110 |
| semantic/0.75 | 0.017 | 0.033 | +0.017 | 0.105 | 0.108 |
| semantic/0.80 | 0.000 | 0.000 | 0.000 | 0.045 | 0.077 |

**The chunking conclusion survives reranking**: structural strategies (heading, paragraph)
remain best and sentence-level semantic splitting remains unusable, whether or not the
reranker is applied. Reranking shifts the level, not the ranking.

---

## 4. Order-sensitive effects at a fixed pool (C2 − main)

These metrics *can* move without changing the retrieved set. Family-wise Holm correction
across seven order-sensitive metrics:

| metric | Δ | 95% CI | p (Holm) |
|---|---|---|---|
| Recall@3 | +0.058 | [+0.023, +0.095] | **0.014** |
| Recall@5 | +0.044 | [+0.015, +0.074] | 0.065 |
| MRR | +0.036 | [+0.003, +0.070] | 0.259 |
| Recall@1 | +0.033 | [−0.003, +0.070] | 0.370 |
| token-budget recall @512 | +0.040 | [+0.001, +0.078] | 0.282 |
| @1024 | +0.040 | [+0.006, +0.075] | 0.210 |
| @2048 | +0.031 | [+0.006, +0.057] | 0.143 |

So reranking a ten-chunk pool buys a modest, mostly non-significant improvement in *early*
recall, and changes nothing about whether the pool contains the evidence.

---

## 5. Matched token budgets

Macro over configurations of recall under a complete-chunk budget (admission stops at the
first chunk that does not fit):

| track | @512 | @1024 | @2048 |
|---|---|---|---|
| main | 0.215 | 0.283 | 0.348 |
| C1 | 0.215 | 0.284 | 0.363 |
| C2 | 0.253 | 0.321 | 0.378 |
| C3 | **0.254** | **0.326** | **0.420** |

At 512 and 1024 tokens the pool size is irrelevant (the budget runs out before ten chunks),
so C2 ≈ C3; at 2048 the larger pool starts to pay off.

---

## 6. Heterogeneity

**Per document** (best configuration per track, documents weighted equally — the corpus is
concentrated, with the Glasgow assessment guide supplying 26.8% of clause references):

| track | config | macro over documents | pooled |
|---|---|---|---|
| main | heading | 0.732 | 0.733 |
| C3 | heading | **0.894** | 0.833 |

**Per domain** (heading, Recall@10):

| domain | n | main | C3 |
|---|---|---|---|
| attendance | 8 | 0.500 | 0.750 |
| exam | 13 | 0.769 | 0.885 |
| grading | 12 | 0.833 | 0.833 |
| integrity | 12 | 0.750 | 0.917 |
| late | 8 | 0.875 | 0.812 |
| resources | 7 | 0.857 | 1.000 |

`resources` has only 7 test questions and `attendance` only 8; those rows are indicative.

**Truncation** (1,020 question×configuration cells; the reranker sees at most 512 tokens, so
long chunks are tail-truncated). Median truncated fraction 0.00, p75 0.30. The correlation
between the truncated fraction and the C3 − C1 gain is **ρ = +0.53**: the benefit
concentrates where pairs had to be cut (bottom tercile +0.032, middle +0.118, top +0.067).

---

## 7. Pilot versus formal

The pilot used exactly the protocol now called `main` (dense top-10, no reranker), so the
two can be compared directly as a reproducibility check:

| | pilot | formal main | formal C3 |
|---|---|---|---|
| Complete@10, macro over 17 configurations | 0.3569 | 0.358 | 0.405 |
| Recall@10, macro over 17 configurations | 0.4113 | 0.412 | 0.467 |
| best configuration, Complete@10 | heading 0.733 | heading 0.733 | heading 0.833 |

Pilot and formal `main` agree to within 0.001 on both metrics, which is what a deterministic
re-execution of the same protocol should do. The earlier pilot headline of 0.733/0.767 belongs
to a *single* configuration (heading), not to the macro average; quoting it next to a macro
figure would overstate the system.

This comparison also shows why the split is no longer a clean holdout: the pilot fixed the
expectations, and the formal run was computed by someone who had already seen those numbers.

---

## 8. Limitations

1. **The test split is not unseen.** It was used for the pilot; the formal run is a
   re-evaluation with frozen settings, and no claim of a pristine holdout is made.
2. **Reranking is not established on test.** The primary contrast is +0.047 with a CI
   spanning zero; only the development split showed a significant gain.
3. **Partial fine-tuning.** On a 6 GB card the fp32 weights, gradients and optimizer state
   of all 278 M parameters exhaust VRAM and the driver pages to system memory, so the first
   10 of 12 encoder layers are frozen (85.6 M trainable). The A5000 was unavailable.
4. **Truncation.** 25.3% of development pairs and a median 0–30% of test cells exceed 512
   tokens and are tail-truncated; the sensitivity analysis in §6 shows the gain is larger,
   not smaller, where truncation occurs.
5. **Sparse domains.** `resources` (7 questions) and `attendance` (8) cannot support
   domain-level inference.
6. **Corpus concentration.** Macro-over-documents is reported alongside pooled means for
   this reason.
7. **One reranker for 17 configurations** is a deliberate control, not an oversight: per
   configuration rerankers would confound model and chunking effects.

---

## 9. Reproducing

```
.venv-gpu\Scripts\python.exe scripts\19_rerank_eval.py --split dev
.venv\Scripts\python.exe      scripts\16_stats.py --split dev
.venv-gpu\Scripts\python.exe scripts\19_rerank_eval.py --split test --acknowledge-test-use
.venv\Scripts\python.exe      scripts\16_stats.py --split test
```

Artifacts: `results/final/{main,C1,C2,C3}/*.json`, `summary_<track>.csv`,
`perq_<track>.jsonl`, `stats.json`, `ATTEMPTS.json`, `FINAL_RUN.json`; frozen model metadata
in `models/reranker/FROZEN.json`; run history in `experiments/log.jsonl` and
`experiments/aborted_runs.jsonl`.
