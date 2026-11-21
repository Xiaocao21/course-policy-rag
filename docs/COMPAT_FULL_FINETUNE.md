# Full fine-tuning on the A5000 — run record

Written 2026-11-11 on the A5000 machine, following `docs/HANDOFF_A5000.md` §0 and §6.
Everything here is **compatibility mode**: it does not replace, and must not be confused
with, the canonical results in `results/final*`.

## 0. Summary

| item | value |
|---|---|
| full fine-tuning (0 frozen layers, 278 M trainable, AdamW, batch 32) | **val macro Complete@10 = 0.3930** |
| frozen partial fine-tuning (10/12 layers frozen, 85.6 M trainable) | val 0.3877 (recorded on the 3060) / **0.3904 re-measured here** |
| dev, full − partial, C3 Complete@10 | **+0.004, 95% CI [−0.004, +0.012], p ≈ 1.0** |
| dev, full − partial, C3 MRR | −0.005, CI [−0.015, +0.005] |
| verdict | **no measurable gain from full fine-tuning** |

The honest same-machine comparison is the second and third rows: on this GPU the frozen
checkpoint scores 0.3904, not 0.388, so most of the apparent "+0.005" in the handoff's
suggested comparison is the difference between two GPUs rather than between two models.

## 1. Environment built (handoff §3)

Python 3.12.10 (installed with `winget install Python.Python.3.12`; the machine shipped
only 3.14, which the CUDA wheels do not support).

| env | contents | version |
|---|---|---|
| `.venv` | `requirements.txt` + numpy | numpy 2.5.4 |
| `.venv-gpu` | torch cu124, transformers, sentence-transformers, faiss-cpu | torch **2.6.0+cu124**, CUDA 12.4, transformers 5.22.2, sentence-transformers 6.3.0, faiss-cpu 1.16.0 |
| GPU | NVIDIA RTX A5000 | 24 GB, driver 596.71 |

Torch resolved to `2.6.0+cu124`, i.e. the same build recorded in `FROZEN.json`.

## 2. Reproduction check before changing anything (handoff §4)

```
.\.venv\Scripts\python.exe scripts\16_stats.py --split test
```

| quantity | expected by handoff | observed |
|---|---|---|
| main | 0.358 | 0.358 |
| C3 | 0.405 | 0.405 |
| C3 − C1 | +0.047, Holm p = 0.15 | +0.047, Holm p = 0.150 |
| invariance C1/C2 vs main | 0.0 | 0.0 |
| rank stability (C1 vs C3) | — | 0.9632 |

Also verified byte-for-byte: `models/reranker/FROZEN.json` has sha256
`68e06ad7…`, which equals `frozen_sha256` in `results/final/FINAL_RUN.json`, and the frozen
checkpoint had sha256 `547060da…`, which equals `FROZEN.json → model.checkpoint_sha256`.

## 3. Pre-flight: the run-directory collision

`18_train_reranker.py` derives the output directory from
`rk-{lr}-n{negatives}-s{seed}`, and `select_best` evaluates the same grid as the frozen run.
A full fine-tune at lr 1e-5 / 4 negatives / seed 20261027 therefore targets
`models/reranker/runs/rk-1e-05-n4-s20261027` — **the directory that holds the frozen
canonical checkpoint** — and would have silently overwritten the 1.1 GB weights behind
`FROZEN.json`.

Because `19_rerank_eval.find_checkpoint` locates the model by hash rather than by name, the
frozen directory was renamed before training and its bytes re-verified:

```
models/reranker/runs/rk-1e-05-n4-s20261027
  → models/reranker/runs/rk-1e-05-n4-s20261027-partial-3060-FROZEN
sha256 547060da6310cfd6cb31b75521f75c6af4161aa916196330ef1492ee91719ba2   (unchanged)
```

The new full fine-tune then took the plain `rk-1e-05-n4-s20261027` name. `FROZEN.json` was
not touched (mtime still 2026-10-25) and `results/final` was not written.

## 4. Training (handoff §6)

```
.\.venv-gpu\Scripts\python.exe scripts\18_train_reranker.py --freeze-encoder-layers 0 `
    --optimizer adamw --batch 32 --grad-accum 1 --epochs-max 3 --env-role compatibility
```

Log: `models/reranker/train_compat_full_a5000.log`. The default grid
(`1e-5:4,2e-5:4`) was kept, so the same two runs compete under the pre-registered
selection rule.

| run | epoch | train loss | macro Complete@10 | time (train/eval) |
|---|---|---|---|---|
| lr 1e-5, n 4, seed 20261027 | 1 | 0.4502 | **0.3930** | 64.0 s / 32.9 s |
| | 2 | 0.2801 | 0.3743 | 63.8 s / 32.9 s → early stop |
| lr 2e-5, n 4, seed 20261037 | 1 | 0.4100 | 0.3770 | 63.7 s / 32.9 s |
| | 2 | 0.2153 | 0.3636 | 63.8 s / 32.9 s → early stop |

Peak GPU memory **9.7 GB** for all 278 M parameters in fp32 plus AdamW state and batch 32.
An epoch takes 64 s here against ~1180 s paged on the 6 GB card (`aborted_runs.jsonl`,
2026-10-25T01:59). Selected: lr 1e-5 → `models/reranker/runs/rk-1e-05-n4-s20261027`,
`COMPAT.json` (`run_id c3d403324448`, `frozen_encoder_layers: 0`), and one `training` row
appended to `experiments/log.jsonl`.

The learning-rate result repeats the 3060 pattern: 2e-5 is worse, and validation peaks at
epoch 1 while training loss keeps falling — the model overfits within one epoch at either
depth.

## 5. Dev comparison: full vs partial, same machine

`19_rerank_eval.py` refuses to evaluate anything that is not the canonical freeze, by
design. The comparison was therefore made with two small scripts that reuse the project's
own metric code instead of re-deriving it:

| script | role |
|---|---|
| `scripts/20_compat_dev_compare.py` | runs the four tracks of `19` for two checkpoints in one process, over one shared set of dense top-50 pools, on the dev split |
| `scripts/21_compat_dev_delta.py` | paired, group-level bootstrap of full − partial (same groups, same resample, seed 20261013, B = 2000) |

```
python scripts/20_compat_dev_compare.py --tag dev_full-vs-partial `
    --model partial=…/rk-1e-05-n4-s20261027-partial-3060-FROZEN `
    --model full=…/rk-1e-05-n4-s20261027
python scripts/16_stats.py --split dev --results results/compat_dev/dev_full-vs-partial/<tag>
python scripts/21_compat_dev_delta.py --a …/partial --b …/full
```

Self-checks that passed:

* `main` and `C1` do not touch the reranker and came out **identical** for both checkpoints
  (asserted in `20`, `max|diff| = 0` in `16_stats`), so the dense path is
  checkpoint-independent and matches `19`.
* The frozen checkpoint re-measured here gives C1 0.296 / C3 0.406 / C3−C1 +0.110
  (p_holm 0.001) — **identical to `results/final_dev`** produced on the 3060, so the
  A5000 does not move these metrics at all.
* Restricting to the 22 validation questions reproduces the training-time selection metric
  for the full model exactly (0.3930), which confirms the re-execution is faithful.

### Results (dev, 112 questions, 106 groups, macro over 17 configurations)

| track | metric | partial (frozen) | full | Δ (full − partial) |
|---|---|---|---|---|
| main / C1 | Complete@10 | 0.296 | 0.296 | 0.000 (by construction) |
| C2 | Complete@10 | 0.296 | 0.296 | 0.000 (by construction) |
| C3 | Complete@10 | 0.406 | 0.410 | **+0.004 [−0.004, +0.012], p ≈ 1.0** |
| C3 | Recall@10 | 0.511 | 0.515 | +0.004 [−0.004, +0.013] |
| C3 | MRR | 0.464 | 0.459 | −0.005 [−0.015, +0.005] |
| C3 | tok2048 recall | 0.488 | 0.495 | +0.006 [−0.001, +0.015] |

Within-model contrasts are unchanged in kind: partial C3−C1 = +0.110 [+0.079, +0.141],
full C3−C1 = +0.114 [+0.086, +0.142], both p_holm = 0.001.

How small the difference really is:

* at the level of individual question × configuration cells, 37 of 1,904 flip (22 up, 15
  down) — 1.9%;
* 22 of 106 groups change at all; 7 of 17 configurations are untouched, and the largest
  per-configuration moves are ±0.027, i.e. three questions (`size128_ov64` −3,
  `size512_ov64` +3);
* the ranking of configurations barely moves (Spearman 0.9975 between the two checkpoints
  on C3).

Full fine-tuning buys nothing measurable and costs 3.2× the trainable parameters.

### The machine confound in the suggested comparison

HANDOFF §6 suggests comparing the new validation number against the frozen 0.388. Both
numbers are real, but they were produced on different GPUs, and the frozen checkpoint
re-evaluated here gives **0.3904**, not 0.3877. Two checks pin the cause down:

* the re-evaluation is deterministic — the frozen checkpoint scores 0.3904 twice in one
  process, so this is not run-to-run noise;
* an independent A5000 run of the same recipe and seed produced *different weights* (the
  checkpoint hash differs) which score 0.3877, i.e. training itself is not bit-reproducible
  across machines even with a fixed seed.

So both the weights and the machine contribute, which is precisely why nothing here is
compared with canonical numbers directly:

| comparison | gap | reading |
|---|---|---|
| COMPAT 0.3930 vs FROZEN 0.3877 (as written in the handoff) | +0.0053 | mixes two GPUs |
| same machine, val subset: 0.3930 vs 0.3904 | +0.0026 | ≈ one question |
| same machine, full dev: +0.004 [−0.004, +0.012] | +0.004 | not distinguishable from zero |

The three agree in sign and all are within noise, but only the last two are legitimate
evidence. This is exactly the confound `DECISIONS.md` D12 was written to prevent.

## 6. What this means for the A/B/C decision (handoff §6)

**Path C (promote the A5000 to canonical and re-run test) is now unsupported by evidence.**
Its only justification would be that full fine-tuning is better, and it is not: the dev
difference is +0.004 with a CI spanning zero, and MRR is marginally worse. Path C would
also spend the second and final test use to buy nothing.

**Path A (dev-only comparison) is already done** and costs nothing further. It removes
limitation #2 of the report ("partial fine-tuning was forced by 6 GB") in the honest form:
the hardware limit did not cost anything measurable, at least on this corpus and at this
model size.

**Path B (a new `test2` from the 1,239 usable unused clauses — 1,771 are unused in total) is
the only route to a new confirmatory number**, and it addresses a different limitation — the
exposure of the test split (#3 / D7) — not the fine-tuning depth. Its cost (~2 days) does not
depend on the result above.

### Decision taken

**Path A, landed 2026-11-11** — the dev-only full-versus-partial comparison above stands as
the answer to the full fine-tuning question, with no new test claim. `REPORT_FINAL.md` and
`RUBRIC_CHECK.md` were deliberately left unedited at this point; this file is the record.

**Path B, landed 2026-11-13** — the second held-out set was built, pre-registered, locked and
evaluated once (45 questions / 45 groups). Primary endpoint, structural minus semantic:
**+0.800 [0.666, 0.911]**, confirmed. Secondary, C3 − C1: **+0.052 [0.020, 0.089]**, positive
with the magnitude unresolved at this n. The question this answered was the split exposure,
not the fine-tuning depth; the full fine-tune remains the null result recorded above. See
`results/test2/OUTCOME.json` and §5.1 of `docs/PROJECT_REPORT.docx`.

Path C was rejected: it would spend the second and final test use to promote a model that
full fine-tuning did not improve.

If B is opened later, the entry points are already in place and none of the work above needs
repeating: the full and partial checkpoints, their `COMPAT.json` / `FROZEN.json` metadata,
the dense pools (rebuilt in seconds from `indexes/`), and `scripts/20_compat_dev_compare.py`
+ `scripts/21_compat_dev_delta.py`, which take any two result directories. B's cost sits
almost entirely in drafting and signing off 30–40 questions over unused clauses, locking the
set, and spending its single evaluation — not in modelling.

## 7. What was deliberately not done

* The test split was **not** re-evaluated. `results/final/FINAL_RUN.json` is untouched and
  still records attempt 1 of 2026-10-28.
* `work1_Grouped_Test_Revision.docx` was not modified.
* `FROZEN.json`, `results/final`, and the frozen checkpoint are unchanged.
* `results/final_dev/stats.json` and `results/final/stats.json` were rewritten by the
  deterministic re-run of `16_stats.py` during the §2 check (same inputs, same seed, same
  values).

## 8. Note on code fingerprints

`scripts/` gained five files after the freeze — `20_compat_dev_compare.py`,
`21_compat_dev_delta.py`, `22_union_coverage_audit.py`, `23_make_report_figures.py`,
`24_preregister_test2.py`, `25_lr_grid.py` (six, counting the grid driver) — so a hash of
`scripts/*.py` computed today will not match `FROZEN.json → code.scripts_sha256`, and
`COMPAT.json`'s fingerprint reflects the scripts directory as it stood at training time. The
training code path (`18_train_reranker.py`) gained only additive flags (`--run-tag`,
`--no-freeze`, `--seed-mode`) whose defaults reproduce the frozen behaviour, and the metric
modules (`13_evaluate.py`, `goldset.py`, `common.py`) are unmodified.

The six T07 experiment runs keep their `train_log.json` (the metrics quoted in the report)
but not their 1.1 GB checkpoints: they were `--no-freeze` experiment runs, and `22`, `19` and
`24` locate the canonical checkpoint by walking `runs/`, so retaining unused weights would
only slow every later evaluation. The canonical checkpoint and the T06 full fine-tune
checkpoint are untouched.

