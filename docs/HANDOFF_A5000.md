# Handoff to the A5000 machine

Written 2026-11-11. This file is the entry point for a fresh session on another computer:
the chat history does not travel, but every decision, number and artefact does, because they
live in the repository.

## 1. What the project is

A controlled comparison of 17 chunking configurations for course-policy RAG retrieval, plus a
single cross-encoder reranker as a sensitivity check.

| item | value |
|---|---|
| corpus | v0.1.5, 36 documents, 1,988 clauses, 2,007 retrieval units |
| questions | 172 (112 development / 60 test, 59 test groups) |
| encoder | `BAAI/bge-small-en-v1.5`, frozen, 384-d, FAISS `IndexFlatIP` |
| reranker | `BAAI/bge-reranker-base`, partial fine-tune (10/12 layers frozen), frozen on 2026-10-25 |
| test split | locked 2026-10-24, formally evaluated once on 2026-10-28 |

Read these first, in this order:

1. `docs/DECISIONS.md` — every methodological decision and why.
2. `docs/RERANKER_PROTOCOL.md` — labels, selection metric, canonical environment.
3. `results/REPORT_FINAL.md` — the results and their limitations.
4. `docs/RUBRIC_CHECK.md` — requirement-by-requirement self-check.
5. `experiments/aborted_runs.jsonl` — the training runs that failed, and why.

## 2. What to copy

Copy the project directory, **excluding** `.venv` and `.venv-gpu` (they are machine-specific).

| directory | size | copy? |
|---|---|---|
| `raw/` `extracted/` `cleaned/` | 107 MB | optional (rebuildable, but needed for a corpus rebuild) |
| `chunks/` `indexes/` | 92 MB | yes — saves rebuilding 17 indexes |
| `models/` | 2.2 GB | yes — the 1 GB base model and the 1.1 GB frozen checkpoint |
| everything else | ~20 MB | yes |

Total ≈ 2.5 GB. From the current machine:

```powershell
robocopy <repo> \\A5000\share\course-policy-rag /E /XD .venv .venv-gpu __pycache__
```

or zip it first if there is no share:

```powershell
Compress-Archive -Path <repo>\* -DestinationPath .\cps5801.zip
```

## 3. Set up on the A5000 machine

Requires Python 3.12.

```powershell
# CPU environment for analysis, statistics and report generation
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# CUDA environment for the reranker (cu124 works with any driver >= 525)
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.\.venv-gpu\Scripts\python.exe -m pip install transformers numpy tqdm faiss-cpu sentence-transformers
```

On Linux the same commands work with `python3.12` and `.venv/bin/python`.

Verify:

```powershell
.\.venv-gpu\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 4. Reproduce the existing numbers (do this before changing anything)

```powershell
# statistics on the finished test results
.\.venv\Scripts\python.exe scripts\16_stats.py --split test

# the results page and the Word report
.\.venv\Scripts\python.exe scripts\make_dashboard.py
.\.venv\Scripts\python.exe scripts\make_project_report.py
```

Expected: C3 = 0.405, main = 0.358, C3 − C1 = +0.047 with Holm p = 0.15. If those differ, stop
and find out why before running anything new.

## 5. Guardrails — read before running experiments

1. **The test split cannot be evaluated again.** `19_rerank_eval.py --split test` refuses once
   `results/final/FINAL_RUN.json` exists. There is no `--force`; this is deliberate.
2. **The canonical environment is the RTX 3060.** `FROZEN.json` says `env_role: canonical` and
   `18_train_reranker.py` refuses to write `FROZEN.json` on any other GPU. On the A5000 use
   `--env-role compatibility`, which writes `COMPAT.json` instead. Numbers from a different
   GPU are **not** comparable to the reported ones, even with the same checkpoint.
3. **Evaluation on a different GPU is a different measurement.** `19_rerank_eval.py` will run
   on the A5000 with the existing 3060 checkpoint, but fp16 kernels differ; if you report those
   numbers, say which machine produced them.

## 6. The experiment that is now possible (and was not before)

> **Status 2026-11-12: done.** Full fine-tuning was run on the A5000 and produced no
> measurable gain (val 0.3930 against 0.3904 for the partial model re-measured in the same
> process; dev Δ +0.004, 95% CI [−0.004, +0.012]). See `docs/COMPAT_FULL_FINETUNE.md`.
>
> **Status 2026-11-13: path A and path B are both done.** A second held-out set was built
> (45 questions / 45 groups), pre-registered, locked and evaluated once: primary
> structural-minus-semantic **+0.800 [0.666, 0.911]** (confirmed), secondary C3 − C1
> **+0.052 [0.020, 0.089]** (positive, magnitude unresolved). See
> `results/test2/OUTCOME.json` and §5.1 of `docs/PROJECT_REPORT.docx`. Path C was rejected:
> full fine-tuning bought no gain, so promoting the A5000 would spend the second test use for
> nothing.

Full fine-tuning of the reranker, without freezing layers. On the 6 GB card this paged to
system memory and took ~20 min per epoch; the A5000 should make it minutes.

```powershell
.\.venv-gpu\Scripts\python.exe scripts\18_train_reranker.py --freeze-encoder-layers 0 `
    --optimizer adamw --batch 32 --grad-accum 1 --epochs-max 3 --env-role compatibility
```

This writes `COMPAT.json`. Compare its validation number against the frozen 0.388
(`FROZEN.json → val_metrics.macro_complete@10`).

**Then you must choose one of two paths**, because the test split has already been used once:

| path | what it means | cost |
|---|---|---|
| A. dev-only comparison | report full vs partial fine-tuning on the development split only; no new test claim | ~1 h |
| B. new held-out set (`test2`) | build 30–40 questions from the 1,239 usable unused clauses (1,771 unused in total), lock it, evaluate once | ~2 days |
| C. replace the canonical model | retrain on the A5000, promote A5000 to canonical, re-freeze, and re-run test | a second test use; must be disclosed |

Path A is the safe default; path B is the only one that produces a new confirmatory number.

## 7. Known open issues

- The test split had been used once as a pilot before the formal run, so all test numbers are
  re-evaluations of a seen split. This is disclosed everywhere.
- The reranking gain is +0.047 on test (CI includes zero) against +0.110 on development
  (p = 0.001). The effect is suggestive, not established.
- `resources` has only 7 test questions; domain-level inference there is weak.
- 25.3% of reranker training pairs exceed 512 tokens and are tail-truncated.

## 8. If you want the chat history on the other machine

DSH stores sessions under `~/.dsh/sessions/<bucket>/session-*/`, where `<bucket>` encodes the
workspace path (for example `--C-Users-QJC_Wing-Documents-deepseek-harness-default-workspace--`).
Copying that folder to a matching bucket on the new machine can surface the old session, but
the transcript contains absolute paths from the original machine and they will be stale. The
repository is the reliable handoff; this document exists so a fresh session does not need the
transcript.
