# Reproduce the reported results

Three levels, deliberately separated:

| level | claim | needs |
|---|---|---|
| **R1 statistical recomputation** | every statistic in the paper, recomputed from committed per-question results | CPU + corpus snapshot |
| **R2 retrieval replay** | rerun the retrieval/reranking and get the same per-question results | GPU + checkpoint + official indexes + corpus |
| **R3 retraining** | retrain the reranker as a method check (not an exact replay) | GPU |

## 0. Install

```powershell
git clone https://github.com/Xiaocao21/course-policy-rag.git
cd course-policy-rag
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 1. Acquire corpus, indexes and checkpoint

```powershell
$rel = "https://github.com/Xiaocao21/course-policy-rag/releases/download/v0.1.5-repro"
.\.venv\Scripts\python tools\fetch_data.py    --snapshot "$rel/corpus_v0.1.5.tar.gz" --verify
.\.venv\Scripts\python tools\check_data.py   --snapshot
.\.venv\Scripts\python tools\fetch_indexes.py
.\.venv\Scripts\python tools\fetch_indexes.py --check
```

Expected: `snapshot verified: True files: 110 missing: 0 mismatched: 0`,
`snapshot layers: cleaned 108/108, corpus_safe_512 2/2`, `indexes: ok`.

The indexes are **required** for exact replay: rebuilding them changes the bytes and the test2
pre-registration will refuse them.

## 2. R1 — statistical recomputation

```powershell
.\.venv\Scripts\python src\verify.py --split test
.\.venv\Scripts\python src\verify.py --split test2
```

Expected for both: `stats PASS (diffs: 0)`, `official hashes unchanged: True`; test2 also
`outcome PASS (diffs: 0)`. Without the corpus the command refuses and tells you to run step 1,
because `by_length_tercile` needs clause texts.

## 3. R2 — retrieval replay

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python -m pip install -r requirements-gpu.txt
.\.venv-gpu\Scripts\python tools\fetch_models.py --download --base
.\.venv-gpu\Scripts\python src\evaluate.py --config configs\proposed_model.yaml
```

Expected: `per-question identical: 3060/3060 (different 0)`, `official hashes unchanged: True`,
and `results/repro/proposed_model_test2/` containing `REPRO_RUN.json`, `perq_*.jsonl`,
`summary_*.csv`, `compare.json`, `run_meta.json`. There is deliberately **no**
`FINAL_RUN.json`.

The run uses a temporary staging root, so `results/final*`, `results/test2/OUTCOME.json` and
`experiments/log.jsonl` are untouched and checked by hash before and after.

Diagnostic mode (`--diagnostic`, or automatically when `data/STATUS.json` is not
`reproduction_grade`) writes to the system temp directory, prefixes `diag_`, adds
`NONCOMPARABLE.json`, and can never enter a result table.

## 4. R3 — retraining (optional, not required to verify numbers)

```powershell
.\.venv-gpu\Scripts\python scripts\17_build_rerank_data.py
.\.venv-gpu\Scripts\python scripts\18_train_reranker.py --env-role compatibility ^
    --freeze-encoder-layers 0 --optimizer adamw --batch 32 --seed 20261027
```

Run it in a scratch copy: `18` writes `FROZEN.json`/`COMPAT.json` under `models/reranker/`.
Log-derived timing: 2 lr candidates × 2 epochs, 63.7–64.0 s train + 32.9 s validation per
epoch (≈6.5 min compute, excluding pair construction). Retraining is a method check; it does
not reproduce the exact reported numbers.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `REFUSING: indexes ... hash mismatch` | indexes rebuilt locally | `python tools/fetch_indexes.py` |
| R1 refuses "corpus snapshot" | `cleaned/` absent | step 1 |
| many `by_length_tercile` diffs | ran R1 before the corpus | step 1, then rerun |
| `ModuleNotFoundError: torch` | CPU venv used for R2 | use `.venv-gpu` |
| pre-registration complains about `TEST2_REVIEW.csv` | stale clone | `git pull` (the file is committed) |
