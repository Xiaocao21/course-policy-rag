# Acceptance record

Measured on 2026-11-22 on Windows 11 (Python 3.12.14, torch 2.6.0+cu124, CUDA 12.4,
NVIDIA RTX A5000 24 GB).

## Clean clone

| step | command | result |
|---|---|---|
| clone + CPU env | `git clone …`, `pip install -r requirements.txt` | ok |
| R1 test | `python src/verify.py --split test` | `stats PASS (diffs: 0)`, official hashes unchanged |
| R1 test2 | `python src/verify.py --split test2` | `stats PASS`, `outcome PASS`, official hashes unchanged |
| snapshot | `tools/fetch_data.py --snapshot … --verify` | 110 files, 0 missing, 0 mismatched |
| gate | `tools/check_data.py --snapshot` | `cleaned 108/108`, `corpus_safe_512 2/2` |
| indexes | `tools/fetch_indexes.py` | verification ok (34 files) |
| models | `tools/fetch_models.py --download` | `assets: 6/6 ok` |
| R2 | `python src/evaluate.py --config configs/proposed_model.yaml` | `per-question identical: 3060/3060`, official hashes unchanged |

## Isolation checks

* R2 writes `REPRO_RUN.json`, never `FINAL_RUN.json`; official `results/final*`,
  `results/test2/OUTCOME.json` and `experiments/log.jsonl` are hashed before/after and unchanged.
* Diagnostic mode writes to the system temp directory with `diag_perq_*` names and
  `NONCOMPARABLE.json`; `perq_*.jsonl` count is 0, so `16_stats` cannot consume it.
* A fresh output directory is accepted; an output inside `results/final*`/`results/test2` is
  refused (`tests/test_repro_guard.py`).

## Repository checks

| check | value |
|---|---|
| tracked files | 511 |
| tracked size | 26 MB |
| largest tracked file | 1.3 MB (`results/compat_dev/.../perq_C3.jsonl`) |
| unit tests | 174 OK |
| secrets scan | 0 hits (policy text matches were reviewed and are not credentials) |
| public release | HTTP 200 without signing in; 11 assets; hashes match `results/HASHES.csv` |
| question file bytes | `sha256(annotations/questions.jsonl)` equals `LOCKED.json` |

## What was **not** verified

* Full R3 retraining was not executed end-to-end in the clean clone; `src/train.py --dry-run`
  and the module's staging logic were checked, and the A5000 recipe is documented. The reported
  numbers never depend on R3.
* Linux was not tested (Windows 11 only).
* Live corpus rebuild was tested and **failed** (1/36 documents matched); see
  `data/REBUILD_REPORT.md`. The snapshot is the supported path.
* The 1.06 GB checkpoint download was verified by hash, not by a second independent network.
