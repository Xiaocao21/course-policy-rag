# course-policy-rag — chunking × reranking for evidence-complete retrieval

Code, frozen data packages and result files for the CPS5801 study *How do chunking strategies
affect retrieval accuracy, evidence completeness and efficiency in a RAG system for university
course policies?*

**Reported result.** Structural chunking (heading / paragraph) is far better than sentence-level
semantic chunking at returning a whole policy clause: on the second held-out set the
structural-minus-semantic gap in Complete@10 is **+0.800, 95% CI [0.666, 0.911]**. Reranking
adds **+0.052 [0.020, 0.089]** there (positive, magnitude unresolved); on the first test split
it was **+0.047 [−0.005, +0.095]** (not significant).

## Research question

Does the chunking strategy change retrieval accuracy and evidence completeness on course-policy
text, and does a fine-tuned cross-encoder change that ranking? The main comparison varies only
chunking (17 configurations, 4 families); reranking is a secondary 2×2 contrast.

## Repository structure

```
├── README.md  REPRODUCE.md  LICENSE  NOTICE.md  AI_DISCLOSURE.md  PROVENANCE.md
├── requirements.txt            # CPU: R1 and the report
├── requirements-gpu.txt        # A5000: torch cu124, sentence-transformers, faiss
├── configs/                    # proposed_model.yaml, baseline.yaml, a5000.yaml
├── data/                       # README, REQUIRED_SOURCES.csv, snapshot instructions
├── src/                        # entry points: verify, evaluate, train, inference, preprocess, utils
├── tools/                      # fetch_data, fetch_indexes, fetch_models, check_data, exporters
├── scripts/                    # the pipeline (01–28): collection → chunking → indexing → stats
├── results/                    # per-question scores, stats.json, OUTCOME.json, figures, HASHES.csv
├── annotations/                # question sets, locks, review attestation, quote provenance
├── docs/                       # PROJECT_REPORT.docx and analysis notes
├── models/                     # FROZEN.json, COMPAT.json, run metadata (weights are release assets)
├── MANIFEST.json               # frozen corpus hashes
└── sources.csv  COMPLIANCE.md  # crawl provenance and licence notes
```

## Hardware

| role | hardware | what ran there |
|---|---|---|
| canonical | NVIDIA RTX 3060 Laptop, 6 GB | corpus/index build; training; the first test evaluation |
| target | **NVIDIA RTX A5000, 24 GB** | full fine-tune (T06), matched-seed lr grid (T07), test2 evaluation |
| CPU only | any | statistical recomputation (R1) and report generation |

The two official runs used the **same frozen checkpoint** (`547060da…`) but different GPUs and
different code versions; test-vs-test2 differences therefore mix split, hardware and code. This
is recorded per row in `results/artifact_map.csv` and in `ENVIRONMENT_EVIDENCE.md`.

## Software and dependency requirements

Python 3.12.14; CPU path: `requirements.txt`. GPU path (A5000):

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python -m pip install -r requirements-gpu.txt
```

Recorded versions: torch 2.6.0+cu124, CUDA 12.4, transformers 5.20.2,
sentence-transformers 6.2.0, faiss 1.15.1, numpy 2.5.4. Windows 11 was verified; Linux is
expected but not tested.

## Installation

```powershell
git clone https://github.com/Xiaocao21/course-policy-rag.git
cd course-policy-rag
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

For R2/R3 (A5000), also create the GPU environment:

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python -m pip install -r requirements-gpu.txt
```

`requirements.txt` gives supported ranges; the exact versions used for the reported results
(torch 2.6.0+cu124, CUDA 12.4, transformers 5.20.2, sentence-transformers 6.2.0, faiss 1.15.1,
numpy 2.5.4) are recorded in `ENVIRONMENT_EVIDENCE.md` and `requirements-gpu.txt`.

## Dataset

36 public university policy documents (14 course / 9 faculty / 13 institution, 12 institutions,
1,988 clauses). Sources, HTTP validators, robots status and recorded licence notes are in
`data/REQUIRED_SOURCES.csv`; the frozen hashes are in `MANIFEST.json`.

* **Retrieval replay** uses the author-provided snapshot from release `v0.1.5-repro`
  (`corpus_v0.1.5.tar.gz`, 3.4 MB, 110 files, byte-identical to `MANIFEST.json`).
* **Live rebuild was measured and fails**: 1 of 36 documents still matches its original hash,
  17 are unavailable, 18 changed (`data/REBUILD_REPORT.md`). It is reported as a limitation,
  not offered as a reproduction path.
* `raw/`, `extracted/`, `chunks/`, `indexes/` and model weights are not in git; they are
  rebuilt or downloaded.

## Data preprocessing

```powershell
python tools/fetch_data.py --snapshot <release-url>/corpus_v0.1.5.tar.gz --verify
python tools/check_data.py --snapshot          # writes data/STATUS.json
python tools/fetch_indexes.py                  # official FAISS indexes (required for exact replay)
```

`src/preprocess.py` rebuilds `chunks/` and `indexes/` from the corpus for exploration. Rebuilt
indexes have different bytes from the official ones (float32/BLAS), so they cannot pass the
test2 pre-registration check.

## Training (optional; not needed to verify reported numbers)

The reranker used for every reported result is already frozen and published, so retraining is
not required. To retrain:

```powershell
python scripts/17_build_rerank_data.py
python scripts/18_train_reranker.py --env-role compatibility --freeze-encoder-layers 0 ^
    --optimizer adamw --batch 32 --seed 20261027
```

Run this in a scratch copy: `18` writes `FROZEN.json`/`COMPAT.json` under `models/reranker/`,
which would overwrite the official metadata. Settings are recorded in `configs/a5000.yaml`.
The A5000 log shows 2 lr candidates × 2 epochs, 63.7–64.0 s train + 32.9 s validation per epoch
(≈6.5 min compute, excluding pair construction). Retraining is a method check and does not
reproduce the exact numbers.

## Evaluation

```powershell
# R1 statistical recomputation (CPU; needs the corpus snapshot)
python src/verify.py --split test
python src/verify.py --split test2

# R2 retrieval replay (GPU; needs checkpoint + official indexes)
python src/evaluate.py --config configs/proposed_model.yaml
```

R1 recomputes every field of `stats.json` with the same script and seed and compares with the
committed official file; test and test2 both reproduce with zero differences, and `OUTCOME.json`
matches except its `created_at`. R2 runs `19_rerank_eval.py` inside a temporary staging root and
compares per-question results; the official results and `experiments/log.jsonl` are not touched.
`--diagnostic` output is written to a temp directory with `diag_` prefixes and
`NONCOMPARABLE.json`, and can never enter a result table.

## Expected results

| track | test Complete@10 [95% CI] | test2 Complete@10 [95% CI] |
|---|---|---|
| main (dense top-10) | 0.358 [0.299, 0.418] | 0.490 [0.422, 0.557] |
| C1 (dense top-50) | 0.358 [0.299, 0.418] | 0.490 [0.422, 0.557] |
| C2 (top-10 + rerank) | 0.358 [0.299, 0.418] | 0.490 [0.422, 0.557] |
| C3 (top-50 + rerank) | 0.405 [0.347, 0.464] | 0.542 [0.481, 0.601] |

C3 − C1: test +0.047 [−0.005, +0.095], Holm p = 0.150; test2 +0.052 [0.020, 0.089],
p = 0.004. test2 primary: structural − semantic +0.800 [0.666, 0.911],
`confirmed_at_least_0.30`; secondary `positive_magnitude_unresolved`.

## Pretrained models and checkpoints

| item | identifier |
|---|---|
| encoder | Hugging Face `BAAI/bge-small-en-v1.5` |
| reranker base | Hugging Face `BAAI/bge-reranker-base` |
| frozen reranker | release asset `reranker-FROZEN-model.safetensors`, sha256 `547060da…` |
| tokenizer / config of the checkpoint | release assets, hashes in `results/HASHES.csv` |
| download | `python tools/fetch_models.py --download --base`; verify with `--check` |

## Reproduction quickstart

```powershell
git clone https://github.com/Xiaocao21/course-policy-rag.git
cd course-policy-rag
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

python tools/fetch_data.py --snapshot https://github.com/Xiaocao21/course-policy-rag/releases/download/v0.1.5-repro/corpus_v0.1.5.tar.gz --verify
python tools/check_data.py --snapshot
python src/verify.py --split test
python src/verify.py --split test2
```

`REPRODUCE.md` covers R1/R2 in full, with expected outputs and failure modes. `LIMITATIONS.md`
lists what this package cannot claim.

## Code provenance

CODE_MANIFEST.csv lists every script with its original and public SHA-256; CODE_DELTA.md
explains the three path-only edits; RESULTS_CODE_SHA256.json shows the eight result-generating
files are byte-identical. The historical FINAL_RUN.json/FROZEN.json were not modified.
ACCEPTANCE.md records what was actually run and what was not.

## Citation and attribution

If you use this code or data, cite the project report (`docs/PROJECT_REPORT.docx`) and this
repository. Third-party libraries and models are listed in `NOTICE.md`; no external repository
code was copied.

## AI use

AI assistance was used for code, question drafting and text; the author reviewed and is
responsible for everything. The first 172 questions were student-reviewed; test2 was jointly
reviewed (attestation in `annotations/TEST2_REVIEW_ATTESTATION.md`). Full details:
`AI_DISCLOSURE.md`.

## Licence

MIT for the original code only; corpus text and third-party components keep their own terms
(`LICENSE`, `NOTICE.md`).
