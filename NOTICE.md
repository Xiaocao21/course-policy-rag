# Third-party notices and attribution

This repository contains original code plus third-party software, models and corpus text. The
MIT licence in `LICENSE` covers **only the original code** written for this project; it does not
cover the items below.

## Software libraries

| library | use | licence |
|---|---|---|
| PyTorch | tensor runtime | BSD-3-Clause |
| transformers (Hugging Face) | model loading | Apache-2.0 |
| sentence-transformers | bi-encoder | Apache-2.0 |
| FAISS (`faiss-cpu`) | exact inner-product search | MIT |
| NumPy | arrays/statistics | BSD-3-Clause |
| matplotlib | figures | PSF-based |
| python-docx | report generation | MIT |
| PyYAML | configuration | MIT |
| selectolax | HTML parsing | MIT |
| pdfplumber / pypdf | PDF extraction | MIT / BSD-3-Clause |
| requests | HTTP | Apache-2.0 |

No external repository code was copied into this project; the pipeline is original, written for
this study.

## Models

| model | source | licence |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | Hugging Face `BAAI/bge-small-en-v1.5` | MIT |
| `BAAI/bge-reranker-base` | Hugging Face `BAAI/bge-reranker-base` | MIT |

The fine-tuned reranker checkpoint (`547060da…`) is a derivative of `BAAI/bge-reranker-base`
and is distributed under the same terms, as a release asset of this repository.

## Corpus text

The corpus is 36 publicly accessible university policy pages collected in October 2026. Every
document has its source URL, retrieval time and content hash in `data/REQUIRED_SOURCES.csv`.

| status | documents | basis |
|---|---|---|
| CC BY-NC-SA 4.0 | 16 | licence recorded at collection (`sources.csv → license_note`) |
| public institutional policy text, academic-research use | 20 | recorded at collection; **not individually verified for redistribution** |

The corpus snapshot and the questions' `quote_evidence` fields contain short excerpts of these
pages for verification. Each question's source is listed in
`annotations/QUOTE_PROVENANCE.csv`. If you are a rights holder and want an excerpt removed,
open an issue in this repository or contact the author through the GitHub profile; the affected
question will be withdrawn and the statistics recomputed.

`data/REQUIRED_SOURCES.csv` records the per-document licence note exactly as observed at
collection; it is not a legal determination.
