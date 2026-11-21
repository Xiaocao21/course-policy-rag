# Data: course-policy corpus v0.1.5

## What it is

36 public policy documents (14 course / 9 faculty / 13 institution, 12 institutions, 1,988
clauses) collected in October 2026. The 36 documents are the ones used by the 172-question
first set and the 45-question second set; they were selected from 265 collected documents by
`reports/curation_decisions.csv`.

`data/REQUIRED_SOURCES.csv` is the **frozen** list of those 36 documents: `doc_id`, tier,
institution, `seed_id`, `final_url`, byte size, `sha256_raw`, HTTP validators, robots status
and the recorded licence note. Its SHA-256 is in `data/REQUIRED_SOURCES.sha256`.

## Two ways to obtain the corpus

### A. Snapshot (used for retrieval replay)

The live sites have changed since collection, so the snapshot is the only way to replay
retrieval under the original conditions. It contains `cleaned/` (36 docs × evidence/retrieval/
charmap) and `corpus_safe_512/`, byte-identical to `MANIFEST.json`, and **no other documents**.

```text
python tools/fetch_data.py --snapshot https://github.com/Xiaocao21/course-policy-rag/releases/download/v0.1.5-repro/corpus_v0.1.5.tar.gz --verify
python tools/check_data.py --snapshot            # writes data/STATUS.json
```

The archive and its SHA-256 are published with the reproduction release
(`corpus_v0.1.5.tar.gz`); `SNAPSHOT_MANIFEST.json` inside lists every file with its hash and
source URL. `fetch_data.py` refuses any archive whose files do not match `MANIFEST.json`.

**Redistribution note.** The snapshot is provided by the author for academic verification.
16 documents carry CC BY-NC-SA 4.0; the other 20 record "public institutional policy text;
academic research use" and have **not** been individually verified for redistribution
(`annotations/QUOTE_PROVENANCE.csv` marks this per question). If you are a rights holder and
want a document removed, see `NOTICE.md`.

### B. Live rebuild (measured; currently fails)

`python tools/check_data.py --require --work <dir>` after running the collector is the
supported path, but on 2026-11-21 the rebuild produced:

| check | result |
|---|---|
| required documents matched (`sha256_raw`) | **1 / 36** |
| unavailable | **17 / 36** |
| content changed | **18 / 36** |

The detail is in `data/REBUILD_REPORT.md`. Because the gate fails, a live rebuild must not be
used for retrieval replay; it is reported as a limitation, not as a reproduction.

## Expected directory tree after a snapshot install

```text
cleaned/
  <doc_id>.evidence.jsonl     # character-faithful layer; all gold spans anchor here
  <doc_id>.retrieval.jsonl    # normalised retrieval layer
  <doc_id>.charmap.jsonl      # mapping between the two layers
corpus_safe_512/
  corpus_safe_512.jsonl       # 2,007 chunks, 512-token cap
  chunk_index.jsonl
```

`raw/`, `extracted/`, `indexes/` and `chunks/` are **not** shipped: `raw/` is the crawl output,
`extracted/` and `chunks/` are deterministic pipeline products, and `indexes/` is rebuilt by
`src/preprocess.py` from the corpus.

## Provenance and licensing

Every document has a `url`, `retrieved_at`, `etag`, `last_modified` and `sha256_raw` in
`data/REQUIRED_SOURCES.csv`; the frozen corpus hashes are in `MANIFEST.json`. Licence status
per document, robots decisions and refused sources are in `sources.csv`, `COMPLIANCE.md` and
`reports/robots_audit.json`. The questions quote only the evidence clauses they cite
(`quote_evidence`); `annotations/README.md` explains that scope.
