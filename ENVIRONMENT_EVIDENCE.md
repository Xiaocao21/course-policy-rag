# Environment evidence for the reported results

This file records what is **proven** by the artefacts, what is only inferred, and what is
unknown. `same_env` in the reproduction comparison may be `true` only when every required
field below has evidence and matches.

## Official runs

| run | split | device | recorded environment fields | evidence | unknown |
|---|---|---|---|---|---|
| E02–E05 (2026-10-28) | test | RTX 3060 Laptop 6 GB | `torch 2.6.0+cu124`, `cuda 12.4`, `precision fp16`, `batch 8 × accum 2` | `FROZEN.env` — this is the **training** environment recorded on 2026-10-25; `FINAL_RUN.json` (test) has no `inputs` block | evaluation-process Python, transformers, sentence-transformers, faiss, numpy, OMP/BLAS threads, faiss threads, determinism flags, eval batch |
| test2 (2026-11-13) | test2 | RTX A5000 24 GB | `torch 2.6.0+cu124`, `cuda 12.4`, `precision fp16`, `gpu`, `evaluation_host` | `results/final_test2/FINAL_RUN.json → inputs` | Python, transformers, sentence-transformers, faiss, numpy, OMP/BLAS threads, faiss threads, determinism flags, eval batch |

`experiments/env.json` was captured on 2026-10-25 (CPU: torch 2.14.1+cpu, transformers 5.20.2,
sentence-transformers 6.1.0, faiss 1.15.1; GPU: torch 2.6.0+cu124, transformers 5.20.2,
sentence-transformers 6.2.0). It is a **point-in-time snapshot**, not a record of the
2026-10-28 or 2026-11-13 evaluation processes, and the two runs may have used different
package versions. It must not be used to infer them.

## Consequence for comparison

* `test` and `test2` also used **different code versions** — `FINAL_RUN.code_sha256` is
  `6f0bcff6…` for test and `da6a039b…` for test2 (training recorded `cefb847e…`). The
  evaluation code advanced between the runs (22–28 were added, 19 gained input hashing), so
  test-vs-test2 differences cannot be attributed to the split alone. `results/artifact_map.csv`
  carries the code hash per row.
* For a reproduction run, `src/repro/compare.py` writes the **current** environment to
  `repro/env.json`; because the historical process environment is unknown, `same_env` will be
  `unknown` unless a future run records all fields. In that case the comparison reports deltas
  and whether the pre-registered verdict is unchanged; it does not require per-question
  equality.

## Required fields for a future `same_env = true`

`python`, `torch`, `cuda`, `precision`, `gpu`, `transformers`, `sentence-transformers`,
`faiss`, `numpy`, `OMP_NUM_THREADS`, `faiss.omp_get_max_threads()`,
`torch.backends.cudnn.deterministic`, `torch.backends.cudnn.benchmark`,
`torch.get_num_threads()`, `eval_batch`, `device`.
