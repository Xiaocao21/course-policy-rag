# Pilot (exploratory) test results - NOT the final result

Archived 2026-10-25T00:28:58+08:00 from `results/`.

This is the first evaluation of the locked test split: frozen `BAAI/bge-small-en-v1.5`,
dense retrieval with a top-10 candidate pool, no reranker, 17 chunking configurations.

It is kept for provenance and to document what was seen before the reranker work. The formal
result will be a single four-track evaluation executed once after `FROZEN.json` exists:

| track | retrieval | reranker |
|---|---|---|
| main | dense top-10 | no |
| C1 | dense top-50 | no |
| C2 | dense top-10 | yes |
| C3 | dense top-50 | yes |

Because the test split must be used once, this pilot is not reported as a confirmatory result.

## Files

- `summary_test.csv`  sha256 `eb600176596750b2...`
- `perq_test.jsonl`  sha256 `4746af9a5359b435...`
- `run_test.json`  sha256 `24d30c8ed40bbd74...`
- `metrics_fixed_size128_ov0.json`  sha256 `a7a06068e04b5c6c...`
- `metrics_fixed_size128_ov32.json`  sha256 `ae87d2de023e8833...`
- `metrics_fixed_size128_ov64.json`  sha256 `96f4df74352d9416...`
- `metrics_fixed_size256_ov0.json`  sha256 `c5e981690d2d3125...`
- `metrics_fixed_size256_ov32.json`  sha256 `7f789176e755a836...`
- `metrics_fixed_size256_ov64.json`  sha256 `306b6054f3a78ace...`
- `metrics_fixed_size384_ov0.json`  sha256 `4fe4386b5ea6af35...`
- `metrics_fixed_size384_ov32.json`  sha256 `eb0c69fc334438dc...`
- `metrics_fixed_size384_ov64.json`  sha256 `2cbedcdbe693e63f...`
- `metrics_fixed_size512_ov0.json`  sha256 `a3ab2467af23e799...`
- `metrics_fixed_size512_ov32.json`  sha256 `a4493f4e344bf701...`
- `metrics_fixed_size512_ov64.json`  sha256 `a9d1aa2f33e2955e...`
- `metrics_heading_heading.json`  sha256 `0cfb2c61fea54009...`
- `metrics_paragraph_paragraph.json`  sha256 `8eaef60bcda79a80...`
- `metrics_semantic_thr0.70.json`  sha256 `4cd2d347bd806124...`
- `metrics_semantic_thr0.75.json`  sha256 `ceda46c6e0429652...`
- `metrics_semantic_thr0.80.json`  sha256 `da73e81cc53f27d2...`
