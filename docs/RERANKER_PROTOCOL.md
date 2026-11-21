# Reranker protocol (stage 1 and stage 2)

This note fixes the decisions that must not drift between now and the final evaluation.
It covers why one reranker is trained for all seventeen chunking configurations, what a
training label means, how the model is selected, and which environment counts.

## 1. Two-stage retrieval

```
question ──► frozen bge-small-en-v1.5 ──► per-configuration FAISS index ──► dense top-50
                                                                              │
                                                                              ▼
                                                              bge-reranker-base (one model)
                                                                              │
                                                                              ▼
                                                     reranked list ──► admit complete chunks
                                                                       in rank order, stop at the
                                                                       first chunk that does not
                                                                       fit the remaining budget
```

The bi-encoder is **frozen** and identical across all configurations, so the comparison
isolates chunking. The reranker rescores `(question, chunk)` pairs and therefore needs no
new vectors and no index rebuild.

## 2. Why exactly one reranker

Seventeen rerankers would confound the chunking effect with the model effect: a
configuration could win because its reranker happened to fit its own chunk granularity.
One shared model keeps the independent variable singular. The cost is that the reranker
sees a mixed distribution of chunk sizes, which is the intended regularisation.

## 3. Labels: clause-level only

| label | judged on | criterion | used for |
|---|---|---|---|
| `positive_clause` | one chunk | its evidence intervals fully contain at least one annotated clause span of **any** valid minimal sufficient set | training positive |
| `partial` | one chunk | overlaps a required span without containing it | **excluded** from binary training, retained for analysis |
| `negative` | one chunk | dense-ranked but neither of the above | training negative |
| `complete_evidence` | the retrieved **set** | one whole sufficient set is covered | final metric only, never a label |

A clause-level positive is *necessary but not sufficient* for answering the question: the
reranker never sees the rest of the ranking, so set-level completeness is an emergent
property of the ordering rather than something the loss can optimise directly.

Alternative sufficient sets are handled naturally: covering any clause of any valid set is
a positive, because the annotation declares several acceptable evidence sets.

## 4. Candidate identity and deduplication

```
key = (qid, doc_id, corpus_version, char_start, char_end, sha1(text))
```

Offsets are part of the key. Two chunks with identical normalised text but different
boundaries cover different evidence, so merging them would corrupt the labels. When the
same identity appears in several configurations the rows are merged, but
`config_ids`, `dense_rank_by_config`, `source_offsets` (the original `ev_ranges`) and
`merged_from_chunk_ids` are all retained so a later analysis can ask where a chunk came
from and how each configuration ranked it.

## 5. Model selection (the part that is easy to get wrong)

Reranking a **fixed** top-10 pool cannot change Recall@10, Hit@10 or Complete@10: the
retrieved set is the same set, only its order differs. A selection metric built on that
pool would be a constant and would select nothing.

Selection therefore works on the **top-50** pool:

```
for each configuration c, each validation question q:
    rerank the 50 dense candidates, take the top 10 of the reordered list
    complete(c, q) = 1 iff one whole sufficient set is covered by that top-10

S(θ) = (1/17) · Σ_c mean_q complete(c, q)
```

`S(θ)` is a **macro average over configurations**, so a single large configuration cannot
dominate the decision. Early stopping monitors `S(θ)` with patience 1.

Pre-registered tie-breaking, applied in this order:

1. macro reranked `Complete@10`
2. macro reranked `MRR`
3. macro reranked `Recall@10`
4. smaller learning rate
5. fewer negatives per positive

The top-10 pool is still useful, but only for order-sensitive quantities: `@1/@3/@5`,
`MRR`, and the token-budget metrics (admission stops at the first chunk that does not fit,
so order matters there too). Its `@10` set-level numbers are expected to be identical to
the dense baseline, and will be labelled that way rather than reported as an improvement.

## 6. Fixed hyper-parameters and truncation

| item | value | reason |
|---|---|---|
| model | `BAAI/bge-reranker-base` | 512 position embeddings, fits 6 GB in fp16 |
| tokenizer | the same repository | never mixed with the bi-encoder tokenizer |
| question cap | 128 tokens | intent sits early in a question |
| pair cap | 512 tokens | model limit |
| chunk rule | `512 - len(question) - 3`, tail cut | deterministic and reproducible |
| dense candidates | 50 | reranking needs a pool larger than the reported k |
| objective | `BCEWithLogits` on a single logit | stable at this data size; pairwise margin ranking kept as a documented alternative |
| lr | 1e-5, 2e-5 | selected on validation |
| negatives per positive | 4, 8 | selected on validation |
| batch / accumulation | 8 / 2 on 6 GB | OOM fallback to 4 / 4 |
| precision | fp16 | canonical |
| seed | 20261013 | from `settings.yaml` |

Truncation is unavoidable: corpus chunks reach 509 tokens and the model accepts 512, so a
question can never be paired with a full longest chunk. The measured rate on the built
data is in `models/reranker/data_stats.json` (`truncation`), with a per-pair
`truncated_tokens` field so a sensitivity analysis on the untruncated subset remains
possible.

## 7. Canonical environment

Canonical training environment: **RTX 3060 Laptop 6 GB, CUDA, fp16, batch 8, grad
accumulation 2**.

Any other GPU (including the lab A5000) runs in compatibility mode: it writes
`COMPAT.json` instead of `FROZEN.json`, and its numbers must never be compared with
canonical numbers. fp16 kernels and different parallelism produce different last digits,
and mixing them would silently break reproducibility.

`FROZEN.json` records model and tokenizer ids, checkpoint and data hashes, the code
snapshot, corpus version, GPU model and memory, CUDA and PyTorch versions, precision,
batch, accumulation, seed, the full training configuration, the selection protocol and the
validation metrics.

## 8. What is deferred to stage 3

* Four-track evaluation on the locked test split, executed **once** after freezing:
  main (dense top-10), C1 (dense top-50), C2 (reranked top-10), C3 (reranked top-50).
* Bootstrap confidence intervals, paired comparisons and the per-document macro average.
* Spearman correlation between the main-track and C3 configuration rankings, which is the
  direct answer to "does reranking change the chunking conclusion".
* Sensitivity analysis on the untruncated subset if the truncation rate stays high.
