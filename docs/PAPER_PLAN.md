# Final Conference-Style Paper — Plan (v4)

Status: plan for the 20% final paper. All numbers below are already verified against the
repository; no new experiments are planned.

## 1. Goal and claim boundaries

Write a two-column, 6–8 page (target about **7 pages** of main content; 6–8 is acceptable, references excluded)
conference-style paper on evidence-complete retrieval for course-policy RAG.

**What the paper may claim**

> Under a fixed corpus, a frozen encoder, a fixed set of 17 chunking configurations and one
> shared reranker, the tested structure-aware configurations outperformed the tested semantic
> configurations on this course-policy corpus under the stated evaluation protocol.

**What it must not claim**

* that union-based completeness is a new metric (position it against prior evidence-coverage work);
* that the result generalises beyond this corpus/protocol;
* that the reranker effect is established on the first test split (it is not);
* that test2 validates multi-clause completeness (it contains only single-clause questions);
* "proves", "state of the art", or any statement stronger than the intervals support.

**Three contributions**

1. **Evaluation design** — gold evidence annotated on the original text (independent of any
   chunking), interval-union merging across chunks, and separate evidence-recall / partial /
   Complete@10 reporting, with MRR kept as a single-chunk ordering metric.
2. **Controlled comparison** — 17 configurations × four tracks (pool 10/50 × reranker), which
   separates "larger candidate pool" from "reranking".
3. **Empirical findings** — the structural-versus-semantic difference on this corpus and a
   pre-registered supplementary check on a second, single-clause question set.

## 2. Structure and page budget

| Section | Pages | Content |
|---|---|---|
| Title + Abstract + 1 Introduction | 1.2 | motivation, problem, gap, RQs, contributions; a clearly labelled fictitious late-submission rule+exception example |
| 2 Related Work | 0.7 | two core chunking papers → chunking comparisons → evidence completeness / structured QA → RAG and retrieval foundations |
| 3 Methodology | 1.3 | corpus and annotation; 17 configurations; metric definitions; reranker |
| 4 Experimental Setup | 0.8 | splits, baselines, hyperparameters, seeds, software, hardware, statistical protocol |
| 5 Results | 1.3 | Table 1 (four tracks, test); Table 2 (test2 endpoints); token-budget analysis; per-family table |
| 6 Analysis + 7 Limitations + 8 Conclusion + AI Disclosure | 1.7 | mechanism, alternative explanations, limitations, conclusion, disclosure |
| References | separate | 15–18 verified entries |
| Appendix (optional) | separate | reproduction commands, artifact map |

Tables/figures are de-duplicated: Table 1 replaces a four-track bar chart; ρ = 0.963 is one
sentence, not a figure.

## 3. Dataset and test2 scope

* Corpus v0.1.5: 36 documents (14 course / 9 faculty / 13 institution, 12 institutions),
  1,988 clauses; evidence spans anchored on the original text.
* First set: 172 questions (112 dev / 60 test, 59 test groups); evidence types include
  single-clause, rule-and-exception and cross-clause.
* Second set (test2): **45 questions / 45 groups, all single-clause**.
* Timeline for test2: answer revisions and composition amendments recorded 2026-11-13 01:45;
  lock 01:56; the single official evaluation 02:01. The joint student–AI review attestation is
  dated **2026-11-21, after the evaluation**; no locked question was changed. The machine
  record at lock time says `agent_self_audit`; the attestation is a later addition.

**test2 supports**: family differences on single-clause questions, retrieval of **complete gold
evidence for single-clause questions**, and whether a reranking gain appears under that
composition.

**test2 cannot support**: whether all evidence for multi-clause questions is retrieved.
Limitations must say this explicitly.

Use the phrase **"pre-registered supplementary validation on a single-clause question set"**,
not "confirmatory replication of the method".

## 3b. Implementation facts that must appear in the main text

* **Reranker**: `BAAI/bge-reranker-base`, an XLM-RoBERTa-base cross-encoder (12 layers, hidden
  size 768), trained with `BCEWithLogitsLoss` on query–chunk pairs.
* **Dev split**: 89 train / 22 validation questions, 85 / 21 groups, group-disjoint.
* **Hard negatives**: drawn from the top-50 pools of the 17 configurations, four negatives per
  positive; partial-overlap pairs are excluded from the loss (1,812 partials).
* **First test split**: used once as a pilot before the reranker stage, so its results are
  exploratory re-evaluations of a seen split, not a pristine holdout.
* **test2 composition**: 45 questions / 45 groups; integrity-heavy (25/45); Glasgow 15/45 (33%);
  all single-clause. These are recorded pre-evaluation design amendments.
* **Hardware roles**: canonical results (training, first test run) on the RTX 3060 6 GB; A5000
  24 GB for T06, T07 and the single test2 run. Do not compare their numbers as if the hardware
  were the same.
* **Scope**: the study evaluates retrieval of evidence completeness; it does **not** measure
  answer-generation quality.

## 4. Metric definitions (must match `scripts/13_evaluate.py`)

1. **Gold spans**: `(doc_id, char_start, char_end)` on the original evidence layer; a question may
   provide several alternative sufficient evidence sets.
2. **Interval union**: for each document, merge all `ev_ranges` of retrieved chunks; overlapping
   chunks do not double-count; a span is covered only if the union contains it entirely.
3. **Hit@k**: the union of the first k chunks fully covers **at least one** required gold clause
   (from any sufficient set). It is not "one chunk covers a clause".
4. **Evidence recall@k**: within a sufficient set, the fraction of required clauses fully covered;
   the maximum over sufficient sets is reported.
5. **Complete@k**: binary; some sufficient set is fully covered by the union.
6. **Partial coverage**: per-span covered-character fraction, averaged over all spans of all
   sufficient sets; kept separate from recall (which is per-clause and takes the best set).
7. **MRR**: reciprocal rank of the **first single chunk** whose `ev_ranges` fully cover any
   required clause; partial overlap does not count; missing = 0.
8. **Relationship**: Complete/Hit are cumulative-set metrics; MRR is a single-position metric.
   The paper must say they measure different objects.

## 5. Aggregation order (`scripts/16_stats.py`)

1. compute each metric per question;
2. average questions equally **within a group**;
3. average configurations equally within the group (17 overall; the family's configurations for
   family results);
4. paired bootstrap over groups (B = 2000, seed 20261013, shared resampling indices), percentile
   2.5/97.5.

Consequences to state: questions are equally weighted within a group, groups are equally
weighted in the bootstrap, and configurations are equally weighted; this is **not** equal
weighting over all questions globally.

The secondary endpoint C3 − C1 averages the **same 17 configurations** in each track, per group,
then pairs groups.

## 6. Units

Methods statement:

> Metric values are proportions in [0, 1], while differences between metrics lie in [−1, 1].
> Confidence intervals use the same proportion scale. For readability, a difference of 0.800
> corresponds to 80.0 percentage points; where percentage points are used, the interval is
> converted to the same scale.

Rules:

* tables and abstract use proportions (`0.405`, `+0.047`, `[−0.005, +0.095]`);
* percentage points only for a standalone magnitude, with a pp interval
  (`80.0 pp, 95% CI [66.6, 91.1] pp`);
* never put `80.0 pp` next to a proportion-scale interval.

## 7. Token-budget analysis (main control, not appendix)

Report in Results, as a compact table or figure:

* tokenizer `BAAI/bge-small-en-v1.5` (same as indexing);
* budget = sum of each admitted chunk's `token_count` (content tokens, no query/special tokens);
* chunks are admitted in rank order as **complete** chunks; stop at the first chunk that does not
  fit — no truncation, no skipping;
* therefore **overlapping chunks double-charge their shared text**; "chunk IDs are unique" is not
  "text is de-duplicated";
* the budget metrics score the **complete admitted chunks** (all their evidence ranges), not the
  text the reranker actually read; the reranker input is query ≤128 + chunk tail-truncated to a
  512-token pair, so some evidence never reached the reranker;
* report 512/1024/2048 recall and complete, plus `tokens_unused_mean` and `zero_chunk_queries`
  where available; do not invent missing fields.

## 8. Table A — 17 configurations (must count to 17)

| family | configurations | count |
|---|---|---|
| fixed | `size ∈ {128, 256, 384, 512}` × `overlap ∈ {0, 32, 64}`; all 12 combinations are valid (overlap < size), step = size − overlap | **12** |
| paragraph | merge corpus units at paragraph boundaries; over-long units split further | 1 |
| heading | merge at heading boundaries; over-long units split further | 1 |
| semantic | sentence-embedding cosine thresholds `0.70 / 0.75 / 0.80`; merge adjacent sentences above threshold; over-long units split further | 3 |
| **total** | | **17** |

Two distinct limits, both 512 tokens but conceptually different:

* **chunk length cap = 512 tokens** — pipeline `safe_corpus.max_input_tokens`; any family's
  over-long unit is split further;
* **encoder max input = 512 tokens** — `bge-small-en-v1.5` `model_max_length`.

The fixed sizes 128–512 are **target window sizes**, not encoder limits.

## 9. Table B — training and model selection

| item | formal partial checkpoint | T06 full FT (compatibility) | T07 lr grid (compatibility) |
|---|---|---|---|
| frozen encoder layers | 10 / 12 | 0 | 10 / 12 |
| optimizer | Adafactor | AdamW | Adafactor |
| lr | 1e-5 | 1e-5 | {1e-5, 2e-5} |
| negatives / positive | 4 | 4 | 4 |
| batch × accum | 8 × 2 | 32 × 1 | 8 × 2 |
| epochs / early stop | ≤3 / patience 1 | ≤3 / patience 1 | ≤3 |
| selection metric | macro reranked Complete@10 (17 configs, top-50 val pool) | same | per-run monitoring; not used for checkpoint selection |
| epoch choice within run | strict improvement; ties keep the **earlier** epoch; reports best val epoch | same | same; each run reports its best validation epoch, not the last |
| across-run tie-break | Complete@10 → MRR → Recall@10 → smaller lr → fewer negatives | same | paired within seed; 1e-5 vs 2e-5 |
| train / val groups | 85 / 21 | 85 / 21 | 85 / 21 |
| seed | 20261027 | 20261027 | 20261027 / 47 / 57 |
| truncation | query ≤128, pair ≤512, chunk tail cut | same | same |
| role | **main result** | compatibility only | compatibility only |

T07 values: 1e-5 mean 0.3824; 2e-5 mean 0.3610; per-seed paired differences (2e-5 − 1e-5)
−0.0241 / −0.0267 / −0.0134.

**Three validation numbers, three different measurements** (never merge them):

| value | source and conditions |
|---|---|
| 0.3877 | `FROZEN.val_metrics` — the selected checkpoint's validation score recorded during canonical training on the RTX 3060 |
| 0.3904 | the **same partial checkpoint** re-measured in the T06 process on the A5000 (`COMPAT_FULL_FINETUNE.md`); a different device/process, not the selection score |
| 0.3930 | T06 full fine-tune (0 frozen layers, AdamW, batch 32), compatibility result; not the paper's checkpoint |

Only 0.3877 belongs to the main model. The other two go to the discussion/appendix as
compatibility evidence.

Unknown software versions are marked `unknown`; never inferred from the current environment.

## 10. Statistics: pre-registered decisions vs supplementary p-values

**First test split (exploratory):**

* Holm correction over {C3−C1, C3−C2} for Complete@10; ordering metrics corrected separately by
  family;
* report Δ, 95% CI, `p_holm`; C3−C1 is +0.047 [−0.005, +0.095], `p_holm` = 0.150.

**test2 (pre-registered):**

* fixed-sequence gatekeeping: primary endpoint first — C1 structural (heading+paragraph, 2
  configs) minus semantic (thr0.70/0.75/0.80, 3 configs), confirmation requires **CI lower
  bound > 0.30** (strict inequality);
* the secondary endpoint C3−C1 is interpreted only if the primary confirms. Once the primary
  passes, a **positive gain** is supported when the secondary **CI lower bound > 0**; the
  magnitude comparisons are strictly the frozen ones (CI lower > 0.06 for "exceeds 0.06"; the
  numerical comparison to the first split's +0.047 is descriptive only). The paper must not
  invent a new threshold or reinterpret zero as a magnitude claim;
* the reported `p_holm = 0.004` is an **additional Holm-adjusted p** over {C3−C1, C3−C2}; it is a
  supplementary statistic, **not** the pre-registered decision rule. The paper must keep the two
  separate and must not imply that p = 0.004 is what confirmed the result.

**Abstract** reports effect sizes and CIs only (no p-values):

> +0.800, 95% CI [0.666, 0.911] for structural minus semantic on C1; +0.052, 95% CI
> [0.020, 0.089] for C3 − C1.

## 11. Related work and references

Priority order:

1. **Khalib et al. 2026** (IJACSA, structured institutional documents) and
   **Moreno-Cediel et al. 2026** (Knowledge-Based Systems, growing-window semantic chunking) —
   the two core papers; compare task, variable, metrics and gap in a short table or paragraph.
2. Direct chunking comparisons: Fadillah 2024 (regulation documents), Gomez-Cabello 2025
   (clinical), Kiss 2025 (Max–Min semantic), Mazzitelli & Zimeo 2025 (paragraph-based).
3. Evidence completeness / structured QA: UDA 2024, TableRAG 2025, RAGAs 2024.
4. Retrieval foundations: Lewis 2020 (RAG), Karpukhin 2020 (DPR), Nogueira & Cho 2019,
   Thakur 2021 (BEIR), Reimers & Gurevych 2019.

Constraints:

* 10 references are already student-verified (the literature review); target **15–18** total;
* new entries are added only after the student opens and verifies them; record DOI/URL and date in
  `docs/REFERENCES_VERIFIED.md`;
* ColBERT is a multi-vector retrieval method, not a cross-encoder reranker;
* BGE-M3 does not replace the actually used `bge-small-en-v1.5` and `bge-reranker-base`; cite the
  respective model cards/reports;
* technical reports supplement, but recent peer-reviewed work is the main body.

## 12. Analysis and discussion wording

* Section title: **Boundary fragmentation and incomplete evidence retrieval**.
* The union audit uses the **test split, track C1** (`main` is identical for set-level metrics);
  C3 is not audited because reranking a test pool would be a second test use. The denominator is
  the set of **Complete@10-successful question×configuration cells, pooled within the stated
  family** (main: 364 complete cells, 71 needing union → 19.51%; fixed/128: 76.2% of that
  family's complete cells; heading/paragraph: 0%). These are pooled cell counts, not a macro
  average of per-configuration shares.
* The audit shows an **association** between fragmentation and completeness failure, not a unique
  cause; it is indirect evidence about the mechanism.
* Discuss alternatives: chunk length, truncation, candidate-pool coverage, ranking quality.
* Length terciles: Δ +0.276, 95% CI [−0.019, +0.572] → "not statistically resolvable at this n",
  not "no length effect".
* test reranking is not significant; test2 is positive. Do **not** run or imply a formal
  between-split significance test: different questions, hardware and code versions.
* ρ = 0.963: high similarity, not invariance.
* T06/T07 are compatibility experiments; keep them out of the main result and never describe a
  retrained compatibility model as the paper's checkpoint.

## 13. Figures and tables (final set)

1. **Table 1** — four tracks on the first test split: Complete@10, 95% CI, MRR.
2. **Table 2** — test2 four tracks plus primary/secondary endpoints (Δ, CI, verdict).
3. **Figure 1** — matched token budget (512/1024/2048 recall and complete; note double charging
   and reranker truncation).
4. **Table 3** — union audit by family (Complete@10, share needing union, median chunks).
5. **Table 4** (if space) — per-family C1→C3 on the test split.
6. No duplicate four-track bar chart; ρ = 0.963 in one sentence.

## 14. Reproducibility and AI disclosure

* Reproducibility paragraph: repo + release URLs; R1 statistical recomputation reproduces both
  splits with zero field differences; **R2** reproduces **3060 row-level comparisons**
  (4 tracks × 17 configurations × 45 questions) for test2 with per-question values identical and
  official result hashes unchanged. Be explicit about the devices: the official test2 run was on
  the A5000 and the replay was also on the A5000, while the first test run was on the 3060.
  Row-level agreement does **not** prove that every historical environment detail matched; the
  evaluation environment is only partially recorded (`ENVIRONMENT_EVIDENCE.md`).
  `ACCEPTANCE.md` records what was and was not run.
* AI Disclosure section: tools used; what AI did (code assistance, question drafting, editing);
  how outputs were checked; first set student-reviewed (172); test2 jointly reviewed (attestation,
  after evaluation); the student is responsible for all claims.

## 15. Number validation

* `tools/make_paper_tables.py` generates table rows and key body numbers from
  `results/results.csv`, `stats.json`, `OUTCOME.json`, `summary_*.csv`.
* Each number is bound to `(split, track, config, metric, source_file)` in `paper/NUMBERS.json`.
* `tools/check_paper_numbers.py` checks table cells and bound body values against that mapping.
* PDF text extraction is an auxiliary check for unbound numbers and unit consistency; no global
  ban on values such as 0.833/0.717/0.750.

## 16. Build and acceptance

* Preferred: IEEEtran two-column via Tectonic; fallback: two-column DOCX + LibreOffice PDF.
* Deliverables: `paper/main.tex`, `paper/refs.bib`, `paper/figures/`, `docs/PAPER.pdf`,
  `docs/REFERENCES_VERIFIED.md`, `docs/PAPER_NOTES.md`.
* Acceptance:
  1. about 7 pages of main content (6–8 acceptable), two columns, references separate;
  2. 15–18 references, all in `REFERENCES_VERIFIED.md` with DOI/URL and date; both core papers
     substantively discussed;
  3. metric definitions match code (especially MRR = first **single chunk** fully covering a
     clause);
  4. test2 labelled single-clause in abstract, results and limitations;
  5. token-budget analysis in the main text with the accounting rules;
  6. statistics section separates pre-registered gatekeeping from supplementary Holm p;
  7. all numbers generated or checked via `NUMBERS.json`;
  8. no unsupported claims; effect-size units consistent;
  9. AI Disclosure present.

## 17. Format decisions (settled)

1. **IEEEtran**, English, two columns. Main content target **about 7 pages**, with **6–8 pages
   acceptable**; do not force exactly 7.0.
2. If Tectonic/LaTeX is unavailable, fall back to the **official IEEE Word conference template**
   (two columns), not a generic DOCX.
3. References are already student-verified (10) and the joint test2 review is complete; neither
   is a blocker. New references are still verified before they enter `refs.bib`.
