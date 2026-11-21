# course-policy-rag — data collection and preprocessing

Pipeline that builds the frozen, auditable corpus for the study *Effects of Text
Chunking Strategies on Retrieval Accuracy in RAG-Based Question Answering for
University Course Policies* (CPS5801).

The corpus is the single shared input to all four chunking strategies, so this
repository's job is to make that input **reproducible** and **fair**: identical
bytes for every strategy, gold spans anchored character-exactly, and no strategy
silently receiving more model input than another.

Project root: `E:\CPS5801\course-policy-rag` (kept off `C:` which had ~2.6 GB free).

---

## Quick start

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py run.py test          # offline unit tests, no network
& $py run.py probe         # score candidate seed pages (robots, density, capability)
& $py run.py ocw           # harvest MIT OpenCourseWare syllabus candidates
& $py run.py template      # create annotation scaffolding

& $py scripts\build_registry.py --include-course-selected
& $py run.py collect       # crawl registry -> raw/ + sources.csv
& $py run.py extract       # structure + dual text layers + dedup
& $py run.py corpus        # shared 512-token corpus
& $py run.py validate      # invariant checks
& $py run.py freeze        # MANIFEST.json + COMPLIANCE.md
& $py run.py report        # what exists so far
```

---

## Corpus design: three tiers

Measured finding that drove this design: MIT OCW syllabus pages are thin
(5–13 policy keywords on a typical page), while institutional conduct codes and
regulations are dense (BU's academic conduct code: 47,509 chars, 195 keyword
hits, 14 exception clauses). No single tier covers all six question domains.

| tier | scope | who sets it | supplies |
|---|---|---|---|
| `course` | one course section | instructor | late work, course resources, grade weighting |
| `faculty` | faculty / department / college | faculty committee | grading, exam, attendance rules with exceptions |
| `institution` | whole university | provost / registrar / integrity office | academic integrity, deferred exams, appeals |

Tier shares are checked at validation time and no tier may exceed 40%.

**`late work` can only come from the course tier**, so that tier is never dropped
below its share — it is the one domain institutional regulations cannot supply.

---

## Two text layers (do not collapse these)

| layer | file | property |
|---|---|---|
| **evidence** | `cleaned/<doc_id>.evidence.jsonl` | character-faithful to the source. **All gold spans anchor here.** |
| **retrieval** | `cleaned/<doc_id>.retrieval.jsonl` | normalised (NFKC, quotes, dashes, ligatures, whitespace) for embedding and retrieval |

`cleaned/<doc_id>.charmap.jsonl` maps every retrieval character back to its
evidence index, so a retrieved hit can be tested for **full-span containment**
against a gold span.

Why it matters: the revised protocol scores a clause as fully covered only when the
union of retrieved offsets contains its *entire* annotated span, and tracks partial
overlap separately as `partial_span_coverage`. If normalisation changed character
counts without a map, both numbers would be silently wrong.

---

## Rules the pipeline enforces

1. **robots.txt is evaluated before every host is touched.** A disallowed URL is
   never requested. Decisions and raw robots policies are stored in
   `reports/robots_audit.json`. Sources that cannot be crawled are documented in
   `reports/manual_only_sources.json` and in `COMPLIANCE.md`.
2. **Raw bytes are immutable.** `raw/<tier>/<doc_id>/raw.{html,pdf}` plus
   `meta.json` with URL, retrieval time, ETag/Last-Modified and SHA-256.
3. **Exception clauses survive cleaning.** Header/footer lines containing
   `except|unless|provided that|notwithstanding|subject to|shall|may be|waive|
   extension|penalt|deadline` are marked `REVIEW_PROTECTED` and kept, never
   auto-deleted. Every removal is logged in `cleaned/removed_lines.log`.
4. **Clause spans are stable.** Chunking may split an over-length clause into
   several chunks, but it never redefines the clause boundary, so the annotated
   span stays valid and containment tests stay meaningful.
5. **Nothing is silently truncated.** Blocks over 512 tokens are split recursively
   (sentence boundary → token boundary → character) with the reason recorded.
6. **Polite crawling**: per-host delay, `Crawl-Delay` honoured, exponential backoff,
   a global page cap, and a self-identifying user agent.

---

## Layout

```
config/     settings.yaml, domain_taxonomy.yaml, source_registry.csv, landings.csv
raw/        immutable source bytes + meta.json, grouped by tier
extracted/  structure tree: heading / clause / paragraph / list / table + provenance
cleaned/    evidence + retrieval layers, charmap, removed_lines.log
corpus_safe_512/  the shared corpus and chunk index
annotations/      questions, gold worksheet, split audit, kappa sample, SCHEMA.md
reports/          probe results, fetch log, quality report, capability, validation
tests/            offline unit tests (no network)
sources.csv       provenance and decisions, one row per document
MANIFEST.json     frozen hashes for everything
COMPLIANCE.md     robots decisions, licence notes, documented limitations
```

---

## Scripts

| script | role |
|---|---|
| `probe_seeds.py` | score candidates: robots, reachability, domain coverage, exception/crossref/alt-evidence counts |
| `harvest_ocw.py` | walk the OCW sitemap index, score syllabus pages, emit course-tier candidates |
| `select_course_seeds.py` | keep the densest N course pages so the tier stays a deliberate share |
| `build_registry.py` | convert probe/discovery output into collector-ready seeds (with `crawl_depth`) |
| `01_collect.py` | polite crawl into `raw/`, content gate, capability metrics, incremental flush |
| `02_extract_clean.py` | extraction, header/footer protection, dual layers, dedup, clause records |
| `03_build_safe_corpus.py` | 512-token shared corpus, split reasons, token distribution figure |
| `curate_corpus.py` | select the documented corpus, enforce tier balance, write `include=` back |
| `05_audit_splits.py` | group/split discipline, leakage checks, containment self-test |
| `06_validate.py` | corpus invariants and acceptance gates |
| `04_freeze_manifest.py` | `MANIFEST.json` + `COMPLIANCE.md` |
| `make_annotation_template.py` | schema, worked examples, gold worksheet, kappa sample |
| `kappa_agreement.py` | Cohen's kappa for the second-annotator review |
| `report_content.py` | single source of truth for every figure in the reports, with consistency assertions |
| `make_progress_report.py` | render the Chinese supervisor progress report (needs the bundled Python, not the venv) |
| `make_status_report.py` | thin wrapper that delegates to `make_progress_report.py` |

`crawl_depth` matters: course seeds use `0` (the syllabus page is the document;
following links would pull in lecture notes), while faculty and institution seeds
use `2` so the crawler can walk from an index page to the actual policies.

---

## Environment

```powershell
python -m venv .venv                      # done
.\.venv\Scripts\python.exe -m pip install -r requirements.txt     # stage A
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt  # stage B, later
```

Stage A (installed) is enough for collection and preprocessing.
Stage B adds `sentence-transformers`, `transformers`, `torch`, `faiss-cpu` for
embeddings, indexing and the real `bge-small-en-v1.5` tokenizer.

Until stage B is installed, `03_build_safe_corpus.py` falls back to approximate
tokenisation and writes `*_approx` files, so an approximate run can never be
mistaken for the frozen corpus.

### Dependency traps already hit

- `selectolax` 1.0 **removed** `selectolax.parser` (the "modest" backend). Scripts
  use `selectolax.lexbor.LexborHTMLParser` with a legacy fallback.
- `lxml` is required by the probe and is **not** part of the stdlib.
- Both of the above originally failed *silently* — a missing parser returned `""`,
  making real pages look empty and get classified `pending_review`. Parser helpers
  now raise instead of returning empty.

---

## What this pipeline deliberately does not do

- It does not write questions, gold spans or reference answers. It only produces
  the schema, a worked-example template, the clause worksheet and the agreement
  sample. The proposal's AI disclosure assigns annotation to the student.
- It does not collect WKU material; only public external sources are used.
- It does not bypass logins, paywalls or robots restrictions.
- It does not implement the four chunking strategies or any retrieval metric.

---

## Current status

**Corpus v0.1.5 is frozen, the first question set is signed off (172 reviewed), the test split is locked and evaluated once, and a second held-out set (45 questions) was pre-registered, locked and evaluated once on 2026-11-13.**

265 documents collected across 13 institutions; 36 selected into the corpus
(14 course / 9 faculty / 13 institution, all tiers within the 40% cap), 12 institutions,
2,007 clause units, 1,988 clauses, maximum unit 509 tokens (real `bge-small-en-v1.5`
tokenizer), 0 over the model limit, 0 clause-span errors, 0 validation blockers.

**Questions**: 172 after review repairs (`annotations/questions.jsonl`) - 112 dev / 60 test,
59 independent test groups. Evidence types 68 single / 60 rule-and-exception / 44 cross-clause;
six domains in both splits, with resources at 7 test questions. All drafted by AI from corpus
clauses; the student reviewed and accepted all 172 (`manual_reviewed: true`, recorded in
`reports/lock_audit.jsonl`), and the test split was locked before its single evaluation.
**Second held-out set (test2)**: 45 questions in 45 groups drawn from clauses that no existing
question uses, by a pre-registered length-stratified rule; locked in `annotations/LOCKED_TEST2/`
and evaluated once (`results/final_test2/`, a single run). Drafted and self-audited by the agent
(`agent_self_audit`), **no human sign-off**. Composition deviations (integrity-heavy, Glasgow
33%, single-clause only) were recorded before evaluation in
`results/test2/DESIGN_AMENDMENTS.json`. Outcome (`results/test2/OUTCOME.json`): primary
structural-minus-semantic **+0.800 [0.666, 0.911]** (confirmed at least 0.30); secondary
C3 - C1 **+0.052 [0.020, 0.089]** (positive, magnitude unresolved).

**Review repairs applied** (2026-10-22, `annotations/repair_patch.json`,
`reports/repair_report.json`): 29 content fixes (six wrong/incomplete answers, nine
non-minimal evidence sets, four unnatural cross-clause pairings), 79 source qualifiers so
multi-institution questions name their source, and a structural fix for split leakage —
questions that cite the same gold clause are kept in one split (a split constraint), while
`group_id` is clustered by question wording so a group means a real paraphrase/variant
family rather than an evidence tangle. **Zero gold clauses and zero groups cross the
split** (previously 20 clauses did). Six new independent grading questions were added and
three weak examination items removed, so both splits gained grading coverage without
changing the test-set size.

**Development results** (`results/REPORT.md`):

| strategy | config | R@1 | R@5 | R@10 | Complete@10 | MRR | CV R@10 |
|---|---|---|---|---|---|---|---|
| paragraph | paragraph | 0.346 | 0.601 | **0.719** | **0.580** | 0.598 | 0.719 |
| heading | heading | 0.322 | 0.547 | 0.679 | 0.562 | 0.541 | 0.680 |
| fixed | size512_ov64 | 0.245 | 0.499 | 0.616 | 0.509 | 0.431 | 0.607 |
| semantic | thr0.70 | 0.044 | 0.112 | 0.143 | 0.054 | 0.119 | 0.145 |

**Held-out test results** (configurations chosen on development only):

| strategy | config | dev R@10 | test R@10 | dev Complete@10 | test Complete@10 | test MRR |
|---|---|---|---|---|---|---|
| heading | heading | 0.679 | **0.767** | 0.562 | **0.733** | 0.613 |
| fixed | size512_ov64 | 0.616 | 0.692 | 0.509 | 0.617 | 0.483 |
| paragraph | paragraph | **0.719** | 0.692 | **0.580** | 0.667 | 0.541 |
| semantic | thr0.70 | 0.143 | 0.092 | 0.054 | 0.033 | 0.086 |

The dev and test orderings of the two structural strategies do not agree: paragraph leads on
development (0.719 vs 0.679) while heading leads on test (0.767 vs 0.692). The dev-test gap is
within what 60 test questions can resolve, so the honest claim is that **paragraph and heading
are tied within noise and both beat fixed-length and semantic chunking**, not that one is
better than the other.

Two findings worth stating: under **matched retrieved-token budgets** the order changes
(paragraph leads: 0.404 / 0.505 / 0.619 at 512 / 1024 / 2048, ahead of heading's
0.330 / 0.454 / 0.560), which is exactly the confound the protocol wanted to control. And
**semantic chunking at 0.70–0.80 thresholds degenerates to near-sentence units** (5.5–7.0k
chunks), which cannot contain a multi-sentence clause, so full-span coverage collapses —
a substantive result for an evidence-completeness metric, not a tuning accident.

### Final four-track evaluation (stage 3 complete)

Test split evaluated once with frozen settings; the split had already been used for the
exploratory pilot, so this is a formal **re-evaluation**, not an unseen holdout.

| track | pool | reranker | Complete@10 [95% CI] | MRR |
|---|---|---|---|---|
| main | 10 | no | 0.358 [0.299, 0.418] | 0.276 |
| C1 | 50 | no | 0.358 [0.299, 0.418] | 0.282 |
| C2 | 10 | yes | 0.358 [0.299, 0.418] | 0.312 |
| C3 | 50 | yes | **0.405 [0.347, 0.464]** | **0.322** |

C2 = main and C1 = main on the @10 set metrics are **asserted** (max difference 0.0), not
tested. Primary contrast C3 − C1 = +0.047 [−0.005, +0.095], Holm p = 0.15: suggestive, not
established. Chunking ranking survives reranking (Spearman ρ = 0.963).

Deliverables: `results/REPORT_FINAL.md`, `results/final/` (68 result files + stats),
`docs/TEACHER_BRIEF.docx` (3 pages), `docs/DECISIONS.md`, `docs/RERANKER_PROTOCOL.md`.

### Reranker stage (stage 1 and 2 complete)

One cross-encoder (`bge-reranker-base`) is trained for **all 17 configurations** on the
development split and frozen; the frozen bi-encoder indexes are untouched.

| step | artifact |
|---|---|
| candidate data | `models/reranker/pairs_dev.jsonl` - 22,200 pairs (1,581 clause-positives, 18,807 negatives, 1,812 partials), deduplicated by `(qid, doc_id, version, char_start, char_end, text_sha1)` with `config_ids` / `dense_rank_by_config` / `source_offsets` retained |
| group-disjoint split | 85 train groups (89 questions) / 21 validation groups (22 questions), zero overlap, asserted |
| training | 2 learning rates x 4 negatives, early stopping on macro reranked `Complete@10` over a top-50 pool |
| frozen model | `models/reranker/FROZEN.json` (canonical RTX 3060, fp16, Adafactor, first 10 of 12 encoder layers frozen = 85.6 M trainable parameters) |

Validation (top-50 pool, macro over 17 configurations):

| condition | Complete@10 | Recall@10 | MRR |
|---|---|---|---|
| dense, no reranker | 0.358 | 0.437 | 0.327 |
| off-the-shelf reranker | 0.358 | 0.445 | 0.329 |
| fine-tuned reranker (frozen) | **0.388** | **0.468** | **0.387** |

Two constraints worth remembering: reranking a fixed top-10 pool cannot change any @10
set-level metric (enforced by `tests/test_rerank.py::TestSetInvariance`), so model selection
had to use the top-50 pool; and on a 6 GB card AdamW states plus fp32 gradients do not fit,
which is why the optimizer and the frozen-layer count are recorded in `FROZEN.json`.

### Annotation and experiment toolchain

| step | command | what it does |
|---|---|---|
| 1 | `run.py feasibility` | capacity vs quota per cell; refuses to bless a corpus that cannot support 40 test groups |
| 2 | `run.py template` | schema, worked examples, gold worksheet (scoped to the corpus only) |
| 3 | `run.py candidates` | proposes clause / rule+exception pair / alternative-evidence candidates |
| 4 | `run.py annotate` | local workbench on 127.0.0.1 with live validation |
| 5 | `run.py review` | builds `REVIEW.md` / `REVIEW.csv`: every question with its cited clause text |
| 6 | `run.py reviewed` | **your sign-off**: sets `manual_reviewed=true` (recorded in `reports/lock_audit.jsonl`) |
| 7 | `run.py audit` | split, group, leakage and manual-review checks |
| 8 | `run.py lock` | freezes the held-out test split and records its hash |
| 9 | `run.py chunk` → `index` → `evaluate` → `analyze` | four strategies, FAISS indexes, metrics, `results/REPORT.md` |
| 10 | `run.py agreement` → `kappa` | question-level second annotation and Cohen's kappa |


**Candidate honesty.** `08_prefill_candidates.py` writes
`prefilled=true, manual_reviewed=false` on every artefact. A prefilled question cannot
enter the test split until `manual_reviewed` is set by hand; `goldset.validate_question`
and the workbench both enforce this. The gold annotation itself remains the researcher's
own work, as the proposal's AI disclosure requires.

**Why a workbench instead of a spreadsheet.** Spans must be exact character offsets in
the evidence layer, and a clause counts as covered only when the retrieved union contains
the whole span. Selecting text in the browser produces the offset; the server re-checks
quote fidelity, clause existence, span bounds and group/split discipline before saving.

### State repair (why this version exists)

The corpus had drifted: `MANIFEST.json` described 32 documents while `sources.csv`
selected 31, `validation.json` described a third state, the gold worksheet still
referenced a dropped document, and 9 documents carried the *seed* tier inside the corpus
instead of the curated tier. Two root causes were fixed:

- `03_build_safe_corpus.py` now takes tier/institution from `sources.csv` (curation is
  authoritative) and **fails** if the corpus and curation disagree.
- `04_freeze_manifest.py` now refuses to freeze unless five contract gates hold: the
  corpus document set equals `include=TRUE`; corpus tiers equal curated tiers;
  `validation.json` describes this corpus; `settings.yaml` exists; and
  `gold_worksheet.csv` references only corpus documents.

### Figure provenance

`sources.csv` carries two families of column and they must not be confused:

- `clause_count_raw_hits`, `exception_count_raw_hits`, … — page-level regex hits
  gathered at crawl time, navigation included. **Not** clause counts.
- `clauses_usable`, `clauses_exception`, `clauses_crossref`, `clauses_alt` — authoritative
  clause-level counts measured on the cleaned evidence layer, written back by
  `curate_corpus.py`, sourced from `reports/clause_stats.csv`.

`06_validate.py` reads only the authoritative family (`reports/clause_stats.csv`).
`01_collect.py` no longer writes a capability report, because the crawl-time numbers are
not clause counts.

### Pending

1. **Institution concentration.** Glasgow supplies ~53% of usable clauses and the top-3
   documents hold ~34% of corpus characters. Report per-document macro averages alongside
   pooled scores. The second held-out set inherits this (Glasgow 33%, recorded as a design
   amendment).
2. **Optional: expand the `resources` domain.** It has 7 questions in the first test split
   and only exists in a thin form in test2; expanding it needs new questions and therefore a
   new pre-registration and lock, not a re-run of a frozen split.

Experiments are complete: corpus and 17 indexes frozen, both splits evaluated, the reranker
frozen (plus the A5000 compatibility runs T06/T07), and the second held-out set pre-registered,
locked and evaluated once. The authoritative summary is `docs/PROJECT_REPORT.docx` (revision 4).

Run `python run.py report` for live state. Validation gates are in
`config/settings.yaml` under `acceptance:`.

### Reports

```powershell
# figures come from report_content.py; generation is refused if they disagree
& "<bundled python>" scripts\make_progress_report.py
```

Do not restate corpus numbers by hand in a document — add them to
`report_content.py` so the consistency check covers them.
