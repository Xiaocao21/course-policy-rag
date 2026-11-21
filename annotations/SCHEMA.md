# Gold annotation schema

All spans are **character offsets into the evidence layer** of a document
(`cleaned/<doc_id>.evidence.jsonl`). The evidence layer is character-faithful to
the source, so a span can be quoted back verbatim. Never annotate against the
retrieval layer: normalisation there can change character counts.

## Containment rule (revision section 7)

A gold clause counts as **fully covered** only when the union of retrieved source
offsets from the correct document version contains its **entire** annotated span.

- Retrieving part of it is `partial_span_coverage` = fraction of annotated source
  characters present. It is **not** a full-clause hit.
- A span therefore has to be the smallest unit that actually makes the clause
  self-sufficient. If a qualifier inside the same sentence matters, include it in
  the span rather than annotating a shorter fragment.

## Multiple sufficient evidence sets

Some questions can be answered from more than one minimal set of clauses. Record
each set as a separate element of `evidence_sets`. Metric behaviour:

| metric | one set | several sets |
|---|---|---|
| Recall@k | share of required clauses fully covered | max share over sets |
| Hit@k | any required clause fully covered | any required clause in any set fully covered |
| complete evidence | all clauses covered | all clauses of at least one set covered |

If the corpus genuinely cannot support alternative sets, leave one set and say so
in the Limitations section rather than inventing variants.

## Field reference - `questions.jsonl`

| field | type | allowed values / notes |
|---|---|---|
| `qid` | string | unique, e.g. `Q0001` |
| `question` | string | student-style natural question |
| `reference_answer` | string | the defensible answer, written by you |
| `group_id` | string | paraphrases and template variants share a group. A broad topic or an incidentally shared clause is NOT a group |
| `split` | enum | `dev` (100 target) or `test` (50 target) |
| `evidence_type` | enum | `single-clause`, `rule-and-exception`, `cross-clause` |
| `source_level` | enum | `course`, `faculty`, `institution` |
| `domain` | enum | `grading`, `late`, `attendance`, `exam`, `integrity`, `resources` |
| `condition` | enum | `baseline`, `exception`, `conflicting-version`, `alternative-evidence` |
| `evidence_sets` | array | each element `{"set_id": "...", "clauses": [clause_ref, ...]}` |
| `audit` | object | `manual_reviewed` (bool), `near_duplicate_of` (list), `note` |

### `clause_ref`

| field | notes |
|---|---|
| `doc_id` | must equal the corpus document id |
| `version` | source version string |
| `clause_id` | must exist in `cleaned/<doc_id>.evidence.jsonl` |
| `section_path` | copied from the clause record |
| `char_start`, `char_end` | half-open span in the evidence layer |
| `quote_evidence` | exact text of `evidence[char_start:char_end]`; validated by the audit |
| `token_start`, `token_end` | optional token offsets under the frozen tokenizer |

## Split discipline (revision section 7)

- Whole groups go to one split. `group_id` must not appear in both.
- Development is **not** sub-split: tuning uses group-disjoint cross-validation
  inside the 100 development questions.
- Target at least 40 independent test groups. If you cannot reach that, add
  genuinely new situations or question types. Do not multiply paraphrases, and do
  not move existing groups across the split.
- The held-out test set is evaluated **once**, after all settings are frozen.

### Worked leakage examples that must stay inside one split

| pair | why it is one group |
|---|---|
| "May I hand in work late?" / "Can I submit after the deadline?" | same template, same inference |
| "...miss an exam because of illness..." / "...because of travel..." | one missed-exam template, one rule plus different applicability conditions |

## Checklist before running `05_audit_splits.py`

1. Every `clause_id` exists and every `quote_evidence` matches its span exactly.
2. Every `group_id` is confined to one split.
3. Test questions represent >= 40 independent groups.
4. Evidence-type and domain distributions are balanced across the split.
5. Cross-split near-duplicates have been reviewed by hand.
6. A second annotator labelled at least 20% independently; report Cohen's kappa.

## Inter-annotator agreement

The double-annotated unit is the **question**, not the clause: for a sample of
questions the second annotator independently selects the minimal sufficient evidence
sets, and agreement is measured on the clause sets chosen (per-clause inclusion
decisions). `agreement_sample.csv` is the clause sampling frame that scopes which
questions to double-annotate; add one row per candidate clause decision once the
questions exist, then run `scripts/kappa_agreement.py`.
