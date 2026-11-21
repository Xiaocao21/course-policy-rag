"""make_annotation_template.py - create annotation scaffolding the researcher fills in.

This script does NOT generate questions, gold spans or answers. The proposal's AI
disclosure states that the student personally annotates gold source spans and
reference answers, so this only produces:

    annotations/questions.template.jsonl   one worked example per evidence type
    annotations/SCHEMA.md                  field-by-field specification
    annotations/gold_worksheet.csv         clause inventory to pick spans from
    annotations/agreement_sample.csv       sampling frame for the second annotator

Every evidence reference must resolve against cleaned/*.evidence.jsonl and satisfy
full-span containment (see SCHEMA.md).

Usage:
    .venv\\Scripts\\python.exe scripts\\make_annotation_template.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, load_taxonomy, project_paths, read_jsonl, write_jsonl,
)

SCHEMA_MD = """# Gold annotation schema

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
"""


def included_doc_ids(root: Path) -> set[str]:
    """Documents that are actually in the frozen corpus (include=TRUE in sources.csv).

    Annotation scaffolding must cover the corpus, not everything the crawl happened
    to collect; otherwise the worksheet and agreement sample are mostly documents the
    study does not analyse.
    """
    sources_csv = root / "sources.csv"
    if not sources_csv.exists():
        return set()
    out: set[str] = set()
    with sources_csv.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("include", "")).upper() in ("TRUE", "YES", "1"):
                out.add(r["doc_id"])
    return out


def build_examples(paths: dict, allowed: set[str] | None = None) -> list[dict]:
    """One worked example per evidence type, wired to real clause ids when possible."""
    clauses: list[dict] = []
    for p in sorted(paths["cleaned"].glob("*.evidence.jsonl")):
        doc_id = p.name.replace(".evidence.jsonl", "")
        if allowed and doc_id not in allowed:
            continue
        clauses.extend(read_jsonl(p))

    def pick(pred) -> dict | None:
        for c in clauses:
            if pred(c):
                return c
        return None

    examples: list[dict] = []

    def ref(c: dict) -> dict:
        quote = c["text"]
        return {
            "doc_id": c["doc_id"], "version": "", "clause_id": c["clause_id"],
            "section_path": c.get("section_path", ""),
            "char_start": c["char_start"], "char_end": c["char_end"],
            "quote_evidence": quote,
            "token_start": None, "token_end": None,
        }

    single = pick(lambda c: not c.get("has_exception") and not c.get("has_crossref")
                  and c.get("char_len", 0) > 80)
    if single:
        examples.append({
            "qid": "Q0001",
            "question": "<<write a student-style question whose answer is fully supported by this single clause>>",
            "reference_answer": "<<write the defensible answer>>",
            "group_id": "G0001",
            "split": "dev",
            "evidence_type": "single-clause",
            "source_level": single.get("tier", ""),
            "domain": "<<grading|late|attendance|exam|integrity|resources>>",
            "condition": "baseline",
            "evidence_sets": [{"set_id": "S1", "clauses": [ref(single)]}],
            "audit": {"manual_reviewed": False, "near_duplicate_of": [], "note": "template example"},
        })

    exc = pick(lambda c: c.get("has_exception"))
    if exc:
        examples.append({
            "qid": "Q0002",
            "question": "<<write a question that needs the general rule AND the exception clause>>",
            "reference_answer": "<<answer must cite both clauses>>",
            "group_id": "G0002",
            "split": "dev",
            "evidence_type": "rule-and-exception",
            "source_level": exc.get("tier", ""),
            "domain": "<<domain>>",
            "condition": "exception",
            "evidence_sets": [{"set_id": "S1", "clauses": [ref(exc)]}],
            "audit": {"manual_reviewed": False, "near_duplicate_of": [],
                      "note": "add the general-rule clause to this set; the example shows one ref"},
        })

    xr = pick(lambda c: c.get("has_crossref"))
    if xr:
        examples.append({
            "qid": "Q0003",
            "question": "<<write a question requiring several clauses, e.g. a rule, a condition and a deadline>>",
            "reference_answer": "<<answer must cite every clause in the set>>",
            "group_id": "G0003",
            "split": "test",
            "evidence_type": "cross-clause",
            "source_level": xr.get("tier", ""),
            "domain": "<<domain>>",
            "condition": "baseline",
            "evidence_sets": [
                {"set_id": "S1", "clauses": [ref(xr)]},
                {"set_id": "S2", "clauses": [ref(xr)], "_comment": "delete if no alternative set exists"},
            ],
            "audit": {"manual_reviewed": False, "near_duplicate_of": [],
                      "note": "shows two alternative sufficient sets"},
        })

    return examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)
    ann = cfg["annotation"]

    paths["annotations"].mkdir(parents=True, exist_ok=True)

    allowed = included_doc_ids(root)
    if allowed:
        print(f"restricting annotation scaffolding to {len(allowed)} corpus documents")
    else:
        print("WARNING: no include=TRUE documents found; scaffolding covers every "
              "cleaned document. Run curate_corpus.py first for a corpus-scoped sheet.")

    # SCHEMA.md
    (paths["annotations"] / "SCHEMA.md").write_text(SCHEMA_MD, encoding="utf-8")

    # questions.template.jsonl (never overwrite questions.jsonl)
    examples = build_examples(paths, allowed)
    tpl = paths["annotations"] / "questions.template.jsonl"
    write_jsonl(tpl, examples)

    if not (paths["annotations"] / "questions.jsonl").exists():
        write_jsonl(paths["annotations"] / "questions.jsonl", [])

    # gold worksheet: clause inventory to choose spans from
    rows: list[dict] = []
    for p in sorted(paths["cleaned"].glob("*.evidence.jsonl")):
        doc_id = p.name.replace(".evidence.jsonl", "")
        if allowed and doc_id not in allowed:
            continue
        for c in read_jsonl(p):
            rows.append({
                "doc_id": c["doc_id"], "tier": c.get("tier", ""),
                "institution": c.get("institution", ""),
                "section_path": c.get("section_path", ""),
                "clause_id": c["clause_id"], "clause_label": c.get("clause_label", ""),
                "char_start": c["char_start"], "char_end": c["char_end"],
                "char_len": c.get("char_len", 0),
                "has_exception": c.get("has_exception", False),
                "has_crossref": c.get("has_crossref", False),
                "has_alt_evidence": c.get("has_alt_evidence", False),
                "boilerplate": c.get("boilerplate", False),
                "text": (c.get("text", "")[:400].replace("\n", " ")),
            })
    ws = paths["annotations"] / "gold_worksheet.csv"
    with ws.open("w", encoding="utf-8", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # agreement sampling frame for the second annotator (20% by default)
    rng = random.Random(cfg["project"]["random_seed"])
    n = max(1, int(len(rows) * ann["kappa_sample_ratio"])) if rows else 0
    sample = rng.sample(rows, min(n, len(rows))) if rows else []
    agree = paths["annotations"] / "agreement_sample.csv"
    with agree.open("w", encoding="utf-8", newline="") as f:
        fields = ["clause_id", "doc_id", "label_annotator1", "label_annotator2", "notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sample:
            w.writerow({"clause_id": r["clause_id"], "doc_id": r["doc_id"],
                        "label_annotator1": "", "label_annotator2": "", "notes": ""})

    print("=== annotation scaffolding ===")
    print(f"  clauses available        : {len(rows)}")
    print(f"  template questions       : {len(examples)} worked examples")
    print(f"  kappa sample (target {ann['kappa_sample_ratio']:.0%}) : {len(sample)} clauses")
    print(f"  targets: {ann['target_dev_questions']} dev + {ann['target_test_questions']} test, "
          f">= {ann['target_test_groups_min']} test groups")
    print(f"\nwrote {paths['annotations'] / 'SCHEMA.md'}")
    print(f"wrote {tpl}")
    print(f"wrote {paths['annotations'] / 'questions.jsonl'} (empty; fill this in)")
    print(f"wrote {ws}")
    print(f"wrote {agree}")
    print("\nThe examples in questions.template.jsonl are structural placeholders only.")
    print("Write your own questions, gold spans and reference answers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
