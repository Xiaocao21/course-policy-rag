"""kappa_agreement.py - inter-annotator agreement for the gold evidence annotation.

The proposal commits to a second annotation review. The double-annotated unit is the
QUESTION: for a sample of questions the second annotator independently selects the
minimal sufficient evidence sets, and agreement is measured on the per-clause inclusion
decisions (is this clause required to answer the question, yes or no).

Two measures are reported:
    * Cohen's kappa over every (question, clause) inclusion decision   <- the headline
    * per-question set agreement (Jaccard of the selected clause sets)  <- diagnostic

Inputs (new, question-level):
    annotations/agreement_worksheet.csv
        qid, question, split, annotator1_clauses, annotator2_clauses, ...
        clause lists are semicolon separated as "setIndex:clause_id"
        e.g. "0:DOC#c3;0:DOC#c7;1:DOC#c3"

Legacy input (still accepted, clause-level):
    annotations/agreement_sample.csv
        clause_id, doc_id, label_annotator1, label_annotator2, notes
        where labels are include/exclude judgements.

Usage:
    .venv\\Scripts\\python.exe scripts/kappa_agreement.py
    .venv\\Scripts\\python.exe scripts/kappa_agreement.py --input annotations/agreement_worksheet.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, load_settings, project_paths, write_json  # noqa: E402


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two nominal raters. None when undefined."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    if len(labels) < 2:
        # every decision is the same label: kappa is 0/0, report 1.0 (perfect)
        return 1.0
    obs = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    exp = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if exp >= 1.0:
        return 1.0
    return (obs - exp) / (1 - exp)


def parse_clause_list(value: str) -> set[str]:
    """'0:DOC#c3;1:DOC#c7' -> {'DOC#c3', 'DOC#c7'} (set index is ignored for agreement)."""
    out: set[str] = set()
    for part in (value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part and part.split(":", 1)[0].strip().isdigit():
            part = part.split(":", 1)[1]
        out.add(part.strip())
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def question_level(rows: list[dict]) -> dict:
    """Build per-clause binary decisions per question, then one pooled kappa."""
    decisions: list[tuple[str, str]] = []
    per_question = []
    for r in rows:
        qid = (r.get("qid") or "").strip()
        a1 = parse_clause_list(r.get("annotator1_clauses", ""))
        a2 = parse_clause_list(r.get("annotator2_clauses", ""))
        if not qid or not a1 or not a2:
            continue
        for clause in sorted(a1 | a2):
            decisions.append(("include" if clause in a1 else "exclude",
                              "include" if clause in a2 else "exclude"))
        per_question.append({
            "qid": qid,
            "annotator1": sorted(a1),
            "annotator2": sorted(a2),
            "jaccard": round(jaccard(a1, a2), 3),
            "identical": a1 == a2,
        })
    kappa = cohen_kappa(decisions)
    obs = (sum(1 for a, b in decisions if a == b) / len(decisions)) if decisions else None
    return {
        "questions_compared": len(per_question),
        "decisions": len(decisions),
        "observed_agreement": round(obs, 4) if obs is not None else None,
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "identical_sets": sum(1 for q in per_question if q["identical"]),
        "mean_set_jaccard": (round(sum(q["jaccard"] for q in per_question) / len(per_question), 4)
                             if per_question else None),
        "per_question": per_question,
        "disagreements": [q for q in per_question if not q["identical"]],
    }


def legacy_clause_level(rows: list[dict]) -> dict:
    pairs = []
    for r in rows:
        a = (r.get("label_annotator1") or "").strip().lower()
        b = (r.get("label_annotator2") or "").strip().lower()
        if a and b:
            pairs.append((a, b))
    kappa = cohen_kappa(pairs)
    obs = (sum(1 for a, b in pairs if a == b) / len(pairs)) if pairs else None
    return {
        "decisions": len(pairs),
        "observed_agreement": round(obs, 4) if obs is not None else None,
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "disagreements": [
            {"clause_id": r.get("clause_id", ""), "annotator1": r.get("label_annotator1", ""),
             "annotator2": r.get("label_annotator2", ""), "notes": r.get("notes", "")}
            for r in rows
            if (r.get("label_annotator1") or "").strip()
            and (r.get("label_annotator2") or "").strip()
            and r["label_annotator1"].strip().lower() != r["label_annotator2"].strip().lower()
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--input", default="")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    target = cfg["annotation"]["kappa_target"]

    candidates = ([Path(args.input)] if args.input else
                  [paths["annotations"] / "agreement_worksheet.csv",
                   paths["annotations"] / "agreement_sample.csv"])
    path = None
    for c in candidates:
        p = c if c.is_absolute() else root / c
        if p.exists():
            path = p
            break
    if path is None:
        print("no agreement file found. Generate one with:", file=sys.stderr)
        print("  scripts/05_audit_splits.py --make-agreement-sample 20", file=sys.stderr)
        return 2

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    mode = ("question-level" if "annotator1_clauses" in (rows[0].keys() if rows else [])
            else "clause-level(legacy)")

    if mode == "question-level":
        result = question_level(rows)
        total_questions = len([r for r in rows if (r.get("qid") or "").strip()])
        labelled = result["questions_compared"]
        unlabelled = total_questions - labelled
    else:
        result = legacy_clause_level(rows)
        total_questions = len(rows)
        labelled = sum(1 for r in rows
                       if (r.get("label_annotator1") or "").strip()
                       and (r.get("label_annotator2") or "").strip())
        unlabelled = total_questions - labelled

    kappa = result.get("cohen_kappa")
    report = {
        "file": str(path),
        "mode": mode,
        "rows": total_questions,
        "labelled": labelled,
        "unlabelled": unlabelled,
        "target_kappa": target,
        "meets_target": bool(kappa is not None and kappa >= target),
        **result,
    }
    out = paths["annotations"] / "agreement_report.json"
    write_json(out, report)

    print("=== inter-annotator agreement ===")
    print(f"  input              : {path.name}  ({mode})")
    print(f"  rows               : {total_questions}")
    print(f"  double-annotated   : {labelled}")
    print(f"  not yet annotated  : {unlabelled}")
    if result.get("decisions") is not None:
        print(f"  clause decisions   : {result['decisions']}")
    if result.get("observed_agreement") is not None:
        print(f"  observed agreement : {result['observed_agreement']:.4f}")
    print(f"  Cohen's kappa      : {kappa}  (target >= {target})")
    if mode == "question-level":
        print(f"  identical sets     : {result['identical_sets']} of {result['questions_compared']}")
        print(f"  mean set Jaccard   : {result['mean_set_jaccard']}")
    print(f"  meets target       : {report['meets_target']}")
    dis = result.get("disagreements") or []
    print(f"  disagreements      : {len(dis)}")
    for d in dis[:8]:
        if mode == "question-level":
            print(f"    - {d['qid']}: Jaccard {d['jaccard']}")
        else:
            print(f"    - {d.get('clause_id')}: '{d.get('annotator1')}' vs '{d.get('annotator2')}'")
    print(f"\nwrote {out}")
    if not labelled:
        print("\nNothing double-annotated yet. Fill the annotator2 column, then re-run.")
        return 0
    return 0 if report["meets_target"] else 1


if __name__ == "__main__":
    sys.exit(main())
