"""10c_make_review.py - build the human review packet for the AI-drafted questions.

The student reviews the annotation and is the one who sets `manual_reviewed`. This
script produces a readable packet so that review is a matter of reading and ticking,
not clicking through 150+ records one at a time.

Outputs
    annotations/REVIEW.md    readable packet, grouped by domain, with clause quotes
    annotations/REVIEW.csv   the same content flattened for filtering in a spreadsheet
    annotations/review_checklist.txt  one qid per line, for `--mark-reviewed`

Nothing here modifies the questions; it only reads them.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, project_paths, read_jsonl  # noqa: E402
import goldset  # noqa: E402

DOMAIN_ORDER = ["grading", "integrity", "late", "exam", "attendance", "resources"]


def quote(corpus: goldset.Corpus, ref: dict, limit: int = 260) -> str:
    clause = corpus.clauses.get(ref.get("clause_id", ""))
    if clause is None:
        return f"[UNRESOLVED {ref.get('clause_id')}]"
    text = goldset.evidence_text(corpus, clause.doc_id)[clause.char_start:clause.char_end]
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--quote-chars", type=int, default=260)
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    questions = read_jsonl(paths["annotations"] / "questions.jsonl")
    if not questions:
        print("no questions to review", file=sys.stderr)
        return 2
    corpus = goldset.load_corpus(root)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_domain[q.get("domain", "?")].append(q)

    stats = {
        "total": len(questions),
        "dev": sum(1 for q in questions if q.get("split") == "dev"),
        "test": sum(1 for q in questions if q.get("split") == "test"),
        "types": dict(Counter(q.get("evidence_type") for q in questions)),
        "domains": dict(Counter(q.get("domain") for q in questions)),
        "levels": dict(Counter(q.get("source_level") for q in questions)),
    }

    lines: list[str] = []
    lines.append("# Annotation review packet")
    lines.append("")
    lines.append(f"Corpus: **{corpus.corpus_version}**  ·  questions: **{stats['total']}** "
                 f"(dev {stats['dev']} / test {stats['test']})")
    lines.append("")
    lines.append("Every question below was **drafted by AI** from clauses in the frozen corpus "
                 "and is marked `manual_reviewed: false`. Read each one, check that the "
                 "reference answer really follows from the quoted clause text, then set "
                 "`manual_reviewed` to true for the ones you accept.")
    lines.append("")
    lines.append("How to accept:")
    lines.append("")
    lines.append("```powershell")
    lines.append("# accept everything in the checklist (all questions)")
    lines.append(".\\\\.venv\\\\Scripts\\\\python.exe scripts\\\\05_audit_splits.py --mark-reviewed")
    lines.append("")
    lines.append("# or accept a subset you have verified:")
    lines.append(".\\\\.venv\\\\Scripts\\\\python.exe scripts\\\\05_audit_splits.py --mark-reviewed Q0001 Q0002 Q0007")
    lines.append("")
    lines.append("# or review interactively in the workbench")
    lines.append(".\\\\.venv\\\\Scripts\\\\python.exe run.py annotate")
    lines.append("```")
    lines.append("")
    lines.append("Distribution: " + " · ".join(f"{k} {v}" for k, v in stats["types"].items())
                 + " | " + " · ".join(f"{k} {v}" for k, v in stats["domains"].items()))
    lines.append("")
    lines.append("---")
    lines.append("")

    seen_domains = [d for d in DOMAIN_ORDER if d in by_domain] + \
                   [d for d in by_domain if d not in DOMAIN_ORDER]
    for domain in seen_domains:
        qs = sorted(by_domain[domain], key=lambda q: q.get("qid", ""))
        lines.append(f"## {domain} ({len(qs)})")
        lines.append("")
        for q in qs:
            split = q.get("split", "?")
            et = q.get("evidence_type", "?")
            level = q.get("source_level", "?")
            cond = q.get("condition", "?")
            lines.append(f"### {q.get('qid')} · {split} · {et} · {level} · {cond}")
            lines.append("")
            lines.append(f"**Q:** {q.get('question', '')}")
            lines.append("")
            lines.append(f"**A:** {q.get('reference_answer', '')}")
            lines.append("")
            n_sets = len(q.get("evidence_sets") or [])
            if n_sets > 1:
                lines.append(f"*{n_sets} alternative sufficient sets*")
                lines.append("")
            for si, eset in enumerate(q.get("evidence_sets") or []):
                label = f"set {si + 1}" if n_sets > 1 else "evidence"
                for ref in eset.get("clauses") or []:
                    cid = ref.get("clause_id", "")
                    lines.append(f"- `{cid}` — {quote(corpus, ref, args.quote_chars)}")
                if n_sets > 1:
                    lines.append(f"  *({label})*")
            lines.append("")
        lines.append("---")
        lines.append("")

    review_md = paths["annotations"] / "REVIEW.md"
    review_md.write_text("\n".join(lines), encoding="utf-8")

    csv_path = paths["annotations"] / "REVIEW.csv"
    fields = ["qid", "split", "domain", "evidence_type", "source_level", "condition",
              "question", "reference_answer", "clause_ids", "group_id", "manual_reviewed"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for q in sorted(questions, key=lambda q: (DOMAIN_ORDER.index(q["domain"])
                                                  if q.get("domain") in DOMAIN_ORDER else 99,
                                                  q.get("qid", ""))):
            cids = ";".join(r.get("clause_id", "")
                            for s in q.get("evidence_sets") or []
                            for r in s.get("clauses") or [])
            w.writerow({
                "qid": q.get("qid", ""), "split": q.get("split", ""),
                "domain": q.get("domain", ""), "evidence_type": q.get("evidence_type", ""),
                "source_level": q.get("source_level", ""), "condition": q.get("condition", ""),
                "question": q.get("question", ""),
                "reference_answer": q.get("reference_answer", ""),
                "clause_ids": cids, "group_id": q.get("group_id", ""),
                "manual_reviewed": q.get("manual_reviewed", False),
            })

    checklist = paths["annotations"] / "review_checklist.txt"
    checklist.write_text("\n".join(q.get("qid", "") for q in questions) + "\n", encoding="utf-8")

    print("=== review packet ===")
    print(f"  questions     : {stats['total']} (dev {stats['dev']} / test {stats['test']})")
    print(f"  types         : {stats['types']}")
    print(f"  domains       : {stats['domains']}")
    print(f"\nwrote {review_md}")
    print(f"wrote {csv_path}")
    print(f"wrote {checklist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
