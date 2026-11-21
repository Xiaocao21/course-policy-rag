"""quote_provenance.py - per-question provenance for the quoted evidence text.

`annotations/questions*.jsonl` contains `quote_evidence`, a short excerpt of the clause each
question cites. This tool maps every question to its document, source URL and the licence note
recorded at collection time, so the repository states exactly what text it exposes and on what
recorded basis.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "annotations" / "QUOTE_PROVENANCE.csv"


def sources() -> dict[str, dict]:
    rows = {}
    with (ROOT / "sources.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["doc_id"]] = row
    return rows


def status(note: str) -> str:
    note = (note or "").lower()
    if "cc by" in note:
        return "cc-licence-recorded"
    if note:
        return "recorded_unverified"
    return "unknown"


def main() -> int:
    src = sources()
    rows = []
    for split, name in (("test/dev", "questions.jsonl"), ("test2", "questions_test2.jsonl")):
        path = ROOT / "annotations" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            for clause in (q.get("evidence_sets") or [{}])[0].get("clauses", []):
                doc = clause.get("doc_id", "")
                meta = src.get(doc, {})
                note = meta.get("license_note", "")
                rows.append({
                    "qid": q["qid"], "split": split, "doc_id": doc,
                    "clause_id": clause.get("clause_id", ""),
                    "url": meta.get("final_url", ""),
                    "license_note": note,
                    "robots_allowed": meta.get("robots_allowed", ""),
                    "status": status(note),
                    "included_in_public": "yes",
                })
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} quotations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
