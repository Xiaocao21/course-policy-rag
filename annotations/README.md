# Annotations

| file | contents |
|---|---|
| `questions.jsonl` | first question set: 172 questions (112 dev / 60 test, 59 test groups) |
| `questions_test2.jsonl` | second held-out set: 45 questions / 45 groups |
| `LOCKED/LOCKED.json`, `LOCKED_TEST2/LOCKED2.json` | lock metadata (question hashes, group counts) |
| `TEST2_REVIEW.csv` | the 45-row review sheet; required by the test2 pre-registration hash |
| `TEST2_SELF_AUDIT.md` | what the agent self-audit checked |
| `TEST2_REVIEW_ATTESTATION.md` | student's joint-review attestation |
| `QUOTE_PROVENANCE.csv` | every `quote_evidence` clause mapped to document, URL and recorded licence note |
| `SCHEMA.md`, `split_audit.json`, `repair_patch.json`, `used_clause_ids.txt` | schema, split audit and repair record |

`REVIEW.md`, `REVIEW.csv`, `gold_worksheet.csv`, the pre-repair backup, the skeleton file and
the test2 review Markdown are **not** published: they repeat full clause text already present in
the question files. The questions themselves keep `quote_evidence` because it is the evidence
the reference answer is checked against; `QUOTE_PROVENANCE.csv` states the source and recorded
licence status of every quotation. See `NOTICE.md` for the takedown route.
