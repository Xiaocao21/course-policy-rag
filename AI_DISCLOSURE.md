# AI use disclosure

**Author:** Juncheng Qian (student, CPS5801). The author is responsible for every claim,
number and file in this repository.

## What the AI did

| Area | AI involvement | Human check |
|---|---|---|
| Pipeline code (`scripts/`, `src/`, `tools/`) | written with AI assistance (DeepSeek and Codex); the author directed the design and reviewed the code | the author can explain every module; tests and assertions cover the transformations |
| First question set (172 questions, 112 dev / 60 test) | drafted by the AI from frozen corpus clauses | **the student reviewed all 172**, raised 23 defects that were corrected, and accepted every question (timestamped in `reports/lock_audit.jsonl`) |
| Second held-out set (45 questions, test2) | drafted by the AI from frozen evidence bundles | **jointly reviewed by the student and the AI**; two answers were corrected (T2Q003, T2Q021). See `annotations/TEST2_REVIEW_ATTESTATION.md` |
| Evidence selection | AI proposed candidate clauses and bundles | the author froze the bundles before writing questions; the freeze is hashed and dated |
| Report and documentation text | drafted with AI assistance | reviewed and edited by the author |
| Results | produced only by executing the code | every number is traceable to a committed result file (`results/`, `results/HASHES.csv`) |

## What the AI did **not** do

* It did not run experiments on its own or invent numbers. Every reported value comes from a
  run whose inputs are hashed (`results/final*/FINAL_RUN.json`, `results/test2/OUTCOME.json`).
* It did not decide the research question, the evaluation protocol or the interpretation; those
  were fixed by the author and, where applicable, pre-registered before evaluation.
* No inter-annotator agreement is claimed: there was no second human annotator.

## Honest limits

* The first test split had been used for a pilot before the formal runs; those results are
  re-evaluations of a seen split (`REPORT` §5).
* test2 is a first-use split, but its questions were AI-drafted; the joint review is recorded
  in the attestation, and the machine record at lock time says `agent_self_audit` because that
  was the identity written before the joint review was added.
* The live corpus can no longer be rebuilt byte-identically (see `data/REBUILD_REPORT.md`);
  retrieval replay uses the author-provided snapshot and official indexes.

## Tools

The assistant was DeepSeek (via the DeepSeek Harness coding agent). Codex also assisted with
code development. These are the only AI systems used for code, questions or text in this
repository.
