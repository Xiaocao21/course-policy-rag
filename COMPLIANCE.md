# Compliance and provenance - course-policy-rag v0.1.5

Frozen at: 2026-10-21T18:46:45+0800
Documents in the frozen corpus: 36 (collected during this pass: 265)

## Data sources

Only publicly accessible web pages and PDFs are collected. No login, paywall or
credential is bypassed. No private student information is collected or stored.
Course-level material comes from open courseware published under an open licence;
institutional and faculty-level material is public policy text reproduced for
non-commercial academic research, with the source URL and retrieval timestamp kept
for every document in `sources.csv`.

### Licence notes in use

- 16 document(s): CC BY-NC-SA 4.0
- 249 document(s): public institutional policy text; academic research use

## Crawling policy

- User agent: `CPS5801-research-bot/0.1 (+academic research; contact: Xiaocao21@users.noreply.github.com)`
- Per-host delay: 1.5 s minimum, `Crawl-Delay` honoured when larger
- Global page cap per run: 400
- `robots.txt` is fetched once per host and evaluated before any request to that host.
  A disallowed URL is never requested, and the decision is recorded in
  `reports/robots_audit.json` together with the raw policy text.
- Hosts audited: 10

## Known access restrictions

The following sources are documented as manual-only because their robots policy
disallows the relevant paths, or the host is unreachable from the collection machine.
They are listed so that a reader can reproduce or extend the corpus by hand:

- `apps.ualberta.ca` - `Disallow: /catalogue/syllabus` (syllabus downloads disallowed)
- `ro.umich.edu` and `cse.engin.umich.edu` - blanket `Disallow: /`
- `www.qaa.ac.uk` - the quality-code path is disallowed
- `www.ucl.ac.uk` - examinations path disallowed
- Several university hosts resolve only from unrestricted networks and were
  unreachable from the collection machine; these are recorded as
  `unreachable` in `reports/seed_probe_round2.csv` rather than silently skipped.

## Tier composition (frozen corpus)

Selection is performed by `scripts/curate_corpus.py`; every inclusion and
exclusion decision is recorded in `reports/curation_decisions.csv`.

- course: 14 document(s)  (39%)
- faculty: 9 document(s)  (25%)
- institution: 13 document(s)  (36%)

Collected but not selected (retained as provenance): 229

### Documents per institution (frozen corpus)

- MIT OpenCourseWare: 14
- University of Glasgow: 9
- Stanford University: 3
- Boston University: 2
- Carnegie Mellon University: 1
- Imperial College London: 1
- University of Florida: 1
- University of Manchester: 1
- University of Maryland: 1
- University of North Carolina: 1
- University of Washington: 1
- University of Wisconsin-Madison: 1

## AI assistance and review status

The question set and gold evidence spans were drafted with AI assistance
(`scripts/10_ai_annotate.py`) from clauses in this frozen corpus, under an
arrangement approved by the project supervisor for this course assignment, with the
student responsible for review.

How that is recorded:

- Every question carries `provenance.generated_by = "ai-assisted"`,
  `provenance.model`, `provenance.generated_at` and
  `provenance.review_status`.
- Questions are written with `manual_reviewed = false`. The held-out test split
  refuses them until the student sets `manual_reviewed = true`
  (`scripts/05_audit_splits.py --mark-reviewed`), and that action is appended to
  `reports/lock_audit.jsonl` with a timestamp.
- `annotations/REVIEW.md` and `annotations/REVIEW.csv` are the review packet:
  every question with its reference answer and the verbatim clause text it cites.

This project is a course assignment rather than publishable data, so no
inter-annotator agreement is claimed for the AI-drafted set. If the results are
ever written up for publication, the annotation provenance above is what a reader
must be told.

## Reproducibility

`MANIFEST.json` records a SHA-256 for every raw file, extracted structure, cleaned
clause set and corpus artefact, plus the hashes of the settings and taxonomy that
produced them. Every chunking strategy in this study consumes the single corpus file
recorded under `chunk_stats.file`; therefore corpus differences cannot confound the
comparison. Re-running the pipeline with the same settings reproduces the same hashes.

## Validation state at freeze

- corpus validation: pass
- split audit: blocked
- chunks over model maximum: 0

---

## Addendum after the evaluation (not part of the freeze-time record)

This section was appended after `MANIFEST.json` was written; it does not alter any frozen
artefact or hash.

### Test-split exposure

The held-out split was evaluated once as an exploratory pilot on 2026-10-24
(`results/pilot/`, `run_test.json` status `pilot`). It was then evaluated again, once, as a
formal four-track run on 2026-10-28 with every setting frozen beforehand. The split is
therefore **no longer unseen**, and the formal run is reported as a re-evaluation. No claim
of a pristine holdout is made anywhere in the reports.

### AI assistance

* Questions and reference answers were drafted with AI assistance from corpus clauses; every
  question records `provenance.generated_by`, model and timestamp. The student reviewed all
  172 questions, raised 23 specific defects that were corrected, and accepted every question.
  The sign-off is timestamped in `reports/lock_audit.jsonl`.
* Pipeline code was written with AI assistance; data transformations are asserted and covered
  by 108 automated tests.
* `docs/TEACHER_BRIEF.docx` and the analysis narrative were drafted with AI assistance and
  reviewed by the student.
* No inter-annotator agreement is claimed: there was no second human annotator.
