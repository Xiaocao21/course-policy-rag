# test2 self-audit (agent)

Scope: a second-pass read of all 45 questions against their frozen clause text. This is an
**agent self-audit**, not the student sign-off that manual_reviewed is meant to record.

Questions: 45  ·  approved: 45  ·  rejected: 0  ·  revised during audit: 2

## What was checked, per question

| check | result |
|---|---|
| question answerable from the cited clause alone | 45/45 |
| reference answer entailed by the clause, no outside facts | 45/45 |
| evidence set minimal (one clause, and the clause is the gold unit) | 45/45 |
| evidence_type single-clause correct | 45/45 |
| version applicable (v0.1.5, quote verbatim) | 45/45 |
| near-duplicate check (Jaccard >= 0.85 against 172 existing + within test2) | 45/45 clean |

## Revisions made during the audit

* T2Q003 — the answer restated the definition of "Admitted"; trimmed to the question asked.
* T2Q021 — the answer stated the petition content without the clause's "better chance of
  being approved" nuance; restored.

## Frozen-span residue (not entailment failures)

* T2Q001 — frozen span also carries BU contact/website boilerplate; question uses only the rule sentence
* T2Q003 — answer tightened during self-audit to the question asked
* T2Q012 — clause also lists committee composition; question targets the Chair vote only
* T2Q018 — clause opens with 1977/2002 historical framing; answer quotes the definition
* T2Q021 — answer tightened during self-audit; keeps the "better chance of approval" nuance
* T2Q030 — frozen clause includes a course-code list after the appeal rule; span is the whole clause
* T2Q036 — clause begins with examiner names; answer uses the definition that follows
* T2Q043 — long clause (3.7k chars); question targets paragraphs (a) and (b)

## Limit of this audit

An agent can verify that an answer follows from the quoted text and that no outside fact was
introduced. It cannot supply the independent judgement that a human reviewer provides (is this
a question a student would really ask; is the citation the one a human would choose). If the
split is signed off on the strength of this audit, the report must say that test2 was
agent-reviewed and not human-reviewed.