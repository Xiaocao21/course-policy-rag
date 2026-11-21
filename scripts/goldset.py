"""goldset.py - the single implementation of gold-span resolution and validation.

Every consumer of the gold annotation goes through this module:
    05_audit_splits.py      split/group discipline + leakage checks
    08_prefill_candidates.py candidate generation (needs the same span arithmetic)
    09_annotate_server.py   live validation while annotating
    12_evaluate.py          containment tests when scoring retrieval

Keeping one implementation matters: the protocol defines a clause as covered only when
the union of retrieved offsets from the correct document version *contains the entire*
annotated span. Three slightly different re-implementations of that rule would drift,
and the drift would be invisible in the metrics.

Vocabulary
    evidence layer   character-faithful text of a document; all spans live here
    clause           a record from cleaned/<doc_id>.evidence.jsonl with char_start/char_end
    sufficient set   one minimal set of clauses that answers a question
    span             (start, end) half-open character range in the evidence layer

Public API
    load_corpus(root)                        -> Corpus
    evidence_text(corpus, doc_id)            -> str
    resolve_clause_ref(ref, corpus)          -> (clause|None, problems)
    validate_question(q, corpus)             -> problems
    resolve_evidence_set(q, corpus, i)       -> [clause]
    iter_gold_spans(q, corpus)               -> [(set_index, clause, span)]
    span_of(ref)                             -> (start, end)
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    contains_span, covered_token_fraction, project_paths, read_json, read_jsonl,
    sha256_text,
)

# Question fields the protocol requires.
REQUIRED_QUESTION_FIELDS = ("qid", "question", "group_id", "split", "evidence_type",
                           "domain", "source_level", "evidence_sets")
# `test2` is the second held-out set. It lives in its own questions file and its own lock
# so the original test questions - and the hash that guards their one formal use - are not
# touched.
VALID_SPLITS = ("dev", "test", "test2")
HELD_OUT_SPLITS = ("test", "test2")


@dataclass
class Clause:
    doc_id: str
    clause_id: str
    char_start: int
    char_end: int
    text: str
    tier: str = ""
    institution: str = ""
    section_path: str = ""
    clause_label: str = ""
    block_type: str = ""
    has_exception: bool = False
    has_crossref: bool = False
    has_alt_evidence: bool = False
    boilerplate: bool = False
    evidence_sha256: str = ""

    @property
    def span(self) -> tuple[int, int]:
        return self.char_start, self.char_end

    @property
    def char_len(self) -> int:
        return self.char_end - self.char_start


@dataclass
class Corpus:
    root: Path
    corpus_version: str = ""
    doc_tier: dict[str, str] = field(default_factory=dict)
    doc_institution: dict[str, str] = field(default_factory=dict)
    clauses: dict[str, Clause] = field(default_factory=dict)          # clause_id -> Clause
    by_doc: dict[str, list[Clause]] = field(default_factory=dict)     # doc_id -> clauses
    _evidence_cache: dict[str, str] = field(default_factory=dict)

    @property
    def doc_ids(self) -> set[str]:
        return set(self.by_doc)

    def clauses_for(self, doc_id: str) -> list[Clause]:
        return self.by_doc.get(doc_id, [])


# --------------------------------------------------------------------- loading
def _load_evidence(paths: dict, doc_id: str) -> str:
    """Evidence text = retained blocks joined by newlines (exactly what spans index).

    Verified empirically: for every document in the corpus, joining the extracted
    blocks with "\\n" reproduces the text such that slicing a clause span and hashing
    it equals the stored evidence_sha256.
    """
    ext = paths["extracted"] / f"{doc_id}.json"
    if not ext.exists():
        return ""
    data = read_json(ext)
    return "\n".join(b.get("text", "") for b in data.get("blocks", []))


def classify_domain(text: str, taxonomy: dict, context: str = "") -> tuple[str, int]:
    """Return (best policy domain, keyword hit count) for a piece of text.

    `context` (usually the section path) is searched too and its hits are weighted
    once: a clause under "Attendance" belongs to attendance even when the clause
    itself repeats few domain keywords. Shared so the feasibility report, the
    candidate generator and any later analysis agree on the assignment.
    """
    low = text.lower()
    ctx = (context or "").lower()
    best, best_n = "", 0
    for name, spec in taxonomy.get("domains", {}).items():
        n = sum(low.count(kw.lower()) for kw in spec.get("keywords", []))
        if ctx:
            n += sum(1 for kw in spec.get("keywords", []) if kw.lower() in ctx)
        if n > best_n:
            best, best_n = name, n
    return best, best_n


def load_corpus(root: Path, doc_ids: Iterable[str] | None = None) -> Corpus:
    """Load the frozen corpus: documents with include=TRUE in sources.csv.

    A caller may pass an explicit doc_id list to override (tests, subsets). Documents
    that have no cleaned clause file are simply absent, which the validators report.
    """
    root = Path(root)
    paths = project_paths(root)
    corpus = Corpus(root=root)

    manifest_path = root / "MANIFEST.json"
    if manifest_path.exists():
        corpus.corpus_version = read_json(manifest_path).get("corpus_version", "")

    wanted: set[str] | None = set(doc_ids) if doc_ids is not None else None
    sources_csv = root / "sources.csv"
    if sources_csv.exists():
        with sources_csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                doc_id = row.get("doc_id", "")
                if not doc_id:
                    continue
                if str(row.get("include", "")).upper() not in ("TRUE", "YES", "1"):
                    continue
                if wanted is not None and doc_id not in wanted:
                    continue
                corpus.doc_tier[doc_id] = row.get("tier", "")
                corpus.doc_institution[doc_id] = row.get("institution", "")

    if wanted is not None and not corpus.doc_tier:
        # explicit ids without sources.csv rows: still load them
        for d in wanted:
            corpus.doc_tier.setdefault(d, "")

    for doc_id in list(corpus.doc_tier):
        ev = paths["cleaned"] / f"{doc_id}.evidence.jsonl"
        if not ev.exists():
            continue
        rows = read_jsonl(ev)
        corpus.by_doc[doc_id] = []
        for r in rows:
            c = Clause(
                doc_id=doc_id,
                clause_id=r["clause_id"],
                char_start=int(r["char_start"]),
                char_end=int(r["char_end"]),
                text=r.get("text", ""),
                tier=corpus.doc_tier.get(doc_id, r.get("tier", "")),
                institution=corpus.doc_institution.get(doc_id, r.get("institution", "")),
                section_path=r.get("section_path", ""),
                clause_label=r.get("clause_label", ""),
                block_type=r.get("block_type", ""),
                has_exception=bool(r.get("has_exception")),
                has_crossref=bool(r.get("has_crossref")),
                has_alt_evidence=bool(r.get("has_alt_evidence")),
                boilerplate=bool(r.get("boilerplate")),
                evidence_sha256=r.get("evidence_sha256", ""),
            )
            corpus.clauses[c.clause_id] = c
            corpus.by_doc[doc_id].append(c)
        if not corpus.by_doc[doc_id]:
            corpus.by_doc.pop(doc_id)

    return corpus


def evidence_text(corpus: Corpus, doc_id: str) -> str:
    if doc_id not in corpus._evidence_cache:
        corpus._evidence_cache[doc_id] = _load_evidence(project_paths(corpus.root), doc_id)
    return corpus._evidence_cache[doc_id]


# --------------------------------------------------------------------- refs
def span_of(ref: dict) -> tuple[int, int]:
    try:
        return int(ref.get("char_start", -1)), int(ref.get("char_end", -1))
    except (TypeError, ValueError):
        return -1, -1


def resolve_clause_ref(ref: dict, corpus: Corpus) -> tuple[Clause | None, list[str]]:
    """Validate one clause reference against the corpus. Returns (clause, problems)."""
    problems: list[str] = []
    cid = ref.get("clause_id") or ""
    if not cid:
        return None, ["clause_id is missing"]
    clause = corpus.clauses.get(cid)
    if clause is None:
        return None, [f"unknown clause_id {cid!r} (not in the frozen corpus)"]

    if ref.get("doc_id") and ref["doc_id"] != clause.doc_id:
        problems.append(f"doc_id mismatch for {cid}: ref={ref['doc_id']} corpus={clause.doc_id}")

    start, end = span_of(ref)
    if start < 0 or end < 0 or end <= start:
        problems.append(f"{cid}: invalid span ({start}, {end})")
        return clause, problems

    text = evidence_text(corpus, clause.doc_id)
    if not text:
        problems.append(f"{cid}: evidence text missing for {clause.doc_id}")
        return clause, problems
    if start < 0 or end > len(text):
        problems.append(f"{cid}: span ({start}, {end}) outside evidence length {len(text)}")
        return clause, problems

    quote = ref.get("quote_evidence", "")
    if quote is not None and quote != "":
        actual = text[start:end]
        if actual != quote:
            problems.append(
                f"{cid}: quote_evidence does not match evidence[{start}:{end}] "
                f"(expected {actual[:60]!r}, got {str(quote)[:60]!r})")

    # the corpus-side clause span must itself be sane
    if clause.char_end > len(text):
        problems.append(f"{cid}: corpus clause span exceeds evidence length")
    elif clause.evidence_sha256 and sha256_text(text[clause.char_start:clause.char_end]) != clause.evidence_sha256:
        problems.append(f"{cid}: corpus clause span hash does not match its text")

    return clause, problems


def resolve_evidence_set(question: dict, corpus: Corpus, set_index: int) -> list[Clause]:
    """Return the clauses of one sufficient set (skipping unresolvable refs)."""
    sets = question.get("evidence_sets") or []
    if not (0 <= set_index < len(sets)):
        return []
    out: list[Clause] = []
    for ref in sets[set_index].get("clauses") or []:
        clause, _problems = resolve_clause_ref(ref, corpus)
        if clause is not None:
            out.append(clause)
    return out


def iter_gold_spans(question: dict, corpus: Corpus) -> Iterator[tuple[int, Clause, tuple[int, int]]]:
    """Yield (set_index, clause, span) for every resolvable clause reference."""
    for si, eset in enumerate(question.get("evidence_sets") or []):
        for ref in eset.get("clauses") or []:
            clause, _ = resolve_clause_ref(ref, corpus)
            if clause is None:
                continue
            yield si, clause, span_of(ref)


# --------------------------------------------------------------------- questions
def validate_question(question: dict, corpus: Corpus, *,
                      valid_evidence_types: Iterable[str] | None = None,
                      valid_domains: Iterable[str] | None = None,
                      valid_levels: Iterable[str] | None = None,
                      require_manual_review_for_test: bool = True) -> list[str]:
    """Return every problem with one question. Empty list means it is well formed."""
    problems: list[str] = []
    qid = question.get("qid") or "<missing qid>"

    for field_name in REQUIRED_QUESTION_FIELDS:
        if field_name not in question or question[field_name] in (None, ""):
            problems.append(f"{qid}: missing required field {field_name!r}")

    if question.get("split") not in VALID_SPLITS:
        problems.append(f"{qid}: split must be one of {VALID_SPLITS}, got "
                        f"{question.get('split')!r}")

    # held-out questions may only enter the split once a human has verified them
    if (require_manual_review_for_test and question.get("split") in HELD_OUT_SPLITS
            and question.get("prefilled") and not question.get("manual_reviewed")):
        problems.append(
            f"{qid}: a prefilled question may not enter the test split (or any other "
            f"held-out split) until manual_reviewed is true")

    if valid_evidence_types is not None and question.get("evidence_type") not in set(valid_evidence_types):
        problems.append(f"{qid}: unknown evidence_type {question.get('evidence_type')!r}")
    if valid_domains is not None and question.get("domain") not in set(valid_domains):
        problems.append(f"{qid}: unknown domain {question.get('domain')!r}")
    if valid_levels is not None and question.get("source_level") not in set(valid_levels):
        problems.append(f"{qid}: unknown source_level {question.get('source_level')!r}")

    sets = question.get("evidence_sets") or []
    if not sets:
        problems.append(f"{qid}: no evidence_sets")
        return problems

    for si, eset in enumerate(sets):
        refs = eset.get("clauses") or []
        if not refs:
            problems.append(f"{qid}: sufficient set {si} has no clauses")
            continue
        for ci, ref in enumerate(refs):
            _clause, ref_problems = resolve_clause_ref(ref, corpus)
            for p in ref_problems:
                problems.append(f"{qid}: set {si} clause {ci}: {p}")

    return problems


# --------------------------------------------------------------------- coverage
@dataclass
class CoverageResult:
    """Outcome of checking one question's evidence against retrieved spans."""

    full_sets: list[int] = field(default_factory=list)     # set indices fully covered
    clause_full: dict[int, list[bool]] = field(default_factory=dict)
    recall: float = 0.0
    hit: bool = False
    complete: bool = False
    partial: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "recall": self.recall,
            "hit": self.hit,
            "complete": self.complete,
            "partial_span_coverage": self.partial,
            "fully_covered_sets": self.full_sets,
        }


def score_question(question: dict, corpus: Corpus,
                   retrieved: list[tuple[str, int, int]]) -> CoverageResult:
    """Score one question against retrieved (doc_id, start, end) spans.

    Implements the protocol exactly:
      * a clause is covered only if the union of retrieved spans *from the same
        document* contains its entire annotated span
      * Recall@k with several sufficient sets = the largest covered-clause share
        over sets; with one set it is that set's share
      * Hit = 1 when any required clause in any set is fully covered
      * Complete evidence = 1 only when one whole sufficient set is covered
      * partial_span_coverage = fraction of annotated characters covered, reported
        separately and never promoted to a full hit
    """
    result = CoverageResult()
    sets = question.get("evidence_sets") or []
    if not sets:
        return result

    by_doc: dict[str, list[tuple[int, int]]] = {}
    for doc_id, start, end in retrieved:
        by_doc.setdefault(doc_id, []).append((start, end))

    set_shares: list[float] = []
    partials: list[float] = []

    for si, eset in enumerate(sets):
        flags: list[bool] = []
        shares: list[float] = []
        for ref in eset.get("clauses") or []:
            clause, _ = resolve_clause_ref(ref, corpus)
            if clause is None:
                flags.append(False)
                shares.append(0.0)
                continue
            covers = by_doc.get(clause.doc_id, [])
            span = span_of(ref)
            text = evidence_text(corpus, clause.doc_id)
            full = contains_span(text, covers, span)
            frac = covered_token_fraction(text, covers, span)
            flags.append(full)
            shares.append(1.0 if full else 0.0)
            partials.append(frac)
        result.clause_full[si] = flags
        share = sum(shares) / len(shares) if shares else 0.0
        set_shares.append(share)
        if flags and all(flags):
            result.full_sets.append(si)

    result.recall = max(set_shares) if set_shares else 0.0
    result.hit = any(any(flags) for flags in result.clause_full.values() if flags)
    result.complete = bool(result.full_sets)
    result.partial = (sum(partials) / len(partials)) if partials else 0.0
    return result


def test_group_counts(questions: list[dict]) -> dict[str, Any]:
    """Group/split bookkeeping used by both the audit and the workbench.

    Splits are aggregated dynamically so a second held-out set (`test2`) is counted and
    checked like any other split. `dev` and `test` keys are always present, so existing
    callers and printouts keep working unchanged.
    """
    seen_splits = {q.get("split") for q in questions if q.get("split") in VALID_SPLITS}
    ordered = ["dev", "test"] + [s for s in VALID_SPLITS if s not in ("dev", "test")
                                 and s in seen_splits]
    by_split: dict[str, list[dict]] = {s: [] for s in ordered}
    for q in questions:
        split = q.get("split")
        if split in by_split:
            by_split[split].append(q)
    groups = {s: {str(q.get("group_id")) for q in qs} for s, qs in by_split.items()}
    # a group must never straddle two splits, whichever pair it is
    all_groups: dict[str, set[str]] = {}
    for split, gs in groups.items():
        for g in gs:
            all_groups.setdefault(g, set()).add(split)
    crossing = {g for g, ss in all_groups.items() if len(ss) > 1}
    return {
        "counts": {k: len(v) for k, v in by_split.items()},
        "groups": {k: len(v) for k, v in groups.items()},
        "crossing_groups": sorted(crossing),
        "manual_reviewed": sum(1 for q in questions if q.get("manual_reviewed")),
        "prefilled": sum(1 for q in questions if q.get("prefilled")),
    }
