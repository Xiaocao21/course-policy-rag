"""10d_repair_questions.py - apply the reviewer's findings to the question set.

Three repairs, in order.

1. **Content** (from `annotations/repair_patch.json`): wrong answers, evidence sets that
   were not minimal or not sufficient, and questions whose pairing was unnatural are
   replaced with versions supported by the cited clauses.

2. **Source qualifiers.** The corpus spans several institutions and academic years, so a
   question with no source named has no unique answer. Where the question text does not
   already name the institution (or, for course-level documents, the course and term),
   a natural qualifier is prefixed.

3. **Grouping and splits.** Two questions that cite the same gold clause are variants of
   the same evidence situation: they receive **one group_id** and are kept in **one
   split**. Components are computed by union-find over shared clause ids, which removes
   the tuning/test leakage the reviewer found (20 clauses used on both sides). Splits are
   then reassigned component-wise, keeping roughly two thirds in development while
   preserving at least 40 independent test groups.

Outputs
    annotations/questions.jsonl        repaired set
    reports/repair_report.json         what changed and the resulting distributions
    annotations/used_clause_ids.txt    refreshed

Usage
    .venv\\Scripts\\python.exe scripts/10d_repair_questions.py --dry-run
    .venv\\Scripts\\python.exe scripts/10d_repair_questions.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, project_paths, read_jsonl, write_json, write_jsonl,
)
import goldset  # noqa: E402

ALIASES = {
    "University of Glasgow": ["glasgow"],
    "Boston University": ["boston university", "bu "],
    "MIT OpenCourseWare": ["mit", "14.0", "15.0", "16.0", "1.00", "10.37", "6.0"],
    "Stanford University": ["stanford"],
    "Imperial College London": ["imperial"],
    "Carnegie Mellon University": ["carnegie mellon", "cmu"],
    "University of Wisconsin-Madison": ["wisconsin"],
    "University of Washington": ["washington"],
    "University of Maryland": ["maryland"],
    "University of Florida": ["florida"],
    "University of Manchester": ["manchester"],
    "University of North Carolina": ["north carolina", "unc"],
    "University of Nottingham": ["nottingham"],
}

CODE_RE = re.compile(r"COURSES-(\d+)-(\d+)-")
TERM_RE = re.compile(r"(?i)\b(fall|spring|summer|winter|january|iap)-(\d{4})\b")
QUESTION_STARTERS = {"what", "which", "can", "do", "does", "is", "are", "how", "may",
                     "must", "if", "could", "should", "where", "when", "who", "why",
                     "will", "would", "am", "in", "at", "our", "my", "no", "not", "there"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def course_label(doc_id: str, url: str = "") -> str | None:
    """Course code plus term.

    doc_id values were truncated when generated, so the term is often cut off; the
    source URL still carries it (e.g. .../spring-2004/...), so that is the fallback.
    """
    m = CODE_RE.search(doc_id)
    if not m:
        return None
    code = f"{m.group(1)}.{m.group(2)}"
    t = TERM_RE.search(doc_id) or TERM_RE.search(url or "")
    term = f"{t.group(1).capitalize()} {t.group(2)}" if t else ""
    return f"{code} ({term})" if term else code


def mentions_source(question: str, institution: str, doc_id: str, url: str = "") -> bool:
    """Does the question already name its source?

    Aliases match on word boundaries: a plain substring test let the MIT alias "mit"
    match inside ordinary words such as "submit", which made unsourced questions look
    sourced and silently skipped the qualifier.
    """
    q = question.lower()
    if course_label(doc_id, url):
        m = CODE_RE.search(doc_id)
        if m and f"{m.group(1)}.{m.group(2)}" in question:
            return True
    for alias in ALIASES.get(institution, []):
        a = alias.strip().lower()
        if a and re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", q):
            return True
    if institution and institution.lower() in q:
        return True
    return False


def add_qualifier(question: str, institution: str, doc_id: str, url: str = "") -> str:
    label = course_label(doc_id, url)
    prefix = f"In {label}, " if label else f"At {institution}, "
    rest = question.strip()
    if not rest:
        return prefix.rstrip(", ") + "?"
    words = rest.split()
    first = re.sub(r"[^A-Za-z]", "", words[0]).lower()
    if first in QUESTION_STARTERS:
        words[0] = words[0][0].lower() + words[0][1:]
    return prefix + " ".join(words)


def build_ref(corpus: goldset.Corpus, clause_id: str) -> dict:
    clause = corpus.clauses[clause_id]
    text = goldset.evidence_text(corpus, clause.doc_id)
    return {
        "doc_id": clause.doc_id, "version": corpus.corpus_version,
        "clause_id": clause.clause_id, "section_path": clause.section_path,
        "char_start": clause.char_start, "char_end": clause.char_end,
        "quote_evidence": text[clause.char_start:clause.char_end],
        "token_start": None, "token_end": None,
    }


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--patch", default="annotations/repair_patch.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-share", type=float, default=1 / 3)
    ap.add_argument("--group-similarity", type=float, default=0.25,
                    help="question Jaccard above which two questions in the same evidence "
                         "component share a group_id (paraphrase variants)")
    ap.add_argument("--keep-splits", action="store_true",
                    help="do not reassign splits; only regroup and verify (used when new "
                         "questions were ingested with an intended split)")
    ap.add_argument("--min-test-groups", type=int, default=40)
    ap.add_argument("--test-group-ratio", type=float, default=0.82,
                    help="target test groups as a share of test questions (<1 groups variants)")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    corpus = goldset.load_corpus(root)
    questions = read_jsonl(paths["annotations"] / "questions.jsonl")
    by_qid = {q["qid"]: q for q in questions}

    doc_url: dict[str, str] = {}
    sources_csv = root / "sources.csv"
    if sources_csv.exists():
        import csv as _csv
        with sources_csv.open(encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                doc_url[row.get("doc_id", "")] = row.get("url", "")

    patch_path = Path(args.patch)
    if not patch_path.is_absolute():
        patch_path = root / patch_path
    patch = json.loads(patch_path.read_text(encoding="utf-8"))

    report: dict = {"at": now(), "patched": [], "qualifiers_added": [],
                    "regrouped": 0, "splits_changed": 0}

    # ---------------------------------------------------------------- 1. content
    deleted: list[str] = []
    for qid, changes in patch.items():
        if qid.startswith("_"):
            continue
        q = by_qid.get(qid)
        if q is None:
            report["patched"].append({"qid": qid, "status": "MISSING"})
            continue
        if changes.get("delete"):
            deleted.append(qid)
            report["patched"].append({"qid": qid, "status": "deleted",
                                      "note": changes.get("_note", "")})
            continue
        applied = []
        for field in ("question", "reference_answer", "evidence_type", "condition"):
            if field in changes and changes[field] != q.get(field):
                q[field] = changes[field]
                applied.append(field)
        if "clause_ids" in changes:
            ids = changes["clause_ids"]
            q["evidence_sets"] = [{"set_id": "S1", "clauses": [build_ref(corpus, c) for c in ids]}]
            q["evidence_type"] = changes.get("evidence_type",
                                             "single-clause" if len(ids) == 1 else "cross-clause")
            applied.append("clause_ids")
        first = q["evidence_sets"][0]["clauses"][0]
        q["source_level"] = corpus.doc_tier.get(first["doc_id"], q.get("source_level", ""))
        q.setdefault("provenance", {})["repaired_at"] = now()
        if changes.get("_note"):
            q["provenance"]["repair_note"] = changes["_note"]
        report["patched"].append({"qid": qid, "status": "ok", "fields": applied})

    if deleted:
        drop = set(deleted)
        questions = [q for q in questions if q["qid"] not in drop]
        report["deleted"] = deleted

    # ---------------------------------------------------------------- 2. qualifiers
    for q in questions:
        ref = q["evidence_sets"][0]["clauses"][0]
        doc_id = ref["doc_id"]
        inst = corpus.doc_institution.get(doc_id, "")
        if not inst:
            continue
        label = course_label(doc_id, doc_url.get(doc_id, ""))
        if label and "(" in label:
            # course-level sources also carry a term; a bare course code does not
            # identify which year's syllabus the answer comes from
            code, term = label.split(" (", 1)
            term = term.rstrip(")")
            if code in q["question"] and term not in q["question"]:
                q["question"] = q["question"].replace(code, f"{code} ({term})", 1)
                report["qualifiers_added"].append({"qid": q["qid"], "institution": inst,
                                                   "added": f"term {term}"})
                continue
        if not mentions_source(q["question"], inst, doc_id, doc_url.get(doc_id, "")):
            before = q["question"]
            q["question"] = add_qualifier(before, inst, doc_id, doc_url.get(doc_id, ""))
            report["qualifiers_added"].append({"qid": q["qid"], "institution": inst,
                                               "before": before[:80],
                                               "after": q["question"][:80]})

    # ---------------------------------------------------------------- 3. grouping
    uf = UnionFind(len(questions))
    owners: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(questions):
        for es in q["evidence_sets"]:
            for r in es["clauses"]:
                owners[r["clause_id"]].append(i)
    for cid, idxs in owners.items():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    # ---- grouping: two different relations, deliberately kept apart -------------
    # Components (shared gold clause) are a *split* constraint: they must not be
    # divided across dev/test, or the held-out score is contaminated.
    # A group_id, by contrast, means "paraphrase / variant of the same template".
    # Equating the two produced groups of 11 and 13 questions that were mere evidence
    # tangles rather than template families, so groups are now clustered inside a
    # component by question wording.
    comp_of: dict[int, int] = {}
    for i in range(len(questions)):
        comp_of[i] = uf.find(i)
    comps: dict[int, list[int]] = defaultdict(list)
    for i, c in comp_of.items():
        comps[c].append(i)
    ordered_comps = sorted(comps.values(), key=lambda idxs: questions[min(idxs)]["qid"])

    from common import jaccard as _jaccard, shingles as _shingles
    sh = {i: _shingles(questions[i]["question"]) for i in range(len(questions))}

    n_groups = 0
    for idxs in ordered_comps:
        # greedy clustering by question similarity inside the component
        assigned: set[int] = set()
        for i in idxs:
            if i in assigned:
                continue
            n_groups += 1
            gid = f"G{n_groups:03d}"
            cluster = [i]
            assigned.add(i)
            for j in idxs:
                if j in assigned:
                    continue
                if _jaccard(sh[i], sh[j]) >= args.group_similarity:
                    cluster.append(j)
                    assigned.add(j)
            for k in cluster:
                if questions[k].get("group_id") != gid:
                    report["regrouped"] += 1
                questions[k]["group_id"] = gid

    # ---------------------------------------------------------------- 4. splits
    if args.keep_splits:
        test_ids = {i for i, q in enumerate(questions) if q.get("split") == "test"}
        comp_info = []
        for idxs in ordered_comps:
            dom = Counter(questions[i]["domain"] for i in idxs).most_common(1)[0][0]
            comp_info.append({"idxs": idxs, "size": len(idxs), "domain": dom,
                              "gid": questions[idxs[0]]["group_id"]})
        test_components = [c for c in comp_info if c["idxs"][0] in test_ids]
    else:
            # Stratified, group-aware assignment in three phases:
        #   A. guarantee every domain appears in the test split (singletons first)
        #   B. add a few two-question clusters so the test set exercises the grouping rule
        #   C. top up with singletons to reach the question target
        # Components are never split, so no gold clause and no group crosses the boundary.
        target_test = int(round(len(questions) * args.test_share))
        target_groups = int(round(target_test * args.test_group_ratio))
        comp_info = []
        for idxs in ordered_comps:
            dom = Counter(questions[i]["domain"] for i in idxs).most_common(1)[0][0]
            comp_info.append({"idxs": idxs, "size": len(idxs), "domain": dom,
                              "gid": questions[idxs[0]]["group_id"]})

        total_by_dom = Counter(questions[i]["domain"] for i in range(len(questions)))
        singletons = [c for c in comp_info if c["size"] == 1]
        pairs = [c for c in comp_info if c["size"] == 2]
        larger = [c for c in comp_info if c["size"] > 2]

        test_components: list[dict] = []
        chosen_ids: set[int] = set()
        test_dom = Counter()

        def take(info: dict) -> None:
            test_components.append(info)
            chosen_ids.add(id(info))
            test_dom[info["domain"]] += info["size"]

        # Phase A: per-domain floor, singletons only, smallest-domain-share first
        for dom in sorted(total_by_dom, key=lambda d: total_by_dom[d]):
            floor = max(3, int(round(total_by_dom[dom] * args.test_share * 0.6)))
            for info in sorted([c for c in singletons if c["domain"] == dom],
                               key=lambda c: c["gid"]):
                if test_dom[dom] >= floor:
                    break
                if id(info) in chosen_ids:
                    continue
                take(info)

        # Phase B: add two-question clusters until the group target is reached
        for info in sorted(pairs, key=lambda c: (c["domain"], c["gid"])):
            if len(test_components) >= target_groups or sum(
                    c["size"] for c in test_components) >= target_test:
                break
            take(info)

        # Phase C: top up to the question target with singletons, keeping domains balanced
        for info in sorted(singletons, key=lambda c: (total_by_dom[c["domain"]], c["gid"])):
            if sum(c["size"] for c in test_components) >= target_test:
                break
            if id(info) in chosen_ids:
                continue
            cap = max(4, int(round(total_by_dom[info["domain"]] * args.test_share)) + 3)
            if test_dom[info["domain"]] + 1 > cap:
                continue
            take(info)

        # Phase D: if still short of the group floor, add any remaining components
        for info in sorted(larger, key=lambda c: (c["size"], c["gid"])):
            if len(test_components) >= args.min_test_groups:
                break
            take(info)


    test_ids = {i for info in test_components for i in info["idxs"]}
    for i, q in enumerate(questions):
        new_split = "test" if i in test_ids else "dev"
        if q.get("split") != new_split:
            report["splits_changed"] += 1
        q["split"] = new_split

    # ---------------------------------------------------------------- verification
    clause_splits: dict[str, set[str]] = defaultdict(set)
    for q in questions:
        for es in q["evidence_sets"]:
            for r in es["clauses"]:
                clause_splits[r["clause_id"]].add(q["split"])
    leaking = [c for c, s in clause_splits.items() if len(s) > 1]

    test_groups = {q["group_id"] for q in questions if q["split"] == "test"}
    dev_groups = {q["group_id"] for q in questions if q["split"] == "dev"}
    overlap_groups = test_groups & dev_groups

    report.update({
        "questions": len(questions),
        "groups": len({q["group_id"] for q in questions}),
        "evidence_components": len(ordered_comps),
        "split_counts": dict(Counter(q["split"] for q in questions)),
        "group_counts": {"dev": len(dev_groups), "test": len(test_groups)},
        "clauses_shared_across_splits": len(leaking),
        "groups_spanning_splits": sorted(overlap_groups),
        "types": dict(Counter(q["evidence_type"] for q in questions)),
        "domains": dict(Counter(q["domain"] for q in questions)),
        "test_domains": dict(Counter(q["domain"] for q in questions if q["split"] == "test")),
    })

    if not args.dry_run:
        tmp = (paths["annotations"] / "questions.jsonl").with_suffix(".jsonl.tmp")
        write_jsonl(tmp, questions)
        tmp.replace(paths["annotations"] / "questions.jsonl")

        used = sorted({r["clause_id"] for q in questions
                       for es in q["evidence_sets"] for r in es["clauses"]})
        (paths["annotations"] / "used_clause_ids.txt").write_text(
            "\n".join(used) + "\n", encoding="utf-8")

        write_json(paths["reports"] / "repair_report.json", report)
        log = paths["reports"] / "lock_audit.jsonl"
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": now(), "event": "repair_questions",
                                "patched": len([p for p in report["patched"]
                                                if p["status"] == "ok"]),
                                "qualifiers": len(report["qualifiers_added"]),
                                "groups": report["groups"],
                                "clauses_crossing_splits": len(leaking)},
                               ensure_ascii=False) + "\n")

    print("=== repair ===")
    print(f"  content patches   : {len([p for p in report['patched'] if p['status']=='ok'])}")
    print(f"  qualifiers added  : {len(report['qualifiers_added'])}")
    print(f"  regrouped         : {report['regrouped']} questions into {report['groups']} groups")
    print(f"  splits changed    : {report['splits_changed']}")
    print(f"  split counts      : {report['split_counts']}")
    print(f"  group counts      : dev {report['group_counts']['dev']} / "
          f"test {report['group_counts']['test']} (need >= {args.min_test_groups})")
    print(f"  clauses crossing  : {len(leaking)}  (must be 0)")
    print(f"  groups crossing   : {len(overlap_groups)}  (must be 0)")
    print(f"  types             : {report['types']}")
    print(f"  test domains      : {report['test_domains']}")
    if args.dry_run:
        print("\n  dry run: nothing written")
    else:
        print(f"\nwrote {paths['annotations'] / 'questions.jsonl'}")
        print(f"wrote {paths['reports'] / 'repair_report.json'}")
    return 0 if not leaking and not overlap_groups else 1


if __name__ == "__main__":
    sys.exit(main())
