"""Verify the clause size bound worked and check current corpus balance."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

out = {}
rows = list(csv.DictReader((ROOT / "annotations" / "gold_worksheet.csv").open(encoding="utf-8")))
lens = sorted(int(r["char_len"]) for r in rows)
def pct(p):
    return lens[min(len(lens) - 1, int(len(lens) * p))]
out["n_clauses"] = len(rows)
out["char_len"] = {"min": lens[0], "median": pct(.5), "p90": pct(.90),
                   "p99": pct(.99), "max": lens[-1]}
for b in (1200, 2000, 5000):
    out[f"over_{b}"] = sum(1 for L in lens if L > b)
out["under_80"] = sum(1 for L in lens if L < 80)

# curation tier balance
cur = list(csv.DictReader((ROOT / "reports" / "curation_decisions.csv").open(encoding="utf-8")))
chosen = [r for r in cur if r["in_corpus"].lower() == "true"]
tc = Counter(r["tier"] for r in chosen)
n = len(chosen)
out["curated"] = {"n": n, "tiers": dict(tc),
                  "shares": {t: round(c / n, 4) for t, c in tc.items()}}

# corpus membership match
src = {r["doc_id"]: r for r in csv.DictReader((ROOT / "sources.csv").open(encoding="utf-8"))}
inc = {d for d, r in src.items() if r["include"] == "TRUE"}
corpus = {json.loads(l)["doc_id"]
          for l in (ROOT / "corpus_safe_512" / "corpus_safe_512_approx.jsonl").open(encoding="utf-8")}
out["membership_match"] = inc == corpus

# acceptance gates vs settings
cap = list(csv.DictReader((ROOT / "reports" / "clause_stats.csv").open(encoding="utf-8")))
byd = {r["doc_id"]: r for r in cap}
out["docs_with_exception"] = sum(1 for d in corpus if int(byd[d]["clauses_exception"]) > 0)
out["docs_with_crossref"] = sum(1 for d in corpus if int(byd[d]["clauses_crossref"]) > 0)
out["docs_with_alt"] = sum(1 for d in corpus if int(byd[d]["clauses_alt"]) > 0)
out["corpus_clauses"] = sum(int(byd[d]["clauses_total"]) for d in corpus)
out["institutions"] = len({src[d]["institution"] for d in corpus})

(ROOT / "reports" / "_spancheck.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("written")
