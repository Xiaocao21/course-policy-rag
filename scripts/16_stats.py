"""16_stats.py - group-level statistics for the four-track evaluation.

Why group level. The test split has 60 questions but only 59 groups, and every question is
evaluated under all 17 configurations. Treating questions, or (question, configuration)
pairs, as independent observations would understate the variance: the same paraphrase
family contributes correlated numbers, and the same question appears seventeen times. So
the resampling unit is `group_id`, configurations are first collapsed by the pre-registered
macro-average, and only then are paired comparisons made.

What is tested and what is not. Two contrasts are arithmetic identities, not hypotheses:

    C2 - main   reranking a fixed ten-chunk set cannot change Recall@10 / Hit@10 /
                Complete@10, because the set is the same set
    C1 - main   the first ten of a dense top-50 are the dense top-10, so the same holds

Those are reported as "invariance assertion passed" and never given a p-value. The
informative set-level comparisons are C3 - C1 (fixed pool, reranking can pull chunks 11-50
into the top ten) and C3 - C2 (pool effect with reranking active); the primary comparison is
pre-specified as C3 - C1 on Complete@10, with Holm correction over the two.

Order-sensitive quantities (MRR, Recall@1/3/5, token-budget recall) are informative for the
fixed-pool contrasts and are tested as separate families with Holm correction inside each.

Derived summaries the report reads instead of hard-coded numbers:

    by_strategy        one row per chunking family (fixed split by size), so the
                       per-strategy comparison is reported next to the macro average
    family_contrasts   structural (heading+paragraph) minus semantic, group-level,
                       paired bootstrap; this is the primary endpoint of a future
                       unseen set, because it is large enough to be detectable there
    by_length_tercile  short-vs-long clause-length contrast, symmetric by k so the
                       comparison carries no remainder-block artefact; the precision
                       estimate is a Welch t interval, with a bootstrap interval as a
                       secondary column (see the docstring of that block)

Usage
    .venv\\Scripts\\python.exe scripts/16_stats.py --split dev
    .venv\\Scripts\\python.exe scripts/16_stats.py --split test
    .venv\\Scripts\\python.exe scripts/16_stats.py --split test2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import PROJECT_ROOT, load_settings, project_paths, read_jsonl, write_json  # noqa: E402
import goldset  # noqa: E402

TRACKS = ("main", "C1", "C2", "C3")
SET_METRICS = ("recall@10", "hit@10", "complete@10")
ORDER_METRICS = ("mrr", "recall@1", "recall@3", "recall@5",
                 "tok512_recall", "tok1024_recall", "tok2048_recall")
PRIMARY = ("C3", "C1", "complete@10")
SECONDARY = ("C3", "C2", "complete@10")
STRUCTURAL_STRATEGIES = ("heading", "paragraph")
SEMANTIC_STRATEGY = "semantic"


def t_quantile_975(df: int) -> float:
    """Two-sided 95% t quantile via the Cornish-Fisher expansion.

    `scipy` is not a dependency of this project (the CPU environment installs only
    `requirements.txt`), and `wilcoxon()` below already avoids it, so the quantile is
    computed here. The expansion is accurate to <0.001 for df >= 19 (df=19: 2.0932 vs the
    exact 2.0930; df=35: 2.0301 vs 2.0301), which covers every bucket this script sees.
    """
    if df <= 0:
        return float("nan")
    if df > 2000:
        return 1.959964
    z = 1.959964
    return (z + (z ** 3 + z) / (4 * df)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def welch_interval(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference of means `b - a` with a Welch t 95% interval (no p-value on purpose)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"delta": None, "ci_low": None, "ci_high": None}
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"delta": round(float(b.mean() - a.mean()), 4), "ci_low": None,
                "ci_high": None}
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = t_quantile_975(int(round(df)))
    delta = float(b.mean() - a.mean())
    return {"delta": round(delta, 4),
            "ci_low": round(delta - t * se, 4),
            "ci_high": round(delta + t * se, 4),
            "se": round(se, 4), "df": int(round(df))}


def family_of(strategy: str, config: str) -> str:
    """Chunking family key. `fixed` is split by nominal size so 128/256/384/512 differ."""
    if strategy == "fixed":
        return f"fixed/{config.split('_')[0]}"
    return strategy


def mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvals[idx])
        running = max(running, value)
        adj[idx] = running
    return adj


def bootstrap_p(deltas: np.ndarray) -> float:
    """Two-sided bootstrap p for a paired difference centered at zero."""
    b = len(deltas)
    p = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return float(min(1.0, max(p, 1.0 / b)))


def wilcoxon(deltas: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p via normal approximation (no scipy)."""
    d = deltas[np.abs(deltas) > 0]
    n = len(d)
    if n < 6:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_plus = float(np.sum(ranks[d > 0]))
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return float("nan")
    z = (w_plus - mu) / sigma
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--split", default="dev", choices=["dev", "test", "test2"])
    ap.add_argument("--results", default="", help="results root (default: final for test, "
                                                  "final_dev for dev, final_test2 for test2)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--questions", default="",
                    help="questions file. Required for --split test2: there the supplied "
                         "file is the sole source of question metadata (groups, domain, "
                         "evidence_type, documents, gold lengths). The original "
                         "questions.jsonl is never used for test2 statistics - only 27's "
                         "leakage check reads it.")
    args = ap.parse_args()

    if args.split == "test2" and not args.questions:
        raise SystemExit("REFUSING: --split test2 requires --questions; there is no "
                         "fallback to annotations/questions.jsonl")

    root = Path(args.root)
    paths = project_paths(root)
    settings = load_settings(root)
    seed = int(args.seed if args.seed is not None else
               settings.get("project", {}).get("random_seed", 20261013))
    default_out = {"test": "results/final", "dev": "results/final_dev",
                   "test2": "results/final_test2"}[args.split]
    out_root = Path(args.results) if args.results else root / default_out

    # The metadata source. For test2 this is the split's own file and nothing else; the
    # canonical questions.jsonl is only ever read by 27's leakage check.
    qpath = Path(args.questions) if args.questions else paths["annotations"] / "questions.jsonl"
    if not qpath.is_absolute():
        qpath = root / qpath
    if not qpath.exists():
        raise SystemExit(f"REFUSING: questions file not found: {qpath}")

    rows_by_track = {}
    for track in TRACKS:
        path = out_root / f"perq_{track}.jsonl"
        if not path.exists():
            print(f"missing {path}; run 19_rerank_eval.py first", file=sys.stderr)
            return 2
        rows_by_track[track] = read_jsonl(path)

    # ---------------------------------------------------------------- group means
    # data[track][config][group] = metric value (mean over the group's questions)
    data: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        t: defaultdict(dict) for t in TRACKS}
    groups_by_config: dict[str, set[str]] = defaultdict(set)
    for track, rows in rows_by_track.items():
        acc: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        for r in rows:
            g = r.get("group_id") or r["qid"]
            groups_by_config[r["config"]].add(g)
            for metric in SET_METRICS + ORDER_METRICS:
                if metric in r:
                    acc[r["config"]][g][metric].append(float(r[metric]))
        for cfg, gmap in acc.items():
            for g, metrics in gmap.items():
                data[track][cfg][g] = {m: mean(v) for m, v in metrics.items()}

    configs = sorted(set().union(*[set(data[t]) for t in TRACKS]))
    groups = sorted(set().union(*[set(data[t].get(c, {})) for t in TRACKS for c in configs]))
    print(f"=== stats ({args.split}) ===")
    print(f"  configs {len(configs)}  groups {len(groups)}  boot {args.boot}  seed {seed}")

    # per-group macro over configs, per track per metric
    macro: dict[str, dict[str, np.ndarray]] = {t: {} for t in TRACKS}
    for track in TRACKS:
        for metric in SET_METRICS + ORDER_METRICS:
            vals = []
            for g in groups:
                per_cfg = [data[track].get(c, {}).get(g, {}).get(metric)
                           for c in configs]
                per_cfg = [v for v in per_cfg if v is not None]
                vals.append(mean(per_cfg) if per_cfg else np.nan)
            macro[track][metric] = np.asarray(vals, dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(args.boot, len(groups)))
    # one shared resample for every track keeps the contrasts paired
    boot_cache: dict[tuple[str, str], np.ndarray] = {}
    for track in TRACKS:
        for metric, arr in macro[track].items():
            boot_cache[(track, metric)] = np.array(
                [np.nanmean(arr[i]) for i in idx])

    def ci(track: str, metric: str) -> dict:
        b = boot_cache[(track, metric)]
        return {"mean": round(float(np.nanmean(macro[track][metric])), 4),
                "ci_low": round(float(np.nanpercentile(b, 2.5)), 4),
                "ci_high": round(float(np.nanpercentile(b, 97.5)), 4)}

    tracks_summary = {t: {m: ci(t, m) for m in SET_METRICS + ORDER_METRICS}
                      for t in TRACKS}

    # ---------------------------------------------------------------- invariance
    invariance = {}
    for track, ref in (("C1", "main"), ("C2", "main")):
        max_diff = 0.0
        for c in configs:
            for metric in SET_METRICS:
                for g in groups:
                    a = data[track].get(c, {}).get(g, {}).get(metric)
                    b = data[ref].get(c, {}).get(g, {}).get(metric)
                    if a is not None and b is not None:
                        max_diff = max(max_diff, abs(a - b))
        invariance[f"{track}-{ref}"] = {
            "max_abs_diff": max_diff, "passed": max_diff < 1e-9,
            "note": "arithmetic identity; no p-value is computed"}

    # ---------------------------------------------------------------- contrasts
    def contrast(a: str, b: str, metric: str) -> dict:
        ba, bb = boot_cache[(a, metric)], boot_cache[(b, metric)]
        deltas = ba - bb
        point = float(np.nanmean(macro[a][metric]) - np.nanmean(macro[b][metric]))
        return {
            "metric": metric, "delta": round(point, 4),
            "ci_low": round(float(np.nanpercentile(deltas, 2.5)), 4),
            "ci_high": round(float(np.nanpercentile(deltas, 97.5)), 4),
            "p_boot": round(bootstrap_p(deltas), 4),
            "p_wilcoxon": round(wilcoxon(macro[a][metric] - macro[b][metric]), 4),
            "n_groups": len(groups),
        }

    contrasts = {}
    family_set = ["C3-C1", "C3-C2"]
    raw = []
    for name in family_set:
        a, b = name.split("-")
        contrasts[name] = {**contrast(a, b, "complete@10"), "family": "set_level_primary"}
        raw.append(contrasts[name]["p_boot"])
    for name, adj in zip(family_set, holm(raw)):
        contrasts[name]["p_holm"] = round(adj, 4)

    for family, pairs in (("order_fixed_pool", [("C2", "main"), ("C3", "C1")]),
                          ("pool_with_rerank", [("C3", "C2")])):
        raw = []
        names = []
        for a, b in pairs:
            for metric in ORDER_METRICS:
                name = f"{a}-{b}:{metric}"
                contrasts[name] = {**contrast(a, b, metric), "family": family}
                raw.append(contrasts[name]["p_boot"])
                names.append(name)
        for name, adj in zip(names, holm(raw)):
            contrasts[name]["p_holm"] = round(adj, 4)

    # ---------------------------------------------------------------- best configs
    best: dict[str, str] = {}
    per_config: dict[str, dict[str, float]] = {}
    for track in TRACKS:
        scored = {c: mean([data[track][c][g]["complete@10"] for g in data[track][c]
                           if "complete@10" in data[track][c][g]]) for c in data[track]}
        best[track] = max(scored, key=scored.get)
        per_config[track] = {c: round(v, 4) for c, v in scored.items()}

    # ---------------------------------------------------------------- per document
    # Single metadata source: the file named by --questions (canonical for dev/test, the
    # test2 file for test2). Nothing here may read annotations/questions.jsonl directly.
    questions = {q["qid"]: q for q in read_jsonl(qpath)}
    doc_of = {}
    for r in rows_by_track["main"]:
        q = questions.get(r["qid"], {})
        clauses = [c["clause_id"] for es in q.get("evidence_sets", [])
                   for c in es.get("clauses", [])]
        doc_of[r["qid"]] = clauses[0].split("#")[0] if clauses else "?"

    per_doc = {}
    for track in TRACKS:
        cfg = best[track]
        acc: dict[str, list[float]] = defaultdict(list)
        for g in data[track][cfg]:
            for r in rows_by_track[track]:
                if r["config"] == cfg and (r.get("group_id") or r["qid"]) == g:
                    acc[doc_of.get(r["qid"], "?")].append(float(r["complete@10"]))
        doc_means = [mean(v) for v in acc.values()]
        per_doc[track] = {
            "config": cfg, "documents": len(doc_means),
            "macro_over_documents": round(mean(doc_means), 4),
            "pooled": round(mean([x for v in acc.values() for x in v]), 4),
        }

    # ---------------------------------------------------------------- rank stability
    rank_stability = {
        "C1_vs_C3": round(spearman([per_config["C1"][c] for c in configs],
                                   [per_config["C3"][c] for c in configs]), 4),
        "main_vs_C3": round(spearman([per_config["main"][c] for c in configs],
                                     [per_config["C3"][c] for c in configs]), 4),
        "note": ("C1 vs C3 isolates reranking at a fixed top-50 pool; main vs C3 mixes the "
                 "pool change with reranking and must not be attributed to reranking alone."),
    }

    # ---------------------------------------------------------------- truncation strata
    trunc = {}
    c3rows = [r for r in rows_by_track["C3"] if "trunc_frac" in r]
    if c3rows:
        # The unit that matters is a (question, configuration) cell: truncation depends on
        # the chunk lengths that configuration produced, so averaging over configurations
        # first would wash the variation out.
        c1 = {(r["qid"], r["config"]): float(r["complete@10"])
              for r in rows_by_track["C1"]}
        cells = []
        for r in c3rows:
            key = (r["qid"], r["config"])
            if key not in c1:
                continue
            cells.append({"qid": r["qid"], "config": r["config"],
                          "group": r.get("group_id") or r["qid"],
                          "frac": float(r["trunc_frac"]),
                          "delta": float(r["complete@10"]) - c1[key]})
        fr = np.array([c["frac"] for c in cells])
        dl = np.array([c["delta"] for c in cells])
        q1, q2 = np.percentile(fr, [33.333, 66.667])
        tiers = {"low": fr <= q1, "mid": (fr > q1) & (fr <= q2), "high": fr > q2}
        trunc = {
            "n_cells": len(cells),
            "trunc_frac": {"median": round(float(np.median(fr)), 4),
                           "p25": round(float(np.percentile(fr, 25)), 4),
                           "p75": round(float(np.percentile(fr, 75)), 4)},
            "spearman_trunc_vs_C3_minus_C1": round(spearman(fr.tolist(), dl.tolist()), 4),
            "by_tercile": {
                name: {"n_cells": int(mask.sum()),
                       "mean_trunc_frac": round(float(fr[mask].mean()), 4) if mask.any() else None,
                       "mean_C3_minus_C1_complete@10": round(float(dl[mask].mean()), 4) if mask.any() else None}
                for name, mask in tiers.items()},
            "note": ("Cells are (question, configuration); the Spearman correlation answers "
                     "whether the gain concentrates where pairs were truncated."),
        }
    else:
        trunc = {"available": False}

    # ---------------------------------------------------------------- per domain
    per_domain = {}
    for track in TRACKS:
        cfg = best[track]
        acc: dict[str, list[float]] = defaultdict(list)
        for r in rows_by_track[track]:
            if r["config"] == cfg:
                acc[r.get("domain", "?")].append(float(r["recall@10"]))
        per_domain[track] = {d: {"n": len(v), "recall@10": round(mean(v), 4)}
                             for d, v in sorted(acc.items())}

    # ---------------------------------------------------------------- per strategy
    # The macro average hides the fact that the families differ by orders of magnitude.
    # The report therefore carries a per-family table next to the macro figure, and both
    # come from this block so they cannot drift apart.
    config_strategy: dict[str, str] = {}
    for rows in rows_by_track.values():
        for r in rows:
            config_strategy.setdefault(r["config"], r.get("strategy", ""))
    families: dict[str, list[str]] = defaultdict(list)
    for c in configs:
        families[family_of(config_strategy.get(c, ""), c)].append(c)

    def family_array(track: str, cfgs: list[str], metric: str = "complete@10") -> np.ndarray:
        out = []
        for g in groups:
            vals = [data[track].get(c, {}).get(g, {}).get(metric) for c in cfgs]
            vals = [v for v in vals if v is not None]
            out.append(mean(vals) if vals else np.nan)
        return np.asarray(out, dtype=float)

    def family_array_rows(track: str, cfgs: list[str], metric: str) -> np.ndarray:
        """Same collapse, but read straight from the per-question rows.

        Needed for metrics that are not part of the aggregated `data` table (partial@10).
        """
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows_by_track[track]:
            if r["config"] in cfgs and metric in r:
                acc[r["config"]][r.get("group_id") or r["qid"]].append(float(r[metric]))
        out = []
        for g in groups:
            vals = [mean(acc[c][g]) for c in cfgs if g in acc[c]]
            out.append(mean(vals) if vals else np.nan)
        return np.asarray(out, dtype=float)

    family_order = ["heading", "paragraph", "fixed/size512", "fixed/size384",
                    "fixed/size256", "fixed/size128", "semantic"]
    by_strategy = {}
    for fam in family_order:
        if fam not in families:
            continue
        cfgs = sorted(families[fam])
        entry: dict = {"configs": cfgs, "n_configs": len(cfgs)}
        for track in TRACKS:
            arr = family_array(track, cfgs)
            boot = np.array([np.nanmean(arr[i]) for i in idx])
            # Per-question mean as well: the per-configuration table in the report quotes
            # per-question means (summary_*.csv), while the intervals above are group-level.
            # Both are emitted so the table can show the familiar level next to the CI.
            qvals = [float(r["complete@10"]) for r in rows_by_track[track]
                     if r["config"] in cfgs and "complete@10" in r]
            entry[track] = {
                "complete@10": round(float(np.nanmean(arr)), 4),
                "mean_questions": round(mean(qvals), 4),
                "ci_low": round(float(np.nanpercentile(boot, 2.5)), 4),
                "ci_high": round(float(np.nanpercentile(boot, 97.5)), 4),
                # two robustness metrics: union character coverage and equal-token recall.
                # Both answer "is the semantic family failing because of chunk granularity?"
                "partial@10": round(float(np.nanmean(
                    family_array_rows(track, cfgs, "partial@10"))), 4),
                "tok512_recall": round(float(np.nanmean(
                    family_array(track, cfgs, "tok512_recall"))), 4),
            }
        by_strategy[fam] = entry

    # ---------------------------------------------------------------- family contrasts
    # structural vs semantic: the contrast a future unseen set can actually resolve,
    # because it is an order of magnitude larger than the reranking gain.
    struct_cfgs = sorted(c for c in configs
                         if config_strategy.get(c) in STRUCTURAL_STRATEGIES)
    sem_cfgs = sorted(c for c in configs if config_strategy.get(c) == SEMANTIC_STRATEGY)
    family_contrasts = {}
    if struct_cfgs and sem_cfgs:
        for track in TRACKS:
            a = family_array(track, struct_cfgs)
            b = family_array(track, sem_cfgs)
            delta = b - a
            boot = np.array([np.nanmean(delta[i]) for i in idx])
            family_contrasts[f"{track}:semantic-minus-structural"] = {
                "structural_configs": struct_cfgs,
                "semantic_configs": sem_cfgs,
                "structural_mean": round(float(np.nanmean(a)), 4),
                "semantic_mean": round(float(np.nanmean(b)), 4),
                "delta": round(float(np.nanmean(delta)), 4),
                "ci_low": round(float(np.nanpercentile(boot, 2.5)), 4),
                "ci_high": round(float(np.nanpercentile(boot, 97.5)), 4),
                "p_boot": round(bootstrap_p(boot), 4),
                "n_groups": len(groups),
            }

    # ---------------------------------------------------------------- length strata
    # Short vs long clauses are compared symmetrically: `short` is the first k groups and
    # `long` the last k, both after sorting by (mean gold clause length, group_id). Only k
    # (floor vs ceil of n/3) varies between the two panels, so the middle/remainder block
    # of an uneven split never enters the comparison.
    #
    # Precision is reported as a Welch t interval, deliberately *not* as a percentile
    # bootstrap interval: at k=19 on the test split the bootstrap lower bound moves
    # between -0.013 and +0.009 across seeds (B=4000, six seeds), i.e. it would sometimes
    # appear to exclude zero while the t interval (-0.019, +0.571) and the corresponding
    # p=0.065 agree that it does not. The bootstrap interval is kept as a secondary column
    # with a `stable` flag so the instability is visible rather than hidden.
    corpus = goldset.load_corpus(root)

    def clause_chars(clause_id: str) -> int:
        clause = corpus.clauses.get(clause_id)
        return (clause.char_end - clause.char_start) if clause else 0

    question_len = {}
    for qid, q in questions.items():
        lens = [clause_chars(ref.get("clause_id", ""))
                for eset in q.get("evidence_sets") or []
                for ref in eset.get("clauses") or []]
        question_len[qid] = max(lens) if lens else 0

    group_of: dict[str, str] = {}
    for rows in rows_by_track.values():
        for r in rows:
            group_of.setdefault(r["qid"], r.get("group_id") or r["qid"])
    per_group_len: dict[str, list[int]] = defaultdict(list)
    for qid, g in group_of.items():
        per_group_len[g].append(question_len.get(qid, 0))
    group_len = {g: float(np.mean(v)) for g, v in per_group_len.items() if v}
    len_arr = np.array(sorted(group_len.values()), dtype=float)

    # Sampling frame: how long the clauses already used by questions are, against the
    # clauses still available for a future held-out set (unused, >=150 chars, not
    # boilerplate - the definition `24_preregister_test2.py` must use too).
    #
    # This is a *design-time* statistic of the canonical question set. It is meaningless for
    # a split that was itself drawn from that pool under a pre-registered design, so the
    # test2 run records it as not applicable instead of recomputing it from its own rows.
    if args.questions:
        sampling_frame = {
            "applicable": False,
            "note": ("not applicable: this run's metadata comes from the split's own "
                     "questions file; the sampling frame is recorded in the pre-registration"),
        }
    else:
        used_ids = {ref.get("clause_id")
                    for q in questions.values()
                    for eset in (q.get("evidence_sets") or [])
                    for ref in (eset.get("clauses") or [])}
        all_clauses = list(corpus.clauses.values())
        used_lens = [c.char_end - c.char_start for c in all_clauses if c.clause_id in used_ids]
        pool_lens = [c.char_end - c.char_start for c in all_clauses
                     if c.clause_id not in used_ids and not c.boilerplate
                     and len(c.text) >= 150]
        sampling_frame = {
            "applicable": True,
            "used_clauses": len(used_lens),
            "used_median_chars": round(float(np.median(used_lens)), 1) if used_lens else None,
            "available_clauses": len(pool_lens),
            "available_median_chars": round(float(np.median(pool_lens)), 1) if pool_lens else None,
            "note": ("available = unused by any question, >=150 characters, not boilerplate; "
                     "the future held-out set is drawn from it and must be length-matched"),
        }

    def length_panel(track: str, k: int) -> dict:
        a = family_array(track, struct_cfgs) if struct_cfgs else None
        b = family_array(track, sem_cfgs) if sem_cfgs else None
        if a is None or b is None:
            return {"available": False}
        cells = [(float(a[i] - b[i]), group_len.get(g, 0.0), g)
                 for i, g in enumerate(groups) if not (np.isnan(a[i]) or np.isnan(b[i]))]
        cells.sort(key=lambda t: (t[1], t[2]))          # ties broken by group_id
        k = max(1, min(k, len(cells) // 2))             # never let the two ends overlap
        short = np.array([c[0] for c in cells[:k]])
        long = np.array([c[0] for c in cells[-k:]])
        t_int = welch_interval(short, long)
        rng = np.random.default_rng(seed)
        bs = np.array([rng.choice(long, size=k).mean() - rng.choice(short, size=k).mean()
                       for _ in range(args.boot)])
        ci_low, ci_high = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        return {
            "k": k, "n_groups": len(cells),
            "short_mean": round(float(short.mean()), 4),
            "long_mean": round(float(long.mean()), 4),
            "short_len_median": round(float(np.median([c[1] for c in cells[:k]])), 1),
            "long_len_median": round(float(np.median([c[1] for c in cells[-k:]])), 1),
            "welch": t_int,
            "bootstrap": {"ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
                          "boot": args.boot, "seed": seed,
                          "stable": bool(abs(ci_low) < 0.02)},
        }

    by_length_tercile: dict = {
        "rule": ("short = first k groups, long = last k groups, ordered by "
                 "(mean gold clause length, group_id); k in {floor(n/3), ceil(n/3)}; "
                 "the middle block never enters the comparison"),
        "length_statistic": ("question length = max annotated clause span (characters) "
                             "over all evidence sets; group length = mean over its questions"),
        "n_groups": len(group_len),
        "length_distribution": {
            "median": round(float(np.median(len_arr)), 1),
            "p25": round(float(np.percentile(len_arr, 25)), 1),
            "p75": round(float(np.percentile(len_arr, 75)), 1),
        },
        "sampling_frame": sampling_frame,
        "tracks": {},
    }
    if struct_cfgs and sem_cfgs and group_len:
        for track in ("C1", "C3"):
            n = len(group_len)
            by_length_tercile["tracks"][track] = {
                "k_floor": length_panel(track, n // 3),
                "k_ceil": length_panel(track, -(-n // 3)),
            }

    stats = {
        "split": args.split, "boot": args.boot, "seed": seed,
        "questions_source": qpath.relative_to(root).as_posix()
        if str(qpath).startswith(str(root)) else str(qpath),
        "legacy_questions_used_for_statistics": not bool(args.questions),
        "groups": len(groups), "configs": len(configs),
        "unit": "group_id (configurations collapsed by macro-average first)",
        "tracks": tracks_summary,
        "invariance": invariance,
        "contrasts": contrasts,
        "primary_comparison": "C3-C1 on complete@10",
        "best_config": best,
        "per_config_complete@10": per_config,
        "by_strategy": by_strategy,
        "family_contrasts": family_contrasts,
        "by_length_tercile": by_length_tercile,
        "per_document": per_doc,
        "rank_stability": rank_stability,
        "truncation_strata": trunc,
        "per_domain": per_domain,
    }
    write_json(out_root / "stats.json", stats)

    print("\n  track macro complete@10 (95% CI):")
    for t in TRACKS:
        c = tracks_summary[t]["complete@10"]
        print(f"    {t:<5} {c['mean']:.3f}  [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
    print("\n  set-level contrasts (informative):")
    for name in family_set:
        c = contrasts[name]
        print(f"    {name:<6} {c['delta']:+.3f}  [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
              f"  p_holm={c['p_holm']:.3f}")
    print("\n  invariance:")
    for k, v in invariance.items():
        print(f"    {k:<6} max|diff|={v['max_abs_diff']:.2e}  passed={v['passed']}")
    print(f"\n  rank stability: {rank_stability['C1_vs_C3']} (C1 vs C3)")
    if by_strategy:
        print("\n  per strategy, complete@10 (C1 -> C3):")
        for fam, e in by_strategy.items():
            print(f"    {fam:<15} n={e['n_configs']}  "
                  f"{e['C1']['complete@10']:.3f} -> {e['C3']['complete@10']:.3f}"
                  f"  C3 CI [{e['C3']['ci_low']:.3f}, {e['C3']['ci_high']:.3f}]")
    for name, c in family_contrasts.items():
        print(f"  family contrast {name:<32} {c['delta']:+.3f} "
              f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")
    if by_length_tercile.get("tracks"):
        print("\n  clause-length short vs long (Welch 95% interval, no p-value):")
        for track, panels in by_length_tercile["tracks"].items():
            for label in ("k_floor", "k_ceil"):
                p = panels[label]
                w = p["welch"]
                print(f"    {track} {label} k={p['k']:<3} delta {w['delta']:+.3f} "
                      f"[{w['ci_low']:+.3f}, {w['ci_high']:+.3f}]"
                      f"  bootstrap [{p['bootstrap']['ci_low']:+.3f}, "
                      f"{p['bootstrap']['ci_high']:+.3f}]"
                      f"{'*unstable' if p['bootstrap']['stable'] else ''}")
    print(f"  wrote {out_root / 'stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
