"""14_analyze.py - tune on development, summarise, and write the results report.

Two things this script is careful about.

1. **Group-disjoint tuning.** Parameter selection happens on development questions only,
   using k-fold cross-validation with folds defined by `group_id`, so a paraphrase family
   can never sit on both sides of a fold. The cross-validated score of the *procedure*
   (which includes picking the best configuration) is reported alongside the score of the
   configuration finally chosen on the whole development set.
2. **The test split is not inspected here.** It is summarised only if
   `results/perq_test.jsonl` already exists, i.e. after a locked, one-shot evaluation.

Outputs
    results/REPORT.md                     the narrative report with tables
    results/tables.json                   the same numbers, machine readable
    results/figures/recall_at_k.png
    results/figures/token_budget.png
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, load_settings, project_paths, read_jsonl, write_json  # noqa: E402

METRIC_KEYS = ("recall@1", "recall@5", "recall@10", "complete@10", "hit@10",
               "partial@10", "tok512_recall", "tok1024_recall", "tok2048_recall")


def mean(xs):
    return round(statistics.fmean(xs), 4) if xs else 0.0


def std(xs):
    return round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0


def load_perq(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


def group_cv(rows: list[dict], folds: int = 5) -> dict:
    """Cross-validate the *selection procedure* over group-disjoint folds."""
    by_config: dict[str, dict[str, float]] = defaultdict(dict)
    groups: dict[str, str] = {}
    for r in rows:
        by_config[r["config"]][r["qid"]] = r["recall@10"]
        groups[r["qid"]] = str(r.get("group_id") or r["qid"])
    gids = sorted(set(groups.values()))
    if len(gids) < folds:
        return {"folds": 0, "mean": None, "per_strategy": {}}

    fold_of = {g: i % folds for i, g in enumerate(gids)}
    qids_by_fold: dict[int, list[str]] = defaultdict(list)
    for qid, g in groups.items():
        qids_by_fold[fold_of[g]].append(qid)

    # per strategy: pick best config on the training folds, score it on the held-out fold
    per_strategy: dict[str, list[float]] = defaultdict(list)
    config_strategy = {r["config"]: r["strategy"] for r in rows}
    for f in range(folds):
        held = qids_by_fold.get(f, [])
        if not held:
            continue
        train = [q for q in groups if q not in set(held)]
        best_by_strategy: dict[str, tuple[str, float]] = {}
        for cfg, scores in by_config.items():
            train_scores = [scores[q] for q in train if q in scores]
            if not train_scores:
                continue
            val = statistics.fmean(train_scores)
            strat = config_strategy.get(cfg, "?")
            if strat not in best_by_strategy or val > best_by_strategy[strat][1]:
                best_by_strategy[strat] = (cfg, val)
        for strat, (cfg, _val) in best_by_strategy.items():
            scores = by_config[cfg]
            held_scores = [scores[q] for q in held if q in scores]
            if held_scores:
                per_strategy[strat].append(statistics.fmean(held_scores))

    return {
        "folds": folds,
        "groups": len(gids),
        "per_strategy": {k: {"cv_recall@10": mean(v), "sd": std(v), "n_folds": len(v)}
                         for k, v in sorted(per_strategy.items())},
    }


def best_per_strategy(rows: list[dict]) -> dict[str, str]:
    """Best configuration name per strategy, ranked on development Recall@10."""
    grouped: dict[str, set[str]] = defaultdict(set)
    stats: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        grouped[r["strategy"]].add(r["config"])
        stats[r["config"]].append(r["recall@10"])
    out: dict[str, str] = {}
    for strat, configs in grouped.items():
        out[strat] = max(configs, key=lambda c: (statistics.fmean(stats[c]), c))
    return out


def summarise(rows: list[dict]) -> dict:
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    out = {}
    for cfg, rs in by_cfg.items():
        out[cfg] = {k: {"mean": mean([r[k] for r in rs]), "sd": std([r[k] for r in rs])}
                    for k in METRIC_KEYS}
        out[cfg]["n"] = len(rs)
        out[cfg]["strategy"] = rs[0]["strategy"]
        out[cfg]["evidence_type"] = rs[0].get("evidence_type")
    return out


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    res = root / "results"

    dev = load_perq(res / "perq_dev.jsonl")
    test = load_perq(res / "perq_test.jsonl")
    if not dev:
        print("no results/perq_dev.jsonl; run 13_evaluate.py --split dev first", file=sys.stderr)
        return 2

    def load_summary_csv(split: str) -> dict[str, dict]:
        import csv
        path = res / f"summary_{split}.csv"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8", newline="") as f:
            return {f"{r['strategy']}/{r['config']}": r for r in csv.DictReader(f)}

    csv_dev = load_summary_csv("dev")
    csv_test = load_summary_csv("test")

    dev_summary = summarise(dev)
    test_summary = summarise(test) if test else {}
    dev_best = best_per_strategy(dev)
    cv = group_cv(dev, args.folds)

    manifest = {}
    mpath = root / "MANIFEST.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        manifest = {"corpus_version": m.get("corpus_version"),
                    "documents": m.get("documents_included"),
                    "chunks": m.get("chunk_stats", {}).get("chunks"),
                    "tokens_max": m.get("chunk_stats", {}).get("token_max")}

    run_meta = {}
    rpath = res / "run_dev.json"
    if rpath.exists():
        run_meta = json.loads(rpath.read_text(encoding="utf-8"))

    metrics_by_cfg: dict[str, dict] = {}
    for mp in (root / "results").glob("metrics_*.json"):
        try:
            d = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics_by_cfg[f"{d.get('strategy')}/{d.get('config')}"] = d

    index_meta = {}
    ipath = paths["reports"] / "index_summary.json"
    if ipath.exists():
        index_meta = json.loads(ipath.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- tables
    main_rows = []
    for strat in sorted(dev_best):
        cfg_name = dev_best[strat]
        s = dev_summary[cfg_name]
        idx = index_meta.get("configs", {}).get(cfg_name, {})
        main_rows.append([
            strat, cfg_name.split("/")[-1], s["n"], idx.get("n_chunks", ""),
            f"{s['recall@1']['mean']:.3f}",
            f"{s['recall@5']['mean']:.3f}",
            f"{s['recall@10']['mean']:.3f} ± {s['recall@10']['sd']:.3f}",
            f"{s['complete@10']['mean']:.3f}",
            f"{s['partial@10']['mean']:.3f}",
            f"{float(csv_dev.get(cfg_name, {}).get('mrr', 'nan')):.3f}",
            f"{idx.get('index_size_bytes', 0) / 1e6:.2f}",
            f"{idx.get('embedding_time_s', 0):.1f}",
            f"{float(csv_dev.get(cfg_name, {}).get('latency_ms', 0)):.3f}",
        ])

    token_rows = []
    for strat in sorted(dev_best):
        cfg_name = dev_best[strat]
        s = dev_summary[cfg_name]
        token_rows.append([
            strat, cfg_name.split("/")[-1],
            f"{s['tok512_recall']['mean']:.3f}", f"{s['tok1024_recall']['mean']:.3f}",
            f"{s['tok2048_recall']['mean']:.3f}",
        ])

    # per evidence type / domain for the chosen configurations
    type_rows = []
    for strat in sorted(dev_best):
        chosen = dev_best[strat]
        subset = [r for r in dev if r["config"] == chosen]
        by_type = defaultdict(list)
        by_dom = defaultdict(list)
        for r in subset:
            by_type[r.get("evidence_type")].append(r["recall@5"])
            by_dom[r.get("domain")].append(r["recall@5"])
        type_rows.append([strat, chosen.split("/")[-1],
                          *[f"{mean(by_type[t]):.3f}" for t in
                            ("single-clause", "rule-and-exception", "cross-clause")]])
    domain_keys = sorted({r.get("domain") for r in dev})
    domain_rows = []
    for strat in sorted(dev_best):
        chosen = dev_best[strat]
        subset = [r for r in dev if r["config"] == chosen]
        by_dom = defaultdict(list)
        for r in subset:
            by_dom[r.get("domain")].append(r["recall@5"])
        domain_rows.append([strat, chosen.split("/")[-1], *[f"{mean(by_dom[d]):.3f}" for d in domain_keys]])

    all_rows = []
    for cfg_name, s in sorted(dev_summary.items(),
                              key=lambda kv: -kv[1]["recall@10"]["mean"]):
        all_rows.append([s["strategy"], cfg_name.split("/")[-1], s["n"],
                         f"{s['recall@10']['mean']:.3f}",
                         f"{s['complete@10']['mean']:.3f}"])

    # dev-selected configuration scored once on the locked test split
    test_rows = []
    for strat in sorted(dev_best):
        cfg_name = dev_best[strat]
        ds = dev_summary.get(cfg_name, {})
        ts = test_summary.get(cfg_name, {})
        if not ts:
            continue
        test_rows.append([
            strat, cfg_name.split("/")[-1],
            f"{ds['recall@10']['mean']:.3f}", f"{ts['recall@10']['mean']:.3f}",
            f"{ds['complete@10']['mean']:.3f}", f"{ts['complete@10']['mean']:.3f}",
            f"{ts['recall@1']['mean']:.3f}", f"{ts['recall@5']['mean']:.3f}",
            f"{float(csv_test.get(cfg_name, {}).get('mrr', 'nan')):.3f}",
        ])

    tables = {
        "main": main_rows, "token_budget": token_rows,
        "by_evidence_type": type_rows, "by_domain": domain_rows,
        "all_configs": all_rows, "cv": cv, "manifest": manifest, "run": run_meta,
    }
    write_json(res / "tables.json", tables)

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = res / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        labels = ["R@1", "R@5", "R@10"]
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / max(1, len(dev_best))
        for i, strat in enumerate(sorted(dev_best)):
            s = dev_summary[dev_best[strat]]
            vals = [s["recall@1"]["mean"], s["recall@5"]["mean"], s["recall@10"]["mean"]]
            ax.bar([x + i * width for x in range(len(labels))], vals, width,
                   label=f"{strat} ({dev_best[strat].split('/')[-1]})")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("evidence recall")
        ax.set_title("Evidence Recall@k by chunking strategy (development set)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / "recall_at_k.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4))
        budgets = ["512", "1024", "2048"]
        width = 0.8 / max(1, len(dev_best))
        for i, strat in enumerate(sorted(dev_best)):
            s = dev_summary[dev_best[strat]]
            vals = [s[f"tok{b}_recall"]["mean"] for b in budgets]
            ax.bar([x + i * width for x in range(len(budgets))], vals, width, label=strat)
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"{b} tokens" for b in budgets])
        ax.set_ylabel("evidence recall")
        ax.set_title("Recall under matched retrieved-token budgets (complete chunks only)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / "token_budget.png", dpi=140)
        plt.close(fig)
        figures_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: figures not rendered ({type(e).__name__}: {e})", file=sys.stderr)
        figures_ok = False

    # ---------------------------------------------------------------- report
    lines: list[str] = []
    lines.append("# Chunking strategies for course-policy retrieval - results\n")
    lines.append(f"Corpus **{manifest.get('corpus_version')}** · "
                 f"{manifest.get('documents')} documents · "
                 f"{manifest.get('chunks')} clause units · "
                 f"max chunk {manifest.get('tokens_max')} tokens\n")
    lines.append(f"Embedding model: `{run_meta.get('model', 'BAAI/bge-small-en-v1.5')}` · "
                 f"device `{index_meta.get('device', 'cpu')}` · "
                 f"index `FAISS IndexFlatIP` (exact search)\n")
    lines.append(f"Development questions: **{run_meta.get('questions', len({r['qid'] for r in dev}))}** "
                 f"· configurations evaluated: **{run_meta.get('configs', len(dev_summary))}**\n")
    lines.append("## How to read this\n")
    lines.append("- `Recall@k` counts a clause only when the retrieved evidence intervals contain "
                 "its **entire** annotated span. Partial overlap is reported separately as "
                 "`partial@10` and never counted as a hit.")
    lines.append("- `Complete@10` is 1 only when one whole sufficient evidence set is covered.")
    lines.append("- Parameter selection used the development set with **group-disjoint** "
                 "cross-validation; the test split is untouched until it is locked.")
    lines.append("")

    lines.append("## Chosen configuration per strategy (development)\n")
    lines.append(md_table(
        ["strategy", "config", "questions", "chunks", "R@1", "R@5", "R@10", "Complete@10",
         "partial@10", "MRR", "index MB", "embed s", "latency ms"], main_rows))
    lines.append("")
    if cv.get("per_strategy"):
        cv_rows = [[k, f"{v['cv_recall@10']:.3f}", f"{v['sd']:.3f}", v["n_folds"]]
                   for k, v in cv["per_strategy"].items()]
        lines.append(f"### Cross-validated procedure score ({cv['folds']}-fold, "
                     f"{cv['groups']} groups)\n")
        lines.append(md_table(["strategy", "CV R@10", "sd", "folds"], cv_rows))
        lines.append("")
        lines.append("This is the honest estimate of *tuning plus selection*: the best "
                     "configuration is picked on each training fold and scored on the "
                     "held-out fold, so it cannot benefit from having seen the evaluation "
                     "questions.\n")

    if test_rows:
        lines.append("## Held-out test results — PILOT / exploratory, not final\n")
        lines.append("> **Status: pilot.** This was the first test-split evaluation (top-10 dense "
                     "retrieval, no reranker, 2026-10-24). It is retained for provenance only. "
                     "The formal result will come from a single four-track run (dense top-10 / "
                     "dense top-50 / reranked top-10 / reranked top-50) executed once after the "
                     "reranker is frozen. Archived copy: `results/pilot/`.\n")
        lines.append("Configurations were selected on the development split only; these are "
                     "the same configurations scored once on the locked test split.\n")
        lines.append(md_table(["strategy", "config", "dev R@10", "test R@10",
                               "dev Complete@10", "test Complete@10",
                               "test R@1", "test R@5", "test MRR"], test_rows))
        lines.append("")

    lines.append("## Matched retrieved-token budgets\n")
    lines.append(md_table(["strategy", "config", "recall @512", "recall @1024",
                           "recall @2048"], token_rows))
    lines.append("")
    lines.append("Chunks are admitted in rank order and only as complete chunks: if the next "
                 "chunk does not fit the remaining budget, selection stops there. Nothing is "
                 "truncated and no smaller lower-ranked chunk is skipped ahead to.\n")

    lines.append("## By evidence type (Recall@5)\n")
    lines.append(md_table(["strategy", "config", "single-clause", "rule-and-exception",
                           "cross-clause"], type_rows))
    lines.append("")
    lines.append("## By policy domain (Recall@5)\n")
    lines.append(md_table(["strategy", "config", *domain_keys], domain_rows))
    lines.append("")

    lines.append("## All configurations (development, ranked)\n")
    lines.append(md_table(["strategy", "config", "questions", "R@10", "Complete@10"], all_rows))
    lines.append("")

    lines.append("## Limitations to state in the write-up\n")
    lines.append("- Questions were drafted with AI assistance from the frozen corpus and "
                 "reviewed by the student; they are not independently double-annotated, so "
                 "no inter-annotator kappa is reported for this run.")
    lines.append("- The corpus is concentrated: a few large documents supply most clauses, "
                 "so pooled means should be read alongside the per-document breakdown.")
    lines.append("- `resources` is the thinnest domain; domain-level comparisons there are "
                 "based on few questions.")
    if test:
        lines.append("- The test split has been evaluated; see `summary_test.csv`.")
    else:
        lines.append("- The test split has **not** been evaluated yet; it is summarised only "
                     "after the questions are reviewed and the split is locked.")
    lines.append("")
    if figures_ok:
        lines.append("![Recall@k](figures/recall_at_k.png)\n")
        lines.append("![Token budgets](figures/token_budget.png)\n")

    (res / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== analysis ===")
    for r in main_rows:
        print(f"  {r[0]:<10} {r[1]:<16} R@10={r[6]}  Complete@10={r[7]}")
    print(f"  CV folds: {cv.get('folds')} over {cv.get('groups')} groups")
    print(f"\nwrote {res / 'REPORT.md'}")
    print(f"wrote {res / 'tables.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
