"""19_rerank_eval.py - the single formal evaluation of the locked test split.

Four tracks form a 2x2 factorial design over (candidate pool size) x (reranker on/off):

    track  pool  rerank
    main   top-10  no      baseline, same protocol as the pilot run
    C1     top-50  no      pool effect only
    C2     top-10  yes     order effect only at a fixed pool
    C3     top-50  yes     the proposed configuration

Two facts are asserted rather than tested, because they are arithmetic:

  * reranking a fixed set of ten chunks cannot change Recall@10 / Hit@10 / Complete@10,
    so C2 must equal main on those three;
  * the first ten of a dense top-50 are exactly the dense top-10, so C1 must equal main
    on the same three.

If either assertion fails, the implementation is wrong and the run aborts before writing a
result. The informative comparisons for the set-level metrics are therefore C3 - C1
(fixed pool, reranking may pull chunks 11-50 into the top ten) and C3 - C2 (pool effect
with reranking active).

Test-use protocol. The test split was already used once for the exploratory pilot, so it is
no longer unseen; the report must say so. To keep the formal run honest anyway:

    --split test --acknowledge-test-use   writes IN_PROGRESS.json, then results
    success                               writes FINAL_RUN.json
    crash                                 keeps IN_PROGRESS.json and partial output
    --resume                              archives the failed attempt under attempts/ and
                                          increments the attempt counter

There is no --force. A second successful run is impossible; only continuation of an
incomplete attempt is.

Usage
    .venv-gpu\\Scripts\\python.exe scripts/19_rerank_eval.py --split dev
    .venv-gpu\\Scripts\\python.exe scripts/19_rerank_eval.py --split test --acknowledge-test-use
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import PROJECT_ROOT, load_settings, project_paths, read_jsonl, write_json, write_jsonl  # noqa: E402
import goldset  # noqa: E402

TRACKS = {
    "main": {"pool": 10, "rerank": False},
    "C1": {"pool": 50, "rerank": False},
    "C2": {"pool": 10, "rerank": True},
    "C3": {"pool": 50, "rerank": True},
}
KS = (1, 3, 5, 10)
TOKEN_BUDGETS = (512, 1024, 2048)
SET_METRIC_KEYS = ("recall@10", "hit@10", "complete@10")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(rel: str, name: str, root: Path):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def find_checkpoint(root: Path, frozen: dict) -> Path:
    """Locate the frozen checkpoint by hash, so evaluation cannot pick the wrong model."""
    want = frozen["model"].get("checkpoint_sha256")
    runs = root / "models" / "reranker" / "runs"
    for candidate in sorted(runs.iterdir() if runs.exists() else []):
        weights = candidate / "model.safetensors"
        if weights.exists() and sha256_file(weights) == want:
            return candidate
    raise SystemExit(
        f"REFUSING: no checkpoint under {runs} matches FROZEN.json "
        f"checkpoint_sha256={want}")


def verify_preregistration(root: Path, pre_path: Path, questions_path: Path) -> list[dict]:
    """Compare every hash recorded at pre-registration time with the file on disk."""
    if not pre_path.exists():
        return [{"artifact": str(pre_path), "problem": "pre-registration file is missing"}]
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    problems: list[dict] = []
    for rel, want in (pre.get("artifacts") or {}).items():
        path = root / rel
        if not path.exists():
            problems.append({"artifact": rel, "problem": "missing"})
        elif sha256_file(path) != want:
            problems.append({"artifact": rel, "problem": "hash mismatch"})
    want_q = pre.get("questions_sha256")
    if want_q and sha256_file(questions_path) != want_q:
        problems.append({"artifact": str(questions_path), "problem": "hash mismatch"})
    return problems


def code_sha(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted((root / "scripts").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def metric_block(per_k: dict, per_budget: dict, budgets_unused: dict, budgets_zero: dict,
                 total: int) -> dict:
    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else 0.0

    out = {}
    for k in KS:
        out[f"recall@{k}"] = mean(per_k[k]["recall"])
        out[f"hit@{k}"] = mean(per_k[k]["hit"])
        out[f"complete@{k}"] = mean(per_k[k]["complete"])
        out[f"partial@{k}"] = mean(per_k[k]["partial"])
    out["mrr"] = mean(per_k["mrr"])
    out["questions"] = total
    out["by_token_budget"] = {}
    for b in TOKEN_BUDGETS:
        out["by_token_budget"][str(b)] = {
            "recall": mean(per_budget[b]["recall"]),
            "hit": mean(per_budget[b]["hit"]),
            "complete": mean(per_budget[b]["complete"]),
            "tokens_unused_mean": round(float(np.mean(budgets_unused[b])), 1)
            if budgets_unused[b] else 0.0,
            "zero_chunk_queries": budgets_zero[b],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--split", default="dev", choices=["dev", "test", "test2"])
    ap.add_argument("--acknowledge-test-use", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--strategy", action="append", default=[])
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--questions", default="",
                    help="questions file (default annotations/questions.jsonl; a second "
                         "held-out split lives in its own file so the locked test hash stays "
                         "valid)")
    ap.add_argument("--locked-dir", default="",
                    help="lock directory (default annotations/LOCKED for test, "
                         "annotations/LOCKED_TEST2 for test2)")
    ap.add_argument("--preregistration", default="",
                    help="pre-registration json; when given, every recorded hash must still "
                         "match or the run is refused")
    ap.add_argument("--out", default="",
                    help="output root (default results/final, results/final_dev, "
                         "results/final_test2)")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    qpath = Path(args.questions) if args.questions else paths["annotations"] / "questions.jsonl"
    if not qpath.is_absolute():
        qpath = root / qpath
    default_out = {"dev": "results/final_dev", "test": "results/final",
                   "test2": "results/final_test2"}[args.split]
    out_root = root / (args.out or default_out)
    out_root.mkdir(parents=True, exist_ok=True)

    frozen_path = root / "models" / "reranker" / "FROZEN.json"
    if not frozen_path.exists():
        print("REFUSING: models/reranker/FROZEN.json is missing; train and freeze first.",
              file=sys.stderr)
        return 2
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("env_role") != "canonical":
        print("REFUSING: FROZEN.json is not from the canonical environment.", file=sys.stderr)
        return 2

    if args.split == "test" and not args.acknowledge_test_use:
        print("REFUSING: --split test requires --acknowledge-test-use.", file=sys.stderr)
        return 2

    attempts_path = out_root / "ATTEMPTS.json"
    in_progress = out_root / "IN_PROGRESS.json"
    final_run = out_root / "FINAL_RUN.json"
    attempts = json.loads(attempts_path.read_text(encoding="utf-8")) if attempts_path.exists() else {"attempts": []}

    if final_run.exists():
        print("REFUSING: this split already has a completed formal run "
              f"({final_run}). There is no --force; a second run is not permitted.",
              file=sys.stderr)
        return 2
    if in_progress.exists() and not args.resume:
        print("REFUSING: an unfinished attempt exists. Re-run with --resume to archive it "
              "and continue; the failure stays on the record.", file=sys.stderr)
        return 2

    # lock integrity: test and test2 each have their own lock and their own questions file
    lock: dict = {}
    if args.split in ("test", "test2"):
        layout = {"test": ("LOCKED", "LOCKED.json"), "test2": ("LOCKED_TEST2", "LOCKED2.json")}
        dirname, meta_name = layout[args.split]
        lock_dir = Path(args.locked_dir) if args.locked_dir else paths["annotations"] / dirname
        if not lock_dir.is_absolute():
            lock_dir = root / lock_dir
        locked = lock_dir / meta_name
        if not locked.exists():
            print(f"REFUSING: the {args.split} split is not locked ({locked}).", file=sys.stderr)
            return 2
        lock = json.loads(locked.read_text(encoding="utf-8"))
        current = sha256_file(qpath)
        if lock.get("questions_jsonl_sha256") and current != lock["questions_jsonl_sha256"]:
            print(f"REFUSING: {qpath.name} does not match the locked hash.", file=sys.stderr)
            return 2

    # pre-registration: a second held-out set is evaluated once, against frozen settings
    if args.preregistration:
        pre_path = Path(args.preregistration)
        if not pre_path.is_absolute():
            pre_path = root / pre_path
        problems = verify_preregistration(root, pre_path, qpath)
        if problems:
            print("REFUSING: pre-registration does not match the repository:", file=sys.stderr)
            for p in problems[:10]:
                print(f"  {p['artifact']}: {p['problem']}", file=sys.stderr)
            return 2
        print(f"  preregistered : {pre_path}")

    attempt_no = 1
    if args.resume and in_progress.exists():
        prev = json.loads(in_progress.read_text(encoding="utf-8"))
        attempt_no = int(prev.get("attempt", 1)) + 1
        archive = out_root / "attempts" / f"attempt-{prev.get('attempt', 1)}"
        archive.mkdir(parents=True, exist_ok=True)
        for item in out_root.iterdir():
            if item.name in ("attempts", "ATTEMPTS.json", "IN_PROGRESS.json", "FINAL_RUN.json"):
                continue
            shutil.move(str(item), str(archive / item.name))
        prev.update({"status": "incomplete", "archived_to": str(archive),
                     "archived_at": now()})
        attempts["attempts"].append(prev)
        write_json(attempts_path, attempts)
        in_progress.unlink()

    # Inputs that make this evaluation reproducible: every artefact this run depends on is
    # hashed before the run and written into FINAL_RUN.json, so a later reader can prove the
    # numbers came from these questions, this bundle list and this checkpoint.
    def _sha(path: Path) -> str | None:
        return sha256_file(path) if path.exists() else None

    prereg_path = (Path(args.preregistration) if args.preregistration else None)
    if prereg_path is not None and not prereg_path.is_absolute():
        prereg_path = root / prereg_path
    input_hashes = {
        "questions": _sha(qpath),
        "bundles": _sha(root / "results" / "test2" / "selected_bundles.json"),
        "amendments": _sha(root / "results" / "test2" / "bundle_amendments.jsonl"),
        "lock": _sha(locked) if args.split in ("test", "test2") else None,
        "preregistration": _sha(prereg_path) if prereg_path else None,
        "frozen": sha256_file(frozen_path),
        "code": code_sha(root),
    }

    attempt = {"attempt": attempt_no, "split": args.split, "started_at": now(),
               "status": "running",
               "frozen_sha256": input_hashes["frozen"],
               "lock_sha256": lock.get("test_jsonl_sha256", ""),
               "code_sha256": input_hashes["code"],
               "inputs": input_hashes}
    write_json(in_progress, attempt)
    print(f"=== attempt {attempt_no} ({args.split}) -> {out_root} ===")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt_dir = find_checkpoint(root, frozen)
    print(f"  checkpoint : {ckpt_dir}")
    device = args.device if torch.cuda.is_available() else "cpu"
    env_info = {
        "evaluation_host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": "fp16" if device == "cuda" else "fp32",
        "checkpoint_sha256": sha256_file(ckpt_dir / "model.safetensors"),
    }
    print(f"  host/gpu   : {env_info['evaluation_host']} / {env_info['gpu']}")
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), num_labels=1)
    model.to(device).eval()

    ev = load_module("scripts/13_evaluate.py", "eval13", root)
    tr = load_module("scripts/18_train_reranker.py", "train18", root)

    corpus = goldset.load_corpus(root)
    questions = [q for q in read_jsonl(qpath) if q.get("split") == args.split]
    if not questions:
        print(f"no {args.split} questions", file=sys.stderr)
        return 2
    gold = [ev.gold_spans(q, corpus) for q in questions]
    qtext = {q["qid"]: q["question"] for q in questions}

    summary = json.loads((paths["reports"] / "index_summary.json").read_text(encoding="utf-8"))
    configs = sorted(summary.get("configs", {}))
    if args.strategy:
        configs = [c for c in configs if c.split("/")[0] in set(args.strategy)]
    if args.config:
        configs = [c for c in configs if c.split("/")[-1] in set(args.config)]
    print(f"  questions  : {len(questions)}  configs: {len(configs)}  device: {device}")

    import faiss
    from common import load_sentence_transformer

    settings = load_settings(root)
    embed_model = load_sentence_transformer(
        settings.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5"), device="cpu")
    qemb = embed_model.encode([q["question"] for q in questions], batch_size=32,
                              normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True)

    trunc_cache: dict[tuple[str, str], bool] = {}

    def pair_truncated(qid: str, text: str, chunk_text: str) -> bool:
        """Would this (question, chunk) pair exceed the 512-token limit?"""
        key = (qid, text[:64] + str(len(chunk_text)))
        if key not in trunc_cache:
            q_ids = tok(text, add_special_tokens=False)["input_ids"]
            c_ids = tok(chunk_text, add_special_tokens=False)["input_ids"]
            q_len = min(len(q_ids), tr.QUERY_MAX)
            trunc_cache[key] = (q_len + len(c_ids) + 3) > tr.PAIR_MAX
        return trunc_cache[key]

    perq: dict[str, list[dict]] = {t: [] for t in TRACKS}
    results: dict[str, dict[str, dict]] = {t: {} for t in TRACKS}
    invariance_errors: list[str] = []

    for cfg in configs:
        strategy, config = cfg.split("/", 1)
        t_cfg = time.time()
        index = faiss.read_index(str(root / "indexes" / strategy / f"{config}.faiss"))
        meta = json.loads((root / "indexes" / strategy / f"{config}.meta.json")
                          .read_text(encoding="utf-8"))
        chunks = {c["chunk_id"]: c for c in
                  read_jsonl(root / "chunks" / strategy / f"{config}.jsonl")}

        t0 = time.time()
        _D, I = index.search(qemb, 50)
        dense_ms = round((time.time() - t0) / max(1, len(questions)) * 1000, 3)

        dense_per_q: list[list[dict]] = []
        for qi, row in enumerate(I):
            cands = []
            for idx in row:
                if idx < 0:
                    continue
                c = chunks.get(meta["chunk_ids"][int(idx)])
                if c is None:
                    continue
                cands.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                              "token_count": int(c.get("token_count", 0)),
                              "ev_ranges": c.get("ev_ranges", []), "text": c["text"]})
            dense_per_q.append(cands)

        for track, spec in TRACKS.items():
            t_track = time.time()
            per_k = {k: defaultdict(list) for k in KS}
            per_budget = {b: defaultdict(list) for b in TOKEN_BUDGETS}
            budgets_unused = {b: [] for b in TOKEN_BUDGETS}
            budgets_zero = {b: 0 for b in TOKEN_BUDGETS}
            per_k["mrr"] = []
            rows: list[dict] = []

            for qi, q in enumerate(questions):
                base = dense_per_q[qi]
                if spec["rerank"]:
                    pool = base[: spec["pool"]]
                    ranked = tr.rerank(model, tok, q["question"], pool, device,
                                       args.eval_batch)
                    cut = sum(1 for c in pool
                              if pair_truncated(q["qid"], q["question"], c["text"]))
                    trunc_frac = round(cut / max(1, len(pool)), 4)
                else:
                    ranked = base[: spec["pool"]]
                    trunc_frac = None

                rank = ev.first_hit_rank(gold[qi], ranked, corpus)
                per_k["mrr"].append(1.0 / rank if rank else 0.0)
                per_q = {"qid": q["qid"], "group_id": q.get("group_id"),
                         "domain": q.get("domain"), "evidence_type": q.get("evidence_type"),
                         "strategy": strategy, "config": config, "track": track,
                         "split": args.split, "pool": spec["pool"],
                         "rerank": spec["rerank"], "first_hit_rank": rank}
                for k in KS:
                    rec, hit, comp, part = ev.coverage_for_set(gold[qi], ranked[:k], corpus)
                    per_k[k]["recall"].append(rec)
                    per_k[k]["hit"].append(1.0 if hit else 0.0)
                    per_k[k]["complete"].append(1.0 if comp else 0.0)
                    per_k[k]["partial"].append(part)
                    per_q[f"recall@{k}"] = round(rec, 6)
                    per_q[f"hit@{k}"] = 1.0 if hit else 0.0
                    per_q[f"complete@{k}"] = 1.0 if comp else 0.0
                    per_q[f"partial@{k}"] = round(part, 6)
                for b in TOKEN_BUDGETS:
                    taken, used, _stopped = ev.admission_token_budget(ranked, b)
                    rec, hit, comp, _part = ev.coverage_for_set(gold[qi], taken, corpus)
                    per_budget[b]["recall"].append(rec)
                    per_budget[b]["hit"].append(1.0 if hit else 0.0)
                    per_budget[b]["complete"].append(1.0 if comp else 0.0)
                    budgets_unused[b].append(b - used)
                    if not taken:
                        budgets_zero[b] += 1
                    per_q[f"tok{b}_recall"] = round(rec, 6)
                    per_q[f"tok{b}_complete"] = 1.0 if comp else 0.0
                per_q["mrr"] = round(1.0 / rank if rank else 0.0, 6)
                if trunc_frac is not None:
                    per_q["trunc_frac"] = trunc_frac
                rows.append(per_q)

            block = metric_block(per_k, per_budget, budgets_unused, budgets_zero,
                                 len(questions))
            for b in TOKEN_BUDGETS:
                bb = block["by_token_budget"][str(b)]
                block[f"tok{b}_recall"] = bb["recall"]
                block[f"tok{b}_complete"] = bb["complete"]
                block[f"tok{b}_hit"] = bb["hit"]
            block.update({"strategy": strategy, "config": config, "track": track,
                          "pool": spec["pool"], "rerank": spec["rerank"],
                          "split": args.split, "latency_ms": dense_ms,
                          "eval_time_s": round(time.time() - t_track, 1)})
            results[track][f"{strategy}_{config}"] = block
            perq[track].extend(rows)

        # ---- arithmetic assertions: C2 == main and C1 == main on set metrics
        if "main" in results and f"{strategy}_{config}" in results["main"]:
            key = f"{strategy}_{config}"
            main_rows = {(r["qid"]): r for r in perq["main"] if r["config"] == config
                         and r["strategy"] == strategy}
            for track in ("C1", "C2"):
                for r in perq[track]:
                    if r["config"] != config or r["strategy"] != strategy:
                        continue
                    m = main_rows.get(r["qid"])
                    if m is None:
                        continue
                    for metric in SET_METRIC_KEYS:
                        if abs(float(r[metric]) - float(m[metric])) > 1e-9:
                            invariance_errors.append(
                                f"{track} != main on {key} {r['qid']} {metric}: "
                                f"{r[metric]} vs {m[metric]}")
        print(f"  {cfg:<28} " + "  ".join(
            f"{t}:{results[t].get(f'{strategy}_{config}', {}).get('complete@10', float('nan')):.3f}"
            for t in TRACKS), flush=True)

    if invariance_errors:
        print("\nINVARIANCE ASSERTION FAILED - no results written:", file=sys.stderr)
        for err in invariance_errors[:10]:
            print(f"  {err}", file=sys.stderr)
        return 3
    print(f"\n  invariance: C1 == main and C2 == main on {', '.join(SET_METRIC_KEYS)} "
          f"for all {len(configs)} configs")

    # ---- write
    for track in TRACKS:
        (out_root / track).mkdir(parents=True, exist_ok=True)
        for name, block in results[track].items():
            write_json(out_root / track / f"{name}.json", block)
        fields = ["strategy", "config", "track", "pool", "rerank",
                  "recall@1", "recall@5", "recall@10", "hit@10", "complete@10",
                  "partial@10", "mrr", "latency_ms", "eval_time_s",
                  "tok512_recall", "tok1024_recall", "tok2048_recall",
                  "tok512_complete", "tok1024_complete", "tok2048_complete"]
        with (out_root / f"summary_{track}.csv").open("w", encoding="utf-8",
                                                      newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for name in sorted(results[track]):
                w.writerow(results[track][name])
        write_jsonl(out_root / f"perq_{track}.jsonl", perq[track])

    attempt.update({"status": "completed", "completed_at": now(),
                    "configs": len(configs), "questions": len(questions)})
    # `test2` must keep the same "one run, no force" guarantee as `test`; the note about the
    # pilot exposure applies only to the original split.
    note = ("The test split was evaluated once before (exploratory pilot, results/pilot/). "
            "This run is a formal re-evaluation with settings frozen beforehand; it is NOT "
            "a first, unseen use of the split.") if args.split == "test" else (
            "First and only evaluation of the second held-out split, against a design frozen "
            "in results/test2/PREREGISTRATION.json before the questions were drafted.")
    write_json(final_run, {
        "attempt": attempt_no, "split": args.split, "completed_at": now(),
        "frozen_sha256": attempt["frozen_sha256"],
        "lock_sha256": attempt["lock_sha256"],
        "code_sha256": attempt["code_sha256"],
        "inputs": {**input_hashes, **env_info},
        "configs": len(configs), "questions": len(questions), "tracks": list(TRACKS),
        "test_previously_used": args.split == "test",
        "note": note,
    })
    in_progress.unlink()
    print(f"  wrote {out_root}")

    # log
    import importlib
    logmod = importlib.import_module("15_experiment_log")
    logmod.append_log(root, {"kind": "eval", "run_id": logmod.run_id(
        corpus_version=corpus.corpus_version, strategy="all", config="all",
        split=args.split, track="four_track",
        code_sha=attempt["code_sha256"]),
        "track": "four_track", "split": args.split, "attempt": attempt_no,
        "configs": len(configs), "questions": len(questions),
        "corpus_version": corpus.corpus_version,
        "note": "formal four-track evaluation"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
