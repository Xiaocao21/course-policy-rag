"""evaluate.py - retrieval replay (R2) with isolation and a diagnostic-only fallback.

Two modes:

formal reproduction  (needs data/STATUS.json reproduction_grade=true)
    runs `19_rerank_eval.py` inside a temporary staging root, so the official results and the
    official `experiments/log.jsonl` are untouched; copies the per-question results to
    `results/repro/<name>/`, writes `REPRO_RUN.json` (never `FINAL_RUN.json`) and a comparison.

diagnostic           (--diagnostic, or when the gate is not satisfied)
    runs under the system temp directory, renames the outputs to `diag_perq_*`/`diag_summary_*`,
    writes `NONCOMPARABLE.json` and `DIAGNOSTIC.json`, and never produces a `FINAL_RUN.json`.

Usage
    python src/evaluate.py --config configs/proposed_model.yaml --dry-run
    python src/evaluate.py --config configs/proposed_model.yaml
    python src/evaluate.py --config configs/proposed_model.yaml --diagnostic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from repro import stage  # noqa: E402
from repro.compare import compare  # noqa: E402
import config as cfg  # noqa: E402

OFFICIAL = ("results/final", "results/final_test2", "results/test2")
GUARDED = ("models/reranker/FROZEN.json", "results/final/FINAL_RUN.json",
           "results/final_test2/FINAL_RUN.json", "results/final/stats.json",
           "results/final_test2/stats.json", "results/test2/OUTCOME.json",
           "experiments/log.jsonl")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def guarded_hashes(root: Path) -> dict[str, str]:
    return {rel: sha256(root / rel) for rel in GUARDED}


def resolve_out(root: Path, out: str) -> Path:
    path = Path(out)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    for official in OFFICIAL:
        base = (root / official).resolve()
        if resolved == base or base in resolved.parents:
            raise SystemExit(f"REFUSING: output {resolved} is inside the official tree {base}")
    if (resolved / "FINAL_RUN.json").exists():
        raise SystemExit(f"REFUSING: {resolved} already contains FINAL_RUN.json")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default="",
                    help="interpreter with torch (default: .venv-gpu if present, else this one)")
    args = ap.parse_args()
    conf = cfg.load(ROOT / args.config)

    status_path = ROOT / "data" / "STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    grade = bool(status.get("reproduction_grade"))
    diagnostic = args.diagnostic or not grade
    if not grade and not args.diagnostic:
        print("NOTE: data/STATUS.json is not reproduction_grade; running in DIAGNOSTIC mode.\n"
              "      Output is marked NONCOMPARABLE and cannot enter any result table.")

    if args.out:
        out = args.out
    elif diagnostic:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = str(Path(tempfile.gettempdir()) / "cpr-diagnostic" / f"{conf['name']}-{stamp}")
    else:
        out = conf.get("out", f"results/repro/{conf['name']}")
    out_path = resolve_out(ROOT, out)

    python = Path(args.python) if args.python else None
    if python is None:
        gpu = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
        python = gpu if gpu.exists() else Path(sys.executable)
    if not Path(python).exists():
        raise SystemExit(f"REFUSING: interpreter {python} does not exist")

    work = Path(tempfile.mkdtemp(prefix="cpr-stage-"))
    stage.build(ROOT, work)
    staged_out = work / "results" / "repro" / conf["name"]
    cmd = [str(python), str(ROOT / "scripts" / "19_rerank_eval.py"),
           "--root", str(work), "--split", conf["split"],
           "--questions", conf["questions"], "--locked-dir", conf["locked_dir"],
           "--device", conf.get("device", "cuda"),
           "--eval-batch", str(conf.get("eval_batch", 32)),
           "--out", str(staged_out)]
    if conf.get("preregistration"):
        cmd += ["--preregistration", conf["preregistration"]]
    if conf["split"] == "test":
        cmd += ["--acknowledge-test-use"]
    print("$ " + " ".join(cmd[:5]) + " ...")
    if args.dry_run:
        print(f"staging root: {work}\noutput: {out_path}")
        stage.cleanup(work)
        return 0

    before = guarded_hashes(ROOT)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        stage.cleanup(work)
        print("REFUSING: the evaluation did not complete; see the log above", file=sys.stderr)
        return result.returncode

    final_run = json.loads((staged_out / "FINAL_RUN.json").read_text(encoding="utf-8"))
    for track in ("main", "C1", "C2", "C3"):
        src = staged_out / f"perq_{track}.jsonl"
        name = f"diag_perq_{track}.jsonl" if diagnostic else f"perq_{track}.jsonl"
        shutil.copy2(src, out_path / name)
        summary = staged_out / f"summary_{track}.csv"
        if summary.exists():
            shutil.copy2(summary, out_path / (f"diag_summary_{track}.csv" if diagnostic
                                              else f"summary_{track}.csv"))
    payload = {**final_run, "official": False,
               "mode": "diagnostic" if diagnostic else "reproduction",
               "source_run": str(conf.get("official_run", ""))}
    if diagnostic:
        (out_path / "NONCOMPARABLE.json").write_text(json.dumps({
            "official": False, "comparable": False, "not_for_results": True,
            "reasons": status.get("reasons", ["data gate not satisfied"]),
        }, indent=1), encoding="utf-8")
        (out_path / "DIAGNOSTIC.json").write_text(json.dumps(payload, indent=1),
                                                  encoding="utf-8")
    else:
        (out_path / "REPRO_RUN.json").write_text(json.dumps(payload, indent=1),
                                                 encoding="utf-8")
        report = compare(ROOT / conf["official_results"], out_path)
        (out_path / "compare.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"per-question identical: {report['totals']['identical']}/"
              f"{report['totals']['rows']} (different {report['totals']['different']})")
    after = guarded_hashes(ROOT)
    unchanged = before == after
    (out_path / "run_meta.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config, "staging_root": str(work),
        "official_hashes_unchanged": unchanged, "mode": payload["mode"],
    }, indent=1), encoding="utf-8")
    stage.cleanup(work)
    if not unchanged:
        print("FAIL: an official artefact changed during the run", file=sys.stderr)
        return 1
    print(f"wrote {out_path} ({payload['mode']}); official hashes unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
