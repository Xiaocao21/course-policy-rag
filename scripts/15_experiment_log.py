"""15_experiment_log.py - environment capture and a structured experiment log.

Two jobs, both required before any training result can be trusted.

`capture_env` records the machine as it actually is, separately for the CPU evaluation
interpreter and for the CUDA training interpreter, because the study now runs on two
different environments and they must never be mixed in one comparison.

`append_log` writes one JSON object per run to `experiments/log.jsonl`. A run id is a
deterministic hash of what the run consumed (corpus version, strategy, config, split,
track, code snapshot), so the same run always gets the same id and a different run
cannot silently reuse one.

Usage
    .venv\\Scripts\\python.exe scripts/15_experiment_log.py --env
    .venv\\Scripts\\python.exe scripts/15_experiment_log.py --show
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, project_paths, write_json  # noqa: E402

ENV_KEYS = ("torch", "cuda", "cuda_available", "transformers", "sentence_transformers",
            "faiss", "numpy", "python", "gpu_name", "gpu_memory_gb", "driver")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_json(py: Path, code: str) -> dict:
    """Evaluate `code` in another interpreter and return its JSON output."""
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True,
                             timeout=300, check=False)
    except (OSError, subprocess.SubprocessError):
        return {}
    text = (out.stdout or "").strip().splitlines()
    for line in reversed(text):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


PROBE = r"""
import json, sys
d = {"python": sys.version.split()[0]}
try:
    import torch
    d["torch"] = torch.__version__
    d["cuda"] = torch.version.cuda
    d["cuda_available"] = bool(torch.cuda.is_available())
    if d["cuda_available"]:
        d["gpu_name"] = torch.cuda.get_device_name(0)
        d["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
except Exception as exc:
    d["torch_error"] = f"{type(exc).__name__}: {exc}"
try:
    import transformers
    d["transformers"] = transformers.__version__
except Exception:
    pass
try:
    import sentence_transformers
    d["sentence_transformers"] = sentence_transformers.__version__
except Exception:
    pass
try:
    import faiss
    d["faiss"] = faiss.__version__
except Exception:
    pass
try:
    import numpy
    d["numpy"] = numpy.__version__
except Exception:
    pass
print(json.dumps(d))
"""


def driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or "").strip().splitlines()
    return line[0].strip() if line else None


def capture_env(root: Path) -> dict:
    paths = project_paths(root)
    cpu_py = root / ".venv" / "Scripts" / "python.exe"
    gpu_py = root / ".venv-gpu" / "Scripts" / "python.exe"

    cpu = run_json(cpu_py, PROBE) if cpu_py.exists() else {}
    gpu = run_json(gpu_py, PROBE) if gpu_py.exists() else {}

    drv = driver_version()
    if drv:
        cpu.setdefault("driver", drv)
        gpu.setdefault("driver", drv)

    return {
        "captured_at": now(),
        "host": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "total_memory_gb": _total_memory_gb(),
            "driver": drv,
        },
        "cpu_eval_env": {"interpreter": str(cpu_py), **cpu},
        "canonical_training_env": {"interpreter": str(gpu_py), **gpu},
        "canonical_policy": (
            "RTX 3060 Laptop 6GB with fp16 is the canonical training environment. "
            "Other GPUs may be used only for compatibility checks and must write "
            "COMPAT.json, never FROZEN.json; numbers from different environments are "
            "not comparable."
        ),
    }


def _total_memory_gb() -> float | None:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / 1024**3, 1)
    except Exception:  # noqa: BLE001
        return None


def code_sha256(root: Path) -> str:
    """Aggregate hash of the code snapshot that produced a run."""
    h = hashlib.sha256()
    for path in sorted((root / "scripts").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    for name in ("run.py", "MANIFEST.json"):
        p = root / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def run_id(*, corpus_version: str, strategy: str | None, config: str | None,
           split: str, track: str, code_sha: str) -> str:
    key = "|".join([corpus_version, strategy or "-", config or "-", split, track, code_sha])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def append_log(root: Path, row: dict) -> Path:
    paths = project_paths(root)
    log = paths["reports"].parent / "experiments" / "log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    row = {"logged_at": now(), **row}
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--env", action="store_true", help="capture and write experiments/env.json")
    ap.add_argument("--show", action="store_true", help="print the log summary")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    exp = paths["reports"].parent / "experiments"

    if args.env or not args.show:
        env = capture_env(root)
        write_json(exp / "env.json", env)
        print("=== environment ===")
        cpu = env["cpu_eval_env"]
        gpu = env["canonical_training_env"]
        print(f"  cpu eval   : torch {cpu.get('torch')} cuda {cpu.get('cuda')} "
              f"available {cpu.get('cuda_available')}")
        print(f"  gpu train  : torch {gpu.get('torch')} cuda {gpu.get('cuda')} "
              f"available {gpu.get('cuda_available')} "
              f"gpu {gpu.get('gpu_name')}")
        print(f"  driver     : {env['host']['driver']}")
        print(f"wrote {exp / 'env.json'}")
        print(f"code sha256  : {code_sha256(root)[:16]}...")

    if args.show:
        log = exp / "log.jsonl"
        if not log.exists():
            print("no experiments/log.jsonl yet")
            return 0
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"=== log: {len(rows)} rows ===")
        for r in rows[-12:]:
            print(f"  {r.get('logged_at','')[:19]} {r.get('kind','?'):<12} "
                  f"{r.get('run_id','?')} {r.get('track','')} "
                  f"{str(r.get('config',''))[:24]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
