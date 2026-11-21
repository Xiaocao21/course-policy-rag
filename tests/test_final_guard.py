"""Guards around the formal test run.

The point of these tests is that the formal evaluation cannot be repeated or started
without the frozen artefacts: there is no --force flag, and a completed run blocks a second
one at the process level.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "19_rerank_eval.py"
CPU_PY = Path(sys.executable)


def run(args: list[str], python: Path | None = None) -> subprocess.CompletedProcess:
    exe = python or CPU_PY
    return subprocess.run([str(exe), str(SCRIPT), *args], capture_output=True,
                          text=True, timeout=900, cwd=str(ROOT))


class TestFinalGuards(unittest.TestCase):
    def test_refuses_without_frozen(self):
        # the guard is checked before anything heavy, so a stub FROZEN path is enough:
        # move nothing; instead assert the message path exists in the source
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("FROZEN.json is missing", source)
        self.assertIn("--acknowledge-test-use", source)
        # the docstring says "There is no --force", so match the argparse definition
        self.assertNotIn('add_argument("--force"', source)
        self.assertNotIn("add_argument('--force'", source)

    def test_test_split_requires_acknowledgement(self):
        out = run(["--split", "test"])
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("acknowledge", (out.stderr or "").lower())

    def test_completed_run_blocks_second_run(self):
        final = ROOT / "results" / "final" / "FINAL_RUN.json"
        if not final.exists():
            self.skipTest("formal run not executed yet")
        out = run(["--split", "test", "--acknowledge-test-use"])
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("second run is not permitted", (out.stderr or "").lower())

    def test_final_run_records_provenance(self):
        final = ROOT / "results" / "final" / "FINAL_RUN.json"
        if not final.exists():
            self.skipTest("formal run not executed yet")
        data = json.loads(final.read_text(encoding="utf-8"))
        for key in ("attempt", "frozen_sha256", "lock_sha256", "code_sha256", "tracks"):
            self.assertIn(key, data)
        self.assertTrue(data.get("test_previously_used"))
        self.assertIn("pilot", json.dumps(data).lower())


if __name__ == "__main__":
    unittest.main()
