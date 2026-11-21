"""28_decide_test2.py - apply the frozen decision matrix to the test2 statistics.

Why a script. The interpretation of test2 must not be chosen after seeing the intervals. This
script reads the pre-registered endpoints from `results/test2/PREREGISTRATION.json`, reads the
statistics produced by `16_stats.py --split test2`, and writes the claim, using strict
inequalities only (equality never supports a claim).

Endpoints (sign fixed ex ante):
    primary    delta_p = Complete@10(structural) - Complete@10(semantic)   on C1
               structural = {heading/heading, paragraph/paragraph}
               semantic   = {semantic/thr0.70, thr0.75, thr0.80}
    secondary  delta_s = C3 - C1 Complete@10
    unit = group_id, percentile bootstrap (B=2000, seed 20261013)
    16_stats reports semantic-minus-structural, so delta_p = -(that value); the script
    asserts the sign rather than assuming it.

Multiplicity: fixed-sequence gatekeeping. The secondary is interpreted only if the primary
confirms; otherwise it is reported as descriptive.

Usage
    .venv\\Scripts\\python.exe scripts/28_decide_test2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, write_json  # noqa: E402

PRIMARY_THRESHOLD = 0.30
SECONDARY_DELTA = 0.06
SECONDARY_SMALLER = 0.047


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def classify_primary(lower: float, upper: float) -> tuple[str, str]:
    if lower > PRIMARY_THRESHOLD:
        return ("confirmed_at_least_0.30",
                "The structural family leads the semantic family by at least 0.30 "
                f"on the unseen set (95% CI [{lower:.3f}, {upper:.3f}]).")
    if upper < PRIMARY_THRESHOLD:
        return ("confirmed_below_0.30",
                "The gap is confirmed to be below 0.30 "
                f"(95% CI [{lower:.3f}, {upper:.3f}]); the ordering conclusion is weaker than "
                "on the first test split.")
    if lower > 0:
        return ("positive_not_confirmed",
                "The direction is positive but the pre-registered 0.30 threshold is not "
                f"confirmed (95% CI [{lower:.3f}, {upper:.3f}]). This is not evidence that "
                "structural chunking is 'significantly better'.")
    return ("indeterminate",
            "No directional difference was confirmed at this sample size "
            f"(95% CI [{lower:.3f}, {upper:.3f}]); the interval is compatible with zero and "
            "with small positive effects.")


def classify_secondary(lower: float, upper: float, primary_confirmed: bool) -> tuple[str, str]:
    if not primary_confirmed:
        return ("descriptive_only",
                "Secondary endpoint reported descriptively only, because the primary endpoint "
                "did not confirm (gatekeeping).")
    if lower > SECONDARY_DELTA:
        return ("exceeds_0.06",
                f"Reranking effect above 0.06 supported (95% CI [{lower:.3f}, {upper:.3f}]).")
    if upper < SECONDARY_SMALLER:
        return ("below_0.047",
                "The reranking effect is confirmed to be smaller than the first test split's "
                f"+0.047 (95% CI [{lower:.3f}, {upper:.3f}]).")
    if lower > 0:
        return ("positive_magnitude_unresolved",
                "Reranking direction is positive (95% CI "
                f"[{lower:.3f}, {upper:.3f}]); the magnitude is not resolved at this n.")
    return ("indeterminate",
            "Reranking is not confirmed; the interval cannot exclude effects below the MDE, "
            f"including +0.047 (95% CI [{lower:.3f}, {upper:.3f}]). This is not evidence of no "
            "effect and not a failed replication.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--stats", default="results/final_test2/stats.json")
    ap.add_argument("--prereg", default="results/test2/PREREGISTRATION.json")
    ap.add_argument("--out", default="results/test2/OUTCOME.json")
    args = ap.parse_args()

    root = Path(args.root)
    stats_path = root / args.stats
    pre_path = root / args.prereg
    if not stats_path.exists():
        print(f"REFUSING: missing {stats_path}; run 16_stats.py --split test2 first",
              file=sys.stderr)
        return 2
    if not pre_path.exists():
        print(f"REFUSING: missing {pre_path}; the decision rules must be pre-registered",
              file=sys.stderr)
        return 2

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if pre.get("status") != "finalised":
        print("REFUSING: the pre-registration is not finalised", file=sys.stderr)
        return 2
    if stats.get("split") != "test2":
        print(f"REFUSING: stats are for split {stats.get('split')!r}, not test2", file=sys.stderr)
        return 2
    if not stats.get("legacy_questions_used_for_statistics") is False:
        print("REFUSING: test2 statistics were not produced from the test2 questions file",
              file=sys.stderr)
        return 2

    fam = stats["family_contrasts"]["C1:semantic-minus-structural"]
    gap = {"delta": -fam["delta"],
           "ci_low": -fam["ci_high"],
           "ci_high": -fam["ci_low"],
           "n_groups": fam["n_groups"]}
    contrast = stats["contrasts"]["C3-C1"]
    secondary = {"delta": contrast["delta"], "ci_low": contrast["ci_low"],
                 "ci_high": contrast["ci_high"]}

    # sign assertion: the registered primary is structural minus semantic
    assert abs(gap["delta"] - (-fam["delta"])) < 1e-9, "sign mismatch"

    primary_code, primary_claim = classify_primary(gap["ci_low"], gap["ci_high"])
    primary_confirmed = primary_code == "confirmed_at_least_0.30"
    secondary_code, secondary_claim = classify_secondary(
        secondary["ci_low"], secondary["ci_high"], primary_confirmed)

    power = pre.get("power", {})
    out = {
        "kind": "test2_outcome",
        "created_at": now(),
        "split": "test2",
        "questions": stats.get("questions"),
        "groups": stats.get("groups"),
        "questions_source": stats.get("questions_source"),
        "ci_method": pre["endpoints"]["statistics"]["ci_method"],
        "primary": {"name": pre["endpoints"]["primary"]["name"], **gap,
                    "threshold": PRIMARY_THRESHOLD, "verdict": primary_code,
                    "claim": primary_claim},
        "secondary": {"name": pre["endpoints"]["secondary"]["name"], **secondary,
                      "verdict": secondary_code, "claim": secondary_claim,
                      "gatekeeping": ("interpreted only because the primary confirmed"
                                      if primary_confirmed else
                                      "descriptive only; primary did not confirm")},
        "power": {"n_groups": power.get("n_groups"),
                  "primary_mde": power.get("structural_vs_semantic", {}).get("mde_at_n"),
                  "secondary_mde": power.get("reranking_gain", {}).get("mde_at_n"),
                  "secondary_power_at_0.047":
                      power.get("reranking_gain", {}).get("power_at_test_effect_0.047")},
        "limits": ("test2 can resolve a development-sized effect and cannot reliably exclude "
                   "a +0.047 effect; a CI that includes zero is neither evidence of no effect "
                   "nor a failed replication."),
        "disclosure": ("test2 was drafted and reviewed by the agent (agent_self_audit), with "
                       "no human sign-off; the composition amendments in "
                       "results/test2/DESIGN_AMENDMENTS.json apply."),
        "note": ("Rules were fixed in results/test2/PREREGISTRATION.json before evaluation; "
                 "this file only applies them."),
    }
    target = root / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, out)

    print("=== test2 outcome ===")
    print(f"  primary  : {gap['delta']:+.3f} [{gap['ci_low']:+.3f}, {gap['ci_high']:+.3f}]"
          f"  -> {primary_code}")
    print(f"  secondary: {secondary['delta']:+.3f} [{secondary['ci_low']:+.3f}, "
          f"{secondary['ci_high']:+.3f}]  -> {secondary_code}")
    print(f"  {primary_claim}")
    print(f"  {secondary_claim}")
    print(f"  wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
