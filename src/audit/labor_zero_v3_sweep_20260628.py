"""
src/audit/labor_zero_v3_sweep_20260628.py
=========================================
MAIN deterministic v3 sweep over ALL 46 starts (1975-2020), strict labor-zero.
This is the independent grid the Workflow agents are cross-checked against.

Strict pass = labor_years_total == 0 over all 46 starts at data-max horizon.
Among passers, rank by max min_floor_M (safety), then term_median_M.

Stages:
  STAGE 0  diagnostic: does SPLIT-TRANCHE ALONE (sc2.6, run14/res26 bond) save
           2012-2014? (isolates the single most-promising lever)
  STAGE 1  full combined grid (strat x run0 x tranches x reserve_mix), eval all 46.
  Output the PASS set (if any) and, regardless, the min-fail frontier.

ASCII-only. No commit. Output: audit_results/labor_zero_v3_sweep_20260628.csv
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_THIS_DIR, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
M = v3.M
AR_DIR = os.path.join(_REPO_DIR, "audit_results")


# reserve mixes (name -> weights)
RESERVE_MIXES = {
    "bond": {"bond": 1.0},
    "b70g20c10": {"bond": 0.7, "gold": 0.2, "cash": 0.1},
    "b50c50": {"bond": 0.5, "cash": 0.5},
    "g50b50": {"gold": 0.5, "bond": 0.5},
    "b60g20s20": {"bond": 0.6, "gold": 0.2, "sofr": 0.2},
    "cash": {"cash": 1.0},
}

# tranche schedules: list of (threshold_M, frac_of_remaining_reserve). thr descending.
# thresholds are clipped to <= run0 at eval time.
def tranche_sets(run0_M):
    return {
        "v2_allin": [(14, 1.0)],
        "split2_half": [(14, 0.5), (8, 1.0)],
        "split2_60": [(14, 0.6), (7, 1.0)],
        "split3_even": [(16, 0.34), (10, 0.5), (6, 1.0)],
        "split3_front": [(18, 0.5), (11, 0.5), (6, 1.0)],
        "split3_back": [(14, 0.25), (9, 0.4), (5, 1.0)],
        "split4": [(18, 0.3), (13, 0.4), (8, 0.5), (4, 1.0)],
    }


def _mk_tranches(sched_M, run0):
    """Convert M-thresholds to absolute, clip thr<=run0, keep order/fracs."""
    out = []
    for thr_M, frac in sched_M:
        thr = min(thr_M * M, run0)
        out.append((thr, frac))
    return out


def eval_cfg(rets, reserve_series, strat, run0, reserve0, tranches):
    return v3.run_all_starts_v3(rets, reserve_series, strat=strat, run0=run0,
                                reserve0=reserve0, tranches=tranches,
                                starts=v3.ALL_STARTS, horizon=20)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 95)
    print("v3 MAIN SWEEP -- strict labor-zero over ALL 46 starts (1975-2020)")
    print("=" * 95)
    rets = v3.load_rets()
    mixes = {k: v3.load_mixed_reserve(w) for k, w in RESERVE_MIXES.items()}

    # ---- STAGE 0: split-tranche ALONE (sc2.6, run14/res26 bond) ----
    print("\n--- STAGE 0: does split-tranche alone save 2012-2014? (sc2.6 run14/res26 bond) ---")
    for name, sched in tranche_sets(14 * M).items():
        tr = _mk_tranches(sched, 14 * M)
        r = eval_cfg(rets, mixes["bond"], "sc2.6", 14 * M, 26 * M, tr)
        print("  %-12s labor=%-4d n_fail=%-2d/46 floor=%7.2fM fails=%s"
              % (name, r["labor_years_total"], r["n_fail"], r["min_floor_M"],
                 r["starts_fail"]))

    # ---- STAGE 1: full combined grid ----
    print("\n--- STAGE 1: full combined grid (strat x run0 x tranche x reserve_mix) ---")
    STRATS = ["sc2.6", "sc2.4", "sc2.2", "sc2.0", "X4eq_sc2.2"]
    RUN0_M = [10, 12, 14, 16, 18]
    rows = []
    n = 0
    for strat in STRATS:
        for run0_M in RUN0_M:
            run0 = run0_M * M
            reserve0 = 40 * M - run0
            for tname, sched in tranche_sets(run0_M).items():
                tr = _mk_tranches(sched, run0)
                for mname, mser in mixes.items():
                    r = eval_cfg(rets, mser, strat, run0, reserve0, tr)
                    rows.append(dict(
                        strat=strat, run0_M=run0_M, reserve0_M=40 - run0_M,
                        tranche=tname, reserve_mix=mname,
                        labor_total=r["labor_years_total"], n_fail=r["n_fail"],
                        min_floor_M=r["min_floor_M"], term_median_M=r["term_median_M"],
                        fails=";".join(str(x) for x in r["starts_fail"]),
                    ))
                    n += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v3_sweep_20260628.csv"),
              index=False, encoding="utf-8-sig")
    print("  evaluated %d configs." % n)

    # ---- PASS set ----
    passers = df[df.labor_total == 0].sort_values(
        ["min_floor_M", "term_median_M"], ascending=[False, False])
    print("\n" + "=" * 95)
    print("STRICT PASS configs (labor_total==0 over all 46 starts): %d" % len(passers))
    print("=" * 95)
    if len(passers):
        print("  %-11s %-6s %-6s %-12s %-10s %-9s %s"
              % ("strat", "run0M", "resM", "tranche", "mix", "floorM", "termMedM"))
        for _, r in passers.head(15).iterrows():
            print("  %-11s %-6d %-6d %-12s %-10s %-9.2f %.1f"
                  % (r["strat"], r["run0_M"], r["reserve0_M"], r["tranche"],
                     r["reserve_mix"], r["min_floor_M"], r["term_median_M"]))
    else:
        print("  NONE. Strict all-46 labor-zero not found in this grid.")

    # ---- min-fail frontier (best even if not strict-pass) ----
    print("\n--- min-fail frontier (fewest failing starts, then highest floor) ---")
    minfail = df.sort_values(["n_fail", "min_floor_M"], ascending=[True, False])
    for _, r in minfail.head(12).iterrows():
        print("  n_fail=%-2d floor=%7.2fM %-11s run%d/res%d %-12s %-10s fails=%s"
              % (r["n_fail"], r["min_floor_M"], r["strat"], r["run0_M"],
                 r["reserve0_M"], r["tranche"], r["reserve_mix"], r["fails"]))

    block = {
        "script": "labor_zero_v3_sweep_20260628.py", "date": "2026-06-28",
        "n_configs": n, "n_pass_strict": int(len(passers)),
        "best_pass": (passers.head(1).to_dict("records")[0] if len(passers) else None),
        "min_n_fail": int(df["n_fail"].min()),
        "best_minfail": minfail.head(1).to_dict("records")[0],
    }
    print("\n" + "=" * 95)
    print("RETURN_BLOCK")
    print("=" * 95)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
