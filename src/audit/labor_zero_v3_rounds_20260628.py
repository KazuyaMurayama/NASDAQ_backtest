"""
src/audit/labor_zero_v3_rounds_20260628.py
==========================================
Multi-round exploration of NEW method families after the v3 static sweep proved
strict all-46 labor-zero INFEASIBLE at fixed 7.2M / 40M (18%):
  - 0 of 1050 static configs reach labor-zero over all 46 starts.
  - Even a perfect-foresight oracle over the scale ladder cannot save 2013/2014
    (labor 3 / 6); only an oracle that ALSO picks N4/X4 with foresight clears them
    => no CAUSAL rule works. The wall is the 18% withdrawal vs the 2013-2016 return
    sequence (sc2.6 +29/+2/-38/-2), with bonds ALSO down in 2015.

So the static-allocation lever family is exhausted. This script tests genuinely
DIFFERENT lever families, in rounds, each measured against the same 46 starts:

ROUND A  FLEXIBLE / GUARDRAIL SPENDING (the most promising untested family):
  spend the full 7.2M normally, but in years AFTER a drawdown, cut spend toward a
  floor (Guyton-Klinger style). A "labor year" now = total can't fund even the
  FLOOR. Metric: (i) does all-46 labor-zero become feasible, (ii) how much total
  lifetime spend is sacrificed vs the rigid 7.2M*20 baseline (the PRICE of safety),
  (iii) worst-case realized spend.

ROUND B  DIFFERENT OBJECTIVE (zero is impossible -> minimize the damage):
  keep spend rigid 7.2M, but find the static config that MINIMIZES total labor-years
  and worst-start labor across all 46 starts (the best-effort fixed design).

ROUND C  HYBRID FLOOR + FLEXIBLE TOP:
  guarantee a fixed floor f every year (never cut below f), spend more (up to 7.2M)
  only when wealth supports it. Tests whether a modest guaranteed floor with an
  opportunistic top buys all-46 floor-survival.

Each round prints a frontier and writes a CSV. ASCII-only. No commit.
Outputs:
  audit_results/labor_zero_v3_roundA_flexspend_20260628.csv
  audit_results/labor_zero_v3_roundB_minlabor_20260628.csv
  audit_results/labor_zero_v3_roundC_hybrid_20260628.csv
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

FULL_SPEND = 7.2 * M
DATA_END = 2025
ALL_STARTS = list(range(1975, 2021))   # 46


def sim_flex(rets, reserve_series, start, horizon, *, strat, run0, reserve0, thr,
             full_spend=FULL_SPEND, floor_spend, dd_trigger, cut_to,
             data_end=DATA_END):
    """ROUND A / C engine: guardrail spending.
    Each year: compute this-year spend:
      - default = full_spend
      - if the run sleeve is in a drawdown >= dd_trigger vs its running peak,
        spend = max(floor_spend, full_spend * cut_to)   (a Guyton-Klinger-style cut)
    A LABOR year = total can't fund even THIS year's (possibly reduced) spend.
    Reserve top-up: single all-in at run<thr (the v2 mechanic, which dominated).
    Returns dict(labor_years, floor_M, total_spend_M, min_spend_M).
    """
    run, res = float(run0), float(reserve0)
    s = rets[strat]
    n = min(horizon, data_end - start + 1)
    peak = run
    labor = 0
    floor_tot = run + res
    total_spend = 0.0
    min_spend = full_spend
    fired = False
    for k in range(n):
        yr = start + k
        # top-up
        if (not fired) and run < thr and res > 1e-6:
            run += res
            res = 0.0
            fired = True
        # determine spend (guardrail on run drawdown)
        peak = max(peak, run)
        dd = (peak - run) / peak if peak > 1e-9 else 0.0
        spend = full_spend
        if dd >= dd_trigger:
            spend = max(floor_spend, full_spend * cut_to)
        # strict spend of THIS year's amount
        total = run + res
        if total + 1e-6 < spend:
            labor += 1
            run = res = 0.0
            spent = 0.0
        else:
            spent = spend
            if run >= spend:
                run -= spend
            else:
                res -= (spend - run)
                run = 0.0
        total_spend += spent
        min_spend = min(min_spend, spent if spent > 0 else min_spend)
        # growth
        run *= (1.0 + float(s.loc[yr]))
        res *= (1.0 + float(reserve_series.loc[yr]))
        if run + res < floor_tot:
            floor_tot = run + res
    return dict(labor_years=labor, floor_M=round(floor_tot / M, 2),
                total_spend_M=round(total_spend / M, 1),
                min_spend_M=round(min_spend / M, 2), n_years=n)


def agg_flex(rets, reserve_series, **kw):
    labor = 0
    fails = []
    floors = []
    spends = []
    minspends = []
    for sy in ALL_STARTS:
        r = sim_flex(rets, reserve_series, sy, 20, **kw)
        labor += r["labor_years"]
        if r["labor_years"] > 0:
            fails.append(sy)
        floors.append(r["floor_M"])
        spends.append(r["total_spend_M"])
        minspends.append(r["min_spend_M"])
    return dict(labor_total=labor, n_fail=len(fails), fails=fails,
                min_floor_M=float(np.min(floors)),
                med_total_spend_M=float(np.median(spends)),
                min_total_spend_M=float(np.min(spends)),
                worst_min_spend_M=float(np.min(minspends)))


def round_A(rets, bond):
    print("\n" + "=" * 95)
    print("ROUND A: FLEXIBLE / GUARDRAIL SPENDING (full 7.2M normally, cut after drawdowns)")
    print("=" * 95)
    print("  baseline rigid 7.2M*20 = 144.0M lifetime spend. Question: what flex reaches all-46 labor-zero,")
    print("  and how much lifetime spend is sacrificed (the PRICE of never working)?")
    rows = []
    # grid: floor_spend (M), dd_trigger, cut_to ; config = best v2 base (sc2.6 14/26 bond all-in@14M)
    for floor_M in [3.0, 4.0, 4.5, 5.0, 5.5, 6.0]:
        for dd_trig in [0.15, 0.25, 0.35]:
            for cut_to in [0.5, 0.6, 0.7, 1.0]:
                r = agg_flex(rets, bond, strat="sc2.6", run0=14 * M, reserve0=26 * M,
                             thr=14 * M, floor_spend=floor_M * M, dd_trigger=dd_trig,
                             cut_to=cut_to)
                rows.append(dict(floor_spend_M=floor_M, dd_trigger=dd_trig, cut_to=cut_to,
                                 **{k: r[k] for k in ("labor_total", "n_fail", "min_floor_M",
                                    "med_total_spend_M", "min_total_spend_M", "worst_min_spend_M")},
                                 fails=";".join(str(x) for x in r["fails"])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v3_roundA_flexspend_20260628.csv"),
              index=False, encoding="utf-8-sig")
    passers = df[df.labor_total == 0].sort_values(
        ["med_total_spend_M", "min_floor_M"], ascending=[False, False])
    print("\n  ALL-46 labor-zero (never below the year's reduced spend) configs: %d" % len(passers))
    if len(passers):
        print("  %-9s %-9s %-7s %-9s %-12s %-12s %s"
              % ("floorM", "dd_trig", "cut", "floorM_tot", "medSpendM", "minSpendM", "worstYrSpendM"))
        for _, r in passers.head(12).iterrows():
            print("  %-9.1f %-9.2f %-7.1f %-9.2f %-12.1f %-12.1f %.2f"
                  % (r["floor_spend_M"], r["dd_trigger"], r["cut_to"], r["min_floor_M"],
                     r["med_total_spend_M"], r["min_total_spend_M"], r["worst_min_spend_M"]))
        best = passers.iloc[0]
        print("\n  => best (max lifetime spend among all-46-safe): floor %.1fM, dd>=%.0f%%, cut to %.0f%% of 7.2M"
              % (best["floor_spend_M"], best["dd_trigger"] * 100, best["cut_to"] * 100))
        print("     median lifetime spend %.1fM (vs rigid 144.0M = %.1f%% of full), worst single-year spend %.2fM"
              % (best["med_total_spend_M"], best["med_total_spend_M"] / 144.0 * 100, best["worst_min_spend_M"]))
    else:
        print("  NONE even with flexible spending down to the floors tested.")
    return df, passers


def round_B(rets, mixes):
    print("\n" + "=" * 95)
    print("ROUND B: DIFFERENT OBJECTIVE -- rigid 7.2M, minimize total labor-years & worst-start labor")
    print("=" * 95)
    STRATS = ["sc2.6", "sc2.4", "sc2.2", "sc2.0", "X4eq_sc2.2", "N4"]
    RUN0_M = [10, 12, 14, 16, 18, 20]
    rows = []
    for strat in STRATS:
        for run0_M in RUN0_M:
            run0 = run0_M * M
            res0 = 40 * M - run0
            for mname, mser in mixes.items():
                worst = 0
                tot = 0
                fails = []
                for sy in ALL_STARTS:
                    r = v3.simulate_v3(rets, mser, sy, 20, strat=strat, run0=run0,
                                       reserve0=res0, tranches=[(min(14 * M, run0), 1.0)])
                    tot += r["labor_years"]
                    worst = max(worst, r["labor_years"])
                    if r["labor_years"] > 0:
                        fails.append(sy)
                rows.append(dict(strat=strat, run0_M=run0_M, reserve_mix=mname,
                                 labor_total=tot, worst_start_labor=worst, n_fail=len(fails),
                                 fails=";".join(str(x) for x in fails)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v3_roundB_minlabor_20260628.csv"),
              index=False, encoding="utf-8-sig")
    best = df.sort_values(["labor_total", "worst_start_labor", "n_fail"]).head(8)
    print("\n  best static configs by (total labor, worst-start labor):")
    print("  %-11s %-6s %-10s %-7s %-7s %-7s %s"
          % ("strat", "run0M", "mix", "totLab", "worst", "nFail", "fails"))
    for _, r in best.iterrows():
        print("  %-11s %-6d %-10s %-7d %-7d %-7d %s"
              % (r["strat"], r["run0_M"], r["reserve_mix"], r["labor_total"],
                 r["worst_start_labor"], r["n_fail"], r["fails"]))
    return df, best


def round_C(rets, bond):
    print("\n" + "=" * 95)
    print("ROUND C: HYBRID -- guarantee a fixed floor f every year, top up toward 7.2M when wealth allows")
    print("=" * 95)
    print("  (modelled as guardrail with cut_to=floor/7.2 always-applied below a wealth test; here we sweep")
    print("   a pure floor f that is ALWAYS spent, never more -> the max constant spend that is all-46 safe)")
    rows = []
    for f_M in [4.0, 4.4, 4.43, 4.5, 4.7, 5.0, 5.5, 6.0, 6.5, 7.0]:
        r = agg_flex(rets, bond, strat="sc2.6", run0=14 * M, reserve0=26 * M, thr=14 * M,
                     floor_spend=f_M * M, dd_trigger=0.0, cut_to=f_M / 7.2)
        # dd_trigger=0 with cut_to=f/7.2 => always spend exactly f_M (constant)
        rows.append(dict(constant_spend_M=f_M, labor_total=r["labor_total"], n_fail=r["n_fail"],
                         min_floor_M=r["min_floor_M"], fails=";".join(str(x) for x in r["fails"])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v3_roundC_hybrid_20260628.csv"),
              index=False, encoding="utf-8-sig")
    print("\n  constant-spend frontier (the guaranteed floor you can ALWAYS take, all-46 labor-zero):")
    print("  %-14s %-9s %-7s %-10s %s" % ("constSpendM", "totLabor", "nFail", "floorM", "fails"))
    for _, r in df.iterrows():
        mark = "  <- max safe" if r["labor_total"] == 0 else ""
        print("  %-14.2f %-9d %-7d %-10.2f %s%s"
              % (r["constant_spend_M"], r["labor_total"], r["n_fail"], r["min_floor_M"],
                 r["fails"], mark))
    safe = df[df.labor_total == 0]["constant_spend_M"]
    max_safe = float(safe.max()) if len(safe) else None
    print("\n  => max guaranteed constant spend (all-46 labor-zero) = %s M/yr"
          % ("%.2f" % max_safe if max_safe else "none"))
    return df, max_safe


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 95)
    print("LABOR-ZERO v3 -- MULTI-ROUND NEW-METHOD EXPLORATION  (2026-06-28)")
    print("static allocation exhausted (infeasible at 18%); testing flexible-spend / objective / hybrid")
    print("=" * 95)
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    mixes = {"bond": v3.load_mixed_reserve({"bond": 1.0}),
             "b70g20c10": v3.load_mixed_reserve({"bond": 0.7, "gold": 0.2, "cash": 0.1}),
             "cash": v3.load_mixed_reserve({"cash": 1.0})}

    dfA, passA = round_A(rets, bond)
    dfB, bestB = round_B(rets, mixes)
    dfC, max_safe_C = round_C(rets, bond)

    block = {
        "script": "labor_zero_v3_rounds_20260628.py", "date": "2026-06-28",
        "roundA_n_pass": int(len(passA)),
        "roundA_best": (passA.iloc[0].to_dict() if len(passA) else None),
        "roundB_best": bestB.iloc[0].to_dict(),
        "roundC_max_safe_constant_spend_M": max_safe_C,
    }
    print("\n" + "=" * 95)
    print("RETURN_BLOCK")
    print("=" * 95)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True, default=str))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
