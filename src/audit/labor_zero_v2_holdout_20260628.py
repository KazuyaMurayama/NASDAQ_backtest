"""
src/audit/labor_zero_v2_holdout_20260628.py
===========================================
OUT-OF-SAMPLE / REPRODUCIBILITY check of the labor-zero "v2" design on start
years the v2 search NEVER used.

v2 design (frozen):
    P09_STR scale2.6 run 14M / reserve 26M in 1x BOND / top-up-ALL when run<14M,
    spend 7.2M strict every year. Budget = run+reserve = 40M.

v2 was tuned on starts 1975..2005 (x20y). The user's concern is over-fitting to
that 30-year window. This script re-runs the SAME frozen v2 on:
    - HOLDOUT starts 2006..2020 (never in the v2 sweep), and
    - because data ends in 2025, the horizon is capped at min(H, 2025-start+1);
      we therefore report survival at MULTIPLE horizons H in {5,10,15,20} (each
      start uses up to as many years as data allows), per the user request
      ("5-20 years, do they survive, and what is the minimum total assets").

For each (start, horizon) we record:
    labor_years  : years external earning is forced (the failure metric)
    floor_M      : minimum of (run+reserve) over the path  (how low assets dip)
    floor_yr     : the calendar year of that minimum
    terminal_M   : run+reserve at the end of the horizon
    n_years      : actual years simulated (<= horizon if data-limited)

It also re-runs the IN-SAMPLE 1975..2005 at the same multi-horizon grid so the
holdout can be compared apples-to-apples (does the design degrade out of sample?).

ASCII-only prints. No commit. Outputs:
  audit_results/labor_zero_v2_holdout_20260628.csv        (per start x horizon)
  audit_results/labor_zero_v2_holdout_summary_20260628.csv (per horizon, IS vs holdout)
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
    "h2", os.path.join(_THIS_DIR, "labor_zero_harness_v2_20260627.py"))
h2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h2)

AR_DIR = os.path.join(_REPO_DIR, "audit_results")
M = 1_000_000.0

# ----- v2 frozen config -----
V2 = dict(strat="sc2.6", run0=14 * M, reserve0=26 * M, rmode="bond", thr=14 * M)
SPEND = 7.2 * M
DATA_END = 2025

IS_STARTS = list(range(1975, 2006))       # in-sample (v2 was tuned here)
HOLDOUT_STARTS = list(range(2006, 2021))  # 2006..2020 (never used by v2)
HORIZONS = [5, 10, 15, 20]


def sim_one(rets, sleeves, start, horizon, *, strat, run0, reserve0, rmode, thr,
            spend=SPEND, data_end=DATA_END):
    """One start, capped at min(horizon, data_end-start+1). Returns dict.
    Same mechanic as harness simulate_v2 (one-way top-up-all)."""
    run, res = float(run0), float(reserve0)
    s = rets[strat]
    b = sleeves[rmode]
    n = min(horizon, data_end - start + 1)
    labor = 0
    floor = run + res
    floor_yr = start
    topups = 0
    for k in range(n):
        yr = start + k
        # 1. top-up (one-way, all-in) when run < thr
        if run < thr and res > 1e-6:
            run += res
            res = 0.0
            topups += 1
        # 2. strict spend
        total = run + res
        if total + 1e-6 < spend:
            labor += 1
            run = res = 0.0
        else:
            if run >= spend:
                run -= spend
            else:
                res -= (spend - run)
                run = 0.0
        # 3. growth
        run *= (1.0 + float(s.loc[yr]))
        res *= (1.0 + float(b.loc[yr]))
        tot = run + res
        if tot < floor:
            floor = tot
            floor_yr = yr
    return dict(start=start, horizon=horizon, n_years=n, labor_years=labor,
                topups=topups, floor_M=round(floor / M, 2), floor_yr=floor_yr,
                terminal_M=round((run + res) / M, 1))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("LABOR-ZERO v2 HOLDOUT / REPRODUCIBILITY CHECK  (2026-06-28)")
    print("v2 = sc2.6 run14M / reserve26M BOND / top-up-all @14M, spend 7.2M strict")
    print("=" * 100)

    rets = h2.load_returns()
    rets.update(h2.load_extended())
    sleeves = h2.load_sleeve_returns()

    # sanity: v2 in-sample 20y reproduces labor=0 / floor 18.12 (matches the report)
    is20 = [sim_one(rets, sleeves, sy, 20, **V2) for sy in IS_STARTS]
    is20_labor = sum(r["labor_years"] for r in is20)
    is20_floor = min(r["floor_M"] for r in is20)
    print("\n--- SANITY: v2 in-sample 1975-2005 x20y ---")
    print("  labor_years_total=%d (expect 0)   floor=%.2fM (expect ~18.12)"
          % (is20_labor, is20_floor))
    ok = (is20_labor == 0 and abs(is20_floor - 18.12) < 0.2)
    print("  -> %s" % ("MATCH report" if ok else "MISMATCH -- investigate"))

    rows = []
    for grp, starts in (("IS_1975_2005", IS_STARTS), ("HOLDOUT_2006_2020", HOLDOUT_STARTS)):
        for sy in starts:
            for H in HORIZONS:
                r = sim_one(rets, sleeves, sy, H, **V2)
                r["group"] = grp
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v2_holdout_20260628.csv"),
              index=False, encoding="utf-8-sig")

    # ---- holdout detail per start (the headline of the user's question) ----
    print("\n" + "=" * 100)
    print("HOLDOUT 2006-2020: per-start survival (data ends 2025, so horizon is capped)")
    print("=" * 100)
    print("  %-6s %-7s %-7s %-9s %-10s %-9s %s"
          % ("start", "horizon", "n_yrs", "labor", "floor_M", "floor_yr", "terminal_M"))
    for sy in HOLDOUT_STARTS:
        for H in HORIZONS:
            r = df[(df.group == "HOLDOUT_2006_2020") & (df.start == sy) & (df.horizon == H)].iloc[0]
            tag = "" if r["labor_years"] == 0 else "  <-- LABOR"
            # only print horizons that are not pure duplicates of the data-capped one
            if H == HORIZONS[0] or r["n_years"] != df[(df.group == "HOLDOUT_2006_2020") & (df.start == sy) & (df.horizon == HORIZONS[HORIZONS.index(H) - 1])].iloc[0]["n_years"]:
                print("  %-6d %-7d %-7d %-9d %-10.2f %-9d %.1f%s"
                      % (sy, H, r["n_years"], r["labor_years"], r["floor_M"],
                         r["floor_yr"], r["terminal_M"], tag))

    # ---- summary per horizon: IS vs holdout ----
    print("\n" + "=" * 100)
    print("SUMMARY per horizon  (IS = 1975-2005 ; HOLDOUT = 2006-2020)")
    print("=" * 100)
    print("  %-7s | %-26s | %-26s"
          % ("horizon", "IS 1975-2005", "HOLDOUT 2006-2020"))
    print("  %-7s | %-26s | %-26s"
          % ("", "labor / starts_fail / minFloor", "labor / starts_fail / minFloor"))
    sum_rows = []
    for H in HORIZONS:
        out = {}
        for grp in ("IS_1975_2005", "HOLDOUT_2006_2020"):
            sub = df[(df.group == grp) & (df.horizon == H)]
            labor = int(sub["labor_years"].sum())
            nfail = int((sub["labor_years"] > 0).sum())
            minfloor = float(sub["floor_M"].min())
            out[grp] = (labor, nfail, minfloor, len(sub))
        il, ifa, ifl, instarts = out["IS_1975_2005"]
        hl, hfa, hfl, hstarts = out["HOLDOUT_2006_2020"]
        print("  %-7d | %2d / %2d of %2d / %7.2fM     | %2d / %2d of %2d / %7.2fM"
              % (H, il, ifa, instarts, ifl, hl, hfa, hstarts, hfl))
        sum_rows.append(dict(horizon=H,
                             IS_labor=il, IS_starts_fail=ifa, IS_n=instarts, IS_minFloor_M=ifl,
                             HOLDOUT_labor=hl, HOLDOUT_starts_fail=hfa, HOLDOUT_n=hstarts,
                             HOLDOUT_minFloor_M=hfl))
    pd.DataFrame(sum_rows).to_csv(
        os.path.join(AR_DIR, "labor_zero_v2_holdout_summary_20260628.csv"),
        index=False, encoding="utf-8-sig")

    # ---- worst holdout case at the data-max horizon ----
    print("\n--- HOLDOUT worst cases (largest labor, then lowest floor) at each start's data-max horizon ---")
    worst = []
    for sy in HOLDOUT_STARTS:
        # data-max horizon = the largest H whose n_years == 2025-sy+1 (or just min(20, ...))
        maxH = min(20, DATA_END - sy + 1)
        r = sim_one(rets, sleeves, sy, maxH, **V2)
        worst.append(r)
    worst_df = pd.DataFrame(worst).sort_values(["labor_years", "floor_M"], ascending=[False, True])
    for _, r in worst_df.head(6).iterrows():
        print("  start %d (%d yrs to 2025): labor=%d floor=%.2fM@%d term=%.1fM"
              % (r["start"], r["n_years"], r["labor_years"], r["floor_M"],
                 r["floor_yr"], r["terminal_M"]))

    block = {
        "script": "labor_zero_v2_holdout_20260628.py", "date": "2026-06-28",
        "v2": {"strat": "sc2.6", "run0_M": 14, "reserve0_M": 26, "rmode": "bond",
               "thr_M": 14, "spend_M": 7.2},
        "is20_sanity_labor": is20_labor, "is20_sanity_floor_M": is20_floor,
        "summary": sum_rows,
        "holdout_any_labor": int(worst_df["labor_years"].sum()),
        "holdout_min_floor_M_datamax": float(worst_df["floor_M"].min()),
        "holdout_worst_start": int(worst_df.iloc[0]["start"]),
    }
    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
