"""
src/audit/retirement_survival_strongmap_6M_backfill_20260624.py
==============================================================
Retirement-survival simulation on the P09_STR (STRONG-map) leverage dial,
ANNUAL SPEND = 6,000,000 JPY, WITH A BACKFILL RULE:

  BACKFILL (user rule a/b/c):
    (a) If the asset balance is below 20,000,000 JPY, that year's 6,000,000 JPY
        spending is set to ZERO (the shortfall is earned externally).
    (b) Interpretation: in bad markets the retiree earns 6M JPY outside and
        offsets the year's living cost, so the portfolio is NOT drawn down.
    (c) As long as the balance is still below 20M at the start of a year, that
        year's spending is again zero -- repeated every year while below 20M.

  Decision rule (faithful to a/b/c):
    - At the START of each year, look at the balance BEFORE withdrawing.
    - If balance < 20,000,000 -> spend 0 this year (backfill year), count it.
    - Else -> withdraw 6,000,000 as usual.
    - Then apply that year's market return to the remaining balance.

  Backfilled years are counted per (start_year, strategy). Ruin = balance <= 0.
  (With backfill, ruin can only occur after a year where balance >= 20M, then
  a -6M withdrawal plus a bad return pushes it <= 0.)

Base case is identical to retirement_survival_strongmap_6M_20260624.py except
for the backfill rule. init 30M, 31 starts (1975-2005), 20yr, strong-map
after-tax returns. Half-asset threshold = 15M (initial half, unchanged).

ASCII-only prints. Does NOT commit, no temp files.
Outputs:
  audit_results/retirement_survival_strongmap_6M_backfill_grid_20260624.csv
  audit_results/retirement_survival_strongmap_6M_backfill_paths_20260624.csv
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)

ANNUAL_CSV = os.path.join(_REPO_DIR, "audit_results",
                          "p09_strongmap_scale_dial_annual_20260623.csv")

INIT_ASSET = 30_000_000.0
ANNUAL_SPEND = 6_000_000.0
BACKFILL_THRESHOLD = 20_000_000.0   # balance < 20M at year start -> spend 0
HALF_ASSET = 15_000_000.0           # half of INITIAL (1500万) -- unchanged
HORIZON = 20
START_YEARS = list(range(1975, 2006))
SCALES_ALL = [1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
REP_STARTS = [1975, 1980, 1985, 1990, 1995, 2000, 2005]


def simulate(cy_ret_series, start_year, init=INIT_ASSET,
             spend=ANNUAL_SPEND, horizon=HORIZON,
             backfill_threshold=BACKFILL_THRESHOLD):
    """Withdraw-first with backfill: if balance (before withdrawal) < threshold,
    that year's spend is 0 (external income offsets it). Counts backfill years.
    """
    bal = float(init)
    path = []
    ruin_elapsed = None
    asset_y10 = None
    backfill_years = []          # 1-based elapsed years where spend was skipped
    for k in range(horizon):
        yr = start_year + k
        # ---- backfill decision at year start (balance before withdrawal) ----
        if bal < backfill_threshold:
            this_spend = 0.0
            backfill_years.append(k + 1)
        else:
            this_spend = spend
        bal -= this_spend
        if bal <= 0:
            if ruin_elapsed is None:
                ruin_elapsed = k + 1
            bal = 0.0
            path.append(bal)
            if k + 1 == 10:
                asset_y10 = 0.0
            continue
        r = float(cy_ret_series.loc[yr])
        bal *= (1.0 + r)
        if bal <= 0:
            if ruin_elapsed is None:
                ruin_elapsed = k + 1
            bal = 0.0
        path.append(bal)
        if k + 1 == 10:
            asset_y10 = bal
    survived = (ruin_elapsed is None)
    if asset_y10 is None:
        asset_y10 = bal
    below_half_years = [k + 1 for k, b in enumerate(path) if b < HALF_ASSET]
    return {"survived": survived, "ruin_year_elapsed": ruin_elapsed,
            "asset_y10": asset_y10, "asset_y20": path[-1], "path": path,
            "ever_below_half": len(below_half_years) > 0,
            "first_below_half_elapsed": below_half_years[0] if below_half_years else None,
            "backfill_count": len(backfill_years),
            "backfill_years": backfill_years,
            "ever_backfilled": len(backfill_years) > 0}


def main():
    print("=" * 110)
    print("RETIREMENT SURVIVAL (STRONG-MAP, SPEND 6M, BACKFILL<20M)  2026-06-24")
    print("init=%.0f  spend=%.0f/yr  backfill if balance<%.0f -> spend 0 (external income)"
          % (INIT_ASSET, ANNUAL_SPEND, BACKFILL_THRESHOLD))
    print("Source: %s" % os.path.basename(ANNUAL_CSV))
    print("=" * 110)

    df = pd.read_csv(ANNUAL_CSV).set_index("year")
    cy = {}
    for sc in SCALES_ALL:
        s = df["sc%.1f_strong_aftertax_pct" % sc].dropna() / 100.0
        cy[sc] = s[(s.index >= 1975) & (s.index <= 2025)]
    ndx = df["NASDAQ_1x_BH_aftertax_pct"].dropna() / 100.0
    ndx = ndx[(ndx.index >= 1975) & (ndx.index <= 2025)]

    # ---- SANITY ----
    print("\n--- SANITY: OUT-year scale invariance ---")
    ok = True
    for yr in (2008, 2001, 2002, 2022):
        vals = [cy[sc].loc[yr] * 100 for sc in SCALES_ALL]
        if max(vals) - min(vals) >= 0.05:
            ok = False
        print("  %d: spread=%.4fpp" % (yr, max(vals) - min(vals)))
    if abs(cy[1.0].loc[2008] * 100 - 20.34) >= 0.1:
        ok = False
    print("  SANITY: %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        print("HALT")
        sys.exit(1)

    strat_series = {("P09_STR_sc%.1f" % sc): cy[sc] for sc in SCALES_ALL}
    strat_series["NASDAQ_1x_BH"] = ndx
    strat_order = ["P09_STR_sc1.0", "P09_STR_sc1.4", "P09_STR_sc1.6",
                   "P09_STR_sc1.8", "P09_STR_sc2.0", "P09_STR_sc2.2",
                   "P09_STR_sc2.4", "NASDAQ_1x_BH"]

    grid_rows = []
    results = {s: {} for s in strat_order}
    for s in strat_order:
        ser = strat_series[s]
        for sy in START_YEARS:
            if not all(sy + k in ser.index for k in range(HORIZON)):
                raise RuntimeError("%s missing years for %d" % (s, sy))
            res = simulate(ser, sy)
            results[s][sy] = res
            grid_rows.append({
                "start_year": sy, "strategy": s,
                "survived": int(res["survived"]),
                "ruin_year_elapsed": res["ruin_year_elapsed"] if res["ruin_year_elapsed"] else "",
                "asset_y10_yen": round(res["asset_y10"]),
                "asset_y20_yen": round(res["asset_y20"]),
                "ever_below_15M": int(res["ever_below_half"]),
                "first_below_15M_elapsed": res["first_below_half_elapsed"] if res["first_below_half_elapsed"] else "",
                "backfill_count": res["backfill_count"],
                "backfill_years": ";".join(str(x) for x in res["backfill_years"]),
            })
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(out_dir, "retirement_survival_strongmap_6M_backfill_grid_20260624.csv"),
        index=False, encoding="utf-8-sig")
    print("\nSaved grid CSV.")

    path_rows = []
    for sy in REP_STARTS:
        for s in strat_order:
            for k, bal in enumerate(results[s][sy]["path"]):
                path_rows.append({"start_year": sy, "strategy": s,
                                  "elapsed_year": k + 1, "calendar_year": sy + k,
                                  "asset_yen": round(bal),
                                  "backfilled": int((k + 1) in results[s][sy]["backfill_years"])})
    pd.DataFrame(path_rows).to_csv(
        os.path.join(out_dir, "retirement_survival_strongmap_6M_backfill_paths_20260624.csv"),
        index=False, encoding="utf-8-sig")
    print("Saved paths CSV.")

    # ---- summary ----
    print("\n" + "=" * 110)
    print("SUMMARY (spend 6M + backfill<20M, of %d starts)" % len(START_YEARS))
    print("%-14s | %8s | %10s | %11s | %14s | %s"
          % ("strategy", "survived", "below15M", "backfill_yrs", "y20_med(surv)", "ruined"))
    summary = {}
    for s in strat_order:
        survs = [results[s][sy]["survived"] for sy in START_YEARS]
        y20s = [results[s][sy]["asset_y20"] for sy in START_YEARS if results[s][sy]["survived"]]
        ruined = [sy for sy in START_YEARS if not results[s][sy]["survived"]]
        below = [sy for sy in START_YEARS if results[s][sy]["ever_below_half"]]
        bf_counts = [results[s][sy]["backfill_count"] for sy in START_YEARS]
        bf_total = int(np.sum(bf_counts))
        bf_starts = [sy for sy in START_YEARS if results[s][sy]["ever_backfilled"]]
        bf_max = int(np.max(bf_counts)) if bf_counts else 0
        med = float(np.median(y20s)) if y20s else 0.0
        lo = float(np.min(y20s)) if y20s else 0.0
        hi = float(np.max(y20s)) if y20s else 0.0
        summary[s] = {"survived": int(np.sum(survs)), "y20_median": med,
                      "y20_min": lo, "y20_max": hi, "ruined_starts": ruined,
                      "below_15M_count": len(below), "below_15M_starts": below,
                      "backfill_total_years": bf_total,
                      "backfill_starts_count": len(bf_starts),
                      "backfill_starts": bf_starts,
                      "backfill_max_in_one_start": bf_max,
                      "ruin_years_elapsed": {sy: results[s][sy]["ruin_year_elapsed"] for sy in ruined}}
        print("%-14s | %4d/%-3d | %2d/31 | tot %3d (max %2d) | %14.0f | %s"
              % (s, int(np.sum(survs)), len(START_YEARS), len(below),
                 bf_total, bf_max, med,
                 ",".join(str(x) for x in ruined) if ruined else "(none)"))

    block = {
        "script": "retirement_survival_strongmap_6M_backfill_20260624.py", "date": "2026-06-24",
        "source_annual_csv": os.path.basename(ANNUAL_CSV),
        "init_asset": INIT_ASSET, "annual_spend": ANNUAL_SPEND,
        "backfill_threshold": BACKFILL_THRESHOLD, "half_threshold": HALF_ASSET,
        "horizon": HORIZON, "sanity_pass": bool(ok),
        "summary": {s: {
            "survived": summary[s]["survived"], "n_starts": len(START_YEARS),
            "y20_median": round(summary[s]["y20_median"]),
            "y20_min": round(summary[s]["y20_min"]),
            "y20_max": round(summary[s]["y20_max"]),
            "ruined_starts": summary[s]["ruined_starts"],
            "below_15M_count": summary[s]["below_15M_count"],
            "below_15M_starts": summary[s]["below_15M_starts"],
            "backfill_total_years": summary[s]["backfill_total_years"],
            "backfill_starts_count": summary[s]["backfill_starts_count"],
            "backfill_starts": summary[s]["backfill_starts"],
            "backfill_max_in_one_start": summary[s]["backfill_max_in_one_start"],
            "ruin_years_elapsed": summary[s]["ruin_years_elapsed"],
        } for s in strat_order},
        "grid": grid_rows,
    }
    print("\n" + "=" * 110)
    print("RETURN_BLOCK")
    print("=" * 110)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
