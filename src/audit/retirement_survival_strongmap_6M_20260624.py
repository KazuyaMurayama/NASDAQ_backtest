"""
src/audit/retirement_survival_strongmap_6M_20260624.py
======================================================
Retirement-survival simulation on the P09_STR (STRONG-map) leverage dial,
with ANNUAL SPEND = 6,000,000 JPY (vs the 5,000,000 base case in
retirement_survival_strongmap_20260623.py). Same engine, same strong-map
after-tax calendar-year returns, same 31 start years x 8 strategies x 20yr.

  init = 30,000,000 JPY ; spend = 6,000,000 JPY/yr (initial withdrawal rate 20.0%)
  half-asset threshold = 15,000,000 JPY (= half of INITIAL, unchanged).

Withdraw-first (conservative). Ruin = balance <= 0 (floored at 0). After-tax
returns from audit_results/p09_strongmap_scale_dial_annual_20260623.csv (already
FRONTIER-sanity-validated). ASCII-only prints. Does NOT commit, no temp files.

Outputs:
  audit_results/retirement_survival_strongmap_6M_grid_20260624.csv
  audit_results/retirement_survival_strongmap_6M_paths_20260624.csv
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
ANNUAL_SPEND = 6_000_000.0   # 600万/年 (this version)
HALF_ASSET = 15_000_000.0    # half of INITIAL (1500万) -- unchanged
HORIZON = 20
START_YEARS = list(range(1975, 2006))
SCALES_ALL = [1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
REP_STARTS = [1975, 1980, 1985, 1990, 1995, 2000, 2005]


def simulate(cy_ret_series, start_year, init=INIT_ASSET,
             spend=ANNUAL_SPEND, horizon=HORIZON):
    bal = float(init)
    path = []
    ruin_elapsed = None
    asset_y10 = None
    for k in range(horizon):
        yr = start_year + k
        bal -= spend
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
    ever_below_half = len(below_half_years) > 0
    first_below_half = below_half_years[0] if below_half_years else None
    return {"survived": survived, "ruin_year_elapsed": ruin_elapsed,
            "asset_y10": asset_y10, "asset_y20": path[-1], "path": path,
            "ever_below_half": ever_below_half,
            "first_below_half_elapsed": first_below_half}


def main():
    print("=" * 110)
    print("RETIREMENT SURVIVAL SIMULATION (STRONG-MAP, SPEND 6M)  2026-06-24")
    print("init=%.0f JPY  spend=%.0f JPY/yr (rate %.1f%%)  horizon=%d yr  starts=%d..%d"
          % (INIT_ASSET, ANNUAL_SPEND, 100 * ANNUAL_SPEND / INIT_ASSET,
             HORIZON, START_YEARS[0], START_YEARS[-1]))
    print("Source: %s" % os.path.basename(ANNUAL_CSV))
    print("=" * 110)

    df = pd.read_csv(ANNUAL_CSV).set_index("year")

    cy = {}
    for sc in SCALES_ALL:
        col = "sc%.1f_strong_aftertax_pct" % sc
        s = df[col].dropna() / 100.0
        cy[sc] = s[(s.index >= 1975) & (s.index <= 2025)]
    ndx = df["NASDAQ_1x_BH_aftertax_pct"].dropna() / 100.0
    ndx = ndx[(ndx.index >= 1975) & (ndx.index <= 2025)]

    # ---- SANITY: OUT years scale-invariant + 2008 == +20.34 ----
    print("\n--- SANITY: OUT-year scale invariance ---")
    ok = True
    for yr in (2008, 2001, 2002, 2022):
        vals = [cy[sc].loc[yr] * 100 for sc in SCALES_ALL]
        if max(vals) - min(vals) >= 0.05:
            ok = False
        print("  %d: spread=%.4fpp" % (yr, max(vals) - min(vals)))
    v2008 = cy[1.0].loc[2008] * 100
    print("  sc1.0 2008 = %+.4f%% (expect ~+20.34)" % v2008)
    if abs(v2008 - 20.34) >= 0.1:
        ok = False
    print("  SANITY: %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        print("HALT: sanity failed.")
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
                raise RuntimeError("%s missing years for start %d" % (s, sy))
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
            })
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(out_dir, "retirement_survival_strongmap_6M_grid_20260624.csv"),
        index=False, encoding="utf-8-sig")
    print("\nSaved grid CSV.")

    path_rows = []
    for sy in REP_STARTS:
        for s in strat_order:
            for k, bal in enumerate(results[s][sy]["path"]):
                path_rows.append({"start_year": sy, "strategy": s,
                                  "elapsed_year": k + 1, "calendar_year": sy + k,
                                  "asset_yen": round(bal)})
    pd.DataFrame(path_rows).to_csv(
        os.path.join(out_dir, "retirement_survival_strongmap_6M_paths_20260624.csv"),
        index=False, encoding="utf-8-sig")
    print("Saved paths CSV.")

    # ---- summary ----
    print("\n" + "=" * 110)
    print("SURVIVAL SUMMARY (spend 6M, of %d start years)" % len(START_YEARS))
    print("%-14s | %8s | %12s | %14s | %14s | %14s | %s"
          % ("strategy", "survived", "below15M", "y20_med(surv)", "y20_min(surv)",
             "y20_max(surv)", "ruined"))
    summary = {}
    for s in strat_order:
        survs = [results[s][sy]["survived"] for sy in START_YEARS]
        y20s = [results[s][sy]["asset_y20"] for sy in START_YEARS if results[s][sy]["survived"]]
        ruined = [sy for sy in START_YEARS if not results[s][sy]["survived"]]
        below_half_starts = [sy for sy in START_YEARS if results[s][sy]["ever_below_half"]]
        med = float(np.median(y20s)) if y20s else 0.0
        lo = float(np.min(y20s)) if y20s else 0.0
        hi = float(np.max(y20s)) if y20s else 0.0
        summary[s] = {"survived": int(np.sum(survs)), "y20_median": med,
                      "y20_min": lo, "y20_max": hi, "ruined_starts": ruined,
                      "ruin_years_elapsed": {sy: results[s][sy]["ruin_year_elapsed"] for sy in ruined},
                      "below_15M_count": len(below_half_starts),
                      "below_15M_starts": below_half_starts}
        print("%-14s | %4d/%-3d | below %2d/%-2d | %14.0f | %14.0f | %14.0f | %s"
              % (s, int(np.sum(survs)), len(START_YEARS),
                 len(below_half_starts), len(START_YEARS), med, lo, hi,
                 ",".join(str(x) for x in ruined) if ruined else "(none)"))

    block = {
        "script": "retirement_survival_strongmap_6M_20260624.py", "date": "2026-06-24",
        "source_annual_csv": os.path.basename(ANNUAL_CSV),
        "init_asset": INIT_ASSET, "annual_spend": ANNUAL_SPEND,
        "withdrawal_rate_pct": round(100 * ANNUAL_SPEND / INIT_ASSET, 2),
        "horizon": HORIZON, "half_threshold": HALF_ASSET,
        "sanity_pass": bool(ok),
        "summary": {s: {
            "survived": summary[s]["survived"], "n_starts": len(START_YEARS),
            "y20_median": round(summary[s]["y20_median"]),
            "y20_min": round(summary[s]["y20_min"]),
            "y20_max": round(summary[s]["y20_max"]),
            "ruined_starts": summary[s]["ruined_starts"],
            "ruin_years_elapsed": summary[s]["ruin_years_elapsed"],
            "below_15M_count": summary[s]["below_15M_count"],
            "below_15M_starts": summary[s]["below_15M_starts"],
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
