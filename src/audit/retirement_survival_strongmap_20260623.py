"""
src/audit/retirement_survival_strongmap_20260623.py
===================================================
Retirement-survival simulation on the P09_STR (STRONG-map) leverage dial.

REPLACES src/audit/retirement_survival_20260623.py (which used the inferior
DEFAULT-map dial). Drives the simulation directly from the strong-map after-tax
calendar-year returns
  audit_results/p09_strongmap_scale_dial_annual_20260623.csv
(already FRONTIER-sanity-validated by p09_strongmap_scale_dial_20260623.py).

QUESTION (user): init 30,000,000 JPY; spend 5,000,000 JPY/yr; for each start
year 1975..2005 run 20 yr. Survive (never hit zero)? Balance at yr10 / yr20?

MODELING (faithful to request):
  - Withdraw 5,000,000 JPY at START of each year, then apply that year's
    after-tax return to the remaining balance (withdraw-first, conservative).
  - Fixed nominal 5,000,000 JPY/yr (no inflation indexing).
  - Ruin = balance <= 0 (floored at 0 thereafter).
  - 31 start years x 8 strategies x 20-year horizon.

Strategies: P09_STR scale 1.0 / 1.4 / 1.6 / 1.8 / 2.0 / 2.2 / 2.4 + NASDAQ 1x B&H.
ASCII-only prints. Does NOT commit, no temp files.

Outputs:
  audit_results/retirement_survival_strongmap_grid_20260623.csv
  audit_results/retirement_survival_strongmap_paths_20260623.csv
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
ANNUAL_SPEND = 5_000_000.0
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
    return {"survived": survived, "ruin_year_elapsed": ruin_elapsed,
            "asset_y10": asset_y10, "asset_y20": path[-1], "path": path}


def main():
    print("=" * 110)
    print("RETIREMENT SURVIVAL SIMULATION (STRONG-MAP)  2026-06-23")
    print("init=%.0f JPY  spend=%.0f JPY/yr  horizon=%d yr  starts=%d..%d"
          % (INIT_ASSET, ANNUAL_SPEND, HORIZON, START_YEARS[0], START_YEARS[-1]))
    print("Source: %s" % os.path.basename(ANNUAL_CSV))
    print("=" * 110)

    df = pd.read_csv(ANNUAL_CSV)
    df = df.set_index("year")

    # after-tax fractions per scale
    cy = {}
    for sc in SCALES_ALL:
        col = "sc%.1f_strong_aftertax_pct" % sc
        s = df[col].dropna() / 100.0
        s = s[(s.index >= 1975) & (s.index <= 2025)]
        cy[sc] = s
    ndx = df["NASDAQ_1x_BH_aftertax_pct"].dropna() / 100.0
    ndx = ndx[(ndx.index >= 1975) & (ndx.index <= 2025)]

    # ---- SANITY: OUT years scale-invariant + 2008 == +20.34 ----
    print("\n--- SANITY: OUT-year scale invariance ---")
    ok = True
    for yr in (2008, 2001, 2002, 2022):
        vals = [cy[sc].loc[yr] * 100 for sc in SCALES_ALL]
        spread = max(vals) - min(vals)
        if spread >= 0.05:
            ok = False
        print("  %d: spread=%.4fpp -> %s" % (yr, spread, "OK" if spread < 0.05 else "FAIL"))
    v2008 = cy[1.0].loc[2008] * 100
    print("  sc1.0 2008 = %+.4f%% (expect ~+20.34) -> %s"
          % (v2008, "OK" if abs(v2008 - 20.34) < 0.1 else "FAIL"))
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
            need = [sy + k for k in range(HORIZON)]
            if not all(y in ser.index for y in need):
                raise RuntimeError("%s missing years for start %d" % (s, sy))
            res = simulate(ser, sy)
            results[s][sy] = res
            grid_rows.append({
                "start_year": sy, "strategy": s,
                "survived": int(res["survived"]),
                "ruin_year_elapsed": res["ruin_year_elapsed"] if res["ruin_year_elapsed"] else "",
                "asset_y10_yen": round(res["asset_y10"]),
                "asset_y20_yen": round(res["asset_y20"]),
            })
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(out_dir, "retirement_survival_strongmap_grid_20260623.csv"),
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
        os.path.join(out_dir, "retirement_survival_strongmap_paths_20260623.csv"),
        index=False, encoding="utf-8-sig")
    print("Saved paths CSV.")

    # ---- summary ----
    print("\n" + "=" * 110)
    print("SURVIVAL SUMMARY (of %d start years)" % len(START_YEARS))
    print("%-14s | %8s | %14s | %14s | %14s | %s"
          % ("strategy", "survived", "y20_med(surv)", "y20_min(surv)", "y20_max(surv)", "ruined"))
    summary = {}
    for s in strat_order:
        survs = [results[s][sy]["survived"] for sy in START_YEARS]
        y20s = [results[s][sy]["asset_y20"] for sy in START_YEARS if results[s][sy]["survived"]]
        ruined = [sy for sy in START_YEARS if not results[s][sy]["survived"]]
        med = float(np.median(y20s)) if y20s else 0.0
        lo = float(np.min(y20s)) if y20s else 0.0
        hi = float(np.max(y20s)) if y20s else 0.0
        summary[s] = {"survived": int(np.sum(survs)), "y20_median": med,
                      "y20_min": lo, "y20_max": hi, "ruined_starts": ruined,
                      "ruin_years_elapsed": {sy: results[s][sy]["ruin_year_elapsed"] for sy in ruined}}
        print("%-14s | %4d/%-3d | %14.0f | %14.0f | %14.0f | %s"
              % (s, int(np.sum(survs)), len(START_YEARS), med, lo, hi,
                 ",".join(str(x) for x in ruined) if ruined else "(none)"))

    block = {
        "script": "retirement_survival_strongmap_20260623.py", "date": "2026-06-23",
        "source_annual_csv": os.path.basename(ANNUAL_CSV),
        "init_asset": INIT_ASSET, "annual_spend": ANNUAL_SPEND, "horizon": HORIZON,
        "sanity_pass": bool(ok),
        "summary": {s: {
            "survived": summary[s]["survived"], "n_starts": len(START_YEARS),
            "y20_median": round(summary[s]["y20_median"]),
            "y20_min": round(summary[s]["y20_min"]),
            "y20_max": round(summary[s]["y20_max"]),
            "ruined_starts": summary[s]["ruined_starts"],
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
