"""
src/audit/retirement_survival_x4n4_6M_backfill_20260627.py
==========================================================
NEW VERSION of the 6M-spend + backfill retirement-survival report, with the
strategy set CHANGED per user request (2026-06-27):

  ADD     : X4  (scale=3.0,  in_gold_w=0.32, in_bond_w=0.05)
            N4  (scale=2.85, in_gold_w=0.28, in_bond_w=0.07)
  REMOVE  : scale2.2, scale2.4
  KEEP    : P09_STR sc1.0 / 1.4 / 1.6 / 1.8 / 2.0  +  NASDAQ 1x B&H

Everything else is IDENTICAL to retirement_survival_strongmap_6M_backfill_20260624.py:
  init 30,000,000 JPY, spend 6,000,000/yr, BACKFILL if balance < 20,000,000 at
  year-start -> spend 0 that year (earned externally); 31 starts 1975-2005, 20yr
  horizon; calendar-year after-tax (x0.8273) returns; half threshold 15M; ruin = bal<=0.

WHY a new generator (not just edit the CSV): X4/N4 are NOT simple scale dials --
they blend 1x Gold/Bond INTO the leveraged IN leg (LEVER B). Their calendar-year
after-tax returns are built here with the SAME validated builders + the SAME
_calendar_year_returns x AFTER_TAX convention used by the existing scale-dial
annual CSV. We SANITY-CHECK that the harness sc2.0 column reproduces the existing
p09_strongmap_scale_dial_annual_20260623.csv sc2.0 column year-by-year, which
proves X4/N4 use the identical convention before they enter the sim.

ASCII-only prints. Does NOT commit, no temp files.
Outputs:
  audit_results/retirement_survival_x4n4_6M_backfill_annual_20260627.csv  (calendar-year after-tax, new set)
  audit_results/retirement_survival_x4n4_6M_backfill_grid_20260627.csv
  audit_results/retirement_survival_x4n4_6M_backfill_paths_20260627.csv
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
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.audit.dd_reduction_harness_20260626 as H
from src.audit.run_p01_backtest_20260611 import _calendar_year_returns

AFTER_TAX = 0.8273

# Existing validated annual CSV (scale dial, after-tax calendar-year). We reuse its
# sc1.0/1.4/1.6/1.8/2.0 + NASDAQ columns verbatim (already FRONTIER-validated) and
# only ADD X4/N4 computed here. sc2.0 column is also used as the cross-check anchor.
EXISTING_ANNUAL = os.path.join(_REPO_DIR, "audit_results",
                               "p09_strongmap_scale_dial_annual_20260623.csv")

# New strategy set (order = report column order).
KEEP_SCALES = [1.0, 1.4, 1.6, 1.8, 2.0]          # dropped 2.2 / 2.4
ADD_BLENDS = [
    ("N4", dict(scale=2.85, in_gold_w=0.28, in_bond_w=0.07)),
    ("X4", dict(scale=3.0,  in_gold_w=0.32, in_bond_w=0.05)),
]

INIT_ASSET = 30_000_000.0
ANNUAL_SPEND = 6_000_000.0
BACKFILL_THRESHOLD = 20_000_000.0
HALF_ASSET = 15_000_000.0
HORIZON = 20
START_YEARS = list(range(1975, 2006))
REP_STARTS = [1975, 1980, 1985, 1990, 1995, 2000, 2005]

# strategy display order for the new report
STRAT_ORDER = (["P09_STR_sc1.0", "P09_STR_sc1.4", "P09_STR_sc1.6",
                "P09_STR_sc1.8", "P09_STR_sc2.0", "N4", "X4", "NASDAQ_1x_BH"])


def simulate(cy_ret_series, start_year, init=INIT_ASSET,
             spend=ANNUAL_SPEND, horizon=HORIZON,
             backfill_threshold=BACKFILL_THRESHOLD):
    """Withdraw-first with backfill (identical logic to the 20260624 6M version)."""
    bal = float(init)
    path = []
    ruin_elapsed = None
    asset_y10 = None
    backfill_years = []
    for k in range(horizon):
        yr = start_year + k
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
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 110)
    print("RETIREMENT SURVIVAL (NEW SET: +X4 +N4, -sc2.2 -sc2.4)  SPEND 6M + BACKFILL<20M  2026-06-27")
    print("init=%.0f spend=%.0f/yr backfill if bal<%.0f -> spend 0" %
          (INIT_ASSET, ANNUAL_SPEND, BACKFILL_THRESHOLD))
    print("=" * 110)

    # ---- 1. existing validated scale-dial annual (after-tax calendar-year) ----
    ex = pd.read_csv(EXISTING_ANNUAL).set_index("year")

    # ---- 2. build X4/N4 calendar-year after-tax with the SAME convention ----
    print("\n--- building X4 / N4 daily NAV via harness (LEVER B IN-leg blend) ---")
    ctx = H.setup()
    dates_dt = ctx["dates_dt"]
    blend_cy = {}
    blend_meta = {}
    for name, kw in ADD_BLENDS:
        nav_dt, r, tpy, exc = H.build(ctx, **kw)
        cy = _calendar_year_returns(nav_dt)        # same fn the scale-dial CSV used
        cy = cy[cy.index <= 2025]
        blend_cy[name] = cy * AFTER_TAX            # x0.8273, same as scale dial
        blend_meta[name] = dict(tpy=float(tpy), kw=kw)
        print("  %-3s scale=%.2f gold=%.2f bond=%.2f  tpy=%.2f  cy2008(at)=%+.2f%%  cy_worst(at)=%+.2f%%(%d)"
              % (name, kw["scale"], kw["in_gold_w"], kw["in_bond_w"], tpy,
                 float(blend_cy[name].get(2008, np.nan)) * 100,
                 float(blend_cy[name].min()) * 100, int(blend_cy[name].idxmin())))

    # ---- 3. SANITY: harness sc2.0 reproduces existing CSV sc2.0 column ----
    print("\n--- SANITY: harness sc2.0 calendar-year vs existing scale-dial CSV (must match) ---")
    nav20, r20, tpy20, _ = H.build(ctx, scale=2.0)
    cy20 = _calendar_year_returns(nav20)
    cy20 = (cy20[cy20.index <= 2025] * AFTER_TAX) * 100.0
    max_abs = 0.0
    worst_y = None
    for y in range(1975, 2026):
        if y in cy20.index and y in ex.index:
            a = float(cy20.loc[y])
            b = float(ex.loc[y, "sc2.0_strong_aftertax_pct"])
            d = abs(a - b)
            if d > max_abs:
                max_abs, worst_y = d, y
    print("  max |harness_sc2.0 - csv_sc2.0| over 1975-2025 = %.4fpp (year %s)" % (max_abs, worst_y))
    sane = max_abs <= 0.02   # 0.02pp rounding (csv stored at 2 decimals)
    print("  SANITY: %s" % ("PASS (X4/N4 use the identical convention)" if sane else "FAIL"))
    if not sane:
        print("HALT -- convention mismatch; do not trust X4/N4 series.")
        sys.exit(1)

    # ---- 4. assemble the NEW annual after-tax matrix (fractions) ----
    cy = {}
    for sc in KEEP_SCALES:
        s = ex["sc%.1f_strong_aftertax_pct" % sc].dropna() / 100.0
        cy["P09_STR_sc%.1f" % sc] = s[(s.index >= 1975) & (s.index <= 2025)]
    for name, _ in ADD_BLENDS:
        cy[name] = blend_cy[name][(blend_cy[name].index >= 1975) & (blend_cy[name].index <= 2025)]
    ndx = ex["NASDAQ_1x_BH_aftertax_pct"].dropna() / 100.0
    cy["NASDAQ_1x_BH"] = ndx[(ndx.index >= 1975) & (ndx.index <= 2025)]

    # write the new annual CSV (percent, 2 decimals, matching the existing format)
    years = list(range(1974, 2026))
    ar_rows = []
    for y in years:
        row = {"year": y}
        for sc in KEEP_SCALES:
            ser = ex["sc%.1f_strong_aftertax_pct" % sc]
            row["P09_STR_sc%.1f_aftertax_pct" % sc] = (
                round(float(ser.loc[y]), 2) if y in ser.index and ser.loc[y] == ser.loc[y] else np.nan)
        for name, _ in ADD_BLENDS:
            v = blend_cy[name].get(y, np.nan)
            row["%s_aftertax_pct" % name] = round(float(v) * 100, 2) if v == v else np.nan
        nv = ex["NASDAQ_1x_BH_aftertax_pct"]
        row["NASDAQ_1x_BH_aftertax_pct"] = (
            round(float(nv.loc[y]), 2) if y in nv.index and nv.loc[y] == nv.loc[y] else np.nan)
        ar_rows.append(row)
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    annual_path = os.path.join(out_dir, "retirement_survival_x4n4_6M_backfill_annual_20260627.csv")
    pd.DataFrame(ar_rows).to_csv(annual_path, index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\nSaved new annual CSV: %s" % annual_path)

    # ---- 5. run backfill sim on the NEW set ----
    grid_rows = []
    results = {s: {} for s in STRAT_ORDER}
    for s in STRAT_ORDER:
        ser = cy[s]
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
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(out_dir, "retirement_survival_x4n4_6M_backfill_grid_20260627.csv"),
        index=False, encoding="utf-8-sig")
    print("Saved grid CSV.")

    path_rows = []
    for sy in REP_STARTS:
        for s in STRAT_ORDER:
            for k, bal in enumerate(results[s][sy]["path"]):
                path_rows.append({"start_year": sy, "strategy": s,
                                  "elapsed_year": k + 1, "calendar_year": sy + k,
                                  "asset_yen": round(bal),
                                  "backfilled": int((k + 1) in results[s][sy]["backfill_years"])})
    pd.DataFrame(path_rows).to_csv(
        os.path.join(out_dir, "retirement_survival_x4n4_6M_backfill_paths_20260627.csv"),
        index=False, encoding="utf-8-sig")
    print("Saved paths CSV.")

    # ---- 6. summary ----
    print("\n" + "=" * 110)
    print("SUMMARY (spend 6M + backfill<20M, NEW set, of %d starts)" % len(START_YEARS))
    print("%-14s | %8s | %9s | %13s | %14s | %s"
          % ("strategy", "survived", "below15M", "backfill_yrs", "y20_med(surv)", "ruined"))
    summary = {}
    for s in STRAT_ORDER:
        survs = [results[s][sy]["survived"] for sy in START_YEARS]
        y20s = [results[s][sy]["asset_y20"] for sy in START_YEARS if results[s][sy]["survived"]]
        ruined = [sy for sy in START_YEARS if not results[s][sy]["survived"]]
        below = [sy for sy in START_YEARS if results[s][sy]["ever_below_half"]]
        bf_counts = [results[s][sy]["backfill_count"] for sy in START_YEARS]
        bf_total = int(np.sum(bf_counts))
        bf_starts = [sy for sy in START_YEARS if results[s][sy]["ever_backfilled"]]
        bf_max = int(np.max(bf_counts)) if bf_counts else 0
        med = float(np.median(y20s)) if y20s else 0.0
        summary[s] = {"survived": int(np.sum(survs)), "y20_median": med,
                      "ruined_starts": ruined, "below_15M_count": len(below),
                      "below_15M_starts": below, "backfill_total_years": bf_total,
                      "backfill_starts_count": len(bf_starts), "backfill_starts": bf_starts,
                      "backfill_max_in_one_start": bf_max,
                      "ruin_years_elapsed": {sy: results[s][sy]["ruin_year_elapsed"] for sy in ruined}}
        print("%-14s | %4d/%-3d | %3d/31 | tot %3d (max %2d) | %14.0f | %s"
              % (s, int(np.sum(survs)), len(START_YEARS), len(below),
                 bf_total, bf_max, med,
                 ",".join(str(x) for x in ruined) if ruined else "(none)"))

    block = {
        "script": "retirement_survival_x4n4_6M_backfill_20260627.py", "date": "2026-06-27",
        "source_existing_annual": os.path.basename(EXISTING_ANNUAL),
        "new_annual_csv": os.path.basename(annual_path),
        "blend_meta": blend_meta, "sanity_sc2.0_maxabs_pp": round(max_abs, 4),
        "init_asset": INIT_ASSET, "annual_spend": ANNUAL_SPEND,
        "backfill_threshold": BACKFILL_THRESHOLD, "half_threshold": HALF_ASSET,
        "horizon": HORIZON, "strat_order": STRAT_ORDER,
        "summary": {s: {
            "survived": summary[s]["survived"], "n_starts": len(START_YEARS),
            "y20_median": round(summary[s]["y20_median"]),
            "ruined_starts": summary[s]["ruined_starts"],
            "below_15M_count": summary[s]["below_15M_count"],
            "below_15M_starts": summary[s]["below_15M_starts"],
            "backfill_total_years": summary[s]["backfill_total_years"],
            "backfill_starts_count": summary[s]["backfill_starts_count"],
            "backfill_starts": summary[s]["backfill_starts"],
            "backfill_max_in_one_start": summary[s]["backfill_max_in_one_start"],
            "ruin_years_elapsed": summary[s]["ruin_years_elapsed"],
        } for s in STRAT_ORDER},
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
