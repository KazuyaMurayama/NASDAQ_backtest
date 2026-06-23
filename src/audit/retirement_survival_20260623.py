"""
src/audit/retirement_survival_20260623.py
==========================================
Retirement-survival simulation on the P09_C1 leverage-scale dial.

QUESTION (user)
  Initial assets 30,000,000 JPY; fixed spending 5,000,000 JPY/year.
  For each start year 1975..2005, begin "living off the portfolio" at the
  start of that year and run 20 years. Does the portfolio survive (never hit
  zero)? What is the balance at year 10 and year 20?

STRATEGIES
  P09_C1 scale dial: 1.0 (base) / 1.4 / 1.6 / 1.8 / 2.0 / 2.2 / 2.4
  + NASDAQ 1x B&H (reference). All after-tax (x0.8273) calendar-year returns.

MODELING (faithful to the request; see plan
  docs/superpowers/plans/2026-06-23-withdrawal-survival-sim.md)
  - Withdraw 5,000,000 JPY at the START of each year, then apply that year's
    after-tax return to the remaining balance (withdraw-first convention).
  - Fixed nominal 5,000,000 JPY/yr (no inflation indexing in base case).
  - Ruin = balance <= 0 at any point; balance floored at 0 thereafter.
  - 31 start years x 8 strategies x 20-year horizon.

ALSO emits the P09_C1(scale1.0) after-tax calendar-year column that was missing
from audit_results/p09_scale_dial_annual_20260623.csv (so it can be added to the
dial report's annual table).

Reuses validated builders ONLY (no reimplementation). ASCII-only prints.
Does NOT commit, no temp files.

Outputs:
  audit_results/p09_scale10_annual_20260623.csv          (sc1.0 after-tax annual)
  audit_results/retirement_survival_grid_20260623.csv    (start_year x strategy)
  audit_results/retirement_survival_paths_20260623.csv   (representative paths)
"""

from __future__ import annotations

import json
import os
import sys
import types

if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import src.audit.strategy_runners as sr
from src.audit.unified_metrics import IS_END, OOS_START  # noqa: F401 (kept for parity)

from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base
from src.audit.leverup_b1c1_20260612 import _build_p09_on_base_c1
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)

AFTER_TAX = 0.8273
NASDAQ_CSV_PATH = os.path.join(_REPO_DIR, "NASDAQ_extended_to_2026.csv")

# ---- survival simulation constants ----
INIT_ASSET = 30_000_000.0    # 3000 万円
ANNUAL_SPEND = 5_000_000.0   # 500 万円/年
HORIZON = 20                 # 20 年
START_YEARS = list(range(1975, 2006))   # 1975 .. 2005 (31 starts)
SCALES_ALL = [1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
REP_STARTS = [1975, 1980, 1985, 1990, 1995, 2000, 2005]


# ---------------------------------------------------------------------------
# Inputs (mirror p09_scale_dial_20260623.py main() L122-148; reuse only)
# ---------------------------------------------------------------------------
def build_inputs():
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    return dict(shared=shared, dates_dt=dates_dt, n_years=n_years,
                ret_gold=ret_gold, ret_bond=ret_bond, fund_active=fund_active,
                wg=wg, wb=wb, bond_on=bond_on, sofr_arr=sofr_arr)


def _build_p09_scaled(inp, lev_scale, cfd_excess):
    """Genuine P09_C1 daily NAV at lev_scale (V7 default map x scale + C1 fill)."""
    base_nav, r_base, tpy_base, exc = _build_tqqq_base(
        inp["shared"], inp["dates_dt"], v7_map=None,
        lev_scale=lev_scale, cfd_excess=cfd_excess)
    nav_dt, r_p09c1, tpy = _build_p09_on_base_c1(
        r_base, inp["ret_gold"], inp["ret_bond"], inp["fund_active"],
        inp["wg"], inp["wb"], inp["bond_on"], inp["sofr_arr"],
        inp["dates_dt"], tpy_base, inp["n_years"])
    return nav_dt


def _load_nasdaq_bh_annual():
    df = pd.read_csv(NASDAQ_CSV_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    close = df["Close"].dropna()
    yearly = close.resample("YE").last()
    ann = yearly.pct_change().dropna()
    ann = ann[ann.index.year <= 2025]
    return pd.Series(ann.values, index=ann.index.year)


# ---------------------------------------------------------------------------
# Survival simulation (withdraw-first)
# ---------------------------------------------------------------------------
def simulate(cy_ret_series, start_year, init=INIT_ASSET,
             spend=ANNUAL_SPEND, horizon=HORIZON):
    """Year-start withdrawal then apply that year's after-tax return.

    Ruin = balance <= 0 (floored at 0). ruin_year_elapsed = 1-based elapsed
    year at which ruin first occurs (None if survived).
    """
    bal = float(init)
    path = []
    ruin_elapsed = None
    asset_y10 = None
    for k in range(horizon):            # k = 0..19 (elapsed years from start)
        yr = start_year + k
        bal -= spend                    # withdraw at start of year
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
    asset_y20 = path[-1]
    if asset_y10 is None:
        asset_y10 = bal
    return {"survived": survived, "ruin_year_elapsed": ruin_elapsed,
            "asset_y10": asset_y10, "asset_y20": asset_y20, "path": path}


def main():
    print("=" * 110)
    print("RETIREMENT SURVIVAL SIMULATION  2026-06-23")
    print("init=%.0f JPY  spend=%.0f JPY/yr  horizon=%d yr  starts=%d..%d"
          % (INIT_ASSET, ANNUAL_SPEND, HORIZON, START_YEARS[0], START_YEARS[-1]))
    print("Strategies: P09_C1 scale %s + NASDAQ 1x B&H ; after-tax x%.4f"
          % (SCALES_ALL, AFTER_TAX))
    print("=" * 110)

    inp = build_inputs()

    # ---- after-tax calendar-year returns per scale ----
    cy_aftertax = {}    # scale -> Series(year -> after-tax fraction)
    for sc in SCALES_ALL:
        cfd = (sc > 1.0)
        nav_dt = _build_p09_scaled(inp, lev_scale=sc, cfd_excess=cfd)
        cy = _calendar_year_returns(nav_dt)
        cy = cy[(cy.index >= 1975) & (cy.index <= 2025)]
        cy_aftertax[sc] = cy * AFTER_TAX

    ndx = _load_nasdaq_bh_annual()
    ndx_aftertax = (ndx[(ndx.index >= 1975) & (ndx.index <= 2025)]) * AFTER_TAX

    # ---- SANITY: OUT-year returns are scale-invariant (DH-W1 OUT -> C1 fill) ----
    print("\n--- SANITY: OUT-year scale invariance (2008/2001/2002/2022) ---")
    ok_sanity = True
    for yr in (2008, 2001, 2002, 2022):
        vals = [cy_aftertax[sc].loc[yr] * 100 for sc in SCALES_ALL]
        spread = max(vals) - min(vals)
        flag = "OK" if spread < 0.05 else "FAIL"
        if spread >= 0.05:
            ok_sanity = False
        print("  %d: %s  spread=%.4fpp -> %s"
              % (yr, ["%+.2f%%" % v for v in vals], spread, flag))
    # scale1.0 2008 should match dial CSV sc1.4 2008 (+20.34%)
    s10_2008 = cy_aftertax[1.0].loc[2008] * 100
    print("  scale1.0 2008 = %+.4f%% (expect ~+20.34%%, OUT-year scale-invariant)"
          % s10_2008)
    if abs(s10_2008 - 20.34) > 0.10:
        ok_sanity = False
        print("  WARNING: scale1.0 2008 deviates from expected +20.34%%")
    print("  SANITY: %s" % ("PASS" if ok_sanity else "FAIL"))

    # =====================================================================
    # OUTPUT 1: P09_C1(scale1.0) after-tax annual CSV (missing from dial CSV)
    # =====================================================================
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)

    s10 = cy_aftertax[1.0]
    df_s10 = pd.DataFrame({
        "year": s10.index.astype(int),
        "sc1.0_aftertax_pct": np.round(s10.values * 100, 2),
    })
    p_s10 = os.path.join(out_dir, "p09_scale10_annual_20260623.csv")
    df_s10.to_csv(p_s10, index=False, encoding="utf-8-sig")
    print("\nSaved: %s" % p_s10)

    # ---- annual stats for sc1.0 (for dial report 2.1) ----
    v10 = s10[(s10.index >= 1975) & (s10.index <= 2025)].values * 100
    print("\n--- sc1.0 annual stats (after-tax, 1975-2025, ddof=1) ---")
    print("  mean=%+.2f%%  median=%+.2f%%  std=%.2f%%  max=%+.2f%%  min=%+.2f%%  pos=%d  neg=%d"
          % (np.mean(v10), np.median(v10), np.std(v10, ddof=1),
             np.max(v10), np.min(v10), int(np.sum(v10 > 0)), int(np.sum(v10 < 0))))
    print("  (annual after-tax pct per year:)")
    for y in range(1975, 2026):
        if y in s10.index:
            print("    %d: %+.2f%%" % (y, s10.loc[y] * 100))

    # =====================================================================
    # OUTPUT 2: survival grid (start_year x strategy)
    # =====================================================================
    strat_series = {("P09_C1_sc%.1f" % sc): cy_aftertax[sc] for sc in SCALES_ALL}
    strat_series["NASDAQ_1x_BH"] = ndx_aftertax
    strat_order = ["P09_C1_sc1.0", "P09_C1_sc1.4", "P09_C1_sc1.6", "P09_C1_sc1.8",
                   "P09_C1_sc2.0", "P09_C1_sc2.2", "P09_C1_sc2.4", "NASDAQ_1x_BH"]

    grid_rows = []
    results = {s: {} for s in strat_order}   # strat -> start_year -> sim dict
    for s in strat_order:
        cy = strat_series[s]
        for sy in START_YEARS:
            # ensure required years present
            need = [sy + k for k in range(HORIZON)]
            if not all(y in cy.index for y in need):
                missing = [y for y in need if y not in cy.index]
                raise RuntimeError("%s missing years %s for start %d" % (s, missing, sy))
            res = simulate(cy, sy)
            results[s][sy] = res
            grid_rows.append({
                "start_year": sy, "strategy": s,
                "survived": int(res["survived"]),
                "ruin_year_elapsed": res["ruin_year_elapsed"] if res["ruin_year_elapsed"] else "",
                "asset_y10_yen": round(res["asset_y10"]),
                "asset_y20_yen": round(res["asset_y20"]),
            })
    df_grid = pd.DataFrame(grid_rows)
    p_grid = os.path.join(out_dir, "retirement_survival_grid_20260623.csv")
    df_grid.to_csv(p_grid, index=False, encoding="utf-8-sig")
    print("\nSaved: %s" % p_grid)

    # =====================================================================
    # OUTPUT 3: representative paths
    # =====================================================================
    path_rows = []
    for sy in REP_STARTS:
        for s in strat_order:
            res = results[s][sy]
            for k, bal in enumerate(res["path"]):
                path_rows.append({
                    "start_year": sy, "strategy": s, "elapsed_year": k + 1,
                    "calendar_year": sy + k, "asset_yen": round(bal),
                })
    df_paths = pd.DataFrame(path_rows)
    p_paths = os.path.join(out_dir, "retirement_survival_paths_20260623.csv")
    df_paths.to_csv(p_paths, index=False, encoding="utf-8-sig")
    print("Saved: %s" % p_paths)

    # =====================================================================
    # SUMMARY per strategy
    # =====================================================================
    print("\n" + "=" * 110)
    print("SURVIVAL SUMMARY (of %d start years 1975-2005)" % len(START_YEARS))
    print("%-14s | %8s | %14s | %14s | %14s | %s"
          % ("strategy", "survived", "y20_median", "y20_min", "y20_max", "ruined_starts"))
    print("-" * 130)
    summary = {}
    for s in strat_order:
        survs = [results[s][sy]["survived"] for sy in START_YEARS]
        n_surv = int(np.sum(survs))
        y20_surv = [results[s][sy]["asset_y20"] for sy in START_YEARS
                    if results[s][sy]["survived"]]
        ruined = [sy for sy in START_YEARS if not results[s][sy]["survived"]]
        med = float(np.median(y20_surv)) if y20_surv else 0.0
        lo = float(np.min(y20_surv)) if y20_surv else 0.0
        hi = float(np.max(y20_surv)) if y20_surv else 0.0
        summary[s] = {
            "survived": n_surv, "n_starts": len(START_YEARS),
            "y20_median": med, "y20_min": lo, "y20_max": hi,
            "ruined_starts": ruined,
            "ruin_years_elapsed": {sy: results[s][sy]["ruin_year_elapsed"] for sy in ruined},
        }
        print("%-14s | %4d/%-3d | %14.0f | %14.0f | %14.0f | %s"
              % (s, n_surv, len(START_YEARS), med, lo, hi,
                 ",".join(str(x) for x in ruined) if ruined else "(none)"))

    # =====================================================================
    # RETURN BLOCK (for report authoring)
    # =====================================================================
    block = {
        "script": "retirement_survival_20260623.py", "date": "2026-06-23",
        "init_asset": INIT_ASSET, "annual_spend": ANNUAL_SPEND,
        "horizon": HORIZON, "start_years": [START_YEARS[0], START_YEARS[-1]],
        "after_tax": AFTER_TAX,
        "sanity_pass": bool(ok_sanity),
        "sc10_annual_stats": {
            "mean": round(float(np.mean(v10)), 4),
            "median": round(float(np.median(v10)), 4),
            "std": round(float(np.std(v10, ddof=1)), 4),
            "max": round(float(np.max(v10)), 4),
            "min": round(float(np.min(v10)), 4),
            "pos": int(np.sum(v10 > 0)), "neg": int(np.sum(v10 < 0)),
        },
        "summary": {s: {
            "survived": summary[s]["survived"], "n_starts": summary[s]["n_starts"],
            "y20_median": round(summary[s]["y20_median"]),
            "y20_min": round(summary[s]["y20_min"]),
            "y20_max": round(summary[s]["y20_max"]),
            "ruined_starts": summary[s]["ruined_starts"],
            "ruin_years_elapsed": summary[s]["ruin_years_elapsed"],
        } for s in strat_order},
        "grid": [{
            "start_year": r["start_year"], "strategy": r["strategy"],
            "survived": r["survived"], "ruin_year_elapsed": r["ruin_year_elapsed"],
            "asset_y10_yen": r["asset_y10_yen"], "asset_y20_yen": r["asset_y20_yen"],
        } for r in grid_rows],
        "sc10_annual_pct": {int(y): round(float(s10.loc[y] * 100), 2)
                            for y in s10.index},
    }
    print("\n" + "=" * 110)
    print("RETURN_BLOCK")
    print("=" * 110)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
