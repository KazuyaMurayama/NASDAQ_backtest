"""
src/audit/horizon_returns_20260611.py
=====================================
5-year / 10-year horizon-return deep dive for the v7 candidates, plus the full
standard-10 metrics, for a horizon-focused investor.

For V7_TQQQ / P09_TQQQ / LU1_cfd / vz065_l5 (after-tax x0.8273 on CAGR):
  - rolling 5y CAGR (daily, 1260d): min, p10, p25, median, %>0, %>10%
  - rolling 10y CAGR (calendar-year, section 3.5 "star"): min(=Worst10Y*), p25, median, %>0
  - standard-10 echo from compute_10metrics (CAGR_IS/OOS, Sharpe, MaxDD, Worst10Y*,
    P10_5Y, Worst5Y, Trades_yr) + WFA CI95_lo.

Reuses the validated NAV builders. ASCII-only. Saves CSV; no commit.
"""
from __future__ import annotations

import os
import sys
import types
import json

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
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base, _build_p09_on_base, LU1_MAP
from src.audit.run_p09_tqqq_validate_20260611 import _run_wfa
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, _ret_from_nav_level, _inverse_vol_weights,
)
from src.audit.run_p02_p09_backtest_20260611 import GATE_DELAY, _load_macro_signal
from compute_cfd_worst10y import nav_to_annual, rolling_nY_cagr

AFTER_TAX = 0.8273


def _roll5y(nav):
    v = np.asarray(nav.values, float)
    n = len(v); h = 1260
    out = []
    for i in range(h, n):
        if v[i - h] > 0:
            out.append(((v[i] / v[i - h]) ** (1.0 / 5.0) - 1.0) * AFTER_TAX)
    return np.asarray(out, float)


def _roll10y_star(nav):
    dates = pd.Series(nav.index, index=nav.index)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, n=10)
    return np.asarray(r10, float) * AFTER_TAX


def _stats5(a):
    return dict(min=float(np.min(a)), p10=float(np.percentile(a, 10)),
                p25=float(np.percentile(a, 25)), median=float(np.median(a)),
                frac_pos=float(np.mean(a > 0)), frac_gt10=float(np.mean(a > 0.10)))


def _stats10(a):
    a = a[~np.isnan(a)]
    return dict(min=float(np.min(a)), p25=float(np.percentile(a, 25)),
                median=float(np.median(a)), frac_pos=float(np.mean(a > 0)))


def main():
    print("=" * 92)
    print("HORIZON RETURNS (5y / 10y) + STANDARD-10   v7 candidates   2026-06-11")
    print("=" * 92)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt); n_years = n / 252.0

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x); ret_bond = _ret_from_nav_level(bond_1x)
    mask = np.asarray(shared["mask"], float)
    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool); fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    # Build the 4 NAVs
    v7_nav, _, tpy_v7, _ = _build_tqqq_base(shared, dates_dt, cfd_excess=False)
    p09b, r_p09b, tpy_p9b, _ = _build_tqqq_base(shared, dates_dt, cfd_excess=False)
    p09_nav, _, tpy_p9 = _build_p09_on_base(r_p09b, ret_gold, ret_bond, fund_active, wg, wb, bond_on, dates_dt, tpy_p9b, n_years)
    lu1b, r_lu1b, tpy_l1b, _ = _build_tqqq_base(shared, dates_dt, v7_map=LU1_MAP, cfd_excess=True)
    lu1_nav, _, tpy_l1 = _build_p09_on_base(r_lu1b, ret_gold, ret_bond, fund_active, wg, wb, bond_on, dates_dt, tpy_l1b, n_years)
    vz = sr.run_vz065(5.0, "realistic"); vz_nav = vz["nav"]

    navs = {"V7_TQQQ": (v7_nav, tpy_v7), "P09_TQQQ": (p09_nav, tpy_p9),
            "LU1_cfd": (lu1_nav, tpy_l1), "vz065_l5": (vz_nav, vz["trades_per_year"])}
    order = ["V7_TQQQ", "P09_TQQQ", "LU1_cfd", "vz065_l5"]

    rows = []
    print("\n--- ROLLING 5-YEAR CAGR (after-tax) ---")
    print("%-9s | %7s | %7s | %7s | %7s | %7s | %7s" % ("cand","min","p10","p25","median",">0%",">10%"))
    print("-" * 72)
    s5 = {}
    for k in order:
        a5 = _roll5y(navs[k][0]); s5[k] = _stats5(a5)
        x = s5[k]
        print("%-9s | %+6.2f%% | %+6.2f%% | %+6.2f%% | %+6.2f%% | %5.0f%% | %5.0f%%"
              % (k, 100*x["min"], 100*x["p10"], 100*x["p25"], 100*x["median"], 100*x["frac_pos"], 100*x["frac_gt10"]))

    print("\n--- ROLLING 10-YEAR CAGR (calendar-year, star; after-tax) ---")
    print("%-9s | %7s | %7s | %7s | %7s" % ("cand","min(W10*)","p25","median",">0%"))
    print("-" * 56)
    s10 = {}
    for k in order:
        a10 = _roll10y_star(navs[k][0]); s10[k] = _stats10(a10)
        x = s10[k]
        print("%-9s | %+7.2f%% | %+6.2f%% | %+6.2f%% | %5.0f%%"
              % (k, 100*x["min"], 100*x["p25"], 100*x["median"], 100*x["frac_pos"]))

    print("\n--- STANDARD-10 (after-tax CAGR family; Sharpe/MaxDD pretax) ---")
    print("%-9s | %8s | %8s | %7s | %8s | %8s | %7s | %7s | %6s | %8s"
          % ("cand","CAGR_IS","CAGR_OOS","Sharpe","MaxDD","W10Y*","P10_5Y","W5Y","Trd","WFA_CI95"))
    print("-" * 100)
    std = {}
    for k in order:
        nav, tpy = navs[k]
        m = compute_10metrics(nav, tpy)
        w = _run_wfa(nav, k)
        std[k] = dict(m=m, wfa=w)
        print("%-9s | %+7.2f%% | %+7.2f%% | %6.3f | %+7.2f%% | %+6.2f%% | %+5.2f%% | %+5.2f%% | %5.1f | %+7.2f%%"
              % (k, 100*m["CAGR_IS"]*AFTER_TAX, 100*m["CAGR_OOS"]*AFTER_TAX, m["Sharpe_OOS"],
                 100*m["MaxDD_FULL"], 100*m["Worst10Y_star"]*AFTER_TAX, 100*m["P10_5Y"]*AFTER_TAX,
                 100*m["Worst5Y"]*AFTER_TAX, m["Trades_yr"], 100*w["CI95_lo"]))
        rows.append(dict(candidate=k,
            roll5_min=s5[k]["min"], roll5_p10=s5[k]["p10"], roll5_p25=s5[k]["p25"],
            roll5_median=s5[k]["median"], roll5_fracpos=s5[k]["frac_pos"], roll5_fracgt10=s5[k]["frac_gt10"],
            roll10_min=s10[k]["min"], roll10_p25=s10[k]["p25"], roll10_median=s10[k]["median"], roll10_fracpos=s10[k]["frac_pos"],
            CAGR_IS_at=m["CAGR_IS"]*AFTER_TAX, CAGR_OOS_at=m["CAGR_OOS"]*AFTER_TAX,
            Sharpe_OOS=m["Sharpe_OOS"], MaxDD=m["MaxDD_FULL"],
            Worst10Y_star_at=m["Worst10Y_star"]*AFTER_TAX, P10_5Y_at=m["P10_5Y"]*AFTER_TAX,
            Worst5Y_at=m["Worst5Y"]*AFTER_TAX, Trades_yr=m["Trades_yr"], WFA_CI95_lo=w["CI95_lo"]))

    out_csv = os.path.join(_REPO_DIR, "audit_results", "horizon_returns_20260611.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
    print("\nSaved CSV: %s" % out_csv)
    return rows


if __name__ == "__main__":
    main()
