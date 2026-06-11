"""
src/audit/yearly_returns_20260611.py
====================================
Calendar-year (annual) returns for the HORIZON_AND_SCORECARD report, three series:
  - LU1        (TQQQ base + CFD-excess re-cost on >3x, P09 OUT-fill)
  - P09_TQQQ   (Active candidate; TQQQ base, P09 OUT-fill, ~0% CFD)
  - NASDAQ 1X B&H (underlying index buy & hold, benchmark, no cost)

Reuses the validated NAV builders from lu_cfd_recost_20260611. Annual returns are
PRETAX gross calendar-year total returns (section 3.5 convention): per-year tax is
not well defined (you don't pay tax in a down year), so the standard yearly table
is gross -- consistent with B9_YEARLY_RETURNS / CFD_YEARLY_RETURNS prior reports.
Summary CAGRs are echoed both pretax and after-tax for context.

ASCII-only. Saves CSV; no commit.
"""
from __future__ import annotations

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
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base, _build_p09_on_base, LU1_MAP
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS, _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns,
)
from src.audit.run_p02_p09_backtest_20260611 import GATE_DELAY, _load_macro_signal

AFTER_TAX = 0.8273
OOS_START_YEAR = 2021  # OOS begins 2021-05-08


def _cagr(nav_dt):
    v = np.asarray(nav_dt.values, float)
    yrs = (nav_dt.index[-1] - nav_dt.index[0]).days / 365.25
    return (v[-1] / v[0]) ** (1.0 / yrs) - 1.0


def main():
    print("=" * 84)
    print("YEARLY (calendar-year) RETURNS  LU1 / P09_TQQQ / NASDAQ 1X B&H   2026-06-11")
    print("=" * 84)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt); n_years = n / TRADING_DAYS

    # Gold/Bond 1x legs + OUT mask + fund lag (same wiring as P09)
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

    # ---- P09_TQQQ ----
    p09b, r_p09b, tpy_p9b, _ = _build_tqqq_base(shared, dates_dt, cfd_excess=False)
    p09_nav, _, _ = _build_p09_on_base(
        r_p09b, ret_gold, ret_bond, fund_active, wg, wb, bond_on, dates_dt, tpy_p9b, n_years)

    # ---- LU1 (CFD-excess recost) ----
    lu1b, r_lu1b, tpy_l1b, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=LU1_MAP, cfd_excess=True)
    lu1_nav, _, _ = _build_p09_on_base(
        r_lu1b, ret_gold, ret_bond, fund_active, wg, wb, bond_on, dates_dt, tpy_l1b, n_years)

    # ---- NASDAQ 1X B&H (underlying index buy & hold, no cost) ----
    close = np.asarray(a["close"].values, float)
    bh_nav = pd.Series(close / close[0], index=dates_dt)

    series = {"LU1": lu1_nav, "P09_TQQQ": p09_nav, "NASDAQ_1X_BH": bh_nav}
    order = ["LU1", "P09_TQQQ", "NASDAQ_1X_BH"]

    # Calendar-year returns
    cy = {k: _calendar_year_returns(series[k]) for k in order}
    years = sorted(cy["LU1"].index.tolist())

    # ---- Print yearly table ----
    print("\n--- ANNUAL CALENDAR-YEAR RETURNS (pretax gross, %) ---")
    print("%-7s | %9s | %9s | %12s" % ("year", "LU1", "P09_TQQQ", "NASDAQ_1X_BH"))
    print("-" * 50)
    rows = []
    for y in years:
        tag = " [OOS]" if y >= OOS_START_YEAR else ""
        vals = {k: (cy[k].get(y, np.nan)) for k in order}
        print("%-7s | %+8.2f%% | %+8.2f%% | %+11.2f%%%s"
              % (y, 100 * vals["LU1"], 100 * vals["P09_TQQQ"], 100 * vals["NASDAQ_1X_BH"], tag))
        rows.append(dict(year=y, LU1=vals["LU1"], P09_TQQQ=vals["P09_TQQQ"],
                         NASDAQ_1X_BH=vals["NASDAQ_1X_BH"], oos=(y >= OOS_START_YEAR)))

    # ---- Summary stats ----
    print("\n--- SUMMARY (pretax gross unless noted) ---")
    print("%-16s | %9s | %9s | %12s" % ("stat", "LU1", "P09_TQQQ", "NASDAQ_1X_BH"))
    print("-" * 56)
    def _line(name, fn):
        print("%-16s | %+8.2f%% | %+8.2f%% | %+11.2f%%"
              % (name, 100 * fn("LU1"), 100 * fn("P09_TQQQ"), 100 * fn("NASDAQ_1X_BH")))
    _line("CAGR(FULL) pre", lambda k: _cagr(series[k]))
    _line("CAGR(FULL) aft", lambda k: _cagr(series[k]) * AFTER_TAX)
    _line("median yr", lambda k: float(cy[k].median()))
    _line("best yr", lambda k: float(cy[k].max()))
    _line("worst yr", lambda k: float(cy[k].min()))
    print("%-16s | %9d | %9d | %12d"
          % ("positive yrs", int((cy["LU1"] > 0).sum()),
             int((cy["P09_TQQQ"] > 0).sum()), int((cy["NASDAQ_1X_BH"] > 0).sum())))
    print("%-16s | %9d | %9d | %12d"
          % ("negative yrs", int((cy["LU1"] <= 0).sum()),
             int((cy["P09_TQQQ"] <= 0).sum()), int((cy["NASDAQ_1X_BH"] <= 0).sum())))

    out_csv = os.path.join(_REPO_DIR, "audit_results", "yearly_returns_20260611.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
    print("\nSaved CSV: %s" % out_csv)
    return rows


if __name__ == "__main__":
    main()
