"""
src/audit/p09_forward_return_20260612.py
========================================
P09-strategy-specific 5-business-day forward-return bin table (1974-2026),
replacing the Dyn2x3x regime table previously borrowed by the GAS notifier.

Bins (decision-level signals as shown in the notification):
  IN  x mom63 bucket Q0..Q3 (prev-day, frozen quartiles) x lev band (lo<=0.6 / hi>0.6)
  OUT x bond_on (ON / OFF)
  -> 10 bins.

For each bin over the FULL history:
  n_days, median 5d return (%), annualized CAGR from mean 5d log-return,
  pct positive. Computed for the P09_TQQQ NAV (pretax gross) and the LU1
  (CFD-recost) NAV for reference.

Outputs:
  - audit_results/p09_forward_return_20260612.csv
  - prints a ready-to-paste GAS constant block (FORWARD_RETURN_P09_)

ASCII-only prints.
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
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base, _build_p09_on_base, LU1_MAP
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS, _ret_from_nav_level, _inverse_vol_weights,
)
from src.audit.run_p02_p09_backtest_20260611 import GATE_DELAY, _load_macro_signal

H = 5  # forward horizon (business days)

# frozen quartiles (p09_live_spec_20260611.json)
Q25 = -0.028802750000000002
Q50 = 0.0375715
Q75 = 0.09098425
LEV_SPLIT = 0.6


def _bucket(m):
    if np.isnan(m):
        return -1
    if m <= Q25:
        return 0
    if m <= Q50:
        return 1
    if m <= Q75:
        return 2
    return 3


def _bin_stats(r5):
    """r5: array of 5d simple forward returns for one bin."""
    r5 = r5[np.isfinite(r5)]
    n = len(r5)
    if n == 0:
        return dict(n=0, median_5d=np.nan, cagr=np.nan, pct_pos=np.nan)
    med = float(np.median(r5))
    mean_log = float(np.mean(np.log1p(r5)))
    cagr = float(np.expm1(mean_log * TRADING_DAYS / H))
    return dict(n=n, median_5d=med * 100.0, cagr=cagr * 100.0,
                pct_pos=100.0 * float(np.mean(r5 > 0)))


def main():
    print("=" * 86)
    print("P09 STRATEGY-SPECIFIC 5D FORWARD-RETURN BINS (1974-2026)   2026-06-12")
    print("=" * 86)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt); n_years = n / TRADING_DAYS

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

    # NAVs
    p09b, r_p09b, tpy_p9b, _ = _build_tqqq_base(shared, dates_dt, cfd_excess=False)
    p09_nav, _, _ = _build_p09_on_base(
        r_p09b, ret_gold, ret_bond, fund_active, wg, wb, bond_on, dates_dt, tpy_p9b, n_years)
    lu1b, r_lu1b, tpy_l1b, _ = _build_tqqq_base(shared, dates_dt, v7_map=LU1_MAP, cfd_excess=True)
    lu1_nav, _, _ = _build_p09_on_base(
        r_lu1b, ret_gold, ret_bond, fund_active, wg, wb, bond_on, dates_dt, tpy_l1b, n_years)

    # decision-level signals (as the notification reports them)
    lev_raw = np.asarray(a["lev_raw"], float)
    macro = pd.read_csv(os.path.join(_REPO_DIR, "data", "macro_features.csv"),
                        parse_dates=["date"], index_col="date").sort_index()
    mom63_raw = macro["nasdaq_mom63"].dropna()
    mom63 = mom63_raw.reindex(
        mom63_raw.index.union(dates_dt)).ffill().reindex(dates_dt).values

    def _fwd5(nav):
        v = np.asarray(nav.values, float)
        out = np.full(n, np.nan)
        out[:n - H] = v[H:] / v[:n - H] - 1.0
        return out

    f_p09 = _fwd5(p09_nav)
    f_lu1 = _fwd5(lu1_nav)

    # bin keys per day (live rule: prev-day mom63 bucket)
    keys = np.empty(n, dtype=object)
    for i in range(1, n):
        if mask[i] >= 0.5:
            q = _bucket(mom63[i - 1])
            if q < 0:
                keys[i] = None
                continue
            band = "lo" if lev_raw[i] <= LEV_SPLIT else "hi"
            keys[i] = "IN_Q%d_%s" % (q, band)
        else:
            keys[i] = "OUT_%s" % ("bondON" if bond_on[i] else "bondOFF")
    keys[0] = None

    order = (["IN_Q%d_%s" % (q, b) for q in range(4) for b in ("lo", "hi")]
             + ["OUT_bondON", "OUT_bondOFF"])

    rows = []
    gas_p09 = {}
    print("\n%-14s | %6s | %8s | %10s | %7s || %8s | %10s" %
          ("bin", "n", "med5d%", "CAGR%(P09)", ">0%", "med5d%", "CAGR%(LU1)"))
    print("-" * 86)
    for k in order:
        sel = np.array([keys[i] == k for i in range(n)])
        sp = _bin_stats(f_p09[sel])
        sl = _bin_stats(f_lu1[sel])
        print("%-14s | %6d | %+7.2f%% | %+9.1f%% | %6.1f%% || %+7.2f%% | %+9.1f%%" %
              (k, sp["n"], sp["median_5d"], sp["cagr"], sp["pct_pos"],
               sl["median_5d"], sl["cagr"]))
        rows.append(dict(bin=k, n=sp["n"],
                         p09_median_5d_pct=sp["median_5d"], p09_cagr_pct=sp["cagr"],
                         p09_pct_pos=sp["pct_pos"],
                         lu1_median_5d_pct=sl["median_5d"], lu1_cagr_pct=sl["cagr"],
                         lu1_pct_pos=sl["pct_pos"]))
        gas_p09[k] = {"cagr": round(sp["cagr"], 1), "median": round(sp["median_5d"], 2)}

    out_csv = os.path.join(_REPO_DIR, "audit_results", "p09_forward_return_20260612.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.4f")
    print("\nSaved CSV: %s" % out_csv)

    print("\n--- GAS constant block (paste into P09Signal.gs) ---")
    print("var FORWARD_RETURN_P09_ = " + json.dumps(gas_p09, indent=2) + ";")
    return rows


if __name__ == "__main__":
    main()
