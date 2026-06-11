"""
src/audit/regime_labeler_20260611.py
====================================
Phase 1 of the evaluation-methodology upgrade (EVALUATION_METHODOLOGY_UPGRADE_PLAN
_20260611.md). Produce per-day REGIME labels on three orthogonal axes so any NAV
can be evaluated regime-by-regime ("does it hold up in every regime?").

Axes (each day gets one label per axis):
  trend : 'bull' / 'bear'      -- NASDAQ close vs its 200-day moving average.
  vol   : 'calm' / 'highvol'   -- 63-day realized vol vs its full-sample median.
  rate  : 'rate_up'/'rate_down'-- SOFR(annualized) level today vs 252 trading days ago.

Plus fixed, hand-labelled STRESS event windows (calendar):
  dotcom (2000-03..2002-10), gfc (2007-10..2009-03), covid (2020-02..2020-04),
  bear2022 (2022-01..2022-12), tri2015 (2015-01..2015-12, stock+bond+gold all soft).

Causality note: the labels are descriptive (used for ex-post stratification, NOT as
a trading signal), so the vol median is full-sample. trend/rate use only trailing
data at each day. No NAV is fed in here; labels are aligned by date downstream.

ASCII-only prints. No file writes here (pure library + __main__ smoke test).
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

TRADING_DAYS = 252
MA_TREND = 200
VOL_WIN = 63
RATE_LOOKBACK = 252

# Fixed stress windows (inclusive calendar dates).
STRESS_WINDOWS = {
    "dotcom_2000": ("2000-03-10", "2002-10-09"),
    "gfc_2008":    ("2007-10-09", "2009-03-09"),
    "covid_2020":  ("2020-02-19", "2020-04-07"),
    "bear_2022":   ("2022-01-03", "2022-12-30"),
    "tri_2015":    ("2015-01-02", "2015-12-31"),  # stock/bond/gold all soft
}


def build_regime_labels(close: pd.Series, sofr_daily, dates_dt: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a DataFrame indexed by dates_dt with columns trend/vol/rate (+ helper
    numeric columns). close is the NASDAQ close (same length/order as dates_dt);
    sofr_daily is the per-day SOFR rate series (load_sofr output, ~DTB3/252)."""
    close = pd.Series(np.asarray(close, float), index=dates_dt)
    ret = close.pct_change().fillna(0.0)

    # --- trend: close vs 200d MA (trailing) ---
    ma = close.rolling(MA_TREND, min_periods=MA_TREND).mean()
    trend = np.where(close.values > ma.values, "bull", "bear")
    trend = np.where(np.isnan(ma.values), "n/a", trend)

    # --- vol: 63d realized vol (annualized) vs full-sample median ---
    rv = ret.rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=1) * np.sqrt(TRADING_DAYS)
    vol_med = np.nanmedian(rv.values)
    vol = np.where(rv.values > vol_med, "highvol", "calm")
    vol = np.where(np.isnan(rv.values), "n/a", vol)

    # --- rate: SOFR annual level today vs 252 trading days ago ---
    sofr_annual = pd.Series(np.asarray(sofr_daily, float) * TRADING_DAYS, index=dates_dt)
    sofr_past = sofr_annual.shift(RATE_LOOKBACK)
    rate = np.where(sofr_annual.values > sofr_past.values, "rate_up", "rate_down")
    rate = np.where(np.isnan(sofr_past.values), "n/a", rate)

    df = pd.DataFrame({
        "trend": trend, "vol": vol, "rate": rate,
        "ma200": ma.values, "rv63": rv.values, "sofr_annual": sofr_annual.values,
    }, index=dates_dt)
    df.attrs["vol_median"] = float(vol_med)
    return df


def stress_masks(dates_dt: pd.DatetimeIndex) -> dict:
    """Return {name: boolean mask aligned to dates_dt} for each STRESS window."""
    out = {}
    for name, (a, b) in STRESS_WINDOWS.items():
        lo, hi = pd.Timestamp(a), pd.Timestamp(b)
        out[name] = (dates_dt >= lo) & (dates_dt <= hi)
    return out


def _smoke():
    import src.audit.strategy_runners as sr
    sr._load_dhw1_shared()
    a = sr._DHW1_SHARED["assets"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(a["dates"].values))
    lab = build_regime_labels(a["close"], a["sofr"], dates_dt)
    print("Regime label coverage (fraction of valid days):")
    for ax in ("trend", "vol", "rate"):
        vc = lab[ax][lab[ax] != "n/a"].value_counts(normalize=True)
        print("  %-6s : %s" % (ax, dict(vc.round(3))))
    print("  vol median (annualized) = %.3f" % lab.attrs["vol_median"])
    sm = stress_masks(dates_dt)
    print("Stress window day counts:")
    for k, m in sm.items():
        print("  %-12s : %d days" % (k, int(m.sum())))


if __name__ == "__main__":
    _smoke()
