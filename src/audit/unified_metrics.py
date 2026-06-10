"""
Unified 10-metric harness for NASDAQ backtest strategies.

Computes the standard 10 indicators from a NAV pd.Series (DatetimeIndex)
by importing canonical functions without modifying source files.

Split constants (public):
  IS_END    = 2021-05-07
  OOS_START = 2021-05-08

Returns dict with keys:
  CAGR_IS, CAGR_OOS, CAGR_FULL, IS_OOS_gap_pp,
  Sharpe_OOS, MaxDD_FULL, Worst10Y_star, P10_5Y, Worst5Y, Trades_yr
"""
import sys
import os
import types

# ---------------------------------------------------------------------------
# Patch multitasking (yfinance dependency) before any src imports
# ---------------------------------------------------------------------------
if 'multitasking' not in sys.modules:
    _m = types.ModuleType('multitasking')
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules['multitasking'] = _m

# Ensure src/ is importable
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Split constants (module-level public API)
# ---------------------------------------------------------------------------
IS_END    = pd.Timestamp('2021-05-07')
OOS_START = pd.Timestamp('2021-05-08')

# ---------------------------------------------------------------------------
# Canonical imports (read-only, never modified)
# ---------------------------------------------------------------------------
from cfd_leverage_backtest import calc_7metrics  # noqa: E402
from compute_cfd_worst10y import nav_to_annual, rolling_nY_cagr  # noqa: E402
from calculate_p10_5y import compute_p10_5y, compute_worst5y  # noqa: E402


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_10metrics(nav: pd.Series, trades_per_year: float) -> dict:
    """Compute standard 10 metrics from a NAV series.

    Parameters
    ----------
    nav : pd.Series
        Daily NAV with DatetimeIndex.
    trades_per_year : float
        Number of round-trip trades per year (passed through as Trades_yr).

    Returns
    -------
    dict with keys:
        CAGR_IS, CAGR_OOS, CAGR_FULL, IS_OOS_gap_pp,
        Sharpe_OOS, MaxDD_FULL, Worst10Y_star, P10_5Y, Worst5Y, Trades_yr
    """
    # ---- Sanitise input ----
    nav = nav.dropna().sort_index()
    if len(nav) == 0:
        raise ValueError("NAV series is empty after dropna().")

    # ---- Build dates series required by calc_7metrics ----
    # calc_7metrics expects: nav (pd.Series, positional), dates (pd.Series of Timestamps)
    # It filters internally using its own module constants IS_START/IS_END/OOS_START/FULL_END.
    dates = pd.Series(nav.index, index=nav.index)

    # ---- Call canonical 7-metric function ----
    m7 = calc_7metrics(nav, dates, trades_per_year=trades_per_year)

    # ---- Worst10Y_star via calendar-year rolling ----
    try:
        ann = nav_to_annual(nav, dates)
        r10 = rolling_nY_cagr(ann, n=10)
        worst10y_star = float(r10.min()) if len(r10) > 0 else np.nan
    except Exception:
        worst10y_star = np.nan

    # ---- P10_5Y and Worst5Y via daily rolling ----
    try:
        p10_5y = compute_p10_5y(nav.values)
    except Exception:
        p10_5y = np.nan

    try:
        worst5y = compute_worst5y(nav.values)
    except Exception:
        worst5y = np.nan

    # ---- IS-OOS gap ----
    cagr_is  = m7.get('CAGR_IS',   np.nan)
    cagr_oos = m7.get('CAGR_OOS',  np.nan)
    if not (np.isnan(cagr_is) or np.isnan(cagr_oos)):
        is_oos_gap_pp = (cagr_is - cagr_oos) * 100.0
    else:
        is_oos_gap_pp = np.nan

    return {
        'CAGR_IS':        m7.get('CAGR_IS',    np.nan),
        'CAGR_OOS':       m7.get('CAGR_OOS',   np.nan),
        'CAGR_FULL':      m7.get('CAGR_FULL',  np.nan),
        'IS_OOS_gap_pp':  is_oos_gap_pp,
        'Sharpe_OOS':     m7.get('Sharpe_OOS', np.nan),
        'MaxDD_FULL':     m7.get('MaxDD_FULL', np.nan),
        'Worst10Y_star':  worst10y_star,
        'P10_5Y':         p10_5y,
        'Worst5Y':        worst5y,
        'Trades_yr':      trades_per_year,
    }
