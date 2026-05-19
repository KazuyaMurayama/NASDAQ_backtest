"""
Turtle Trading System — data loading and synthetic NAV builders.

Builds TQQQ / SQQQ synthetic OHLC by applying 3x (or -3x) daily reset
to NASDAQ daily relative moves, deducting TER + 2*SOFR + swap_spread.

Pre-2000 NASDAQ has Open=High=Low=Close; post-2000 has real intraday OHLC.
The 3x scaling preserves the intraday H/L pattern via:
  tqqq_X[t] = nav[t-1] * (1 + 3 * (nasdaq_X[t] / nasdaq_close[t-1] - 1))
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from product_costs import TQQQ

_BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE, 'data')
_NASDAQ_CSV = os.path.join(_BASE, 'NASDAQ_extended_to_2026.csv')

TRADING_DAYS = 252


def load_nasdaq() -> pd.DataFrame:
    """Load NASDAQ daily 1974-2026 with [Open, High, Low, Close] indexed by Date."""
    df = pd.read_csv(_NASDAQ_CSV, parse_dates=['Date'])
    df = df.set_index('Date')[['Open', 'High', 'Low', 'Close']].astype(float)
    df = df.sort_index()
    return df


def load_dtb3_aligned(dates: pd.DatetimeIndex) -> pd.Series:
    """SOFR proxy (DTB3) annualised decimal, aligned to NASDAQ trading dates."""
    path = os.path.join(_DATA_DIR, 'dtb3_daily.csv')
    s = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    s = pd.to_numeric(s['yield_pct'], errors='coerce').ffill(limit=5)
    s = (s / 100.0).reindex(dates).ffill(limit=10).bfill(limit=10)
    return s


def build_tqqq_synthetic_ohlc(nasdaq: pd.DataFrame, sofr_annual: pd.Series,
                               initial_price: float = 1.0,
                               leverage: float = 3.0) -> pd.DataFrame:
    """
    Build TQQQ-equivalent synthetic OHLC.

    Close-to-close daily return:
        r_tqqq[t] = leverage * r_nas[t] - daily_cost
        daily_cost = (TER + leverage_factor*SOFR + swap_spread)/252
    For TQQQ (lev=3): financing = 2x SOFR + 0.50% swap (per product_costs.py).

    OHLC scaling for intraday bars (relative to previous nav):
        tqqq_X[t] = nav[t-1] * (1 + leverage * (nasdaq_X[t]/nasdaq_close[t-1] - 1))
    where X ∈ {Open, High, Low, Close}.
    """
    nas_close = nasdaq['Close'].values
    nas_open  = nasdaq['Open'].values
    nas_high  = nasdaq['High'].values
    nas_low   = nasdaq['Low'].values
    n = len(nas_close)

    sofr = sofr_annual.values
    ter_d   = TQQQ.ter / TRADING_DAYS
    swap_d  = TQQQ.swap_spread / TRADING_DAYS

    nav    = np.empty(n); nav[0] = initial_price
    tqqq_o = np.empty(n); tqqq_h = np.empty(n); tqqq_l = np.empty(n); tqqq_c = np.empty(n)
    tqqq_o[0] = tqqq_h[0] = tqqq_l[0] = tqqq_c[0] = initial_price

    for i in range(1, n):
        prev_close = nas_close[i-1]
        if prev_close <= 0 or not np.isfinite(prev_close):
            nav[i] = nav[i-1]
            tqqq_o[i] = tqqq_h[i] = tqqq_l[i] = tqqq_c[i] = nav[i-1]
            continue

        r_close = nas_close[i] / prev_close - 1.0
        cost_d  = ter_d + TQQQ.sofr_multiplier * sofr[i] / TRADING_DAYS + swap_d
        r_tqqq  = leverage * r_close - cost_d
        nav[i]  = nav[i-1] * (1.0 + r_tqqq)

        r_o = nas_open[i]  / prev_close - 1.0
        r_h = nas_high[i]  / prev_close - 1.0
        r_l = nas_low[i]   / prev_close - 1.0
        tqqq_o[i] = nav[i-1] * (1.0 + leverage * r_o)
        tqqq_h[i] = nav[i-1] * (1.0 + leverage * r_h)
        tqqq_l[i] = nav[i-1] * (1.0 + leverage * r_l)
        tqqq_c[i] = nav[i]

        tqqq_h[i] = max(tqqq_h[i], tqqq_o[i], tqqq_c[i])
        tqqq_l[i] = min(tqqq_l[i], tqqq_o[i], tqqq_c[i])

    return pd.DataFrame({'Open': tqqq_o, 'High': tqqq_h,
                         'Low': tqqq_l, 'Close': tqqq_c},
                        index=nasdaq.index)


def build_sqqq_synthetic_ohlc(nasdaq: pd.DataFrame, sofr_annual: pd.Series,
                               initial_price: float = 1.0) -> pd.DataFrame:
    """
    Build SQQQ-equivalent synthetic OHLC (-3x daily reset).
    Same cost structure as TQQQ. Note: for negative leverage,
    nasdaq's High becomes sqqq's Low and vice versa.
    """
    nas_close = nasdaq['Close'].values
    nas_open  = nasdaq['Open'].values
    nas_high  = nasdaq['High'].values
    nas_low   = nasdaq['Low'].values
    n = len(nas_close)

    sofr = sofr_annual.values
    ter_d   = TQQQ.ter / TRADING_DAYS
    swap_d  = TQQQ.swap_spread / TRADING_DAYS
    leverage = -3.0

    nav    = np.empty(n); nav[0] = initial_price
    sq_o = np.empty(n); sq_h = np.empty(n); sq_l = np.empty(n); sq_c = np.empty(n)
    sq_o[0] = sq_h[0] = sq_l[0] = sq_c[0] = initial_price

    for i in range(1, n):
        prev_close = nas_close[i-1]
        if prev_close <= 0 or not np.isfinite(prev_close):
            nav[i] = nav[i-1]
            sq_o[i] = sq_h[i] = sq_l[i] = sq_c[i] = nav[i-1]
            continue

        r_close = nas_close[i] / prev_close - 1.0
        cost_d  = ter_d + TQQQ.sofr_multiplier * sofr[i] / TRADING_DAYS + swap_d
        r_sqqq  = leverage * r_close - cost_d
        nav[i]  = nav[i-1] * (1.0 + r_sqqq)

        r_o = nas_open[i]  / prev_close - 1.0
        r_h = nas_high[i]  / prev_close - 1.0
        r_l = nas_low[i]   / prev_close - 1.0
        sq_o[i] = nav[i-1] * (1.0 + leverage * r_o)
        sq_h[i] = nav[i-1] * (1.0 + leverage * r_l)
        sq_l[i] = nav[i-1] * (1.0 + leverage * r_h)
        sq_c[i] = nav[i]

        sq_h[i] = max(sq_h[i], sq_o[i], sq_c[i])
        sq_l[i] = min(sq_l[i], sq_o[i], sq_c[i])

    return pd.DataFrame({'Open': sq_o, 'High': sq_h,
                         'Low': sq_l, 'Close': sq_c},
                        index=nasdaq.index)


def make_pseudo_ohlc(close: pd.Series) -> pd.DataFrame:
    """For close-only series, set Open=High=Low=Close."""
    return pd.DataFrame({'Open': close, 'High': close, 'Low': close, 'Close': close},
                        index=close.index)
