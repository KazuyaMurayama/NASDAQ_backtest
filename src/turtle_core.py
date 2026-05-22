"""
Turtle Trading System — core computational primitives.

All functions are stateless and operate on numpy/pandas data.
ATR formula: Wilder SMMA  N_t = ((period-1)*N_{t-1} + TR_t) / period
Seed: simple mean of the first `period` TR values (bars 1..period).
"""
import numpy as np
import pandas as pd


def wilder_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               period: int = 20) -> np.ndarray:
    """
    Wilder SMMA ATR (called N in turtle literature).

    Returns array of same length as inputs; NaN for index < period.
    tr[0] falls back to H[0]-L[0] (no previous close available).
    Seed is mean(tr[1 : period+1]) — first `period` valid TR values.
    """
    n = len(highs)
    hl = highs - lows
    # vectorised true-range components; index-0 gets 0 placeholder → overwritten below
    hc = np.concatenate([[0.0], np.abs(highs[1:] - closes[:-1])])
    lc = np.concatenate([[0.0], np.abs(lows[1:]  - closes[:-1])])
    tr = np.maximum(hl, np.maximum(hc, lc))
    tr[0] = highs[0] - lows[0]

    atr = np.full(n, np.nan)
    if n <= period:
        return atr

    atr[period] = tr[1:period + 1].mean()   # seed from bars 1..period
    a_prev = (period - 1) / period           # 19/20 for period=20
    a_new  = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = a_prev * atr[i - 1] + a_new * tr[i]
    return atr


def compute_donchian_high(highs: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling max of the previous `window` HIGH bars (exclusive of current bar).

    Breakout signal at bar t: close[t] > compute_donchian_high(highs, window)[t]
    Returns NaN for t < window.
    """
    n = len(highs)
    result = np.full(n, np.nan)
    for t in range(window, n):
        result[t] = highs[t - window:t].max()
    return result


def compute_donchian_low(lows: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling min of the previous `window` LOW bars (exclusive of current bar).

    Exit signal at bar t: close[t] < compute_donchian_low(lows, window)[t]
    Returns NaN for t < window.
    """
    n = len(lows)
    result = np.full(n, np.nan)
    for t in range(window, n):
        result[t] = lows[t - window:t].min()
    return result


def unit_size(equity: float, n_value: float, dollar_per_point: float = 1.0) -> float:
    """
    Turtle Unit size in shares/contracts.

    Unit = (equity * 0.01) / (N * dollar_per_point)
    For TQQQ (ETF): dollar_per_point = 1.0 (price in $/share).
    N from TQQQ synthetic ATR already embeds 3x scaling — no manual adjustment needed.
    """
    if n_value <= 0 or not np.isfinite(n_value):
        return 0.0
    return (equity * 0.01) / (n_value * dollar_per_point)
