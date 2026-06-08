"""Causal timing-signal builders for the single-asset hold-vs-cash sweep.

Every function returns a position series in {0.0, 1.0} (1 = hold the asset,
0 = cash) and is **causal**: the position acted on day t uses only
information available through day t-1 (a trailing `.shift(1)`), so a future
value can never change an earlier position. This is verified by the
"is_causal" tests in tests/multi_asset/test_bond_signals.py.

See docs/multiasset/SIGNAL_TAXONOMY.md.
"""
from __future__ import annotations

import pandas as pd


def ma_cross_position(price: pd.Series, window: int) -> pd.Series:
    """Hold when price is above its trailing moving average (B-TR)."""
    ma = price.rolling(window).mean()
    raw = (price > ma).astype(float)
    raw[ma.isna()] = float('nan')
    return raw.shift(1)


def momentum_position(price: pd.Series, lookback: int) -> pd.Series:
    """Hold when trailing `lookback`-period return is positive (B-TR / mom)."""
    mom = price / price.shift(lookback) - 1.0
    raw = (mom > 0).astype(float)
    raw[mom.isna()] = float('nan')
    return raw.shift(1)


def zscore_position(series: pd.Series, window: int, enter: float,
                    invert: bool = False) -> pd.Series:
    """Hold when the trailing z-score of `series` is at/above `enter`.

    z_t = (x_t - rolling_mean) / rolling_std over `window`. With invert=True
    the rule flips (hold when z <= enter), e.g. for negatively-correlated
    drivers like real yields for gold.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std
    if invert:
        raw = (z <= enter).astype(float)
    else:
        raw = (z >= enter).astype(float)
    raw[z.isna()] = float('nan')
    return raw.shift(1)
