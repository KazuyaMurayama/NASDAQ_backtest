"""Forward log returns for signal IC evaluation.

Builds (asset, horizon) forward return matrix from a price DataFrame.
Used by Phase B screening (signals.ic, signals.hit_rate) to compute
the dependent variable for predictive power tests.

Note: Uses unleveraged underlying prices (NDX / IEF / GLD spot) - not the
leveraged products (TQQQ / TMF / UGL). The Phase C strategy runner re-scales.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd


_DEFAULT_HORIZONS = [5, 20, 60, 252]


def build_forward_returns(
    prices: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Return log returns shifted backward by each horizon.

    Args:
        prices: DataFrame with columns like ['TQQQ', 'TLT', 'GLD'] indexed by date.
                Each column = adjusted close prices (positive, no NaN at start; NaN OK in middle).
        horizons: forward periods in business days.

    Returns:
        DataFrame with MultiIndex columns (asset, horizon).
        Value at index t = log(price[t+horizon] / price[t]).
        Last `horizon` rows of each (asset, horizon) column are NaN.
    """
    if horizons is None:
        horizons = _DEFAULT_HORIZONS
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive (got zero or negative)")

    log_prices = np.log(prices)
    cols = {}
    for asset in prices.columns:
        lp = log_prices[asset]
        for h in horizons:
            cols[(asset, h)] = lp.shift(-h) - lp

    out = pd.DataFrame(cols, index=prices.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=['asset', 'horizon'])
    return out


def load_default_prices() -> pd.DataFrame:
    """Helper: load NDX/IEF/GLD from existing repo data files.

    Returns DataFrame with columns ['NDX', 'IEF', 'GLD'] indexed by date.
    """
    from pathlib import Path
    base = Path(__file__).resolve().parents[2]

    # NDX from NASDAQ_extended_to_2026.csv (root)
    ndx_path = base / 'NASDAQ_extended_to_2026.csv'
    ndx_df = pd.read_csv(ndx_path, parse_dates=['Date'], index_col='Date').sort_index()
    ndx_col = next((c for c in ndx_df.columns if c.lower() in ('close', 'adj close', 'price')), ndx_df.columns[0])
    ndx = ndx_df[ndx_col].rename('NDX')

    # IEF from data/ief_daily.csv
    ief_df = pd.read_csv(base / 'data' / 'ief_daily.csv', parse_dates=['Date'], index_col='Date').sort_index()
    ief = ief_df.iloc[:, 0].rename('IEF')

    # Gold from data/lbma_gold_daily.csv
    gold_df = pd.read_csv(base / 'data' / 'lbma_gold_daily.csv', parse_dates=['Date'], index_col='Date').sort_index()
    gold = gold_df.iloc[:, 0].rename('GLD')

    df = pd.concat([ndx, ief, gold], axis=1).dropna()
    return df
