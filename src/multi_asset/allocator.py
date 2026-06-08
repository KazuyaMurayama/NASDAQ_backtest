"""Portfolio allocation logics across asset strategy-return streams (Phase 3).

Each input column is one asset's *timed* daily return (the asset's confirmed
hold/cash strategy). Allocators return a weights DataFrame (rows sum to 1) that
is **causal** (uses only trailing data). combine_portfolio applies the weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equal_weights(ret_df: pd.DataFrame) -> pd.DataFrame:
    """Static 1/N weights."""
    n = ret_df.shape[1]
    return pd.DataFrame(1.0 / n, index=ret_df.index, columns=ret_df.columns)


def inverse_vol_weights(ret_df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Risk-parity proxy: weight ∝ 1 / trailing volatility, normalized. Causal."""
    vol = ret_df.rolling(window).std()
    inv = 1.0 / vol
    w = inv.div(inv.sum(axis=1), axis=0)
    return w.shift(1)


def sharpe_tilt_weights(ret_df: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Weight ∝ max(trailing Sharpe, 0), normalized. If all non-positive,
    fall back to equal weights. Causal."""
    mean = ret_df.rolling(window).mean()
    std = ret_df.rolling(window).std()
    sharpe = (mean / std).clip(lower=0.0)
    s = sharpe.sum(axis=1)
    w = sharpe.div(s, axis=0)
    n = ret_df.shape[1]
    # rows where all sharpe<=0 -> equal weight
    w = w.where(s > 0, other=1.0 / n)
    return w.shift(1)


def combine_portfolio(ret_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.Series:
    """Portfolio daily return = sum_i weight_i * ret_i (weights already causal)."""
    w = weights_df.reindex(ret_df.index).reindex(columns=ret_df.columns)
    return (ret_df * w).sum(axis=1)
