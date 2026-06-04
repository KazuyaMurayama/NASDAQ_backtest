"""Information Coefficient utilities for signal-vs-forward-return evaluation.

Spearman (rank) correlation is used to handle quantized signals (0/1 or 0/1/2/3)
and robust against non-normal forward return distributions.

Newey-West HAC t-stat handles overlapping rolling-window observations
(rolling 252d on daily IC creates serial correlation).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm


def compute_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    window: int = 252,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Rolling Spearman IC between signal and forward returns.

    Both series are aligned by date (inner join). Within each rolling window
    of `window` observations, compute Spearman rank correlation between
    signal[t-window+1..t] and forward_returns[t-window+1..t].

    Returns:
        Series indexed by date; first `window-1` values NaN.
    """
    if min_periods is None:
        min_periods = window

    df = pd.concat(
        [signal.rename('s'), forward_returns.rename('r')],
        axis=1,
        join='inner',
    ).dropna()
    if df.empty:
        return pd.Series(dtype='float64', index=signal.index, name='ic')

    s_vals = df['s'].values
    r_vals = df['r'].values
    n = len(df)
    out_idx = df.index
    out_vals = np.full(n, np.nan, dtype='float64')

    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        s_win = s_vals[start:i + 1]
        r_win = r_vals[start:i + 1]
        # Skip if either is constant (no rank variance)
        if len(set(s_win)) < 2 or len(set(r_win)) < 2:
            continue
        rho, _ = spearmanr(s_win, r_win)
        out_vals[i] = rho

    return pd.Series(out_vals, index=out_idx, name='ic').reindex(signal.index)


def ic_tstat_newey_west(ic_series: pd.Series, lags: int = 20) -> float:
    """Newey-West HAC t-stat of mean IC (regression on constant).

    Mean is constant regression with HAC standard errors (Newey-West kernel).

    Returns:
        scalar t-stat for H0: mean IC = 0.
    """
    ic_clean = ic_series.dropna()
    if len(ic_clean) < lags + 2:
        return float('nan')
    # If series has no variance, HAC SE is zero → t-stat undefined
    if ic_clean.std() == 0:
        return float('nan')
    X = np.ones((len(ic_clean), 1))
    y = ic_clean.values
    try:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        return float(model.tvalues[0])
    except Exception:
        return float('nan')


def ic_summary(
    signal: pd.Series,
    forward_returns: pd.Series,
    window: int = 252,
    lags: int = 20,
) -> dict:
    """Convenience: compute_ic + ic_tstat_newey_west + descriptive stats.

    Returns:
        {
          'mean_ic': float,
          'std_ic': float,
          't_stat': float,
          'n_obs': int,
          'ic_series': pd.Series,
        }
    """
    ic = compute_ic(signal, forward_returns, window=window)
    ic_clean = ic.dropna()
    return {
        'mean_ic': float(ic_clean.mean()) if len(ic_clean) else float('nan'),
        'std_ic': float(ic_clean.std()) if len(ic_clean) else float('nan'),
        't_stat': ic_tstat_newey_west(ic, lags=lags),
        'n_obs': int(len(ic_clean)),
        'ic_series': ic,
    }
