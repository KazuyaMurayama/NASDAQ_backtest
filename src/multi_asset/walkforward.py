"""Generic Walk-Forward Analysis + stationary block bootstrap for hold/cash NAVs.

House-consistent definitions (cf. src/integration/phase_d_wfa.py /
phase_d_bootstrap.py):
  - WFA windows = calendar-year windows (like g14.generate_windows).
  - WFE = mean(per-window Sharpe) / full-sample Sharpe.
  - CI95_lo = t-distribution lower 95% bound of per-window CAGR.
  - Bootstrap = stationary block bootstrap, block=60 trading days,
    n_resamples=10000, paired draws for diff. PASS gate P(diff>0) > 0.90.

Applied across the FULL sample (every yearly window is a legitimate OOS
observation because the signals are fixed-rule, not refit per window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
BLOCK_SIZE = 60
N_BOOTSTRAP = 10000
RNG_SEED = 42


def yearly_windows(index: pd.DatetimeIndex, min_days: int = 60) -> list:
    """Return [(year, boolean mask ndarray)] for each year with >= min_days."""
    years = sorted(set(index.year))
    out = []
    for y in years:
        mask = (index.year == y).values if hasattr(index.year == y, 'values') \
            else np.asarray(index.year == y)
        if mask.sum() >= min_days:
            out.append((y, mask))
    return out


def _cagr_from_ret(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float('nan')
    end = float(np.prod(1.0 + r))
    if end <= 0:
        return float('nan')
    years = len(r) / TRADING_DAYS
    return end ** (1.0 / years) - 1.0


def _sharpe_from_ret(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if len(r) < 20:
        return float('nan')
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return float('nan')
    return float(np.mean(r) / sd * np.sqrt(TRADING_DAYS))


def wfa_stats(nav: pd.Series, baseline_nav: pd.Series | None = None,
              min_days: int = 60, min_vol: float = 0.02) -> dict:
    """Per-calendar-year WFA aggregate stats for a candidate NAV.

    min_vol: annualized-vol floor for including a window in the Sharpe-based
    WFE. Hold/cash strategies sit fully in cash for whole years; those
    near-riskless years have degenerate (huge) Sharpe that would corrupt WFE,
    so they are excluded from the WFE numerator (kept for CAGR/CI95).
    """
    ret = nav.pct_change().dropna()
    wins = yearly_windows(ret.index, min_days=min_days)
    cagrs, sharpes = [], []
    for year, _mask in wins:
        r = ret[ret.index.year == year].values
        cagrs.append(_cagr_from_ret(r))
        r_clean = r[~np.isnan(r)]
        win_vol = float(np.std(r_clean, ddof=1)) * np.sqrt(TRADING_DAYS) \
            if len(r_clean) >= 20 else 0.0
        if win_vol >= min_vol:
            sharpes.append(_sharpe_from_ret(r))
    cagrs = np.array([c for c in cagrs if not np.isnan(c)])
    sharpes_arr = np.array([s for s in sharpes if not np.isnan(s)])

    full_sharpe = _sharpe_from_ret(ret.values)
    wfe = (float(np.mean(sharpes_arr)) / full_sharpe
           if full_sharpe and not np.isnan(full_sharpe) and len(sharpes_arr)
           else float('nan'))

    n = len(cagrs)
    mean_cagr = float(np.mean(cagrs)) if n else float('nan')
    if n >= 3:
        se = float(np.std(cagrs, ddof=1)) / np.sqrt(n)
        ci95_lo = mean_cagr - stats.t.ppf(0.975, n - 1) * se
    else:
        ci95_lo = float('nan')

    pass_ci = bool(ci95_lo > 0) if not np.isnan(ci95_lo) else False
    pass_wfe = bool(0.5 <= wfe <= 2.0) if not np.isnan(wfe) else False
    return {
        'n_windows': n,
        'mean_window_cagr': mean_cagr,
        'mean_window_sharpe': float(np.mean(sharpes_arr)) if len(sharpes_arr) else float('nan'),
        'full_sharpe': full_sharpe,
        'wfe': wfe,
        'ci95_lo_cagr': ci95_lo,
        'pct_pos_windows': float(np.mean(cagrs > 0)) if n else float('nan'),
        'pass_ci': pass_ci,
        'pass_wfe': pass_wfe,
        'passed': pass_ci and pass_wfe,
    }


def _block_indices(n: int, block: int, rng: np.random.RandomState) -> np.ndarray:
    """Stationary-ish block bootstrap index vector of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.randint(0, n, n_blocks)
    idx = np.concatenate([np.arange(s, s + block) % n for s in starts])
    return idx[:n]


def block_bootstrap(ret: pd.Series, n_boot: int = N_BOOTSTRAP,
                    block: int = BLOCK_SIZE, seed: int = RNG_SEED) -> dict:
    """Distribution of CAGR under stationary block resampling of `ret`."""
    arr = ret.dropna().values
    n = len(arr)
    rng = np.random.RandomState(seed)
    cagrs = np.empty(n_boot)
    for i in range(n_boot):
        cagrs[i] = _cagr_from_ret(arr[_block_indices(n, block, rng)])
    return {
        'cagr_median': float(np.median(cagrs)),
        'cagr_lo': float(np.percentile(cagrs, 2.5)),
        'cagr_hi': float(np.percentile(cagrs, 97.5)),
        'p_pos': float(np.mean(cagrs > 0)),
    }


def paired_block_bootstrap(ret_a: pd.Series, ret_b: pd.Series,
                           n_boot: int = N_BOOTSTRAP, block: int = BLOCK_SIZE,
                           seed: int = RNG_SEED) -> dict:
    """Paired block bootstrap of CAGR(a) - CAGR(b) on the common index."""
    common = ret_a.dropna().index.intersection(ret_b.dropna().index)
    a = ret_a.reindex(common).values
    b = ret_b.reindex(common).values
    n = len(common)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(n, block, rng)
        diffs[i] = _cagr_from_ret(a[idx]) - _cagr_from_ret(b[idx])
    return {
        'diff_median': float(np.median(diffs)),
        'diff_lo': float(np.percentile(diffs, 2.5)),
        'diff_hi': float(np.percentile(diffs, 97.5)),
        'p_a_gt_b': float(np.mean(diffs > 0)),
    }
