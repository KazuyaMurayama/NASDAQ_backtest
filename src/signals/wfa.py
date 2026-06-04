"""Simplified G1-G10 WFA protocol for signal-augmented strategies.

Phase C (C3): WFA wrapper.

The original Phase A plan calls for wrapping the existing ``src/audit/`` G-series
scripts (g20_*, g30_*, etc.), but those are research scripts with hardcoded paths
and ad-hoc CLI surfaces — they are not cleanly importable. For C3 scope, we
implement minimal, self-contained versions that operate on a strategy NAV series
and a benchmark NAV series.

NAV convention: pd.Series indexed by date, NAV[t0] = 1.0,
    NAV[t] / NAV[t-1] = 1 + daily_return.

Tests implemented:
    G1: Static IS/OOS split → Sharpe, CAGR, MaxDD per window.
    G3: Rolling N-window WFA → mean Sharpe + 95% CI + WFE (mean/full ratio).
    G7: Block bootstrap OOS → P(candidate CAGR > benchmark CAGR).
    G8: Year-by-year contribution → max single-year share / concentration check.
    G9: Permutation test → p-value of CAGR difference vs random shuffle.
    G10: Parameter-sweep robustness (caller passes 5-point sweep) → all-pass check.

NOTE on tradeoffs vs g20_* scripts:
    - The audit g20/g30 family uses fixed splice rules (TQQQ→QQQ→NDX), exact
      regime definitions (VZ-Gated, LT2-N750, E4), and the live cost model
      (src/product_costs.py). This module assumes the caller has already
      produced a final NAV series; we treat NAV as opaque.
    - G3 here uses non-overlapping windows for simplicity. The audit
      g20_wfa_rolling.py uses anchored/rolling overlapping windows with
      walk-forward refit. For signal-discovery sanity checks this is
      sufficient; for production WFA promotion of a new strategy, the
      audit scripts remain authoritative.
    - G7 uses a circular/wraparound block bootstrap on excess CAGR;
      audit g30_bootstrap.py uses stationary bootstrap on full IS-OOS.
    - G9 permutes pooled daily returns; audit g30_permutation.py preserves
      regime labels.

API:
    run_g_series(candidate_nav, benchmark_nav, split_date, series) -> dict
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal metric helpers
# ---------------------------------------------------------------------------
def _cagr(nav: pd.Series) -> float:
    """Compound annual growth rate from NAV[0] to NAV[-1]."""
    n = nav.dropna()
    if len(n) < 2:
        return float('nan')
    years = (n.index[-1] - n.index[0]).days / 365.25
    if years <= 0:
        return float('nan')
    base = n.iloc[-1] / n.iloc[0]
    if base <= 0:
        return float('nan')
    return float(base ** (1 / years) - 1)


def _sharpe(nav: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio from daily NAV. Returns NaN if <30 obs or zero vol."""
    r = nav.pct_change().dropna()
    if len(r) < 30 or r.std() == 0:
        return float('nan')
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))


def _maxdd(nav: pd.Series) -> float:
    """Maximum drawdown (negative number, e.g., -0.40 = -40%)."""
    n = nav.dropna()
    if len(n) < 2:
        return float('nan')
    peak = n.cummax()
    dd = n / peak - 1
    return float(dd.min())


# ---------------------------------------------------------------------------
# G1: Static IS/OOS split
# ---------------------------------------------------------------------------
def g1_static_split(nav: pd.Series, split_date: str) -> dict:
    """Static IS / OOS split metrics.

    IS window:  [start, split_date]
    OOS window: [split_date, end]
    """
    is_nav = nav.loc[:split_date]
    oos_nav = nav.loc[split_date:]
    return {
        'is_sharpe': _sharpe(is_nav),
        'is_cagr': _cagr(is_nav),
        'is_maxdd': _maxdd(is_nav),
        'oos_sharpe': _sharpe(oos_nav),
        'oos_cagr': _cagr(oos_nav),
        'oos_maxdd': _maxdd(oos_nav),
        'split_date': split_date,
    }


# ---------------------------------------------------------------------------
# G3: Rolling WFA
# ---------------------------------------------------------------------------
def g3_rolling_wfa(nav: pd.Series, n_windows: int = 50) -> dict:
    """Rolling WFA: split NAV into n_windows non-overlapping segments.

    Computes Sharpe per window; reports mean, 95% CI, and walk-forward
    efficiency (WFE = mean rolling Sharpe / full-sample Sharpe).
    """
    nav_clean = nav.dropna()
    n_obs = len(nav_clean)
    if n_obs < n_windows * 60:
        return {
            'mean_sharpe': float('nan'),
            'ci95_lo': float('nan'),
            'ci95_hi': float('nan'),
            'wfe': float('nan'),
            'n_windows': 0,
        }
    win_size = n_obs // n_windows
    sharpes = []
    for i in range(n_windows):
        start = i * win_size
        end = (i + 1) * win_size if i < n_windows - 1 else n_obs
        sharpes.append(_sharpe(nav_clean.iloc[start:end]))
    sharpes = [s for s in sharpes if not np.isnan(s)]
    if not sharpes:
        return {
            'mean_sharpe': float('nan'),
            'ci95_lo': float('nan'),
            'ci95_hi': float('nan'),
            'wfe': float('nan'),
            'n_windows': 0,
        }
    arr = np.array(sharpes)
    mean = float(arr.mean())
    sd = float(arr.std())
    ci95 = 1.96 * sd / np.sqrt(len(arr))
    full_sharpe = _sharpe(nav_clean)
    return {
        'mean_sharpe': mean,
        'ci95_lo': mean - ci95,
        'ci95_hi': mean + ci95,
        'wfe': float(mean / full_sharpe) if full_sharpe and not np.isnan(full_sharpe) else float('nan'),
        'n_windows': len(arr),
    }


# ---------------------------------------------------------------------------
# G7: Bootstrap OOS
# ---------------------------------------------------------------------------
def g7_bootstrap_oos(
    candidate_nav: pd.Series,
    benchmark_nav: pd.Series,
    n_boot: int = 1000,
    block_size: int = 60,
    seed: int = 42,
) -> dict:
    """Block bootstrap: probability candidate CAGR > benchmark CAGR.

    Uses circular block resampling on common dates. Block size defaults to
    60 trading days (~quarter); seed ensures reproducibility.
    """
    rng = np.random.default_rng(seed)
    c_ret = candidate_nav.pct_change().dropna()
    b_ret = benchmark_nav.pct_change().dropna()
    common = c_ret.index.intersection(b_ret.index)
    c_ret = c_ret.loc[common].values
    b_ret = b_ret.loc[common].values

    n = len(c_ret)
    if n < block_size * 5:
        return {'p_cand_gt_bench': float('nan'), 'n_boot': 0}

    n_blocks = n // block_size + 1
    wins = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        c_resample = np.concatenate([c_ret[s:s + block_size] for s in starts])[:n]
        b_resample = np.concatenate([b_ret[s:s + block_size] for s in starts])[:n]
        c_total = np.prod(1 + c_resample) - 1
        b_total = np.prod(1 + b_resample) - 1
        if c_total > b_total:
            wins += 1

    return {'p_cand_gt_bench': wins / n_boot, 'n_boot': n_boot}


# ---------------------------------------------------------------------------
# G8: Year contribution
# ---------------------------------------------------------------------------
def g8_year_contribution(nav: pd.Series) -> dict:
    """Year-by-year return contribution → maximum single-year share of |total|.

    concentration_pass = True iff max share < 35% (no single year dominates).
    """
    yearly = nav.resample('YE').last().pct_change().dropna()
    if len(yearly) < 2:
        return {'max_year_share': float('nan'), 'concentration_pass': False, 'n_years': 0}
    abs_total = yearly.abs().sum()
    if abs_total == 0:
        return {'max_year_share': float('nan'), 'concentration_pass': False, 'n_years': len(yearly)}
    max_share = float(yearly.abs().max() / abs_total)
    return {
        'max_year_share': max_share,
        'concentration_pass': max_share < 0.35,
        'n_years': len(yearly),
    }


# ---------------------------------------------------------------------------
# G9: Permutation test
# ---------------------------------------------------------------------------
def g9_permutation(
    candidate_nav: pd.Series,
    benchmark_nav: pd.Series,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test on the cumulative-return difference of candidate vs benchmark.

    Pool candidate + benchmark daily returns, shuffle, split into two halves of
    equal length, compute the cumulative return difference. p-value is the
    fraction of permutations whose difference is >= the observed difference.
    """
    rng = np.random.default_rng(seed)
    c_cagr_obs = _cagr(candidate_nav)
    b_cagr_obs = _cagr(benchmark_nav)
    obs_diff = float(c_cagr_obs - b_cagr_obs) if not (np.isnan(c_cagr_obs) or np.isnan(b_cagr_obs)) else float('nan')

    c_ret = candidate_nav.pct_change().dropna()
    b_ret = benchmark_nav.pct_change().dropna()
    common = c_ret.index.intersection(b_ret.index)
    pooled = np.concatenate([c_ret.loc[common].values, b_ret.loc[common].values])
    n_each = len(common)

    if n_each < 100:
        return {'p_value': float('nan'), 'n_perm': 0, 'obs_diff_cagr': obs_diff}

    # Observed pooled-cumulative-difference (matches the permutation statistic).
    c_total_obs = np.prod(1 + c_ret.loc[common].values) - 1
    b_total_obs = np.prod(1 + b_ret.loc[common].values) - 1
    obs_stat = c_total_obs - b_total_obs

    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        c_perm = np.prod(1 + shuffled[:n_each]) - 1
        b_perm = np.prod(1 + shuffled[n_each:n_each * 2]) - 1
        perm_diffs[i] = c_perm - b_perm

    p_val = float((perm_diffs >= obs_stat).mean())
    return {'p_value': p_val, 'n_perm': n_perm, 'obs_diff_cagr': obs_diff}


# ---------------------------------------------------------------------------
# G10: Parameter robustness
# ---------------------------------------------------------------------------
def g10_param_sweep(navs_by_param: dict[str, pd.Series], benchmark_nav: pd.Series) -> dict:
    """5-point parameter robustness: do all sweep points beat the benchmark CAGR?

    Args:
        navs_by_param: label → NAV series for each parameter setting.
        benchmark_nav: benchmark NAV.

    Returns:
        n_total, n_better_than_bench, all_pass, per-label CAGRs, bench CAGR.
    """
    bench_cagr = _cagr(benchmark_nav)
    results = {}
    n_better = 0
    for label, nav in navs_by_param.items():
        c = _cagr(nav)
        results[label] = c
        if pd.notna(c) and pd.notna(bench_cagr) and c > bench_cagr:
            n_better += 1
    n_total = len(navs_by_param)
    return {
        'n_total': n_total,
        'n_better_than_bench': n_better,
        'all_pass': n_better == n_total and n_total > 0,
        'param_cagrs': results,
        'bench_cagr': bench_cagr,
    }


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------
def run_g_series(
    candidate_nav: pd.Series,
    benchmark_nav: pd.Series,
    split_date: str = '2020-01-01',
    series: Optional[list[str]] = None,
) -> dict:
    """Run requested subset of G1-G10 tests; return dict keyed by test name.

    G10 requires a parameter sweep dict that this dispatcher cannot synthesize,
    so it is NOT included in the default series. Call g10_param_sweep directly.
    """
    if series is None:
        series = ['G1', 'G3', 'G7', 'G8', 'G9']
    out: dict = {}
    if 'G1' in series:
        out['G1'] = g1_static_split(candidate_nav, split_date)
    if 'G3' in series:
        out['G3'] = g3_rolling_wfa(candidate_nav, n_windows=50)
    if 'G7' in series:
        out['G7'] = g7_bootstrap_oos(candidate_nav, benchmark_nav, n_boot=500)
    if 'G8' in series:
        out['G8'] = g8_year_contribution(candidate_nav)
    if 'G9' in series:
        out['G9'] = g9_permutation(candidate_nav, benchmark_nav, n_perm=500)
    return out
