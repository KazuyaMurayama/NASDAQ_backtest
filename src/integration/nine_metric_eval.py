"""9+1 metric evaluation per docs/rules/08_evaluation-metrics.md.

For each candidate NAV vs baseline NAV, compute and diff:
  CAGR_OOS, IS-OOS gap, Sharpe_OOS, MaxDD,
  Worst10Y_CAGR, P10_5Y_CAGR.

Trades/yr, WFE, CI95_lo are NOT computed at this stage (they require
WFA execution, which is Session S3/S4 work). Phase B PASS judgment
relies on the 6 metrics above + degradation guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Primitive metric calculators
# ------------------------------------------------------------------

def _cagr(nav: pd.Series) -> float:
    n = nav.dropna()
    if len(n) < 2 or float(n.iloc[0]) <= 0:
        return float('nan')
    years = max((n.index[-1] - n.index[0]).days / 365.25, 1e-9)
    return float((n.iloc[-1] / n.iloc[0]) ** (1 / years) - 1)


def _sharpe(nav: pd.Series) -> float:
    r = nav.pct_change().dropna()
    if len(r) < 30 or float(r.std()) == 0:
        return float('nan')
    return float((r.mean() / r.std()) * np.sqrt(252))


def _maxdd(nav: pd.Series) -> float:
    n = nav.dropna()
    if len(n) < 2:
        return float('nan')
    return float((n / n.cummax() - 1).min())


def _worst_window_cagr(nav: pd.Series, years: int) -> float:
    """Worst rolling N-year CAGR (returns the minimum)."""
    n = nav.dropna()
    window = years * 252
    if len(n) < window + 1:
        return float('nan')
    rolling_cagr = (n / n.shift(window)) ** (1 / years) - 1
    return float(rolling_cagr.min())


def _pct_window_cagr(nav: pd.Series, years: int, pct: float = 0.10) -> float:
    """N-year rolling CAGR at given percentile (default P10)."""
    n = nav.dropna()
    window = years * 252
    if len(n) < window + 1:
        return float('nan')
    rolling_cagr = (n / n.shift(window)) ** (1 / years) - 1
    return float(rolling_cagr.quantile(pct))


def _is_oos_split(nav: pd.Series, split_date: str = '2018-01-01') -> tuple:
    return nav.loc[:split_date], nav.loc[split_date:]


# ------------------------------------------------------------------
# Public evaluator
# ------------------------------------------------------------------

def evaluate(
    candidate_nav: pd.Series,
    baseline_nav: pd.Series,
    split_date: str = '2018-01-01',
) -> dict:
    """Compute 6 axes (each: candidate, baseline, diff) → 18 metric fields."""
    cand_is, cand_oos = _is_oos_split(candidate_nav, split_date)
    base_is, base_oos = _is_oos_split(baseline_nav, split_date)

    m: dict = {}

    # CAGR_OOS
    m['cand_cagr_oos'] = _cagr(cand_oos)
    m['base_cagr_oos'] = _cagr(base_oos)
    m['cagr_oos_diff'] = m['cand_cagr_oos'] - m['base_cagr_oos']

    # IS-OOS gap (abs(IS) - abs(OOS) of CAGR for each, then diff)
    m['cand_cagr_is'] = _cagr(cand_is)
    m['base_cagr_is'] = _cagr(base_is)
    m['cand_is_oos_gap'] = m['cand_cagr_is'] - m['cand_cagr_oos']
    m['base_is_oos_gap'] = m['base_cagr_is'] - m['base_cagr_oos']
    m['is_oos_gap_diff'] = abs(m['cand_is_oos_gap']) - abs(m['base_is_oos_gap'])

    # Sharpe (OOS)
    m['cand_sharpe_oos'] = _sharpe(cand_oos)
    m['base_sharpe_oos'] = _sharpe(base_oos)
    m['sharpe_diff'] = m['cand_sharpe_oos'] - m['base_sharpe_oos']

    # MaxDD (full series)
    m['cand_maxdd'] = _maxdd(candidate_nav)
    m['base_maxdd'] = _maxdd(baseline_nav)
    # diff > 0 means candidate has less-negative DD (improvement)
    m['maxdd_diff'] = m['cand_maxdd'] - m['base_maxdd']

    # Worst 10-year CAGR
    m['cand_worst10y'] = _worst_window_cagr(candidate_nav, 10)
    m['base_worst10y'] = _worst_window_cagr(baseline_nav, 10)
    m['worst10y_diff'] = m['cand_worst10y'] - m['base_worst10y']

    # P10 5-year rolling CAGR
    m['cand_p10_5y'] = _pct_window_cagr(candidate_nav, 5, pct=0.10)
    m['base_p10_5y'] = _pct_window_cagr(baseline_nav, 5, pct=0.10)
    m['p10_5y_diff'] = m['cand_p10_5y'] - m['base_p10_5y']

    return m


# ------------------------------------------------------------------
# PASS / FAIL judgment per Plan §5
# ------------------------------------------------------------------

# Improvement thresholds (positive diff required)
IMP_THR = {
    'cagr_oos':     0.005,
    'sharpe':       0.03,
    'maxdd':        0.02,   # MaxDD diff positive = less-negative = better
    'worst10y':     0.005,
    'p10_5y':       0.005,
}
# Severe-degradation thresholds (any one triggers failure)
DEG_THR = {
    'cagr_oos':     -0.01,
    'sharpe':       -0.03,
    'maxdd':        -0.05,
    'worst10y':     -0.01,
    'p10_5y':       -0.01,
}
# IS-OOS gap: negative diff = candidate gap shrunk = improvement
GAP_IMP_THR = -0.005
GAP_DEG_THR = 0.015


def judge_improvement(metrics: dict) -> dict:
    """Apply Plan §5 thresholds.

    Returns dict with:
      improved_axes  : '|'-separated axes that improved
      degraded_axes  : '|'-separated axes that severely degraded
      n_improved     : int
      n_degraded     : int
      judgment       : 'STRONG_PASS' | 'STANDARD_PASS' | 'MARGINAL' | 'FAIL'
    """
    improvements: list = []
    degradations: list = []

    diff_map = {
        'cagr_oos':  metrics.get('cagr_oos_diff', 0.0),
        'sharpe':    metrics.get('sharpe_diff', 0.0),
        'maxdd':     metrics.get('maxdd_diff', 0.0),
        'worst10y':  metrics.get('worst10y_diff', 0.0),
        'p10_5y':    metrics.get('p10_5y_diff', 0.0),
    }
    for axis, diff in diff_map.items():
        if pd.isna(diff):
            continue
        if diff >= IMP_THR[axis]:
            improvements.append(axis)
        elif diff < DEG_THR[axis]:
            degradations.append(axis)

    gap_diff = metrics.get('is_oos_gap_diff', 0.0)
    if not pd.isna(gap_diff):
        if gap_diff <= GAP_IMP_THR:
            improvements.append('is_oos_gap')
        elif gap_diff > GAP_DEG_THR:
            degradations.append('is_oos_gap')

    n_imp = len(improvements)
    n_deg = len(degradations)
    has_severe_degradation = n_deg > 0

    if n_imp >= 4 and not has_severe_degradation:
        judgment = 'STRONG_PASS'
    elif n_imp >= 2 and not has_severe_degradation:
        judgment = 'STANDARD_PASS'
    elif n_imp >= 1:
        judgment = 'MARGINAL'
    else:
        judgment = 'FAIL'

    return {
        'improved_axes': '|'.join(improvements),
        'degraded_axes': '|'.join(degradations),
        'n_improved': n_imp,
        'n_degraded': n_deg,
        'judgment': judgment,
    }
