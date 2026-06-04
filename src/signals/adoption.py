"""Pareto-style adoption judgment for Phase C strategy variants.

Spec criteria (plan §6.3):
  Pareto judge: candidate must improve on >=2 axes AND not degrade >threshold on any axis.
  Hard requirements: G3 PASS (CI95_lo > 0) + G7 P > 0.90 + G9 p < 0.10 + (G11 SPA p < 0.10 batch-wise)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


_DEFAULT_THRESHOLDS = {
    'cagr_improve_min': 0.02,         # +2pp
    'cagr_degrade_max': -0.01,        # -1pp tolerated
    'sharpe_improve_min': 0.05,
    'sharpe_degrade_max': -0.05,
    'maxdd_improve_min': 0.05,         # -5pp DD improvement (absolute value reduction)
    'maxdd_degrade_max': -0.08,        # -8pp DD worsening tolerated
    'trades_yr_max': 200,
}


def pareto_judge(
    candidate_metrics: dict,
    baseline_metrics: dict,
    thresholds: dict | None = None,
) -> dict:
    """Compare candidate vs baseline on 4 metrics; return adoption verdict.

    Metrics expected (in both dicts):
      - cagr (decimal, e.g., 0.30)
      - sharpe (Sharpe ratio)
      - maxdd (negative decimal, e.g., -0.65)
      - trades_yr (int, optional; default 100)

    Returns:
        {
          'pareto_pass': bool,
          'improved_axes': list[str],
          'degraded_axes': list[str],
          'detail': {axis: {cand, base, diff, status}}
        }
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS

    detail = {}
    improved = []
    degraded = []

    # CAGR
    diff = candidate_metrics['cagr'] - baseline_metrics['cagr']
    if diff >= thresholds['cagr_improve_min']:
        improved.append('cagr')
        status = 'improved'
    elif diff < thresholds['cagr_degrade_max']:
        degraded.append('cagr')
        status = 'degraded_beyond'
    else:
        status = 'neutral'
    detail['cagr'] = {'cand': candidate_metrics['cagr'], 'base': baseline_metrics['cagr'], 'diff': diff, 'status': status}

    # Sharpe
    diff = candidate_metrics['sharpe'] - baseline_metrics['sharpe']
    if diff >= thresholds['sharpe_improve_min']:
        improved.append('sharpe')
        status = 'improved'
    elif diff < thresholds['sharpe_degrade_max']:
        degraded.append('sharpe')
        status = 'degraded_beyond'
    else:
        status = 'neutral'
    detail['sharpe'] = {'cand': candidate_metrics['sharpe'], 'base': baseline_metrics['sharpe'], 'diff': diff, 'status': status}

    # MaxDD: improvement = candidate has SMALLER absolute DD (less negative).
    # diff = candidate - baseline. For MaxDD which is negative, improvement = positive diff.
    diff = candidate_metrics['maxdd'] - baseline_metrics['maxdd']
    if diff >= thresholds['maxdd_improve_min']:
        improved.append('maxdd')
        status = 'improved'
    elif diff < thresholds['maxdd_degrade_max']:
        degraded.append('maxdd')
        status = 'degraded_beyond'
    else:
        status = 'neutral'
    detail['maxdd'] = {'cand': candidate_metrics['maxdd'], 'base': baseline_metrics['maxdd'], 'diff': diff, 'status': status}

    # Trades/yr: hard cap only
    trades = candidate_metrics.get('trades_yr', 0)
    detail['trades_yr'] = {'cand': trades, 'cap': thresholds['trades_yr_max']}

    # Pareto pass: >=2 improvements AND no degradation
    pareto_pass = (len(improved) >= 2) and (len(degraded) == 0) and (trades <= thresholds['trades_yr_max'])

    return {
        'pareto_pass': pareto_pass,
        'improved_axes': improved,
        'degraded_axes': degraded,
        'detail': detail,
    }


def hard_requirements_check(g_results: dict) -> dict:
    """Check Phase C hard requirements (G3/G7/G9 from wfa, optionally G11 SPA).

    Returns:
        {'pass': bool, 'failures': list[str]}
    """
    failures = []

    g3 = g_results.get('G3', {})
    if g3.get('ci95_lo', float('-inf')) <= 0 or g3.get('wfe', 0) < 1.0:
        failures.append(f"G3: ci95_lo={g3.get('ci95_lo'):.3f}, wfe={g3.get('wfe'):.3f}")

    g7 = g_results.get('G7', {})
    if g7.get('p_cand_gt_bench', 0) < 0.90:
        failures.append(f"G7: P(cand>bench)={g7.get('p_cand_gt_bench'):.3f}")

    g9 = g_results.get('G9', {})
    if g9.get('p_value', 1.0) >= 0.10:
        failures.append(f"G9: p={g9.get('p_value'):.3f}")

    return {'pass': len(failures) == 0, 'failures': failures}
