"""Standalone single-asset hold-vs-cash sweep runner.

For one asset (NASDAQ / Gold / Bond), evaluate a set of timing signals that
each decide "hold the asset" vs "sit in cash", using the house 9-metric
standard (src/integration/nine_metric_eval.py).

A *signal* here is a position series in [0, 1] aligned to the asset's dates:
  1.0 = fully invested in the asset, 0.0 = fully in cash.
Callers are responsible for applying publication lag
(src/signals/timing.apply_publication_lag) before passing positions in.

See MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md, Phase 2.1.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from integration.nine_metric_eval import evaluate, judge_improvement

# House canonical OOS split (matches phase_d_wfa / phase_d_bootstrap).
CANONICAL_SPLIT = '2021-05-08'

# 9-metric candidate columns surfaced by the sweep (docs/rules/08).
_METRIC_COLS = [
    'cand_cagr_oos', 'cand_is_oos_gap', 'cand_sharpe_oos', 'cand_maxdd',
    'cand_worst10y', 'cand_p10_5y', 'cand_trades_yr', 'cand_wfe',
    'cand_ci95_lo',
]


def build_holdcash_nav(asset_ret: pd.Series,
                       cash_ret: pd.Series,
                       position: pd.Series) -> pd.Series:
    """Build a NAV from a hold-vs-cash position series.

    daily_return = position * asset_ret + (1 - position) * cash_ret

    position is reindexed to the asset's dates; missing/NaN positions are
    treated as cash (0.0) and positions are clipped to [0, 1].
    """
    idx = asset_ret.dropna().index
    a = asset_ret.reindex(idx).fillna(0.0)
    c = cash_ret.reindex(idx).fillna(0.0)
    p = position.reindex(idx).fillna(0.0).clip(0.0, 1.0)
    strat_ret = p * a + (1.0 - p) * c
    return (1.0 + strat_ret).cumprod()


def buy_and_hold_nav(asset_ret: pd.Series) -> pd.Series:
    """NAV of always-invested baseline."""
    idx = asset_ret.dropna().index
    return (1.0 + asset_ret.reindex(idx).fillna(0.0)).cumprod()


def all_cash_nav(cash_ret: pd.Series) -> pd.Series:
    """NAV of always-in-cash baseline."""
    idx = cash_ret.dropna().index
    return (1.0 + cash_ret.reindex(idx).fillna(0.0)).cumprod()


def run_single_asset_sweep(asset_ret: pd.Series,
                           cash_ret: pd.Series,
                           signals: Mapping[str, pd.Series],
                           split_date: str = CANONICAL_SPLIT,
                           baseline: str = 'bh') -> pd.DataFrame:
    """Evaluate each signal's hold-vs-cash NAV vs a baseline.

    Args:
        asset_ret: daily simple returns of the asset.
        cash_ret: daily simple returns of the cash/risk-free leg.
        signals: name -> position series (0..1), pre-lagged by caller.
        split_date: IS/OOS boundary (default: house canonical 2021-05-08).
        baseline: 'bh' (buy & hold) or 'cash' (all-cash) comparison.

    Returns:
        DataFrame, one row per signal, with the 9 candidate metrics plus the
        judge_improvement verdict, sorted by cand_cagr_oos descending.
    """
    if baseline == 'bh':
        base_nav = buy_and_hold_nav(asset_ret)
    elif baseline == 'cash':
        base_nav = all_cash_nav(cash_ret)
    else:
        raise ValueError(f"baseline must be 'bh' or 'cash', got {baseline!r}")

    rows = []
    for name, position in signals.items():
        nav = build_holdcash_nav(asset_ret, cash_ret, position)
        m = evaluate(nav, base_nav, split_date=split_date)
        verdict = judge_improvement(m)
        row = {'signal': name}
        row.update({k: m.get(k) for k in _METRIC_COLS})
        row['base_cagr_oos'] = m.get('base_cagr_oos')
        row['judgment'] = verdict.get('judgment')
        row['n_improved'] = verdict.get('n_improved')
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values('cand_cagr_oos', ascending=False,
                          na_position='last').reset_index(drop=True)
