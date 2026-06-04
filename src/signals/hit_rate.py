"""Conditional hit rate + Wilson CI for signal screening.

Compares P(forward_return > 0 | signal == k) against unconditional base rate.
Wilson CI is preferred over normal approximation for small n / extreme p.

Used in Phase B screening (plan §5.1): signals must show
  Wilson_lower_95 > base_rate + 3pp
to advance.
"""
from __future__ import annotations
from typing import Literal
import pandas as pd
import statsmodels.stats.proportion as smp


_VALID_DIRS = {'positive', 'negative'}


def _hit_mask(returns: pd.Series, direction: str) -> pd.Series:
    if direction == 'positive':
        return returns > 0
    if direction == 'negative':
        return returns < 0
    raise ValueError(f"direction must be 'positive'/'negative', got {direction}")


def hit_rate_conditional(
    signal: pd.Series,
    forward_returns: pd.Series,
    signal_value: int,
    direction: str = 'positive',
) -> dict:
    """Conditional hit rate when signal == signal_value.

    Returns:
        {
          'hit_rate': float,           # conditional probability
          'wilson_lower_95': float,    # Wilson CI lower bound
          'wilson_upper_95': float,
          'n_conditional': int,        # observations where signal == signal_value
          'n_hits': int,
        }
    """
    if direction not in _VALID_DIRS:
        raise ValueError(f"direction must be 'positive'/'negative', got {direction}")

    df = pd.concat([signal.rename('s'), forward_returns.rename('r')], axis=1, join='inner').dropna()
    sub = df[df['s'] == signal_value]
    n = len(sub)
    if n == 0:
        return {
            'hit_rate': float('nan'),
            'wilson_lower_95': float('nan'),
            'wilson_upper_95': float('nan'),
            'n_conditional': 0,
            'n_hits': 0,
        }
    hits = int(_hit_mask(sub['r'], direction).sum())
    p = hits / n
    lo, hi = smp.proportion_confint(hits, n, alpha=0.05, method='wilson')
    return {
        'hit_rate': float(p),
        'wilson_lower_95': float(lo),
        'wilson_upper_95': float(hi),
        'n_conditional': n,
        'n_hits': hits,
    }


def base_rate(forward_returns: pd.Series, direction: str = 'positive') -> float:
    """Unconditional P(forward_return > 0) — to compare against hit_rate_conditional."""
    if direction not in _VALID_DIRS:
        raise ValueError(f"direction must be 'positive'/'negative', got {direction}")
    clean = forward_returns.dropna()
    if len(clean) == 0:
        return float('nan')
    return float(_hit_mask(clean, direction).mean())


def hit_rate_lift(
    signal: pd.Series,
    forward_returns: pd.Series,
    signal_value: int,
    direction: str = 'positive',
) -> dict:
    """hit_rate_conditional + base_rate + lift (hit_rate - base_rate).

    Returns:
        Full dict from hit_rate_conditional PLUS:
        + 'base_rate': float,
        + 'lift_pp': float,    # (hit_rate - base_rate) * 100
        + 'wilson_lift_pp': float,  # (wilson_lower_95 - base_rate) * 100
    """
    cond = hit_rate_conditional(signal, forward_returns, signal_value, direction)
    base = base_rate(forward_returns, direction)
    return {
        **cond,
        'base_rate': base,
        'lift_pp': (cond['hit_rate'] - base) * 100 if cond['n_conditional'] else float('nan'),
        'wilson_lift_pp': (cond['wilson_lower_95'] - base) * 100 if cond['n_conditional'] else float('nan'),
    }
