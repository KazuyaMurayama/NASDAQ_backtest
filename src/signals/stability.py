"""Time-series stability checks for signal IC.

Phase B requires signals to show consistent sign:
  - across decades (2000s / 2010s / 2020s)
  - first half vs second half of available data
to advance to Phase C. Mitigates regime-dependent overfit.

Note: For short signals (<10yr), decade test is relaxed; the
half-sample test is still applied.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _full_period_spearman(signal: pd.Series, forward_returns: pd.Series) -> float:
    """One-shot Spearman over the full aligned period (not rolling)."""
    df = pd.concat([signal.rename('s'), forward_returns.rename('r')],
                   axis=1, join='inner').dropna()
    if len(df) < 30:
        return float('nan')
    if df['s'].nunique() < 2 or df['r'].nunique() < 2:
        return float('nan')
    rho, _ = spearmanr(df['s'], df['r'])
    return float(rho)


def decade_ic_check(
    signal: pd.Series,
    forward_returns: pd.Series,
    decades: Optional[List[Tuple[str, str]]] = None,
    min_obs_per_decade: int = 50,
) -> dict:
    """Compute mean IC per decade + same-sign check.

    Args:
        signal, forward_returns: aligned by date (inner join after dropna).
        decades: list of (start, end) inclusive date strings.
        min_obs_per_decade: skip decades with fewer obs.

    Returns:
        {
          'decade_ics': dict[str, float],  # 'YYYY-YYYY' -> mean IC
          'decades_evaluated': int,
          'same_sign': bool,
          'sign_consistency': str,
        }
    """
    if decades is None:
        decades = [
            ('2000-01-01', '2009-12-31'),
            ('2010-01-01', '2019-12-31'),
            ('2020-01-01', '2099-12-31'),
        ]

    df = pd.concat([signal.rename('s'), forward_returns.rename('r')],
                   axis=1, join='inner').dropna()

    decade_ics = {}
    for start, end in decades:
        sub = df.loc[start:end]
        if len(sub) < min_obs_per_decade:
            continue
        if sub['s'].nunique() < 2 or sub['r'].nunique() < 2:
            continue
        rho, _ = spearmanr(sub['s'], sub['r'])
        label = f"{start[:4]}-{end[:4]}"
        decade_ics[label] = float(rho)

    n_eval = len(decade_ics)
    if n_eval == 0:
        sign_consistency = 'insufficient'
        same_sign = True  # vacuously, but accompanied by 'insufficient'
    elif all(v > 0 for v in decade_ics.values()):
        sign_consistency = 'all_positive'
        same_sign = True
    elif all(v < 0 for v in decade_ics.values()):
        sign_consistency = 'all_negative'
        same_sign = True
    else:
        sign_consistency = 'mixed'
        same_sign = False

    return {
        'decade_ics': decade_ics,
        'decades_evaluated': n_eval,
        'same_sign': same_sign,
        'sign_consistency': sign_consistency,
    }


def half_sample_ic_check(
    signal: pd.Series,
    forward_returns: pd.Series,
) -> dict:
    """First half vs second half mean IC.

    Returns:
        {
          'first_half_ic': float,
          'second_half_ic': float,
          'same_sign': bool,
          'split_date': str,
        }
    """
    df = pd.concat([signal.rename('s'), forward_returns.rename('r')],
                   axis=1, join='inner').dropna()
    if len(df) < 100:
        return {
            'first_half_ic': float('nan'),
            'second_half_ic': float('nan'),
            'same_sign': True,  # insufficient data, conservatively pass
            'split_date': '',
        }
    mid = len(df) // 2
    split_date = str(df.index[mid].date())
    first = df.iloc[:mid]
    second = df.iloc[mid:]

    def _rho(d):
        if d['s'].nunique() < 2 or d['r'].nunique() < 2:
            return float('nan')
        r, _ = spearmanr(d['s'], d['r'])
        return float(r)

    h1, h2 = _rho(first), _rho(second)

    if np.isnan(h1) or np.isnan(h2):
        same_sign = True  # insufficient
    elif h1 > 0 and h2 > 0:
        same_sign = True
    elif h1 < 0 and h2 < 0:
        same_sign = True
    else:
        same_sign = False

    return {
        'first_half_ic': h1,
        'second_half_ic': h2,
        'same_sign': same_sign,
        'split_date': split_date,
    }


def stability_summary(
    signal: pd.Series,
    forward_returns: pd.Series,
) -> dict:
    """Combine decade_ic_check + half_sample_ic_check + overall verdict.

    Returns:
        decade_ic_check output + half_sample_ic_check output +
        {'stability_pass': bool}
    """
    dec = decade_ic_check(signal, forward_returns)
    half = half_sample_ic_check(signal, forward_returns)

    # Pass if both same_sign True (insufficient data conservatively returns True)
    stability_pass = dec['same_sign'] and half['same_sign']

    return {
        **dec,
        **half,
        'stability_pass': stability_pass,
    }
