"""Hansen Superior Predictive Ability (SPA) test for multiple strategy variants.

Phase C (C4): SPA wrapper.

Wraps ``arch.bootstrap.SPA``. Used to honestly evaluate 'best-of-K' performance
across many candidate strategies (e.g., 90+ Phase B variants), correcting for
multiple-comparison bias in the maximum-of-K SR / mean-excess statistic.

API:
    run_spa(benchmark_returns, candidate_returns, n_bootstrap, block_size, seed) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_spa(
    benchmark_returns: pd.Series,
    candidate_returns: pd.DataFrame,
    n_bootstrap: int = 1000,
    block_size: int = 60,
    seed: int = 42,
) -> dict:
    """Hansen SPA test.

    H0: benchmark is at least as good as the best candidate.
    Low p-value → reject H0 → at least one candidate is genuinely better.

    Args:
        benchmark_returns: 1D Series of benchmark daily returns.
        candidate_returns: DataFrame of candidate daily returns
            (columns = variant names).
        n_bootstrap: # of stationary bootstrap samples.
        block_size: bootstrap block size (~quarter = 60).
        seed: RNG seed for reproducibility.

    Returns:
        {
          'spa_p_lower':       float,  # lower bound (most powerful, may be optimistic)
          'spa_p_consistent':  float,  # asymptotic consistent (recommended)
          'spa_p_upper':       float,  # upper bound (most conservative)
          'best_variant':      str,    # column with highest mean excess
          'best_mean_excess':  float,  # daily mean excess of best variant
        }
    """
    from arch.bootstrap import SPA

    df = pd.concat(
        [benchmark_returns.rename('bench')]
        + [candidate_returns[c].rename(c) for c in candidate_returns.columns],
        axis=1,
        join='inner',
    ).dropna()
    bench = df['bench'].values
    cand_cols = [c for c in df.columns if c != 'bench']
    cand_matrix = df[cand_cols].values  # n × K

    if len(bench) < block_size * 5:
        return {
            'spa_p_lower': float('nan'),
            'spa_p_consistent': float('nan'),
            'spa_p_upper': float('nan'),
            'best_variant': '',
            'best_mean_excess': float('nan'),
        }

    excess = cand_matrix - bench.reshape(-1, 1)
    best_idx = int(np.argmax(excess.mean(axis=0)))
    best_variant = cand_cols[best_idx]
    best_excess = float(excess[:, best_idx].mean())

    # arch.bootstrap.SPA expects LOSSES (lower = better).  For returns,
    # we negate so higher return → lower loss → "better model".  Without
    # this transform, the p-values reverse and the test loses power.
    spa = SPA(-bench, -cand_matrix, block_size=block_size, reps=n_bootstrap, seed=seed)
    spa.compute()

    pvals = spa.pvalues
    # arch returns a pd.Series indexed by ['lower','consistent','upper'].
    return {
        'spa_p_lower': float(pvals['lower']),
        'spa_p_consistent': float(pvals['consistent']),
        'spa_p_upper': float(pvals['upper']),
        'best_variant': best_variant,
        'best_mean_excess': best_excess,
    }
