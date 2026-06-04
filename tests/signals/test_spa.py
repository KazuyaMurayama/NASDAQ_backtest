"""Tests for src/signals/spa_test.py (Phase C C4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import numpy as np
import pandas as pd

from signals.spa_test import run_spa


def test_spa_dominant_candidate_low_p():
    """A clearly better candidate should yield low SPA p-value."""
    rng = np.random.default_rng(0)
    n = 1000
    idx = pd.date_range('2015-01-01', periods=n, freq='B')
    bench_ret = pd.Series(rng.normal(0.0001, 0.01, n), index=idx)
    # Two candidates, one clearly better
    cand_df = pd.DataFrame({
        'good': rng.normal(0.001, 0.01, n),   # mean ~10x bench
        'bad': rng.normal(0.0, 0.01, n),
    }, index=idx)
    out = run_spa(bench_ret, cand_df, n_bootstrap=200)
    assert out['best_variant'] == 'good'
    assert out['spa_p_consistent'] < 0.10


def test_spa_no_better_candidate_high_p():
    """If no candidate beats bench, p should be high."""
    rng = np.random.default_rng(1)
    n = 1000
    idx = pd.date_range('2015-01-01', periods=n, freq='B')
    bench_ret = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
    # All candidates worse
    cand_df = pd.DataFrame({
        'a': rng.normal(0.0, 0.01, n),
        'b': rng.normal(0.0, 0.01, n),
    }, index=idx)
    out = run_spa(bench_ret, cand_df, n_bootstrap=200)
    assert out['spa_p_consistent'] > 0.20


def test_spa_returns_all_pvals():
    rng = np.random.default_rng(2)
    n = 800
    idx = pd.date_range('2015-01-01', periods=n, freq='B')
    bench_ret = pd.Series(rng.normal(0.0001, 0.01, n), index=idx)
    cand_df = pd.DataFrame({
        'c1': rng.normal(0.0005, 0.01, n),
        'c2': rng.normal(0.0002, 0.01, n),
    }, index=idx)
    out = run_spa(bench_ret, cand_df, n_bootstrap=200)
    for k in ('spa_p_lower', 'spa_p_consistent', 'spa_p_upper'):
        assert 0 <= out[k] <= 1
    assert out['best_variant'] in ('c1', 'c2')


def test_spa_short_series_returns_nan():
    """Series shorter than 5*block_size should return NaN gracefully."""
    n = 100  # < 60*5 = 300
    idx = pd.date_range('2015-01-01', periods=n, freq='B')
    bench_ret = pd.Series(np.random.randn(n) * 0.01, index=idx)
    cand_df = pd.DataFrame({'a': np.random.randn(n) * 0.01}, index=idx)
    out = run_spa(bench_ret, cand_df, n_bootstrap=100, block_size=60)
    assert np.isnan(out['spa_p_consistent'])
