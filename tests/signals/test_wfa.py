"""Tests for src/signals/wfa.py (Phase C C3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import numpy as np
import pandas as pd

from signals.wfa import (
    g1_static_split,
    g3_rolling_wfa,
    g7_bootstrap_oos,
    g8_year_contribution,
    g9_permutation,
    g10_param_sweep,
    run_g_series,
)


def _ramp_nav(n: int = 1000, rate: float = 0.0005, vol: float = 0.01, seed: int = 0) -> pd.Series:
    """Synthetic NAV: cumulative product of N(rate, vol) daily returns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2010-01-01', periods=n, freq='B')
    ret = rng.normal(rate, vol, n)
    return pd.Series(np.cumprod(1 + ret), index=idx)


def test_g1_split_returns_both_windows():
    nav = _ramp_nav(2000)
    out = g1_static_split(nav, '2014-01-01')
    assert 'is_cagr' in out and 'oos_cagr' in out
    assert not np.isnan(out['is_sharpe'])
    assert not np.isnan(out['oos_sharpe'])


def test_g3_rolling_wfa_returns_ci():
    nav = _ramp_nav(5000)
    out = g3_rolling_wfa(nav, n_windows=20)
    assert out['n_windows'] > 0
    assert not np.isnan(out['mean_sharpe'])
    assert out['ci95_lo'] <= out['mean_sharpe'] <= out['ci95_hi']


def test_g7_bootstrap_strong_candidate():
    cand = _ramp_nav(2000, rate=0.001, seed=1)
    bench = _ramp_nav(2000, rate=0.0001, seed=2)
    # Align dates
    bench.index = cand.index
    out = g7_bootstrap_oos(cand, bench, n_boot=200)
    assert out['p_cand_gt_bench'] > 0.8  # strong candidate dominates


def test_g8_year_contribution():
    nav = _ramp_nav(2500)
    out = g8_year_contribution(nav)
    assert out['n_years'] >= 8
    assert 0 <= out['max_year_share'] <= 1


def test_g9_permutation_returns_p():
    cand = _ramp_nav(1000, rate=0.001, seed=3)
    bench = _ramp_nav(1000, rate=0.0001, seed=4)
    bench.index = cand.index
    out = g9_permutation(cand, bench, n_perm=200)
    assert 0 <= out['p_value'] <= 1


def test_g10_param_sweep_all_pass():
    bench = _ramp_nav(1500, rate=0.0001, seed=10)
    sweep = {
        f'p{i}': _ramp_nav(1500, rate=0.001, seed=20 + i) for i in range(5)
    }
    # Align all to bench dates
    for k in sweep:
        sweep[k].index = bench.index
    out = g10_param_sweep(sweep, bench)
    assert out['n_total'] == 5
    assert out['all_pass'] is True


def test_g10_param_sweep_partial_fail():
    bench = _ramp_nav(1500, rate=0.001, seed=30)
    sweep = {
        'good': _ramp_nav(1500, rate=0.002, seed=31),
        'bad': _ramp_nav(1500, rate=0.0001, seed=32),
    }
    for k in sweep:
        sweep[k].index = bench.index
    out = g10_param_sweep(sweep, bench)
    assert out['all_pass'] is False
    assert out['n_better_than_bench'] == 1


def test_run_g_series_runs_subset():
    cand = _ramp_nav(2500, rate=0.001, seed=5)
    bench = _ramp_nav(2500, rate=0.0001, seed=6)
    bench.index = cand.index
    out = run_g_series(cand, bench, split_date='2014-01-01', series=['G1', 'G3', 'G8'])
    assert set(out.keys()) == {'G1', 'G3', 'G8'}
