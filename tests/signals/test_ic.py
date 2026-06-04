import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
import pytest
from signals.ic import compute_ic, ic_tstat_newey_west, ic_summary


def _aligned(seed=0, n=500):
    """Two random series aligned by index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    s = pd.Series(rng.integers(0, 4, size=n), index=idx, name='signal')
    r = pd.Series(rng.normal(0, 0.01, size=n), index=idx, name='ret')
    return s, r


def test_compute_ic_random_signal_has_small_mean_ic():
    s, r = _aligned(seed=42, n=600)
    ic = compute_ic(s, r, window=252)
    ic_clean = ic.dropna()
    # Random signal: |mean IC| should be small, well under 0.1
    assert abs(ic_clean.mean()) < 0.1
    # First window-1 values NaN
    assert ic.iloc[:251].isna().all()
    assert not ic.iloc[251:].isna().all()  # at least some non-NaN


def test_compute_ic_perfect_signal_yields_ic_near_one():
    n = 500
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    # signal = rank of forward return (so monotone)
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0, 0.01, size=n), index=idx)
    s = r.rank()  # perfect rank → Spearman = 1
    ic = compute_ic(s, r, window=252)
    ic_clean = ic.dropna()
    assert ic_clean.mean() > 0.95


def test_compute_ic_constant_signal_returns_nan():
    n = 300
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    s = pd.Series(np.ones(n), index=idx)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, n), index=idx)
    ic = compute_ic(s, r, window=252)
    # All NaN because signal has no variance
    assert ic.dropna().empty


def test_newey_west_tstat_constant_zero_returns_nan_or_zero():
    n = 300
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    ic = pd.Series(np.zeros(n), index=idx)
    t = ic_tstat_newey_west(ic)
    # All zeros → no variance → t-stat undefined (NaN) or ~0
    assert np.isnan(t) or abs(t) < 1e-3


def test_newey_west_tstat_strong_positive_ic():
    n = 500
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    rng = np.random.default_rng(11)
    ic = pd.Series(rng.normal(0.1, 0.05, size=n), index=idx)
    t = ic_tstat_newey_west(ic, lags=20)
    # Mean 0.1, std 0.05, n=500 → very high t-stat
    assert t > 10


def test_ic_summary_returns_all_keys():
    s, r = _aligned(seed=0, n=500)
    out = ic_summary(s, r, window=252)
    assert set(out.keys()) == {'mean_ic', 'std_ic', 't_stat', 'n_obs', 'ic_series'}
    assert isinstance(out['ic_series'], pd.Series)
    assert out['n_obs'] >= 0
