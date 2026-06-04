import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
import pytest
from signals.stability import decade_ic_check, half_sample_ic_check, stability_summary


def _persistent_signal(seed, n, ic_direction=+1):
    """Signal positively (or negatively) correlated with forward returns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2000-01-01', periods=n, freq='B')
    s = pd.Series(rng.integers(0, 4, size=n), index=idx)
    # Forward return = signal_value * direction * 0.005 + noise
    r = pd.Series(ic_direction * s.values * 0.005 + rng.normal(0, 0.01, size=n), index=idx)
    return s, r


def test_decade_check_persistent_positive_signal():
    """20+ years of consistent +IC signal: all decades positive."""
    s, r = _persistent_signal(seed=1, n=5500, ic_direction=+1)  # ~22yr -> covers 3 decades
    out = decade_ic_check(s, r)
    assert out['decades_evaluated'] >= 2
    assert out['sign_consistency'] == 'all_positive'
    assert out['same_sign'] is True


def test_decade_check_short_signal_marks_insufficient():
    """3 years (one decade only): decades_evaluated <= 1 -> still same_sign True."""
    s, r = _persistent_signal(seed=2, n=750, ic_direction=+1)  # ~3yr, all in 2000-2009
    out = decade_ic_check(s, r)
    assert out['decades_evaluated'] <= 1
    # With only 1 decade, all_positive (no mixing possible)
    assert out['same_sign'] is True


def test_decade_check_regime_flip_detected():
    """Signal flips sign between decades -> same_sign False."""
    n = 5500
    rng = np.random.default_rng(3)
    idx = pd.date_range('2000-01-01', periods=n, freq='B')
    s = pd.Series(rng.integers(0, 4, size=n), index=idx)
    # First decade: +IC, second/third: -IC
    direction = np.where(idx < pd.Timestamp('2010-01-01'), +1, -1)
    r = pd.Series(direction * s.values * 0.005 + rng.normal(0, 0.005, size=n), index=idx)
    out = decade_ic_check(s, r)
    assert out['same_sign'] is False
    assert out['sign_consistency'] == 'mixed'


def test_half_sample_persistent_signal():
    s, r = _persistent_signal(seed=4, n=2000, ic_direction=+1)
    out = half_sample_ic_check(s, r)
    assert out['first_half_ic'] > 0
    assert out['second_half_ic'] > 0
    assert out['same_sign'] is True


def test_half_sample_flip_detected():
    n = 2000
    rng = np.random.default_rng(5)
    idx = pd.date_range('2000-01-01', periods=n, freq='B')
    s = pd.Series(rng.integers(0, 4, size=n), index=idx)
    mid = n // 2
    direction = np.array([+1] * mid + [-1] * (n - mid))
    r = pd.Series(direction * s.values * 0.005 + rng.normal(0, 0.005, size=n), index=idx)
    out = half_sample_ic_check(s, r)
    assert out['same_sign'] is False


def test_half_sample_too_short_passes_conservatively():
    idx = pd.date_range('2020-01-01', periods=30, freq='B')
    s = pd.Series([0]*15 + [1]*15, index=idx)
    r = pd.Series([0.01]*30, index=idx)
    out = half_sample_ic_check(s, r)
    assert out['same_sign'] is True  # insufficient data -> conservative pass


def test_stability_summary_combines_both():
    s, r = _persistent_signal(seed=6, n=5500, ic_direction=+1)
    out = stability_summary(s, r)
    assert 'decade_ics' in out
    assert 'first_half_ic' in out
    assert out['stability_pass'] is True
