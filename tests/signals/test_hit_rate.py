import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
import pytest
from signals.hit_rate import hit_rate_conditional, base_rate, hit_rate_lift


def test_base_rate_balanced_returns_half():
    idx = pd.date_range('2020-01-01', periods=100, freq='B')
    r = pd.Series([0.01] * 50 + [-0.01] * 50, index=idx)
    br = base_rate(r)
    assert abs(br - 0.5) < 1e-9


def test_conditional_perfect_predictor():
    """signal=1 → always positive return; should give hit_rate=1.0."""
    idx = pd.date_range('2020-01-01', periods=200, freq='B')
    rng = np.random.default_rng(0)
    s = pd.Series(rng.integers(0, 2, size=200), index=idx)
    r = pd.Series(np.where(s.values == 1, 0.01, -0.01), index=idx)
    result = hit_rate_conditional(s, r, signal_value=1)
    assert result['hit_rate'] == 1.0
    assert result['wilson_lower_95'] > 0.9  # large n → tight CI


def test_conditional_no_signal_match_returns_zero_n():
    idx = pd.date_range('2020-01-01', periods=50, freq='B')
    s = pd.Series([0] * 50, index=idx)
    r = pd.Series([0.01] * 50, index=idx)
    result = hit_rate_conditional(s, r, signal_value=99)
    assert result['n_conditional'] == 0
    assert np.isnan(result['hit_rate'])


def test_lift_zero_for_random_signal():
    """Random signal should have lift near 0."""
    rng = np.random.default_rng(42)
    idx = pd.date_range('2020-01-01', periods=1000, freq='B')
    s = pd.Series(rng.integers(0, 4, size=1000), index=idx)
    r = pd.Series(rng.normal(0.0001, 0.01, size=1000), index=idx)
    result = hit_rate_lift(s, r, signal_value=2)
    # |lift| should be small for random signal at large n
    assert abs(result['lift_pp']) < 5


def test_negative_direction():
    """signal=1 → return always negative; with direction='negative' → hit_rate=1."""
    idx = pd.date_range('2020-01-01', periods=100, freq='B')
    s = pd.Series([1] * 100, index=idx)
    r = pd.Series([-0.01] * 100, index=idx)
    result = hit_rate_conditional(s, r, signal_value=1, direction='negative')
    assert result['hit_rate'] == 1.0


def test_invalid_direction_raises():
    s = pd.Series([1, 0], index=pd.date_range('2020-01-01', periods=2, freq='B'))
    r = pd.Series([0.01, -0.01], index=s.index)
    with pytest.raises(ValueError, match="positive"):
        hit_rate_conditional(s, r, signal_value=1, direction='sideways')


def test_wilson_lift_pp_calculation():
    """Verify wilson_lift_pp = (wilson_lower_95 - base_rate) * 100."""
    idx = pd.date_range('2020-01-01', periods=500, freq='B')
    s = pd.Series([1] * 500, index=idx)
    r = pd.Series([0.01] * 400 + [-0.01] * 100, index=idx)  # 80% positive
    result = hit_rate_lift(s, r, signal_value=1)
    assert result['hit_rate'] == 0.8
    assert abs(result['wilson_lift_pp'] - (result['wilson_lower_95'] - result['base_rate']) * 100) < 1e-9
