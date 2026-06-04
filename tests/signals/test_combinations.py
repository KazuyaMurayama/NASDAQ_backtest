import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pandas as pd
import pytest
from signals.combinations import combine_signals


def test_combine_and():
    idx = pd.date_range('2024-01-01', periods=4, freq='B')
    s1 = pd.Series([0, 1, 1, 0], index=idx)
    s2 = pd.Series([0, 0, 1, 1], index=idx)
    out = combine_signals(s1, s2, operator='AND')
    assert list(out.values) == [0, 0, 1, 0]


def test_combine_or():
    idx = pd.date_range('2024-01-01', periods=4, freq='B')
    s1 = pd.Series([0, 1, 1, 0], index=idx)
    s2 = pd.Series([0, 0, 1, 1], index=idx)
    out = combine_signals(s1, s2, operator='OR')
    assert list(out.values) == [0, 1, 1, 1]


def test_combine_threshold_levels():
    idx = pd.date_range('2024-01-01', periods=4, freq='B')
    s1 = pd.Series([0, 1, 2, 3], index=idx)
    s2 = pd.Series([3, 2, 1, 0], index=idx)
    out = combine_signals(s1, s2, operator='AND', threshold1=2, threshold2=2)
    assert list(out.values) == [0, 0, 0, 0]


def test_invalid_operator_raises():
    s1 = pd.Series([0, 1])
    s2 = pd.Series([1, 0])
    with pytest.raises(ValueError):
        combine_signals(s1, s2, operator='XOR')
