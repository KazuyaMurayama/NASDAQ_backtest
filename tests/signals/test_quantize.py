import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from signals.quantize import binary_threshold, quantile_cut, zscore_band


def test_binary_threshold_above():
    s = pd.Series([10, 20, 30, 40])
    out = binary_threshold(s, threshold=25, direction='above')
    assert list(out) == [0, 0, 1, 1]


def test_binary_threshold_below():
    s = pd.Series([10, 20, 30, 40])
    out = binary_threshold(s, threshold=25, direction='below')
    assert list(out) == [1, 1, 0, 0]


def test_quantile_cut_4levels():
    s = pd.Series(np.arange(100))
    out = quantile_cut(s, levels=4, window=None)
    assert set(out.dropna().unique()) <= {0, 1, 2, 3}
    counts = out.value_counts().sort_index()
    assert all(abs(c - 25) <= 1 for c in counts)


def test_quantile_cut_rolling_window():
    s = pd.Series(np.arange(252 * 3))
    out = quantile_cut(s, levels=4, window=252)
    assert out.iloc[:252].isna().all()
    assert not out.iloc[252:].isna().any()


def test_zscore_band_3levels():
    s = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 10, 0, -10])
    out = zscore_band(s, lower=-1.0, upper=1.0)
    assert set(out.dropna().unique()) <= {0, 1, 2}
    assert out.iloc[12] == 2
    assert out.iloc[14] == 0


def test_binary_threshold_raises_on_bad_direction():
    import pytest
    with pytest.raises(ValueError):
        binary_threshold(pd.Series([1, 2]), threshold=1, direction='sideways')
