import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
import pytest
from signals.composite import build_composite, quantize_composite, COMPOSITE_BLOCKS


def _correlated_signals(seed=0, n=500, n_signals=3, correlation=0.7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    base = rng.normal(size=n)
    cols = {}
    for i in range(n_signals):
        noise = rng.normal(size=n)
        cols[f's{i}'] = correlation * base + np.sqrt(1 - correlation**2) * noise
    return pd.DataFrame(cols, index=idx)


def test_build_composite_yields_single_series():
    df = _correlated_signals(seed=1, n=500)
    out = build_composite(df)
    assert isinstance(out, pd.Series)
    assert out.name == 'composite'
    assert len(out) == 500


def test_composite_loads_positively_with_components():
    """Sign-normalized composite should correlate positively with sum of components."""
    df = _correlated_signals(seed=2, n=500)
    out = build_composite(df).dropna()
    summed = df.sum(axis=1).reindex(out.index)
    corr = np.corrcoef(out.values, summed.values)[0, 1]
    assert corr > 0


def test_composite_handles_anticorrelated_components():
    """If a component is negatively correlated with others, PCA still works."""
    rng = np.random.default_rng(3)
    n = 500
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    base = rng.normal(size=n)
    df = pd.DataFrame({
        's0': base + rng.normal(0, 0.1, n),
        's1': -base + rng.normal(0, 0.1, n),  # negatively correlated
        's2': base + rng.normal(0, 0.1, n),
    }, index=idx)
    out = build_composite(df)
    assert out.notna().sum() > 400


def test_single_column_returns_empty_or_fallback():
    """With <2 columns, PCA isn't meaningful — return empty series."""
    df = _correlated_signals(seed=4, n=100, n_signals=1)
    out = build_composite(df)
    assert out.dropna().empty


def test_quantize_composite_yields_0_to_3():
    df = _correlated_signals(seed=5, n=500)
    raw = build_composite(df)
    q = quantize_composite(raw, levels=4)
    assert set(q.dropna().unique()) <= {0, 1, 2, 3}


def test_unsupported_method_raises():
    df = _correlated_signals(seed=6, n=100)
    with pytest.raises(NotImplementedError):
        build_composite(df, method='ica')


def test_composite_blocks_has_4_entries():
    assert set(COMPOSITE_BLOCKS.keys()) == {'sentiment', 'credit_stress', 'macro_nowcast', 'yield_curve'}
    assert COMPOSITE_BLOCKS['sentiment'] == [12, 13, 14, 15]
