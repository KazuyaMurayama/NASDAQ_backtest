import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
import pytest
from signals.forward_returns import build_forward_returns, load_default_prices


def _ramp_prices(n=30):
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    p = pd.DataFrame({
        'A': np.linspace(100, 130, n),
        'B': np.linspace(50, 55, n),
    }, index=idx)
    return p


def test_5d_forward_return_known_values():
    p = _ramp_prices(20)
    fr = build_forward_returns(p, horizons=[5])
    expected_a_t0 = np.log(p['A'].iloc[5] / p['A'].iloc[0])
    assert abs(fr[('A', 5)].iloc[0] - expected_a_t0) < 1e-9


def test_horizons_yield_multiindex_columns():
    p = _ramp_prices(30)
    fr = build_forward_returns(p, horizons=[5, 20])
    assert fr.columns.names == ['asset', 'horizon']
    assert ('A', 5) in fr.columns
    assert ('B', 20) in fr.columns


def test_last_h_rows_are_nan():
    p = _ramp_prices(30)
    fr = build_forward_returns(p, horizons=[5])
    assert fr[('A', 5)].iloc[-5:].isna().all()
    assert not fr[('A', 5)].iloc[:-5].isna().any()


def test_rejects_nonpositive_prices():
    p = _ramp_prices(10)
    p.iloc[3, 0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        build_forward_returns(p)


def test_load_default_prices_returns_3_columns():
    """Smoke test against real repo data files."""
    p = load_default_prices()
    assert list(p.columns) == ['NDX', 'IEF', 'GLD']
    assert len(p) > 100  # should have plenty of overlap
    assert (p > 0).all().all()
