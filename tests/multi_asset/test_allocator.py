import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
from multi_asset.allocator import (
    equal_weights,
    inverse_vol_weights,
    sharpe_tilt_weights,
    combine_portfolio,
)


def _rets(seed, n=400, mu=0.0003, sig=0.01, cols=('A', 'B', 'C')):
    rng = np.random.RandomState(seed)
    idx = pd.date_range('2015-01-01', periods=n, freq='B')
    return pd.DataFrame({c: rng.normal(mu, sig, n) for c in cols}, index=idx)


def test_equal_weights_sum_to_one_and_equal():
    df = _rets(1)
    w = equal_weights(df)
    assert np.allclose(w.iloc[-1].values, [1/3, 1/3, 1/3])
    assert np.allclose(w.sum(axis=1).dropna(), 1.0)


def test_inverse_vol_underweights_high_vol():
    df = _rets(2)
    df['B'] = df['B'] * 5.0   # B much more volatile
    w = inverse_vol_weights(df, window=60)
    last = w.dropna().iloc[-1]
    assert last['B'] < last['A'] and last['B'] < last['C']
    assert abs(last.sum() - 1.0) < 1e-9


def test_inverse_vol_is_causal():
    df = _rets(3)
    w1 = inverse_vol_weights(df, window=60)
    df2 = df.copy(); df2.iloc[-1] = 9.0
    w2 = inverse_vol_weights(df2, window=60)
    assert w1.iloc[:-1].equals(w2.iloc[:-1])


def test_sharpe_tilt_overweights_higher_sharpe():
    df = _rets(4)
    df['A'] = df['A'] + 0.001    # higher mean → higher Sharpe
    w = sharpe_tilt_weights(df, window=120)
    last = w.dropna().iloc[-1]
    assert last['A'] >= last['B'] and last['A'] >= last['C']
    assert abs(last.sum() - 1.0) < 1e-9


def test_sharpe_tilt_weights_nonnegative():
    df = _rets(5)
    w = sharpe_tilt_weights(df, window=120).dropna()
    assert (w.values >= -1e-12).all()


def test_combine_portfolio_matches_manual():
    df = _rets(6, n=100)
    w = equal_weights(df)               # causal already (constant); no extra shift
    port = combine_portfolio(df, w)
    manual = (df * (1/3)).sum(axis=1)
    assert np.allclose(port.values, manual.values, atol=1e-9)
