import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
from multi_asset.bond_signals import (
    ma_cross_position,
    momentum_position,
    zscore_position,
)


def _price(values, start='2010-01-04'):
    idx = pd.date_range(start, periods=len(values), freq='B')
    return pd.Series(values, index=idx, dtype=float)


def test_ma_cross_holds_when_above_mean():
    # steadily rising price → above its trailing mean → hold (1) once warmed up
    p = _price(list(range(1, 30)))
    pos = ma_cross_position(p, window=5)
    assert pos.dropna().iloc[-1] == 1.0


def test_ma_cross_is_causal():
    # changing a FUTURE price must not alter an earlier position
    p = _price([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    pos1 = ma_cross_position(p, window=3)
    p2 = p.copy()
    p2.iloc[-1] = 999.0  # perturb only the last day
    pos2 = ma_cross_position(p2, window=3)
    # all positions before the last day are identical
    assert pos1.iloc[:-1].equals(pos2.iloc[:-1])


def test_momentum_positive_lookback_holds():
    p = _price([1, 2, 3, 4, 5, 6, 7, 8])
    pos = momentum_position(p, lookback=3)
    assert pos.dropna().iloc[-1] == 1.0


def test_momentum_is_causal():
    p = _price([5, 4, 6, 7, 5, 8, 9, 6, 10, 11])
    pos1 = momentum_position(p, lookback=2)
    p2 = p.copy()
    p2.iloc[-1] = -100.0
    pos2 = momentum_position(p2, lookback=2)
    assert pos1.iloc[:-1].equals(pos2.iloc[:-1])


def test_zscore_responds_to_regime_shift():
    # noisy-low regime then higher regime → both hold(1) and cash(0) occur
    rng = np.random.RandomState(3)
    vals = list(rng.normal(0.0, 1.0, 60)) + list(rng.normal(5.0, 1.0, 30))
    s = _price(vals)
    pos = zscore_position(s, window=20, enter=1.0)
    assert set(pos.dropna().unique()) == {0.0, 1.0}


def test_zscore_invert_is_complement():
    # invert flips z>=enter into z<=enter, i.e. complementary positions
    rng = np.random.RandomState(4)
    s = _price(rng.normal(0.0, 1.0, 90).cumsum())
    pos = zscore_position(s, window=20, enter=0.5)
    pos_inv = zscore_position(s, window=20, enter=0.5, invert=True)
    both = pos.dropna().index.intersection(pos_inv.dropna().index)
    assert (pos.loc[both] + pos_inv.loc[both] == 1.0).all()


def test_zscore_is_causal():
    rng = np.random.RandomState(0)
    s = _price(rng.normal(0, 1, 60).cumsum())
    pos1 = zscore_position(s, window=10, enter=0.5)
    s2 = s.copy()
    s2.iloc[-1] = 500.0
    pos2 = zscore_position(s2, window=10, enter=0.5)
    assert pos1.iloc[:-1].equals(pos2.iloc[:-1])
