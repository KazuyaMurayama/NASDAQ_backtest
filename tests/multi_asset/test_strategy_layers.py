import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
from multi_asset.strategy_layers import (
    vol_target_scale,
    vol_regime_gate,
    dual_ma_position,
    donchian_breakout_position,
    hysteresis,
    deadband,
    compose,
    rebalance_periodic,
    apply_exec_lag,
    trades_per_year,
)


def _ret(values, start='2010-01-04'):
    idx = pd.date_range(start, periods=len(values), freq='B')
    return pd.Series(values, index=idx, dtype=float)


def _price(values, start='2010-01-04'):
    idx = pd.date_range(start, periods=len(values), freq='B')
    return pd.Series(values, index=idx, dtype=float)


# ---- vol targeting ----
def test_vol_target_lower_when_vol_higher():
    calm = _ret([0.001, -0.001] * 40)            # low vol
    wild = _ret([0.05, -0.05] * 40)              # high vol
    s_calm = vol_target_scale(calm, target_vol=0.10, vol_window=10).dropna().iloc[-1]
    s_wild = vol_target_scale(wild, target_vol=0.10, vol_window=10).dropna().iloc[-1]
    assert s_calm > s_wild


def test_vol_target_capped_at_max_leverage():
    calm = _ret([0.0001, -0.0001] * 40)          # extremely low vol
    s = vol_target_scale(calm, target_vol=0.10, vol_window=10, max_leverage=1.0)
    assert s.dropna().max() <= 1.0 + 1e-9


def test_vol_target_is_causal():
    r = _ret(list(np.random.RandomState(0).normal(0, 0.01, 50)))
    p1 = vol_target_scale(r, vol_window=10)
    r2 = r.copy(); r2.iloc[-1] = 5.0
    p2 = vol_target_scale(r2, vol_window=10)
    assert p1.iloc[:-1].equals(p2.iloc[:-1])


# ---- vol regime gate ----
def test_vol_regime_gate_cuts_on_high_vol():
    r = _ret(list(np.random.RandomState(1).normal(0, 0.005, 300))
             + list(np.random.RandomState(2).normal(0, 0.05, 30)))
    g = vol_regime_gate(r, vol_window=20, z_window=120, z_thresh=1.0, gate_min=0.0)
    assert g.dropna().iloc[-1] == 0.0     # last = high vol → cut
    assert set(g.dropna().unique()) <= {0.0, 1.0}


# ---- dual MA ----
def test_dual_ma_holds_uptrend():
    p = _price(list(range(1, 80)))
    pos = dual_ma_position(p, n_short=5, n_long=20)
    assert pos.dropna().iloc[-1] == 1.0


def test_dual_ma_is_causal():
    p = _price(list(np.random.RandomState(3).normal(100, 5, 60)))
    a = dual_ma_position(p, 5, 20)
    p2 = p.copy(); p2.iloc[-1] = 999
    b = dual_ma_position(p2, 5, 20)
    assert a.iloc[:-1].equals(b.iloc[:-1])


# ---- donchian breakout ----
def test_donchian_enters_on_new_high_exits_on_new_low():
    # rise to new highs then fall to new lows
    p = _price(list(range(1, 40)) + list(range(39, 0, -1)))
    pos = donchian_breakout_position(p, entry_n=10, exit_n=5)
    vals = pos.dropna()
    assert vals.max() == 1.0 and vals.min() == 0.0


def test_donchian_is_causal():
    p = _price(list(np.random.RandomState(4).normal(100, 5, 80)))
    a = donchian_breakout_position(p, 10, 5)
    p2 = p.copy(); p2.iloc[-1] = 999
    b = donchian_breakout_position(p2, 10, 5)
    assert a.iloc[:-1].equals(b.iloc[:-1])


# ---- hysteresis ----
def test_hysteresis_holds_between_thresholds():
    raw = _price([0.0, 0.8, 0.5, 0.4, 0.2, 0.35, 0.9])
    pos = hysteresis(raw, enter=0.7, exit=0.3)
    # 0->0 (below enter), 0.8->1 (enter), 0.5->1 (between, hold), 0.4->1, 0.2->0 (exit), 0.35->0 (between, hold 0), 0.9->1
    assert list(pos.values) == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]


# ---- deadband ----
def test_deadband_suppresses_small_changes():
    raw = _price([0.0, 0.4, 0.45, 0.7, 0.72])
    pos = deadband(raw, eps=0.1)
    # 0; jump to 0.4 (>0.1); 0.45 within band (hold 0.4); 0.7 (>0.1 → 0.7); 0.72 hold
    assert list(pos.values) == [0.0, 0.4, 0.4, 0.7, 0.7]


# ---- compose ----
def test_compose_multiplies_aligned_layers():
    a = _price([1.0, 1.0, 0.0, 1.0])
    b = _price([0.5, 1.0, 1.0, 0.0])
    out = compose(a, b)
    assert list(out.values) == [0.5, 1.0, 0.0, 0.0]


# ---- realistic execution helpers ----
def test_rebalance_periodic_holds_between_rebalance_days():
    p = _price([0.0, 1.0, 0.2, 0.9, 0.3, 0.7])
    out = rebalance_periodic(p, every=2)
    # update at idx 0,2,4; hold in between
    assert list(out.values) == [0.0, 0.0, 0.2, 0.2, 0.3, 0.3]


def test_rebalance_periodic_cuts_trades():
    daily = _price(list(np.random.RandomState(5).uniform(0, 1, 252)))
    monthly = rebalance_periodic(daily, every=21)
    assert trades_per_year(monthly) < trades_per_year(daily)


def test_apply_exec_lag_shifts_and_is_causal():
    p = _price([0.0, 1.0, 1.0, 0.0, 1.0])
    out = apply_exec_lag(p, lag=2)
    assert out.iloc[2] == p.iloc[0] and out.iloc[4] == p.iloc[2]
    assert out.iloc[:2].isna().all()


def test_trades_per_year_counts_meaningful_changes():
    # binary position flipping 4 times across exactly one trading year
    vals = [0.0] * 63 + [1.0] * 63 + [0.0] * 63 + [1.0] * 63  # 252 days, 3 flips
    p = _price(vals)
    tpy = trades_per_year(p, min_delta=0.05)
    assert abs(tpy - 3.0) < 1e-6


def test_trades_per_year_ignores_tiny_drift():
    p = _price(list(np.linspace(0.0, 0.05, 252)))   # drifts < min_delta daily
    assert trades_per_year(p, min_delta=0.05) == 0.0
