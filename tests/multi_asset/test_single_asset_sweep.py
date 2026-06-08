import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
import pytest
from multi_asset.single_asset_sweep import (
    build_holdcash_nav,
    buy_and_hold_nav,
    all_cash_nav,
    run_single_asset_sweep,
)


def _ret(values, start='2010-01-04'):
    idx = pd.date_range(start, periods=len(values), freq='B')
    return pd.Series(values, index=idx)


def test_holds_asset_when_position_one():
    asset = _ret([0.01] * 5)
    cash = _ret([0.0] * 5)
    pos = _ret([1, 1, 1, 1, 1])
    nav = build_holdcash_nav(asset, cash, pos)
    # fully invested → compounds asset return
    assert abs(nav.iloc[-1] - (1.01 ** 5)) < 1e-9


def test_uses_cash_when_position_zero():
    asset = _ret([0.01] * 5)
    cash = _ret([0.002] * 5)
    pos = _ret([0, 0, 0, 0, 0])
    nav = build_holdcash_nav(asset, cash, pos)
    assert abs(nav.iloc[-1] - (1.002 ** 5)) < 1e-9


def test_switches_between_asset_and_cash():
    asset = _ret([0.01, 0.01, 0.01, 0.01])
    cash = _ret([0.0, 0.0, 0.0, 0.0])
    pos = _ret([1, 0, 1, 0])
    nav = build_holdcash_nav(asset, cash, pos)
    # only days 1 and 3 earn 1%
    assert abs(nav.iloc[-1] - (1.01 * 1.0 * 1.01 * 1.0)) < 1e-9


def test_position_fillna_defaults_to_cash():
    asset = _ret([0.01] * 4)
    cash = _ret([0.0] * 4)
    pos = pd.Series([1.0, np.nan, 1.0, np.nan], index=asset.index)
    nav = build_holdcash_nav(asset, cash, pos)
    # NaN positions treated as cash (0)
    assert abs(nav.iloc[-1] - (1.01 * 1.0 * 1.01 * 1.0)) < 1e-9


def test_buy_and_hold_equals_full_position():
    asset = _ret([0.01, -0.02, 0.03, 0.01])
    cash = _ret([0.0] * 4)
    pos = _ret([1, 1, 1, 1])
    assert np.allclose(buy_and_hold_nav(asset).values,
                       build_holdcash_nav(asset, cash, pos).values)


def test_all_cash_compounds_cash():
    cash = _ret([0.001] * 6)
    nav = all_cash_nav(cash)
    assert abs(nav.iloc[-1] - (1.001 ** 6)) < 1e-9


def _long_series(seed, n=1600):
    rng = np.random.RandomState(seed)
    idx = pd.date_range('2015-01-01', periods=n, freq='B')
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


def test_sweep_returns_one_row_per_signal_with_metric_columns():
    asset = _long_series(1)
    cash = pd.Series(0.0, index=asset.index)
    signals = {
        'always_in': pd.Series(1.0, index=asset.index),
        'always_out': pd.Series(0.0, index=asset.index),
    }
    df = run_single_asset_sweep(asset, cash, signals, split_date='2018-01-01')
    assert len(df) == 2
    for col in ['signal', 'cand_cagr_oos', 'cand_sharpe_oos', 'cand_maxdd',
                'cand_worst10y', 'cand_p10_5y', 'cand_trades_yr',
                'cand_wfe', 'cand_ci95_lo', 'judgment']:
        assert col in df.columns
    assert set(df['signal']) == {'always_in', 'always_out'}


def test_sweep_sorted_by_cagr_oos_desc():
    asset = _long_series(2)
    cash = pd.Series(0.0, index=asset.index)
    signals = {
        'always_in': pd.Series(1.0, index=asset.index),
        'always_out': pd.Series(0.0, index=asset.index),
    }
    df = run_single_asset_sweep(asset, cash, signals, split_date='2018-01-01')
    vals = df['cand_cagr_oos'].dropna().tolist()
    assert vals == sorted(vals, reverse=True)
