import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
import product_costs as pc
from multi_asset.leverage_eval import strategy_net_returns, after_tax_cagr


def _series(values, start='2010-01-04'):
    idx = pd.date_range(start, periods=len(values), freq='B')
    return pd.Series(values, index=idx, dtype=float)


def test_higher_leverage_higher_return_in_up_market():
    r = _series([0.001] * 60)           # steady up
    pos = _series([1.0] * 60)
    cash = _series([0.0] * 60)
    sofr = _series([0.0] * 60)          # no financing cost
    k1 = strategy_net_returns(r, pos, cash, sofr, 1.0, pc.TQQQ, exec_lag=0).sum()
    k3 = strategy_net_returns(r, pos, cash, sofr, 3.0, pc.TQQQ, exec_lag=0).sum()
    assert k3 > k1


def test_financing_cost_drags_when_sofr_high():
    r = _series([0.0] * 60)             # flat market
    pos = _series([1.0] * 60)
    cash = _series([0.0] * 60)
    low = strategy_net_returns(r, pos, cash, _series([0.0] * 60), 3.0, pc.TQQQ, exec_lag=0).sum()
    high = strategy_net_returns(r, pos, cash, _series([0.06] * 60), 3.0, pc.TQQQ, exec_lag=0).sum()
    assert high < low      # higher SOFR -> more financing drag -> lower return


def test_one_x_fund_cheaper_than_leveraged_at_k1_flat_market():
    r = _series([0.0] * 60)
    pos = _series([1.0] * 60)
    cash = _series([0.0] * 60)
    sofr = _series([0.04] * 60)
    fund = strategy_net_returns(r, pos, cash, sofr, 1.0, pc.NASDAQ1X, exec_lag=0).sum()
    lev = strategy_net_returns(r, pos, cash, sofr, 1.0, pc.TQQQ, exec_lag=0).sum()
    assert fund > lev      # fund: no SOFR/swap, lower TER


def test_exec_lag_is_causal():
    r = _series(list(np.random.RandomState(0).normal(0, 0.01, 40)))
    pos = _series([1.0] * 40)
    cash = _series([0.0] * 40)
    sofr = _series([0.03] * 40)
    a = strategy_net_returns(r, pos, cash, sofr, 2.0, pc.TQQQ, exec_lag=2)
    # first `lag` entries have zero invested fraction -> equal to cash (0)
    assert a.iloc[:2].abs().sum() < 1e-9


def test_after_tax_cagr_reduces_gains_not_losses():
    up = _series([0.001] * 252 * 3)       # all-positive years
    down = _series([-0.0005] * 252 * 3)   # all-negative years
    # tax reduces positive years
    assert after_tax_cagr(up) < after_tax_cagr(up, tax_factor=1.0)
    # losing years are untouched by tax (same with/without tax factor)
    assert abs(after_tax_cagr(down) - after_tax_cagr(down, tax_factor=1.0)) < 1e-12
