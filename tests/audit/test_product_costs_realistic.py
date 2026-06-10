from src.audit.product_costs_realistic_20260610 import (
    SOFR_2026, R_USD_FINANCING, FX_HEDGE_COST, cfd_overnight_annual,
    implicit_financing_annual, us_etf_trade_cost_annual, JP_TAX,
)

def test_sofr_and_financing():
    assert SOFR_2026 == 0.0363
    assert abs(R_USD_FINANCING - 0.0403) < 1e-9

def test_cfd_overnight_scales_with_leverage():
    assert abs(cfd_overnight_annual(3.0) - (0.0363 + 0.030) * 3.0) < 1e-9
    assert abs(cfd_overnight_annual(7.0) - (0.0363 + 0.030) * 7.0) < 1e-9

def test_implicit_financing():
    assert abs(implicit_financing_annual(3.0) - 2.0 * 0.0403) < 1e-9

def test_fx_hedge_and_tax():
    assert FX_HEDGE_COST == 0.029
    assert JP_TAX == 0.20315

def test_us_etf_trade_cost():
    # $22上限 ¥3190 × 20回 / ¥3000万 ≒ 0.2127%
    v = us_etf_trade_cost_annual(20.0)
    assert abs(v - (3190.0 * 20.0 / 30_000_000.0)) < 1e-9
