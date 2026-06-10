from src.audit.product_costs_realistic_20260610 import (
    SOFR_2026, R_USD_FINANCING, FX_HEDGE_COST, cfd_overnight_annual,
    cfd_overnight_daily, CFD_OVERNIGHT_SPREAD,
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


def test_cfd_overnight_daily_formula():
    # cfd_overnight_daily(sofr_daily, L) = (sofr_daily + 3.0%/252) * L
    sofr_d = 0.0363 / 252.0
    L = 3.0
    expected = (sofr_d + 0.030 / 252.0) * L
    result = cfd_overnight_daily(sofr_d, L)
    assert abs(result - expected) < 1e-12


def test_cfd_overnight_daily_full_notional():
    # フルNotional: L倍に課金 (L-1倍ではない)
    sofr_d = 0.05 / 252.0
    L = 5.0
    result_full = cfd_overnight_daily(sofr_d, L)          # full notional
    result_partial = cfd_overnight_daily(sofr_d, L - 1.0)  # (L-1) notional
    # full > partial for L > 1
    assert result_full > result_partial


def test_cfd_overnight_daily_scales_with_leverage():
    sofr_d = 0.04 / 252.0
    assert cfd_overnight_daily(sofr_d, 1.0) < cfd_overnight_daily(sofr_d, 2.0)
    assert cfd_overnight_daily(sofr_d, 2.0) < cfd_overnight_daily(sofr_d, 3.0)


def test_cfd_overnight_daily_uses_cfd_spread_constant():
    # CFD_OVERNIGHT_SPREAD/252 が上乗せされていることを確認
    sofr_d = 0.0
    L = 1.0
    # sofr_daily=0 の時: result = CFD_OVERNIGHT_SPREAD / 252.0 * L
    expected = CFD_OVERNIGHT_SPREAD / 252.0 * L
    assert abs(cfd_overnight_daily(sofr_d, L) - expected) < 1e-14
