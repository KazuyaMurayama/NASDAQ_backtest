"""2026-06-10 realistic cost constants. Source: PRODUCT_COST_COMPARISON_2026-06-10.md.
ANNUAL decimal rates. Daily = annual / 252."""
SOFR_2026 = 0.0363
MGR_SPREAD = 0.0040
R_USD_FINANCING = SOFR_2026 + MGR_SPREAD   # 4.03%
FX_HEDGE_COST = 0.029
CFD_SPREAD_ONE_WAY = 0.00028
CFD_OVERNIGHT_SPREAD = 0.030               # CFD金利 = SOFR + 3.0%
US_ETF_TRADE_CAP_JPY = 3190.0
PORTFOLIO_JPY = 30_000_000.0
JP_TAX = 0.20315
US_WHT_DIV = 0.10

def implicit_financing_annual(leverage: float) -> float:
    return (leverage - 1.0) * R_USD_FINANCING

def cfd_overnight_annual(leverage: float) -> float:
    return (SOFR_2026 + CFD_OVERNIGHT_SPREAD) * leverage


def cfd_overnight_daily(sofr_daily: float, leverage: float) -> float:
    """Daily CFD financing on full notional with time-varying SOFR. (sofr_daily + 3.0%/252) × L."""
    return (sofr_daily + CFD_OVERNIGHT_SPREAD / 252.0) * leverage

def us_etf_trade_cost_annual(trades_per_year: float, portfolio_jpy: float = PORTFOLIO_JPY) -> float:
    return (US_ETF_TRADE_CAP_JPY * trades_per_year) / portfolio_jpy


def cfd_overnight_daily_borrowed(sofr_daily: float, leverage: float) -> float:
    """Daily CFD financing on borrowed amount only (L-1)× notional.

    Sensitivity mode: finances only the borrowed portion (L-1), not full notional L.
    cost = (sofr_daily + CFD_OVERNIGHT_SPREAD/252) * max(L-1, 0)

    Contrast with cfd_overnight_daily which uses full notional L.
    """
    return (sofr_daily + CFD_OVERNIGHT_SPREAD / 252.0) * max(leverage - 1.0, 0.0)
