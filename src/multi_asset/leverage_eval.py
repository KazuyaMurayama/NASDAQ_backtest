"""Phase 4 — net-of-cost-after-tax return engine for product/leverage choice.

Models holding a signal-driven position via either a leveraged ETF (TQQQ 3x /
2036 2x / TMF 3x; T+2) or a 1x fund (T+5, no SOFR/swap), at a chosen *effective*
leverage k. Lets us compare, per asset, which product and how much leverage is
optimal on net-cost-after-tax expected return / risk.

Daily mechanics (when signal position p, effective leverage k via product L):
  f = p * (k / L)                      # capital fraction in the product
  in_unit = L*r - (sofr_mult*SOFR + swap)/252 - TER/252
  daily   = f*in_unit + (1-f)*cash - per_trade_cost*|Δf|

Cost constants come from src/product_costs.py (single source of truth).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
TAX_FACTOR = 0.8273   # after-tax factor on positive yearly returns (20.315% x 85%)
PER_TRADE_COST = 0.0010


def strategy_net_returns(asset_ret: pd.Series, position: pd.Series,
                         cash_ret: pd.Series, sofr_annual: pd.Series,
                         target_leverage: float, product,
                         exec_lag: int = 2,
                         per_trade_cost: float = PER_TRADE_COST) -> pd.Series:
    """Net-of-cost (pre-tax) daily returns of holding `position` in `product`
    at effective leverage `target_leverage`."""
    idx = asset_ret.dropna().index
    r = asset_ret.reindex(idx).fillna(0.0)
    cash = cash_ret.reindex(idx).fillna(0.0)
    sofr = sofr_annual.reindex(idx).ffill().fillna(0.0)
    pos = position.reindex(idx).shift(exec_lag).clip(0.0, 1.0).fillna(0.0)

    L = product.leverage
    fin_daily = (product.sofr_multiplier * sofr + product.swap_spread) / TRADING_DAYS
    ter_daily = product.ter / TRADING_DAYS

    f = pos * (target_leverage / L)
    in_unit = L * r - fin_daily - ter_daily
    daily = f * in_unit + (1.0 - f) * cash
    tc = per_trade_cost * f.diff().abs().fillna(0.0)
    return daily - tc


def _cagr_simple(nav: pd.Series) -> float:
    n = nav.dropna()
    if len(n) < 2 or float(n.iloc[0]) <= 0:
        return float('nan')
    years = max((n.index[-1] - n.index[0]).days / 365.25, 1e-9)
    return float((n.iloc[-1] / n.iloc[0]) ** (1.0 / years) - 1.0)


def after_tax_cagr(daily_ret: pd.Series, tax_factor: float = TAX_FACTOR) -> float:
    """CAGR after the house tax model: positive yearly returns x tax_factor,
    negative years untouched, then chained."""
    r = daily_ret.dropna()
    if r.empty:
        return float('nan')
    yearly = (1.0 + r).groupby(r.index.year).prod() - 1.0
    taxed = yearly.where(yearly <= 0, yearly * tax_factor)
    nav_end = float((1.0 + taxed).prod())
    n_years = len(taxed)
    if n_years == 0 or nav_end <= 0:
        return float('nan')
    return nav_end ** (1.0 / n_years) - 1.0
