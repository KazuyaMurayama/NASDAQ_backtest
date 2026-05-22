"""
Turtle Trading System — transaction costs and holding costs.

Slippage: 0.30 % per side (user decision B, 2026-05-18).
  Applied on: entry, every pyramid add, stop exit, Donchian exit.

Daily holding cost for TQQQ position:
  TER/252 + (2 * SOFR + swap_spread) / 252
  sourced from product_costs.py (same constants as existing backtests).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from product_costs import TQQQ, daily_financing_cost, daily_ter

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

SLIPPAGE_RATE: float = 0.0030   # 0.30 % per side — user-confirmed choice B

# ------------------------------------------------------------------ #
# Entry / exit price functions                                        #
# ------------------------------------------------------------------ #

def long_entry_price(signal_price: float) -> float:
    """Effective buy price for a long entry: pays slippage above signal."""
    return signal_price * (1.0 + SLIPPAGE_RATE)


def long_exit_price(signal_price: float) -> float:
    """Effective sell price for a long exit: receives slippage below signal."""
    return signal_price * (1.0 - SLIPPAGE_RATE)


def short_entry_price(signal_price: float) -> float:
    """Effective sell price for a short entry (sells below signal)."""
    return signal_price * (1.0 - SLIPPAGE_RATE)


def short_exit_price(signal_price: float) -> float:
    """Effective buy price to cover a short (pays above signal)."""
    return signal_price * (1.0 + SLIPPAGE_RATE)


# ------------------------------------------------------------------ #
# Holding cost                                                        #
# ------------------------------------------------------------------ #

def tqqq_daily_holding_cost_rate(sofr_annual: float) -> float:
    """
    Daily fractional cost rate for a TQQQ position.

    rate = TER/252 + (2*SOFR + swap_spread)/252
    Multiply by TQQQ position value to get dollar cost for that day.
    """
    return daily_ter(TQQQ) + daily_financing_cost(TQQQ, sofr_annual)


def deduct_tqqq_holding_cost(equity: float, tqqq_position_value: float,
                              sofr_annual: float) -> float:
    """
    Return equity after deducting one day's TQQQ holding cost.

    tqqq_position_value: shares * price at start of day (or prev close).
    """
    cost = tqqq_position_value * tqqq_daily_holding_cost_rate(sofr_annual)
    return equity - cost


# ------------------------------------------------------------------ #
# Slippage round-trip summary                                         #
# ------------------------------------------------------------------ #

def round_trip_slippage_cost(position_value: float) -> float:
    """Total slippage cost for a complete trade (entry + exit), in dollars."""
    return position_value * 2.0 * SLIPPAGE_RATE
