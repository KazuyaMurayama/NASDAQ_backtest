"""
Turtle Trading System — position state machine.

TurtleState tracks discrete position state: units held, stop level,
pyramiding progress, and the S1 skip flag. It is market-agnostic
(caller supplies pre-slippage-adjusted prices and current equity).

Stop rule: after each pyramid add, ALL units share the same stop =
  last_entry_price - 2 * last_N  (long)
  last_entry_price + 2 * last_N  (short)
This is enforced via the stop_level property which always reads the last unit.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turtle_core import unit_size


@dataclass
class UnitEntry:
    date: object       # pd.Timestamp or str label
    price: float       # post-slippage entry price recorded at entry time
    n_value: float     # ATR frozen at entry (not updated daily)
    shares: float      # number of shares for this unit


class TurtleState:
    """
    Discrete position state for a single turtle market sleeve.

    Lifecycle:
      flat → long  via enter_long()
      long → long  via add_pyramid_unit()   (up to max_units)
      long → flat  via exit_all()
      flat → short via enter_short()        (T2 only)
      short→ flat  via exit_all()

    skip_s1: set externally by the simulation loop based on exit reason and P&L.
      True  → skip next S1 (20-day) breakout signal.
      False → take S1 signals normally.
    S2 (55-day) signals always enter regardless of skip_s1 (original turtle failsafe).
    """

    def __init__(self, max_units: int = 4):
        self.max_units: int = max_units
        self.units: list[UnitEntry] = []
        self.skip_s1: bool = False
        self.direction: str = 'flat'       # 'long', 'flat', 'short'
        self.entry_system: Optional[str] = None   # 'S1' or 'S2'

    # ------------------------------------------------------------------ #
    # Read-only properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def num_units(self) -> int:
        return len(self.units)

    @property
    def is_flat(self) -> bool:
        return len(self.units) == 0

    @property
    def stop_level(self) -> Optional[float]:
        """
        Current hard stop price.

        For long:  last_entry_price - 2 * last_N
        For short: last_entry_price + 2 * last_N
        After each pyramid add the stop rises (long) / falls (short)
        because last unit moves closer to current price.
        """
        if not self.units:
            return None
        last = self.units[-1]
        if self.direction == 'long':
            return last.price - 2.0 * last.n_value
        if self.direction == 'short':
            return last.price + 2.0 * last.n_value
        return None

    @property
    def next_pyramid_price(self) -> Optional[float]:
        """Price that triggers the next pyramid unit (0.5N from last entry)."""
        if not self.units or self.num_units >= self.max_units:
            return None
        last = self.units[-1]
        if self.direction == 'long':
            return last.price + 0.5 * last.n_value
        if self.direction == 'short':
            return last.price - 0.5 * last.n_value
        return None

    @property
    def total_shares(self) -> float:
        return sum(u.shares for u in self.units)

    @property
    def cost_basis(self) -> float:
        """Sum of (entry_price * shares) across all units."""
        return sum(u.price * u.shares for u in self.units)

    # ------------------------------------------------------------------ #
    # Mutating methods                                                     #
    # ------------------------------------------------------------------ #

    def enter_long(self, date: object, entry_price: float, n_value: float,
                   equity: float, system: str = 'S1') -> UnitEntry:
        """
        Open a new long position (first unit). Caller must ensure is_flat.
        entry_price should already include slippage (long_entry_price(signal)).
        """
        assert self.is_flat, "enter_long called with open position"
        shares = unit_size(equity, n_value)
        unit = UnitEntry(date, entry_price, n_value, shares)
        self.units = [unit]
        self.direction = 'long'
        self.entry_system = system
        return unit

    def enter_short(self, date: object, entry_price: float, n_value: float,
                    equity: float, system: str = 'S1') -> UnitEntry:
        """
        Open a new short position (T2 only). Caller must ensure is_flat.
        entry_price should already include slippage (short_entry_price(signal)).
        """
        assert self.is_flat, "enter_short called with open position"
        shares = unit_size(equity, n_value)
        unit = UnitEntry(date, entry_price, n_value, shares)
        self.units = [unit]
        self.direction = 'short'
        self.entry_system = system
        return unit

    def add_pyramid_unit(self, date: object, entry_price: float, n_value: float,
                         equity: float) -> Optional[UnitEntry]:
        """
        Add one pyramid unit if below max_units and direction is active.

        After adding, stop_level automatically reflects the new last unit
        (i.e., all prior units share the new, tighter stop — by design).
        Returns None if pyramid limit already reached.
        """
        if self.num_units >= self.max_units or self.direction == 'flat':
            return None
        shares = unit_size(equity, n_value)
        unit = UnitEntry(date, entry_price, n_value, shares)
        self.units.append(unit)
        return unit

    def exit_all(self, exit_price: float) -> tuple[float, float]:
        """
        Exit all units and return to flat.

        exit_price should include slippage (long_exit_price / short_exit_price).

        Returns:
            gross_pnl   : sum of (exit_price - entry_price) * shares
                          (positive = profit for long, negative = loss)
                          Note: for a short, profit when exit < entry.
            total_shares: shares liquidated (for trade log)
        """
        if not self.units:
            return 0.0, 0.0

        total_sh = self.total_shares

        if self.direction == 'long':
            gross_pnl = sum((exit_price - u.price) * u.shares for u in self.units)
        else:  # short
            gross_pnl = sum((u.price - exit_price) * u.shares for u in self.units)

        self.units = []
        self.direction = 'flat'
        self.entry_system = None
        return gross_pnl, total_sh

    def reset(self) -> None:
        """Force-reset to flat (used for forced liquidation or re-initialisation)."""
        self.units = []
        self.direction = 'flat'
        self.entry_system = None
        self.skip_s1 = False
