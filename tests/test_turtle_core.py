"""
Unit tests for turtle_core, turtle_state, turtle_costs.

Known-value checks derived from original turtle rule specifications:
  - Wilder SMMA: N_t = (19*N_{t-1} + TR_t) / 20  (period=20)
  - Stop level:  last_entry_price - 2 * last_N
  - Pyramid:     last_entry_price + 0.5 * last_N
"""
import os
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from turtle_core import wilder_atr, compute_donchian_high, compute_donchian_low, unit_size
from turtle_state import TurtleState, UnitEntry
from turtle_costs import (
    SLIPPAGE_RATE,
    long_entry_price, long_exit_price,
    short_entry_price, short_exit_price,
    tqqq_daily_holding_cost_rate, round_trip_slippage_cost,
)


# ================================================================== #
# turtle_core — wilder_atr                                           #
# ================================================================== #

class TestWilderAtr:
    def _flat_series(self, n, level=100.0, half_spread=1.0):
        """Constant price series: H = level+spread, L = level-spread, C = level."""
        c = np.full(n, level)
        h = c + half_spread
        lo = c - half_spread
        return h, lo, c

    def test_nan_below_period(self):
        h, lo, c = self._flat_series(25)
        atr = wilder_atr(h, lo, c, period=20)
        assert np.all(np.isnan(atr[:20]))

    def test_seed_value_constant_tr(self):
        # With H=C+1, L=C-1 and constant C: TR=max(2, 1, 1)=2 for all bars.
        # Seed at index 20 = mean(tr[1:21]) = 2.0
        h, lo, c = self._flat_series(30)
        atr = wilder_atr(h, lo, c, period=20)
        assert math.isclose(atr[20], 2.0, rel_tol=1e-9)

    def test_smma_stays_constant_for_constant_tr(self):
        h, lo, c = self._flat_series(50)
        atr = wilder_atr(h, lo, c, period=20)
        # All values from period onward should equal 2.0
        valid = atr[20:]
        assert np.allclose(valid, 2.0)

    def test_smma_single_step(self):
        # First 21 bars: H=L=C=100 (TR=0), then one bar with TR=20
        n = 30
        c  = np.full(n, 100.0)
        h  = np.full(n, 100.0)
        lo = np.full(n, 100.0)
        h[21]  = 110.0   # bar 21: H-L=10, but C_prev=100 → TR = max(10, 10, 10) = 10
        lo[21] = 90.0    # wait: H-C_prev=|110-100|=10, L-C_prev=|90-100|=10 → TR=10
        # Actually h[21]-lo[21] = 20, |h-c_prev|=10, |l-c_prev|=10 → TR = 20
        h[21]  = 120.0   # H-L=20, |H-C_prev|=20, |L-C_prev|=10 → TR=20
        lo[21] = 100.0
        atr = wilder_atr(h, lo, c, period=20)
        # seed atr[20] = mean(tr[1:21])
        # tr[1..20] = max(H-L=0, |H-C_prev|=0, |L-C_prev|=0) = 0 for all
        # tr[21] = max(120-100=20, |120-100|=20, |100-100|=0) = 20
        # atr[20] = 0.0  (mean of zeros)
        # atr[21] = (19/20)*0 + (1/20)*20 = 1.0
        assert math.isclose(atr[20], 0.0, abs_tol=1e-9)
        assert math.isclose(atr[21], 1.0, rel_tol=1e-9)

    def test_smma_convergence_toward_new_tr(self):
        # ATR starts at 1.0, then TR permanently jumps to 3.0 — ATR should converge to 3
        n = 200
        h, lo, c = self._flat_series(n, half_spread=0.5)   # TR=1
        h[21:] += 1.0   # after bar 21, H expands: TR=max(3,1.5,1.5)=3
        lo[21:] -= 1.0
        atr = wilder_atr(h, lo, c, period=20)
        # Should converge within 100 bars of the jump
        assert atr[150] > 2.9

    def test_returns_same_length_as_input(self):
        h, lo, c = self._flat_series(100)
        atr = wilder_atr(h, lo, c)
        assert len(atr) == 100

    def test_short_series_all_nan(self):
        h, lo, c = self._flat_series(5)
        atr = wilder_atr(h, lo, c, period=20)
        assert np.all(np.isnan(atr))


# ================================================================== #
# turtle_core — compute_donchian_high / low                          #
# ================================================================== #

class TestDonchian:
    def test_high_basic(self):
        highs = np.array([3.0, 1.0, 2.0, 5.0, 4.0, 6.0])
        result = compute_donchian_high(highs, window=3)
        assert np.all(np.isnan(result[:3]))
        assert result[3] == 3.0   # max(3,1,2)
        assert result[4] == 5.0   # max(1,2,5)
        assert result[5] == 5.0   # max(2,5,4)

    def test_low_basic(self):
        lows = np.array([10.0, 8.0, 9.0, 6.0, 7.0, 5.0])
        result = compute_donchian_low(lows, window=3)
        assert np.all(np.isnan(result[:3]))
        assert result[3] == 8.0   # min(10,8,9)
        assert result[4] == 6.0   # min(8,9,6)
        assert result[5] == 6.0   # min(9,6,7)

    def test_high_exclusive_of_current_bar(self):
        # Spike on bar 5 should NOT appear in result[5]; only in result[6+]
        highs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 99.0, 1.0, 1.0])
        result = compute_donchian_high(highs, window=3)
        assert result[5] == 1.0    # window [2,3,4] — spike not included
        assert result[6] == 99.0   # window [3,4,5] — spike included

    def test_length_preserved(self):
        prices = np.arange(50, dtype=float)
        assert len(compute_donchian_high(prices, 20)) == 50
        assert len(compute_donchian_low(prices, 10)) == 50


# ================================================================== #
# turtle_core — unit_size                                            #
# ================================================================== #

class TestUnitSize:
    def test_basic_formula(self):
        # Unit = equity * 0.01 / N
        equity = 100_000.0
        n = 5.0
        assert math.isclose(unit_size(equity, n), 200.0)

    def test_zero_n_returns_zero(self):
        assert unit_size(100_000.0, 0.0) == 0.0

    def test_negative_n_returns_zero(self):
        assert unit_size(100_000.0, -1.0) == 0.0

    def test_nan_n_returns_zero(self):
        assert unit_size(100_000.0, float('nan')) == 0.0

    def test_dollar_per_point_scaling(self):
        equity = 100_000.0
        n = 10.0
        # DPP=1: unit=100, DPP=2: unit=50
        assert math.isclose(unit_size(equity, n, 1.0), 100.0)
        assert math.isclose(unit_size(equity, n, 2.0), 50.0)

    def test_scales_with_equity(self):
        n = 5.0
        assert math.isclose(unit_size(200_000.0, n), 2 * unit_size(100_000.0, n))


# ================================================================== #
# turtle_state — TurtleState                                         #
# ================================================================== #

class TestTurtleState:
    def test_initial_state_is_flat(self):
        s = TurtleState()
        assert s.is_flat
        assert s.num_units == 0
        assert s.stop_level is None
        assert s.next_pyramid_price is None
        assert not s.skip_s1

    def test_enter_long_sets_direction(self):
        s = TurtleState()
        s.enter_long('2026-01-01', 100.0, 5.0, 100_000.0, system='S1')
        assert s.direction == 'long'
        assert s.num_units == 1
        assert not s.is_flat
        assert s.entry_system == 'S1'

    def test_stop_level_long(self):
        s = TurtleState()
        s.enter_long('2026-01-01', 100.0, 5.0, 100_000.0)
        # stop = 100 - 2*5 = 90
        assert math.isclose(s.stop_level, 90.0)

    def test_next_pyramid_price_long(self):
        s = TurtleState()
        s.enter_long('2026-01-01', 100.0, 5.0, 100_000.0)
        # next pyramid = 100 + 0.5*5 = 102.5
        assert math.isclose(s.next_pyramid_price, 102.5)

    def test_pyramid_raises_stop(self):
        s = TurtleState()
        s.enter_long('2026-01-01', 100.0, 5.0, 100_000.0)
        s.add_pyramid_unit('2026-01-05', 102.5, 5.0, 102_000.0)
        # stop = 102.5 - 2*5 = 92.5  (tighter stop after raise)
        assert math.isclose(s.stop_level, 92.5)
        assert s.num_units == 2

    def test_full_pyramid_four_units(self):
        s = TurtleState(max_units=4)
        s.enter_long('d1', 100.0, 5.0, 100_000.0)
        s.add_pyramid_unit('d2', 102.5, 5.0, 102_000.0)
        s.add_pyramid_unit('d3', 105.0, 5.0, 104_000.0)
        s.add_pyramid_unit('d4', 107.5, 5.0, 106_000.0)
        assert s.num_units == 4
        # stop = 107.5 - 2*5 = 97.5
        assert math.isclose(s.stop_level, 97.5)
        # next_pyramid_price is None (max reached)
        assert s.next_pyramid_price is None

    def test_add_unit_beyond_max_returns_none(self):
        s = TurtleState(max_units=2)
        s.enter_long('d1', 100.0, 5.0, 100_000.0)
        s.add_pyramid_unit('d2', 102.5, 5.0, 102_000.0)
        result = s.add_pyramid_unit('d3', 105.0, 5.0, 104_000.0)
        assert result is None
        assert s.num_units == 2

    def test_exit_all_profitable_long(self):
        s = TurtleState()
        s.enter_long('d1', 100.0, 5.0, 100_000.0)
        shares = s.total_shares   # (100000*0.01)/5 = 200
        pnl, sold = s.exit_all(110.0)
        # pnl = (110-100)*200 = 2000
        assert math.isclose(pnl, (110.0 - 100.0) * shares)
        assert math.isclose(sold, shares)
        assert s.is_flat

    def test_exit_all_loss_long(self):
        s = TurtleState()
        s.enter_long('d1', 100.0, 5.0, 100_000.0)
        shares = s.total_shares
        pnl, _ = s.exit_all(90.0)
        assert pnl < 0
        assert math.isclose(pnl, (90.0 - 100.0) * shares)

    def test_exit_resets_state(self):
        s = TurtleState()
        s.enter_long('d1', 100.0, 5.0, 100_000.0)
        s.exit_all(110.0)
        assert s.is_flat
        assert s.direction == 'flat'
        assert s.entry_system is None
        assert s.stop_level is None

    def test_enter_short_stop_level(self):
        s = TurtleState()
        s.enter_short('d1', 100.0, 5.0, 100_000.0)
        # stop = 100 + 2*5 = 110
        assert math.isclose(s.stop_level, 110.0)

    def test_enter_short_next_pyramid(self):
        s = TurtleState()
        s.enter_short('d1', 100.0, 5.0, 100_000.0)
        # next pyramid = 100 - 0.5*5 = 97.5
        assert math.isclose(s.next_pyramid_price, 97.5)

    def test_exit_short_profit(self):
        s = TurtleState()
        s.enter_short('d1', 100.0, 5.0, 100_000.0)
        shares = s.total_shares
        pnl, _ = s.exit_all(90.0)  # price fell — short wins
        # pnl = (entry - exit) * shares = (100-90)*shares
        assert pnl > 0
        assert math.isclose(pnl, (100.0 - 90.0) * shares)

    def test_skip_s1_is_external_flag(self):
        s = TurtleState()
        assert not s.skip_s1
        s.skip_s1 = True
        assert s.skip_s1
        s.skip_s1 = False
        assert not s.skip_s1

    def test_reset_clears_everything(self):
        s = TurtleState()
        s.enter_long('d1', 100.0, 5.0, 100_000.0)
        s.skip_s1 = True
        s.reset()
        assert s.is_flat
        assert not s.skip_s1
        assert s.direction == 'flat'

    def test_total_shares_multi_unit(self):
        s = TurtleState()
        # equity ~constant for simplicity; unit_size = equity*0.01/N
        s.enter_long('d1', 100.0, 10.0, 100_000.0)   # 100 shares
        s.add_pyramid_unit('d2', 105.0, 10.0, 100_000.0)  # 100 shares
        assert math.isclose(s.total_shares, 200.0)

    def test_enter_requires_flat(self):
        s = TurtleState()
        s.enter_long('d1', 100.0, 5.0, 100_000.0)
        with pytest.raises(AssertionError):
            s.enter_long('d2', 110.0, 5.0, 102_000.0)


# ================================================================== #
# turtle_costs                                                        #
# ================================================================== #

class TestTurtleCosts:
    def test_slippage_rate_value(self):
        assert math.isclose(SLIPPAGE_RATE, 0.003)

    def test_long_entry_price_above_signal(self):
        price = 100.0
        entry = long_entry_price(price)
        assert entry > price
        assert math.isclose(entry, 100.3)

    def test_long_exit_price_below_signal(self):
        price = 100.0
        ex = long_exit_price(price)
        assert ex < price
        assert math.isclose(ex, 99.7)

    def test_short_entry_price_below_signal(self):
        price = 100.0
        entry = short_entry_price(price)
        assert entry < price
        assert math.isclose(entry, 99.7)

    def test_short_exit_price_above_signal(self):
        price = 100.0
        ex = short_exit_price(price)
        assert ex > price
        assert math.isclose(ex, 100.3)

    def test_round_trip_slippage_2x_slippage_rate(self):
        val = 10_000.0
        assert math.isclose(round_trip_slippage_cost(val), val * 2 * SLIPPAGE_RATE)

    def test_tqqq_daily_holding_cost_positive(self):
        # At SOFR=5%: TER/252 + (2*0.05+0.005)/252 > 0
        rate = tqqq_daily_holding_cost_rate(sofr_annual=0.05)
        assert rate > 0
        # Sanity: roughly (0.86% + 10% + 0.5%) / 252 ≈ 0.045%/day
        assert rate < 0.001   # < 0.1%/day is a sane upper bound

    def test_tqqq_daily_holding_cost_scales_with_sofr(self):
        low_sofr  = tqqq_daily_holding_cost_rate(0.01)
        high_sofr = tqqq_daily_holding_cost_rate(0.10)
        assert high_sofr > low_sofr


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
