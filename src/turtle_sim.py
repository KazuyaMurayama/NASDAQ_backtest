"""
Turtle Trading System — daily simulator (T1 long-only, T2 long/short).

Daily order of operations per bar t (after warm-up):
  1. Apply holding cost on TQQQ position (prev close basis) + MTM gain
  2. Check exits: 2N stop (vs intraday low) > Donchian exit (10d/20d low vs close)
  3. Pyramid adds: while high[t] >= next_pyramid_price and num_units < max
  4. New entry (if flat): S2 (close > H55) wins over S1 (close > H20) — S2 failsafe
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turtle_core import (
    wilder_atr, compute_donchian_high, compute_donchian_low, unit_size,
)
from turtle_state import TurtleState
from turtle_costs import (
    long_entry_price, long_exit_price,
    short_entry_price, short_exit_price,
    tqqq_daily_holding_cost_rate,
)


@dataclass
class TurtleSimResult:
    equity_series: pd.Series
    trade_log: pd.DataFrame
    daily_state: pd.DataFrame
    n_entries: int = 0
    n_2n_stops: int = 0
    n_donchian_exits_s1: int = 0
    n_donchian_exits_s2: int = 0
    n_s1_skips: int = 0
    n_pyramid_adds: int = 0
    n_short_entries: int = 0


def _precompute_signals(ohlc: pd.DataFrame, s1_period: int, s2_period: int,
                        s1_exit_period: int, s2_exit_period: int,
                        atr_period: int):
    h = ohlc['High'].values
    lo = ohlc['Low'].values
    c = ohlc['Close'].values
    atr = wilder_atr(h, lo, c, period=atr_period)
    h20 = compute_donchian_high(h, s1_period)
    h55 = compute_donchian_high(h, s2_period)
    l10 = compute_donchian_low(lo, s1_exit_period)
    l20 = compute_donchian_low(lo, s2_exit_period)
    return atr, h20, h55, l10, l20


def simulate_turtle_long_only(
    ohlc: pd.DataFrame,
    sofr_annual: pd.Series,
    initial_equity: float = 100_000.0,
    s1_period: int = 20,
    s2_period: int = 55,
    s1_exit_period: int = 10,
    s2_exit_period: int = 20,
    atr_period: int = 20,
    max_units: int = 4,
) -> TurtleSimResult:
    """
    T1: Pure Turtle Long-Only on a single OHLC series (e.g. TQQQ synthetic).
    """
    atr, h20, h55, l10, l20 = _precompute_signals(
        ohlc, s1_period, s2_period, s1_exit_period, s2_exit_period, atr_period)

    dates = ohlc.index
    open_ = ohlc['Open'].values
    high  = ohlc['High'].values
    low   = ohlc['Low'].values
    close = ohlc['Close'].values
    sofr  = sofr_annual.reindex(dates).ffill().bfill().values

    warm_up = max(s2_period, atr_period) + 1

    state  = TurtleState(max_units=max_units)
    equity = initial_equity
    equity_arr = np.full(len(dates), np.nan)
    equity_arr[:warm_up] = initial_equity

    trade_log: list[dict] = []
    daily_state: list[dict] = []

    cur_trade: Optional[dict] = None
    counters = dict(n_entries=0, n_2n_stops=0,
                    n_donchian_exits_s1=0, n_donchian_exits_s2=0,
                    n_s1_skips=0, n_pyramid_adds=0)

    def _start_trade(date, system, price, n_val):
        return dict(entry_date=date, system=system, entry_price=price,
                    n_at_entry=n_val, pyramid_adds=0, exit_date=None,
                    exit_price=None, exit_reason=None, gross_pnl=None,
                    holding_days=None)

    def _close_trade(t_dict, date, price, reason, pnl, holding_days):
        t_dict.update(exit_date=date, exit_price=price, exit_reason=reason,
                      gross_pnl=pnl, holding_days=holding_days)

    for t in range(warm_up, len(dates)):
        date_t = dates[t]
        c_t = close[t]; h_t = high[t]; l_t = low[t]
        n_t = atr[t]
        if not np.isfinite(n_t) or n_t <= 0:
            equity_arr[t] = equity
            continue

        # 1. holding cost + MTM
        if not state.is_flat:
            tqqq_val_prev = state.total_shares * close[t-1]
            cost = tqqq_val_prev * tqqq_daily_holding_cost_rate(sofr[t])
            equity -= cost
            equity += state.total_shares * (c_t - close[t-1])

        # 2. exit checks
        exited_today = False
        if not state.is_flat and state.direction == 'long':
            stop_lvl = state.stop_level
            if l_t <= stop_lvl:
                exit_signal = stop_lvl
                exit_px = long_exit_price(exit_signal)
                # Adjust equity: MTM assumed close, actual fill at exit_px
                equity += (exit_px - c_t) * state.total_shares
                pnl, _ = state.exit_all(exit_px)
                state.skip_s1 = False
                exited_today = True
                counters['n_2n_stops'] += 1
                if cur_trade is not None:
                    holding = (date_t - cur_trade['entry_date']).days
                    _close_trade(cur_trade, date_t, exit_px, '2N_stop', pnl, holding)
                    trade_log.append(cur_trade); cur_trade = None
            else:
                l_exit = l10[t] if state.entry_system == 'S1' else l20[t]
                if np.isfinite(l_exit) and c_t < l_exit:
                    exit_signal = l_exit
                    exit_px = long_exit_price(exit_signal)
                    equity += (exit_px - c_t) * state.total_shares
                    pnl, _ = state.exit_all(exit_px)
                    reason = '10d_exit' if state.entry_system == 'S1' else '20d_exit'
                    # NOTE: state.entry_system already reset by exit_all → use cur_trade
                    if cur_trade is not None:
                        reason = '10d_exit' if cur_trade['system'] == 'S1' else '20d_exit'
                        if cur_trade['system'] == 'S1':
                            counters['n_donchian_exits_s1'] += 1
                            state.skip_s1 = (pnl > 0)
                        else:
                            counters['n_donchian_exits_s2'] += 1
                            state.skip_s1 = False
                        holding = (date_t - cur_trade['entry_date']).days
                        _close_trade(cur_trade, date_t, exit_px, reason, pnl, holding)
                        trade_log.append(cur_trade); cur_trade = None
                    exited_today = True

        # 3. pyramid adds (allow multiple in one bar if high pierced several levels)
        if not state.is_flat and state.direction == 'long':
            safety = 0
            while (state.num_units < max_units
                   and state.next_pyramid_price is not None
                   and h_t >= state.next_pyramid_price
                   and safety < max_units):
                pyr_signal = state.next_pyramid_price
                entry_px = long_entry_price(pyr_signal)
                new_sh = unit_size(equity, n_t)
                if new_sh <= 0:
                    break
                # slippage paid on the new unit only
                equity -= (entry_px - pyr_signal) * new_sh
                state.add_pyramid_unit(date_t, entry_px, n_t, equity)
                counters['n_pyramid_adds'] += 1
                if cur_trade is not None:
                    cur_trade['pyramid_adds'] += 1
                safety += 1

        # 4. new entry (only if flat AND not exited today to avoid same-bar round-trip)
        if state.is_flat and not exited_today:
            h55_t = h55[t]; h20_t = h20[t]
            if np.isfinite(h55_t) and c_t > h55_t:
                signal_px = h55_t
                entry_px = long_entry_price(signal_px)
                new_sh = unit_size(equity, n_t)
                if new_sh > 0:
                    equity -= (entry_px - signal_px) * new_sh
                    state.enter_long(date_t, entry_px, n_t, equity, system='S2')
                    state.skip_s1 = False
                    counters['n_entries'] += 1
                    cur_trade = _start_trade(date_t, 'S2', entry_px, n_t)
            elif np.isfinite(h20_t) and c_t > h20_t:
                if not state.skip_s1:
                    signal_px = h20_t
                    entry_px = long_entry_price(signal_px)
                    new_sh = unit_size(equity, n_t)
                    if new_sh > 0:
                        equity -= (entry_px - signal_px) * new_sh
                        state.enter_long(date_t, entry_px, n_t, equity, system='S1')
                        counters['n_entries'] += 1
                        cur_trade = _start_trade(date_t, 'S1', entry_px, n_t)
                else:
                    state.skip_s1 = False
                    counters['n_s1_skips'] += 1

        equity_arr[t] = equity
        daily_state.append(dict(date=date_t, equity=equity,
                                num_units=state.num_units,
                                direction=state.direction,
                                stop_level=state.stop_level,
                                skip_s1=state.skip_s1,
                                n=n_t, h20=h20[t], h55=h55[t]))

    # Forced exit at end of series for any open position
    if not state.is_flat:
        final_px = long_exit_price(close[-1])
        pnl, _ = state.exit_all(final_px)
        if cur_trade is not None:
            holding = (dates[-1] - cur_trade['entry_date']).days
            _close_trade(cur_trade, dates[-1], final_px, 'series_end', pnl, holding)
            trade_log.append(cur_trade)

    equity_series = pd.Series(equity_arr, index=dates).ffill()
    return TurtleSimResult(
        equity_series=equity_series,
        trade_log=pd.DataFrame(trade_log),
        daily_state=pd.DataFrame(daily_state).set_index('date'),
        **counters,
    )


def simulate_turtle_long_short(
    long_ohlc: pd.DataFrame,    # TQQQ synthetic
    short_ohlc: pd.DataFrame,   # SQQQ synthetic — symmetric sleeve for shorts
    nasdaq_ohlc: pd.DataFrame,  # underlying NASDAQ for breakout/exit signals
    sofr_annual: pd.Series,
    initial_equity: float = 100_000.0,
    s1_period: int = 20, s2_period: int = 55,
    s1_exit_period: int = 10, s2_exit_period: int = 20,
    atr_period: int = 20,
    max_units: int = 4,
) -> TurtleSimResult:
    """
    T2: Pure Turtle Long/Short on NASDAQ-driven signals.
      - Long entry  on NASDAQ 20/55-day high break  → hold TQQQ synthetic
      - Short entry on NASDAQ 20/55-day low break   → hold SQQQ synthetic
      - At most one side open at any time (single sleeve).
      - Signals computed on NASDAQ OHLC; sizing N from the TRADED sleeve's ATR.
    """
    # Signals from NASDAQ
    h_n = nasdaq_ohlc['High'].values
    l_n = nasdaq_ohlc['Low'].values
    c_n = nasdaq_ohlc['Close'].values
    h20_n = compute_donchian_high(h_n, s1_period)
    h55_n = compute_donchian_high(h_n, s2_period)
    l20_n = compute_donchian_low(l_n, s1_period)        # short S1 entry
    l55_n = compute_donchian_low(l_n, s2_period)        # short S2 entry
    h10_n = compute_donchian_high(h_n, s1_exit_period)  # short exit (against)
    h20x  = compute_donchian_high(h_n, s2_exit_period)
    l10_n = compute_donchian_low(l_n, s1_exit_period)   # long exit (S1)
    l20x  = compute_donchian_low(l_n, s2_exit_period)   # long exit (S2)

    # ATRs from each sleeve
    atr_long  = wilder_atr(long_ohlc['High'].values, long_ohlc['Low'].values,
                           long_ohlc['Close'].values, period=atr_period)
    atr_short = wilder_atr(short_ohlc['High'].values, short_ohlc['Low'].values,
                           short_ohlc['Close'].values, period=atr_period)

    dates = long_ohlc.index
    sofr  = sofr_annual.reindex(dates).ffill().bfill().values

    long_h  = long_ohlc['High'].values
    long_l  = long_ohlc['Low'].values
    long_c  = long_ohlc['Close'].values
    short_h = short_ohlc['High'].values
    short_l = short_ohlc['Low'].values
    short_c = short_ohlc['Close'].values

    warm_up = max(s2_period, atr_period) + 1

    state  = TurtleState(max_units=max_units)
    equity = initial_equity
    equity_arr = np.full(len(dates), np.nan)
    equity_arr[:warm_up] = initial_equity
    blew_up = False
    EQUITY_FLOOR = 0.05 * initial_equity   # liquidate at 95% drawdown (broker margin)

    trade_log: list[dict] = []
    daily_state: list[dict] = []
    cur_trade: Optional[dict] = None
    counters = dict(n_entries=0, n_2n_stops=0,
                    n_donchian_exits_s1=0, n_donchian_exits_s2=0,
                    n_s1_skips=0, n_pyramid_adds=0, n_short_entries=0)

    for t in range(warm_up, len(dates)):
        date_t = dates[t]
        c_nas = c_n[t]

        # Account ruin check — once blown up, freeze equity and stop trading
        if blew_up or equity <= EQUITY_FLOOR:
            if not blew_up and cur_trade is not None:
                cur_trade.update(exit_date=date_t, exit_price=np.nan,
                                 exit_reason='account_ruin', gross_pnl=np.nan,
                                 holding_days=(date_t - cur_trade['entry_date']).days)
                trade_log.append(cur_trade); cur_trade = None
            blew_up = True
            equity = max(equity, EQUITY_FLOOR)
            equity_arr[t] = equity
            continue

        # Use the active sleeve's data for MTM/stops/pyramid
        is_long  = state.direction == 'long'
        is_short = state.direction == 'short'

        if is_long:
            sleeve_c_prev = long_c[t-1]; sleeve_c = long_c[t]
            sleeve_h = long_h[t]; sleeve_l = long_l[t]
            n_t = atr_long[t]
        elif is_short:
            sleeve_c_prev = short_c[t-1]; sleeve_c = short_c[t]
            sleeve_h = short_h[t]; sleeve_l = short_l[t]
            n_t = atr_short[t]
        else:
            n_t = atr_long[t]   # for sizing potential entry

        if not np.isfinite(n_t) or n_t <= 0:
            equity_arr[t] = equity
            continue

        # 1. holding cost + MTM (only when in a sleeve)
        if is_long:
            sleeve_val_prev = state.total_shares * sleeve_c_prev
            equity -= sleeve_val_prev * tqqq_daily_holding_cost_rate(sofr[t])
            equity += state.total_shares * (sleeve_c - sleeve_c_prev)
        elif is_short:
            sleeve_val_prev = state.total_shares * sleeve_c_prev
            equity -= sleeve_val_prev * tqqq_daily_holding_cost_rate(sofr[t])
            equity += state.total_shares * (sleeve_c - sleeve_c_prev)
            # NOTE: state holds SQQQ shares (long SQQQ to be short NASDAQ)
            # so MTM is + (current - prev) on SQQQ position itself.

        # 2. exit checks
        exited_today = False
        if is_long:
            stop_lvl = state.stop_level
            if sleeve_l <= stop_lvl:
                exit_px = long_exit_price(stop_lvl)
                equity += (exit_px - sleeve_c) * state.total_shares
                pnl, _ = state.exit_all(exit_px)
                state.skip_s1 = False
                counters['n_2n_stops'] += 1
                exited_today = True
                if cur_trade is not None:
                    holding = (date_t - cur_trade['entry_date']).days
                    cur_trade.update(exit_date=date_t, exit_price=exit_px,
                                     exit_reason='2N_stop', gross_pnl=pnl,
                                     holding_days=holding)
                    trade_log.append(cur_trade); cur_trade = None
            else:
                # Donchian exit on NASDAQ low against position
                l_exit = l10_n[t] if cur_trade and cur_trade['system'] == 'S1' else l20x[t]
                if np.isfinite(l_exit) and c_nas < l_exit:
                    # Translate signal: when NASDAQ breaks down, SELL the TQQQ sleeve
                    # at its current close (approx). Use sleeve close for fill.
                    exit_px = long_exit_price(sleeve_c)
                    equity += (exit_px - sleeve_c) * state.total_shares
                    pnl, _ = state.exit_all(exit_px)
                    if cur_trade is not None:
                        if cur_trade['system'] == 'S1':
                            counters['n_donchian_exits_s1'] += 1
                            reason = '10d_exit'
                            state.skip_s1 = (pnl > 0)
                        else:
                            counters['n_donchian_exits_s2'] += 1
                            reason = '20d_exit'
                            state.skip_s1 = False
                        holding = (date_t - cur_trade['entry_date']).days
                        cur_trade.update(exit_date=date_t, exit_price=exit_px,
                                         exit_reason=reason, gross_pnl=pnl,
                                         holding_days=holding)
                        trade_log.append(cur_trade); cur_trade = None
                    exited_today = True
        elif is_short:
            # Short stop: SQQQ price falls (we're long SQQQ → losing)
            stop_lvl = state.stop_level   # for short = entry + 2N (SQQQ price up)
            # Actually for our model: SQQQ is held LONG when shorting NASDAQ.
            # The 2N stop on a short position is when SQQQ FALLS below entry-2N.
            # But TurtleState.short stop_level = entry+2N (designed for true short).
            # Workaround: for T2 we use long SQQQ as the short sleeve, so we should
            # use the LONG stop semantics on the SQQQ position.
            # To simplify: track this as a "long SQQQ" position internally.
            # For correctness, we instead use stop = entry - 2N (long semantics).
            # Compute manually:
            if state.units:
                last = state.units[-1]
                effective_stop = last.price - 2.0 * last.n_value
            else:
                effective_stop = None

            if effective_stop is not None and sleeve_l <= effective_stop:
                exit_px = long_exit_price(effective_stop)
                equity += (exit_px - sleeve_c) * state.total_shares
                # exit_all for short direction computes (entry-exit)*shares,
                # but we want (exit-entry)*shares for long-SQQQ.
                # → compute PnL manually and reset state
                pnl_manual = sum((exit_px - u.price) * u.shares for u in state.units)
                state.units = []
                state.direction = 'flat'
                state.entry_system = None
                state.skip_s1 = False
                counters['n_2n_stops'] += 1
                exited_today = True
                if cur_trade is not None:
                    holding = (date_t - cur_trade['entry_date']).days
                    cur_trade.update(exit_date=date_t, exit_price=exit_px,
                                     exit_reason='2N_stop', gross_pnl=pnl_manual,
                                     holding_days=holding)
                    trade_log.append(cur_trade); cur_trade = None
            else:
                # Donchian exit on NASDAQ high (against short)
                h_exit = h10_n[t] if cur_trade and cur_trade['system'] == 'S1' else h20x[t]
                if np.isfinite(h_exit) and c_nas > h_exit:
                    exit_px = long_exit_price(sleeve_c)
                    equity += (exit_px - sleeve_c) * state.total_shares
                    pnl_manual = sum((exit_px - u.price) * u.shares for u in state.units)
                    if cur_trade is not None:
                        if cur_trade['system'] == 'S1':
                            counters['n_donchian_exits_s1'] += 1
                            reason = '10d_exit'
                        else:
                            counters['n_donchian_exits_s2'] += 1
                            reason = '20d_exit'
                        holding = (date_t - cur_trade['entry_date']).days
                        cur_trade.update(exit_date=date_t, exit_price=exit_px,
                                         exit_reason=reason, gross_pnl=pnl_manual,
                                         holding_days=holding)
                        trade_log.append(cur_trade); cur_trade = None
                    state.units = []
                    state.direction = 'flat'
                    state.entry_system = None
                    state.skip_s1 = False
                    exited_today = True

        # 3. pyramid (long sleeve only — keep T2 simple, no short pyramiding)
        if state.direction == 'long':
            safety = 0
            while (state.num_units < max_units
                   and state.next_pyramid_price is not None
                   and sleeve_h >= state.next_pyramid_price
                   and safety < max_units):
                pyr_signal = state.next_pyramid_price
                entry_px = long_entry_price(pyr_signal)
                new_sh = unit_size(equity, n_t)
                if new_sh <= 0:
                    break
                equity -= (entry_px - pyr_signal) * new_sh
                state.add_pyramid_unit(date_t, entry_px, n_t, equity)
                counters['n_pyramid_adds'] += 1
                if cur_trade is not None:
                    cur_trade['pyramid_adds'] += 1
                safety += 1

        # 4. new entry (flat only, no same-bar round-trip)
        if state.is_flat and not exited_today:
            # Long entry on NASDAQ high break (use TQQQ sleeve)
            if np.isfinite(h55_n[t]) and c_nas > h55_n[t]:
                signal_px = long_c[t]   # enter at TQQQ close (signal triggered)
                entry_px = long_entry_price(signal_px)
                n_t = atr_long[t]
                new_sh = unit_size(equity, n_t)
                if new_sh > 0:
                    equity -= (entry_px - signal_px) * new_sh
                    state.enter_long(date_t, entry_px, n_t, equity, system='S2')
                    state.skip_s1 = False
                    counters['n_entries'] += 1
                    cur_trade = dict(entry_date=date_t, system='S2', side='long',
                                     entry_price=entry_px, n_at_entry=n_t,
                                     pyramid_adds=0)
            elif np.isfinite(h20_n[t]) and c_nas > h20_n[t]:
                if not state.skip_s1:
                    signal_px = long_c[t]
                    entry_px = long_entry_price(signal_px)
                    n_t = atr_long[t]
                    new_sh = unit_size(equity, n_t)
                    if new_sh > 0:
                        equity -= (entry_px - signal_px) * new_sh
                        state.enter_long(date_t, entry_px, n_t, equity, system='S1')
                        counters['n_entries'] += 1
                        cur_trade = dict(entry_date=date_t, system='S1', side='long',
                                         entry_price=entry_px, n_at_entry=n_t,
                                         pyramid_adds=0)
                else:
                    state.skip_s1 = False
                    counters['n_s1_skips'] += 1
            # Short entry on NASDAQ low break (use SQQQ sleeve as long SQQQ)
            elif np.isfinite(l55_n[t]) and c_nas < l55_n[t]:
                signal_px = short_c[t]
                entry_px = long_entry_price(signal_px)   # buying SQQQ
                n_t = atr_short[t]
                new_sh = unit_size(equity, n_t)
                if new_sh > 0:
                    equity -= (entry_px - signal_px) * new_sh
                    # Treat as long SQQQ; reuse enter_long for tracking
                    state.enter_long(date_t, entry_px, n_t, equity, system='S2')
                    counters['n_entries'] += 1
                    counters['n_short_entries'] += 1
                    cur_trade = dict(entry_date=date_t, system='S2', side='short',
                                     entry_price=entry_px, n_at_entry=n_t,
                                     pyramid_adds=0)
            elif np.isfinite(l20_n[t]) and c_nas < l20_n[t]:
                if not state.skip_s1:
                    signal_px = short_c[t]
                    entry_px = long_entry_price(signal_px)
                    n_t = atr_short[t]
                    new_sh = unit_size(equity, n_t)
                    if new_sh > 0:
                        equity -= (entry_px - signal_px) * new_sh
                        state.enter_long(date_t, entry_px, n_t, equity, system='S1')
                        counters['n_entries'] += 1
                        counters['n_short_entries'] += 1
                        cur_trade = dict(entry_date=date_t, system='S1', side='short',
                                         entry_price=entry_px, n_at_entry=n_t,
                                         pyramid_adds=0)
                else:
                    state.skip_s1 = False
                    counters['n_s1_skips'] += 1

        equity_arr[t] = equity
        daily_state.append(dict(date=date_t, equity=equity,
                                num_units=state.num_units,
                                direction=state.direction,
                                stop_level=state.stop_level,
                                n=n_t))

    # Forced exit at series end
    if not state.is_flat:
        if state.direction == 'long':
            final_px = long_exit_price(long_c[-1])
            pnl_manual = sum((final_px - u.price) * u.shares for u in state.units)
        else:
            final_px = long_exit_price(short_c[-1])
            pnl_manual = sum((final_px - u.price) * u.shares for u in state.units)
        if cur_trade is not None:
            holding = (dates[-1] - cur_trade['entry_date']).days
            cur_trade.update(exit_date=dates[-1], exit_price=final_px,
                             exit_reason='series_end', gross_pnl=pnl_manual,
                             holding_days=holding)
            trade_log.append(cur_trade)

    equity_series = pd.Series(equity_arr, index=dates).ffill()
    return TurtleSimResult(
        equity_series=equity_series,
        trade_log=pd.DataFrame(trade_log),
        daily_state=pd.DataFrame(daily_state).set_index('date'),
        **counters,
    )


# ---------------------------------------------------------------------- #
# Metrics                                                                  #
# ---------------------------------------------------------------------- #

def compute_metrics(equity: pd.Series, name: str = "") -> dict:
    """Compute CAGR, Sharpe, MaxDD, Worst5Y from an equity series."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    daily_ret = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr  = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    vol   = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / vol if vol > 0 else 0.0
    peaks = equity.cummax()
    dd    = (equity / peaks - 1.0)
    maxdd = dd.min()
    # Worst rolling 5Y CAGR
    w5y_window = 5 * 252
    if len(equity) >= w5y_window:
        roll_cagr = (equity / equity.shift(w5y_window)) ** (1.0 / 5.0) - 1.0
        worst5y = roll_cagr.min()
    else:
        worst5y = np.nan
    return dict(name=name, cagr=cagr, sharpe=sharpe, maxdd=maxdd, worst5y=worst5y,
                vol=vol, years=years, final_equity=equity.iloc[-1])


def yearly_returns(equity: pd.Series) -> pd.Series:
    """Calendar year returns from equity curve."""
    eq = equity.dropna()
    annual = eq.resample('YE').last()
    rets = annual.pct_change().dropna()
    first_eoy = annual.iloc[0]
    first_year = eq.index[0].year
    first_ret = first_eoy / eq.iloc[0] - 1.0
    rets = pd.concat([pd.Series({pd.Timestamp(f'{first_year}-12-31'): first_ret}), rets])
    rets.index = rets.index.year
    return rets
