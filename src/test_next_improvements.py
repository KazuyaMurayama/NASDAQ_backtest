"""
Next Improvements: Dual DD, Rolling Sharpe, ATR-Adaptive DD
============================================================
All built on top of A2 (VIX MeanRevert + MD(60/180)) as the new baseline.

Direction 1: Dual Timeframe DD — short(100d) + long(200d) DD
Direction 2: Rolling Sharpe Gate — market confidence filter
Direction 3: ATR-Adaptive DD — vol-adjusted exit thresholds
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import (
    load_data, calc_dd_signal, calc_dual_dd_signal, calc_triple_dd_signal,
    calc_metrics, calc_ewma_vol, calc_rolling_sharpe, calc_atr
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult
)
from test_improvements import DATA_PATH, DEFAULT_DELAY, ANNUAL_COST
from test_vix_integration import calc_vix_proxy

DEFAULT_THRESHOLD = 0.20


# =============================================================================
# A2 Baseline (current best: VIX MeanRevert + MD(60/180))
# =============================================================================
def strategy_a2_baseline(close, returns):
    """A2: VIX MeanRevert + MD(60/180) — current best."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_original_baseline(close, returns):
    """Original baseline: MD(40/120) + Ens2(S+T)."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Direction 1: Dual Timeframe DD + A2
# =============================================================================
def strategy_d1_dual_dd(close, returns):
    """D1: Dual DD (100d+200d) + A2 layers.
    Short-term DD catches early declines, long-term prevents whipsaws."""
    dd_signal = calc_dual_dd_signal(close, short_lb=100, long_lb=200,
                                     short_exit=0.87, long_exit=0.85, reentry=0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_d1b_dual_dd_tight(close, returns):
    """D1b: Dual DD with tighter thresholds (earlier exit)."""
    dd_signal = calc_dual_dd_signal(close, short_lb=100, long_lb=200,
                                     short_exit=0.90, long_exit=0.87, reentry=0.93)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Direction 2: Rolling Sharpe Gate + A2
# =============================================================================
def strategy_d2_rolling_sharpe(close, returns):
    """D2: Rolling Sharpe gate on A2.
    When 60-day Sharpe < 0, reduce leverage (market losing confidence)."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    # Rolling Sharpe gate
    roll_sharpe = calc_rolling_sharpe(returns, 60)
    # Sharpe > 0.5 -> 1.0, Sharpe = 0 -> 0.7, Sharpe < -1 -> 0.3
    sharpe_mult = (0.7 + 0.6 * roll_sharpe).clip(0.3, 1.1).fillna(1.0)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult * sharpe_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_d2b_sharpe_binary(close, returns):
    """D2b: Binary Sharpe gate — simpler version.
    Sharpe > 0 -> full, Sharpe <= 0 -> 50%."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    roll_sharpe = calc_rolling_sharpe(returns, 60)
    sharpe_gate = pd.Series(1.0, index=close.index)
    sharpe_gate[roll_sharpe <= 0] = 0.5
    sharpe_gate = sharpe_gate.fillna(1.0)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult * sharpe_gate
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Direction 3: ATR-Adaptive DD + A2
# =============================================================================
def strategy_d3_atr_adaptive(close, returns):
    """D3: ATR-Adaptive DD thresholds + A2 layers.
    High ATR -> loosen exit (avoid false exits in volatile but trending markets).
    Low ATR -> tighten exit (protect capital in calm markets)."""

    # Need High/Low data — approximate from Close
    high_approx = close.rolling(2).max()
    low_approx = close.rolling(2).min()
    atr = calc_atr(high_approx, low_approx, close, 20)
    atr_pct = atr / close

    # ATR percentile over 252 days
    atr_rank = atr_pct.rolling(252).rank(pct=True).fillna(0.5)

    # Dynamic exit threshold: low ATR (calm) -> 0.84, high ATR (volatile) -> 0.80
    dynamic_exit = 0.84 - 0.04 * atr_rank  # Range: 0.80 to 0.84
    # Dynamic reentry: low ATR -> 0.90, high ATR -> 0.94
    dynamic_reentry = 0.90 + 0.04 * atr_rank  # Range: 0.90 to 0.94

    # DD with dynamic thresholds
    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    for i in range(len(close)):
        ex = dynamic_exit.iloc[i] if not pd.isna(dynamic_exit.iloc[i]) else 0.82
        re = dynamic_reentry.iloc[i] if not pd.isna(dynamic_reentry.iloc[i]) else 0.92
        if state == 'HOLD' and ratio.iloc[i] <= ex:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= re:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    raw_leverage = position * vt_lev * slope_mult * mom_decel * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, position


# =============================================================================
# Combinations
# =============================================================================
def strategy_combo_dual_dd_sharpe(close, returns):
    """Combo: Dual DD + Rolling Sharpe + A2 VIX layers."""
    dd_signal = calc_dual_dd_signal(close, short_lb=100, long_lb=200,
                                     short_exit=0.87, long_exit=0.85, reentry=0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    roll_sharpe = calc_rolling_sharpe(returns, 60)
    sharpe_mult = (0.7 + 0.6 * roll_sharpe).clip(0.3, 1.1).fillna(1.0)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult * sharpe_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_combo_atr_sharpe(close, returns):
    """Combo: ATR-Adaptive DD + Rolling Sharpe + A2 VIX layers."""
    # ATR-adaptive DD
    high_approx = close.rolling(2).max()
    low_approx = close.rolling(2).min()
    atr = calc_atr(high_approx, low_approx, close, 20)
    atr_pct = atr / close
    atr_rank = atr_pct.rolling(252).rank(pct=True).fillna(0.5)
    dynamic_exit = 0.84 - 0.04 * atr_rank
    dynamic_reentry = 0.90 + 0.04 * atr_rank

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    for i in range(len(close)):
        ex = dynamic_exit.iloc[i] if not pd.isna(dynamic_exit.iloc[i]) else 0.82
        re = dynamic_reentry.iloc[i] if not pd.isna(dynamic_reentry.iloc[i]) else 0.92
        if state == 'HOLD' and ratio.iloc[i] <= ex:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= re:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    roll_sharpe = calc_rolling_sharpe(returns, 60)
    sharpe_mult = (0.7 + 0.6 * roll_sharpe).clip(0.3, 1.1).fillna(1.0)

    raw_leverage = position * vt_lev * slope_mult * mom_decel * vix_mult * sharpe_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, position


# =============================================================================
# Test Runner
# =============================================================================
def run_full_and_oos(close, dates, strategy_func, name, threshold=DEFAULT_THRESHOLD):
    returns = close.pct_change()
    raw_leverage, dd_signal = strategy_func(close, returns)
    lev = rebalance_threshold(raw_leverage, threshold)
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)
    m_full = calc_metrics(nav, strat_ret, dd_signal, dates)
    rebal = count_rebalances(lev)

    split_date = '2021-05-07'
    split_idx = dates[dates >= split_date].index[0]
    nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
    ret_oos = strat_ret.iloc[split_idx:]
    oos_years = len(nav_oos) / 252
    oos_cagr = (nav_oos.iloc[-1] ** (1 / oos_years)) - 1
    oos_sharpe = (ret_oos.mean() * 252) / (ret_oos.std() * np.sqrt(252)) if ret_oos.std() > 0 else 0
    oos_maxdd = (nav_oos / nav_oos.cummax() - 1).min()

    mask_2022 = (dates.dt.year == 2022)
    nav_2022 = nav[mask_2022]
    ret_2022 = (nav_2022.iloc[-1] / nav_2022.iloc[0] - 1) * 100 if mask_2022.any() else np.nan

    return {
        'Strategy': name,
        'Sharpe': m_full['Sharpe'], 'CAGR': m_full['CAGR'],
        'MaxDD': m_full['MaxDD'], 'Worst5Y': m_full['Worst5Y'],
        'OOS_Sharpe': oos_sharpe, 'OOS_CAGR': oos_cagr, 'OOS_MaxDD': oos_maxdd,
        'Y2022': ret_2022, 'Trades': m_full['Trades'], 'Rebalances': rebal,
    }


def main():
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)\n")

    strategies = [
        (strategy_original_baseline, "Original: MD(40/120)"),
        (strategy_a2_baseline, "A2: VIX+MD(60/180)"),
        # Direction 1: Dual DD
        (strategy_d1_dual_dd, "D1: DualDD+A2"),
        (strategy_d1b_dual_dd_tight, "D1b: DualDD(tight)+A2"),
        # Direction 2: Rolling Sharpe
        (strategy_d2_rolling_sharpe, "D2: RollSharpe+A2"),
        (strategy_d2b_sharpe_binary, "D2b: SharpeBinary+A2"),
        # Direction 3: ATR-Adaptive
        (strategy_d3_atr_adaptive, "D3: ATR-DD+A2"),
        # Combinations
        (strategy_combo_dual_dd_sharpe, "Combo: DualDD+Sharpe+A2"),
        (strategy_combo_atr_sharpe, "Combo: ATR+Sharpe+A2"),
    ]

    results = []
    for i, (func, name) in enumerate(strategies, 1):
        print(f"[{i}/{len(strategies)}] {name}...")
        r = run_full_and_oos(close, dates, func, name)
        results.append(r)

    # Print comparison
    orig = results[0]
    a2 = results[1]
    print(f"\n{'Strategy':<28} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} | {'OOS_Sh':>7} {'OOS_CAGR':>9} {'Y2022':>7}")
    print("-" * 105)
    for r in results:
        flag = ""
        all4_vs_orig = (r['Sharpe'] > orig['Sharpe'] and r['CAGR'] > orig['CAGR']
                        and r['MaxDD'] > orig['MaxDD'] and r['Worst5Y'] > orig['Worst5Y'])
        all4_vs_a2 = (r['Sharpe'] > a2['Sharpe'] and r['CAGR'] > a2['CAGR']
                      and r['MaxDD'] > a2['MaxDD'] and r['Worst5Y'] > a2['Worst5Y'])
        oos_vs_orig = r['OOS_Sharpe'] > orig['OOS_Sharpe']
        oos_vs_a2 = r['OOS_Sharpe'] > a2['OOS_Sharpe']
        if all4_vs_a2 and oos_vs_a2:
            flag = " ★★★(vsA2)"
        elif all4_vs_orig and oos_vs_orig:
            flag = " ★★★(vsOrig)"
        elif all4_vs_a2:
            flag = " ★★(vsA2)"
        elif oos_vs_a2:
            flag = " ★(OOS>A2)"
        print(f"{r['Strategy']:<28} {r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% "
              f"{r['Worst5Y']*100:>6.2f}% | {r['OOS_Sharpe']:>7.4f} {r['OOS_CAGR']*100:>8.2f}% "
              f"{r['Y2022']:>6.1f}%{flag}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.DataFrame(results).to_csv(os.path.join(base_dir, 'improvement_results_next.csv'), index=False)
    print(f"\nSaved to improvement_results_next.csv")


if __name__ == '__main__':
    main()
