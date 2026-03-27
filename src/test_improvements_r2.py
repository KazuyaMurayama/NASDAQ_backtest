"""
Strategy Improvement Round 2
=============================
Based on:
- Round 1 results: CPPI improved MaxDD, AsymHAR improved Worst5Y
- Web research: Vol ratio filter, Dynamic CPPI, Jump penalty regime

Tests improvements that COMBINE multiple ideas.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol,
    calc_vt_leverage
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult, calc_soft_dd_with_floor
)
from test_improvements import (
    strategy_baseline_momdecel_ens2_st, calc_asym_har_vol,
    DATA_PATH, DEFAULT_DELAY, DEFAULT_THRESHOLD, ANNUAL_COST
)

# =============================================================================
# R2-1: Vol Ratio Filter (QuantPedia inspired)
# Compare short-term vol to medium-term vol. When short-term exceeds
# medium-term significantly, reduce leverage.
# =============================================================================
def strategy_r2_vol_ratio_filter(close, returns):
    """R2-1: Vol Ratio Filter — reduce exposure when short-term vol spikes above trend."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # Vol ratio filter: short-term EWMA(5) vs medium-term EWMA(63)
    vol_short = calc_ewma_vol(returns, 5)
    vol_medium = calc_ewma_vol(returns, 63)
    vol_ratio = vol_short / vol_medium

    # When short-term vol is 1.5x+ the medium-term, reduce leverage by 40%
    vol_filter = pd.Series(1.0, index=close.index)
    vol_filter[vol_ratio > 2.0] = 0.4
    vol_filter[(vol_ratio > 1.5) & (vol_ratio <= 2.0)] = 0.7

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vol_filter
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R2-2: Dynamic CPPI + MomDecel+Ens2 (PIMCO/QuantPedia inspired)
# CPPI with vol-adaptive multiplier and drawdown-based floor update
# =============================================================================
def strategy_r2_dynamic_cppi(close, returns):
    """R2-2: Dynamic CPPI with vol-adaptive cushion multiplier.

    - Floor is 80% of rolling peak (drawdown-based CPPI)
    - Multiplier adjusts with vol: low vol -> higher m, high vol -> lower m
    - Combined with Ens2 VT layers
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # Drawdown-based floor: 80% of rolling 200-day peak
    peak = close.rolling(200, min_periods=1).max()
    floor_ratio = 0.80
    cushion = (close / peak - floor_ratio) / (1.0 - floor_ratio)
    cushion = cushion.clip(0, 1.0)

    # Dynamic multiplier based on vol (low vol -> aggressive, high vol -> conservative)
    ewma_vol_21 = calc_ewma_vol(returns, 21)
    vol_ma_126 = ewma_vol_21.rolling(126).mean()
    vol_relative = ewma_vol_21 / vol_ma_126.replace(0, 0.001)

    # Map vol_relative to multiplier: low vol (0.5x) -> m=5, high vol (2.0x) -> m=2
    dynamic_m = (5.0 - 2.0 * (vol_relative - 0.5) / 1.5).clip(2.0, 5.0).fillna(3.5)

    # CPPI leverage = m * cushion, capped at 1.0
    cppi_lev = (dynamic_m * cushion).clip(0, 1.0)

    # Blend CPPI with Ens2 VT
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # Use minimum of CPPI and VT leverage (conservative blend)
    ens2_lev = vt_lev * slope_mult * mom_decel
    blended_lev = np.minimum(cppi_lev, ens2_lev)

    raw_leverage = dd_signal * blended_lev
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R2-3: Soft DD + Ens2(S+T) + MomDecel (from delay_robust.py)
# Already implemented but not tested with MomDecel
# =============================================================================
def strategy_r2_soft_dd_momdecel(close, returns):
    """R2-3: Soft DD with hard floor + MomDecel + Ens2 VT layers.

    Soft DD starts reducing leverage at -5% from peak (ratio=0.95),
    reaching 0 at -20% (ratio=0.80), with hard floor at -22% (ratio=0.78).
    """
    soft_dd = calc_soft_dd_with_floor(close, 200, lower=0.80, upper=0.95, floor_ratio=0.78)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = soft_dd * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)

    # For trade counting, approximate binary signal from soft_dd
    dd_binary = (soft_dd > 0.5).astype(float)
    return raw_leverage, dd_binary


# =============================================================================
# R2-4: Jump Penalty Regime (arXiv 2024 inspired)
# Simplified version: require N consecutive days of regime signal before switching
# =============================================================================
def calc_persistent_regime_signal(close, ma_period=200, confirm_days=10):
    """Persistent regime: require N consecutive days above/below MA before switching.
    This acts like a jump penalty — prevents rapid regime flipping."""
    ma = close.rolling(ma_period).mean()
    above_ma = (close > ma).astype(int)

    # Count consecutive days above/below
    regime = pd.Series(1.0, index=close.index)  # Start in HOLD
    state = 1.0
    count_above = 0
    count_below = 0

    for i in range(len(close)):
        if above_ma.iloc[i] == 1:
            count_above += 1
            count_below = 0
        else:
            count_below += 1
            count_above = 0

        # Only switch after N consecutive days of confirmation
        if state == 1.0 and count_below >= confirm_days:
            state = 0.5  # Reduce to 50% (not full exit)
        elif state == 0.5 and count_above >= confirm_days:
            state = 1.0

        regime.iloc[i] = state

    return regime

def strategy_r2_persistent_regime(close, returns):
    """R2-4: Persistent Regime Filter (jump penalty inspired).

    Add a low-frequency regime overlay that requires 10 consecutive days
    of confirmation before switching. Reduces to 50% (not 0) in bear regime.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    regime = calc_persistent_regime_signal(close, 200, 10)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * regime
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R2-5: Combined Best — AsymHAR + CPPI Floor + Vol Ratio Filter
# =============================================================================
def strategy_r2_combined_best(close, returns):
    """R2-5: Combine the best elements from R1 and R2.

    - AsymHAR Vol (R1: improved Worst5Y)
    - CPPI floor cushion (R1: improved MaxDD)
    - Vol ratio filter (R2: research-backed)
    - MomDecel + DD unchanged
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # AsymHAR Vol (from R1)
    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)

    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # CPPI cushion (gentler version)
    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    cppi_mult = ((ratio - 0.82) / (1.0 - 0.82)).clip(0.5, 1.0)

    # Vol ratio filter (gentler version)
    vol_short = calc_ewma_vol(returns, 5)
    vol_medium = calc_ewma_vol(returns, 63)
    vol_ratio = vol_short / vol_medium
    vol_filter = pd.Series(1.0, index=close.index)
    vol_filter[vol_ratio > 1.8] = 0.6
    vol_filter[(vol_ratio > 1.4) & (vol_ratio <= 1.8)] = 0.8

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult * vol_filter
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R2-6: Combined Best V2 — more aggressive MaxDD protection
# =============================================================================
def strategy_r2_combined_v2(close, returns):
    """R2-6: Aggressive MaxDD protection variant.

    Like R2-5 but with:
    - Softer CPPI (starts at -8% instead of -18%)
    - Stronger vol filter
    - Persistent regime overlay at 50%
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)

    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # Earlier CPPI cushion (starts reducing at -8% from peak)
    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    cppi_mult = ((ratio - 0.82) / (0.92 - 0.82)).clip(0.3, 1.0)

    # Stronger vol filter
    vol_short = calc_ewma_vol(returns, 5)
    vol_medium = calc_ewma_vol(returns, 63)
    vol_ratio = vol_short / vol_medium
    vol_filter = pd.Series(1.0, index=close.index)
    vol_filter[vol_ratio > 1.5] = 0.5
    vol_filter[(vol_ratio > 1.2) & (vol_ratio <= 1.5)] = 0.75

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult * vol_filter
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Test Runner
# =============================================================================
def run_test(close, dates, strategy_func, name):
    returns = close.pct_change()
    raw_leverage, dd_signal = strategy_func(close, returns)

    lev = rebalance_threshold(raw_leverage, DEFAULT_THRESHOLD)
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)
    metrics = calc_metrics(nav, strat_ret, dd_signal, dates)
    rebal = count_rebalances(lev)
    metrics['Strategy'] = name
    metrics['Rebalances'] = rebal
    return metrics


def print_results(results):
    df = pd.DataFrame(results)
    cols = ['Strategy', 'CAGR', 'Sharpe', 'MaxDD', 'Worst5Y', 'Sortino', 'Trades', 'Rebalances']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()

    df['CAGR'] = df['CAGR'].apply(lambda x: f"{x*100:.2f}%")
    df['Sharpe'] = df['Sharpe'].apply(lambda x: f"{x:.4f}")
    df['MaxDD'] = df['MaxDD'].apply(lambda x: f"{x*100:.2f}%")
    df['Worst5Y'] = df['Worst5Y'].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A")
    if 'Sortino' in df.columns:
        df['Sortino'] = df['Sortino'].apply(lambda x: f"{x:.4f}")
    df['Trades'] = df['Trades'].astype(int)
    df['Rebalances'] = df['Rebalances'].astype(int)

    print("\n" + "="*115)
    print("ROUND 2: STRATEGY IMPROVEMENT COMPARISON (1974-2026, 5-day delay, 1.5% cost)")
    print("="*115)
    print(df.to_string(index=False))
    print("="*115)


def main():
    print(f"Loading data from {DATA_PATH}")
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data range: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)")

    results = []
    strategies = [
        (strategy_baseline_momdecel_ens2_st, "Baseline: MomDecel+Ens2(S+T)"),
        (strategy_r2_vol_ratio_filter, "R2-1: Vol Ratio Filter"),
        (strategy_r2_dynamic_cppi, "R2-2: Dynamic CPPI"),
        (strategy_r2_soft_dd_momdecel, "R2-3: Soft DD + MomDecel"),
        (strategy_r2_persistent_regime, "R2-4: Persistent Regime"),
        (strategy_r2_combined_best, "R2-5: Combined Best"),
        (strategy_r2_combined_v2, "R2-6: Combined V2 (aggr)"),
    ]

    for i, (func, name) in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}] Testing {name}...")
        r = run_test(close, dates, func, name)
        results.append(r)
        print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, "
              f"MaxDD: {r['MaxDD']*100:.2f}%, Worst5Y: {r['Worst5Y']*100:.2f}%")

    print_results(results)

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'improvement_results_r2.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
