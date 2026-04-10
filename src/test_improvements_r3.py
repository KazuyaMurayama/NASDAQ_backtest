"""
Strategy Improvement Round 3 - Refinement
==========================================
Focus on the two most promising findings from R2:
- R2-2 Dynamic CPPI: Only strategy to improve Sharpe + MaxDD simultaneously
- R2-3 Soft DD: Best Worst5Y (-0.46%) and MaxDD (-47.59%), but too many trades

Round 3 goals:
1. Refine Dynamic CPPI parameters
2. Fix Soft DD trade count issue
3. Combine best elements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult, calc_soft_dd_with_floor
)
from test_improvements import (
    DATA_PATH, DEFAULT_DELAY, ANNUAL_COST,
    strategy_baseline_momdecel_ens2_st, calc_asym_har_vol
)

DEFAULT_THRESHOLD = 0.20


# =============================================================================
# R3-1: Refined Dynamic CPPI (best from R2)
# Wider parameter sweep on cushion formula
# =============================================================================
def strategy_r3_cppi_v2(close, returns):
    """R3-1: Dynamic CPPI V2 — more sensitive cushion.

    Key change: cushion starts reducing earlier (at -10% instead of -18%)
    but with gentler slope, so it doesn't kill CAGR as much.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak

    # Gentler CPPI: starts at ratio=0.90 (10% DD), reaches min at 0.82 (18% DD)
    # At 0.90: mult = 1.0
    # At 0.86: mult = 0.5
    # At 0.82: mult = 0.0 (DD kicks in anyway)
    cppi_mult = ((ratio - 0.82) / (0.90 - 0.82)).clip(0.3, 1.0)

    # Dynamic vol adjustment: low vol = aggressive CPPI, high vol = gentler
    ewma_vol_21 = calc_ewma_vol(returns, 21)
    vol_ma_126 = ewma_vol_21.rolling(126).mean()
    vol_rel = (ewma_vol_21 / vol_ma_126.replace(0, 0.001)).clip(0.5, 2.0).fillna(1.0)

    # When vol is low, don't reduce as much (boost cppi_mult toward 1)
    # When vol is high, reduce more aggressively
    vol_adjustment = (1.5 - 0.5 * vol_rel).clip(0.7, 1.3)
    cppi_adjusted = (cppi_mult * vol_adjustment).clip(0.3, 1.0)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_adjusted
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R3-2: Soft DD with higher rebalance threshold (fix trade count)
# =============================================================================
def strategy_r3_soft_dd_fixed(close, returns):
    """R3-2: Soft DD with higher rebalance threshold to reduce trade count.

    Use Th=30% instead of 20% for the soft DD component.
    Also make the soft DD range wider (0.75 to 0.98).
    """
    soft_dd = calc_soft_dd_with_floor(close, 200, lower=0.75, upper=0.98, floor_ratio=0.73)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = soft_dd * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    dd_binary = (soft_dd > 0.5).astype(float)
    return raw_leverage, dd_binary


# =============================================================================
# R3-3: Dynamic CPPI + AsymHAR (combine best R1 + R2)
# =============================================================================
def strategy_r3_cppi_asym_har(close, returns):
    """R3-3: Dynamic CPPI + AsymHAR Vol.

    Combine:
    - AsymHAR (R1-3: best Worst5Y)
    - Dynamic CPPI (R2-2: best balanced improvement)
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    cppi_mult = ((ratio - 0.82) / (0.90 - 0.82)).clip(0.3, 1.0)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R3-4: Dynamic CPPI + Soft DD hybrid
# Use Soft DD for gradual protection, CPPI for vol-adjusted scaling
# =============================================================================
def strategy_r3_soft_cppi_hybrid(close, returns):
    """R3-4: Hybrid Soft DD + CPPI.

    Use soft DD for the binary exit (gradual instead of sudden),
    combined with CPPI vol-adjusted cushion for the VT layer.
    """
    # Soft DD replaces binary DD
    soft_dd = calc_soft_dd_with_floor(close, 200, lower=0.78, upper=0.95, floor_ratio=0.76)

    # CPPI vol scaling
    ewma_vol_21 = calc_ewma_vol(returns, 21)
    vol_ma_126 = ewma_vol_21.rolling(126).mean()
    vol_rel = (ewma_vol_21 / vol_ma_126.replace(0, 0.001)).clip(0.5, 2.0).fillna(1.0)
    vol_scale = (1.3 - 0.3 * vol_rel).clip(0.7, 1.2)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = soft_dd * vt_lev * slope_mult * mom_decel * vol_scale
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    dd_binary = (soft_dd > 0.5).astype(float)
    return raw_leverage, dd_binary


# =============================================================================
# R3-5: MomDecel parameter optimization (40/120 -> test alternatives)
# =============================================================================
def strategy_r3_momdecel_30_90(close, returns):
    """R3-5a: MomDecel(30/90) — faster detection of momentum shifts."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=30, long=90,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_r3_momdecel_60_180(close, returns):
    """R3-5b: MomDecel(60/180) — slower, more stable detection."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R3-6: Best Candidate — CPPI V2 + AsymHAR + MomDecel(30/90)
# =============================================================================
def strategy_r3_best_candidate(close, returns):
    """R3-6: Best Candidate combining top refinements.

    - CPPI V2 cushion (gentle, vol-adjusted)
    - AsymHAR vol (multi-timescale + asymmetric)
    - MomDecel(30/90) (if faster detection helps)
    - Standard DD + SlopeMult
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    cppi_mult = ((ratio - 0.82) / (0.90 - 0.82)).clip(0.3, 1.0)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=30, long=90,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Test Runner
# =============================================================================
def run_test(close, dates, strategy_func, name, threshold=DEFAULT_THRESHOLD):
    returns = close.pct_change()
    raw_leverage, dd_signal = strategy_func(close, returns)
    lev = rebalance_threshold(raw_leverage, threshold)
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)
    metrics = calc_metrics(nav, strat_ret, dd_signal, dates)
    rebal = count_rebalances(lev)
    metrics['Strategy'] = name
    metrics['Rebalances'] = rebal
    return metrics


def print_results(results):
    df = pd.DataFrame(results)
    cols = ['Strategy', 'CAGR', 'Sharpe', 'MaxDD', 'Worst5Y', 'Sortino', 'Calmar', 'Trades', 'Rebalances']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()
    df['CAGR'] = df['CAGR'].apply(lambda x: f"{x*100:.2f}%")
    df['Sharpe'] = df['Sharpe'].apply(lambda x: f"{x:.4f}")
    df['MaxDD'] = df['MaxDD'].apply(lambda x: f"{x*100:.2f}%")
    df['Worst5Y'] = df['Worst5Y'].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A")
    df['Sortino'] = df['Sortino'].apply(lambda x: f"{x:.4f}")
    df['Calmar'] = df['Calmar'].apply(lambda x: f"{x:.4f}")
    df['Trades'] = df['Trades'].astype(int)
    df['Rebalances'] = df['Rebalances'].astype(int)
    print("\n" + "="*130)
    print("ROUND 3: REFINEMENT (1974-2026, 5-day delay, 1.5% cost)")
    print("="*130)
    print(df.to_string(index=False))
    print("="*130)


def main():
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)")

    results = []
    strategies = [
        (strategy_baseline_momdecel_ens2_st, "Baseline: MomDecel(40/120)+Ens2(S+T)", DEFAULT_THRESHOLD),
        (strategy_r3_cppi_v2, "R3-1: CPPI V2 (vol-adjusted)", DEFAULT_THRESHOLD),
        (strategy_r3_soft_dd_fixed, "R3-2: Soft DD (Th30%)", 0.30),
        (strategy_r3_cppi_asym_har, "R3-3: CPPI + AsymHAR", DEFAULT_THRESHOLD),
        (strategy_r3_soft_cppi_hybrid, "R3-4: Soft DD + CPPI hybrid", 0.25),
        (strategy_r3_momdecel_30_90, "R3-5a: MomDecel(30/90)", DEFAULT_THRESHOLD),
        (strategy_r3_momdecel_60_180, "R3-5b: MomDecel(60/180)", DEFAULT_THRESHOLD),
        (strategy_r3_best_candidate, "R3-6: CPPI+AsymHAR+MD(30/90)", DEFAULT_THRESHOLD),
    ]

    for i, (func, name, th) in enumerate(strategies, 1):
        print(f"[{i}/{len(strategies)}] {name}...")
        r = run_test(close, dates, func, name, th)
        results.append(r)
        delta_sharpe = r['Sharpe'] - results[0]['Sharpe'] if i > 1 else 0
        delta_maxdd = (r['MaxDD'] - results[0]['MaxDD']) * 100 if i > 1 else 0
        print(f"  Sharpe: {r['Sharpe']:.4f} ({delta_sharpe:+.4f}), "
              f"MaxDD: {r['MaxDD']*100:.2f}% ({delta_maxdd:+.2f}pp), "
              f"Worst5Y: {r['Worst5Y']*100:.2f}%")

    print_results(results)

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'improvement_results_r3.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
