"""
Strategy Improvement Round 4 - Final Combination
=================================================
Combine the two best findings from R3:
1. MomDecel(60/180): Sharpe 0.8788, Worst5Y -0.40% (best Sharpe)
2. CPPI+AsymHAR: MaxDD -47.85%, Sharpe 0.8748 (best MaxDD with Sharpe gain)

Test various combinations and parameter refinements.
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
    calc_momentum_decel_mult
)
from test_improvements import (
    DATA_PATH, DEFAULT_DELAY, ANNUAL_COST,
    strategy_baseline_momdecel_ens2_st, calc_asym_har_vol
)

DEFAULT_THRESHOLD = 0.20


# =============================================================================
# R4-1: MomDecel(60/180) + CPPI + AsymHAR (full combination)
# =============================================================================
def strategy_r4_full_combo(close, returns):
    """R4-1: MomDecel(60/180) + CPPI cushion + AsymHAR Vol.

    Combines ALL three best improvements:
    - AsymHAR vol (R1: Worst5Y improvement)
    - CPPI cushion (R2/R3: MaxDD improvement)
    - MomDecel(60/180) (R3: Sharpe + Worst5Y improvement)
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # CPPI cushion
    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    cppi_mult = ((ratio - 0.82) / (0.90 - 0.82)).clip(0.3, 1.0)

    # AsymHAR Vol
    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)

    slope_mult = calc_slope_multiplier(close)

    # MomDecel(60/180)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R4-2: MomDecel(60/180) + CPPI (original AsymEWMA vol, not HAR)
# =============================================================================
def strategy_r4_md60_cppi_asym(close, returns):
    """R4-2: MomDecel(60/180) + CPPI + standard AsymEWMA.

    Keep original AsymEWMA vol (proven), add CPPI + slower MomDecel.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    cppi_mult = ((ratio - 0.82) / (0.90 - 0.82)).clip(0.3, 1.0)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)

    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R4-3: MomDecel(60/180) only (simplest change from baseline)
# =============================================================================
def strategy_r4_md60_only(close, returns):
    """R4-3: Just change MomDecel from (40/120) to (60/180). Minimal change."""
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
# R4-4: CPPI gentler (0.85-0.92 range) + MomDecel(60/180) + AsymHAR
# =============================================================================
def strategy_r4_cppi_gentle(close, returns):
    """R4-4: Gentler CPPI (starts at -8% DD) + MomDecel(60/180) + AsymHAR."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    # Gentler: starts reducing at ratio=0.92 (8% DD), min at 0.85 (15% DD)
    cppi_mult = ((ratio - 0.85) / (0.92 - 0.85)).clip(0.4, 1.0)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)

    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R4-5: MomDecel sensitivity sweep (0.2 vs 0.3 vs 0.4) with (60/180)
# =============================================================================
def strategy_r4_md60_sens02(close, returns):
    """R4-5a: MomDecel(60/180) with lower sensitivity (0.2)."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.2, min_mult=0.6, max_mult=1.2)
    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_r4_md60_sens04(close, returns):
    """R4-5b: MomDecel(60/180) with higher sensitivity (0.4)."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.4, min_mult=0.4, max_mult=1.4)
    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# R4-6: Best candidate with vol-adjusted CPPI
# =============================================================================
def strategy_r4_best_vol_cppi(close, returns):
    """R4-6: MomDecel(60/180) + vol-adjusted CPPI + AsymHAR.

    CPPI multiplier adjusts with vol: low vol -> less CPPI reduction.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    base_cppi = ((ratio - 0.82) / (0.90 - 0.82)).clip(0.3, 1.0)

    # Vol-adjusted: low vol -> boost CPPI toward 1, high vol -> keep aggressive
    ewma_vol_21 = calc_ewma_vol(returns, 21)
    vol_ma_126 = ewma_vol_21.rolling(126).mean()
    vol_rel = (ewma_vol_21 / vol_ma_126.replace(0, 0.001)).clip(0.5, 2.0).fillna(1.0)
    vol_adj = (1.3 - 0.3 * vol_rel).clip(0.8, 1.2)
    cppi_mult = (base_cppi * vol_adj).clip(0.3, 1.0)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)

    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
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
    print("\n" + "="*135)
    print("ROUND 4: FINAL COMBINATION (1974-2026, 5-day delay, 1.5% cost)")
    print("="*135)
    print(df.to_string(index=False))
    print("="*135)

    # Highlight improvements vs baseline
    baseline = results[0]
    print("\n--- Delta vs Baseline ---")
    for r in results[1:]:
        ds = r['Sharpe'] - baseline['Sharpe']
        dm = (r['MaxDD'] - baseline['MaxDD']) * 100
        dw = (r['Worst5Y'] - baseline['Worst5Y']) * 100
        dc = (r['CAGR'] - baseline['CAGR']) * 100
        print(f"  {r['Strategy']}: Sharpe {ds:+.4f}, MaxDD {dm:+.2f}pp, "
              f"Worst5Y {dw:+.2f}pp, CAGR {dc:+.2f}pp")


def main():
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)")

    results = []
    strategies = [
        (strategy_baseline_momdecel_ens2_st, "Baseline: MD(40/120)+Ens2(S+T)"),
        (strategy_r4_md60_only, "R4-3: MD(60/180) only"),
        (strategy_r4_md60_sens02, "R4-5a: MD(60/180) sens=0.2"),
        (strategy_r4_md60_sens04, "R4-5b: MD(60/180) sens=0.4"),
        (strategy_r4_md60_cppi_asym, "R4-2: MD(60/180)+CPPI+AsymEWMA"),
        (strategy_r4_full_combo, "R4-1: MD(60/180)+CPPI+AsymHAR"),
        (strategy_r4_cppi_gentle, "R4-4: MD(60/180)+GentleCPPI+AsymHAR"),
        (strategy_r4_best_vol_cppi, "R4-6: MD(60/180)+VolCPPI+AsymHAR"),
    ]

    for i, (func, name) in enumerate(strategies, 1):
        print(f"[{i}/{len(strategies)}] {name}...")
        r = run_test(close, dates, func, name)
        results.append(r)
        print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, "
              f"MaxDD: {r['MaxDD']*100:.2f}%, Worst5Y: {r['Worst5Y']*100:.2f}%")

    print_results(results)

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'improvement_results_r4.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
