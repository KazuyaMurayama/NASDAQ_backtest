"""
Strategy Validation: Out-of-Sample + Crisis Analysis + All-4-Metric Winner
==========================================================================
1. Out-of-sample: train on 1974-2021, test on 2021-2026
2. Find strategy that beats baseline on ALL 4 metrics
3. Crisis year breakdown for top candidates
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
from test_improvements_r4 import (
    strategy_r4_md60_only, strategy_r4_full_combo,
    strategy_r4_best_vol_cppi, strategy_r4_cppi_gentle,
    strategy_r4_md60_cppi_asym
)

DEFAULT_THRESHOLD = 0.20


# =============================================================================
# NEW: Strategies targeting ALL 4 metrics beating baseline
# Baseline: Sharpe 0.8728, CAGR 23.21%, MaxDD -50.58%, Worst5Y -0.91%
#
# Problem: CPPI improves Sharpe/CAGR/MaxDD but hurts Worst5Y
# Solution: Make CPPI less aggressive (higher floor) so it protects MaxDD
#           without creating a drag during the worst 5-year periods
# =============================================================================

def strategy_v1_md60_micro_cppi(close, returns):
    """V1: MD(60/180) + Micro CPPI — minimal CPPI that preserves Worst5Y.

    Very gentle CPPI: only reduces at extreme drawdowns (ratio < 0.86).
    This should protect MaxDD without affecting Worst5Y much.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    # Only activate near DD exit: ratio 0.82-0.86
    cppi_mult = ((ratio - 0.82) / (0.86 - 0.82)).clip(0.5, 1.0)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_v2_md60_asym_har_only(close, returns):
    """V2: MD(60/180) + AsymHAR Vol (no CPPI).

    MD(60/180) improves Sharpe+Worst5Y. AsymHAR may also help Worst5Y.
    Without CPPI, Worst5Y should stay good. MaxDD might improve from AsymHAR
    reacting faster in crisis.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_v3_md60_micro_cppi_asym_har(close, returns):
    """V3: MD(60/180) + Micro CPPI + AsymHAR.

    Combine the safest CPPI (micro) with AsymHAR.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    cppi_mult = ((ratio - 0.82) / (0.86 - 0.82)).clip(0.5, 1.0)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


def strategy_v4_md50_150(close, returns):
    """V4: MD(50/150) — between baseline (40/120) and best (60/180)."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=50, long=150,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Out-of-Sample Test
# =============================================================================
def run_oos_test(close_full, dates_full, strategy_func, name,
                 split_date='2021-05-07', threshold=DEFAULT_THRESHOLD):
    """Run backtest on full period and out-of-sample period separately."""
    returns_full = close_full.pct_change()
    raw_leverage, dd_signal = strategy_func(close_full, returns_full)
    lev = rebalance_threshold(raw_leverage, threshold)

    # Full period
    nav_full, strat_ret_full = run_backtest_realistic(close_full, lev, DEFAULT_DELAY, ANNUAL_COST)
    metrics_full = calc_metrics(nav_full, strat_ret_full, dd_signal, dates_full)

    # In-sample (1974 to split_date)
    split_idx = dates_full[dates_full >= split_date].index[0]
    nav_is = nav_full.iloc[:split_idx]
    ret_is = strat_ret_full.iloc[:split_idx]
    dates_is = dates_full.iloc[:split_idx]
    dd_is = dd_signal.iloc[:split_idx]
    metrics_is = calc_metrics(nav_is, ret_is, dd_is, dates_is)

    # Out-of-sample (split_date to end)
    # Need to renormalize NAV for OOS period
    nav_oos_raw = nav_full.iloc[split_idx:]
    nav_oos = nav_oos_raw / nav_oos_raw.iloc[0]  # Normalize to 1.0 at start
    ret_oos = strat_ret_full.iloc[split_idx:]
    dates_oos = dates_full.iloc[split_idx:]
    dd_oos = dd_signal.iloc[split_idx:]

    # OOS metrics (manual for short period)
    oos_days = len(nav_oos)
    oos_years = oos_days / 252
    oos_cagr = (nav_oos.iloc[-1] ** (1 / oos_years)) - 1
    oos_annual_ret = ret_oos.mean() * 252
    oos_annual_vol = ret_oos.std() * np.sqrt(252)
    oos_sharpe = oos_annual_ret / oos_annual_vol if oos_annual_vol > 0 else 0
    oos_maxdd_series = nav_oos / nav_oos.cummax() - 1
    oos_maxdd = oos_maxdd_series.min()

    return {
        'Strategy': name,
        'Full_Sharpe': metrics_full['Sharpe'],
        'Full_CAGR': metrics_full['CAGR'],
        'Full_MaxDD': metrics_full['MaxDD'],
        'Full_Worst5Y': metrics_full['Worst5Y'],
        'IS_Sharpe': metrics_is['Sharpe'],
        'IS_CAGR': metrics_is['CAGR'],
        'OOS_Sharpe': oos_sharpe,
        'OOS_CAGR': oos_cagr,
        'OOS_MaxDD': oos_maxdd,
        'OOS_Years': oos_years,
        'Trades': metrics_full['Trades'],
    }


# =============================================================================
# Crisis Year Analysis
# =============================================================================
def run_crisis_analysis(close, dates, strategy_func, name, threshold=DEFAULT_THRESHOLD):
    """Calculate yearly returns for crisis years."""
    returns = close.pct_change()
    raw_leverage, dd_signal = strategy_func(close, returns)
    lev = rebalance_threshold(raw_leverage, threshold)
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)

    df = pd.DataFrame({'nav': nav.values, 'date': dates.values})
    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly_nav = df.groupby('year')['nav'].agg(['first', 'last'])
    yearly_returns = (yearly_nav['last'] / yearly_nav['first'] - 1) * 100

    crisis_years = [1974, 1987, 2000, 2001, 2002, 2008, 2011, 2018, 2020, 2022, 2025]
    result = {'Strategy': name}
    for y in crisis_years:
        if y in yearly_returns.index:
            result[str(y)] = yearly_returns[y]
        else:
            result[str(y)] = np.nan
    return result


# =============================================================================
# Main
# =============================================================================
def main():
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)\n")

    strategies = [
        (strategy_baseline_momdecel_ens2_st, "Baseline: MD(40/120)"),
        (strategy_r4_md60_only, "R4-3: MD(60/180)"),
        (strategy_v1_md60_micro_cppi, "V1: MD(60)+MicroCPPI"),
        (strategy_v2_md60_asym_har_only, "V2: MD(60)+AsymHAR"),
        (strategy_v3_md60_micro_cppi_asym_har, "V3: MD(60)+MicroCPPI+AsymHAR"),
        (strategy_v4_md50_150, "V4: MD(50/150)"),
        (strategy_r4_best_vol_cppi, "R4-6: MD(60)+VolCPPI+AsymHAR"),
        (strategy_r4_cppi_gentle, "R4-4: MD(60)+GentleCPPI+AsymHAR"),
    ]

    # =========================================================================
    # Part 1: Full period + all-4-metrics check
    # =========================================================================
    print("=" * 100)
    print("PART 1: FULL PERIOD METRICS (1974-2026)")
    print("=" * 100)

    full_results = []
    for func, name in strategies:
        returns = close.pct_change()
        raw_leverage, dd_signal = func(close, returns)
        lev = rebalance_threshold(raw_leverage, DEFAULT_THRESHOLD)
        nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)
        m = calc_metrics(nav, strat_ret, dd_signal, dates)
        m['Strategy'] = name
        full_results.append(m)

    baseline = full_results[0]
    print(f"\n{'Strategy':<35} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'Worst5Y':>8}  All4?")
    print("-" * 80)
    for r in full_results:
        beats_sharpe = r['Sharpe'] > baseline['Sharpe']
        beats_cagr = r['CAGR'] > baseline['CAGR']
        beats_maxdd = r['MaxDD'] > baseline['MaxDD']
        beats_w5y = r['Worst5Y'] > baseline['Worst5Y']
        all4 = beats_sharpe and beats_cagr and beats_maxdd and beats_w5y
        flag = " *** ALL 4 ***" if all4 else ""
        if r == baseline:
            flag = " (baseline)"
        print(f"{r['Strategy']:<35} {r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>7.2f}%{flag}")

    # =========================================================================
    # Part 2: Out-of-Sample Validation
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("PART 2: OUT-OF-SAMPLE VALIDATION (train: 1974-2021, test: 2021-2026)")
    print("=" * 100)

    oos_results = []
    for func, name in strategies:
        r = run_oos_test(close, dates, func, name)
        oos_results.append(r)

    print(f"\n{'Strategy':<35} {'IS_Sharpe':>10} {'OOS_Sharpe':>11} {'OOS_CAGR':>9} {'OOS_MaxDD':>10} {'Degradation':>12}")
    print("-" * 95)
    for r in oos_results:
        degrad = (r['OOS_Sharpe'] - r['IS_Sharpe']) / abs(r['IS_Sharpe']) * 100 if r['IS_Sharpe'] != 0 else 0
        print(f"{r['Strategy']:<35} {r['IS_Sharpe']:>10.4f} {r['OOS_Sharpe']:>11.4f} "
              f"{r['OOS_CAGR']*100:>8.2f}% {r['OOS_MaxDD']*100:>9.2f}% {degrad:>11.1f}%")

    # =========================================================================
    # Part 3: Crisis Year Analysis
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("PART 3: CRISIS YEAR RETURNS (%)")
    print("=" * 100)

    crisis_results = []
    for func, name in strategies:
        r = run_crisis_analysis(close, dates, func, name)
        crisis_results.append(r)

    crisis_df = pd.DataFrame(crisis_results)
    crisis_df = crisis_df.set_index('Strategy')
    for col in crisis_df.columns:
        crisis_df[col] = crisis_df[col].apply(lambda x: f"{x:>7.1f}" if not pd.isna(x) else "   N/A")
    print(f"\n{crisis_df.to_string()}")

    # Save all results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.DataFrame(full_results).to_csv(os.path.join(base_dir, 'validation_full.csv'), index=False)
    pd.DataFrame(oos_results).to_csv(os.path.join(base_dir, 'validation_oos.csv'), index=False)
    pd.DataFrame(crisis_results).to_csv(os.path.join(base_dir, 'validation_crisis.csv'), index=False)
    print(f"\nResults saved to validation_full.csv, validation_oos.csv, validation_crisis.csv")


if __name__ == '__main__':
    main()
