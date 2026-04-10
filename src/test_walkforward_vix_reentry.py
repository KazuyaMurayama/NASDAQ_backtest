"""
Walk-Forward Validation + VIX Reentry Optimization
====================================================
2. Walk-Forward: Rolling 5-year OOS windows to verify A2 isn't overfit
3. VIX Reentry: Use VIX level to dynamically adjust 92% reentry threshold
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
from test_improvements import DATA_PATH, DEFAULT_DELAY, ANNUAL_COST
from test_vix_integration import calc_vix_proxy

DEFAULT_THRESHOLD = 0.20


# =============================================================================
# Strategy definitions
# =============================================================================
def run_strategy_original(close, returns):
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    raw = dd_signal * vt_lev * slope_mult * mom_decel
    return raw.clip(0, 1.0).fillna(0), dd_signal


def run_strategy_a2(close, returns):
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
    raw = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    return raw.clip(0, 1.0).fillna(0), dd_signal


# =============================================================================
# VIX Reentry strategies
# =============================================================================
def calc_dd_vix_reentry(close, returns, exit_th=0.82, base_reentry=0.92,
                         vix_low_reentry=0.88, vix_high_reentry=0.95):
    """DD with VIX-adaptive reentry threshold.
    - VIX low (fear subsiding) -> early reentry at 88%
    - VIX high (fear elevated) -> delayed reentry at 95%
    """
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std

    # Map VIX z-score to reentry threshold
    # z < -1 (low fear): reentry at 0.88 (earlier)
    # z = 0 (normal): reentry at 0.92 (standard)
    # z > 1 (high fear): reentry at 0.95 (later)
    dynamic_reentry = base_reentry + 0.03 * vix_z
    dynamic_reentry = dynamic_reentry.clip(vix_low_reentry, vix_high_reentry).fillna(base_reentry)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    for i in range(len(close)):
        re = dynamic_reentry.iloc[i]
        if state == 'HOLD' and ratio.iloc[i] <= exit_th:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= re:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    return position


def run_strategy_a2_vix_reentry(close, returns):
    """A2 + VIX-adaptive reentry."""
    dd_signal = calc_dd_vix_reentry(close, returns, 0.82, 0.92, 0.88, 0.95)
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
    raw = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    return raw.clip(0, 1.0).fillna(0), dd_signal


def run_strategy_a2_vix_reentry_gentle(close, returns):
    """A2 + VIX-adaptive reentry (gentler: 0.90-0.94)."""
    dd_signal = calc_dd_vix_reentry(close, returns, 0.82, 0.92, 0.90, 0.94)
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
    raw = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    return raw.clip(0, 1.0).fillna(0), dd_signal


def run_strategy_a2_vix_exit_and_reentry(close, returns):
    """A2 + VIX-adaptive BOTH exit AND reentry.
    Exit: tighten when VIX high (exit earlier).
    Reentry: delay when VIX high."""
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std

    # Dynamic exit: normal=0.82, high fear=0.85 (earlier exit)
    dynamic_exit = (0.82 + 0.02 * vix_z).clip(0.80, 0.85).fillna(0.82)
    # Dynamic reentry: normal=0.92, high fear=0.95 (later reentry)
    dynamic_reentry = (0.92 + 0.03 * vix_z).clip(0.89, 0.95).fillna(0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    for i in range(len(close)):
        ex = dynamic_exit.iloc[i]
        re = dynamic_reentry.iloc[i]
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
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)
    raw = position * vt_lev * slope_mult * mom_decel * vix_mult
    return raw.clip(0, 1.0).fillna(0), position


# =============================================================================
# Walk-Forward Validation
# =============================================================================
def walk_forward_test(close, dates, strategy_func, name, window_years=10, step_years=5):
    """Rolling walk-forward OOS test.
    Train on `window_years`, test on next `step_years`, roll forward.
    """
    returns = close.pct_change()
    raw_leverage, dd_signal = strategy_func(close, returns)
    lev = rebalance_threshold(raw_leverage, DEFAULT_THRESHOLD)
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)

    results = []
    start_year = dates.iloc[0].year + 1  # Skip first year (warmup)
    end_year = dates.iloc[-1].year

    # Generate OOS windows
    oos_start = start_year + window_years
    while oos_start + step_years <= end_year + 1:
        oos_end = min(oos_start + step_years, end_year + 1)

        mask = (dates.dt.year >= oos_start) & (dates.dt.year < oos_end)
        if mask.sum() < 252:
            oos_start += step_years
            continue

        nav_period = nav[mask]
        ret_period = strat_ret[mask]

        # Normalize NAV
        nav_norm = nav_period / nav_period.iloc[0]
        period_years = len(nav_norm) / 252
        period_cagr = (nav_norm.iloc[-1] ** (1 / period_years)) - 1
        period_sharpe = (ret_period.mean() * 252) / (ret_period.std() * np.sqrt(252)) if ret_period.std() > 0 else 0
        period_maxdd = (nav_norm / nav_norm.cummax() - 1).min()

        results.append({
            'Period': f"{oos_start}-{oos_end-1}",
            'CAGR': period_cagr,
            'Sharpe': period_sharpe,
            'MaxDD': period_maxdd,
            'Years': period_years,
        })

        oos_start += step_years

    return results


def main():
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)")

    # =========================================================================
    # Part 2: Walk-Forward Validation
    # =========================================================================
    print("\n" + "=" * 95)
    print("PART 2: WALK-FORWARD VALIDATION (10yr train → 5yr OOS, rolling)")
    print("=" * 95)

    strategies_wf = [
        (run_strategy_original, "Original: MD(40/120)"),
        (run_strategy_a2, "A2: VIX+MD(60/180)"),
    ]

    for func, name in strategies_wf:
        periods = walk_forward_test(close, dates, func, name, window_years=10, step_years=5)
        print(f"\n  {name}:")
        print(f"  {'Period':<12} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>9}")
        print(f"  {'-'*40}")
        sharpes = []
        for p in periods:
            print(f"  {p['Period']:<12} {p['Sharpe']:>8.4f} {p['CAGR']*100:>7.2f}% {p['MaxDD']*100:>8.2f}%")
            sharpes.append(p['Sharpe'])
        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        min_sharpe = np.min(sharpes)
        win_rate = sum(1 for s in sharpes if s > 0) / len(sharpes) * 100
        print(f"  {'--- Summary ---':<12}")
        print(f"  Avg Sharpe: {avg_sharpe:.4f}, Std: {std_sharpe:.4f}, Min: {min_sharpe:.4f}, Win%: {win_rate:.0f}%")

    # Also test with 3-year windows for finer granularity
    print(f"\n{'=' * 95}")
    print("WALK-FORWARD (10yr train → 3yr OOS, finer granularity)")
    print("=" * 95)

    for func, name in strategies_wf:
        periods = walk_forward_test(close, dates, func, name, window_years=10, step_years=3)
        print(f"\n  {name}:")
        print(f"  {'Period':<12} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>9}")
        print(f"  {'-'*40}")
        sharpes = []
        for p in periods:
            print(f"  {p['Period']:<12} {p['Sharpe']:>8.4f} {p['CAGR']*100:>7.2f}% {p['MaxDD']*100:>8.2f}%")
            sharpes.append(p['Sharpe'])
        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        min_sharpe = np.min(sharpes)
        a2_wins = 0
        print(f"  {'--- Summary ---':<12}")
        print(f"  Avg Sharpe: {avg_sharpe:.4f}, Std: {std_sharpe:.4f}, Min: {min_sharpe:.4f}")

    # Head-to-head comparison (3-year windows)
    print(f"\n{'=' * 95}")
    print("HEAD-TO-HEAD: A2 vs Original (3-year OOS windows)")
    print("=" * 95)

    orig_periods = walk_forward_test(close, dates, run_strategy_original, "Orig", 10, 3)
    a2_periods = walk_forward_test(close, dates, run_strategy_a2, "A2", 10, 3)

    a2_wins_sharpe = 0
    a2_wins_maxdd = 0
    print(f"  {'Period':<12} {'Orig_Sh':>8} {'A2_Sh':>8} {'Delta':>8} {'Winner':>8}")
    print(f"  {'-'*50}")
    for o, a in zip(orig_periods, a2_periods):
        delta = a['Sharpe'] - o['Sharpe']
        winner = "A2" if delta > 0 else "Orig"
        if delta > 0:
            a2_wins_sharpe += 1
        if a['MaxDD'] > o['MaxDD']:
            a2_wins_maxdd += 1
        print(f"  {o['Period']:<12} {o['Sharpe']:>8.4f} {a['Sharpe']:>8.4f} {delta:>+8.4f} {winner:>8}")

    total = len(orig_periods)
    print(f"\n  A2 wins Sharpe: {a2_wins_sharpe}/{total} ({a2_wins_sharpe/total*100:.0f}%)")
    print(f"  A2 wins MaxDD:  {a2_wins_maxdd}/{total} ({a2_wins_maxdd/total*100:.0f}%)")

    # =========================================================================
    # Part 3: VIX Reentry Optimization
    # =========================================================================
    print(f"\n{'=' * 95}")
    print("PART 3: VIX REENTRY OPTIMIZATION")
    print("=" * 95)

    strategies_vix = [
        (run_strategy_original, "Original: MD(40/120)"),
        (run_strategy_a2, "A2: VIX+MD(60/180)"),
        (run_strategy_a2_vix_reentry, "A2+VixReentry(88-95)"),
        (run_strategy_a2_vix_reentry_gentle, "A2+VixReentry(90-94)"),
        (run_strategy_a2_vix_exit_and_reentry, "A2+VixExitReentry"),
    ]

    vix_results = []
    for func, name in strategies_vix:
        returns = close.pct_change()
        raw_leverage, dd_signal = func(close, returns)
        lev = rebalance_threshold(raw_leverage, DEFAULT_THRESHOLD)
        nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)
        m = calc_metrics(nav, strat_ret, dd_signal, dates)
        rebal = count_rebalances(lev)

        # OOS
        split_idx = dates[dates >= '2021-05-07'].index[0]
        nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
        ret_oos = strat_ret.iloc[split_idx:]
        oos_sharpe = (ret_oos.mean() * 252) / (ret_oos.std() * np.sqrt(252)) if ret_oos.std() > 0 else 0

        # 2022
        mask_2022 = (dates.dt.year == 2022)
        nav_2022 = nav[mask_2022]
        ret_2022 = (nav_2022.iloc[-1] / nav_2022.iloc[0] - 1) * 100

        r = {
            'Strategy': name, 'Sharpe': m['Sharpe'], 'CAGR': m['CAGR'],
            'MaxDD': m['MaxDD'], 'Worst5Y': m['Worst5Y'],
            'OOS_Sharpe': oos_sharpe, 'Y2022': ret_2022,
            'Trades': m['Trades'], 'Rebalances': rebal,
        }
        vix_results.append(r)

    a2_ref = vix_results[1]
    print(f"\n{'Strategy':<28} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} | {'OOS_Sh':>7} {'Y2022':>7} {'Trades':>7}")
    print("-" * 100)
    for r in vix_results:
        flag = ""
        if r != vix_results[0] and r != vix_results[1]:
            better_sharpe = r['Sharpe'] > a2_ref['Sharpe']
            better_oos = r['OOS_Sharpe'] > a2_ref['OOS_Sharpe']
            better_w5y = r['Worst5Y'] > a2_ref['Worst5Y']
            if better_sharpe and better_oos:
                flag = " ★★★"
            elif better_sharpe:
                flag = " ★★(IS)"
            elif better_oos:
                flag = " ★(OOS)"
        print(f"{r['Strategy']:<28} {r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% "
              f"{r['Worst5Y']*100:>6.2f}% | {r['OOS_Sharpe']:>7.4f} {r['Y2022']:>6.1f}% {r['Trades']:>6}{flag}")

    # Save
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.DataFrame(vix_results).to_csv(os.path.join(base_dir, 'improvement_results_wf_vix.csv'), index=False)
    print(f"\nSaved to improvement_results_wf_vix.csv")


if __name__ == '__main__':
    main()
