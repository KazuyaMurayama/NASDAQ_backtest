"""
Dynamic Portfolio Allocation
==============================
A2 signal-driven dynamic allocation between NASDAQ 3x, Gold, and Bonds.
Uses raw_leverage and vix_z from A2 to shift weights dynamically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import load_data, calc_dd_signal, calc_ewma_vol
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult
)
from test_improvements import DATA_PATH, DEFAULT_DELAY, ANNUAL_COST
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import (
    prepare_gold_data, prepare_bond_data, calc_portfolio_metrics
)

DEFAULT_THRESHOLD = 0.20


# =============================================================================
# A2 signal extraction (raw_leverage, vix_z, dd_state)
# =============================================================================
def get_a2_signals(close, dates):
    """Get A2 strategy NAV plus internal signals for dynamic allocation."""
    returns = close.pct_change()
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

    lev = rebalance_threshold(raw_leverage, DEFAULT_THRESHOLD)
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)

    return {
        'nav': nav.values,
        'ret': strat_ret.values,
        'raw_leverage': raw_leverage.values,
        'dd_signal': dd_signal.values,
        'vix_z': vix_z.fillna(0).values,
    }


# =============================================================================
# Dynamic portfolio builder
# =============================================================================
def build_dynamic_portfolio(nasdaq_nav, gold_prices, bond_nav,
                            w_nasdaq_daily, w_gold_daily, w_bond_daily):
    """Build portfolio with daily-varying target weights.
    Rebalances whenever target weights change significantly (>5%)."""
    n = len(nasdaq_nav)
    nasdaq_ret = np.zeros(n)
    gold_ret = np.zeros(n)
    bond_ret = np.zeros(n)

    for i in range(1, n):
        nasdaq_ret[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        gold_ret[i] = gold_prices[i] / gold_prices[i-1] - 1 if gold_prices[i-1] > 0 else 0
        bond_ret[i] = bond_nav[i] / bond_nav[i-1] - 1 if bond_nav[i-1] > 0 else 0

    portfolio_nav = np.ones(n)
    cw_n, cw_g, cw_b = w_nasdaq_daily[0], w_gold_daily[0], w_bond_daily[0]

    for i in range(1, n):
        port_ret = cw_n * nasdaq_ret[i] + cw_g * gold_ret[i] + cw_b * bond_ret[i]
        portfolio_nav[i] = portfolio_nav[i-1] * (1 + port_ret)

        total = cw_n * (1 + nasdaq_ret[i]) + cw_g * (1 + gold_ret[i]) + cw_b * (1 + bond_ret[i])
        if total > 0:
            cw_n = cw_n * (1 + nasdaq_ret[i]) / total
            cw_g = cw_g * (1 + gold_ret[i]) / total
            cw_b = cw_b * (1 + bond_ret[i]) / total

        # Rebalance when target changed significantly OR periodic (quarterly)
        tw_n, tw_g, tw_b = w_nasdaq_daily[i], w_gold_daily[i], w_bond_daily[i]
        drift = abs(cw_n - tw_n) + abs(cw_g - tw_g) + abs(cw_b - tw_b)
        if drift > 0.10 or i % 63 == 0:
            cw_n, cw_g, cw_b = tw_n, tw_g, tw_b

    return portfolio_nav


# =============================================================================
# Static portfolio builder (from previous test, for reference)
# =============================================================================
def build_static_portfolio(nasdaq_nav, gold_prices, bond_nav, wn, wg, wb):
    n = len(nasdaq_nav)
    nasdaq_ret = np.zeros(n)
    gold_ret = np.zeros(n)
    bond_ret = np.zeros(n)
    for i in range(1, n):
        nasdaq_ret[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        gold_ret[i] = gold_prices[i] / gold_prices[i-1] - 1 if gold_prices[i-1] > 0 else 0
        bond_ret[i] = bond_nav[i] / bond_nav[i-1] - 1 if bond_nav[i-1] > 0 else 0
    portfolio_nav = np.ones(n)
    cw_n, cw_g, cw_b = wn, wg, wb
    for i in range(1, n):
        port_ret = cw_n * nasdaq_ret[i] + cw_g * gold_ret[i] + cw_b * bond_ret[i]
        portfolio_nav[i] = portfolio_nav[i-1] * (1 + port_ret)
        total = cw_n * (1 + nasdaq_ret[i]) + cw_g * (1 + gold_ret[i]) + cw_b * (1 + bond_ret[i])
        if total > 0:
            cw_n = cw_n * (1 + nasdaq_ret[i]) / total
            cw_g = cw_g * (1 + gold_ret[i]) / total
            cw_b = cw_b * (1 + bond_ret[i]) / total
        if i % 63 == 0:
            cw_n, cw_g, cw_b = wn, wg, wb
    return portfolio_nav


# =============================================================================
# Dynamic allocation strategies
# =============================================================================
def alloc_dyn_lev1(signals):
    """Dyn-Lev1: raw_leverage-based allocation (aggressive shifts)."""
    n = len(signals['raw_leverage'])
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lev = signals['raw_leverage'][i]
        if lev > 0.7:
            wn[i], wg[i], wb[i] = 0.85, 0.08, 0.07
        elif lev > 0.3:
            wn[i], wg[i], wb[i] = 0.70, 0.15, 0.15
        else:
            wn[i], wg[i], wb[i] = 0.40, 0.30, 0.30
    return wn, wg, wb


def alloc_dyn_lev2(signals):
    """Dyn-Lev2: raw_leverage-based allocation (moderate shifts)."""
    n = len(signals['raw_leverage'])
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lev = signals['raw_leverage'][i]
        if lev > 0.7:
            wn[i], wg[i], wb[i] = 0.80, 0.10, 0.10
        elif lev > 0.3:
            wn[i], wg[i], wb[i] = 0.70, 0.15, 0.15
        else:
            wn[i], wg[i], wb[i] = 0.55, 0.22, 0.23
    return wn, wg, wb


def alloc_dyn_vix(signals):
    """Dyn-VIX: vix_z-based allocation."""
    n = len(signals['vix_z'])
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        vz = signals['vix_z'][i]
        if vz < -0.5:
            wn[i], wg[i], wb[i] = 0.85, 0.08, 0.07
        elif vz < 1.0:
            wn[i], wg[i], wb[i] = 0.70, 0.15, 0.15
        else:
            wn[i], wg[i], wb[i] = 0.50, 0.25, 0.25
    return wn, wg, wb


def alloc_dyn_dd(signals):
    """Dyn-DD: When DD=CASH, shift NASDAQ allocation to Gold/Bond."""
    n = len(signals['dd_signal'])
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        if signals['dd_signal'][i] > 0.5:  # HOLD
            wn[i], wg[i], wb[i] = 0.70, 0.15, 0.15
        else:  # CASH
            wn[i], wg[i], wb[i] = 0.00, 0.50, 0.50
    return wn, wg, wb


def alloc_dyn_hybrid(signals):
    """Dyn-Hybrid: Combined leverage + VIX signal."""
    n = len(signals['raw_leverage'])
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lev = signals['raw_leverage'][i]
        vz = max(signals['vix_z'][i], 0)  # Only penalize high VIX
        w = 0.50 + 0.30 * lev - 0.10 * vz
        w = np.clip(w, 0.30, 0.90)
        wn[i] = w
        wg[i] = (1 - w) * 0.55  # Slightly more Gold than Bond
        wb[i] = (1 - w) * 0.45
    return wn, wg, wb


# =============================================================================
# Grid search for optimal static allocation
# =============================================================================
def grid_search_static(nasdaq_nav, gold_prices, bond_nav, dates):
    """Test all 5% increments of NASDAQ/Gold/Bond allocation."""
    best = None
    results = []
    for wn in range(50, 95, 5):
        for wg in range(5, 35, 5):
            wb = 100 - wn - wg
            if wb < 0 or wb > 30:
                continue
            nav = build_static_portfolio(nasdaq_nav, gold_prices, bond_nav,
                                          wn/100, wg/100, wb/100)
            m = calc_portfolio_metrics(nav, dates, f"{wn}/{wg}/{wb}")
            m['wn'], m['wg'], m['wb'] = wn, wg, wb
            results.append(m)
            if best is None or m['Sharpe'] > best['Sharpe']:
                best = m

    return results, best


# =============================================================================
# Main
# =============================================================================
def main():
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)")

    # Get A2 signals
    print("\nGenerating A2 signals...")
    signals = get_a2_signals(close, dates)
    print(f"  NAV end: {signals['nav'][-1]:.0f}, "
          f"raw_lev mean: {np.mean(signals['raw_leverage']):.3f}")

    # Get Gold/Bond data
    print("Fetching Gold/Bond data...")
    gold_prices = prepare_gold_data(dates)
    bond_nav = prepare_bond_data(dates)

    # =========================================================================
    # Part 1: Grid search for optimal static allocation
    # =========================================================================
    print(f"\n{'=' * 110}")
    print("PART 1: GRID SEARCH — Optimal Static Allocation (5% increments)")
    print("=" * 110)

    grid_results, grid_best = grid_search_static(
        signals['nav'], gold_prices, bond_nav, dates)

    # Top 10 by Sharpe
    grid_df = pd.DataFrame(grid_results).sort_values('Sharpe', ascending=False)
    print(f"\nTop 10 Static Allocations by Sharpe:")
    print(f"{'Alloc':<12} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} {'OOS_Sh':>8}")
    print("-" * 55)
    for _, r in grid_df.head(10).iterrows():
        print(f"{r['Strategy']:<12} {r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% "
              f"{r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>6.2f}% {r['OOS_Sharpe']:>8.4f}")

    print(f"\n  Grid Best: {grid_best['Strategy']} → Sharpe {grid_best['Sharpe']:.4f}")

    # =========================================================================
    # Part 2: Dynamic allocation strategies
    # =========================================================================
    print(f"\n{'=' * 110}")
    print("PART 2: DYNAMIC ALLOCATION STRATEGIES")
    print("=" * 110)

    dyn_strategies = [
        ("A2 100% (reference)", None),
        ("Static 70/15/15", None),
        (f"Static Best ({grid_best['Strategy']})", None),
        ("Dyn-Lev1 (aggressive)", alloc_dyn_lev1),
        ("Dyn-Lev2 (moderate)", alloc_dyn_lev2),
        ("Dyn-VIX", alloc_dyn_vix),
        ("Dyn-DD (CASH→Gold/Bond)", alloc_dyn_dd),
        ("Dyn-Hybrid (lev+vix)", alloc_dyn_hybrid),
    ]

    results = []
    for name, alloc_func in dyn_strategies:
        if name == "A2 100% (reference)":
            nav = signals['nav']
        elif name == "Static 70/15/15":
            nav = build_static_portfolio(signals['nav'], gold_prices, bond_nav, 0.70, 0.15, 0.15)
        elif name.startswith("Static Best"):
            nav = build_static_portfolio(signals['nav'], gold_prices, bond_nav,
                                          grid_best['wn']/100, grid_best['wg']/100, grid_best['wb']/100)
        else:
            wn, wg, wb = alloc_func(signals)
            nav = build_dynamic_portfolio(signals['nav'], gold_prices, bond_nav, wn, wg, wb)

        m = calc_portfolio_metrics(nav, dates, name)
        results.append(m)

    # Print
    ref = results[0]
    static_best = results[1]
    print(f"\n{'Strategy':<30} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} | {'OOS_Sh':>7} {'Y2022':>7}")
    print("-" * 100)
    for r in results:
        flag = ""
        if r != ref:
            vs_static = (r['Sharpe'] > static_best['Sharpe'] and
                        r['MaxDD'] > static_best['MaxDD'] and
                        r['Worst5Y'] > static_best['Worst5Y'])
            if vs_static and r['OOS_Sharpe'] > static_best['OOS_Sharpe']:
                flag = " ★★★(vs Static)"
            elif r['Sharpe'] > ref['Sharpe'] and r['MaxDD'] > ref['MaxDD']:
                flag = " ★★"
        print(f"{r['Strategy']:<30} {r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% "
              f"{r['Worst5Y']*100:>6.2f}% | {r['OOS_Sharpe']:>7.4f} {r['Y2022']:>6.1f}%{flag}")

    # Crisis year analysis for top candidates
    print(f"\n{'=' * 110}")
    print("CRISIS YEAR RETURNS (%)")
    print("=" * 110)
    crisis_years = [1974, 1987, 2000, 2008, 2011, 2022]
    print(f"{'Strategy':<30}", end="")
    for y in crisis_years:
        print(f" {y:>7}", end="")
    print()
    print("-" * 80)

    for name, alloc_func in dyn_strategies:
        if name == "A2 100% (reference)":
            nav = signals['nav']
        elif name == "Static 70/15/15":
            nav = build_static_portfolio(signals['nav'], gold_prices, bond_nav, 0.70, 0.15, 0.15)
        elif name.startswith("Static Best"):
            nav = build_static_portfolio(signals['nav'], gold_prices, bond_nav,
                                          grid_best['wn']/100, grid_best['wg']/100, grid_best['wb']/100)
        else:
            wn, wg, wb = alloc_func(signals)
            nav = build_dynamic_portfolio(signals['nav'], gold_prices, bond_nav, wn, wg, wb)

        nav_s = pd.Series(nav, index=dates.index)
        print(f"{name:<30}", end="")
        for y in crisis_years:
            mask = (dates.dt.year == y)
            if mask.any():
                yr_nav = nav_s[mask]
                yr_ret = (yr_nav.iloc[-1] / yr_nav.iloc[0] - 1) * 100
                print(f" {yr_ret:>6.1f}%", end="")
        print()

    # Save
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.DataFrame(results).to_csv(os.path.join(base_dir, 'dynamic_portfolio_results.csv'), index=False)
    grid_df.to_csv(os.path.join(base_dir, 'grid_search_results.csv'), index=False)
    print(f"\nSaved to dynamic_portfolio_results.csv, grid_search_results.csv")


if __name__ == '__main__':
    main()
