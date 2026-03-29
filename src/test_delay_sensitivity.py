"""
Delay Sensitivity Analysis for A2 + Dyn-Hybrid
================================================
Tests execution delays of [0, 1, 2, 3, 5, 7] business days
to quantify the benefit of faster execution (e.g., TQQQ vs 投資信託).

Current assumption: 5 business days (NASDAQ100 3倍ブル 投信)
Target: 1-2 business days (TQQQ 米国ETF)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import load_data, calc_dd_signal, calc_ewma_vol, calc_metrics
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult, ANNUAL_COST, BASE_LEVERAGE
)
from test_improvements import DATA_PATH
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import (
    prepare_gold_data, prepare_bond_data, calc_portfolio_metrics
)

DEFAULT_THRESHOLD = 0.20
DELAYS_TO_TEST = [0, 1, 2, 3, 5, 7]


# =============================================================================
# A2 signal extraction (delay-independent: signals computed once)
# =============================================================================
def get_a2_raw_signals(close):
    """Compute A2 raw signals (before delay/threshold application)."""
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

    return raw_leverage, dd_signal, vix_z.fillna(0)


def alloc_dyn_hybrid(raw_leverage_vals, vix_z_vals):
    """Dyn-Hybrid portfolio weights from A2 signals."""
    n = len(raw_leverage_vals)
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lev = raw_leverage_vals[i]
        vz = max(vix_z_vals[i], 0)
        w = 0.50 + 0.30 * lev - 0.10 * vz
        w = np.clip(w, 0.30, 0.90)
        wn[i] = w
        wg[i] = (1 - w) * 0.55
        wb[i] = (1 - w) * 0.45
    return wn, wg, wb


def build_dynamic_portfolio(nasdaq_nav, gold_prices, bond_nav,
                            w_nasdaq, w_gold, w_bond):
    """Build portfolio with dynamic weights, rebalance on >10% drift or quarterly."""
    n = len(nasdaq_nav)
    nasdaq_ret = np.zeros(n)
    gold_ret = np.zeros(n)
    bond_ret = np.zeros(n)
    for i in range(1, n):
        nasdaq_ret[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        gold_ret[i] = gold_prices[i] / gold_prices[i-1] - 1 if gold_prices[i-1] > 0 else 0
        bond_ret[i] = bond_nav[i] / bond_nav[i-1] - 1 if bond_nav[i-1] > 0 else 0

    portfolio_nav = np.ones(n)
    cw_n, cw_g, cw_b = w_nasdaq[0], w_gold[0], w_bond[0]
    for i in range(1, n):
        port_ret = cw_n * nasdaq_ret[i] + cw_g * gold_ret[i] + cw_b * bond_ret[i]
        portfolio_nav[i] = portfolio_nav[i-1] * (1 + port_ret)
        total = cw_n * (1 + nasdaq_ret[i]) + cw_g * (1 + gold_ret[i]) + cw_b * (1 + bond_ret[i])
        if total > 0:
            cw_n = cw_n * (1 + nasdaq_ret[i]) / total
            cw_g = cw_g * (1 + gold_ret[i]) / total
            cw_b = cw_b * (1 + bond_ret[i]) / total
        tw_n, tw_g, tw_b = w_nasdaq[i], w_gold[i], w_bond[i]
        drift = abs(cw_n - tw_n) + abs(cw_g - tw_g) + abs(cw_b - tw_b)
        if drift > 0.10 or i % 63 == 0:
            cw_n, cw_g, cw_b = tw_n, tw_g, tw_b
    return portfolio_nav


def calc_worst_5year_return(nav):
    """Calculate worst 5-year rolling return."""
    if len(nav) < 1260:
        return None
    worst = float('inf')
    for i in range(1260, len(nav)):
        ret = nav[i] / nav[i - 1260] - 1
        annual = (1 + ret) ** (1/5) - 1
        if annual < worst:
            worst = annual
    return worst


def main():
    print("=" * 90)
    print("DELAY SENSITIVITY ANALYSIS: A2 + Dyn-Hybrid")
    print("=" * 90)
    print("Purpose: Quantify Sharpe improvement from faster execution")
    print("  NASDAQ100 3倍ブル (投信): delay=5-6 days")
    print("  TQQQ (米国ETF): delay=1-2 days")
    print("  CFD: delay=0 days")
    print()

    # Load data
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    returns = close.pct_change()
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)\n")

    # ==========================================================================
    # Part 1: A2-only delay sensitivity
    # ==========================================================================
    print("=" * 90)
    print("PART 1: A2 (NASDAQ 3x only) — Delay Sensitivity")
    print("=" * 90)

    raw_leverage, dd_signal, vix_z = get_a2_raw_signals(close)
    lev_thresholded = rebalance_threshold(raw_leverage, DEFAULT_THRESHOLD)

    a2_results = {}
    for delay in DELAYS_TO_TEST:
        nav, strat_ret = run_backtest_realistic(close, lev_thresholded, delay, ANNUAL_COST)
        metrics = calc_metrics(nav, strat_ret, dd_signal, dates)
        w5y = calc_worst_5year_return(nav.values)
        a2_results[delay] = {
            'Sharpe': metrics['Sharpe'],
            'CAGR': metrics['CAGR'],
            'MaxDD': metrics['MaxDD'],
            'Worst5Y': w5y,
        }

    print(f"\n{'Delay':>6} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Worst5Y':>8} {'vs d5':>8}")
    print("-" * 55)
    baseline_sharpe = a2_results[5]['Sharpe']
    for delay in DELAYS_TO_TEST:
        r = a2_results[delay]
        diff = r['Sharpe'] - baseline_sharpe
        w5y_str = f"{r['Worst5Y']:.2%}" if r['Worst5Y'] is not None else 'N/A'
        print(f"{delay:>6d} {r['Sharpe']:>8.4f} {r['CAGR']:>7.1%} {r['MaxDD']:>7.1%} "
              f"{w5y_str:>8} {diff:>+8.4f}")

    # ==========================================================================
    # Part 2: Dyn-Hybrid portfolio — Delay Sensitivity
    # ==========================================================================
    print("\n" + "=" * 90)
    print("PART 2: Dyn-Hybrid Portfolio (NASDAQ 3x + Gold + Bond) — Delay Sensitivity")
    print("=" * 90)

    # Prepare Gold & Bond data
    print("Fetching Gold/Bond data...")
    gold_prices = prepare_gold_data(dates)
    bond_nav = prepare_bond_data(dates)

    hybrid_results = {}
    for delay in DELAYS_TO_TEST:
        # Run A2 backtest with this delay
        nav_a2, _ = run_backtest_realistic(close, lev_thresholded, delay, ANNUAL_COST)

        # Compute portfolio weights from raw signals (delayed too)
        delayed_lev = raw_leverage.shift(delay).fillna(0).values
        delayed_vix_z = vix_z.shift(delay).fillna(0).values
        wn, wg, wb = alloc_dyn_hybrid(delayed_lev, delayed_vix_z)

        # Build dynamic portfolio
        portfolio_nav = build_dynamic_portfolio(
            nav_a2.values, gold_prices, bond_nav, wn, wg, wb
        )

        m = calc_portfolio_metrics(portfolio_nav, dates, f"Dyn-Hybrid d={delay}")
        w5y = calc_worst_5year_return(portfolio_nav)
        hybrid_results[delay] = {
            'Sharpe': m['Sharpe'],
            'CAGR': m['CAGR'],
            'MaxDD': m['MaxDD'],
            'Worst5Y': w5y,
        }

    print(f"\n{'Delay':>6} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Worst5Y':>8} {'vs d5':>8}")
    print("-" * 55)
    baseline_sharpe_h = hybrid_results[5]['Sharpe']
    for delay in DELAYS_TO_TEST:
        r = hybrid_results[delay]
        diff = r['Sharpe'] - baseline_sharpe_h
        w5y_str = f"{r['Worst5Y']:.2%}" if r['Worst5Y'] is not None else 'N/A'
        print(f"{delay:>6d} {r['Sharpe']:>8.4f} {r['CAGR']:>7.1%} {r['MaxDD']:>7.1%} "
              f"{w5y_str:>8} {diff:>+8.4f}")

    # ==========================================================================
    # Part 3: OOS analysis (2021-2026) at each delay
    # ==========================================================================
    print("\n" + "=" * 90)
    print("PART 3: Out-of-Sample (2021-2026) — Delay Sensitivity")
    print("=" * 90)

    oos_start = dates[dates.dt.year >= 2021].index[0]

    print(f"\n{'Delay':>6} {'A2 OOS':>10} {'Hybrid OOS':>12}")
    print("-" * 35)
    for delay in DELAYS_TO_TEST:
        # A2 OOS
        nav_a2, strat_ret_a2 = run_backtest_realistic(close, lev_thresholded, delay, ANNUAL_COST)
        oos_nav_a2 = nav_a2.iloc[oos_start:]
        oos_ret_a2 = strat_ret_a2.iloc[oos_start:]
        oos_nav_norm = oos_nav_a2 / oos_nav_a2.iloc[0]
        n_years = len(oos_nav_a2) / 252
        oos_ret_annual = oos_nav_norm.iloc[-1] ** (1/n_years) - 1
        oos_std = oos_ret_a2.std() * np.sqrt(252)
        a2_oos_sharpe = oos_ret_annual / oos_std if oos_std > 0 else 0

        # Hybrid OOS
        delayed_lev = raw_leverage.shift(delay).fillna(0).values
        delayed_vix_z = vix_z.shift(delay).fillna(0).values
        wn, wg, wb = alloc_dyn_hybrid(delayed_lev, delayed_vix_z)
        portfolio_nav = build_dynamic_portfolio(
            nav_a2.values, gold_prices, bond_nav, wn, wg, wb
        )
        oos_pnav = portfolio_nav[oos_start:]
        oos_pnav_norm = oos_pnav / oos_pnav[0]
        oos_pret = np.diff(oos_pnav) / oos_pnav[:-1]
        oos_pret_annual = oos_pnav_norm[-1] ** (1/n_years) - 1
        oos_pstd = np.std(oos_pret) * np.sqrt(252)
        hybrid_oos_sharpe = oos_pret_annual / oos_pstd if oos_pstd > 0 else 0

        print(f"{delay:>6d} {a2_oos_sharpe:>10.4f} {hybrid_oos_sharpe:>12.4f}")

    # ==========================================================================
    # Summary & Recommendations
    # ==========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY: EXECUTION SPEED vs SHARPE")
    print("=" * 90)

    print("""
Product Mapping:
  delay=0  : CFD (即時約定)
  delay=1  : TQQQ 成行注文→翌日反映 (best case)
  delay=2  : TQQQ 通常ケース / 東証ETF
  delay=3  : TQQQ worst case (祝日等)
  delay=5  : NASDAQ100 3倍ブル (投資信託) — 現在の前提
  delay=7  : 投資信託 worst case (申込不可日等)
""")

    print(f"{'':>20} {'A2 Sharpe':>12} {'Hybrid Sharpe':>14} {'A2 MaxDD':>10} {'Hybrid MaxDD':>12}")
    print("-" * 70)
    for delay in DELAYS_TO_TEST:
        a2 = a2_results[delay]
        hy = hybrid_results[delay]
        label = {0: 'CFD', 1: 'TQQQ best', 2: 'TQQQ/東証ETF',
                 3: 'TQQQ worst', 5: '投信 (現行)', 7: '投信 worst'}
        name = f"d={delay} ({label.get(delay, '')})"
        print(f"{name:>20} {a2['Sharpe']:>12.4f} {hy['Sharpe']:>14.4f} "
              f"{a2['MaxDD']:>9.1%} {hy['MaxDD']:>11.1%}")

    # Improvement from d=5 → d=2
    a2_improvement = a2_results[2]['Sharpe'] - a2_results[5]['Sharpe']
    hy_improvement = hybrid_results[2]['Sharpe'] - hybrid_results[5]['Sharpe']
    print(f"\n★ TQQQ採用 (d=5→d=2) の改善:")
    print(f"  A2 Sharpe: {a2_results[5]['Sharpe']:.4f} → {a2_results[2]['Sharpe']:.4f} ({a2_improvement:+.4f})")
    print(f"  Hybrid Sharpe: {hybrid_results[5]['Sharpe']:.4f} → {hybrid_results[2]['Sharpe']:.4f} ({hy_improvement:+.4f})")
    print(f"  A2 MaxDD: {a2_results[5]['MaxDD']:.1%} → {a2_results[2]['MaxDD']:.1%}")
    print(f"  Hybrid MaxDD: {hybrid_results[5]['MaxDD']:.1%} → {hybrid_results[2]['MaxDD']:.1%}")


if __name__ == '__main__':
    main()
