"""
Realistic Product Backtest: Updated with actual product constraints.

New conditions:
  1. Execution delay: 5 business days from signal to position change
  2. Annual fee: 1.5% (daily = 1.5% / 252)

Strategies (focused set):
  [Base]       DD-Only: HOLD 3x or CASH (simplest, ~30 trades in 47yr)
  [Base]       DD+VT(25%): DD control + volatility targeting
  [High Perf]  Ens2(Asym+Slope): Best Sharpe strategy
  [High Perf]  Ens2(Slope+TrendTV): 2nd best Sharpe strategy
  [Comparison] DD+VT+VolSpike(1.5x): 3rd ranked, simpler structure
  [Reference]  Buy & Hold 3x: No strategy, pure leverage

Rebalancing models for partial-sell strategies:
  - Daily (ideal, impractical)
  - Threshold: 10%, 15%, 20% (practical)
  - Periodic: 2-week, monthly
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol,
    calc_vt_leverage, strategy_baseline_bh3x, strategy_baseline_dd_only,
    strategy_baseline_dd_vt, strategy_dd_vt_volspike
)
from test_ens2_strategies import (
    strategy_ens2_asym_slope, strategy_ens2_slope_trendtv
)

# =============================================================================
# Constants - UPDATED
# =============================================================================
ANNUAL_COST = 0.015        # 1.5% annual fee
EXECUTION_DELAY = 5        # 5 business days
BASE_LEVERAGE = 3.0

# =============================================================================
# Backtest with configurable delay and cost
# =============================================================================
def run_backtest_realistic(close: pd.Series, leverage: pd.Series,
                           base_leverage: float = BASE_LEVERAGE,
                           annual_cost: float = ANNUAL_COST,
                           exec_delay: int = EXECUTION_DELAY) -> tuple:
    """
    Realistic backtest:
    - leverage.shift(exec_delay): signal takes N days to become effective
    - Cost charged only when invested
    """
    returns = close.pct_change()
    leveraged_returns = returns * base_leverage
    daily_cost = annual_cost / 252

    # Apply execution delay
    delayed_leverage = leverage.shift(exec_delay)
    strategy_returns = delayed_leverage * (leveraged_returns - daily_cost)
    strategy_returns = strategy_returns.fillna(0)

    nav = (1 + strategy_returns).cumprod()
    return nav, strategy_returns


# =============================================================================
# Rebalancing models (from previous analysis)
# =============================================================================
def rebalance_threshold(leverage: pd.Series, threshold: float) -> pd.Series:
    """Only rebalance when target changes by more than threshold.
    DD exit/reentry always triggers immediately.
    """
    result = pd.Series(0.0, index=leverage.index)
    current = leverage.iloc[0]
    result.iloc[0] = current

    for i in range(1, len(leverage)):
        target = leverage.iloc[i]
        if target == 0.0 and current > 0.0:
            current = 0.0
        elif current == 0.0 and target > 0.0:
            current = target
        elif abs(target - current) > threshold:
            current = target
        result.iloc[i] = current

    return result


def rebalance_periodic(leverage: pd.Series, period_days: int) -> pd.Series:
    """Rebalance every N trading days. DD exits still happen immediately."""
    result = pd.Series(0.0, index=leverage.index)
    current = leverage.iloc[0]
    result.iloc[0] = current
    days_since = 0

    for i in range(1, len(leverage)):
        target = leverage.iloc[i]
        days_since += 1

        if target == 0.0 and current > 0.0:
            current = 0.0
            days_since = 0
        elif current == 0.0 and target > 0.0:
            current = target
            days_since = 0
        elif days_since >= period_days:
            current = target
            days_since = 0

        result.iloc[i] = current

    return result


def count_rebalances(leverage: pd.Series) -> int:
    changes = leverage.diff().abs()
    return (changes > 0.01).sum()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 130)
    print("REALISTIC PRODUCT BACKTEST")
    print(f"  Execution delay: {EXECUTION_DELAY} business days")
    print(f"  Annual cost:     {ANNUAL_COST*100:.1f}%")
    print(f"  Base leverage:   {BASE_LEVERAGE}x")
    print("=" * 130)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'NASDAQ_Dairy_since1973.csv')
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total: {len(df)} days (~{len(df)/252:.0f} years)\n")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    results_all = []

    def test(lev, dd_sig, name, category, delay=EXECUTION_DELAY, cost=ANNUAL_COST):
        nav, strat_ret = run_backtest_realistic(close, lev, BASE_LEVERAGE, cost, delay)
        metrics = calc_metrics(nav, strat_ret, dd_sig, dates)
        rebal = count_rebalances(lev)
        metrics.update({
            'Strategy': name, 'Category': category,
            'Rebalances': rebal, 'Rebal/Year': rebal / (len(df) / 252),
            'Delay': delay, 'Cost': cost
        })
        results_all.append(metrics)
        print(f"  {name:<55s} Sharpe={metrics['Sharpe']:.3f}  CAGR={metrics['CAGR']*100:.1f}%  "
              f"MaxDD={metrics['MaxDD']*100:.1f}%  W5Y={metrics['Worst5Y']*100:.1f}%  "
              f"Rebal={rebal} ({metrics['Rebal/Year']:.1f}/yr)")
        return metrics

    # =====================================================================
    # PART 1: Old conditions vs New conditions (impact comparison)
    # =====================================================================
    print("=" * 130)
    print("PART 1: CONDITION IMPACT COMPARISON (Old vs New)")
    print("  Old: delay=1day, cost=0.9%  |  New: delay=5day, cost=1.5%")
    print("=" * 130)

    strategies = {
        'Buy&Hold 3x': strategy_baseline_bh3x(close, returns),
        'DD-Only': strategy_baseline_dd_only(close, returns, 0.82, 0.92),
        'DD+VT(25%)': strategy_baseline_dd_vt(close, returns, 0.82, 0.92, 0.25, 10),
        'DD+VT+VolSpike': strategy_dd_vt_volspike(close, returns, 0.82, 0.92, 0.25, 10, 1.5),
        'Ens2(Asym+Slope)': strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0),
        'Ens2(Slope+TrendTV)': strategy_ens2_slope_trendtv(close, returns, 0.82, 0.92, 20, 5, 1.0),
    }

    for label, (lev, dd) in strategies.items():
        print(f"\n  [{label}]")
        test(lev, dd, f'{label} (OLD: 1d/0.9%)', 'Old', delay=1, cost=0.009)
        test(lev, dd, f'{label} (NEW: 5d/1.5%)', 'New', delay=5, cost=0.015)

    # =====================================================================
    # PART 2: New conditions - Rebalancing frequency analysis
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 2: REBALANCING FREQUENCY (New conditions: 5-day delay, 1.5% cost)")
    print("=" * 130)

    # --- Ens2(Asym+Slope) ---
    lev_as, dd_as = strategies['Ens2(Asym+Slope)']
    print("\n--- Ens2(Asym+Slope) ---")
    print("[Daily]")
    test(lev_as, dd_as, 'Ens2(A+S) Daily', 'S1-Daily')
    print("[Threshold-based]")
    for th in [0.05, 0.10, 0.15, 0.20, 0.25]:
        lev_th = rebalance_threshold(lev_as, th)
        test(lev_th, dd_as, f'Ens2(A+S) Th={th:.0%}', 'S1-Thresh')
    print("[Periodic]")
    for period, label in [(5, '1wk'), (10, '2wk'), (21, '1mo')]:
        lev_p = rebalance_periodic(lev_as, period)
        test(lev_p, dd_as, f'Ens2(A+S) Per={label}', 'S1-Period')

    # --- Ens2(Slope+TrendTV) ---
    lev_st, dd_st = strategies['Ens2(Slope+TrendTV)']
    print("\n--- Ens2(Slope+TrendTV) ---")
    print("[Daily]")
    test(lev_st, dd_st, 'Ens2(S+T) Daily', 'S2-Daily')
    print("[Threshold-based]")
    for th in [0.05, 0.10, 0.15, 0.20, 0.25]:
        lev_th = rebalance_threshold(lev_st, th)
        test(lev_th, dd_st, f'Ens2(S+T) Th={th:.0%}', 'S2-Thresh')
    print("[Periodic]")
    for period, label in [(5, '1wk'), (10, '2wk'), (21, '1mo')]:
        lev_p = rebalance_periodic(lev_st, period)
        test(lev_p, dd_st, f'Ens2(S+T) Per={label}', 'S2-Period')

    # --- DD+VT+VolSpike ---
    lev_vs, dd_vs = strategies['DD+VT+VolSpike']
    print("\n--- DD+VT+VolSpike(1.5x) ---")
    print("[Daily]")
    test(lev_vs, dd_vs, 'VolSpike Daily', 'S3-Daily')
    print("[Threshold-based]")
    for th in [0.05, 0.10, 0.15, 0.20]:
        lev_th = rebalance_threshold(lev_vs, th)
        test(lev_th, dd_vs, f'VolSpike Th={th:.0%}', 'S3-Thresh')
    print("[Periodic]")
    for period, label in [(5, '1wk'), (10, '2wk'), (21, '1mo')]:
        lev_p = rebalance_periodic(lev_vs, period)
        test(lev_p, dd_vs, f'VolSpike Per={label}', 'S3-Period')

    # --- DD-Only & DD+VT (base) ---
    print("\n--- Base Strategies ---")
    lev_dd, dd_dd = strategies['DD-Only']
    test(lev_dd, dd_dd, 'DD-Only (3x or Cash)', 'Base')

    lev_dv, dd_dv = strategies['DD+VT(25%)']
    test(lev_dv, dd_dv, 'DD+VT(25%) Daily', 'Base')
    for th in [0.10, 0.15, 0.20]:
        lev_th = rebalance_threshold(lev_dv, th)
        test(lev_th, dd_dv, f'DD+VT(25%) Th={th:.0%}', 'Base-Thresh')

    lev_bh, dd_bh = strategies['Buy&Hold 3x']
    test(lev_bh, dd_bh, 'Buy&Hold 3x', 'Ref')

    # =====================================================================
    # PART 3: Summary
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 3: FULL RANKING (New conditions only, sorted by Sharpe)")
    print("=" * 130)

    # Filter to new-condition results only (exclude Old comparisons)
    new_results = [r for r in results_all
                   if r.get('Category') not in ('Old',)]
    # Also remove the Part1 "New" entries since they duplicate Part2
    new_results = [r for r in new_results
                   if not r['Strategy'].endswith('(NEW: 5d/1.5%)')]
    # Add back the Part1 New entries for strategies not in Part2
    for r in results_all:
        if r['Strategy'] == 'Buy&Hold 3x (NEW: 5d/1.5%)':
            r2 = r.copy()
            r2['Strategy'] = 'Buy&Hold 3x'
            r2['Category'] = 'Ref'

    results_df = pd.DataFrame(new_results)
    cols = ['Category', 'Strategy', 'Sharpe', 'CAGR', 'MaxDD', 'Worst5Y', 'Rebalances', 'Rebal/Year']
    results_df = results_df[cols].sort_values('Sharpe', ascending=False)

    display = results_df.copy()
    for col in ['CAGR', 'MaxDD', 'Worst5Y']:
        display[col] = display[col].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "N/A")
    display['Sharpe'] = display['Sharpe'].apply(lambda x: f"{x:.3f}")
    display['Rebalances'] = display['Rebalances'].astype(int)
    display['Rebal/Year'] = display['Rebal/Year'].apply(lambda x: f"{x:.1f}")
    print(display.to_string(index=False))

    # =====================================================================
    # PART 4: Condition Impact Summary
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 4: CONDITION IMPACT SUMMARY")
    print(f"{'Strategy':<30s} {'Old Sharpe':>10s} {'New Sharpe':>10s} {'Change':>8s} "
          f"{'Old MaxDD':>10s} {'New MaxDD':>10s} {'Change':>8s}")
    print("-" * 130)

    for label in ['Buy&Hold 3x', 'DD-Only', 'DD+VT(25%)', 'DD+VT+VolSpike',
                   'Ens2(Asym+Slope)', 'Ens2(Slope+TrendTV)']:
        old = [r for r in results_all if r['Strategy'] == f'{label} (OLD: 1d/0.9%)']
        new = [r for r in results_all if r['Strategy'] == f'{label} (NEW: 5d/1.5%)']
        if old and new:
            o, n = old[0], new[0]
            print(f"{label:<30s} {o['Sharpe']:>10.3f} {n['Sharpe']:>10.3f} {n['Sharpe']-o['Sharpe']:>+8.3f} "
                  f"{o['MaxDD']*100:>9.1f}% {n['MaxDD']*100:>9.1f}% {(n['MaxDD']-o['MaxDD'])*100:>+7.1f}%")

    # =====================================================================
    # PART 5: Practical Recommendations
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 5: PRACTICAL RECOMMENDATIONS (under new conditions)")
    print("=" * 130)

    budgets = [
        ("~30 trades/47yr (< 1/yr)", 50),
        ("~300 trades (~ 6/yr)", 400),
        ("~600 trades (~ 12/yr)", 700),
        ("~1000 trades (~ 20/yr)", 1200),
        ("Unlimited", 99999),
    ]

    print(f"\n{'Budget':<30s} {'Best Strategy':<55s} {'Sharpe':>7s} {'CAGR':>7s} "
          f"{'MaxDD':>7s} {'W5Y':>7s} {'Rebal':>6s}")
    print("-" * 130)

    for label, max_r in budgets:
        cands = results_df[results_df['Rebalances'] <= max_r]
        if len(cands) > 0:
            best = cands.iloc[0]
            print(f"{label:<30s} {best['Strategy']:<55s} {best['Sharpe']:.3f} "
                  f"{best['CAGR']*100:+.1f}% {best['MaxDD']*100:.1f}% "
                  f"{best['Worst5Y']*100:+.1f}% {int(best['Rebalances']):>5d}")

    # Save
    output_path = os.path.join(script_dir, '..', 'realistic_product_results.csv')
    pd.DataFrame(results_all).to_csv(output_path, index=False)
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
