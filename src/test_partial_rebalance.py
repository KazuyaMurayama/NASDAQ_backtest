"""
Partial Rebalance Backtest: 3x product + cash ratio adjustment.

Assumption:
- User holds ONE 3x leveraged NASDAQ product + cash
- Can partially sell/buy the 3x product (next-day execution)
- No need to switch between 1x/2x/3x products
- Already 1-day delayed (backtest uses leverage.shift(1))

Key question: How often do you need to rebalance?
- Daily rebalancing = original backtest (ideal but impractical)
- Threshold-based: only rebalance when target changes significantly
- Periodic: rebalance every N days

Compare: Ens2(Asym+Slope) #1, Ens2(Slope+TrendTV) #2, DD+VT+VolSpike #3
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol,
    calc_vt_leverage, strategy_dd_vt_volspike, run_backtest
)
from test_ens2_strategies import (
    strategy_ens2_asym_slope, strategy_ens2_slope_trendtv
)


# =============================================================================
# Rebalancing Models
# =============================================================================
def rebalance_threshold(leverage: pd.Series, threshold: float) -> pd.Series:
    """Only rebalance when target leverage changes by more than threshold.
    e.g., threshold=0.10 means rebalance when |new - current| > 10%.
    """
    result = pd.Series(0.0, index=leverage.index)
    current = leverage.iloc[0]
    result.iloc[0] = current

    for i in range(1, len(leverage)):
        target = leverage.iloc[i]
        # DD exit (target=0) always triggers immediately
        if target == 0.0 and current > 0.0:
            current = 0.0
        # DD reentry (from 0 to positive) also triggers immediately
        elif current == 0.0 and target > 0.0:
            current = target
        # Within HOLD: only rebalance if change exceeds threshold
        elif abs(target - current) > threshold:
            current = target
        result.iloc[i] = current

    return result


def rebalance_periodic(leverage: pd.Series, period_days: int) -> pd.Series:
    """Rebalance every N trading days. DD exits still happen immediately.
    """
    result = pd.Series(0.0, index=leverage.index)
    current = leverage.iloc[0]
    result.iloc[0] = current
    days_since_rebalance = 0

    for i in range(1, len(leverage)):
        target = leverage.iloc[i]
        days_since_rebalance += 1

        # DD exit/reentry always immediate
        if target == 0.0 and current > 0.0:
            current = 0.0
            days_since_rebalance = 0
        elif current == 0.0 and target > 0.0:
            current = target
            days_since_rebalance = 0
        elif days_since_rebalance >= period_days:
            current = target
            days_since_rebalance = 0

        result.iloc[i] = current

    return result


def count_rebalances(leverage: pd.Series, min_change: float = 0.01) -> int:
    """Count number of actual rebalance events."""
    changes = leverage.diff().abs()
    return (changes > min_change).sum()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 120)
    print("PARTIAL REBALANCE BACKTEST - 3x Product + Cash Ratio Adjustment")
    print("=" * 120)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'NASDAQ_Dairy_since1973.csv')
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total: {len(df)} trading days, ~{len(df)/252:.0f} years\n")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    results = []

    def add_result(lev, dd_signal, name, category):
        nav, strat_ret = run_backtest(close, lev)
        metrics = calc_metrics(nav, strat_ret, dd_signal, dates)
        rebalances = count_rebalances(lev)
        metrics['Strategy'] = name
        metrics['Category'] = category
        metrics['Rebalances'] = rebalances
        metrics['Rebal/Year'] = rebalances / (len(df) / 252)
        results.append(metrics)
        print(f"  {name:<50s} Sharpe={metrics['Sharpe']:.3f}  CAGR={metrics['CAGR']*100:.1f}%  "
              f"MaxDD={metrics['MaxDD']*100:.1f}%  W5Y={metrics['Worst5Y']*100:.1f}%  "
              f"Rebal={rebalances} ({metrics['Rebal/Year']:.1f}/yr)")
        return nav

    # =========================================================================
    # Strategy 3: DD+VT+VolSpike (reference)
    # =========================================================================
    print("=" * 120)
    print("STRATEGY 3 (Reference): DD+VT+VolSpike(1.5x)")
    print("=" * 120)

    lev_vs, dd_vs = strategy_dd_vt_volspike(close, returns, 0.82, 0.92, 0.25, 10, 1.5)

    print("\n[Original - Daily Rebalance]")
    add_result(lev_vs, dd_vs, 'VolSpike Daily', 'S3-Original')

    print("\n[Threshold-Based Rebalance]")
    for th in [0.05, 0.10, 0.15, 0.20]:
        lev_th = rebalance_threshold(lev_vs, th)
        add_result(lev_th, dd_vs, f'VolSpike Threshold={th:.0%}', 'S3-Threshold')

    print("\n[Periodic Rebalance]")
    for period in [5, 10, 21, 63]:
        label = {5: '1wk', 10: '2wk', 21: '1mo', 63: '3mo'}[period]
        lev_p = rebalance_periodic(lev_vs, period)
        add_result(lev_p, dd_vs, f'VolSpike Periodic={label}', 'S3-Periodic')

    # =========================================================================
    # Strategy 1: Ens2(Asym+Slope)
    # =========================================================================
    print("\n" + "=" * 120)
    print("STRATEGY 1: Ens2(Asym+Slope)")
    print("=" * 120)

    lev_as, dd_as = strategy_ens2_asym_slope(
        close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)

    print("\n[Original - Daily Rebalance]")
    add_result(lev_as, dd_as, 'Ens2(A+S) Daily', 'S1-Original')

    print("\n[Threshold-Based Rebalance]")
    for th in [0.05, 0.10, 0.15, 0.20]:
        lev_th = rebalance_threshold(lev_as, th)
        add_result(lev_th, dd_as, f'Ens2(A+S) Threshold={th:.0%}', 'S1-Threshold')

    print("\n[Periodic Rebalance]")
    for period in [5, 10, 21, 63]:
        label = {5: '1wk', 10: '2wk', 21: '1mo', 63: '3mo'}[period]
        lev_p = rebalance_periodic(lev_as, period)
        add_result(lev_p, dd_as, f'Ens2(A+S) Periodic={label}', 'S1-Periodic')

    # =========================================================================
    # Strategy 2: Ens2(Slope+TrendTV)
    # =========================================================================
    print("\n" + "=" * 120)
    print("STRATEGY 2: Ens2(Slope+TrendTV)")
    print("=" * 120)

    lev_st, dd_st = strategy_ens2_slope_trendtv(
        close, returns, 0.82, 0.92, 20, 5, 1.0)

    print("\n[Original - Daily Rebalance]")
    add_result(lev_st, dd_st, 'Ens2(S+T) Daily', 'S2-Original')

    print("\n[Threshold-Based Rebalance]")
    for th in [0.05, 0.10, 0.15, 0.20]:
        lev_th = rebalance_threshold(lev_st, th)
        add_result(lev_th, dd_st, f'Ens2(S+T) Threshold={th:.0%}', 'S2-Threshold')

    print("\n[Periodic Rebalance]")
    for period in [5, 10, 21, 63]:
        label = {5: '1wk', 10: '2wk', 21: '1mo', 63: '3mo'}[period]
        lev_p = rebalance_periodic(lev_st, period)
        add_result(lev_p, dd_st, f'Ens2(S+T) Periodic={label}', 'S2-Periodic')

    # =========================================================================
    # DD-Only Reference
    # =========================================================================
    print("\n" + "=" * 120)
    print("REFERENCE: DD-Only (always 100% in 3x or 100% cash)")
    print("=" * 120)
    dd_only = calc_dd_signal(close, 0.82, 0.92)
    add_result(dd_only, dd_only, 'DD-Only (3x or Cash)', 'Ref-DD')

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 120)
    print("FULL COMPARISON (sorted by Sharpe)")
    print("=" * 120)

    results_df = pd.DataFrame(results)
    cols = ['Category', 'Strategy', 'Sharpe', 'CAGR', 'MaxDD', 'Worst5Y', 'Rebalances', 'Rebal/Year']
    results_df = results_df[cols].sort_values('Sharpe', ascending=False)

    display = results_df.copy()
    for col in ['CAGR', 'MaxDD', 'Worst5Y']:
        display[col] = display[col].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "N/A")
    display['Sharpe'] = display['Sharpe'].apply(lambda x: f"{x:.3f}")
    display['Rebalances'] = display['Rebalances'].astype(int)
    display['Rebal/Year'] = display['Rebal/Year'].apply(lambda x: f"{x:.1f}")

    print(display.to_string(index=False))

    # =========================================================================
    # Practical Recommendation Table
    # =========================================================================
    print("\n" + "=" * 120)
    print("PRACTICAL RECOMMENDATION: Best options by rebalancing budget")
    print("=" * 120)

    budgets = [
        ("~30回/47年 (年1回以下)", 50),
        ("~100回 (年2回)", 150),
        ("~300回 (年6回)", 400),
        ("~600回 (年12回=月1回)", 700),
        ("制限なし (毎日可)", 99999),
    ]

    print(f"\n{'Rebal Budget':<30s} {'Best Strategy':<50s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'W5Y':>8s} {'Rebal':>6s}")
    print("-" * 120)

    for label, max_rebal in budgets:
        candidates = results_df[results_df['Rebalances'] <= max_rebal]
        if len(candidates) > 0:
            best = candidates.iloc[0]
            print(f"{label:<30s} {best['Strategy']:<50s} {best['Sharpe']:.3f} "
                  f"{best['CAGR']*100:+.1f}% {best['MaxDD']*100:.1f}% "
                  f"{best['Worst5Y']*100:+.1f}% {int(best['Rebalances']):>5d}")

    # =========================================================================
    # Save
    # =========================================================================
    output_path = os.path.join(script_dir, '..', 'partial_rebalance_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
