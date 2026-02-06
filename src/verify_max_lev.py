"""
Verify Sharpe discrepancy: max_lev=1.0 vs max_lev=3.0
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    load_data, calc_dd_signal, calc_ewma_vol, run_backtest, calc_metrics
)

def calc_vt_leverage_custom(vol, target_vol, max_lev):
    """VT with custom max_lev"""
    leverage = (target_vol / vol).clip(0, max_lev)
    return leverage.fillna(1.0)

def main():
    print("=" * 80)
    print("SHARPE DISCREPANCY VERIFICATION: max_lev=1.0 vs max_lev=3.0")
    print("=" * 80)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    # DD Signal (common)
    dd_signal_18 = calc_dd_signal(close, 0.82, 0.92)  # DD-18/92
    dd_signal_15 = calc_dd_signal(close, 0.85, 0.90)  # DD-15/90

    # EWMA Vol
    ewma_vol = calc_ewma_vol(returns, 10)

    results = []

    # Test both max_lev values for both DD parameters
    for dd_name, dd_signal in [('DD-18/92', dd_signal_18), ('DD-15/90', dd_signal_15)]:
        for max_lev in [1.0, 3.0]:
            vt_lev = calc_vt_leverage_custom(ewma_vol, 0.25, max_lev)
            leverage = dd_signal * vt_lev

            nav, strat_ret = run_backtest(close, leverage)
            metrics = calc_metrics(nav, strat_ret, dd_signal, dates)

            # Also calculate effective leverage stats
            eff_lev = leverage * 3.0  # base leverage is 3x

            results.append({
                'Strategy': f"{dd_name}+VT(25%)",
                'max_lev': max_lev,
                'CAGR': metrics['CAGR'],
                'Sharpe': metrics['Sharpe'],
                'Sortino': metrics['Sortino'],
                'MaxDD': metrics['MaxDD'],
                'Trades': metrics['Trades'],
                'AvgEffLev': eff_lev.mean(),
                'MaxEffLev': eff_lev.max()
            })

    # Display results
    print("COMPARISON TABLE")
    print("-" * 100)
    print(f"{'Strategy':<25} {'max_lev':>8} {'CAGR':>10} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>10} {'AvgLev':>8} {'MaxLev':>8}")
    print("-" * 100)

    for r in results:
        print(f"{r['Strategy']:<25} {r['max_lev']:>8.1f} {r['CAGR']*100:>9.2f}% {r['Sharpe']:>8.3f} {r['Sortino']:>8.3f} {r['MaxDD']*100:>9.2f}% {r['AvgEffLev']:>8.2f}x {r['MaxEffLev']:>7.2f}x")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Compare max_lev=1.0 vs max_lev=3.0 for DD-15/90
    r1 = [r for r in results if r['Strategy'] == 'DD-15/90+VT(25%)' and r['max_lev'] == 1.0][0]
    r3 = [r for r in results if r['Strategy'] == 'DD-15/90+VT(25%)' and r['max_lev'] == 3.0][0]

    print(f"""
DD-15/90+VT(25%) Comparison:

[max_lev=1.0] (Our implementation - Conservative)
  Sharpe: {r1['Sharpe']:.3f}
  CAGR: {r1['CAGR']*100:.2f}%
  MaxDD: {r1['MaxDD']*100:.2f}%
  Effective Leverage: {r1['AvgEffLev']:.2f}x avg, {r1['MaxEffLev']:.2f}x max

[max_lev=3.0] (Original R3 implementation)
  Sharpe: {r3['Sharpe']:.3f}
  CAGR: {r3['CAGR']*100:.2f}%
  MaxDD: {r3['MaxDD']*100:.2f}%
  Effective Leverage: {r3['AvgEffLev']:.2f}x avg, {r3['MaxEffLev']:.2f}x max

Sharpe Difference: {r3['Sharpe'] - r1['Sharpe']:+.3f}
CAGR Difference: {(r3['CAGR'] - r1['CAGR'])*100:+.2f}%
MaxDD Difference: {(r3['MaxDD'] - r1['MaxDD'])*100:+.2f}%
""")

    if r3['Sharpe'] > 1.5:
        print("CONFIRMED: max_lev=3.0 reproduces Sharpe > 1.5 (matching original R3)")
    else:
        print("NOTE: max_lev=3.0 still doesn't reach Sharpe 1.8+, other factors may exist")

if __name__ == "__main__":
    main()
