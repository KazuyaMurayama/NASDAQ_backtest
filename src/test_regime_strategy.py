"""
Test DD+Regime(MA200)+VT strategy
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import *

def main():
    print("=" * 80)
    print("DD+Regime(MA200)+VT Strategy Analysis")
    print("=" * 80)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    results = []

    # ==========================================================================
    # Baseline strategies for comparison
    # ==========================================================================
    print("\n--- Baseline Strategies ---")

    # DD-18/92 + VT(25%)
    lev, pos = strategy_baseline_dd_vt(close, returns, 0.82, 0.92, 0.25, 10)
    nav, strat_ret = run_backtest(close, lev)
    metrics = calc_metrics(nav, strat_ret, pos, dates)
    metrics['Strategy'] = 'DD(-18/92)+VT(25%) [Baseline]'
    results.append(metrics)
    print(f"DD(-18/92)+VT(25%): CAGR={metrics['CAGR']*100:.2f}%, Sharpe={metrics['Sharpe']:.3f}, MaxDD={metrics['MaxDD']*100:.2f}%, Trades={metrics['Trades']}")

    # DD-18/92 + VT(30%)
    lev, pos = strategy_baseline_dd_vt(close, returns, 0.82, 0.92, 0.30, 10)
    nav, strat_ret = run_backtest(close, lev)
    metrics = calc_metrics(nav, strat_ret, pos, dates)
    metrics['Strategy'] = 'DD(-18/92)+VT(30%)'
    results.append(metrics)
    print(f"DD(-18/92)+VT(30%): CAGR={metrics['CAGR']*100:.2f}%, Sharpe={metrics['Sharpe']:.3f}, MaxDD={metrics['MaxDD']*100:.2f}%, Trades={metrics['Trades']}")

    # ==========================================================================
    # DD + Regime + VT strategies
    # ==========================================================================
    print("\n--- DD + Regime(MA200) + VT Strategies ---")

    # Main target: DD-18/92 + Regime(MA200) + VT(30%)
    lev, pos = strategy_dd_regime_vt(close, returns, 0.82, 0.92, 200, 0.30, 10)
    nav, strat_ret = run_backtest(close, lev)
    metrics = calc_metrics(nav, strat_ret, pos, dates)
    metrics['Strategy'] = 'DD(-18/92)+Regime(MA200)+VT(30%)'
    results.append(metrics)
    print(f"DD(-18/92)+Regime(MA200)+VT(30%): CAGR={metrics['CAGR']*100:.2f}%, Sharpe={metrics['Sharpe']:.3f}, MaxDD={metrics['MaxDD']*100:.2f}%, Trades={metrics['Trades']}")

    # Variants with different parameters
    variants = [
        (0.82, 0.92, 200, 0.25, 10, 'DD(-18/92)+Regime(MA200)+VT(25%)'),
        (0.82, 0.92, 200, 0.35, 10, 'DD(-18/92)+Regime(MA200)+VT(35%)'),
        (0.85, 0.90, 200, 0.30, 10, 'DD(-15/90)+Regime(MA200)+VT(30%)'),
        (0.82, 0.92, 150, 0.30, 10, 'DD(-18/92)+Regime(MA150)+VT(30%)'),
        (0.82, 0.92, 250, 0.30, 10, 'DD(-18/92)+Regime(MA250)+VT(30%)'),
        (0.80, 0.92, 200, 0.30, 10, 'DD(-20/92)+Regime(MA200)+VT(30%)'),
    ]

    for exit_th, reentry_th, ma_lb, tv, span, name in variants:
        lev, pos = strategy_dd_regime_vt(close, returns, exit_th, reentry_th, ma_lb, tv, span)
        nav, strat_ret = run_backtest(close, lev)
        metrics = calc_metrics(nav, strat_ret, pos, dates)
        metrics['Strategy'] = name
        results.append(metrics)
        print(f"{name}: CAGR={metrics['CAGR']*100:.2f}%, Sharpe={metrics['Sharpe']:.3f}, MaxDD={metrics['MaxDD']*100:.2f}%, Trades={metrics['Trades']}")

    # ==========================================================================
    # Results DataFrame
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FULL RESULTS TABLE")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    cols = ['Strategy', 'CAGR', 'Worst5Y', 'Sharpe', 'Sortino', 'Calmar', 'MaxDD', 'Trades', 'WinRate']
    results_df = results_df[cols]

    # Sort by Sharpe
    results_df = results_df.sort_values('Sharpe', ascending=False)

    # Format for display
    display_df = results_df.copy()
    for col in ['CAGR', 'Worst5Y', 'MaxDD', 'WinRate']:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    for col in ['Sharpe', 'Sortino', 'Calmar']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    display_df['Trades'] = display_df['Trades'].astype(int)

    print(display_df.to_string(index=False))

    # ==========================================================================
    # Detailed analysis of main strategy
    # ==========================================================================
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: DD(-18/92)+Regime(MA200)+VT(30%)")
    print("=" * 80)

    lev, pos = strategy_dd_regime_vt(close, returns, 0.82, 0.92, 200, 0.30, 10)
    nav, strat_ret = run_backtest(close, lev)

    # Trade analysis
    position_changes = pos.diff().abs()
    trade_dates = dates[position_changes > 0.5]
    print(f"\nTotal trades: {len(trade_dates)}")
    print(f"Trades per year: {len(trade_dates) / 47:.1f}")

    # Regime analysis
    ma200 = close.rolling(200).mean()
    regime = (close > ma200).astype(int)
    regime_changes = regime.diff().abs()
    regime_crossings = (regime_changes > 0).sum()
    print(f"\nMA200 regime crossings: {regime_crossings}")
    print(f"Regime crossings per year: {regime_crossings / 47:.1f}")

    # Time in market
    time_in_market = (pos > 0).mean() * 100
    print(f"\nTime in market: {time_in_market:.1f}%")

    # Crisis year performance
    print("\nCrisis Year Returns:")
    crisis_returns = calc_crisis_returns(nav, dates)
    for year, ret in sorted(crisis_returns.items()):
        print(f"  {year}: {ret:.1f}%")

    # Save results
    results_df.to_csv(r"C:\Users\user\Desktop\nasdaq_backtest\regime_strategy_results.csv", index=False)
    print(f"\nResults saved to regime_strategy_results.csv")

    return results_df


if __name__ == "__main__":
    results = main()
