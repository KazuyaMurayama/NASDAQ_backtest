"""
Compare DD+Regime+VT strategies with top R4 strategies
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import *

def main():
    print("=" * 80)
    print("DD+Regime(MA200)+VT vs Top R4 Strategies Comparison")
    print("=" * 80)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    results = []

    def add_result(lev, pos, name, category):
        nav, strat_ret = run_backtest(close, lev)
        metrics = calc_metrics(nav, strat_ret, pos, dates)
        metrics['Strategy'] = name
        metrics['Category'] = category
        results.append(metrics)
        return nav

    # ==========================================================================
    # Top R4 Strategies (for comparison)
    # ==========================================================================
    print("--- Top R4 Strategies (<=100 trades) ---")

    # Baseline
    lev, pos = strategy_baseline_dd_vt(close, returns, 0.82, 0.92, 0.25, 10)
    add_result(lev, pos, 'DD(-18/92)+VT(25%) [Baseline]', 'Baseline')

    # Vol Spike (Top 1)
    lev, pos = strategy_dd_vt_volspike(close, returns, 0.82, 0.92, 0.25, 10, 1.5)
    add_result(lev, pos, 'DD+VT+VolSpike(1.5x) [R4 Top1]', 'R4-Top')

    # Seasonal Q3Q4 (Top 2)
    lev, pos = strategy_dd_quarterly_filter(close, returns, dates, 0.82, 0.92, 0.25, 10, [7,8,9,10], 0.8)
    add_result(lev, pos, 'DD+VT+Q3Q4weak(0.8x) [R4 Top2]', 'R4-Top')

    # ==========================================================================
    # DD + Regime + VT Strategies (High Trade Count)
    # ==========================================================================
    print("\n--- DD+Regime(MA200)+VT Strategies (High Trade Count) ---")

    # DD+Regime+VT(25%)
    lev, pos = strategy_dd_regime_vt(close, returns, 0.82, 0.92, 200, 0.25, 10)
    nav_25 = add_result(lev, pos, 'DD+Regime(MA200)+VT(25%)', 'Regime')

    # DD+Regime+VT(30%)
    lev, pos = strategy_dd_regime_vt(close, returns, 0.82, 0.92, 200, 0.30, 10)
    nav_30 = add_result(lev, pos, 'DD+Regime(MA200)+VT(30%)', 'Regime')

    # DD+Regime+VT(35%)
    lev, pos = strategy_dd_regime_vt(close, returns, 0.82, 0.92, 200, 0.35, 10)
    add_result(lev, pos, 'DD+Regime(MA200)+VT(35%)', 'Regime')

    # DD+Regime+VT(40%)
    lev, pos = strategy_dd_regime_vt(close, returns, 0.82, 0.92, 200, 0.40, 10)
    nav_40 = add_result(lev, pos, 'DD+Regime(MA200)+VT(40%)', 'Regime')

    # DD(-15/90)+Regime+VT(30%)
    lev, pos = strategy_dd_regime_vt(close, returns, 0.85, 0.90, 200, 0.30, 10)
    add_result(lev, pos, 'DD(-15/90)+Regime(MA200)+VT(30%)', 'Regime')

    # DD(-15/90)+Regime+VT(40%)
    lev, pos = strategy_dd_regime_vt(close, returns, 0.85, 0.90, 200, 0.40, 10)
    add_result(lev, pos, 'DD(-15/90)+Regime(MA200)+VT(40%)', 'Regime')

    # ==========================================================================
    # Results Table
    # ==========================================================================
    print("\n" + "=" * 100)
    print("FULL COMPARISON TABLE (Sorted by Sharpe)")
    print("=" * 100)

    results_df = pd.DataFrame(results)
    cols = ['Category', 'Strategy', 'CAGR', 'Worst5Y', 'Sharpe', 'Sortino', 'Calmar', 'MaxDD', 'Trades', 'WinRate']
    results_df = results_df[cols]
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
    # Trade-Adjusted Analysis
    # ==========================================================================
    print("\n" + "=" * 100)
    print("TRADE-ADJUSTED ANALYSIS")
    print("=" * 100)

    print("\nSharpe per 100 trades (efficiency metric):")
    for _, row in results_df.iterrows():
        trades = row['Trades']
        sharpe = row['Sharpe']
        if trades > 0:
            sharpe_per_100 = sharpe / (trades / 100)
            print(f"  {row['Strategy'][:45]:<45}: Sharpe/100trades = {sharpe_per_100:.3f}")
        else:
            print(f"  {row['Strategy'][:45]:<45}: N/A (BH)")

    # ==========================================================================
    # Incremental Sharpe per Trade Analysis
    # ==========================================================================
    print("\n" + "=" * 100)
    print("INCREMENTAL VALUE ANALYSIS (vs Baseline)")
    print("=" * 100)

    baseline_sharpe = results_df[results_df['Strategy'].str.contains('Baseline')]['Sharpe'].values[0]
    baseline_trades = results_df[results_df['Strategy'].str.contains('Baseline')]['Trades'].values[0]

    print(f"\nBaseline: Sharpe={baseline_sharpe:.3f}, Trades={baseline_trades}")
    print("\nIncremental Sharpe per additional trade:")

    for _, row in results_df.iterrows():
        if 'Baseline' not in row['Strategy']:
            delta_sharpe = row['Sharpe'] - baseline_sharpe
            delta_trades = row['Trades'] - baseline_trades
            if delta_trades > 0:
                incremental = delta_sharpe / delta_trades * 100  # per 100 additional trades
                print(f"  {row['Strategy'][:45]:<45}: ΔSharpe={delta_sharpe:+.3f}, ΔTrades={delta_trades:+3d}, Incr/100t={incremental:+.3f}")
            elif delta_trades == 0:
                print(f"  {row['Strategy'][:45]:<45}: ΔSharpe={delta_sharpe:+.3f}, ΔTrades={delta_trades:+3d}, (same trades)")

    # ==========================================================================
    # Crisis Year Comparison
    # ==========================================================================
    print("\n" + "=" * 100)
    print("CRISIS YEAR RETURNS COMPARISON")
    print("=" * 100)

    # Recalculate for crisis analysis
    strategies_for_crisis = [
        ('DD+VT(25%) Baseline', strategy_baseline_dd_vt(close, returns, 0.82, 0.92, 0.25, 10)),
        ('DD+VT+VolSpike(1.5x)', strategy_dd_vt_volspike(close, returns, 0.82, 0.92, 0.25, 10, 1.5)),
        ('DD+Regime+VT(30%)', strategy_dd_regime_vt(close, returns, 0.82, 0.92, 200, 0.30, 10)),
        ('DD+Regime+VT(40%)', strategy_dd_regime_vt(close, returns, 0.82, 0.92, 200, 0.40, 10)),
    ]

    crisis_years = [1974, 1987, 2000, 2001, 2002, 2008, 2011, 2020]
    crisis_data = []

    for name, (lev, pos) in strategies_for_crisis:
        nav, _ = run_backtest(close, lev)
        crisis_ret = calc_crisis_returns(nav, dates)
        row = {'Strategy': name}
        for year in crisis_years:
            row[str(year)] = crisis_ret.get(year, np.nan)
        crisis_data.append(row)

    crisis_df = pd.DataFrame(crisis_data)
    print("\nCrisis Year Returns (%):")
    display_crisis = crisis_df.copy()
    for col in [str(y) for y in crisis_years]:
        display_crisis[col] = display_crisis[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    print(display_crisis.to_string(index=False))

    # ==========================================================================
    # Conclusion
    # ==========================================================================
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)

    regime_best = results_df[results_df['Category'] == 'Regime'].iloc[0]
    r4_best = results_df[results_df['Category'] == 'R4-Top'].iloc[0]

    print(f"""
Regime Strategy vs R4 Top Strategy:

[Regime Best] {regime_best['Strategy']}
  - Sharpe: {regime_best['Sharpe']:.3f}
  - CAGR: {regime_best['CAGR']*100:.2f}%
  - MaxDD: {regime_best['MaxDD']*100:.2f}%
  - Trades: {int(regime_best['Trades'])}

[R4 Top] {r4_best['Strategy']}
  - Sharpe: {r4_best['Sharpe']:.3f}
  - CAGR: {r4_best['CAGR']*100:.2f}%
  - MaxDD: {r4_best['MaxDD']*100:.2f}%
  - Trades: {int(r4_best['Trades'])}

Sharpe Delta: {regime_best['Sharpe'] - r4_best['Sharpe']:+.3f}
Additional Trades: {int(regime_best['Trades']) - int(r4_best['Trades'])}
""")

    # Save results
    results_df.to_csv(r"C:\Users\user\Desktop\nasdaq_backtest\regime_vs_r4_comparison.csv", index=False)
    print("Results saved to regime_vs_r4_comparison.csv")

    return results_df


if __name__ == "__main__":
    results = main()
