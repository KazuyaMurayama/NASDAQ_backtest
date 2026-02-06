"""
DD+VT+VolSpike(1.5x) Strategy - Yearly Returns Analysis
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    load_data, strategy_dd_vt_volspike, strategy_baseline_dd_vt,
    run_backtest, calc_dd_signal, calc_ewma_vol, calc_vt_leverage
)

def main():
    print("=" * 80)
    print("DD+VT+VolSpike(1.5x) - Yearly Returns Analysis")
    print("=" * 80)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    # ==========================================================================
    # Run strategies
    # ==========================================================================

    # Strategy 1: DD+VT+VolSpike(1.5x) [Top Recommended]
    lev_volspike, pos_volspike = strategy_dd_vt_volspike(
        close, returns, 0.82, 0.92, 0.25, 10, 1.5
    )
    nav_volspike, ret_volspike = run_backtest(close, lev_volspike)

    # Strategy 2: DD+VT Baseline (for comparison)
    lev_baseline, pos_baseline = strategy_baseline_dd_vt(
        close, returns, 0.82, 0.92, 0.25, 10
    )
    nav_baseline, ret_baseline = run_backtest(close, lev_baseline)

    # Strategy 3: Buy & Hold 3x (for comparison)
    lev_bh = pd.Series(1.0, index=close.index)
    nav_bh, ret_bh = run_backtest(close, lev_bh)

    # Strategy 4: NASDAQ (1x, no leverage) for reference
    nasdaq_returns = returns.fillna(0)
    nav_nasdaq = (1 + nasdaq_returns).cumprod()

    # ==========================================================================
    # Calculate yearly returns
    # ==========================================================================

    # Create DataFrame with all NAVs
    nav_df = pd.DataFrame({
        'Date': dates.values,
        'VolSpike': nav_volspike.values,
        'Baseline': nav_baseline.values,
        'BH_3x': nav_bh.values,
        'NASDAQ_1x': nav_nasdaq.values
    })
    nav_df['Year'] = pd.to_datetime(nav_df['Date']).dt.year

    # Get year-end NAV for each year
    yearly_nav = nav_df.groupby('Year').last()

    # Calculate yearly returns
    yearly_returns = pd.DataFrame()
    yearly_returns['Year'] = yearly_nav.index

    for col in ['VolSpike', 'Baseline', 'BH_3x', 'NASDAQ_1x']:
        nav_series = yearly_nav[col]
        # First year: return from initial NAV (1.0)
        first_year_return = nav_series.iloc[0] - 1
        # Subsequent years: year-over-year return
        yoy_returns = nav_series.pct_change()
        yoy_returns.iloc[0] = first_year_return
        yearly_returns[col] = yoy_returns.values

    # Also get year-end NAV values
    yearly_nav_values = yearly_nav.reset_index()

    # ==========================================================================
    # Calculate cumulative NAV at each year end
    # ==========================================================================

    yearly_results = pd.DataFrame()
    yearly_results['Year'] = yearly_nav_values['Year']
    yearly_results['VolSpike_Return'] = (yearly_returns['VolSpike'] * 100).round(2)
    yearly_results['VolSpike_NAV'] = yearly_nav_values['VolSpike'].round(2)
    yearly_results['Baseline_Return'] = (yearly_returns['Baseline'] * 100).round(2)
    yearly_results['Baseline_NAV'] = yearly_nav_values['Baseline'].round(2)
    yearly_results['BH3x_Return'] = (yearly_returns['BH_3x'] * 100).round(2)
    yearly_results['BH3x_NAV'] = yearly_nav_values['BH_3x'].round(2)
    yearly_results['NASDAQ_Return'] = (yearly_returns['NASDAQ_1x'] * 100).round(2)
    yearly_results['NASDAQ_NAV'] = yearly_nav_values['NASDAQ_1x'].round(2)

    # Add strategy state (HOLD/CASH) at year end
    pos_df = pd.DataFrame({
        'Date': dates.values,
        'Position': pos_volspike.values
    })
    pos_df['Year'] = pd.to_datetime(pos_df['Date']).dt.year
    year_end_pos = pos_df.groupby('Year')['Position'].last()
    yearly_results['YearEnd_State'] = year_end_pos.apply(
        lambda x: 'HOLD' if x > 0.5 else 'CASH'
    ).values

    # ==========================================================================
    # Display Results
    # ==========================================================================

    print("=" * 120)
    print("YEARLY RETURNS TABLE")
    print("=" * 120)
    print(f"{'Year':<6} {'VolSpike':>12} {'VolSpike':>12} {'Baseline':>12} {'BH 3x':>12} {'NASDAQ':>10} {'State':>8}")
    print(f"{'':>6} {'Return(%)':>12} {'NAV':>12} {'Return(%)':>12} {'Return(%)':>12} {'Return(%)':>10} {'YearEnd':>8}")
    print("-" * 120)

    for _, row in yearly_results.iterrows():
        print(f"{int(row['Year']):<6} {row['VolSpike_Return']:>+12.2f} {row['VolSpike_NAV']:>12.2f} "
              f"{row['Baseline_Return']:>+12.2f} {row['BH3x_Return']:>+12.2f} "
              f"{row['NASDAQ_Return']:>+10.2f} {row['YearEnd_State']:>8}")

    print("-" * 120)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for strategy, col in [('VolSpike', 'VolSpike_Return'),
                          ('Baseline', 'Baseline_Return'),
                          ('BH 3x', 'BH3x_Return'),
                          ('NASDAQ 1x', 'NASDAQ_Return')]:
        rets = yearly_results[col] / 100
        positive_years = (rets > 0).sum()
        negative_years = (rets < 0).sum()
        best_year = rets.max() * 100
        worst_year = rets.min() * 100
        avg_return = rets.mean() * 100
        median_return = rets.median() * 100

        print(f"\n[{strategy}]")
        print(f"  Positive Years: {positive_years} / {len(rets)} ({positive_years/len(rets)*100:.1f}%)")
        print(f"  Negative Years: {negative_years} / {len(rets)}")
        print(f"  Best Year: {best_year:+.2f}%")
        print(f"  Worst Year: {worst_year:+.2f}%")
        print(f"  Average Return: {avg_return:+.2f}%")
        print(f"  Median Return: {median_return:+.2f}%")

    # ==========================================================================
    # Decade Analysis
    # ==========================================================================

    print("\n" + "=" * 80)
    print("DECADE ANALYSIS (VolSpike Strategy)")
    print("=" * 80)

    yearly_results['Decade'] = (yearly_results['Year'] // 10) * 10
    decade_stats = yearly_results.groupby('Decade').agg({
        'VolSpike_Return': ['mean', 'min', 'max', 'count'],
        'Year': 'count'
    })

    print(f"\n{'Decade':<10} {'Avg Return':>12} {'Best Year':>12} {'Worst Year':>12} {'Years':>8}")
    print("-" * 60)

    for decade in sorted(yearly_results['Decade'].unique()):
        decade_data = yearly_results[yearly_results['Decade'] == decade]
        avg_ret = decade_data['VolSpike_Return'].mean()
        best = decade_data['VolSpike_Return'].max()
        worst = decade_data['VolSpike_Return'].min()
        count = len(decade_data)
        print(f"{int(decade)}s{'':<5} {avg_ret:>+12.2f}% {best:>+12.2f}% {worst:>+12.2f}% {count:>8}")

    # ==========================================================================
    # Crisis Year Performance
    # ==========================================================================

    print("\n" + "=" * 80)
    print("CRISIS YEAR PERFORMANCE")
    print("=" * 80)

    crisis_years = [1974, 1987, 2000, 2001, 2002, 2008, 2011, 2020]
    crisis_df = yearly_results[yearly_results['Year'].isin(crisis_years)]

    print(f"\n{'Year':<6} {'VolSpike':>12} {'Baseline':>12} {'BH 3x':>12} {'NASDAQ':>10} {'State':>8}")
    print("-" * 70)

    for _, row in crisis_df.iterrows():
        print(f"{int(row['Year']):<6} {row['VolSpike_Return']:>+12.2f}% {row['Baseline_Return']:>+12.2f}% "
              f"{row['BH3x_Return']:>+12.2f}% {row['NASDAQ_Return']:>+10.2f}% {row['YearEnd_State']:>8}")

    # ==========================================================================
    # Save to Excel
    # ==========================================================================

    excel_path = r"C:\Users\user\Desktop\nasdaq_backtest\VolSpike_Yearly_Returns.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Full yearly results
        yearly_results.to_excel(writer, sheet_name='Yearly_Returns', index=False)

        # Sheet 2: Summary comparison
        summary_data = []
        for strategy, ret_col, nav_col in [
            ('DD+VT+VolSpike(1.5x)', 'VolSpike_Return', 'VolSpike_NAV'),
            ('DD+VT Baseline', 'Baseline_Return', 'Baseline_NAV'),
            ('Buy & Hold 3x', 'BH3x_Return', 'BH3x_NAV'),
            ('NASDAQ 1x', 'NASDAQ_Return', 'NASDAQ_NAV')
        ]:
            rets = yearly_results[ret_col] / 100
            final_nav = yearly_results[nav_col].iloc[-1] if nav_col else None
            summary_data.append({
                'Strategy': strategy,
                'Final_NAV': final_nav,
                'Total_Return_Pct': (final_nav - 1) * 100 if final_nav else None,
                'CAGR_Pct': ((final_nav ** (1/47)) - 1) * 100 if final_nav else None,
                'Avg_Annual_Return_Pct': rets.mean() * 100,
                'Median_Annual_Return_Pct': rets.median() * 100,
                'Best_Year_Pct': rets.max() * 100,
                'Worst_Year_Pct': rets.min() * 100,
                'Positive_Years': (rets > 0).sum(),
                'Negative_Years': (rets < 0).sum(),
                'Win_Rate_Pct': (rets > 0).mean() * 100
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 3: Crisis years
        crisis_df.to_excel(writer, sheet_name='Crisis_Years', index=False)

        # Sheet 4: Decade analysis
        decade_summary = yearly_results.groupby('Decade').agg({
            'VolSpike_Return': ['mean', 'min', 'max', 'std', 'count']
        }).round(2)
        decade_summary.columns = ['Avg_Return', 'Min_Return', 'Max_Return', 'Std_Dev', 'Years']
        decade_summary.to_excel(writer, sheet_name='Decade_Analysis')

    print(f"\n" + "=" * 80)
    print(f"Results saved to: {excel_path}")
    print("=" * 80)

    # Final NAV comparison
    print(f"\n[FINAL NAV (After 47 years)]")
    print(f"  VolSpike:  ${yearly_results['VolSpike_NAV'].iloc[-1]:,.2f} (from $1.00)")
    print(f"  Baseline:  ${yearly_results['Baseline_NAV'].iloc[-1]:,.2f}")
    print(f"  BH 3x:     ${yearly_results['BH3x_NAV'].iloc[-1]:,.2f}")
    print(f"  NASDAQ 1x: ${yearly_results['NASDAQ_NAV'].iloc[-1]:,.2f}")

    return yearly_results

if __name__ == "__main__":
    results = main()
