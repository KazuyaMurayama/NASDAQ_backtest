"""
Final 6-Strategy Comparison with Yearly Returns.

Realistic conditions: 5-day delay, 1.5% annual cost.

Strategies:
  NEW:  MomDecel(40/120)+Ens2(S+T) Th20%  [Best found - delay-robust]
  1st:  Ens2(Slope+TrendTV) Th=20%         [Previous best risk-adjusted]
  4th:  DD+VT(25%) Th=15%                  [Base + VT]
  6th:  DD-Only (3x or Cash)               [Simplest]
  Ref:  Buy & Hold 3x                      [Full leverage, no strategy]
  Ref:  Buy & Hold 1x                      [No leverage, no strategy]
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics,
    strategy_baseline_bh3x, strategy_baseline_dd_only,
    strategy_baseline_dd_vt, calc_ewma_vol, calc_vt_leverage
)
from test_ens2_strategies import strategy_ens2_slope_trendtv
from test_delay_robust import (
    strategy_momentum_decel_ens2_st, calc_momentum_decel_mult
)

# Constants
ANNUAL_COST = 0.015
EXEC_DELAY = 5
BASE_LEV = 3.0


def run_backtest(close, leverage, base_lev=BASE_LEV, cost=ANNUAL_COST, delay=EXEC_DELAY):
    returns = close.pct_change()
    lev_returns = returns * base_lev
    daily_cost = cost / 252
    delayed = leverage.shift(delay)
    strat_ret = delayed * (lev_returns - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret


def rebalance_threshold(leverage, threshold):
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


def calc_yearly_returns(nav, dates):
    """Calculate year-by-year returns."""
    nav_df = pd.DataFrame({'nav': nav.values, 'date': dates.values})
    nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year

    yearly_last = nav_df.groupby('year')['nav'].last()
    yearly_ret = yearly_last.pct_change()
    # First year: return from 1.0
    yearly_ret.iloc[0] = yearly_last.iloc[0] - 1.0
    return yearly_ret


def count_rebalances(leverage):
    return (leverage.diff().abs() > 0.01).sum()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'NASDAQ_Dairy_since1973.csv')
    df = load_data(data_path)

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    # =================================================================
    # Build 6 strategies
    # =================================================================

    # NEW: MomDecel(40/120)+Ens2(S+T) Th=20%
    lev_raw_new, dd_new = strategy_momentum_decel_ens2_st(
        close, returns, short=40, long=120, sensitivity=0.3)
    lev_new = rebalance_threshold(lev_raw_new, 0.20)

    # 1st: Ens2(Slope+TrendTV) Th=20%
    lev_raw_1, dd_1 = strategy_ens2_slope_trendtv(
        close, returns, 0.82, 0.92, 20, 5, 1.0)
    lev_1 = rebalance_threshold(lev_raw_1, 0.20)

    # 4th: DD+VT(25%) Th=15%
    lev_raw_4, dd_4 = strategy_baseline_dd_vt(
        close, returns, 0.82, 0.92, 0.25, 10)
    lev_4 = rebalance_threshold(lev_raw_4, 0.15)

    # 6th: DD-Only
    lev_6, dd_6 = strategy_baseline_dd_only(close, returns, 0.82, 0.92)

    # Ref: Buy & Hold 3x
    lev_bh3, dd_bh3 = strategy_baseline_bh3x(close, returns)

    # Ref: Buy & Hold 1x (base_lev=1.0, no cost since it's index itself)
    lev_bh1 = pd.Series(1.0, index=close.index)

    strategies = [
        ('MomDecel+Ens2(S+T)',  lev_new,  dd_new, BASE_LEV, ANNUAL_COST, EXEC_DELAY),
        ('Ens2(S+T) Th20%',    lev_1,    dd_1,   BASE_LEV, ANNUAL_COST, EXEC_DELAY),
        ('DD+VT Th15%',        lev_4,    dd_4,   BASE_LEV, ANNUAL_COST, EXEC_DELAY),
        ('DD-Only',            lev_6,    dd_6,   BASE_LEV, ANNUAL_COST, EXEC_DELAY),
        ('B&H 3x',            lev_bh3,  dd_bh3, BASE_LEV, ANNUAL_COST, EXEC_DELAY),
        ('B&H 1x',            lev_bh1,  lev_bh1, 1.0, 0.0, 0),
    ]

    # =================================================================
    # Part 1: Summary Metrics
    # =================================================================
    print("=" * 140)
    print("FINAL 6-STRATEGY COMPARISON")
    print(f"Conditions: {EXEC_DELAY}-day execution delay, {ANNUAL_COST*100:.1f}% annual cost")
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()} ({len(df)/252:.0f} years)")
    print("=" * 140)

    metrics_list = []
    navs = {}

    for name, lev, dd, blev, cost, delay in strategies:
        nav, strat_ret = run_backtest(close, lev, blev, cost, delay)
        m = calc_metrics(nav, strat_ret, dd, dates)
        rebal = count_rebalances(lev)
        m['Strategy'] = name
        m['Rebalances'] = rebal
        m['Rebal/Year'] = rebal / (len(df) / 252)
        metrics_list.append(m)
        navs[name] = nav

    print(f"\n{'Rank':<6s} {'Strategy':<24s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} "
          f"{'Worst5Y':>8s} {'Sortino':>8s} {'Calmar':>8s} {'WinRate':>8s} {'Rebal/yr':>9s}")
    print("-" * 140)

    ranks = ['NEW', '1st', '4th', '6th', 'Ref', 'Ref']
    for rank, m in zip(ranks, metrics_list):
        print(f"{rank:<6s} {m['Strategy']:<24s} {m['Sharpe']:>7.3f} {m['CAGR']*100:>+7.1f}% "
              f"{m['MaxDD']*100:>7.1f}% {m['Worst5Y']*100:>+7.1f}% "
              f"{m['Sortino']:>8.3f} {m['Calmar']:>8.3f} "
              f"{m['WinRate']*100:>7.1f}% {m['Rebal/Year']:>8.1f}")

    # Final NAV
    print(f"\n{'':6s} {'Strategy':<24s} {'Final NAV':>15s} {'$10,000 →':>15s}")
    print("-" * 70)
    for rank, m in zip(ranks, metrics_list):
        name = m['Strategy']
        final = navs[name].iloc[-1]
        print(f"{rank:<6s} {name:<24s} {final:>15,.1f} ${final*10000:>14,.0f}")

    # =================================================================
    # Part 2: Yearly Returns
    # =================================================================
    print("\n" + "=" * 140)
    print("YEARLY RETURNS (%)")
    print("=" * 140)

    yearly_data = {}
    for name, lev, dd, blev, cost, delay in strategies:
        nav = navs[name]
        yr = calc_yearly_returns(nav, dates)
        yearly_data[name] = yr

    yearly_df = pd.DataFrame(yearly_data)
    yearly_df.index.name = 'Year'

    # Print header
    names = [s[0] for s in strategies]
    header = f"{'Year':<6s}"
    for n in names:
        header += f" {n:>18s}"
    print(header)
    print("-" * (6 + 19 * len(names)))

    # Print each year
    for year in yearly_df.index:
        row = f"{year:<6d}"
        for n in names:
            val = yearly_df.loc[year, n]
            if pd.notna(val):
                row += f" {val*100:>+17.1f}%"
            else:
                row += f" {'N/A':>18s}"
        print(row)

    # Summary stats at bottom
    print("-" * (6 + 19 * len(names)))

    # Mean
    row = f"{'Mean':<6s}"
    for n in names:
        row += f" {yearly_df[n].mean()*100:>+17.1f}%"
    print(row)

    # Median
    row = f"{'Med.':<6s}"
    for n in names:
        row += f" {yearly_df[n].median()*100:>+17.1f}%"
    print(row)

    # Std
    row = f"{'Std':<6s}"
    for n in names:
        row += f" {yearly_df[n].std()*100:>17.1f}%"
    print(row)

    # Best
    row = f"{'Best':<6s}"
    for n in names:
        row += f" {yearly_df[n].max()*100:>+17.1f}%"
    print(row)

    # Worst
    row = f"{'Worst':<6s}"
    for n in names:
        row += f" {yearly_df[n].min()*100:>+17.1f}%"
    print(row)

    # Positive years
    row = f"{'Win':<6s}"
    for n in names:
        pos = (yearly_df[n] > 0).sum()
        total = yearly_df[n].notna().sum()
        row += f" {pos:>12d}/{total:<4d}   "
    print(row)

    # Negative years
    row = f"{'Lose':<6s}"
    for n in names:
        neg = (yearly_df[n] < 0).sum()
        total = yearly_df[n].notna().sum()
        row += f" {neg:>12d}/{total:<4d}   "
    print(row)

    # =================================================================
    # Part 3: Crisis & Boom Years Highlight
    # =================================================================
    print("\n" + "=" * 140)
    print("NOTABLE YEARS HIGHLIGHT")
    print("=" * 140)

    notable = {
        'Crashes': [1974, 1987, 2000, 2001, 2002, 2008],
        'Booms':   [1975, 1991, 1995, 1998, 1999, 2003, 2009, 2019, 2020],
    }

    for label, years in notable.items():
        print(f"\n--- {label} ---")
        header = f"{'Year':<6s}"
        for n in names:
            header += f" {n:>18s}"
        print(header)
        print("-" * (6 + 19 * len(names)))
        for year in years:
            if year in yearly_df.index:
                row = f"{year:<6d}"
                for n in names:
                    val = yearly_df.loc[year, n]
                    if pd.notna(val):
                        row += f" {val*100:>+17.1f}%"
                    else:
                        row += f" {'N/A':>18s}"
                print(row)

    # =================================================================
    # Part 4: MomDecel vs Ens2(S+T) Year-by-Year Difference
    # =================================================================
    print("\n" + "=" * 140)
    print("MomDecel ADVANTAGE: Year-by-Year Difference vs Ens2(S+T)")
    print("=" * 140)

    diff = yearly_df['MomDecel+Ens2(S+T)'] - yearly_df['Ens2(S+T) Th20%']
    print(f"\n{'Year':<6s} {'MomDecel+Ens2':>14s} {'Ens2(S+T)':>14s} {'Diff':>10s} {'Note':>8s}")
    print("-" * 60)
    for year in yearly_df.index:
        md = yearly_df.loc[year, 'MomDecel+Ens2(S+T)']
        e2 = yearly_df.loc[year, 'Ens2(S+T) Th20%']
        d = diff.loc[year]
        note = ""
        if abs(d) > 0.05:
            note = "***" if d > 0 else "!!!"
        elif abs(d) > 0.02:
            note = "**" if d > 0 else "!!"
        print(f"{year:<6d} {md*100:>+13.1f}% {e2*100:>+13.1f}% {d*100:>+9.1f}% {note:>8s}")

    better = (diff > 0.001).sum()
    worse = (diff < -0.001).sum()
    tied = len(diff) - better - worse
    print(f"\nMomDecel better: {better} years / Worse: {worse} years / Similar: {tied} years")
    print(f"Average diff: {diff.mean()*100:+.2f}%")

    # Save yearly returns CSV
    output_yearly = os.path.join(script_dir, '..', 'final_6strategy_yearly_returns.csv')
    save_df = yearly_df.copy()
    save_df.columns = [f'{n} (%)' for n in save_df.columns]
    save_df = save_df * 100
    save_df.to_csv(output_yearly, float_format='%.2f')
    print(f"\nYearly returns saved to {output_yearly}")


if __name__ == "__main__":
    main()
