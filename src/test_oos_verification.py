"""
Out-of-Sample Verification: 2021/06 - Present

Fetches post-sample NASDAQ data from Yahoo Finance,
merges with existing backtest data, and computes monthly returns
for MomDecel+Ens2(S+T) vs B&H benchmarks.
"""
import pandas as pd
import numpy as np
import urllib.request
import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics,
    strategy_baseline_bh3x, strategy_baseline_dd_only
)
from test_ens2_strategies import (
    strategy_ens2_slope_trendtv, calc_asym_ewma_vol,
    calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    strategy_momentum_decel_ens2_st, rebalance_threshold
)

ANNUAL_COST = 0.015
EXEC_DELAY = 5
BASE_LEV = 3.0


# =================================================================
# Step 1: Fetch NASDAQ data from Yahoo Finance
# =================================================================
def fetch_yahoo_finance(symbol="^IXIC", start_date="2020-01-01"):
    """Fetch daily close prices from Yahoo Finance."""
    start_ts = int(time.mktime(time.strptime(start_date, '%Y-%m-%d')))
    end_ts = int(time.time())
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
           f'?period1={start_ts}&period2={end_ts}&interval=1d')

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())

    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    closes = result['indicators']['quote'][0]['close']

    df = pd.DataFrame({
        'Date': [pd.Timestamp.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
        'Close': closes
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    return df


# =================================================================
# Step 2: Merge with existing data
# =================================================================
def merge_datasets(existing_df, new_df):
    """Merge existing CSV data with newly fetched data."""
    # Find overlap period
    existing_end = existing_df['Date'].max()
    new_start = new_df['Date'].min()

    print(f"  Existing data: {existing_df['Date'].min().date()} to {existing_end.date()} "
          f"({len(existing_df)} rows)")
    print(f"  New data:      {new_start.date()} to {new_df['Date'].max().date()} "
          f"({len(new_df)} rows)")

    # Check overlap consistency
    overlap = pd.merge(existing_df, new_df, on='Date', suffixes=('_old', '_new'))
    if len(overlap) > 0:
        pct_diff = ((overlap['Close_old'] - overlap['Close_new']).abs() /
                    overlap['Close_old'] * 100)
        max_diff = pct_diff.max()
        print(f"  Overlap: {len(overlap)} days, max price diff: {max_diff:.4f}%")
        if max_diff > 1.0:
            print("  WARNING: Large price discrepancy in overlap period!")
    else:
        print("  No overlap period found")

    # Combine: use existing data where available, append new data after
    new_only = new_df[new_df['Date'] > existing_end].copy()
    combined = pd.concat([existing_df[['Date', 'Close']], new_only[['Date', 'Close']]],
                         ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)
    print(f"  Combined: {combined['Date'].min().date()} to {combined['Date'].max().date()} "
          f"({len(combined)} rows)")
    return combined


# =================================================================
# Backtest functions
# =================================================================
def run_backtest(close, leverage, base_lev=BASE_LEV, cost=ANNUAL_COST, delay=EXEC_DELAY):
    returns = close.pct_change()
    lev_returns = returns * base_lev
    daily_cost = cost / 252
    delayed = leverage.shift(delay)
    strat_ret = delayed * (lev_returns - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret


def calc_monthly_returns(nav, dates):
    """Calculate month-by-month returns."""
    nav_df = pd.DataFrame({'nav': nav.values, 'date': dates.values})
    nav_df['ym'] = pd.to_datetime(nav_df['date']).dt.to_period('M')
    monthly_last = nav_df.groupby('ym')['nav'].last()
    monthly_ret = monthly_last.pct_change()
    monthly_ret.iloc[0] = monthly_last.iloc[0] - 1.0
    return monthly_ret


# =================================================================
# Main
# =================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 120)
    print("OUT-OF-SAMPLE VERIFICATION: MomDecel+Ens2(S+T) vs Benchmarks")
    print("Conditions: 5-day execution delay, 1.5% annual cost")
    print("=" * 120)

    # --- Step 1: Fetch data ---
    print("\n--- Step 1: Fetch NASDAQ data from Yahoo Finance ---")
    new_df = fetch_yahoo_finance("^IXIC", "2020-01-01")
    print(f"  Fetched {len(new_df)} rows: {new_df['Date'].min().date()} to "
          f"{new_df['Date'].max().date()}")

    # --- Step 2: Merge ---
    print("\n--- Step 2: Merge with existing data ---")
    data_path = os.path.join(script_dir, '..', 'NASDAQ_Dairy_since1973.csv')
    existing_df = load_data(data_path)
    combined = merge_datasets(existing_df, new_df)

    # Save extended dataset
    output_csv = os.path.join(script_dir, '..', 'NASDAQ_extended.csv')
    combined.to_csv(output_csv, index=False)
    print(f"  Saved to {output_csv}")

    close = combined['Close']
    returns = close.pct_change()
    dates = combined['Date']

    # --- Step 3: Run strategies ---
    print("\n--- Step 3: Run strategies on full dataset ---")

    # MomDecel+Ens2(S+T) Th20%
    lev_raw_md, dd_md = strategy_momentum_decel_ens2_st(
        close, returns, short=40, long=120, sensitivity=0.3)
    lev_md = rebalance_threshold(lev_raw_md, 0.20)
    nav_md, sr_md = run_backtest(close, lev_md)
    print(f"  MomDecel+Ens2(S+T): computed")

    # B&H 3x
    lev_bh3, dd_bh3 = strategy_baseline_bh3x(close, returns)
    nav_bh3, sr_bh3 = run_backtest(close, lev_bh3)
    print(f"  B&H 3x: computed")

    # B&H 1x
    lev_bh1 = pd.Series(1.0, index=close.index)
    nav_bh1, sr_bh1 = run_backtest(close, lev_bh1, base_lev=1.0, cost=0.0, delay=0)
    print(f"  B&H 1x: computed")

    # --- Step 4: Monthly returns for OOS period ---
    print("\n--- Step 4: Out-of-Sample Monthly Returns ---")

    # OOS start: 2021-06-01 (first full month after sample ends 2021-05-07)
    oos_start = pd.Timestamp('2021-06-01')
    oos_mask = dates >= oos_start
    oos_start_idx = oos_mask.idxmax()

    # Normalize NAV to 1.0 at OOS start
    nav_md_oos = nav_md / nav_md.iloc[oos_start_idx]
    nav_bh3_oos = nav_bh3 / nav_bh3.iloc[oos_start_idx]
    nav_bh1_oos = nav_bh1 / nav_bh1.iloc[oos_start_idx]

    # Monthly returns from OOS start
    oos_dates = dates[oos_mask].reset_index(drop=True)
    oos_nav_md = nav_md_oos[oos_mask].reset_index(drop=True)
    oos_nav_bh3 = nav_bh3_oos[oos_mask].reset_index(drop=True)
    oos_nav_bh1 = nav_bh1_oos[oos_mask].reset_index(drop=True)

    monthly_md = calc_monthly_returns(oos_nav_md, oos_dates)
    monthly_bh3 = calc_monthly_returns(oos_nav_bh3, oos_dates)
    monthly_bh1 = calc_monthly_returns(oos_nav_bh1, oos_dates)

    # Print monthly returns table
    print(f"\n{'Month':<10s} {'MomDecel+Ens2':>14s} {'B&H 3x':>14s} {'B&H 1x':>14s} "
          f"{'MD vs 3x':>10s} {'MD vs 1x':>10s}")
    print("-" * 80)

    all_months = monthly_md.index
    for m in all_months:
        md_val = monthly_md[m] if m in monthly_md.index else np.nan
        bh3_val = monthly_bh3[m] if m in monthly_bh3.index else np.nan
        bh1_val = monthly_bh1[m] if m in monthly_bh1.index else np.nan

        diff3 = md_val - bh3_val if not (np.isnan(md_val) or np.isnan(bh3_val)) else np.nan
        diff1 = md_val - bh1_val if not (np.isnan(md_val) or np.isnan(bh1_val)) else np.nan

        print(f"{str(m):<10s} {md_val*100:>+13.1f}% {bh3_val*100:>+13.1f}% "
              f"{bh1_val*100:>+13.1f}% {diff3*100:>+9.1f}% {diff1*100:>+9.1f}%")

    # Summary stats
    print("-" * 80)
    print(f"{'Cum.':>10s} {(oos_nav_md.iloc[-1]-1)*100:>+13.1f}% "
          f"{(oos_nav_bh3.iloc[-1]-1)*100:>+13.1f}% "
          f"{(oos_nav_bh1.iloc[-1]-1)*100:>+13.1f}%")

    n_oos_years = len(oos_dates) / 252
    cagr_md = oos_nav_md.iloc[-1] ** (1/n_oos_years) - 1
    cagr_bh3 = oos_nav_bh3.iloc[-1] ** (1/n_oos_years) - 1
    cagr_bh1 = oos_nav_bh1.iloc[-1] ** (1/n_oos_years) - 1

    print(f"{'CAGR':>10s} {cagr_md*100:>+13.1f}% {cagr_bh3*100:>+13.1f}% "
          f"{cagr_bh1*100:>+13.1f}%")

    # Monthly Sharpe
    sharpe_md = monthly_md.mean() / monthly_md.std() * np.sqrt(12)
    sharpe_bh3 = monthly_bh3.mean() / monthly_bh3.std() * np.sqrt(12)
    sharpe_bh1 = monthly_bh1.mean() / monthly_bh1.std() * np.sqrt(12)
    print(f"{'Sharpe(M)':>10s} {sharpe_md:>14.3f} {sharpe_bh3:>14.3f} {sharpe_bh1:>14.3f}")

    # Max drawdown in OOS
    def calc_maxdd(nav_series):
        peak = nav_series.cummax()
        dd = (nav_series - peak) / peak
        return dd.min()

    maxdd_md = calc_maxdd(oos_nav_md)
    maxdd_bh3 = calc_maxdd(oos_nav_bh3)
    maxdd_bh1 = calc_maxdd(oos_nav_bh1)
    print(f"{'MaxDD':>10s} {maxdd_md*100:>13.1f}% {maxdd_bh3*100:>13.1f}% "
          f"{maxdd_bh1*100:>13.1f}%")

    # Win rate (months with positive return)
    win_md = (monthly_md > 0).mean()
    win_bh3 = (monthly_bh3 > 0).mean()
    win_bh1 = (monthly_bh1 > 0).mean()
    print(f"{'Win%':>10s} {win_md*100:>13.1f}% {win_bh3*100:>13.1f}% "
          f"{win_bh1*100:>13.1f}%")

    # --- Step 5: Year-by-year OOS summary ---
    print("\n" + "=" * 120)
    print("YEARLY OOS RETURNS")
    print("=" * 120)

    nav_df = pd.DataFrame({
        'date': oos_dates.values,
        'md': oos_nav_md.values,
        'bh3': oos_nav_bh3.values,
        'bh1': oos_nav_bh1.values
    })
    nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year

    print(f"\n{'Year':<6s} {'MomDecel+Ens2':>14s} {'B&H 3x':>14s} {'B&H 1x':>14s}")
    print("-" * 55)

    for year in sorted(nav_df['year'].unique()):
        yr_data = nav_df[nav_df['year'] == year]
        for col, label in [('md', 'MomDecel'), ('bh3', 'B&H 3x'), ('bh1', 'B&H 1x')]:
            start_nav = yr_data[col].iloc[0]
            end_nav = yr_data[col].iloc[-1]
            yr_ret = end_nav / start_nav - 1
            if col == 'md':
                row = f"{year:<6d} {yr_ret*100:>+13.1f}%"
            else:
                row += f" {yr_ret*100:>+13.1f}%"
        print(row)

    # --- OOS vs In-Sample comparison ---
    print("\n" + "=" * 120)
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("=" * 120)

    # In-sample: up to 2021-05-07
    is_end_idx = (dates <= pd.Timestamp('2021-05-07')).sum() - 1
    nav_md_is = nav_md.iloc[:is_end_idx+1]
    sr_md_is = sr_md.iloc[:is_end_idx+1]
    is_years = (is_end_idx + 1) / 252

    is_cagr = nav_md_is.iloc[-1] ** (1/is_years) - 1
    is_sharpe = sr_md_is.mean() * 252 / (sr_md_is.std() * np.sqrt(252))
    is_maxdd = calc_maxdd(nav_md_is)

    print(f"\n{'Period':<20s} {'CAGR':>10s} {'Sharpe':>10s} {'MaxDD':>10s} {'Years':>8s}")
    print("-" * 65)
    print(f"{'In-Sample':<20s} {is_cagr*100:>+9.1f}% {is_sharpe:>10.3f} "
          f"{is_maxdd*100:>9.1f}% {is_years:>7.1f}")
    print(f"{'Out-of-Sample':<20s} {cagr_md*100:>+9.1f}% {sharpe_md:>10.3f} "
          f"{maxdd_md*100:>9.1f}% {n_oos_years:>7.1f}")

    # Save results
    results_df = pd.DataFrame({
        'Month': [str(m) for m in all_months],
        'MomDecel+Ens2 (%)': [monthly_md[m]*100 for m in all_months],
        'B&H 3x (%)': [monthly_bh3[m]*100 for m in all_months],
        'B&H 1x (%)': [monthly_bh1[m]*100 for m in all_months],
    })
    results_path = os.path.join(script_dir, '..', 'oos_monthly_returns.csv')
    results_df.to_csv(results_path, index=False, float_format='%.2f')
    print(f"\nMonthly returns saved to {results_path}")


if __name__ == "__main__":
    main()
