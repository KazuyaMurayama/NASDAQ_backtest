"""Compute DH[A] Scenario D calendar-year rolling Worst10Y / Best10Y.

Uses only local data (no internet required):
  data/NASDAQ_extended_to_2026.csv
  data/lbma_gold_daily.csv
  data/dtb3_daily.csv, dgs10_daily.csv, dgs30_daily.csv
"""
import sys
import os
import types

# Patch multitasking before any yfinance imports
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from corrected_strategy_backtest import (
    load_data, load_sofr,
    build_bond_1x_nav_corrected, build_bond_3x, build_gold_2x, build_nav,
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, DATA_DIR,
)


def prepare_gold_local(nasdaq_dates):
    """Build gold price series aligned to NASDAQ dates from local lbma_gold_daily.csv."""
    gold_path = os.path.join(DATA_DIR, 'lbma_gold_daily.csv')
    g = pd.read_csv(gold_path, parse_dates=['Date'])
    g = g.rename(columns={'USD': 'Gold'})
    nasdaq_df = pd.DataFrame({'Date': nasdaq_dates})
    merged = nasdaq_df.merge(g, on='Date', how='left')
    merged['Gold'] = merged['Gold'].ffill().bfill()
    return merged['Gold'].values


def rolling_nY_cagr(annual_ret_series, n=10):
    """Rolling n-year CAGR from annual return series (decimal)."""
    r = annual_ret_series.values
    results = []
    for i in range(len(r) - n + 1):
        prod = np.prod(1 + r[i:i+n])
        cagr = prod ** (1.0 / n) - 1
        results.append(cagr)
    return np.array(results)


def nav_to_annual(nav_series, dates_series):
    """Convert daily NAV to annual returns (calendar-year basis)."""
    s = pd.Series(nav_series.values, index=pd.to_datetime(dates_series.values))
    yearly = s.resample('YE').last()
    ann = yearly.pct_change().dropna()
    # Drop partial current year if last bar is not Dec 31
    last_date = pd.to_datetime(dates_series.values[-1])
    if last_date.month < 12 or last_date.day < 28:
        ann = ann.iloc[:-1]
    return ann


def main():
    print("=" * 60)
    print("DH[A] Scenario D  Worst10Y/Best10Y  (calendar-year basis)")
    print("=" * 60)

    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']

    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(dates)} days)")

    raw, vz = build_a2_signal(close, ret)
    lev, wn_A, wg_A, wb_A, n_trades = simulate_rebalance_A(raw, vz, threshold=0.15)
    print(f"Trades: {n_trades}, {n_trades/52.26:.1f}/yr")

    sofr = load_sofr(dates)

    print("Loading gold data from local CSV...")
    gold_1x = prepare_gold_local(dates)

    print("Building gold 2x NAV...")
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)

    print("Building bond NAV...")
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    print("Building DH[A] Scenario D portfolio NAV...")
    nav_D = build_nav(close, lev, wn_A, wg_A, wb_A, dates,
                      gold_2x, bond_3x, sofr_daily=sofr, apply_tqqq_sofr=True)

    ann = nav_to_annual(nav_D, dates)
    print(f"Annual returns: {ann.index[0].year}-{ann.index[-1].year} ({len(ann)} years)")

    r10 = rolling_nY_cagr(ann, n=10)
    r5  = rolling_nY_cagr(ann, n=5)

    worst10_idx = r10.argmin()
    best10_idx  = r10.argmax()
    years = ann.index.year

    worst10 = r10.min() * 100
    best10  = r10.max() * 100
    worst5  = r5.min() * 100

    w_start = years[worst10_idx]
    w_end   = years[worst10_idx + 9]
    b_start = years[best10_idx]
    b_end   = years[best10_idx + 9]

    print()
    print("=" * 60)
    print("RESULTS  (DH[A] Scenario D, calendar-year rolling window)")
    print("=" * 60)
    print(f"  Worst10Y : {worst10:+.2f}%  (window {w_start}-{w_end})")
    print(f"  Best10Y  : {best10:+.2f}%  (window {b_start}-{b_end})")
    print(f"  Worst5Y  : {worst5:+.2f}%")
    print()
    print("Annual returns in worst 10Y window:")
    for yr in range(w_start, w_end + 1):
        idx = ann.index[ann.index.year == yr]
        if len(idx):
            print(f"    {yr}: {ann[idx[0]]*100:+.2f}%")


if __name__ == '__main__':
    main()
