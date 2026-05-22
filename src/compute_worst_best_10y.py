"""
Compute annual returns and rolling 10Y Worst/Best CAGR for all 6 strategies.
Strategies: DH[A] Scenario D, P01_Dyn×HY, P02_Dyn×CPI, P05_HY×CPI, T1, T2
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
    DATA_PATH, DATA_DIR, TRADING_DAYS, SWAP_SPREAD, DELAY,
)
from test_portfolio_diversification import prepare_gold_data

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rolling_nY_cagr(annual_ret_series, n=10):
    """Rolling n-year CAGR from annual return series (decimal form)."""
    r = annual_ret_series.values
    results = []
    for i in range(len(r) - n + 1):
        prod = np.prod(1 + r[i:i+n])
        cagr = prod ** (1.0 / n) - 1
        results.append(cagr)
    return np.array(results)


def nav_to_annual(nav_series, dates_series):
    """Convert daily NAV series to annual returns (decimal)."""
    nav_df = pd.Series(nav_series.values, index=pd.to_datetime(dates_series.values))
    yearly = nav_df.resample('YE').last()
    ann = yearly.pct_change().dropna()
    # Drop partial current year if last bar is not Dec 31
    last_date = pd.to_datetime(dates_series.values[-1])
    if last_date.month < 12 or last_date.day < 28:
        ann = ann.iloc[:-1]  # drop partial last year
    return ann


def compute_worst_best_10y(annual_ret_series, name):
    r10 = rolling_nY_cagr(annual_ret_series, n=10)
    r5  = rolling_nY_cagr(annual_ret_series, n=5)
    if len(r10) == 0:
        return None
    worst10 = r10.min() * 100
    best10  = r10.max() * 100
    worst5  = r5.min() * 100 if len(r5) > 0 else np.nan
    # Find worst window years
    idx_w = r10.argmin()
    # Annual returns available for reference
    years = annual_ret_series.index.year
    w_start = years[idx_w] if idx_w < len(years) else "?"
    w_end   = years[idx_w + 9] if idx_w + 9 < len(years) else "?"
    return dict(name=name, worst10=worst10, best10=best10, worst5=worst5,
                worst10_period=f"{w_start}–{w_end}")


def main():
    print("=" * 70)
    print("Loading NASDAQ data...")
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    print(f"  {dates.iloc[0].date()} → {dates.iloc[-1].date()} ({len(df):,} bars)")

    print("Building DH[A] signal (Approach A)...")
    raw, vz = build_a2_signal(close, ret)
    lev, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw, vz)
    wn_A_arr = wn_A  # numpy array

    print("Loading gold & SOFR...")
    gold_1x  = prepare_gold_data(dates)
    sofr     = load_sofr(dates)

    print("Building Scenario D asset NAVs...")
    gold_2x  = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x  = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x  = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    print("Building DH[A] Scenario D NAV...")
    nav_D = build_nav(close, lev, wn_A_arr, wg_A, wb_A, dates,
                      gold_2x, bond_3x, sofr_daily=sofr, apply_tqqq_sofr=True)

    # --- Timing signals ---
    print("\nLoading timing signals...")
    sig_path = os.path.join(DATA_DIR, '..', 'data', 'timing_signals_raw.csv')
    sig_path = os.path.normpath(sig_path)
    sig_df = pd.read_csv(sig_path, parse_dates=['DATE'], index_col='DATE')

    # Align signals to NASDAQ trading dates
    dates_idx = pd.to_datetime(dates.values)

    # HY gate: z_thresh=1.0, slope=0.5, min_gate=0.2
    hy_raw = sig_df['hy_spread'].reindex(dates_idx).ffill(limit=30).bfill(limit=60)
    hy_mean = hy_raw.rolling(252, min_periods=63).mean()
    hy_std  = hy_raw.rolling(252, min_periods=63).std().replace(0, np.nan)
    hy_z    = (hy_raw - hy_mean) / hy_std
    hy_gate = np.clip(1.0 - np.maximum(0, hy_z.fillna(0) - 1.0) * 0.5, 0.2, 1.0).values

    # CPI gate: cpi_thresh=5.0, reduce_factor=0.3
    cpi_raw = sig_df['cpi_yoy'].reindex(dates_idx).ffill(limit=60).bfill(limit=60)
    cpi_gate = np.where(cpi_raw.fillna(0).values < 5.0, 1.0, 0.3)

    # Dyn_Corr gate: NASDAQ-Bond 60-day rolling correlation, min_gate=0.2
    bond_ret_1x = pd.Series(bond_1x).pct_change().fillna(0)
    rho_nb = ret.rolling(60, min_periods=30).corr(bond_ret_1x).fillna(0)
    dyn_gate = np.clip(np.maximum(0.0, -rho_nb.values), 0.2, 1.0)

    def apply_combo_gates(wn_base, nas_gates=None, bond_gate_arr=None):
        wn = wn_base.copy()
        if nas_gates is not None:
            for g in nas_gates:
                wn = np.clip(wn * g, 0.0, 1.0)
        rest = 1.0 - wn
        wg = rest * 0.5
        wb = rest * 0.5
        if bond_gate_arr is not None:
            wb = wb * bond_gate_arr
        return wn, wg, wb

    combos = [
        ('P01_Dyn×HY', [hy_gate], dyn_gate),
        ('P02_Dyn×CPI', [cpi_gate], dyn_gate),
        ('P05_HY×CPI', [hy_gate, cpi_gate], None),
    ]

    combo_navs = {}
    for name, nas_gates, bond_gate_arr in combos:
        print(f"Building {name} NAV...")
        wn_c, wg_c, wb_c = apply_combo_gates(wn_A_arr, nas_gates, bond_gate_arr)
        nav_c = build_nav(close, lev, wn_c, wg_c, wb_c, dates,
                          gold_2x, bond_3x, sofr_daily=sofr, apply_tqqq_sofr=True)
        combo_navs[name] = nav_c

    # --- DH[A] and combos annual returns ---
    all_results = []

    print("\nComputing DH[A] Scenario D rolling 10Y...")
    ann_D = nav_to_annual(nav_D, dates)
    res_D = compute_worst_best_10y(ann_D, 'DH[A] Scenario D')
    all_results.append(res_D)

    for name, nav_c in combo_navs.items():
        print(f"Computing {name} rolling 10Y...")
        ann_c = nav_to_annual(nav_c, dates)
        res_c = compute_worst_best_10y(ann_c, name)
        all_results.append(res_c)

    # --- T1 annual returns ---
    print("\nLoading T1 yearly returns...")
    t1_path = os.path.join(BASE, 't1_yearly_returns.csv')
    t1_df = pd.read_csv(t1_path, index_col=0)
    t1_df.index = t1_df.index.astype(int)
    t1_ret = t1_df['return']
    # 2026 is partial (value 0.0), drop it
    t1_ret = t1_ret[t1_ret.index <= 2025]
    t1_ret.index = pd.to_datetime([f"{y}-12-31" for y in t1_ret.index])
    res_T1 = compute_worst_best_10y(t1_ret, 'T1 Pure Turtle Long')
    all_results.append(res_T1)

    # --- T2 annual returns ---
    print("Loading T2 yearly returns...")
    t2_path = os.path.join(BASE, 't2_yearly_returns.csv')
    t2_df = pd.read_csv(t2_path, index_col=0)
    t2_df.index = t2_df.index.astype(int)
    t2_ret = t2_df['return']
    t2_ret = t2_ret[t2_ret.index <= 2025]
    t2_ret.index = pd.to_datetime([f"{y}-12-31" for y in t2_ret.index])
    res_T2 = compute_worst_best_10y(t2_ret, 'T2 Pure Turtle L/S')
    all_results.append(res_T2)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("ROLLING 10Y CAGR RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<28} {'Worst10Y':>10} {'Best10Y':>10} {'Worst5Y':>10}  Worst10Y Period")
    print("-" * 75)
    for r in all_results:
        if r is None:
            continue
        w10_str = f"{r['worst10']:+.2f}%"
        b10_str = f"{r['best10']:+.2f}%"
        w5_str  = f"{r['worst5']:+.2f}%" if not np.isnan(r['worst5']) else "  n/a"
        print(f"{r['name']:<28} {w10_str:>10} {b10_str:>10} {w5_str:>10}  {r['worst10_period']}")

    # Also print DH[A] annual returns for reference
    print("\n--- DH[A] Scenario D annual returns (selected) ---")
    for yr, val in ann_D.items():
        if yr.year in [1974, 1975, 1981, 1982, 1988, 1994, 2000, 2008, 2015, 2022, 2024]:
            print(f"  {yr.year}: {val*100:+.2f}%")

    return all_results


if __name__ == '__main__':
    main()
