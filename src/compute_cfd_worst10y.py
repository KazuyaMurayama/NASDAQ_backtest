"""Compute calendar-year rolling Worst10Y for CFD strategies and BH 1x.

Strategies: S2_VZGated, P2 best (tv=0.8), S4_RelVol, CFD 7x [fixed], BH 1x
Method: calendar-year window (same as compute_dha_worst10y_only.py)
Uses local data only -- no internet required.
"""
import sys
import os
import types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    DATA_PATH, DATA_DIR, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_cfd_nas_sleeve,
    build_nav_strategy,
    CFD_SPREAD_LOW,
)
from dynamic_leverage_strategies import (
    compute_L_vol_target,
    compute_L_s2_vz_gated,
    compute_L_s4_relvol,
)


def prepare_gold_local(nasdaq_dates):
    """Build gold price series from local lbma_gold_daily.csv."""
    gold_path = os.path.join(DATA_DIR, 'lbma_gold_daily.csv')
    g = pd.read_csv(gold_path, parse_dates=['Date'])
    g = g.rename(columns={'USD': 'Gold'})
    nasdaq_df = pd.DataFrame({'Date': nasdaq_dates})
    merged = nasdaq_df.merge(g, on='Date', how='left')
    merged['Gold'] = merged['Gold'].ffill().bfill()
    return merged['Gold'].values


def nav_to_annual(nav_series, dates_series):
    """Convert daily NAV to annual returns (calendar-year)."""
    s = pd.Series(nav_series.values, index=pd.to_datetime(dates_series.values))
    yearly = s.resample('YE').last()
    ann = yearly.pct_change().dropna()
    last_date = pd.to_datetime(dates_series.values[-1])
    if last_date.month < 12 or last_date.day < 28:
        ann = ann.iloc[:-1]
    return ann


def rolling_nY_cagr(annual_ret_series, n=10):
    r = annual_ret_series.values
    results = []
    for i in range(len(r) - n + 1):
        prod = np.prod(1 + r[i:i+n])
        results.append(prod ** (1.0 / n) - 1)
    return np.array(results)


def report(name, nav, dates):
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    r5  = rolling_nY_cagr(ann, 5)
    w10 = r10.min() * 100
    b10 = r10.max() * 100
    w5  = r5.min() * 100
    idx_w = r10.argmin()
    idx_b = r10.argmax()
    years = ann.index.year
    w_start = years[idx_w]; w_end = years[idx_w + 9]
    b_start = years[idx_b]; b_end = years[idx_b + 9]
    print(f"  {name:<30}  Worst10Y: {w10:+.2f}% ({w_start}-{w_end})  "
          f"Best10Y: {b10:+.2f}% ({b_start}-{b_end})  Worst5Y: {w5:+.2f}%")
    return {'name': name, 'worst10': w10, 'best10': b10, 'worst5': w5,
            'worst_window': f"{w_start}-{w_end}"}


def main():
    print("=" * 70)
    print("CFD Strategies  Worst10Y/Best10Y  (calendar-year basis)")
    print("=" * 70)

    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(dates)} days)")

    # Shared assets (Scenario D)
    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    # DH Dyn signal (shared for all CFD portfolio strategies)
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    print(f"DH Dyn signal: {n_tr} trades, {n_tr/52.26:.1f}/yr")

    results = {}

    # --- S2_VZGated (tv=0.8, k_vz=0.3, gate_min=0.5) ---
    print("\nBuilding S2_VZGated NAV...")
    L_s2 = compute_L_s2_vz_gated(ret, vz, target_vol=0.8, k_vz=0.3, gate_min=0.5,
                                   n=20, l_min=1.0, l_max=7.0, step=0.5)
    nav_s2 = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                  gold_2x, bond_3x, sofr,
                                  nas_mode='CFD', cfd_leverage=L_s2.values,
                                  cfd_spread=CFD_SPREAD_LOW)
    results['S2_VZGated'] = report('S2_VZGated', nav_s2, dates)

    # --- P2 best (vol-target, tv=0.8) ---
    print("Building P2 best NAV...")
    L_p2 = compute_L_vol_target(ret, target_vol=0.8, n=20, l_min=1.0, l_max=7.0, step=0.5)
    nav_p2 = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                  gold_2x, bond_3x, sofr,
                                  nas_mode='CFD', cfd_leverage=L_p2.values,
                                  cfd_spread=CFD_SPREAD_LOW)
    results['P2 best'] = report('P2 best (tv=0.8)', nav_p2, dates)

    # --- S4_RelVol (l_base=7, k_rel=2.0) ---
    print("Building S4_RelVol NAV...")
    L_s4 = compute_L_s4_relvol(ret, vz, l_base=7.0, k_rel=2.0,
                                  l_min=1.0, step=0.5)
    nav_s4 = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                  gold_2x, bond_3x, sofr,
                                  nas_mode='CFD', cfd_leverage=L_s4.values,
                                  cfd_spread=CFD_SPREAD_LOW)
    results['S4_RelVol'] = report('S4_RelVol (l_base=7, k_rel=2.0)', nav_s4, dates)

    # --- CFD 7x fixed ---
    print("Building CFD 7x [fixed] NAV...")
    nav_7x = build_nav_strategy(close, lev_A, wn_A, wg_A, wb_A, dates,
                                  gold_2x, bond_3x, sofr,
                                  nas_mode='CFD', cfd_leverage=7.0,
                                  cfd_spread=CFD_SPREAD_LOW)
    results['CFD 7x'] = report('CFD 7x [fixed]', nav_7x, dates)

    # --- BH 1x ---
    print("Building BH 1x NAV...")
    nav_bh = (1 + close.pct_change().fillna(0)).cumprod()
    nav_bh_s = pd.Series(nav_bh.values, index=nav_bh.index)
    nav_bh_s.attrs = {}
    results['BH 1x'] = report('BH 1x (NASDAQ)', nav_bh_s, dates)

    print()
    print("=" * 70)
    print("SUMMARY  (calendar-year window)")
    print("=" * 70)
    print(f"{'Strategy':<32} {'Worst10Y':>10} {'Best10Y':>10} {'Worst5Y':>9}  Worst Window")
    print("-" * 75)
    for k, r in results.items():
        print(f"{r['name']:<32} {r['worst10']:>+9.2f}% {r['best10']:>+9.2f}% "
              f"{r['worst5']:>+8.2f}%  {r['worst_window']}")


if __name__ == '__main__':
    main()
