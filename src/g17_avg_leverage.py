"""
G17: 戦略別 平均レバレッジ (NAV加重) 算出
=================================================================
取引コスト計算のために、4戦略の OOS 期間平均レバレッジ (NAV加重) を算出。

出力:
  - g17_avg_leverage.csv (戦略別 L_avg)
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

from g14_wfa_sbi_cfd import (
    load_shared_assets, _make_nav,
    S2_LMAX5, S2_LMAX5P5, S2_LMAX7, SBI_CFD_SPREAD,
    BASE, TODAY,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated

OOS_START = pd.Timestamp('2021-05-08')
OOS_END   = pd.Timestamp('2026-03-26')


def nav_weighted_mean(L_series, nav_series, mask):
    """NAV-weighted mean of L over the masked period."""
    L = np.asarray(L_series)[mask]
    n = np.asarray(nav_series)[mask]
    n = np.where(n > 0, n, 0)
    if n.sum() == 0:
        return float(np.mean(L))
    return float(np.average(L, weights=n))


def trade_count_three_defs(L_series):
    """3 definitions of trades count (annualized)."""
    L = pd.Series(L_series)
    dL = L.diff().abs()
    n_total = len(L)
    years = n_total / 252.0
    return dict(
        trades_yr_lev_gt05  = int((dL > 0.5).sum())  / years,
        trades_yr_lev_gt001 = int((dL > 0.01).sum()) / years,
        sum_abs_dL_per_yr   = float(dL.sum())        / years,
    )


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 72)
    print(f'G17: 戦略別 平均レバレッジ (NAV加重) — OOS期間: {OOS_START.date()} ~ {OOS_END.date()}')
    print(f'実行日: {TODAY}')
    print('=' * 72)

    print('\n[S1] Loading shared assets (via g14)...')
    a = load_shared_assets()
    dates = a['dates']
    is_oos = (dates >= OOS_START) & (dates <= OOS_END)
    is_full = pd.Series(True, index=dates.index)
    print(f'  OOS days: {is_oos.sum()}, FULL days: {len(dates)}')

    # ---- Build NAVs (for NAV-weighted L_avg) ----
    print('\n[S2] Building NAVs for each strategy...')

    nav_e4 = _make_nav(a['close'], a['lev_mod_e4'], a['wn_A'], a['wg_A'], a['wb_A'],
                       a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                       a['L_s2_lmax7'].values)
    nav_f10 = _make_nav(a['close'], a['lev_mod_e4'], a['wn_f10'], a['wg_f10'], a['wb_f10'],
                        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                        a['L_s2_lmax7'].values)
    nav_d5 = _make_nav(a['close'], a['lev_mod_065'], a['wn_A'], a['wg_A'], a['wb_A'],
                       a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                       a['L_s2_lmax5p5'].values)
    print('  NAVs built.')

    # ---- L_s2 series for 3 CFD strategies ----
    L_e4  = a['L_s2_lmax7'].values
    L_f10 = a['L_s2_lmax7'].values  # same S2 signal as E4
    L_d5  = a['L_s2_lmax5p5'].values

    # ---- Effective Notional Leverage = L_s2 × (1 - lev_mod position scale) ---
    # In NAV, the actual CFD notional = L_s2 * wn (NASDAQ sleeve weight) * lev_mod
    # For simplicity, use L_s2 directly (worst-case proxy for trading cost)
    # Note: actual notional = L_s2 * (wn * lev_mod), but we use L_s2 as
    # the standard CFD leverage descriptor.

    # ---- DH Dyn [A]: lev_raw is signal [0,1], NDX leverage via TQQQ 3x = lev_raw × 3 ----
    L_dh = np.asarray(a['lev_raw']) * 3.0

    # ---- Compute L_avg (FULL + OOS) ----
    rows = []
    for sid, L, nav in [
        ('E4 Regime k_lt',           L_e4,  nav_e4),
        ('F10 eps015',               L_f10, nav_f10),
        ('D5 vz065 lmax5p5',         L_d5,  nav_d5),
        ('DH Dyn 2x3x [A]',          L_dh,  None),  # NAV-weighted not strictly needed
    ]:
        if nav is not None:
            nav_arr = nav.values
            L_avg_full = nav_weighted_mean(L, nav_arr, is_full.values)
            L_avg_oos  = nav_weighted_mean(L, nav_arr, is_oos.values)
        else:
            # DH [A]: simple time-weighted
            L_avg_full = float(np.mean(L))
            L_avg_oos  = float(np.mean(L[is_oos.values]))
        L_max_full = float(np.max(L))
        L_max_oos  = float(np.max(L[is_oos.values]))
        trades_full = trade_count_three_defs(L)
        rows.append(dict(
            strategy=sid,
            L_avg_full=L_avg_full,
            L_avg_oos=L_avg_oos,
            L_max_full=L_max_full,
            L_max_oos=L_max_oos,
            trades_yr_lev_gt05=trades_full['trades_yr_lev_gt05'],
            trades_yr_lev_gt001=trades_full['trades_yr_lev_gt001'],
            sum_abs_dL_per_yr=trades_full['sum_abs_dL_per_yr'],
        ))

    df = pd.DataFrame(rows)
    out_path = os.path.join(BASE, 'g17_avg_leverage.csv')
    df.to_csv(out_path, index=False, float_format='%.4f')

    print('\n[S3] Results:')
    print(df.to_string(index=False))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
