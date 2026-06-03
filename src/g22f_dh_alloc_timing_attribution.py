"""G22F: DH-Z2 (F10 tilt + binary vz) 年次寄与分解 + 統計サマリ + OOS 累積
=================================================================
v4 レポート (§5 列 / §6 列 / §6-2 行) 貼付用 CSV を出力。

出力:
  - g22f_dh_alloc_timing_attribution.csv  (1974-2026 Z2 vs REF 年次比較)
  - g22f_dh_z2_yearly_returns_aftertax.csv (1977-2026 Z2 単独・税後年次%)
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets, BASE
from g18_daily_trade_cost_wfa import metrics_from_nav, apply_tax_etf_decimal
from g22a_dh_alloc_timing_variants import build_variant


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G22F: DH-Z2 (F10 tilt + binary vz) 年次寄与 + 統計サマリ + OOS 累積')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']

    nav_ref, _, _, _, _ = build_variant(a, 'always_in', 'dh_base')
    nav_z2,  _, _, _, _ = build_variant(a, 'vz_binary', 'f10')

    m_ref = metrics_from_nav(nav_ref, dates, a['ret'])
    m_z2  = metrics_from_nav(nav_z2,  dates, a['ret'])
    yr_ref_pre = m_ref['yearly']; yr_z2_pre = m_z2['yearly']
    yr_ref_aft = yr_ref_pre.apply(apply_tax_etf_decimal)
    yr_z2_aft  = yr_z2_pre.apply(apply_tax_etf_decimal)

    diff = (yr_z2_aft - yr_ref_aft).dropna()
    df = pd.DataFrame({
        'DH-REF_pre_pct':  yr_ref_pre * 100,
        'DH-Z2_pre_pct':   yr_z2_pre  * 100,
        'DH-REF_aft_pct':  yr_ref_aft * 100,
        'DH-Z2_aft_pct':   yr_z2_aft  * 100,
        'diff_aft_pp':     diff       * 100,
    }).dropna()
    df.index.name = 'year'

    csv = os.path.join(BASE, 'g22f_dh_alloc_timing_attribution.csv')
    df.to_csv(csv, float_format='%.4f')
    print(f'\n→ Yearly attribution CSV: {csv}')

    # Z2 単独 1977-2026 年次税後（v4 レポート §5 貼付用）
    yr_only = yr_z2_aft.loc[(yr_z2_aft.index >= 1977) & (yr_z2_aft.index <= 2026)] * 100
    out2 = pd.DataFrame({'DH-Z2_aft_pct': yr_only})
    csv2 = os.path.join(BASE, 'g22f_dh_z2_yearly_returns_aftertax.csv')
    out2.to_csv(csv2, float_format='%.4f')
    print(f'→ Z2 yearly CSV: {csv2}')

    # 統計サマリ (1974-2026)
    print('\n[Z2 統計サマリ 1974-2026 (after-tax, moderate)]')
    s = (yr_z2_aft.dropna() * 100)
    print(f'  Mean: {s.mean():+.2f}%, Median: {s.median():+.2f}%, Std: {s.std():.2f}%')
    print(f'  Max: {s.max():+.2f}%, Min: {s.min():+.2f}%')
    print(f'  Pos yrs: {int((s>0).sum())}, Neg yrs: {int((s<0).sum())}, Total: {len(s)}')

    # OOS 累積
    print('\n[Z2 vs REF OOS (2021-2026) 年次差分]')
    oos = df.loc[df.index >= 2021]
    for y, row in oos.iterrows():
        print(f'  {int(y)}: Z2={row["DH-Z2_aft_pct"]:+7.2f}%  REF={row["DH-REF_aft_pct"]:+7.2f}%  diff={row["diff_aft_pp"]:+7.2f}pp')
    sum_oos = oos['diff_aft_pp'].sum()
    mean_oos = oos['diff_aft_pp'].mean()
    print(f'\n  OOS 6yr diff sum: {sum_oos:+.2f}pp, 平均: {mean_oos:+.2f}pp/年')

    z2_oos = yr_z2_aft.loc[(yr_z2_aft.index>=2021)&(yr_z2_aft.index<=2026)]
    cum_z2 = float((1+z2_oos).prod())
    print(f'\n  Z2 OOS 6 年累積倍率: ×{cum_z2:.2f} (100 万円→ {cum_z2*100:.0f} 万円)')


if __name__ == '__main__':
    main()
