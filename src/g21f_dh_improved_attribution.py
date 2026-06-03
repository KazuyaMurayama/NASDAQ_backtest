"""G21F: DH-T4 vs DH-REF 年次寄与分解 + DH-T4 単独の年次税後リターン CSV 出力
=================================================================
出力:
  - g21f_dh_t4_vs_ref_yearly_attribution.csv  (1974-2026 年次差分)
  - g21f_dh_t4_yearly_returns_aftertax.csv     (DH-T4 単独・税後年次%・後で v2 レポート貼付用)
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
from g18_daily_trade_cost_wfa import (
    build_dh_nav_with_cost, metrics_from_nav, apply_tax_etf_decimal,
)
from g21a_dh_improved_variants import VARIANT_SPECS, build_variant, DH_PER_UNIT


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G21F: DH-T4 vs DH-REF 年次寄与分解')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']

    nav_ref, _ = build_dh_nav_with_cost(
        a['close'], a['lev_raw'], a['wn_A'], a['wg_A'], a['wb_A'],
        dates, a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    nav_t4, _ = build_variant(a, VARIANT_SPECS['DH-T4 (Full vz+lmax+F10e)'])

    m_ref = metrics_from_nav(nav_ref, dates, a['ret'])
    m_t4  = metrics_from_nav(nav_t4,  dates, a['ret'])
    yr_ref_pre = m_ref['yearly']
    yr_t4_pre  = m_t4['yearly']
    yr_ref_aft = yr_ref_pre.apply(apply_tax_etf_decimal)
    yr_t4_aft  = yr_t4_pre.apply(apply_tax_etf_decimal)

    diff = (yr_t4_aft - yr_ref_aft).dropna()
    df = pd.DataFrame({
        'DH-REF_pre_pct':  yr_ref_pre * 100,
        'DH-T4_pre_pct':   yr_t4_pre  * 100,
        'DH-REF_aft_pct':  yr_ref_aft * 100,
        'DH-T4_aft_pct':   yr_t4_aft  * 100,
        'diff_aft_pp':     diff       * 100,
    }).dropna()
    df.index.name = 'year'

    csv = os.path.join(BASE, 'g21f_dh_t4_vs_ref_yearly_attribution.csv')
    df.to_csv(csv, float_format='%.4f')
    print(f'\n→ Yearly diff CSV: {csv}')

    # DH-T4 単独 1977-2026 年次 after-tax % (v2 レポート貼付用)
    yr_only = yr_t4_aft.loc[(yr_t4_aft.index >= 1977) & (yr_t4_aft.index <= 2026)] * 100
    out2 = pd.DataFrame({'DH-T4_aft_pct': yr_only})
    csv2 = os.path.join(BASE, 'g21f_dh_t4_yearly_returns_aftertax.csv')
    out2.to_csv(csv2, float_format='%.4f')
    print(f'→ DH-T4 yearly CSV: {csv2}')

    # 統計サマリ
    print('\n[DH-T4 (1977-2026, after-tax, moderate) 統計サマリ]')
    s = yr_only.dropna()
    print(f'  Mean       : {s.mean():+.2f}%')
    print(f'  Median     : {s.median():+.2f}%')
    print(f'  Std        : {s.std():.2f}%')
    print(f'  Min        : {s.min():+.2f}%')
    print(f'  Max        : {s.max():+.2f}%')
    print(f'  Positive yrs: {int((s > 0).sum())} / {len(s)} ({int((s > 0).sum())/len(s)*100:.1f}%)')
    print(f'  Negative yrs: {int((s < 0).sum())} / {len(s)} ({int((s < 0).sum())/len(s)*100:.1f}%)')

    print('\n[OOS (2021-2026) 年次差分: DH-T4 - DH-REF]')
    oos = df.loc[df.index >= 2021]
    for y, row in oos.iterrows():
        print(f'  {int(y)}: T4={row["DH-T4_aft_pct"]:+7.2f}% REF={row["DH-REF_aft_pct"]:+7.2f}%  diff={row["diff_aft_pp"]:+7.2f}pp')
    sum_oos = oos['diff_aft_pp'].sum()
    mean_oos = oos['diff_aft_pp'].mean()
    print(f'\n  OOS 6yr diff sum: {sum_oos:+.2f}pp, 平均: {mean_oos:+.2f}pp')

    # 単年集中度
    if abs(sum_oos) > 0.1:
        max_year = oos['diff_aft_pp'].abs().idxmax()
        max_val = oos.loc[max_year, 'diff_aft_pp']
        total_abs = sum([abs(v) for v in oos['diff_aft_pp'].values])
        if total_abs > 0:
            concentration = abs(max_val) / total_abs * 100
            print(f'  最大絶対 diff 年: {int(max_year)} ({max_val:+.2f}pp), 集中度: {concentration:.1f}%')


if __name__ == '__main__':
    main()
