"""G22B: DH-Z シリーズ 9 指標標準比較表
=================================================================
EVALUATION_STANDARD §3.12 v1.4 準拠の指標を 6 戦略 (REF + Z1〜Z5) で計算。
WFE / CI95_lo は g22c (WFA) で別途取得。

出力:
  - g22b_dh_alloc_timing_9metrics.csv
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
from g22a_dh_alloc_timing_variants import VARIANT_SPECS, build_variant


def calc_9metrics(nav, dates, ret_nas):
    m = metrics_from_nav(nav, dates, ret_nas)
    yr_pre = m['yearly']
    yr_aft = yr_pre.apply(apply_tax_etf_decimal)
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return float('nan')
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is, cagr_oos = _geo(is_subset), _geo(oos_subset)
    return dict(
        CAGR_OOS_pct=cagr_oos*100,
        IS_OOS_gap_pp=(cagr_is - cagr_oos)*100,
        Sharpe_OOS=m['Sharpe_OOS'],
        MaxDD_FULL_pct=m['MaxDD_FULL']*100,
        Worst10Y_CAGR_pct=m['Worst10Y_star']*100,
        P10_5Y_CAGR_pct=m['P10_5Y']*100,
    )


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G22B: DH-Z シリーズ 9 指標標準 (REF + Z1〜Z5)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    ret_nas = a['ret']

    rows = []

    # REF (always_in + dh_base)
    nav_ref, _, _, _, _ = build_variant(a, 'always_in', 'dh_base')
    m = calc_9metrics(nav_ref, dates, ret_nas)
    m['Strategy'] = 'DH Dyn 2x3x [A] (REF)'
    rows.append(m)
    print(f'\n[REF] gap={m["IS_OOS_gap_pp"]:+.2f}pp, CAGR_OOS={m["CAGR_OOS_pct"]:+.2f}%, '
          f'Sharpe={m["Sharpe_OOS"]:+.3f}, MaxDD={m["MaxDD_FULL_pct"]:+.2f}%, '
          f'W10Y={m["Worst10Y_CAGR_pct"]:+.2f}%, P10_5Y={m["P10_5Y_CAGR_pct"]:+.2f}%')

    for label, (tkey, akey) in VARIANT_SPECS.items():
        nav, _, _, _, _ = build_variant(a, tkey, akey)
        m = calc_9metrics(nav, dates, ret_nas)
        m['Strategy'] = label
        rows.append(m)
        print(f'[{label[:42]:42s}] gap={m["IS_OOS_gap_pp"]:+.2f}pp, CAGR_OOS={m["CAGR_OOS_pct"]:+.2f}%, '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}, MaxDD={m["MaxDD_FULL_pct"]:+.2f}%, '
              f'W10Y={m["Worst10Y_CAGR_pct"]:+.2f}%, P10_5Y={m["P10_5Y_CAGR_pct"]:+.2f}%')

    df = pd.DataFrame(rows)
    cols = ['Strategy', 'CAGR_OOS_pct', 'IS_OOS_gap_pp', 'Sharpe_OOS',
            'MaxDD_FULL_pct', 'Worst10Y_CAGR_pct', 'P10_5Y_CAGR_pct']
    df = df[cols]
    csv = os.path.join(BASE, 'g22b_dh_alloc_timing_9metrics.csv')
    df.to_csv(csv, index=False, float_format='%.4f')
    print(f'\n→ 9-metric CSV: {csv}')

    print('\n[一次評価サマリ — gap 縮小 vs REF + 防御性能チェック]')
    ref_gap = float(df.iloc[0]['IS_OOS_gap_pp'])
    ref_w10y = float(df.iloc[0]['Worst10Y_CAGR_pct'])
    ref_cagr_oos = float(df.iloc[0]['CAGR_OOS_pct'])
    print(f'  REF: gap={ref_gap:+.2f}pp, Worst10Y={ref_w10y:+.2f}%, CAGR_OOS={ref_cagr_oos:+.2f}%')
    print()
    for i in range(1, len(df)):
        r = df.iloc[i]
        gap = float(r['IS_OOS_gap_pp']); w10y = float(r['Worst10Y_CAGR_pct']); cagr = float(r['CAGR_OOS_pct'])
        d_gap = gap - ref_gap; d_w10y = w10y - ref_w10y; d_cagr = cagr - ref_cagr_oos
        gap_ok = '✅縮小' if d_gap < -2.0 else ('⚠中立' if d_gap < 1.0 else '❌悪化')
        def_ok = '✅維持' if w10y > 10.0 else ('⚠減少' if w10y > 8.0 else '❌失格')
        cagr_ok = '✅向上' if d_cagr > 1.0 else ('⚠同等' if d_cagr > -1.0 else '❌劣化')
        print(f'  {r["Strategy"][:40]:40s}  gap={gap:+5.2f}({d_gap:+5.2f}){gap_ok}  '
              f'W10Y={w10y:+5.2f}{def_ok}  CAGR_OOS={cagr:+5.2f}({d_cagr:+5.2f}){cagr_ok}')


if __name__ == '__main__':
    main()
