"""G21B: DH 改善 4 変種 × 9 指標標準 比較表生成
=================================================================
EVALUATION_STANDARD.md §3.12 v1.4 準拠の 10 列ヘッダで CSV 出力。

列順:
  1. CAGR_OOS  ⓽
  2. IS-OOS gap CAGR
  3. Sharpe_OOS  ⓒ
  4. MaxDD_FULL  ⓒ
  5. Worst10Y★_CAGR  ⓽
  6. P10_5Y▷_CAGR  ⓽
  7. Trades_yr  ⓞ
  8. Overfit(WFE)  ⓞ
  9. WFA_CI95_lo  ⓡ

ただし WFE / CI95_lo は g21c (WFA) の結果待ち。本スクリプトでは前 7 列を計算。

出力:
  - g21b_dh_improved_9metrics.csv  (5 戦略 × 9 指標)
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
    build_dh_nav_with_cost, build_dh_nav_with_timing_cost,
    metrics_from_nav, apply_tax_etf_decimal,
)
from g21a_dh_improved_variants import VARIANT_SPECS, build_variant, DH_PER_UNIT


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
        # Trades_yr / WFE / CI95_lo は g21c で算出（ここでは reference として 27 を記録）
    )


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G21B: DH 改善 4 変種 × 9 指標標準（一次根拠）')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    ret_nas = a['ret']

    rows = []

    # REF: 現行 DH
    nav_ref, _ = build_dh_nav_with_cost(
        a['close'], a['lev_raw'], a['wn_A'], a['wg_A'], a['wb_A'],
        dates, a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    m = calc_9metrics(nav_ref, dates, ret_nas)
    m['Strategy'] = 'DH Dyn 2x3x [A] (現行 REF)'
    rows.append(m)
    print(f'\n[REF] DH Dyn 2x3x [A]:')
    print(f'  CAGR_OOS={m["CAGR_OOS_pct"]:+.2f}%, gap={m["IS_OOS_gap_pp"]:+.2f}pp, '
          f'Sharpe={m["Sharpe_OOS"]:+.3f}, MaxDD={m["MaxDD_FULL_pct"]:+.2f}%')
    print(f'  Worst10Y={m["Worst10Y_CAGR_pct"]:+.2f}%, P10_5Y={m["P10_5Y_CAGR_pct"]:+.2f}%')

    for label, spec in VARIANT_SPECS.items():
        nav_v, _ = build_variant(a, spec)
        m = calc_9metrics(nav_v, dates, ret_nas)
        m['Strategy'] = label
        rows.append(m)
        print(f'\n[{label}]:')
        print(f'  CAGR_OOS={m["CAGR_OOS_pct"]:+.2f}%, gap={m["IS_OOS_gap_pp"]:+.2f}pp, '
              f'Sharpe={m["Sharpe_OOS"]:+.3f}, MaxDD={m["MaxDD_FULL_pct"]:+.2f}%')
        print(f'  Worst10Y={m["Worst10Y_CAGR_pct"]:+.2f}%, P10_5Y={m["P10_5Y_CAGR_pct"]:+.2f}%')

    df = pd.DataFrame(rows)
    cols_order = ['Strategy', 'CAGR_OOS_pct', 'IS_OOS_gap_pp', 'Sharpe_OOS',
                  'MaxDD_FULL_pct', 'Worst10Y_CAGR_pct', 'P10_5Y_CAGR_pct']
    df = df[cols_order]
    csv = os.path.join(BASE, 'g21b_dh_improved_9metrics.csv')
    df.to_csv(csv, index=False, float_format='%.4f')
    print(f'\n→ 9-metric CSV: {csv}')

    print('\n[一次評価サマリ — gap 縮小 vs REF]')
    ref_gap = float(df.iloc[0]['IS_OOS_gap_pp'])
    print(f'  REF gap = {ref_gap:+.2f}pp')
    for i in range(1, len(df)):
        row = df.iloc[i]
        diff = float(row['IS_OOS_gap_pp']) - ref_gap
        flag = '✅縮小' if diff < -2.0 else ('⚠中立' if diff < 1.0 else '❌悪化')
        print(f'  {row["Strategy"]:35s}  gap={float(row["IS_OOS_gap_pp"]):+.2f}pp '
              f'(Δ={diff:+.2f}pp) {flag}')


if __name__ == '__main__':
    main()
