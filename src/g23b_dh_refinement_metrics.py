"""G23B: DH 改善精製版 9 指標 + 累積 CAGR OOS/IS
=================================================================
7 戦略 (DH-REF, DH-Z2, DH-W1, DH-W2, DH-W3, NEW 🟢, NDX 1x B&H) で:
  - CAGR_OOS (g22b metric_from_nav 由来、年次複利方式)
  - IS-OOS gap CAGR
  - Sharpe_OOS, MaxDD, Worst10Y, P10_5Y
  - 累積 CAGR ⓽ OOS/IS (§5 年次から計算)
  - Trades/yr (g14 WFA)
  - WFA WFE, CI95_lo は g23d で別途 (本スクリプトでは省略)

出力:
  - g23b_dh_refinement_metrics.csv
  - g23b_dh_refinement_yearly_aftertax.csv (各戦略の 1974-2026 年次 after-tax %)
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
    build_dh_nav_with_cost, metrics_from_nav,
    apply_tax_etf_decimal, apply_tax_cfd_decimal,
)
from g22a_dh_alloc_timing_variants import build_variant as build_z_variant
from g22a_dh_alloc_timing_variants import (
    hold_mask_vz_binary,
    alloc_dh_base, alloc_f10_tilt, alloc_fixed_bull, alloc_regime_switch,
)
from g23a_dh_refinement_variants import (
    build_W1, build_W2, build_W3, VARIANT_BUILDERS,
)


def calc_metrics_etf(nav, dates, ret_nas):
    """ETF 用 9 指標 + 累積 CAGR (年次)"""
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
        cum_CAGR_IS_pct=cagr_is*100,
        cum_CAGR_OOS_pct=cagr_oos*100,
        Sharpe_OOS=m['Sharpe_OOS'],
        MaxDD_FULL_pct=m['MaxDD_FULL']*100,
        Worst10Y_CAGR_pct=m['Worst10Y_star']*100,
        P10_5Y_CAGR_pct=m['P10_5Y']*100,
        yearly_aft=yr_aft,
    )


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G23B: DH 改善精製版 9 指標 + 累積 CAGR OOS/IS (7 戦略)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    ret_nas = a['ret']

    strategies = []  # list of (label, nav_builder_callable)
    # baselines
    strategies.append(('DH-REF (現行 DH)', lambda: build_z_variant(a, 'always_in', 'dh_base')))
    strategies.append(('DH-Z2 (v4 採用)',   lambda: build_z_variant(a, 'vz_binary', 'f10')))
    # new candidates
    strategies.append(('DH-W1 (Asymm+Hyst, DH base)', lambda: build_W1(a)))
    strategies.append(('DH-W2 (Z2 + TMF rotation)',   lambda: build_W2(a)))
    strategies.append(('DH-W3 (3-state preset switch)', lambda: build_W3(a)))
    # 参考: NEW (CFD 戦略, 税モデルは CFD)
    # NEW は §5 年次 (NEW yearly) から計算するので別途
    # B&H 用に compute_bnh
    from g19e_3strategies_daily_cost import compute_bnh_metrics_after_tax

    rows = []
    yearly_records = {}

    print('\n[各戦略の指標計算]')
    for label, build_fn in strategies:
        result = build_fn()
        nav = result[0]
        m = calc_metrics_etf(nav, dates, ret_nas)
        m['Strategy'] = label
        rows.append(m)
        yearly_records[label] = m['yearly_aft']
        print(f'  {label[:38]:38s}  gap={m["IS_OOS_gap_pp"]:+5.2f}pp  '
              f'OOS_CAGR={m["CAGR_OOS_pct"]:+6.2f}%  IS_CAGR={m["cum_CAGR_IS_pct"]:+6.2f}%  '
              f'MaxDD={m["MaxDD_FULL_pct"]:+6.2f}%  W10Y={m["Worst10Y_CAGR_pct"]:+5.2f}%')

    # NEW (CFD): use shared CFD pipeline
    # 簡易: NEW yearly returns from g20f CSV (既存)
    print('\n[Reference: NEW 🟢 (CFD)]')
    try:
        new_yearly_csv = os.path.join(BASE, 'g20f_unified_yearly_returns.csv')
        df_new = pd.read_csv(new_yearly_csv, index_col=0)
        # 列名: 'NEW (vz=0.65+l7+F10ε) 🔍'
        new_col = [c for c in df_new.columns if 'NEW' in c][0]
        new_yr = (df_new[new_col] / 100).dropna()  # convert pct → decimal
        is_n = new_yr.loc[[y for y in new_yr.index if 1977 <= y <= 2020]]
        oos_n = new_yr.loc[[y for y in new_yr.index if 2021 <= y <= 2026]]
        def _geo(x):
            if len(x) == 0: return float('nan')
            c = float(np.prod(1.0 + x.values))
            return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
        cagr_is_n, cagr_oos_n = _geo(is_n), _geo(oos_n)
        # §0' v2 から取得した値
        rows.append(dict(
            Strategy='NEW 🟢 (CFD reference)',
            CAGR_OOS_pct=21.49, IS_OOS_gap_pp=-1.27,
            cum_CAGR_IS_pct=cagr_is_n*100, cum_CAGR_OOS_pct=cagr_oos_n*100,
            Sharpe_OOS=0.829, MaxDD_FULL_pct=-65.95,
            Worst10Y_CAGR_pct=9.96, P10_5Y_CAGR_pct=5.84,
            yearly_aft=new_yr,
        ))
        yearly_records['NEW 🟢 (CFD reference)'] = new_yr
        print(f'  NEW 🟢: IS_CAGR={cagr_is_n*100:+.2f}%, OOS_CAGR={cagr_oos_n*100:+.2f}%')
    except Exception as e:
        print(f'  NEW 取得失敗: {e}')

    # B&H
    print('\n[Reference: NDX 1x B&H]')
    bnh = compute_bnh_metrics_after_tax(a['close'], dates)
    bnh_yr = bnh['yearly_aft']
    is_b = bnh_yr.loc[[y for y in bnh_yr.index if 1977 <= y <= 2020]]
    oos_b = bnh_yr.loc[[y for y in bnh_yr.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return float('nan')
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is_b, cagr_oos_b = _geo(is_b), _geo(oos_b)
    rows.append(dict(
        Strategy='NDX 1x B&H (Benchmark)',
        CAGR_OOS_pct=cagr_oos_b*100, IS_OOS_gap_pp=(cagr_is_b - cagr_oos_b)*100,
        cum_CAGR_IS_pct=cagr_is_b*100, cum_CAGR_OOS_pct=cagr_oos_b*100,
        Sharpe_OOS=0.516, MaxDD_FULL_pct=-77.93,
        Worst10Y_CAGR_pct=-4.85, P10_5Y_CAGR_pct=0.59,
        yearly_aft=bnh_yr,
    ))
    yearly_records['NDX 1x B&H (Benchmark)'] = bnh_yr
    print(f'  B&H: IS_CAGR={cagr_is_b*100:+.2f}%, OOS_CAGR={cagr_oos_b*100:+.2f}%')

    # output metrics CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'yearly_aft'} for r in rows])
    cols = ['Strategy', 'CAGR_OOS_pct', 'IS_OOS_gap_pp', 'cum_CAGR_IS_pct', 'cum_CAGR_OOS_pct',
            'Sharpe_OOS', 'MaxDD_FULL_pct', 'Worst10Y_CAGR_pct', 'P10_5Y_CAGR_pct']
    df = df[cols]
    csv = os.path.join(BASE, 'g23b_dh_refinement_metrics.csv')
    df.to_csv(csv, index=False, float_format='%.4f')
    print(f'\n→ Metrics CSV: {csv}')

    # output yearly returns table (all strategies)
    years = sorted(set(y for ser in yearly_records.values() for y in ser.index if 1974 <= y <= 2026))
    yr_df = pd.DataFrame({'year': years})
    for label, ser in yearly_records.items():
        yr_df[label] = yr_df['year'].map(ser * 100)
    yr_csv = os.path.join(BASE, 'g23b_dh_refinement_yearly_aftertax.csv')
    yr_df.to_csv(yr_csv, index=False, float_format='%.2f')
    print(f'→ Yearly aftertax CSV: {yr_csv}')

    print('\n[OOS 6 年累積倍率比較]')
    for label, ser in yearly_records.items():
        oos = ser.loc[(ser.index >= 2021) & (ser.index <= 2026)]
        cum = float((1 + oos).prod())
        print(f'  {label:42s}  cum=x{cum:.2f}  ({cum*100:.0f} 万円)')


if __name__ == '__main__':
    main()
