"""G21A: DH Dyn 2x3x [A] 改善 4 変種 NAV 生成 (1974-2026)
=================================================================
DH-T1: vz=0.65 Gate only      (lev_mod_065 のみ)
DH-T2: vz=0.65 + lmax=5.5 cap (lev_mod_065 + L_s2_lmax5p5)
DH-T3: vz=0.65 + F10 ε=0.015  (lev_mod_065 + wn_f10/wb_f10)
DH-T4: Full                   (lev_mod_065 + L_s2_lmax5p5 + wn_f10/wb_f10)

商品は現状と同一: TQQQ + TMF + WisdomTree 2036。
税モデル: ETF (apply_tax_etf_decimal, ×0.8273)。
取引コスト: per_unit_cost = 0.0010 (moderate)。

出力:
  - g21a_dh_improved_navs.csv  (5 列 × ~13169 日)
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
)

DH_PER_UNIT = 0.0010  # moderate

VARIANT_SPECS = {
    'DH-T1 (vz=0.65 Gate)':            dict(use_levmod=True,  use_lmax=False, use_f10=False),
    'DH-T2 (vz=0.65 + lmax=5.5)':       dict(use_levmod=True,  use_lmax=True,  use_f10=False),
    'DH-T3 (vz=0.65 + F10 e=0.015)':    dict(use_levmod=True,  use_lmax=False, use_f10=True),
    'DH-T4 (Full vz+lmax+F10e)':        dict(use_levmod=True,  use_lmax=True,  use_f10=True),
}


def build_variant(a, spec):
    lev_mod_arr = np.asarray(a['lev_mod_065']) if spec['use_levmod'] else None
    L_s2_vals = np.asarray(a['L_s2_lmax5p5']) if spec['use_lmax'] else None
    if spec['use_f10']:
        wn, wg, wb = a['wn_f10'], a['wg_f10'], a['wb_f10']
    else:
        wn, wg, wb = a['wn_A'], a['wg_A'], a['wb_A']
    nav_adj, yr_cost = build_dh_nav_with_timing_cost(
        a['close'], a['lev_raw'], wn, wg, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
        lev_mod=lev_mod_arr, L_s2_values=L_s2_vals,
    )
    return nav_adj, yr_cost


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G21A: DH Dyn 2x3x [A] 改善 4 変種 NAV 生成 (moderate cost = 0.10%)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']

    # Reference: 現行 DH Dyn 2x3x [A]
    nav_ref, cost_ref = build_dh_nav_with_cost(
        a['close'], a['lev_raw'], a['wn_A'], a['wg_A'], a['wb_A'],
        dates, a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    print(f'\n[REF] DH Dyn 2x3x [A] (現行)')
    print(f'  NAV final: {nav_ref.iloc[-1]:.2f}')
    print(f'  Avg yr cost (approx): {cost_ref*100:.3f}% / yr')

    navs = {'DH-REF (現行)': nav_ref}
    for label, spec in VARIANT_SPECS.items():
        nav_adj, yr_cost = build_variant(a, spec)
        navs[label] = nav_adj
        print(f'\n[{label}]')
        print(f'  NAV final: {nav_adj.iloc[-1]:.2f}')
        print(f'  Avg yr cost: {yr_cost*100:.3f}% / yr')

    out = pd.DataFrame(navs)  # auto-align by Series index (date)
    out.index.name = 'date'
    csv = os.path.join(BASE, 'g21a_dh_improved_navs.csv')
    out.to_csv(csv, float_format='%.6f')
    print(f'\n→ NAVs CSV: {csv}')
    print(f'  shape: {out.shape}')

    # Sanity: 全変種が REF と異なる NAV を返すこと
    print('\n[Sanity check: NAV final distinctness]')
    ref_final = float(nav_ref.iloc[-1])
    for col in out.columns:
        final = float(out[col].iloc[-1])
        diff_pct = (final / ref_final - 1.0) * 100
        flag = '✓' if col == 'DH-REF (現行)' else ('✓DISTINCT' if abs(diff_pct) > 0.1 else '⚠IDENTICAL')
        print(f'  {col:40s}  final={final:>12,.2f}  vs REF {diff_pct:+7.2f}%  {flag}')


if __name__ == '__main__':
    main()
