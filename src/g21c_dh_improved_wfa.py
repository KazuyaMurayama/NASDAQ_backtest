"""G21C: DH 改善 4 変種 + 現行 DH の WFA 50 窓検証
=================================================================
g18 と同じ wfa_metrics で各変種を評価し、CI95_lo / WFE / mean_Trades_yr を CSV 出力。

出力:
  - g21c_dh_improved_wfa.csv
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets, BASE, generate_windows
from g18_daily_trade_cost_wfa import (
    build_dh_nav_with_cost, build_dh_nav_with_timing_cost, wfa_metrics,
)
from g21a_dh_improved_variants import VARIANT_SPECS, build_variant, DH_PER_UNIT


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G21C: DH 改善 4 変種 + REF の WFA 50 窓検証')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    windows = generate_windows(dates)
    print(f'\n[Windows: {len(windows)}]')

    rows = []

    # REF
    print('\n[1/5] DH-REF (現行)...')
    nav_ref, _ = build_dh_nav_with_cost(
        a['close'], a['lev_raw'], a['wn_A'], a['wg_A'], a['wb_A'],
        dates, a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    L_eff_ref = np.asarray(a['lev_raw']) * 3.0
    wfa = wfa_metrics(nav_ref, dates, windows, lev_arr=L_eff_ref,
                       wn_arr=a['wn_A'], wb_arr=a['wb_A'])
    rows.append(dict(
        Strategy='DH Dyn 2x3x [A] (現行 REF)',
        WFA_CI95_lo=wfa.get('WFA_CI95_lo', np.nan)*100,
        WFA_CI95_hi=wfa.get('WFA_CI95_hi', np.nan)*100,
        WFA_WFE=wfa.get('WFA_WFE', np.nan),
        t_stat=wfa.get('t_stat', np.nan),
        p_value=wfa.get('t_pvalue', np.nan),
        Trades_yr=wfa.get('mean_Trades_yr', np.nan),
    ))
    print(f'  CI95_lo={wfa.get("WFA_CI95_lo")*100:+.3f}%, WFE={wfa.get("WFA_WFE"):.3f}, '
          f'p={wfa.get("t_pvalue"):.4f}, Trades/yr={wfa.get("mean_Trades_yr"):.1f}')

    # 4 variants
    L_s2_55 = np.asarray(a['L_s2_lmax5p5'])
    for i, (label, spec) in enumerate(VARIANT_SPECS.items(), start=2):
        print(f'\n[{i}/5] {label}...')
        nav_v, _ = build_variant(a, spec)
        # WFA に渡すレバ配列: lmax cap ありなら min(lev_raw*3, L_s2_55)、なしなら lev_raw*3 * lev_mod_065
        L_eff = np.asarray(a['lev_raw']) * 3.0
        if spec['use_lmax']:
            L_eff = np.minimum(L_eff, L_s2_55)
        if spec['use_levmod']:
            lm = np.asarray(a['lev_mod_065'])
            L_eff = L_eff * np.nan_to_num(lm, nan=1.0)
        wn = a['wn_f10'] if spec['use_f10'] else a['wn_A']
        wb = a['wb_f10'] if spec['use_f10'] else a['wb_A']
        wfa = wfa_metrics(nav_v, dates, windows, lev_arr=L_eff, wn_arr=wn, wb_arr=wb)
        rows.append(dict(
            Strategy=label,
            WFA_CI95_lo=wfa.get('WFA_CI95_lo', np.nan)*100,
            WFA_CI95_hi=wfa.get('WFA_CI95_hi', np.nan)*100,
            WFA_WFE=wfa.get('WFA_WFE', np.nan),
            t_stat=wfa.get('t_stat', np.nan),
            p_value=wfa.get('t_pvalue', np.nan),
            Trades_yr=wfa.get('mean_Trades_yr', np.nan),
        ))
        print(f'  CI95_lo={wfa.get("WFA_CI95_lo")*100:+.3f}%, WFE={wfa.get("WFA_WFE"):.3f}, '
              f'p={wfa.get("t_pvalue"):.4f}, Trades/yr={wfa.get("mean_Trades_yr"):.1f}')

    df = pd.DataFrame(rows)
    csv = os.path.join(BASE, 'g21c_dh_improved_wfa.csv')
    df.to_csv(csv, index=False, float_format='%.6f')
    print(f'\n→ WFA CSV: {csv}')

    print('\n[PASS 判定]')
    for _, r in df.iterrows():
        ci = r['WFA_CI95_lo']; wfe = r['WFA_WFE']; p = r['p_value']
        ci_ok = ci > 0
        wfe_ok = (0.5 <= wfe <= 2.0)
        p_ok = p < 0.05
        verdict = '✅PASS' if (ci_ok and wfe_ok and p_ok) else (
                  '⚠MARGINAL' if (ci_ok or (wfe_ok and p_ok)) else '❌FAIL')
        print(f'  {r["Strategy"]:35s}  CI95_lo={ci:+.3f}% WFE={wfe:.3f} p={p:.4f}  {verdict}')


if __name__ == '__main__':
    main()
