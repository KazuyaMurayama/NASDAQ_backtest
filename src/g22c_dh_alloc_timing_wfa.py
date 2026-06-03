"""G22C: DH-Z シリーズ WFA 50 窓検証
=================================================================
6 戦略 (REF + Z1〜Z5) で CI95_lo, WFE, t_stat, p_value を取得。

出力:
  - g22c_dh_alloc_timing_wfa.csv
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
from g18_daily_trade_cost_wfa import wfa_metrics
from g22a_dh_alloc_timing_variants import VARIANT_SPECS, build_variant


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G22C: DH-Z シリーズ WFA 50 窓検証 (REF + Z1〜Z5)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    windows = generate_windows(dates)
    print(f'\n[Windows: {len(windows)}]')

    rows = []

    # REF
    print('\n[1/6] DH-REF (always_in + dh_base)...')
    nav_ref, _, mask_ref, wn_ref, lev_ref = build_variant(a, 'always_in', 'dh_base')
    # WFA に渡す lev: lev_raw * 3 (DH 既存標準)
    L_eff_ref = lev_ref * 3.0
    wfa = wfa_metrics(nav_ref, dates, windows, lev_arr=L_eff_ref,
                       wn_arr=wn_ref, wb_arr=np.asarray(a['wb_A']))
    rows.append(dict(
        Strategy='DH Dyn 2x3x [A] (REF)',
        WFA_CI95_lo_pct=wfa.get('WFA_CI95_lo', np.nan)*100,
        WFA_CI95_hi_pct=wfa.get('WFA_CI95_hi', np.nan)*100,
        WFA_WFE=wfa.get('WFA_WFE', np.nan),
        t_stat=wfa.get('t_stat', np.nan),
        p_value=wfa.get('t_pvalue', np.nan),
        Trades_yr=wfa.get('mean_Trades_yr', np.nan),
    ))
    print(f'  CI95_lo={wfa.get("WFA_CI95_lo")*100:+.3f}%, WFE={wfa.get("WFA_WFE"):.3f}, '
          f'p={wfa.get("t_pvalue"):.4f}, Trades/yr={wfa.get("mean_Trades_yr"):.1f}')

    for i, (label, (tkey, akey)) in enumerate(VARIANT_SPECS.items(), start=2):
        print(f'\n[{i}/6] {label}...')
        nav, _, mask, wn_m, lev_m = build_variant(a, tkey, akey)
        L_eff = lev_m * 3.0
        wb_m = np.asarray(a['wb_A']) * mask  # wb for WFA trade counting
        # Z2: use wn_f10 for wn_arr (WFA trade detection uses wn changes)
        if akey == 'f10':
            wn_for_wfa = np.asarray(a['wn_f10']) * mask
            wb_for_wfa = np.asarray(a['wb_f10']) * mask
        elif akey == 'dh_base':
            wn_for_wfa = wn_m
            wb_for_wfa = np.asarray(a['wb_A']) * mask
        elif akey == 'fixed_bull':
            wn_for_wfa = wn_m
            wb_for_wfa = np.full(len(a['close']), 0.10) * mask
        elif akey == 'regime':
            wn_for_wfa = wn_m
            a2_arr = np.asarray(a['raw_a2'].values if hasattr(a['raw_a2'], 'values') else a['raw_a2'])
            wb_for_wfa = np.where(a2_arr > 0.5, 0.10, 0.25) * mask
        wfa = wfa_metrics(nav, dates, windows, lev_arr=L_eff,
                           wn_arr=wn_for_wfa, wb_arr=wb_for_wfa)
        rows.append(dict(
            Strategy=label,
            WFA_CI95_lo_pct=wfa.get('WFA_CI95_lo', np.nan)*100,
            WFA_CI95_hi_pct=wfa.get('WFA_CI95_hi', np.nan)*100,
            WFA_WFE=wfa.get('WFA_WFE', np.nan),
            t_stat=wfa.get('t_stat', np.nan),
            p_value=wfa.get('t_pvalue', np.nan),
            Trades_yr=wfa.get('mean_Trades_yr', np.nan),
        ))
        print(f'  CI95_lo={wfa.get("WFA_CI95_lo")*100:+.3f}%, WFE={wfa.get("WFA_WFE"):.3f}, '
              f'p={wfa.get("t_pvalue"):.4f}, Trades/yr={wfa.get("mean_Trades_yr"):.1f}')

    df = pd.DataFrame(rows)
    csv = os.path.join(BASE, 'g22c_dh_alloc_timing_wfa.csv')
    df.to_csv(csv, index=False, float_format='%.6f')
    print(f'\n→ WFA CSV: {csv}')

    print('\n[PASS 判定]')
    for _, r in df.iterrows():
        ci = r['WFA_CI95_lo_pct']; wfe = r['WFA_WFE']; p = r['p_value']
        ci_ok = ci > 0
        wfe_ok = (0.5 <= wfe <= 2.0)
        p_ok = p < 0.05
        verdict = '✅PASS' if (ci_ok and wfe_ok and p_ok) else (
                  '⚠MARGINAL' if (ci_ok or (wfe_ok and p_ok)) else '❌FAIL')
        print(f'  {r["Strategy"][:42]:42s}  CI95_lo={ci:+6.3f}% WFE={wfe:.3f} p={p:.4f} '
              f'Trades={r["Trades_yr"]:5.1f}  {verdict}')


if __name__ == '__main__':
    main()
