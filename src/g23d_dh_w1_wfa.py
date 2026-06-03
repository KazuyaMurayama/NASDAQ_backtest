"""G23D: DH-W1 (Asymm+Hysteresis, DH base) WFA 50 窓検証
=================================================================
DH-REF + DH-W1 の WFA 50 窓で CI95_lo, WFE, p_value, Trades/yr を取得
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
from g22a_dh_alloc_timing_variants import build_variant as build_z_variant
from g23a_dh_refinement_variants import build_W1, hold_mask_W1


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G23D: DH-W1 WFA 50 窓検証 (REF + W1)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    windows = generate_windows(dates)
    print(f'\n[Windows: {len(windows)}]')

    rows = []

    # REF
    print('\n[1/2] DH-REF...')
    nav_ref, _, _, wn_ref, lev_ref = build_z_variant(a, 'always_in', 'dh_base')
    L_eff = lev_ref * 3.0
    wfa = wfa_metrics(nav_ref, dates, windows, lev_arr=L_eff,
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

    # W1
    print('\n[2/2] DH-W1 (Asymm+Hysteresis, DH base)...')
    nav_w1, _, mask_w1, wn_w1, lev_w1 = build_W1(a)
    L_eff_w1 = lev_w1 * 3.0
    wb_w1 = np.asarray(a['wb_A']) * mask_w1
    wfa_w1 = wfa_metrics(nav_w1, dates, windows, lev_arr=L_eff_w1,
                          wn_arr=wn_w1, wb_arr=wb_w1)
    rows.append(dict(
        Strategy='DH-W1 (Asymm+Hyst, DH base)',
        WFA_CI95_lo_pct=wfa_w1.get('WFA_CI95_lo', np.nan)*100,
        WFA_CI95_hi_pct=wfa_w1.get('WFA_CI95_hi', np.nan)*100,
        WFA_WFE=wfa_w1.get('WFA_WFE', np.nan),
        t_stat=wfa_w1.get('t_stat', np.nan),
        p_value=wfa_w1.get('t_pvalue', np.nan),
        Trades_yr=wfa_w1.get('mean_Trades_yr', np.nan),
    ))
    print(f'  CI95_lo={wfa_w1.get("WFA_CI95_lo")*100:+.3f}%, WFE={wfa_w1.get("WFA_WFE"):.3f}, '
          f'p={wfa_w1.get("t_pvalue"):.4f}, Trades/yr={wfa_w1.get("mean_Trades_yr"):.1f}')

    df = pd.DataFrame(rows)
    csv = os.path.join(BASE, 'g23d_dh_w1_wfa.csv')
    df.to_csv(csv, index=False, float_format='%.6f')
    print(f'\n→ WFA CSV: {csv}')

    print('\n[PASS 判定]')
    for _, r in df.iterrows():
        ci = r['WFA_CI95_lo_pct']; wfe = r['WFA_WFE']; p = r['p_value']
        ci_ok = ci > 0; wfe_ok = (0.5 <= wfe <= 2.0); p_ok = p < 0.05
        verdict = '✅PASS' if (ci_ok and wfe_ok and p_ok) else (
                  '⚠MARGINAL' if (ci_ok or (wfe_ok and p_ok)) else '❌FAIL')
        print(f'  {r["Strategy"]:35s}  CI95_lo={ci:+6.3f}% WFE={wfe:.3f} p={p:.4f} {verdict}')


if __name__ == '__main__':
    main()
