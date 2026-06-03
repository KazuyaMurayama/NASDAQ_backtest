"""G24: 5 戦略全指標の再検証
=================================================================
親レポート (STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md v4.x) の
§0' 5 戦略の各 9 指標 + 累積 CAGR を canonical 計算で再算出し、
レポート掲載値との差分を CSV で出力。

5 戦略:
  1. NEW 🟢 (vz=0.65+l7+F10ε)  ← CFD, g20f/g20b 由来
  2. D5 vz=0.65/lmax=5.5         ← CFD, g20f 由来
  3. DH-W1 (Asymm+Hyst, DH base) ← ETF, g23a 由来 (新規)
  4. DH Dyn 2x3x [A] (REF)        ← ETF, g18 由来
  5. NDX 1x B&H                   ← Benchmark, g19e 由来

出力:
  - g24_5strategies_verification.csv (報告書値 vs 再計算値 + diff)
  - g24_5strategies_yearly.csv (5 戦略 1974-2026 年次税後 %)
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
    build_dh_nav_with_cost, build_cfd_nav_with_cost, wfa_metrics,
    metrics_from_nav, apply_tax_etf_decimal, apply_tax_cfd_decimal,
)
from g19e_3strategies_daily_cost import compute_bnh_metrics_after_tax
from g19a_f10_eps_extended import build_f10_wn_for_eps
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b
from g22a_dh_alloc_timing_variants import build_variant as build_z_variant
from g23a_dh_refinement_variants import build_W1

# Constants used for CFD NEW / D5 builders
from g14_wfa_sbi_cfd import K_LO, K_HI, K_MID, THRESHOLD
S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
SPREAD_RT = 0.00050  # moderate CFD spread


def build_lev_mod_for_vz(a, vz_thr):
    vz_arr = a['vz_arr']
    lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > vz_thr, K_HI,
            np.where(vz_arr < -vz_thr, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    return apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)


def build_new_candidate_nav(a):
    """NEW: vz=0.65+lmax=7+F10ε=0.015 (CFD)"""
    dates = a['dates']; close = a['close']
    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': 7.0})
    lev_mod = build_lev_mod_for_vz(a, 0.65)
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
        np.asarray(a['lev_raw']), bull_mask, 0.015,
    )
    wg = a['wg_A']
    nav_adj, _ = build_cfd_nav_with_cost(
        close, lev_mod, wn_f10, wg, wb_f10, dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
        SPREAD_RT / 2.0,
    )
    return nav_adj, lev_mod, wn_f10, wg, wb_f10, L_s2.values


def build_d5_nav(a):
    """D5: vz=0.65 + lmax=5.5 (CFD, no F10 tilt, DH base weights wn_A/wb_A)"""
    dates = a['dates']; close = a['close']
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': 5.5})
    lev_mod = build_lev_mod_for_vz(a, 0.65)
    nav_adj, _ = build_cfd_nav_with_cost(
        close, lev_mod, a['wn_A'], a['wg_A'], a['wb_A'], dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
        SPREAD_RT / 2.0,
    )
    return nav_adj, lev_mod, a['wn_A'], a['wg_A'], a['wb_A'], L_s2.values


def calc_full_metrics(label, nav, dates, ret_nas, is_cfd, lev_arr_for_wfa, wn_arr, wb_arr,
                       windows):
    """9 indicators + 累積 CAGR + WFA"""
    m = metrics_from_nav(nav, dates, ret_nas)
    yr_pre = m['yearly']
    tax_fn = apply_tax_cfd_decimal if is_cfd else apply_tax_etf_decimal
    yr_aft = yr_pre.apply(tax_fn)
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return float('nan')
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is, cagr_oos = _geo(is_subset), _geo(oos_subset)
    # WFA
    wfa = wfa_metrics(nav, dates, windows, lev_arr=lev_arr_for_wfa,
                       wn_arr=wn_arr, wb_arr=wb_arr)
    return dict(
        Strategy=label,
        CAGR_OOS_pct=cagr_oos*100,
        IS_OOS_gap_pp=(cagr_is - cagr_oos)*100,
        cum_CAGR_IS_pct=cagr_is*100,
        cum_CAGR_OOS_pct=cagr_oos*100,
        Sharpe_OOS=m['Sharpe_OOS'],
        MaxDD_FULL_pct=m['MaxDD_FULL']*100,
        Worst10Y_CAGR_pct=m['Worst10Y_star']*100,
        P10_5Y_CAGR_pct=m['P10_5Y']*100,
        Trades_yr=wfa.get('mean_Trades_yr', np.nan),
        WFA_WFE=wfa.get('WFA_WFE', np.nan),
        WFA_CI95_lo_pct=wfa.get('WFA_CI95_lo', np.nan)*100,
        WFA_p_value=wfa.get('t_pvalue', np.nan),
        yearly_aft=yr_aft,
    )


# 報告書 (v4.2 §0') 掲載値
REPORTED = {
    'NEW (vz=0.65+l7+F10ε)': dict(
        CAGR_OOS_pct=21.49, IS_OOS_gap_pp=-1.27,
        Sharpe_OOS=0.829, MaxDD_FULL_pct=-65.95,
        Worst10Y_CAGR_pct=9.96, P10_5Y_CAGR_pct=5.84,
        Trades_yr=52, WFA_WFE=1.369, WFA_CI95_lo_pct=19.9,
    ),
    'D5 vz=0.65/lmax=5.5': dict(
        CAGR_OOS_pct=17.86, IS_OOS_gap_pp=2.22,
        Sharpe_OOS=0.79, MaxDD_FULL_pct=-55.88,
        Worst10Y_CAGR_pct=12.21, P10_5Y_CAGR_pct=6.76,
        Trades_yr=28, WFA_WFE=1.30, WFA_CI95_lo_pct=19.2,
    ),
    'DH Dyn 2x3x [A] (REF)': dict(
        CAGR_OOS_pct=9.56, IS_OOS_gap_pp=10.29,
        Sharpe_OOS=0.60, MaxDD_FULL_pct=-41.57,
        Worst10Y_CAGR_pct=12.57, P10_5Y_CAGR_pct=8.77,
        Trades_yr=27, WFA_WFE=0.66, WFA_CI95_lo_pct=17.5,
    ),
    'NDX 1x B&H': dict(
        CAGR_OOS_pct=8.27, IS_OOS_gap_pp=1.64,
        Sharpe_OOS=0.516, MaxDD_FULL_pct=-77.93,
        Worst10Y_CAGR_pct=-4.85, P10_5Y_CAGR_pct=0.59,
        Trades_yr=0, WFA_WFE=None, WFA_CI95_lo_pct=None,
    ),
}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G24: 5 戦略 全指標 再検証 (NEW / D5 / DH-W1 / DH-REF / B&H)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    ret_nas = a['ret']
    windows = generate_windows(dates)
    print(f'\n[Windows: {len(windows)}]')

    rows = []
    yearly_records = {}

    # 1. NEW CANDIDATE (CFD)
    print('\n[1/5] NEW 🟢 (vz=0.65+l7+F10ε)...')
    nav, lev_mod, wn, wg, wb, L_s2 = build_new_candidate_nav(a)
    m = calc_full_metrics('NEW (vz=0.65+l7+F10ε)', nav, dates, ret_nas,
                          is_cfd=True, lev_arr_for_wfa=L_s2,
                          wn_arr=wn, wb_arr=wb, windows=windows)
    rows.append(m); yearly_records['NEW 🟢'] = m['yearly_aft']

    # 2. D5
    print('\n[2/5] D5 vz=0.65/lmax=5.5...')
    nav, lev_mod, wn, wg, wb, L_s2 = build_d5_nav(a)
    m = calc_full_metrics('D5 vz=0.65/lmax=5.5', nav, dates, ret_nas,
                          is_cfd=True, lev_arr_for_wfa=L_s2,
                          wn_arr=wn, wb_arr=wb, windows=windows)
    rows.append(m); yearly_records['D5'] = m['yearly_aft']

    # 3. DH-W1 (NEW)
    print('\n[3/5] DH-W1 (Asymm+Hyst, DH base)...')
    nav, _, mask, wn_w1, lev_w1 = build_W1(a)
    L_eff = lev_w1 * 3.0
    wb_w1 = np.asarray(a['wb_A']) * mask
    m = calc_full_metrics('DH-W1 (Asymm+Hyst)', nav, dates, ret_nas,
                          is_cfd=False, lev_arr_for_wfa=L_eff,
                          wn_arr=wn_w1, wb_arr=wb_w1, windows=windows)
    rows.append(m); yearly_records['DH-W1'] = m['yearly_aft']

    # 4. DH-REF
    print('\n[4/5] DH Dyn 2x3x [A] (REF)...')
    nav, _, _, wn_ref, lev_ref = build_z_variant(a, 'always_in', 'dh_base')
    L_eff = lev_ref * 3.0
    m = calc_full_metrics('DH Dyn 2x3x [A] (REF)', nav, dates, ret_nas,
                          is_cfd=False, lev_arr_for_wfa=L_eff,
                          wn_arr=wn_ref, wb_arr=np.asarray(a['wb_A']),
                          windows=windows)
    rows.append(m); yearly_records['DH [A]'] = m['yearly_aft']

    # 5. B&H
    print('\n[5/5] NDX 1x B&H...')
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
        Strategy='NDX 1x B&H', CAGR_OOS_pct=cagr_oos_b*100,
        IS_OOS_gap_pp=(cagr_is_b - cagr_oos_b)*100,
        cum_CAGR_IS_pct=cagr_is_b*100, cum_CAGR_OOS_pct=cagr_oos_b*100,
        Sharpe_OOS=bnh.get('Sharpe_OOS', 0.516),
        MaxDD_FULL_pct=bnh.get('MaxDD_FULL', -0.7793)*100,
        Worst10Y_CAGR_pct=bnh.get('Worst10Y_star', -0.0485)*100,
        P10_5Y_CAGR_pct=bnh.get('P10_5Y', 0.0059)*100,
        Trades_yr=0, WFA_WFE=np.nan, WFA_CI95_lo_pct=np.nan, WFA_p_value=np.nan,
        yearly_aft=bnh_yr,
    ))
    yearly_records['B&H'] = bnh_yr

    # Output verification CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'yearly_aft'} for r in rows])
    csv = os.path.join(BASE, 'g24_5strategies_verification.csv')
    df.to_csv(csv, index=False, float_format='%.4f')
    print(f'\n→ Verification CSV: {csv}')

    # Output yearly CSV
    years = sorted(set(y for ser in yearly_records.values() for y in ser.index if 1974 <= y <= 2026))
    yr_df = pd.DataFrame({'year': years})
    for label, ser in yearly_records.items():
        yr_df[label] = yr_df['year'].map(ser * 100)
    yr_csv = os.path.join(BASE, 'g24_5strategies_yearly.csv')
    yr_df.to_csv(yr_csv, index=False, float_format='%.2f')
    print(f'→ Yearly CSV: {yr_csv}')

    # 報告書値との diff 表示
    print('\n' + '=' * 80)
    print('REPORTED vs RECOMPUTED 差分 (報告書 sec0prime 値との比較)')
    print('=' * 80)
    label_map = {
        'NEW (vz=0.65+l7+F10ε)': 'NEW (vz=0.65+l7+F10ε)',
        'D5 vz=0.65/lmax=5.5': 'D5 vz=0.65/lmax=5.5',
        'DH Dyn 2x3x [A] (REF)': 'DH Dyn 2x3x [A] (REF)',
        'NDX 1x B&H': 'NDX 1x B&H',
    }
    for r in rows:
        label = r['Strategy']
        rep_key = label_map.get(label)
        if rep_key is None:
            print(f'\n[{label}] — 新規 (報告書 v4.2 にはまだ無し):')
            print(f'  CAGR_OOS={r["CAGR_OOS_pct"]:+.2f}%, gap={r["IS_OOS_gap_pp"]:+.2f}pp, '
                  f'IS_CAGR={r["cum_CAGR_IS_pct"]:+.2f}%, OOS_CAGR(cum)={r["cum_CAGR_OOS_pct"]:+.2f}%')
            print(f'  Sharpe={r["Sharpe_OOS"]:+.3f}, MaxDD={r["MaxDD_FULL_pct"]:+.2f}%, '
                  f'W10Y={r["Worst10Y_CAGR_pct"]:+.2f}%, P10_5Y={r["P10_5Y_CAGR_pct"]:+.2f}%')
            print(f'  Trades/yr={r["Trades_yr"]:.1f}, WFE={r["WFA_WFE"]:.3f}, CI95={r["WFA_CI95_lo_pct"]:+.2f}%')
            continue
        rep = REPORTED[rep_key]
        print(f'\n[{label}] vs 報告書 v4.2 値:')
        for key, rep_v in rep.items():
            if rep_v is None: continue
            calc_v = r.get(key)
            if calc_v is None or (isinstance(calc_v, float) and np.isnan(calc_v)):
                continue
            diff = calc_v - rep_v
            flag = '✅一致' if abs(diff) < 0.05 else ('⚠近似' if abs(diff) < 0.3 else '❌乖離')
            print(f'  {key:22s}: 報告={rep_v:>+8.3f}  再計算={calc_v:>+8.3f}  diff={diff:>+7.3f}  {flag}')


if __name__ == '__main__':
    main()
