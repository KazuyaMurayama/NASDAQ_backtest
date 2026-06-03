"""G21E: Permutation test — vz_thr のランダム化で DH-T4 の gap が偶然か検定
=================================================================
方法: vz_thr を uniform[0.40, 1.00] から 100 サンプリングし、各 sample で
       DH-T4 構成 (vz_gate + lmax5.5 + F10ε=0.015) を組み立て、IS-OOS gap を計算。
       実 vz=0.65 の gap が NULL 分布のどこに位置するかを評価。

帰無仮説 H0: vz_thr に最適性なし (任意の閾値で同等の gap)
判定:
  - 実 gap が NULL 下位 5% 以内 → 統計的に異常 (偶然でない)
  - 実 gap が NULL 下位 15% 以内 → marginal
  - 実 gap が NULL 下位 15% 超    → 偶然の範囲

出力:
  - g21e_dh_permutation_summary.csv
  - g21e_dh_permutation_detail.csv
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets, BASE, K_LO, K_HI, K_MID, THRESHOLD
from g18_daily_trade_cost_wfa import (
    build_dh_nav_with_timing_cost, metrics_from_nav, apply_tax_etf_decimal,
)
from g19a_f10_eps_extended import build_f10_wn_for_eps
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b
from g21a_dh_improved_variants import DH_PER_UNIT

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
EPS_F10 = 0.015
LMAX = 5.5
N_PERMS = 100
VZ_LOW = 0.40
VZ_HIGH = 1.00
RNG_SEED = 42


def build_lev_mod_for_vz(a, vz_thr):
    vz_arr = a['vz_arr']; lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > vz_thr, K_HI,
            np.where(vz_arr < -vz_thr, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    return apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)


def compute_gap_for_vz(a, dates, vz_thr):
    lev_mod_obj = build_lev_mod_for_vz(a, vz_thr)
    lev_mod = lev_mod_obj.values if hasattr(lev_mod_obj, 'values') else np.asarray(lev_mod_obj)
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': LMAX})
    raw_a2_vals = a['raw_a2'].values; vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
        np.asarray(a['lev_raw']), bull_mask, EPS_F10,
    )
    nav_adj, _ = build_dh_nav_with_timing_cost(
        a['close'], a['lev_raw'], wn_f10, a['wg_f10'], wb_f10,
        dates, a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
        lev_mod=lev_mod, L_s2_values=L_s2.values,
    )
    m = metrics_from_nav(nav_adj, dates, a['ret'])
    yr_aft = m['yearly'].apply(apply_tax_etf_decimal)
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x)==0: return np.nan
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c>0 else -1.0
    cagr_is, cagr_oos = _geo(is_subset), _geo(oos_subset)
    return cagr_is - cagr_oos, cagr_oos


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G21E: Permutation test (vz_thr random sampling, DH-T4 base)')
    print('=' * 80)
    print(f'  n_permutations = {N_PERMS}, vz_thr ∈ [{VZ_LOW}, {VZ_HIGH}], seed = {RNG_SEED}')

    a = load_shared_assets()
    dates = a['dates']

    actual_gap, actual_cagr_oos = compute_gap_for_vz(a, dates, 0.65)
    print(f'\n  実 vz=0.65 gap: {actual_gap*100:+.2f}pp, CAGR_OOS: {actual_cagr_oos*100:+.2f}%')

    rng = np.random.default_rng(RNG_SEED)
    null_gaps = []
    null_cagrs = []
    print(f'\n[Permutation 実行中... ({N_PERMS} samples)]')
    for i in range(N_PERMS):
        vz_thr_rand = rng.uniform(VZ_LOW, VZ_HIGH)
        gap, cagr = compute_gap_for_vz(a, dates, vz_thr_rand)
        null_gaps.append(gap)
        null_cagrs.append(cagr)
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{N_PERMS} 完了')

    null_gaps = np.array(null_gaps)
    null_cagrs = np.array(null_cagrs)

    pct_le_actual = float((null_gaps <= actual_gap).mean()) * 100
    pct_lt_zero = float((null_gaps < 0).mean()) * 100
    null_gap_low  = float(np.percentile(null_gaps, 5))
    null_gap_high = float(np.percentile(null_gaps, 95))
    null_gap_mean = float(null_gaps.mean())

    pct_ge_actual_cagr = float((null_cagrs >= actual_cagr_oos).mean()) * 100
    null_cagr_low  = float(np.percentile(null_cagrs, 5))
    null_cagr_high = float(np.percentile(null_cagrs, 95))

    print(f'\n[結果]')
    print(f'  実 vz=0.65 gap: {actual_gap*100:+.2f}pp')
    print(f'  NULL gap mean: {null_gap_mean*100:+.2f}pp')
    print(f'  NULL gap 5-95%: [{null_gap_low*100:+.2f}pp, {null_gap_high*100:+.2f}pp]')
    print(f'  P(NULL gap ≤ 実 vz=0.65) : {pct_le_actual:.1f}%')
    print(f'  実 vz=0.65 CAGR_OOS: {actual_cagr_oos*100:+.2f}%')
    print(f'  NULL CAGR_OOS 5-95%: [{null_cagr_low*100:+.2f}%, {null_cagr_high*100:+.2f}%]')
    print(f'  P(NULL CAGR ≥ 実): {pct_ge_actual_cagr:.1f}%')

    out_summary = pd.DataFrame({
        'metric': ['actual_gap_pp', 'actual_cagr_oos_pct',
                   'null_gap_mean_pp', 'null_gap_5pct_pp', 'null_gap_95pct_pp',
                   'P_gap_lt_zero_under_null_pct', 'P_gap_le_actual_under_null_pct',
                   'null_cagr_mean_pct', 'null_cagr_5pct', 'null_cagr_95pct',
                   'P_cagr_ge_actual_under_null_pct'],
        'value': [actual_gap*100, actual_cagr_oos*100,
                  null_gap_mean*100, null_gap_low*100, null_gap_high*100,
                  pct_lt_zero, pct_le_actual,
                  float(null_cagrs.mean())*100, null_cagr_low*100, null_cagr_high*100,
                  pct_ge_actual_cagr],
    })
    csv_s = os.path.join(BASE, 'g21e_dh_permutation_summary.csv')
    out_summary.to_csv(csv_s, index=False, float_format='%.6f')
    csv_d = os.path.join(BASE, 'g21e_dh_permutation_detail.csv')
    pd.DataFrame({'idx': range(N_PERMS),
                  'null_gap_pp': null_gaps*100,
                  'null_cagr_oos_pct': null_cagrs*100}).to_csv(csv_d, index=False, float_format='%.6f')
    print(f'\n→ Summary CSV: {csv_s}')
    print(f'→ Detail CSV : {csv_d}')

    print('\n[判定]')
    if pct_le_actual <= 5:
        print(f'  ✅ 実 gap は NULL 下位 {pct_le_actual:.1f}% (≤5%) → 統計的に異常、偶然でない')
    elif pct_le_actual <= 15:
        print(f'  ⚠ 実 gap は NULL 下位 {pct_le_actual:.1f}% → marginal')
    else:
        print(f'  ❌ 実 gap は NULL 下位 {pct_le_actual:.1f}% → 偶然の範囲')


if __name__ == '__main__':
    main()
