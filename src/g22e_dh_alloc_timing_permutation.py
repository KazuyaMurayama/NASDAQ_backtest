"""G22E: Permutation test — Z2 (F10 tilt + binary vz, θ=0.5) の binary 閾値 θ 偶然性検定
=================================================================
方法: θ ∈ [0.1, 0.9] uniform を 100 サンプリングし、各 θ で
       hold_mask = (lev_mod_065 ≥ θ) を生成、F10 tilt 配分で NAV 構築、
       IS-OOS gap を計算。実 θ=0.5 の gap が NULL 分布のどこに位置するか。

判定:
  - 実 gap が NULL 下位 5% 以内 → 統計的有意
  - 5〜15% → marginal
  - > 15% → 偶然の範囲

出力:
  - g22e_dh_alloc_timing_permutation_summary.csv
  - g22e_dh_alloc_timing_permutation_detail.csv
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
from g22a_dh_alloc_timing_variants import DH_PER_UNIT

N_PERMS = 100
THETA_LOW = 0.10
THETA_HIGH = 0.90
RNG_SEED = 42


def compute_gap_for_theta(a, dates, theta):
    """θ を使って binary mask 生成 + F10 配分で NAV 構築 + IS-OOS gap 算出"""
    lm = np.nan_to_num(np.asarray(a['lev_mod_065']), nan=0.0)
    mask = (lm >= theta).astype(float)
    wn = np.asarray(a['wn_f10']) * mask
    wg = np.asarray(a['wg_f10']) * mask
    wb = np.asarray(a['wb_f10']) * mask
    lev = np.asarray(a['lev_raw']) * mask
    nav, _ = build_dh_nav_with_cost(
        a['close'], lev, wn, wg, wb,
        dates, a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    m = metrics_from_nav(nav, dates, a['ret'])
    yr_aft = m['yearly'].apply(apply_tax_etf_decimal)
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return np.nan
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is, cagr_oos = _geo(is_subset), _geo(oos_subset)
    return cagr_is - cagr_oos, cagr_oos


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G22E: Permutation test — Z2 binary 閾値 θ (lev_mod_065 ≥ θ)')
    print('=' * 80)
    print(f'  n_permutations = {N_PERMS}, θ ∈ [{THETA_LOW}, {THETA_HIGH}], seed = {RNG_SEED}')

    a = load_shared_assets()
    dates = a['dates']

    actual_gap, actual_cagr_oos = compute_gap_for_theta(a, dates, 0.5)
    print(f'\n  実 θ=0.5 gap: {actual_gap*100:+.2f}pp, CAGR_OOS: {actual_cagr_oos*100:+.2f}%')

    rng = np.random.default_rng(RNG_SEED)
    null_gaps = []; null_cagrs = []
    print(f'\n[Permutation 実行中... ({N_PERMS} samples)]')
    for i in range(N_PERMS):
        theta_rand = rng.uniform(THETA_LOW, THETA_HIGH)
        g, c = compute_gap_for_theta(a, dates, theta_rand)
        null_gaps.append(g); null_cagrs.append(c)
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{N_PERMS} 完了')

    null_gaps = np.array(null_gaps); null_cagrs = np.array(null_cagrs)

    pct_le = float((null_gaps <= actual_gap).mean()) * 100
    pct_lt0 = float((null_gaps < 0).mean()) * 100
    g_lo = float(np.percentile(null_gaps, 5))
    g_hi = float(np.percentile(null_gaps, 95))
    g_mean = float(null_gaps.mean())
    pct_ge_cagr = float((null_cagrs >= actual_cagr_oos).mean()) * 100
    c_lo = float(np.percentile(null_cagrs, 5))
    c_hi = float(np.percentile(null_cagrs, 95))

    print(f'\n[結果]')
    print(f'  実 θ=0.5 gap: {actual_gap*100:+.2f}pp')
    print(f'  NULL gap mean: {g_mean*100:+.2f}pp')
    print(f'  NULL gap 5-95%: [{g_lo*100:+.2f}pp, {g_hi*100:+.2f}pp]')
    print(f'  P(NULL ≤ 実): {pct_le:.1f}%')
    print(f'  実 θ=0.5 CAGR_OOS: {actual_cagr_oos*100:+.2f}%')
    print(f'  NULL CAGR_OOS 5-95%: [{c_lo*100:+.2f}%, {c_hi*100:+.2f}%]')
    print(f'  P(NULL CAGR ≥ 実): {pct_ge_cagr:.1f}%')

    summary = pd.DataFrame({
        'metric': ['actual_gap_pp', 'actual_cagr_oos_pct',
                   'null_gap_mean_pp', 'null_gap_5pct_pp', 'null_gap_95pct_pp',
                   'P_gap_lt_zero_under_null_pct', 'P_gap_le_actual_under_null_pct',
                   'null_cagr_mean_pct', 'null_cagr_5pct', 'null_cagr_95pct',
                   'P_cagr_ge_actual_under_null_pct'],
        'value': [actual_gap*100, actual_cagr_oos*100,
                  g_mean*100, g_lo*100, g_hi*100,
                  pct_lt0, pct_le,
                  float(null_cagrs.mean())*100, c_lo*100, c_hi*100,
                  pct_ge_cagr],
    })
    csv_s = os.path.join(BASE, 'g22e_dh_alloc_timing_permutation_summary.csv')
    summary.to_csv(csv_s, index=False, float_format='%.6f')
    csv_d = os.path.join(BASE, 'g22e_dh_alloc_timing_permutation_detail.csv')
    pd.DataFrame({'idx': range(N_PERMS),
                  'null_gap_pp': null_gaps*100,
                  'null_cagr_oos_pct': null_cagrs*100}).to_csv(csv_d, index=False, float_format='%.6f')
    print(f'\n→ Summary: {csv_s}')
    print(f'→ Detail : {csv_d}')

    print('\n[判定]')
    if pct_le <= 5:
        print(f'  ✅ 実 gap NULL 下位 {pct_le:.1f}% (≤5%) → 統計的有意')
    elif pct_le <= 15:
        print(f'  ⚠ 実 gap NULL 下位 {pct_le:.1f}% → marginal')
    else:
        print(f'  ❌ 実 gap NULL 下位 {pct_le:.1f}% → 偶然の範囲')


if __name__ == '__main__':
    main()
