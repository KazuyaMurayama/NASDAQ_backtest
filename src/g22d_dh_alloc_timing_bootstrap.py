"""G22D: Bootstrap on OOS — DH-Z2 (F10 tilt + binary vz) vs DH-REF
=================================================================
方法: paired block bootstrap (block_size=21日, n=10000, seed=42)
帰無仮説 H0: CAGR_diff (Z2 - REF) = 0

出力:
  - g22d_dh_alloc_timing_bootstrap_results.csv
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
from g18_daily_trade_cost_wfa import OOS_START_TS, OOS_END_TS
from g22a_dh_alloc_timing_variants import build_variant
from corrected_strategy_backtest import TRADING_DAYS

N_BOOTSTRAP = 10000
BLOCK_SIZE = 21
RNG_SEED = 42


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G22D: OOS Bootstrap — DH-Z2 (F10 tilt + binary vz) vs DH-REF')
    print('=' * 80)
    print(f'  n_bootstrap = {N_BOOTSTRAP}, block_size = {BLOCK_SIZE}, seed = {RNG_SEED}')

    a = load_shared_assets()
    dates = a['dates']

    print('\n[NAV 構築]')
    nav_ref, _, _, _, _ = build_variant(a, 'always_in', 'dh_base')
    nav_z2,  _, _, _, _ = build_variant(a, 'vz_binary', 'f10')

    oos_mask = (dates >= OOS_START_TS) & (dates <= OOS_END_TS)
    ret_ref = nav_ref.pct_change().fillna(0).values[oos_mask.values]
    ret_z2  = nav_z2.pct_change().fillna(0).values[oos_mask.values]
    n_oos = len(ret_ref)
    years_oos = n_oos / TRADING_DAYS
    print(f'  OOS 日数: {n_oos}, OOS 年数: {years_oos:.2f}')

    cum_ref = float(np.prod(1.0 + ret_ref))
    cum_z2  = float(np.prod(1.0 + ret_z2))
    cagr_ref = cum_ref**(1.0/years_oos) - 1.0
    cagr_z2  = cum_z2**(1.0/years_oos) - 1.0
    actual_diff_pp = (cagr_z2 - cagr_ref) * 100
    print(f'\n  実 CAGR_OOS Z2 (raw daily): {cagr_z2*100:+.2f}%')
    print(f'  実 CAGR_OOS REF (raw daily): {cagr_ref*100:+.2f}%')
    print(f'  実 diff (Z2 - REF): {actual_diff_pp:+.2f}pp')

    print(f'\n[Bootstrap (paired block, {N_BOOTSTRAP} resamples)]')
    rng = np.random.default_rng(RNG_SEED)
    n_blocks = int(np.ceil(n_oos / BLOCK_SIZE))
    paired_diff = []
    for i in range(N_BOOTSTRAP):
        block_starts = rng.integers(0, n_oos - BLOCK_SIZE + 1, size=n_blocks)
        s_ref = np.concatenate([ret_ref[s:s+BLOCK_SIZE] for s in block_starts])[:n_oos]
        s_z2  = np.concatenate([ret_z2 [s:s+BLOCK_SIZE] for s in block_starts])[:n_oos]
        c_ref = float(np.prod(1.0 + s_ref))
        c_z2  = float(np.prod(1.0 + s_z2))
        cagr_r = c_ref**(1.0/years_oos) - 1.0 if c_ref > 0 else -1.0
        cagr_z = c_z2 **(1.0/years_oos) - 1.0 if c_z2  > 0 else -1.0
        paired_diff.append((cagr_z - cagr_r) * 100)
        if (i+1) % 2000 == 0:
            print(f'  {i+1}/{N_BOOTSTRAP} done')

    diff_arr = np.array(paired_diff)
    ci_lo  = float(np.percentile(diff_arr,  2.5))
    ci_md  = float(np.percentile(diff_arr, 50))
    ci_hi  = float(np.percentile(diff_arr, 97.5))
    p_pos  = float((diff_arr > 0).mean()) * 100

    print(f'\n[結果]')
    print(f'  diff (Z2 - REF) median: {ci_md:+.2f}pp')
    print(f'  diff 95% CI: [{ci_lo:+.2f}pp, {ci_hi:+.2f}pp]')
    print(f'  P(diff > 0): {p_pos:.1f}%')

    out = pd.DataFrame({
        'metric': ['actual_diff_pp', 'paired_median_pp', 'paired_CI_low_pp',
                   'paired_CI_high_pp', 'P_diff_gt0_pct'],
        'value':  [actual_diff_pp, ci_md, ci_lo, ci_hi, p_pos],
    })
    csv = os.path.join(BASE, 'g22d_dh_alloc_timing_bootstrap_results.csv')
    out.to_csv(csv, index=False, float_format='%.6f')
    print(f'\n→ CSV: {csv}')

    print('\n[判定]')
    if ci_lo > 0:
        print(f'  ✅ CI95 下端 {ci_lo:+.2f}pp > 0 → H0 棄却、Z2 は REF を統計的に上回る')
    elif ci_lo > -0.5:
        print(f'  ⚠ CI95 下端 {ci_lo:+.2f}pp ≈ 0 → marginal')
    else:
        print(f'  ❌ CI95 下端 {ci_lo:+.2f}pp < 0 → 偶然の範囲')
    if p_pos > 95:
        print(f'  ✅ P(diff>0)={p_pos:.1f}% > 95%')
    elif p_pos > 90:
        print(f'  ⚠ P(diff>0)={p_pos:.1f}% borderline')
    else:
        print(f'  ❌ P(diff>0)={p_pos:.1f}% < 95%')


if __name__ == '__main__':
    main()
