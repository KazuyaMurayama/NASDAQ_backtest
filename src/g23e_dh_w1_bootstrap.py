"""G23E: Bootstrap on OOS — DH-W1 (Asymm+Hysteresis) vs DH-REF
方法: paired block bootstrap (block=21d, n=10000, seed=42)
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
from g22a_dh_alloc_timing_variants import build_variant as build_z_variant
from g23a_dh_refinement_variants import build_W1
from corrected_strategy_backtest import TRADING_DAYS

N_BOOTSTRAP = 10000
BLOCK_SIZE = 21
RNG_SEED = 42


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G23E: OOS Bootstrap — DH-W1 (Asymm+Hysteresis) vs DH-REF')
    print('=' * 80)
    a = load_shared_assets()
    dates = a['dates']

    nav_ref, _, _, _, _ = build_z_variant(a, 'always_in', 'dh_base')
    nav_w1,  _, _, _, _ = build_W1(a)

    oos_mask = (dates >= OOS_START_TS) & (dates <= OOS_END_TS)
    ret_ref = nav_ref.pct_change().fillna(0).values[oos_mask.values]
    ret_w1  = nav_w1.pct_change().fillna(0).values[oos_mask.values]
    n_oos = len(ret_ref)
    years_oos = n_oos / TRADING_DAYS
    print(f'OOS days={n_oos}, years={years_oos:.2f}')

    cum_ref = float(np.prod(1.0 + ret_ref))
    cum_w1  = float(np.prod(1.0 + ret_w1))
    cagr_ref = cum_ref**(1.0/years_oos) - 1.0
    cagr_w1  = cum_w1**(1.0/years_oos) - 1.0
    actual_diff_pp = (cagr_w1 - cagr_ref) * 100
    print(f'\n  実 CAGR_OOS W1 (raw daily): {cagr_w1*100:+.2f}%')
    print(f'  実 CAGR_OOS REF: {cagr_ref*100:+.2f}%')
    print(f'  実 diff: {actual_diff_pp:+.2f}pp')

    rng = np.random.default_rng(RNG_SEED)
    n_blocks = int(np.ceil(n_oos / BLOCK_SIZE))
    paired_diff = []
    print(f'\n[Bootstrap {N_BOOTSTRAP}...]')
    for i in range(N_BOOTSTRAP):
        block_starts = rng.integers(0, n_oos - BLOCK_SIZE + 1, size=n_blocks)
        s_ref = np.concatenate([ret_ref[s:s+BLOCK_SIZE] for s in block_starts])[:n_oos]
        s_w1  = np.concatenate([ret_w1 [s:s+BLOCK_SIZE] for s in block_starts])[:n_oos]
        c_r = float(np.prod(1.0 + s_ref))
        c_w = float(np.prod(1.0 + s_w1))
        cagr_r = c_r**(1.0/years_oos) - 1.0 if c_r > 0 else -1.0
        cagr_w = c_w**(1.0/years_oos) - 1.0 if c_w > 0 else -1.0
        paired_diff.append((cagr_w - cagr_r) * 100)
        if (i+1) % 2000 == 0: print(f'  {i+1}/{N_BOOTSTRAP}')

    arr = np.array(paired_diff)
    ci_lo = float(np.percentile(arr, 2.5))
    ci_md = float(np.percentile(arr, 50))
    ci_hi = float(np.percentile(arr, 97.5))
    p_pos = float((arr > 0).mean()) * 100
    print(f'\n[結果]')
    print(f'  diff (W1-REF) median: {ci_md:+.2f}pp')
    print(f'  diff 95% CI: [{ci_lo:+.2f}, {ci_hi:+.2f}]pp')
    print(f'  P(diff > 0): {p_pos:.1f}%')

    out = pd.DataFrame({
        'metric': ['actual_diff_pp', 'paired_median_pp', 'paired_CI_low_pp',
                   'paired_CI_high_pp', 'P_diff_gt0_pct'],
        'value':  [actual_diff_pp, ci_md, ci_lo, ci_hi, p_pos],
    })
    csv = os.path.join(BASE, 'g23e_dh_w1_bootstrap_results.csv')
    out.to_csv(csv, index=False, float_format='%.6f')
    print(f'\n→ CSV: {csv}')

    print('\n[判定]')
    if ci_lo > 0: print(f'  ✅ CI95 下端 {ci_lo:+.2f}pp > 0 → H0 棄却、W1 が REF 統計的に上回る')
    elif ci_lo > -0.5: print(f'  ⚠ CI95 下端 {ci_lo:+.2f}pp ≈ 0 marginal')
    else: print(f'  ❌ CI95 下端 {ci_lo:+.2f}pp 偶然範囲')
    if p_pos > 95: print(f'  ✅ P(diff>0)={p_pos:.1f}% > 95%')
    elif p_pos > 90: print(f'  ⚠ P(diff>0)={p_pos:.1f}% borderline')
    else: print(f'  ❌ P(diff>0)={p_pos:.1f}% < 95%')


if __name__ == '__main__':
    main()
