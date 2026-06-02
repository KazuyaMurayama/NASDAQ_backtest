"""
G20D: Bootstrap on OOS — NEW CANDIDATE vs F10 の +2.05pp 95%CI 検定
=================================================================
QC Agent 3 指摘:
  +2.05pp 改善の 95%CI が 0 を跨ぐか統計検定。

方法:
  OOS 日次差分リターン d_t = r_065(t) - r_070(t) を block bootstrap
  (block size = 21 営業日 ≈ 1 month) で 10000 サンプリングし、
  年率 CAGR 差分の 95%CI を構築。

帰無仮説 H0: CAGR_diff = 0 (差なし)
判定:
  - 95%CI 下端 > 0 → H0 棄却、NEW CANDIDATE 統計的に優位
  - 95%CI 下端 ≤ 0 → 偶然の可能性、棄却不可
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import (
    load_shared_assets, BASE,
    K_LO, K_HI, K_MID, THRESHOLD,
)
from g18_daily_trade_cost_wfa import (
    build_cfd_nav_with_cost,
    IS_END_TS, OOS_START_TS, OOS_END_TS,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b
from g19a_f10_eps_extended import build_f10_wn_for_eps
from corrected_strategy_backtest import TRADING_DAYS

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
LMAX = 7.0
EPS = 0.015
SPREAD_RT = 0.00050

N_BOOTSTRAP = 10000
BLOCK_SIZE = 21
RNG_SEED = 42  # deterministic


def build_lev_mod_for_vz(a, vz_thr):
    vz_arr = a['vz_arr']
    lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > vz_thr, K_HI,
            np.where(vz_arr < -vz_thr, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    return apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)


def build_nav_for_vz(a, dates, close, vz_thr):
    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': LMAX})
    lev_mod = build_lev_mod_for_vz(a, vz_thr)
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
        np.asarray(a['lev_raw']), bull_mask, EPS,
    )
    spread_ow = SPREAD_RT / 2.0
    nav_adj, _ = build_cfd_nav_with_cost(
        close, lev_mod, wn_f10, a['wg_A'], wb_f10, dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
        spread_ow,
    )
    return nav_adj


def block_bootstrap(returns_arr, block_size, n_resamples, rng):
    """日次リターン配列を block bootstrap。各 resample は同じ長さで NAV CAGR を返す。"""
    n = len(returns_arr)
    cagrs = []
    n_blocks = int(np.ceil(n / block_size))
    for _ in range(n_resamples):
        # blockの開始 indices を n - block_size の範囲からランダム選択
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled = np.concatenate([returns_arr[s:s+block_size] for s in block_starts])[:n]
        # cumulative product
        cum = float(np.prod(1.0 + sampled))
        years = n / TRADING_DAYS
        if cum > 0 and years > 0:
            cagr = cum**(1.0/years) - 1.0
        else:
            cagr = -1.0
        cagrs.append(cagr)
    return np.array(cagrs)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G20D: OOS Bootstrap — NEW CANDIDATE (vz=0.65) vs F10 (vz=0.70)')
    print('=' * 80)
    print(f'  n_bootstrap = {N_BOOTSTRAP}, block_size = {BLOCK_SIZE} 日, seed = {RNG_SEED}')

    a = load_shared_assets()
    dates = a['dates']
    close = a['close']

    print('\n[NAV 構築中...]')
    nav_065 = build_nav_for_vz(a, dates, close, 0.65)
    nav_070 = build_nav_for_vz(a, dates, close, 0.70)

    # OOS 期間で抽出
    oos_mask = (dates >= OOS_START_TS) & (dates <= OOS_END_TS)
    ret_065 = nav_065.pct_change().fillna(0).values[oos_mask.values]
    ret_070 = nav_070.pct_change().fillna(0).values[oos_mask.values]
    diff = ret_065 - ret_070
    n_oos = len(diff)
    years_oos = n_oos / TRADING_DAYS
    print(f'  OOS 日数: {n_oos}, OOS 年数: {years_oos:.2f}')

    # 実 CAGR_diff (NEW - F10)
    cum_065 = float(np.prod(1.0 + ret_065))
    cum_070 = float(np.prod(1.0 + ret_070))
    cagr_065 = cum_065**(1.0/years_oos) - 1.0
    cagr_070 = cum_070**(1.0/years_oos) - 1.0
    actual_diff_pp = (cagr_065 - cagr_070) * 100
    print(f'\n  実 CAGR_OOS NEW (vz=0.65): {cagr_065*100:+.2f}%')
    print(f'  実 CAGR_OOS F10 (vz=0.70): {cagr_070*100:+.2f}%')
    print(f'  実 diff: {actual_diff_pp:+.2f}pp')

    # Block bootstrap on diff series
    rng = np.random.default_rng(RNG_SEED)
    print(f'\n[Bootstrap 実行中... ({N_BOOTSTRAP} resamples)]')
    boot_cagrs = block_bootstrap(diff, BLOCK_SIZE, N_BOOTSTRAP, rng)
    # diff 日次リターンを bootstrap → cum product で diff CAGR を直接近似
    # 注: diff = r065 - r070、cum(1+diff) は厳密には CAGR_065 - CAGR_070 と一致しない
    # → r065 と r070 の各 cum を別途取って差分する block-paired bootstrap が厳密
    # 簡易版では cum(1+diff) を近似指標として使う
    boot_diff_pp = boot_cagrs * 100  # pp 単位

    ci_low = np.percentile(boot_diff_pp, 2.5)
    ci_high = np.percentile(boot_diff_pp, 97.5)
    ci_median = np.percentile(boot_diff_pp, 50)
    pct_above_zero = (boot_diff_pp > 0).mean() * 100

    print(f'\n[Bootstrap 結果 (近似版 1: cum(1+diff))]')
    print(f'  median: {ci_median:+.2f}pp')
    print(f'  95% CI: [{ci_low:+.2f}pp, {ci_high:+.2f}pp]')
    print(f'  P(diff > 0): {pct_above_zero:.1f}%')

    # 厳密版: paired block bootstrap on (r065[i], r070[i]) pair
    print(f'\n[Bootstrap 実行中... (厳密版 — paired block)]')
    rng2 = np.random.default_rng(RNG_SEED + 1)
    paired_cagr_065 = []
    paired_cagr_070 = []
    n_blocks = int(np.ceil(n_oos / BLOCK_SIZE))
    for _ in range(N_BOOTSTRAP):
        block_starts = rng2.integers(0, n_oos - BLOCK_SIZE + 1, size=n_blocks)
        s_065 = np.concatenate([ret_065[s:s+BLOCK_SIZE] for s in block_starts])[:n_oos]
        s_070 = np.concatenate([ret_070[s:s+BLOCK_SIZE] for s in block_starts])[:n_oos]
        c065 = float(np.prod(1.0 + s_065))
        c070 = float(np.prod(1.0 + s_070))
        cagr_a = c065**(1.0/years_oos) - 1.0 if c065 > 0 else -1.0
        cagr_b = c070**(1.0/years_oos) - 1.0 if c070 > 0 else -1.0
        paired_cagr_065.append(cagr_a)
        paired_cagr_070.append(cagr_b)
    paired_diff_pp = (np.array(paired_cagr_065) - np.array(paired_cagr_070)) * 100

    ci_low2 = np.percentile(paired_diff_pp, 2.5)
    ci_high2 = np.percentile(paired_diff_pp, 97.5)
    ci_median2 = np.percentile(paired_diff_pp, 50)
    pct_above_zero2 = (paired_diff_pp > 0).mean() * 100

    print(f'\n[Bootstrap 結果 (厳密版 paired)]')
    print(f'  median: {ci_median2:+.2f}pp')
    print(f'  95% CI: [{ci_low2:+.2f}pp, {ci_high2:+.2f}pp]')
    print(f'  P(diff > 0): {pct_above_zero2:.1f}%')

    # 保存
    out = pd.DataFrame({
        'metric': ['actual_diff_pp', 'cum1+diff_median', 'cum1+diff_CI_low', 'cum1+diff_CI_high',
                    'cum1+diff_P(>0)%',
                    'paired_median', 'paired_CI_low', 'paired_CI_high', 'paired_P(>0)%'],
        'value': [actual_diff_pp, ci_median, ci_low, ci_high, pct_above_zero,
                   ci_median2, ci_low2, ci_high2, pct_above_zero2],
    })
    csv_out = os.path.join(BASE, 'g20d_bootstrap_oos_results.csv')
    out.to_csv(csv_out, index=False)
    print(f'\n→ CSV: {csv_out}')

    print('\n[判定]')
    if ci_low2 > 0:
        print(f'  ✅ 95% CI 下端 {ci_low2:+.2f}pp > 0 → H0 棄却、NEW CANDIDATE は統計的に F10 を上回る')
    elif ci_low2 > -0.5:
        print(f'  ⚠ 95% CI 下端 {ci_low2:+.2f}pp が 0 近傍 → marginal significance')
    else:
        print(f'  ❌ 95% CI 下端 {ci_low2:+.2f}pp ≤ 0 → 偶然の可能性、棄却不可')

    if pct_above_zero2 > 95:
        print(f'  ✅ P(diff > 0) = {pct_above_zero2:.1f}% > 95%')
    elif pct_above_zero2 > 90:
        print(f'  ⚠ P(diff > 0) = {pct_above_zero2:.1f}% borderline')
    else:
        print(f'  ❌ P(diff > 0) = {pct_above_zero2:.1f}% < 95%')


if __name__ == '__main__':
    main()
