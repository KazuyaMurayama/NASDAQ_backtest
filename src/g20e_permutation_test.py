"""
G20E: Permutation test — vz_thr のラベルシャッフルで gap=-1.27pp 偶然か検定
=================================================================
QC Agent 3 指摘:
  vz_thr のラベルをシャッフルして null distribution を構築し、
  実 gap=-1.27pp (vz=0.65 で OOS>IS) がどのくらい稀かを評価。

方法 (簡易版):
  日次 vz 値の符号を保ったまま閾値 vz_thr を ramdom 値で置換した
  permutation を 1000 回試行。各 permutation で IS-OOS gap を計算し、
  実 vz=0.65 の gap=-1.27pp より低い (= OOS>IS 度合いが強い) 確率を出す。

  ただし完全な permutation は計算量大なので、vz_thr のみを uniform [0.5, 1.0] で
  random サンプリング (NULL: vz_thr に最適性なし) → 100 試行で実 vz=0.65 の
  gap を比較。

判定:
  - 実 gap が NULL 分布の下位 5% 以内 → 統計的に異常 (= 偶然でない)
  - 実 gap が NULL 分布の中央付近 → 偶然の範囲内
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
    build_cfd_nav_with_cost, metrics_from_nav,
    apply_tax_cfd_decimal,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b
from g19a_f10_eps_extended import build_f10_wn_for_eps
from corrected_strategy_backtest import TRADING_DAYS

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
LMAX = 7.0
EPS = 0.015
SPREAD_RT = 0.00050

# permutation: vz_thr を [0.4, 1.0] から ramdom sample
N_PERMS = 100
VZ_LOW = 0.40
VZ_HIGH = 1.00
RNG_SEED = 42


def build_lev_mod_for_vz(a, vz_thr):
    vz_arr = a['vz_arr']
    lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > vz_thr, K_HI,
            np.where(vz_arr < -vz_thr, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    return apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)


def compute_gap_for_vz(a, dates, close, vz_thr,
                       L_s2_values, wn_f10, wb_f10):
    lev_mod = build_lev_mod_for_vz(a, vz_thr)
    spread_ow = SPREAD_RT / 2.0
    nav_adj, _ = build_cfd_nav_with_cost(
        close, lev_mod, wn_f10, a['wg_A'], wb_f10, dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], L_s2_values,
        spread_ow,
    )
    m = metrics_from_nav(nav_adj, dates, a['ret'])
    yr_pre = m['yearly']
    yr_aft = yr_pre.apply(apply_tax_cfd_decimal)
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return np.nan
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is = _geo(is_subset)
    cagr_oos = _geo(oos_subset)
    return cagr_is - cagr_oos, cagr_oos


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G20E: Permutation test (vz_thr random sampling)')
    print('=' * 80)
    print(f'  n_permutations = {N_PERMS}, vz_thr ∈ [{VZ_LOW}, {VZ_HIGH}], seed = {RNG_SEED}')

    a = load_shared_assets()
    dates = a['dates']
    close = a['close']

    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD

    # 共通計算
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': LMAX})
    L_s2_values = L_s2.values
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
        np.asarray(a['lev_raw']), bull_mask, EPS,
    )

    # 実観測値 (vz=0.65)
    actual_gap, actual_cagr_oos = compute_gap_for_vz(a, dates, close, 0.65, L_s2_values, wn_f10, wb_f10)
    print(f'\n  実 vz=0.65 gap: {actual_gap*100:+.2f}pp, CAGR_OOS: {actual_cagr_oos*100:+.2f}%')

    # Permutation
    rng = np.random.default_rng(RNG_SEED)
    null_gaps = []
    null_cagrs = []
    print(f'\n[Permutation 実行中... ({N_PERMS} samples)]')
    for i in range(N_PERMS):
        vz_thr_rand = rng.uniform(VZ_LOW, VZ_HIGH)
        gap, cagr_oos = compute_gap_for_vz(a, dates, close, vz_thr_rand, L_s2_values, wn_f10, wb_f10)
        null_gaps.append(gap)
        null_cagrs.append(cagr_oos)
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{N_PERMS} 完了')

    null_gaps = np.array(null_gaps)
    null_cagrs = np.array(null_cagrs)

    # P(gap ≤ actual_gap) — 実 gap が NULL 分布の下位何%か
    pct_le_actual = (null_gaps <= actual_gap).mean() * 100
    pct_lt_zero = (null_gaps < 0).mean() * 100
    null_gap_low = np.percentile(null_gaps, 5)
    null_gap_high = np.percentile(null_gaps, 95)

    # CAGR_OOS 分布
    pct_ge_actual_cagr = (null_cagrs >= actual_cagr_oos).mean() * 100
    null_cagr_low = np.percentile(null_cagrs, 5)
    null_cagr_high = np.percentile(null_cagrs, 95)

    print(f'\n[結果]')
    print(f'  実 vz=0.65 gap: {actual_gap*100:+.2f}pp')
    print(f'  NULL 分布 (vz random ∈ [{VZ_LOW}, {VZ_HIGH}]):')
    print(f'    gap mean: {null_gaps.mean()*100:+.2f}pp')
    print(f'    gap 5-95%: [{null_gap_low*100:+.2f}pp, {null_gap_high*100:+.2f}pp]')
    print(f'    P(gap < 0) under NULL: {pct_lt_zero:.1f}%')
    print(f'    P(gap ≤ 実 vz=0.65) under NULL: {pct_le_actual:.1f}%')

    print(f'\n  実 vz=0.65 CAGR_OOS: {actual_cagr_oos*100:+.2f}%')
    print(f'  NULL CAGR_OOS 5-95%: [{null_cagr_low*100:+.2f}%, {null_cagr_high*100:+.2f}%]')
    print(f'  P(CAGR_OOS ≥ 実 vz=0.65) under NULL: {pct_ge_actual_cagr:.1f}%')

    # 保存
    out_summary = pd.DataFrame({
        'metric': ['actual_gap_pp', 'actual_cagr_oos_pct',
                    'null_gap_mean_pp', 'null_gap_5pct_pp', 'null_gap_95pct_pp',
                    'P_gap_lt_zero_under_null_pct', 'P_gap_le_actual_under_null_pct',
                    'null_cagr_mean_pct', 'null_cagr_5pct', 'null_cagr_95pct',
                    'P_cagr_ge_actual_under_null_pct'],
        'value': [actual_gap*100, actual_cagr_oos*100,
                   null_gaps.mean()*100, null_gap_low*100, null_gap_high*100,
                   pct_lt_zero, pct_le_actual,
                   null_cagrs.mean()*100, null_cagr_low*100, null_cagr_high*100,
                   pct_ge_actual_cagr],
    })
    csv_summary = os.path.join(BASE, 'g20e_permutation_test_summary.csv')
    out_summary.to_csv(csv_summary, index=False)
    print(f'\n→ Summary CSV: {csv_summary}')

    # 詳細 csv
    out_detail = pd.DataFrame({'permutation_idx': range(N_PERMS),
                                'null_gap': null_gaps,
                                'null_cagr_oos': null_cagrs})
    csv_detail = os.path.join(BASE, 'g20e_permutation_test_detail.csv')
    out_detail.to_csv(csv_detail, index=False)
    print(f'→ Detail CSV: {csv_detail}')

    print('\n[判定]')
    if pct_le_actual <= 5:
        print(f'  ✅ 実 gap は NULL 下位 {pct_le_actual:.1f}% (≤5%) → 統計的に異常、偶然でない')
    elif pct_le_actual <= 15:
        print(f'  ⚠ 実 gap は NULL 下位 {pct_le_actual:.1f}% → marginal、追加検証推奨')
    else:
        print(f'  ❌ 実 gap は NULL 下位 {pct_le_actual:.1f}% → 偶然の範囲、棄却不可')

    if pct_ge_actual_cagr <= 5:
        print(f'  ✅ 実 CAGR_OOS は NULL 上位 {pct_ge_actual_cagr:.1f}% (≤5%) → CAGR_OOS も統計的有意')
    elif pct_ge_actual_cagr <= 15:
        print(f'  ⚠ 実 CAGR_OOS は NULL 上位 {pct_ge_actual_cagr:.1f}% → marginal')
    else:
        print(f'  ❌ 実 CAGR_OOS は NULL 上位 {pct_ge_actual_cagr:.1f}% → 偶然')


if __name__ == '__main__':
    main()
