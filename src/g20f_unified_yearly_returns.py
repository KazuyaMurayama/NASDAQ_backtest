"""
G20F: 7 戦略 + NEW CANDIDATE = 8 戦略 統合年次リターン表生成 (1974-2026)
=================================================================
v6.2 で 4→7 戦略に拡張、本 g20f で NEW CANDIDATE を含む合計 8 戦略の
年次リターン (税後・日次取引コスト後) + 統計サマリーを生成。

戦略順序 (採用判断順):
  1. NEW CANDIDATE (vz=0.65+lmax=7+F10ε=0.015)  ← v6.3 検証対象
  2. F10 ε=0.015 ★                                ← v6.1 推奨
  3. F8 R5_CALM_BOOST                              ← F10 同等
  4. F7v3+E4 A:tilt=2.0                            ← tilt 構造差別化
  5. D5 vz=0.65/lmax=5.5                           ← MaxDD 最浅候補
  6. E4 Regime k_lt ◆                              ← 現 Active
  7. DH Dyn 2x3x [A]                               ← ETF 戦略
  8. NDX 1x B&H                                    ← ベンチマーク

出力:
  - g20f_unified_yearly_returns.csv
  - g20f_unified_stats_summary.csv
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
    build_cfd_nav_with_cost, build_dh_nav_with_cost, metrics_from_nav,
    apply_tax_cfd_decimal, apply_tax_etf_decimal,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b
from g19a_f10_eps_extended import build_f10_wn_for_eps
from g19e_3strategies_daily_cost import build_f7v3_wn, compute_bnh_metrics_after_tax
from corrected_strategy_backtest import TRADING_DAYS

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
LMAX_7 = 7.0
LMAX_55 = 5.5
EPS_F10 = 0.015
SPREAD_RT = 0.00050  # moderate
DH_PER_UNIT = 0.0010  # moderate ETF cost


def build_lev_mod_for_vz(a, vz_thr):
    vz_arr = a['vz_arr']
    lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > vz_thr, K_HI,
            np.where(vz_arr < -vz_thr, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    return apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)


def get_yearly_aft_cfd(a, dates, close, vz_thr, lmax, use_f10=True, use_f7v3=False):
    """CFD 戦略の年次税後リターン (×0.8273 - 0.66%)。"""
    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD

    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': lmax})
    lev_mod = build_lev_mod_for_vz(a, vz_thr)

    if use_f10:
        wn, wb, _, _ = build_f10_wn_for_eps(
            raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
            np.asarray(a['lev_raw']), bull_mask, EPS_F10,
        )
        wg = a['wg_A']
    elif use_f7v3:
        wn, wb = build_f7v3_wn(raw_a2_vals, a['wn_A'], a['wb_A'])
        wg = a['wg_A']
    else:
        wn = a['wn_A']
        wb = a['wb_A']
        wg = a['wg_A']

    spread_ow = SPREAD_RT / 2.0
    nav_adj, _ = build_cfd_nav_with_cost(
        close, lev_mod, wn, wg, wb, dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
        spread_ow,
    )
    m = metrics_from_nav(nav_adj, dates, a['ret'])
    yr_aft = m['yearly'].apply(apply_tax_cfd_decimal)
    return yr_aft, nav_adj


def get_yearly_aft_f8r5(a, dates, close):
    """F8 R5 CALM_BOOST (F10 ε=0 baseline) — load_shared_assets で wn_ref_f8 として準備済み。"""
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': LMAX_7})
    lev_mod = a['lev_mod_e4']
    wn = a['wn_ref_f8']
    wg = a['wg_ref_f8']
    wb = a['wb_ref_f8']
    spread_ow = SPREAD_RT / 2.0
    nav_adj, _ = build_cfd_nav_with_cost(
        close, lev_mod, wn, wg, wb, dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
        spread_ow,
    )
    m = metrics_from_nav(nav_adj, dates, a['ret'])
    return m['yearly'].apply(apply_tax_cfd_decimal), nav_adj


def get_yearly_aft_dh(a, dates, close):
    """DH Dyn [A] — ETF 税モデル (×0.8273)。"""
    nav_adj, _ = build_dh_nav_with_cost(
        close, a['lev_raw'], a['wn_A'], a['wg_A'], a['wb_A'], dates,
        a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    m = metrics_from_nav(nav_adj, dates, a['ret'])
    return m['yearly'].apply(apply_tax_etf_decimal), nav_adj


def compute_stats(yr_series):
    """年次リターンから統計量を出す。"""
    y = yr_series.dropna()
    if len(y) == 0:
        return {k: np.nan for k in ['mean', 'median', 'std', 'min', 'max', 'positive_yrs', 'negative_yrs']}
    return dict(
        mean=float(y.mean()),
        median=float(y.median()),
        std=float(y.std()),
        min=float(y.min()),
        max=float(y.max()),
        positive_yrs=int((y > 0).sum()),
        negative_yrs=int((y < 0).sum()),
    )


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G20F: 8 戦略統合年次リターン (税後・日次取引コスト後・moderate)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    close = a['close']

    yearly_dict = {}

    print('\n[1/8] NEW CANDIDATE (vz=0.65+lmax=7+F10ε=0.015)...')
    yr, _ = get_yearly_aft_cfd(a, dates, close, vz_thr=0.65, lmax=LMAX_7, use_f10=True)
    yearly_dict['NEW (vz=0.65+l7+F10ε) 🔍'] = yr

    print('[2/8] F10 ε=0.015 ★ (vz=0.70+lmax=7+F10ε=0.015)...')
    yr, _ = get_yearly_aft_cfd(a, dates, close, vz_thr=0.70, lmax=LMAX_7, use_f10=True)
    yearly_dict['F10 ε=0.015 ★'] = yr

    print('[3/8] F8 R5_CALM_BOOST (vz=0.70+lmax=7, ε=0)...')
    yr, _ = get_yearly_aft_f8r5(a, dates, close)
    yearly_dict['F8 R5_CALM_BOOST'] = yr

    print('[4/8] F7v3+E4 A:tilt=2.0 (vz=0.70+lmax=7+F7v3)...')
    yr, _ = get_yearly_aft_cfd(a, dates, close, vz_thr=0.70, lmax=LMAX_7,
                                 use_f10=False, use_f7v3=True)
    yearly_dict['F7v3+E4 A:tilt=2.0'] = yr

    print('[5/8] D5 vz=0.65/lmax=5.5 (no F10 tilt)...')
    yr, _ = get_yearly_aft_cfd(a, dates, close, vz_thr=0.65, lmax=LMAX_55,
                                 use_f10=False)
    yearly_dict['D5 vz=0.65/lmax=5.5'] = yr

    print('[6/8] E4 Regime k_lt ◆ (vz=0.70+lmax=7, no F10)...')
    yr, _ = get_yearly_aft_cfd(a, dates, close, vz_thr=0.70, lmax=LMAX_7,
                                 use_f10=False)
    yearly_dict['E4 Regime k_lt ◆'] = yr

    print('[7/8] DH Dyn 2x3x [A] (TQQQ+TMF+2036)...')
    yr, _ = get_yearly_aft_dh(a, dates, close)
    yearly_dict['DH Dyn 2x3x [A]'] = yr

    print('[8/8] NDX 1x B&H (Benchmark, cost=0)...')
    bnh = compute_bnh_metrics_after_tax(close, dates)
    yearly_dict['NDX 1x B&H'] = bnh['yearly_aft']

    # 統合 DataFrame
    df = pd.DataFrame(yearly_dict)
    df.index.name = 'year'
    df = df.sort_index()
    df = df * 100  # to percent

    # 全期間 (1974-2026) で結合
    csv_yr = os.path.join(BASE, 'g20f_unified_yearly_returns.csv')
    df.to_csv(csv_yr)
    print(f'\n→ Yearly returns CSV: {csv_yr}')

    # 統計サマリ
    stats_rows = []
    for col, yr_pct in yearly_dict.items():
        s = compute_stats(yr_pct * 100 if not isinstance(yr_pct, pd.Series) else yr_pct.dropna() * 100)
        # Actually yearly_dict values are decimal (not pct); we multiplied df only
        # Recompute stats from decimal directly
        y = yr_pct.dropna()
        stats_rows.append({
            'strategy': col,
            'mean': y.mean() * 100,
            'median': y.median() * 100,
            'std': y.std() * 100,
            'max': y.max() * 100,
            'min': y.min() * 100,
            'positive_yrs': int((y > 0).sum()),
            'negative_yrs': int((y < 0).sum()),
            'n_yrs': len(y),
        })
    stats_df = pd.DataFrame(stats_rows)
    csv_stats = os.path.join(BASE, 'g20f_unified_stats_summary.csv')
    stats_df.to_csv(csv_stats, index=False)
    print(f'→ Stats CSV: {csv_stats}')

    # コンソール表示
    print('\n[統計サマリ (1974-2026、税後・日次取引コスト後、moderate)]')
    print(stats_df.to_string(index=False))

    print('\n[年次表 - 最初 5 年 & 直近 6 年 (OOS)]')
    head = df.head(5)
    oos = df.loc[df.index >= 2021]
    print('\n--- 最初 5 年 (1974-1978) ---')
    print(head.to_string())
    print('\n--- OOS (2021-2026) ---')
    print(oos.to_string())


if __name__ == '__main__':
    main()
