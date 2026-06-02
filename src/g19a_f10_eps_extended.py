"""
G19A: F10 ε-deadband 拡張 sweep + 日次取引コスト評価
=================================================================
v6.1 で確定した F10 ε=0.015 の周辺で隣接 ε 値の頑健性を確認する。
g18_daily_trade_cost_wfa.py と同等の日次コスト評価を全 ε で適用。

ε grid (拡張版):
  {0.000, 0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.050}
  → 11 点 (既存 7点 + 新規 4点 = 0.008, 0.012, 0.018, 0.025)

評価コスト: g18 と同一 4 spread ケース (CFD round-trip):
  measured (GMO 0.020%) / optimistic (0.030%) / moderate (0.050%) / conservative (0.100%)

出力指標 (各 ε × 各 spread): CAGR_OOS_pre_tax, CAGR_OOS_net, Sharpe_OOS,
  MaxDD, Worst10Y★, P10_5Y▷, IS-OOS gap CAGR, Trades_yr
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
    load_shared_assets, SBI_CFD_SPREAD, BASE, TODAY,
    compute_tilt_with_deadband,
    K_LO, K_HI, VZ_THR_E4, K_MID, N_LT2,
    TILT_R5, EPS_F10,
    THRESHOLD,
)
from g18_daily_trade_cost_wfa import (
    build_cfd_nav_with_cost, metrics_from_nav,
    SPREAD_CASES_CFD, apply_tax_cfd_decimal,
    UNCAPTURED_DRAG_CFD, TAX_FACTOR,
    IS_END_TS, OOS_START_TS, OOS_END_TS,
)
from corrected_strategy_backtest import (
    simulate_rebalance_A, TRADING_DAYS,
)

# ε grid 拡張版
EPS_GRID_EXT = [0.000, 0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.050]


def build_f10_wn_for_eps(raw_a2_vals, vz_vals, wn_A, wb_A, lev_raw_arr, bull_mask, eps):
    """F10 ε deadband を適用した wn / wb を返す。

    F10 構造:
      tilt_confirmed = compute_tilt_with_deadband(raw_a2, vz, bull_mask, eps)
      wn_f10 = clip(wn_A + tilt_confirmed, 0, 1)
      wb_f10 = 1 - wn_f10  (gold/bond 配分は wn の残りを bond に振る簡略仕様)
    """
    tilt_conf, n_upd = compute_tilt_with_deadband(raw_a2_vals, vz_vals, bull_mask, eps)
    wn_arr = np.asarray(wn_A, dtype=float)
    wb_arr = np.asarray(wb_A, dtype=float)
    wn_f10 = np.clip(wn_arr + tilt_conf, 0.0, 1.0)
    # 余剰を bond へ ((1 - wn) のうち wb を維持比率で再分配)
    delta = wn_f10 - wn_arr  # 正(NDX に振り直す量)
    wb_f10 = np.maximum(wb_arr - delta, 0.0)  # bond を削減
    # wg は維持 (g14 / g7 と同じ)
    return wn_f10, wb_f10, tilt_conf, n_upd


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print(f'G19A: F10 ε-deadband 拡張 sweep + 日次取引コスト評価')
    print(f'実行日: 2026-06-02')
    print('=' * 80)

    print('\n[S1] Loading shared assets (g14 同等)...')
    a = load_shared_assets()
    dates = a['dates']
    close = a['close']
    ret_nas = a['ret']

    raw_a2_vals = a['raw_a2'].values if hasattr(a['raw_a2'], 'values') else np.asarray(a['raw_a2'])
    vz_vals = a['vz'].values if hasattr(a['vz'], 'values') else np.asarray(a['vz'])
    lev_raw_arr = np.asarray(a['lev_raw'])
    bull_mask = raw_a2_vals > THRESHOLD

    # F10 は S2_lmax7 + E4 regime lev_mod (= a['lev_mod_e4'])
    lev_mod_e4 = a['lev_mod_e4']
    L_s2_lmax7 = a['L_s2_lmax7']

    print('\n[S2] Sweeping F10 ε × spread (11 ε × 4 spread = 44 configs)...')
    rows = []
    for eps in EPS_GRID_EXT:
        # F10 wn/wb with deadband
        wn_f10, wb_f10, tilt_conf, n_upd = build_f10_wn_for_eps(
            raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'], lev_raw_arr, bull_mask, eps,
        )
        # 各 spread ケースで NAV 計算
        for case, spread_rt in SPREAD_CASES_CFD.items():
            spread_ow = spread_rt / 2.0
            nav_adj, yr_cost = build_cfd_nav_with_cost(
                close, lev_mod_e4, wn_f10, a['wg_A'], wb_f10, dates,
                a['gold_2x'], a['bond_3x'], a['sofr'], L_s2_lmax7.values,
                spread_ow,
            )
            m = metrics_from_nav(nav_adj, dates, ret_nas)
            # tax 適用 — 年次 pre-tax を税後に変換し、税後 CAGR を再計算
            yearly_pre = m['yearly']
            yearly_aft = yearly_pre.apply(apply_tax_cfd_decimal)
            # OOS CAGR after tax (calendar year compound)
            yr_oos = yearly_aft[yearly_aft.index >= 2021]
            cum_oos = float(np.prod(1.0 + yr_oos.values))
            n_yr_oos = len(yr_oos)
            cagr_oos_net = cum_oos ** (1.0/n_yr_oos) - 1.0 if cum_oos > 0 and n_yr_oos > 0 else np.nan
            # IS CAGR after tax
            yr_is = yearly_aft[yearly_aft.index < 2021]
            cum_is = float(np.prod(1.0 + yr_is.values))
            n_yr_is = len(yr_is)
            cagr_is_net = cum_is ** (1.0/n_yr_is) - 1.0 if cum_is > 0 and n_yr_is > 0 else np.nan
            is_oos_gap_net = cagr_is_net - cagr_oos_net

            # Trades_yr (g14 流儀 = lev_mod / wn / wb いずれかの変化)
            wn_arr_calc = np.asarray(wn_f10)
            wb_arr_calc = np.asarray(wb_f10)
            lev_calc = np.asarray(lev_mod_e4)
            n_changes = int(((np.diff(wn_arr_calc) != 0) | (np.diff(wb_arr_calc) != 0) |
                              (np.diff(lev_calc) != 0)).sum())
            trades_yr = n_changes / (len(dates) / TRADING_DAYS)

            rows.append(dict(
                eps=eps, spread_rt=spread_rt, case=case,
                CAGR_IS_pre=m['CAGR_IS'], CAGR_OOS_pre=m['CAGR_OOS'],
                CAGR_IS_net=cagr_is_net, CAGR_OOS_net=cagr_oos_net,
                IS_OOS_gap_net=is_oos_gap_net,
                Sharpe_OOS=m['Sharpe_OOS'], MaxDD_FULL=m['MaxDD_FULL'],
                Worst10Y_star=m['Worst10Y_star'], P10_5Y=m['P10_5Y'],
                Trades_yr=trades_yr, tilt_updates=n_upd,
                yr_cost_approx=yr_cost,
            ))

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(BASE, 'g19a_f10_eps_extended_results.csv')
    df_out.to_csv(out_csv, index=False)
    print(f'\n→ CSV saved: {out_csv}')

    # Summary table: moderate ケース (0.050%) で全 ε の比較
    print('\n[S3] Summary (moderate ケース: spread=0.05%):')
    sub = df_out[df_out['case'] == 'moderate'].sort_values('eps').reset_index(drop=True)
    print(f'{"eps":>6s} {"CAGR_OOS_net":>13s} {"IS-OOS_gap":>12s} {"Sharpe_OOS":>11s} {"MaxDD":>8s} {"Trades_yr":>10s} {"tilt_upd":>9s}')
    print('-' * 80)
    for _, r in sub.iterrows():
        marker = ' ★' if abs(r['eps'] - 0.015) < 1e-6 else ''
        print(f'{r["eps"]:>6.3f} {r["CAGR_OOS_net"]*100:>+11.2f}%  '
              f'{r["IS_OOS_gap_net"]*100:>+10.2f}pp  '
              f'{r["Sharpe_OOS"]:>+11.3f} {r["MaxDD_FULL"]*100:>+7.2f}% '
              f'{r["Trades_yr"]:>10.1f} {r["tilt_updates"]:>9d}{marker}')

    # 全 spread ケース併記
    print('\n[S4] Full grid (CAGR_OOS_net %, all spread cases):')
    pivot = df_out.pivot_table(index='eps', columns='case', values='CAGR_OOS_net') * 100
    case_order = ['measured (GMO 2026/4)', 'optimistic', 'moderate', 'conservative (base)']
    pivot = pivot[case_order]
    print(pivot.round(2).to_string())

    # Best ε per spread
    print('\n[S5] Best ε per spread case:')
    for case in case_order:
        sub = df_out[df_out['case'] == case].sort_values('CAGR_OOS_net', ascending=False).head(3)
        best = sub.iloc[0]
        print(f'  {case:>30s}: best ε={best["eps"]:.3f} → CAGR_OOS_net={best["CAGR_OOS_net"]*100:+.2f}%, '
              f'gap={best["IS_OOS_gap_net"]*100:+.2f}pp, Sharpe={best["Sharpe_OOS"]:+.3f}')


if __name__ == '__main__':
    main()
