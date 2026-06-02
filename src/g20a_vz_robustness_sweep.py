"""
G20A: NEW CANDIDATE vz_thr robustness sweep (5点)
=================================================================
QC Agent 3 指摘:
  vz=0.65/lmax=7/ε=0.015 で gap=-1.27pp (OOS>IS) は本物か lucky か?
  vz_thr の隣接値 {0.625, 0.65, 0.675, 0.70, 0.725} × lmax=7+ε=0.015
  で 5 点 sweep し、vz=0.65 のみ勝つなら overfit 確定。

合格基準:
  vz=0.65 周辺 (0.625 / 0.675) も CAGR_OOS = +20.5%以上、gap < +1.0pp なら
  「vz=0.65 は frontier の頂点で、その周辺も健全」→ 構造的優位を支持
  vz=0.65 単独が突出する場合 → overfit 確定 → 棄却
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

# vz_thr 隣接 5 点
VZ_THR_GRID = [0.625, 0.65, 0.675, 0.70, 0.725]
LMAX = 7.0
EPS = 0.015
SPREAD_RT = 0.00050  # moderate

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)


def build_lev_mod_for_vz(a, vz_thr):
    vz_arr = a['vz_arr']
    lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > vz_thr, K_HI,
            np.where(vz_arr < -vz_thr, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    return apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G20A: vz_thr robustness sweep (5 点) — NEW CANDIDATE 周辺検証')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    close = a['close']
    ret_nas = a['ret']

    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD

    # L_s2 (lmax=7) 共通
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': LMAX})
    print(f'  L_s2 lmax={LMAX} done')

    # F10 wn (ε=0.015) は vz_thr に依存しない
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
        np.asarray(a['lev_raw']), bull_mask, EPS,
    )

    rows = []
    for vz_thr in VZ_THR_GRID:
        lev_mod = build_lev_mod_for_vz(a, vz_thr)
        spread_ow = SPREAD_RT / 2.0
        nav_adj, yr_cost = build_cfd_nav_with_cost(
            close, lev_mod, wn_f10, a['wg_A'], wb_f10, dates,
            a['gold_2x'], a['bond_3x'], a['sofr'], L_s2.values,
            spread_ow,
        )
        m = metrics_from_nav(nav_adj, dates, ret_nas)
        yr_pre = m['yearly']
        yr_aft = yr_pre.apply(apply_tax_cfd_decimal)

        is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
        oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
        def _geo(x):
            if len(x) == 0: return np.nan
            c = float(np.prod(1.0 + x.values))
            return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
        cagr_is_aft = _geo(is_subset)
        cagr_oos_aft = _geo(oos_subset)

        rows.append(dict(
            vz_thr=vz_thr,
            CAGR_IS=cagr_is_aft, CAGR_OOS=cagr_oos_aft,
            IS_OOS_gap=cagr_is_aft - cagr_oos_aft,
            Sharpe_OOS=m['Sharpe_OOS'], MaxDD_FULL=m['MaxDD_FULL'],
            Worst10Y_star=m['Worst10Y_star'], P10_5Y=m['P10_5Y'],
        ))

    df = pd.DataFrame(rows)
    csv_out = os.path.join(BASE, 'g20a_vz_robustness_sweep.csv')
    df.to_csv(csv_out, index=False)
    print(f'\n→ CSV: {csv_out}\n')

    print(f'{"vz_thr":>7s} {"CAGR_IS":>10s} {"CAGR_OOS":>10s} {"IS-OOS gap":>11s} {"Sharpe":>8s} {"MaxDD":>9s} {"Worst10Y":>10s}')
    print('-' * 85)
    for _, r in df.iterrows():
        marker = ' ★' if abs(r['vz_thr'] - 0.65) < 1e-6 else ''
        print(f'{r["vz_thr"]:>7.3f} {r["CAGR_IS"]*100:>+8.2f}% {r["CAGR_OOS"]*100:>+8.2f}% '
              f'{r["IS_OOS_gap"]*100:>+9.2f}pp {r["Sharpe_OOS"]:>+8.3f} '
              f'{r["MaxDD_FULL"]*100:>+7.2f}% {r["Worst10Y_star"]*100:>+8.2f}%{marker}')

    # 検証ロジック
    print('\n[判定]')
    vz65_row = df[df['vz_thr']==0.65].iloc[0]
    vz625_row = df[df['vz_thr']==0.625].iloc[0]
    vz675_row = df[df['vz_thr']==0.675].iloc[0]
    cagr_diff_625 = (vz65_row['CAGR_OOS'] - vz625_row['CAGR_OOS']) * 100
    cagr_diff_675 = (vz65_row['CAGR_OOS'] - vz675_row['CAGR_OOS']) * 100
    print(f'  vz=0.65 vs 0.625 CAGR_OOS 差: {cagr_diff_625:+.2f}pp')
    print(f'  vz=0.65 vs 0.675 CAGR_OOS 差: {cagr_diff_675:+.2f}pp')

    neighbors_avg = df[df['vz_thr'].isin([0.625, 0.675])]['CAGR_OOS'].mean() * 100
    vz65_oos = vz65_row['CAGR_OOS'] * 100
    spike_pp = vz65_oos - neighbors_avg
    print(f'  vz=0.625/0.675 平均: {neighbors_avg:+.2f}%, vz=0.65: {vz65_oos:+.2f}%, spike: {spike_pp:+.2f}pp')

    # gap 単調性確認
    gaps = df['IS_OOS_gap'].values
    neg_gap_count = int((gaps < 0).sum())
    print(f'  負 gap (OOS>IS) の vz 数: {neg_gap_count}/5')
    if neg_gap_count == 1 and vz65_row['IS_OOS_gap'] < 0:
        print(f'  → vz=0.65 だけが負 gap → spurious fit の疑い濃厚')
    elif neg_gap_count >= 2:
        print(f'  → 複数 vz で負 gap → 構造的優位を支持')

    if abs(spike_pp) > 1.0:
        print(f'\n  ❌ vz=0.65 が周辺と {abs(spike_pp):.1f}pp 以上突出 → spike pattern (overfit 疑い)')
    else:
        print(f'\n  ✅ vz=0.65 周辺で smooth な勾配 (spike < 1pp) → 構造的優位を支持')


if __name__ == '__main__':
    main()
