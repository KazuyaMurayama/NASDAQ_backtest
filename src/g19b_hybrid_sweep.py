"""
G19B: F10 + D5 ハイブリッド 3D sweep (vz_thr × l_max × ε)
=================================================================
v6.1 で F10 (vz=0.7, lmax=7, ε=0.015) vs D5 (vz=0.65, lmax=5.5, ε=なし) が拮抗。
中間に SOTA が存在するか 3D grid 探索。

グリッド:
  vz_thr  ∈ {0.60, 0.65, 0.70}      # 3点 (D5/E4周辺)
  l_max   ∈ {5.0, 5.5, 7.0}         # 3点 (D5/E4の代表値)
  eps_f10 ∈ {0.010, 0.015, 0.020}   # 3点 (Task A 確認の頑健範囲)
  合計: 27 configs

評価: g18 と同等の日次取引コスト (moderate spread=0.05%)
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
    load_shared_assets, SBI_CFD_SPREAD, BASE,
    compute_tilt_with_deadband,
    K_LO, K_HI, K_MID, N_LT2,
    TILT_R5, THRESHOLD,
)
from g18_daily_trade_cost_wfa import (
    build_cfd_nav_with_cost, metrics_from_nav,
    apply_tax_cfd_decimal,
    IS_END_TS, OOS_START_TS, OOS_END_TS,
)
from corrected_strategy_backtest import TRADING_DAYS
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b
from g19a_f10_eps_extended import build_f10_wn_for_eps

VZ_THR_GRID = [0.60, 0.65, 0.70]
LMAX_GRID   = [5.0, 5.5, 7.0]
EPS_GRID    = [0.010, 0.015, 0.020]

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)

# moderate spread case
SPREAD_RT = 0.00050  # 0.05% round-trip


def build_lev_mod_for_vz(a, vz_thr):
    """vz_thr で k_dyn を構築し、lev_mod を返す。"""
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
    print(f'G19B: Hybrid 3D sweep (vz × lmax × ε) - 27 configs')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    close = a['close']
    ret_nas = a['ret']

    raw_a2_vals = a['raw_a2'].values
    vz_vals = a['vz_arr']
    bull_mask = raw_a2_vals > THRESHOLD

    # L_s2 各 lmax を事前計算
    L_s2_cache = {}
    for lmax in LMAX_GRID:
        params = {**S2_BASE, 'l_max': lmax}
        L_s2_cache[lmax] = compute_L_s2_vz_gated(a['ret'], a['vz'], **params)
        print(f'  L_s2 lmax={lmax} done')

    # lev_mod 各 vz_thr を事前計算
    lev_mod_cache = {}
    for vz_thr in VZ_THR_GRID:
        lev_mod_cache[vz_thr] = build_lev_mod_for_vz(a, vz_thr)
        print(f'  lev_mod vz_thr={vz_thr} done')

    rows = []
    total = len(VZ_THR_GRID) * len(LMAX_GRID) * len(EPS_GRID)
    i = 0
    for vz_thr in VZ_THR_GRID:
        for lmax in LMAX_GRID:
            for eps in EPS_GRID:
                i += 1
                # F10 wn/wb with deadband ε
                wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
                    raw_a2_vals, vz_vals, a['wn_A'], a['wb_A'],
                    np.asarray(a['lev_raw']), bull_mask, eps,
                )
                lev_mod = lev_mod_cache[vz_thr]
                L_s2 = L_s2_cache[lmax]

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
                    vz_thr=vz_thr, l_max=lmax, eps=eps,
                    CAGR_IS=cagr_is_aft, CAGR_OOS=cagr_oos_aft,
                    IS_OOS_gap=cagr_is_aft - cagr_oos_aft,
                    Sharpe_OOS=m['Sharpe_OOS'], MaxDD_FULL=m['MaxDD_FULL'],
                    Worst10Y_star=m['Worst10Y_star'], P10_5Y=m['P10_5Y'],
                    yr_cost_approx=yr_cost,
                ))
                print(f'  [{i:2d}/{total}] vz={vz_thr}/lmax={lmax}/ε={eps}: '
                      f'CAGR_OOS={cagr_oos_aft*100:+.2f}%, gap={(cagr_is_aft-cagr_oos_aft)*100:+.2f}pp, '
                      f'Sharpe={m["Sharpe_OOS"]:+.3f}, MaxDD={m["MaxDD_FULL"]*100:+.1f}%')

    df_out = pd.DataFrame(rows)
    csv_out = os.path.join(BASE, 'g19b_hybrid_sweep_results.csv')
    df_out.to_csv(csv_out, index=False)
    print(f'\n→ CSV saved: {csv_out}')

    print('\n[結果ランキング — CAGR_OOS 降順 TOP 10]')
    top = df_out.sort_values('CAGR_OOS', ascending=False).head(10)
    print(f'{"vz":>5s} {"lmax":>5s} {"ε":>6s} {"CAGR_OOS":>10s} {"gap":>9s} {"Sharpe":>8s} {"MaxDD":>9s} {"Worst10Y":>10s}')
    print('-' * 80)
    for _, r in top.iterrows():
        print(f'{r["vz_thr"]:>5.2f} {r["l_max"]:>5.1f} {r["eps"]:>6.3f} '
              f'{r["CAGR_OOS"]*100:>+8.2f}% {r["IS_OOS_gap"]*100:>+7.2f}pp '
              f'{r["Sharpe_OOS"]:>+8.3f} {r["MaxDD_FULL"]*100:>+7.2f}% '
              f'{r["Worst10Y_star"]*100:>+8.2f}%')

    print('\n[Sharpe_OOS 降順 TOP 5]')
    top_s = df_out.sort_values('Sharpe_OOS', ascending=False).head(5)
    for _, r in top_s.iterrows():
        print(f'  vz={r["vz_thr"]:.2f}/lmax={r["l_max"]:.1f}/ε={r["eps"]:.3f}: '
              f'Sharpe={r["Sharpe_OOS"]:+.3f}, CAGR_OOS={r["CAGR_OOS"]*100:+.2f}%, gap={r["IS_OOS_gap"]*100:+.2f}pp')

    print('\n[MaxDD shallowest TOP 5]')
    top_dd = df_out.sort_values('MaxDD_FULL', ascending=False).head(5)
    for _, r in top_dd.iterrows():
        print(f'  vz={r["vz_thr"]:.2f}/lmax={r["l_max"]:.1f}/ε={r["eps"]:.3f}: '
              f'MaxDD={r["MaxDD_FULL"]*100:+.2f}%, CAGR_OOS={r["CAGR_OOS"]*100:+.2f}%, Sharpe={r["Sharpe_OOS"]:+.3f}')


if __name__ == '__main__':
    main()
