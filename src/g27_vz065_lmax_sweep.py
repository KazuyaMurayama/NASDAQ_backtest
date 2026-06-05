"""G27: vz=0.65 + F10ε=0.015 の lmax sensitivity sweep (l5, l5.5, l7)
=================================================================
ユーザー要望 (2026-06-05): vz=0.65+l7+F10ε に対し lmax を 5.0/5.5/7.0 で比較。
既存検索結果:
  - S2_VZGated+LT2_N750_vz065_lmax5 (vz=0.65+l5, F10 なし) — 存在
  - S2_VZGated+LT2_N750_F10lmax5 (F10+l5、vz=0.70=E4) — 存在
  - vz=0.65+l5+F10ε / vz=0.65+l5.5+F10ε (CFD) — **両方とも存在せず → 本スクリプトで初実装**

REF: vz=0.65+l7+F10ε (現 CFD 環境 Active 候補、g26 REF と同一)

コスト前提: CFD spread = 0.05% (moderate), §3-A 税モデル

出力:
  - g27_vz065_lmax_sweep_metrics.csv (3 戦略 × 10 指標)
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets, BASE, generate_windows, K_LO, K_HI, K_MID, THRESHOLD
from g18_daily_trade_cost_wfa import (
    build_cfd_nav_with_cost, wfa_metrics, metrics_from_nav,
    apply_tax_cfd_decimal,
)
from g19a_f10_eps_extended import build_f10_wn_for_eps
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from long_cycle_signal import apply_lt_mode_b

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
SPREAD_RT = 0.00050  # moderate
EPS_F10 = 0.015
VZ_THR = 0.65


def build_lev_mod_065(a):
    vz_arr = a['vz_arr']; lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > VZ_THR, K_HI,
            np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    obj = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    return obj.values if hasattr(obj, 'values') else np.asarray(obj)


def build_variant(a, lmax):
    """vz=0.65 + lmax=X + F10ε=0.015"""
    raw_a2 = a['raw_a2'].values; vz = a['vz_arr']
    bull_mask = raw_a2 > THRESHOLD
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': lmax}).values
    lev_mod = build_lev_mod_065(a)
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2, vz, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw']), bull_mask, EPS_F10,
    )
    wg = np.asarray(a['wg_A'])
    nav, _ = build_cfd_nav_with_cost(
        a['close'], lev_mod, wn_f10, wg, wb_f10,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
        L_s2, SPREAD_RT/2.0,
    )
    return nav, lev_mod, wn_f10, wg, wb_f10, L_s2


def calc_10metrics(label, nav, dates, ret_nas, wn, wb, lev_arr, windows):
    m = metrics_from_nav(nav, dates, ret_nas)
    yr_pre = m['yearly']
    yr_aft = yr_pre.apply(apply_tax_cfd_decimal)
    is_subset  = yr_aft.loc[[y for y in yr_aft.index if 1977 <= y <= 2020]]
    oos_subset = yr_aft.loc[[y for y in yr_aft.index if 2021 <= y <= 2026]]
    def _geo(x):
        if len(x) == 0: return float('nan')
        c = float(np.prod(1.0 + x.values))
        return c**(1.0/len(x)) - 1.0 if c > 0 else -1.0
    cagr_is, cagr_oos = _geo(is_subset), _geo(oos_subset)
    wfa = wfa_metrics(nav, dates, windows, lev_arr=lev_arr, wn_arr=wn, wb_arr=wb)
    return dict(
        Strategy=label,
        CAGR_OOS_pct=cagr_oos*100,
        cum_CAGR_OOS_pct=cagr_oos*100,
        cum_CAGR_IS_pct=cagr_is*100,
        min_CAGR_pct=min(cagr_is, cagr_oos)*100,
        IS_OOS_gap_pp=(cagr_is - cagr_oos)*100,
        Sharpe_OOS=m['Sharpe_OOS'],
        MaxDD_FULL_pct=m['MaxDD_FULL']*100,
        Worst10Y_CAGR_pct=m['Worst10Y_star']*100,
        P10_5Y_CAGR_pct=m['P10_5Y']*100,
        Trades_yr=wfa.get('mean_Trades_yr', np.nan),
        WFA_WFE=wfa.get('WFA_WFE', np.nan),
        WFA_CI95_lo_pct=wfa.get('WFA_CI95_lo', np.nan)*100,
    )


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G27: vz=0.65+F10ε lmax sensitivity sweep (l5.0 / l5.5 / l7.0)')
    print('=' * 80)
    a = load_shared_assets()
    dates = a['dates']
    ret_nas = a['ret']
    windows = generate_windows(dates)
    print(f'\n[Windows: {len(windows)}]')

    rows = []
    for lmax in [5.0, 5.5, 7.0]:
        label = f'vz=0.65+l{lmax:g}+F10ε'
        if lmax == 7.0:
            label += ' (REF)'
        elif lmax == 5.5:
            label += ' ★new'
        else:
            label += ' ★new'
        print(f'\n[Building {label}...]')
        nav, lev_mod, wn, wg, wb, L_s2 = build_variant(a, lmax)
        m = calc_10metrics(label, nav, dates, ret_nas, wn, wb, L_s2, windows)
        rows.append(m)
        print(f'  CAGR_OOS={m["CAGR_OOS_pct"]:+.2f}%  IS={m["cum_CAGR_IS_pct"]:+.2f}%  '
              f'min={m["min_CAGR_pct"]:+.2f}%  gap={m["IS_OOS_gap_pp"]:+.2f}pp')
        print(f'  Sharpe={m["Sharpe_OOS"]:+.3f}  MaxDD={m["MaxDD_FULL_pct"]:+.2f}%  '
              f'W10Y={m["Worst10Y_CAGR_pct"]:+.2f}%  P10_5Y={m["P10_5Y_CAGR_pct"]:+.2f}%')
        print(f'  Trades={m["Trades_yr"]:.0f}  WFE={m["WFA_WFE"]:.3f}  '
              f'CI95={m["WFA_CI95_lo_pct"]:+.2f}%')

    df = pd.DataFrame(rows)
    csv = os.path.join(BASE, 'g27_vz065_lmax_sweep_metrics.csv')
    df.to_csv(csv, index=False, float_format='%.4f')
    print(f'\n→ Metrics CSV: {csv}')

    # min ルール ranking
    print('\n[min(IS, OOS) CAGR ranking]')
    df_sorted = df.sort_values('min_CAGR_pct', ascending=False)
    for i, (_, r) in enumerate(df_sorted.iterrows(), 1):
        print(f'  {i}. {r["Strategy"]:35s}  min={r["min_CAGR_pct"]:+.2f}%  '
              f'(IS={r["cum_CAGR_IS_pct"]:+.2f}, OOS={r["CAGR_OOS_pct"]:+.2f})')


if __name__ == '__main__':
    main()
