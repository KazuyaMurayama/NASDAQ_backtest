"""G26: vz=0.65+l7+F10ε に DH-W1 の非対称機構を移植する 3 候補
=================================================================
NEW base = vz=0.65+lmax=7+F10ε=0.015 (CFD strategy)
  lev_mod = build_lev_mod_for_vz(a, 0.65)  # continuous 0-1
  L_s2    = compute_L_s2_vz_gated(l_max=7.0)  # 1-7
  weights = wn_f10, wb_f10 (F10 ε tilt)

3 候補:
  vz0.65+l7+F10ε-AH (Asymm Hysteresis, full binary HOLD/OUT):
    DH-W1 と同じ hysteresis 状態機械 (Enter lev_mod_065 ≥ 0.7, Exit ≤ 0.3)
    HOLD: NEW base そのまま
    OUT: 全部 0 (lev_mod=0, weights=0, L_s2=0) → cash
    → Z2 と同じく cash 退避型

  vz0.65+l7+F10ε-AT (Asymm Continuous Tilt):
    binary mask なし、連続 scaling
    confidence_factor = clip((lev_mod_065 - 0.3) / 0.4, 0, 1) — 0.3 → 0, 0.7 → 1
    全パラメータを factor で scale (lev_mod, weights, L_s2)
    → 過学習 NEW を smooth に減衰させる

  vz0.65+l7+F10ε-HL (Hyst-Lite, 部分 OUT 30%):
    HOLD: NEW base 100%
    OUT (hysteresis state=0): NEW base × 0.30 (完全退避でなく 30% 残す)
    → cash 削減・配分維持の折衷

参考: NEW-REF (vz=0.65+l7+F10ε base、改変なし)、DH-W1 (ETF binary HOLD/OUT)

出力:
  - g26_new_asymmetric_metrics.csv
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
from g23a_dh_refinement_variants import hold_mask_W1

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
SPREAD_RT = 0.00050  # CFD moderate spread
EPS_F10 = 0.015


def build_lev_mod_065(a):
    vz_arr = a['vz_arr']; lt_sig_arr = a['lt_sig_arr']
    k_dyn = np.where(vz_arr > 0.65, K_HI,
            np.where(vz_arr < -0.65, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod_obj = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    return lev_mod_obj.values if hasattr(lev_mod_obj, 'values') else np.asarray(lev_mod_obj)


def build_NEW_REF(a):
    """NEW base — vz=0.65+l7+F10ε そのまま"""
    raw_a2 = a['raw_a2'].values; vz = a['vz_arr']
    bull_mask = raw_a2 > THRESHOLD
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': 7.0}).values
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


def build_NEW_AH(a):
    """NEW + Asymm Hysteresis (full binary HOLD/OUT, like DH-W1)"""
    mask = hold_mask_W1(a, enter_thr=0.7, exit_thr=0.3)
    raw_a2 = a['raw_a2'].values; vz = a['vz_arr']
    bull_mask = raw_a2 > THRESHOLD
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': 7.0}).values
    lev_mod_base = build_lev_mod_065(a)
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2, vz, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw']), bull_mask, EPS_F10,
    )
    wg = np.asarray(a['wg_A'])
    # Apply binary mask
    lev_mod = lev_mod_base * mask
    wn = wn_f10 * mask
    wb = wb_f10 * mask
    wg_m = wg * mask
    L_s2_m = L_s2 * mask  # 0 when OUT → nas_ret(0) = funding rebate
    nav, _ = build_cfd_nav_with_cost(
        a['close'], lev_mod, wn, wg_m, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
        L_s2_m, SPREAD_RT/2.0,
    )
    return nav, lev_mod, wn, wg_m, wb, L_s2_m, mask


def build_NEW_AT(a):
    """NEW + Asymm Continuous Tilt — smooth scaling without binary mask
    confidence_factor = clip((lev_mod_065 - 0.3) / 0.4, 0, 1)
    → 0.3 → 0 (no exposure), 0.7 → 1 (full NEW), smooth between"""
    lm = build_lev_mod_065(a)
    factor = np.clip((lm - 0.3) / 0.4, 0.0, 1.0)
    raw_a2 = a['raw_a2'].values; vz = a['vz_arr']
    bull_mask = raw_a2 > THRESHOLD
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': 7.0}).values
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2, vz, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw']), bull_mask, EPS_F10,
    )
    wg = np.asarray(a['wg_A'])
    # Apply continuous scaling
    lev_mod = lm * factor
    wn = wn_f10 * factor
    wb = wb_f10 * factor
    wg_m = wg * factor
    L_s2_m = L_s2 * factor
    nav, _ = build_cfd_nav_with_cost(
        a['close'], lev_mod, wn, wg_m, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
        L_s2_m, SPREAD_RT/2.0,
    )
    return nav, lev_mod, wn, wg_m, wb, L_s2_m, factor


def build_NEW_HL(a):
    """NEW + Hyst-Lite (HOLD: NEW base 100%, OUT: 30% of NEW)"""
    mask = hold_mask_W1(a, enter_thr=0.7, exit_thr=0.3)
    factor = mask + (1.0 - mask) * 0.30  # 1.0 when HOLD, 0.30 when OUT
    raw_a2 = a['raw_a2'].values; vz = a['vz_arr']
    bull_mask = raw_a2 > THRESHOLD
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': 7.0}).values
    lev_mod_base = build_lev_mod_065(a)
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2, vz, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw']), bull_mask, EPS_F10,
    )
    wg = np.asarray(a['wg_A'])
    lev_mod = lev_mod_base * factor
    wn = wn_f10 * factor
    wb = wb_f10 * factor
    wg_m = wg * factor
    L_s2_m = L_s2 * factor
    nav, _ = build_cfd_nav_with_cost(
        a['close'], lev_mod, wn, wg_m, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
        L_s2_m, SPREAD_RT/2.0,
    )
    return nav, lev_mod, wn, wg_m, wb, L_s2_m, factor


def calc_full_metrics(label, nav, dates, ret_nas, wn, wb, lev_arr_for_wfa, windows):
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
    wfa = wfa_metrics(nav, dates, windows, lev_arr=lev_arr_for_wfa, wn_arr=wn, wb_arr=wb)
    return dict(
        Strategy=label,
        CAGR_OOS_pct=cagr_oos*100,
        cum_CAGR_OOS_pct=cagr_oos*100,
        cum_CAGR_IS_pct=cagr_is*100,
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
    print('G26: vz=0.65+l7+F10ε に非対称機構を移植する 3 候補')
    print('=' * 80)
    a = load_shared_assets()
    dates = a['dates']
    ret_nas = a['ret']
    windows = generate_windows(dates)
    print(f'\n[Windows: {len(windows)}]')

    rows = []

    # REF (NEW base)
    print('\n[1/4] vz=0.65+l7+F10ε (REF, base)...')
    nav, lev_mod, wn, wg, wb, L_s2 = build_NEW_REF(a)
    m = calc_full_metrics('vz=0.65+l7+F10ε (REF)', nav, dates, ret_nas, wn, wb, L_s2, windows)
    rows.append(m)
    print(f'  CAGR_OOS={m["CAGR_OOS_pct"]:+.2f}%  IS={m["cum_CAGR_IS_pct"]:+.2f}%  '
          f'gap={m["IS_OOS_gap_pp"]:+.2f}pp  MaxDD={m["MaxDD_FULL_pct"]:+.2f}%  '
          f'WFE={m["WFA_WFE"]:.3f}')

    # AH
    print('\n[2/4] vz=0.65+l7+F10ε-AH (Asymm Hysteresis full HOLD/OUT)...')
    nav_ah, lev_mod_ah, wn_ah, wg_ah, wb_ah, L_s2_ah, mask = build_NEW_AH(a)
    hold_ratio = float(mask.mean()) * 100
    m_ah = calc_full_metrics('vz=0.65+l7+F10ε-AH', nav_ah, dates, ret_nas, wn_ah, wb_ah, L_s2_ah, windows)
    rows.append(m_ah)
    print(f'  HOLD={hold_ratio:.1f}%  CAGR_OOS={m_ah["CAGR_OOS_pct"]:+.2f}%  IS={m_ah["cum_CAGR_IS_pct"]:+.2f}%  '
          f'gap={m_ah["IS_OOS_gap_pp"]:+.2f}pp  MaxDD={m_ah["MaxDD_FULL_pct"]:+.2f}%  '
          f'WFE={m_ah["WFA_WFE"]:.3f}')

    # AT
    print('\n[3/4] vz=0.65+l7+F10ε-AT (Asymm Continuous Tilt)...')
    nav_at, lev_mod_at, wn_at, wg_at, wb_at, L_s2_at, factor = build_NEW_AT(a)
    avg_factor = float(np.nanmean(factor)) * 100
    m_at = calc_full_metrics('vz=0.65+l7+F10ε-AT', nav_at, dates, ret_nas, wn_at, wb_at, L_s2_at, windows)
    rows.append(m_at)
    print(f'  avg_factor={avg_factor:.1f}%  CAGR_OOS={m_at["CAGR_OOS_pct"]:+.2f}%  '
          f'IS={m_at["cum_CAGR_IS_pct"]:+.2f}%  gap={m_at["IS_OOS_gap_pp"]:+.2f}pp  '
          f'MaxDD={m_at["MaxDD_FULL_pct"]:+.2f}%  WFE={m_at["WFA_WFE"]:.3f}')

    # HL
    print('\n[4/4] vz=0.65+l7+F10ε-HL (Hyst-Lite, 部分 OUT 30%)...')
    nav_hl, lev_mod_hl, wn_hl, wg_hl, wb_hl, L_s2_hl, factor_hl = build_NEW_HL(a)
    avg_factor_hl = float(np.nanmean(factor_hl)) * 100
    m_hl = calc_full_metrics('vz=0.65+l7+F10ε-HL', nav_hl, dates, ret_nas, wn_hl, wb_hl, L_s2_hl, windows)
    rows.append(m_hl)
    print(f'  avg_factor={avg_factor_hl:.1f}%  CAGR_OOS={m_hl["CAGR_OOS_pct"]:+.2f}%  '
          f'IS={m_hl["cum_CAGR_IS_pct"]:+.2f}%  gap={m_hl["IS_OOS_gap_pp"]:+.2f}pp  '
          f'MaxDD={m_hl["MaxDD_FULL_pct"]:+.2f}%  WFE={m_hl["WFA_WFE"]:.3f}')

    df = pd.DataFrame(rows)
    csv = os.path.join(BASE, 'g26_new_asymmetric_metrics.csv')
    df.to_csv(csv, index=False, float_format='%.4f')
    print(f'\n→ Metrics CSV: {csv}')


if __name__ == '__main__':
    main()
