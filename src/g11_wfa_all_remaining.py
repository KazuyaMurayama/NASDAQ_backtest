"""
G11: WFA — 14戦略 一括 Walk-Forward Analysis
=============================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-27)

目的:
  STRATEGY_PERFORMANCE_COMPARISON_2026-05-27 v1.7 で WFA 未実施だった
  14戦略について CI95_lo / WFE を一括計算し、21行全 WFA 完了を目指す。

対象戦略 (14本):
  B4-klo0          : discrete k_lt (k_lo=0.0, k_hi=0.7, vz_thr=0.7)
  A1-alpha2/3/5/8  : sigmoid k_lt (alpha=2,3,5,8, K_LO=0.1, K_HI=0.8)
  A2-lmax6vs2      : dynamic lmax (lmax_base=6.0, vol_sens=2.0, VOL_REF=0.20)
  A2B-rolling      : dynamic lmax + rolling VOL_REF (lmax_base=6.0, vol_sens=2.0)
  A3-vov           : VoV+vz dual gate (vov_thr=1.3, alpha_max=0.2)
  C2-adaptive      : adaptive deadband (eps_0=0.020, sigma_lookback=250)
  C3-yz            : Yang-Zhang vol (yz_n=10, vz_thr=0.7)
  D5-vz060-lmax45/50, D5-vz065-lmax45, D5-vz070-lmax50

WFAインフラ: g10_wfa_vz065_lmax_row.py の generate_windows / compute_window_metrics /
            compute_summary_stats / evaluate_criteria を流用（窓設計完全同一）。

出力:
  - g11_wfa_all_remaining_per_window.csv
  - g11_wfa_all_remaining_summary.csv
"""

import sys
import os
import types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    CFD_SPREAD_LOW,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, apply_lt_mode_b

# G10 WFAインフラを流用
from g10_wfa_vz065_lmax_row import (
    generate_windows,
    compute_window_metrics,
    compute_summary_stats,
    evaluate_criteria,
    WINDOW_DAYS, STEP_DAYS, EVAL_START, EVAL_END,
    IS_END_REF, OOS_START_REF,
)

# A2 dynamic lmax 関数（共通インフラ）
from a2_dyn_lmax import compute_L_s2_dyn_lmax

# A3 VoV computation
from a3_regime_asset_tilt import compute_vov_zscore, apply_regime_tilt

# C2 adaptive deadband helpers
from c2_adaptive_deadband import (
    compute_tilt_with_adaptive_deadband,
    compute_tilt_with_deadband,
    count_trades_tilted,
    TILT as C2_TILT, VZ_REG as C2_VZ_REG,
    CAP_CALM as C2_CAP_CALM, CAP_BULL as C2_CAP_BULL, CAP_BEAR as C2_CAP_BEAR,
    SIGMA_LOOKBACK as C2_SIGMA_LOOKBACK, VOL_WINDOW as C2_VOL_WINDOW,
)

# C3 Yang-Zhang
from c3_yang_zhang_vol import build_vz_from_yz, load_ohlc, OHLC_PATH

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
K_LO_E4   = 0.1
K_HI_E4   = 0.8
VZ_THR_E4 = 0.70
K_MID     = 0.5
N_LT2     = 750

S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
               l_min=1.0, step=0.5)

TODAY = '2026-05-27'

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 共有資産ロード（全14戦略で使い回す）
# ---------------------------------------------------------------------------

def load_shared_assets():
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'  Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n_years:.2f} yr)')

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Scenario D assets done.')

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    print(f'  DH Dyn A: {n_tr} trades ({n_tr/n_years:.1f}/yr)')

    lt_sig_raw = build_lt_signal(close, 'LT2', N=N_LT2)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values

    return dict(
        close=close, ret=ret, dates=dates, n=n, n_years=n_years, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        raw_a2=raw_a2, vz=vz, vz_arr=vz_arr,
        lev_raw=lev_raw, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        lt_sig_raw=lt_sig_raw, lt_sig_arr=lt_sig_arr,
    )


# ---------------------------------------------------------------------------
# 戦略 NAV ビルダ
# ---------------------------------------------------------------------------

def _make_nav(close, lev_mod, wn, wg, wb, dates, gold_2x, bond_3x, sofr,
              L_s2_values):
    return build_nav_strategy(
        close, lev_mod, wn, wg, wb, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2_values,
        cfd_spread=CFD_SPREAD_LOW,
    )


def _e4_like_klt(a, k_lo, k_hi, vz_thr, l_max):
    """E4 風 discrete k_lt + S2 vz_gated leverage で NAV を構築。"""
    vz_arr = a['vz_arr']
    k_dyn = np.where(vz_arr >  vz_thr, k_hi,
              np.where(vz_arr < -vz_thr, k_lo, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)

    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **S2_BASE, l_max=l_max)
    nav = _make_nav(a['close'], lev_mod, a['wn_A'], a['wg_A'], a['wb_A'],
                    a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                    L_s2.values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_b4_klo0(a):
    return _e4_like_klt(a, k_lo=0.0, k_hi=0.7, vz_thr=0.7, l_max=7.0)


def build_d5(a, vz_thr, l_max):
    """D5 グリッドの各点（E4と同じ k_lo=0.1, k_hi=0.8 で vz_thr / l_max のみ変更）"""
    return _e4_like_klt(a, k_lo=K_LO_E4, k_hi=K_HI_E4, vz_thr=vz_thr, l_max=l_max)


def build_a1(a, alpha):
    """A1 sigmoid k_lt"""
    vz_arr = a['vz_arr']
    x = np.clip(alpha * vz_arr, -500.0, 500.0)
    k_arr = K_LO_E4 + (K_HI_E4 - K_LO_E4) / (1.0 + np.exp(-x))
    lt_bias = pd.Series(np.clip(-k_arr * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)

    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **S2_BASE, l_max=7.0)
    nav = _make_nav(a['close'], lev_mod, a['wn_A'], a['wg_A'], a['wb_A'],
                    a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                    L_s2.values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_a2(a, lmax_base=6.0, vol_sens=2.0, vol_ref=0.20):
    """A2: 252日vol 連動 dynamic l_max + E4 離散 k_lt (vz_thr=0.7)"""
    ret = a['ret']

    vol252 = ret.rolling(252, min_periods=126).std() * np.sqrt(TRADING_DAYS)
    vol252_exp = ret.expanding(min_periods=30).std() * np.sqrt(TRADING_DAYS)
    vol252 = vol252.fillna(vol252_exp)

    L_FLOOR, L_CEIL = 4.5, 6.5
    l_max_t = (lmax_base - vol_sens * (vol252 / vol_ref - 1)).clip(L_FLOOR, L_CEIL)
    l_max_t = l_max_t.reindex(ret.index).fillna(lmax_base)

    L_s2 = compute_L_s2_dyn_lmax(ret, a['vz'], l_max_t, **S2_BASE)

    # k_lt は E4 と同じ離散
    vz_arr = a['vz_arr']
    k_dyn = np.where(vz_arr >  VZ_THR_E4, K_HI_E4,
              np.where(vz_arr < -VZ_THR_E4, K_LO_E4, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)

    nav = _make_nav(a['close'], lev_mod, a['wn_A'], a['wg_A'], a['wb_A'],
                    a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                    L_s2.values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_a2b(a, lmax_base=6.0, vol_sens=2.0):
    """A2B: A2 と同じだが VOL_REF を 10年 rolling median に置換"""
    ret = a['ret']

    vol252 = ret.rolling(252, min_periods=126).std() * np.sqrt(TRADING_DAYS)
    vol252_exp = ret.expanding(min_periods=30).std() * np.sqrt(TRADING_DAYS)
    vol252 = vol252.fillna(vol252_exp)

    vol_ref_t = vol252.rolling(2520, min_periods=504).median()
    vol_ref_t = vol_ref_t.fillna(vol252.expanding(min_periods=5).median())
    vol_ref_t = vol_ref_t.fillna(vol252)
    vol_ref_t = vol_ref_t.clip(lower=0.05)

    L_FLOOR, L_CEIL = 4.5, 6.5
    l_max_t = (lmax_base - vol_sens * (vol252 / vol_ref_t - 1)).clip(L_FLOOR, L_CEIL)
    l_max_t = l_max_t.reindex(ret.index).fillna(lmax_base)

    L_s2 = compute_L_s2_dyn_lmax(ret, a['vz'], l_max_t, **S2_BASE)

    vz_arr = a['vz_arr']
    k_dyn = np.where(vz_arr >  VZ_THR_E4, K_HI_E4,
              np.where(vz_arr < -VZ_THR_E4, K_LO_E4, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)

    nav = _make_nav(a['close'], lev_mod, a['wn_A'], a['wg_A'], a['wb_A'],
                    a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                    L_s2.values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_a3(a, vov_thr=1.3, alpha_max=0.2):
    """A3: E4 base + VoV+vz dual gate でストレス時 NASDAQ→Gold/Bond シフト"""
    ret = a['ret']
    vz_arr = a['vz_arr']

    # E4 k_lt
    k_dyn = np.where(vz_arr >  VZ_THR_E4, K_HI_E4,
              np.where(vz_arr < -VZ_THR_E4, K_LO_E4, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)

    # E4 と同じ l_max=7.0 固定
    L_s2 = compute_L_s2_vz_gated(ret, a['vz'], **S2_BASE, l_max=7.0)

    # VoV z-score + ストレス判定
    vov_z = compute_vov_zscore(ret)
    vov_arr = vov_z.values
    vz_pos = (a['vz'].fillna(0).values > VZ_THR_E4)
    stress = (vov_arr > vov_thr) & vz_pos

    # NASDAQ → Gold/Bond シフト
    wn_p, wg_p, wb_p = apply_regime_tilt(a['wn_A'], a['wg_A'], a['wb_A'],
                                          stress, alpha_max)

    nav = _make_nav(a['close'], lev_mod, wn_p, wg_p, wb_p,
                    a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                    L_s2.values)
    # trade-count 用に元 wn/wb を返す（asset tilt は連続変化）
    return nav, wn_p, wb_p, np.asarray(a['lev_raw'])


def build_c2(a, eps_0=0.020):
    """C2: F10 系 + adaptive deadband (eps_t = eps_0 × vol_ratio)"""
    ret = a['ret']
    vz_arr = a['vz_arr']

    # E4 base
    k_dyn = np.where(vz_arr >  VZ_THR_E4, K_HI_E4,
              np.where(vz_arr < -VZ_THR_E4, K_LO_E4, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    lev_raw_arr = np.asarray(a['lev_raw'])

    L_s2 = compute_L_s2_vz_gated(ret, a['vz'], **S2_BASE, l_max=7.0)

    raw_a2 = a['raw_a2']
    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    bull_mask   = raw_a2_vals > THRESHOLD

    # vol_ratio: 20日vol / 250日平均
    vol_20d  = ret.rolling(C2_VOL_WINDOW).std() * np.sqrt(252)
    vol_mean = vol_20d.rolling(C2_SIGMA_LOOKBACK).mean()
    vol_ratio = (vol_20d / vol_mean).fillna(1.0).clip(0.3, 3.0)
    vol_ratio_vals = vol_ratio.values

    eps_t = eps_0 * vol_ratio_vals

    tilt_confirmed, _n_upd = compute_tilt_with_adaptive_deadband(
        raw_a2_vals, vz_arr, bull_mask, eps_t,
    )

    wn_tilted = a['wn_A'] + tilt_confirmed
    wb_tilted = np.clip(a['wb_A'] - tilt_confirmed, 0.0, a['wb_A'])

    nav = _make_nav(a['close'], lev_mod, wn_tilted, a['wg_A'], wb_tilted,
                    a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                    L_s2.values)
    return nav, wn_tilted, wb_tilted, lev_raw_arr


def build_c3(a, yz_n=10, vz_thr=0.7):
    """C3: Yang-Zhang vz （cc-vol を YZ-vol に置換した E4 ベスト相当）"""
    close = a['close']
    ret   = a['ret']
    dates = a['dates']

    df_ohlc = load_ohlc(OHLC_PATH)
    vz_yz = build_vz_from_yz(close, df_ohlc, dates, yz_n=yz_n)

    raw_a2_yz, _ = build_a2_signal(close, ret)
    lev_raw_yz, wn_A_yz, wg_A_yz, wb_A_yz, _ = simulate_rebalance_A(
        raw_a2_yz, vz_yz, THRESHOLD,
    )
    L_s2 = compute_L_s2_vz_gated(ret, vz_yz, **S2_BASE, l_max=7.0)

    vz_yz_arr = vz_yz.values
    k_dyn = np.where(vz_yz_arr >  vz_thr, K_HI_E4,
              np.where(vz_yz_arr < -vz_thr, K_LO_E4, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(lev_raw_yz, lt_bias, l_min=0.0, l_max=1.0)

    nav = _make_nav(close, lev_mod, wn_A_yz, wg_A_yz, wb_A_yz,
                    dates, a['gold_2x'], a['bond_3x'], a['sofr'],
                    L_s2.values)
    return nav, wn_A_yz, wb_A_yz, np.asarray(lev_raw_yz)


# ---------------------------------------------------------------------------
# 戦略リスト
# ---------------------------------------------------------------------------

STRATEGY_BUILDERS = [
    ('B4-klo0',             lambda a: build_b4_klo0(a)),
    ('A1-alpha2',           lambda a: build_a1(a, alpha=2.0)),
    ('A1-alpha3',           lambda a: build_a1(a, alpha=3.0)),
    ('A1-alpha5',           lambda a: build_a1(a, alpha=5.0)),
    ('A1-alpha8',           lambda a: build_a1(a, alpha=8.0)),
    ('A2-lmax6vs2',         lambda a: build_a2(a, lmax_base=6.0, vol_sens=2.0)),
    ('A2B-rolling',         lambda a: build_a2b(a, lmax_base=6.0, vol_sens=2.0)),
    ('A3-vov',              lambda a: build_a3(a, vov_thr=1.3, alpha_max=0.2)),
    ('C2-adaptive',         lambda a: build_c2(a, eps_0=0.020)),
    ('C3-yz',               lambda a: build_c3(a, yz_n=10, vz_thr=0.7)),
    ('D5-vz060-lmax45',     lambda a: build_d5(a, vz_thr=0.60, l_max=4.5)),
    ('D5-vz060-lmax50',     lambda a: build_d5(a, vz_thr=0.60, l_max=5.0)),
    ('D5-vz065-lmax45',     lambda a: build_d5(a, vz_thr=0.65, l_max=4.5)),
    ('D5-vz070-lmax50',     lambda a: build_d5(a, vz_thr=0.70, l_max=5.0)),
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 72)
    print('G11: WFA — 14戦略 一括 Walk-Forward Analysis')
    print(f'実行日: {TODAY}')
    print('=' * 72)

    print('\n[S1] Loading shared assets...')
    assets = load_shared_assets()
    dates  = assets['dates']

    print('\n[S2] Generating windows (G10 と同一設計)...')
    windows = generate_windows(dates)
    print(f'  {len(windows)} windows: '
          f'{windows[0]["start_date"].date()} ... {windows[-1]["end_date"].date()}')
    short_wins = sum(w['short_flag'] for w in windows)
    print(f'  Short windows: {short_wins}')

    all_pw_rows = []
    all_sm_rows = []
    results = {}

    print('\n[S3] Building NAV + WFA evaluation for 14 strategies...')
    print('-' * 72)
    for sid, builder in STRATEGY_BUILDERS:
        print(f'  [{sid}] building NAV ...', end='', flush=True)
        try:
            nav, wn, wb, lev_arr = builder(assets)
        except Exception as exc:
            print(f' FAILED: {exc}')
            raise

        # Per-window metrics
        per_rows = []
        for w in windows:
            m = compute_window_metrics(nav, w, wn=wn, wb=wb, lev_arr=lev_arr)
            m.update(dict(
                strategy=sid, window_id=w['window_id'],
                start_date=w['start_date'], end_date=w['end_date'],
                short_flag=w['short_flag'],
            ))
            per_rows.append(m)
            all_pw_rows.append(m)

        per_df  = pd.DataFrame(per_rows)
        summary = compute_summary_stats(per_df)
        verdict, crits = evaluate_criteria(summary)

        results[sid] = dict(per_window=per_df, summary=summary,
                            verdict=verdict, criteria=crits)

        sm_row = {'strategy': sid, 'verdict': verdict,
                  **summary, **{f'crit_{k}': v for k, v in crits.items()}}
        all_sm_rows.append(sm_row)

        ci_lo  = summary.get('WFA_CI95_lo', np.nan)
        ci_hi  = summary.get('WFA_CI95_hi', np.nan)
        wfe    = summary.get('WFA_WFE', np.nan)
        mean_c = summary.get('mean_CAGR', np.nan)
        tp     = summary.get('t_pvalue', np.nan)
        print(f'  mean_CAGR={mean_c*100:+.2f}%  '
              f'CI95=[{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]  '
              f't_p={tp:.4f}  WFE={wfe:+.3f}  => {verdict}')

    print('-' * 72)

    print('\n[S4] Summary:')
    print('=' * 100)
    print(f'{"Strategy":<22} {"mean_CAGR":>10} {"CI95_lo":>9} {"CI95_hi":>9}'
          f' {"t_p":>9} {"WFE":>7} {"Verdict":>8}')
    print('-' * 100)
    for sid, _ in STRATEGY_BUILDERS:
        s = results[sid]['summary']
        v = results[sid]['verdict']
        print(
            f'{sid:<22}'
            f' {s.get("mean_CAGR", 0)*100:>+9.2f}%'
            f' {s.get("WFA_CI95_lo", np.nan)*100:>+8.2f}%'
            f' {s.get("WFA_CI95_hi", np.nan)*100:>+8.2f}%'
            f' {s.get("t_pvalue", np.nan):>9.6f}'
            f' {s.get("WFA_WFE", np.nan):>7.3f}'
            f' {v:>8}'
        )
    print('=' * 100)

    print('\n[S5] Saving CSVs...')
    pw_path = os.path.join(BASE, 'g11_wfa_all_remaining_per_window.csv')
    sm_path = os.path.join(BASE, 'g11_wfa_all_remaining_summary.csv')
    pd.DataFrame(all_pw_rows).to_csv(pw_path, index=False, float_format='%.6f')
    pd.DataFrame(all_sm_rows).to_csv(sm_path, index=False, float_format='%.6f')
    print(f'  Per-window: {pw_path}')
    print(f'  Summary:    {sm_path}')

    print('\nDone.')
    return results


if __name__ == '__main__':
    main()
