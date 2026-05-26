"""
P10: F10 ε=0.015 + l_max=5.0 — フル9指標計算
=============================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-26)

目的:
  `INTEGRATION_DEBATE_2026-05-26.md` §2 / §3.3 / §3.4 で `—` / `†` 推定値のままに
  なっている F10+lmax5 の以下指標を実測値として埋める:
    - CAGR_OOS (実測)
    - CAGR_IS
    - IS-OOS gap = CAGR_IS - CAGR_OOS (e4_regime_klt と同一符号慣行)
    - Sharpe_OOS
    - MaxDD_FULL
    - Worst10Y★
    - P10_5Y
    - Trades/yr (lev_raw 基準)

実装:
  - `src/g8_wfa_lmax5.py` の F10-eps015-lmax5 と完全に同一の NAV 構築手順を使用
  - `src/e4_regime_klt.py` の calc_all_metrics と完全に同一の指標計算を使用

出力:
  - コンソールに全指標
  - f10lmax5_fullmetrics.csv
"""

import sys
import os
import types

# multitasking スタブ (yfinance 依存回避)
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
    build_nav_strategy, calc_7metrics,
    CFD_SPREAD_LOW, IS_START, IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local, nav_to_annual, rolling_nY_cagr
from long_cycle_signal import build_lt_signal, apply_lt_mode_b

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 定数 (g8_wfa_lmax5.py F10-eps015-lmax5 と完全同一)
# ---------------------------------------------------------------------------
# E4 base
K_LO   = 0.1
K_HI   = 0.8
VZ_THR = 0.70
K_MID  = 0.5
N_LT2  = 750

# F10 tilt (G7 と同一)
TILT_R5            = 10.0
VZ_REG             = 0.70
TILT_CAP_CALM      = 0.15
TILT_CAP_BULL_VZ   = 0.10
TILT_CAP_BEAR_VZ   = 0.05
EPS_F10            = 0.015

# CFD 設定 (l_max=5.0)
S2_LMAX5 = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5,
                n=20, l_min=1.0, l_max=5.0, step=0.5)


# ---------------------------------------------------------------------------
# ε-Deadband (g8_wfa_lmax5.py と同一)
# ---------------------------------------------------------------------------

def compute_tilt_with_deadband(raw_a2, vz, bull_mask, eps):
    n = len(raw_a2)
    cap_eff = np.where(np.abs(vz) < VZ_REG, TILT_CAP_CALM,
              np.where(vz > VZ_REG, TILT_CAP_BULL_VZ, TILT_CAP_BEAR_VZ))

    tilt_raw    = TILT_R5 * (raw_a2 - THRESHOLD) * (1.0 - raw_a2)
    tilt_target = np.minimum(np.maximum(tilt_raw, 0.0), cap_eff)
    tilt_target = np.where(bull_mask, tilt_target, 0.0)

    confirmed = np.zeros(n, dtype=float)
    cur = 0.0
    n_updates = 0
    for i in range(n):
        if i == 0 or abs(tilt_target[i] - cur) >= eps:
            cur = tilt_target[i]
            n_updates += 1
        confirmed[i] = cur
    return confirmed, n_updates


# ---------------------------------------------------------------------------
# P10_5Y (e4_regime_klt.py と同一)
# ---------------------------------------------------------------------------

def compute_p10_5y(nav, td=252):
    s = pd.Series(np.asarray(nav, dtype=float))
    return float(((s / s.shift(td * 5)) ** 0.2 - 1).dropna().quantile(0.10))


# ---------------------------------------------------------------------------
# 全指標計算 (e4_regime_klt.py calc_all_metrics と同一)
# ---------------------------------------------------------------------------

def calc_all_metrics(nav, dates, trades_yr):
    m = calc_7metrics(nav, dates, trades_per_year=trades_yr)
    ann = nav_to_annual(nav, dates)
    r10 = rolling_nY_cagr(ann, 10)
    return {**m,
            'Worst10Y_star': float(r10.min()) if len(r10) > 0 else float('nan'),
            'P10_5Y':        compute_p10_5y(nav.values),
            'IS_OOS_gap':    m['CAGR_IS'] - m['CAGR_OOS']}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 72)
    print('P10: F10 ε=0.015 + l_max=5.0 — フル9指標計算')
    print('=' * 72)
    print(f'  E4 base: k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, k_mid={K_MID}, LT2-N{N_LT2}')
    print(f'  F10 tilt: ε={EPS_F10}, tilt={TILT_R5}, cap=calm{TILT_CAP_CALM}/'
          f'bullVZ{TILT_CAP_BULL_VZ}/bearVZ{TILT_CAP_BEAR_VZ}, vz_reg={VZ_REG}')
    print(f'  CFD: target_vol={S2_LMAX5["target_vol"]}, k_vz={S2_LMAX5["k_vz"]}, '
          f'gate_min={S2_LMAX5["gate_min"]}, n={S2_LMAX5["n"]}, '
          f'l_min={S2_LMAX5["l_min"]}, **l_max={S2_LMAX5["l_max"]}**, step={S2_LMAX5["step"]}')

    # --- Data load
    print('\n[S1] Loading data and shared assets...')
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'  Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n_years:.2f} yr)')

    sofr    = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates,
                                          use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Scenario D assets done.')

    # DH Dyn A シグナル
    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    trades_yr_lev_raw = n_tr / n_years
    print(f'  DH Dyn A: {n_tr} trades, {trades_yr_lev_raw:.2f}/yr (lev_raw 基準)')

    # LT2 シグナル
    lt_sig_raw = build_lt_signal(close, 'LT2', N=N_LT2)
    lt_sig_arr = lt_sig_raw.values
    vz_arr     = vz.values
    print('  LT2 signal done.')

    # E4 Regime k_lt
    k_dyn = np.where(vz_arr >  VZ_THR, K_HI,
             np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias_e4 = pd.Series(
        np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
        index=lt_sig_raw.index,
    )
    lev_mod_e4 = apply_lt_mode_b(lev_raw, lt_bias_e4, l_min=0.0, l_max=1.0)

    # CFD レバレッジ l_max=5.0
    L_s2_lmax5 = compute_L_s2_vz_gated(ret, vz, **S2_LMAX5)
    print('  S2 leverage done (l_max=5.0).')

    # F10 tilt (ε=0.015)
    raw_a2_vals = raw_a2.values
    bull_mask   = raw_a2_vals > THRESHOLD
    tilt_f10, n_updates_f10 = compute_tilt_with_deadband(
        raw_a2_vals, vz_arr, bull_mask, EPS_F10
    )
    wn_f10 = wn_A + tilt_f10
    wb_f10 = np.clip(wb_A - tilt_f10, 0.0, wb_A)
    wg_f10 = wg_A
    print(f'  F10 tilt done (ε={EPS_F10}, updates={n_updates_f10:,}).')

    # --- NAV 構築 (F10-eps015-lmax5)
    print('\n[S2] Building F10+lmax5 NAV...')
    nav = build_nav_strategy(
        close, lev_mod_e4, wn_f10, wg_f10, wb_f10, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD',
        cfd_leverage=L_s2_lmax5.values,
        cfd_spread=CFD_SPREAD_LOW,
    )
    print(f'  NAV range: [{nav.min():.4f}, {nav.max():.4f}]  '
          f'final: {nav.iloc[-1]:.4f}')

    # --- 指標計算
    print('\n[S3] Computing full metrics...')
    m = calc_all_metrics(nav, dates, trades_yr_lev_raw)

    # --- 出力
    print('\n' + '=' * 72)
    print('F10 ε=0.015 + l_max=5.0 — フル指標 (実測値)')
    print('=' * 72)
    print(f'  IS:  {IS_START} ~ {IS_END}')
    print(f'  OOS: {OOS_START} ~ (FULL_END)')
    print('-' * 72)
    print(f'  CAGR_FULL      : {m["CAGR_FULL"]*100:+8.4f}%')
    print(f'  CAGR_IS        : {m["CAGR_IS"]*100:+8.4f}%')
    print(f'  CAGR_OOS       : {m["CAGR_OOS"]*100:+8.4f}%')
    print(f'  IS_OOS_gap     : {m["IS_OOS_gap"]*100:+8.4f} pp  '
          f'(= CAGR_IS - CAGR_OOS;  負値 = OOS > IS = 好ましい)')
    print(f'  Sharpe_FULL    : {m["Sharpe_FULL"]:+8.4f}')
    print(f'  Sharpe_IS      : {m["Sharpe_IS"]:+8.4f}')
    print(f'  Sharpe_OOS     : {m["Sharpe_OOS"]:+8.4f}')
    print(f'  MaxDD_FULL     : {m["MaxDD_FULL"]*100:+8.4f}%')
    print(f'  Worst5Y        : {m["Worst5Y"]*100:+8.4f}%')
    print(f'  Worst10Y       : {m["Worst10Y"]*100:+8.4f}%')
    print(f'  Worst10Y_star  : {m["Worst10Y_star"]*100:+8.4f}%   '
          f'(nav_to_annual + rolling_nY_cagr 10y .min())')
    print(f'  P10_5Y         : {m["P10_5Y"]*100:+8.4f}%')
    print(f'  WinRate        : {m["WinRate"]*100:+8.4f}%')
    print(f'  Trades_yr (raw): {trades_yr_lev_raw:+8.4f}')
    print('=' * 72)

    # --- CSV 保存
    out_row = {
        'strategy':       'F10-eps015-lmax5',
        'CAGR_FULL':      m['CAGR_FULL'],
        'CAGR_IS':        m['CAGR_IS'],
        'CAGR_OOS':       m['CAGR_OOS'],
        'IS_OOS_gap':     m['IS_OOS_gap'],
        'Sharpe_FULL':    m['Sharpe_FULL'],
        'Sharpe_IS':      m['Sharpe_IS'],
        'Sharpe_OOS':     m['Sharpe_OOS'],
        'MaxDD_FULL':     m['MaxDD_FULL'],
        'Worst5Y':        m['Worst5Y'],
        'Worst10Y':       m['Worst10Y'],
        'Worst10Y_star':  m['Worst10Y_star'],
        'P10_5Y':         m['P10_5Y'],
        'WinRate':        m['WinRate'],
        'Trades_yr':      trades_yr_lev_raw,
        'EPS_F10':        EPS_F10,
        'l_max':          S2_LMAX5['l_max'],
    }
    csv_path = os.path.join(BASE, 'f10lmax5_fullmetrics.csv')
    pd.DataFrame([out_row]).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved: {csv_path}')
    print('Done.')


if __name__ == '__main__':
    main()
