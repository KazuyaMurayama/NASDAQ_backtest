"""
gen_f8r5_yearly_returns.py — F8-R5 CALM_BOOST (tilt=10.0 step-func, regime cap) の年次リターン生成
=================================================================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-24)

対象戦略:
  F8-R5 CALM_BOOST bull-tilt (TILT_A=10.0 step-func, calm cap=0.15 / bull_vz cap=0.10 / bear_vz cap=0.05)
  ベース: E4 採用 config (k_lo=0.1, k_hi=0.8, vz_thr=0.7, LT2-N750, mode B)
  → Shortlisted ✅ (G5 WFA PASS 2026-05-24)
  CURRENT_BEST は E4 Regime k_lt（◆）。本戦略は次善候補1。

処理:
  1. F8-R5 CALM_BOOST NAV を構築（E4 base + regime-conditional bull tilt）
  2. 各年の年次リターン (%) を計算
  3. CSV `f8r5_yearly_returns.csv` を出力
  4. サニティチェック: OOS 期間 (2021-2026) の幾何平均が CAGR_OOS +36.83% と概ね一致するか確認

出力:
  - <BASE>/f8r5_yearly_returns.csv (columns: year, annual_return)
  - stdout: 各年リターン、OOS 期間サニティチェック

参考: gen_strategy_comparison.py から呼ばれることも想定。
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr, build_bond_1x_nav_corrected, build_gold_2x, build_bond_3x,
    build_a2_signal, simulate_rebalance_A,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import build_nav_strategy, CFD_SPREAD_LOW
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, apply_lt_mode_b

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 採用 config (CURRENT_BEST_STRATEGY.md 2026-05-24)
# E4 base
K_LO, K_HI, VZ_THR, K_MID = 0.1, 0.8, 0.7, 0.5
S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20,
                l_min=1.0, l_max=7.0, step=0.5)

# F8-R5 CALM_BOOST bull-tilt
TILT_FORMULA  = 'R5_CALM_BOOST'
TILT_A        = 10.0
CALM_CAP      = 0.15   # |vz| < 0.7
BULL_VZ_CAP   = 0.10   # vz > 0.7
BEAR_VZ_CAP   = 0.05   # vz < -0.7

# サニティ参照値
REF_CAGR_OOS = 0.3683   # F8-R5 CAGR_OOS +36.83%
OOS_FIRST_YEAR = 2021
OOS_LAST_YEAR  = 2026


def build_f8r5_nav():
    """F8-R5 CALM_BOOST (tilt=10.0 step-func, regime cap) NAV を構築して返す。

    Returns
    -------
    nav   : pd.Series
    dates : pd.Series
    """
    df = load_data(DATA_PATH)
    close = df['Close']
    ret = close.pct_change().fillna(0)
    dates = df['Date']

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, _ = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)

    # E4 base lev_mod
    lt_sig_raw = build_lt_signal(close, 'LT2', N=750)
    lt_sig_arr = lt_sig_raw.values
    vz_arr = vz.values
    k_dyn = np.where(vz_arr > VZ_THR, K_HI,
                     np.where(vz_arr < -VZ_THR, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * lt_sig_arr * 0.5, -0.5, 0.5),
                        index=lt_sig_raw.index)
    lev_mod = apply_lt_mode_b(lev_raw, lt_bias, l_min=0.0, l_max=1.0)

    # F8-R5 CALM_BOOST step-func + regime-conditional cap
    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    bull_mask = raw_a2_vals > THRESHOLD
    tilt_raw = TILT_A * (raw_a2_vals - THRESHOLD) * (1.0 - raw_a2_vals)
    cap_eff = np.where(np.abs(vz_arr) < VZ_THR, CALM_CAP,
              np.where(vz_arr > VZ_THR, BULL_VZ_CAP, BEAR_VZ_CAP))
    tilt_amount = np.where(bull_mask, np.clip(tilt_raw, 0.0, cap_eff), 0.0)

    wn_tilted = wn_A + tilt_amount
    wb_tilted = np.clip(wb_A - tilt_amount, 0.0, wb_A)
    wg_tilted = wg_A  # unchanged

    nav = build_nav_strategy(
        close, lev_mod, wn_tilted, wg_tilted, wb_tilted, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD', cfd_leverage=L_s2.values, cfd_spread=CFD_SPREAD_LOW,
    )
    return nav, dates


def nav_to_yearly_returns(nav: pd.Series, dates: pd.Series) -> dict:
    """各年の年次リターン (%) を計算。

    年内最初の取引日 NAV → 年内最後の取引日 NAV の変化率を年次リターンとする。
    OOS 開始（2021-05-08）以前/以後の境界に関わらず、暦年ベースで集計する。
    """
    df2 = pd.DataFrame({'nav': nav.values}, index=pd.to_datetime(dates.values))
    yearly = {}
    for year, grp in df2.groupby(df2.index.year):
        nav_start = grp['nav'].iloc[0]
        nav_end = grp['nav'].iloc[-1]
        pct = (nav_end / nav_start - 1) * 100
        yearly[year] = round(float(pct), 1)
    return yearly


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 70)
    print('gen_f8r5_yearly_returns.py — F8-R5 CALM_BOOST 年次リターン生成')
    print(f'  E4 base: k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, k_mid={K_MID}')
    print(f'  F8-R5 {TILT_FORMULA}: tilt_A={TILT_A} step-func, '
          f'calm cap={CALM_CAP}/bullVZ cap={BULL_VZ_CAP}/bearVZ cap={BEAR_VZ_CAP}, '
          f'THRESHOLD={THRESHOLD}')
    print('=' * 70)

    print('Building F8-R5 CALM_BOOST NAV...')
    nav, dates = build_f8r5_nav()
    print(f'NAV built. length={len(nav)}, '
          f'dates={pd.to_datetime(dates.iloc[0]).date()} → {pd.to_datetime(dates.iloc[-1]).date()}')

    yearly = nav_to_yearly_returns(nav, dates)
    years = sorted(yearly.keys())

    # CSV 出力
    out_csv = os.path.join(BASE, 'f8r5_yearly_returns.csv')
    pd.DataFrame({
        'year': years,
        'annual_return': [yearly[y] for y in years],
    }).to_csv(out_csv, index=False, float_format='%.1f')
    print(f'\nSaved: {out_csv}')

    # 全年表示
    print('\n=== F8-R5 CALM_BOOST 年次リターン (1974–2026) ===')
    for y in years:
        tag = ' [OOS]' if y >= OOS_FIRST_YEAR else ''
        print(f'  {y}{tag}: {yearly[y]:+.1f}%')

    # サニティチェック: OOS 期間の幾何平均 vs CAGR_OOS
    print('\n=== サニティチェック ===')
    print(f'  参照: CAGR_OOS (公式) = +{REF_CAGR_OOS*100:.2f}%')
    oos_years = [y for y in years if OOS_FIRST_YEAR <= y <= OOS_LAST_YEAR]
    if oos_years:
        cum = 1.0
        for y in oos_years:
            cum *= (1.0 + yearly[y] / 100.0)
        n_years = len(oos_years)
        geo_mean = cum ** (1.0 / n_years) - 1.0
        diff_pp = (geo_mean - REF_CAGR_OOS) * 100
        print(f'  暦年 OOS (2021-2026, {n_years}年): 幾何平均 = {geo_mean*100:+.2f}%')
        print(f'  diff vs CAGR_OOS = {diff_pp:+.2f}pp '
              f'(暦年基準 vs OOS_START=2021-05-08 のため厳密一致しない。±5pp 内なら OK)')
        ok = abs(diff_pp) <= 5.0
        print(f'  判定: {"OK" if ok else "WARN"}')

    print('\nDone.')
    return yearly


if __name__ == '__main__':
    main()
