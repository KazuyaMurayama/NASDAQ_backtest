"""
G14: WFA — SBI店頭CFD スプレッド (3.0%/yr) で全 26 戦略再評価
=================================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-29)

目的:
  STRATEGY_PERFORMANCE_COMPARISON_20260529.md (v3) の Phase 2 TODO として、
  g3/g7/g8/g10/g11 で計算済みの 26 戦略を SBI店頭CFD スプレッド
  (SBI_CFD_SPREAD = 0.0300 = 3.0%/yr, IG基準0.50% + 推定マージン2.5%) で
  WFA 再評価し、CI95_lo / WFE / Trades_yr を取得する。

旧スプレッド (CFD_SPREAD_LOW = 0.0020 = 0.20%/yr くりっく株365) との差分を
v3 MD の CI95_lo / Overfit(WFE) 列に反映する。

設計方針: 全 26 戦略を本ファイル内で inline 再実装。副作用なし。
       既存 g3/g7/g8/g10/g11 の CSV は上書きしない。

対象戦略 (26本):
  [g3]  E4-RegimeKLT, REF-N750
  [g7]  F10-eps015, REF-F8R5-eps0
  [g8]  REF-E4, E4-lmax5, F10-eps015-lmax5
  [g10] vz065-lmax5, vz065-lmax5p5, vz065-lmax6, vz065-lmax7
  [g11] B4-klo0, A1-alpha2/3/5/8, A2-lmax6vs2, A2B-rolling, A3-vov,
        C2-adaptive, C3-yz, D5-vz060-lmax45/50, D5-vz065-lmax45, D5-vz070-lmax50

窓設計 (G3/G7/G8/G10/G11 と完全同一):
  - 評価開始: 1977-01-03 / 評価終了: 2026-03-26
  - 窓長: 252営業日 / ステップ: 252営業日 (非重複, カレンダー年アンカー)

判定基準 (EVALUATION_STANDARD v1.1):
  α: WFA_CI95_lo > 0 AND t_p < 0.05
  β: WFA_WFE ∈ [0.5, 2.0]
  総合: α+β → PASS / α のみ → WARN / α FAIL → FAIL

出力:
  - g14_wfa_sbi_cfd_per_window.csv
  - g14_wfa_sbi_cfd_summary.csv
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
from scipy import stats

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
    IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b
from a2_dyn_lmax import compute_L_s2_dyn_lmax
from a3_regime_asset_tilt import compute_vov_zscore, apply_regime_tilt
from c2_adaptive_deadband import (
    compute_tilt_with_adaptive_deadband,
    SIGMA_LOOKBACK as C2_SIGMA_LOOKBACK,
    VOL_WINDOW as C2_VOL_WINDOW,
)
from c3_yang_zhang_vol import build_vz_from_yz, load_ohlc, OHLC_PATH

# ---------------------------------------------------------------------------
# Phase 2 重要変更: SBI店頭CFD スプレッド (3.0%/yr)
# ---------------------------------------------------------------------------
SBI_CFD_SPREAD = 0.0300

# ---------------------------------------------------------------------------
# 定数 (g3/g7/g8/g10/g11 と完全同一)
# ---------------------------------------------------------------------------
K_LO       = 0.1
K_HI       = 0.8
VZ_THR_E4  = 0.70
VZ_THR_065 = 0.65
K_MID      = 0.5
K_REF_N750 = 0.5
N_LT2      = 750

# F10 / F8-R5 tilt パラメータ (g7/g8 と同一)
TILT_R5            = 10.0
VZ_REG             = 0.70
TILT_CAP_CALM      = 0.15
TILT_CAP_BULL_VZ   = 0.10
TILT_CAP_BEAR_VZ   = 0.05
EPS_F10            = 0.015
EPS_REF            = 0.000

# S2 CFD レバ設定
S2_BASE = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, step=0.5)
S2_LMAX5   = {**S2_BASE, 'l_max': 5.0}
S2_LMAX5P5 = {**S2_BASE, 'l_max': 5.5}
S2_LMAX6   = {**S2_BASE, 'l_max': 6.0}
S2_LMAX7   = {**S2_BASE, 'l_max': 7.0}

# WFA 窓設計
WINDOW_DAYS   = 252
STEP_DAYS     = 252
EVAL_START    = '1977-01-03'
EVAL_END      = '2026-03-26'
IS_END_REF    = IS_END
OOS_START_REF = OOS_START
TODAY         = '2026-05-29'

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# F10 / F8-R5 ε-Deadband (g7/g8 と完全同一)
# ---------------------------------------------------------------------------

def compute_tilt_with_deadband(raw_a2, vz, bull_mask, eps):
    """F8-R5 (CALM_BOOST) cap_eff + ε-デッドバンドで確定 tilt 系列を返す。"""
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


def count_trades_in_window(wn, wb, lev_arr, s, e):
    n_tr = 0
    for i in range(s + 1, e):
        if (wn[i] != wn[i-1] or wb[i] != wb[i-1] or lev_arr[i] != lev_arr[i-1]):
            n_tr += 1
    return n_tr


# ---------------------------------------------------------------------------
# Window 生成 / 窓内指標 / 統計集計 / 判定 (g10 と完全同一)
# ---------------------------------------------------------------------------

def generate_windows(dates, eval_start=EVAL_START, eval_end=EVAL_END,
                     window_days=WINDOW_DAYS, step_days=STEP_DAYS):
    eval_start_ts = pd.Timestamp(eval_start)
    eval_end_ts   = pd.Timestamp(eval_end)
    first_year    = eval_start_ts.year
    last_year     = eval_end_ts.year

    windows = []
    for year in range(first_year, last_year + 1):
        yr_start_ts = max(pd.Timestamp(f'{year}-01-01'), eval_start_ts)
        yr_end_ts   = min(pd.Timestamp(f'{year}-12-31'), eval_end_ts)

        mask_s = dates >= yr_start_ts
        if not mask_s.any():
            break
        s_idx = int(mask_s.values.argmax())

        mask_e = dates <= yr_end_ts
        if not mask_e.any():
            break
        e_idx = int(np.where(mask_e.values)[0][-1])

        if e_idx < s_idx:
            break

        n_days = e_idx - s_idx + 1
        windows.append(dict(
            window_id=year, start_idx=s_idx, end_idx=e_idx,
            start_date=dates.iloc[s_idx], end_date=dates.iloc[e_idx],
            n_days=n_days, short_flag=n_days < int(window_days * 0.8),
        ))
    return windows


def compute_window_metrics(nav, window, wn=None, wb=None, lev_arr=None,
                           trading_days=TRADING_DAYS):
    s = window['start_idx']
    e = window['end_idx'] + 1
    nav_arr = nav.iloc[s:e].values.astype(float)
    n_days  = len(nav_arr)

    if n_days < 5 or nav_arr[0] <= 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, MaxDD=np.nan,
                    Vol=np.nan, PosDay_pct=np.nan, n_days=n_days, Trades_yr=np.nan)

    daily_ret = np.diff(nav_arr) / nav_arr[:-1]
    years     = n_days / trading_days
    cagr      = (nav_arr[-1] / nav_arr[0]) ** (1.0 / years) - 1
    r_std     = np.std(daily_ret, ddof=1)
    sharpe    = (np.mean(daily_ret) / r_std * np.sqrt(trading_days)
                 if r_std > 1e-10 else np.nan)
    running_max = np.maximum.accumulate(nav_arr)
    max_dd  = float((nav_arr / running_max - 1).min())
    vol     = float(r_std * np.sqrt(trading_days))
    pos_pct = float(np.mean(daily_ret > 0))

    trades_yr = np.nan
    if wn is not None and wb is not None and lev_arr is not None:
        n_tr = count_trades_in_window(wn, wb, lev_arr, s, e)
        trades_yr = n_tr / years if years > 0 else np.nan

    return dict(CAGR=float(cagr), Sharpe=float(sharpe), MaxDD=max_dd,
                Vol=vol, PosDay_pct=pos_pct, n_days=n_days,
                Trades_yr=float(trades_yr) if not np.isnan(trades_yr) else np.nan)


def compute_summary_stats(per_df):
    valid   = per_df[~per_df['short_flag']].copy()
    cagrs   = valid['CAGR'].dropna().values
    sharpes = valid['Sharpe'].dropna().values
    n       = len(cagrs)

    if n == 0:
        return {}

    mean_c = float(np.mean(cagrs))
    med_c  = float(np.median(cagrs))
    std_c  = float(np.std(cagrs, ddof=1)) if n > 1 else np.nan
    se     = std_c / np.sqrt(n) if (not np.isnan(std_c) and n > 1) else np.nan
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else np.nan

    ci95_lo = mean_c - t_crit * se if not np.isnan(se) else np.nan
    ci95_hi = mean_c + t_crit * se if not np.isnan(se) else np.nan
    t_stat  = mean_c / se if (not np.isnan(se) and se > 0) else np.nan
    t_pval  = float(stats.t.sf(t_stat, df=n - 1)) if not np.isnan(t_stat) else np.nan

    mean_s = float(np.mean(sharpes)) if len(sharpes) > 0 else np.nan
    std_s  = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else np.nan

    oos_start_ts = pd.Timestamp(OOS_START_REF)
    is_mask   = valid['start_date'] < oos_start_ts
    post_mask = valid['start_date'] >= oos_start_ts
    is_cagrs   = valid.loc[is_mask,   'CAGR'].dropna().values
    post_cagrs = valid.loc[post_mask, 'CAGR'].dropna().values
    mean_is   = float(np.mean(is_cagrs))   if len(is_cagrs)   > 0 else np.nan
    mean_post = float(np.mean(post_cagrs)) if len(post_cagrs) > 0 else np.nan
    wfe_post  = (mean_post / mean_is) if (not np.isnan(mean_is) and mean_is != 0) else np.nan

    tr = valid['Trades_yr'].dropna().values if 'Trades_yr' in valid.columns else []
    mean_tr = float(np.mean(tr)) if len(tr) > 0 else np.nan

    return dict(
        n_windows=n, mean_CAGR=mean_c, median_CAGR=med_c, std_CAGR=std_c,
        min_CAGR=float(np.min(cagrs)), max_CAGR=float(np.max(cagrs)),
        P05_CAGR=float(np.percentile(cagrs, 5)),
        P25_CAGR=float(np.percentile(cagrs, 25)),
        P75_CAGR=float(np.percentile(cagrs, 75)),
        P95_CAGR=float(np.percentile(cagrs, 95)),
        WFA_CI95_lo=ci95_lo, WFA_CI95_hi=ci95_hi,
        t_stat=t_stat, t_pvalue=t_pval,
        mean_Sharpe=mean_s, std_Sharpe=std_s,
        mean_CAGR_IS=mean_is, mean_CAGR_postIS=mean_post, WFA_WFE=wfe_post,
        n_windows_IS=len(is_cagrs), n_windows_postIS=len(post_cagrs),
        mean_Trades_yr=mean_tr,
    )


def evaluate_criteria(summary):
    crit_alpha = (summary.get('WFA_CI95_lo', -1) > 0 and
                  summary.get('t_pvalue', 1.0) < 0.05)
    wfe    = summary.get('WFA_WFE', np.nan)
    n_post = summary.get('n_windows_postIS', 0)
    if n_post < 3:
        crit_beta = True
    else:
        crit_beta = (not np.isnan(wfe)) and (0.5 <= wfe <= 2.0)

    crits = dict(alpha=crit_alpha, beta=crit_beta)
    verdict = 'PASS' if (crit_alpha and crit_beta) else ('WARN' if crit_alpha else 'FAIL')
    return verdict, crits


# ---------------------------------------------------------------------------
# 共有資産ロード (g3+g7+g8+g10+g11 の全要求を満たすマスタ)
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
    print('  LT2 signal done.')

    # E4 Regime k_lt (vz_thr=0.70)
    k_dyn_e4 = np.where(vz_arr >  VZ_THR_E4, K_HI,
                np.where(vz_arr < -VZ_THR_E4, K_LO, K_MID))
    lt_bias_e4 = pd.Series(np.clip(-k_dyn_e4 * lt_sig_arr * 0.5, -0.5, 0.5),
                           index=lt_sig_raw.index)
    lev_mod_e4 = apply_lt_mode_b(lev_raw, lt_bias_e4, l_min=0.0, l_max=1.0)

    # vz065 Regime (vz_thr=0.65)
    k_dyn_065 = np.where(vz_arr >  VZ_THR_065, K_HI,
                 np.where(vz_arr < -VZ_THR_065, K_LO, K_MID))
    lt_bias_065 = pd.Series(np.clip(-k_dyn_065 * lt_sig_arr * 0.5, -0.5, 0.5),
                            index=lt_sig_raw.index)
    lev_mod_065 = apply_lt_mode_b(lev_raw, lt_bias_065, l_min=0.0, l_max=1.0)

    # REF-N750 (fixed k=0.5)
    lt_bias_ref = signal_to_bias(lt_sig_raw, k_lt=K_REF_N750)
    lev_mod_ref = apply_lt_mode_b(lev_raw, lt_bias_ref, l_min=0.0, l_max=1.0)

    # CFD レバ (l_max 4種)
    L_s2_lmax7   = compute_L_s2_vz_gated(ret, vz, **S2_LMAX7)
    L_s2_lmax5   = compute_L_s2_vz_gated(ret, vz, **S2_LMAX5)
    L_s2_lmax5p5 = compute_L_s2_vz_gated(ret, vz, **S2_LMAX5P5)
    L_s2_lmax6   = compute_L_s2_vz_gated(ret, vz, **S2_LMAX6)
    print('  S2 leverage done (l_max=5.0, 5.5, 6.0, 7.0).')

    # F10 tilt (ε=0.015)
    raw_a2_vals = raw_a2.values
    bull_mask   = raw_a2_vals > THRESHOLD
    tilt_f10, _ = compute_tilt_with_deadband(raw_a2_vals, vz_arr, bull_mask, EPS_F10)
    wn_f10 = wn_A + tilt_f10
    wb_f10 = np.clip(wb_A - tilt_f10, 0.0, wb_A)
    wg_f10 = wg_A

    # F8-R5 REF tilt (ε=0)
    tilt_ref, _ = compute_tilt_with_deadband(raw_a2_vals, vz_arr, bull_mask, EPS_REF)
    wn_ref_f8 = wn_A + tilt_ref
    wb_ref_f8 = np.clip(wb_A - tilt_ref, 0.0, wb_A)
    wg_ref_f8 = wg_A
    print(f'  F10/F8-R5 tilt done.')

    return dict(
        close=close, ret=ret, dates=dates, n=n, n_years=n_years, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        raw_a2=raw_a2, vz=vz, vz_arr=vz_arr,
        lev_raw=lev_raw, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        lt_sig_raw=lt_sig_raw, lt_sig_arr=lt_sig_arr,
        lev_mod_e4=lev_mod_e4, lev_mod_065=lev_mod_065, lev_mod_ref=lev_mod_ref,
        wn_f10=wn_f10, wg_f10=wg_f10, wb_f10=wb_f10,
        wn_ref_f8=wn_ref_f8, wg_ref_f8=wg_ref_f8, wb_ref_f8=wb_ref_f8,
        L_s2_lmax5=L_s2_lmax5, L_s2_lmax5p5=L_s2_lmax5p5,
        L_s2_lmax6=L_s2_lmax6, L_s2_lmax7=L_s2_lmax7,
    )


# ---------------------------------------------------------------------------
# NAV ビルダ共通ヘルパ (SBI_CFD_SPREAD 固定)
# ---------------------------------------------------------------------------

def _make_nav(close, lev_mod, wn, wg, wb, dates, gold_2x, bond_3x, sofr,
              L_s2_values):
    return build_nav_strategy(
        close, lev_mod, wn, wg, wb, dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD',
        cfd_leverage=L_s2_values,
        cfd_spread=SBI_CFD_SPREAD,
    )


# ---------------------------------------------------------------------------
# 戦略ビルダ (26本, 各々 (nav, wn, wb, lev_arr) を返す)
# ---------------------------------------------------------------------------

# [g3] E4-RegimeKLT (Active 確定戦略)
def build_g3_e4_regimeklt(a):
    nav = _make_nav(a['close'], a['lev_mod_e4'], a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax7'].values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


# [g3] REF-N750 (固定 k=0.5, サニティ)
def build_g3_ref_n750(a):
    nav = _make_nav(a['close'], a['lev_mod_ref'], a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax7'].values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


# [g7] F10-eps015 (F8-R5 + ε=0.015 deadband)
def build_g7_f10_eps015(a):
    nav = _make_nav(a['close'], a['lev_mod_e4'], a['wn_f10'], a['wg_f10'], a['wb_f10'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax7'].values)
    return nav, a['wn_f10'], a['wb_f10'], np.asarray(a['lev_raw'])


# [g7] REF-F8R5-eps0 (G5 と完全同一)
def build_g7_ref_f8r5_eps0(a):
    nav = _make_nav(a['close'], a['lev_mod_e4'], a['wn_ref_f8'], a['wg_ref_f8'],
                     a['wb_ref_f8'], a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax7'].values)
    return nav, a['wn_ref_f8'], a['wb_ref_f8'], np.asarray(a['lev_raw'])


# [g8] REF-E4 (g3 と同設定, サニティ)
def build_g8_ref_e4(a):
    return build_g3_e4_regimeklt(a)


# [g8] E4-lmax5
def build_g8_e4_lmax5(a):
    nav = _make_nav(a['close'], a['lev_mod_e4'], a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax5'].values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


# [g8] F10-eps015-lmax5
def build_g8_f10_eps015_lmax5(a):
    nav = _make_nav(a['close'], a['lev_mod_e4'], a['wn_f10'], a['wg_f10'], a['wb_f10'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax5'].values)
    return nav, a['wn_f10'], a['wb_f10'], np.asarray(a['lev_raw'])


# [g10] vz065 × l_max 4種
def build_g10_vz065_lmax5(a):
    nav = _make_nav(a['close'], a['lev_mod_065'], a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax5'].values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_g10_vz065_lmax5p5(a):
    nav = _make_nav(a['close'], a['lev_mod_065'], a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax5p5'].values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_g10_vz065_lmax6(a):
    nav = _make_nav(a['close'], a['lev_mod_065'], a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax6'].values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_g10_vz065_lmax7(a):
    nav = _make_nav(a['close'], a['lev_mod_065'], a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     a['L_s2_lmax7'].values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


# [g11] 共通ヘルパ
def _e4_like_klt(a, k_lo, k_hi, vz_thr, l_max):
    vz_arr = a['vz_arr']
    k_dyn = np.where(vz_arr >  vz_thr, k_hi,
              np.where(vz_arr < -vz_thr, k_lo, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **{**S2_BASE, 'l_max': l_max})
    nav = _make_nav(a['close'], lev_mod, a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     L_s2.values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_g11_b4_klo0(a):
    return _e4_like_klt(a, k_lo=0.0, k_hi=0.7, vz_thr=0.7, l_max=7.0)


def build_g11_a1(a, alpha):
    vz_arr = a['vz_arr']
    x = np.clip(alpha * vz_arr, -500.0, 500.0)
    k_arr = K_LO + (K_HI - K_LO) / (1.0 + np.exp(-x))
    lt_bias = pd.Series(np.clip(-k_arr * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    L_s2 = compute_L_s2_vz_gated(a['ret'], a['vz'], **S2_LMAX7)
    nav = _make_nav(a['close'], lev_mod, a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     L_s2.values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_g11_a2(a, lmax_base=6.0, vol_sens=2.0, vol_ref=0.20):
    ret = a['ret']
    vol252 = ret.rolling(252, min_periods=126).std() * np.sqrt(TRADING_DAYS)
    vol252_exp = ret.expanding(min_periods=30).std() * np.sqrt(TRADING_DAYS)
    vol252 = vol252.fillna(vol252_exp)
    L_FLOOR, L_CEIL = 4.5, 6.5
    l_max_t = (lmax_base - vol_sens * (vol252 / vol_ref - 1)).clip(L_FLOOR, L_CEIL)
    l_max_t = l_max_t.reindex(ret.index).fillna(lmax_base)
    L_s2 = compute_L_s2_dyn_lmax(ret, a['vz'], l_max_t, **S2_BASE)

    vz_arr = a['vz_arr']
    k_dyn = np.where(vz_arr >  VZ_THR_E4, K_HI,
              np.where(vz_arr < -VZ_THR_E4, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    nav = _make_nav(a['close'], lev_mod, a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     L_s2.values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_g11_a2b(a, lmax_base=6.0, vol_sens=2.0):
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
    k_dyn = np.where(vz_arr >  VZ_THR_E4, K_HI,
              np.where(vz_arr < -VZ_THR_E4, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    nav = _make_nav(a['close'], lev_mod, a['wn_A'], a['wg_A'], a['wb_A'],
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     L_s2.values)
    return nav, a['wn_A'], a['wb_A'], np.asarray(a['lev_raw'])


def build_g11_a3(a, vov_thr=1.3, alpha_max=0.2):
    ret = a['ret']
    vz_arr = a['vz_arr']
    k_dyn = np.where(vz_arr >  VZ_THR_E4, K_HI,
              np.where(vz_arr < -VZ_THR_E4, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    L_s2 = compute_L_s2_vz_gated(ret, a['vz'], **S2_LMAX7)

    vov_z = compute_vov_zscore(ret)
    vov_arr = vov_z.values
    vz_pos = (a['vz'].fillna(0).values > VZ_THR_E4)
    stress = (vov_arr > vov_thr) & vz_pos

    wn_p, wg_p, wb_p = apply_regime_tilt(a['wn_A'], a['wg_A'], a['wb_A'],
                                          stress, alpha_max)
    nav = _make_nav(a['close'], lev_mod, wn_p, wg_p, wb_p,
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     L_s2.values)
    return nav, wn_p, wb_p, np.asarray(a['lev_raw'])


def build_g11_c2(a, eps_0=0.020):
    ret = a['ret']
    vz_arr = a['vz_arr']
    k_dyn = np.where(vz_arr >  VZ_THR_E4, K_HI,
              np.where(vz_arr < -VZ_THR_E4, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(a['lev_raw'], lt_bias, l_min=0.0, l_max=1.0)
    L_s2 = compute_L_s2_vz_gated(ret, a['vz'], **S2_LMAX7)

    raw_a2 = a['raw_a2']
    raw_a2_vals = raw_a2.values if hasattr(raw_a2, 'values') else np.asarray(raw_a2)
    bull_mask   = raw_a2_vals > THRESHOLD

    vol_20d  = ret.rolling(C2_VOL_WINDOW).std() * np.sqrt(252)
    vol_mean = vol_20d.rolling(C2_SIGMA_LOOKBACK).mean()
    vol_ratio = (vol_20d / vol_mean).fillna(1.0).clip(0.3, 3.0)
    eps_t = eps_0 * vol_ratio.values

    tilt_confirmed, _ = compute_tilt_with_adaptive_deadband(
        raw_a2_vals, vz_arr, bull_mask, eps_t,
    )
    wn_tilted = a['wn_A'] + tilt_confirmed
    wb_tilted = np.clip(a['wb_A'] - tilt_confirmed, 0.0, a['wb_A'])
    nav = _make_nav(a['close'], lev_mod, wn_tilted, a['wg_A'], wb_tilted,
                     a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'],
                     L_s2.values)
    return nav, wn_tilted, wb_tilted, np.asarray(a['lev_raw'])


def build_g11_c3(a, yz_n=10, vz_thr=0.7):
    close = a['close']
    ret   = a['ret']
    dates = a['dates']
    df_ohlc = load_ohlc(OHLC_PATH)
    vz_yz = build_vz_from_yz(close, df_ohlc, dates, yz_n=yz_n)

    raw_a2_yz, _ = build_a2_signal(close, ret)
    lev_raw_yz, wn_A_yz, wg_A_yz, wb_A_yz, _ = simulate_rebalance_A(
        raw_a2_yz, vz_yz, THRESHOLD,
    )
    L_s2 = compute_L_s2_vz_gated(ret, vz_yz, **S2_LMAX7)

    vz_yz_arr = vz_yz.values
    k_dyn = np.where(vz_yz_arr >  vz_thr, K_HI,
              np.where(vz_yz_arr < -vz_thr, K_LO, K_MID))
    lt_bias = pd.Series(np.clip(-k_dyn * a['lt_sig_arr'] * 0.5, -0.5, 0.5),
                        index=a['lt_sig_raw'].index)
    lev_mod = apply_lt_mode_b(lev_raw_yz, lt_bias, l_min=0.0, l_max=1.0)
    nav = _make_nav(close, lev_mod, wn_A_yz, wg_A_yz, wb_A_yz,
                     dates, a['gold_2x'], a['bond_3x'], a['sofr'],
                     L_s2.values)
    return nav, wn_A_yz, wb_A_yz, np.asarray(lev_raw_yz)


def build_g11_d5(a, vz_thr, l_max):
    return _e4_like_klt(a, k_lo=K_LO, k_hi=K_HI, vz_thr=vz_thr, l_max=l_max)


# ---------------------------------------------------------------------------
# 戦略レジストリ (26本)
# ---------------------------------------------------------------------------
STRATEGY_BUILDERS = [
    # [g3] 2本
    ('g3-E4-RegimeKLT',         build_g3_e4_regimeklt),
    ('g3-REF-N750',             build_g3_ref_n750),
    # [g7] 2本
    ('g7-F10-eps015',           build_g7_f10_eps015),
    ('g7-REF-F8R5-eps0',        build_g7_ref_f8r5_eps0),
    # [g8] 3本
    ('g8-REF-E4',               build_g8_ref_e4),
    ('g8-E4-lmax5',             build_g8_e4_lmax5),
    ('g8-F10-eps015-lmax5',     build_g8_f10_eps015_lmax5),
    # [g10] 4本 (REF-E4 は g8 と重複なので省略)
    ('g10-vz065-lmax5',         build_g10_vz065_lmax5),
    ('g10-vz065-lmax5p5',       build_g10_vz065_lmax5p5),
    ('g10-vz065-lmax6',         build_g10_vz065_lmax6),
    ('g10-vz065-lmax7',         build_g10_vz065_lmax7),
    # [g11] 14本
    ('g11-B4-klo0',             build_g11_b4_klo0),
    ('g11-A1-alpha2',           lambda a: build_g11_a1(a, alpha=2.0)),
    ('g11-A1-alpha3',           lambda a: build_g11_a1(a, alpha=3.0)),
    ('g11-A1-alpha5',           lambda a: build_g11_a1(a, alpha=5.0)),
    ('g11-A1-alpha8',           lambda a: build_g11_a1(a, alpha=8.0)),
    ('g11-A2-lmax6vs2',         lambda a: build_g11_a2(a, lmax_base=6.0, vol_sens=2.0)),
    ('g11-A2B-rolling',         lambda a: build_g11_a2b(a, lmax_base=6.0, vol_sens=2.0)),
    ('g11-A3-vov',              lambda a: build_g11_a3(a, vov_thr=1.3, alpha_max=0.2)),
    ('g11-C2-adaptive',         lambda a: build_g11_c2(a, eps_0=0.020)),
    ('g11-C3-yz',               lambda a: build_g11_c3(a, yz_n=10, vz_thr=0.7)),
    ('g11-D5-vz060-lmax45',     lambda a: build_g11_d5(a, vz_thr=0.60, l_max=4.5)),
    ('g11-D5-vz060-lmax50',     lambda a: build_g11_d5(a, vz_thr=0.60, l_max=5.0)),
    ('g11-D5-vz065-lmax45',     lambda a: build_g11_d5(a, vz_thr=0.65, l_max=4.5)),
    ('g11-D5-vz070-lmax50',     lambda a: build_g11_d5(a, vz_thr=0.70, l_max=5.0)),
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    n_strats = len(STRATEGY_BUILDERS)
    print('=' * 72)
    print(f'G14: WFA — SBI店頭CFD スプレッド ({SBI_CFD_SPREAD*100:.2f}%/yr)')
    print(f'      全 {n_strats} 戦略再評価')
    print(f'実行日: {TODAY}')
    print('=' * 72)

    print('\n[S1] Loading shared assets...')
    assets = load_shared_assets()
    dates  = assets['dates']

    print('\n[S2] Generating windows (252日非重複, カレンダー年アンカー)...')
    windows = generate_windows(dates)
    print(f'  {len(windows)} windows: '
          f'{windows[0]["start_date"].date()} ... {windows[-1]["end_date"].date()}')
    short_wins = sum(w['short_flag'] for w in windows)
    print(f'  Short windows: {short_wins}')

    all_pw_rows = []
    all_sm_rows = []
    results = {}

    print(f'\n[S3] Evaluating {n_strats} strategies with SBI_CFD_SPREAD={SBI_CFD_SPREAD}...')
    print('-' * 72)
    for sid, builder in STRATEGY_BUILDERS:
        print(f'  [{sid}] ', end='', flush=True)
        try:
            nav, wn, wb, lev_arr = builder(assets)
        except Exception as exc:
            print(f'FAILED: {exc}')
            raise

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
        print(f'CAGR={mean_c*100:+6.2f}%  '
              f'CI95=[{ci_lo*100:+6.2f}%, {ci_hi*100:+6.2f}%]  '
              f't_p={tp:.4f}  WFE={wfe:+.3f}  => {verdict}')

    print('-' * 72)

    print('\n[S4] Summary:')
    print('=' * 115)
    print(f'{"Strategy":<28} {"mean_CAGR":>10} {"CI95_lo":>9} {"CI95_hi":>9}'
          f' {"t_p":>9} {"WFE":>7} {"Trades/yr":>9} {"Verdict":>8}')
    print('-' * 115)
    for sid, _ in STRATEGY_BUILDERS:
        s = results[sid]['summary']
        v = results[sid]['verdict']
        print(
            f'{sid:<28}'
            f' {s.get("mean_CAGR", 0)*100:>+9.2f}%'
            f' {s.get("WFA_CI95_lo", np.nan)*100:>+8.2f}%'
            f' {s.get("WFA_CI95_hi", np.nan)*100:>+8.2f}%'
            f' {s.get("t_pvalue", np.nan):>9.6f}'
            f' {s.get("WFA_WFE", np.nan):>7.3f}'
            f' {s.get("mean_Trades_yr", np.nan):>9.1f}'
            f' {v:>8}'
        )
    print('=' * 115)

    print('\n[S5] Saving CSVs...')
    pw_path = os.path.join(BASE, 'g14_wfa_sbi_cfd_per_window.csv')
    sm_path = os.path.join(BASE, 'g14_wfa_sbi_cfd_summary.csv')
    pd.DataFrame(all_pw_rows).to_csv(pw_path, index=False, float_format='%.6f')
    pd.DataFrame(all_sm_rows).to_csv(sm_path, index=False, float_format='%.6f')
    print(f'  Per-window: {pw_path}')
    print(f'  Summary:    {sm_path}')

    print('\nDone.')
    return results


if __name__ == '__main__':
    main()
