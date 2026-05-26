"""
G8: Walk-Forward Analysis — l_max=5.0 バリアント × 50窓
========================================================
EVALUATION_STANDARD §3.12 準拠 (v1.1, 2026-05-26)

目的:
  `l_max=5.0` という MaxDD 改善レバーが、WFA50窓で統計的に有効かを確認する。

対象戦略 (3本):
  REF-E4:            l_max=7.0 (G3 サニティ参照, CI95_lo=+26.51%, WFE=+1.131 と一致すべき)
  E4-lmax5:          E4 base + l_max=5.0 (MaxDD改善レバーの単独WFA)
  F10-eps015-lmax5:  F10 tilt + l_max=5.0 (CAGR向上 + MaxDD改善の複合)

窓設計: G3/G4/G5/G7 と完全同一
  - 評価開始: 1977-01-03 (LT2 warmup)
  - 評価終了: 2026-03-26
  - 窓長: 252営業日 / ステップ 252営業日（非重複、カレンダー年アンカー）
  - 総窓数: ~50

判定基準 (EVALUATION_STANDARD v1.1 §3.9/§3.10):
  α: WFA_CI95_lo > 0 AND t_p < 0.05
  β: WFA_WFE ∈ [0.5, 2.0]
  総合: α+β→PASS / αのみ→WARN / α FAIL→FAIL

出力:
  - g8_wfa_lmax5_per_window.csv
  - g8_wfa_lmax5_summary.csv
  - G8_WFA_LMAX5_2026-05-26.md
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
    CFD_SPREAD_LOW,
    IS_END, OOS_START,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, apply_lt_mode_b

# ---------------------------------------------------------------------------
# 定数
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

# CFD 設定 (l_max のみ差分)
S2_LMAX7 = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5,
                n=20, l_min=1.0, l_max=7.0, step=0.5)
S2_LMAX5 = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5,
                n=20, l_min=1.0, l_max=5.0, step=0.5)

# WFA 設定 (G3/G4/G5/G7 と完全同一)
WINDOW_DAYS   = 252
STEP_DAYS     = 252
EVAL_START    = '1977-01-03'
EVAL_END      = '2026-03-26'
IS_END_REF    = IS_END
OOS_START_REF = OOS_START
TODAY         = '2026-05-26'

STRATEGY_IDS = ['REF-E4', 'E4-lmax5', 'F10-eps015-lmax5']

STRATEGY_LABELS = {
    'REF-E4':            'REF E4 (l_max=7.0, G3 sanity)',
    'E4-lmax5':          'E4 + l_max=5.0',
    'F10-eps015-lmax5':  f'F10 (ε={EPS_F10}) + l_max=5.0',
}

# G3 サニティ参照値 (REF-E4 = G3 と同一戦略設定なので一致すべき)
REF_G3_CI95_LO = 0.265093    # G3: E4 WFA_CI95_lo = +26.51%
REF_G3_WFE     = 1.130664    # G3: E4 WFA_WFE     = +1.131

# G7 参照 (F10-eps015 with l_max=7.0)
REF_G7_CI95_LO = 0.279162    # G7 (F10-eps015): WFA_CI95_lo (注: G5 の REF と同値で公開済み, F10 でも同等)
REF_G7_WFE     = 1.207959

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# ε-Deadband (G7 と同一)
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


def count_trades_in_window(wn_tilted, wb_tilted, lev_arr, s, e):
    n_tr = 0
    for i in range(s + 1, e):
        if (wn_tilted[i] != wn_tilted[i-1] or
            wb_tilted[i] != wb_tilted[i-1] or
            lev_arr[i]   != lev_arr[i-1]):
            n_tr += 1
    return n_tr


# ---------------------------------------------------------------------------
# 共有資産ロード
# ---------------------------------------------------------------------------

def load_shared_assets(data_path: str) -> dict:
    df    = load_data(data_path)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    n_years = n / TRADING_DAYS
    print(f'  Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n_years:.2f} yr)')

    sofr = load_sofr(dates)

    # Scenario D 資産
    gold_1x_local = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x_local, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                          bond_maturity=22.0)
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Scenario D assets done.')

    # DH Dyn A シグナル
    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    print(f'  DH Dyn A: {n_tr} trades, {n_tr/n_years:.1f}/yr (lev_raw 基準)')

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

    # CFD レバレッジ (l_max=7.0 と l_max=5.0 の両方)
    L_s2_lmax7 = compute_L_s2_vz_gated(ret, vz, **S2_LMAX7)
    L_s2_lmax5 = compute_L_s2_vz_gated(ret, vz, **S2_LMAX5)
    print('  S2 leverage done (l_max=7.0 and l_max=5.0).')

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

    return dict(
        close=close, ret=ret, dates=dates, n=n, n_years=n_years, sofr=sofr,
        gold_2x=gold_2x, bond_3x=bond_3x,
        raw_a2=raw_a2, vz=vz,
        lev_raw=lev_raw, lev_mod_e4=lev_mod_e4,
        wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        wn_f10=wn_f10, wg_f10=wg_f10, wb_f10=wb_f10,
        L_s2_lmax7=L_s2_lmax7, L_s2_lmax5=L_s2_lmax5,
        tilt_updates_f10=n_updates_f10,
    )


# ---------------------------------------------------------------------------
# 戦略 NAV 構築
# ---------------------------------------------------------------------------

def build_all_navs(assets: dict) -> dict:
    a = assets
    close, dates, sofr = a['close'], a['dates'], a['sofr']
    gold_2x, bond_3x   = a['gold_2x'], a['bond_3x']
    lev_mod_e4         = a['lev_mod_e4']
    lev_raw_arr        = np.asarray(a['lev_raw'])

    out = {}

    # REF-E4: E4 base, l_max=7.0 (G3 サニティ用)
    nav_ref = build_nav_strategy(
        close, lev_mod_e4, a['wn_A'], a['wg_A'], a['wb_A'], dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD',
        cfd_leverage=a['L_s2_lmax7'].values,
        cfd_spread=CFD_SPREAD_LOW,
    )
    out['REF-E4'] = dict(
        nav=nav_ref,
        wn=np.asarray(a['wn_A']), wb=np.asarray(a['wb_A']),
        lev_raw_arr=lev_raw_arr,
    )

    # E4-lmax5: E4 base, l_max=5.0
    nav_e4l5 = build_nav_strategy(
        close, lev_mod_e4, a['wn_A'], a['wg_A'], a['wb_A'], dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD',
        cfd_leverage=a['L_s2_lmax5'].values,
        cfd_spread=CFD_SPREAD_LOW,
    )
    out['E4-lmax5'] = dict(
        nav=nav_e4l5,
        wn=np.asarray(a['wn_A']), wb=np.asarray(a['wb_A']),
        lev_raw_arr=lev_raw_arr,
    )

    # F10-eps015-lmax5: F10 tilt + l_max=5.0
    nav_f10l5 = build_nav_strategy(
        close, lev_mod_e4, a['wn_f10'], a['wg_f10'], a['wb_f10'], dates,
        gold_2x, bond_3x, sofr,
        nas_mode='CFD',
        cfd_leverage=a['L_s2_lmax5'].values,
        cfd_spread=CFD_SPREAD_LOW,
    )
    out['F10-eps015-lmax5'] = dict(
        nav=nav_f10l5,
        wn=np.asarray(a['wn_f10']), wb=np.asarray(a['wb_f10']),
        lev_raw_arr=lev_raw_arr,
    )

    return out


# ---------------------------------------------------------------------------
# Window 生成 (G7 と完全同一 — カレンダー年アンカー方式)
# ---------------------------------------------------------------------------

def generate_windows(dates: pd.Series,
                     eval_start: str = EVAL_START,
                     eval_end: str = EVAL_END,
                     window_days: int = WINDOW_DAYS,
                     step_days: int = STEP_DAYS) -> list:
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
            window_id  = year,
            start_idx  = s_idx,
            end_idx    = e_idx,
            start_date = dates.iloc[s_idx],
            end_date   = dates.iloc[e_idx],
            n_days     = n_days,
            short_flag = n_days < int(window_days * 0.8),
        ))

    return windows


# ---------------------------------------------------------------------------
# 窓内指標計算 (G7 と完全同一)
# ---------------------------------------------------------------------------

def compute_window_metrics(nav: pd.Series, window: dict,
                            wn=None, wb=None, lev_arr=None,
                            trading_days: int = TRADING_DAYS) -> dict:
    s = window['start_idx']
    e = window['end_idx'] + 1
    nav_arr = nav.iloc[s:e].values.astype(float)
    n_days  = len(nav_arr)

    if n_days < 5 or nav_arr[0] <= 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, MaxDD=np.nan,
                    Vol=np.nan, PosDay_pct=np.nan, n_days=n_days,
                    Trades_yr=np.nan)

    daily_ret = np.diff(nav_arr) / nav_arr[:-1]
    years     = n_days / trading_days
    cagr      = (nav_arr[-1] / nav_arr[0]) ** (1.0 / years) - 1

    r_std  = np.std(daily_ret, ddof=1)
    sharpe = (np.mean(daily_ret) / r_std * np.sqrt(trading_days)
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


# ---------------------------------------------------------------------------
# 統計集計 (G7 と完全同一)
# ---------------------------------------------------------------------------

def compute_summary_stats(per_df: pd.DataFrame) -> dict:
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

    mean_s  = float(np.mean(sharpes)) if len(sharpes) > 0 else np.nan
    std_s   = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else np.nan

    oos_start_ts = pd.Timestamp(OOS_START_REF)
    is_mask   = valid['start_date'] < oos_start_ts
    post_mask = valid['start_date'] >= oos_start_ts
    is_cagrs   = valid.loc[is_mask,   'CAGR'].dropna().values
    post_cagrs = valid.loc[post_mask, 'CAGR'].dropna().values
    mean_is   = float(np.mean(is_cagrs))   if len(is_cagrs)   > 0 else np.nan
    mean_post = float(np.mean(post_cagrs)) if len(post_cagrs) > 0 else np.nan
    wfe_post  = (mean_post / mean_is) if (not np.isnan(mean_is) and mean_is != 0) else np.nan

    if 'Trades_yr' in valid.columns:
        tr = valid['Trades_yr'].dropna().values
        mean_tr = float(np.mean(tr)) if len(tr) > 0 else np.nan
    else:
        mean_tr = np.nan

    return dict(
        n_windows=n,
        mean_CAGR=mean_c, median_CAGR=med_c, std_CAGR=std_c,
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


# ---------------------------------------------------------------------------
# 判定基準 (G7 と完全同一)
# ---------------------------------------------------------------------------

def evaluate_criteria(summary: dict) -> tuple:
    crit_alpha = (summary.get('WFA_CI95_lo', -1) > 0 and
                  summary.get('t_pvalue', 1.0) < 0.05)
    wfe    = summary.get('WFA_WFE', np.nan)
    n_post = summary.get('n_windows_postIS', 0)
    if n_post < 3:
        crit_beta = True
    else:
        crit_beta = (not np.isnan(wfe)) and (0.5 <= wfe <= 2.0)

    crits = dict(alpha=crit_alpha, beta=crit_beta)
    if crit_alpha and crit_beta:
        verdict = 'PASS'
    elif crit_alpha:
        verdict = 'WARN'
    else:
        verdict = 'FAIL'

    return verdict, crits


# ---------------------------------------------------------------------------
# フォーマットヘルパー
# ---------------------------------------------------------------------------

def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v * 100:+.{d}f}%'


def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:+.{d}f}'


# ---------------------------------------------------------------------------
# Markdown レポート
# ---------------------------------------------------------------------------

def generate_md_report(results: dict, sanity: dict, windows: list,
                        tilt_updates: int, base_dir: str) -> str:

    ref   = results.get('REF-E4', {}).get('summary', {})
    e4l5  = results.get('E4-lmax5', {}).get('summary', {})
    f10l5 = results.get('F10-eps015-lmax5', {}).get('summary', {})

    ref_v   = results.get('REF-E4', {}).get('verdict', 'N/A')
    e4l5_v  = results.get('E4-lmax5', {}).get('verdict', 'N/A')
    f10l5_v = results.get('F10-eps015-lmax5', {}).get('verdict', 'N/A')

    ref_cr   = results.get('REF-E4', {}).get('criteria', {})
    e4l5_cr  = results.get('E4-lmax5', {}).get('criteria', {})
    f10l5_cr = results.get('F10-eps015-lmax5', {}).get('criteria', {})

    lines = []
    lines += [
        f'# G8: Walk-Forward Analysis — l_max=5.0 バリアント × {len(windows)}窓',
        '',
        f'作成日: {TODAY}',
        f'EVALUATION_STANDARD: v1.1',
        '',
        '## 目的',
        '',
        '`l_max=5.0` という MaxDD 改善レバーが WFA50窓で統計的に有効かを確認する。',
        '',
        '対象戦略 (3本):',
        '',
        '1. **REF-E4** (l_max=7.0): G3 サニティ参照。G3 結果 (CI95_lo=+26.51%, WFE=+1.131) と一致すべき。',
        '2. **E4-lmax5** (E4 base + l_max=5.0): MaxDD 改善レバーの単独 WFA。',
        '3. **F10-eps015-lmax5** (F10 tilt + l_max=5.0): CAGR 向上 + MaxDD 改善の複合。',
        '',
        '背景:',
        '- 現行ベスト E4 (l_max=7.0): バックテスト CAGR_OOS=+33.53%, Sharpe=+0.891, MaxDD=-60.01% / WFA G3 PASS (CI95_lo=+26.51%, WFE=+1.131)',
        '- F10 ε=0.015 (l_max=7.0): バックテスト CAGR_OOS=+36.84%, Sharpe=+0.935, MaxDD=-63.09% / WFA G7 PASS (CI95_lo=+27.91%, WFE=+1.208)',
        '- E5b 実測 (vz_thr=0.70 + l_max=5.0): バックテスト CAGR_OOS=+30.58%, MaxDD=-54.07% (CAGR -2.95pp トレードオフで MaxDD +5.94pp 改善)',
        '',
        '---', '',
        '## 1. セットアップ', '',
        '| 項目 | 値 |',
        '|------|-----|',
        f'| 評価開始 | {EVAL_START} (LT2 warmup完了) |',
        f'| 評価終了 | {EVAL_END} |',
        f'| 窓長 | {WINDOW_DAYS}営業日 (1年) |',
        f'| ステップ | {STEP_DAYS}営業日 (非重複) |',
        f'| 総窓数 | {len(windows)} |',
        f'| IS境界 | {IS_END_REF} |',
        f'| OOS開始 | {OOS_START_REF} |',
        f'| E4 base | k_lo={K_LO}, k_hi={K_HI}, vz_thr={VZ_THR}, k_mid={K_MID}, LT2-N{N_LT2} |',
        f'| F10 tilt | ε={EPS_F10}, tilt={TILT_R5}, cap=calm{TILT_CAP_CALM}/bullVZ{TILT_CAP_BULL_VZ}/bearVZ{TILT_CAP_BEAR_VZ} |',
        f'| CFD (REF-E4) | target_vol=0.8, k_vz=0.3, gate_min=0.5, n=20, l_min=1.0, **l_max=7.0**, step=0.5 |',
        f'| CFD (E4-lmax5 / F10-lmax5) | 上記同一だが **l_max=5.0** |',
        f'| Trades/yr 基準 | `lev_raw` (discrete) |',
        '',
        f'**判定基準 (v1.1 §3.9/§3.10)**: α (CI95_lo>0 ∧ t_p<0.05) ∩ β (WFE∈[0.5, 2.0])',
        '',
        '---', '',
    ]

    # §2 WFA サマリ
    lines += ['## 2. WFA サマリ（3戦略）', '']
    lines += [
        '| 戦略 | n | mean_ret | std_ret | t_stat | t_p | CI95_lo | CI95_hi | WFE | Tr/yr | 判定 |',
        '|------|--:|---------:|--------:|-------:|----:|--------:|--------:|----:|------:|:---:|',
    ]
    for sid in STRATEGY_IDS:
        if sid not in results:
            continue
        s = results[sid]['summary']
        v = results[sid]['verdict']
        label = STRATEGY_LABELS.get(sid, sid)
        lines.append(
            f'| {label} '
            f'| {s.get("n_windows", "N/A")} '
            f'| {_fp(s.get("mean_CAGR"))} '
            f'| {_fp(s.get("std_CAGR"))} '
            f'| {_ff(s.get("t_stat"), d=3)} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_fp(s.get("WFA_CI95_hi"))} '
            f'| {_ff(s.get("WFA_WFE"))} '
            f'| {_ff(s.get("mean_Trades_yr"), d=1)} '
            f'| **{v}** |'
        )
    lines += ['', '---', '']

    # §3 判定詳細
    lines += ['## 3. 判定詳細 (α∩β 2基準)', '']
    lines += [
        '| 戦略 | α: WFA_CI95_lo | α: t_p | β: WFA_WFE | 判定 |',
        '|------|---------------:|-------:|-----------:|:---:|',
    ]
    for sid in STRATEGY_IDS:
        if sid not in results:
            continue
        s  = results[sid]['summary']
        cr = results[sid]['criteria']
        v  = results[sid]['verdict']
        a_mark = 'OK' if cr.get('alpha') else 'NG'
        b_mark = 'OK' if cr.get('beta')  else 'NG'
        lines.append(
            f'| {STRATEGY_LABELS.get(sid, sid)} '
            f'| [{a_mark}] {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| [{b_mark}] {_ff(s.get("WFA_WFE"))} '
            f'| **{v}** |'
        )
    lines += ['', '---', '']

    # §4 サニティ
    ci_diff_pp = sanity.get('ci_diff_pp', np.nan)
    wfe_diff   = sanity.get('wfe_diff',   np.nan)
    ci_ok      = (not np.isnan(ci_diff_pp))  and abs(ci_diff_pp)  <= 0.1
    wfe_ok     = (not np.isnan(wfe_diff))    and abs(wfe_diff)    <= 0.001
    lines += ['## 4. サニティチェック (REF-E4 vs G3)', '']
    lines += [
        '| 項目 | G3 値 | G8 実測 (REF-E4) | 差分 | 許容範囲 | 判定 |',
        '|------|------:|-----------------:|-----:|---------:|:---:|',
        f'| CI95_lo | +{REF_G3_CI95_LO*100:.4f}% | {_fp(ref.get("WFA_CI95_lo"), d=4)} | {ci_diff_pp:+.4f} pp | ±0.1 pp | {"OK" if ci_ok else "WARN"} |',
        f'| WFA_WFE | +{REF_G3_WFE:.6f} | {_ff(ref.get("WFA_WFE"), d=6)} | {wfe_diff:+.6f} | ±0.001 | {"OK" if wfe_ok else "WARN"} |',
        '',
        'REF-E4 (l_max=7.0) は G3 の E4 と完全同一の戦略設定。完全一致 (±0pp / ±0.000) が期待される。',
        '',
        '---', '',
    ]

    # §5 E4-lmax5 vs REF-E4 (MaxDD 改善レバー単独効果)
    lines += ['## 5. E4-lmax5 vs REF-E4 比較（l_max=5.0 の単独効果）', '']
    if e4l5 and ref:
        d_mean = (e4l5.get('mean_CAGR', np.nan) - ref.get('mean_CAGR', np.nan)) * 100
        d_ci   = (e4l5.get('WFA_CI95_lo', np.nan) - ref.get('WFA_CI95_lo', np.nan)) * 100
        d_wfe  = e4l5.get('WFA_WFE', np.nan) - ref.get('WFA_WFE', np.nan)
        d_sh   = e4l5.get('mean_Sharpe', np.nan) - ref.get('mean_Sharpe', np.nan)
        d_tr   = e4l5.get('mean_Trades_yr', np.nan) - ref.get('mean_Trades_yr', np.nan)
        lines += [
            '| 指標 | REF-E4 (l_max=7.0) | E4-lmax5 (l_max=5.0) | 差分 (lmax5 − ref) |',
            '|------|-------------------:|---------------------:|--------------------:|',
            f'| mean_CAGR  | {_fp(ref.get("mean_CAGR"))}   | {_fp(e4l5.get("mean_CAGR"))}   | {d_mean:+.4f} pp |',
            f'| CI95_lo    | {_fp(ref.get("WFA_CI95_lo"))} | {_fp(e4l5.get("WFA_CI95_lo"))} | {d_ci:+.4f} pp |',
            f'| CI95_hi    | {_fp(ref.get("WFA_CI95_hi"))} | {_fp(e4l5.get("WFA_CI95_hi"))} | |',
            f'| t_p        | {_ff(ref.get("t_pvalue"), d=6)} | {_ff(e4l5.get("t_pvalue"), d=6)} | |',
            f'| WFA_WFE    | {_ff(ref.get("WFA_WFE"), d=4)} | {_ff(e4l5.get("WFA_WFE"), d=4)} | {d_wfe:+.4f} |',
            f'| mean_Sharpe | {_ff(ref.get("mean_Sharpe"))} | {_ff(e4l5.get("mean_Sharpe"))} | {d_sh:+.4f} |',
            f'| mean_Trades/yr | {_ff(ref.get("mean_Trades_yr"), d=1)} | {_ff(e4l5.get("mean_Trades_yr"), d=1)} | {d_tr:+.2f} |',
            '',
            '**バックテスト実測 (参考)**:',
            '- REF-E4 (l_max=7.0): CAGR_OOS=+33.53%, Sharpe=+0.891, MaxDD=-60.01%',
            '- vz_thr=0.70 + l_max=5.0 (E5b): CAGR_OOS=+30.58%, MaxDD=-54.07% (CAGR -2.95pp / MaxDD +5.94pp)',
            '',
        ]
    lines += ['---', '']

    # §6 F10-eps015-lmax5 vs F10-eps015 (G7)
    lines += ['## 6. F10-eps015-lmax5 vs F10-eps015 (G7) 比較', '']
    if f10l5:
        d_ci_g7  = (f10l5.get('WFA_CI95_lo', np.nan) - REF_G7_CI95_LO) * 100
        d_wfe_g7 = f10l5.get('WFA_WFE', np.nan) - REF_G7_WFE
        lines += [
            '| 指標 | F10-eps015 (G7, l_max=7.0) ‡ | F10-eps015-lmax5 (G8, l_max=5.0) | 差分 |',
            '|------|----------------------------:|----------------------------------:|-----:|',
            f'| CI95_lo | +{REF_G7_CI95_LO*100:.2f}% | {_fp(f10l5.get("WFA_CI95_lo"))} | {d_ci_g7:+.4f} pp |',
            f'| WFA_WFE | +{REF_G7_WFE:.3f}  | {_ff(f10l5.get("WFA_WFE"))} | {d_wfe_g7:+.4f} |',
            f'| t_p     | (G7 参照)            | {_ff(f10l5.get("t_pvalue"), d=6)} | |',
            f'| mean_CAGR | (G7 参照)            | {_fp(f10l5.get("mean_CAGR"))} | |',
            f'| mean_Sharpe | (G7 参照)            | {_ff(f10l5.get("mean_Sharpe"))} | |',
            f'| mean_Trades/yr | (G7 参照)            | {_ff(f10l5.get("mean_Trades_yr"), d=1)} | |',
            '',
            '‡ G7 結果は `G7_WFA_F10_2026-05-26.md` 参照。',
            '',
            '`l_max=5.0` 追加により CI95_lo / WFE がどう変化したかを確認する。',
            '',
        ]
    lines += ['---', '']

    # §7 正式昇格判定
    lines += ['## 7. 正式昇格判定', '']
    lines += [
        '| 戦略 | CI95_lo | WFE | t_p | α | β | 総合 | 昇格判断 |',
        '|------|--------:|----:|----:|:--:|:--:|:---:|:--------|',
    ]
    E4_CI = REF_G3_CI95_LO   # 0.2651

    def _decision(v, ci, label):
        if v == 'PASS':
            if not np.isnan(ci) and ci > E4_CI:
                return (f'**正式 Active 昇格候補** ({label}, E4 CI95_lo=+{E4_CI*100:.2f}% を上回る)。'
                        f'MaxDD 改善も併せ確認し `CURRENT_BEST_STRATEGY.md` 更新候補。')
            else:
                return (f'PASS 達成。E4 CI95_lo=+{E4_CI*100:.2f}% を超えず → SHORTLISTED 候補 '
                        f'(Active は E4 維持)。MaxDD 改善トレードオフ評価は別途検討。')
        elif v == 'WARN':
            return f'SHORTLISTED 維持 ({label}, β 基準未達)。'
        else:
            return f'昇格保留 ({label}, α 基準未達, E4 Active 維持)。'

    for sid in STRATEGY_IDS:
        s  = results[sid]['summary']
        cr = results[sid]['criteria']
        v  = results[sid]['verdict']
        a_mark = 'PASS' if cr.get('alpha') else 'FAIL'
        b_mark = 'PASS' if cr.get('beta')  else 'FAIL'
        ci = s.get('WFA_CI95_lo', np.nan)

        if sid == 'REF-E4':
            decision = '(参考: G3 で同 PASS 既確認。サニティ用)'
        else:
            decision = _decision(v, ci, sid)

        lines.append(
            f'| {STRATEGY_LABELS.get(sid, sid)} '
            f'| {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_ff(s.get("WFA_WFE"))} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| {a_mark} | {b_mark} '
            f'| **{v}** | {decision} |'
        )
    lines += ['', '---', '']

    # §8 再現コマンド
    lines += [
        '## 8. 再現コマンド', '',
        '```bash',
        'cd "C:\\Users\\user\\Desktop\\投資・不動産\\nasdaq_backtest"',
        'python -X utf8 src/g8_wfa_lmax5.py',
        '```', '',
        '出力:',
        f'- `g8_wfa_lmax5_per_window.csv` — {len(windows)}窓 × 3戦略の窓内指標',
        '- `g8_wfa_lmax5_summary.csv` — 3戦略の WFA サマリ',
        f'- `G8_WFA_LMAX5_{TODAY}.md` — 本レポート',
        '',
        '参照:',
        '- `src/g3_wfa_e4_regime.py` — G3 WFA (REF-E4 サニティ参照)',
        '- `src/g7_wfa_f10.py` — G7 WFA (F10-eps015 with l_max=7.0)',
        '- `g3_wfa_e4_summary.csv` — G3 結果',
        '- `g7_wfa_f10_summary.csv` — G7 結果',
        '- `CURRENT_BEST_STRATEGY.md` — E4 現行ベスト',
        '- `EVALUATION_STANDARD.md` §3.9/§3.10/§3.12',
        '',
        '---', '',
        f'*生成スクリプト: `src/g8_wfa_lmax5.py`  '
        f'準拠: `EVALUATION_STANDARD.md v1.1`*',
    ]

    md_text = '\n'.join(lines)
    md_path = os.path.join(base_dir, f'G8_WFA_LMAX5_{TODAY}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    return md_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 72)
    print('G8: Walk-Forward Analysis — l_max=5.0 バリアント × 非重複1年窓')
    print(f'実行日: {TODAY}')
    print('=' * 72)

    # S1: 共有資産ロード
    print('\n[S1] Loading shared assets...')
    assets = load_shared_assets(DATA_PATH)
    dates  = assets['dates']

    # S2: 3戦略 NAV 構築
    print('\n[S2] Building NAVs for 3 strategies...')
    nav_data = build_all_navs(assets)
    for sid in STRATEGY_IDS:
        d = nav_data[sid]
        print(f'  {sid:<20}: NAV range [{d["nav"].min():.4f}, {d["nav"].max():.4f}]')

    # S3: Window 生成
    print('\n[S3] Generating windows...')
    windows = generate_windows(dates, EVAL_START, EVAL_END, WINDOW_DAYS, STEP_DAYS)
    print(f'  {len(windows)} windows: '
          f'{windows[0]["start_date"].date()} ... {windows[-1]["end_date"].date()}')
    short_wins = sum(w['short_flag'] for w in windows)
    print(f'  Short windows (< 80% of target): {short_wins}')

    for i in range(len(windows) - 1):
        assert windows[i]['end_idx'] < windows[i+1]['start_idx'], \
            f'Overlapping windows at {i} and {i+1}'
    print('  Non-overlap check: OK')

    # S4: 戦略評価
    print('\n[S4] Evaluating all strategies across windows...')
    results     = {}
    all_pw_rows = []
    all_sm_rows = []

    for sid in STRATEGY_IDS:
        nav     = nav_data[sid]['nav']
        wn      = nav_data[sid]['wn']
        wb      = nav_data[sid]['wb']
        lev_arr = nav_data[sid]['lev_raw_arr']

        per_rows = []
        for w in windows:
            m = compute_window_metrics(nav, w, wn=wn, wb=wb, lev_arr=lev_arr)
            m.update(dict(
                strategy   = sid,
                window_id  = w['window_id'],
                start_date = w['start_date'],
                end_date   = w['end_date'],
                short_flag = w['short_flag'],
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

        mean_c = summary.get('mean_CAGR', np.nan)
        ci_lo  = summary.get('WFA_CI95_lo', np.nan)
        ci_hi  = summary.get('WFA_CI95_hi', np.nan)
        tp     = summary.get('t_pvalue', np.nan)
        wfe    = summary.get('WFA_WFE', np.nan)
        mean_tr = summary.get('mean_Trades_yr', np.nan)
        print(
            f'  {sid:<20}: mean_CAGR={mean_c*100:+.2f}%  '
            f'CI95=[{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]  '
            f't_p={tp:.4f}  WFE={wfe:+.3f}  '
            f'mean_Tr/yr={mean_tr:.1f}  => {verdict}'
        )

    # S5: サニティ
    print('\n[S5] Sanity checks (REF-E4 vs G3)...')
    ref_ci  = results['REF-E4']['summary'].get('WFA_CI95_lo', np.nan)
    ref_wfe = results['REF-E4']['summary'].get('WFA_WFE', np.nan)
    ci_diff_pp  = (ref_ci  - REF_G3_CI95_LO) * 100 if not np.isnan(ref_ci)  else np.nan
    wfe_diff    = (ref_wfe - REF_G3_WFE)            if not np.isnan(ref_wfe) else np.nan
    ci_ok       = (not np.isnan(ci_diff_pp))  and abs(ci_diff_pp)  <= 0.1
    wfe_ok      = (not np.isnan(wfe_diff))    and abs(wfe_diff)    <= 0.001
    print(f'  REF-E4 CI95_lo: {ref_ci*100:+.4f}%  (G3 {REF_G3_CI95_LO*100:+.4f}%,'
          f' diff {ci_diff_pp:+.4f} pp) => {"OK" if ci_ok else "WARN"}')
    print(f'  REF-E4 WFA_WFE: {ref_wfe:+.6f}  (G3 {REF_G3_WFE:+.6f},'
          f' diff {wfe_diff:+.6f}) => {"OK" if wfe_ok else "WARN"}')

    sanity = {'ci_diff_pp': ci_diff_pp, 'wfe_diff': wfe_diff,
              'ci_ok': ci_ok, 'wfe_ok': wfe_ok}

    # S6: コンソールサマリ
    print('\n[S6] Summary:')
    print('=' * 110)
    print(f'{"Strategy":<20} {"mean_CAGR":>10} {"CI95_lo":>9} {"CI95_hi":>9}'
          f' {"t_p":>9} {"Sharpe":>7} {"WFE":>7} {"Tr/yr":>7} {"Verdict":>8}')
    print('-' * 110)
    for sid in STRATEGY_IDS:
        s = results[sid]['summary']
        v = results[sid]['verdict']
        print(
            f'{sid:<20}'
            f' {s.get("mean_CAGR", 0)*100:>+9.2f}%'
            f' {s.get("WFA_CI95_lo", np.nan)*100:>+8.2f}%'
            f' {s.get("WFA_CI95_hi", np.nan)*100:>+8.2f}%'
            f' {s.get("t_pvalue", np.nan):>9.6f}'
            f' {s.get("mean_Sharpe", np.nan):>7.3f}'
            f' {s.get("WFA_WFE", np.nan):>7.3f}'
            f' {s.get("mean_Trades_yr", np.nan):>7.1f}'
            f' {v:>8}'
        )
    print('=' * 110)

    # S7: CSV 保存
    print('\n[S7] Saving CSVs...')
    pw_path = os.path.join(BASE, 'g8_wfa_lmax5_per_window.csv')
    sm_path = os.path.join(BASE, 'g8_wfa_lmax5_summary.csv')
    pd.DataFrame(all_pw_rows).to_csv(pw_path, index=False, float_format='%.6f')
    pd.DataFrame(all_sm_rows).to_csv(sm_path, index=False, float_format='%.6f')
    print(f'  Per-window: {pw_path}')
    print(f'  Summary:    {sm_path}')

    # S8: MD レポート
    print('[S8] Generating Markdown report...')
    md_path = generate_md_report(results, sanity, windows,
                                 assets.get('tilt_updates_f10', 0), BASE)
    print(f'  Report: {md_path}')

    print('\nDone.')
    return results


if __name__ == '__main__':
    main()
