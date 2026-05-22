"""
G1: Walk-Forward Analysis (WFA) — 11戦略 × 非重複1年窓
========================================================
目的: 11戦略の真のOOS性能を統計的に推定する。
     単一OOS区間(5年)の限界を、非重複1年窓49本で克服する。

窓設計:
  - 評価開始: 1977-01-03 (LT2 N=750 ウォームアップ完了後)
  - 評価終了: 2026-03-26
  - 窓長: 252営業日 (1年)
  - ステップ: 252営業日 (非重複)
  - 総窓数: 約49窓

判定基準 (EVALUATION_STANDARD v1.1 準拠):
  α: WFA_CI95_lo>0 AND t_p<0.05  (§3.9 統計的有意性)
  β: WFA_WFE ∈ [0.5, 2.0]        (§3.10 IS↔OOS安定性)
  総合: α+β PASS → PASS / α のみ PASS → WARN / α FAIL → FAIL

出力:
  g1_wfa_per_window.csv   (長形式: strategy × window の全レコード)
  g1_wfa_summary.csv      (戦略ごと1行サマリ)
  G1_WFA_2026-05-21.md
"""

import sys
import os
import types
import warnings

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
    build_nav,
    DATA_PATH, TRADING_DAYS, THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    CFD_SPREAD_LOW,
    IS_END, OOS_START,
)
from dynamic_leverage_strategies import (
    compute_L_s2_vz_gated,
    compute_L_vol_target,
    compute_L_s4_relvol,
)
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, signal_to_bias, apply_lt_mode_b
from sleeves_extended import build_gold_tocom
from test_portfolio_diversification import prepare_gold_data

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
WARMUP_DAYS   = 750
WINDOW_DAYS   = 252
STEP_DAYS     = 252
EVAL_START    = '1977-01-03'
EVAL_END      = '2026-03-26'
IS_END_REF    = IS_END       # '2021-05-07'
OOS_START_REF = OOS_START    # '2021-05-08'
TODAY         = '2026-05-21'

S2_FIXED = dict(target_vol=0.8, k_vz=0.3, gate_min=0.5,
                n=20, l_min=1.0, l_max=7.0, step=0.5)

STRATEGY_IDS = [
    'S2+LT2', 'S2', 'P2', 'S4', 'CFD7x',
    'DHA', 'BH1x', 'P02', 'P05', 'P01', 'DHA+LT2',
]

STRATEGY_LABELS = {
    'S2+LT2':  'S2_VZGated+LT2-N750-k0.5-modeB ◆ BEST',
    'S2':      'S2_VZGated (tv=0.8, k=0.3, gate=0.5)',
    'P2':      'P2 vol-target (tv=0.8)',
    'S4':      'S4_RelVol (l_base=7, k_rel=2.0)',
    'CFD7x':   'CFD 7x固定 (DH Dyn+7x)',
    'DHA':     'DH Dyn 2x3x [A] (TQQQ, Scenario D)',
    'BH1x':    'BH 1x (NASDAQ買持ち)',
    'P02':     'P02_Dyn×CPI [mult]',
    'P05':     'P05_HY×CPI [mult]',
    'P01':     'P01_Dyn×HY [mult]',
    'DHA+LT2': 'DH Dyn 2x3x [A+LT2] (LT2-N750-k0.5-modeB)',
}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS_PATH = os.path.join(BASE, 'data', 'timing_signals_raw.csv')

# サニティ参照値
REF_S2LT2_CAGR_OOS  = 0.3116   # B1確定
REF_S2LT2_SHARPE    = 0.858
REF_BH1X_CAGR_IS    = 0.1113   # STRATEGY_COMPARISON表より


# ---------------------------------------------------------------------------
# Gate ビルダ (calculate_p10_5y.py と同一ロジック)
# ---------------------------------------------------------------------------

def build_hy_gate(hy, z_thresh=1.0, slope=0.5):
    mu = hy.rolling(252, min_periods=126).mean()
    sd = hy.rolling(252, min_periods=126).std().clip(lower=0.01)
    z = (hy - mu) / sd
    g = (1.0 - np.maximum(0.0, z - z_thresh) * slope).clip(0.2, 1.0)
    return g.fillna(1.0)


def build_cpi_gate(cpi_yoy, cpi_accel, cpi_thresh=5.0, reduce_factor=0.3):
    infl_regime = ((cpi_yoy - cpi_thresh) / 5.0).clip(0.0, 1.0)
    accel_norm  = (cpi_accel / 2.0).clip(0.0, 1.0)
    g = (1.0 - reduce_factor * np.maximum(infl_regime, accel_norm)).clip(
        1.0 - reduce_factor, 1.0
    )
    return g.fillna(1.0)


def build_corr_gate(close, bond_3x, gold_2x, window=60, min_gate=0.2):
    ret = pd.Series(close.pct_change().fillna(0).values, index=close.index)
    bond_ret = pd.Series(bond_3x, index=close.index).pct_change().fillna(0)
    gold_ret = pd.Series(gold_2x, index=close.index).pct_change().fillna(0)
    rho_nb = ret.rolling(window).corr(bond_ret)
    rho_ng = ret.rolling(window).corr(gold_ret)
    hedge  = (-rho_nb).clip(lower=0.0) + (-rho_ng).clip(lower=0.0)
    g = hedge.clip(lower=min_gate, upper=1.0)
    return g.fillna(1.0)


def apply_gates(wn_A_arr, nas_gate=None, bond_gate=None):
    ones  = np.ones(len(wn_A_arr))
    g_nas  = np.where(np.isnan(nas_gate), 1.0, nas_gate)  if nas_gate  is not None else ones
    g_bond = np.where(np.isnan(bond_gate), 1.0, bond_gate) if bond_gate is not None else ones
    wn   = np.clip(wn_A_arr * g_nas, 0.0, 1.0)
    rest = 1.0 - wn
    wg   = rest * 0.5
    wb   = rest * 0.5 * g_bond
    return wn, wg, wb


# ---------------------------------------------------------------------------
# 共有資産ロード (1回のみ)
# ---------------------------------------------------------------------------

def load_shared_assets(data_path: str) -> dict:
    df    = load_data(data_path)
    close = df['Close']
    ret   = close.pct_change().fillna(0)
    dates = df['Date']
    n     = len(df)
    print(f'  Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} '
          f'({n:,} days, {n/TRADING_DAYS:.2f} yr)')

    sofr = load_sofr(dates)

    # Scenario D 資産 (CFD + DHA 戦略用)
    gold_1x_local = prepare_gold_local(dates)
    gold_2x_sd    = build_gold_2x(gold_1x_local, sofr_daily=sofr, apply_sofr=True)
    bond_1x       = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                                 bond_maturity=22.0)
    bond_3x_sd    = build_bond_3x(bond_1x, sofr, apply_sofr=True)
    print('  Scenario D assets done.')

    # P系コストモデル (gold_tocom + bond_3x SOFR off)
    dates_dt   = pd.DatetimeIndex(pd.to_datetime(dates.values))
    gold_1x_pf = prepare_gold_data(dates_dt)
    gold_2x_p  = build_gold_tocom(gold_1x_pf, 2.0, sofr)
    bond_3x_p  = build_bond_3x(bond_1x, sofr, apply_sofr=False)
    print('  P-series assets done.')

    # DH Dyn A シグナル
    raw_a2, vz = build_a2_signal(close, ret)
    lev_A, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_years = n / TRADING_DAYS
    print(f'  DH Dyn A: {n_tr} trades, {n_tr/n_years:.1f}/yr')

    # LT2 シグナル
    lt_sig  = build_lt_signal(close, 'LT2', N=750)
    lt_bias = signal_to_bias(lt_sig, k_lt=0.5)
    lev_mod = apply_lt_mode_b(lev_A, lt_bias, l_min=0.0, l_max=1.0)
    print('  LT2 signal done.')

    # P系タイミングシグナル
    sig_raw = pd.read_csv(SIGNALS_PATH, index_col=0, parse_dates=True)
    sig     = sig_raw.reindex(dates_dt)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hy_s    = pd.Series(sig['hy_spread'].values, index=close.index).fillna(method='ffill').fillna(4.0)
    cpi_yoy = pd.Series(sig['cpi_yoy'].fillna(0.0).values, index=close.index)
    cpi_acc = pd.Series(sig['cpi_accel'].fillna(0.0).values, index=close.index)
    print('  Timing signals done.')

    # レバレッジ系列
    L_s2 = compute_L_s2_vz_gated(ret, vz, **S2_FIXED)
    L_p2 = compute_L_vol_target(ret, target_vol=0.8, n=20, l_min=1.0, l_max=7.0, step=0.5)
    L_s4 = compute_L_s4_relvol(ret, vz, l_base=7.0, k_rel=2.0, l_min=1.0, step=0.5)
    print('  Leverage series done.')

    return dict(
        close=close, ret=ret, dates=dates, dates_dt=dates_dt,
        n=n, n_years=n_years, sofr=sofr,
        gold_2x_sd=gold_2x_sd, bond_3x_sd=bond_3x_sd,
        gold_2x_p=gold_2x_p, bond_3x_p=bond_3x_p,
        raw_a2=raw_a2, vz=vz,
        lev_A=lev_A, wn_A=wn_A, wg_A=wg_A, wb_A=wb_A,
        n_tr=n_tr, n_trades_yr=n_tr / n_years,
        lt_sig=lt_sig, lt_bias=lt_bias, lev_mod=lev_mod,
        hy_s=hy_s, cpi_yoy=cpi_yoy, cpi_acc=cpi_acc,
        L_s2=L_s2, L_p2=L_p2, L_s4=L_s4,
    )


# ---------------------------------------------------------------------------
# 11戦略の FULL NAV 構築
# ---------------------------------------------------------------------------

def build_all_navs(assets: dict) -> dict:
    a = assets
    close, dates, sofr = a['close'], a['dates'], a['sofr']
    lev_A, wn_A, wg_A, wb_A = a['lev_A'], a['wn_A'], a['wg_A'], a['wb_A']
    lev_mod = a['lev_mod']
    gold_2x_sd, bond_3x_sd = a['gold_2x_sd'], a['bond_3x_sd']
    gold_2x_p,  bond_3x_p  = a['gold_2x_p'],  a['bond_3x_p']

    # P系 gate
    g_hy   = build_hy_gate(a['hy_s'])
    g_cpi  = build_cpi_gate(a['cpi_yoy'], a['cpi_acc'])
    g_corr = build_corr_gate(close, bond_3x_p, gold_2x_p)

    navs = {}

    # 1. S2+LT2 (CURRENT BEST)
    navs['S2+LT2'] = build_nav_strategy(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x_sd, bond_3x_sd, sofr,
        nas_mode='CFD', cfd_leverage=a['L_s2'].values, cfd_spread=CFD_SPREAD_LOW,
    )

    # 2. S2 (baseline, no LT2)
    navs['S2'] = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x_sd, bond_3x_sd, sofr,
        nas_mode='CFD', cfd_leverage=a['L_s2'].values, cfd_spread=CFD_SPREAD_LOW,
    )

    # 3. P2 (vol-target)
    navs['P2'] = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x_sd, bond_3x_sd, sofr,
        nas_mode='CFD', cfd_leverage=a['L_p2'].values, cfd_spread=CFD_SPREAD_LOW,
    )

    # 4. S4 (RelVol)
    navs['S4'] = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x_sd, bond_3x_sd, sofr,
        nas_mode='CFD', cfd_leverage=a['L_s4'].values, cfd_spread=CFD_SPREAD_LOW,
    )

    # 5. CFD 7x 固定
    navs['CFD7x'] = build_nav_strategy(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x_sd, bond_3x_sd, sofr,
        nas_mode='CFD', cfd_leverage=7.0, cfd_spread=CFD_SPREAD_LOW,
    )

    # 6. DHA (TQQQ, Scenario D)
    navs['DHA'] = build_nav(
        close, lev_A, wn_A, wg_A, wb_A, dates,
        gold_2x_sd, bond_3x_sd, sofr_daily=sofr, apply_tqqq_sofr=True,
    )

    # 7. BH 1x
    navs['BH1x'] = (1 + a['ret']).cumprod()

    # 8. P02 (Dyn×CPI, bond_gate=corr)
    wn8, wg8, wb8 = apply_gates(wn_A, nas_gate=g_cpi.values, bond_gate=g_corr.values)
    navs['P02'] = build_nav_strategy(
        close, lev_A, wn8, wg8, wb8, dates,
        gold_2x_p, bond_3x_p, sofr,
        nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002,
    )

    # 9. P05 (HY×CPI, bond_gate=None)
    g_nas_p05 = np.clip(g_hy.values * g_cpi.values, 0.2, 1.0)
    wn9, wg9, wb9 = apply_gates(wn_A, nas_gate=g_nas_p05, bond_gate=None)
    navs['P05'] = build_nav_strategy(
        close, lev_A, wn9, wg9, wb9, dates,
        gold_2x_p, bond_3x_p, sofr,
        nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002,
    )

    # 10. P01 (Dyn×HY, bond_gate=corr)
    wn10, wg10, wb10 = apply_gates(wn_A, nas_gate=g_hy.values, bond_gate=g_corr.values)
    navs['P01'] = build_nav_strategy(
        close, lev_A, wn10, wg10, wb10, dates,
        gold_2x_p, bond_3x_p, sofr,
        nas_mode='TQQQ', cfd_leverage=3.0, cfd_spread=0.002,
    )

    # 11. DHA+LT2 (TQQQ + LT2 overlay)
    navs['DHA+LT2'] = build_nav(
        close, lev_mod, wn_A, wg_A, wb_A, dates,
        gold_2x_sd, bond_3x_sd, sofr_daily=sofr, apply_tqqq_sofr=True,
    )

    return navs


# ---------------------------------------------------------------------------
# Window 生成 (カレンダー年アンカー方式)
# ---------------------------------------------------------------------------

def generate_windows(dates: pd.Series,
                     eval_start: str = EVAL_START,
                     eval_end: str = EVAL_END,
                     window_days: int = WINDOW_DAYS,
                     step_days: int = STEP_DAYS) -> list:
    """カレンダー年アンカー: 各年1月1日(≒最初の営業日)〜12月31日(≒最後の営業日)で窓を生成。
    252日ステップの Dec-to-Dec ズレ問題を回避し、IS/OOS 境界と年次窓を整合させる。
    短窓 (n_days < 0.8 × 252 = 201日) は short_flag=True。"""
    eval_start_ts = pd.Timestamp(eval_start)
    eval_end_ts   = pd.Timestamp(eval_end)
    first_year    = eval_start_ts.year
    last_year     = eval_end_ts.year

    windows = []
    for year in range(first_year, last_year + 1):
        yr_start_ts = max(pd.Timestamp(f'{year}-01-01'), eval_start_ts)
        yr_end_ts   = min(pd.Timestamp(f'{year}-12-31'), eval_end_ts)

        # 最初の営業日 >= yr_start
        mask_s = dates >= yr_start_ts
        if not mask_s.any():
            break
        s_idx = int(mask_s.values.argmax())

        # 最後の営業日 <= yr_end
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
            short_flag = n_days < int(window_days * 0.8),  # < 201日
        ))

    return windows


# ---------------------------------------------------------------------------
# 窓内指標計算
# ---------------------------------------------------------------------------

def compute_window_metrics(nav: pd.Series, window: dict,
                            trading_days: int = TRADING_DAYS) -> dict:
    s = window['start_idx']
    e = window['end_idx'] + 1  # exclusive
    nav_arr = nav.iloc[s:e].values.astype(float)
    n_days  = len(nav_arr)

    if n_days < 5 or nav_arr[0] <= 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, MaxDD=np.nan,
                    Vol=np.nan, PosDay_pct=np.nan, n_days=n_days)

    # 日次リターン (n_days - 1 個)
    daily_ret = np.diff(nav_arr) / nav_arr[:-1]

    # CAGR (年率化)
    years = n_days / trading_days
    cagr  = (nav_arr[-1] / nav_arr[0]) ** (1.0 / years) - 1

    # Sharpe
    r_std = np.std(daily_ret, ddof=1)
    sharpe = (np.mean(daily_ret) / r_std * np.sqrt(trading_days)
              if r_std > 1e-10 else np.nan)

    # MaxDD
    running_max = np.maximum.accumulate(nav_arr)
    max_dd = float((nav_arr / running_max - 1).min())

    # Vol & PosDay
    vol      = float(r_std * np.sqrt(trading_days))
    pos_pct  = float(np.mean(daily_ret > 0))

    return dict(CAGR=float(cagr), Sharpe=float(sharpe), MaxDD=max_dd,
                Vol=vol, PosDay_pct=pos_pct, n_days=n_days)


# ---------------------------------------------------------------------------
# 統計集計
# ---------------------------------------------------------------------------

def compute_summary_stats(per_df: pd.DataFrame) -> dict:
    """短窓 (short_flag=True) は全指標から除外して集計。
    EVALUATION_STANDARD v1.1 準拠: 標準7指標 + WFA補助2指標 (§3.9, §3.10) のみ出力。
    IS窓: start_date < OOS_START (straddle年=2021を含む)
    postIS窓: start_date >= OOS_START AND NOT short_flag"""
    valid = per_df[~per_df['short_flag']].copy()
    cagrs   = valid['CAGR'].dropna().values
    sharpes = valid['Sharpe'].dropna().values
    n       = len(cagrs)

    if n == 0:
        return {}

    mean_c  = float(np.mean(cagrs))
    med_c   = float(np.median(cagrs))
    std_c   = float(np.std(cagrs, ddof=1)) if n > 1 else np.nan
    se      = std_c / np.sqrt(n) if (not np.isnan(std_c) and n > 1) else np.nan
    t_crit  = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else np.nan

    # §3.9 WFA_CI95_lo / hi
    ci95_lo = mean_c - t_crit * se if not np.isnan(se) else np.nan
    ci95_hi = mean_c + t_crit * se if not np.isnan(se) else np.nan
    t_stat  = mean_c / se if (not np.isnan(se) and se > 0) else np.nan
    t_pval  = float(stats.t.sf(t_stat, df=n - 1)) if not np.isnan(t_stat) else np.nan

    mean_s  = float(np.mean(sharpes)) if len(sharpes) > 0 else np.nan
    std_s   = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else np.nan

    # §3.10 WFA_WFE
    oos_start_ts = pd.Timestamp(OOS_START_REF)
    is_mask   = valid['start_date'] < oos_start_ts
    post_mask = valid['start_date'] >= oos_start_ts
    is_cagrs   = valid.loc[is_mask,   'CAGR'].dropna().values
    post_cagrs = valid.loc[post_mask, 'CAGR'].dropna().values
    mean_is   = float(np.mean(is_cagrs))   if len(is_cagrs)   > 0 else np.nan
    mean_post = float(np.mean(post_cagrs)) if len(post_cagrs) > 0 else np.nan
    wfe_post  = (mean_post / mean_is) if (not np.isnan(mean_is) and mean_is != 0) else np.nan

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
    )


# ---------------------------------------------------------------------------
# 判定基準
# ---------------------------------------------------------------------------

def evaluate_criteria(summary: dict) -> tuple:
    """EVALUATION_STANDARD v1.1: α (§3.9) + β (§3.10) の2基準のみ。
    α PASS + β PASS → PASS
    α PASS + β FAIL → WARN  (IS過学習の兆候)
    α FAIL           → FAIL"""
    crit_alpha = (summary.get('WFA_CI95_lo', -1) > 0 and
                  summary.get('t_pvalue', 1.0) < 0.05)
    wfe    = summary.get('WFA_WFE', np.nan)
    n_post = summary.get('n_windows_postIS', 0)
    if n_post < 3:
        crit_beta = True   # postIS窓不足のため N/A → PASS扱い
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
# CSV 保存
# ---------------------------------------------------------------------------

def save_results_csv(all_rows: list, all_summaries: list, base_dir: str) -> tuple:
    pw_path = os.path.join(base_dir, 'g1_wfa_per_window.csv')
    sm_path = os.path.join(base_dir, 'g1_wfa_summary.csv')

    pd.DataFrame(all_rows).to_csv(pw_path, index=False, float_format='%.6f')
    pd.DataFrame(all_summaries).to_csv(sm_path, index=False, float_format='%.6f')

    return pw_path, sm_path


# ---------------------------------------------------------------------------
# Markdown レポート生成
# ---------------------------------------------------------------------------

def _fp(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v * 100:+.{d}f}%'


def _ff(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:+.{d}f}'


def generate_md_report(results: dict, sanity: dict, windows: list,
                        base_dir: str) -> str:
    """EVALUATION_STANDARD v1.1 準拠: 9指標 (§3標準7 + WFA補助2) でスクロールなし表示。"""
    lines = []
    lines += [
        f'# G1: Walk-Forward Analysis — 11戦略 × {len(windows)}窓',
        '',
        f'**実行日**: {TODAY}',
        f'**EVALUATION_STANDARD**: v1.1',
        '**目的**: 非重複1年窓 (252日) による11戦略の真のOOS性能統計推定。',
        '       単一OOS区間(5年)の統計的脆弱性を解消する (§2.3 補助ロバストネス確認)。',
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
        '',
        '**判定基準 (EVALUATION_STANDARD v1.1 §3.9/§3.10)**',
        '',
        '| 基準 | 条件 | 参照 |',
        '|------|------|------|',
        '| α (統計的有意性) | WFA_CI95_lo > 0 AND t_p < 0.05 | §3.9 |',
        '| β (IS↔OOS安定) | WFA_WFE ∈ [0.5, 2.0] | §3.10 |',
        '| **総合** | α+β PASS → PASS / α のみ → WARN / α FAIL → FAIL | |',
        '',
        '---', '',
    ]

    # メインサマリ表 (コンパクト: スクロールなし)
    lines += ['## 2. 戦略別サマリ (mean_CAGR降順)', '']
    lines += [
        '| # | 戦略 | mean_CAGR | CI95[lo,hi] | mean_Sharpe | WFE | 判定 |',
        '|---|------|----------:|------------:|------------:|----:|:---:|',
    ]

    ranked = sorted(results.items(),
                    key=lambda kv: kv[1]['summary'].get('mean_CAGR', -99),
                    reverse=True)
    for rank, (sid, res) in enumerate(ranked, 1):
        s = res['summary']
        v = res['verdict']
        label = STRATEGY_LABELS.get(sid, sid)
        ci_lo = s.get('WFA_CI95_lo')
        ci_hi = s.get('WFA_CI95_hi')
        lines.append(
            f'| {rank} | {label} '
            f'| {_fp(s.get("mean_CAGR"))} '
            f'| [{_fp(ci_lo)}, {_fp(ci_hi)}] '
            f'| {_ff(s.get("mean_Sharpe"))} '
            f'| {_ff(s.get("WFA_WFE"))} '
            f'| **{v}** |'
        )
    lines += ['', '---', '']

    # 判定詳細 (α+β 2基準)
    lines += ['## 3. 判定詳細 (α+β 2基準)', '']
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
        a_mark = '✅' if cr.get('alpha') else '❌'
        b_mark = '✅' if cr.get('beta')  else '❌'
        lines.append(
            f'| {STRATEGY_LABELS.get(sid, sid)} '
            f'| {a_mark} {_fp(s.get("WFA_CI95_lo"))} '
            f'| {_ff(s.get("t_pvalue"), d=4)} '
            f'| {b_mark} {_ff(s.get("WFA_WFE"))} '
            f'| **{v}** |'
        )
    lines += ['', '---', '']

    # IS vs postIS (WFA_WFE の根拠)
    lines += ['## 4. IS vs postIS — WFA_WFE 根拠 (§3.10)', '']
    lines += [
        '| 戦略 | n_IS | CAGR_IS | n_postIS | CAGR_postIS | WFA_WFE |',
        '|------|-----:|--------:|---------:|------------:|--------:|',
    ]
    for sid in STRATEGY_IDS:
        if sid not in results:
            continue
        s = results[sid]['summary']
        lines.append(
            f'| {STRATEGY_LABELS.get(sid, sid)} '
            f'| {s.get("n_windows_IS", "N/A")} '
            f'| {_fp(s.get("mean_CAGR_IS"))} '
            f'| {s.get("n_windows_postIS", "N/A")} '
            f'| {_fp(s.get("mean_CAGR_postIS"))} '
            f'| {_ff(s.get("WFA_WFE"))} |'
        )
    lines += ['', '---', '']

    # サニティチェック
    lines += ['## 5. サニティチェック', '']
    s2lt2_post = results.get('S2+LT2', {}).get('summary', {}).get('mean_CAGR_postIS', np.nan)
    bh_mean    = results.get('BH1x',   {}).get('summary', {}).get('mean_CAGR',       np.nan)
    s2lt2_ok   = (not np.isnan(s2lt2_post)) and abs(s2lt2_post - REF_S2LT2_CAGR_OOS) <= 0.15
    bh_ok      = (not np.isnan(bh_mean))    and abs(bh_mean    - REF_BH1X_CAGR_IS)   <= 0.05
    lines += [
        f'- {"✅" if s2lt2_ok else "⚠️"} S2+LT2 mean_CAGR_postIS = **{_fp(s2lt2_post)}**'
        f' (参照 +31.16%: 算術平均≠複利CAGR のため乖離は想定内)',
        f'- {"✅" if bh_ok else "⚠️"} BH1x mean_CAGR = **{_fp(bh_mean)}**'
        f' (参照 +11.13%, 許容 ±5pp)',
        '',
        '---', '',
    ]

    # 考察
    lines += [
        '## 6. 考察', '',
        '### 6.1 WFA_CI95_lo (§3.9) の読み方', '',
        '- CI95_lo > 0 ⇒ 49窓で「真の期待リターン > 0」が統計的に支持される（α PASS）。',
        '- CI95 の幅が広い戦略（高分散レバレッジ系）はサンプルリスクも大きい。',
        '',
        '### 6.2 WFA_WFE (§3.10) の読み方', '',
        '- WFE ≈ 1.0: IS/postIS で均一なパフォーマンス → 過学習なし。',
        '- WFE < 0.5: postIS が IS の半分以下 → IS 最適化の過剰フィット疑い（β WARN）。',
        '- WFE > 2.0: postIS が異常に良い → レジーム変化か幸運（要注意）。',
        '',
        '### 6.3 §3 標準指標との関係', '',
        '- 本 WFA レポートは §2.3 の補助ロバストネス確認。CAGR_OOS / Sharpe_OOS / MaxDD 等',
        '  の主要判断は `STRATEGY_COMPARISON_INTEGRATED_*.md` の §3 標準指標テーブルを参照。',
        '- WFA_CI95_lo と WFA_WFE は IS-OOS gap (§3.8) の統計的根拠を補強する位置付け。',
        '',
        '---', '',
        '## 7. 再現コマンド', '',
        '```',
        'python -X utf8 src/g1_wfa.py',
        '```', '',
        '---', '',
        f'*生成スクリプト: `src/g1_wfa.py`  '
        f'準拠: `EVALUATION_STANDARD.md v1.1`*',
        f'*参照: `CURRENT_BEST_STRATEGY.md`, `B1_S2_LT2_{TODAY}.md`*',
    ]

    md_text = '\n'.join(lines)
    md_path = os.path.join(base_dir, f'G1_WFA_{TODAY}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    return md_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    print('=' * 70)
    print('G1: Walk-Forward Analysis (11戦略 × 非重複1年窓)')
    print(f'実行日: {TODAY}')
    print('=' * 70)

    # S1: 共有資産ロード
    print('\n[S1] Loading shared assets...')
    assets = load_shared_assets(DATA_PATH)
    dates  = assets['dates']
    ret    = assets['ret']

    # S2: 全戦略 FULL NAV 構築
    print('\n[S2] Building NAVs for all 11 strategies...')
    navs = build_all_navs(assets)
    for sid in STRATEGY_IDS:
        print(f'  {sid}: NAV range [{navs[sid].min():.4f}, {navs[sid].max():.4f}]')

    # S3: Window 生成
    print('\n[S3] Generating windows...')
    windows = generate_windows(dates, EVAL_START, EVAL_END, WINDOW_DAYS, STEP_DAYS)
    print(f'  {len(windows)} windows: '
          f'{windows[0]["start_date"].date()} ... {windows[-1]["end_date"].date()}')
    short_wins = sum(w['short_flag'] for w in windows)
    print(f'  Short windows (< 80% of target): {short_wins}')

    # 重複なし assert
    for i in range(len(windows) - 1):
        assert windows[i]['end_idx'] < windows[i+1]['start_idx'], \
            f'Overlapping windows at {i} and {i+1}'
    print('  Non-overlap check: OK')

    # S5: 全戦略評価
    print('\n[S5] Evaluating all strategies across windows...')
    results      = {}
    all_pw_rows  = []
    all_sm_rows  = []

    for sid in STRATEGY_IDS:
        nav = navs[sid]
        per_rows = []
        for w in windows:
            m = compute_window_metrics(nav, w)
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

        sm_row = {'strategy': sid, 'verdict': verdict, **summary, **{f'crit_{k}': v for k, v in crits.items()}}
        all_sm_rows.append(sm_row)

        mean_c = summary.get('mean_CAGR', np.nan)
        ci_lo  = summary.get('WFA_CI95_lo', np.nan)
        ci_hi  = summary.get('WFA_CI95_hi', np.nan)
        tp     = summary.get('t_pvalue', np.nan)
        wfe    = summary.get('WFA_WFE', np.nan)
        print(
            f'  {sid:<10}: mean_CAGR={mean_c*100:+.1f}%  '
            f'CI95=[{ci_lo*100:+.1f}%, {ci_hi*100:+.1f}%]  '
            f't_p={tp:.3f}  WFE={wfe:+.2f}  => {verdict}'
        )

    # S6: サニティチェック
    print('\n[S6] Sanity checks...')
    s2lt2_post = results['S2+LT2']['summary'].get('mean_CAGR_postIS', np.nan)
    bh_mean    = results['BH1x']['summary'].get('mean_CAGR', np.nan)
    sanity = {
        'S2+LT2 mean_CAGR_postIS': f'{s2lt2_post*100:+.2f}% (ref +31.16%)',
        'BH1x mean_CAGR_all':      f'{bh_mean*100:+.2f}% (ref +11.13%)',
    }
    # 算術平均 ≠ 複利CAGR のため許容幅を緩める (±15pp / ±5pp)
    s2_ok = abs(s2lt2_post - REF_S2LT2_CAGR_OOS) <= 0.15
    bh_ok = abs(bh_mean - REF_BH1X_CAGR_IS) <= 0.05
    print(f'  S2+LT2 postIS CAGR: {s2lt2_post*100:+.2f}%  {"OK" if s2_ok else "WARN"} '
          f'(参照+31.16%: 算術平均≠複利CAGRのため乖離は想定内)')
    print(f'  BH1x mean_CAGR:     {bh_mean*100:+.2f}%  {"OK" if bh_ok else "WARN"}')

    # S7: 全戦略コンソールサマリ (EVALUATION_STANDARD v1.1 準拠9指標)
    print('\n[S7] Per-strategy console summary (9 unified metrics):')
    print('=' * 90)
    print(f'{"Strategy":<12} {"mean_CAGR":>10} {"CI95_lo":>9} {"CI95_hi":>9}'
          f' {"t_p":>7} {"Sharpe":>7} {"WFE":>7} {"Verdict":>8}')
    print('-' * 90)
    for sid in STRATEGY_IDS:
        s = results[sid]['summary']
        v = results[sid]['verdict']
        print(
            f'{sid:<12}'
            f' {s.get("mean_CAGR",0)*100:>+9.1f}%'
            f' {s.get("WFA_CI95_lo",np.nan)*100:>+8.1f}%'
            f' {s.get("WFA_CI95_hi",np.nan)*100:>+8.1f}%'
            f' {s.get("t_pvalue",np.nan):>7.3f}'
            f' {s.get("mean_Sharpe",np.nan):>7.3f}'
            f' {s.get("WFA_WFE",np.nan):>7.2f}'
            f' {v:>8}'
        )
    print('=' * 90)

    # S8: CSV 保存
    print('\n[S8] Saving CSV...')
    pw_path, sm_path = save_results_csv(all_pw_rows, all_sm_rows, BASE)
    print(f'  Per-window: {pw_path}')
    print(f'  Summary:    {sm_path}')

    # S9: MD レポート
    print('[S9] Generating Markdown report...')
    md_path = generate_md_report(results, sanity, windows, BASE)
    print(f'  Report: {md_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
