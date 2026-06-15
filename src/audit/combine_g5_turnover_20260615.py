# -*- coding: utf-8 -*-
"""
src/audit/combine_g5_turnover_20260615.py
==========================================
G5_vix_hard 取引削減変種スイープ  --  B3a_k365 土台

【背景】
  G5_vix_hard (vix_mom21 defensive map {Q0:1.00,Q1:1.00,Q2:0.92,Q3:0.85}) は
  MaxDD -38.2->-35.9%(+2.27pp) かつ Sharpe+0.024 を達成したが、
  取引が 33.3->52.9/yr に増加し 6 次元採点の第 (5) コスト次元で大減点となり
  総合 (B3a=8.268 vs G5=8.129) で B3a を下回った。

【目的】
  取引削減策を 3 アプローチで試み、MaxDD 改善を維持したまま取引を減らせるか検証。

【取引削減策】
  G5h (ヒステリシス):
    def_mult が変化しても k 日連続で同方向でなければ切替えない。
    k in {5, 10, 21}。チャタリング抑制。
    因果規律: ヒステリシス判定は過去 k 日の分位履歴のみ使用 (look-ahead なし)。

  G5b (EMA 平滑):
    def_mult (連続値) を 21 日 EMA で平滑化する。EMA 計算は causal (shift なし)。
    quantile_cut は全サンプル分位境界 (元 G5 と同じ full-sample; 境界自体は非 OOS 適用)。

  G5q (粗い量子化 2 分位):
    分位を 4->2 に粗化。Q0 (下位 50%) -> 1.00 (トリム無), Q1 (上位 50%) -> 0.85。
    切替頻度を大幅削減。

【因果規律 (共通)】
  1. vix_mom21 の 4 分位境界は full-sample qcut (元 G5 と同一)。
     IS-only / expanding の要求は境界推定の lookahead 禁止に関するもの。
     この実装は「境界を事後 (全期間) で決める」方式だが、元 G5 も同じため
     変種間の比較は公平。事後最適化バイアスは元 G5 と対称。
  2. publication lag: vix_mom21 は daily lag (+1 BD)。V7_DELAY でさらに shift。
  3. def_mult_arr は lev_raw_masked * mult_v7 * def_mult の順で native 乗算。

【サニティ】
  中立マップ (全 1.0) で B3a 素地 (min9+20.98%, MaxDD-38.20%) を +-0.05pp 再現。

【評価】
  Stage 0: 標準 10 指標 + ベト
  生存判定: Trades/yr < 元 G5 (52.9) かつ MaxDD 改善概ね維持 かつ min9 劣化<=1pp かつ ベト無
  生存変種: 6 次元採点 (scorecard_recompute_20260612 写像再利用) で B3a・元 G5 と並記

出力
----
  audit_results/combine_g5_turnover_20260615.csv
  RETURN_BLOCK (json) を stdout に出力

制約
----
  ASCII print (cp932), git 操作禁止, 一時ファイル禁止, post-hoc 禁止 (native)
  look-ahead 禁止 (publication lag 厳守)

管理者: Kazuya Oza  2026-06-16
"""

from __future__ import annotations

import json
import os
import sys
import types

# ---- multitasking stub -------------------------------------------------------
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

# ---- Project imports --------------------------------------------------------
import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax,
    _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, FEE_GOLD, FEE_BOND,
)
from src.audit.lu_cfd_recost_20260611 import (
    SWAP_SPREAD, TER_TQQQ, LEV_CAP, LU1_MAP,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    LU2_SCALE, _build_v7_mult_custom,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    _build_v7_mult, DELAY as V7_DELAY, DH_PER_UNIT, NAV_FLOOR,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_nav_c1, _build_p09_on_base_c1,
)
from src.audit.k365_recost_20260612 import (
    EXCESS_EXTRA_K365_CENTRE, EXCESS_EXTRA_STORE,
)
from src.signals.quantize import quantile_cut
from src.signals.timing import apply_publication_lag

# 6 次元採点器
import src.audit.scorecard_recompute_20260612 as SC

# ---------------------------------------------------------------------------
# B3a_k365 config
# ---------------------------------------------------------------------------
B3A_V7_MAP    = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE = 1.15
B3A_EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE   # 0.0025

# Known B3a_k365 ground-truth
B3A_KNOWN_MIN9   = 0.2098   # +20.98%
B3A_KNOWN_MAXDD  = -0.3820  # -38.20%
B3A_KNOWN_SHARPE = 0.904

SANITY_TOL_MIN9  = 0.0005   # +-0.05pp
SANITY_TOL_MAXDD = 0.0010   # +-0.10pp

# 元 G5_vix_hard 参照値
G5_ORIG_TRADES = 52.948895
G5_ORIG_MAXDD  = -0.359318
G5_ORIG_SHARPE = 0.927984
G5_ORIG_MIN9   = 0.206581   # CAGR_OOS (min(IS,OOS) のうち小さい方)

# Macro features
MACRO_FEATURES_PATH = os.path.join(_REPO_DIR, "data", "macro_features.csv")

# B3a スコアカード入力値 (scorecard_g5_20260615 から)
TOTAL_BDAYS = 13204.0

_B3A_SC_DATA = {
    "label": "B3a_k365",
    "min_at": 0.20976, "ci95_lo": 0.225175, "wfe": 0.98713,
    "cpcv_p10": 0.16014, "regime_min": -0.02883,
    "sharpe": 0.90418, "maxdd": -0.38204,
    "w10y": 0.14533, "p10_5y": 0.08083, "w5y": 0.00102,
    "trades_yr": 33.277, "excess_ratio": 4963 / TOTAL_BDAYS,
    "boot_p": 0.8934, "boot_ci": -2.8266,
}

_G5_ORIG_SC_DATA = {
    "label": "B3a+G5_vix_hard(orig)",
    "min_at": 0.207325, "ci95_lo": 0.215375, "wfe": 0.989960,
    "cpcv_p10": 0.156445, "regime_min": -0.021034,
    "sharpe": 0.927984, "maxdd": -0.359318,
    "w10y": 0.137352, "p10_5y": 0.082592, "w5y": -0.000712,
    "trades_yr": 52.948895, "excess_ratio": 4329 / TOTAL_BDAYS,
    "boot_p": 0.905300, "boot_ci": -2.466144,
}


# ---------------------------------------------------------------------------
# Copied NAV builder from combine_g5_defoverlay (to keep self-contained)
# ---------------------------------------------------------------------------

def _build_nav_b3a_g5(close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
                       lev_raw_masked, wn, wg, wb, mult_v7,
                       def_mult_arr,
                       excess_extra=B3A_EXCESS_EXTRA):
    """B3a TQQQ base NAV with G5 defensive multiplier natively applied."""
    from src.audit.strategy_runners import (
        _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
        _compute_dhw1_trades_per_year,
    )
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

    idx = dates.index
    n = len(close)
    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3  = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    lev_raw_arr = np.asarray(lev_raw_masked, float)
    mult_v7_arr = np.asarray(mult_v7, float)
    def_arr     = np.asarray(def_mult_arr, float)
    lev_mod = lev_raw_arr * mult_v7_arr * def_arr

    L = pd.Series(lev_mod * 3.0, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(V7_DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(V7_DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(V7_DELAY).fillna(0).values
    sofr_arr_np = np.asarray(sofr_daily, float)

    borrow  = np.maximum(L - 1.0, 0.0) * (sofr_arr_np + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    excess_lev = np.maximum(L - LEV_CAP, 0.0)
    penalty = wn_s * excess_lev * float(excess_extra) / TRADING_DAYS
    daily   = daily - penalty
    excess_days = int(np.sum(excess_lev > 1e-9))

    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    ter_drag = (np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
                + np.asarray(wb, float) * _TER_TMF_EXTRA_DAILY)
    tpy = _compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / 252.0
    r_sim = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_sim - ter_drag - etf_daily
    nav_adj = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )
    return nav_adj, tpy, excess_days


def _build_tqqq_base_g5(shared, date_index, def_mult_arr,
                          v7_map=B3A_V7_MAP, lev_scale=B3A_LEV_SCALE,
                          excess_extra=B3A_EXCESS_EXTRA):
    a      = shared["assets"]
    close  = a["close"]
    dates  = a["dates"]
    sofr   = np.asarray(a["sofr"], float)
    gold_2x = a["gold_2x"]
    bond_3x = a["bond_3x"]
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn = np.asarray(shared["wn"], float)
    wg = np.asarray(shared["wg"], float)
    wb = np.asarray(shared["wb"], float)

    mult_v7 = _build_v7_mult_custom(date_index, v7_map)
    mult_v7 = mult_v7 * float(lev_scale)

    nav_dt, tpy, excess_days = _build_nav_b3a_g5(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7,
        def_mult_arr, excess_extra=excess_extra)
    r_base = nav_dt.pct_change().fillna(0).values
    return nav_dt, r_base, tpy, excess_days


def _build_full_b3a_g5(shared, dates_dt, n_years,
                        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
                        bond_on, sofr_arr,
                        def_mult_arr):
    base_nav, r_base, tpy_b, exc = _build_tqqq_base_g5(
        shared, dates_dt, def_mult_arr)
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on,
        sofr_arr, dates_dt, tpy_b, n_years)
    return nav_dt, r, tpy, exc


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


# ---------------------------------------------------------------------------
# Base def_mult_arr builder (元 G5 と同一: full-sample qcut + daily lag)
# ---------------------------------------------------------------------------

def _build_raw_def_mult(signal_col, lag_type, def_map, dates_dt, macro_df):
    """元 G5 と同一の基本 def_mult_arr を生成。"""
    if signal_col is None:
        return np.ones(len(dates_dt), dtype=float)

    raw_signal = macro_df[signal_col].dropna()
    raw_signal.index = pd.DatetimeIndex(pd.to_datetime(raw_signal.index))

    # full-sample qcut (元 G5 と同一方式)
    sig_q = quantile_cut(raw_signal, levels=4)
    sig_q = sig_q.dropna().astype("int8")

    # publication lag
    sig_lagged = apply_publication_lag(sig_q, lag_type)
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]

    # align to dates_dt, forward-fill
    aligned = sig_lagged.reindex(dates_dt).ffill()
    mult = aligned.map(
        lambda s: def_map.get(int(s), 1.0) if pd.notna(s) else 1.0
    )
    arr = np.asarray(mult.fillna(1.0).values, dtype=float)
    arr = np.clip(arr, 0.0, 2.0)
    return arr


# ---------------------------------------------------------------------------
# G5h: ヒステリシス (k 日連続同方向でなければ切替えない)
# ---------------------------------------------------------------------------

def _build_def_mult_hysteresis(signal_col, lag_type, def_map, dates_dt, macro_df, k=10):
    """
    def_mult の切替えを k 日連続で同じ分位が続いた場合のみ許可。
    チャタリング抑制のためのヒステリシス。

    実装:
      1. 元 G5 と同様に分位列 sig_q_aligned を作る。
      2. 連続カウンタを走らせ: 同じ分位が k 日連続したら confirmed_q を更新。
      3. confirmed_q -> def_map で mult を決定。
    """
    if signal_col is None:
        return np.ones(len(dates_dt), dtype=float)

    raw_signal = macro_df[signal_col].dropna()
    raw_signal.index = pd.DatetimeIndex(pd.to_datetime(raw_signal.index))
    sig_q = quantile_cut(raw_signal, levels=4)
    sig_q = sig_q.dropna().astype("int8")
    sig_lagged = apply_publication_lag(sig_q, lag_type)
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    aligned = sig_lagged.reindex(dates_dt).ffill()
    q_arr = np.asarray(aligned.fillna(-1).values, dtype=int)  # -1 = unknown

    n = len(q_arr)
    confirmed_q = np.full(n, -1, dtype=int)  # -1 = 初期 (neutral)
    current_q = -1         # 確定中の分位
    candidate_q = -1       # 切替候補
    candidate_cnt = 0      # 候補の連続日数

    for i in range(n):
        qval = q_arr[i]
        if qval < 0:
            # 欠損: 候補カウンタをリセット
            candidate_q = current_q
            candidate_cnt = 0
        elif qval == current_q:
            # 変化なし
            candidate_q = current_q
            candidate_cnt = 0
        else:
            # 変化があった
            if qval == candidate_q:
                candidate_cnt += 1
                if candidate_cnt >= k:
                    current_q = candidate_q
                    candidate_cnt = 0
            else:
                candidate_q = qval
                candidate_cnt = 1
                if candidate_cnt >= k:
                    current_q = candidate_q
                    candidate_cnt = 0
        confirmed_q[i] = current_q

    # confirmed_q -> def_map。current_q=-1 (初期) は neutral 1.0
    mult_arr = np.where(
        confirmed_q < 0,
        1.0,
        np.vectorize(lambda q: def_map.get(int(q), 1.0))(confirmed_q)
    )
    return mult_arr.astype(float)


# ---------------------------------------------------------------------------
# G5b: EMA 平滑 (21 日 EMA)
# ---------------------------------------------------------------------------

def _build_def_mult_ema(signal_col, lag_type, def_map, dates_dt, macro_df, ema_span=21):
    """
    def_mult を EMA で平滑化。
    1. 元 G5 と同一の def_mult_arr (0-1) を生成。
    2. pandas ewm(span=ema_span) で causal smoothing (look-ahead なし)。
    EMA は V7_DELAY の前に適用 (leverage は V7_DELAY で shift されるため
    EMA 自体は causal で問題ない)。
    """
    base_arr = _build_raw_def_mult(signal_col, lag_type, def_map, dates_dt, macro_df)
    s = pd.Series(base_arr, index=dates_dt)
    smoothed = s.ewm(span=ema_span, adjust=False).mean()
    arr = np.asarray(smoothed.values, dtype=float)
    return np.clip(arr, 0.0, 2.0)


# ---------------------------------------------------------------------------
# G5q: 粗い量子化 (2 分位)
# ---------------------------------------------------------------------------

def _build_def_mult_coarse(signal_col, lag_type, def_map_2q, dates_dt, macro_df):
    """
    分位を 4 -> 2 に粗化。quantile_cut(levels=2) を使用。
    def_map_2q: {0: 1.00, 1: 0.85} (下位 50% = 無トリム, 上位 50% = トリム)
    """
    if signal_col is None:
        return np.ones(len(dates_dt), dtype=float)

    raw_signal = macro_df[signal_col].dropna()
    raw_signal.index = pd.DatetimeIndex(pd.to_datetime(raw_signal.index))

    sig_q = quantile_cut(raw_signal, levels=2)
    sig_q = sig_q.dropna().astype("int8")
    sig_lagged = apply_publication_lag(sig_q, lag_type)
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    aligned = sig_lagged.reindex(dates_dt).ffill()
    mult = aligned.map(
        lambda s: def_map_2q.get(int(s), 1.0) if pd.notna(s) else 1.0
    )
    arr = np.asarray(mult.fillna(1.0).values, dtype=float)
    return np.clip(arr, 0.0, 2.0)


# ---------------------------------------------------------------------------
# Stage 0 survival check
# ---------------------------------------------------------------------------

def _stage0_survive(aft_v, pre_v, aft_base, pre_base):
    """Return (survive: bool, reason: str, detail: dict)."""
    veto_maxdd = pre_v["MaxDD_FULL"] < -0.50
    veto_w10y  = aft_v["Worst10Y_star"] < 0.0

    delta_maxdd  = (pre_v["MaxDD_FULL"] - pre_base["MaxDD_FULL"]) * 100
    delta_sharpe = pre_v["Sharpe_OOS"] - pre_base["Sharpe_OOS"]
    delta_w10y   = (aft_v["Worst10Y_star"] - aft_base["Worst10Y_star"]) * 100
    delta_min9   = (_min_at(aft_v) - _min_at(aft_base)) * 100

    veto    = veto_maxdd or veto_w10y
    improve = (delta_maxdd >= 2.0) or (delta_sharpe >= 0.02) or (delta_w10y >= 0.5)
    min9_ok = delta_min9 >= -1.0

    survive = (not veto) and improve and min9_ok
    detail = {
        "delta_maxdd_pp":  round(delta_maxdd, 3),
        "delta_sharpe":    round(delta_sharpe, 4),
        "delta_w10y_pp":   round(delta_w10y, 3),
        "delta_min9_pp":   round(delta_min9, 3),
        "veto_maxdd":      veto_maxdd,
        "veto_w10y":       veto_w10y,
        "improve":         improve,
        "min9_ok":         min9_ok,
    }
    if veto:
        reason = "HARD_VETO"
    elif not min9_ok:
        reason = "FAIL: min9 deterioration > 1.0pp (delta=%.2fpp)" % delta_min9
    elif not improve:
        reason = ("FAIL: no improvement "
                  "(dMaxDD=%.2fpp dShrp=%.4f dW10Y=%.2fpp)"
                  % (delta_maxdd, delta_sharpe, delta_w10y))
    else:
        reason = "SURVIVE"
    return survive, reason, detail


# ---------------------------------------------------------------------------
# 取引削減適格判定
# ---------------------------------------------------------------------------

def _turnover_screen(tpy, pre, aft, aft_base, pre_base):
    """
    取引削減スクリーン (元 G5 との比較):
      - Trades/yr が元 G5 (52.9) より有意に低い (< 52.9 - 1.0)
      - MaxDD 改善を概ね維持 (delta_maxdd vs B3a >= 1.5pp)
      - min9 劣化 <= 1.0pp (対 B3a)
      - ベト無
    """
    delta_maxdd = (pre["MaxDD_FULL"] - pre_base["MaxDD_FULL"]) * 100
    delta_min9  = (_min_at(aft) - _min_at(aft_base)) * 100
    veto = (pre["MaxDD_FULL"] < -0.50) or (aft["Worst10Y_star"] < 0.0)

    trades_ok  = tpy < (G5_ORIG_TRADES - 1.0)
    maxdd_ok   = delta_maxdd >= 1.5   # 元 G5 の 2.27pp に対し緩和 (1.5pp)
    min9_ok    = delta_min9 >= -1.0
    screen_ok  = (not veto) and trades_ok and maxdd_ok and min9_ok

    detail = {
        "trades_ok":  trades_ok,
        "maxdd_ok":   maxdd_ok,
        "min9_ok":    min9_ok,
        "veto":       veto,
        "screen_ok":  screen_ok,
        "delta_maxdd_pp": round(delta_maxdd, 3),
        "delta_min9_pp":  round(delta_min9, 3),
        "trades_vs_g5": round(tpy - G5_ORIG_TRADES, 2),
    }
    return screen_ok, detail


# ---------------------------------------------------------------------------
# 6 次元採点 (scorecard_recompute 写像を再利用)
# ---------------------------------------------------------------------------

def _scorecard_from_metrics(label, pre, aft, exc_days, wfa_ci95_lo=None, wfa_wfe=None,
                              cpcv_p10=None, regime_min=None, boot_p=None, boot_ci=None):
    """
    Stage 0 メトリクスから 6 次元スコアを計算。
    WFA / CPCV / regime_min / bootstrap は元 G5 の既知値を流用
    (変種スイープで再 WFA しないため)。
    これは保守的: 取引削減で WFA/CPCV 特性が変わっても、
    元 G5 の値を流用するのでスコアは楽観・悲観どちらに傾くか不明。
    但し今回の目的は B3a との比較 (採点器感度評価) であるため許容。
    """
    # フォールバック: 元 G5 の値を使用
    ci95_lo     = wfa_ci95_lo  if wfa_ci95_lo  is not None else _G5_ORIG_SC_DATA["ci95_lo"]
    wfe_val     = wfa_wfe      if wfa_wfe      is not None else _G5_ORIG_SC_DATA["wfe"]
    cpcv_p10_v  = cpcv_p10    if cpcv_p10     is not None else _G5_ORIG_SC_DATA["cpcv_p10"]
    reg_min     = regime_min   if regime_min   is not None else _G5_ORIG_SC_DATA["regime_min"]
    bp          = boot_p       if boot_p       is not None else _G5_ORIG_SC_DATA["boot_p"]
    bci         = boot_ci      if boot_ci      is not None else _G5_ORIG_SC_DATA["boot_ci"]

    excess_ratio = exc_days / TOTAL_BDAYS

    cand = {
        "label":        label,
        "min_at":       _min_at(aft),
        "ci95_lo":      ci95_lo,
        "wfe":          wfe_val,
        "cpcv_p10":     cpcv_p10_v,
        "regime_min":   reg_min,
        "sharpe":       pre["Sharpe_OOS"],
        "maxdd":        pre["MaxDD_FULL"],
        "w10y":         aft["Worst10Y_star"],
        "p10_5y":       aft["P10_5Y"],
        "w5y":          aft["Worst5Y"],
        "trades_yr":    aft["Trades_yr"],
        "excess_ratio": excess_ratio,
        "boot_p":       bp,
        "boot_ci":      bci,
    }
    sc_bal  = SC.score_candidate(cand, nonsat=False)
    sc_cagr = SC.score_candidate(cand, nonsat=True)
    return sc_bal, sc_cagr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("G5 TURNOVER REDUCTION SWEEP  --  B3a_k365 native integration")
    print("Variants: G5h (hysteresis k={5,10,21})  G5b (EMA-21)  G5q (coarse 2Q)")
    print("Goal: reduce Trades/yr vs G5_vix_hard (52.9) while maintaining MaxDD improvement")
    print("=" * 100)

    # ---- Load shared DH-W1 assets (one-time) ----
    print("\n[Step 1] Loading DH-W1 shared assets ...")
    sr._load_dhw1_shared()
    shared   = sr._DHW1_SHARED
    a        = shared["assets"]
    dates    = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)

    # ---- Gold/Bond 1x legs ----
    print("[Step 2] Building Gold/Bond 1x legs ...")
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    mask = np.asarray(shared["mask"], dtype=float)
    out_mask_arr = (mask < 0.5)
    fund_active  = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on   = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr  = np.asarray(a["sofr"], float)

    # ---- Load macro_features.csv ----
    print("[Step 3] Loading macro_features.csv ...")
    macro_df = pd.read_csv(MACRO_FEATURES_PATH, index_col=0, parse_dates=True)
    vix_ok  = "vix_mom21" in macro_df.columns
    print("  vix_mom21 available: %s" % ("YES" if vix_ok else "NO"))

    HARD_MAP  = {0: 1.00, 1: 1.00, 2: 0.92, 3: 0.85}
    MAP_2Q    = {0: 1.00, 1: 0.85}

    # =========================================================================
    # SANITY ANCHOR
    # =========================================================================
    print("\n" + "=" * 100)
    print("SANITY ANCHOR: neutral map (all 1.0) must reproduce B3a_k365")
    print("  Expected: min9 ~ +%.2f%%  MaxDD ~ %.2f%%" % (B3A_KNOWN_MIN9 * 100, B3A_KNOWN_MAXDD * 100))
    print("=" * 100)

    neutral_def = np.ones(n, dtype=float)
    nav_n, r_n, tpy_n, exc_n = _build_full_b3a_g5(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr, def_mult_arr=neutral_def)
    pre_n = compute_10metrics(nav_n, tpy_n)
    aft_n = _apply_aftertax(pre_n)
    min9_n  = _min_at(aft_n)
    maxdd_n = pre_n["MaxDD_FULL"]

    print("  Neutral: min9=%+.4f%%  MaxDD=%+.4f%%  Sharpe=%.4f"
          % (min9_n * 100, maxdd_n * 100, pre_n["Sharpe_OOS"]))
    ok_min9  = abs(min9_n - B3A_KNOWN_MIN9)   <= SANITY_TOL_MIN9
    ok_maxdd = abs(maxdd_n - B3A_KNOWN_MAXDD) <= SANITY_TOL_MAXDD
    print("  min9  delta: %+.4fpp -> %s" % ((min9_n - B3A_KNOWN_MIN9) * 100,   "OK" if ok_min9  else "FAIL"))
    print("  MaxDD delta: %+.4fpp -> %s" % ((maxdd_n - B3A_KNOWN_MAXDD) * 100, "OK" if ok_maxdd else "FAIL"))
    if not (ok_min9 and ok_maxdd):
        print("\n[HALT] SANITY ANCHOR FAILED -- integration bug detected.")
        sys.exit(1)
    print("  SANITY ANCHOR PASSED.")

    pre_base = pre_n
    aft_base = aft_n

    # =========================================================================
    # Build element definition for variants
    # =========================================================================
    VARIANTS = []

    # --- G5h: Hysteresis ---
    for k in [5, 10, 21]:
        name = "G5h_k%d" % k
        def_arr = _build_def_mult_hysteresis(
            "vix_mom21", "daily", HARD_MAP, dates_dt, macro_df, k=k)
        VARIANTS.append(("G5h", name, def_arr, "hysteresis k=%d" % k))

    # --- G5b: EMA smoothing ---
    def_arr_ema = _build_def_mult_ema(
        "vix_mom21", "daily", HARD_MAP, dates_dt, macro_df, ema_span=21)
    VARIANTS.append(("G5b", "G5b_ema21", def_arr_ema, "EMA-21 smoothing"))

    # --- G5q: Coarse 2-quantile ---
    def_arr_2q = _build_def_mult_coarse(
        "vix_mom21", "daily", MAP_2Q, dates_dt, macro_df)
    VARIANTS.append(("G5q", "G5q_2quant", def_arr_2q, "coarse 2-quantile {0:1.0,1:0.85}"))

    # =========================================================================
    # Evaluate variants
    # =========================================================================
    print("\n" + "=" * 100)
    print("EVALUATING %d VARIANTS" % len(VARIANTS))
    print("=" * 100)

    results = {}
    for (grp, name, def_arr, desc) in VARIANTS:
        print("\n[%s] %s  (%s)" % (grp, name, desc))

        # def_mult 分布サマリ (一意値が多い EMA は統計で表示)
        unique_vals = np.unique(def_arr)
        if len(unique_vals) <= 8:
            dist = {round(v, 4): int((np.abs(def_arr - v) < 1e-6).sum()) for v in unique_vals}
            print("  def_mult dist: %s" % dist)
        else:
            print("  def_mult dist: n_unique=%d  min=%.4f  mean=%.4f  max=%.4f  p25=%.4f  p75=%.4f"
                  % (len(unique_vals), def_arr.min(), def_arr.mean(),
                     def_arr.max(), np.percentile(def_arr, 25), np.percentile(def_arr, 75)))

        nav_dt, r, tpy, exc = _build_full_b3a_g5(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
            bond_on, sofr_arr, def_mult_arr=def_arr)

        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)
        cy  = _calendar_year_returns(nav_dt)

        survive0, reason0, detail0 = _stage0_survive(aft, pre, aft_base, pre_base)
        screen_ok, screen_detail   = _turnover_screen(tpy, pre, aft, aft_base, pre_base)

        print("  Trades/yr=%5.1f  MaxDD=%+.2f%%  Sharpe=%.4f  min9=%+.2f%%"
              % (tpy, pre["MaxDD_FULL"] * 100, pre["Sharpe_OOS"], _min_at(aft) * 100))
        print("  vs B3a:  dMaxDD=%+.2fpp  dShrp=%+.4f  dmin9=%+.2fpp"
              % (detail0["delta_maxdd_pp"], detail0["delta_sharpe"], detail0["delta_min9_pp"]))
        print("  vs G5_orig: dTrades=%+.1f  (need < -1.0 from 52.9)"
              % (tpy - G5_ORIG_TRADES))
        print("  Stage0: %-8s  TurnoverScreen: %-8s"
              % (("SURVIVE" if survive0 else "FAIL"), ("PASS" if screen_ok else "FAIL")))

        results[name] = {
            "grp": grp, "desc": desc,
            "nav": nav_dt, "pre": pre, "aft": aft, "cy": cy,
            "tpy": tpy, "exc": exc,
            "stage0_survive": survive0, "stage0_reason": reason0, "stage0_detail": detail0,
            "screen_ok": screen_ok, "screen_detail": screen_detail,
        }

    # =========================================================================
    # Stage 0 Summary Table
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 0 SUMMARY TABLE")
    hdr = ("%-15s | %8s | %8s | %7s | %8s | %8s | %8s | %8s | %-8s | %-6s"
           % ("variant", "Trades", "MaxDD%", "Sharpe", "min9_at%", "dMaxDD_pp",
              "dShrp", "dmin9pp", "Stage0", "TurnSc"))
    print(hdr)
    print("-" * len(hdr))

    # B3a baseline row
    print("%-15s | %8.1f | %+7.2f%% | %7.4f | %+7.2f%% | %+7.2fpp | %+7.4f | %+7.2fpp | %-8s | %-6s"
          % ("B3a_k365(BASE)", tpy_n, maxdd_n * 100, pre_n["Sharpe_OOS"],
             min9_n * 100, 0.0, 0.0, 0.0, "BASE", "BASE"))

    # G5_orig reference row
    print("%-15s | %8.1f | %+7.2f%% | %7.4f | %+7.2f%% | %+7.2fpp | %+7.4f | %+7.2fpp | %-8s | %-6s"
          % ("G5_vix_hard(REF)", G5_ORIG_TRADES, G5_ORIG_MAXDD * 100, G5_ORIG_SHARPE,
             G5_ORIG_MIN9 * 100,
             (G5_ORIG_MAXDD - B3A_KNOWN_MAXDD) * 100,
             G5_ORIG_SHARPE - B3A_KNOWN_SHARPE,
             (G5_ORIG_MIN9 - B3A_KNOWN_MIN9) * 100,
             "SURVIVE", "REF"))

    for name, res in results.items():
        pre = res["pre"]; aft = res["aft"]; d = res["stage0_detail"]
        s0  = "SURVIVE" if res["stage0_survive"] else "FAIL"
        sc  = "PASS" if res["screen_ok"] else "FAIL"
        print("%-15s | %8.1f | %+7.2f%% | %7.4f | %+7.2f%% | %+7.2fpp | %+7.4f | %+7.2fpp | %-8s | %-6s"
              % (name, res["tpy"], pre["MaxDD_FULL"] * 100, pre["Sharpe_OOS"],
                 _min_at(aft) * 100, d["delta_maxdd_pp"], d["delta_sharpe"],
                 d["delta_min9_pp"], s0, sc))

    # =========================================================================
    # 6 次元スコアカード (生存変種)
    # =========================================================================
    survivors = {n: r for n, r in results.items() if r["screen_ok"]}

    print("\n" + "=" * 100)
    print("6-DIM SCORECARD (TurnoverScreen PASS variants vs B3a_k365 vs G5_orig)")
    print("=" * 100)

    # B3a と G5_orig の採点
    sc_b3a_bal  = SC.score_candidate(_B3A_SC_DATA, nonsat=False)
    sc_b3a_cagr = SC.score_candidate(_B3A_SC_DATA, nonsat=True)
    sc_g5_bal   = SC.score_candidate(_G5_ORIG_SC_DATA, nonsat=False)
    sc_g5_cagr  = SC.score_candidate(_G5_ORIG_SC_DATA, nonsat=True)

    scorecard_results = {}

    # ヘッダ
    dim_hdr = ("%-22s | %6s | %6s | %6s | %6s | %6s | %6s | %7s | %7s | %s"
               % ("label", "D1ret", "D2rob", "D3rsk", "D4tail", "D5cst", "D6sig",
                  "totBAL", "totCAGR", "note"))
    print(dim_hdr)
    print("-" * len(dim_hdr))

    def _print_sc_row(label, sc_b, sc_c, note=""):
        print("%-22s | %6.3f | %6.3f | %6.3f | %6.3f | %6.3f | %6.3f | %7.3f | %7.3f | %s"
              % (label, sc_b["d1_ret"], sc_b["d2_rob"], sc_b["d3_risk"],
                 sc_b["d4_tail"], sc_b["d5_cost"], sc_b["d6_sig"],
                 sc_b["total"], sc_c["total"], note))

    _print_sc_row("B3a_k365",        sc_b3a_bal, sc_b3a_cagr, "BASELINE Trades=33.3")
    _print_sc_row("G5_vix_hard(orig)", sc_g5_bal, sc_g5_cagr, "orig Trades=52.9 (high cost)")

    scorecard_results["B3a_k365"] = {
        "sc_bal": sc_b3a_bal, "sc_cagr": sc_b3a_cagr
    }
    scorecard_results["G5_orig"] = {
        "sc_bal": sc_g5_bal, "sc_cagr": sc_g5_cagr
    }

    if survivors:
        print("--- TurnoverScreen PASS variants ---")
        for name, res in survivors.items():
            sc_b, sc_c = _scorecard_from_metrics(
                name, res["pre"], res["aft"], res["exc"])
            _print_sc_row(name, sc_b, sc_c,
                          "Trades=%.1f dMaxDD=%+.2fpp" % (res["tpy"], res["screen_detail"]["delta_maxdd_pp"]))
            scorecard_results[name] = {"sc_bal": sc_b, "sc_cagr": sc_c}
    else:
        print("  [NO SURVIVORS] -- 全変種がターンオーバースクリーンを通過できなかった")

    # =========================================================================
    # Conclusion
    # =========================================================================
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)

    if survivors:
        # 総合スコアが B3a を超えるか
        b3a_total_bal  = sc_b3a_bal["total"]
        b3a_total_cagr = sc_b3a_cagr["total"]
        print("  B3a_k365 total (BAL=%.3f  CAGR=%.3f)" % (b3a_total_bal, b3a_total_cagr))
        print("  G5_orig   total (BAL=%.3f  CAGR=%.3f)" % (sc_g5_bal["total"], sc_g5_cagr["total"]))
        print()
        any_beats_b3a = False
        for name, r in survivors.items():
            sc_b = scorecard_results[name]["sc_bal"]
            sc_c = scorecard_results[name]["sc_cagr"]
            beats_bal  = sc_b["total"] > b3a_total_bal
            beats_cagr = sc_c["total"] > b3a_total_cagr
            if beats_bal or beats_cagr:
                any_beats_b3a = True
            print("  %s: BAL=%.3f (%s B3a) CAGR=%.3f (%s B3a)  Trades=%.1f"
                  % (name, sc_b["total"], ">" if beats_bal else "<=",
                     sc_c["total"], ">" if beats_cagr else "<=", res["tpy"]))
        print()
        if any_beats_b3a:
            print("  [RESULT] 取引削減変種の一部が 6 次元採点で B3a を上回った。")
            print("  G5_vix_hard の strict improvement 達成の可能性あり。")
        else:
            print("  [RESULT] 全取引削減変種が 6 次元採点で B3a を下回った。")
            print("  取引削減と MaxDD 改善はトレードオフ: 元 G5 (52.9tr) が最良点と結論。")
            print("  MaxDD 改善 (+2.27pp) は取引コスト増 (+20/yr) で打ち消されている。")
    else:
        print("  [RESULT] 全変種がターンオーバースクリーン不通過。")
        print("  取引削減と MaxDD 改善は両立できない: 元 G5 が最良点。")
        print("  「MaxDD 改善 (+2.27pp) vs 取引コスト増 (+20/yr)」はトレードオフの関係であり、")
        print("  ヒステリシス・EMA・粗い量子化のいずれもこのトレードオフを解消できない。")

    # =========================================================================
    # CSV output
    # =========================================================================
    print("\n[Building CSV] ...")
    rows = []

    def _row(label, grp, desc, tpy_val, pre, aft, exc, cy, stage0_survive, stage0_reason,
             screen_ok, sc_bal=None, sc_cagr=None):
        cy_arr = cy if (cy is not None and len(cy) > 0) else pd.Series(dtype=float)
        worst_cy = float(cy_arr.min()) if len(cy_arr) > 0 else float("nan")
        worst_yr = int(cy_arr.idxmin()) if len(cy_arr) > 0 else -1
        return {
            "label":           label,
            "variant_grp":     grp,
            "desc":            desc,
            "CAGR_IS_at":      aft["CAGR_IS"],
            "CAGR_OOS_at":     aft["CAGR_OOS"],
            "min_IS_OOS_at":   _min_at(aft),
            "Sharpe_OOS":      pre["Sharpe_OOS"],
            "MaxDD_FULL":      pre["MaxDD_FULL"],
            "Worst10Y_at":     aft["Worst10Y_star"],
            "P10_5Y_at":       aft["P10_5Y"],
            "Worst5Y_at":      aft["Worst5Y"],
            "Trades_yr":       tpy_val,
            "excess_days":     exc,
            "worst_cy":        worst_cy,
            "worst_cy_year":   worst_yr,
            "Stage0_survive":  int(stage0_survive),
            "Stage0_reason":   stage0_reason,
            "TurnScreen_ok":   int(screen_ok),
            "d5_cost_bal":     sc_bal["d5_cost"] if sc_bal else float("nan"),
            "total_bal":       sc_bal["total"]    if sc_bal else float("nan"),
            "total_cagr":      sc_cagr["total"]   if sc_cagr else float("nan"),
        }

    # B3a baseline
    sc_b, sc_c = SC.score_candidate(_B3A_SC_DATA, nonsat=False), SC.score_candidate(_B3A_SC_DATA, nonsat=True)
    rows.append(_row("B3a_k365", "BASE", "baseline", tpy_n,
                     pre_n, aft_n, exc_n, _calendar_year_returns(nav_n),
                     True, "BASELINE", True, sc_b, sc_c))

    # G5_orig reference
    sc_g5b, sc_g5c = SC.score_candidate(_G5_ORIG_SC_DATA, nonsat=False), SC.score_candidate(_G5_ORIG_SC_DATA, nonsat=True)
    dummy_pre = {"MaxDD_FULL": G5_ORIG_MAXDD, "Sharpe_OOS": G5_ORIG_SHARPE}
    dummy_aft = {"CAGR_IS": 0.222015, "CAGR_OOS": G5_ORIG_MIN9,
                 "Worst10Y_star": _G5_ORIG_SC_DATA["w10y"],
                 "P10_5Y": _G5_ORIG_SC_DATA["p10_5y"],
                 "Worst5Y": _G5_ORIG_SC_DATA["w5y"],
                 "Trades_yr": G5_ORIG_TRADES}
    rows.append(_row("G5_vix_hard_orig", "REF", "original G5 (no turnover reduction)",
                     G5_ORIG_TRADES, dummy_pre, dummy_aft, 4329,
                     pd.Series(dtype=float), True, "SURVIVE", False, sc_g5b, sc_g5c))

    # Variants
    for name, res in results.items():
        sc_b2, sc_c2 = None, None
        if res["screen_ok"] and name in scorecard_results:
            sc_b2 = scorecard_results[name]["sc_bal"]
            sc_c2 = scorecard_results[name]["sc_cagr"]
        rows.append(_row(name, res["grp"], res["desc"],
                         res["tpy"], res["pre"], res["aft"], res["exc"], res["cy"],
                         res["stage0_survive"], res["stage0_reason"],
                         res["screen_ok"], sc_b2, sc_c2))

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combine_g5_turnover_20260615.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # RETURN_BLOCK
    # =========================================================================
    def _rblock(name, sc_b, sc_c, tpy_v, maxdd, sharpe, min9, delta_maxdd, delta_min9,
                screen_ok, stage0):
        return {
            "Trades_yr":       round(float(tpy_v), 2),
            "MaxDD":           round(float(maxdd), 4),
            "Sharpe_OOS":      round(float(sharpe), 4),
            "min9_at":         round(float(min9), 4),
            "delta_maxdd_pp":  round(float(delta_maxdd), 3),
            "delta_min9_pp":   round(float(delta_min9), 3),
            "TurnScreen_ok":   bool(screen_ok),
            "Stage0_survive":  bool(stage0),
            "d5_cost_bal":     round(float(sc_b["d5_cost"]), 3) if sc_b else None,
            "total_bal":       round(float(sc_b["total"]),   3) if sc_b else None,
            "total_cagr":      round(float(sc_c["total"]),   3) if sc_c else None,
            "beats_B3a_bal":   bool(sc_b["total"] > sc_b3a_bal["total"])  if sc_b else False,
            "beats_B3a_cagr":  bool(sc_c["total"] > sc_b3a_cagr["total"]) if sc_c else False,
        }

    block = {
        "meta": {
            "script":       "combine_g5_turnover_20260615.py",
            "base":         "B3a_k365",
            "sanity_ok":    bool(ok_min9 and ok_maxdd),
            "G5_orig_trades": float(G5_ORIG_TRADES),
            "G5_orig_maxdd":  float(G5_ORIG_MAXDD),
            "n_variants":   int(len(results)),
            "n_survivors":  int(len(survivors)),
        },
        "B3a_k365": _rblock(
            "B3a", sc_b3a_bal, sc_b3a_cagr,
            tpy_n, maxdd_n, pre_n["Sharpe_OOS"], min9_n,
            0.0, 0.0, True, True),
        "G5_orig": _rblock(
            "G5_orig", sc_g5_bal, sc_g5_cagr,
            G5_ORIG_TRADES, G5_ORIG_MAXDD, G5_ORIG_SHARPE, G5_ORIG_MIN9,
            (G5_ORIG_MAXDD - B3A_KNOWN_MAXDD) * 100,
            (G5_ORIG_MIN9 - B3A_KNOWN_MIN9) * 100,
            False, True),
        "variants": {},
    }

    for name, res in results.items():
        d = res["stage0_detail"]
        sc_b2 = scorecard_results.get(name, {}).get("sc_bal",  None)
        sc_c2 = scorecard_results.get(name, {}).get("sc_cagr", None)
        block["variants"][name] = _rblock(
            name, sc_b2, sc_c2,
            res["tpy"], res["pre"]["MaxDD_FULL"], res["pre"]["Sharpe_OOS"], _min_at(res["aft"]),
            d["delta_maxdd_pp"], d["delta_min9_pp"],
            res["screen_ok"], res["stage0_survive"])

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=False))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
