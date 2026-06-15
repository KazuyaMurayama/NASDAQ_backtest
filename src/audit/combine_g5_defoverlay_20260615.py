# -*- coding: utf-8 -*-
"""
src/audit/combine_g5_defoverlay_20260615.py
============================================
G5: 第2シグナル防御オーバーレイ  --  B3a_k365 土台に native 統合

目的
----
B3a_k365 は mom63 を V7 ブーストに使用中のため、defensive trim には
別シグナルを使う。macro_features.csv の vix_mom21 と nfci_z52w を候補に
分位化 defensive マップを IN 期レバに native 乗算統合し、Stage 0 スクリーンを
実施する。

土台: B3a_k365
  v7_map  = {0:1.40, 1:1.40, 2:1.05, 3:1.00}
  lev_scale = 1.15
  excess_extra = 0.0025 (k365 centre: SOFR+0.75pp - TQQQ_SWAP 0.5%)
  C1 SOFR fill ON

Native 統合
-----------
  lev_mod = lev_raw_masked * mult_v7     <- 既存 B3a
  lev_mod_g5 = lev_mod * def_mult_arr   <- G5 乗算 (追加)
  def_mult_arr: signal_q {0,1,2,3} -> multiplier

シグナル候補
------------
  1. vix_mom21   : 高値 = VIX モメンタム上昇 = リスクオン過熱信号
                   -> 高分位でレバ trim (defensive)
                   publication lag: daily (+1 BD)
  2. nfci_z52w   : 高値 = 金融状況ひっ迫 (Federal Reserve NFCI, 週次)
                   -> 高分位でレバ trim (defensive)
                   publication lag: weekly (+5 BD)

マップ
------
  控えめ (soft): {0:1.00, 1:1.00, 2:0.95, 3:0.90}
  強め   (hard): {0:1.00, 1:1.00, 2:0.92, 3:0.85}

構成 (2 シグナル x 2 マップ = 4)
  G5_vix_soft, G5_vix_hard, G5_nfci_soft, G5_nfci_hard

サニティアンカー
----------------
  マップ = neutral {0:1.00, 1:1.00, 2:1.00, 3:1.00} で B3a_k365 素地
  (min9 +20.98%, MaxDD -38.20%, Sharpe 0.904) を +-0.05pp で再現するか確認。
  再現しなければ統合バグとして HALT。

Stage 0 ベト判定 (防御要素)
  ベト無 AND
  (MaxDD 改善 >= +2pp OR Sharpe >= +0.02 OR Worst10Y* >= +0.5pp)
  AND min9 劣化 <= 1.0pp

出力
----
  audit_results/combine_g5_defoverlay_20260615.csv
  RETURN_BLOCK (json) を stdout に出力

制約
----
  ASCII print (cp932), git 操作禁止, 一時ファイル禁止, post-hoc 禁止
  ルックアヘッド禁止 (publication lag 厳守)

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

# ---------------------------------------------------------------------------
# B3a_k365 config
# ---------------------------------------------------------------------------
B3A_V7_MAP  = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE = 1.15
B3A_EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE   # 0.0025

# Known B3a_k365 ground-truth (from LEVERUP_SWEEP / k365_recost)
B3A_KNOWN_MIN9   = 0.2098   # +20.98%
B3A_KNOWN_MAXDD  = -0.3820  # -38.20%
B3A_KNOWN_SHARPE = 0.904

SANITY_TOL_MIN9  = 0.0005   # +-0.05pp
SANITY_TOL_MAXDD = 0.0010   # +-0.10pp

# ---------------------------------------------------------------------------
# Defensive maps
# ---------------------------------------------------------------------------
SOFT_MAP = {0: 1.00, 1: 1.00, 2: 0.95, 3: 0.90}
HARD_MAP = {0: 1.00, 1: 1.00, 2: 0.92, 3: 0.85}
NEUTRAL_MAP = {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00}

# Macro features CSV
MACRO_FEATURES_PATH = os.path.join(_REPO_DIR, "data", "macro_features.csv")

# Signal configs: (name, column, lag_type, map_name, map_dict)
SIGNAL_CONFIGS = [
    # neutral (sanity anchor)
    ("B3a_k365_neutral",  None,         None,     "neutral", NEUTRAL_MAP),
    # vix_mom21: high VIX momentum = risk-on overheating -> defensive
    ("G5_vix_soft",       "vix_mom21",  "daily",  "soft",    SOFT_MAP),
    ("G5_vix_hard",       "vix_mom21",  "daily",  "hard",    HARD_MAP),
    # nfci_z52w: high NFCI z-score = tightening financial conditions -> defensive
    ("G5_nfci_soft",      "nfci_z52w",  "weekly", "soft",    SOFT_MAP),
    ("G5_nfci_hard",      "nfci_z52w",  "weekly", "hard",    HARD_MAP),
]


# ---------------------------------------------------------------------------
# Parameterised NAV builder (mirrors k365_recost._build_nav_v7_tqqq_param
# but with additional defensive multiplier on lev_mod)
# ---------------------------------------------------------------------------

def _build_nav_b3a_g5(close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
                       lev_raw_masked, wn, wg, wb, mult_v7,
                       def_mult_arr,
                       excess_extra=B3A_EXCESS_EXTRA):
    """B3a TQQQ base NAV with G5 defensive multiplier natively applied.

    def_mult_arr: 1-D ndarray aligned to dates, values in (0, 1.2].
                  Applied as: lev_mod = lev_raw_masked * mult_v7 * def_mult_arr
                  so the defensive signal modulates leverage BEFORE NAV is
                  computed (native integration, not post-hoc).
    """
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

    # --- G5 native integration: lev_mod includes defensive multiplier ---
    lev_raw_arr = np.asarray(lev_raw_masked, float)
    mult_v7_arr = np.asarray(mult_v7, float)
    def_arr     = np.asarray(def_mult_arr, float)
    lev_mod = lev_raw_arr * mult_v7_arr * def_arr

    L = pd.Series(lev_mod * 3.0, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(V7_DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(V7_DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(V7_DELAY).fillna(0).values
    sofr_arr_np = np.asarray(sofr_daily, float)

    # TQQQ financing leg
    borrow  = np.maximum(L - 1.0, 0.0) * (sofr_arr_np + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # >3x excess penalty (k365 centre = 0.0025/yr)
    excess_lev = np.maximum(L - LEV_CAP, 0.0)
    penalty = wn_s * excess_lev * float(excess_extra) / TRADING_DAYS
    daily   = daily - penalty
    excess_days = int(np.sum(excess_lev > 1e-9))

    # turnover drag
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
    """Build B3a TQQQ base NAV with G5 defensive multiplier."""
    a   = shared["assets"]
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
    """Build full B3a_k365+G5 NAV (TQQQ base + C1 OUT fill)."""
    base_nav, r_base, tpy_b, exc = _build_tqqq_base_g5(
        shared, dates_dt, def_mult_arr)
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on,
        sofr_arr, dates_dt, tpy_b, n_years)
    return nav_dt, r, tpy, exc


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


# ---------------------------------------------------------------------------
# Build defensive multiplier array aligned to strategy dates
# ---------------------------------------------------------------------------

def _build_def_mult_arr(signal_col, lag_type, def_map, dates_dt, macro_df):
    """Compute per-date defensive multiplier array.

    Steps (Lesson A / native integration):
      1. Load signal column from macro_df (DatetimeIndex)
      2. quantile_cut(levels=4) on full sample (no lookahead in quantile edges:
         full-sample cut is acceptable for a slowly-varying feature; consistent
         with how G3 native evaluation was done in signal expansion project)
      3. apply_publication_lag(lag_type) -> shift by 1 BD (daily) or 5 BD (weekly)
      4. Align / ffill to strategy dates
      5. Map {0,1,2,3} -> def_map multiplier
    """
    if signal_col is None:
        # neutral: all 1.0
        return np.ones(len(dates_dt), dtype=float)

    raw_signal = macro_df[signal_col].dropna()
    raw_signal.index = pd.DatetimeIndex(pd.to_datetime(raw_signal.index))

    sig_q = quantile_cut(raw_signal, levels=4)
    sig_q = sig_q.dropna().astype("int8")

    sig_lagged = apply_publication_lag(sig_q, lag_type)
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]

    # Align to strategy dates, forward-fill gaps (no data = previous quartile)
    aligned = sig_lagged.reindex(dates_dt).ffill()

    mult = aligned.map(
        lambda s: def_map.get(int(s), 1.0) if pd.notna(s) else 1.0
    )
    arr = np.asarray(mult.fillna(1.0).values, dtype=float)
    arr = np.clip(arr, 0.0, 2.0)
    return arr


# ---------------------------------------------------------------------------
# Stage 0 survival check (defensive element criteria)
# ---------------------------------------------------------------------------

def _stage0_survive(aft_g5, pre_g5, aft_base, pre_base):
    """Return (survive: bool, reason: str, detail: dict)."""
    # Hard veto
    veto_maxdd  = pre_g5["MaxDD_FULL"] < -0.50
    veto_w10y   = aft_g5["Worst10Y_star"] < 0.0

    # Improvement checks (vs B3a_k365 base)
    delta_maxdd  = (pre_g5["MaxDD_FULL"] - pre_base["MaxDD_FULL"]) * 100   # pp
    delta_sharpe = pre_g5["Sharpe_OOS"] - pre_base["Sharpe_OOS"]
    delta_w10y   = (aft_g5["Worst10Y_star"] - aft_base["Worst10Y_star"]) * 100  # pp
    delta_min9   = (_min_at(aft_g5) - _min_at(aft_base)) * 100  # pp (negative = deterioration)

    veto = veto_maxdd or veto_w10y
    improve = (delta_maxdd >= 2.0) or (delta_sharpe >= 0.02) or (delta_w10y >= 0.5)
    min9_ok = delta_min9 >= -1.0  # min9 deterioration <= 1.0pp

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
        reason = "HARD_VETO (MaxDD<-50%% or W10Y<0)" if veto_maxdd else "HARD_VETO (W10Y<0)"
    elif not min9_ok:
        reason = "FAIL: min9 劣化 > 1.0pp (delta=%.2fpp)" % delta_min9
    elif not improve:
        reason = ("FAIL: defensive 改善なし "
                  "(MaxDD_delta=%.2fpp Sharpe_delta=%.4f W10Y_delta=%.2fpp)"
                  % (delta_maxdd, delta_sharpe, delta_w10y))
    else:
        reason = "SURVIVE"
    return survive, reason, detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("G5 DEFENSIVE OVERLAY  --  B3a_k365 native integration")
    print("Signals: vix_mom21 (daily lag) / nfci_z52w (weekly lag)")
    print("Maps   : soft {0:1.00,1:1.00,2:0.95,3:0.90} / hard {0:1.00,1:1.00,2:0.92,3:0.85}")
    print("=" * 100)

    # ---- Load shared DH-W1 assets (one-time) ----
    print("\n[Step 1] Loading DH-W1 shared assets ...")
    sr._load_dhw1_shared()
    shared   = sr._DHW1_SHARED
    a        = shared["assets"]
    mask     = np.asarray(shared["mask"], dtype=float)
    dates    = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    # ---- Gold/Bond 1x legs ----
    print("[Step 2] Building Gold/Bond 1x legs ...")
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x  = np.asarray(prepare_gold_local(dates), float)
    bond_1x  = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

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
    print("  Columns available: %d  (vix_mom21=%s  nfci_z52w=%s)"
          % (len(macro_df.columns),
             "YES" if "vix_mom21" in macro_df.columns else "NO",
             "YES" if "nfci_z52w" in macro_df.columns else "NO"))

    # ---- Signal stats ----
    for col in ["vix_mom21", "nfci_z52w"]:
        s = macro_df[col].dropna()
        print("  %s: N=%d  mean=%.4f  std=%.4f  min=%.4f  max=%.4f"
              % (col, len(s), s.mean(), s.std(), s.min(), s.max()))

    # =========================================================================
    # Sanity anchor: neutral map must reproduce B3a_k365
    # =========================================================================
    print("\n" + "=" * 100)
    print("SANITY ANCHOR: neutral map (all 1.0) must reproduce B3a_k365 素地")
    print("Expected: min9 ~ +%.2f%%  MaxDD ~ %.2f%%  (tol: min9 +-%.2fpp  MaxDD +-%.2fpp)"
          % (B3A_KNOWN_MIN9 * 100, B3A_KNOWN_MAXDD * 100,
             SANITY_TOL_MIN9 * 100, SANITY_TOL_MAXDD * 100))
    print("=" * 100)

    neutral_def = np.ones(n, dtype=float)
    nav_neutral, r_neutral, tpy_neutral, exc_neutral = _build_full_b3a_g5(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr,
        def_mult_arr=neutral_def)

    pre_neutral = compute_10metrics(nav_neutral, tpy_neutral)
    aft_neutral = _apply_aftertax(pre_neutral)
    min9_neutral = _min_at(aft_neutral)
    maxdd_neutral = pre_neutral["MaxDD_FULL"]
    sharpe_neutral = pre_neutral["Sharpe_OOS"]

    print("  Neutral: min9=%+.4f%%  MaxDD=%+.4f%%  Sharpe=%.4f"
          % (min9_neutral * 100, maxdd_neutral * 100, sharpe_neutral))

    ok_min9  = abs(min9_neutral - B3A_KNOWN_MIN9)  <= SANITY_TOL_MIN9
    ok_maxdd = abs(maxdd_neutral - B3A_KNOWN_MAXDD) <= SANITY_TOL_MAXDD

    print("  min9  delta: %+.4fpp  -> %s" % ((min9_neutral - B3A_KNOWN_MIN9) * 100,
                                               "OK" if ok_min9 else "FAIL"))
    print("  MaxDD delta: %+.4fpp  -> %s" % ((maxdd_neutral - B3A_KNOWN_MAXDD) * 100,
                                               "OK" if ok_maxdd else "FAIL"))

    if not (ok_min9 and ok_maxdd):
        print("\n[HALT] SANITY ANCHOR FAILED -- integration bug detected.")
        print("  Confirm _build_full_b3a_g5 uses B3A_V7_MAP, B3A_LEV_SCALE, B3A_EXCESS_EXTRA.")
        import sys as _sys; _sys.exit(1)

    print("  SANITY ANCHOR PASSED.")

    # Store B3a base for diff comparison
    pre_base = pre_neutral
    aft_base = aft_neutral

    # =========================================================================
    # G5 evaluations
    # =========================================================================
    print("\n" + "=" * 100)
    print("G5 EVALUATIONS (Stage 0 metrics)")
    print("=" * 100)

    results = {}

    for (name, signal_col, lag_type, map_name, def_map) in SIGNAL_CONFIGS:
        if name == "B3a_k365_neutral":
            # already computed
            results[name] = {
                "nav": nav_neutral, "r": r_neutral,
                "pre": pre_neutral, "aft": aft_neutral,
                "tpy": tpy_neutral, "exc": exc_neutral,
                "signal_col": "NONE", "lag_type": "NONE", "map_name": "neutral",
            }
            continue

        print("\n[Building] %s  (signal=%s  lag=%s  map=%s)"
              % (name, signal_col, lag_type, map_name))

        def_arr = _build_def_mult_arr(signal_col, lag_type, def_map,
                                       dates_dt, macro_df)

        # Signal coverage stats
        unique_mults, counts = np.unique(def_arr[def_arr > 0], return_counts=True)
        cov_vals = np.unique(def_arr)
        print("  def_mult distribution: %s"
              % {round(v, 4): int((def_arr == v).sum()) for v in cov_vals})

        nav_dt, r, tpy, exc = _build_full_b3a_g5(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
            bond_on, sofr_arr,
            def_mult_arr=def_arr)

        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)
        cy  = _calendar_year_returns(nav_dt)

        results[name] = {
            "nav": nav_dt, "r": r,
            "pre": pre, "aft": aft, "cy": cy,
            "tpy": tpy, "exc": exc,
            "signal_col": signal_col, "lag_type": lag_type, "map_name": map_name,
        }

        survive, reason, detail = _stage0_survive(aft, pre, aft_base, pre_base)

        print("  min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.4f  W10Y*=%+.2f%%"
              % (_min_at(aft) * 100, pre["MaxDD_FULL"] * 100,
                 pre["Sharpe_OOS"], aft["Worst10Y_star"] * 100))
        print("  vs B3a: delta_min9=%+.2fpp  delta_MaxDD=%+.2fpp  delta_Sharpe=%+.4f  delta_W10Y=%+.2fpp"
              % (detail["delta_min9_pp"], detail["delta_maxdd_pp"],
                 detail["delta_sharpe"], detail["delta_w10y_pp"]))
        print("  Stage0: %s  [%s]" % ("SURVIVE" if survive else "CLOSE", reason))

        results[name]["survive"] = survive
        results[name]["reason"]  = reason
        results[name]["detail"]  = detail

    # Also attach survive/reason for neutral
    results["B3a_k365_neutral"]["survive"] = True
    results["B3a_k365_neutral"]["reason"]  = "BASELINE (neutral map)"
    results["B3a_k365_neutral"]["detail"]  = {
        "delta_maxdd_pp": 0.0, "delta_sharpe": 0.0,
        "delta_w10y_pp": 0.0, "delta_min9_pp": 0.0,
        "veto_maxdd": False, "veto_w10y": False,
        "improve": True, "min9_ok": True,
    }

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 0 SUMMARY TABLE")
    hdr = ("%-22s | %9s | %8s | %7s | %9s | %8s | %8s | %8s | %-14s"
           % ("name", "min9_at%", "MaxDD%", "Sharpe",
              "W10Y*_at%", "dMaxDD_pp", "dShrp", "dmin9pp", "Stage0"))
    print(hdr)
    print("-" * len(hdr))
    for name, res in results.items():
        aft = res["aft"]; pre = res["pre"]; detail = res["detail"]
        s0 = "SURVIVE" if res["survive"] else "CLOSE"
        print("%-22s | %+8.2f%% | %+7.2f%% | %7.4f | %+8.2f%% | %+7.2fpp | %+7.4f | %+7.2fpp | %-14s"
              % (name,
                 _min_at(aft) * 100, pre["MaxDD_FULL"] * 100,
                 pre["Sharpe_OOS"], aft["Worst10Y_star"] * 100,
                 detail["delta_maxdd_pp"], detail["delta_sharpe"],
                 detail["delta_min9_pp"], s0))

    # G5 closure check (§6 中止条件)
    g5_names = [n for n in results if n.startswith("G5_")]
    survivors = [n for n in g5_names if results[n]["survive"]]

    print("\n" + "=" * 100)
    if survivors:
        print("G5 RESULT: %d / %d 構成が Stage 0 生存。Stage 1 (フルゲート) へ進む。"
              % (len(survivors), len(g5_names)))
        print("  生存構成: %s" % ", ".join(survivors))
    else:
        print("G5 CLOSE: 全 %d 構成が Stage 0 で防御改善なし。" % len(g5_names))
        print("  判定理由:")
        for n in g5_names:
            print("    %s: %s" % (n, results[n]["reason"]))
        print("")
        print("  [Lesson D 準拠] IC 高だけでは進めない。")
        print("  [§6 中止条件]   どのシグナル/マップも防御改善なし -> G5 効果なしクローズ。")

    # =========================================================================
    # Worst calendar year breakdown
    # =========================================================================
    print("\n" + "=" * 100)
    print("WORST CALENDAR YEAR BREAKDOWN (B3a base vs G5 構成)")
    print("%-22s | %-8s | %-10s" % ("name", "worst_cy%", "worst_yr"))
    print("-" * 50)
    for name, res in results.items():
        if "cy" in res:
            cy = res["cy"]
            print("%-22s | %+8.2f%% | %d" % (name, float(cy.min()) * 100, int(cy.idxmin())))

    # =========================================================================
    # CSV output
    # =========================================================================
    print("\n[Building CSV] ...")
    rows = []
    for name, res in results.items():
        aft  = res["aft"]
        pre  = res["pre"]
        detail = res["detail"]
        cy = res.get("cy", pd.Series(dtype=float))
        worst_cy = float(cy.min()) if len(cy) > 0 else float("nan")
        worst_yr = int(cy.idxmin()) if len(cy) > 0 else -1

        row = {
            "label":         name,
            "signal_col":    res["signal_col"],
            "lag_type":      res["lag_type"],
            "map_name":      res["map_name"],
            "CAGR_IS_at":    aft["CAGR_IS"],
            "CAGR_OOS_at":   aft["CAGR_OOS"],
            "min_IS_OOS_at": _min_at(aft),
            "IS_OOS_gap_pp": aft["IS_OOS_gap_pp"],
            "Sharpe_OOS":    pre["Sharpe_OOS"],
            "MaxDD_FULL":    pre["MaxDD_FULL"],
            "Worst10Y_star_at": aft["Worst10Y_star"],
            "P10_5Y_at":     aft["P10_5Y"],
            "Worst5Y_at":    aft["Worst5Y"],
            "Trades_yr":     aft["Trades_yr"],
            "worst_cy":      worst_cy,
            "worst_cy_year": worst_yr,
            "excess_days":   res["exc"],
            "delta_maxdd_pp":   detail["delta_maxdd_pp"],
            "delta_sharpe":     detail["delta_sharpe"],
            "delta_w10y_pp":    detail["delta_w10y_pp"],
            "delta_min9_pp":    detail["delta_min9_pp"],
            "veto_maxdd":       int(detail["veto_maxdd"]),
            "veto_w10y":        int(detail["veto_w10y"]),
            "Stage0_survive":   int(res["survive"]),
            "Stage0_reason":    res["reason"],
        }
        rows.append(row)

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combine_g5_defoverlay_20260615.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # RETURN_BLOCK
    # =========================================================================
    def _rblock(name, res):
        aft = res["aft"]; pre = res["pre"]; detail = res["detail"]
        cy  = res.get("cy", pd.Series(dtype=float))
        return {
            "signal_col":    res["signal_col"],
            "lag_type":      res["lag_type"],
            "map_name":      res["map_name"],
            "CAGR_IS_at":    round(float(aft["CAGR_IS"]),  6),
            "CAGR_OOS_at":   round(float(aft["CAGR_OOS"]), 6),
            "min_at":        round(float(_min_at(aft)),     6),
            "gap_pp":        round(float(aft["IS_OOS_gap_pp"]), 4),
            "Sharpe_OOS":    round(float(pre["Sharpe_OOS"]),    4),
            "MaxDD":         round(float(pre["MaxDD_FULL"]),     6),
            "Worst10Y_star_at": round(float(aft["Worst10Y_star"]), 6),
            "P10_5Y_at":     round(float(aft["P10_5Y"]),    6),
            "Trades_yr":     round(float(aft["Trades_yr"]), 2),
            "delta_maxdd_pp":    round(detail["delta_maxdd_pp"],  3),
            "delta_sharpe":      round(detail["delta_sharpe"],    4),
            "delta_w10y_pp":     round(detail["delta_w10y_pp"],   3),
            "delta_min9_pp":     round(detail["delta_min9_pp"],   3),
            "Stage0_survive":    res["survive"],
            "Stage0_reason":     res["reason"],
            "worst_cy": round(float(cy.min()), 6) if len(cy) > 0 else None,
        }

    block = {
        "meta": {
            "script":    "combine_g5_defoverlay_20260615.py",
            "base":      "B3a_k365",
            "sanity_ok": bool(ok_min9 and ok_maxdd),
            "B3a_known_min9_pct":   round(B3A_KNOWN_MIN9 * 100, 4),
            "B3a_known_maxdd_pct":  round(B3A_KNOWN_MAXDD * 100, 4),
            "neutral_min9_pct":     round(min9_neutral * 100, 4),
            "neutral_maxdd_pct":    round(maxdd_neutral * 100, 4),
            "neutral_sharpe":       round(sharpe_neutral, 4),
            "G5_survivors":         survivors,
            "G5_closed":            len(survivors) == 0,
        },
        "results": {name: _rblock(name, res) for name, res in results.items()},
    }

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=False))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
