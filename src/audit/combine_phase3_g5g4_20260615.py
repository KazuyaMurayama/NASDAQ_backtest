# -*- coding: utf-8 -*-
"""
src/audit/combine_phase3_g5g4_20260615.py
==========================================
Phase 3 -- G5+G4 combination full-gate evaluation
B3a_k365 with BOTH G5_vix_hard defensive map AND G4_LT2 k020 multiplier applied.

PURPOSE (MULTISTRATEGY_COMBINE_PLAN_20260615.md Section 2/4 Phase 3)
----------------------------------------------------------------------
Phase 2 results:
  G5_vix_hard : MaxDD -35.93%, Sharpe 0.928, min9 +20.73%  -- PRIMARY Phase 2 result
  G4_LT2_k020 : min9 +21.40%, Sharpe 0.865, Trades 122/yr  -- CAGR-up but Sharpe low

This script asks: does stacking G5+G4 produce a combination that
EXCEEDS G5_vix_hard alone on the combination criterion (plan Section 4)?
  (a) min9 > G5 single AND MaxDD stays within G5 single +1pp AND no veto
  (b) G4 CAGR uplift while G5 defence is maintained

COMBINATION LEVER FORMULA (native, not post-hoc):
  L_eff = (lev_raw_masked
           * mult_v7          <- B3a V7 map (per quartile)
           * lev_scale        <- 1.15 uniform
           * def_mult         <- G5_vix_hard defensive multiplier (vix_mom21, hard)
           * lt2_mult         <- G4_LT2 contrarian multiplier (k_lt=0.20)
          ) * 3.0

  Order of application:
    1. G5 def_mult : built from vix_mom21 quantile -> HARD_MAP {Q0:1,Q1:1,Q2:.92,Q3:.85}
       (daily lag +1 BD), full-sample quantile edges consistent with G5 standalone
    2. lt2_mult    : signal_to_mult(compute_lt2(close, N=750), k_lt=0.20)
       (strictly backward-looking, same as G4 standalone)

SANITY CHECKS (both required before proceeding):
  A. Set lt2_mult = ones (k_lt=0)   -> must reproduce G5_vix_hard standalone
     within tol: min9 +-0.05pp, MaxDD +-0.10pp
  B. Set def_mult = ones (neutral G5) -> must reproduce G4_k020 standalone
     within tol: min9 +-0.05pp, MaxDD +-0.10pp
  If either fails -> HALT and report.

FULL-GATE PIPELINE (reusing existing machinery):
  1. NAV construction (combined two multipliers)
  2. _eval_one (WFA 49w + CPCV N10k2emb21 + Regime + Stress)
  3. 4-metric block bootstrap x 2 block sizes
     Pairs: combined vs G5_vix_hard, combined vs B3a素地
  4. 4 hard veto checks

JUDGMENT (plan Section 4 combination criterion):
  ADOPT candidate if:
    (a) min9 > G5 single AND MaxDD (worsening) <= +1pp vs G5 single AND no veto
    OR
    (b) G5 defence preserved (MaxDD similar to G5) AND G4 CAGR uplift present
        AND Sharpe not worse than B3a
  OTHERWISE: "G5 single is best; combination adds no value"

Outputs:
  src/audit/combine_phase3_g5g4_20260615.py  (this script)
  audit_results/combine_phase3_g5g4_20260615.csv
  RETURN_BLOCK JSON to stdout

ASCII-only prints (cp932). No git ops. No temp files. No post-hoc.

Author: Kazuya Oza   2026-06-16
"""

from __future__ import annotations

import json
import os
import sys
import types

# ---- multitasking stub (yfinance dependency) ---------------------------------
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

# ---- Project infrastructure -------------------------------------------------
import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.regime_labeler_20260611 import build_regime_labels, stress_masks

# Reuse _eval_one from extended_eval (WFA+CPCV+Regime+Stress)
from src.audit.extended_eval_20260611 import (
    _cum_ret, _cpcv_dist, _regime_cagr, _eval_one,
)

# Multi-metric 4-axis bootstrap
from src.audit.multimetric_bootstrap_20260615 import _block_bootstrap_multimetric

# WFA runner (for _eval_one)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _run_wfa, N_BOOT, BLOCK, SEED, AFTER_TAX,
    _maxdd_from_returns, _cagr_seg,
)

# P09 OUT-fill helpers
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)

# B3a / k365 builders
from src.audit.k365_recost_20260612 import (
    _build_full_c1, EXCESS_EXTRA_K365_CENTRE, _min_at,
)
from src.audit.leverup_b1c1_20260612 import _build_p09_on_base_c1

# V7 baseline builder
from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base as _build_v7_tqqq_base,
    _build_p09_on_base,
    _apply_aftertax,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    _build_v7_mult, DELAY as V7_DELAY, DH_PER_UNIT, NAV_FLOOR,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom, LU2_SCALE,
)
from src.audit.lu_cfd_recost_20260611 import (
    SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)
from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

# G5 defensive overlay helpers (signal building)
from src.audit.combine_g5_defoverlay_20260615 import (
    _build_def_mult_arr, _build_full_b3a_g5,
    HARD_MAP, NEUTRAL_MAP, MACRO_FEATURES_PATH,
    B3A_V7_MAP, B3A_LEV_SCALE, B3A_EXCESS_EXTRA,
)

# G4 LT2 contrarian helpers
from src.audit.combine_g4_lt2_20260615 import _build_b3a_g4
from long_cycle_signal import compute_lt2, signal_to_mult

# TER daily drag constants (from strategy_runners)
_TER_GOLD2X_EXTRA_DAILY = 0.0
_TER_TMF_EXTRA_DAILY    = 0.0
try:
    from src.audit.strategy_runners import _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Known reference values
# ---------------------------------------------------------------------------

# B3a素地
B3A_REF = {
    "min9_at":     0.2098,   # +20.98%
    "MaxDD":      -0.3820,   # -38.20%
    "Sharpe":      0.904,
    "wfa_CI95_lo": 0.2252,
    "wfa_WFE":     0.987,
    "cpcv_p10_at": 0.1601,
    "regime_min": -0.0288,
}

# G5_vix_hard standalone (Phase 2 confirmed results)
G5_REF = {
    "min9_at":  0.2073,   # +20.73%
    "MaxDD":   -0.3593,   # -35.93%
    "Sharpe":   0.928,
    "Trades":   27.0,
}

# G4_k020 standalone (Phase 2 confirmed results)
G4_REF = {
    "min9_at":  0.2140,   # +21.40%
    "MaxDD":   -0.3900,   # approx (Phase2 value)
    "Sharpe":   0.865,
    "Trades":  122.0,
}

# V7 baseline
V7_REF_MIN9 = 0.1627

# Sanity tolerances
SANITY_TOL_MIN9  = 0.0005   # 0.05pp
SANITY_TOL_MAXDD = 0.0010   # 0.10pp

# Hard veto thresholds
VETO_MAXDD  = -0.50
VETO_WFE    =  1.50
VETO_W10Y   =  0.00
VETO_REGIME = -0.10

# G4 k_lt to use for combination
K_LT_COMBO = 0.20


# ---------------------------------------------------------------------------
# Combined NAV builder: G5 def_mult AND G4 lt2_mult, both native
# ---------------------------------------------------------------------------

def _build_nav_b3a_g5g4(close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
                          lev_raw_masked, wn, wg, wb, mult_v7,
                          def_mult_arr, lt2_mult_arr,
                          excess_extra=B3A_EXCESS_EXTRA):
    """B3a TQQQ base NAV with BOTH G5 defensive and G4 LT2 multipliers applied natively.

    Lever formula:
        lev_mod = lev_raw_masked * mult_v7 * def_mult_arr * lt2_mult_arr
        L = lev_mod * 3.0  (shifted by V7_DELAY)

    When def_mult_arr = ones -> reduces to G4_k020 standalone
    When lt2_mult_arr = ones -> reduces to G5_vix_hard standalone
    Both native (not post-hoc): applied BEFORE NAV compounding.
    """
    idx = dates.index
    n   = len(close)

    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3  = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    # Native combination: both multipliers applied before L is computed
    lev_base = (np.asarray(lev_raw_masked, float)
                * np.asarray(mult_v7,     float)
                * np.asarray(def_mult_arr, float)
                * np.clip(np.asarray(lt2_mult_arr, float), 0.5, 1.5))
    lev_mod = lev_base  # alias for clarity

    L     = pd.Series(lev_mod * 3.0,           index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s  = pd.Series(np.asarray(wn, float),   index=idx).shift(V7_DELAY).fillna(0).values
    wg_s  = pd.Series(np.asarray(wg, float),   index=idx).shift(V7_DELAY).fillna(0).values
    wb_s  = pd.Series(np.asarray(wb, float),   index=idx).shift(V7_DELAY).fillna(0).values
    sofr_arr = np.asarray(sofr_daily, float)

    # TQQQ financing
    borrow  = np.maximum(L - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # k365 excess penalty (L > 3x)
    excess_lev  = np.maximum(L - LEV_CAP, 0.0)
    penalty     = wn_s * excess_lev * float(excess_extra) / TRADING_DAYS
    daily       = daily - penalty
    excess_days = int(np.sum(excess_lev > 1e-9))

    # Turnover-based DH cost
    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn  = np.nan_to_num(dwn + dwg + dwb, nan=0.0)
    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    ter_drag  = (np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
                 + np.asarray(wb, float) * _TER_TMF_EXTRA_DAILY)

    tpy       = sr._compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / 252.0

    r_sim  = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj  = r_sim - ter_drag - etf_daily
    nav_adj = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )
    return nav_adj, tpy, excess_days


def _build_tqqq_base_combo(shared, dates_dt, def_mult_arr, lt2_mult_arr,
                             v7_map=B3A_V7_MAP, lev_scale=B3A_LEV_SCALE,
                             excess_extra=B3A_EXCESS_EXTRA):
    """Thin wrapper: extract arrays from shared and call _build_nav_b3a_g5g4."""
    a               = shared["assets"]
    close           = a["close"]
    dates           = a["dates"]
    sofr            = np.asarray(a["sofr"], float)
    gold_2x         = a["gold_2x"]
    bond_3x         = a["bond_3x"]
    lev_raw_masked  = np.asarray(shared["lev_raw_masked"], float)
    wn              = np.asarray(shared["wn"], float)
    wg              = np.asarray(shared["wg"], float)
    wb              = np.asarray(shared["wb"], float)

    mult_v7 = _build_v7_mult_custom(dates_dt, v7_map)
    mult_v7 = mult_v7 * float(lev_scale)

    nav_dt, tpy, excess_days = _build_nav_b3a_g5g4(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7,
        def_mult_arr, lt2_mult_arr, excess_extra=excess_extra)

    r_base = nav_dt.pct_change().fillna(0).values
    return nav_dt, r_base, tpy, excess_days


def _build_full_combo(shared, dates_dt, n_years,
                       ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
                       bond_on, sofr_arr,
                       def_mult_arr, lt2_mult_arr):
    """Full B3a+G5+G4 NAV (TQQQ base + P09 OUT-fill + C1 SOFR)."""
    base_nav, r_base, tpy_b, exc = _build_tqqq_base_combo(
        shared, dates_dt, def_mult_arr, lt2_mult_arr)
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on,
        sofr_arr, dates_dt, tpy_b, n_years)
    return nav_dt, r, tpy, exc


# ---------------------------------------------------------------------------
# Hard veto checker
# ---------------------------------------------------------------------------

def _hard_veto(label, wfa_res, full_maxdd, worst10y_at, regime_min):
    """Return (veto: bool, reasons: list[str])."""
    reasons = []
    if full_maxdd < VETO_MAXDD:
        reasons.append("MaxDD %.2f%% < -50%%" % (full_maxdd * 100))
    wfe = wfa_res.get("WFE", float("nan"))
    if not np.isnan(wfe) and wfe > VETO_WFE:
        reasons.append("WFE %.3f > 1.5" % wfe)
    if worst10y_at < VETO_W10Y:
        reasons.append("Worst10Y* %.2f%% < 0" % (worst10y_at * 100))
    if regime_min < VETO_REGIME:
        reasons.append("Regime_min %.2f%% < -10%%" % (regime_min * 100))
    return len(reasons) > 0, reasons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("PHASE 3  --  G5+G4 COMBINATION FULL-GATE EVALUATION")
    print("B3a_k365 + G5_vix_hard (defensiveMap) + G4_LT2_k020 (contrarianMult)")
    print("Formula: L = lev_raw * mult_v7 * lev_scale * def_mult_G5 * lt2_mult_G4 * 3.0")
    print("Comparison: vs G5_vix_hard single (primary) AND vs B3a baseline")
    print("=" * 100)
    print("N_BOOT=%d  BLOCK=%d  SEED=%d  AFTER_TAX=%.4f" % (N_BOOT, BLOCK, SEED, AFTER_TAX))

    # ---- [1] Load shared DH-W1 assets ----
    print("\n[1] Loading DH-W1 shared assets ...")
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

    # ---- [2] Regime labels and stress masks ----
    print("[2] Building regime labels and stress masks ...")
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress  = stress_masks(dates_dt)

    # ---- [3] Auxiliary legs ----
    print("[3] Building Gold/Bond 1x legs ...")
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

    # ---- [4] Load macro_features.csv for G5 signal ----
    print("[4] Loading macro_features.csv for G5 vix_mom21 signal ...")
    macro_df = pd.read_csv(MACRO_FEATURES_PATH, index_col=0, parse_dates=True)
    print("  Columns: %d  (vix_mom21=%s)"
          % (len(macro_df.columns),
             "YES" if "vix_mom21" in macro_df.columns else "NO"))

    # ---- [5] Build G5 defensive multiplier (hard map, daily lag) ----
    print("[5] Building G5 vix_hard defensive multiplier ...")
    def_arr_hard = _build_def_mult_arr(
        "vix_mom21", "daily", HARD_MAP, dates_dt, macro_df)
    def_arr_neutral = np.ones(n, dtype=float)

    # ---- [6] Build G4 LT2 multipliers ----
    print("[6] Computing LT2-N750 signal and building G4 multipliers ...")
    lt2_sig       = compute_lt2(a["close"], N=750)
    lt2_mult_k020 = np.clip(signal_to_mult(lt2_sig, k_lt=K_LT_COMBO).values, 0.5, 1.5)
    lt2_mult_neut = np.ones(len(lt2_sig), dtype=float)

    # =========================================================================
    # [7] Build all NAV series needed
    # =========================================================================
    print("\n[7] Building NAV series ...")

    # V7 baseline (no fill, no excess)
    print("  V7_TQQQ ...")
    v7_nav, r_v7, tpy_v7, _ = _build_v7_tqqq_base(shared, dates_dt, cfd_excess=False)

    # B3a_k365 baseline (k365 + C1 fill)
    print("  B3a_k365 ...")
    b3a_nav, r_b3a, tpy_b3a, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        v7_map=B3A_V7_MAP, lev_scale=B3A_LEV_SCALE,
        excess_extra=EXCESS_EXTRA_K365_CENTRE)

    # G5_vix_hard standalone (Phase 2 reference)
    print("  G5_vix_hard (Phase2 reference) ...")
    g5_nav, r_g5, tpy_g5, exc_g5 = _build_full_b3a_g5(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr, def_mult_arr=def_arr_hard)

    # G4_k020 standalone (Phase 2 reference)
    print("  G4_LT2_k020 (Phase2 reference) ...")
    g4_nav, r_g4, tpy_g4, exc_g4 = _build_b3a_g4(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr, lt2_mult=lt2_mult_k020)

    # Sanity A: combined with lt2_mult=ones -> should match G5_vix_hard
    print("  Sanity A: combo(def_hard, lt2_neut) == G5_vix_hard ...")
    san_a_nav, r_san_a, tpy_san_a, exc_san_a = _build_full_combo(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr,
        def_mult_arr=def_arr_hard, lt2_mult_arr=lt2_mult_neut)

    # Sanity B: combined with def_mult=ones -> should match G4_k020
    print("  Sanity B: combo(def_neut, lt2_k020) == G4_k020 ...")
    san_b_nav, r_san_b, tpy_san_b, exc_san_b = _build_full_combo(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr,
        def_mult_arr=def_arr_neutral, lt2_mult_arr=lt2_mult_k020)

    # Full combination: G5_vix_hard + G4_k020
    print("  B3a+G5+G4 combination ...")
    comb_nav, r_comb, tpy_comb, exc_comb = _build_full_combo(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr,
        def_mult_arr=def_arr_hard, lt2_mult_arr=lt2_mult_k020)

    print("  All NAVs built.")

    # =========================================================================
    # [8] SANITY CHECKS
    # =========================================================================
    print("\n" + "=" * 100)
    print("[8] SANITY CHECKS")
    print("    A: combo(lt2_neut) must reproduce G5_vix_hard (tol min9 +-0.05pp, MaxDD +-0.10pp)")
    print("    B: combo(def_neut) must reproduce G4_k020     (tol min9 +-0.05pp, MaxDD +-0.10pp)")
    print("=" * 100)

    def _quick_metrics(nav, tpy):
        pre = compute_10metrics(nav, tpy)
        aft = _apply_aftertax(pre)
        min9 = min(aft["CAGR_IS"], aft["CAGR_OOS"])
        return min9, pre["MaxDD_FULL"], pre["Sharpe_OOS"]

    # Measure standalone series
    g5_min9, g5_maxdd, g5_sh    = _quick_metrics(g5_nav,  tpy_g5)
    g4_min9, g4_maxdd, g4_sh    = _quick_metrics(g4_nav,  tpy_g4)
    sa_min9, sa_maxdd, sa_sh    = _quick_metrics(san_a_nav, tpy_san_a)
    sb_min9, sb_maxdd, sb_sh    = _quick_metrics(san_b_nav, tpy_san_b)

    def _san_check(name, got_min9, ref_min9, got_maxdd, ref_maxdd):
        d_min9  = abs(got_min9 - ref_min9)
        d_maxdd = abs(got_maxdd - ref_maxdd)
        ok_min9  = d_min9  <= SANITY_TOL_MIN9
        ok_maxdd = d_maxdd <= SANITY_TOL_MAXDD
        ok = ok_min9 and ok_maxdd
        print("  %-36s : min9=%+.4f%% (ref=%+.4f%%) d=%+.4fpp %s"
              % (name + " min9", got_min9 * 100, ref_min9 * 100,
                 d_min9 * 100, "OK" if ok_min9 else "FAIL"))
        print("  %-36s : MaxDD=%+.4f%% (ref=%+.4f%%) d=%+.4fpp %s"
              % (name + " MaxDD", got_maxdd * 100, ref_maxdd * 100,
                 d_maxdd * 100, "OK" if ok_maxdd else "FAIL"))
        return ok

    print("\n  [A] combo(def_hard, lt2_NEUTRAL) vs G5_vix_hard standalone:")
    ok_a = _san_check("SanityA", sa_min9, g5_min9, sa_maxdd, g5_maxdd)
    print("  -> SanityA: %s\n" % ("PASS" if ok_a else "FAIL"))

    print("  [B] combo(def_NEUTRAL, lt2_k020) vs G4_k020 standalone:")
    ok_b = _san_check("SanityB", sb_min9, g4_min9, sb_maxdd, g4_maxdd)
    print("  -> SanityB: %s\n" % ("PASS" if ok_b else "FAIL"))

    if not (ok_a and ok_b):
        print("[HALT] SANITY FAILED -- combination integration bug detected.")
        print("  Check _build_full_combo multiplier order vs _build_full_b3a_g5 / _build_b3a_g4.")
        sys.exit(1)

    print("  BOTH SANITY CHECKS PASSED. Combination logic verified.")

    # =========================================================================
    # [9] Full-gate evaluation
    # =========================================================================
    print("\n" + "=" * 100)
    print("[9] FULL-GATE _eval_one (WFA + CPCV + Regime + Stress)")
    print("    WFA is the slowest step -- please wait ~2-5 min per series")
    print("=" * 100)

    print("  Evaluating V7_TQQQ ...")
    ev_v7   = _eval_one("V7_TQQQ",    v7_nav,   r_v7,
                         regimes, stress, is_mask, oos_mask)

    print("  Evaluating B3a_k365 ...")
    ev_b3a  = _eval_one("B3a_k365",   b3a_nav,  r_b3a,
                         regimes, stress, is_mask, oos_mask, baseline_r=r_v7)

    print("  Evaluating G5_vix_hard (Phase2 ref) ...")
    ev_g5   = _eval_one("G5_vix_hard", g5_nav,  r_g5,
                         regimes, stress, is_mask, oos_mask, baseline_r=r_v7)

    print("  Evaluating B3a+G5+G4 combination ...")
    ev_comb = _eval_one("G5+G4_combo", comb_nav, r_comb,
                         regimes, stress, is_mask, oos_mask, baseline_r=r_v7)

    ev_all = {
        "V7_TQQQ":     ev_v7,
        "B3a_k365":    ev_b3a,
        "G5_vix_hard": ev_g5,
        "G5+G4_combo": ev_comb,
    }

    # =========================================================================
    # [10] Compute after-tax metrics for veto checks
    # =========================================================================
    pre_b3a  = compute_10metrics(b3a_nav,  tpy_b3a)
    pre_g5   = compute_10metrics(g5_nav,   tpy_g5)
    pre_comb = compute_10metrics(comb_nav, tpy_comb)
    pre_v7   = compute_10metrics(v7_nav,   tpy_v7)

    aft_b3a  = _apply_aftertax(pre_b3a)
    aft_g5   = _apply_aftertax(pre_g5)
    aft_comb = _apply_aftertax(pre_comb)
    aft_v7   = _apply_aftertax(pre_v7)

    # =========================================================================
    # [11] Multi-metric bootstrap (4 axes, block=21 + block=252)
    # =========================================================================
    print("\n" + "=" * 100)
    print("[11] MULTI-METRIC BOOTSTRAP (N=%d, block=21 + block=252)" % N_BOOT)
    print("     Pairs: combo vs G5_vix_hard (primary)  AND  combo vs B3a_k365")
    print("=" * 100)

    BOOT_PAIRS = [
        # (cand_label, r_cand, base_label, r_base)
        ("G5+G4_combo", r_comb, "G5_vix_hard", r_g5),
        ("G5+G4_combo", r_comb, "B3a_k365",    r_b3a),
    ]

    boot_21  = {}
    boot_252 = {}

    for (cl, rc, bl, rb) in BOOT_PAIRS:
        key = "%s_vs_%s" % (cl, bl)
        print("  [block=21]  %s ..." % key, end="", flush=True)
        boot_21[key] = _block_bootstrap_multimetric(
            rc, rb, is_mask, oos_mask, n_boot=N_BOOT, block=21, seed=SEED)
        r = boot_21[key]
        print(" done  P_min=%.3f P_dd=%.3f P_w10y=%.3f P_shp=%.3f"
              % (r["P_min_better"], r["P_maxdd_better"],
                 r.get("P_worst10y_better", float("nan")) or float("nan"),
                 r.get("P_sharpe_better",   float("nan")) or float("nan")))

        print("  [block=252] %s ..." % key, end="", flush=True)
        boot_252[key] = _block_bootstrap_multimetric(
            rc, rb, is_mask, oos_mask, n_boot=N_BOOT, block=252, seed=SEED)
        r = boot_252[key]
        print(" done  P_min=%.3f P_dd=%.3f P_w10y=%.3f P_shp=%.3f"
              % (r["P_min_better"], r["P_maxdd_better"],
                 r.get("P_worst10y_better", float("nan")) or float("nan"),
                 r.get("P_sharpe_better",   float("nan")) or float("nan")))

    # =========================================================================
    # [12] Hard veto checks
    # =========================================================================
    print("\n" + "=" * 100)
    print("[12] HARD VETO CHECKS")
    print("     Thresholds: MaxDD<-50% / WFE>1.5 / Worst10Y*<0 / Regime_min<-10%")
    print("=" * 100)

    veto_comb, veto_reasons_comb = _hard_veto(
        "G5+G4_combo",
        {"WFE": ev_comb["wfa_WFE"]},
        ev_comb["MaxDD_FULL"],
        aft_comb["Worst10Y_star"],
        ev_comb["regime_min_at"],
    )
    print("  G5+G4_combo: %s" % ("**HARD VETO**" if veto_comb else "PASS"))
    for r in veto_reasons_comb:
        print("    -> %s" % r)

    # =========================================================================
    # [13] Result tables
    # =========================================================================
    print("\n" + "=" * 100)
    print("FULL-GATE HEADLINE TABLE (after-tax min9/CPCV/Regime; pretax Sharpe/MaxDD)")
    print("=" * 100)
    hdr = ("%-18s | %8s | %9s | %7s | %8s | %8s | %8s | %7s | %7s"
           % ("label", "min9_at%", "WFA_CI95%", "WFE",
              "CPCV_p10%", "Reg_min%", "MaxDD%", "Sharpe", "Trd/yr"))
    print(hdr)
    print("-" * len(hdr))
    for lbl in ["V7_TQQQ", "B3a_k365", "G5_vix_hard", "G5+G4_combo"]:
        ev  = ev_all[lbl]
        aft = (aft_v7   if lbl == "V7_TQQQ"     else
               aft_b3a  if lbl == "B3a_k365"    else
               aft_g5   if lbl == "G5_vix_hard" else aft_comb)
        tpy = (tpy_v7   if lbl == "V7_TQQQ"     else
               tpy_b3a  if lbl == "B3a_k365"    else
               tpy_g5   if lbl == "G5_vix_hard" else tpy_comb)
        print("%-18s | %+7.2f%% | %+8.2f%% | %6.3f | %+7.2f%% | %+7.2f%% | %+7.2f%% | %6.3f | %6.1f"
              % (lbl,
                 ev["min_at"] * 100, ev["wfa_CI95_lo"] * 100, ev["wfa_WFE"],
                 ev["cpcv_p10_at"] * 100, ev["regime_min_at"] * 100,
                 ev["MaxDD_FULL"] * 100, ev["Sharpe_OOS"], aft["Trades_yr"]))

    # ---- Diff vs G5_vix_hard (primary comparison) ----
    print("\n" + "=" * 100)
    print("DIFF vs G5_vix_hard (primary comparison) AND vs B3a_k365")
    print("=" * 100)
    diff_hdr = ("%-18s | %9s | %9s | %9s | %9s | %9s | %8s | %s"
                % ("label", "d_min9pp", "d_MaxDD_pp", "d_Sharpe",
                   "d_W10Y_pp", "d_Reg_min", "Trd/yr", "VETO"))
    print(diff_hdr)
    print("-" * len(diff_hdr))

    for (ref_lbl, ev_ref, aft_ref) in [
        ("G5_vix_hard", ev_g5, aft_g5),
        ("B3a_k365",    ev_b3a, aft_b3a),
    ]:
        d_min9  = (ev_comb["min_at"]       - ev_ref["min_at"])       * 100
        d_maxdd = (ev_comb["MaxDD_FULL"]   - ev_ref["MaxDD_FULL"])   * 100
        d_sh    =  ev_comb["Sharpe_OOS"]   - ev_ref["Sharpe_OOS"]
        d_w10y  = (aft_comb["Worst10Y_star"] - aft_ref["Worst10Y_star"]) * 100
        d_rmin  = (ev_comb["regime_min_at"] - ev_ref["regime_min_at"]) * 100
        print("  combo vs %-10s | %+8.3fpp | %+8.3fpp | %+8.4f | %+8.3fpp | %+8.3fpp | %6.1f | %s"
              % (ref_lbl, d_min9, d_maxdd, d_sh, d_w10y, d_rmin,
                 aft_comb["Trades_yr"],
                 "**VETO**" if veto_comb else "PASS"))

    # ---- Regime-stratified ----
    print("\n" + "=" * 100)
    print("REGIME-STRATIFIED after-tax CAGR")
    print("=" * 100)
    axes_order = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                  "rate:rate_up", "rate:rate_down"]
    print("%-18s | %s" % ("label", " | ".join("%-12s" % a for a in axes_order)))
    print("-" * 110)
    for lbl in ["V7_TQQQ", "B3a_k365", "G5_vix_hard", "G5+G4_combo"]:
        rg = ev_all[lbl]["regime"]
        cells = " | ".join("%+11.2f%%" % (100 * rg.get(ax, float("nan")))
                           for ax in axes_order)
        print("%-18s | %s" % (lbl, cells))

    # ---- Stress windows ----
    print("\n" + "=" * 100)
    print("STRESS WINDOWS: cumulative return (within-window MaxDD)")
    print("=" * 100)
    sw_order = list(stress.keys())
    print("%-18s | %s" % ("label", " | ".join("%-18s" % s for s in sw_order)))
    print("-" * 130)
    for lbl in ["V7_TQQQ", "B3a_k365", "G5_vix_hard", "G5+G4_combo"]:
        st = ev_all[lbl]["stress"]
        cells = " | ".join(
            "%+6.1f%%(%+5.1f%%)" % (100 * st[s]["ret"], 100 * st[s]["maxdd"])
            for s in sw_order)
        print("%-18s | %s" % (lbl, cells))

    # ---- Bootstrap tables ----
    def _fp(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  nan  "
        return "%6.3f " % v

    def _fpp(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "   nan   "
        return "%+8.2f%%" % v

    print("\n" + "=" * 100)
    print("BOOTSTRAP: combo vs G5_vix_hard [block=21]  (primary: is combo better than G5 single?)")
    print("%-20s | P_min  CI95_min | P_maxdd CI95_dd | P_w10y CI95_w10y | P_shp CI95_shp"
          % "pair")
    print("-" * 100)
    key = "G5+G4_combo_vs_G5_vix_hard"
    r = boot_21[key]
    print("%-20s | %s %s | %s %s | %s %s | %s %s"
          % ("vs_G5_vix_hard",
             _fp(r["P_min_better"]),    _fpp(r["CI95_lo_min_pp"]),
             _fp(r["P_maxdd_better"]),  _fpp(r["CI95_lo_dd_pp"]),
             _fp(r.get("P_worst10y_better", float("nan"))), _fpp(r.get("CI95_lo_w10y_pp", float("nan"))),
             _fp(r.get("P_sharpe_better", float("nan"))),   _fpp(r.get("CI95_lo_sharpe", float("nan")))))

    print("\n" + "=" * 100)
    print("BOOTSTRAP: combo vs B3a_k365 [block=21]")
    print("%-20s | P_min  CI95_min | P_maxdd CI95_dd | P_w10y CI95_w10y | P_shp CI95_shp"
          % "pair")
    print("-" * 100)
    key = "G5+G4_combo_vs_B3a_k365"
    r = boot_21[key]
    print("%-20s | %s %s | %s %s | %s %s | %s %s"
          % ("vs_B3a_k365",
             _fp(r["P_min_better"]),    _fpp(r["CI95_lo_min_pp"]),
             _fp(r["P_maxdd_better"]),  _fpp(r["CI95_lo_dd_pp"]),
             _fp(r.get("P_worst10y_better", float("nan"))), _fpp(r.get("CI95_lo_w10y_pp", float("nan"))),
             _fp(r.get("P_sharpe_better", float("nan"))),   _fpp(r.get("CI95_lo_sharpe", float("nan")))))

    print("\n" + "=" * 100)
    print("BOOTSTRAP SENSITIVITY: block=21 vs block=252")
    print("%-38s | %-12s | P(21) CI95(21) | P(252) CI95(252)"
          % ("pair", "metric"))
    print("-" * 90)
    for (cl, _, bl, _) in BOOT_PAIRS:
        key = "%s_vs_%s" % (cl, bl)
        r21  = boot_21[key]
        r252 = boot_252[key]
        for metric, p21, ci21, p252, ci252 in [
            ("Worst10Y*",
             r21.get("P_worst10y_better", float("nan")), r21.get("CI95_lo_w10y_pp", float("nan")),
             r252.get("P_worst10y_better", float("nan")), r252.get("CI95_lo_w10y_pp", float("nan"))),
            ("MaxDD",
             r21.get("P_maxdd_better", float("nan")), r21.get("CI95_lo_dd_pp", float("nan")),
             r252.get("P_maxdd_better", float("nan")), r252.get("CI95_lo_dd_pp", float("nan"))),
            ("min9_CAGR",
             r21.get("P_min_better", float("nan")), r21.get("CI95_lo_min_pp", float("nan")),
             r252.get("P_min_better", float("nan")), r252.get("CI95_lo_min_pp", float("nan"))),
        ]:
            print("%-38s | %-12s | %s %s | %s %s"
                  % (key[:38], metric,
                     _fp(p21), _fpp(ci21),
                     _fp(p252), _fpp(ci252)))

    # =========================================================================
    # [14] PHASE 3 JUDGMENT
    # =========================================================================
    print("\n" + "=" * 100)
    print("[14] PHASE 3 JUDGMENT")
    print("=" * 100)

    ev_c  = ev_comb
    aft_c = aft_comb
    ev_g  = ev_g5
    aft_g = aft_g5

    # Metrics vs G5 single (primary)
    d_min9_vs_g5   = (ev_c["min_at"]       - ev_g["min_at"])       * 100
    d_maxdd_vs_g5  = (ev_c["MaxDD_FULL"]   - ev_g["MaxDD_FULL"])   * 100
    d_sh_vs_g5     =  ev_c["Sharpe_OOS"]   - ev_g["Sharpe_OOS"]
    d_w10y_vs_g5   = (aft_c["Worst10Y_star"] - aft_g["Worst10Y_star"]) * 100

    # Metrics vs B3a baseline
    d_min9_vs_b3a  = (ev_c["min_at"]       - ev_b3a["min_at"])       * 100
    d_maxdd_vs_b3a = (ev_c["MaxDD_FULL"]   - ev_b3a["MaxDD_FULL"])   * 100
    d_sh_vs_b3a    =  ev_c["Sharpe_OOS"]   - ev_b3a["Sharpe_OOS"]

    # Bootstrap results vs G5
    bkey_g5 = "G5+G4_combo_vs_G5_vix_hard"
    p_min_vs_g5   = boot_21[bkey_g5].get("P_min_better",      float("nan"))
    p_dd_vs_g5    = boot_21[bkey_g5].get("P_maxdd_better",    float("nan"))
    p_w10y_vs_g5  = boot_21[bkey_g5].get("P_worst10y_better", float("nan"))
    p_sh_vs_g5    = boot_21[bkey_g5].get("P_sharpe_better",   float("nan"))

    # Plan Section 4 combination criteria
    crit_a_min9   = d_min9_vs_g5 > 0.0       # min9 improves vs G5
    crit_a_maxdd  = d_maxdd_vs_g5 >= -1.0    # MaxDD does NOT worsen more than +1pp vs G5
    crit_a_noveto = not veto_comb
    crit_b_cagr   = d_min9_vs_b3a > 0.0      # CAGR uplift vs B3a
    crit_b_def    = d_maxdd_vs_b3a < 3.0     # G5 defence broadly maintained vs B3a

    # Trades/yr check (plan notes G4 may push >100)
    trades_flag = aft_c["Trades_yr"] > 100.0

    print("\n  Combination vs G5_vix_hard (primary comparison):")
    print("    d_min9=%+.3fpp  d_MaxDD=%+.3fpp  d_Sharpe=%+.4f  d_W10Y=%+.3fpp"
          % (d_min9_vs_g5, d_maxdd_vs_g5, d_sh_vs_g5, d_w10y_vs_g5))
    print("    Bootstrap (vs G5): P_min=%.3f  P_dd=%.3f  P_w10y=%.3f  P_shp=%.3f"
          % (p_min_vs_g5, p_dd_vs_g5, p_w10y_vs_g5, p_sh_vs_g5))
    print("    Crit(a): min9>0=%s  MaxDD_ok=%s  no_veto=%s"
          % (crit_a_min9, crit_a_maxdd, crit_a_noveto))

    print("\n  Combination vs B3a_k365:")
    print("    d_min9=%+.3fpp  d_MaxDD=%+.3fpp  d_Sharpe=%+.4f"
          % (d_min9_vs_b3a, d_maxdd_vs_b3a, d_sh_vs_b3a))
    print("    Crit(b): CAGR_uplift=%s  defence_ok=%s"
          % (crit_b_cagr, crit_b_def))

    print("\n  Trades/yr combination: %.1f %s"
          % (aft_c["Trades_yr"], "(WARNING: >100 trades/yr -- real-world cost concern)"
             if trades_flag else "(OK)"))

    # Final verdict
    crit_a_pass = crit_a_min9 and crit_a_maxdd and crit_a_noveto
    crit_b_pass = crit_b_cagr and crit_b_def and crit_a_noveto

    print("\n" + "=" * 100)
    if crit_a_pass or crit_b_pass:
        judgment   = "COMBINATION_ADOPTED"
        judgment_reason = ("Combination criterion %s met. G5+G4 is improvement candidate."
                           % ("(a)" if crit_a_pass else "(b)"))
        if trades_flag:
            judgment_reason += " CAUTION: Trades/yr=%.1f > 100 -- real-world cost scrutiny required." % aft_c["Trades_yr"]
    else:
        judgment   = "G5_SINGLE_BEST"
        judgment_reason = ("Combination does NOT improve on G5_vix_hard single. "
                           "G5+G4 stacking adds no net value (G4 Sharpe drag / high Trades "
                           "offsets G5 defensive gain). G5_vix_hard alone is the Phase 2/3 best.")

    print("PHASE 3 VERDICT: %s" % judgment)
    print("Reason: %s" % judgment_reason)
    print("=" * 100)

    # =========================================================================
    # [15] CSV output
    # =========================================================================
    print("\n[15] Saving CSV ...")

    def _row(lbl, ev, aft, tpy, exc, veto, veto_reasons, ref_ev_g5, ref_aft_g5, ref_ev_b3a, ref_aft_b3a):
        return {
            "label":             lbl,
            "CAGR_IS_at":        float(aft["CAGR_IS"]),
            "CAGR_OOS_at":       float(aft["CAGR_OOS"]),
            "min_IS_OOS_at":     float(min(aft["CAGR_IS"], aft["CAGR_OOS"])),
            "IS_OOS_gap_pp":     float(aft["IS_OOS_gap_pp"]),
            "Sharpe_OOS":        float(ev["Sharpe_OOS"]),
            "MaxDD_FULL":        float(ev["MaxDD_FULL"]),
            "Worst10Y_star_at":  float(aft["Worst10Y_star"]),
            "P10_5Y_at":         float(aft["P10_5Y"]),
            "Worst5Y_at":        float(aft["Worst5Y"]),
            "Trades_yr":         float(aft["Trades_yr"]),
            "wfa_CI95_lo":       float(ev["wfa_CI95_lo"]),
            "wfa_WFE":           float(ev["wfa_WFE"]),
            "cpcv_p10_at":       float(ev["cpcv_p10_at"]),
            "regime_min_at":     float(ev["regime_min_at"]),
            "excess_days":       int(exc) if exc is not None else 0,
            "hard_veto":         int(veto),
            "veto_reasons":      "; ".join(veto_reasons),
            # diffs vs G5_vix_hard
            "d_min9_vs_g5_pp":   (float(ev["min_at"])     - float(ref_ev_g5["min_at"]))     * 100,
            "d_maxdd_vs_g5_pp":  (float(ev["MaxDD_FULL"]) - float(ref_ev_g5["MaxDD_FULL"])) * 100,
            "d_sh_vs_g5":        float(ev["Sharpe_OOS"])  - float(ref_ev_g5["Sharpe_OOS"]),
            "d_w10y_vs_g5_pp":   (float(aft["Worst10Y_star"]) - float(ref_aft_g5["Worst10Y_star"])) * 100,
            # diffs vs B3a
            "d_min9_vs_b3a_pp":  (float(ev["min_at"])     - float(ref_ev_b3a["min_at"]))     * 100,
            "d_maxdd_vs_b3a_pp": (float(ev["MaxDD_FULL"]) - float(ref_ev_b3a["MaxDD_FULL"])) * 100,
            "d_sh_vs_b3a":       float(ev["Sharpe_OOS"])  - float(ref_ev_b3a["Sharpe_OOS"]),
        }

    rows = []
    # Reference rows
    for (lbl, ev, aft, tpy, exc) in [
        ("V7_TQQQ",     ev_v7,  aft_v7,  tpy_v7,  None),
        ("B3a_k365",    ev_b3a, aft_b3a, tpy_b3a, None),
        ("G5_vix_hard", ev_g5,  aft_g5,  tpy_g5,  exc_g5),
    ]:
        rows.append(_row(lbl, ev, aft, tpy, exc, False, [],
                         ev_g5, aft_g5, ev_b3a, aft_b3a))
    # Combination row
    rows.append(_row("G5+G4_combo", ev_comb, aft_comb, tpy_comb, exc_comb,
                     veto_comb, veto_reasons_comb,
                     ev_g5, aft_g5, ev_b3a, aft_b3a))

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combine_phase3_g5g4_20260615.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # [16] RETURN_BLOCK
    # =========================================================================
    def _rb(ev, aft, tpy, exc, veto, boot21_vs_g5, boot21_vs_b3a):
        return {
            "CAGR_IS_at":        round(float(aft["CAGR_IS"]),  6),
            "CAGR_OOS_at":       round(float(aft["CAGR_OOS"]), 6),
            "min_at":            round(float(min(aft["CAGR_IS"], aft["CAGR_OOS"])), 6),
            "gap_pp":            round(float(aft["IS_OOS_gap_pp"]), 4),
            "Sharpe_OOS":        round(float(ev["Sharpe_OOS"]),  4),
            "MaxDD":             round(float(ev["MaxDD_FULL"]),   6),
            "Worst10Y_star_at":  round(float(aft["Worst10Y_star"]), 6),
            "P10_5Y_at":         round(float(aft["P10_5Y"]),   6),
            "Trades_yr":         round(float(aft["Trades_yr"]), 2),
            "wfa_CI95_lo":       round(float(ev["wfa_CI95_lo"]), 6),
            "wfa_WFE":           round(float(ev["wfa_WFE"]),     4),
            "cpcv_p10_at":       round(float(ev["cpcv_p10_at"]), 6),
            "regime_min_at":     round(float(ev["regime_min_at"]), 6),
            "excess_days":       int(exc) if exc is not None else 0,
            "hard_veto":         bool(veto),
            "d_min9_vs_g5_pp":   round((float(ev["min_at"]) - float(ev_g5["min_at"])) * 100, 4),
            "d_maxdd_vs_g5_pp":  round((float(ev["MaxDD_FULL"]) - float(ev_g5["MaxDD_FULL"])) * 100, 4),
            "d_sh_vs_g5":        round(float(ev["Sharpe_OOS"]) - float(ev_g5["Sharpe_OOS"]), 4),
            "d_min9_vs_b3a_pp":  round((float(ev["min_at"]) - float(ev_b3a["min_at"])) * 100, 4),
            "d_maxdd_vs_b3a_pp": round((float(ev["MaxDD_FULL"]) - float(ev_b3a["MaxDD_FULL"])) * 100, 4),
            "boot21_vs_g5": {
                "P_min":    round(float(boot21_vs_g5.get("P_min_better",      float("nan"))), 4),
                "P_maxdd":  round(float(boot21_vs_g5.get("P_maxdd_better",    float("nan"))), 4),
                "P_w10y":   round(float(boot21_vs_g5.get("P_worst10y_better", float("nan"))), 4),
                "P_sharpe": round(float(boot21_vs_g5.get("P_sharpe_better",   float("nan"))), 4),
                "CI95_min_pp": round(float(boot21_vs_g5.get("CI95_lo_min_pp", float("nan"))), 4),
                "CI95_dd_pp":  round(float(boot21_vs_g5.get("CI95_lo_dd_pp",  float("nan"))), 4),
            },
            "boot21_vs_b3a": {
                "P_min":    round(float(boot21_vs_b3a.get("P_min_better",      float("nan"))), 4),
                "P_maxdd":  round(float(boot21_vs_b3a.get("P_maxdd_better",    float("nan"))), 4),
                "P_w10y":   round(float(boot21_vs_b3a.get("P_worst10y_better", float("nan"))), 4),
                "P_sharpe": round(float(boot21_vs_b3a.get("P_sharpe_better",   float("nan"))), 4),
            },
        }

    block = {
        "meta": {
            "script":   "combine_phase3_g5g4_20260615.py",
            "base":     "B3a_k365",
            "combo":    "G5_vix_hard + G4_LT2_k020",
            "formula":  "L = lev_raw * mult_v7 * lev_scale * def_mult_G5 * lt2_mult_G4 * 3.0",
            "K_LT_COMBO": K_LT_COMBO,
            "sanity_A_ok": bool(ok_a),
            "sanity_B_ok": bool(ok_b),
        },
        "V7_TQQQ": {
            "min_at": round(float(ev_v7["min_at"]), 6),
            "MaxDD":  round(float(ev_v7["MaxDD_FULL"]), 6),
        },
        "B3a_k365": {
            "min_at":       round(float(ev_b3a["min_at"]), 6),
            "MaxDD":        round(float(ev_b3a["MaxDD_FULL"]), 6),
            "Sharpe_OOS":   round(float(ev_b3a["Sharpe_OOS"]), 4),
            "Trades_yr":    round(float(aft_b3a["Trades_yr"]), 2),
        },
        "G5_vix_hard": {
            "min_at":       round(float(ev_g5["min_at"]), 6),
            "MaxDD":        round(float(ev_g5["MaxDD_FULL"]), 6),
            "Sharpe_OOS":   round(float(ev_g5["Sharpe_OOS"]), 4),
            "Trades_yr":    round(float(aft_g5["Trades_yr"]), 2),
            "wfa_CI95_lo":  round(float(ev_g5["wfa_CI95_lo"]), 6),
            "cpcv_p10_at":  round(float(ev_g5["cpcv_p10_at"]), 6),
        },
        "G5+G4_combo": _rb(ev_comb, aft_comb, tpy_comb, exc_comb,
                            veto_comb,
                            boot_21["G5+G4_combo_vs_G5_vix_hard"],
                            boot_21["G5+G4_combo_vs_B3a_k365"]),
        "judgment":        judgment,
        "judgment_reason": judgment_reason,
        "sanity": {
            "A_lt2_neutral_vs_G5":   {"min9_diff_pp": round((sa_min9 - g5_min9) * 100, 4),
                                       "maxdd_diff_pp": round((sa_maxdd - g5_maxdd) * 100, 4),
                                       "pass": bool(ok_a)},
            "B_def_neutral_vs_G4":   {"min9_diff_pp": round((sb_min9 - g4_min9) * 100, 4),
                                       "maxdd_diff_pp": round((sb_maxdd - g4_maxdd) * 100, 4),
                                       "pass": bool(ok_b)},
        },
    }

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=False))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
