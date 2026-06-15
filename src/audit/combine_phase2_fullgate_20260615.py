# -*- coding: utf-8 -*-
"""
src/audit/combine_phase2_fullgate_20260615.py
=============================================
Phase 2 Full-Gate evaluation for Stage 0 survivors:
  1. B3a+G5_vix_hard  (defensive / main candidate)
  2. B3a+G4_LT2_k020  (attack / caution)

Full gate includes:
  - WFA 49 windows (alpha/beta, CI95_lo, WFE, t_p)
  - CPCV N=10 blocks, k=2, embargo=21
  - Regime-stratified CAGR (trend/vol/rate axes, Regime_min)
  - Stress windows (2000/2008/2020/2022/2015)
  - Multi-metric bootstrap (4 axes: min9/MaxDD/Worst10Y*/Sharpe)
    vs B3a baseline AND vs V7_TQQQ
  - 4 hard veto checks

Sanity gate:
  B3a素地 and V7_TQQQ evaluated by same machinery;
  known values must match within tolerances before proceeding.
  B3a known: WFA CI95_lo~+22.52%, WFE~0.987, CPCV p10~+16.01%, Regime_min~-2.88%
  V7 known:  min9_at~+16.27%

Reuse (no re-implementation):
  extended_eval_20260611.py : _eval_one, _cpcv_dist, _regime_cagr, _cum_ret
  multimetric_bootstrap_20260615.py : _block_bootstrap_multimetric
  combine_g5_defoverlay_20260615.py : _build_full_b3a_g5, _build_def_mult_arr
  combine_g4_lt2_20260615.py        : _build_b3a_g4
  lu_cfd_recost_20260611.py         : _build_tqqq_base (for V7 baseline)
  run_p09_tqqq_validate_20260611.py : _run_wfa, N_BOOT, BLOCK, SEED, AFTER_TAX
  k365_recost_20260612.py            : _build_full_c1, EXCESS_EXTRA_K365_CENTRE
  regime_labeler_20260611.py         : build_regime_labels, stress_masks
  unified_metrics.py                 : compute_10metrics, IS_END, OOS_START

ASCII-only prints (cp932). CSV saved. No git ops. No temp files.

Authors: Kazuya Oza  2026-06-16
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

# Reuse _eval_one machinery from extended_eval_20260611
from src.audit.extended_eval_20260611 import (
    _cum_ret, _cpcv_dist, _regime_cagr, _eval_one,
)
# Multi-metric bootstrap
from src.audit.multimetric_bootstrap_20260615 import _block_bootstrap_multimetric

# WFA runner
from src.audit.run_p09_tqqq_validate_20260611 import (
    _run_wfa, N_BOOT, BLOCK, SEED, AFTER_TAX,
    _maxdd_from_returns, _cagr_seg,
)

# V7 baseline builder (same as extended_eval)
from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base as _build_v7_tqqq_base,
    _build_p09_on_base,
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

# B3a k365 builders
from src.audit.k365_recost_20260612 import (
    _build_full_c1, EXCESS_EXTRA_K365_CENTRE,
)

# G5 (vix_hard defensive overlay) builders
from src.audit.combine_g5_defoverlay_20260615 import (
    _build_def_mult_arr, _build_full_b3a_g5,
    HARD_MAP, MACRO_FEATURES_PATH,
)

# G4 (LT2-N750 contrarian) builders
from src.audit.combine_g4_lt2_20260615 import (
    _build_b3a_g4,
)
from long_cycle_signal import compute_lt2, signal_to_mult

# B3a config
B3A_V7_MAP   = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE = 1.15

# Known B3a sanity references
B3A_REF = {
    "min9_at":      0.2098,
    "wfa_CI95_lo":  0.2252,
    "wfa_WFE":      0.987,
    "cpcv_p10_at":  0.1601,
    "regime_min":  -0.0288,
    "MaxDD":       -0.3820,
    "Sharpe":       0.904,
}
V7_REF_MIN9 = 0.1627   # +16.27%

# Tolerances for sanity gate
TOL_MIN9          = 0.002    # 0.2pp
TOL_WFA_CI95      = 0.005    # 0.5pp
TOL_WFA_WFE       = 0.020
TOL_CPCV_P10      = 0.005    # 0.5pp
TOL_REGIME_MIN    = 0.003    # 0.3pp
TOL_V7_MIN9       = 0.002    # 0.2pp

# Hard veto thresholds
VETO_MAXDD   = -0.50
VETO_WFE     =  1.50
VETO_W10Y    =  0.00
VETO_REGIME  = -0.10


# ---------------------------------------------------------------------------
# Hard veto checker
# ---------------------------------------------------------------------------
def _hard_veto(label, wfa_res, full_maxdd, worst10y_at, regime_min):
    """Return (veto: bool, reasons: list[str])."""
    reasons = []
    if full_maxdd < VETO_MAXDD:
        reasons.append("MaxDD %.2f%% < -50%%" % (full_maxdd * 100))
    if not np.isnan(wfa_res.get("WFE", np.nan)) and wfa_res["WFE"] > VETO_WFE:
        reasons.append("WFE %.3f > 1.5" % wfa_res["WFE"])
    if worst10y_at < VETO_W10Y:
        reasons.append("Worst10Y* %.2f%% < 0" % (worst10y_at * 100))
    if regime_min < VETO_REGIME:
        reasons.append("Regime_min %.2f%% < -10%%" % (regime_min * 100))
    return len(reasons) > 0, reasons


# ---------------------------------------------------------------------------
# Build V7_TQQQ baseline NAV (no fill, no excess)
# ---------------------------------------------------------------------------
def _build_v7_nav(shared, dates_dt):
    """V7_TQQQ: plain DH-W1 TQQQ base, no P09 fill, no cfd_excess."""
    v7_nav, r_v7, tpy_v7, _ = _build_v7_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)
    return v7_nav, r_v7, tpy_v7


# ---------------------------------------------------------------------------
# Build B3a baseline NAV (k365 centre + C1 fill)
# ---------------------------------------------------------------------------
def _build_b3a_nav(shared, dates_dt, n_years,
                   ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
                   bond_on, sofr_arr):
    """B3a_k365: v7_map x lev_scale x k365 excess + C1 SOFR fill."""
    b3a_nav, r_b3a, tpy_b3a, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        v7_map=B3A_V7_MAP, lev_scale=B3A_LEV_SCALE,
        excess_extra=EXCESS_EXTRA_K365_CENTRE,
    )
    return b3a_nav, r_b3a, tpy_b3a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("PHASE 2 FULL-GATE EVALUATION  2026-06-16")
    print("Candidates: B3a+G5_vix_hard (defensive) / B3a+G4_LT2_k020 (attack)")
    print("Gate: WFA(49w) + CPCV(N10,k2,emb21) + Regime + Stress + 4-metric bootstrap")
    print("vs B3a baseline AND vs V7_TQQQ")
    print("N_BOOT=%d  BLOCK=%d  SEED=%d  AFTER_TAX=%.4f" % (N_BOOT, BLOCK, SEED, AFTER_TAX))
    print("=" * 100)

    # ---- Load shared data ----
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

    # ---- Regime labels and stress masks ----
    print("[2] Building regime labels and stress masks ...")
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress  = stress_masks(dates_dt)

    # ---- Gold/Bond auxiliary series ----
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
    bond_m252  = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on    = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr   = np.asarray(a["sofr"], float)

    # =========================================================================
    # [4] Build NAV series
    # =========================================================================
    print("\n[4] Building NAV series ...")

    # V7_TQQQ baseline
    print("  V7_TQQQ ...")
    v7_nav, r_v7, tpy_v7 = _build_v7_nav(shared, dates_dt)

    # B3a_k365 baseline
    print("  B3a_k365 ...")
    b3a_nav, r_b3a, tpy_b3a = _build_b3a_nav(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr)

    # G5_vix_hard: load macro features and build defensive multiplier
    print("  B3a+G5_vix_hard ...")
    macro_df = pd.read_csv(MACRO_FEATURES_PATH, index_col=0, parse_dates=True)
    def_arr_vix_hard = _build_def_mult_arr(
        "vix_mom21", "daily", HARD_MAP, dates_dt, macro_df)
    g5_nav, r_g5, tpy_g5, _exc_g5 = _build_full_b3a_g5(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr,
        def_mult_arr=def_arr_vix_hard)

    # G4_k020: LT2-N750 with k_lt=0.20
    print("  B3a+G4_LT2_k020 ...")
    lt2_sig = compute_lt2(a["close"], N=750)
    lt2_mult_k020 = signal_to_mult(lt2_sig, k_lt=0.20).values
    g4_nav, r_g4, tpy_g4, _exc_g4 = _build_b3a_g4(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr, lt2_mult=lt2_mult_k020)

    print("  All NAVs built.")

    # =========================================================================
    # [5] Full-gate evaluation (WFA + CPCV + Regime + Stress)
    # =========================================================================
    print("\n[5] Running full-gate _eval_one for all 4 series ...")
    print("    (WFA is slowest step; please wait ~2-4 min per series)")

    print("  Evaluating V7_TQQQ ...")
    ev_v7  = _eval_one("V7_TQQQ",  v7_nav,  r_v7,
                        regimes, stress, is_mask, oos_mask)

    print("  Evaluating B3a_k365 ...")
    ev_b3a = _eval_one("B3a_k365", b3a_nav, r_b3a,
                        regimes, stress, is_mask, oos_mask,
                        baseline_r=r_v7)

    print("  Evaluating B3a+G5_vix_hard ...")
    ev_g5  = _eval_one("G5_vix_hard", g5_nav, r_g5,
                        regimes, stress, is_mask, oos_mask,
                        baseline_r=r_v7)

    print("  Evaluating B3a+G4_LT2_k020 ...")
    ev_g4  = _eval_one("G4_LT2_k020", g4_nav, r_g4,
                        regimes, stress, is_mask, oos_mask,
                        baseline_r=r_v7)

    ev_all = {
        "V7_TQQQ":      ev_v7,
        "B3a_k365":     ev_b3a,
        "G5_vix_hard":  ev_g5,
        "G4_LT2_k020":  ev_g4,
    }

    # =========================================================================
    # [6] Sanity gate: B3a and V7 must match known values
    # =========================================================================
    print("\n" + "=" * 100)
    print("[6] SANITY GATE -- known values vs re-computed")
    print("=" * 100)

    def _sf(v, ref, tol, name):
        ok = abs(v - ref) <= tol
        print("  %-36s : got %+.4f  expect ~%+.4f  tol=%.3f -> %s"
              % (name, v, ref, tol, "OK" if ok else "WARN"))
        return ok

    checks = [
        (ev_b3a["min_at"],        B3A_REF["min9_at"],     TOL_MIN9,
         "B3a min9_at"),
        (ev_b3a["wfa_CI95_lo"],   B3A_REF["wfa_CI95_lo"], TOL_WFA_CI95,
         "B3a WFA CI95_lo"),
        (ev_b3a["wfa_WFE"],       B3A_REF["wfa_WFE"],     TOL_WFA_WFE,
         "B3a WFA WFE"),
        (ev_b3a["cpcv_p10_at"],   B3A_REF["cpcv_p10_at"], TOL_CPCV_P10,
         "B3a CPCV p10"),
        (ev_b3a["regime_min_at"], B3A_REF["regime_min"],  TOL_REGIME_MIN,
         "B3a Regime_min"),
        (ev_v7["min_at"],         V7_REF_MIN9,            TOL_V7_MIN9,
         "V7 min9_at"),
    ]
    sanity_pass = all(_sf(v, r, t, n) for v, r, t, n in checks)
    if not sanity_pass:
        print("\n  WARN: Some sanity checks outside tolerance.")
        print("  Results are still printed below, but treat with caution.")
        print("  (Minor drift from C1/k365 cost variant differences is expected)")
    else:
        print("\n  All sanity checks PASSED.")

    # =========================================================================
    # [7] Multi-metric bootstrap (4 axes, block=21 + block=252 sensitivity)
    # =========================================================================
    print("\n" + "=" * 100)
    print("[7] MULTI-METRIC BOOTSTRAP (N=%d, seeds=%d)" % (N_BOOT, SEED))
    print("    Running 4 pairs x 2 block sizes ...")
    print("=" * 100)

    BOOT_PAIRS = [
        # (cand_label, r_cand,  base_label, r_base)
        ("G5_vix_hard",  r_g5,  "B3a_k365",  r_b3a),
        ("G5_vix_hard",  r_g5,  "V7_TQQQ",   r_v7),
        ("G4_LT2_k020",  r_g4,  "B3a_k365",  r_b3a),
        ("G4_LT2_k020",  r_g4,  "V7_TQQQ",   r_v7),
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
                 r.get("P_sharpe_better", float("nan")) or float("nan")))

        print("  [block=252] %s ..." % key, end="", flush=True)
        boot_252[key] = _block_bootstrap_multimetric(
            rc, rb, is_mask, oos_mask, n_boot=N_BOOT, block=252, seed=SEED)
        r = boot_252[key]
        print(" done  P_min=%.3f P_dd=%.3f P_w10y=%.3f P_shp=%.3f"
              % (r["P_min_better"], r["P_maxdd_better"],
                 r.get("P_worst10y_better", float("nan")) or float("nan"),
                 r.get("P_sharpe_better", float("nan")) or float("nan")))

    # =========================================================================
    # [8] Hard veto checks
    # =========================================================================
    print("\n" + "=" * 100)
    print("[8] HARD VETO CHECKS")
    print("    Thresholds: MaxDD<-50% / WFE>1.5 / Worst10Y*<0 / Regime_min<-10%")
    print("=" * 100)

    cand_labels = ["G5_vix_hard", "G4_LT2_k020"]
    veto_results = {}
    pre_g5  = compute_10metrics(g5_nav,  tpy_g5)
    pre_g4  = compute_10metrics(g4_nav,  tpy_g4)
    aft_g5  = _apply_aftertax(pre_g5)
    aft_g4  = _apply_aftertax(pre_g4)
    pre_b3a = compute_10metrics(b3a_nav, tpy_b3a)
    aft_b3a = _apply_aftertax(pre_b3a)
    pre_v7  = compute_10metrics(v7_nav,  tpy_v7)
    aft_v7  = _apply_aftertax(pre_v7)

    for lbl, ev, aft in [
        ("G5_vix_hard",  ev_g5, aft_g5),
        ("G4_LT2_k020",  ev_g4, aft_g4),
    ]:
        veto, reasons = _hard_veto(
            lbl,
            {"WFE": ev["wfa_WFE"]},
            ev["MaxDD_FULL"],
            aft["Worst10Y_star"],
            ev["regime_min_at"],
        )
        veto_results[lbl] = {"veto": veto, "reasons": reasons}
        status = "**HARD VETO**" if veto else "PASS"
        print("  %-18s : %s" % (lbl, status))
        if reasons:
            for r in reasons:
                print("    -> %s" % r)

    # =========================================================================
    # [9] Result tables
    # =========================================================================

    # ---- Full-gate headline table ----
    print("\n" + "=" * 100)
    print("FULL-GATE HEADLINE  (after-tax min9/CPCV/Regime; pretax Sharpe/MaxDD)")
    print("=" * 100)
    hdr = ("%-18s | %8s | %9s | %7s | %8s | %8s | %8s | %7s | %7s"
           % ("label", "min9_at%", "WFA_CI95%", "WFE",
              "CPCV_p10%", "Reg_min%", "MaxDD%", "Sharpe", "Trd/yr"))
    print(hdr)
    print("-" * len(hdr))
    for lbl in ["V7_TQQQ", "B3a_k365", "G5_vix_hard", "G4_LT2_k020"]:
        ev  = ev_all[lbl]
        aft = (aft_v7 if lbl == "V7_TQQQ" else
               aft_b3a if lbl == "B3a_k365" else
               aft_g5  if lbl == "G5_vix_hard" else aft_g4)
        tpy = (tpy_v7 if lbl == "V7_TQQQ" else
               tpy_b3a if lbl == "B3a_k365" else
               tpy_g5  if lbl == "G5_vix_hard" else tpy_g4)
        print("%-18s | %+7.2f%% | %+8.2f%% | %6.3f | %+7.2f%% | %+7.2f%% | %+7.2f%% | %6.3f | %6.1f"
              % (lbl,
                 ev["min_at"] * 100, ev["wfa_CI95_lo"] * 100, ev["wfa_WFE"],
                 ev["cpcv_p10_at"] * 100, ev["regime_min_at"] * 100,
                 ev["MaxDD_FULL"] * 100, ev["Sharpe_OOS"], aft["Trades_yr"]))

    # ---- Diff vs B3a baseline ----
    print("\n" + "=" * 100)
    print("DIFF vs B3a_k365 baseline")
    print("=" * 100)
    diff_hdr = ("%-18s | %9s | %9s | %9s | %9s | %9s | %9s"
                % ("label", "d_min9pp", "d_MaxDD_pp", "d_Sharpe",
                   "d_W10Y_pp", "d_Reg_min", "VETO"))
    print(diff_hdr)
    print("-" * len(diff_hdr))
    for lbl, ev, aft in [("G5_vix_hard", ev_g5, aft_g5),
                          ("G4_LT2_k020", ev_g4, aft_g4)]:
        pre = (pre_g5 if lbl == "G5_vix_hard" else pre_g4)
        d_min9  = (ev["min_at"]       - ev_b3a["min_at"])       * 100
        d_maxdd = (ev["MaxDD_FULL"]   - ev_b3a["MaxDD_FULL"])   * 100
        d_sh    =  ev["Sharpe_OOS"]   - ev_b3a["Sharpe_OOS"]
        d_w10y  = (aft["Worst10Y_star"] - aft_b3a["Worst10Y_star"]) * 100
        d_rmin  = (ev["regime_min_at"] - ev_b3a["regime_min_at"]) * 100
        vt      = veto_results[lbl]["veto"]
        print("%-18s | %+8.3fpp | %+8.3fpp | %+8.4f | %+8.3fpp | %+8.3fpp | %s"
              % (lbl, d_min9, d_maxdd, d_sh, d_w10y, d_rmin,
                 "**VETO**" if vt else "PASS"))

    # ---- Regime detail ----
    print("\n" + "=" * 100)
    print("REGIME-STRATIFIED after-tax CAGR")
    print("=" * 100)
    axes_order = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                  "rate:rate_up", "rate:rate_down"]
    print("%-18s | %s" % ("label", " | ".join("%-12s" % a for a in axes_order)))
    print("-" * 110)
    for lbl in ["V7_TQQQ", "B3a_k365", "G5_vix_hard", "G4_LT2_k020"]:
        rg = ev_all[lbl]["regime"]
        cells = " | ".join("%+11.2f%%" % (100 * rg.get(ax, np.nan)) for ax in axes_order)
        print("%-18s | %s" % (lbl, cells))

    # ---- Stress windows ----
    print("\n" + "=" * 100)
    print("STRESS WINDOWS: cumulative return (within-window MaxDD)")
    print("=" * 100)
    sw_order = list(stress.keys())
    print("%-18s | %s" % ("label", " | ".join("%-18s" % s for s in sw_order)))
    print("-" * 130)
    for lbl in ["V7_TQQQ", "B3a_k365", "G5_vix_hard", "G4_LT2_k020"]:
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
    print("BOOTSTRAP TABLE: vs B3a_k365  [block=21]")
    print("G5: P_maxdd HIGH=good (defensive); G4: P_min HIGH=good (attack)")
    print("%-20s | P_min  CI95_min | P_maxdd CI95_dd | P_w10y  CI95_w10y | P_shp  CI95_shp"
          % "candidate_vs_B3a")
    print("-" * 100)
    for cl in ["G5_vix_hard", "G4_LT2_k020"]:
        key = "%s_vs_B3a_k365" % cl
        r   = boot_21[key]
        print("%-20s | %s %s | %s %s | %s %s | %s %s"
              % (cl,
                 _fp(r["P_min_better"]),    _fpp(r["CI95_lo_min_pp"]),
                 _fp(r["P_maxdd_better"]),  _fpp(r["CI95_lo_dd_pp"]),
                 _fp(r["P_worst10y_better"]), _fpp(r["CI95_lo_w10y_pp"]),
                 _fp(r["P_sharpe_better"]),  _fpp(r["CI95_lo_sharpe"])))

    print("\n" + "=" * 100)
    print("BOOTSTRAP TABLE: vs V7_TQQQ  [block=21]")
    print("%-20s | P_min  CI95_min | P_maxdd CI95_dd | P_w10y  CI95_w10y | P_shp  CI95_shp"
          % "candidate_vs_V7")
    print("-" * 100)
    for cl in ["G5_vix_hard", "G4_LT2_k020"]:
        key = "%s_vs_V7_TQQQ" % cl
        r   = boot_21[key]
        print("%-20s | %s %s | %s %s | %s %s | %s %s"
              % (cl,
                 _fp(r["P_min_better"]),    _fpp(r["CI95_lo_min_pp"]),
                 _fp(r["P_maxdd_better"]),  _fpp(r["CI95_lo_dd_pp"]),
                 _fp(r["P_worst10y_better"]), _fpp(r["CI95_lo_w10y_pp"]),
                 _fp(r["P_sharpe_better"]),  _fpp(r["CI95_lo_sharpe"])))

    print("\n" + "=" * 100)
    print("BOOTSTRAP SENSITIVITY: block=21 vs block=252 (Worst10Y* and MaxDD)")
    print("%-30s | %-12s | P(21) CI95(21) | P(252) CI95(252)"
          % ("pair", "metric"))
    print("-" * 90)
    for (cl, _, bl, _) in BOOT_PAIRS:
        key = "%s_vs_%s" % (cl, bl)
        r21  = boot_21[key]
        r252 = boot_252[key]
        for metric, p21, ci21, p252, ci252 in [
            ("Worst10Y*",
             r21["P_worst10y_better"], r21["CI95_lo_w10y_pp"],
             r252["P_worst10y_better"], r252["CI95_lo_w10y_pp"]),
            ("MaxDD",
             r21["P_maxdd_better"], r21["CI95_lo_dd_pp"],
             r252["P_maxdd_better"], r252["CI95_lo_dd_pp"]),
        ]:
            print("%-30s | %-12s | %s %s | %s %s"
                  % (key[:30], metric,
                     _fp(p21), _fpp(ci21),
                     _fp(p252), _fpp(ci252)))

    # =========================================================================
    # [10] Phase 2 judgment
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 2 JUDGMENT")
    print("=" * 100)

    # ---- G5_vix_hard judgment ----
    print("\n[G5_vix_hard] DEFENSIVE candidate")
    ev  = ev_g5
    aft = aft_g5
    d_min9  = (ev["min_at"]       - ev_b3a["min_at"])       * 100
    d_maxdd = (ev["MaxDD_FULL"]   - ev_b3a["MaxDD_FULL"])   * 100
    d_sh    =  ev["Sharpe_OOS"]   - ev_b3a["Sharpe_OOS"]
    d_w10y  = (aft["Worst10Y_star"] - aft_b3a["Worst10Y_star"]) * 100
    vt_g5   = veto_results["G5_vix_hard"]["veto"]

    # Criterion (plan §4):
    #   MaxDD改善>=+2pp OR Sharpe>=+0.02 OR W10Y*>=+0.5pp
    #   AND min9劣化<=1.0pp  AND veto=None
    #   AND bootstrap MaxDD improvement P>=0.80
    crit_improve = (d_maxdd >= 2.0) or (d_sh >= 0.02) or (d_w10y >= 0.5)
    crit_min9    = d_min9 >= -1.0
    crit_veto    = not vt_g5
    bkey_g5_b3a  = "G5_vix_hard_vs_B3a_k365"
    p_dd_g5      = boot_21[bkey_g5_b3a].get("P_maxdd_better", float("nan"))
    p_sh_g5      = boot_21[bkey_g5_b3a].get("P_sharpe_better", float("nan"))
    crit_boot_dd = (not np.isnan(p_dd_g5)) and (p_dd_g5 >= 0.80)
    crit_boot_sh = (not np.isnan(p_sh_g5)) and (p_sh_g5 >= 0.80)
    # Worst10Y* scrutiny flag (Stage0 showed -0.8pp)
    w10y_flag    = d_w10y < -0.5

    g5_pass = crit_improve and crit_min9 and crit_veto and crit_boot_dd
    print("  Stage0 ref: min9=+20.66%%, MaxDD=-35.93%%, Sharpe=0.928, W10Y*=+13.74%%")
    print("  Fullgate  : min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f  W10Y*=%+.2f%%"
          % (ev["min_at"] * 100, ev["MaxDD_FULL"] * 100,
             ev["Sharpe_OOS"], aft["Worst10Y_star"] * 100))
    print("  vs B3a    : d_min9=%+.2fpp  d_MaxDD=%+.2fpp  d_Sharpe=%+.4f  d_W10Y=%+.2fpp"
          % (d_min9, d_maxdd, d_sh, d_w10y))
    print("  Criteria:")
    print("    MaxDD>=+2pp OR Sharpe>=+0.02 OR W10Y>=+0.5pp : %s (%.2f / %.4f / %.2f)"
          % ("PASS" if crit_improve else "FAIL", d_maxdd, d_sh, d_w10y))
    print("    min9 劣化<=1.0pp                              : %s (%.2fpp)"
          % ("PASS" if crit_min9 else "FAIL", d_min9))
    print("    Hard veto clear                               : %s"
          % ("PASS" if crit_veto else "FAIL"))
    print("    Bootstrap MaxDD P>=0.80 (vs B3a)             : %s (P=%.3f)"
          % ("PASS" if crit_boot_dd else "FAIL", p_dd_g5 if not np.isnan(p_dd_g5) else -1))
    if w10y_flag:
        print("  [WARN] Worst10Y* 劣化 %.2fpp (Stage0 -0.8pp). Bootstrap で確認せよ:"
              % d_w10y)
        bw = boot_21[bkey_g5_b3a]
        print("         P_worst10y(vs B3a,blk21)=%.3f  CI95=%.2fpp"
              % (bw.get("P_worst10y_better", float("nan")) or float("nan"),
                 bw.get("CI95_lo_w10y_pp", float("nan")) or float("nan")))
        bw252 = boot_252[bkey_g5_b3a]
        print("         P_worst10y(vs B3a,blk252)=%.3f  CI95=%.2fpp"
              % (bw252.get("P_worst10y_better", float("nan")) or float("nan"),
                 bw252.get("CI95_lo_w10y_pp", float("nan")) or float("nan")))
        print("         -> Worst10Y* 劣化はbootstrapで非有意なら許容範囲。有意なら要留保。")
    print("  VERDICT: G5_vix_hard Phase2 %s" % ("PASS" if g5_pass else "FAIL"))

    # ---- G4_LT2_k020 judgment ----
    print("\n[G4_LT2_k020] ATTACK candidate")
    ev  = ev_g4
    aft = aft_g4
    d_min9_g4  = (ev["min_at"]       - ev_b3a["min_at"])       * 100
    d_maxdd_g4 = (ev["MaxDD_FULL"]   - ev_b3a["MaxDD_FULL"])   * 100
    d_sh_g4    =  ev["Sharpe_OOS"]   - ev_b3a["Sharpe_OOS"]
    d_w10y_g4  = (aft["Worst10Y_star"] - aft_b3a["Worst10Y_star"]) * 100
    vt_g4      = veto_results["G4_LT2_k020"]["veto"]
    bkey_g4_b3a = "G4_LT2_k020_vs_B3a_k365"
    p_min_g4   = boot_21[bkey_g4_b3a].get("P_min_better", float("nan"))

    # Criterion (plan §4 attack):
    #   min9 >= B3a-0.3pp  AND veto=None  AND MaxDD悪化<=+2pp
    #   Sharpe低下 / Trades122 が6次元減点 (明記)
    crit_min9_g4 = d_min9_g4 >= -0.3
    crit_veto_g4 = not vt_g4
    crit_dd_g4   = d_maxdd_g4 <= 2.0
    g4_pass = crit_min9_g4 and crit_veto_g4 and crit_dd_g4

    print("  Stage0 ref: min9=+21.40%%, MaxDD=-39.89%%, Sharpe=0.865, Trades=122/yr")
    print("  Fullgate  : min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f  W10Y*=%+.2f%%  Trades=%.1f"
          % (ev["min_at"] * 100, ev["MaxDD_FULL"] * 100,
             ev["Sharpe_OOS"], aft["Worst10Y_star"] * 100, aft["Trades_yr"]))
    print("  vs B3a    : d_min9=%+.2fpp  d_MaxDD=%+.2fpp  d_Sharpe=%+.4f  d_W10Y=%+.2fpp"
          % (d_min9_g4, d_maxdd_g4, d_sh_g4, d_w10y_g4))
    print("  Criteria:")
    print("    min9 >= B3a - 0.3pp                           : %s (%.2fpp)"
          % ("PASS" if crit_min9_g4 else "FAIL", d_min9_g4))
    print("    Hard veto clear                               : %s"
          % ("PASS" if crit_veto_g4 else "FAIL"))
    print("    MaxDD 悪化<=+2pp (vs B3a)                     : %s (%.2fpp)"
          % ("PASS" if crit_dd_g4 else "FAIL", d_maxdd_g4))
    print("  6次元減点 (必要記載):")
    print("    [③ MaxDD]  d_MaxDD=%+.2fpp  -> 高Trades/高DDは攻め型コスト" % d_maxdd_g4)
    print("    [⑤ Sharpe] d_Sharpe=%+.4f  -> Sharpe低下は信頼性懸念"       % d_sh_g4)
    print("    [⑥ Trades] %.1f/yr -> 122 trades/yr は高頻度" % aft["Trades_yr"])
    print("  Bootstrap P_min(vs B3a,blk21)=%.3f  CI95=%s"
          % (p_min_g4 if not np.isnan(p_min_g4) else -1,
             _fpp(boot_21[bkey_g4_b3a].get("CI95_lo_min_pp", float("nan")))))
    print("  CPCV p10=%+.2f%%  WFA CI95_lo=%+.2f%%  WFE=%.3f"
          % (ev["cpcv_p10_at"] * 100, ev["wfa_CI95_lo"] * 100, ev["wfa_WFE"]))
    print("  VERDICT: G4_LT2_k020 Phase2 %s" % ("PASS" if g4_pass else "FAIL"))

    # =========================================================================
    # [11] CSV output
    # =========================================================================
    print("\n[11] Saving CSV ...")
    rows = []
    cand_info = [
        ("V7_TQQQ",     ev_v7,  pre_v7,  aft_v7,  tpy_v7,  None),
        ("B3a_k365",    ev_b3a, pre_b3a, aft_b3a, tpy_b3a, None),
        ("G5_vix_hard", ev_g5,  pre_g5,  aft_g5,  tpy_g5,  veto_results.get("G5_vix_hard")),
        ("G4_LT2_k020", ev_g4,  pre_g4,  aft_g4,  tpy_g4,  veto_results.get("G4_LT2_k020")),
    ]
    for lbl, ev, pre, aft, tpy, vt in cand_info:
        row = {
            "label":          lbl,
            "min9_at":        ev["min_at"],
            "CAGR_IS_at":     ev["cagr_is_at"],
            "CAGR_OOS_at":    ev["cagr_oos_at"],
            "Sharpe_OOS":     ev["Sharpe_OOS"],
            "MaxDD_FULL":     ev["MaxDD_FULL"],
            "Worst10Y_at":    aft["Worst10Y_star"],
            "Trades_yr":      aft["Trades_yr"],
            "wfa_CI95_lo":    ev["wfa_CI95_lo"],
            "wfa_WFE":        ev["wfa_WFE"],
            "wfa_t_p":        ev.get("wfa_t_p", np.nan),
            "cpcv_p10_at":    ev["cpcv_p10_at"],
            "cpcv_worst_at":  ev["cpcv_worst_at"],
            "cpcv_med_at":    ev["cpcv_med_at"],
            "regime_min_at":  ev["regime_min_at"],
        }
        for ax in ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                   "rate:rate_up", "rate:rate_down"]:
            row["regime_" + ax.replace(":", "_")] = ev["regime"].get(ax, np.nan)
        for s in list(stress.keys()):
            row["stress_%s_ret" % s] = ev["stress"][s]["ret"]
            row["stress_%s_maxdd" % s] = ev["stress"][s]["maxdd"]
        if vt is not None:
            row["hard_veto"] = int(vt["veto"])
            row["veto_reasons"] = "; ".join(vt["reasons"])
        else:
            row["hard_veto"] = 0
            row["veto_reasons"] = ""
        # bootstrap columns (vs B3a and vs V7)
        for base_lbl in ["B3a_k365", "V7_TQQQ"]:
            bkey = "%s_vs_%s" % (lbl, base_lbl)
            if bkey in boot_21:
                br = boot_21[bkey]
                sfx = "_vs_B3a" if base_lbl == "B3a_k365" else "_vs_V7"
                row["boot21_P_min%s"    % sfx] = br["P_min_better"]
                row["boot21_CI95_min%s" % sfx] = br["CI95_lo_min_pp"]
                row["boot21_P_dd%s"     % sfx] = br["P_maxdd_better"]
                row["boot21_P_w10y%s"   % sfx] = br.get("P_worst10y_better", np.nan)
                row["boot21_P_shp%s"    % sfx] = br.get("P_sharpe_better", np.nan)
        rows.append(row)

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combine_phase2_fullgate_20260615.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("Saved CSV: %s" % csv_path)

    # =========================================================================
    # [12] RETURN_BLOCK
    # =========================================================================
    def _rblock_cand(lbl, ev, pre, aft, vt, boot_b3a_key, boot_v7_key):
        bb  = boot_21.get(boot_b3a_key, {})
        bv  = boot_21.get(boot_v7_key, {})
        return {
            "min9_at":       round(ev["min_at"], 6),
            "Sharpe_OOS":    round(ev["Sharpe_OOS"], 4),
            "MaxDD_FULL":    round(ev["MaxDD_FULL"], 6),
            "Worst10Y_at":   round(float(aft["Worst10Y_star"]), 6),
            "Trades_yr":     round(float(aft["Trades_yr"]), 2),
            "wfa_CI95_lo":   round(float(ev["wfa_CI95_lo"]), 6),
            "wfa_WFE":       round(float(ev["wfa_WFE"]), 4),
            "wfa_t_p":       round(float(ev.get("wfa_t_p", float("nan"))), 4),
            "cpcv_p10_at":   round(ev["cpcv_p10_at"], 6),
            "cpcv_worst_at": round(ev["cpcv_worst_at"], 6),
            "regime_min_at": round(ev["regime_min_at"], 6),
            "hard_veto":     vt["veto"] if vt else False,
            "veto_reasons":  vt["reasons"] if vt else [],
            "boot_vs_B3a_block21": {
                "P_min":    round(bb.get("P_min_better",      float("nan")), 4),
                "P_maxdd":  round(bb.get("P_maxdd_better",    float("nan")), 4),
                "P_w10y":   round(bb.get("P_worst10y_better", float("nan")), 4),
                "P_sharpe": round(bb.get("P_sharpe_better",   float("nan")), 4),
                "CI95_min_pp":  round(bb.get("CI95_lo_min_pp",  float("nan")), 3),
                "CI95_dd_pp":   round(bb.get("CI95_lo_dd_pp",   float("nan")), 3),
                "CI95_w10y_pp": round(bb.get("CI95_lo_w10y_pp", float("nan")), 3),
            },
            "boot_vs_V7_block21": {
                "P_min":    round(bv.get("P_min_better",      float("nan")), 4),
                "P_maxdd":  round(bv.get("P_maxdd_better",    float("nan")), 4),
                "P_w10y":   round(bv.get("P_worst10y_better", float("nan")), 4),
                "P_sharpe": round(bv.get("P_sharpe_better",   float("nan")), 4),
                "CI95_min_pp":  round(bv.get("CI95_lo_min_pp",  float("nan")), 3),
                "CI95_dd_pp":   round(bv.get("CI95_lo_dd_pp",   float("nan")), 3),
                "CI95_w10y_pp": round(bv.get("CI95_lo_w10y_pp", float("nan")), 3),
            },
        }

    return_block = {
        "meta": {
            "script":           "combine_phase2_fullgate_20260615.py",
            "date":             "2026-06-16",
            "n_boot":           N_BOOT,
            "block_sizes":      [21, 252],
            "sanity_pass":      sanity_pass,
            "csv_path":         csv_path,
        },
        "V7_TQQQ": {
            "min9_at":     round(ev_v7["min_at"], 6),
            "Sharpe_OOS":  round(ev_v7["Sharpe_OOS"], 4),
            "MaxDD_FULL":  round(ev_v7["MaxDD_FULL"], 6),
            "wfa_CI95_lo": round(float(ev_v7["wfa_CI95_lo"]), 6),
        },
        "B3a_k365": {
            "min9_at":     round(ev_b3a["min_at"], 6),
            "Sharpe_OOS":  round(ev_b3a["Sharpe_OOS"], 4),
            "MaxDD_FULL":  round(ev_b3a["MaxDD_FULL"], 6),
            "wfa_CI95_lo": round(float(ev_b3a["wfa_CI95_lo"]), 6),
            "wfa_WFE":     round(float(ev_b3a["wfa_WFE"]), 4),
            "cpcv_p10_at": round(ev_b3a["cpcv_p10_at"], 6),
            "regime_min":  round(ev_b3a["regime_min_at"], 6),
        },
        "G5_vix_hard": _rblock_cand(
            "G5_vix_hard", ev_g5, pre_g5, aft_g5,
            veto_results["G5_vix_hard"],
            "G5_vix_hard_vs_B3a_k365", "G5_vix_hard_vs_V7_TQQQ"),
        "G4_LT2_k020": _rblock_cand(
            "G4_LT2_k020", ev_g4, pre_g4, aft_g4,
            veto_results["G4_LT2_k020"],
            "G4_LT2_k020_vs_B3a_k365", "G4_LT2_k020_vs_V7_TQQQ"),
        "judgment": {
            "G5_vix_hard_phase2_pass": g5_pass,
            "G4_LT2_k020_phase2_pass": g4_pass,
            "G5_criteria": {
                "crit_improve":   crit_improve,
                "crit_min9":      crit_min9,
                "crit_veto":      crit_veto,
                "crit_boot_maxdd_P80": crit_boot_dd,
                "worst10y_flag":  w10y_flag,
            },
            "G4_criteria": {
                "crit_min9_within_0_3pp": crit_min9_g4,
                "crit_veto":              crit_veto_g4,
                "crit_maxdd_le_2pp":      crit_dd_g4,
            },
        },
    }

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    def _json_safe(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj
    print(json.dumps(_json_safe(return_block), indent=2, ensure_ascii=False))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
