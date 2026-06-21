"""
src/audit/a7dd_stage1_20260621.py
=================================
A7-based drawdown-reduction Stage-1 runner WITH timing-vs-delever decomposition.

Builds A0 (== canonical P09_C1), A7 (== prior IN-leg vol brake, reproduced) and
B1-B4 IN-leg drawdown-reduction overlays on top of the SAME P09_C1 base strategy
return. Each overlay runs the standard 10-metric battery + the full Stage-1 gate
(WFA 49w + CPCV + regime-stratified CAGR + stress windows + multi-metric block
bootstrap vs V7_TQQQ and vs B3a_k365).

THE NEW PART -- TIMING vs PLAIN DE-LEVER (the "G5 lesson")
---------------------------------------------------------
A brake's MaxDD improvement can come from two sources:
  (1) TIMING  -- it cut exposure at the RIGHT moments (skill), or
  (2) DE-LEVER -- it just held less exposure on average (any uniform cut would
      have helped equally; no timing skill).
For every overlay we build a "uniform-delever twin": apply the SAME average
IN-leg cash fraction (measured via measure_mean_in_leg_frac) uniformly to every
IN day (build_uniform_delever) = identical average exposure cut but BLIND to
when. We then block-bootstrap the overlay vs its OWN twin on MaxDD. If the
overlay's MaxDD is robustly better than its twin (timing_P_maxdd >= 0.90) the
improvement is a TIMING_EFFECT; otherwise it is DELEVER_ONLY (placebo timing,
the G5 retraction pattern).

P09_C1 base wiring (IN-leg of A0): default V7_MAP (v7_map=None), lev_scale=1.0,
excess_extra=0.0 (== canonical cfd_excess=False). Asserted to 1e-6 vs a direct
_build_full_c1 build. A7_REPRODUCE MaxDD asserted within 0.002 of the prior
project's A7 (audit_results/p09c1_alloc_stage1_20260620.csv, -0.282462).

Hard veto (Stage-1), all 4 axes:
  MaxDD < -50% / WFE > 1.5 / Worst10Y* < 0 / Regime_min(bear) < -10%

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
Output: audit_results/a7dd_stage1_20260621.csv
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

import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.regime_labeler_20260611 import build_regime_labels, stress_masks

# C1 cost-model builders
from src.audit.k365_recost_20260612 import (
    _build_full_c1, _build_tqqq_base_param,
    EXCESS_EXTRA_K365_CENTRE,
)

# Full Stage-1 gate
from src.audit.extended_eval_20260611 import _eval_one

# Multi-metric bootstrap
from src.audit.multimetric_bootstrap_20260615 import _block_bootstrap_multimetric

# Cost/NAV helpers + WFA/bootstrap constants
from src.audit.run_p09_tqqq_validate_20260611 import _run_wfa, N_BOOT, BLOCK, SEED
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _count_fund_transitions,
    LAG_DAYS, TRADING_DAYS,
)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base

# OUT-fill variant machinery (Task 8)
from src.audit.out_fill_variants_20260620 import (
    _build_out_fill_variant, alloc_base, apply_in_leg_vol_brake)

# DD-reduction overlays + timing-vs-delever machinery (Task 6)
from src.audit.dd_reduction_overlays_20260621 import (
    apply_downside_dev_brake, apply_dd_throttle, apply_asym_vol_brake,
    apply_param_vol_brake, measure_mean_in_leg_frac, build_uniform_delever)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Canonical P09_C1 wiring: default V7_MAP (v7_map=None), lev_scale=1.0, and
# NO >3x excess charge. excess_extra=0.0 reproduces cfd_excess=False exactly.
P09C1_V7_MAP       = None
P09C1_LEV_SCALE    = 1.0
P09C1_EXCESS_EXTRA = 0.0

# B3a_k365 reference (default map, scale 1.15, k365 centre excess) for bootstrap
B3A_MAP_DEFAULT    = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE      = 1.15
B3A_EXCESS_EXTRA   = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# Hard veto thresholds
HARD_VETO_MAXDD  = -0.50
HARD_VETO_WFE    = 1.5
HARD_VETO_W10Y   = 0.0
HARD_VETO_REGIME = -0.10

# Timing gate: overlay must beat its uniform-delever twin on MaxDD at >=0.90
TIMING_P_THR = 0.90

# Canonical P09_C1 after-tax reference (CURRENT_BEST v8 / leverup_b1c1_20260612)
P09C1_CANON = {
    "CAGR_IS_at":  0.198838,
    "CAGR_OOS_at": 0.177672,
    "MaxDD":      -0.349879,
}
SANITY_TOL_TIGHT = 1e-6        # A0 vs direct _build_full_c1 P09_C1

# Prior-project A7 (audit_results/p09c1_alloc_stage1_20260620.csv, A7_IN_VOL_BRAKE)
PRIOR_A7_MAXDD   = -0.282462
A7_MAXDD_TOL     = 0.002       # +-0.2pp reproduction tolerance


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def main():
    print("=" * 120)
    print("A7-BASED DD-REDUCTION STAGE-1 + TIMING-vs-DELEVER DECOMPOSITION  2026-06-21")
    print("A0==P09_C1 / A7==prior IN-leg vol brake / B1-B4 DD overlays")
    print("Each overlay bootstrapped vs its UNIFORM-DELEVER twin on MaxDD (G5 lesson)")
    print("Stage-1 gates: WFA (49w) + CPCV + Regime + Bootstrap vs V7/B3a")
    print("Hard veto: MaxDD<-50%% / WFE>1.5 / Worst10Y*<0 / Regime_min(bear)<-10%%")
    print("=" * 120)

    # ---- Load shared data (verbatim from template) ----
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    # ---- Gold/Bond auxiliary series (verbatim from template) ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252_raw = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252_raw), False, bond_m252_raw > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    # =========================================================================
    # P09_C1 IN-leg base return (== A0 IN leg; canonical wiring)
    # =========================================================================
    print("\nBuilding P09_C1 IN-leg base (V7_MAP default, scale=1.0, excess=0.0) ...")
    _, r_base_in, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)

    # Build the base P09_C1 daily return ONCE (== A0). r_strat is unbraked.
    nav0, r_strat, eff0 = _build_out_fill_variant(
        r_base_in, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    tpy0 = tpy_base + _count_fund_transitions(eff0) / n_years

    def braked_triple(r_braked):
        nav = np.cumprod(1.0 + r_braked)
        return pd.Series(nav, index=dates_dt), r_braked, tpy0

    # =========================================================================
    # Baselines for bootstrap: V7_TQQQ and B3a_k365 (verbatim from template)
    # =========================================================================
    print("Building V7_TQQQ baseline (bootstrap baseline) ...")
    _, r_v7, _, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    print("Building B3a_k365 reference (default map, scale=1.15, k365 excess) ...")
    _, r_b3a, _, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3A_MAP_DEFAULT, lev_scale=B3A_LEV_SCALE,
        excess_extra=B3A_EXCESS_EXTRA)

    # =========================================================================
    # Regime labels and stress masks (verbatim from template)
    # =========================================================================
    print("Building regime labels and stress masks ...")
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress  = stress_masks(dates_dt)

    # =========================================================================
    # SANITY GATE 1: A0 must reproduce canonical P09_C1 (direct _build_full_c1)
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE 1: A0 (base r_strat) vs direct _build_full_c1 P09_C1 (tol 1e-6 on CAGR_OOS)")
    print("=" * 120)
    canon_nav, canon_r, canon_tpy, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    canon_pre = compute_10metrics(canon_nav, canon_tpy)
    canon_aft = _apply_aftertax(canon_pre)
    canon_cagr_oos = canon_aft["CAGR_OOS"]

    a0_nav, a0_r, a0_tpy = braked_triple(r_strat)
    a0_pre = compute_10metrics(a0_nav, a0_tpy)
    a0_aft = _apply_aftertax(a0_pre)
    a0_cagr_oos = a0_aft["CAGR_OOS"]

    diff_a0_direct   = abs(a0_cagr_oos - canon_cagr_oos)
    diff_direct_book = abs(canon_cagr_oos - P09C1_CANON["CAGR_OOS_at"])
    print("  A0 CAGR_OOS(at)            : %+.8f%%" % (a0_cagr_oos * 100))
    print("  direct _build_full_c1 OOS : %+.8f%%" % (canon_cagr_oos * 100))
    print("  CURRENT_BEST P09_C1 OOS   : %+.8f%%" % (P09C1_CANON["CAGR_OOS_at"] * 100))
    print("  |A0 - direct|             : %.3e   (tol %.0e)" % (diff_a0_direct, SANITY_TOL_TIGHT))
    print("  |direct - CURRENT_BEST|   : %.3e   (book value, ~rounding)" % diff_direct_book)
    print("  A0 MaxDD=%+.4f%%  direct MaxDD=%+.4f%%  book MaxDD=%+.4f%%"
          % (a0_pre["MaxDD_FULL"] * 100, canon_pre["MaxDD_FULL"] * 100,
             P09C1_CANON["MaxDD"] * 100))
    if diff_a0_direct > SANITY_TOL_TIGHT:
        print("\nSANITY 1 FAILED -- A0 does not match direct _build_full_c1 P09_C1 to 1e-6.")
        print("The base wiring is wrong. Halting.")
        sys.exit(1)
    if diff_direct_book > 0.0005:
        print("\n  WARN: direct P09_C1 CAGR_OOS differs from CURRENT_BEST by >0.05pp.")
    print("  SANITY 1 PASSED (A0 == direct P09_C1 to 1e-6).\n")

    # =========================================================================
    # OVERLAYS (braked returns on top of r_strat)
    # =========================================================================
    OVERLAYS = [
        ("A0_P09_C1_BASE",  r_strat),
        ("A7_REPRODUCE",    apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.5)),
        ("B1_DOWNSIDE_DEV", apply_downside_dev_brake(r_strat, fund_active, sofr_arr, 0.20, 63, 0.5)),
        ("B2_DD_THROTTLE",  apply_dd_throttle(r_strat, fund_active, sofr_arr,
                                tiers=((0.15, 0.25), (0.25, 0.50)))),
        ("B3_ASYM_BRAKE",   apply_asym_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.5, 5)),
        ("B4_VOL020_CAP50", apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.20, 63, 0.5)),
        ("B4_VOL025_CAP50", apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.25, 63, 0.5)),
        ("B4_VOL030_CAP75", apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.75)),
    ]

    # =========================================================================
    # SANITY GATE 2: A7_REPRODUCE MaxDD == prior A7 (-0.282462) within 0.002
    # =========================================================================
    print("=" * 120)
    print("SANITY GATE 2: A7_REPRODUCE MaxDD_FULL vs prior A7 (%.6f) within %.3f"
          % (PRIOR_A7_MAXDD, A7_MAXDD_TOL))
    print("=" * 120)
    a7_r = dict(OVERLAYS)["A7_REPRODUCE"]
    a7_nav = pd.Series(np.cumprod(1.0 + a7_r), index=dates_dt)
    a7_pre = compute_10metrics(a7_nav, tpy0)
    a7_maxdd = a7_pre["MaxDD_FULL"]
    diff_a7 = abs(a7_maxdd - PRIOR_A7_MAXDD)
    print("  A7_REPRODUCE MaxDD_FULL : %+.6f" % a7_maxdd)
    print("  prior A7 MaxDD          : %+.6f" % PRIOR_A7_MAXDD)
    print("  |diff|                  : %.6f   (tol %.3f)" % (diff_a7, A7_MAXDD_TOL))
    if diff_a7 > A7_MAXDD_TOL:
        print("\nSANITY 2 FAILED -- A7_REPRODUCE does not reproduce prior A7 MaxDD.")
        print("The new brake module does NOT reproduce A7. Halting.")
        sys.exit(1)
    print("  SANITY 2 PASSED (A7_REPRODUCE == prior A7 MaxDD within 0.002).\n")

    # =========================================================================
    # STAGE 1: Full gate + timing-vs-delever twin for each overlay
    # =========================================================================
    print("=" * 120)
    print("STAGE 1: Full gate + uniform-delever twin for %d overlays" % len(OVERLAYS))
    print("=" * 120)

    results = []
    for label, r_b in OVERLAYS:
        print("\n  [%s] building NAV + Stage-1 gate ..." % label)
        nav_dt, r, tpy = braked_triple(r_b)
        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)

        ev = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                       baseline_r=r_v7)
        wfe     = float(ev["wfa_WFE"])
        reg_min = float(ev["regime_min_at"])
        w10y    = aft["Worst10Y_star"]
        maxdd   = pre["MaxDD_FULL"]

        print("      bootstrap vs V7 ...")
        boot_v7  = _block_bootstrap_multimetric(r, r_v7, is_mask, oos_mask,
                                                n_boot=N_BOOT, block=BLOCK, seed=SEED)
        print("      bootstrap vs B3a ...")
        boot_b3a = _block_bootstrap_multimetric(r, r_b3a, is_mask, oos_mask,
                                                n_boot=N_BOOT, block=BLOCK, seed=SEED)

        v_maxdd = maxdd < HARD_VETO_MAXDD
        v_wfe   = wfe   > HARD_VETO_WFE
        v_w10y  = w10y  < HARD_VETO_W10Y
        v_reg   = reg_min < HARD_VETO_REGIME
        veto_s1 = v_maxdd or v_wfe or v_w10y or v_reg

        # --- TIMING vs PLAIN DE-LEVER (the G5 lesson) ---
        if label == "A0_P09_C1_BASE":
            fbar = float("nan")
            uni_maxdd = float("nan")
            uni_cagr_oos = float("nan")
            timing_p_maxdd = float("nan")
            timing_ci95_pp = float("nan")
            timing_verdict = ""
        else:
            fbar = measure_mean_in_leg_frac(r_strat, r_b, fund_active, sofr_arr)
            r_uni = build_uniform_delever(r_strat, fund_active, sofr_arr, fbar)
            nav_uni = pd.Series(np.cumprod(1.0 + r_uni), index=dates_dt)
            pre_uni = compute_10metrics(nav_uni, tpy0)
            aft_uni = _apply_aftertax(pre_uni)
            uni_maxdd = pre_uni["MaxDD_FULL"]
            uni_cagr_oos = aft_uni["CAGR_OOS"]
            print("      bootstrap vs uniform-delever twin (timing) ...")
            boot_vs_uni = _block_bootstrap_multimetric(
                r_b, r_uni, is_mask, oos_mask, n_boot=N_BOOT, block=BLOCK, seed=SEED)
            timing_p_maxdd = float(boot_vs_uni["P_maxdd_better"])
            timing_ci95_pp = float(boot_vs_uni["CI95_lo_dd_pp"])
            timing_verdict = ("TIMING_EFFECT" if timing_p_maxdd >= TIMING_P_THR
                              else "DELEVER_ONLY")

        s1 = {
            "label":         label,
            "CAGR_IS_at":    aft["CAGR_IS"],
            "CAGR_OOS_at":   aft["CAGR_OOS"],
            "min9_at":       _min_at(aft),
            "IS_OOS_gap_pp": aft["IS_OOS_gap_pp"],
            "Sharpe_FULL":   pre["Sharpe_FULL"],
            "Sharpe_OOS":    pre["Sharpe_OOS"],
            "MaxDD_FULL":    maxdd,
            "Worst1D":       pre["Worst1D"],
            "Worst1D_date":  pre["Worst1D_date"],
            "Worst10Y_at":   w10y,
            "Worst5Y_at":    aft["Worst5Y"],
            "P10_5Y_at":     aft["P10_5Y"],
            "Trades_yr":     aft["Trades_yr"],
            "wfa_WFE":       wfe,
            "wfa_CI95_lo":   float(ev["wfa_CI95_lo"]),
            "wfa_t_p":       float(ev["wfa_t_p"]),
            "cpcv_p10_at":   float(ev["cpcv_p10_at"]),
            "regime_min_at": reg_min,
            "regime":        ev["regime"],
            "mm_v7_P_min":     boot_v7["P_min_better"],
            "mm_v7_P_maxdd":   boot_v7["P_maxdd_better"],
            "mm_b3a_P_min":    boot_b3a["P_min_better"],
            "mm_b3a_P_maxdd":  boot_b3a["P_maxdd_better"],
            "s1_veto_maxdd": int(v_maxdd),
            "s1_veto_wfe":   int(v_wfe),
            "s1_veto_w10y":  int(v_w10y),
            "s1_veto_reg":   int(v_reg),
            "VETO_s1":       int(veto_s1),
            # --- timing-vs-delever ---
            "fbar":                 fbar,
            "uni_MaxDD":            uni_maxdd,
            "uni_CAGR_OOS":         uni_cagr_oos,
            "timing_P_maxdd":       timing_p_maxdd,
            "timing_CI95_maxdd_pp": timing_ci95_pp,
            "timing_verdict":       timing_verdict,
        }

        shp = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        print("    CAGR_OOS=%+.2f%%  Sharpe=%.4f  MaxDD=%+.2f%%  W1D=%+.2f%%  Trd/yr=%.1f  [VETO=%s]"
              % (s1["CAGR_OOS_at"] * 100, shp, s1["MaxDD_FULL"] * 100,
                 (s1["Worst1D"] * 100) if s1["Worst1D"] is not None else float("nan"),
                 s1["Trades_yr"], "YES" if s1["VETO_s1"] else "no"))
        if label != "A0_P09_C1_BASE":
            print("    fbar=%.4f  uni_MaxDD=%+.2f%%  timing_P_maxdd=%.3f  CI95_dd=%+.2fpp  -> %s"
                  % (fbar, uni_maxdd * 100, timing_p_maxdd, timing_ci95_pp, timing_verdict))

        results.append(s1)

    # =========================================================================
    # STAGE-1 + TIMING TABLE
    # =========================================================================
    print("\n" + "=" * 160)
    print("STAGE 1 + TIMING-vs-DELEVER TABLE")
    print("%-16s | %9s | %8s | %8s | %8s | %7s | %8s | %8s | %-13s | %4s"
          % ("label", "CAGR_OOS%", "MaxDD%", "Worst1D%", "ShpFULL", "fbar",
             "uniMaxDD%", "timP_dd", "timing_verdict", "VETO"))
    print("-" * 160)
    for s1 in results:
        shp = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        w1d = (s1["Worst1D"] * 100) if s1["Worst1D"] is not None else float("nan")
        fbar = s1["fbar"]
        uni = s1["uni_MaxDD"]
        tpm = s1["timing_P_maxdd"]
        veto_str = "VETO" if s1["VETO_s1"] else "PASS"
        print("%-16s | %+8.2f%% | %+6.2f%% | %+6.2f%% | %7.4f | %6s | %8s | %7s | %-13s | %-4s"
              % (s1["label"][:16], s1["CAGR_OOS_at"] * 100, s1["MaxDD_FULL"] * 100,
                 w1d, shp,
                 ("%.4f" % fbar) if np.isfinite(fbar) else "  -  ",
                 ("%+.2f%%" % (uni * 100)) if np.isfinite(uni) else "   -   ",
                 ("%.3f" % tpm) if np.isfinite(tpm) else "  -  ",
                 s1["timing_verdict"] if s1["timing_verdict"] else "(base)",
                 veto_str))

    # =========================================================================
    # CSV OUTPUT
    # =========================================================================
    print("\nBuilding CSV ...")
    rows = []
    for s1 in results:
        shp_full = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        def _r(v, nd=6):
            return round(float(v), nd) if v is not None and np.isfinite(v) else float("nan")
        rows.append({
            "label":          s1["label"],
            "CAGR_IS_at":     _r(s1["CAGR_IS_at"]),
            "CAGR_OOS_at":    _r(s1["CAGR_OOS_at"]),
            "min9_at":        _r(s1["min9_at"]),
            "IS_OOS_gap_pp":  _r(s1["IS_OOS_gap_pp"], 4),
            "Sharpe_FULL":    _r(shp_full, 4),
            "Sharpe_OOS":     _r(s1["Sharpe_OOS"], 4),
            "MaxDD_FULL":     _r(s1["MaxDD_FULL"]),
            "Worst1D":        _r(s1["Worst1D"]),
            "Worst1D_date":   s1["Worst1D_date"] if s1["Worst1D_date"] else "",
            "Worst10Y_at":    _r(s1["Worst10Y_at"]),
            "Worst5Y_at":     _r(s1["Worst5Y_at"]),
            "P10_5Y_at":      _r(s1["P10_5Y_at"]),
            "Trades_yr":      _r(s1["Trades_yr"], 2),
            "wfa_WFE":        _r(s1["wfa_WFE"]),
            "wfa_CI95_lo":    _r(s1["wfa_CI95_lo"]),
            "wfa_t_p":        _r(s1["wfa_t_p"], 10),
            "cpcv_p10_at":    _r(s1["cpcv_p10_at"]),
            "regime_min_at":  _r(s1["regime_min_at"]),
            "mm_v7_P_min":    _r(s1["mm_v7_P_min"], 4),
            "mm_v7_P_maxdd":  _r(s1["mm_v7_P_maxdd"], 4),
            "mm_b3a_P_min":   _r(s1["mm_b3a_P_min"], 4),
            "mm_b3a_P_maxdd": _r(s1["mm_b3a_P_maxdd"], 4),
            "s1_veto_maxdd":  s1["s1_veto_maxdd"],
            "s1_veto_wfe":    s1["s1_veto_wfe"],
            "s1_veto_w10y":   s1["s1_veto_w10y"],
            "s1_veto_reg":    s1["s1_veto_reg"],
            "VETO_s1":        s1["VETO_s1"],
            # --- timing-vs-delever ---
            "fbar":                 _r(s1["fbar"]),
            "uni_MaxDD":            _r(s1["uni_MaxDD"]),
            "uni_CAGR_OOS":         _r(s1["uni_CAGR_OOS"]),
            "timing_P_maxdd":       _r(s1["timing_P_maxdd"], 4),
            "timing_CI95_maxdd_pp": _r(s1["timing_CI95_maxdd_pp"], 4),
            "timing_verdict":       s1["timing_verdict"],
        })

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "a7dd_stage1_20260621.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f",
                              encoding="utf-8-sig")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # RETURN_BLOCK
    # =========================================================================
    return_block = {
        "script": "a7dd_stage1_20260621.py",
        "date": "2026-06-21",
        "p09c1_wiring": {
            "v7_map": "default V7_MAP {0:1.20,1:1.10,2:1.00,3:1.00}",
            "lev_scale": P09C1_LEV_SCALE,
            "excess_extra": P09C1_EXCESS_EXTRA,
        },
        "sanity_gate_1": {
            "A0_CAGR_OOS_at_pct":           round(a0_cagr_oos * 100, 6),
            "direct_P09C1_CAGR_OOS_at_pct": round(canon_cagr_oos * 100, 6),
            "abs_diff_A0_vs_direct":        float(diff_a0_direct),
            "PASS": bool(diff_a0_direct <= SANITY_TOL_TIGHT),
        },
        "sanity_gate_2": {
            "A7_REPRODUCE_MaxDD": round(float(a7_maxdd), 6),
            "prior_A7_MaxDD":     PRIOR_A7_MAXDD,
            "abs_diff":           float(diff_a7),
            "PASS": bool(diff_a7 <= A7_MAXDD_TOL),
        },
        "bootstrap_maxdd_keys": {
            "P_value_key": "P_maxdd_better",
            "CI95_key":    "CI95_lo_dd_pp",
        },
        "overlays": [
            {
                "label": s1["label"],
                "CAGR_OOS_at_pct": round(s1["CAGR_OOS_at"] * 100, 4),
                "MaxDD_pct":       round(s1["MaxDD_FULL"] * 100, 4),
                "Worst1D_pct":     round(s1["Worst1D"] * 100, 4) if s1["Worst1D"] is not None else None,
                "Sharpe_FULL":     round(s1["Sharpe_FULL"], 4) if s1["Sharpe_FULL"] is not None else None,
                "fbar":            round(s1["fbar"], 4) if np.isfinite(s1["fbar"]) else None,
                "uni_MaxDD_pct":   round(s1["uni_MaxDD"] * 100, 4) if np.isfinite(s1["uni_MaxDD"]) else None,
                "timing_P_maxdd":  round(s1["timing_P_maxdd"], 4) if np.isfinite(s1["timing_P_maxdd"]) else None,
                "timing_verdict":  s1["timing_verdict"],
                "VETO_s1":         s1["VETO_s1"],
            }
            for s1 in results
        ],
        "csv": csv_path,
        "n_boot": N_BOOT, "block": BLOCK, "seed": SEED,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))

    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
