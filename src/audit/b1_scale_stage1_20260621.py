"""
src/audit/b1_scale_stage1_20260621.py
=====================================
B1 (downside-deviation) brake x leverage-scale Stage-1 runner.

Builds 8 "B1 at scale S" series + a scale-1.0 sanity series (9 total) = the B1
IN-leg downside-deviation brake applied to the SCALED P09_C1 strategy, across:
  boost map  x lev_scale = {default, strong} x {1.4, 1.6, 1.8, 2.0}
plus B1_DEF_S1.0 (default map, scale 1.0) as a sanity anchor.

THE CORRECT COMPOSITION (verified)
----------------------------------
"B1 at scale S" =
  (1) scale the IN-leg leverage via _build_tqqq_base_param(..., lev_scale=S),
  (2) build the FULL P09_C1 strategy via _build_out_fill_variant(alloc_fn=alloc_base),
  (3) apply apply_downside_dev_brake(target_dvol=0.20, window=63, max_frac_cash=0.5)
      to the SCALED strategy return.
The brake's downside-dev is computed on the SCALED return, so it fires harder at
higher scale (intended -- it adapts to the higher vol). excess_extra = 0.0025
(K365 centre) to MATCH the prior P09 scale runs.

Each series runs the standard 10-metric battery + the full Stage-1 gate (WFA 49w
+ CPCV + regime-stratified CAGR + stress windows + multi-metric block bootstrap
vs V7_TQQQ and vs B3a_k365).

Hard veto (Stage-1), all 4 axes:
  MaxDD < -50% / WFE > 1.5 / Worst10Y* < 0 / Regime_min(bear) < -10%

SANITY GATES (assert; STOP if fail):
  1. A0 == canonical P09_C1 to 1e-6 (direct _build_full_c1, v7_map=None,
     lev_scale=1.0, excess_extra=0.0).
  2. B1_DEF_S1.0 REPRODUCE: build B1 at scale 1.0 with excess_extra=0.0 (NOT
     0.0025), default map, and assert MaxDD_FULL within 0.003 of the prior B1
     (audit_results/a7dd_stage1_20260621.csv, B1_DOWNSIDE_DEV, -0.286842).

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
Output: audit_results/b1_scale_stage1_20260621.csv
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
from src.audit.out_fill_variants_20260620 import _build_out_fill_variant, alloc_base

# B1 downside-deviation brake (Task 6)
from src.audit.dd_reduction_overlays_20260621 import apply_downside_dev_brake

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

# Boost maps + excess for the scaled-B1 series
DEFAULT_MAP = None                          # -> default {1.20,1.10,1.00,1.00}
STRONG_MAP  = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXC         = EXCESS_EXTRA_K365_CENTRE      # 0.0025 -- MATCH prior P09 scale runs

# B1 brake params
B1_TARGET_DVOL   = 0.20
B1_WINDOW        = 63
B1_MAX_FRAC_CASH = 0.5

# Hard veto thresholds
HARD_VETO_MAXDD  = -0.50
HARD_VETO_WFE    = 1.5
HARD_VETO_W10Y   = 0.0
HARD_VETO_REGIME = -0.10

# Canonical P09_C1 after-tax reference (CURRENT_BEST v8 / leverup_b1c1_20260612)
P09C1_CANON = {
    "CAGR_IS_at":  0.198838,
    "CAGR_OOS_at": 0.177672,
    "MaxDD":      -0.349879,
}
SANITY_TOL_TIGHT = 1e-6        # A0 vs direct _build_full_c1 P09_C1

# Prior-project B1 (audit_results/a7dd_stage1_20260621.csv, B1_DOWNSIDE_DEV)
PRIOR_B1_MAXDD = -0.286842
B1_MAXDD_TOL   = 0.003         # +-0.3pp reproduction tolerance


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def main():
    print("=" * 120)
    print("B1 DOWNSIDE-DEV BRAKE x LEVERAGE-SCALE STAGE-1  2026-06-21")
    print("B1 brake applied to SCALED P09_C1: {default,strong} map x scale {1.4,1.6,1.8,2.0}")
    print("excess_extra=0.0025 (K365 centre) matches prior P09 scale; brake on SCALED return")
    print("Stage-1 gates: WFA (49w) + CPCV + Regime + Bootstrap vs V7/B3a")
    print("Hard veto: MaxDD<-50%% / WFE>1.5 / Worst10Y*<0 / Regime_min(bear)<-10%%")
    print("=" * 120)

    # ---- Load shared data (verbatim from a7dd template) ----
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

    # ---- Gold/Bond auxiliary series (verbatim from a7dd template) ----
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
    # Scaled-B1 builder
    # =========================================================================
    def build_b1_scaled(v7_map, lev_scale, excess=EXC):
        """Scale IN-leg leverage, build full P09_C1 strategy, then apply B1 brake.
        _build_tqqq_base_param returns (nav, r_base, tpy, excess_days)."""
        _, r_base_in_s, tpy_base_s, _ = _build_tqqq_base_param(
            shared, dates_dt, v7_map=v7_map, lev_scale=lev_scale,
            excess_extra=excess)
        nav_s, r_strat_s, eff_s = _build_out_fill_variant(
            r_base_in_s, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            alloc_fn=alloc_base)
        r_b1_s = apply_downside_dev_brake(
            r_strat_s, fund_active, sofr_arr,
            B1_TARGET_DVOL, B1_WINDOW, B1_MAX_FRAC_CASH)
        nav_b1 = pd.Series(np.cumprod(1.0 + r_b1_s), index=dates_dt)
        tpy = tpy_base_s + _count_fund_transitions(eff_s) / n_years
        return nav_b1, r_b1_s, tpy

    # =========================================================================
    # Baselines for bootstrap: V7_TQQQ and B3a_k365 (verbatim from template)
    # =========================================================================
    print("\nBuilding V7_TQQQ baseline (bootstrap baseline) ...")
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
    print("SANITY GATE 1: A0 (direct P09_C1) vs canonical book value (tol 1e-6 on CAGR_OOS)")
    print("=" * 120)
    canon_nav, canon_r, canon_tpy, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    canon_pre = compute_10metrics(canon_nav, canon_tpy)
    canon_aft = _apply_aftertax(canon_pre)
    canon_cagr_oos = canon_aft["CAGR_OOS"]

    # A0 built via the SAME plumbing as the scaled series (scale=1.0, excess=0.0):
    # _build_tqqq_base_param + _build_out_fill_variant. Must match direct build.
    _, a0_r_base, a0_tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    a0_nav, a0_r_strat, a0_eff = _build_out_fill_variant(
        a0_r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    a0_tpy = a0_tpy_base + _count_fund_transitions(a0_eff) / n_years
    a0_nav_dt = pd.Series(np.cumprod(1.0 + a0_r_strat), index=dates_dt)
    a0_pre = compute_10metrics(a0_nav_dt, a0_tpy)
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
    # SANITY GATE 2: B1_DEF_S1.0 with excess=0.0 reproduces prior B1 MaxDD
    # =========================================================================
    print("=" * 120)
    print("SANITY GATE 2: B1 (default map, scale=1.0, excess=0.0) MaxDD vs prior B1 (%.6f) within %.3f"
          % (PRIOR_B1_MAXDD, B1_MAXDD_TOL))
    print("=" * 120)
    san_nav, san_r, san_tpy = build_b1_scaled(DEFAULT_MAP, 1.0, excess=0.0)
    san_pre = compute_10metrics(san_nav, san_tpy)
    san_maxdd = san_pre["MaxDD_FULL"]
    diff_b1 = abs(san_maxdd - PRIOR_B1_MAXDD)
    print("  sanity-build MaxDD (excess0.0) : %+.6f" % san_maxdd)
    print("  prior B1 MaxDD                 : %+.6f" % PRIOR_B1_MAXDD)
    print("  |diff|                         : %.6f   (tol %.3f)" % (diff_b1, B1_MAXDD_TOL))
    if diff_b1 > B1_MAXDD_TOL:
        print("\nSANITY 2 FAILED -- B1 scale1.0 (excess0.0) does not reproduce prior B1 MaxDD.")
        print("The scaled-B1 composition does NOT reproduce prior B1. Halting.")
        sys.exit(1)
    print("  SANITY 2 PASSED (B1 scale1.0 excess0.0 == prior B1 MaxDD within 0.003).\n")

    # =========================================================================
    # SERIES: B1 brake x leverage-scale (excess=0.0025 for the OUTPUT rows)
    # =========================================================================
    SERIES = [
        ("B1_DEF_S1.0", DEFAULT_MAP, 1.0),
        ("B1_DEF_S1.4", DEFAULT_MAP, 1.4),
        ("B1_DEF_S1.6", DEFAULT_MAP, 1.6),
        ("B1_DEF_S1.8", DEFAULT_MAP, 1.8),
        ("B1_DEF_S2.0", DEFAULT_MAP, 2.0),
        ("B1_STR_S1.4", STRONG_MAP, 1.4),
        ("B1_STR_S1.6", STRONG_MAP, 1.6),
        ("B1_STR_S1.8", STRONG_MAP, 1.8),
        ("B1_STR_S2.0", STRONG_MAP, 2.0),
    ]

    print("=" * 120)
    print("STAGE 1: Full gate for %d B1xscale series (excess=%.4f)" % (len(SERIES), EXC))
    print("=" * 120)

    results = []
    for label, vmap, scale in SERIES:
        boost_map = "default" if vmap is None else "strong"
        print("\n  [%s] (map=%s, scale=%.1f) building NAV + Stage-1 gate ..."
              % (label, boost_map, scale))
        nav_dt, r, tpy = build_b1_scaled(vmap, scale)
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

        s1 = {
            "label":         label,
            "boost_map":     boost_map,
            "lev_scale":     scale,
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
        }

        shp = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        print("    CAGR_IS=%+.2f%%  CAGR_OOS=%+.2f%%  Sharpe=%.4f  MaxDD=%+.2f%%  W1D=%+.2f%%  W10Y=%+.2f%%  Trd/yr=%.1f  [VETO=%s]"
              % (s1["CAGR_IS_at"] * 100, s1["CAGR_OOS_at"] * 100, shp,
                 s1["MaxDD_FULL"] * 100,
                 (s1["Worst1D"] * 100) if s1["Worst1D"] is not None else float("nan"),
                 (s1["Worst10Y_at"] * 100) if s1["Worst10Y_at"] is not None else float("nan"),
                 s1["Trades_yr"], "YES" if s1["VETO_s1"] else "no"))
        results.append(s1)

    # =========================================================================
    # STAGE-1 TABLE
    # =========================================================================
    print("\n" + "=" * 160)
    print("B1 x SCALE STAGE-1 TABLE")
    print("%-12s | %7s | %5s | %8s | %8s | %7s | %8s | %8s | %8s | %6s | %5s | %8s | %8s | %4s"
          % ("label", "map", "scale", "CAGR_IS%", "CAGR_OOS%", "ShpFULL",
             "MaxDD%", "W1D%", "W10Y%", "Trd/yr", "WFE", "CI95_lo", "reg_min", "VETO"))
    print("-" * 160)
    for s1 in results:
        shp = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        w1d = (s1["Worst1D"] * 100) if s1["Worst1D"] is not None else float("nan")
        w10y = (s1["Worst10Y_at"] * 100) if s1["Worst10Y_at"] is not None else float("nan")
        veto_str = "VETO" if s1["VETO_s1"] else "PASS"
        print("%-12s | %7s | %5.1f | %+7.2f%% | %+7.2f%% | %7.4f | %+6.2f%% | %+6.2f%% | %+6.2f%% | %6.1f | %5.3f | %+7.4f | %+7.4f | %-4s"
              % (s1["label"][:12], s1["boost_map"], s1["lev_scale"],
                 s1["CAGR_IS_at"] * 100, s1["CAGR_OOS_at"] * 100, shp,
                 s1["MaxDD_FULL"] * 100, w1d, w10y, s1["Trades_yr"],
                 s1["wfa_WFE"], s1["wfa_CI95_lo"], s1["regime_min_at"], veto_str))

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
            "boost_map":      s1["boost_map"],
            "lev_scale":      s1["lev_scale"],
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
        })

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "b1_scale_stage1_20260621.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f",
                              encoding="utf-8-sig")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # RETURN_BLOCK
    # =========================================================================
    # output-row B1_DEF_S1.0 (excess 0.0025) for reporting alongside the sanity
    out_b1_def_s1 = next(s for s in results if s["label"] == "B1_DEF_S1.0")

    return_block = {
        "script": "b1_scale_stage1_20260621.py",
        "date": "2026-06-21",
        "composition": {
            "step1": "scale IN-leg via _build_tqqq_base_param(lev_scale=S)",
            "step2": "_build_out_fill_variant(alloc_fn=alloc_base) -> full P09_C1",
            "step3": "apply_downside_dev_brake(0.20,63,0.5) on SCALED return",
            "excess_extra_output_rows": EXC,
        },
        "tqqq_base_param_return_order": "(nav, r_base, tpy, excess_days)",
        "sanity_gate_1": {
            "A0_CAGR_OOS_at_pct":           round(a0_cagr_oos * 100, 6),
            "direct_P09C1_CAGR_OOS_at_pct": round(canon_cagr_oos * 100, 6),
            "abs_diff_A0_vs_direct":        float(diff_a0_direct),
            "PASS": bool(diff_a0_direct <= SANITY_TOL_TIGHT),
        },
        "sanity_gate_2": {
            "sanity_B1_MaxDD_excess0.0":   round(float(san_maxdd), 6),
            "prior_B1_MaxDD":              PRIOR_B1_MAXDD,
            "abs_diff":                    float(diff_b1),
            "output_row_B1_DEF_S1.0_MaxDD_excess0.0025": round(float(out_b1_def_s1["MaxDD_FULL"]), 6),
            "PASS": bool(diff_b1 <= B1_MAXDD_TOL),
        },
        "series": [
            {
                "label": s1["label"],
                "boost_map": s1["boost_map"],
                "lev_scale": s1["lev_scale"],
                "CAGR_IS_at_pct":  round(s1["CAGR_IS_at"] * 100, 4),
                "CAGR_OOS_at_pct": round(s1["CAGR_OOS_at"] * 100, 4),
                "Sharpe_FULL":     round(s1["Sharpe_FULL"], 4) if s1["Sharpe_FULL"] is not None else None,
                "MaxDD_pct":       round(s1["MaxDD_FULL"] * 100, 4),
                "Worst1D_pct":     round(s1["Worst1D"] * 100, 4) if s1["Worst1D"] is not None else None,
                "Worst10Y_pct":    round(s1["Worst10Y_at"] * 100, 4) if s1["Worst10Y_at"] is not None else None,
                "Trades_yr":       round(s1["Trades_yr"], 2),
                "wfa_WFE":         round(s1["wfa_WFE"], 4),
                "wfa_CI95_lo":     round(s1["wfa_CI95_lo"], 4),
                "regime_min_at":   round(s1["regime_min_at"], 4),
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
