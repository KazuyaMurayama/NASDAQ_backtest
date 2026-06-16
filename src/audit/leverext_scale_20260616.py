"""
src/audit/leverext_scale_20260616.py
=====================================
Task #1: B3a uniform leverage extension sweep -- scale 1.20 to 1.35.

Purpose (LEVERUP_EXTENSION_PLAN_20260616.md §1):
  B3a_k365 is the current best strategy (min9+20.98%, MaxDD-38.20%, scale=1.15).
  This script explores scale {1.20, 1.25, 1.30, 1.35} x v7_map {default, strong_boost}
  = 8 configs, searching for the CAGR-DD frontier above scale=1.15.
  Hard veto on MaxDD < -50% / Worst10Y* < 0 (WFE/Regime veto applied in Stage 1 full gate).

Grid (pre-registered, per plan §1):
  uniform_scale in {1.20, 1.25, 1.30, 1.35}
  v7_map (a) B3a default: {0:1.40, 1:1.40, 2:1.05, 3:1.00}
         (b) strong boost: {0:1.60, 1:1.50, 2:1.10, 3:1.00}
  Fixed: C1 SOFR fill on bond-OFF days / k365 EXCESS_EXTRA=0.0025 / P09 OUT fill

Sanity gate:
  scale=1.15, default map must reproduce B3a baseline:
    min9 +20.98% +/-0.05pp, MaxDD -38.20% +/-0.10pp
  Halt if violated.

Stage 0 (all 8 configs + B3a/B3c references):
  Standard 10 metrics + >3x day ratio + worst calendar year + hard veto flags.

Stage 1 (vet-free and min9 > B3a among top candidates):
  _eval_one (WFA canonical 49 windows + CPCV + regime + stress + bootstrap vs V7)
  + multimetric bootstrap (4-axis: min/MaxDD/Worst10Y*/Sharpe) vs B3a and vs V7.

Selection bias note (pre-registered):
  Response surface monotonicity is verified and commented. Point estimates for min9
  at higher scale have increasing selection bias because the grid was designed
  specifically to find higher CAGR; treat CI95_lo and bootstrap P_better as
  primary evidence, not the point estimates.

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.

Outputs:
  audit_results/leverext_scale_20260616.csv
  RETURN_BLOCK printed to stdout (json.dumps)
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

# ---- Reuse builders from k365_recost (B3a cost model) ----------------------
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,  # = 0.0025 (k365 spread 0.75pp - SOFR 0.5pp)
    EXCESS_EXTRA_STORE,
)

# ---- Reuse extended_eval for full gate (_eval_one) --------------------------
from src.audit.extended_eval_20260611 import (
    _eval_one, _cpcv_dist, _regime_cagr,
)

# ---- Reuse bootstrap functions ----------------------------------------------
from src.audit.multimetric_bootstrap_20260615 import (
    _block_bootstrap_multimetric,
)

# ---- Reuse cost/NAV helpers -------------------------------------------------
from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base,
    AFTER_TAX, SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)

# ---- Misc helpers -----------------------------------------------------------
from src.audit.run_p09_tqqq_validate_20260611 import (
    _run_wfa, _block_bootstrap_compare, LU2_SCALE, _cagr_seg, _maxdd_from_returns,
    N_BOOT, BLOCK, SEED,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _count_fund_transitions,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
    FEE_GOLD, FEE_BOND,
)

# ---------------------------------------------------------------------------
# Sweep constants
# ---------------------------------------------------------------------------
B3A_MAP_DEFAULT = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}  # B3a / B3c default
B3A_MAP_STRONG  = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}  # strong boost variant

EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025 (k365 centre)

# Reference: B3a known point (QC independent reimpl, 2026-06-15)
B3A_KNOWN_MIN9   = 0.2098   # +20.98%
B3A_KNOWN_MAXDD  = -0.3820  # -38.20%
B3A_SANITY_TOL_MIN9  = 0.0005   # 0.05pp
B3A_SANITY_TOL_MAXDD = 0.0010   # 0.10pp

HARD_VETO_MAXDD  = -0.50
HARD_VETO_W10Y   = 0.0
HARD_VETO_WFE    = 1.5
HARD_VETO_REGIME = -0.10

# Pre-registered sweep grid
SWEEP_SCALES = [1.20, 1.25, 1.30, 1.35]
REF_SCALES   = [1.15, 1.10]   # B3a + B3c reference

CONFIGS = []
# (a) default map at sweep scales
for sc in SWEEP_SCALES:
    CONFIGS.append({
        "label": "Bext_def_sc%.2f" % sc,
        "v7_map": B3A_MAP_DEFAULT,
        "lev_scale": sc,
        "map_tag": "default",
    })
# (b) strong boost at sweep scales
for sc in SWEEP_SCALES:
    CONFIGS.append({
        "label": "Bext_str_sc%.2f" % sc,
        "v7_map": B3A_MAP_STRONG,
        "lev_scale": sc,
        "map_tag": "strong",
    })
# References: B3a (sc=1.15 default), B3c (sc=1.10 default)
REF_CONFIGS = [
    {"label": "B3a_k365_ref", "v7_map": B3A_MAP_DEFAULT,
     "lev_scale": 1.15, "map_tag": "default"},
    {"label": "B3c_k365_ref", "v7_map": B3A_MAP_DEFAULT,
     "lev_scale": 1.10, "map_tag": "default"},
]


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _stage0_row(label, nav_dt, tpy, exc, n_total, map_tag, lev_scale):
    """Compute standard-10 metrics + veto flags for one config (Stage 0)."""
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    cy = _calendar_year_returns(nav_dt)
    mn = _min_at(aft)

    # >3x day ratio
    excess_ratio = float(exc) / float(n_total) if n_total > 0 else 0.0

    # Veto flags (Stage 0: MaxDD + W10Y only; WFE/Regime need full gate)
    v_maxdd = pre["MaxDD_FULL"] < HARD_VETO_MAXDD
    v_w10y  = aft["Worst10Y_star"] < HARD_VETO_W10Y
    veto    = v_maxdd or v_w10y

    return {
        "label":           label,
        "map_tag":         map_tag,
        "lev_scale":       lev_scale,
        # Standard 10 (after-tax where applicable)
        "CAGR_IS_at":      round(aft["CAGR_IS"], 6),
        "CAGR_OOS_at":     round(aft["CAGR_OOS"], 6),
        "min9_at":         round(mn, 6),
        "IS_OOS_gap_pp":   round(aft["IS_OOS_gap_pp"], 4),
        "Sharpe_OOS":      round(pre["Sharpe_OOS"], 4),
        "MaxDD_FULL":      round(pre["MaxDD_FULL"], 6),
        "Worst10Y_star_at":round(aft["Worst10Y_star"], 6),
        "P10_5Y_at":       round(aft["P10_5Y"], 6),
        "Worst5Y_at":      round(aft["Worst5Y"], 6),
        "Trades_yr":       round(aft["Trades_yr"], 2),
        # Extra columns
        "excess_days":     int(exc),
        "excess_ratio_pct": round(excess_ratio * 100, 2),
        "worst_cy":        round(float(cy.min()), 6),
        "worst_cy_year":   int(cy.idxmin()),
        # Veto
        "veto_maxdd":      int(v_maxdd),
        "veto_w10y":       int(v_w10y),
        "VETO_s0":         int(veto),
    }


def _stage1_full_gate(label, nav_dt, r, tpy, regimes, stress,
                      is_mask, oos_mask, r_v7, r_b3a):
    """
    Full gate: WFA (canonical 49 windows) + CPCV + regime + stress via _eval_one,
    then multi-metric bootstrap vs V7 and vs B3a.
    Returns dict with all fields.
    """
    ev = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                   baseline_r=r_v7)
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)

    wfe = float(ev["wfa_WFE"])
    reg_min = float(ev["regime_min_at"])
    w10y = aft["Worst10Y_star"]
    maxdd = pre["MaxDD_FULL"]

    v_maxdd  = maxdd < HARD_VETO_MAXDD
    v_wfe    = wfe   > HARD_VETO_WFE
    v_w10y   = w10y  < HARD_VETO_W10Y
    v_reg    = reg_min < HARD_VETO_REGIME
    veto_s1  = v_maxdd or v_wfe or v_w10y or v_reg

    # Multi-metric bootstrap vs V7 (baseline)
    print("      bootstrap vs V7 ...")
    boot_v7 = _block_bootstrap_multimetric(r, r_v7, is_mask, oos_mask,
                                            n_boot=N_BOOT, block=BLOCK, seed=SEED)
    # Multi-metric bootstrap vs B3a
    print("      bootstrap vs B3a ...")
    boot_b3a = _block_bootstrap_multimetric(r, r_b3a, is_mask, oos_mask,
                                             n_boot=N_BOOT, block=BLOCK, seed=SEED)

    boot_ev = ev.get("boot") or {}

    return {
        # WFA
        "wfa_WFE":        wfe,
        "wfa_CI95_lo":    float(ev["wfa_CI95_lo"]),
        "wfa_t_p":        float(ev["wfa_t_p"]),
        # CPCV
        "cpcv_p10_at":    float(ev["cpcv_p10_at"]),
        "cpcv_worst_at":  float(ev["cpcv_worst_at"]),
        "cpcv_med_at":    float(ev["cpcv_med_at"]),
        # Regime
        "regime_min_at":  reg_min,
        "regime":         ev["regime"],
        # Stress
        "stress":         ev["stress"],
        # Bootstrap vs V7 (standard 1-axis from _eval_one)
        "boot_v7_P_min_better":    boot_ev.get("P_min_better", np.nan),
        "boot_v7_CI95_lo_min_pp":  boot_ev.get("CI95_lo_min_pp", np.nan),
        # Multi-metric bootstrap vs V7
        "mm_v7_P_min":     boot_v7["P_min_better"],
        "mm_v7_CI95_min":  boot_v7["CI95_lo_min_pp"],
        "mm_v7_P_maxdd":   boot_v7["P_maxdd_better"],
        "mm_v7_CI95_dd":   boot_v7["CI95_lo_dd_pp"],
        "mm_v7_P_w10y":    boot_v7["P_worst10y_better"],
        "mm_v7_CI95_w10y": boot_v7["CI95_lo_w10y_pp"],
        "mm_v7_P_sharpe":  boot_v7["P_sharpe_better"],
        "mm_v7_CI95_shp":  boot_v7["CI95_lo_sharpe"],
        # Multi-metric bootstrap vs B3a
        "mm_b3a_P_min":     boot_b3a["P_min_better"],
        "mm_b3a_CI95_min":  boot_b3a["CI95_lo_min_pp"],
        "mm_b3a_P_maxdd":   boot_b3a["P_maxdd_better"],
        "mm_b3a_CI95_dd":   boot_b3a["CI95_lo_dd_pp"],
        "mm_b3a_P_w10y":    boot_b3a["P_worst10y_better"],
        "mm_b3a_CI95_w10y": boot_b3a["CI95_lo_w10y_pp"],
        "mm_b3a_P_sharpe":  boot_b3a["P_sharpe_better"],
        "mm_b3a_CI95_shp":  boot_b3a["CI95_lo_sharpe"],
        # Veto
        "s1_veto_maxdd":  int(v_maxdd),
        "s1_veto_wfe":    int(v_wfe),
        "s1_veto_w10y":   int(v_w10y),
        "s1_veto_reg":    int(v_reg),
        "VETO_s1":        int(veto_s1),
    }


def _print_stage0_table(rows):
    """Print Stage 0 standard-10 table."""
    print("\n" + "=" * 140)
    print("STAGE 0 -- STANDARD 10 METRICS (after-tax CAGR; Sharpe/MaxDD pretax)")
    hdr = ("%-26s | %5s | %5s | %8s | %9s | %7s | %7s | %8s | %8s | %8s | %8s | %7s | %6s | %6s | %5s"
           % ("label", "scale", "map", "CAGR_IS%", "CAGR_OOS%",
              "min9%", "gap_pp", "Sharpe", "MaxDD%", "W10Y*%",
              "P10_5Y%", "Trd/yr", ">3x%", "wcy%", "VETO"))
    print(hdr)
    print("-" * 140)
    for row in rows:
        veto_str = "VETO" if row["VETO_s0"] else "pass"
        print("%-26s | %5.2f | %-5s | %+7.2f%% | %+8.2f%% | %+6.2f%% | %+6.2f | %7.3f | %+7.2f%% | %+7.2f%% | %+7.2f%% | %7.1f | %5.1f%% | %+5.2f%% | %-4s"
              % (row["label"][:26], row["lev_scale"], row["map_tag"][:5],
                 row["CAGR_IS_at"] * 100, row["CAGR_OOS_at"] * 100,
                 row["min9_at"] * 100, row["IS_OOS_gap_pp"],
                 row["Sharpe_OOS"], row["MaxDD_FULL"] * 100,
                 row["Worst10Y_star_at"] * 100, row["P10_5Y_at"] * 100,
                 row["Trades_yr"], row["excess_ratio_pct"],
                 row["worst_cy"] * 100, veto_str))


def main():
    print("=" * 120)
    print("LEVEREXT SCALE SWEEP  2026-06-16")
    print("Task #1: B3a uniform leverage extension -- scale {1.20, 1.25, 1.30, 1.35}")
    print("v7_map (a) default {1.40,1.40,1.05,1.00} (b) strong {1.60,1.50,1.10,1.00}")
    print("Cost model: k365 EXCESS_EXTRA=0.0025 (%% /yr for L>3x). C1 SOFR OUT fill.")
    print("Sanity gate: scale=1.15 default must reproduce B3a min9+20.98%+/-0.05pp / MaxDD-38.20%+/-0.10pp")
    print("Hard veto: MaxDD<-50%% / Worst10Y*<0. WFE/Regime veto applied in Stage 1.")
    print("=" * 120)

    # ---- Load shared data ----
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

    # ---- Gold/Bond auxiliary series ----
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
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    out_bond_off = fund_active & (~bond_on.astype(bool))
    n_out_bondoff = int(out_bond_off.sum())
    print("\nOUT-and-bondOFF days: %d of %d (%.1f%%)" % (n_out_bondoff, n, 100.0 * n_out_bondoff / n))

    # =========================================================================
    # SANITY GATE: B3a (scale=1.15, default map) must reproduce known values
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: Reproducing B3a_k365 (scale=1.15, default map)")
    print("  Expected: min9 +20.98%% +/-0.05pp  MaxDD -38.20%% +/-0.10pp")
    print("=" * 120)

    san_nav, san_r, san_tpy, san_exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3A_MAP_DEFAULT, lev_scale=1.15,
        excess_extra=EXCESS_EXTRA)

    san_pre = compute_10metrics(san_nav, san_tpy)
    san_aft = _apply_aftertax(san_pre)
    san_min9  = _min_at(san_aft)
    san_maxdd = san_pre["MaxDD_FULL"]

    ok_min9  = abs(san_min9  - B3A_KNOWN_MIN9)  <= B3A_SANITY_TOL_MIN9
    ok_maxdd = abs(san_maxdd - B3A_KNOWN_MAXDD) <= B3A_SANITY_TOL_MAXDD

    print("  min9:  got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_min9 * 100, B3A_KNOWN_MIN9 * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD: got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_maxdd * 100, B3A_KNOWN_MAXDD * 100, "OK" if ok_maxdd else "FAIL"))

    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED -- halting. Check _build_full_c1 / EXCESS_EXTRA wiring.")
        sys.exit(1)
    print("  SANITY PASSED. Proceeding.\n")

    # =========================================================================
    # STAGE 0: Build all 8 sweep configs + 2 references
    # =========================================================================
    print("=" * 120)
    print("STAGE 0: Building Stage-0 NAVs (8 sweep + 2 reference configs)")
    print("=" * 120)

    s0_rows_ordered = []  # preserve display order: refs first, then sweep
    nav_cache = {}  # label -> (nav_dt, r, tpy, exc)

    all_configs_ordered = REF_CONFIGS + CONFIGS

    for cfg in all_configs_ordered:
        lbl = cfg["label"]
        sc  = cfg["lev_scale"]
        mp  = cfg["v7_map"]
        print("  Building %s (scale=%.2f map=%s) ..." % (lbl, sc, cfg["map_tag"]))
        nav_dt, r, tpy, exc = _build_full_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            v7_map=mp, lev_scale=sc, excess_extra=EXCESS_EXTRA)
        nav_cache[lbl] = (nav_dt, r, tpy, exc)
        row = _stage0_row(lbl, nav_dt, tpy, exc, n, cfg["map_tag"], sc)
        s0_rows_ordered.append(row)
        print("    min9=%+.2f%%  MaxDD=%+.2f%%  Trades=%.1f  VETO=%s"
              % (row["min9_at"] * 100, row["MaxDD_FULL"] * 100,
                 row["Trades_yr"], "YES" if row["VETO_s0"] else "no"))

    # Print Stage-0 table
    _print_stage0_table(s0_rows_ordered)

    # Identify veto boundary
    print("\n--- VETO ANALYSIS ---")
    print("MaxDD < -50%% veto hit by:")
    any_maxdd_veto = False
    for row in s0_rows_ordered:
        if row["veto_maxdd"]:
            any_maxdd_veto = True
            print("  %s  scale=%.2f map=%s  MaxDD=%+.2f%%" % (
                row["label"], row["lev_scale"], row["map_tag"], row["MaxDD_FULL"] * 100))
    if not any_maxdd_veto:
        print("  (none in this sweep -- MaxDD-50%% frontier not yet reached)")

    # Response surface check (monotonicity)
    print("\n--- MONOTONICITY CHECK (min9 vs scale, each map_tag) ---")
    print("  Purpose: verify point estimates are monotone in scale (expected) and")
    print("  comment on selection bias risk (plan §3 stage-0).")
    for mtag in ["default", "strong"]:
        rows_mt = [(r["lev_scale"], r["min9_at"], r["MaxDD_FULL"])
                   for r in s0_rows_ordered
                   if r["map_tag"] == mtag and r["lev_scale"] in SWEEP_SCALES + [1.10, 1.15]]
        rows_mt.sort(key=lambda x: x[0])
        print("  map=%s:" % mtag)
        prev_m9 = None
        monotone = True
        for sc_v, m9, dd in rows_mt:
            mono_note = ""
            if prev_m9 is not None and m9 < prev_m9:
                monotone = False
                mono_note = " [NON-MONOTONE ALERT]"
            print("    scale=%.2f  min9=%+.2f%%  MaxDD=%+.2f%%%s"
                  % (sc_v, m9 * 100, dd * 100, mono_note))
            prev_m9 = m9
        if monotone:
            print("    => Monotone (as expected for pure leverage scaling).")
            print("       NOTE: monotone response does NOT eliminate selection bias;")
            print("       the grid was designed to maximize CAGR so point estimates")
            print("       are biased upward. Use CI95_lo / bootstrap as primary evidence.")
        else:
            print("    => NON-MONOTONE -- unusual; check for numerical issues.")

    # =========================================================================
    # V7_TQQQ baseline (for bootstrap)
    # =========================================================================
    print("\nBuilding V7_TQQQ baseline for bootstrap ...")
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    # B3a returns (for relative bootstrap)
    _, r_b3a, _, _ = nav_cache.get("B3a_k365_ref", (None, None, None, None))
    if r_b3a is None:
        # Rebuild B3a
        _, r_b3a, _, _ = _build_full_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            v7_map=B3A_MAP_DEFAULT, lev_scale=1.15, excess_extra=EXCESS_EXTRA)

    # =========================================================================
    # Regime labels and stress masks (for Stage 1)
    # =========================================================================
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress  = stress_masks(dates_dt)

    # =========================================================================
    # STAGE 1: Full gate on non-vetoed configs where min9 > B3a baseline
    # =========================================================================
    print("\n" + "=" * 120)
    print("STAGE 1: Selecting candidates for full gate")
    print("  Criteria: VETO_s0==0 AND min9_at > %.4f%% (B3a baseline)"
          % (B3A_KNOWN_MIN9 * 100))
    print("=" * 120)

    # Map s0_rows by label for quick lookup
    s0_by_label = {r["label"]: r for r in s0_rows_ordered}

    # Sweep rows only (not references)
    sweep_s0 = [r for r in s0_rows_ordered if r["label"].startswith("Bext_")]
    candidates_s1 = [r for r in sweep_s0
                     if r["VETO_s0"] == 0 and r["min9_at"] > B3A_KNOWN_MIN9]

    # Sort by min9 desc, keep top 3 (plan: 2-3)
    candidates_s1 = sorted(candidates_s1, key=lambda x: -x["min9_at"])[:3]

    if len(candidates_s1) == 0:
        print("  No sweep configs passed Stage-0 gate with min9 > B3a. "
              "Stage 1 skipped.")
        s1_results = []
    else:
        print("  %d candidate(s) selected:" % len(candidates_s1))
        for cand in candidates_s1:
            print("    %s  scale=%.2f  map=%s  min9=%+.2f%%  MaxDD=%+.2f%%"
                  % (cand["label"], cand["lev_scale"], cand["map_tag"],
                     cand["min9_at"] * 100, cand["MaxDD_FULL"] * 100))

        print("\n--- Running Stage-1 full gate ---")
        s1_results = []
        for cand in candidates_s1:
            lbl  = cand["label"]
            cfg  = next(c for c in CONFIGS if c["label"] == lbl)
            print("\n  [%s] WFA + CPCV + regime + stress + bootstrap ..." % lbl)
            nav_dt, r, tpy, exc = nav_cache[lbl]
            s1_r = _stage1_full_gate(
                lbl, nav_dt, r, tpy, regimes, stress,
                is_mask, oos_mask, r_v7, r_b3a)
            s1_results.append({"label": lbl, "s0": cand, "s1": s1_r,
                                "nav_dt": nav_dt, "r": r})
            print("    WFE=%.4f  CI95_lo=%+.2f%%  CPCV_p10=%+.2f%%  Regime_min=%+.2f%%  VETO=%s"
                  % (s1_r["wfa_WFE"], s1_r["wfa_CI95_lo"] * 100,
                     s1_r["cpcv_p10_at"] * 100, s1_r["regime_min_at"] * 100,
                     "YES" if s1_r["VETO_s1"] else "no"))
            print("    Bootstrap vs V7:  P_min=%.3f  CI95_min=%+.2f%%  P_maxdd=%.3f  P_w10y=%.3f  P_sharpe=%.3f"
                  % (s1_r["mm_v7_P_min"], s1_r["mm_v7_CI95_min"],
                     s1_r["mm_v7_P_maxdd"], s1_r["mm_v7_P_w10y"], s1_r["mm_v7_P_sharpe"]))
            print("    Bootstrap vs B3a: P_min=%.3f  CI95_min=%+.2f%%  P_maxdd=%.3f  P_w10y=%.3f  P_sharpe=%.3f"
                  % (s1_r["mm_b3a_P_min"], s1_r["mm_b3a_CI95_min"],
                     s1_r["mm_b3a_P_maxdd"], s1_r["mm_b3a_P_w10y"], s1_r["mm_b3a_P_sharpe"]))

    # =========================================================================
    # Stage-1 full gate table (if any)
    # =========================================================================
    if s1_results:
        print("\n" + "=" * 120)
        print("STAGE 1 FULL GATE RESULTS")
        print("%-28s | %6s | %8s | %7s | %8s | %8s | %8s | %8s | %4s"
              % ("label", "scale", "WFE", "CI95%", "CPCV_p10%",
                 "Reg_min%", "W10Y*%", "MaxDD%", "VETO"))
        print("-" * 120)
        for entry in s1_results:
            s0 = entry["s0"]
            s1 = entry["s1"]
            veto_str = "VETO" if s1["VETO_s1"] else "PASS"
            print("%-28s | %5.2f  | %7.4f | %+6.2f%% | %+7.2f%% | %+7.2f%% | %+7.2f%% | %+7.2f%% | %-4s"
                  % (entry["label"][:28], s0["lev_scale"],
                     s1["wfa_WFE"], s1["wfa_CI95_lo"] * 100,
                     s1["cpcv_p10_at"] * 100, s1["regime_min_at"] * 100,
                     s0["Worst10Y_star_at"] * 100, s0["MaxDD_FULL"] * 100,
                     veto_str))

        print("\n--- BOOTSTRAP SUMMARY (Stage 1 candidates) ---")
        print("%-28s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s"
              % ("label (vs V7/B3a)", "P_min_V7", "CI_min_V7", "P_min_B3a", "CI_min_B3a",
                 "P_dd_V7", "P_dd_B3a", "P_w10y_V7", "P_shp_V7"))
        print("-" * 120)
        for entry in s1_results:
            s1 = entry["s1"]
            print("%-28s | %8.3f | %+7.2f%% | %8.3f | %+7.2f%% | %8.3f | %8.3f | %8.3f | %8.3f"
                  % (entry["label"][:28],
                     s1["mm_v7_P_min"],  s1["mm_v7_CI95_min"],
                     s1["mm_b3a_P_min"], s1["mm_b3a_CI95_min"],
                     s1["mm_v7_P_maxdd"], s1["mm_b3a_P_maxdd"],
                     s1["mm_v7_P_w10y"],  s1["mm_v7_P_sharpe"]))

    # =========================================================================
    # CSV output
    # =========================================================================
    print("\nBuilding CSV ...")
    csv_rows = []

    # Stage-0 rows (all 10 configs)
    axes_order = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                  "rate:rate_up", "rate:rate_down"]

    for row in s0_rows_ordered:
        lbl = row["label"]
        r_entry = {"stage": 0}
        r_entry.update(row)
        # Merge Stage-1 columns (empty if not evaluated)
        s1_match = next((e for e in s1_results if e["label"] == lbl), None)
        if s1_match:
            s1 = s1_match["s1"]
            r_entry.update({
                "wfa_WFE":        s1["wfa_WFE"],
                "wfa_CI95_lo":    s1["wfa_CI95_lo"],
                "wfa_t_p":        s1["wfa_t_p"],
                "cpcv_p10_at":    s1["cpcv_p10_at"],
                "cpcv_worst_at":  s1["cpcv_worst_at"],
                "cpcv_med_at":    s1["cpcv_med_at"],
                "regime_min_at":  s1["regime_min_at"],
                "s1_veto_maxdd":  s1["s1_veto_maxdd"],
                "s1_veto_wfe":    s1["s1_veto_wfe"],
                "s1_veto_w10y":   s1["s1_veto_w10y"],
                "s1_veto_reg":    s1["s1_veto_reg"],
                "VETO_s1":        s1["VETO_s1"],
                "mm_v7_P_min":    s1["mm_v7_P_min"],
                "mm_v7_CI95_min": s1["mm_v7_CI95_min"],
                "mm_v7_P_maxdd":  s1["mm_v7_P_maxdd"],
                "mm_v7_CI95_dd":  s1["mm_v7_CI95_dd"],
                "mm_v7_P_w10y":   s1["mm_v7_P_w10y"],
                "mm_v7_CI95_w10y":s1["mm_v7_CI95_w10y"],
                "mm_v7_P_sharpe": s1["mm_v7_P_sharpe"],
                "mm_v7_CI95_shp": s1["mm_v7_CI95_shp"],
                "mm_b3a_P_min":   s1["mm_b3a_P_min"],
                "mm_b3a_CI95_min":s1["mm_b3a_CI95_min"],
                "mm_b3a_P_maxdd": s1["mm_b3a_P_maxdd"],
                "mm_b3a_CI95_dd": s1["mm_b3a_CI95_dd"],
                "mm_b3a_P_w10y":  s1["mm_b3a_P_w10y"],
                "mm_b3a_CI95_w10y":s1["mm_b3a_CI95_w10y"],
                "mm_b3a_P_sharpe":s1["mm_b3a_P_sharpe"],
                "mm_b3a_CI95_shp":s1["mm_b3a_CI95_shp"],
            })
            # Regime breakdown
            for ax in axes_order:
                r_entry["regime_" + ax.replace(":", "_")] = s1["regime"].get(ax, np.nan)
            # Stress
            for sw, sv in s1["stress"].items():
                r_entry["stress_%s_ret" % sw]   = sv["ret"]
                r_entry["stress_%s_maxdd" % sw] = sv["maxdd"]
        else:
            # Fill with empty strings
            for col in ["wfa_WFE", "wfa_CI95_lo", "wfa_t_p", "cpcv_p10_at",
                        "cpcv_worst_at", "cpcv_med_at", "regime_min_at",
                        "s1_veto_maxdd", "s1_veto_wfe", "s1_veto_w10y",
                        "s1_veto_reg", "VETO_s1",
                        "mm_v7_P_min", "mm_v7_CI95_min", "mm_v7_P_maxdd",
                        "mm_v7_CI95_dd", "mm_v7_P_w10y", "mm_v7_CI95_w10y",
                        "mm_v7_P_sharpe", "mm_v7_CI95_shp",
                        "mm_b3a_P_min", "mm_b3a_CI95_min", "mm_b3a_P_maxdd",
                        "mm_b3a_CI95_dd", "mm_b3a_P_w10y", "mm_b3a_CI95_w10y",
                        "mm_b3a_P_sharpe", "mm_b3a_CI95_shp"]:
                r_entry[col] = ""
            for ax in axes_order:
                r_entry["regime_" + ax.replace(":", "_")] = ""

        csv_rows.append(r_entry)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "leverext_scale_20260616.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f",
                                   encoding="utf-8-sig")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(csv_rows)))

    # =========================================================================
    # CONCLUSIONS
    # =========================================================================
    print("\n" + "=" * 120)
    print("CONCLUSIONS")
    print("=" * 120)

    # CAGR-DD frontier
    print("\n1. CAGR-DD FRONTIER (B3a default map, scale sweep):")
    print("   scale  | min9%   | MaxDD%  | W10Y*%  | VETO   | Sharpe")
    print("   -------+---------+---------+---------+--------+-------")
    for row in sorted(
        [r for r in s0_rows_ordered if r["map_tag"] == "default"],
        key=lambda x: x["lev_scale"]
    ):
        veto_str = "VETO" if row["VETO_s0"] else "pass"
        print("   %.2f   | %+6.2f%% | %+6.2f%% | %+6.2f%% | %-6s | %.3f"
              % (row["lev_scale"], row["min9_at"] * 100,
                 row["MaxDD_FULL"] * 100, row["Worst10Y_star_at"] * 100,
                 veto_str, row["Sharpe_OOS"]))

    print("\n2. VETO BOUNDARY:")
    veto_first = next(
        (r for r in sorted([r for r in s0_rows_ordered if r["map_tag"] == "default"],
                            key=lambda x: x["lev_scale"])
         if r["VETO_s0"] and r["lev_scale"] in SWEEP_SCALES),
        None)
    if veto_first:
        print("   MaxDD-50%% veto first hit at scale=%.2f (default map), MaxDD=%+.2f%%"
              % (veto_first["lev_scale"], veto_first["MaxDD_FULL"] * 100))
        print("   -> CAGR-DD frontier upper bound = scale %.2f (just below veto)"
              % veto_first["lev_scale"])
    else:
        prev_sc_default = [r for r in s0_rows_ordered
                           if r["map_tag"] == "default" and r["lev_scale"] in SWEEP_SCALES]
        if prev_sc_default:
            max_ok = max(prev_sc_default, key=lambda x: x["lev_scale"])
            print("   MaxDD-50%% veto NOT triggered through scale=%.2f (default map)."
                  % max_ok["lev_scale"])
            print("   Highest scale tested = %.2f  MaxDD=%+.2f%%  min9=%+.2f%%"
                  % (max_ok["lev_scale"], max_ok["MaxDD_FULL"] * 100,
                     max_ok["min9_at"] * 100))

    print("\n3. SHARPE / TRADES trade-off (procyclical leverage cost):")
    ref_b3a = s0_by_label.get("B3a_k365_ref")
    if ref_b3a:
        print("   B3a (scale=1.15): Sharpe=%.3f  Trades/yr=%.1f"
              % (ref_b3a["Sharpe_OOS"], ref_b3a["Trades_yr"]))
    for row in sorted(
        [r for r in s0_rows_ordered if r["map_tag"] == "default"
         and r["lev_scale"] in SWEEP_SCALES],
        key=lambda x: x["lev_scale"]
    ):
        dd_vs_b3a = (row["MaxDD_FULL"] - B3A_KNOWN_MAXDD) * 100
        print("   scale=%.2f: Sharpe=%.3f  Trades/yr=%.1f  MaxDD delta vs B3a=%+.2fpp"
              % (row["lev_scale"], row["Sharpe_OOS"], row["Trades_yr"], dd_vs_b3a))

    if s1_results:
        print("\n4. BEST STAGE-1 CANDIDATE (min9 > B3a, vet-free):")
        best_s1 = min(s1_results, key=lambda x: x["s1"]["VETO_s1"] * 1000 - x["s0"]["min9_at"])
        s0b = best_s1["s0"]
        s1b = best_s1["s1"]
        print("   %s  scale=%.2f  map=%s"
              % (best_s1["label"], s0b["lev_scale"], s0b["map_tag"]))
        print("   min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f  Trades/yr=%.1f"
              % (s0b["min9_at"] * 100, s0b["MaxDD_FULL"] * 100,
                 s0b["Sharpe_OOS"], s0b["Trades_yr"]))
        print("   WFA: WFE=%.4f  CI95_lo=%+.2f%%  t_p=%.4f"
              % (s1b["wfa_WFE"], s1b["wfa_CI95_lo"] * 100, s1b["wfa_t_p"]))
        print("   CPCV p10=%+.2f%%  worst=%+.2f%%"
              % (s1b["cpcv_p10_at"] * 100, s1b["cpcv_worst_at"] * 100))
        print("   Regime_min=%+.2f%%  VETO=%s"
              % (s1b["regime_min_at"] * 100, "YES" if s1b["VETO_s1"] else "no"))
        print("   Bootstrap vs V7: P_min=%.3f CI95=%+.2f%%  vs B3a: P_min=%.3f CI95=%+.2f%%"
              % (s1b["mm_v7_P_min"], s1b["mm_v7_CI95_min"],
                 s1b["mm_b3a_P_min"], s1b["mm_b3a_CI95_min"]))
        print("   NOTE: Sharpe does NOT improve vs B3a (procyclical leverage -- expected).")
        print("   The CAGR gain vs B3a is the only benefit; DD worsens proportionally.")
    else:
        print("\n4. No Stage-1 candidates (all sweep configs vetoed or min9 <= B3a).")

    print("\n5. SELECTION BIAS NOTE:")
    print("   Grid was pre-registered in LEVERUP_EXTENSION_PLAN_20260616.md.")
    print("   Response surface monotone -> consistent with pure leverage mechanism")
    print("   (not a data-mining artifact). However, the point estimate for min9")
    print("   at high scale is still upward-biased because this grid was designed")
    print("   specifically to maximize CAGR. Treat CI95_lo / bootstrap P_better as")
    print("   primary evidence for any adoption decision.")
    print("   Plan rule: vet緩和禁止 (MaxDD-50%% is absolute). "
          "Adopt only vet-free best.")

    # =========================================================================
    # RETURN_BLOCK JSON
    # =========================================================================
    rb_s0 = []
    for row in s0_rows_ordered:
        rb_s0.append({
            "label":        row["label"],
            "lev_scale":    row["lev_scale"],
            "map_tag":      row["map_tag"],
            "min9_at_pct":  round(row["min9_at"] * 100, 4),
            "MaxDD_pct":    round(row["MaxDD_FULL"] * 100, 4),
            "Sharpe_OOS":   round(row["Sharpe_OOS"], 4),
            "Worst10Y_at_pct": round(row["Worst10Y_star_at"] * 100, 4),
            "Trades_yr":    round(row["Trades_yr"], 2),
            "IS_OOS_gap_pp":round(row["IS_OOS_gap_pp"], 4),
            "excess_ratio_pct": round(row["excess_ratio_pct"], 2),
            "VETO_s0":      row["VETO_s0"],
        })

    rb_s1 = []
    for entry in s1_results:
        s0r = entry["s0"]
        s1r = entry["s1"]
        rb_s1.append({
            "label":        entry["label"],
            "lev_scale":    s0r["lev_scale"],
            "map_tag":      s0r["map_tag"],
            "min9_at_pct":  round(s0r["min9_at"] * 100, 4),
            "MaxDD_pct":    round(s0r["MaxDD_FULL"] * 100, 4),
            "wfa_WFE":      round(s1r["wfa_WFE"], 4),
            "wfa_CI95_lo_pct": round(s1r["wfa_CI95_lo"] * 100, 4),
            "wfa_t_p":      round(s1r["wfa_t_p"], 4),
            "cpcv_p10_at_pct": round(s1r["cpcv_p10_at"] * 100, 4),
            "regime_min_at_pct": round(s1r["regime_min_at"] * 100, 4),
            "VETO_s1":      s1r["VETO_s1"],
            "mm_v7_P_min":  round(float(s1r["mm_v7_P_min"]), 4),
            "mm_v7_CI95_min_pp": round(float(s1r["mm_v7_CI95_min"]), 4),
            "mm_v7_P_maxdd": round(float(s1r["mm_v7_P_maxdd"]), 4),
            "mm_v7_P_w10y":  round(float(s1r["mm_v7_P_w10y"]), 4),
            "mm_v7_P_sharpe":round(float(s1r["mm_v7_P_sharpe"]), 4),
            "mm_b3a_P_min":  round(float(s1r["mm_b3a_P_min"]), 4),
            "mm_b3a_CI95_min_pp": round(float(s1r["mm_b3a_CI95_min"]), 4),
            "mm_b3a_P_maxdd":round(float(s1r["mm_b3a_P_maxdd"]), 4),
            "mm_b3a_P_w10y": round(float(s1r["mm_b3a_P_w10y"]), 4),
            "mm_b3a_P_sharpe":round(float(s1r["mm_b3a_P_sharpe"]), 4),
        })

    return_block = {
        "script":   "leverext_scale_20260616.py",
        "date":     "2026-06-16",
        "sanity":   {
            "B3a_min9_got_pct":   round(san_min9 * 100, 4),
            "B3a_MaxDD_got_pct":  round(san_maxdd * 100, 4),
            "ok_min9":  bool(ok_min9),
            "ok_maxdd": bool(ok_maxdd),
            "SANITY_PASS": bool(ok_min9 and ok_maxdd),
        },
        "stage0":   rb_s0,
        "stage1":   rb_s1,
        "csv_path": csv_path,
        "n_out_bondoff": n_out_bondoff,
        "excess_extra_pct": round(EXCESS_EXTRA * 100, 4),
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
