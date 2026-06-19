"""
src/audit/leverext_high_stage1_20260618.py
==========================================
Stage-1 full gate verification for high leverage scales:
  lev_scale in {1.35 (sanity), 1.40, 1.50, 1.60}
  v7_map = STRONG ONLY: {0:1.60, 1:1.50, 2:1.10, 3:1.00}

Purpose:
  Following Stage-0 (leverext_high_20260618.py), apply the full Stage-1
  gate battery used in leverext_scale_20260616.py to the high-scale configs:
    1. WFA (49 windows): CI95_lo, WFE, t_p
    2. CPCV (N=10, k=2, embargo=21): p10/worst/median after-tax CAGR
    3. Multi-metric bootstrap vs V7 and vs B3a (4 axes: P_min, P_maxdd, P_w10y, P_sharpe)
    4. Regime breakdown: trend/vol/rate CAGR, Regime_min(bear)
    5. Standard 10 metrics v2.0

Hard veto criteria (all 4 axes):
  MaxDD < -50%    -> VETO (Stage-0: also caught here for clarity)
  WFE > 1.5       -> VETO
  Worst10Y* < 0   -> VETO
  Regime_min(bear) < -10% -> VETO

Sanity gate (required before proceeding):
  scale=1.35 strong map must reproduce:
    min9    +23.83% +/-0.10pp  (from leverext_high_20260618.py run)
    MaxDD   -45.04% +/-0.10pp
    WFA CI95_lo ~+26.55%       (from leverext_scale_20260616.py Stage-1)
    WFE     ~0.945             (from leverext_scale_20260616.py Stage-1)

Stage-0 pre-known results (from leverext_high_20260618.csv):
  1.35: min9=23.83%  MaxDD=-45.04%  W10Y*=+17.43%  VETO=0
  1.40: min9=24.34%  MaxDD=-46.48%  W10Y*=+17.87%  VETO=0
  1.50: min9=25.31%  MaxDD=-49.27%  W10Y*=+18.70%  VETO=0  (Worst5Y=-0.02% close to zero)
  1.60: min9=26.21%  MaxDD=-51.95%  W10Y*=+19.36%  VETO=1 (MaxDD breach)

Scale 1.60 is hard-vetoed at Stage-0. Stage-1 is still run for completeness but
will be marked VETO=1 and excluded from adoption.

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
Outputs:
  audit_results/leverext_high_stage1_20260618.csv
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

# Reuse builders from k365_recost (B3a cost model)
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,
    EXCESS_EXTRA_STORE,
)

# Reuse extended_eval for full Stage-1 gate
from src.audit.extended_eval_20260611 import (
    _eval_one, _cpcv_dist, _regime_cagr,
)

# Reuse bootstrap functions
from src.audit.multimetric_bootstrap_20260615 import (
    _block_bootstrap_multimetric,
)

# Reuse cost/NAV helpers
from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base,
    AFTER_TAX, SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)

# Misc helpers
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
# Constants
# ---------------------------------------------------------------------------
B3A_MAP_STRONG = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}  # strong boost map
B3A_MAP_DEFAULT = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}  # B3a default (for B3a reference)

EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# Sanity gate: scale=1.35 strong map (from leverext_high_20260618.csv)
SANITY_SCALE = 1.35
SANITY_MIN9_EXPECT   = 0.2383   # +23.83%
SANITY_MAXDD_EXPECT  = -0.4504  # -45.04%
SANITY_TOL           = 0.0010   # +/-0.10pp

# Stage-1 sanity from leverext_scale_20260616 run
SANITY_WFA_CI95_EXPECT = 0.2655   # ~+26.55%
SANITY_WFE_EXPECT      = 0.945    # ~0.945
SANITY_WFA_CI95_TOL    = 0.015    # soft tolerance
SANITY_WFE_TOL         = 0.05     # soft tolerance

# Hard veto thresholds
HARD_VETO_MAXDD  = -0.50
HARD_VETO_W10Y   = 0.0
HARD_VETO_WFE    = 1.5
HARD_VETO_REGIME = -0.10   # Regime_min(bear) < -10%

# Sweep scales: 1.35 as sanity check, then 1.40, 1.50, 1.60
SWEEP_SCALES = [1.35, 1.40, 1.50, 1.60]

# Stage-0 pre-known results (from leverext_high_20260618.csv, for reference display)
STAGE0_KNOWN = {
    1.35: {"min9": 0.238321, "MaxDD": -0.450414, "W10Y_star": 0.174340,
           "Worst5Y": 0.003281, "VETO": 0},
    1.40: {"min9": 0.243414, "MaxDD": -0.464781, "W10Y_star": 0.178682,
           "Worst5Y": 0.002234, "VETO": 0},
    1.50: {"min9": 0.253111, "MaxDD": -0.492692, "W10Y_star": 0.186984,
           "Worst5Y": -0.000209, "VETO": 0},
    1.60: {"min9": 0.262127, "MaxDD": -0.519521, "W10Y_star": 0.193613,
           "Worst5Y": -0.003115, "VETO": 1},
}


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _stage1_full_gate(label, nav_dt, r, tpy, regimes, stress,
                      is_mask, oos_mask, r_v7, r_b3a):
    """
    Full Stage-1 gate: WFA (49 windows) + CPCV + regime + stress + bootstrap.
    Identical logic to leverext_scale_20260616.py::_stage1_full_gate.
    """
    ev = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                   baseline_r=r_v7)
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)

    wfe     = float(ev["wfa_WFE"])
    reg_min = float(ev["regime_min_at"])
    w10y    = aft["Worst10Y_star"]
    maxdd   = pre["MaxDD_FULL"]

    v_maxdd  = maxdd < HARD_VETO_MAXDD
    v_wfe    = wfe   > HARD_VETO_WFE
    v_w10y   = w10y  < HARD_VETO_W10Y
    v_reg    = reg_min < HARD_VETO_REGIME
    veto_s1  = v_maxdd or v_wfe or v_w10y or v_reg

    # Multi-metric bootstrap vs V7 (baseline)
    print("      bootstrap vs V7 ...")
    boot_v7  = _block_bootstrap_multimetric(r, r_v7, is_mask, oos_mask,
                                             n_boot=N_BOOT, block=BLOCK, seed=SEED)
    # Multi-metric bootstrap vs B3a
    print("      bootstrap vs B3a ...")
    boot_b3a = _block_bootstrap_multimetric(r, r_b3a, is_mask, oos_mask,
                                             n_boot=N_BOOT, block=BLOCK, seed=SEED)

    boot_ev = ev.get("boot") or {}

    return {
        # Standard 10 metrics (after-tax where applicable)
        "CAGR_IS_at":   aft["CAGR_IS"],
        "CAGR_OOS_at":  aft["CAGR_OOS"],
        "min9_at":      _min_at(aft),
        "Sharpe_FULL":  pre["Sharpe_FULL"],
        "Sharpe_OOS":   pre["Sharpe_OOS"],
        "MaxDD_FULL":   maxdd,
        "Worst10Y_at":  w10y,
        "P10_5Y_at":    aft["P10_5Y"],
        "Worst5Y_at":   aft["Worst5Y"],
        "Trades_yr":    aft["Trades_yr"],
        "Worst1D":      pre["Worst1D"],
        "Worst1D_date": pre["Worst1D_date"],
        "IS_OOS_gap_pp":aft["IS_OOS_gap_pp"],
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
        # Veto flags
        "s1_veto_maxdd": int(v_maxdd),
        "s1_veto_wfe":   int(v_wfe),
        "s1_veto_w10y":  int(v_w10y),
        "s1_veto_reg":   int(v_reg),
        "VETO_s1":       int(veto_s1),
    }


def main():
    print("=" * 120)
    print("LEVEREXT HIGH SCALE -- STAGE 1 FULL GATE  2026-06-18")
    print("v7_map STRONG: {0:1.60, 1:1.50, 2:1.10, 3:1.00}")
    print("lev_scale in {1.35 (sanity), 1.40, 1.50, 1.60}")
    print("Stage-1 gates: WFA (49w) + CPCV (N=10,k=2,emb=21) + Regime + Bootstrap vs V7/B3a")
    print("Hard veto: MaxDD<-50%% / WFE>1.5 / Worst10Y*<0 / Regime_min(bear)<-10%%")
    print("Sanity: scale=1.35 strong -> min9 +23.83%%+/-0.10pp / MaxDD -45.04%%+/-0.10pp")
    print("        WFA CI95_lo ~+26.55%% (soft) / WFE ~0.945 (soft)")
    print("NOTE: scale=1.60 is pre-vetoed at Stage-0 (MaxDD=-51.95%%), still run for completeness")
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

    # =========================================================================
    # SANITY GATE: Reproduce scale=1.35 strong map
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: Reproducing Bext_str_sc1.35 (strong map)")
    print("  Expected: min9 +23.83%% +/-0.10pp  MaxDD -45.04%% +/-0.10pp")
    print("=" * 120)

    san_nav, san_r, san_tpy, san_exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3A_MAP_STRONG, lev_scale=SANITY_SCALE,
        excess_extra=EXCESS_EXTRA)

    san_pre = compute_10metrics(san_nav, san_tpy)
    san_aft = _apply_aftertax(san_pre)
    san_min9  = _min_at(san_aft)
    san_maxdd = san_pre["MaxDD_FULL"]

    ok_min9  = abs(san_min9  - SANITY_MIN9_EXPECT)  <= SANITY_TOL
    ok_maxdd = abs(san_maxdd - SANITY_MAXDD_EXPECT) <= SANITY_TOL

    print("  min9:  got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_min9 * 100, SANITY_MIN9_EXPECT * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD: got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_maxdd * 100, SANITY_MAXDD_EXPECT * 100, "OK" if ok_maxdd else "FAIL"))

    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED -- halting. Check _build_full_c1 / EXCESS_EXTRA wiring.")
        sys.exit(1)
    print("  SANITY PASSED. Proceeding to Stage-1.\n")

    # =========================================================================
    # Build V7_TQQQ baseline and B3a reference for bootstrap
    # =========================================================================
    print("Building V7_TQQQ baseline (for bootstrap) ...")
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    print("Building B3a_k365 reference (scale=1.15, default map) ...")
    _, r_b3a, _, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3A_MAP_DEFAULT, lev_scale=1.15, excess_extra=EXCESS_EXTRA)

    # =========================================================================
    # Regime labels and stress masks
    # =========================================================================
    print("Building regime labels and stress masks ...")
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress  = stress_masks(dates_dt)

    # =========================================================================
    # Stage-1 full gate for all sweep scales
    # =========================================================================
    print("\n" + "=" * 120)
    print("STAGE 1: Full gate for all 4 scales (including pre-vetoed 1.60 for completeness)")
    print("=" * 120)

    s1_results = []
    nav_cache  = {}  # scale -> (nav_dt, r, tpy, exc)

    for sc in SWEEP_SCALES:
        lbl = "Bext_str_sc%.2f" % sc
        s0_known = STAGE0_KNOWN[sc]

        print("\n  [%s] Stage-1 full gate (scale=%.2f) ..." % (lbl, sc))
        print("    Stage-0 known: min9=%+.2f%%  MaxDD=%+.2f%%  VETO_s0=%s"
              % (s0_known["min9"] * 100, s0_known["MaxDD"] * 100,
                 "YES" if s0_known["VETO"] else "no"))
        if s0_known["VETO"]:
            print("    NOTE: This scale is already VETOED at Stage-0 (MaxDD breach).")
            print("         Running Stage-1 for completeness; result will be marked VETO.")

        # Build NAV (or reuse sanity for 1.35)
        if abs(sc - SANITY_SCALE) < 0.001:
            nav_dt = san_nav
            r      = san_r
            tpy    = san_tpy
            exc    = san_exc
        else:
            print("    Building NAV for scale=%.2f ..." % sc)
            nav_dt, r, tpy, exc = _build_full_c1(
                shared, dates_dt, n_years,
                ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
                v7_map=B3A_MAP_STRONG, lev_scale=sc, excess_extra=EXCESS_EXTRA)

        nav_cache[sc] = (nav_dt, r, tpy, exc)

        print("    Running WFA + CPCV + regime + stress + bootstrap ...")
        s1_r = _stage1_full_gate(
            lbl, nav_dt, r, tpy, regimes, stress,
            is_mask, oos_mask, r_v7, r_b3a)

        # Inherit Stage-0 veto into Stage-1 if already vetoed
        if s0_known["VETO"] and not s1_r["VETO_s1"]:
            s1_r["VETO_s1"] = 1
            s1_r["s1_veto_maxdd"] = 1

        s1_results.append({"label": lbl, "scale": sc, "s1": s1_r, "s0": s0_known})

        veto_reasons = []
        if s1_r["s1_veto_maxdd"]: veto_reasons.append("MaxDD<-50%%")
        if s1_r["s1_veto_wfe"]:   veto_reasons.append("WFE>1.5")
        if s1_r["s1_veto_w10y"]:  veto_reasons.append("W10Y*<0")
        if s1_r["s1_veto_reg"]:   veto_reasons.append("Regime_min<-10%%")

        print("    WFE=%.4f  CI95_lo=%+.2f%%  t_p=%.4e"
              % (s1_r["wfa_WFE"], s1_r["wfa_CI95_lo"] * 100, s1_r["wfa_t_p"]))
        print("    CPCV p10=%+.2f%%  worst=%+.2f%%  med=%+.2f%%"
              % (s1_r["cpcv_p10_at"] * 100, s1_r["cpcv_worst_at"] * 100,
                 s1_r["cpcv_med_at"] * 100))
        print("    Regime_min=%+.2f%%  [VETO=%s  reasons=%s]"
              % (s1_r["regime_min_at"] * 100,
                 "YES" if s1_r["VETO_s1"] else "no",
                 (", ".join(veto_reasons)) if veto_reasons else "none"))
        print("    Bootstrap vs V7:  P_min=%.3f  CI95_min=%+.2f%%  P_maxdd=%.3f  P_w10y=%.3f  P_sharpe=%.3f"
              % (s1_r["mm_v7_P_min"],  s1_r["mm_v7_CI95_min"],
                 s1_r["mm_v7_P_maxdd"], s1_r["mm_v7_P_w10y"],  s1_r["mm_v7_P_sharpe"]))
        print("    Bootstrap vs B3a: P_min=%.3f  CI95_min=%+.2f%%  P_maxdd=%.3f  P_w10y=%.3f  P_sharpe=%.3f"
              % (s1_r["mm_b3a_P_min"],  s1_r["mm_b3a_CI95_min"],
                 s1_r["mm_b3a_P_maxdd"], s1_r["mm_b3a_P_w10y"], s1_r["mm_b3a_P_sharpe"]))

    # =========================================================================
    # SANITY CHECK: Stage-1 values for scale=1.35
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY CHECK -- Stage-1 values for scale=1.35 vs expected")
    print("  Expected from leverext_scale_20260616 Stage-1: CI95_lo~+26.55%%  WFE~0.945")
    print("=" * 120)
    san_s1 = next(e["s1"] for e in s1_results if abs(e["scale"] - 1.35) < 0.001)
    ok_ci95 = abs(san_s1["wfa_CI95_lo"] - SANITY_WFA_CI95_EXPECT) <= SANITY_WFA_CI95_TOL
    ok_wfe  = abs(san_s1["wfa_WFE"]    - SANITY_WFE_EXPECT)      <= SANITY_WFE_TOL
    print("  WFA CI95_lo: got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_s1["wfa_CI95_lo"] * 100, SANITY_WFA_CI95_EXPECT * 100,
             "OK" if ok_ci95 else "WARN (soft tolerance exceeded)"))
    print("  WFE:         got %.4f     expect ~%.4f     -> %s"
          % (san_s1["wfa_WFE"], SANITY_WFE_EXPECT,
             "OK" if ok_wfe else "WARN (soft tolerance exceeded)"))
    print("  Regime_min:  got %+.4f%%" % (san_s1["regime_min_at"] * 100))
    print("  CPCV p10:    got %+.4f%%" % (san_s1["cpcv_p10_at"] * 100))

    # =========================================================================
    # STAGE-1 GATE TABLE
    # =========================================================================
    print("\n" + "=" * 140)
    print("STAGE 1 FULL GATE TABLE")
    print("%-22s | %5s | %7s | %8s | %8s | %7s | %9s | %9s | %8s | %8s | %4s"
          % ("label", "scale", "min9%", "MaxDD%", "CI95%", "WFE",
             "CPCV_p10%", "Reg_min%", "W10Y*%", "Worst5Y%", "VETO"))
    print("-" * 140)
    for entry in s1_results:
        s1 = entry["s1"]
        s0 = entry["s0"]
        veto_str = "VETO" if s1["VETO_s1"] else "PASS"
        print("%-22s | %5.2f | %+6.2f%% | %+7.2f%% | %+7.2f%% | %7.4f | %+8.2f%% | %+8.2f%% | %+7.2f%% | %+7.2f%% | %-4s"
              % (entry["label"][:22], entry["scale"],
                 s0["min9"] * 100, s0["MaxDD"] * 100,
                 s1["wfa_CI95_lo"] * 100, s1["wfa_WFE"],
                 s1["cpcv_p10_at"] * 100, s1["regime_min_at"] * 100,
                 s0["W10Y_star"] * 100, s0["Worst5Y"] * 100, veto_str))

    print("\n--- VETO DETAILS ---")
    for entry in s1_results:
        s1 = entry["s1"]
        reasons = []
        if s1["s1_veto_maxdd"]: reasons.append("MaxDD<-50%%")
        if s1["s1_veto_wfe"]:   reasons.append("WFE>1.5")
        if s1["s1_veto_w10y"]:  reasons.append("W10Y*<0")
        if s1["s1_veto_reg"]:   reasons.append("Regime_min<-10%%")
        status = ("VETO [%s]" % ", ".join(reasons)) if reasons else "PASS"
        print("  scale=%.2f  %s" % (entry["scale"], status))

    # =========================================================================
    # BOOTSTRAP SUMMARY
    # =========================================================================
    print("\n" + "=" * 120)
    print("BOOTSTRAP SUMMARY (Stage-1 candidates)")
    print("%-22s | %8s | %9s | %8s | %9s | %8s | %8s | %8s | %8s"
          % ("label", "P_min_V7", "CI_min_V7", "P_min_B3a", "CI_min_B3a",
             "P_dd_V7", "P_dd_B3a", "P_w10y_V7", "P_shp_V7"))
    print("-" * 120)
    for entry in s1_results:
        s1 = entry["s1"]
        print("%-22s | %8.3f | %+8.2f%% | %8.3f | %+8.2f%% | %8.3f | %8.3f | %8.3f | %8.3f"
              % (entry["label"][:22],
                 s1["mm_v7_P_min"],  s1["mm_v7_CI95_min"],
                 s1["mm_b3a_P_min"], s1["mm_b3a_CI95_min"],
                 s1["mm_v7_P_maxdd"], s1["mm_b3a_P_maxdd"],
                 s1["mm_v7_P_w10y"],  s1["mm_v7_P_sharpe"]))

    # =========================================================================
    # STANDARD 10 METRICS TABLE
    # =========================================================================
    print("\n" + "=" * 140)
    print("STANDARD 10 METRICS v2.0 -- All Stage-1 scales")
    print("%-22s | %8s | %9s | %7s | %7s | %8s | %8s | %7s | %7s | %7s | %5s"
          % ("label", "CAGR_IS%", "CAGR_OOS%", "Sharpe", "MaxDD%",
             "Worst1D%", "W10Y*%", "P10_5Y%", "W5Y%", "Trd/yr", "s1"))
    print("-" * 140)
    for entry in s1_results:
        s1 = entry["s1"]
        wd = s1["Worst1D"]
        wdd = s1["Worst1D_date"]
        s1_str = "PASS" if not s1["VETO_s1"] else "VETO"
        print("%-22s | %+7.2f%% | %+8.2f%% | %7.4f | %+6.2f%% | %+7.2f%%(%-10s) | %+6.2f%% | %+6.2f%% | %+6.2f%% | %5.1f | %-4s"
              % (entry["label"][:22],
                 s1["CAGR_IS_at"] * 100,  s1["CAGR_OOS_at"] * 100,
                 s1["Sharpe_OOS"],          s1["MaxDD_FULL"] * 100,
                 wd * 100 if wd is not None else float("nan"),
                 str(wdd) if wdd else "N/A",
                 s1["Worst10Y_at"] * 100,  s1["P10_5Y_at"] * 100,
                 s1["Worst5Y_at"] * 100,   s1["Trades_yr"],
                 s1_str))

    # =========================================================================
    # REGIME BREAKDOWN TABLE
    # =========================================================================
    print("\n" + "=" * 120)
    print("REGIME BREAKDOWN (after-tax CAGR)")
    axes_order = [("trend:bull", "bull"), ("trend:bear", "bear"),
                  ("vol:calm",   "calm"), ("vol:highvol", "hvol"),
                  ("rate:rate_up", "r_up"), ("rate:rate_down", "r_dn")]
    hdr_parts = ["%-22s" % "label"] + ["%8s" % tag for _, tag in axes_order] + ["%8s" % "Reg_min"]
    print(" | ".join(hdr_parts))
    print("-" * 120)
    for entry in s1_results:
        s1 = entry["s1"]
        reg = s1["regime"]
        vals = [reg.get(ax, float("nan")) for ax, _ in axes_order]
        row_str = "%-22s | " % entry["label"][:22]
        row_str += " | ".join("%+7.2f%%" % (v * 100) for v in vals)
        row_str += " | %+7.2f%%" % (s1["regime_min_at"] * 100)
        print(row_str)

    # =========================================================================
    # CONCLUSIONS
    # =========================================================================
    print("\n" + "=" * 120)
    print("CONCLUSIONS")
    print("=" * 120)

    pass_entries = [e for e in s1_results if not e["s1"]["VETO_s1"]]
    veto_entries = [e for e in s1_results if e["s1"]["VETO_s1"]]

    print("\n1. VETO SUMMARY:")
    for entry in s1_results:
        s1 = entry["s1"]
        reasons = []
        if s1["s1_veto_maxdd"]: reasons.append("MaxDD=%.2f%%<-50%%" % (s1["MaxDD_FULL"] * 100))
        if s1["s1_veto_wfe"]:   reasons.append("WFE=%.4f>1.5" % s1["wfa_WFE"])
        if s1["s1_veto_w10y"]:  reasons.append("W10Y*=%.2f%%<0" % (s1["Worst10Y_at"] * 100))
        if s1["s1_veto_reg"]:   reasons.append("Reg_min=%.2f%%<-10%%" % (s1["regime_min_at"] * 100))
        status = ("VETO [%s]" % "; ".join(reasons)) if reasons else "PASS"
        print("   scale=%.2f: %s" % (entry["scale"], status))

    print("\n2. WFE / REGIME_MIN LIMIT CONDITIONS:")
    for entry in s1_results:
        s1 = entry["s1"]
        wfe_margin    = HARD_VETO_WFE    - s1["wfa_WFE"]
        reg_margin    = s1["regime_min_at"] - HARD_VETO_REGIME
        maxdd_margin  = HARD_VETO_MAXDD    - s1["MaxDD_FULL"]
        print("   scale=%.2f: WFE=%.4f (margin to veto: %.4f)  Reg_min=%+.2f%% (margin: %.2fpp)  MaxDD=%+.2f%% (margin: %.2fpp)"
              % (entry["scale"],
                 s1["wfa_WFE"], wfe_margin,
                 s1["regime_min_at"] * 100, reg_margin * 100,
                 s1["MaxDD_FULL"] * 100, maxdd_margin * 100))

    print("\n3. WFA CI95_lo BY SCALE:")
    for entry in s1_results:
        s1 = entry["s1"]
        print("   scale=%.2f: CI95_lo=%+.4f%%  t_p=%.2e  WFE=%.4f"
              % (entry["scale"], s1["wfa_CI95_lo"] * 100, s1["wfa_t_p"], s1["wfa_WFE"]))

    print("\n4. CPCV p10 BY SCALE:")
    for entry in s1_results:
        s1 = entry["s1"]
        print("   scale=%.2f: CPCV p10=%+.4f%%  worst=%+.4f%%  med=%+.4f%%"
              % (entry["scale"],
                 s1["cpcv_p10_at"] * 100, s1["cpcv_worst_at"] * 100, s1["cpcv_med_at"] * 100))

    print("\n5. RECOMMENDATION (CAGR priority / MaxDD<=-50%% tolerable / 5y+10y positive):")
    if pass_entries:
        # Find highest scale that passes
        highest_pass = max(pass_entries, key=lambda e: e["scale"])
        print("   Highest scale passing ALL Stage-1 gates: scale=%.2f"
              % highest_pass["scale"])
        s1h = highest_pass["s1"]
        print("   -> min9=%+.2f%%  MaxDD=%+.2f%%  CI95_lo=%+.2f%%  WFE=%.4f  Regime_min=%+.2f%%"
              % (s1h["min9_at"] * 100, s1h["MaxDD_FULL"] * 100,
                 s1h["wfa_CI95_lo"] * 100, s1h["wfa_WFE"],
                 s1h["regime_min_at"] * 100))
        if abs(highest_pass["scale"] - 1.50) < 0.001:
            print("   CAUTION: scale=1.50 has Worst5Y=-0.02%% (marginally negative).")
            print("            If Worst5Y>=0 is hard requirement, ceiling is scale=1.40.")
        elif abs(highest_pass["scale"] - 1.40) < 0.001:
            print("   scale=1.40: min9=%+.2f%%  MaxDD=%+.2f%%  -- RECOMMENDED CEILING"
                  % (STAGE0_KNOWN[1.40]["min9"] * 100, STAGE0_KNOWN[1.40]["MaxDD"] * 100))
    else:
        print("   No scale passed Stage-1 gates.")

    print("\n6. ONE-LINE VERDICT:")
    if pass_entries:
        highest_sc = max(pass_entries, key=lambda e: e["scale"])["scale"]
        if abs(highest_sc - 1.50) < 0.001:
            print("   scale=1.50 PASSES Stage-1 (MaxDD=-49.3%%, WFE<1.5, W10Y*>0, Reg_min>-10%%);")
            print("   but Worst5Y=-0.02%% is marginally negative -- if 5y positivity is hard, adopt scale=1.40.")
        elif abs(highest_sc - 1.40) < 0.001:
            print("   scale=1.40 is the highest Stage-1 PASS (scale=1.50/1.60 fail).")
            print("   Recommended ceiling: scale=1.40.")
        else:
            print("   scale=%.2f is the highest Stage-1 PASS." % highest_sc)
    else:
        print("   All tested scales fail Stage-1. Do not adopt.")

    # =========================================================================
    # CSV OUTPUT
    # =========================================================================
    print("\nBuilding CSV ...")
    csv_rows = []
    axes_order_keys = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                       "rate:rate_up", "rate:rate_down"]

    for entry in s1_results:
        s1 = entry["s1"]
        s0 = entry["s0"]
        row = {
            "label":           entry["label"],
            "lev_scale":       entry["scale"],
            "v7_map":          "strong",
            # Stage-0 (pre-known)
            "s0_min9_at":      round(s0["min9"],     6),
            "s0_MaxDD_FULL":   round(s0["MaxDD"],    6),
            "s0_W10Y_star_at": round(s0["W10Y_star"],6),
            "s0_Worst5Y_at":   round(s0["Worst5Y"],  6),
            "s0_VETO":         s0["VETO"],
            # Standard 10 (Stage-1 computed)
            "CAGR_IS_at":      round(s1["CAGR_IS_at"],   6),
            "CAGR_OOS_at":     round(s1["CAGR_OOS_at"],  6),
            "min9_at":         round(s1["min9_at"],       6),
            "IS_OOS_gap_pp":   round(s1["IS_OOS_gap_pp"], 4),
            "Sharpe_OOS":      round(s1["Sharpe_OOS"],   4),
            "Sharpe_FULL":     round(s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan"), 4),
            "MaxDD_FULL":      round(s1["MaxDD_FULL"],    6),
            "Worst10Y_at":     round(s1["Worst10Y_at"],  6),
            "P10_5Y_at":       round(s1["P10_5Y_at"],    6),
            "Worst5Y_at":      round(s1["Worst5Y_at"],   6),
            "Trades_yr":       round(s1["Trades_yr"],    2),
            "Worst1D":         round(s1["Worst1D"] if s1["Worst1D"] is not None else float("nan"), 6),
            "Worst1D_date":    s1["Worst1D_date"] if s1["Worst1D_date"] else "",
            # WFA
            "wfa_WFE":         round(s1["wfa_WFE"],      6),
            "wfa_CI95_lo":     round(s1["wfa_CI95_lo"],  6),
            "wfa_t_p":         round(s1["wfa_t_p"],      10),
            # CPCV
            "cpcv_p10_at":     round(s1["cpcv_p10_at"],  6),
            "cpcv_worst_at":   round(s1["cpcv_worst_at"],6),
            "cpcv_med_at":     round(s1["cpcv_med_at"],  6),
            # Regime
            "regime_min_at":   round(s1["regime_min_at"],6),
            # Bootstrap vs V7
            "mm_v7_P_min":     round(float(s1["mm_v7_P_min"]),    4),
            "mm_v7_CI95_min":  round(float(s1["mm_v7_CI95_min"]), 4),
            "mm_v7_P_maxdd":   round(float(s1["mm_v7_P_maxdd"]),  4),
            "mm_v7_CI95_dd":   round(float(s1["mm_v7_CI95_dd"]),  4),
            "mm_v7_P_w10y":    round(float(s1["mm_v7_P_w10y"]),   4),
            "mm_v7_CI95_w10y": round(float(s1["mm_v7_CI95_w10y"]),4),
            "mm_v7_P_sharpe":  round(float(s1["mm_v7_P_sharpe"]), 4),
            "mm_v7_CI95_shp":  round(float(s1["mm_v7_CI95_shp"]), 4),
            # Bootstrap vs B3a
            "mm_b3a_P_min":    round(float(s1["mm_b3a_P_min"]),    4),
            "mm_b3a_CI95_min": round(float(s1["mm_b3a_CI95_min"]), 4),
            "mm_b3a_P_maxdd":  round(float(s1["mm_b3a_P_maxdd"]),  4),
            "mm_b3a_CI95_dd":  round(float(s1["mm_b3a_CI95_dd"]),  4),
            "mm_b3a_P_w10y":   round(float(s1["mm_b3a_P_w10y"]),   4),
            "mm_b3a_CI95_w10y":round(float(s1["mm_b3a_CI95_w10y"]),4),
            "mm_b3a_P_sharpe": round(float(s1["mm_b3a_P_sharpe"]), 4),
            "mm_b3a_CI95_shp": round(float(s1["mm_b3a_CI95_shp"]), 4),
            # Veto
            "s1_veto_maxdd":   s1["s1_veto_maxdd"],
            "s1_veto_wfe":     s1["s1_veto_wfe"],
            "s1_veto_w10y":    s1["s1_veto_w10y"],
            "s1_veto_reg":     s1["s1_veto_reg"],
            "VETO_s1":         s1["VETO_s1"],
        }
        # Regime axis breakdown
        reg = s1["regime"]
        for ax in axes_order_keys:
            row["regime_" + ax.replace(":", "_")] = round(reg.get(ax, float("nan")), 6)
        csv_rows.append(row)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "leverext_high_stage1_20260618.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f",
                                  encoding="utf-8-sig")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(csv_rows)))

    # =========================================================================
    # RETURN_BLOCK JSON
    # =========================================================================
    sanity_result = {
        "min9_got_pct":   round(san_min9 * 100, 4),
        "maxdd_got_pct":  round(san_maxdd * 100, 4),
        "ok_min9":        bool(ok_min9),
        "ok_maxdd":       bool(ok_maxdd),
        "SANITY_PASS":    bool(ok_min9 and ok_maxdd),
        "s1_CI95_lo_pct": round(san_s1["wfa_CI95_lo"] * 100, 4),
        "s1_WFE":         round(san_s1["wfa_WFE"], 4),
        "s1_CI95_ok_soft":bool(ok_ci95),
        "s1_WFE_ok_soft": bool(ok_wfe),
    }

    rb_s1 = []
    for entry in s1_results:
        s1 = entry["s1"]
        s0 = entry["s0"]
        rb_s1.append({
            "label":           entry["label"],
            "lev_scale":       entry["scale"],
            "v7_map":          "strong",
            "min9_at_pct":     round(s1["min9_at"] * 100, 4),
            "MaxDD_pct":       round(s1["MaxDD_FULL"] * 100, 4),
            "Sharpe_OOS":      round(s1["Sharpe_OOS"], 4),
            "Worst10Y_at_pct": round(s1["Worst10Y_at"] * 100, 4),
            "Worst5Y_at_pct":  round(s1["Worst5Y_at"] * 100, 4),
            "P10_5Y_at_pct":   round(s1["P10_5Y_at"] * 100, 4),
            "Trades_yr":       round(s1["Trades_yr"], 2),
            "Worst1D_pct":     round(s1["Worst1D"] * 100 if s1["Worst1D"] else float("nan"), 4),
            "Worst1D_date":    s1["Worst1D_date"],
            "wfa_WFE":         round(s1["wfa_WFE"], 4),
            "wfa_CI95_lo_pct": round(s1["wfa_CI95_lo"] * 100, 4),
            "wfa_t_p":         round(s1["wfa_t_p"], 10),
            "cpcv_p10_at_pct": round(s1["cpcv_p10_at"] * 100, 4),
            "cpcv_worst_at_pct":round(s1["cpcv_worst_at"] * 100, 4),
            "cpcv_med_at_pct": round(s1["cpcv_med_at"] * 100, 4),
            "regime_min_at_pct":round(s1["regime_min_at"] * 100, 4),
            "VETO_s1":         s1["VETO_s1"],
            "s1_veto_maxdd":   s1["s1_veto_maxdd"],
            "s1_veto_wfe":     s1["s1_veto_wfe"],
            "s1_veto_w10y":    s1["s1_veto_w10y"],
            "s1_veto_reg":     s1["s1_veto_reg"],
            "mm_v7_P_min":     round(float(s1["mm_v7_P_min"]), 4),
            "mm_v7_CI95_min_pp":round(float(s1["mm_v7_CI95_min"]), 4),
            "mm_v7_P_maxdd":   round(float(s1["mm_v7_P_maxdd"]), 4),
            "mm_v7_P_w10y":    round(float(s1["mm_v7_P_w10y"]), 4),
            "mm_v7_P_sharpe":  round(float(s1["mm_v7_P_sharpe"]), 4),
            "mm_b3a_P_min":    round(float(s1["mm_b3a_P_min"]), 4),
            "mm_b3a_CI95_min_pp":round(float(s1["mm_b3a_CI95_min"]), 4),
            "mm_b3a_P_maxdd":  round(float(s1["mm_b3a_P_maxdd"]), 4),
            "mm_b3a_P_w10y":   round(float(s1["mm_b3a_P_w10y"]), 4),
            "mm_b3a_P_sharpe": round(float(s1["mm_b3a_P_sharpe"]), 4),
        })

    return_block = {
        "script":   "leverext_high_stage1_20260618.py",
        "date":     "2026-06-18",
        "sanity":   sanity_result,
        "stage1":   rb_s1,
        "csv_path": csv_path,
        "excess_extra_pct": round(EXCESS_EXTRA * 100, 4),
        "n_boot":   N_BOOT,
        "block":    BLOCK,
        "seed":     SEED,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
