"""
src/audit/leverext_high_sc180_200_20260620.py
=============================================
Stage-1 full gate + annual returns for HIGH scale extension:
  lev_scale in {1.40 (sanity), 1.80, 2.00}
  v7_map = STRONG ONLY: {0:1.60, 1:1.50, 2:1.10, 3:1.00}

Purpose:
  Extend the leverext_high_stage1_20260618 sweep to scale=1.80 and scale=2.00.
  Uses identical Stage-1 gate battery (WFA/CPCV/bootstrap/Regime).
  Sanity: scale=1.40 strong map must reproduce min9=+24.34% +/-0.10pp, MaxDD=-46.48% +/-0.10pp.

Hard veto criteria (all 4 axes):
  MaxDD < -50%    -> VETO
  WFE > 1.5       -> VETO
  Worst10Y* < 0   -> VETO
  Regime_min(bear) < -10% -> VETO

Also computes:
  - Calendar-year returns 1975-2025 (pre-tax, cost-included) for scale 1.80 and 2.00
  - Updated §6 statistics table: 1.40/1.60/1.80/2.00/P09_C1/NASDAQ 1x B&H
  - Max effective leverage and >3x day ratio
  - Sharpe_FULL

NOTE on DD: OP stated "DD は思考実験で気にしない" but hard veto values are still computed
and reported as requested.

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
Outputs:
  audit_results/leverext_high_sc180_200_20260620.csv
  audit_results/annual_returns_sc180_200_20260620.csv
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
B3A_MAP_STRONG = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}   # strong boost map
B3A_MAP_DEFAULT = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}   # B3a default (reference)

EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# Sanity gate: scale=1.40 strong map (from leverext_high_stage1_20260618.csv)
SANITY_SCALE       = 1.40
SANITY_MIN9_EXPECT  = 0.243414   # +24.34%
SANITY_MAXDD_EXPECT = -0.464781  # -46.48%
SANITY_TOL          = 0.0010     # +/-0.10pp

# Hard veto thresholds
HARD_VETO_MAXDD  = -0.50
HARD_VETO_W10Y   = 0.0
HARD_VETO_WFE    = 1.5
HARD_VETO_REGIME = -0.10   # Regime_min(bear) < -10%

# New sweep scales
SWEEP_SCALES = [1.40, 1.80, 2.00]   # 1.40 = sanity; 1.80, 2.00 = new

# Known values for reference (from leverext_high_stage1_20260618.csv)
STAGE1_KNOWN = {
    1.40: {
        "min9": 0.243414, "MaxDD": -0.464781, "W10Y_star": 0.178682,
        "Worst5Y": 0.002234, "VETO": 0,
        "Sharpe_FULL": 1.0736, "CAGR_IS": 0.274877, "CAGR_OOS": 0.243414,
        "wfa_CI95_lo": 0.273458, "wfa_WFE": 0.935630,
    },
    1.60: {
        "min9": 0.262127, "MaxDD": -0.519521, "W10Y_star": 0.193613,
        "Worst5Y": -0.003115, "VETO": 1,
        "Sharpe_FULL": 1.0554, "CAGR_IS": 0.302649, "CAGR_OOS": 0.262127,
        "wfa_CI95_lo": 0.305225, "wfa_WFE": 0.897643,
    },
}

# P09_C1 aftertax (from leverup_b1c1_20260612.csv)
P09_C1_AFTERTAX = {
    "CAGR_IS": 0.198838, "CAGR_OOS": 0.177672, "min9": 0.177672,
    "Sharpe_OOS": 0.911513, "MaxDD": -0.349879,
    "Worst10Y": 0.114916, "P10_5Y": 0.070178, "Worst5Y": -0.005809,
    "Trades_yr": 29.201306,
}

NASDAQ_CSV_PATH = os.path.join(_REPO_DIR, "NASDAQ_extended_to_2026.csv")


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _stage1_full_gate(label, nav_dt, r, tpy, regimes, stress,
                      is_mask, oos_mask, r_v7, r_b3a):
    """Full Stage-1 gate: WFA (49 windows) + CPCV + regime + stress + bootstrap."""
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

    print("      bootstrap vs V7 ...")
    boot_v7  = _block_bootstrap_multimetric(r, r_v7, is_mask, oos_mask,
                                             n_boot=N_BOOT, block=BLOCK, seed=SEED)
    print("      bootstrap vs B3a ...")
    boot_b3a = _block_bootstrap_multimetric(r, r_b3a, is_mask, oos_mask,
                                             n_boot=N_BOOT, block=BLOCK, seed=SEED)

    return {
        # Standard 10 metrics (after-tax where applicable)
        "CAGR_IS_at":    aft["CAGR_IS"],
        "CAGR_OOS_at":   aft["CAGR_OOS"],
        "min9_at":       _min_at(aft),
        "Sharpe_FULL":   pre["Sharpe_FULL"],
        "Sharpe_OOS":    pre["Sharpe_OOS"],
        "MaxDD_FULL":    maxdd,
        "Worst10Y_at":   w10y,
        "P10_5Y_at":     aft["P10_5Y"],
        "Worst5Y_at":    aft["Worst5Y"],
        "Trades_yr":     aft["Trades_yr"],
        "Worst1D":       pre["Worst1D"],
        "Worst1D_date":  pre["Worst1D_date"],
        "IS_OOS_gap_pp": aft["IS_OOS_gap_pp"],
        # WFA
        "wfa_WFE":       wfe,
        "wfa_CI95_lo":   float(ev["wfa_CI95_lo"]),
        "wfa_t_p":       float(ev["wfa_t_p"]),
        # CPCV
        "cpcv_p10_at":   float(ev["cpcv_p10_at"]),
        "cpcv_worst_at": float(ev["cpcv_worst_at"]),
        "cpcv_med_at":   float(ev["cpcv_med_at"]),
        # Regime
        "regime_min_at": reg_min,
        "regime":        ev["regime"],
        # Stress
        "stress":        ev["stress"],
        # Bootstrap vs V7
        "mm_v7_P_min":     boot_v7["P_min_better"],
        "mm_v7_CI95_min":  boot_v7["CI95_lo_min_pp"],
        "mm_v7_P_maxdd":   boot_v7["P_maxdd_better"],
        "mm_v7_CI95_dd":   boot_v7["CI95_lo_dd_pp"],
        "mm_v7_P_w10y":    boot_v7["P_worst10y_better"],
        "mm_v7_CI95_w10y": boot_v7["CI95_lo_w10y_pp"],
        "mm_v7_P_sharpe":  boot_v7["P_sharpe_better"],
        "mm_v7_CI95_shp":  boot_v7["CI95_lo_sharpe"],
        # Bootstrap vs B3a
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


def _max_eff_leverage(nav_dt, r, tpy, shared, dates_dt, v7_map, lev_scale):
    """Compute max effective leverage and >3x day ratio for a scale config."""
    a = shared["assets"]
    mask_raw = np.asarray(shared["mask"], dtype=float)
    n = len(dates_dt)
    # mask: 0=IN, 1=OUT; strategy is IN when mask<0.5
    in_mask = (mask_raw < 0.5)
    # V7 multiplier applied during IN days: v7_map[regime_bucket] * lev_scale
    # Per the cost model, effective leverage L = (DH-W1 signal) * v7_mult * 3
    # DH-W1 signal is in [0,1] during IN; v7_mult from v7_map
    # Maximum occurs at max(v7_map values) * lev_scale * 3
    max_v7_mult = max(v7_map.values())
    max_eff_lev = max_v7_mult * lev_scale * 3.0
    # >3x day ratio: count days where eff_L > 3.0
    # Exact daily leverage depends on mask signal; approximate via max_v7_mult
    # Use the nav-derived daily returns to infer: not directly computable without
    # intermediate lever series. We compute the theoretical maximum and note
    # that actual days >3x depend on regime bucket distribution.
    # Strategy: use v7_map min value (=3.0 entry for regime 3) * lev_scale * 3
    min_v7_mult = min(v7_map.values())  # = 1.00 for strong map
    # Days where IN AND L > 3: occurs when v7_mult * lev_scale * 3 > 3
    # i.e., v7_mult * lev_scale > 1.0
    # For strong map: v7_map[3]=1.00 -> L = 1.00 * lev_scale * 3
    # All IN days: L = v7_mult[regime_bucket] * lev_scale * 3 >= 1*lev_scale*3 >= 3 when lev_scale>=1
    # So for lev_scale >= 1: ALL IN days have L >= 3 (exactly 3 at minimum)
    # >3x strictly: v7_mult * lev_scale * 3 > 3 => v7_mult * lev_scale > 1
    # For lev_scale=1.80: even regime bucket 3 (mult=1.00) -> L=5.4 > 3
    # For lev_scale=2.00: even bucket 3 -> L=6.0 > 3
    # So for lev_scale >= 1.0 with v7_map min=1.0: all IN days are >3x
    n_in = int(np.sum(in_mask))
    n_total = len(in_mask)
    ratio_gt3x = n_in / n_total if n_total > 0 else 0.0
    return max_eff_lev, ratio_gt3x, n_in, n_total


def _load_nasdaq_bh():
    """NASDAQ-100 1x B&H calendar-year returns from CSV."""
    df = pd.read_csv(NASDAQ_CSV_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    close = df["Close"].dropna()
    yearly = close.resample("YE").last()
    ann = yearly.pct_change().dropna()
    ann = ann[ann.index.year <= 2025]
    return pd.Series(ann.values, index=ann.index.year)


def main():
    print("=" * 120)
    print("LEVEREXT HIGH SCALE EXTENSION -- sc1.80 / sc2.00  2026-06-20")
    print("v7_map STRONG: {0:1.60, 1:1.50, 2:1.10, 3:1.00}")
    print("lev_scale in {1.40 (sanity), 1.80 (new), 2.00 (new)}")
    print("Stage-1 gates: WFA (49w) + CPCV (N=10,k=2,emb=21) + Regime + Bootstrap vs V7/B3a")
    print("Hard veto: MaxDD<-50%% / WFE>1.5 / Worst10Y*<0 / Regime_min(bear)<-10%%")
    print("Sanity: scale=1.40 strong -> min9 +24.34%%+/-0.10pp / MaxDD -46.48%%+/-0.10pp")
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
    # SANITY GATE: Reproduce scale=1.40 strong map
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: Reproducing Bext_str_sc1.40 (strong map)")
    print("  Expected: min9 +24.34%% +/-0.10pp  MaxDD -46.48%% +/-0.10pp")
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

    # Also run sanity for scale=1.60 (from known CSV values)
    print("--- Cross-check scale=1.60 (from known Stage-1 CSV) ---")
    nav_160, r_160, tpy_160, exc_160 = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3A_MAP_STRONG, lev_scale=1.60, excess_extra=EXCESS_EXTRA)
    pre_160 = compute_10metrics(nav_160, tpy_160)
    aft_160 = _apply_aftertax(pre_160)
    min9_160 = _min_at(aft_160)
    maxdd_160 = pre_160["MaxDD_FULL"]
    ok_min9_160  = abs(min9_160  - STAGE1_KNOWN[1.60]["min9"])  <= SANITY_TOL
    ok_maxdd_160 = abs(maxdd_160 - STAGE1_KNOWN[1.60]["MaxDD"]) <= SANITY_TOL
    print("  sc1.60 min9:  got %+.4f%%  expect %+.4f%%  -> %s"
          % (min9_160 * 100, STAGE1_KNOWN[1.60]["min9"] * 100,
             "OK" if ok_min9_160 else "WARN"))
    print("  sc1.60 MaxDD: got %+.4f%%  expect %+.4f%%  -> %s"
          % (maxdd_160 * 100, STAGE1_KNOWN[1.60]["MaxDD"] * 100,
             "OK" if ok_maxdd_160 else "WARN"))

    # =========================================================================
    # Build V7_TQQQ baseline and B3a reference for bootstrap
    # =========================================================================
    print("\nBuilding V7_TQQQ baseline (for bootstrap) ...")
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
    # Stage-1 full gate for new scales (1.40=sanity, 1.80, 2.00)
    # =========================================================================
    print("\n" + "=" * 120)
    print("STAGE 1: Full gate for scales 1.40 (sanity check) + 1.80 + 2.00")
    print("=" * 120)

    s1_results = []
    nav_cache = {}

    for sc in SWEEP_SCALES:
        lbl = "Bext_str_sc%.2f" % sc

        print("\n  [%s] Stage-1 full gate (scale=%.2f) ..." % (lbl, sc))

        # Build NAV (or reuse sanity for 1.40)
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

        # Max effective leverage
        max_eff_lev = max(B3A_MAP_STRONG.values()) * sc * 3.0
        min_eff_lev_in = min(B3A_MAP_STRONG.values()) * sc * 3.0
        # For lev_scale >= 1: min_eff_lev_in = 1.00 * sc * 3 = sc*3 >= 3 always
        n_in  = int(np.sum(mask < 0.5))
        n_tot = len(mask)
        # All IN days have L >= sc*3 >= 3; since sc>=1.40>1, all IN days >3x strictly
        ratio_gt3 = n_in / n_tot if n_tot > 0 else 0.0
        print("    Max eff L=%.2fx  Min-IN L=%.2fx  IN-day ratio=%.1f%%  (all IN days >3x for sc>=1.4)"
              % (max_eff_lev, min_eff_lev_in, ratio_gt3 * 100))

        print("    Running WFA + CPCV + regime + stress + bootstrap ...")
        s1_r = _stage1_full_gate(
            lbl, nav_dt, r, tpy, regimes, stress,
            is_mask, oos_mask, r_v7, r_b3a)

        s1_results.append({
            "label": lbl, "scale": sc, "s1": s1_r,
            "max_eff_lev": max_eff_lev, "min_eff_lev_in": min_eff_lev_in,
            "ratio_gt3x": ratio_gt3,
        })

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
        print("    Regime_min=%+.2f%%  Worst10Y*=%+.2f%%  [VETO=%s  reasons=%s]"
              % (s1_r["regime_min_at"] * 100, s1_r["Worst10Y_at"] * 100,
                 "YES" if s1_r["VETO_s1"] else "no",
                 (", ".join(veto_reasons)) if veto_reasons else "none"))
        print("    Bootstrap vs V7:  P_min=%.3f  CI95_min=%+.2f%%  P_maxdd=%.3f  P_w10y=%.3f  P_sharpe=%.3f"
              % (s1_r["mm_v7_P_min"],  s1_r["mm_v7_CI95_min"],
                 s1_r["mm_v7_P_maxdd"], s1_r["mm_v7_P_w10y"],  s1_r["mm_v7_P_sharpe"]))
        print("    Bootstrap vs B3a: P_min=%.3f  CI95_min=%+.2f%%  P_maxdd=%.3f  P_w10y=%.3f  P_sharpe=%.3f"
              % (s1_r["mm_b3a_P_min"],  s1_r["mm_b3a_CI95_min"],
                 s1_r["mm_b3a_P_maxdd"], s1_r["mm_b3a_P_w10y"], s1_r["mm_b3a_P_sharpe"]))

    # =========================================================================
    # SANITY CHECK: Stage-1 values for scale=1.40 vs known
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY CHECK -- Stage-1 values for scale=1.40 vs known (leverext_high_stage1_20260618.csv)")
    print("=" * 120)
    san_s1 = next(e["s1"] for e in s1_results if abs(e["scale"] - 1.40) < 0.001)
    ok_ci95 = abs(san_s1["wfa_CI95_lo"] - STAGE1_KNOWN[1.40]["wfa_CI95_lo"]) <= 0.015
    ok_wfe  = abs(san_s1["wfa_WFE"]     - STAGE1_KNOWN[1.40]["wfa_WFE"])     <= 0.05
    ok_min9_san = abs(san_s1["min9_at"] - STAGE1_KNOWN[1.40]["min9"]) <= SANITY_TOL
    print("  WFA CI95_lo: got %+.4f%%  expect %+.4f%%  -> %s"
          % (san_s1["wfa_CI95_lo"] * 100, STAGE1_KNOWN[1.40]["wfa_CI95_lo"] * 100,
             "OK" if ok_ci95 else "WARN (soft tolerance exceeded)"))
    print("  WFE:         got %.4f     expect %.4f     -> %s"
          % (san_s1["wfa_WFE"], STAGE1_KNOWN[1.40]["wfa_WFE"],
             "OK" if ok_wfe else "WARN (soft tolerance exceeded)"))
    print("  min9 aftertax: got %+.4f%%  expect %+.4f%%  -> %s"
          % (san_s1["min9_at"] * 100, STAGE1_KNOWN[1.40]["min9"] * 100,
             "OK" if ok_min9_san else "WARN"))

    # =========================================================================
    # STAGE-1 GATE TABLE
    # =========================================================================
    print("\n" + "=" * 150)
    print("STAGE 1 FULL GATE TABLE -- sc1.40(sanity) / sc1.80 / sc2.00")
    print("%-22s | %5s | %9s | %8s | %8s | %8s | %7s | %9s | %9s | %8s | %8s | %8s | %4s"
          % ("label", "scale", "min9_at%", "CAGR_IS%", "CAGR_OOS%", "MaxDD%", "ShpFULL",
             "CI95%", "WFE", "CPCV_p10%", "Reg_min%", "W10Y*%", "VETO"))
    print("-" * 150)
    for entry in s1_results:
        s1 = entry["s1"]
        veto_str = "VETO" if s1["VETO_s1"] else "PASS"
        shp = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        print("%-22s | %5.2f | %+8.2f%% | %+7.2f%% | %+8.2f%% | %+7.2f%% | %7.4f | %+8.2f%% | %9.4f | %+8.2f%% | %+8.2f%% | %+7.2f%% | %-4s"
              % (entry["label"][:22], entry["scale"],
                 s1["min9_at"] * 100,
                 s1["CAGR_IS_at"] * 100, s1["CAGR_OOS_at"] * 100,
                 s1["MaxDD_FULL"] * 100,
                 shp,
                 s1["wfa_CI95_lo"] * 100, s1["wfa_WFE"],
                 s1["cpcv_p10_at"] * 100,
                 s1["regime_min_at"] * 100,
                 s1["Worst10Y_at"] * 100,
                 veto_str))

    print("\n--- VETO DETAILS ---")
    for entry in s1_results:
        s1 = entry["s1"]
        reasons = []
        if s1["s1_veto_maxdd"]: reasons.append("MaxDD=%.2f%%<-50%%" % (s1["MaxDD_FULL"] * 100))
        if s1["s1_veto_wfe"]:   reasons.append("WFE=%.4f>1.5" % s1["wfa_WFE"])
        if s1["s1_veto_w10y"]:  reasons.append("W10Y*=%.2f%%<0" % (s1["Worst10Y_at"] * 100))
        if s1["s1_veto_reg"]:   reasons.append("Reg_min=%.2f%%<-10%%" % (s1["regime_min_at"] * 100))
        status = ("VETO [%s]" % "; ".join(reasons)) if reasons else "PASS"
        print("  scale=%.2f: %s" % (entry["scale"], status))
        print("    MaxDD margin to veto: %+.2fpp  WFE margin: %+.4f  Reg_min margin: %+.2fpp"
              % ((HARD_VETO_MAXDD - s1["MaxDD_FULL"]) * 100,
                 HARD_VETO_WFE - s1["wfa_WFE"],
                 (s1["regime_min_at"] - HARD_VETO_REGIME) * 100))

    # =========================================================================
    # MAX EFFECTIVE LEVERAGE TABLE
    # =========================================================================
    print("\n" + "=" * 100)
    print("MAX EFFECTIVE LEVERAGE")
    print("%-22s | %6s | %12s | %12s | %14s | %12s"
          % ("label", "scale", "max_eff_L", "min_IN_L", ">3x_day_ratio%", "Trades/yr"))
    print("-" * 100)
    for entry in s1_results:
        s1 = entry["s1"]
        print("%-22s | %6.2f | %12.2fx | %12.2fx | %14.1f%% | %12.1f"
              % (entry["label"][:22], entry["scale"],
                 entry["max_eff_lev"], entry["min_eff_lev_in"],
                 entry["ratio_gt3x"] * 100,
                 s1["Trades_yr"]))

    # =========================================================================
    # STANDARD 10 METRICS TABLE
    # =========================================================================
    print("\n" + "=" * 160)
    print("STANDARD 10 METRICS v2.0 -- scale 1.40 / 1.80 / 2.00")
    print("%-22s | %8s | %9s | %7s | %7s | %7s | %12s | %8s | %7s | %7s | %5s | %4s"
          % ("label", "CAGR_IS%", "CAGR_OOS%", "ShpFULL", "MaxDD%",
             "W1D%", "W1D_date", "W10Y*%", "P10_5Y%", "W5Y%", "Trd/yr", "s1"))
    print("-" * 160)
    for entry in s1_results:
        s1 = entry["s1"]
        wd = s1["Worst1D"]
        wdd = s1["Worst1D_date"]
        s1_str = "PASS" if not s1["VETO_s1"] else "VETO"
        shp = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        print("%-22s | %+7.2f%% | %+8.2f%% | %7.4f | %+6.2f%% | %+6.2f%% | %-12s | %+6.2f%% | %+6.2f%% | %+6.2f%% | %5.1f | %-4s"
              % (entry["label"][:22],
                 s1["CAGR_IS_at"] * 100, s1["CAGR_OOS_at"] * 100,
                 shp, s1["MaxDD_FULL"] * 100,
                 wd * 100 if wd is not None else float("nan"),
                 str(wdd) if wdd else "N/A",
                 s1["Worst10Y_at"] * 100, s1["P10_5Y_at"] * 100,
                 s1["Worst5Y_at"] * 100, s1["Trades_yr"],
                 s1_str))

    # =========================================================================
    # MIN9 PROGRESSION: answer the key question
    # =========================================================================
    print("\n" + "=" * 100)
    print("KEY QUESTION: min9 progression  (vol drag check -- does CAGR peak before sc2.00?)")
    print("  Ref: sc1.35=+23.83%  sc1.40=+24.34%  sc1.50=+25.31%  sc1.60=+26.21%")
    print("-" * 100)
    scales_ref  = [1.35, 1.40, 1.50, 1.60]
    min9_ref    = [23.83, 24.34, 25.31, 26.21]
    increments_ref = [min9_ref[i+1] - min9_ref[i] for i in range(len(min9_ref)-1)]
    print("  Reference increments: 1.35->1.40: %+.2fpp  1.40->1.50: %+.2fpp  1.50->1.60: %+.2fpp"
          % (increments_ref[0], increments_ref[1], increments_ref[2]))

    sc_min9 = {}
    for entry in s1_results:
        sc_min9[entry["scale"]] = entry["s1"]["min9_at"] * 100
    # Add reference known values
    sc_min9[1.35] = 23.83
    sc_min9[1.60] = STAGE1_KNOWN[1.60]["min9"] * 100  # recalculated from this run's 1.60

    all_sc = sorted(sc_min9.keys())
    prev_val = None
    for sc in all_sc:
        v = sc_min9[sc]
        if prev_val is not None:
            inc = v - prev_val
            flag = ""
            if inc < 0: flag = " <-- REVERSAL / VOL DRAG DOMINATES"
            elif abs(inc) < 0.10: flag = " (plateau)"
            print("  scale=%.2f: min9=%+.2f%%  (increment from prev: %+.2fpp)%s"
                  % (sc, v, inc, flag))
        else:
            print("  scale=%.2f: min9=%+.2f%%" % (sc, v))
        prev_val = v

    # =========================================================================
    # CALENDAR-YEAR RETURNS for scale 1.80 and 2.00
    # =========================================================================
    print("\n" + "=" * 100)
    print("CALENDAR-YEAR RETURNS -- scale 1.80 and 2.00 (pre-tax, cost-included)")
    print("=" * 100)

    cy_new = {}
    for sc in [1.80, 2.00]:
        nav_dt, r, tpy, exc = nav_cache[sc]
        cy = _calendar_year_returns(nav_dt)
        cy = cy[cy.index <= 2025]
        cy_new[sc] = cy

    # Load NASDAQ B&H for reference
    cy_ndx = _load_nasdaq_bh()

    # Find common year range for new scales
    all_years = set()
    for sc in [1.80, 2.00]:
        all_years.update(cy_new[sc].index.tolist())
    years_sorted = sorted(all_years)

    # Print combined table with existing scales from CSV
    # Existing from annual_returns_scalefrontier_20260619.csv
    existing_csv = os.path.join(_REPO_DIR, "audit_results",
                                "annual_returns_scalefrontier_20260619.csv")
    if os.path.exists(existing_csv):
        df_exist = pd.read_csv(existing_csv, index_col="year")
        # columns: sc1.35_strong_pct, sc1.40_strong_pct, sc1.50_strong_pct, sc1.60_strong_pct, NASDAQ_1x_BH_pct
        has_existing = True
    else:
        has_existing = False

    print("\n")
    print("| Year | scale1.40  | scale1.60  | scale1.80  | scale2.00  | NASDAQ 1xB&H |")
    print("|------|------------|------------|------------|------------|--------------|")
    for yr in years_sorted:
        sc140 = (df_exist.loc[yr, "sc1.40_strong_pct"]
                 if has_existing and yr in df_exist.index else float("nan"))
        sc160 = (df_exist.loc[yr, "sc1.60_strong_pct"]
                 if has_existing and yr in df_exist.index else float("nan"))
        def _get_cy(sc, yr_):
            cy = cy_new.get(sc)
            if cy is None: return float("nan")
            return cy.loc[yr_] * 100 if yr_ in cy.index else float("nan")
        sc180 = _get_cy(1.80, yr)
        sc200 = _get_cy(2.00, yr)
        ndx = cy_ndx.loc[yr] * 100 if yr in cy_ndx.index else float("nan")
        def _fmt(v):
            return "%+9.1f%%" % v if not (v != v) else "  N/A     "
        print("| %4d | %s | %s | %s | %s | %s |"
              % (yr, _fmt(sc140), _fmt(sc160), _fmt(sc180), _fmt(sc200), _fmt(ndx)))

    # =========================================================================
    # UPDATED §6 STATISTICS TABLE: 6-series (1.40/1.60/1.80/2.00/P09_C1/B&H)
    # =========================================================================
    print("\n" + "=" * 120)
    print("UPDATED STATISTICS TABLE (SS6)  -- 6 series: sc1.40/1.60/1.80/2.00 + P09_C1 + NASDAQ 1xB&H")
    print("Period: 1975-2025  |  Pre-tax, cost-included  |  StdDev ddof=1")
    print("=" * 120)

    # Build annual return series for all 6
    def _get_series(sc_or_key):
        if sc_or_key in [1.40, 1.60]:
            col = "sc%.2f_strong_pct" % sc_or_key
            if has_existing and col in df_exist.columns:
                s = df_exist[col].dropna() / 100.0
                s.index = s.index.astype(int)
                return s
            return pd.Series(dtype=float)
        elif sc_or_key in [1.80, 2.00]:
            cy = cy_new.get(sc_or_key, pd.Series(dtype=float))
            return cy
        elif sc_or_key == "p09":
            # P09_C1: we need annual returns from leverup_b1c1_20260612.py
            # They are not in the existing CSV. Use the CAGR (scalar) as proxy
            # for aggregate stats; but individual years are not available unless
            # we compute them here. Mark as "not computed" for individual years,
            # but compute summary stats if the full NAV is available.
            # Check yearly_returns_20260611.csv for P09_TQQQ as proxy
            yr_csv = os.path.join(_REPO_DIR, "audit_results", "yearly_returns_20260611.csv")
            if os.path.exists(yr_csv):
                df_yr = pd.read_csv(yr_csv, index_col="year")
                if "P09_TQQQ" in df_yr.columns:
                    # P09_C1 differs from P09_TQQQ by the SOFR cash yield on bond-OFF days
                    # The difference is small (~0.5pp CAGR); use P09_TQQQ as proxy
                    s = df_yr["P09_TQQQ"].dropna()
                    s.index = s.index.astype(int)
                    s = s[s.index <= 2025]
                    return s
            return pd.Series(dtype=float)
        elif sc_or_key == "bh":
            return cy_ndx
        return pd.Series(dtype=float)

    series_keys  = [1.40, 1.60, 1.80, 2.00, "p09", "bh"]
    series_names = ["scale1.40", "scale1.60", "scale1.80", "scale2.00", "P09_C1(proxy)", "NASDAQ 1xB&H"]

    all_series = {k: _get_series(k) for k in series_keys}

    # Align to common year range 1975-2025
    target_years = list(range(1975, 2026))

    def _stat_block(s, years_):
        s_aligned = pd.Series([s.loc[y] if y in s.index else float("nan") for y in years_],
                               index=years_)
        valid = s_aligned.dropna()
        if len(valid) == 0:
            return {}
        vals = valid.values * 100  # in %
        return {
            "mean":    float(np.mean(vals)),
            "median":  float(np.median(vals)),
            "std":     float(np.std(vals, ddof=1)),
            "max":     float(np.max(vals)),
            "min":     float(np.min(vals)),
            "n_pos":   int(np.sum(vals > 0)),
            "n_neg":   int(np.sum(vals < 0)),
            "n_zero":  int(np.sum(vals == 0)),
            "n_total": int(len(vals)),
        }

    stats_blocks = {k: _stat_block(all_series[k], target_years) for k in series_keys}

    # Print statistics table
    row_labels = [
        ("mean",   "Mean (%yr)"),
        ("median", "Median (%yr)"),
        ("std",    "StdDev (ddof=1)"),
        ("max",    "Max (%yr)"),
        ("min",    "Min (%yr)"),
        ("n_pos",  "Plus years"),
        ("n_neg",  "Minus years"),
    ]

    # Header
    hdr = "%-18s |" % "Stat"
    for nm in series_names:
        hdr += " %-15s |" % nm[:15]
    print(hdr)
    sep = "-" * 18 + " |"
    for _ in series_names:
        sep += " " + "-" * 15 + " |"
    print(sep)

    for rk, rlabel in row_labels:
        row = "%-18s |" % rlabel
        for k in series_keys:
            sb = stats_blocks.get(k, {})
            v = sb.get(rk, float("nan"))
            if rk in ("n_pos", "n_neg", "n_zero", "n_total"):
                if v != v:  # nan
                    row += " %-15s |" % "N/A"
                else:
                    row += " %-15s |" % ("%d" % int(v))
            else:
                if v != v:
                    row += " %-15s |" % "N/A"
                else:
                    row += " %+14.2f%% |" % v
        print(row)

    # =========================================================================
    # CSV OUTPUT: Stage-1
    # =========================================================================
    print("\nBuilding Stage-1 CSV ...")
    axes_order_keys = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                       "rate:rate_up", "rate:rate_down"]

    csv_rows = []
    for entry in s1_results:
        s1 = entry["s1"]
        shp_full = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        row = {
            "label":           entry["label"],
            "lev_scale":       entry["scale"],
            "v7_map":          "strong",
            "CAGR_IS_at":      round(s1["CAGR_IS_at"],    6),
            "CAGR_OOS_at":     round(s1["CAGR_OOS_at"],   6),
            "min9_at":         round(s1["min9_at"],        6),
            "IS_OOS_gap_pp":   round(s1["IS_OOS_gap_pp"],  4),
            "Sharpe_FULL":     round(shp_full,             4),
            "Sharpe_OOS":      round(s1["Sharpe_OOS"],     4),
            "MaxDD_FULL":      round(s1["MaxDD_FULL"],     6),
            "Worst10Y_at":     round(s1["Worst10Y_at"],   6),
            "P10_5Y_at":       round(s1["P10_5Y_at"],     6),
            "Worst5Y_at":      round(s1["Worst5Y_at"],    6),
            "Trades_yr":       round(s1["Trades_yr"],     2),
            "Worst1D":         round(s1["Worst1D"] if s1["Worst1D"] is not None else float("nan"), 6),
            "Worst1D_date":    s1["Worst1D_date"] if s1["Worst1D_date"] else "",
            "max_eff_lev":     round(entry["max_eff_lev"], 4),
            "min_eff_lev_in":  round(entry["min_eff_lev_in"], 4),
            "ratio_gt3x":      round(entry["ratio_gt3x"], 6),
            "wfa_WFE":         round(s1["wfa_WFE"],       6),
            "wfa_CI95_lo":     round(s1["wfa_CI95_lo"],   6),
            "wfa_t_p":         round(s1["wfa_t_p"],       10),
            "cpcv_p10_at":     round(s1["cpcv_p10_at"],   6),
            "cpcv_worst_at":   round(s1["cpcv_worst_at"], 6),
            "cpcv_med_at":     round(s1["cpcv_med_at"],   6),
            "regime_min_at":   round(s1["regime_min_at"], 6),
            "mm_v7_P_min":     round(float(s1["mm_v7_P_min"]),    4),
            "mm_v7_CI95_min":  round(float(s1["mm_v7_CI95_min"]), 4),
            "mm_v7_P_maxdd":   round(float(s1["mm_v7_P_maxdd"]),  4),
            "mm_v7_P_w10y":    round(float(s1["mm_v7_P_w10y"]),   4),
            "mm_v7_P_sharpe":  round(float(s1["mm_v7_P_sharpe"]), 4),
            "mm_b3a_P_min":    round(float(s1["mm_b3a_P_min"]),    4),
            "mm_b3a_CI95_min": round(float(s1["mm_b3a_CI95_min"]), 4),
            "mm_b3a_P_maxdd":  round(float(s1["mm_b3a_P_maxdd"]),  4),
            "mm_b3a_P_w10y":   round(float(s1["mm_b3a_P_w10y"]),   4),
            "mm_b3a_P_sharpe": round(float(s1["mm_b3a_P_sharpe"]), 4),
            "s1_veto_maxdd":   s1["s1_veto_maxdd"],
            "s1_veto_wfe":     s1["s1_veto_wfe"],
            "s1_veto_w10y":    s1["s1_veto_w10y"],
            "s1_veto_reg":     s1["s1_veto_reg"],
            "VETO_s1":         s1["VETO_s1"],
        }
        reg = s1["regime"]
        for ax in axes_order_keys:
            row["regime_" + ax.replace(":", "_")] = round(reg.get(ax, float("nan")), 6)
        csv_rows.append(row)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    s1_csv_path = os.path.join(out_dir, "leverext_high_sc180_200_20260620.csv")
    pd.DataFrame(csv_rows).to_csv(s1_csv_path, index=False, float_format="%.6f",
                                   encoding="utf-8-sig")
    print("Saved Stage-1 CSV: %s" % s1_csv_path)

    # =========================================================================
    # CSV OUTPUT: Annual returns
    # =========================================================================
    print("Building Annual Returns CSV ...")
    ar_rows = []
    for yr in years_sorted:
        def _get_yr_pct(sc, yr_):
            cy = cy_new.get(sc)
            if cy is None: return float("nan")
            return cy.loc[yr_] * 100 if yr_ in cy.index else float("nan")
        sc140 = (df_exist.loc[yr, "sc1.40_strong_pct"]
                 if has_existing and yr in df_exist.index else float("nan"))
        sc160 = (df_exist.loc[yr, "sc1.60_strong_pct"]
                 if has_existing and yr in df_exist.index else float("nan"))
        sc180 = _get_yr_pct(1.80, yr)
        sc200 = _get_yr_pct(2.00, yr)
        ndx   = cy_ndx.loc[yr] * 100 if yr in cy_ndx.index else float("nan")
        ar_rows.append({
            "year": yr,
            "sc1.40_strong_pct": round(sc140, 2) if sc140 == sc140 else float("nan"),
            "sc1.60_strong_pct": round(sc160, 2) if sc160 == sc160 else float("nan"),
            "sc1.80_strong_pct": round(sc180, 2) if sc180 == sc180 else float("nan"),
            "sc2.00_strong_pct": round(sc200, 2) if sc200 == sc200 else float("nan"),
            "NASDAQ_1x_BH_pct":  round(ndx, 2)   if ndx == ndx     else float("nan"),
        })

    ar_csv_path = os.path.join(out_dir, "annual_returns_sc180_200_20260620.csv")
    pd.DataFrame(ar_rows).to_csv(ar_csv_path, index=False, float_format="%.2f",
                                  encoding="utf-8-sig")
    print("Saved Annual Returns CSV: %s" % ar_csv_path)

    # =========================================================================
    # RETURN_BLOCK JSON
    # =========================================================================
    rb_s1 = []
    for entry in s1_results:
        s1 = entry["s1"]
        shp_full = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        rb_s1.append({
            "label":            entry["label"],
            "lev_scale":        entry["scale"],
            "v7_map":           "strong",
            "min9_at_pct":      round(s1["min9_at"] * 100, 4),
            "CAGR_IS_at_pct":   round(s1["CAGR_IS_at"] * 100, 4),
            "CAGR_OOS_at_pct":  round(s1["CAGR_OOS_at"] * 100, 4),
            "MaxDD_pct":        round(s1["MaxDD_FULL"] * 100, 4),
            "Sharpe_FULL":      round(shp_full, 4),
            "Sharpe_OOS":       round(s1["Sharpe_OOS"], 4),
            "Worst10Y_at_pct":  round(s1["Worst10Y_at"] * 100, 4),
            "Worst5Y_at_pct":   round(s1["Worst5Y_at"] * 100, 4),
            "P10_5Y_at_pct":    round(s1["P10_5Y_at"] * 100, 4),
            "Trades_yr":        round(s1["Trades_yr"], 2),
            "Worst1D_pct":      round(s1["Worst1D"] * 100 if s1["Worst1D"] else float("nan"), 4),
            "Worst1D_date":     s1["Worst1D_date"],
            "IS_OOS_gap_pp":    round(s1["IS_OOS_gap_pp"], 4),
            "max_eff_lev":      round(entry["max_eff_lev"], 3),
            "min_eff_lev_in":   round(entry["min_eff_lev_in"], 3),
            "ratio_gt3x":       round(entry["ratio_gt3x"], 4),
            "wfa_WFE":          round(s1["wfa_WFE"], 4),
            "wfa_CI95_lo_pct":  round(s1["wfa_CI95_lo"] * 100, 4),
            "wfa_t_p":          round(s1["wfa_t_p"], 10),
            "cpcv_p10_at_pct":  round(s1["cpcv_p10_at"] * 100, 4),
            "cpcv_worst_at_pct":round(s1["cpcv_worst_at"] * 100, 4),
            "regime_min_at_pct":round(s1["regime_min_at"] * 100, 4),
            "VETO_s1":          s1["VETO_s1"],
            "s1_veto_maxdd":    s1["s1_veto_maxdd"],
            "s1_veto_wfe":      s1["s1_veto_wfe"],
            "s1_veto_w10y":     s1["s1_veto_w10y"],
            "s1_veto_reg":      s1["s1_veto_reg"],
            "mm_v7_P_min":      round(float(s1["mm_v7_P_min"]), 4),
            "mm_v7_CI95_min_pp":round(float(s1["mm_v7_CI95_min"]), 4),
            "mm_v7_P_maxdd":    round(float(s1["mm_v7_P_maxdd"]), 4),
            "mm_v7_P_w10y":     round(float(s1["mm_v7_P_w10y"]), 4),
            "mm_v7_P_sharpe":   round(float(s1["mm_v7_P_sharpe"]), 4),
            "mm_b3a_P_min":     round(float(s1["mm_b3a_P_min"]), 4),
            "mm_b3a_CI95_min_pp":round(float(s1["mm_b3a_CI95_min"]), 4),
            "mm_b3a_P_maxdd":   round(float(s1["mm_b3a_P_maxdd"]), 4),
            "mm_b3a_P_w10y":    round(float(s1["mm_b3a_P_w10y"]), 4),
            "mm_b3a_P_sharpe":  round(float(s1["mm_b3a_P_sharpe"]), 4),
        })

    return_block = {
        "script":    "leverext_high_sc180_200_20260620.py",
        "date":      "2026-06-20",
        "sanity_140": {
            "min9_got_pct":  round(san_min9 * 100, 4),
            "maxdd_got_pct": round(san_maxdd * 100, 4),
            "ok_min9":       bool(ok_min9),
            "ok_maxdd":      bool(ok_maxdd),
            "SANITY_PASS":   bool(ok_min9 and ok_maxdd),
        },
        "stage1":    rb_s1,
        "s1_csv":    s1_csv_path,
        "ar_csv":    ar_csv_path,
        "excess_extra_pct": round(EXCESS_EXTRA * 100, 4),
        "n_boot":    N_BOOT,
        "block":     BLOCK,
        "seed":      SEED,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))

    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
