"""
src/audit/leverext_combo_20260616.py
=====================================
Task #3: Combination sweep -- DH-W1 early re-entry (#2) x uniform scale extension (#1).
Goal: find the CAGR-DD frontier upper bound inside MaxDD veto (-50%).

CAUSAL INTEGRITY
-----------------
1. DH-W1 mask rebuilt from lev_mod_065 using _hold_mask_W1_param(enter_thr) -- same as #2.
   The hysteresis state machine uses only past signal; no future data.
2. v7 multiplier x lev_scale applied uniformly to lev_raw_masked -- same as #1.
3. C1 OUT fill (SOFR cash on bond-OFF OUT days) applied on top -- same as #1/#2.
4. k365 EXCESS_EXTRA=0.0025 for >3x leverage cost.
No post-hoc tuning; no lookahead; no veto relaxation.

GRID (pre-registered in LEVERUP_EXTENSION_PLAN_20260616.md -- Task #3)
-----------------------------------------------------------------------
enter_thr in {0.65, 0.60}   (lower enter -> shorter OUT, from #2)
lev_scale in {1.20, 1.25}   (moderate uplift, no strong map to limit MaxDD, from #1)
v7_map = B3a default {0:1.40, 1:1.40, 2:1.05, 3:1.00}  (no strong_boost)
=> 2 x 2 = 4 combination configs.

SANITY GATE (B3a baseline)
--------------------------
enter=0.70, scale=1.15, default map => known B3a_k365:
  min9 +20.98% +/-0.05pp, MaxDD -38.20% +/-0.10pp.
Halts if violated.

STAGE 0 (all 4 combos + B3a / #1 best / #2 best references)
-------------------------------------------------------------
Standard 10 metrics: CAGR_IS, CAGR_OOS, min9, IS-OOS gap, Sharpe, MaxDD,
  Worst10Y*, P10_5Y, Worst5Y, Trades/yr.
Extra: OUT ratio, >3x ratio, worst calendar year, 4 hard vetos (MaxDD/-50%, W10Y*/0).

STAGE 1 (vet-free AND min9 > #1 single best +23.83%)
------------------------------------------------------
_eval_one (WFA49 canonical + CPCV + regime + stress) + multimetric_bootstrap vs V7 + vs B3a.

HARD VETO (absolute, no relaxation)
------------------------------------
MaxDD < -50%  =>  VETO
Worst10Y* < 0  =>  VETO
WFE > 1.5     =>  VETO (Stage 1 only)
regime_min < -10%  =>  VETO (Stage 1 only)

OUTPUTS
-------
  src/audit/leverext_combo_20260616.py   (this script)
  audit_results/leverext_combo_20260616.csv
  RETURN_BLOCK printed to stdout (json)

ASCII-only (cp932). No git operations. No temp files.
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

# ---- B3a / k365 builders ---------------------------------------------------
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    _build_tqqq_base_param,
    EXCESS_EXTRA_K365_CENTRE,
    EXCESS_EXTRA_STORE,
)

# ---- Extended eval (WFA49 + CPCV + regime + stress) ----------------------
from src.audit.extended_eval_20260611 import (
    _eval_one, _cpcv_dist, _regime_cagr,
)

# ---- Multi-metric bootstrap -----------------------------------------------
from src.audit.multimetric_bootstrap_20260615 import (
    _block_bootstrap_multimetric,
)

# ---- Cost / NAV helpers ---------------------------------------------------
from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base,
    AFTER_TAX, SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)

# ---- Misc helpers ---------------------------------------------------------
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

# ---- DH-W1 mask builder (from #2) -----------------------------------------
# Reuse hold_mask_W1_param from leverext_reentry_20260616
from src.audit.leverext_reentry_20260616 import (
    _hold_mask_W1_param,
    _load_base_shared,
    EXIT_THR,
)

# ---- P09 OUT fill helper (from leverup_b1c1_20260612) --------------------
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_on_base_c1,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B3A_MAP_DEFAULT = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}

EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE   # 0.0025

# B3a_k365 sanity reference
B3A_KNOWN_MIN9  = 0.2098    # +20.98%
B3A_KNOWN_MAXDD = -0.3820   # -38.20%
SANITY_TOL_MIN9  = 0.0005   # 0.05pp
SANITY_TOL_MAXDD = 0.0010   # 0.10pp

# Hard veto thresholds
HARD_VETO_MAXDD  = -0.50
HARD_VETO_W10Y   = 0.0
HARD_VETO_WFE    = 1.5
HARD_VETO_REGIME = -0.10

# Stage-1 gate: min9 must exceed #1 single best
#   Task#1 best: Bext_str_sc1.35 min9+23.83% (strong map, scale=1.35)
STAGE1_MIN9_THRESHOLD = 0.2383

# Pre-registered grid (Task #3, plan)
ENTER_LEVELS = [0.65, 0.60]
SCALE_LEVELS = [1.20, 1.25]

COMBOS = []
for enter in ENTER_LEVELS:
    for scale in SCALE_LEVELS:
        COMBOS.append({
            "label":     "combo_e%.2f_sc%.2f" % (enter, scale),
            "enter_thr": enter,
            "lev_scale": scale,
            "v7_map":    B3A_MAP_DEFAULT,
        })

# Reference configs for Stage-0 table
# B3a: enter=0.70, scale=1.15, default map
# #1 best: enter=0.70, scale=1.35, strong map (from leverext_scale results)
# #2 best: enter=0.60, scale=1.15, default map (from leverext_reentry results)
B3A_MAP_STRONG  = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
REF_CONFIGS = [
    {"label": "B3a_k365_ref",
     "enter_thr": 0.70, "lev_scale": 1.15, "v7_map": B3A_MAP_DEFAULT},
    {"label": "ref_scale135str",        # #1 best (strong map sc=1.35)
     "enter_thr": 0.70, "lev_scale": 1.35, "v7_map": B3A_MAP_STRONG},
    {"label": "ref_enter060",           # #2 best (enter=0.60, sc=1.15)
     "enter_thr": 0.60, "lev_scale": 1.15, "v7_map": B3A_MAP_DEFAULT},
]


# ---------------------------------------------------------------------------
# Per-enter shared dict (rebuilds DH-W1 mask from lev_mod_065)
# ---------------------------------------------------------------------------

_SHARED_CACHE: dict = {}


def _get_shared_for_enter(enter_thr):
    """
    Return a 'shared' dict compatible with _build_full_c1, but with the DH-W1 mask
    rebuilt at enter_thr (hysteresis exit_thr=0.30 fixed).

    On first call per enter_thr, load from g14 and cache the result.
    """
    if enter_thr in _SHARED_CACHE:
        return _SHARED_CACHE[enter_thr]

    a = _load_base_shared()           # raw g14 assets (called once, cached inside reentry)
    lev_mod_065 = np.asarray(a["lev_mod_065"])
    mask = _hold_mask_W1_param(lev_mod_065, enter_thr=enter_thr, exit_thr=EXIT_THR)

    wn = np.asarray(a["wn_A"]) * mask
    wg = np.asarray(a["wg_A"]) * mask
    wb = np.asarray(a["wb_A"]) * mask
    lev_raw_masked = np.asarray(a["lev_raw"]) * mask

    shared = {
        "assets":           a,
        "mask":             mask,
        "wn":               wn,
        "wg":               wg,
        "wb":               wb,
        "lev_raw_masked":   lev_raw_masked,
    }
    _SHARED_CACHE[enter_thr] = shared
    return shared


# ---------------------------------------------------------------------------
# NAV builder (combo): rebuilds DH-W1 mask + applies lev_scale + C1 fill
# ---------------------------------------------------------------------------

def _build_combo_nav(enter_thr, lev_scale, v7_map, dates_dt, n_years,
                     ret_gold, ret_bond, sofr_arr):
    """
    Build full NAV for one combination config:
      1. Rebuild DH-W1 shared dict at enter_thr (causal mask reconstruction, #2).
      2. Apply v7_map x lev_scale via _build_tqqq_base_param (#1 lever scaling).
      3. C1 SOFR cash on bond-OFF OUT days.
      4. k365 excess cost (EXCESS_EXTRA=0.0025).

    Parameters
    ----------
    enter_thr : float
        DH-W1 hysteresis enter threshold (0.65 or 0.60 for combos).
    lev_scale : float
        Uniform leverage scale applied to v7_map multiplier (e.g. 1.20, 1.25).
    v7_map : dict
        V7 regime multipliers {0:..., 1:..., 2:..., 3:...}.
    dates_dt : pd.DatetimeIndex
    n_years : float
    ret_gold, ret_bond : np.ndarray  (gold/bond 1x daily returns)
    sofr_arr : np.ndarray

    Returns
    -------
    nav_dt : pd.Series (NAV)
    r : np.ndarray (daily returns)
    tpy : float (trades per year)
    exc : int (excess >3x days)
    out_ratio : float (fraction of days in OUT state)
    """
    shared = _get_shared_for_enter(enter_thr)
    mask   = shared["mask"]

    n = len(dates_dt)

    # OUT-fill parameters: inverse-vol weights for gold/bond; bond momentum signal
    out_mask_arr = (mask < 0.5)
    fund_active  = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    a = shared["assets"]
    dates = a["dates"]
    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on   = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    # _build_full_c1 uses shared["lev_raw_masked"] / wn / wg / wb for the TQQQ leg,
    # and fund_active / wg_iv / wb_iv / bond_on / sofr_arr for C1 OUT fill.
    nav_dt, r, tpy, exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        v7_map=v7_map, lev_scale=lev_scale, excess_extra=EXCESS_EXTRA,
    )
    out_ratio = float((mask < 0.5).mean())
    return nav_dt, r, tpy, exc, out_ratio


# ---------------------------------------------------------------------------
# Stage 0 metrics row
# ---------------------------------------------------------------------------

def _stage0_row(label, nav_dt, r, tpy, exc, n, out_ratio, enter_thr, lev_scale):
    """Compute standard-10 metrics + hard veto flags for Stage 0."""
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    cy  = _calendar_year_returns(nav_dt)

    min9 = min(aft["CAGR_IS"], aft["CAGR_OOS"])
    maxdd = pre["MaxDD_FULL"]
    w10y  = aft["Worst10Y_star"]

    excess_ratio = float(exc) / float(n) if n > 0 else 0.0

    v_maxdd = (maxdd < HARD_VETO_MAXDD)
    v_w10y  = (w10y  < HARD_VETO_W10Y)
    veto_s0 = v_maxdd or v_w10y

    return {
        "label":            label,
        "enter_thr":        enter_thr,
        "lev_scale":        lev_scale,
        # Standard 10 (after-tax CAGR; Sharpe/MaxDD pretax)
        "CAGR_IS_at":       round(aft["CAGR_IS"],         6),
        "CAGR_OOS_at":      round(aft["CAGR_OOS"],        6),
        "min9_at":          round(min9,                    6),
        "IS_OOS_gap_pp":    round(aft["IS_OOS_gap_pp"],   4),
        "Sharpe_OOS":       round(pre["Sharpe_OOS"],       4),
        "MaxDD_FULL":       round(maxdd,                   6),
        "Worst10Y_star_at": round(w10y,                    6),
        "P10_5Y_at":        round(aft["P10_5Y"],           6),
        "Worst5Y_at":       round(aft["Worst5Y"],          6),
        "Trades_yr":        round(aft["Trades_yr"],        2),
        # Extra
        "OUT_ratio":        round(out_ratio,               4),
        "excess_days":      int(exc),
        "excess_ratio_pct": round(excess_ratio * 100,      2),
        "worst_cy":         round(float(cy.min()),         6),
        "worst_cy_year":    int(cy.idxmin()),
        # Veto
        "veto_maxdd":       int(v_maxdd),
        "veto_w10y":        int(v_w10y),
        "VETO_s0":          int(veto_s0),
    }


# ---------------------------------------------------------------------------
# Stage 1 full gate
# ---------------------------------------------------------------------------

def _stage1_full_gate(label, nav_dt, r, tpy, regimes, stress,
                      is_mask, oos_mask, r_v7, r_b3a):
    """
    WFA49 + CPCV + regime + stress via _eval_one,
    then multi-metric bootstrap vs V7 and vs B3a.
    Returns flat dict.
    """
    ev = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                   baseline_r=r_v7)
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)

    wfe     = float(ev["wfa_WFE"])
    reg_min = float(ev["regime_min_at"])
    w10y    = aft["Worst10Y_star"]
    maxdd   = pre["MaxDD_FULL"]

    v_maxdd = (maxdd < HARD_VETO_MAXDD)
    v_wfe   = (wfe   > HARD_VETO_WFE)
    v_w10y  = (w10y  < HARD_VETO_W10Y)
    v_reg   = (reg_min < HARD_VETO_REGIME)
    veto_s1 = v_maxdd or v_wfe or v_w10y or v_reg

    print("      bootstrap vs V7 ...")
    boot_v7  = _block_bootstrap_multimetric(r, r_v7,  is_mask, oos_mask,
                                             n_boot=N_BOOT, block=BLOCK, seed=SEED)
    print("      bootstrap vs B3a ...")
    boot_b3a = _block_bootstrap_multimetric(r, r_b3a, is_mask, oos_mask,
                                             n_boot=N_BOOT, block=BLOCK, seed=SEED)

    boot_ev = ev.get("boot") or {}

    return {
        # WFA
        "wfa_WFE":         wfe,
        "wfa_CI95_lo":     float(ev["wfa_CI95_lo"]),
        "wfa_t_p":         float(ev["wfa_t_p"]),
        # CPCV
        "cpcv_p10_at":     float(ev["cpcv_p10_at"]),
        "cpcv_worst_at":   float(ev["cpcv_worst_at"]),
        "cpcv_med_at":     float(ev["cpcv_med_at"]),
        # Regime
        "regime_min_at":   reg_min,
        "regime":          ev["regime"],
        # Stress
        "stress":          ev["stress"],
        # Bootstrap vs V7 (1-axis from _eval_one)
        "boot_v7_P_min_better":   boot_ev.get("P_min_better",   np.nan),
        "boot_v7_CI95_lo_min_pp": boot_ev.get("CI95_lo_min_pp", np.nan),
        # Multi-metric bootstrap vs V7
        "mm_v7_P_min":    boot_v7["P_min_better"],
        "mm_v7_CI95_min": boot_v7["CI95_lo_min_pp"],
        "mm_v7_P_maxdd":  boot_v7["P_maxdd_better"],
        "mm_v7_CI95_dd":  boot_v7["CI95_lo_dd_pp"],
        "mm_v7_P_w10y":   boot_v7["P_worst10y_better"],
        "mm_v7_CI95_w10y":boot_v7["CI95_lo_w10y_pp"],
        "mm_v7_P_sharpe": boot_v7["P_sharpe_better"],
        "mm_v7_CI95_shp": boot_v7["CI95_lo_sharpe"],
        # Multi-metric bootstrap vs B3a
        "mm_b3a_P_min":    boot_b3a["P_min_better"],
        "mm_b3a_CI95_min": boot_b3a["CI95_lo_min_pp"],
        "mm_b3a_P_maxdd":  boot_b3a["P_maxdd_better"],
        "mm_b3a_CI95_dd":  boot_b3a["CI95_lo_dd_pp"],
        "mm_b3a_P_w10y":   boot_b3a["P_worst10y_better"],
        "mm_b3a_CI95_w10y":boot_b3a["CI95_lo_w10y_pp"],
        "mm_b3a_P_sharpe": boot_b3a["P_sharpe_better"],
        "mm_b3a_CI95_shp": boot_b3a["CI95_lo_sharpe"],
        # Veto
        "s1_veto_maxdd":   int(v_maxdd),
        "s1_veto_wfe":     int(v_wfe),
        "s1_veto_w10y":    int(v_w10y),
        "s1_veto_reg":     int(v_reg),
        "VETO_s1":         int(veto_s1),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_stage0_table(rows):
    w = 168
    print("\n" + "=" * w)
    print("STAGE 0 -- STANDARD 10 METRICS (after-tax CAGR; Sharpe/MaxDD pretax)")
    hdr = ("%-26s | %5s | %5s | %8s | %9s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %6s | %6s | %4s | %5s"
           % ("label", "enter", "scale", "CAGR_IS%", "CAGR_OOS%",
              "min9%", "gap_pp", "Sharpe", "MaxDD%", "W10Y*%",
              "P10_5Y%", "Trd/yr", "OUT%", ">3x%", "wcy%", "VETO"))
    print(hdr)
    print("-" * w)
    for row in rows:
        veto_str = "VETO" if row["VETO_s0"] else "pass"
        print(
            "%-26s | %5.2f | %5.2f | %+7.2f%% | %+8.2f%% | %+6.2f%% | %+6.2f | %7.3f"
            " | %+7.2f%% | %+7.2f%% | %+7.2f%% | %7.1f | %5.1f%% | %5.1f%% | %+5.2f%% | %-4s"
            % (
                row["label"][:26], row["enter_thr"], row["lev_scale"],
                row["CAGR_IS_at"] * 100, row["CAGR_OOS_at"] * 100,
                row["min9_at"] * 100,    row["IS_OOS_gap_pp"],
                row["Sharpe_OOS"],       row["MaxDD_FULL"] * 100,
                row["Worst10Y_star_at"] * 100, row["P10_5Y_at"] * 100,
                row["Trades_yr"],        row["OUT_ratio"] * 100,
                row["excess_ratio_pct"], row["worst_cy"] * 100,
                veto_str,
            )
        )
    print("=" * w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("LEVEREXT COMBO SWEEP  2026-06-16")
    print("Task #3: DH-W1 early re-entry x uniform scale -- combo frontier")
    print("Grid: enter {0.65, 0.60} x scale {1.20, 1.25} x default v7_map")
    print("      = 4 pre-registered combination configs.")
    print("References: B3a (enter=0.70, sc=1.15), #1_best (enter=0.70, sc=1.35 strong),")
    print("            #2_best (enter=0.60, sc=1.15 default)")
    print("Cost: k365 EXCESS_EXTRA=0.0025, C1 SOFR OUT fill")
    print("Sanity: enter=0.70, sc=1.15, default => B3a min9+20.98+/-0.05pp, MaxDD-38.20+/-0.10pp")
    print("Stage-1 gate: VETO_s0==0 AND min9 > +23.83%% (#1 single best)")
    print("Hard veto (absolute, no relaxation): MaxDD<-50%% / W10Y*<0 / WFE>1.5 / reg<-10%%")
    print("=" * 120)

    # ---- Load base shared assets (g14) ----
    print("\nLoading base shared assets (g14) ...")
    a = _load_base_shared()
    dates    = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)
    print("  n=%d  n_years=%.1f  IS_END=%s  OOS_START=%s"
          % (n, n_years, IS_END.date(), OOS_START.date()))

    # ---- Gold / Bond 1x returns ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)
    sofr_arr = np.asarray(a["sofr"], float)

    # =========================================================================
    # SANITY GATE: B3a (enter=0.70, scale=1.15, default map)
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: enter=0.70, scale=1.15, default map => B3a_k365")
    print("  Expected: min9 +20.98%% +/-0.05pp  MaxDD -38.20%% +/-0.10pp")
    print("=" * 120)

    san_nav, san_r, san_tpy, san_exc, san_or = _build_combo_nav(
        enter_thr=0.70, lev_scale=1.15, v7_map=B3A_MAP_DEFAULT,
        dates_dt=dates_dt, n_years=n_years,
        ret_gold=ret_gold, ret_bond=ret_bond, sofr_arr=sofr_arr,
    )
    san_pre  = compute_10metrics(san_nav, san_tpy)
    san_aft  = _apply_aftertax(san_pre)
    san_min9 = min(san_aft["CAGR_IS"], san_aft["CAGR_OOS"])
    san_maxdd= san_pre["MaxDD_FULL"]

    ok_min9  = abs(san_min9  - B3A_KNOWN_MIN9)  <= SANITY_TOL_MIN9
    ok_maxdd = abs(san_maxdd - B3A_KNOWN_MAXDD) <= SANITY_TOL_MAXDD

    print("  min9:  got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_min9 * 100, B3A_KNOWN_MIN9 * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD: got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_maxdd * 100, B3A_KNOWN_MAXDD * 100, "OK" if ok_maxdd else "FAIL"))

    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED -- halting. Check _build_combo_nav / _build_full_c1 wiring.")
        sys.exit(1)
    print("  SANITY PASSED.\n")

    # =========================================================================
    # STAGE 0: Build all configs (references + combos)
    # =========================================================================
    print("=" * 120)
    print("STAGE 0: Building NAVs for 3 references + 4 combo configs")
    print("=" * 120)

    s0_rows_all  = []   # ordered: refs first, then combos
    nav_cache    = {}   # label -> (nav_dt, r, tpy, exc, out_ratio)

    all_configs_ordered = REF_CONFIGS + COMBOS

    for cfg in all_configs_ordered:
        lbl   = cfg["label"]
        et    = cfg["enter_thr"]
        sc    = cfg["lev_scale"]
        mp    = cfg["v7_map"]
        print("  Building %s (enter=%.2f, scale=%.2f) ..." % (lbl, et, sc))
        nav_dt, r, tpy, exc, out_ratio = _build_combo_nav(
            enter_thr=et, lev_scale=sc, v7_map=mp,
            dates_dt=dates_dt, n_years=n_years,
            ret_gold=ret_gold, ret_bond=ret_bond, sofr_arr=sofr_arr,
        )
        nav_cache[lbl] = (nav_dt, r, tpy, exc, out_ratio)
        row = _stage0_row(lbl, nav_dt, r, tpy, exc, n, out_ratio, et, sc)
        s0_rows_all.append(row)
        print("    min9=%+.2f%%  MaxDD=%+.2f%%  OUT=%.1f%%  Trades=%.1f  VETO=%s"
              % (row["min9_at"] * 100, row["MaxDD_FULL"] * 100,
                 row["OUT_ratio"] * 100, row["Trades_yr"],
                 "YES" if row["VETO_s0"] else "no"))

    _print_stage0_table(s0_rows_all)

    # ---- VETO boundary analysis ----
    print("\n--- VETO ANALYSIS (combo configs only) ---")
    combo_s0 = [r for r in s0_rows_all if r["label"].startswith("combo_")]
    any_maxdd_veto = False
    for row in combo_s0:
        if row["veto_maxdd"]:
            any_maxdd_veto = True
            print("  MaxDD<-50%% VETO: %s  enter=%.2f  scale=%.2f  MaxDD=%+.2f%%"
                  % (row["label"], row["enter_thr"], row["lev_scale"],
                     row["MaxDD_FULL"] * 100))
    if not any_maxdd_veto:
        print("  MaxDD-50%% veto NOT triggered in any combo -- all < -50%% limit not reached.")

    # ---- Interaction analysis (vs single factors) ----
    print("\n--- INTERACTION ANALYSIS (combo vs single-factor extrapolation) ---")
    ref_map = {r["label"]: r for r in s0_rows_all}
    b3a_row = ref_map.get("B3a_k365_ref")
    sc135_row = ref_map.get("ref_scale135str")   # #1 best
    e060_row  = ref_map.get("ref_enter060")       # #2 best

    if b3a_row:
        b3a_min9 = b3a_row["min9_at"]
        print("  B3a baseline (enter=0.70, sc=1.15): min9=%+.2f%%" % (b3a_min9 * 100))

    for row in combo_s0:
        if b3a_row and sc135_row and e060_row:
            # Simple extrapolation: delta from B3a for each single factor
            # delta_scale = (similar enter row with scale vs B3a) -- approximate only
            # We don't have exact single-factor points at same scale, so just show raw
            delta_vs_b3a = (row["min9_at"] - b3a_row["min9_at"]) * 100
            veto_str = "VETO" if row["VETO_s0"] else "pass"
            print("  %s: min9=%+.2f%%  delta_vs_B3a=%+.2fpp  MaxDD=%+.2f%%  [%s]"
                  % (row["label"], row["min9_at"] * 100, delta_vs_b3a,
                     row["MaxDD_FULL"] * 100, veto_str))

    # =========================================================================
    # V7 baseline for bootstrap
    # =========================================================================
    print("\nBuilding V7_TQQQ baseline for bootstrap ...")
    sr._load_dhw1_shared()
    shared_dhw1 = sr._DHW1_SHARED
    v7_nav, r_v7, _, _ = _build_tqqq_base(
        shared_dhw1, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    # B3a returns (enter=0.70, sc=1.15 = sanity point)
    _, r_b3a, _, _, _ = nav_cache["B3a_k365_ref"]

    # Regime labels and stress masks
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress  = stress_masks(dates_dt)

    # =========================================================================
    # STAGE 1: Full gate
    # =========================================================================
    print("\n" + "=" * 120)
    print("STAGE 1: Full gate criterion:")
    print("  VETO_s0==0  AND  min9_at > %.4f%% (#1 single best, scale1.35 strong)"
          % (STAGE1_MIN9_THRESHOLD * 100))
    print("=" * 120)

    s1_results = []
    candidates_s1 = [r for r in combo_s0
                     if r["VETO_s0"] == 0 and r["min9_at"] > STAGE1_MIN9_THRESHOLD]
    candidates_s1 = sorted(candidates_s1, key=lambda x: -x["min9_at"])

    if len(candidates_s1) == 0:
        print("  No combo config passed Stage-1 criterion (min9 > +23.83%%). Stage 1 skipped.")
    else:
        print("  %d candidate(s) selected for full gate:" % len(candidates_s1))
        for cand in candidates_s1:
            print("    %s  enter=%.2f  scale=%.2f  min9=%+.2f%%  MaxDD=%+.2f%%"
                  % (cand["label"], cand["enter_thr"], cand["lev_scale"],
                     cand["min9_at"] * 100, cand["MaxDD_FULL"] * 100))

        print("\n--- Running Stage-1 full gate ---")
        for cand in candidates_s1:
            lbl = cand["label"]
            print("\n  [%s] WFA49 + CPCV + regime + stress + bootstrap ..." % lbl)
            nav_dt, r, tpy, exc, out_ratio = nav_cache[lbl]
            s1_r = _stage1_full_gate(
                lbl, nav_dt, r, tpy, regimes, stress,
                is_mask, oos_mask, r_v7, r_b3a)
            s1_results.append({"label": lbl, "s0": cand, "s1": s1_r})
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

    # Stage-1 summary table
    if s1_results:
        print("\n" + "=" * 120)
        print("STAGE 1 FULL GATE RESULTS")
        print("%-28s | %5s | %5s | %8s | %7s | %8s | %8s | %8s | %4s"
              % ("label", "enter", "scale", "WFE", "CI95%", "CPCV_p10%",
                 "Reg_min%", "W10Y*%", "VETO"))
        print("-" * 120)
        for entry in s1_results:
            s0 = entry["s0"]
            s1 = entry["s1"]
            veto_str = "VETO" if s1["VETO_s1"] else "PASS"
            print("%-28s | %5.2f | %5.2f | %7.4f | %+6.2f%% | %+7.2f%% | %+7.2f%% | %+7.2f%% | %-4s"
                  % (entry["label"][:28],
                     s0["enter_thr"], s0["lev_scale"],
                     s1["wfa_WFE"], s1["wfa_CI95_lo"] * 100,
                     s1["cpcv_p10_at"] * 100, s1["regime_min_at"] * 100,
                     s0["Worst10Y_star_at"] * 100, veto_str))

        print("\n--- BOOTSTRAP SUMMARY ---")
        print("%-28s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s"
              % ("label (vs V7/B3a)", "P_min_V7", "CI_min_V7",
                 "P_min_B3a", "CI_min_B3a",
                 "P_dd_V7", "P_dd_B3a", "P_w10y_V7", "P_shp_V7"))
        print("-" * 120)
        for entry in s1_results:
            s1 = entry["s1"]
            print("%-28s | %8.3f | %+7.2f%% | %8.3f | %+7.2f%% | %8.3f | %8.3f | %8.3f | %8.3f"
                  % (entry["label"][:28],
                     s1["mm_v7_P_min"],   s1["mm_v7_CI95_min"],
                     s1["mm_b3a_P_min"],  s1["mm_b3a_CI95_min"],
                     s1["mm_v7_P_maxdd"], s1["mm_b3a_P_maxdd"],
                     s1["mm_v7_P_w10y"],  s1["mm_v7_P_sharpe"]))

    # =========================================================================
    # CONCLUSIONS
    # =========================================================================
    print("\n" + "=" * 120)
    print("CONCLUSIONS")
    print("=" * 120)

    print("\n1. STAGE-0 FRONTIER (combo configs, min9 vs MaxDD):")
    print("   label                      | enter | scale | min9%   | MaxDD%  | Sharpe | VETO")
    print("   " + "-" * 90)
    for row in sorted(combo_s0, key=lambda x: -x["min9_at"]):
        veto_str = "VETO" if row["VETO_s0"] else "pass"
        print("   %-26s | %5.2f | %5.2f | %+6.2f%% | %+6.2f%% | %.3f  | %s"
              % (row["label"][:26], row["enter_thr"], row["lev_scale"],
                 row["min9_at"] * 100, row["MaxDD_FULL"] * 100,
                 row["Sharpe_OOS"], veto_str))

    print("\n2. VETO BOUNDARY:")
    veto_combo = [r for r in combo_s0 if r["veto_maxdd"]]
    if veto_combo:
        first_veto = sorted(veto_combo, key=lambda x: x["MaxDD_FULL"])[0]
        print("   MaxDD-50%% veto first hit: %s  enter=%.2f  scale=%.2f  MaxDD=%+.2f%%"
              % (first_veto["label"], first_veto["enter_thr"],
                 first_veto["lev_scale"], first_veto["MaxDD_FULL"] * 100))
    else:
        best_no_veto = sorted(combo_s0, key=lambda x: -x["min9_at"])
        if best_no_veto:
            b = best_no_veto[0]
            print("   MaxDD-50%% veto NOT reached in any combo config.")
            print("   Best combo (no veto): %s  enter=%.2f  scale=%.2f  min9=%+.2f%%  MaxDD=%+.2f%%"
                  % (b["label"], b["enter_thr"], b["lev_scale"],
                     b["min9_at"] * 100, b["MaxDD_FULL"] * 100))

    print("\n3. FRONTIER vs SINGLE FACTORS (#1/#2 best):")
    ref_b3a     = ref_map.get("B3a_k365_ref")
    ref_sc135   = ref_map.get("ref_scale135str")
    ref_e060    = ref_map.get("ref_enter060")
    if ref_b3a:
        print("   B3a_k365    (enter=0.70, sc=1.15, default):  min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f"
              % (ref_b3a["min9_at"] * 100, ref_b3a["MaxDD_FULL"] * 100, ref_b3a["Sharpe_OOS"]))
    if ref_sc135:
        print("   #1_best     (enter=0.70, sc=1.35, strong):   min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f"
              % (ref_sc135["min9_at"] * 100, ref_sc135["MaxDD_FULL"] * 100, ref_sc135["Sharpe_OOS"]))
    if ref_e060:
        print("   #2_best     (enter=0.60, sc=1.15, default):  min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f"
              % (ref_e060["min9_at"] * 100, ref_e060["MaxDD_FULL"] * 100, ref_e060["Sharpe_OOS"]))

    best_combo = sorted([r for r in combo_s0 if r["VETO_s0"] == 0],
                        key=lambda x: -x["min9_at"])
    if best_combo:
        bc = best_combo[0]
        print("   Best combo  (%s):  min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f"
              % (bc["label"], bc["min9_at"] * 100, bc["MaxDD_FULL"] * 100, bc["Sharpe_OOS"]))

        # Interaction sign determination
        if ref_sc135 and ref_e060 and ref_b3a:
            # Method: compare best_combo vs max(#1, #2) single
            max_single_min9 = max(ref_sc135["min9_at"], ref_e060["min9_at"])
            interaction_delta = (bc["min9_at"] - max_single_min9) * 100
            print("\n4. INTERACTION SIGN:")
            print("   Best combo min9 = %+.2f%%  vs  max(single-factor) = %+.2f%%"
                  % (bc["min9_at"] * 100, max_single_min9 * 100))
            print("   Interaction delta = %+.2f pp  => %s"
                  % (interaction_delta,
                     "POSITIVE (combination exceeds best single)" if interaction_delta > 0
                     else "NEGATIVE or neutral (combination does NOT exceed best single)"))
            print("   NOTE: Combo uses default map (no strong_boost) and moderate scale;")
            print("   #1 best uses strong map+sc1.35. Direct scale comparison is approximate.")
        else:
            print("\n4. INTERACTION: Could not determine (reference rows missing).")

        if s1_results:
            print("\n5. FRONTIER UPDATE:")
            best_s1 = [e for e in s1_results if e["s1"]["VETO_s1"] == 0]
            if best_s1:
                best_e = best_s1[0]
                s0b = best_e["s0"]
                s1b = best_e["s1"]
                print("   STAGE-1 PASS: %s  enter=%.2f  scale=%.2f"
                      % (best_e["label"], s0b["enter_thr"], s0b["lev_scale"]))
                print("   min9=%+.2f%%  MaxDD=%+.2f%%  WFE=%.4f  CI95_lo=%+.2f%%  CPCV_p10=%+.2f%%"
                      % (s0b["min9_at"] * 100, s0b["MaxDD_FULL"] * 100,
                         s1b["wfa_WFE"], s1b["wfa_CI95_lo"] * 100, s1b["cpcv_p10_at"] * 100))
                print("   Bootstrap vs B3a: P_min=%.3f  CI95_min=%+.2f%%"
                      % (s1b["mm_b3a_P_min"], s1b["mm_b3a_CI95_min"]))
                if s0b["min9_at"] > STAGE1_MIN9_THRESHOLD:
                    print("   => FRONTIER UPPER END UPDATED vs #1 single best (+%.2f%% > +23.83%%)"
                          % (s0b["min9_at"] * 100))
                else:
                    print("   => min9 does NOT exceed #1 best (+23.83%%). Frontier not updated.")
            else:
                print("   All Stage-1 candidates vetoed. Frontier not updated by combo.")
        else:
            print("\n5. Stage 1 skipped (no combo passed min9 > +23.83%% threshold).")
            print("   => Combination does NOT update the CAGR-DD frontier upper end.")
    else:
        print("   All combo configs vetoed at Stage 0. Frontier analysis N/A.")
        print("\n4. INTERACTION: N/A (all combos vetoed).")
        print("\n5. FRONTIER: Not updated (all combos vetoed).")

    print("\n6. SELECTION BIAS NOTE:")
    print("   Grid is pre-registered (LEVERUP_EXTENSION_PLAN_20260616.md Task #3).")
    print("   Combination of two factors (enter + scale) has additive/multiplicative")
    print("   selection bias. Treat CI95_lo and bootstrap P_better as primary evidence.")
    print("   Vet-relaxation is prohibited even for promising combos (MaxDD-50%% is absolute).")

    # =========================================================================
    # CSV output
    # =========================================================================
    print("\nBuilding CSV ...")
    axes_order = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                  "rate:rate_up", "rate:rate_down"]

    csv_rows = []
    s1_by_label = {e["label"]: e["s1"] for e in s1_results}

    for row in s0_rows_all:
        lbl     = row["label"]
        is_combo = lbl.startswith("combo_")
        r_entry  = {"stage": 0, "is_combo": int(is_combo)}
        r_entry.update(row)

        s1_match = s1_by_label.get(lbl)
        if s1_match:
            s1 = s1_match
            r_entry.update({
                "wfa_WFE":         s1["wfa_WFE"],
                "wfa_CI95_lo":     s1["wfa_CI95_lo"],
                "wfa_t_p":         s1["wfa_t_p"],
                "cpcv_p10_at":     s1["cpcv_p10_at"],
                "cpcv_worst_at":   s1["cpcv_worst_at"],
                "cpcv_med_at":     s1["cpcv_med_at"],
                "regime_min_at":   s1["regime_min_at"],
                "s1_veto_maxdd":   s1["s1_veto_maxdd"],
                "s1_veto_wfe":     s1["s1_veto_wfe"],
                "s1_veto_w10y":    s1["s1_veto_w10y"],
                "s1_veto_reg":     s1["s1_veto_reg"],
                "VETO_s1":         s1["VETO_s1"],
                "mm_v7_P_min":     s1["mm_v7_P_min"],
                "mm_v7_CI95_min":  s1["mm_v7_CI95_min"],
                "mm_v7_P_maxdd":   s1["mm_v7_P_maxdd"],
                "mm_v7_CI95_dd":   s1["mm_v7_CI95_dd"],
                "mm_v7_P_w10y":    s1["mm_v7_P_w10y"],
                "mm_v7_CI95_w10y": s1["mm_v7_CI95_w10y"],
                "mm_v7_P_sharpe":  s1["mm_v7_P_sharpe"],
                "mm_v7_CI95_shp":  s1["mm_v7_CI95_shp"],
                "mm_b3a_P_min":    s1["mm_b3a_P_min"],
                "mm_b3a_CI95_min": s1["mm_b3a_CI95_min"],
                "mm_b3a_P_maxdd":  s1["mm_b3a_P_maxdd"],
                "mm_b3a_CI95_dd":  s1["mm_b3a_CI95_dd"],
                "mm_b3a_P_w10y":   s1["mm_b3a_P_w10y"],
                "mm_b3a_CI95_w10y":s1["mm_b3a_CI95_w10y"],
                "mm_b3a_P_sharpe": s1["mm_b3a_P_sharpe"],
                "mm_b3a_CI95_shp": s1["mm_b3a_CI95_shp"],
            })
            for ax in axes_order:
                r_entry["regime_" + ax.replace(":", "_")] = s1["regime"].get(ax, np.nan)
            for sw, sv in s1["stress"].items():
                r_entry["stress_%s_ret" % sw]   = sv["ret"]
                r_entry["stress_%s_maxdd" % sw] = sv["maxdd"]
        else:
            for col in [
                "wfa_WFE", "wfa_CI95_lo", "wfa_t_p",
                "cpcv_p10_at", "cpcv_worst_at", "cpcv_med_at",
                "regime_min_at",
                "s1_veto_maxdd", "s1_veto_wfe", "s1_veto_w10y", "s1_veto_reg", "VETO_s1",
                "mm_v7_P_min",    "mm_v7_CI95_min",  "mm_v7_P_maxdd",
                "mm_v7_CI95_dd",  "mm_v7_P_w10y",    "mm_v7_CI95_w10y",
                "mm_v7_P_sharpe", "mm_v7_CI95_shp",
                "mm_b3a_P_min",   "mm_b3a_CI95_min", "mm_b3a_P_maxdd",
                "mm_b3a_CI95_dd", "mm_b3a_P_w10y",   "mm_b3a_CI95_w10y",
                "mm_b3a_P_sharpe","mm_b3a_CI95_shp",
            ]:
                r_entry[col] = ""
            for ax in axes_order:
                r_entry["regime_" + ax.replace(":", "_")] = ""

        csv_rows.append(r_entry)

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "leverext_combo_20260616.csv")
    pd.DataFrame(csv_rows).to_csv(
        csv_path, index=False, float_format="%.6f", encoding="utf-8-sig")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(csv_rows)))

    # =========================================================================
    # RETURN_BLOCK (json)
    # =========================================================================
    rb_s0 = []
    for row in s0_rows_all:
        rb_s0.append({
            "label":               row["label"],
            "enter_thr":           row["enter_thr"],
            "lev_scale":           row["lev_scale"],
            "is_combo":            row["label"].startswith("combo_"),
            "min9_at_pct":         round(row["min9_at"] * 100,         4),
            "CAGR_IS_at_pct":      round(row["CAGR_IS_at"] * 100,      4),
            "CAGR_OOS_at_pct":     round(row["CAGR_OOS_at"] * 100,     4),
            "IS_OOS_gap_pp":       round(row["IS_OOS_gap_pp"],          4),
            "MaxDD_pct":           round(row["MaxDD_FULL"] * 100,       4),
            "Sharpe_OOS":          round(row["Sharpe_OOS"],             4),
            "Worst10Y_at_pct":     round(row["Worst10Y_star_at"] * 100, 4),
            "P10_5Y_at_pct":       round(row["P10_5Y_at"] * 100,        4),
            "Worst5Y_at_pct":      round(row["Worst5Y_at"] * 100,       4),
            "Trades_yr":           round(row["Trades_yr"],               2),
            "OUT_ratio_pct":       round(row["OUT_ratio"] * 100,         2),
            "excess_ratio_pct":    round(row["excess_ratio_pct"],        2),
            "worst_cy_pct":        round(row["worst_cy"] * 100,          4),
            "VETO_s0":             row["VETO_s0"],
        })

    rb_s1 = []
    for entry in s1_results:
        s0r = entry["s0"]
        s1r = entry["s1"]
        rb_s1.append({
            "label":                  entry["label"],
            "enter_thr":              s0r["enter_thr"],
            "lev_scale":              s0r["lev_scale"],
            "min9_at_pct":            round(s0r["min9_at"] * 100,          4),
            "MaxDD_pct":              round(s0r["MaxDD_FULL"] * 100,        4),
            "wfa_WFE":                round(s1r["wfa_WFE"],                 4),
            "wfa_CI95_lo_pct":        round(s1r["wfa_CI95_lo"] * 100,       4),
            "wfa_t_p":                round(s1r["wfa_t_p"],                  4),
            "cpcv_p10_at_pct":        round(s1r["cpcv_p10_at"] * 100,       4),
            "regime_min_at_pct":      round(s1r["regime_min_at"] * 100,     4),
            "VETO_s1":                s1r["VETO_s1"],
            "mm_v7_P_min":            round(float(s1r["mm_v7_P_min"]),      4),
            "mm_v7_CI95_min_pp":      round(float(s1r["mm_v7_CI95_min"]),   4),
            "mm_v7_P_maxdd":          round(float(s1r["mm_v7_P_maxdd"]),    4),
            "mm_v7_P_w10y":           round(float(s1r["mm_v7_P_w10y"]),     4),
            "mm_v7_P_sharpe":         round(float(s1r["mm_v7_P_sharpe"]),   4),
            "mm_b3a_P_min":           round(float(s1r["mm_b3a_P_min"]),     4),
            "mm_b3a_CI95_min_pp":     round(float(s1r["mm_b3a_CI95_min"]),  4),
            "mm_b3a_P_maxdd":         round(float(s1r["mm_b3a_P_maxdd"]),   4),
            "mm_b3a_P_w10y":          round(float(s1r["mm_b3a_P_w10y"]),    4),
            "mm_b3a_P_sharpe":        round(float(s1r["mm_b3a_P_sharpe"]),  4),
        })

    # Determine overall conclusion
    best_combo_vet_free = [r for r in combo_s0 if r["VETO_s0"] == 0]
    if best_combo_vet_free:
        bcc = max(best_combo_vet_free, key=lambda x: x["min9_at"])
        combo_best_min9 = bcc["min9_at"]
    else:
        combo_best_min9 = None

    # max single-factor min9 from references (excluding B3a)
    single_factor_min9_list = []
    for rn in ["ref_scale135str", "ref_enter060"]:
        if rn in ref_map:
            single_factor_min9_list.append(ref_map[rn]["min9_at"])
    max_single_min9 = max(single_factor_min9_list) if single_factor_min9_list else None

    if combo_best_min9 is not None and max_single_min9 is not None:
        interaction_sign = "positive" if combo_best_min9 > max_single_min9 else "negative_or_neutral"
    else:
        interaction_sign = "undetermined"

    return_block = {
        "script":                  "leverext_combo_20260616.py",
        "date":                    "2026-06-16",
        "sanity": {
            "B3a_min9_got_pct":    round(san_min9 * 100,  4),
            "B3a_MaxDD_got_pct":   round(san_maxdd * 100, 4),
            "ok_min9":             bool(ok_min9),
            "ok_maxdd":            bool(ok_maxdd),
            "SANITY_PASS":         bool(ok_min9 and ok_maxdd),
        },
        "stage1_gate_threshold_pct": round(STAGE1_MIN9_THRESHOLD * 100, 4),
        "stage0":                  rb_s0,
        "stage1":                  rb_s1,
        "csv_path":                csv_path,
        "n_combos":                len(COMBOS),
        "n_refs":                  len(REF_CONFIGS),
        "excess_extra_pct":        round(EXCESS_EXTRA * 100, 4),
        "combo_best_min9_pct":     round(combo_best_min9 * 100, 4) if combo_best_min9 else None,
        "max_single_factor_min9_pct": round(max_single_min9 * 100, 4) if max_single_min9 else None,
        "interaction_sign":        interaction_sign,
        "stage1_candidates":       len(candidates_s1),
        "stage1_pass":             len([e for e in s1_results if e["s1"]["VETO_s1"] == 0]),
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
