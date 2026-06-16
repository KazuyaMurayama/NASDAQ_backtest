"""
src/audit/leverext_reentry_20260616.py
=======================================
Task #2: Early Re-entry Sweep -- DH-W1 enter threshold lowered to shorten OUT periods.

CAUSAL INTEGRITY / DH-W1 RECONSTRUCTION
-----------------------------------------
DH-W1 uses a hysteresis state machine over lev_mod_065 (see g23a_dh_refinement_variants.py):
  - State starts at OUT (0).
  - Transition OUT->HOLD: when lev_mod_065[i] >= enter_thr   (default 0.70)
  - Transition HOLD->OUT: when lev_mod_065[i] <= exit_thr    (default 0.30)
  - Between thresholds: stays in current state (hysteresis band).
Lowering enter_thr (0.65, 0.60) makes the strategy re-enter HOLD sooner after exiting,
reducing OUT days. The signal lev_mod_065 is computed from past-only windows (LT2 N750
+ vz rolling), so the state machine remains causal: at each step only lev_mod_065[0..i]
is used. No future information is introduced.

exit_thr=0.30 is held fixed; only enter_thr varies: {0.70, 0.65, 0.60}.

PIPELINE (identical to k365_recost_20260612 B3a_k365)
------------------------------------------------------
1. _load_dhw1_shared_with_enter(enter_thr) -- rebuild mask/lev_raw_masked/wn/wg/wb
   using hold_mask_W1(a, enter_thr=enter_thr, exit_thr=0.30).
2. V7 map {Q0:1.40, Q1:1.40, Q2:1.05, Q3:1.00} x scale 1.15  (B3a config).
3. C1 SOFR cash on bond-OFF OUT days (from k365_recost._build_full_c1 pattern).
4. k365 excess cost (EXCESS_EXTRA_K365_CENTRE = 0.0025).
5. After-tax x0.8273.

SANITY CHECK
------------
enter=0.70 must reproduce B3a_k365 known values within tolerance:
  min9 +20.98% (+-0.10pp), MaxDD -38.20% (+-0.80pp).
Script halts if sanity fails.

STAGE 0
-------
Standard 10 metrics (after-tax) + OUT ratio + worst calendar year + 4 hard vetos.
All 3 enter water levels reported: 0.70, 0.65, 0.60.

STAGE 1
-------
Configurations that pass Stage 0 with veto=0 AND min9 > B3a_k365 (+20.98%) proceed
to full gate: _eval_one (WFA49 + CPCV + regime + stress) + multimetric_bootstrap
(vs B3a_k365 and vs V7_TQQQ baseline).

OUTPUT
------
  audit_results/leverext_reentry_20260616.csv   -- Stage 0 + Stage 1 columns
  RETURN_BLOCK printed to stdout (json).

ASCII-only prints (cp932 compatible). No temp files. No git operations.
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
from src.audit.extended_eval_20260611 import _eval_one, _cpcv_dist, _regime_cagr
from src.audit.run_p09_tqqq_validate_20260611 import (
    _block_bootstrap_compare, _cagr_seg, _maxdd_from_returns,
    AFTER_TAX, N_BOOT, BLOCK, SEED,
    _build_v7_mult_custom,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _count_fund_transitions,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, _build_p09_nav,
    FEE_GOLD, FEE_BOND,
)
from src.audit.lu_cfd_recost_20260611 import (
    SWAP_SPREAD, TER_TQQQ, LEV_CAP, DH_PER_UNIT, NAV_FLOOR,
    _build_nav_v7_tqqq, EXCESS_EXTRA, AFTER_TAX,
)
from src.audit.k365_recost_20260612 import (
    EXCESS_EXTRA_K365_CENTRE, _build_tqqq_base_param, _build_full_c1,
    _build_nav_v7_tqqq_param,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_nav_c1, _build_p09_on_base_c1,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    _build_v7_mult, DELAY as V7_DELAY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B3A_V7_MAP = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_SCALE = 1.15
EXIT_THR = 0.30           # fixed
ENTER_LEVELS = [0.70, 0.65, 0.60]

# B3a_k365 reference values (from LEVERUP_SWEEP_RESULTS_20260612.md)
B3A_MIN9_REF = 0.2098     # +20.98%
B3A_MAXDD_REF = -0.3820   # -38.20%
SANITY_TOL_MIN9 = 0.001   # 0.10pp tolerance (tight: should match B3a exactly)
SANITY_TOL_MAXDD = 0.008  # 0.80pp tolerance

# Hard veto thresholds (standard)
HARD_VETO_MAXDD  = -0.50
HARD_VETO_WFE    = 1.5
HARD_VETO_W10Y   = 0.0
HARD_VETO_REGIME = -0.10


# ---------------------------------------------------------------------------
# DH-W1 mask builder with parameterisable enter threshold
# ---------------------------------------------------------------------------

def _hold_mask_W1_param(lev_mod_065_arr, enter_thr, exit_thr=EXIT_THR):
    """
    Hysteresis state machine for DH-W1 with configurable enter_thr.

    Causal: at time i, only lev_mod_065[0..i] is used (current and past values).
    No future data is accessed. enter_thr controls how quickly we re-enter HOLD.

    Parameters
    ----------
    lev_mod_065_arr : array-like
        lev_mod_065 signal (from shared assets, already causal).
    enter_thr : float
        Threshold to transition OUT->HOLD. Lower = earlier re-entry.
    exit_thr : float
        Threshold to transition HOLD->OUT. Fixed at 0.30.

    Returns
    -------
    mask : np.ndarray of float (0.0=OUT, 1.0=HOLD)
    """
    lm = np.nan_to_num(np.asarray(lev_mod_065_arr, float), nan=0.0)
    n = len(lm)
    mask = np.zeros(n, dtype=float)
    state = 0  # start OUT
    for i in range(n):
        if state == 0 and lm[i] >= enter_thr:
            state = 1
        elif state == 1 and lm[i] <= exit_thr:
            state = 0
        mask[i] = float(state)
    return mask


# ---------------------------------------------------------------------------
# Load shared assets once, then build per-enter_thr variants
# ---------------------------------------------------------------------------
_SHARED_BASE: dict | None = None


def _load_base_shared():
    """Load g14 shared assets (called once; provides lev_mod_065 etc.)."""
    global _SHARED_BASE
    if _SHARED_BASE is not None:
        return _SHARED_BASE

    # strategy_runners caches _DHW1_SHARED which uses hold_mask_W1(enter=0.70).
    # We need the raw g14 assets to rebuild the mask at different enter thresholds.
    from g14_wfa_sbi_cfd import load_shared_assets
    a = load_shared_assets()
    _SHARED_BASE = a
    return _SHARED_BASE


def _build_shared_for_enter(enter_thr):
    """
    Build a 'shared' dict (compatible with _build_tqqq_base_param) for a given enter_thr.

    The DH-W1 mask is rebuilt from scratch using _hold_mask_W1_param(enter_thr=enter_thr).
    wn/wg/wb/lev_raw_masked are derived from the new mask.
    """
    a = _load_base_shared()
    lev_mod_065 = np.asarray(a["lev_mod_065"])
    mask = _hold_mask_W1_param(lev_mod_065, enter_thr=enter_thr, exit_thr=EXIT_THR)

    wn = np.asarray(a["wn_A"]) * mask
    wg = np.asarray(a["wg_A"]) * mask
    wb = np.asarray(a["wb_A"]) * mask
    lev_raw_masked = np.asarray(a["lev_raw"]) * mask

    # Build scenarioD NAV for completeness (not used in final metrics)
    from g18_daily_trade_cost_wfa import build_dh_nav_with_cost
    from g23a_dh_refinement_variants import DH_PER_UNIT as _DH_PER
    nav_base, _ = build_dh_nav_with_cost(
        a["close"], lev_raw_masked, wn, wg, wb,
        a["dates"], a["gold_2x"], a["bond_3x"], a["sofr"], _DH_PER,
    )

    return {
        "assets": a,
        "mask": mask,
        "wn": wn,
        "wg": wg,
        "wb": wb,
        "lev_raw_masked": lev_raw_masked,
        "nav_base": nav_base,
    }


# ---------------------------------------------------------------------------
# P09 out-fill helpers (C1: SOFR cash on bond-OFF OUT days)
# ---------------------------------------------------------------------------

def _build_out_fill_params(a, mask, ret_gold, ret_bond, n, n_years, dates_dt):
    """Build OUT-fill parameters for C1 NAV: fund_active, wg, wb, bond_on, sofr_arr."""
    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    dates = a["dates"]
    wg_w, wb_w = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    return fund_active, wg_w, wb_w, bond_on, sofr_arr


def _build_nav_enter(shared, dates_dt, n_years,
                     ret_gold, ret_bond, fund_active, wg_w, wb_w, bond_on, sofr_arr):
    """Build full C1 + k365 NAV for a given shared (pre-built for enter_thr)."""
    a = shared["assets"]
    close = a["close"]
    dates = a["dates"]
    sofr = np.asarray(a["sofr"], float)
    gold_2x = a["gold_2x"]
    bond_3x = a["bond_3x"]
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn = np.asarray(shared["wn"], float)
    wg = np.asarray(shared["wg"], float)
    wb = np.asarray(shared["wb"], float)

    # Build V7 mult with B3a config
    mult_v7 = _build_v7_mult_custom(dates_dt, B3A_V7_MAP)
    mult_v7 = mult_v7 * B3A_SCALE

    # Build TQQQ base NAV with k365 excess cost
    from src.audit.k365_recost_20260612 import _build_nav_v7_tqqq_param
    nav_base, tpy_b, excess_days = _build_nav_v7_tqqq_param(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7,
        excess_extra=EXCESS_EXTRA_K365_CENTRE,
    )
    r_base = nav_base.pct_change().fillna(0).values

    # Add C1 OUT fill (SOFR cash on bond-OFF OUT days)
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg_w, wb_w, bond_on,
        sofr_arr, dates_dt, tpy_b, n_years)

    return nav_dt, r, tpy, excess_days


def _compute_out_ratio(mask):
    """Fraction of days in OUT state (mask == 0)."""
    return float((mask < 0.5).mean())


def _compute_trades_per_year(lev_raw_masked, n):
    """Count lev_raw_masked transitions / n_years."""
    n_years = n / float(TRADING_DAYS)
    change = np.zeros(n, dtype=bool)
    change[1:] = lev_raw_masked[1:] != lev_raw_masked[:-1]
    return int(change.sum()) / n_years if n_years > 0 else np.nan


# ---------------------------------------------------------------------------
# Stage 0 metrics row
# ---------------------------------------------------------------------------

def _stage0_row(label, nav_dt, r, tpy, out_ratio, excess_days, enter_thr, is_mask, oos_mask):
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    min9 = min(aft["CAGR_IS"], aft["CAGR_OOS"])
    maxdd = pre["MaxDD_FULL"]
    w10y = aft["Worst10Y_star"]
    cy = _calendar_year_returns(nav_dt)

    v_maxdd = (maxdd < HARD_VETO_MAXDD)
    v_w10y  = (w10y < HARD_VETO_W10Y)
    v_mindd = False   # placeholder; WFE/regime veto comes in Stage 1
    veto = v_maxdd or v_w10y

    row = {
        "label": label,
        "enter_thr": enter_thr,
        "exit_thr": EXIT_THR,
        # Standard 10 metrics (after-tax)
        "CAGR_IS_at":   aft["CAGR_IS"],
        "CAGR_OOS_at":  aft["CAGR_OOS"],
        "min9_at":      min9,
        "IS_OOS_gap_pp": aft["IS_OOS_gap_pp"],
        "Sharpe_OOS":   pre["Sharpe_OOS"],
        "MaxDD_FULL":   maxdd,
        "Worst10Y_star_at": w10y,
        "P10_5Y_at":    aft["P10_5Y"],
        "Worst5Y_at":   aft["Worst5Y"],
        "Trades_yr":    aft["Trades_yr"],
        # Extra columns
        "OUT_ratio":    out_ratio,
        "excess_days":  excess_days,
        "worst_cy":     float(cy.min()),
        "worst_cy_year": int(cy.idxmin()),
        # Veto
        "veto_maxdd":   int(v_maxdd),
        "veto_w10y":    int(v_w10y),
        "VETO":         int(veto),
        "s0_pass":      int(not veto and min9 >= B3A_MIN9_REF - 1e-6),
    }
    return row


# ---------------------------------------------------------------------------
# Stage 1 full gate
# ---------------------------------------------------------------------------

def _stage1_row(s0_row, nav_dt, r, regimes, stress, is_mask, oos_mask,
                r_b3a, r_v7, enter_thr):
    label = s0_row["label"]
    ev = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                   baseline_r=r_v7)

    maxdd = s0_row["MaxDD_FULL"]
    w10y  = s0_row["Worst10Y_star_at"]
    wfe   = float(ev["wfa_WFE"])
    reg_min = float(ev["regime_min_at"])

    v_maxdd = (maxdd < HARD_VETO_MAXDD)
    v_wfe   = (wfe > HARD_VETO_WFE)
    v_w10y  = (w10y < HARD_VETO_W10Y)
    v_reg   = (reg_min < HARD_VETO_REGIME)
    veto = v_maxdd or v_wfe or v_w10y or v_reg

    boot_v7  = _block_bootstrap_compare(r, r_v7, is_mask, oos_mask)
    boot_b3a = _block_bootstrap_compare(r, r_b3a, is_mask, oos_mask)

    row = dict(s0_row)
    row.update({
        "wfa_WFE":         wfe,
        "wfa_CI95_lo":     float(ev["wfa_CI95_lo"]),
        "wfa_t_p":         float(ev["wfa_t_p"]),
        "cpcv_p10_at":     float(ev["cpcv_p10_at"]),
        "cpcv_worst_at":   float(ev["cpcv_worst_at"]),
        "cpcv_med_at":     float(ev["cpcv_med_at"]),
        "regime_min_at":   reg_min,
        # Bootstrap vs V7 baseline
        "boot_v7_P_min_better":    boot_v7.get("P_min_better", np.nan),
        "boot_v7_CI95_lo_min_pp":  boot_v7.get("CI95_lo_min_pp", np.nan),
        # Bootstrap vs B3a_k365
        "boot_b3a_P_min_better":   boot_b3a.get("P_min_better", np.nan),
        "boot_b3a_CI95_lo_min_pp": boot_b3a.get("CI95_lo_min_pp", np.nan),
        # Stage 1 veto
        "s1_veto_maxdd": int(v_maxdd),
        "s1_veto_wfe":   int(v_wfe),
        "s1_veto_w10y":  int(v_w10y),
        "s1_veto_reg":   int(v_reg),
        "S1_VETO":       int(veto),
    })
    # Regime detail
    axes = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
            "rate:rate_up", "rate:rate_down"]
    for ax in axes:
        row["regime_" + ax.replace(":", "_")] = ev["regime"].get(ax, np.nan)
    sw_order = list(stress.keys())
    for sw in sw_order:
        row["stress_%s_ret" % sw]   = ev["stress"][sw]["ret"]
        row["stress_%s_maxdd" % sw] = ev["stress"][sw]["maxdd"]
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    print("=" * 96)
    print("LEVEREXT #2: Early Re-entry Sweep  2026-06-16")
    print("DH-W1 enter_thr = {0.70, 0.65, 0.60}, exit_thr=0.30 fixed")
    print("Stack: V7 {1.40,1.40,1.05,1.00} x scale1.15 + P09 fill + C1 + k365")
    print("=" * 96)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load base shared assets ----
    print("\nLoading base shared assets (g14) ...")
    a = _load_base_shared()
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)
    print("  n=%d  n_years=%.1f  IS_END=%s  OOS_START=%s" % (n, n_years, IS_END.date(), OOS_START.date()))

    # ---- Gold/Bond 1x returns ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    # ---- Regime labels and stress masks ----
    print("  Building regime labels and stress masks ...")
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)

    # ---- Build shared dicts for all enter levels ----
    print("\nBuilding DH-W1 masks for each enter_thr ...")
    shared_by_enter = {}
    for et in ENTER_LEVELS:
        print("  enter_thr=%.2f ..." % et)
        shared_by_enter[et] = _build_shared_for_enter(et)

    # ---- Build OUT-fill params per enter level ----
    nav_by_enter = {}
    r_by_enter = {}
    tpy_by_enter = {}
    exc_by_enter = {}
    out_ratio_by_enter = {}

    for et in ENTER_LEVELS:
        sh = shared_by_enter[et]
        mask = sh["mask"]
        fund_active, wg_w, wb_w, bond_on, sofr_arr = _build_out_fill_params(
            a, mask, ret_gold, ret_bond, n, n_years, dates_dt)
        print("  Building NAV for enter_thr=%.2f ..." % et)
        nav_dt, r, tpy, exc = _build_nav_enter(
            sh, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_w, wb_w, bond_on, sofr_arr)
        nav_by_enter[et] = nav_dt
        r_by_enter[et] = r
        tpy_by_enter[et] = tpy
        exc_by_enter[et] = exc
        out_ratio_by_enter[et] = _compute_out_ratio(mask)
        print("    OUT_ratio=%.1f%%  Trades/yr=%.1f  excess_days=%d" % (
            out_ratio_by_enter[et] * 100, tpy, exc))

    # ---- Sanity check: enter=0.70 must match B3a_k365 ----
    print("\n" + "=" * 70)
    print("SANITY CHECK: enter=0.70 vs B3a_k365 reference")
    print("=" * 70)
    pre_sanity = compute_10metrics(nav_by_enter[0.70], tpy_by_enter[0.70])
    aft_sanity = _apply_aftertax(pre_sanity)
    min9_sanity = min(aft_sanity["CAGR_IS"], aft_sanity["CAGR_OOS"])
    maxdd_sanity = pre_sanity["MaxDD_FULL"]
    ok_min9  = abs(min9_sanity  - B3A_MIN9_REF)  <= SANITY_TOL_MIN9
    ok_maxdd = abs(maxdd_sanity - B3A_MAXDD_REF) <= SANITY_TOL_MAXDD
    print("  min9_at:  got=%+.4f%%  expect~%+.4f%%  -> %s"
          % (min9_sanity * 100, B3A_MIN9_REF * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD:    got=%+.4f%%  expect~%+.4f%%  -> %s"
          % (maxdd_sanity * 100, B3A_MAXDD_REF * 100, "OK" if ok_maxdd else "FAIL"))
    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED -- stopping. Investigate DH-W1 reconstruction or B3a stack.")
        sys.exit(1)
    print("  Sanity PASSED.")

    # ---- Stage 0: standard 10 metrics for all enter levels ----
    print("\n" + "=" * 96)
    print("STAGE 0: Standard 10 Metrics (after-tax x0.8273)")
    print("=" * 96)

    s0_rows = []
    for et in ENTER_LEVELS:
        lbl = "reentry_enter%.2f" % et
        row = _stage0_row(
            lbl, nav_by_enter[et], r_by_enter[et], tpy_by_enter[et],
            out_ratio_by_enter[et], exc_by_enter[et], et, is_mask, oos_mask)
        s0_rows.append(row)

    # Print Stage 0 table
    hdr = ("%-24s | %5s | %7s | %7s | %7s | %7s | %6s | %7s | %7s | %7s | %7s | %7s | %5s | %4s | %s"
           % ("label", "enter", "IS_at%", "OOS_at%", "min9%", "gap_pp",
              "Sharpe", "MaxDD%", "W10Y*%", "P10_5Y%", "Wst5Y%", "Trd/yr", "OUT%", "VETO", "pass"))
    print(hdr)
    print("-" * len(hdr))
    for row in s0_rows:
        print("%-24s | %.2f  | %+6.2f%% | %+6.2f%% | %+6.2f%% | %+6.2f  | %6.3f | %+6.2f%% | %+6.2f%% | %+6.2f%% | %+6.2f%% | %6.1f  | %4.1f%% | %4d | %s"
              % (row["label"], row["enter_thr"],
                 row["CAGR_IS_at"] * 100, row["CAGR_OOS_at"] * 100,
                 row["min9_at"] * 100, row["IS_OOS_gap_pp"],
                 row["Sharpe_OOS"],
                 row["MaxDD_FULL"] * 100, row["Worst10Y_star_at"] * 100,
                 row["P10_5Y_at"] * 100, row["Worst5Y_at"] * 100,
                 row["Trades_yr"],
                 row["OUT_ratio"] * 100,
                 row["VETO"],
                 "PASS" if row["s0_pass"] else ("veto" if row["VETO"] else "min9<B3a")))

    # ---- Stage 1: full gate for s0_pass candidates ----
    candidates = [row for row in s0_rows if row["s0_pass"]]
    print("\nStage-0 pass (veto=0 AND min9 >= +20.98%%): %d / %d" % (len(candidates), len(s0_rows)))

    # Build V7 baseline and B3a baseline NAVs for bootstrap
    print("\nBuilding V7 baseline and B3a_k365 baseline for bootstrap ...")
    # V7 baseline (no fill, no excess, cfd_excess=False)
    from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base
    sr._load_dhw1_shared()
    shared_dhw1 = sr._DHW1_SHARED
    v7_nav, r_v7, _, _ = _build_tqqq_base(shared_dhw1, dates_dt, v7_map=None,
                                           lev_scale=1.0, cfd_excess=False)

    # B3a_k365 baseline (enter=0.70, full stack)
    r_b3a = r_by_enter[0.70]

    s1_rows = []
    if candidates:
        print("\n" + "=" * 96)
        print("STAGE 1: Full Gate (WFA49 + CPCV + regime + stress + multi-metric bootstrap)")
        print("=" * 96)
        for s0_row in candidates:
            et = s0_row["enter_thr"]
            lbl = s0_row["label"]
            print("\n  Evaluating %s (enter=%.2f) ..." % (lbl, et))
            nav_dt = nav_by_enter[et]
            r = r_by_enter[et]
            row = _stage1_row(s0_row, nav_dt, r, regimes, stress, is_mask, oos_mask,
                              r_b3a, r_v7, et)
            s1_rows.append(row)
            print("    min9=%.2f%%  WFE=%.4f  CI95_lo=%.2f%%  CPCV_p10=%.2f%%"
                  "  Reg_min=%.2f%%  S1_VETO=%d"
                  % (row["min9_at"] * 100, row["wfa_WFE"],
                     row["wfa_CI95_lo"] * 100, row["cpcv_p10_at"] * 100,
                     row["regime_min_at"] * 100, row["S1_VETO"]))
            print("    Boot vs V7:  P_better=%.3f  CI95_lo=%+.2fpp"
                  % (row["boot_v7_P_min_better"], row["boot_v7_CI95_lo_min_pp"]))
            print("    Boot vs B3a: P_better=%.3f  CI95_lo=%+.2fpp"
                  % (row["boot_b3a_P_min_better"], row["boot_b3a_CI95_lo_min_pp"]))

        if s1_rows:
            print("\n" + "=" * 120)
            print("STAGE 1 FULL GATE TABLE")
            hdr2 = ("%-24s | %5s | %7s | %7s | %7s | %7s | %8s | %8s | %4s | %s"
                    % ("label", "enter", "min9%", "WFE", "CI95%", "CPCV_p10%",
                       "Reg_min%", "W10Y*%", "VETO", "boot_b3a"))
            print(hdr2)
            print("-" * len(hdr2))
            for row in s1_rows:
                print("%-24s | %.2f  | %+6.2f%% | %7.4f | %+6.2f%% | %+7.2f%% | %+7.2f%% | %+7.2f%% | %4d | P=%.3f CI95=%.2fpp"
                      % (row["label"], row["enter_thr"],
                         row["min9_at"] * 100, row["wfa_WFE"],
                         row["wfa_CI95_lo"] * 100, row["cpcv_p10_at"] * 100,
                         row["regime_min_at"] * 100, row["Worst10Y_star_at"] * 100,
                         row["S1_VETO"],
                         row["boot_b3a_P_min_better"],
                         row["boot_b3a_CI95_lo_min_pp"]))
    else:
        print("\n  No Stage-0 pass candidates; Stage 1 skipped.")

    # ---- Save CSV ----
    # Merge s0 and s1 rows (s1 contains all s0 columns plus extras)
    all_rows_dict = {}
    for row in s0_rows:
        all_rows_dict[row["enter_thr"]] = row
    for row in s1_rows:
        all_rows_dict[row["enter_thr"]] = row  # overwrite with richer s1 row

    all_rows = [all_rows_dict[et] for et in ENTER_LEVELS]
    df_out = pd.DataFrame(all_rows)
    out_csv = os.path.join(out_dir, "leverext_reentry_20260616.csv")
    df_out.to_csv(out_csv, index=False, float_format="%.6f")
    print("\nSaved CSV: %s" % out_csv)

    # ---- RETURN_BLOCK ----
    block = {
        "task": "leverext_reentry_20260616",
        "sanity_ok": True,
        "enter_levels": ENTER_LEVELS,
        "b3a_ref_min9_pct": round(B3A_MIN9_REF * 100, 4),
        "b3a_ref_maxdd_pct": round(B3A_MAXDD_REF * 100, 4),
        "stage0": [],
        "stage1": [],
    }
    for row in s0_rows:
        block["stage0"].append({
            "label":    row["label"],
            "enter_thr": row["enter_thr"],
            "CAGR_IS_at_pct":   round(row["CAGR_IS_at"] * 100, 4),
            "CAGR_OOS_at_pct":  round(row["CAGR_OOS_at"] * 100, 4),
            "min9_at_pct":      round(row["min9_at"] * 100, 4),
            "IS_OOS_gap_pp":    round(row["IS_OOS_gap_pp"], 4),
            "Sharpe_OOS":       round(row["Sharpe_OOS"], 4),
            "MaxDD_pct":        round(row["MaxDD_FULL"] * 100, 4),
            "Worst10Y_star_pct": round(row["Worst10Y_star_at"] * 100, 4),
            "P10_5Y_pct":       round(row["P10_5Y_at"] * 100, 4),
            "Worst5Y_pct":      round(row["Worst5Y_at"] * 100, 4),
            "Trades_yr":        round(row["Trades_yr"], 2),
            "OUT_ratio_pct":    round(row["OUT_ratio"] * 100, 2),
            "VETO":             row["VETO"],
            "s0_pass":          row["s0_pass"],
        })
    for row in s1_rows:
        block["stage1"].append({
            "label":        row["label"],
            "enter_thr":    row["enter_thr"],
            "min9_at_pct":  round(row["min9_at"] * 100, 4),
            "wfa_WFE":      round(row["wfa_WFE"], 4),
            "wfa_CI95_lo_pct": round(row["wfa_CI95_lo"] * 100, 4),
            "cpcv_p10_at_pct": round(row["cpcv_p10_at"] * 100, 4),
            "regime_min_at_pct": round(row["regime_min_at"] * 100, 4),
            "Worst10Y_star_pct": round(row["Worst10Y_star_at"] * 100, 4),
            "MaxDD_pct": round(row["MaxDD_FULL"] * 100, 4),
            "S1_VETO": row["S1_VETO"],
            "boot_b3a_P_min_better": round(row["boot_b3a_P_min_better"], 4),
            "boot_b3a_CI95_lo_min_pp": round(row["boot_b3a_CI95_lo_min_pp"], 4),
            "boot_v7_P_min_better": round(row["boot_v7_P_min_better"], 4),
            "boot_v7_CI95_lo_min_pp": round(row["boot_v7_CI95_lo_min_pp"], 4),
        })

    print("\n" + "=" * 96)
    print("RETURN_BLOCK")
    print("=" * 96)
    print(json.dumps(block, indent=2, ensure_ascii=False))

    print("\nDone. Output CSV: %s" % out_csv)
    return block


if __name__ == "__main__":
    main()
