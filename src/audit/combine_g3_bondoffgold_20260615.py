"""
src/audit/combine_g3_bondoffgold_20260615.py
=============================================
G3: bondOFF-day Gold fill -- native integration on B3a_k365 base.

Plan reference: MULTISTRATEGY_COMBINE_PLAN_20260615.md (Section 1, G3)

Design
------
B3a_k365 (the fixed base) uses _build_p09_nav_c1 for its OUT-day fill.
In the current C1 implementation, on OUT days where bond_on=False (bondOFF),
the w_b portion of the blend earns SOFR (cash):
    cash_yield = where(bond_on, 0.0, w_b) * sofr_daily

G3 replaces that cash portion with a Gold/cash blend:
    gold_yield  = where(bond_on, 0.0, w_b) * g * (ret_gold - FEE_GOLD/TRADING_DAYS)
    sofr_yield  = where(bond_on, 0.0, w_b) * (1-g) * sofr_daily
    total_yield = gold_yield + sofr_yield

where g in {0.0, 0.25, 0.50, 1.00}.
  g=0.0  -> identical to C1 (sanity anchor, must reproduce B3a_k365 within +/-0.05pp)
  g=0.25 -> 25% Gold / 75% SOFR on bondOFF w_b fraction
  g=0.50 -> 50% Gold / 50% SOFR on bondOFF w_b fraction
  g=1.00 -> 100% Gold on bondOFF w_b fraction (full Gold fill)

bondON days and IN days are unchanged across all g values.

Sanity anchor (mandatory)
--------------------------
g=0.0 must reproduce B3a_k365 base metrics within +/-0.05pp on min(IS,OOS) aftertax.
Known B3a_k365 base (from LEVERUP_SWEEP_RESULTS_20260612.md / MEMORY.md):
    min(IS,OOS)_aftertax >= +20.98%
    MaxDD               >= -38.20%
    Sharpe_OOS          >= 0.904
    Worst5Y             >= +0.10%
Tolerance: +/-0.05pp on min9, +/-0.50pp on MaxDD.

Survival criterion (Stage 0)
-----------------------------
  PASS if:
    - No hard veto (MaxDD >= -50%, Worst10Y* >= 0%, Regime_min >= -10%)
    - min9 improvement vs B3a_k365 base >= +0.20pp
    - MaxDD deterioration vs base <= +2.00pp

Hard veto thresholds (same as MULTISTRATEGY plan):
    MaxDD   < -50%
    Worst10Y* < 0%
    Regime_min < -10%
(WFE not evaluated at Stage 0 -- Stage 0 is metrics-only screen)

Output files
------------
  src/audit/combine_g3_bondoffgold_20260615.py  (this file)
  audit_results/combine_g3_bondoffgold_20260615.csv
  RETURN_BLOCK printed to stdout (JSON)

Constraints
-----------
  - ASCII-only prints (Windows cp932)
  - No git operations, no temp files
  - post-hoc evaluation forbidden: native integration only
  - Python timeout: 600 seconds

Native integration boundary
----------------------------
  REUSED (data/signal only):
    - _build_full_c1() from k365_recost_20260612 -> data loading and TQQQ base
    - prepare_gold_local() -> gold_1x NAV levels
    - _ret_from_nav_level() -> daily returns from NAV levels
    - FEE_GOLD, TRADING_DAYS, sofr_arr -> cost/rate constants
  NEW (G3 fill logic, implemented here):
    - _build_p09_nav_c1_g3() -> bondOFF Gold/cash blend, g-parameterised
    - _build_p09_on_base_c1_g3() -> thin wrapper matching existing signature
"""

from __future__ import annotations

import json
import os
import sys
import types

# ---------------------------------------------------------------------------
# multitasking stub (yfinance dependency)
# ---------------------------------------------------------------------------
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
from src.audit.regime_labeler_20260611 import build_regime_labels

# ---------------------------------------------------------------------------
# Import reused builders (data loading / TQQQ base only)
# ---------------------------------------------------------------------------
from src.audit.lu_cfd_recost_20260611 import (
    AFTER_TAX, SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    FEE_GOLD, FEE_BOND,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _count_fund_transitions,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)
from src.audit.k365_recost_20260612 import (
    _build_tqqq_base_param,
    EXCESS_EXTRA_K365_CENTRE,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom,
)

# ---------------------------------------------------------------------------
# B3a_k365 fixed parameters (from MULTISTRATEGY_COMBINE_PLAN_20260615.md)
# ---------------------------------------------------------------------------
B3A_V7_MAP    = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE = 1.15
B3A_EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025 (k365 centre)

# Sanity anchor targets (from MEMORY.md / LEVERUP_SWEEP_RESULTS_20260612.md)
ANCHOR_MIN9    = 0.2098   # +20.98%
ANCHOR_MAXDD   = -0.3820  # -38.20%
TOL_MIN9       = 0.0005   # +/-0.05pp
TOL_MAXDD      = 0.0050   # +/-0.50pp

# Survival thresholds
MIN9_IMPROVE_THRESHOLD  = 0.0020   # >= +0.20pp improvement vs base
MAXDD_DEGRADE_THRESHOLD = 0.0200   # <= +2.00pp degradation (less negative is ok)

# Hard veto
HARD_VETO_MAXDD  = -0.50
HARD_VETO_W10Y   = 0.0
HARD_VETO_REGIME = -0.10

# Gold fill fractions to sweep
G_VALUES = [0.0, 0.25, 0.50, 1.00]


# ---------------------------------------------------------------------------
# G3 native fill: _build_p09_nav_c1_g3
# ---------------------------------------------------------------------------

def _build_p09_nav_c1_g3(r_base, ret_gold, ret_bond, fund_active,
                          w_g, w_b, bond_on, sofr_arr, gold_fraction=0.0):
    """G3 variant of _build_p09_nav_c1.

    On OUT days where bond_on=True:  normal bond return (unchanged from C1).
    On OUT days where bond_on=False: w_b portion splits into
        gold_fraction * Gold1x  +  (1-gold_fraction) * SOFR
    instead of 100% SOFR (which is the C1 baseline, gold_fraction=0).

    Parameters
    ----------
    r_base : np.ndarray  -- TQQQ-base daily returns (IN-day backbone)
    ret_gold : np.ndarray -- Gold 1x daily returns
    ret_bond : np.ndarray -- Bond 1x daily returns
    fund_active : np.ndarray(bool) -- True on OUT days (post-lag)
    w_g : np.ndarray -- inverse-vol Gold weight (Gold leg, always on when OUT)
    w_b : np.ndarray -- inverse-vol Bond weight (Bond leg, conditional)
    bond_on : np.ndarray(bool) -- True when bond_mom252>0 (bond timing gate)
    sofr_arr : np.ndarray -- SOFR daily fraction (annual/252)
    gold_fraction : float in [0,1] -- fraction of idle w_b invested in Gold on bondOFF days

    Returns
    -------
    (nav, r, eff_active) matching _build_p09_nav_c1 signature.
    """
    bond_on   = np.asarray(bond_on, dtype=bool)
    sofr_arr  = np.asarray(sofr_arr, float)
    ret_gold  = np.asarray(ret_gold, float)
    ret_bond  = np.asarray(ret_bond, float)
    w_g       = np.asarray(w_g, float)
    w_b       = np.asarray(w_b, float)
    g         = float(gold_fraction)

    # Effective bond weight (zero on bondOFF days)
    w_b_eff = np.where(bond_on, w_b, 0.0)

    # Fee: Gold leg (always), Bond leg (only when bondON).
    # On bondOFF days, the Gold-fill portion of w_b also pays FEE_GOLD (pro-rata).
    # cash portion pays no fee.
    fee_gold_main  = w_g * FEE_GOLD / TRADING_DAYS          # always-on Gold leg
    fee_bond_leg   = w_b_eff * FEE_BOND / TRADING_DAYS      # bondON only
    fee_gold_fill  = np.where(bond_on, 0.0, w_b) * g * FEE_GOLD / TRADING_DAYS  # bondOFF Gold fill
    fee_daily = fee_gold_main + fee_bond_leg + fee_gold_fill

    # bondON days: normal gold + bond returns (same as C1)
    # bondOFF days: gold return on w_g (always) + gold_fraction*ret_gold + (1-g)*sofr on w_b
    cash_yield = np.where(bond_on, 0.0, w_b) * (1.0 - g) * sofr_arr
    gold_fill  = np.where(bond_on, 0.0, w_b) * g * ret_gold

    r_blend = (w_g * ret_gold
               + w_b_eff * ret_bond
               + cash_yield
               + gold_fill
               - fee_daily)

    r = np.where(fund_active, r_blend, r_base)
    r = np.clip(r, -0.999, None)
    nav = np.cumprod(1.0 + r)
    eff_active = fund_active.copy()
    return nav, r, eff_active


def _build_p09_on_base_c1_g3(r_base, ret_gold, ret_bond, fund_active,
                               wg, wb, bond_on, sofr_arr,
                               dates_dt, tpy_base, n_years,
                               gold_fraction=0.0):
    """Thin wrapper matching _build_p09_on_base_c1 but with G3 fill."""
    nav_arr, r_g3, eff_active = _build_p09_nav_c1_g3(
        r_base, ret_gold, ret_bond, fund_active,
        wg, wb, bond_on, sofr_arr,
        gold_fraction=gold_fraction)
    nav_dt = pd.Series(nav_arr, index=dates_dt)
    flips = _count_fund_transitions(eff_active)
    tpy = tpy_base + flips / n_years
    return nav_dt, r_g3, tpy


def _build_full_g3(shared, dates_dt, n_years,
                   ret_gold, ret_bond, fund_active,
                   wg, wb, bond_on, sofr_arr,
                   gold_fraction=0.0):
    """Build complete G3 NAV (B3a_k365 TQQQ base + G3 OUT fill)."""
    base_nav, r_base, tpy_b, exc = _build_tqqq_base_param(
        shared, dates_dt,
        v7_map=B3A_V7_MAP,
        lev_scale=B3A_LEV_SCALE,
        excess_extra=B3A_EXCESS_EXTRA)
    nav_dt, r, tpy = _build_p09_on_base_c1_g3(
        r_base, ret_gold, ret_bond, fund_active,
        wg, wb, bond_on, sofr_arr,
        dates_dt, tpy_b, n_years,
        gold_fraction=gold_fraction)
    return nav_dt, r, tpy, exc


def _min_at(aft):
    return min(float(aft["CAGR_IS"]), float(aft["CAGR_OOS"]))


def _hard_veto(pre, aft, regime_min_at):
    """Apply hard veto flags. Returns (veto_bool, detail_dict)."""
    v_maxdd = float(pre["MaxDD_FULL"]) < HARD_VETO_MAXDD
    v_w10y  = float(aft["Worst10Y_star"]) < HARD_VETO_W10Y
    v_reg   = float(regime_min_at) < HARD_VETO_REGIME
    veto    = v_maxdd or v_w10y or v_reg
    return veto, {"maxdd": v_maxdd, "w10y": v_w10y, "reg": v_reg, "VETO": veto}


# ---------------------------------------------------------------------------
# Regime-min helper (simplified: use min across all regime CAGRs)
# ---------------------------------------------------------------------------

def _regime_min_cagr(nav_dt, regimes):
    """Compute minimum CAGR across all regime windows for hard-veto check.

    Uses after-tax CAGR inside each regime window (approx: apply AFTER_TAX
    multiplicative factor to annualised raw return).
    Returns float (min CAGR, annualised aftertax).
    """
    r = nav_dt.pct_change().fillna(0).values
    dates_dt = nav_dt.index
    n_years_full = len(dates_dt) / float(TRADING_DAYS)

    min_cagr = np.inf
    for label, mask in regimes.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < 60:
            continue
        r_seg = r[mask]
        nav_seg = np.cumprod(1.0 + r_seg)
        n_days = len(r_seg)
        if n_days < 1:
            continue
        total_ret = float(nav_seg[-1]) - 1.0
        ann = float((1.0 + total_ret) ** (TRADING_DAYS / float(n_days)) - 1.0)
        ann_at = ann * AFTER_TAX
        if ann_at < min_cagr:
            min_cagr = ann_at
    if np.isinf(min_cagr):
        min_cagr = 0.0
    return min_cagr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("G3: bondOFF-day Gold fill -- B3a_k365 native integration  2026-06-15")
    print("Gold fractions: g = %s" % str(G_VALUES))
    print("B3a base: v7_map=%s  lev_scale=%.2f  excess_extra=%.4f%%/yr"
          % (str(B3A_V7_MAP), B3A_LEV_SCALE, B3A_EXCESS_EXTRA * 100))
    print("=" * 100)

    # ---- Load shared data ----
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask_arr = np.asarray(shared["mask"], dtype=float)
    dates    = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)
    sofr_arr = np.asarray(a["sofr"], float)

    # ---- Gold / Bond 1x legs ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x  = np.asarray(prepare_gold_local(dates), float)
    bond_1x  = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    # ---- OUT-day mask (LAG_DAYS shift for causal execution) ----
    out_mask_arr = (mask_arr < 0.5)
    fund_active  = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    # ---- Inverse-vol weights (Gold / Bond, 63-day window) ----
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)

    # ---- Bond timing gate (bond_mom252, GATE_DELAY=2 causal shift) ----
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on   = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    # ---- Regime labels for hard-veto (simplified: trend/vol/rate regimes) ----
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)

    # Diagnostics
    out_bond_off = fund_active & (~bond_on.astype(bool))
    n_out_bondoff = int(out_bond_off.sum())
    print("\nOUT-and-bondOFF days: %d of %d (%.1f%%)" % (n_out_bondoff, n, 100.0 * n_out_bondoff / n))
    print("These are the days where Gold fill (g>0) replaces part of SOFR yield.\n")

    # =========================================================================
    # SANITY ANCHOR: g=0.0 must reproduce B3a_k365 base within tolerance
    # =========================================================================
    print("=" * 100)
    print("SANITY ANCHOR CHECK  (g=0.0 must reproduce B3a_k365 base)")
    print("  Target: min9_at ~ +%.2f%%  MaxDD ~ %+.2f%%"
          % (ANCHOR_MIN9 * 100, ANCHOR_MAXDD * 100))
    print("  Tolerance: min9 +/-%.2fpp  MaxDD +/-%.2fpp"
          % (TOL_MIN9 * 100, TOL_MAXDD * 100))
    print("=" * 100)

    anchor_nav, anchor_r, anchor_tpy, anchor_exc = _build_full_g3(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        gold_fraction=0.0)

    anchor_pre = compute_10metrics(anchor_nav, anchor_tpy)
    anchor_aft = _apply_aftertax(anchor_pre)
    anchor_min9  = _min_at(anchor_aft)
    anchor_maxdd = float(anchor_pre["MaxDD_FULL"])

    ok_min9  = abs(anchor_min9  - ANCHOR_MIN9)  <= TOL_MIN9
    ok_maxdd = abs(anchor_maxdd - ANCHOR_MAXDD) <= TOL_MAXDD

    print("  g=0.0 min9_at  = %+.4f%%  (expect %+.4f%%) -> %s"
          % (anchor_min9 * 100, ANCHOR_MIN9 * 100, "OK" if ok_min9 else "FAIL"))
    print("  g=0.0 MaxDD    = %+.4f%%  (expect %+.4f%%) -> %s"
          % (anchor_maxdd * 100, ANCHOR_MAXDD * 100, "OK" if ok_maxdd else "FAIL"))

    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED -- G3 Gold fill injection has a bug. Halting.")
        print("  Possible causes:")
        print("    1. fee_gold_fill double-counting the main w_g Gold leg")
        print("    2. gold_fill added on non-OUT days")
        print("    3. B3a_k365 base parameters do not match (v7_map/lev_scale/excess_extra)")
        sys.exit(1)

    print("\nSANITY PASSED. Proceeding with G3 sweep.\n")

    # =========================================================================
    # G3 SWEEP: g in {0.0, 0.25, 0.50, 1.00}
    # =========================================================================
    print("=" * 100)
    print("G3 STAGE-0 SWEEP  (standard 10 metrics + calendar year min)")
    print("=" * 100)

    results = {}
    for g in G_VALUES:
        label = "G3_g%.2f" % g
        print("\nBuilding %s ..." % label)
        nav_dt, r, tpy, exc = _build_full_g3(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            gold_fraction=g)

        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)
        cy  = _calendar_year_returns(nav_dt)
        min9  = _min_at(aft)
        maxdd = float(pre["MaxDD_FULL"])
        w10y  = float(aft["Worst10Y_star"])
        worst5y = float(aft["Worst5Y"])
        sharpe  = float(pre["Sharpe_OOS"])
        w5y_worst_cy = float(cy.min())
        w5y_worst_yr = int(cy.idxmin())

        regime_min = _regime_min_cagr(nav_dt, regimes)
        veto_flag, veto_detail = _hard_veto(pre, aft, regime_min)

        # Survival criterion vs B3a base (anchor g=0.0)
        delta_min9  = (min9 - anchor_min9) * 100.0   # pp improvement
        delta_maxdd = (maxdd - anchor_maxdd) * 100.0  # pp change (positive = worse)
        survive = (not veto_flag
                   and delta_min9 >= MIN9_IMPROVE_THRESHOLD * 100
                   and delta_maxdd <= MAXDD_DEGRADE_THRESHOLD * 100)

        results[label] = {
            "g": g,
            "label": label,
            "CAGR_IS_at": float(aft["CAGR_IS"]),
            "CAGR_OOS_at": float(aft["CAGR_OOS"]),
            "min9_at": min9,
            "IS_OOS_gap_pp": float(aft["IS_OOS_gap_pp"]),
            "Sharpe_OOS": sharpe,
            "MaxDD_FULL": maxdd,
            "Worst10Y_star_at": w10y,
            "P10_5Y_at": float(aft["P10_5Y"]),
            "Worst5Y_at": worst5y,
            "Trades_yr": float(aft["Trades_yr"]),
            "worst_cy_ret": w5y_worst_cy,
            "worst_cy_year": w5y_worst_yr,
            "regime_min_at": regime_min,
            "excess_days": exc,
            "n_out_bondoff": n_out_bondoff,
            "delta_min9_pp": round(delta_min9, 4),
            "delta_maxdd_pp": round(delta_maxdd, 4),
            "veto_maxdd": int(veto_detail["maxdd"]),
            "veto_w10y": int(veto_detail["w10y"]),
            "veto_reg": int(veto_detail["reg"]),
            "VETO": int(veto_flag),
            "SURVIVE": int(survive),
        }

        print("  min9_at=%+.3f%%  MaxDD=%+.2f%%  W10Y*=%+.2f%%  Worst5Y=%+.2f%%  Sharpe=%.3f"
              % (min9 * 100, maxdd * 100, w10y * 100, worst5y * 100, sharpe))
        print("  delta_min9=%+.3fpp  delta_maxdd=%+.3fpp  VETO=%s  SURVIVE=%s"
              % (delta_min9, delta_maxdd, "YES" if veto_flag else "no",
                 "YES" if survive else "no"))

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 100)
    print("G3 STAGE-0 SUMMARY TABLE")
    print("%-14s | %8s | %8s | %8s | %8s | %8s | %9s | %8s | %8s | %9s | %-8s"
          % ("label", "min9_at%", "MaxDD%", "W10Y*%", "Worst5Y%", "Sharpe",
             "d_min9pp", "d_MDD%pp", "VETO", "SURVIVE", "w_cy"))
    print("-" * 120)
    for g in G_VALUES:
        label = "G3_g%.2f" % g
        r = results[label]
        print("%-14s | %+7.3f%% | %+7.2f%% | %+7.2f%% | %+7.2f%% | %7.3f | %+8.3f | %+7.3f | %-8s | %-9s | %+.2f%%(%d)"
              % (label,
                 r["min9_at"] * 100,
                 r["MaxDD_FULL"] * 100,
                 r["Worst10Y_star_at"] * 100,
                 r["Worst5Y_at"] * 100,
                 r["Sharpe_OOS"],
                 r["delta_min9_pp"],
                 r["delta_maxdd_pp"],
                 "VETO" if r["VETO"] else "no",
                 "PASS" if r["SURVIVE"] else "fail",
                 r["worst_cy_ret"] * 100,
                 r["worst_cy_year"]))

    # =========================================================================
    # CAGR-DD tradeoff narrative
    # =========================================================================
    print("\n" + "=" * 100)
    print("CAGR-DD TRADEOFF (vs B3a_k365 base at g=0.0)")
    print("  B3a_k365 base: min9=%+.3f%%  MaxDD=%+.2f%%"
          % (anchor_min9 * 100, anchor_maxdd * 100))
    print("-" * 60)
    for g in G_VALUES:
        label = "G3_g%.2f" % g
        r = results[label]
        print("  g=%.2f: min9 delta=%+.3fpp  MaxDD delta=%+.3fpp  (MaxDD=%+.2f%%)"
              % (g, r["delta_min9_pp"], r["delta_maxdd_pp"], r["MaxDD_FULL"] * 100))

    # =========================================================================
    # CSV output
    # =========================================================================
    print("\nBuilding CSV ...")
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combine_g3_bondoffgold_20260615.csv")

    rows = []
    for g in G_VALUES:
        label = "G3_g%.2f" % g
        rows.append(results[label])

    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # RETURN BLOCK (JSON)
    # =========================================================================
    def _rblock(label):
        r = results[label]
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in r.items()}

    sanity_result = {
        "g0_min9_at_pct":  round(anchor_min9 * 100, 4),
        "g0_MaxDD_pct":    round(anchor_maxdd * 100, 4),
        "anchor_target_min9_pct": round(ANCHOR_MIN9 * 100, 4),
        "anchor_target_MaxDD_pct": round(ANCHOR_MAXDD * 100, 4),
        "ok_min9":  ok_min9,
        "ok_maxdd": ok_maxdd,
        "SANITY_PASS": bool(ok_min9 and ok_maxdd),
    }

    survivors = [g for g in G_VALUES
                 if results["G3_g%.2f" % g]["SURVIVE"]]

    block = {
        "meta": {
            "script": "combine_g3_bondoffgold_20260615.py",
            "date": "2026-06-15",
            "base": "B3a_k365",
            "g_values": G_VALUES,
            "n_out_bondoff": n_out_bondoff,
            "n_total_days": n,
            "pct_out_bondoff": round(100.0 * n_out_bondoff / n, 2),
            "survival_criterion": {
                "no_hard_veto": True,
                "min9_improve_pp": MIN9_IMPROVE_THRESHOLD * 100,
                "maxdd_degrade_max_pp": MAXDD_DEGRADE_THRESHOLD * 100,
            },
        },
        "sanity_anchor": sanity_result,
        "g_results": {
            ("G3_g%.2f" % g): _rblock("G3_g%.2f" % g)
            for g in G_VALUES
        },
        "survivors_g": survivors,
        "stage0_verdict": (
            "SURVIVORS_FOUND" if survivors else "NO_SURVIVORS_STAGE0_CLOSE"
        ),
    }

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
