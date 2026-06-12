"""
src/audit/leverup_b1c1_20260612.py
===================================
Two parallel investigations on the DH-W1 + P09 framework:

B1 - LU2 full-gate evaluation
  LU2 (uniform IN NASDAQ leverage x1.15, CFD >3x excess penalty) is subjected to
  the same comprehensive evaluation protocol used for P09_TQQQ in extended_eval:
  WFA (canonical g1_wfa windows), CPCV (N=10, k=2, embargo=21), regime-stratified
  CAGR, stress windows, paired block bootstrap vs V7_TQQQ baseline.
  Previously LU2_cfd had only WFA; this adds the full suite.

C1 - SOFR cash yield on bond-OFF days
  When the P09 bond-timing filter is OFF (bond_mom252 <= 0), the bond weight w_b
  is held in cash (return 0 in the current implementation).  In real operation
  that cash sleeve earns approximately SOFR in a MMF or short-duration fund.
  C1 adds  w_b * sofr_daily  on bond-OFF OUT days.

  sofr_arr in _build_nav_v7_tqqq is already a daily rate (verified below).
  No spread is added (MMF ~ SOFR flat; TER/spread excluded => conservative optimism
  acknowledged in comment).

  The C1 yield ONLY affects fund_active (OUT) days because _build_p09_nav applies
  r = np.where(fund_active, r_blend, r_base) -- IN days use r_base unchanged.

Reuses (imports only, no reimplementation):
  lu_cfd_recost_20260611  : _build_tqqq_base, _build_p09_on_base, _metrics_pack, LU1_MAP
  extended_eval_20260611  : _eval_one, _cpcv_dist, _regime_cagr
  run_p09_tqqq_validate   : _run_wfa, _block_bootstrap_compare, LU1_MAP, LU2_SCALE, AFTER_TAX
  run_p02_p09_backtest    : GATE_DELAY, _load_macro_signal, _build_p09_nav
  run_p01_backtest        : LAG_DAYS, TRADING_DAYS, _ret_from_nav_level,
                            _inverse_vol_weights, _calendar_year_returns, _apply_aftertax,
                            _count_fund_transitions

Outputs:
  audit_results/leverup_b1c1_20260612.csv  -- long table (candidate x metric)
  RETURN_BLOCK JSON printed to stdout

ASCII-only prints (Windows cp932). Does NOT commit.
"""

from __future__ import annotations

import os
import sys
import types
import json

# ---- multitasking stub (yfinance dependency) --------------------------------
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

# ---- validated builders (single source of truth) ----------------------------
from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base, _build_p09_on_base, _metrics_pack, LU1_MAP,
    AFTER_TAX,
)
from src.audit.extended_eval_20260611 import (
    _eval_one, _cpcv_dist, _regime_cagr,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _run_wfa, _block_bootstrap_compare, LU2_SCALE, _cagr_seg, _maxdd_from_returns,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, _build_p09_nav,
    FEE_GOLD, FEE_BOND,
)


# ---------------------------------------------------------------------------
# C1: SOFR cash yield on bond-OFF OUT days
# ---------------------------------------------------------------------------

def _build_p09_nav_c1(r_base, ret_gold, ret_bond, fund_active, w_g, w_b, bond_on, sofr_arr):
    """C1 variant of _build_p09_nav.

    Same as P09 but on bond-OFF days (bond_on=False) the idle w_b portion earns
    sofr_daily instead of 0.

    sofr_arr is already a per-day rate (annual/252 already applied by strategy_runners).
    Verification: _build_nav_v7_tqqq line 115 uses
        borrow = max(L-1,0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
    where SWAP_SPREAD=0.005 is divided by TRADING_DAYS (annualized) while sofr_arr
    is added directly -> sofr_arr is the daily fraction (annual_rate / 252).
    No spread added here (MMF ~ SOFR flat; omitting TER / bid-ask is a mild
    optimism but within 2-3 bp/yr and conservative relative to actual MMF yield
    post-expense-ratio which is typically SOFR - 5..15bp, i.e. still below SOFR).

    Returns (nav, r, eff_active) matching _build_p09_nav signature.
    """
    bond_on = np.asarray(bond_on, dtype=bool)
    sofr_arr = np.asarray(sofr_arr, float)
    w_b_eff = np.where(bond_on, w_b, 0.0)
    fee_daily = (w_g * FEE_GOLD + w_b_eff * FEE_BOND) / TRADING_DAYS
    # Bond-OFF days: bond sleeve earns SOFR (MMF-like); bond fees not charged
    # Bond-ON days: normal bond return - fee
    cash_yield = np.where(bond_on, 0.0, w_b) * sofr_arr
    r_blend = w_g * ret_gold + w_b_eff * ret_bond + cash_yield - fee_daily
    r = np.where(fund_active, r_blend, r_base)
    r = np.clip(r, -0.999, None)
    nav = np.cumprod(1.0 + r)
    eff_active = fund_active.copy()
    return nav, r, eff_active


def _build_p09_on_base_c1(r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
                           sofr_arr, dates_dt, tpy_base, n_years):
    """Thin wrapper matching _build_p09_on_base but uses C1 fill."""
    nav_arr, r_c1, eff_active = _build_p09_nav_c1(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr)
    nav_dt = pd.Series(nav_arr, index=dates_dt)
    flips = _count_fund_transitions(eff_active)
    tpy = tpy_base + flips / n_years
    return nav_dt, r_c1, tpy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 96)
    print("LEVERUP B1+C1 FULL-GATE AUDIT  2026-06-12")
    print("B1: LU2_cfd full evaluation (WFA+CPCV+regime+stress+bootstrap)")
    print("C1: SOFR cash yield on bond-OFF OUT days")
    print("=" * 96)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    is_mask = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    # ---- Regime labels and stress masks (same as extended_eval) ----
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)

    # ---- Gold/Bond 1x + OUT-fill machinery (same as extended_eval lines 171-185) ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]

    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    # sofr_arr: daily fraction (verified: used as-is in borrow = max(L-1,0)*(sofr_arr + SWAP/252))
    sofr_arr = np.asarray(a["sofr"], float)

    # ---- Verify sofr is daily (print a few values to confirm scale) ----
    sofr_nonzero = sofr_arr[sofr_arr > 0]
    print("\nSOFR daily rate verification (should be ~0.0001..0.0002 range for 3-5% annual):")
    print("  mean=%.6f  median=%.6f  max=%.6f  -> annual equiv mean=%.2f%%"
          % (float(sofr_nonzero.mean()), float(np.median(sofr_nonzero)),
             float(sofr_nonzero.max()), float(sofr_nonzero.mean()) * 252 * 100))

    # ---- Count OUT + bondOFF days (expected ~3074, ~23.3% of total) ----
    out_bond_off = fund_active & (~bond_on.astype(bool))
    n_out_bondoff = int(out_bond_off.sum())
    print("\nOUT-and-bondOFF day count: %d of %d total days (%.1f%%)"
          % (n_out_bondoff, n, 100.0 * n_out_bondoff / n))
    if abs(n_out_bondoff - 3074) > 100:
        print("  WARNING: expected ~3074 but got %d -- check bond_on construction" % n_out_bondoff)
    else:
        print("  OK: within 100 days of expected 3074")

    # =========================================================================
    # Build 7 NAV series
    # =========================================================================
    print("\nBuilding 7 NAV variants ...")

    # 1. V7_TQQQ (baseline only, no P09 fill)
    v7_nav, r_v7, tpy_v7, exc_v7 = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    # 2. P09_TQQQ (standard fill, no C1)
    p09_base_nav, r_p09base, tpy_p09b, exc_p09 = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)
    p09_nav, r_p09, tpy_p09 = _build_p09_on_base(
        r_p09base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        dates_dt, tpy_p09b, n_years)

    # 3. P09_C1 (C1 fill)
    p09c1_nav, r_p09c1, tpy_p09c1 = _build_p09_on_base_c1(
        r_p09base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        sofr_arr, dates_dt, tpy_p09b, n_years)

    # 4. LU1_cfd (strong map + CFD excess penalty, standard fill)
    lu1_base_nav, r_lu1base, tpy_lu1b, exc_lu1 = _build_tqqq_base(
        shared, dates_dt, v7_map=LU1_MAP, lev_scale=1.0, cfd_excess=True)
    lu1_nav, r_lu1, tpy_lu1 = _build_p09_on_base(
        r_lu1base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        dates_dt, tpy_lu1b, n_years)

    # 5. LU1_C1 (strong map + CFD excess penalty, C1 fill)
    lu1c1_nav, r_lu1c1, tpy_lu1c1 = _build_p09_on_base_c1(
        r_lu1base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        sofr_arr, dates_dt, tpy_lu1b, n_years)

    # 6. LU2_cfd (uniform lev x1.15 + CFD excess penalty, standard fill)
    lu2_base_nav, r_lu2base, tpy_lu2b, exc_lu2 = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=LU2_SCALE, cfd_excess=True)
    lu2_nav, r_lu2, tpy_lu2 = _build_p09_on_base(
        r_lu2base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        dates_dt, tpy_lu2b, n_years)

    # 7. LU2_C1 (uniform lev x1.15 + CFD excess penalty, C1 fill)
    lu2c1_nav, r_lu2c1, tpy_lu2c1 = _build_p09_on_base_c1(
        r_lu2base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        sofr_arr, dates_dt, tpy_lu2b, n_years)

    print("  Excess (L>3x) days: V7=%d  LU1=%d  LU2=%d  (of %d total)"
          % (exc_v7, exc_lu1, exc_lu2, n))

    # =========================================================================
    # Metrics (standard 10, pretax + aftertax)
    # =========================================================================
    print("\nComputing standard-10 metrics for all 7 variants ...")
    navs = {
        "V7_TQQQ": (v7_nav, tpy_v7, exc_v7),
        "P09_TQQQ": (p09_nav, tpy_p09, exc_p09),
        "P09_C1": (p09c1_nav, tpy_p09c1, exc_p09),
        "LU1_cfd": (lu1_nav, tpy_lu1, exc_lu1),
        "LU1_C1": (lu1c1_nav, tpy_lu1c1, exc_lu1),
        "LU2_cfd": (lu2_nav, tpy_lu2, exc_lu2),
        "LU2_C1": (lu2c1_nav, tpy_lu2c1, exc_lu2),
    }
    order = ["V7_TQQQ", "P09_TQQQ", "P09_C1", "LU1_cfd", "LU1_C1", "LU2_cfd", "LU2_C1"]

    def _min_at(aft):
        return min(aft["CAGR_IS"], aft["CAGR_OOS"])

    # For each variant compute metrics
    packs = {}
    for label in order:
        nav_dt, tpy, exc = navs[label]
        pre = compute_10metrics(nav_dt, tpy)
        # _apply_aftertax is available from run_p01_backtest via lu_cfd_recost import
        from src.audit.run_p01_backtest_20260611 import _apply_aftertax
        aft = _apply_aftertax(pre)
        cy = _calendar_year_returns(nav_dt)
        worst_cy = float(cy.min())
        worst_cy_year = int(cy.idxmin())
        packs[label] = {"pre": pre, "aft": aft, "wcy": worst_cy, "wcyy": worst_cy_year,
                        "tpy": tpy, "excess_days": exc}

    # =========================================================================
    # Sanity check (known values from prior scripts)
    # =========================================================================
    print("\n" + "=" * 96)
    print("SANITY CHECKS vs known values from prior audit scripts")
    print("=" * 96)

    # V7_TQQQ 2022 calendar return ~ -0.27%
    r_v7_series = pd.Series(r_v7, index=dates_dt)
    cy2022_v7 = float(np.prod(1.0 + r_v7_series[dates_dt.year == 2022].values) - 1.0)
    print("V7_TQQQ 2022 calendar return: %+.2f%% (expect ~ -0.27%%)" % (100 * cy2022_v7))
    if abs(cy2022_v7 * 100 - (-0.27)) > 0.5:
        print("  MISMATCH: expected -0.27%%, got %+.2f%% -- check V7 base construction" % (100 * cy2022_v7))
    else:
        print("  OK")

    # P09_TQQQ min_at ~ +17.51%
    p09_min_at = _min_at(packs["P09_TQQQ"]["aft"])
    print("P09_TQQQ min_at: %+.2f%% (expect ~ +17.51%%)" % (100 * p09_min_at))
    if abs(p09_min_at * 100 - 17.51) > 0.5:
        print("  MISMATCH: expected +17.51%%, got %+.2f%% -- investigate" % (100 * p09_min_at))
    else:
        print("  OK")

    # LU2_cfd min_at ~ +18.83%, MaxDD ~ -38.7%, WFE ~ 0.988, CI95_lo ~ +19.74%
    lu2_min_at = _min_at(packs["LU2_cfd"]["aft"])
    lu2_maxdd = packs["LU2_cfd"]["pre"]["MaxDD_FULL"]
    print("LU2_cfd min_at: %+.2f%% (expect ~ +18.83%%)" % (100 * lu2_min_at))
    print("LU2_cfd MaxDD: %+.2f%% (expect ~ -38.7%%)" % (100 * lu2_maxdd))
    if abs(lu2_min_at * 100 - 18.83) > 0.5:
        print("  MISMATCH min_at")
    else:
        print("  min_at OK")
    if abs(lu2_maxdd * 100 - (-38.7)) > 1.0:
        print("  MISMATCH MaxDD")
    else:
        print("  MaxDD OK")

    # =========================================================================
    # Full evaluation via _eval_one (WFA + CPCV + regime + stress + bootstrap)
    # =========================================================================
    print("\n" + "=" * 96)
    print("FULL EVALUATION (_eval_one) for all 7 variants ...")
    print("baseline for bootstrap = V7_TQQQ")
    print("=" * 96)

    returns_map = {
        "V7_TQQQ": r_v7,
        "P09_TQQQ": r_p09,
        "P09_C1": r_p09c1,
        "LU1_cfd": r_lu1,
        "LU1_C1": r_lu1c1,
        "LU2_cfd": r_lu2,
        "LU2_C1": r_lu2c1,
    }

    evals = {}
    for label in order:
        nav_dt, tpy, exc = navs[label]
        r = returns_map[label]
        bl = None if label == "V7_TQQQ" else r_v7
        print("  Evaluating %s ..." % label)
        evals[label] = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                                 baseline_r=bl)

    # ---- WFE / CI95_lo sanity for LU2_cfd ----
    lu2_wfe = evals["LU2_cfd"]["wfa_WFE"]
    lu2_ci95 = evals["LU2_cfd"]["wfa_CI95_lo"]
    print("\nLU2_cfd WFE=%.4f (expect ~0.988)  CI95_lo=%+.4f%% (expect ~+19.74%%)"
          % (float(lu2_wfe), float(lu2_ci95) * 100))
    if abs(float(lu2_wfe) - 0.988) > 0.02:
        print("  WFE MISMATCH -- investigate")
    else:
        print("  WFE OK")
    if abs(float(lu2_ci95) * 100 - 19.74) > 0.5:
        print("  CI95_lo MISMATCH -- investigate")
    else:
        print("  CI95_lo OK")

    # =========================================================================
    # Hard-veto flags
    # =========================================================================
    print("\n" + "=" * 96)
    print("HARD-VETO EVALUATION")
    print("Criteria: MaxDD_FULL < -50% OR WFE > 1.5 OR Worst10Y_star_at < 0 OR Regime_min < -10%")
    print("=" * 96)
    print("%-10s | MaxDD<-50 | WFE>1.5 | W10Y_at<0 | Reg_min<-10%% | VETO" % "label")
    print("-" * 80)
    veto_map = {}
    for label in order:
        p = packs[label]["pre"]
        aft = packs[label]["aft"]
        ev = evals[label]
        maxdd = p["MaxDD_FULL"]
        wfe = ev["wfa_WFE"]
        w10y_at = aft["Worst10Y_star"]
        reg_min = ev["regime_min_at"]

        v_maxdd = maxdd < -0.50
        v_wfe = float(wfe) > 1.5
        v_w10y = w10y_at < 0.0
        v_reg = reg_min < -0.10
        veto = v_maxdd or v_wfe or v_w10y or v_reg

        veto_map[label] = {
            "maxdd_veto": v_maxdd, "wfe_veto": v_wfe,
            "w10y_veto": v_w10y, "reg_veto": v_reg, "VETO": veto,
        }

        print("%-10s | %-9s | %-7s | %-9s | %-13s | %s"
              % (label,
                 "VETO" if v_maxdd else "pass",
                 "VETO" if v_wfe else "pass",
                 "VETO" if v_w10y else "pass",
                 "VETO" if v_reg else "pass",
                 "**VETO**" if veto else "PASS"))

    # =========================================================================
    # C1 effect summary (3 pairs, min(IS,OOS) aftertax diff pp)
    # =========================================================================
    print("\n" + "=" * 96)
    print("C1 EFFECT SUMMARY (min(IS,OOS) aftertax delta pp)")
    print("Expected: +0.5 to +0.9 pp per pair")
    print("=" * 96)
    pairs = [
        ("P09_C1", "P09_TQQQ"),
        ("LU1_C1", "LU1_cfd"),
        ("LU2_C1", "LU2_cfd"),
    ]
    c1_effects = {}
    for c1_label, base_label in pairs:
        d_pp = (_min_at(packs[c1_label]["aft"]) - _min_at(packs[base_label]["aft"])) * 100.0
        c1_effects[c1_label] = d_pp
        note = ""
        if d_pp < 0.3:
            note = " [below expected +0.5pp -- bond-OFF days may be fewer or SOFR low]"
        elif d_pp > 1.5:
            note = " [above expected +0.9pp -- check for double-counting]"
        print("  %s - %s : min_at delta = %+.3f pp%s" % (c1_label, base_label, d_pp, note))

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 96)
    print("FULL RESULTS (after-tax CAGR; Sharpe/MaxDD pretax)")
    print("=" * 96)
    hdr = ("%-10s | %9s | %10s | %8s | %7s | %8s | %10s | %8s | %7s | %8s | %8s"
           % ("label", "CAGR_IS_at", "CAGR_OOS_at", "min_at", "Sharpe",
              "MaxDD", "Worst10Y*_at", "P10_5Y_at", "Trd/yr", "WFE", "CI95_lo"))
    print(hdr)
    print("-" * len(hdr))
    for label in order:
        aft = packs[label]["aft"]
        pre = packs[label]["pre"]
        mn = _min_at(aft)
        ev = evals[label]
        wfe = ev["wfa_WFE"]
        ci = ev["wfa_CI95_lo"]
        print("%-10s | %+8.2f%% | %+9.2f%% | %+7.2f%% | %7.3f | %+7.2f%% | %+9.2f%% | %+7.2f%% | %7.1f | %7.4f | %+7.2f%%"
              % (label,
                 100 * aft["CAGR_IS"], 100 * aft["CAGR_OOS"], 100 * mn,
                 pre["Sharpe_OOS"], 100 * pre["MaxDD_FULL"],
                 100 * aft["Worst10Y_star"], 100 * aft["P10_5Y"],
                 aft["Trades_yr"],
                 float(wfe), float(ci) * 100))

    # =========================================================================
    # CSV output
    # =========================================================================
    axes_order = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol", "rate:rate_up", "rate:rate_down"]
    sw_order = list(stress.keys())

    rows = []
    for label in order:
        aft = packs[label]["aft"]
        pre = packs[label]["pre"]
        exc = packs[label]["excess_days"]
        ev = evals[label]
        vm = veto_map[label]
        boot = ev["boot"]
        mn = _min_at(aft)

        for tax_label, m in (("pretax", pre), ("aftertax", aft)):
            row = {
                "condition": label,
                "tax": tax_label,
                "CAGR_IS": m["CAGR_IS"],
                "CAGR_OOS": m["CAGR_OOS"],
                "CAGR_FULL": m["CAGR_FULL"],
                "min_IS_OOS": min(m["CAGR_IS"], m["CAGR_OOS"]),
                "IS_OOS_gap_pp": m["IS_OOS_gap_pp"],
                "Sharpe_OOS": pre["Sharpe_OOS"],
                "MaxDD_FULL": pre["MaxDD_FULL"],
                "Worst10Y_star": m["Worst10Y_star"],
                "P10_5Y": m["P10_5Y"],
                "Worst5Y": m["Worst5Y"],
                "Trades_yr": m["Trades_yr"],
                "worst_calendar_year_return": packs[label]["wcy"],
                "worst_calendar_year": packs[label]["wcyy"],
                "excess_days_L_gt_3": exc,
                "excess_ratio": exc / n if n > 0 else 0.0,
                "wfa_WFE": ev["wfa_WFE"],
                "wfa_CI95_lo": ev["wfa_CI95_lo"],
                "wfa_t_p": ev["wfa_t_p"],
                "cpcv_p10_at": ev["cpcv_p10_at"],
                "cpcv_worst_at": ev["cpcv_worst_at"],
                "cpcv_med_at": ev["cpcv_med_at"],
                "regime_min_at": ev["regime_min_at"],
                "boot_P_min_better": boot["P_min_better"] if boot else "",
                "boot_CI95_lo_min_pp": boot["CI95_lo_min_pp"] if boot else "",
                "veto_maxdd": int(vm["maxdd_veto"]),
                "veto_wfe": int(vm["wfe_veto"]),
                "veto_w10y": int(vm["w10y_veto"]),
                "veto_reg": int(vm["reg_veto"]),
                "VETO": int(vm["VETO"]),
                "OUT_bondOFF_days": n_out_bondoff,
            }
            for ax in axes_order:
                row["regime_" + ax.replace(":", "_")] = ev["regime"].get(ax, np.nan)
            for sw in sw_order:
                row["stress_%s_ret" % sw] = ev["stress"][sw]["ret"]
                row["stress_%s_maxdd" % sw] = ev["stress"][sw]["maxdd"]
            rows.append(row)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "leverup_b1c1_20260612.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("\nSaved CSV: %s" % csv_path)

    # =========================================================================
    # RETURN BLOCK
    # =========================================================================
    def _rblock(label):
        aft = packs[label]["aft"]
        pre = packs[label]["pre"]
        ev = evals[label]
        boot = ev["boot"]
        vm = veto_map[label]
        return {
            "CAGR_IS_at": round(float(aft["CAGR_IS"]), 6),
            "CAGR_OOS_at": round(float(aft["CAGR_OOS"]), 6),
            "min_at": round(float(_min_at(aft)), 6),
            "gap_pp": round(float(aft["IS_OOS_gap_pp"]), 4),
            "Sharpe": round(float(pre["Sharpe_OOS"]), 4),
            "MaxDD": round(float(pre["MaxDD_FULL"]), 6),
            "Worst10Y_star_at": round(float(aft["Worst10Y_star"]), 6),
            "P10_5Y_at": round(float(aft["P10_5Y"]), 6),
            "Trades_yr": round(float(aft["Trades_yr"]), 2),
            "wfa_WFE": round(float(ev["wfa_WFE"]), 4),
            "wfa_CI95_lo": round(float(ev["wfa_CI95_lo"]), 6),
            "wfa_t_p": round(float(ev["wfa_t_p"]), 4),
            "cpcv_p10_at": round(float(ev["cpcv_p10_at"]), 6),
            "cpcv_worst_at": round(float(ev["cpcv_worst_at"]), 6),
            "cpcv_med_at": round(float(ev["cpcv_med_at"]), 6),
            "regime_min_at": round(float(ev["regime_min_at"]), 6),
            "boot_P_min_better": round(float(boot["P_min_better"]), 4) if boot else None,
            "boot_CI95_lo_min_pp": round(float(boot["CI95_lo_min_pp"]), 4) if boot else None,
            "VETO": vm["VETO"],
            "excess_days": packs[label]["excess_days"],
        }

    block = {label: _rblock(label) for label in order}
    block["C1_effects_pp"] = {
        "P09_C1_vs_P09_TQQQ": round(c1_effects["P09_C1"], 4),
        "LU1_C1_vs_LU1_cfd": round(c1_effects["LU1_C1"], 4),
        "LU2_C1_vs_LU2_cfd": round(c1_effects["LU2_C1"], 4),
    }
    block["OUT_bondOFF_days"] = n_out_bondoff
    block["sanity"] = {
        "V7_TQQQ_2022_cy_pct": round(cy2022_v7 * 100, 4),
        "P09_TQQQ_min_at_pct": round(p09_min_at * 100, 4),
        "LU2_cfd_min_at_pct": round(lu2_min_at * 100, 4),
        "LU2_cfd_MaxDD_pct": round(lu2_maxdd * 100, 4),
        "LU2_cfd_WFE": round(float(evals["LU2_cfd"]["wfa_WFE"]), 4),
        "LU2_cfd_CI95_lo_pct": round(float(evals["LU2_cfd"]["wfa_CI95_lo"]) * 100, 4),
    }

    print("\n" + "=" * 96)
    print("RETURN_BLOCK")
    print("=" * 96)
    print(json.dumps(block, indent=2))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
