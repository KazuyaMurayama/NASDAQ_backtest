"""
src/audit/margin_cash_buffer_recost_20260617.py
================================================
Precautionary cash buffer (Concept B) opportunity cost re-calculation.

NOTE (v4, 2026-06-18): Concept B's output (the -1.68 / -3.42pp "drag") was REJECTED as a
*forced* margin cost. Margin is COLLATERAL backing the position (P&L accrues on the full
notional to your equity), so it does NOT create a continuous CAGR drag; holding f% idle cash
is voluntary UNDER-INVESTMENT, not required by margin mechanics. This script remains useful for
(a) the f_cal / M(t) margin-requirement statistics and (b) quantifying the *voluntary* buffer
scenario. Canonical cost premise: PRODUCT_COST_COMPARISON s10 / EVALUATION_STANDARD s1.5 (v1.9).

BACKGROUND / PURPOSE
--------------------
margin_funded_backtest_v2_20260617.py computed margin opportunity cost as:
    opportunity_cost = margin_deposit * SOFR   (Concept A)
This is the cost of holding regulatory minimum margin at the exchange --
a small fraction (~8%) of notional, earning 0 vs SOFR.  Drag was ~-0.9pp.

Concept B (this script) models a DIFFERENT real-world requirement:
    "Keep f fraction of total AUM in cash at all times as precautionary buffer"
The buffer earns cash_return (SOFR, MMF equivalent).
The strategy leg earns r_strat -- the FULL unconstrained return.
Portfolio blend:
    r_port(t) = (1 - f) * r_strat(t) + f * r_cash(t)

Opportunity cost per day = f * (r_strat(t) - r_cash(t))
This is large because r_strat ~ +24%/yr while r_cash ~ SOFR ~ 3-5%/yr.

COMPUTATION MODEL
-----------------
For each config c and buffer fraction f in {0.00, 0.08, 0.10, 0.12, 0.15}:

  1. r_strat: daily return from _build_full_c1(...) (unconstrained baseline,
     same call/args as SANITY GATE in margin_funded_backtest_v2).
     This IS the final C1 daily return AFTER all existing costs (k365 spread,
     TER, trade costs, aftertax via compute_10metrics + _apply_aftertax).

  2. r_cash:  sofr_arr  (daily SOFR rate, same array used for borrow costs).

  3. Blend:   r_port = (1-f)*r_strat + f*r_cash

  4. NAV from r_port:  pd.Series(np.cumprod(1 + r_port), DatetimeIndex)

  5. metrics = compute_10metrics(nav_port, tpy_strat)
     aft      = _apply_aftertax(metrics)
     min9     = min(aft["CAGR_IS"], aft["CAGR_OOS"])

Note: _apply_aftertax is ALREADY applied inside _build_full_c1 flow via
compute_10metrics / _apply_aftertax pair -- the r_strat already reflects
all the per-day cost deductions before the aftertax scaling is applied at
the metric level.  Blending r_strat with r_cash and then computing metrics
from the blended NAV is internally consistent because _apply_aftertax
scales the CAGR-level output, not the daily return.

RISK-CALIBRATED BUFFER
-----------------------
For each config, compute the k365 required margin fraction M(t) on each day:
    M(t) = 0.08 * wn_s(t) * max(L_req(t) - 3.0, 0.0)
where wn_s, L_req are the V7_DELAY-shifted NASDAQ weight and leverage
(same construction as _build_s1_nav in margin_funded_backtest_v2).
f_cal = 99.5th-percentile of M(t) over the full sample.
Report M(t) statistics (mean, median, max, p99.5) and f_cal min9 / drag.

WARNING (independent QC, 2026-06-17): f_cal is the 99.5th-pct of the INITIAL
MARGIN DEPOSIT requirement (regulatory minimum collateral to OPEN/HOLD the
position).  It is NOT the cash needed to SURVIVE an adverse move without a
margin call / forced liquidation.  Variation margin on a gap can far exceed
the initial deposit: e.g. NASDAQ -10% next day at L=6x, wn~0.55 -> k365 P&L
loss ~16.5% of AUM, which exceeds f_cal(scale1.35)=19.7%'s deposit portion
and triggers a call.  Survival buffer can require 2-5x the initial deposit.
Do NOT interpret f_cal as a risk-survival threshold; it is a "can post initial
margin on 99.5% of days" figure only.  (Historically the crashes coincided
with the strategy being OUT (L=0), so this tail did not fire -- but that is
history+luck, not a structural guarantee.  See MARGIN_CAPACITY §9.)

SANITY GATE
-----------
f=0.00 must reproduce known_min9 +/- sanity_tol for all 4 configs:
    scale1.35_strong  -> +23.83% +/-0.15pp
    scale1.25_default -> +22.07% +/-0.15pp
    B3a               -> +20.98% +/-0.15pp
    P09               -> +17.77% +/-0.25pp   (via _build_full_c1 with k365 EXCESS_EXTRA)

CONFIGS
-------
Same 4 configs as margin_funded_backtest_v2:
    scale1.35_strong / scale1.25_default / B3a / P09

OUTPUTS
-------
  src/audit/margin_cash_buffer_recost_20260617.py   (this file)
  audit_results/margin_cash_buffer_recost_20260617.csv

ASCII-only output (Windows cp932). Does NOT commit. No temp files.
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

import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    DELAY as V7_DELAY,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom,
)
from src.audit.lu_cfd_recost_20260611 import (
    _build_v7_mult,
)

# ---------------------------------------------------------------------------
# Strategy configurations (identical to margin_funded_backtest_v2)
# ---------------------------------------------------------------------------
B3A_MAP_DEFAULT = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_MAP_STRONG  = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}

EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

CONFIGS = [
    {
        "label":       "scale1.35_strong",
        "v7_map":      B3A_MAP_STRONG,
        "lev_scale":   1.35,
        "known_min9":  0.2383,
        "sanity_tol":  0.0015,
    },
    {
        "label":       "scale1.25_default",
        "v7_map":      B3A_MAP_DEFAULT,
        "lev_scale":   1.25,
        "known_min9":  0.2207,
        "sanity_tol":  0.0015,
    },
    {
        "label":       "B3a",
        "v7_map":      B3A_MAP_DEFAULT,
        "lev_scale":   1.15,
        "known_min9":  0.2098,
        "sanity_tol":  0.0015,
    },
    {
        "label":       "P09",
        "v7_map":      None,
        "lev_scale":   1.0,
        "known_min9":  0.1777,
        "sanity_tol":  0.0025,
    },
]

MARGIN_RATE = 0.08  # k365 margin deposit rate (for M(t) computation)

# Buffer fractions to test (Scenario 1: uniform)
BUFFER_FRACTIONS = [0.00, 0.08, 0.10, 0.12, 0.15]
BUFFER_MAIN = 0.10  # primary scenario


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _blend_and_metrics(r_strat, r_cash, f, tpy, dates_dt):
    """Blend strategy return with cash at fraction f, compute metrics."""
    r_port = (1.0 - f) * r_strat + f * r_cash
    nav_port = pd.Series(
        np.cumprod(1.0 + r_port),
        index=dates_dt,
    )
    pre  = compute_10metrics(nav_port, tpy)
    aft  = _apply_aftertax(pre)
    min9 = _min_at(aft)
    return min9, aft, pre


def _compute_M_arr(shared, dates_dt, v7_map, lev_scale, margin_rate=0.08):
    """
    Compute per-day k365 required margin fraction M(t) = margin_rate * wn_s(t) * max(L_req(t)-3, 0).

    Uses V7_DELAY-shifted weights and leverage, same as _build_s1_nav.
    Returns M_arr (numpy array, length n).
    """
    a = shared["assets"]
    dates = a["dates"]
    idx = dates.index
    n = len(idx)

    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn_raw = np.asarray(shared["wn"], float)

    if v7_map is None:
        mult_v7 = _build_v7_mult(dates_dt) * float(lev_scale)
    else:
        mult_v7 = _build_v7_mult_custom(dates_dt, v7_map) * float(lev_scale)

    lev_mod     = lev_raw_masked * mult_v7
    L_unshifted = lev_mod * 3.0

    L_req_shifted = pd.Series(L_unshifted, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s          = pd.Series(wn_raw,      index=idx).shift(V7_DELAY).fillna(0.0).values

    M_arr = margin_rate * wn_s * np.maximum(L_req_shifted - 3.0, 0.0)
    return M_arr


def main():
    print("=" * 100)
    print("MARGIN CASH BUFFER RECOST  2026-06-17")
    print("")
    print("Concept B: r_port = (1-f)*r_strat + f*r_sofr")
    print("Opportunity cost = f * (r_strat - r_sofr)  [large: strat~24%/yr vs SOFR~3-5%/yr]")
    print("")
    print("4 CONFIGS: scale1.35_strong / scale1.25_default / B3a / P09")
    print("Uniform buffer: f in {0.00, 0.08, 0.10, 0.12, 0.15}")
    print("Risk-calibrated: f_cal = 99.5th-pct of M(t) = 0.08*wn*max(L-3,0)")
    print("=" * 100)

    # ---- Load shared data ----
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask     = np.asarray(shared["mask"], dtype=float)
    dates    = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)

    # ---- Gold/Bond 1x legs (same construction as margin_funded_backtest_v2 main) ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active  = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on   = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr  = np.asarray(a["sofr"], float)

    # =========================================================================
    # SANITY GATE: f=0.00 must reproduce known min9 for each config
    # =========================================================================
    print("\n" + "=" * 100)
    print("SANITY GATE (f=0.00 => pure strategy, no buffer): reproduce known min9")
    print("=" * 100)

    sanity_results = {}
    all_sanity_ok  = True

    for cfg in CONFIGS:
        lbl = cfg["label"]
        print("  %s ..." % lbl)
        nav_unc, r_unc, tpy_unc, exc_unc = _build_full_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
            v7_map=cfg["v7_map"], lev_scale=cfg["lev_scale"],
            excess_extra=EXCESS_EXTRA)

        # r_unc is the daily return array from _build_full_c1
        # Verify f=0 gives same result as direct compute
        pre_unc = compute_10metrics(nav_unc, tpy_unc)
        aft_unc = _apply_aftertax(pre_unc)
        got_min9 = _min_at(aft_unc)
        known    = cfg["known_min9"]
        tol      = cfg["sanity_tol"]
        ok = abs(got_min9 - known) <= tol
        if not ok:
            all_sanity_ok = False
        status = "OK" if ok else "FAIL"
        print("    got=%+.4f%%  expect~%+.4f%%  diff=%+.4fpp  -> %s"
              % (got_min9 * 100, known * 100, (got_min9 - known) * 100, status))

        sanity_results[lbl] = {
            "nav_unc":  nav_unc,
            "r_unc":    r_unc,
            "tpy_unc":  tpy_unc,
            "got_min9": got_min9,
            "aft_unc":  aft_unc,
            "ok":       ok,
        }

    if not all_sanity_ok:
        print("\nSANITY FAILED -- halting. Check _build_full_c1 / config parameters.")
        sys.exit(1)
    print("  SANITY PASSED for all %d configs." % len(CONFIGS))

    # =========================================================================
    # SCENARIO 1: Uniform buffer f in {0.00, 0.08, 0.10, 0.12, 0.15}
    # =========================================================================
    print("\n" + "=" * 100)
    print("SCENARIO 1: Uniform cash buffer  r_port = (1-f)*r_strat + f*r_sofr")
    print("=" * 100)

    # results[label][f] = {"min9": ..., "drag_pp": ...}
    uniform_results = {}

    for cfg in CONFIGS:
        lbl = cfg["label"]
        san = sanity_results[lbl]
        r_unc   = san["r_unc"]
        tpy_unc = san["tpy_unc"]
        min9_f0 = san["got_min9"]
        uniform_results[lbl] = {}

        for f in BUFFER_FRACTIONS:
            min9_f, aft_f, pre_f = _blend_and_metrics(r_unc, sofr_arr, f, tpy_unc, dates_dt)
            drag = (min9_f - min9_f0) * 100.0  # pp vs f=0
            uniform_results[lbl][f] = {
                "min9":    min9_f,
                "drag_pp": drag,
                "aft":     aft_f,
                "pre":     pre_f,
            }

    # =========================================================================
    # SCENARIO 2: Risk-calibrated buffer (f_cal = 99.5th-pct of M(t))
    # =========================================================================
    print("\n" + "=" * 100)
    print("SCENARIO 2: Risk-calibrated buffer  f_cal = 99.5th-pct of M(t)")
    print("  M(t) = 0.08 * wn_s(t) * max(L_req(t) - 3.0, 0.0)")
    print("=" * 100)

    calibrated_results = {}

    for cfg in CONFIGS:
        lbl = cfg["label"]
        san = sanity_results[lbl]
        r_unc   = san["r_unc"]
        tpy_unc = san["tpy_unc"]
        min9_f0 = san["got_min9"]

        M_arr = _compute_M_arr(shared, dates_dt, cfg["v7_map"], cfg["lev_scale"], MARGIN_RATE)

        M_mean   = float(np.mean(M_arr))
        M_median = float(np.median(M_arr))
        M_max    = float(np.max(M_arr))
        M_p995   = float(np.percentile(M_arr, 99.5))
        f_cal    = M_p995

        print("  %s: M mean=%.4f  median=%.4f  max=%.4f  p99.5=%.4f  -> f_cal=%.4f"
              % (lbl, M_mean, M_median, M_max, M_p995, f_cal))

        min9_cal, aft_cal, pre_cal = _blend_and_metrics(r_unc, sofr_arr, f_cal, tpy_unc, dates_dt)
        drag_cal = (min9_cal - min9_f0) * 100.0

        calibrated_results[lbl] = {
            "M_mean":   M_mean,
            "M_median": M_median,
            "M_max":    M_max,
            "M_p995":   M_p995,
            "f_cal":    f_cal,
            "min9_cal": min9_cal,
            "drag_pp":  drag_cal,
        }

    # =========================================================================
    # SCENARIO 3: f=0.10, cash=0 (no MMF yield, worst-case, one reference row)
    # =========================================================================
    print("\n" + "=" * 100)
    print("REFERENCE: f=0.10 with cash_return=0 (no-yield, worst-case)")
    print("=" * 100)

    cash_zero_results = {}
    r_cash_zero = np.zeros(n, float)  # zero-return cash

    for cfg in CONFIGS:
        lbl = cfg["label"]
        san = sanity_results[lbl]
        r_unc   = san["r_unc"]
        tpy_unc = san["tpy_unc"]
        min9_f0 = san["got_min9"]

        min9_z, aft_z, pre_z = _blend_and_metrics(r_unc, r_cash_zero, 0.10, tpy_unc, dates_dt)
        drag_z = (min9_z - min9_f0) * 100.0
        cash_zero_results[lbl] = {
            "min9":    min9_z,
            "drag_pp": drag_z,
        }
        print("  %s: f=0.10 cash=0  min9=%+.2f%%  drag=%+.2fpp"
              % (lbl, min9_z * 100, drag_z))

    # =========================================================================
    # PRINT SUMMARY TABLES
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 1: Uniform buffer -- min9 (tax-after, %) and drag vs f=0")
    print("=" * 100)

    # Header
    f_cols = BUFFER_FRACTIONS
    hdr_parts = ["%-22s" % "config"]
    for f in f_cols:
        hdr_parts.append("  f=%4.2f(min9%%)  drag(pp)" % f)
    print("  ".join(hdr_parts))
    print("-" * 120)

    for cfg in CONFIGS:
        lbl = cfg["label"]
        row_parts = ["%-22s" % lbl]
        for f in f_cols:
            res = uniform_results[lbl][f]
            if f == 0.00:
                row_parts.append("  %+7.2f%%          n/a    " % (res["min9"] * 100))
            else:
                row_parts.append("  %+7.2f%%       %+6.2fpp" % (res["min9"] * 100, res["drag_pp"]))
        print("  ".join(row_parts))

    print("\n" + "=" * 100)
    print("TABLE 1b: f=0.10 focus -- all configs, key metrics")
    print("=" * 100)
    print("  %-22s | %9s | %9s | %9s | %7s | %8s | %9s"
          % ("config", "min9(%)", "CAGR_IS%", "CAGR_OOS%", "drag_pp", "MaxDD%", "Sharpe"))
    print("  " + "-" * 100)
    for cfg in CONFIGS:
        lbl = cfg["label"]
        res = uniform_results[lbl][BUFFER_MAIN]
        aft = res["aft"]
        pre = res["pre"]
        print("  %-22s | %+8.2f%% | %+8.2f%% | %+8.2f%% | %+6.2fpp | %+7.2f%% | %8.3f"
              % (lbl, res["min9"] * 100, aft["CAGR_IS"] * 100, aft["CAGR_OOS"] * 100,
                 res["drag_pp"], pre["MaxDD_FULL"] * 100, pre["Sharpe_OOS"]))

    print("\n" + "=" * 100)
    print("TABLE 2: Risk-calibrated buffer (f_cal = 99.5th-pct of M(t))")
    print("=" * 100)
    print("  %-22s | %6s | %8s | %8s | %8s | %8s | %9s | %8s"
          % ("config", "f_cal", "M_mean%", "M_med%", "M_max%", "M_p995%", "min9_cal%", "drag_pp"))
    print("  " + "-" * 110)
    for cfg in CONFIGS:
        lbl = cfg["label"]
        cal = calibrated_results[lbl]
        print("  %-22s | %5.4f | %7.4f%% | %7.4f%% | %7.4f%% | %7.4f%% | %+8.2f%%  | %+7.2fpp"
              % (lbl,
                 cal["f_cal"],
                 cal["M_mean"] * 100, cal["M_median"] * 100,
                 cal["M_max"] * 100, cal["M_p995"] * 100,
                 cal["min9_cal"] * 100, cal["drag_pp"]))

    print("\n" + "=" * 100)
    print("TABLE 3: Reference -- f=0.10 cash=0 (no MMF yield, worst-case)")
    print("=" * 100)
    print("  %-22s | %9s | %8s" % ("config", "min9(%)", "drag_pp"))
    print("  " + "-" * 50)
    for cfg in CONFIGS:
        lbl = cfg["label"]
        cz = cash_zero_results[lbl]
        uf = uniform_results[lbl][BUFFER_MAIN]  # SOFR case for comparison
        print("  %-22s | %+8.2f%%  | %+7.2fpp  (vs SOFR f=0.10: min9=%+.2f%% drag=%+.2fpp)"
              % (lbl, cz["min9"] * 100, cz["drag_pp"],
                 uf["min9"] * 100, uf["drag_pp"]))

    # =========================================================================
    # ONE-LINE FINDING
    # =========================================================================
    print("\n" + "=" * 100)
    print("KEY FINDING")
    print("=" * 100)
    sc135_drag = uniform_results["scale1.35_strong"][BUFFER_MAIN]["drag_pp"]
    p09_drag   = uniform_results["P09"][BUFFER_MAIN]["drag_pp"]
    sc135_min9 = uniform_results["scale1.35_strong"][BUFFER_MAIN]["min9"]
    p09_min9   = uniform_results["P09"][BUFFER_MAIN]["min9"]
    asymmetry  = sc135_drag - p09_drag  # negative = scale1.35 hurts more
    print("  f=0.10 SOFR buffer:")
    print("    scale1.35_strong: min9=%+.2f%%  drag=%+.2fpp"
          % (sc135_min9 * 100, sc135_drag))
    print("    P09:              min9=%+.2f%%  drag=%+.2fpp"
          % (p09_min9 * 100, p09_drag))
    print("  Asymmetry (scale1.35 drag - P09 drag) = %+.2fpp"
          % asymmetry)
    if asymmetry < -0.1:
        print("  => scale1.35_strong suffers MORE drag (higher absolute leverage -> higher r_strat -> larger opp cost per unit buffer).")
    elif asymmetry > 0.1:
        print("  => P09 suffers MORE drag (unexpected).")
    else:
        print("  => Drag is roughly symmetric between configs (within 0.1pp).")

    # =========================================================================
    # CSV OUTPUT
    # =========================================================================
    print("\nBuilding CSV ...")
    rows = []

    for cfg in CONFIGS:
        lbl = cfg["label"]
        # Uniform rows
        for f in BUFFER_FRACTIONS:
            res = uniform_results[lbl][f]
            aft = res["aft"]
            pre = res["pre"]
            rows.append({
                "scenario":  "uniform",
                "config":    lbl,
                "f":         f,
                "cash_type": "SOFR",
                "min9_pct":  round(res["min9"] * 100, 4),
                "drag_pp":   round(res["drag_pp"], 4),
                "CAGR_IS_pct":   round(aft["CAGR_IS"] * 100, 4),
                "CAGR_OOS_pct":  round(aft["CAGR_OOS"] * 100, 4),
                "Sharpe_OOS":    round(pre["Sharpe_OOS"], 4),
                "MaxDD_pct":     round(pre["MaxDD_FULL"] * 100, 4),
                "Worst10Y_star_pct": round(aft["Worst10Y_star"] * 100, 4),
                "P10_5Y_pct":    round(aft["P10_5Y"] * 100, 4),
                "f_cal":     None,
            })

        # Risk-calibrated row
        cal = calibrated_results[lbl]
        rows.append({
            "scenario":  "risk_calibrated",
            "config":    lbl,
            "f":         round(cal["f_cal"], 6),
            "cash_type": "SOFR",
            "min9_pct":  round(cal["min9_cal"] * 100, 4),
            "drag_pp":   round(cal["drag_pp"], 4),
            "CAGR_IS_pct":   None,
            "CAGR_OOS_pct":  None,
            "Sharpe_OOS":    None,
            "MaxDD_pct":     None,
            "Worst10Y_star_pct": None,
            "P10_5Y_pct":    None,
            "f_cal":     round(cal["f_cal"], 6),
        })

        # cash=0 reference
        cz = cash_zero_results[lbl]
        rows.append({
            "scenario":  "uniform",
            "config":    lbl,
            "f":         0.10,
            "cash_type": "zero",
            "min9_pct":  round(cz["min9"] * 100, 4),
            "drag_pp":   round(cz["drag_pp"], 4),
            "CAGR_IS_pct":   None,
            "CAGR_OOS_pct":  None,
            "Sharpe_OOS":    None,
            "MaxDD_pct":     None,
            "Worst10Y_star_pct": None,
            "P10_5Y_pct":    None,
            "f_cal":     None,
        })

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "margin_cash_buffer_recost_20260617.csv")
    df_out   = pd.DataFrame(rows)
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(df_out)))

    # =========================================================================
    # RETURN BLOCK
    # =========================================================================
    return_block = {
        "script": "margin_cash_buffer_recost_20260617.py",
        "date":   "2026-06-17",
        "sanity": {
            lbl: {
                "got_min9_pct":   round(sanity_results[lbl]["got_min9"] * 100, 4),
                "known_min9_pct": round(cfg["known_min9"] * 100, 4),
                "ok":             sanity_results[lbl]["ok"],
            }
            for cfg in CONFIGS
            for lbl in [cfg["label"]]
        },
        "uniform_buffer": {
            lbl: {
                ("f%.2f" % f): {
                    "min9_pct": round(uniform_results[lbl][f]["min9"] * 100, 4),
                    "drag_pp":  round(uniform_results[lbl][f]["drag_pp"], 4),
                }
                for f in BUFFER_FRACTIONS
            }
            for cfg in CONFIGS
            for lbl in [cfg["label"]]
        },
        "risk_calibrated": {
            lbl: {
                "f_cal":       round(calibrated_results[lbl]["f_cal"], 6),
                "M_mean_pct":  round(calibrated_results[lbl]["M_mean"] * 100, 4),
                "M_median_pct": round(calibrated_results[lbl]["M_median"] * 100, 4),
                "M_max_pct":   round(calibrated_results[lbl]["M_max"] * 100, 4),
                "M_p995_pct":  round(calibrated_results[lbl]["M_p995"] * 100, 4),
                "min9_cal_pct": round(calibrated_results[lbl]["min9_cal"] * 100, 4),
                "drag_pp":     round(calibrated_results[lbl]["drag_pp"], 4),
            }
            for cfg in CONFIGS
            for lbl in [cfg["label"]]
        },
        "cash_zero_f010": {
            lbl: {
                "min9_pct": round(cash_zero_results[lbl]["min9"] * 100, 4),
                "drag_pp":  round(cash_zero_results[lbl]["drag_pp"], 4),
            }
            for cfg in CONFIGS
            for lbl in [cfg["label"]]
        },
        "asymmetry_finding": {
            "scale1.35_drag_pp": round(sc135_drag, 4),
            "p09_drag_pp":       round(p09_drag, 4),
            "asymmetry_pp":      round(asymmetry, 4),
        },
        "csv_path": csv_path,
    }

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone -- margin_cash_buffer_recost complete.")
    return return_block


if __name__ == "__main__":
    main()
