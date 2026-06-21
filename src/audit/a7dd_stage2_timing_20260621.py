"""
src/audit/a7dd_stage2_timing_20260621.py
========================================
Stage-2 timing-correction runner for the A7-based DD-reduction brakes.

WHY THIS EXISTS (the QC correction)
-----------------------------------
Stage-1 decided "timing vs plain de-lever" with a block=21 bootstrap of the
FULL-SERIES MaxDD vs each brake's uniform-delever twin. An independent QC found
that test INVALID: block=21 shuffles apart the multi-year crash sequences that
DEFINE MaxDD, so timing_P_maxdd is pinned near 0.5 regardless of true timing
skill (the repo's own multimetric docstring warns this for Worst10Y). This
Stage-2 runner REPLACES that invalid block-21 MaxDD bootstrap with a path-robust
test:

(D) CRISIS-WINDOW (intact-path) TIMING. For each DD-reduction brake, slice the
    INTACT (non-resampled) return path on each crisis/stress window and compare
    the brake's windowed MaxDD to its equal-average-exposure uniform-delever
    twin (build_uniform_delever at the brake's own fbar). A cross-window sign
    test then asks whether the brake systematically lands its cuts inside
    crises (real timing) or not (just brakes less). Small n (=5 windows): result
    is SUGGESTIVE, not a formal proof.

(E) B1 EQUAL-FBAR RETEST. B1 (downside-dev brake) appears to beat A7 on CAGR,
    but B1 also brakes LESS on average (fbar ~1.34% vs A7 ~2.12%). So its CAGR
    edge may just be "weaker brake", not "smarter signal". We sweep B1's
    target_dvol DOWN (lower dvol -> more braking -> higher fbar) until B1's fbar
    MATCHES A7's fbar, then compare B1's MaxDD to A7's at that matched braking
    intensity. If B1 still has a meaningfully shallower MaxDD at EQUAL fbar, B1
    is a smarter signal; if not, B1's edge was just less braking.

The base P09_C1 daily return r_strat is built BYTE-IDENTICALLY to the Stage-1
runner (a7dd_stage1_20260621.py) -- same data prep, same _build_out_fill_variant
wiring -- and asserted to 1e-6 vs a direct _build_full_c1 P09_C1 build (A0).
A7_REPRODUCE MaxDD is asserted within 0.002 of the prior project's A7
(audit_results/p09c1_alloc_stage1_20260620.csv, -0.282462).

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
Outputs:
  audit_results/a7dd_stage2_crisis_timing_20260621.csv   (D)
  audit_results/a7dd_stage2_b1_equalfbar_20260621.csv     (E)
"""

from __future__ import annotations

import json
import math
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

# Cost/NAV helpers + WFA/bootstrap constants
from src.audit.run_p09_tqqq_validate_20260611 import _maxdd_from_returns
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _count_fund_transitions,
    LAG_DAYS, TRADING_DAYS,
)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base

# OUT-fill variant machinery (Task 8)
from src.audit.out_fill_variants_20260620 import (
    _build_out_fill_variant, alloc_base)

# DD-reduction overlays + timing-vs-delever machinery (Task 6)
from src.audit.dd_reduction_overlays_20260621 import (
    apply_downside_dev_brake, apply_dd_throttle, apply_asym_vol_brake,
    apply_param_vol_brake, measure_mean_in_leg_frac, build_uniform_delever)

# Crisis-window (intact-path) timing test (Task 1)
from src.audit.crisis_window_timing_20260621 import (
    crisis_window_dd_compare, sign_test_brake_beats_twin)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Canonical P09_C1 wiring: default V7_MAP (v7_map=None), lev_scale=1.0, and
# NO >3x excess charge. excess_extra=0.0 reproduces cfd_excess=False exactly.
P09C1_V7_MAP       = None
P09C1_LEV_SCALE    = 1.0
P09C1_EXCESS_EXTRA = 0.0

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


def main():
    print("=" * 120)
    print("A7-BASED DD-REDUCTION STAGE-2: CRISIS-WINDOW TIMING + B1 EQUAL-FBAR RETEST  2026-06-21")
    print("(D) intact-window MaxDD sign test (replaces invalid block-21 MaxDD bootstrap)")
    print("(E) B1 downside-dev dvol swept to MATCH A7 fbar -> smarter-signal vs weaker-brake")
    print("=" * 120)

    # ---- Load shared data (verbatim from Stage-1 template) ----
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

    a0_nav = pd.Series(np.cumprod(1.0 + r_strat), index=dates_dt)
    a0_pre = compute_10metrics(a0_nav, tpy0)
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
    # (D) CRISIS-WINDOW TIMING for all DD-reduction brakes
    # =========================================================================
    BRAKES = [
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
    a7_r_san = dict(BRAKES)["A7_REPRODUCE"]
    a7_maxdd_san = _maxdd_from_returns(a7_r_san)
    diff_a7 = abs(a7_maxdd_san - PRIOR_A7_MAXDD)
    print("  A7_REPRODUCE MaxDD_FULL : %+.6f" % a7_maxdd_san)
    print("  prior A7 MaxDD          : %+.6f" % PRIOR_A7_MAXDD)
    print("  |diff|                  : %.6f   (tol %.3f)" % (diff_a7, A7_MAXDD_TOL))
    if diff_a7 > A7_MAXDD_TOL:
        print("\nSANITY 2 FAILED -- A7_REPRODUCE does not reproduce prior A7 MaxDD.")
        print("The new brake module does NOT reproduce A7. Halting.")
        sys.exit(1)
    print("  SANITY 2 PASSED (A7_REPRODUCE == prior A7 MaxDD within 0.002).\n")

    # =========================================================================
    # (D) per-brake crisis-window sign test vs uniform-delever twin
    # =========================================================================
    print("=" * 120)
    print("(D) CRISIS-WINDOW (intact-path) TIMING -- %d brakes, %d stress windows"
          % (len(BRAKES), sum(1 for v in stress.values() if np.asarray(v, bool).sum() > 0)))
    print("=" * 120)
    crisis_rows = []
    summary = {}
    for label, r_b in BRAKES:
        fbar = measure_mean_in_leg_frac(r_strat, r_b, fund_active, sofr_arr)
        r_uni = build_uniform_delever(r_strat, fund_active, sofr_arr, fbar)
        win = crisis_window_dd_compare(r_b, r_uni, stress)
        st = sign_test_brake_beats_twin(win)
        # derived path-robust verdict (small-n: suggestive only)
        nW, nS = st["n_windows"], st["n_shallower"]
        if nW > 0 and nS == nW:
            verdict = "TIMING_CONFIRMED"
        elif nW > 0 and nS >= math.ceil(nW * 0.6) and st["binom_p_onesided"] < 0.20:
            verdict = "TIMING_LIKELY"
        else:
            verdict = "TIMING_WEAK"
        summary[label] = {"fbar": fbar, **st, "crisis_verdict": verdict}
        for w in win:
            crisis_rows.append({"label": label, "fbar": fbar, **w,
                                "n_windows": st["n_windows"], "n_shallower": st["n_shallower"],
                                "binom_p_onesided": st["binom_p_onesided"],
                                "mean_dd_edge_pp": st["mean_dd_edge_pp"],
                                "crisis_verdict": verdict})
        print("  %-16s fbar=%.4f  shallower=%d/%d  mean_edge=%+.2fpp  binom_p=%.4f  -> %s"
              % (label, fbar, nS, nW, st["mean_dd_edge_pp"],
                 st["binom_p_onesided"], verdict))

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    crisis_csv = os.path.join(out_dir, "a7dd_stage2_crisis_timing_20260621.csv")
    pd.DataFrame(crisis_rows).to_csv(crisis_csv, index=False, encoding="utf-8-sig")
    print("\nSaved (D) CSV: %s  (%d rows)" % (crisis_csv, len(crisis_rows)))

    # =========================================================================
    # (E) B1 equal-fbar retest -- sweep target_dvol to match A7's fbar
    # =========================================================================
    print("\n" + "=" * 120)
    print("(E) B1 EQUAL-FBAR RETEST -- sweep downside-dev target_dvol to match A7 fbar")
    print("=" * 120)
    a7_r = dict(BRAKES)["A7_REPRODUCE"]
    a7_fbar = measure_mean_in_leg_frac(r_strat, a7_r, fund_active, sofr_arr)
    a7_maxdd = _maxdd_from_returns(a7_r)
    print("  A7 fbar=%.4f  A7 MaxDD=%+.4f%%" % (a7_fbar, a7_maxdd * 100))

    # base sweep; EXTEND downward if it does not bracket a7_fbar (lower dvol ->
    # more braking -> higher fbar)
    DVOL_SWEEP = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06]
    DVOL_EXTEND = [0.05, 0.04, 0.03, 0.02]

    def _b1_row(dvol):
        rb1 = apply_downside_dev_brake(r_strat, fund_active, sofr_arr, dvol, 63, 0.5)
        fb = measure_mean_in_leg_frac(r_strat, rb1, fund_active, sofr_arr)
        mdd = _maxdd_from_returns(rb1)
        nav = pd.Series(np.cumprod(1.0 + rb1), index=dates_dt)
        m = compute_10metrics(nav, tpy0)
        ma = _apply_aftertax(m)
        return {"target_dvol": dvol, "fbar": fb, "MaxDD_FULL": mdd,
                "CAGR_IS_at": ma["CAGR_IS"], "CAGR_OOS_at": ma["CAGR_OOS"],
                "fbar_minus_a7": fb - a7_fbar, "maxdd_minus_a7": mdd - a7_maxdd}

    rows_e = [_b1_row(dvol) for dvol in DVOL_SWEEP]

    # fbar is MONOTONE DECREASING in target_dvol (lower dvol -> more braking ->
    # higher fbar). Two outcomes need handling to land an fbar ~ a7_fbar:
    #   (a) a7_fbar is ABOVE the whole sweep's fbar range (min fbar still < a7
    #       only at the highest dvol but never reaches a7): EXTEND DOWNWARD
    #       (lower dvol) to raise fbar until it crosses a7_fbar.
    #   (b) a7_fbar is bracketed by two adjacent dvol points but the grid is too
    #       COARSE to be within ~0.002: BISECT in dvol between the bracketing
    #       pair to refine onto a7_fbar.
    def _brackets(rows):
        fbs = [r["fbar"] for r in rows]
        crosses = (min(fbs) <= a7_fbar <= max(fbs))
        nearest = min(abs(r["fbar_minus_a7"]) for r in rows)
        return crosses, nearest

    crosses, nearest = _brackets(rows_e)
    extended = False
    refined = False

    # (a) extend downward if the sweep never reaches up to a7_fbar
    if not crosses and (max(r["fbar"] for r in rows_e) < a7_fbar):
        for dvol in DVOL_EXTEND:
            rows_e.append(_b1_row(dvol))
            crosses, nearest = _brackets(rows_e)
            extended = True
            if crosses:
                break

    # (b) bisect within the bracketing dvol pair to land within ~0.002 of a7_fbar
    if crosses and nearest > 0.002:
        # find the adjacent (dvol_hi above-fbar-low, dvol_lo below-fbar-high) pair
        srt = sorted(rows_e, key=lambda r: r["target_dvol"])  # ascending dvol => descending fbar
        lo_d = hi_d = None
        for i in range(len(srt) - 1):
            f_hi_dvol = srt[i + 1]["fbar"]   # higher dvol -> lower fbar
            f_lo_dvol = srt[i]["fbar"]       # lower dvol  -> higher fbar
            if min(f_hi_dvol, f_lo_dvol) <= a7_fbar <= max(f_hi_dvol, f_lo_dvol):
                lo_d, hi_d = srt[i]["target_dvol"], srt[i + 1]["target_dvol"]
                break
        if lo_d is not None:
            for _ in range(20):
                mid = 0.5 * (lo_d + hi_d)
                rmid = _b1_row(mid)
                rows_e.append(rmid)
                refined = True
                if abs(rmid["fbar_minus_a7"]) <= 0.002:
                    break
                # fbar decreasing in dvol: if mid fbar still > a7, raise dvol (hi side)
                if rmid["fbar"] > a7_fbar:
                    lo_d = mid
                else:
                    hi_d = mid
            crosses, nearest = _brackets(rows_e)

    df_e = pd.DataFrame(rows_e).drop_duplicates("target_dvol").sort_values(
        "target_dvol", ascending=False).reset_index(drop=True)
    df_e["a7_fbar"] = a7_fbar
    df_e["a7_maxdd"] = a7_maxdd
    df_e["a7_cagr_oos"] = _apply_aftertax(compute_10metrics(
        pd.Series(np.cumprod(1.0 + a7_r), index=dates_dt), tpy0))["CAGR_OOS"]

    b1_csv = os.path.join(out_dir, "a7dd_stage2_b1_equalfbar_20260621.csv")
    df_e.to_csv(b1_csv, index=False, encoding="utf-8-sig")
    print("  Sweep brackets a7_fbar(%.4f)? %s   nearest |fbar-a7|=%.4f   extended=%s  refined=%s"
          % (a7_fbar, "YES" if crosses else "NO", nearest, extended, refined))
    print("\n  %-11s | %7s | %9s | %12s | %11s | %12s"
          % ("target_dvol", "fbar", "MaxDD%", "fbar-a7", "maxdd-a7pp", "CAGR_OOS%"))
    print("  " + "-" * 78)
    for _, r in df_e.iterrows():
        print("  %11.3f | %7.4f | %+8.2f%% | %+12.4f | %+10.2fpp | %+10.2f%%"
              % (r["target_dvol"], r["fbar"], r["MaxDD_FULL"] * 100,
                 r["fbar_minus_a7"], r["maxdd_minus_a7"] * 100, r["CAGR_OOS_at"] * 100))

    # locate the row where fbar ~ a7_fbar
    idx_match = int(np.argmin(np.abs(df_e["fbar_minus_a7"].values)))
    match = df_e.iloc[idx_match]
    smarter = match["maxdd_minus_a7"]  # >0 => B1 MaxDD shallower at equal fbar
    print("\n  MATCH (fbar~a7_fbar) at target_dvol=%.3f : B1 fbar=%.4f (a7=%.4f, diff=%+.4f)"
          % (match["target_dvol"], match["fbar"], a7_fbar, match["fbar_minus_a7"]))
    print("  At EQUAL braking: B1 MaxDD=%+.4f%%  vs A7 MaxDD=%+.4f%%  (maxdd_minus_a7=%+.2fpp)"
          % (match["MaxDD_FULL"] * 100, a7_maxdd * 100, smarter * 100))
    if smarter > 0.005:
        print("  -> B1 IS A SMARTER SIGNAL (shallower MaxDD at equal braking, edge >0.5pp).")
    elif smarter < -0.005:
        print("  -> B1 is WORSE at equal braking (A7's signal is better).")
    else:
        print("  -> TIE: B1's apparent edge was JUST WEAKER BRAKING, not a smarter signal.")
    print("  Saved (E) CSV: %s  (%d rows)" % (b1_csv, len(df_e)))

    # =========================================================================
    # RETURN_BLOCK
    # =========================================================================
    return_block = {
        "script": "a7dd_stage2_timing_20260621.py",
        "date": "2026-06-21",
        "sanity_gate_1": {
            "A0_CAGR_OOS_at_pct":           round(a0_cagr_oos * 100, 6),
            "direct_P09C1_CAGR_OOS_at_pct": round(canon_cagr_oos * 100, 6),
            "abs_diff_A0_vs_direct":        float(diff_a0_direct),
            "PASS": bool(diff_a0_direct <= SANITY_TOL_TIGHT),
        },
        "sanity_gate_2": {
            "A7_REPRODUCE_MaxDD": round(float(a7_maxdd_san), 6),
            "prior_A7_MaxDD":     PRIOR_A7_MAXDD,
            "abs_diff":           float(diff_a7),
            "PASS": bool(diff_a7 <= A7_MAXDD_TOL),
        },
        "D_crisis_timing": {
            label: {
                "fbar": round(s["fbar"], 6),
                "n_shallower": s["n_shallower"],
                "n_windows": s["n_windows"],
                "mean_dd_edge_pp": round(s["mean_dd_edge_pp"], 4),
                "binom_p_onesided": round(s["binom_p_onesided"], 6),
                "crisis_verdict": s["crisis_verdict"],
            }
            for label, s in summary.items()
        },
        "E_b1_equalfbar": {
            "a7_fbar": round(float(a7_fbar), 6),
            "a7_maxdd": round(float(a7_maxdd), 6),
            "match_target_dvol": round(float(match["target_dvol"]), 4),
            "match_fbar": round(float(match["fbar"]), 6),
            "match_fbar_minus_a7": round(float(match["fbar_minus_a7"]), 6),
            "match_MaxDD": round(float(match["MaxDD_FULL"]), 6),
            "match_maxdd_minus_a7_pp": round(float(smarter) * 100, 4),
            "match_CAGR_OOS_at_pct": round(float(match["CAGR_OOS_at"]) * 100, 4),
            "match_nearest_abs_fbar_diff": round(float(nearest), 6),
            "sweep_brackets_a7_fbar": bool(crosses),
            "range_extended": bool(extended),
            "range_refined_bisect": bool(refined),
        },
        "csv_D": crisis_csv,
        "csv_E": b1_csv,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))

    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
