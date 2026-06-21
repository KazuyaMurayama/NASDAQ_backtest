"""
src/audit/b1_scale_annual_20260621.py
=====================================
B1 (downside-deviation brake) x leverage-scale: (A) ANNUAL after-tax returns and
(B) a per-scale UNIFORM-DELEVER CONTROL that discloses whether B1xscale's
drawdown cap is timing or plain de-lever.

Builds the SAME 8 "B1 at scale S" series as b1_scale_stage1_20260621.py (the B1
IN-leg downside-deviation brake applied to the SCALED P09_C1 strategy), across:
  boost map x lev_scale = {default, strong} x {1.4, 1.6, 1.8, 2.0}
The scale-1.0 sanity series (B1_DEF_S1.0) is dropped here -- only the 8 scale
series + the NASDAQ 1x B&H benchmark are reported.

PART A -- annual after-tax (calendar-year)
------------------------------------------
For each of the 8 B1xscale series and the NASDAQ 1x B&H, extract calendar-year
returns from the daily NAV, apply the repo empirical after-tax multiplier
(x0.8273), and write:
  audit_results/b1_scale_annual_20260621.csv
  columns: year, B1_DEF_S1.4..S2.0, B1_STR_S1.4..S2.0, NASDAQ_1x_BH_pct

PART B -- uniform-delever control per scale (de-lever vs timing disclosure)
--------------------------------------------------------------------------
A code-quality reviewer flagged: at scale 2.0 the brake removes ~20% of average
exposure, so "B1x2.0 ~= P09 at a lower effective scale + vol-adaptive cash". To
disclose this HONESTLY, for each of the 8 series we:
  1. build the UNBRAKED scaled strategy r_strat_s,
  2. build the braked r_b1_s,
  3. measure fbar = average IN-leg cash fraction the brake applied
     (measure_mean_in_leg_frac),
  4. build the equal-fbar UNIFORM-DELEVER TWIN on the SAME scaled base
     (build_uniform_delever -- same average exposure cut, blind to WHEN),
  5. record both NAVs' after-tax CAGR_OOS + MaxDD and the brake-minus-twin gaps.
Output:
  audit_results/b1_scale_delever_control_20260621.csv
Interpretation: maxdd_brake_minus_twin_pp > 0 means the brake's MaxDD is
SHALLOWER than the equal-fbar uniform twin -> timing helps. ~0 means the DD cap
is explained by de-lever alone (no special timing edge).

SANITY GATE: A0 == canonical P09_C1 to 1e-6 (direct _build_full_c1, v7_map=None,
lev_scale=1.0, excess_extra=0.0) built via the SAME scaled-series plumbing.

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
Output: 2 CSVs (utf-8-sig / BOM).
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

# C1 cost-model builders
from src.audit.k365_recost_20260612 import (
    _build_full_c1, _build_tqqq_base_param,
    EXCESS_EXTRA_K365_CENTRE,
)

# Cost/NAV helpers + calendar-year + after-tax constant + transitions
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _count_fund_transitions,
    _calendar_year_returns,
    LAG_DAYS, TRADING_DAYS,
)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY

# MaxDD-from-returns helper (kept available for cross-checks / parity with task)
from src.audit.run_p09_tqqq_validate_20260611 import _maxdd_from_returns  # noqa: F401

# OUT-fill variant machinery (Task 8)
from src.audit.out_fill_variants_20260620 import _build_out_fill_variant, alloc_base

# B1 downside-deviation brake + de-lever-vs-timing controls (Task 6)
from src.audit.dd_reduction_overlays_20260621 import (
    apply_downside_dev_brake,
    measure_mean_in_leg_frac, build_uniform_delever,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Canonical P09_C1 wiring: default V7_MAP (v7_map=None), lev_scale=1.0, and
# NO >3x excess charge. excess_extra=0.0 reproduces cfd_excess=False exactly.
P09C1_V7_MAP       = None
P09C1_LEV_SCALE    = 1.0
P09C1_EXCESS_EXTRA = 0.0

# Boost maps + excess for the scaled-B1 series (MATCH b1_scale_stage1)
DEFAULT_MAP = None                          # -> default {1.20,1.10,1.00,1.00}
STRONG_MAP  = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXC         = EXCESS_EXTRA_K365_CENTRE      # 0.0025 -- MATCH prior P09 scale runs

# B1 brake params (identical to Stage-1)
B1_TARGET_DVOL   = 0.20
B1_WINDOW        = 63
B1_MAX_FRAC_CASH = 0.5

# repo empirical effective after-tax multiplier
AFTER_TAX = 0.8273  # repo empirical effective after-tax multiplier

# A0 sanity vs direct _build_full_c1
SANITY_TOL_TIGHT = 1e-6

# Cross-check reference (after-tax B&H) from a7dd_annual_20260621.csv /
# annual_returns_sc180_200_20260620.csv (PRE-tax x 0.8273):
#   2008 PRE -40.54 -> AT -33.54 ; 2020 PRE +43.64 -> AT +36.10
XCHECK_BH = {2008: -33.54, 2020: 36.10}
XCHECK_TOL_PP = 0.1


def main():
    print("=" * 120)
    print("B1 DOWNSIDE-DEV BRAKE x LEVERAGE-SCALE  ANNUAL + UNIFORM-DELEVER CONTROL  2026-06-21")
    print("8 B1xscale series ({default,strong} map x scale {1.4,1.6,1.8,2.0}) + NASDAQ 1x B&H")
    print("PART A: calendar-year after-tax (x%.4f)   PART B: equal-fbar uniform-delever twin" % AFTER_TAX)
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

    # ---- Gold/Bond auxiliary series (verbatim from Stage-1) ----
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
    # Scaled-B1 builder (identical composition to b1_scale_stage1)
    # =========================================================================
    def build_b1_scaled(v7_map, lev_scale, excess=EXC):
        """Scale IN-leg leverage, build full P09_C1 strategy, then apply B1 brake.
        Returns (nav_b1 [pd.Series], r_b1_s [np.ndarray], tpy [float]).
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

    def build_b1_components(v7_map, lev_scale, excess=EXC):
        """Return (r_strat_s unbraked, r_b1_s braked, eff_s, tpy_base_s) for a
        scaled series. The UNBRAKED r_strat_s is the SAME scaled base the brake
        delevers, so the equal-fbar uniform twin is built on it directly."""
        _, r_base_in_s, tpy_base_s, _ = _build_tqqq_base_param(
            shared, dates_dt, v7_map=v7_map, lev_scale=lev_scale,
            excess_extra=excess)
        _, r_strat_s, eff_s = _build_out_fill_variant(
            r_base_in_s, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            alloc_fn=alloc_base)
        r_b1_s = apply_downside_dev_brake(
            r_strat_s, fund_active, sofr_arr,
            B1_TARGET_DVOL, B1_WINDOW, B1_MAX_FRAC_CASH)
        return r_strat_s, r_b1_s, eff_s, tpy_base_s

    # =========================================================================
    # SANITY GATE: A0 must reproduce canonical P09_C1 (direct _build_full_c1)
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: A0 (scaled plumbing, scale=1.0, excess=0.0) vs direct _build_full_c1 P09_C1 (tol 1e-6)")
    print("=" * 120)
    canon_nav, canon_r, canon_tpy, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    canon_aft = _apply_aftertax(compute_10metrics(canon_nav, canon_tpy))
    canon_cagr_oos = canon_aft["CAGR_OOS"]

    # A0 built via the SAME plumbing as the scaled series (scale=1.0, excess=0.0).
    _, a0_r_base, a0_tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    a0_nav, a0_r_strat, a0_eff = _build_out_fill_variant(
        a0_r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    a0_tpy = a0_tpy_base + _count_fund_transitions(a0_eff) / n_years
    a0_nav_dt = pd.Series(np.cumprod(1.0 + a0_r_strat), index=dates_dt)
    a0_aft = _apply_aftertax(compute_10metrics(a0_nav_dt, a0_tpy))
    a0_cagr_oos = a0_aft["CAGR_OOS"]

    diff_a0_direct = abs(a0_cagr_oos - canon_cagr_oos)
    print("  A0 CAGR_OOS(at)           : %+.8f%%" % (a0_cagr_oos * 100))
    print("  direct _build_full_c1 OOS : %+.8f%%" % (canon_cagr_oos * 100))
    print("  |A0 - direct|             : %.3e   (tol %.0e)" % (diff_a0_direct, SANITY_TOL_TIGHT))
    if diff_a0_direct > SANITY_TOL_TIGHT:
        print("\nSANITY FAILED -- A0 does not match direct _build_full_c1 P09_C1 to 1e-6.")
        print("The base wiring is wrong. Halting.")
        sys.exit(1)
    print("  SANITY PASSED (A0 == direct P09_C1 to 1e-6).\n")

    # =========================================================================
    # SERIES (drop scale-1.0 sanity; 8 scale series)
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
    SERIES8 = [s for s in SERIES if s[0] != "B1_DEF_S1.0"]   # drop scale-1.0 sanity
    SERIES8_LABELS = [s[0] for s in SERIES8]

    # =========================================================================
    # PART A -- calendar-year AFTER-TAX returns for the 8 series + B&H
    # =========================================================================
    print("=" * 120)
    print("PART A: calendar-year after-tax returns for %d B1xscale series + NASDAQ 1x B&H" % len(SERIES8))
    print("=" * 120)

    series_annual = {}
    for label, vmap, scale in SERIES8:
        print("  [%s] (map=%s, scale=%.1f) ..." % (label, "default" if vmap is None else "strong", scale))
        nav_b1, r_b1, tpy = build_b1_scaled(vmap, scale)
        yr = _calendar_year_returns(nav_b1)            # pd.Series indexed by year (fraction)
        series_annual[label] = {int(y): float(v) * AFTER_TAX * 100.0 for y, v in yr.items()}

    print("  [NASDAQ_1x_BH_pct] from a['ret'] ...")
    ret_1x = np.asarray(a["ret"], float)
    bh_nav = pd.Series(np.cumprod(1.0 + ret_1x), index=dates_dt)
    bh_yr = _calendar_year_returns(bh_nav)
    series_annual["NASDAQ_1x_BH_pct"] = {int(y): float(v) * AFTER_TAX * 100.0 for y, v in bh_yr.items()}

    # ---- CROSS-CHECK: B&H 2008/2020 (after-tax) vs prior CSV (within 0.1pp) ----
    print("\n" + "=" * 120)
    print("CROSS-CHECK: NASDAQ_1x_BH_pct (after-tax) 2008/2020 vs prior CSV (tol %.1fpp)" % XCHECK_TOL_PP)
    print("=" * 120)
    xcheck_ok = True
    for yr_k, ref_pct in sorted(XCHECK_BH.items()):
        our_pct = series_annual["NASDAQ_1x_BH_pct"].get(yr_k, float("nan"))
        diff_pp = abs(our_pct - ref_pct)
        status = "OK" if diff_pp <= XCHECK_TOL_PP else "FAIL"
        if status == "FAIL":
            xcheck_ok = False
        print("  %d: ours=%+.2f%%  ref=%+.2f%%  |diff|=%.2fpp  [%s]"
              % (yr_k, our_pct, ref_pct, diff_pp, status))
    if not xcheck_ok:
        print("\n  WARN: NASDAQ_1x_BH cross-check FAIL on 2008/2020.")
    else:
        print("  CROSS-CHECK PASSED (2008/2020 within %.1fpp)." % XCHECK_TOL_PP)

    # ---- 2008 must be POSITIVE for all B1xscale series (OUT defense) ----
    print("\n2008 after-tax (must be > 0 for all B1xscale series; OUT defense in the crash):")
    all_2008_pos = True
    for label in SERIES8_LABELS:
        v2008 = series_annual[label].get(2008, float("nan"))
        ok = np.isfinite(v2008) and v2008 > 0
        if not ok:
            all_2008_pos = False
        print("  %-12s 2008 = %+.2f%%  [%s]" % (label, v2008, "OK" if ok else "FAIL"))
    assert all_2008_pos, "FAIL: at least one B1xscale series has 2008 <= 0 (expected positive)."
    print("  -> PASS (all B1xscale 2008 > 0)")

    # ---- Build PART A table ----
    all_years = sorted(set().union(*[set(v.keys()) for v in series_annual.values()]))
    colA = ["year"] + SERIES8_LABELS + ["NASDAQ_1x_BH_pct"]
    rowsA = []
    for yr_k in all_years:
        row = {"year": yr_k}
        for label in SERIES8_LABELS:
            row[label] = series_annual[label].get(yr_k, float("nan"))
        row["NASDAQ_1x_BH_pct"] = series_annual["NASDAQ_1x_BH_pct"].get(yr_k, float("nan"))
        rowsA.append(row)
    dfA = pd.DataFrame(rowsA)[colA]

    # ---- spot-check S2.0 2008/2020 ----
    print("\nKEY SPOT-CHECK: scale-2.0 2008 / 2020 after-tax:")
    for label in ("B1_DEF_S2.0", "B1_STR_S2.0"):
        print("  %-12s 2008=%+.2f%%  2020=%+.2f%%"
              % (label, series_annual[label].get(2008, float("nan")),
                 series_annual[label].get(2020, float("nan"))))

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csvA_path = os.path.join(out_dir, "b1_scale_annual_20260621.csv")
    dfA.to_csv(csvA_path, index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\nSaved CSV (PART A): %s  (%d rows, %d columns)" % (csvA_path, len(dfA), len(dfA.columns)))

    # =========================================================================
    # PART B -- uniform-delever control per scale (de-lever vs timing)
    # =========================================================================
    print("\n" + "=" * 120)
    print("PART B: equal-fbar UNIFORM-DELEVER CONTROL per scale (de-lever vs timing)")
    print("For each B1xscale series: measure brake's avg IN-leg cash fraction (fbar),")
    print("build the same-fbar uniform-delever twin on the SAME scaled base (no timing),")
    print("compare MaxDD + after-tax CAGR_OOS. maxdd_brake_minus_twin_pp > 0 => timing helps.")
    print("=" * 120)

    ctrl_rows = []
    for label, vmap, scale in SERIES8:
        r_strat_s, r_b1_s, eff_s, tpy_base_s = build_b1_components(vmap, scale)
        tpy = tpy_base_s + _count_fund_transitions(eff_s) / n_years
        fbar = measure_mean_in_leg_frac(r_strat_s, r_b1_s, fund_active, sofr_arr)
        r_uni = build_uniform_delever(r_strat_s, fund_active, sofr_arr, fbar)  # equal-fbar twin

        nav_b = pd.Series(np.cumprod(1.0 + r_b1_s), index=dates_dt)
        nav_u = pd.Series(np.cumprod(1.0 + r_uni), index=dates_dt)
        pb = compute_10metrics(nav_b, tpy)
        pu = compute_10metrics(nav_u, tpy)
        mb = _apply_aftertax(pb)
        mu = _apply_aftertax(pu)

        maxdd_gap_pp = (pb["MaxDD_FULL"] - pu["MaxDD_FULL"]) * 100.0
        cagr_gap_pp  = (mb["CAGR_OOS"] - mu["CAGR_OOS"]) * 100.0
        ctrl_rows.append({
            "label": label,
            "boost_map": ("default" if vmap is None else "strong"),
            "lev_scale": scale,
            "fbar": fbar,
            "brake_MaxDD": pb["MaxDD_FULL"],
            "twin_MaxDD": pu["MaxDD_FULL"],
            "brake_CAGR_OOS": mb["CAGR_OOS"],
            "twin_CAGR_OOS": mu["CAGR_OOS"],
            "maxdd_brake_minus_twin_pp": maxdd_gap_pp,
            "cagr_brake_minus_twin_pp": cagr_gap_pp,
        })
        print("  %-12s scale=%.1f  fbar=%.4f  brakeDD=%+.2f%%  twinDD=%+.2f%%  "
              "dMaxDD=%+.2fpp  brakeCAGR=%+.2f%%  twinCAGR=%+.2f%%  dCAGR=%+.2fpp"
              % (label, scale, fbar, pb["MaxDD_FULL"] * 100, pu["MaxDD_FULL"] * 100,
                 maxdd_gap_pp, mb["CAGR_OOS"] * 100, mu["CAGR_OOS"] * 100, cagr_gap_pp))

    dfB = pd.DataFrame(ctrl_rows)
    csvB_path = os.path.join(out_dir, "b1_scale_delever_control_20260621.csv")
    dfB.to_csv(csvB_path, index=False, encoding="utf-8-sig")
    print("\nSaved CSV (PART B): %s  (%d rows)" % (csvB_path, len(dfB)))

    # ---- INTERPRETATION: timing vs de-lever per scale ----
    print("\n" + "=" * 120)
    print("INTERPRETATION: is B1xscale's DD cap TIMING or DE-LEVER?")
    print("  maxdd_brake_minus_twin_pp > +0.5pp => brake DD meaningfully SHALLOWER than equal-fbar twin (TIMING)")
    print("  |gap| <= 0.5pp                     => DD cap ~= de-lever only (NO special timing edge)")
    print("=" * 120)
    TIMING_THR = 0.5
    for row in ctrl_rows:
        gap = row["maxdd_brake_minus_twin_pp"]
        if gap > TIMING_THR:
            verdict = "TIMING (brake shallower)"
        elif gap < -TIMING_THR:
            verdict = "WORSE THAN DELEVER (twin shallower)"
        else:
            verdict = "DE-LEVER ONLY (~equal)"
        print("  %-12s scale=%.1f  fbar=%.3f  dMaxDD=%+.2fpp  -> %s"
              % (row["label"], row["lev_scale"], row["fbar"], gap, verdict))

    # focused scale-2.0 statement
    print("\n  SCALE-2.0 FOCUS (the QC caveat):")
    for label in ("B1_DEF_S2.0", "B1_STR_S2.0"):
        row = next(r for r in ctrl_rows if r["label"] == label)
        gap = row["maxdd_brake_minus_twin_pp"]
        special = "shallower than equal-fbar uniform twin -> TIMING adds value" if gap > TIMING_THR \
            else ("~= equal-fbar uniform twin -> DD cap is DE-LEVER, brake adds little at scale"
                  if abs(gap) <= TIMING_THR else "WORSE than equal-fbar twin -> brake hurts DD at scale")
        print("    %-12s fbar=%.3f  brakeDD=%+.2f%%  twinDD=%+.2f%%  dMaxDD=%+.2fpp  => %s"
              % (label, row["fbar"], row["brake_MaxDD"] * 100, row["twin_MaxDD"] * 100, gap, special))

    # =========================================================================
    # RETURN_BLOCK
    # =========================================================================
    return_block = {
        "script": "b1_scale_annual_20260621.py",
        "date": "2026-06-21",
        "after_tax_multiplier": AFTER_TAX,
        "sanity_A0_vs_direct_abs_diff": float(diff_a0_direct),
        "sanity_PASS": bool(diff_a0_direct <= SANITY_TOL_TIGHT),
        "part_a_csv": csvA_path,
        "part_a_rows": int(len(dfA)),
        "part_a_columns": list(dfA.columns),
        "bh_2008_at_pct": round(series_annual["NASDAQ_1x_BH_pct"].get(2008, float("nan")), 2),
        "bh_2020_at_pct": round(series_annual["NASDAQ_1x_BH_pct"].get(2020, float("nan")), 2),
        "xcheck_PASS": bool(xcheck_ok),
        "all_B1_2008_positive": bool(all_2008_pos),
        "s2_2008_2020": {
            "B1_DEF_S2.0": {"y2008": round(series_annual["B1_DEF_S2.0"].get(2008, float("nan")), 2),
                            "y2020": round(series_annual["B1_DEF_S2.0"].get(2020, float("nan")), 2)},
            "B1_STR_S2.0": {"y2008": round(series_annual["B1_STR_S2.0"].get(2008, float("nan")), 2),
                            "y2020": round(series_annual["B1_STR_S2.0"].get(2020, float("nan")), 2)},
        },
        "part_b_csv": csvB_path,
        "part_b_rows": int(len(dfB)),
        "control": [
            {
                "label": r["label"],
                "lev_scale": r["lev_scale"],
                "fbar": round(r["fbar"], 4),
                "brake_MaxDD_pct": round(r["brake_MaxDD"] * 100, 4),
                "twin_MaxDD_pct": round(r["twin_MaxDD"] * 100, 4),
                "maxdd_brake_minus_twin_pp": round(r["maxdd_brake_minus_twin_pp"], 4),
                "brake_CAGR_OOS_at_pct": round(r["brake_CAGR_OOS"] * 100, 4),
                "twin_CAGR_OOS_at_pct": round(r["twin_CAGR_OOS"] * 100, 4),
                "cagr_brake_minus_twin_pp": round(r["cagr_brake_minus_twin_pp"], 4),
            }
            for r in ctrl_rows
        ],
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))

    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
