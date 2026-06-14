"""
src/audit/multimetric_bootstrap_20260615.py
==========================================
Multi-metric paired block bootstrap for leverage-up candidates.
Evaluates 4 axes: min(IS,OOS) after-tax CAGR, MaxDD, Worst10Y*, Sharpe(OOS).

PURPOSE
-------
Resolves QC-留保6 from LEVERUP_SWEEP_RESULTS_20260612.md:
  Prior bootstrap covered only min(IS,OOS) CAGR (1 axis). LESSONS_LEARNED
  教訓C requires multi-metric bootstrap for complete risk disclosure.

IMPORTANT FRAMING -- procyclical leverage asymmetry
-----------------------------------------------------
B3a/B3c are procyclical leverage boosts (uniform x1.15 on IN days). For such
strategies:
  - CAGR / Worst10Y: expected to IMPROVE vs baseline (P_better HIGH)
  - MaxDD: expected to WORSEN vs baseline (P_maxdd_better LOW, i.e. strat DD
    is more negative = worse)
  - Sharpe: uncertain -- depends on whether CAGR gain compensates vol increase
A low P_maxdd_better for B3a is NOT a disqualifier. It quantifies the cost
(DD worsening) of the CAGR gain. The key question is whether the CAGR/Worst10Y
advantage is statistically robust, not whether DD improves.
Contrast with defensive overlays (e.g. nasdaq_mom63 x M6): those are expected
to show P_maxdd_better HIGH (DD improvement is the value).

METRICS COMPUTED PER BOOTSTRAP ITERATION
-----------------------------------------
1. min(IS,OOS) after-tax CAGR -- large = better (existing P_min_better)
2. MaxDD (full path)           -- less negative = better (P_maxdd_better)
3. Worst10Y* after-tax         -- large = better (P_worst10y_better)
   NOTE: block=21 breaks the long-term path structure; Worst10Y* under
   block=21 is an APPROXIMATION. block=252 (annual) preserves multi-year
   autocorrelation better and is reported as sensitivity. Neither is exact
   for a path-dependent metric; treat with caution.
4. Sharpe(OOS) pre-tax         -- large = better (P_sharpe_better)
   Computed from resampled OOS-mask days only.

EVALUATION PAIRS
-----------------
vs V7_TQQQ (no-fill baseline, cfd_excess=False):
  P09_C1, LU2_C1_k365, B3a_k365, B3c_k365
vs P09_C1 (incumbent candidate):
  B3a_k365, B3c_k365

REUSE (no re-implementation)
-----------------------------
  run_p09_tqqq_validate_20260611: _block_bootstrap_compare (compatibility check),
    _maxdd_from_returns, _cagr_seg, N_BOOT, BLOCK, SEED, AFTER_TAX
  k365_recost_20260612: _build_tqqq_base_param, _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE, K365_CENTRE_CONFIGS
  leverup_b1c1_20260612: _build_p09_on_base_c1
  unified_metrics: IS_END, OOS_START (canonical split)

ASCII-only prints (Windows cp932). Saves CSV. Does NOT commit. No temp files.
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
from src.audit.run_p09_tqqq_validate_20260611 import (
    _block_bootstrap_compare,
    _maxdd_from_returns,
    _cagr_seg,
    N_BOOT,
    BLOCK,
    SEED,
    AFTER_TAX,
)
from src.audit.k365_recost_20260612 import (
    _build_tqqq_base_param,
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,
    EXCESS_EXTRA_STORE,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_on_base_c1,
    _build_p09_nav_c1,
)
from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base,
    _build_p09_on_base,
    _apply_aftertax,
    LU1_MAP,
    AFTER_TAX,
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
from src.audit.run_p09_tqqq_validate_20260611 import LU2_SCALE

# ---- Candidate configs (k365 centre, C1 fill) --------------------------------
B3A_V7_MAP = {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00}
B3C_V7_MAP = {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_SCALE = 1.15
B3C_SCALE = 1.10


# ---------------------------------------------------------------------------
# Extended block bootstrap: 4-metric paired comparison
# ---------------------------------------------------------------------------

def _worst10y_from_nav_path(nav_path, n_years_window=10):
    """Rolling 10-year (calendar-year) CAGR minimum from a daily NAV path array.

    Uses calendar-year rolling (consistent with unified_metrics / nav_to_annual /
    rolling_nY_cagr) but approximated from the resampled path:
      - Build cumulative product of the resampled returns -> NAV
      - Compute annual returns from that path
      - Rolling 10-year CAGR minimum

    This is a BOOTSTRAP APPROXIMATION. The resampled path does not preserve
    long-run autocorrelation of multi-year sequences. block=21 (default) is
    especially coarse; block=252 (annual) is preferred for this metric.
    """
    nav = np.cumprod(1.0 + np.clip(nav_path, -0.999, None))
    n = len(nav)
    if n < TRADING_DAYS * (n_years_window + 1):
        return np.nan
    # Compute annual returns: split into ~252-day segments and compound each
    annual_rets = []
    seg_len = TRADING_DAYS
    for start in range(0, n - seg_len + 1, seg_len):
        seg = nav[start:start + seg_len]
        yr = float(seg[-1] / seg[0] - 1.0) if seg[0] > 0 else np.nan
        annual_rets.append(yr)
    if len(annual_rets) < n_years_window + 1:
        return np.nan
    arr = np.array(annual_rets)
    # Rolling n_years_window CAGR
    best_worst = np.inf
    for i in range(len(arr) - n_years_window + 1):
        window = arr[i:i + n_years_window]
        if np.any(np.isnan(window)):
            continue
        cum = np.prod(1.0 + window)
        cagr = cum ** (1.0 / n_years_window) - 1.0
        if cagr < best_worst:
            best_worst = cagr
    if best_worst == np.inf:
        return np.nan
    return float(best_worst)


def _sharpe_oos(r, oos_mask):
    """Pre-tax annualised Sharpe from OOS days of the resampled path."""
    r_oos = r[oos_mask]
    if len(r_oos) < 20:
        return np.nan
    mu = float(np.mean(r_oos)) * TRADING_DAYS
    sigma = float(np.std(r_oos, ddof=1)) * np.sqrt(TRADING_DAYS)
    if sigma < 1e-10:
        return np.nan
    return mu / sigma


def _block_bootstrap_multimetric(r_strat, r_base, is_mask, oos_mask,
                                  n_boot=N_BOOT, block=BLOCK, seed=SEED):
    """Paired stationary-block bootstrap -- 4 metrics.

    Computes, for each bootstrap resample, paired differences (strat - base):
      1. min(IS,OOS) after-tax CAGR  (large = better)
      2. MaxDD full-path              (less negative = better: strat_dd > base_dd)
      3. Worst10Y* after-tax          (large = better) [APPROXIMATION, see docstring]
      4. Sharpe(OOS) pre-tax          (large = better)

    Returns dict with P(strat better) and CI95 (2.5%-ile) for each metric,
    plus mean_diff. Seeds are fixed to SEED for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = len(r_strat)
    r_strat = np.asarray(r_strat, float)
    r_base = np.asarray(r_base, float)
    is_mask = np.asarray(is_mask, bool)
    oos_mask = np.asarray(oos_mask, bool)

    n_blocks = int(np.ceil(n / block))

    # Accumulators
    d_min = np.empty(n_boot)
    d_dd = np.empty(n_boot)
    d_w10y = np.empty(n_boot)
    d_sharpe = np.empty(n_boot)

    cnt_min = 0
    cnt_dd = 0
    cnt_w10y = 0
    cnt_sharpe = 0
    cnt_w10y_valid = 0  # track NaN count

    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]

        rs = r_strat[idx]
        rb = r_base[idx]
        im = is_mask[idx]
        om = oos_mask[idx]

        # ---- 1. min(IS,OOS) after-tax CAGR ----
        s_is = _cagr_seg(rs[im]) * AFTER_TAX
        s_oos = _cagr_seg(rs[om]) * AFTER_TAX
        b_is = _cagr_seg(rb[im]) * AFTER_TAX
        b_oos = _cagr_seg(rb[om]) * AFTER_TAX
        s_min = np.nanmin([s_is, s_oos])
        b_min = np.nanmin([b_is, b_oos])
        d_min[b] = s_min - b_min
        if s_min > b_min:
            cnt_min += 1

        # ---- 2. MaxDD (full path) ----
        s_dd = _maxdd_from_returns(rs)
        b_dd = _maxdd_from_returns(rb)
        d_dd[b] = s_dd - b_dd   # positive means strat DD less negative (better)
        if s_dd > b_dd:
            cnt_dd += 1

        # ---- 3. Worst10Y* after-tax (approximation) ----
        s_w10y = _worst10y_from_nav_path(rs)
        b_w10y = _worst10y_from_nav_path(rb)
        if np.isnan(s_w10y) or np.isnan(b_w10y):
            d_w10y[b] = np.nan
        else:
            d_w10y[b] = (s_w10y - b_w10y) * AFTER_TAX
            if s_w10y * AFTER_TAX > b_w10y * AFTER_TAX:
                cnt_w10y += 1
            cnt_w10y_valid += 1

        # ---- 4. Sharpe(OOS) pre-tax ----
        s_sh = _sharpe_oos(rs, om)
        b_sh = _sharpe_oos(rb, om)
        if np.isnan(s_sh) or np.isnan(b_sh):
            d_sharpe[b] = np.nan
        else:
            d_sharpe[b] = s_sh - b_sh
            if s_sh > b_sh:
                cnt_sharpe += 1

    d_w10y_valid = d_w10y[~np.isnan(d_w10y)]
    d_sharpe_valid = d_sharpe[~np.isnan(d_sharpe)]
    n_valid_w10y = len(d_w10y_valid)
    n_valid_sharpe = len(d_sharpe_valid)

    return {
        # min(IS,OOS)
        "P_min_better":        cnt_min / n_boot,
        "CI95_lo_min_pp":      float(np.percentile(d_min, 2.5)) * 100.0,
        "mean_diff_min_pp":    float(np.mean(d_min)) * 100.0,
        # MaxDD
        "P_maxdd_better":      cnt_dd / n_boot,
        "CI95_lo_dd_pp":       float(np.percentile(d_dd, 2.5)) * 100.0,
        "mean_diff_dd_pp":     float(np.mean(d_dd)) * 100.0,
        # Worst10Y*
        "P_worst10y_better":   (cnt_w10y / n_valid_w10y) if n_valid_w10y > 0 else np.nan,
        "CI95_lo_w10y_pp":     float(np.percentile(d_w10y_valid, 2.5)) * 100.0 if n_valid_w10y > 0 else np.nan,
        "mean_diff_w10y_pp":   float(np.mean(d_w10y_valid)) * 100.0 if n_valid_w10y > 0 else np.nan,
        "n_valid_w10y":        n_valid_w10y,
        # Sharpe(OOS)
        "P_sharpe_better":     (cnt_sharpe / n_valid_sharpe) if n_valid_sharpe > 0 else np.nan,
        "CI95_lo_sharpe":      float(np.percentile(d_sharpe_valid, 2.5)) if n_valid_sharpe > 0 else np.nan,
        "mean_diff_sharpe":    float(np.mean(d_sharpe_valid)) if n_valid_sharpe > 0 else np.nan,
        "n_valid_sharpe":      n_valid_sharpe,
        # Meta
        "n_boot":              n_boot,
        "block":               block,
        "seed":                seed,
    }


def _point_metrics(r, is_mask, oos_mask, nav_dt, tpy):
    """Full-path point estimates for sanity checks."""
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    return {
        "MaxDD":       pre["MaxDD_FULL"],
        "Worst10Y_at": aft["Worst10Y_star"],
        "Sharpe_OOS":  pre["Sharpe_OOS"],
        "min_at":      min(aft["CAGR_IS"], aft["CAGR_OOS"]),
        "CAGR_IS_at":  aft["CAGR_IS"],
        "CAGR_OOS_at": aft["CAGR_OOS"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("MULTI-METRIC PAIRED BLOCK BOOTSTRAP  2026-06-15")
    print("4-axis: min(IS,OOS) CAGR | MaxDD | Worst10Y* | Sharpe(OOS)")
    print("Candidates: P09_C1, LU2_C1_k365, B3a_k365, B3c_k365  vs  V7 & P09_C1")
    print("N_BOOT=%d  BLOCK=%d  SEED=%d  AFTER_TAX=%.4f" % (N_BOOT, BLOCK, SEED, AFTER_TAX))
    print()
    print("FRAMING: B3a/B3c are PROCYCLICAL boosts -> MaxDD expected to WORSEN.")
    print("  P_maxdd_better LOW for B3a/B3c vs V7 is expected and NOT a disqualifier.")
    print("  It quantifies the DD cost of the CAGR gain (trade-off disclosure).")
    print("=" * 100)

    # ---- Load shared data ----
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

    # ---- Gold/Bond auxiliary series ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0), float)
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
    # Build NAV series
    # =========================================================================
    print("\nBuilding NAV series ...")

    # V7_TQQQ (no fill, no cfd excess = TQQQ cost only)
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)
    print("  V7_TQQQ done")

    # P09_C1 (k365 centre, C1 SOFR fill on bond-OFF)
    p09c1_nav, r_p09c1, tpy_p09c1, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=None, lev_scale=1.0, excess_extra=EXCESS_EXTRA_K365_CENTRE)
    print("  P09_C1 done")

    # LU2_C1_k365
    lu2c1_nav, r_lu2c1, tpy_lu2c1, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=None, lev_scale=LU2_SCALE, excess_extra=EXCESS_EXTRA_K365_CENTRE)
    print("  LU2_C1_k365 done")

    # B3a_k365: v7_map={0:1.4,1:1.4,2:1.05,3:1.0} x lev_scale=1.15 x k365 centre x C1
    b3a_nav, r_b3a, tpy_b3a, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3A_V7_MAP, lev_scale=B3A_SCALE, excess_extra=EXCESS_EXTRA_K365_CENTRE)
    print("  B3a_k365 done")

    # B3c_k365: same v7_map but lev_scale=1.10
    b3c_nav, r_b3c, tpy_b3c, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3C_V7_MAP, lev_scale=B3C_SCALE, excess_extra=EXCESS_EXTRA_K365_CENTRE)
    print("  B3c_k365 done")

    # =========================================================================
    # Point estimates (full-path, for sanity check)
    # =========================================================================
    print("\nComputing full-path point estimates ...")
    pm_v7   = _point_metrics(r_v7,    is_mask, oos_mask, v7_nav,    tpy_v7)
    pm_p09  = _point_metrics(r_p09c1, is_mask, oos_mask, p09c1_nav, tpy_p09c1)
    pm_lu2  = _point_metrics(r_lu2c1, is_mask, oos_mask, lu2c1_nav, tpy_lu2c1)
    pm_b3a  = _point_metrics(r_b3a,   is_mask, oos_mask, b3a_nav,   tpy_b3a)
    pm_b3c  = _point_metrics(r_b3c,   is_mask, oos_mask, b3c_nav,   tpy_b3c)

    print("\nPOINT ESTIMATES (full path, pretax DD/Sharpe, aftertax CAGR/W10Y):")
    print("%-18s | %9s | %9s | %9s | %9s | %9s"
          % ("label", "MaxDD%", "W10Y*_at%", "Sharpe", "min_at%", "CAGR_OOS_at%"))
    print("-" * 85)
    for lbl, pm in [("V7_TQQQ", pm_v7), ("P09_C1", pm_p09), ("LU2_C1_k365", pm_lu2),
                    ("B3a_k365", pm_b3a), ("B3c_k365", pm_b3c)]:
        print("%-18s | %+8.2f%% | %+8.2f%% | %8.3f | %+8.2f%% | %+8.2f%%"
              % (lbl, pm["MaxDD"] * 100, pm["Worst10Y_at"] * 100,
                 pm["Sharpe_OOS"], pm["min_at"] * 100, pm["CAGR_OOS_at"] * 100))

    # =========================================================================
    # SANITY CHECK 1: P_min_better compatibility with existing _block_bootstrap_compare
    # =========================================================================
    print("\n" + "=" * 100)
    print("SANITY CHECK 1: P_min_better from new function vs existing _block_bootstrap_compare")
    print("  Comparing P09_C1 vs V7_TQQQ (expect within +-0.05 of each other)")
    print("  NOTE: P09_C1 uses k365 cost; existing compare may have used store-CFD.")
    print("=" * 100)

    existing = _block_bootstrap_compare(r_p09c1, r_v7, is_mask, oos_mask,
                                         n_boot=N_BOOT, block=BLOCK, seed=SEED)
    new_mm = _block_bootstrap_multimetric(r_p09c1, r_v7, is_mask, oos_mask,
                                           n_boot=N_BOOT, block=BLOCK, seed=SEED)
    print("  existing P_min_better: %.4f" % existing["P_min_better"])
    print("  new      P_min_better: %.4f" % new_mm["P_min_better"])
    diff_check = abs(existing["P_min_better"] - new_mm["P_min_better"])
    print("  delta: %.4f -> %s" % (diff_check, "OK (within 0.05)" if diff_check < 0.05 else "WARN (>0.05)"))

    # =========================================================================
    # SANITY CHECK 2: Point estimates vs known values
    # =========================================================================
    print("\n" + "=" * 100)
    print("SANITY CHECK 2: Point estimates vs known values from k365_recost CSV")
    print("  B3a_k365:  MaxDD~-38.2%  Worst10Y*~+14.53%  Sharpe~0.904")
    print("  P09_C1:    MaxDD~-35.0%  Worst10Y*~+11.49%  Sharpe~0.912")
    print("  V7_TQQQ:   MaxDD~-34.5%  Worst10Y*~+10.08%  Sharpe~0.877")
    print("=" * 100)
    checks = [
        ("B3a_k365 MaxDD",       pm_b3a["MaxDD"] * 100,      -38.2, 1.5),
        ("B3a_k365 Worst10Y_at", pm_b3a["Worst10Y_at"] * 100, 14.53, 2.0),
        ("B3a_k365 Sharpe",      pm_b3a["Sharpe_OOS"],         0.904, 0.05),
        ("P09_C1 MaxDD",         pm_p09["MaxDD"] * 100,       -35.0, 1.5),
        ("P09_C1 Worst10Y_at",   pm_p09["Worst10Y_at"] * 100,  11.49, 2.0),
        ("P09_C1 Sharpe",        pm_p09["Sharpe_OOS"],          0.912, 0.05),
        ("V7_TQQQ MaxDD",        pm_v7["MaxDD"] * 100,         -34.5, 1.5),
        ("V7_TQQQ Worst10Y_at",  pm_v7["Worst10Y_at"] * 100,   10.08, 2.0),
        ("V7_TQQQ Sharpe",       pm_v7["Sharpe_OOS"],           0.877, 0.05),
    ]
    any_fail = False
    for name, val, expect, tol in checks:
        ok = abs(val - expect) <= tol
        if not ok:
            any_fail = True
        print("  %-32s got %+.3f  expect ~%+.3f  tol=%.3f  -> %s"
              % (name, val, expect, tol, "OK" if ok else "WARN"))
    if any_fail:
        print("\n  WARN: Some sanity checks outside tolerance. Results still printed.")
        print("  (Tolerance is generous; minor deviations from C1/k365 cost variant are normal.)")
    else:
        print("\n  All sanity checks PASSED.")

    # =========================================================================
    # MAIN BOOTSTRAP RUNS: block=21 and block=252
    # =========================================================================
    PAIRS = [
        # (strat_label, r_strat, base_label, r_base)
        ("P09_C1",      r_p09c1, "V7_TQQQ",  r_v7),
        ("LU2_C1_k365", r_lu2c1, "V7_TQQQ",  r_v7),
        ("B3a_k365",    r_b3a,   "V7_TQQQ",  r_v7),
        ("B3c_k365",    r_b3c,   "V7_TQQQ",  r_v7),
        ("B3a_k365",    r_b3a,   "P09_C1",   r_p09c1),
        ("B3c_k365",    r_b3c,   "P09_C1",   r_p09c1),
    ]

    results_21 = {}
    results_252 = {}

    print("\n" + "=" * 100)
    print("BOOTSTRAP BLOCK=21 (standard, %d iterations)" % N_BOOT)
    print("=" * 100)
    for (sl, rs, bl, rb) in PAIRS:
        key = "%s_vs_%s" % (sl, bl)
        print("  Running %s ..." % key, end="", flush=True)
        res = _block_bootstrap_multimetric(rs, rb, is_mask, oos_mask,
                                            n_boot=N_BOOT, block=21, seed=SEED)
        results_21[key] = {"strat": sl, "base": bl, **res}
        print(" done  P_min=%.3f  P_dd=%.3f  P_w10y=%.3f  P_sharpe=%.3f"
              % (res["P_min_better"], res["P_maxdd_better"],
                 res["P_worst10y_better"] if not np.isnan(res["P_worst10y_better"]) else -1,
                 res["P_sharpe_better"] if not np.isnan(res["P_sharpe_better"]) else -1))

    print("\n" + "=" * 100)
    print("BOOTSTRAP BLOCK=252 (annual sensitivity, %d iterations)" % N_BOOT)
    print("NOTE: block=252 better preserves multi-year structure for Worst10Y*")
    print("=" * 100)
    for (sl, rs, bl, rb) in PAIRS:
        key = "%s_vs_%s" % (sl, bl)
        print("  Running %s ..." % key, end="", flush=True)
        res = _block_bootstrap_multimetric(rs, rb, is_mask, oos_mask,
                                            n_boot=N_BOOT, block=252, seed=SEED)
        results_252[key] = {"strat": sl, "base": bl, **res}
        print(" done  P_min=%.3f  P_dd=%.3f  P_w10y=%.3f  P_sharpe=%.3f"
              % (res["P_min_better"], res["P_maxdd_better"],
                 res["P_worst10y_better"] if not np.isnan(res["P_worst10y_better"]) else -1,
                 res["P_sharpe_better"] if not np.isnan(res["P_sharpe_better"]) else -1))

    # =========================================================================
    # Result tables
    # =========================================================================
    def _fmt_p(v):
        if isinstance(v, float) and np.isnan(v):
            return "  nan  "
        return "%6.3f " % v

    def _fmt_pp(v):
        if isinstance(v, float) and np.isnan(v):
            return "   nan  "
        return "%+7.2f%%" % (v)

    print("\n" + "=" * 100)
    print("TABLE 1: vs V7_TQQQ  [block=21]")
    print("(P_min/P_sharpe/P_worst10y HIGH=good, P_maxdd LOW=expected for procyclical leverage)")
    print("%-20s | %7s | %8s | %7s | %8s | %7s | %8s | %7s | %8s"
          % ("strat vs V7", "P_min", "CI95_min", "P_maxdd", "CI95_dd", "P_w10y", "CI95_w10y", "P_shp", "CI95_shp"))
    print("-" * 100)
    for sl in ["P09_C1", "LU2_C1_k365", "B3a_k365", "B3c_k365"]:
        key = "%s_vs_V7_TQQQ" % sl
        r = results_21[key]
        print("%-20s | %s | %s | %s | %s | %s | %s | %s | %s"
              % (sl,
                 _fmt_p(r["P_min_better"]), _fmt_pp(r["CI95_lo_min_pp"]),
                 _fmt_p(r["P_maxdd_better"]), _fmt_pp(r["CI95_lo_dd_pp"]),
                 _fmt_p(r["P_worst10y_better"]), _fmt_pp(r["CI95_lo_w10y_pp"]),
                 _fmt_p(r["P_sharpe_better"]), _fmt_pp(r["CI95_lo_sharpe"])))

    print("\n" + "=" * 100)
    print("TABLE 2: vs P09_C1  [block=21]")
    print("('Does B3a/B3c beat P09_C1 on CAGR? How much worse is MaxDD?')")
    print("%-20s | %7s | %8s | %7s | %8s | %7s | %8s | %7s | %8s"
          % ("strat vs P09_C1", "P_min", "CI95_min", "P_maxdd", "CI95_dd", "P_w10y", "CI95_w10y", "P_shp", "CI95_shp"))
    print("-" * 100)
    for sl in ["B3a_k365", "B3c_k365"]:
        key = "%s_vs_P09_C1" % sl
        r = results_21[key]
        print("%-20s | %s | %s | %s | %s | %s | %s | %s | %s"
              % (sl,
                 _fmt_p(r["P_min_better"]), _fmt_pp(r["CI95_lo_min_pp"]),
                 _fmt_p(r["P_maxdd_better"]), _fmt_pp(r["CI95_lo_dd_pp"]),
                 _fmt_p(r["P_worst10y_better"]), _fmt_pp(r["CI95_lo_w10y_pp"]),
                 _fmt_p(r["P_sharpe_better"]), _fmt_pp(r["CI95_lo_sharpe"])))

    print("\n" + "=" * 100)
    print("TABLE 3: block=21 vs block=252 sensitivity (Worst10Y* and MaxDD rows)")
    print("%-20s | %-12s | %7s(21) | %8s(21) | %7s(252) | %8s(252)"
          % ("strat_vs_base", "metric", "P_better", "CI95_lo", "P_better", "CI95_lo"))
    print("-" * 100)
    for (sl, _, bl, _) in PAIRS:
        key = "%s_vs_%s" % (sl, bl)
        r21  = results_21[key]
        r252 = results_252[key]
        for metric, p21, ci21, p252, ci252 in [
            ("Worst10Y*",
             r21["P_worst10y_better"], r21["CI95_lo_w10y_pp"],
             r252["P_worst10y_better"], r252["CI95_lo_w10y_pp"]),
            ("MaxDD",
             r21["P_maxdd_better"], r21["CI95_lo_dd_pp"],
             r252["P_maxdd_better"], r252["CI95_lo_dd_pp"]),
        ]:
            print("%-20s | %-12s | %s | %s | %s | %s"
                  % (key[:20], metric,
                     _fmt_p(p21), _fmt_pp(ci21),
                     _fmt_p(p252), _fmt_pp(ci252)))

    # =========================================================================
    # CONCLUSION BLOCK
    # =========================================================================
    print("\n" + "=" * 100)
    print("CONCLUSION: B3a_k365 exchange-rate summary (CAGR premium vs DD cost)")
    print("=" * 100)
    r_b3a_v7  = results_21["B3a_k365_vs_V7_TQQQ"]
    r_b3a_p09 = results_21["B3a_k365_vs_P09_C1"]
    r_b3c_v7  = results_21["B3c_k365_vs_V7_TQQQ"]
    r_b3c_p09 = results_21["B3c_k365_vs_P09_C1"]

    def _sig(p, thr=0.95):
        if np.isnan(p):
            return "n/a"
        return "SIGNIFICANT" if p >= thr else ("borderline" if p >= 0.80 else "NOT significant")

    print()
    print("B3a_k365 vs V7_TQQQ:")
    print("  P_min_better   = %.3f  CI95_lo=%s  -> CAGR advantage %s"
          % (r_b3a_v7["P_min_better"], _fmt_pp(r_b3a_v7["CI95_lo_min_pp"]),
             _sig(r_b3a_v7["P_min_better"])))
    print("  P_worst10y     = %.3f  CI95_lo=%s  -> Worst10Y* advantage %s"
          % (r_b3a_v7["P_worst10y_better"], _fmt_pp(r_b3a_v7["CI95_lo_w10y_pp"]),
             _sig(r_b3a_v7["P_worst10y_better"])))
    print("  P_sharpe       = %.3f  CI95_lo=%s  -> Sharpe advantage %s"
          % (r_b3a_v7["P_sharpe_better"], _fmt_pp(r_b3a_v7["CI95_lo_sharpe"]),
             _sig(r_b3a_v7["P_sharpe_better"])))
    print("  P_maxdd_better = %.3f  [EXPECTED LOW -- DD worsens for procyclical lev]"
          % r_b3a_v7["P_maxdd_better"])
    print("  DD cost: mean_diff_dd = %s  (negative = B3a DD more negative)"
          % _fmt_pp(r_b3a_v7["mean_diff_dd_pp"]))

    print()
    print("B3a_k365 vs P09_C1 (incumbent):")
    print("  P_min_better   = %.3f  CI95_lo=%s  -> CAGR advantage %s"
          % (r_b3a_p09["P_min_better"], _fmt_pp(r_b3a_p09["CI95_lo_min_pp"]),
             _sig(r_b3a_p09["P_min_better"])))
    print("  P_maxdd_better = %.3f  [EXPECTED LOW -- B3a DD > P09 DD cost]"
          % r_b3a_p09["P_maxdd_better"])
    print("  DD premium vs P09: mean_diff_dd = %s"
          % _fmt_pp(r_b3a_p09["mean_diff_dd_pp"]))

    print()
    print("B3c_k365 vs V7_TQQQ (lighter version at x1.10):")
    print("  P_min_better   = %.3f  P_maxdd_better=%.3f  P_worst10y=%.3f  P_sharpe=%.3f"
          % (r_b3c_v7["P_min_better"], r_b3c_v7["P_maxdd_better"],
             r_b3c_v7["P_worst10y_better"] if not np.isnan(r_b3c_v7["P_worst10y_better"]) else float("nan"),
             r_b3c_v7["P_sharpe_better"] if not np.isnan(r_b3c_v7["P_sharpe_better"]) else float("nan")))

    # =========================================================================
    # Save CSV
    # =========================================================================
    rows = []
    for blk_label, res_dict in [("21", results_21), ("252", results_252)]:
        for key, r in res_dict.items():
            for metric, p_col, ci_col, mean_col in [
                ("min_IS_OOS_CAGR_at",  "P_min_better",     "CI95_lo_min_pp",  "mean_diff_min_pp"),
                ("MaxDD",               "P_maxdd_better",   "CI95_lo_dd_pp",   "mean_diff_dd_pp"),
                ("Worst10Y_star_at",    "P_worst10y_better","CI95_lo_w10y_pp", "mean_diff_w10y_pp"),
                ("Sharpe_OOS",          "P_sharpe_better",  "CI95_lo_sharpe",  "mean_diff_sharpe"),
            ]:
                rows.append({
                    "strat":        r["strat"],
                    "base":         r["base"],
                    "metric":       metric,
                    "block":        int(blk_label),
                    "P_better":     r[p_col],
                    "CI95_lo":      r[ci_col],
                    "mean_diff":    r[mean_col],
                    "n_boot":       r["n_boot"],
                    "seed":         r["seed"],
                })

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "multimetric_bootstrap_20260615.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nCSV saved: %s" % csv_path)

    # =========================================================================
    # RETURN_BLOCK JSON
    # =========================================================================
    return_block = {
        "script": "multimetric_bootstrap_20260615.py",
        "date": "2026-06-15",
        "n_boot": N_BOOT,
        "blocks_tested": [21, 252],
        "sanity_p_min_delta": float(diff_check),
        "point_estimates": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in [("V7_TQQQ", pm_v7), ("P09_C1", pm_p09),
                         ("LU2_C1_k365", pm_lu2), ("B3a_k365", pm_b3a),
                         ("B3c_k365", pm_b3c)]
        },
        "block21": {k: {kk: (float(vv) if not isinstance(vv, int) else vv)
                        for kk, vv in v.items() if kk not in ("strat", "base")}
                    for k, v in results_21.items()},
        "block252": {k: {kk: (float(vv) if not isinstance(vv, int) else vv)
                         for kk, vv in v.items() if kk not in ("strat", "base")}
                     for k, v in results_252.items()},
        "csv_path": csv_path,
    }
    print("\nRETURN_BLOCK:")
    print(json.dumps(return_block, indent=2, ensure_ascii=False, allow_nan=True))

    return return_block


if __name__ == "__main__":
    main()
