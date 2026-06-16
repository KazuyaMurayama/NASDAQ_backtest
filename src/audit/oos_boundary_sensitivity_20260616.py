"""
src/audit/oos_boundary_sensitivity_20260616.py
===============================================
OOS Boundary Sensitivity Analysis -- 2026-06-16

PURPOSE
-------
Directly address the top-priority methodological gap identified in the audit:
"single OOS (2021-05 to 2026) covers mostly a bull market -- is the levert
extension min9 advantage merely an artifact of this split?"

For each of 4 target configurations (B3a / scale1.25 / scale1.35 / combo
enter0.60 x scale1.25), we shift OOS_START to 5 different boundary points
and recompute min(IS,OOS) after-tax CAGR, CAGR_IS, CAGR_OOS, IS-OOS gap.
The key question: is the leverage extension's advantage over B3a STABLE across
boundary shifts, or does it disappear when the OOS window includes weak periods?

CONFIGS
-------
  B3a      : scale=1.15, v7_map default, enter=0.70  (baseline)
  scale125 : scale=1.25, v7_map default, enter=0.70
  scale135 : scale=1.35, v7_map strong,  enter=0.70
  combo    : enter=0.60, scale=1.25, v7_map default

OOS_START LEVELS (5 boundary points)
--------------------------------------
  2019-05-01 -- pre-COVID, includes 2020 crash in OOS
  2020-05-01 -- includes COVID recovery in OOS
  2021-05-08 -- canonical (current standard)
  2022-05-01 -- shorter OOS (post-2021 bear included in IS)
  2023-05-01 -- shortest OOS (5 bear years in IS)

KEY METRIC
----------
  Delta = min9(config, boundary) - min9(B3a, boundary)  [pp]
  We report Delta for each config x boundary, plus std(Delta) and range(Delta)
  across boundaries. A config whose advantage is driven purely by the 2021-05
  strong-bull OOS will show high variability in Delta.

TASK 2: REGIME STRATIFICATION + WORST5Y
-----------------------------------------
Using the CANONICAL boundary (2021-05-08) for each config:
  - trend:bear CAGR (after-tax) via _regime_cagr
  - regime_min (minimum across all 6 axes)
  - Worst5Y (5-year rolling minimum, after-tax)
  - Whether scale1.25 / scale1.35 / combo go negative on Worst5Y

SANITY GATE
-----------
  Canonical boundary, B3a config must reproduce min9 +20.98% +/-0.05pp
  and MaxDD -38.20% +/-0.10pp.

LOOKAHEAD NOTE
--------------
  Boundary shifts change ONLY which segment of the already-computed NAV is
  labelled IS vs OOS for CAGR reporting purposes.  The NAV itself is built
  once over the full history; no retrain / re-optimise occurs for each boundary.
  This is correct (avoids lookahead): the strategy rules are fixed; we are
  only changing the EVALUATION window, not the strategy.

OUTPUTS
-------
  audit_results/oos_boundary_sensitivity_20260616.csv
  RETURN_BLOCK (json) printed to stdout.

ASCII-only (cp932). No git. No temp files.
"""

from __future__ import annotations

import csv
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

# ---- B3a / k365 NAV builder --------------------------------------------------
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,
    EXCESS_EXTRA_STORE,
)

# ---- Misc helpers ------------------------------------------------------------
from src.audit.run_p09_tqqq_validate_20260611 import (
    _cagr_seg, _maxdd_from_returns,
    N_BOOT, BLOCK, SEED,
    LU2_SCALE,
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

# ---- combo: DH-W1 mask with tunable enter threshold -------------------------
from src.audit.leverext_reentry_20260616 import (
    _hold_mask_W1_param,
    _load_base_shared,
    _build_shared_for_enter as _reentry_build_shared_for_enter,
    EXIT_THR,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_on_base_c1,
)
from src.audit.lu_cfd_recost_20260611 import (
    AFTER_TAX, SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)

# ---- Extended eval for regime (Task 2) -----------------------------------
from src.audit.extended_eval_20260611 import (
    _regime_cagr,
)

# (compute_worst5y imported inside _worst5y_at via calculate_p10_5y)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B3A_MAP_DEFAULT = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_MAP_STRONG  = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}

EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025 (k365 centre)

# Canonical sanity reference
B3A_KNOWN_MIN9   = 0.2098
B3A_KNOWN_MAXDD  = -0.3820
SANITY_TOL_MIN9  = 0.0005
SANITY_TOL_MAXDD = 0.0010

# OOS boundary sweep (5 points)
OOS_BOUNDARIES = [
    ("2019-05-01", "2019-04-30"),
    ("2020-05-01", "2020-04-30"),
    ("2021-05-08", "2021-05-07"),   # canonical
    ("2022-05-01", "2022-04-30"),
    ("2023-05-01", "2023-04-30"),
]
# Each tuple: (oos_start_str, is_end_str)

# Config definitions
CONFIGS = [
    {
        "label":     "B3a",
        "enter_thr": 0.70,
        "lev_scale": 1.15,
        "v7_map":    B3A_MAP_DEFAULT,
    },
    {
        "label":     "scale125",
        "enter_thr": 0.70,
        "lev_scale": 1.25,
        "v7_map":    B3A_MAP_DEFAULT,
    },
    {
        "label":     "scale135",
        "enter_thr": 0.70,
        "lev_scale": 1.35,
        "v7_map":    B3A_MAP_STRONG,
    },
    {
        "label":     "combo_e060_sc125",
        "enter_thr": 0.60,
        "lev_scale": 1.25,
        "v7_map":    B3A_MAP_DEFAULT,
    },
]

OUT_CSV = os.path.join(_REPO_DIR, "audit_results",
                       "oos_boundary_sensitivity_20260616.csv")


# ---------------------------------------------------------------------------
# NAV builders
# ---------------------------------------------------------------------------

def _build_shared_for_enter(enter_thr: float):
    """Delegate to leverext_reentry's validated _build_shared_for_enter."""
    return _reentry_build_shared_for_enter(enter_thr)


def _build_nav_for_config(cfg, dates_dt, n_years,
                          ret_gold, ret_bond, wg, wb, bond_on, sofr_arr):
    """Build full C1 NAV for a given config. Returns (nav_dt, r, tpy, exc)."""
    enter_thr = cfg["enter_thr"]
    lev_scale = cfg["lev_scale"]
    v7_map    = cfg["v7_map"]

    if abs(enter_thr - 0.70) < 1e-9:
        # Standard B3a-style: use the default sr._DHW1_SHARED
        shared = sr._DHW1_SHARED
    else:
        shared = _build_shared_for_enter(enter_thr)

    nav_dt, r, tpy, exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond,
        _get_fund_active(shared, n_years, dates_dt),
        wg, wb, bond_on, sofr_arr,
        v7_map=v7_map, lev_scale=lev_scale,
        excess_extra=EXCESS_EXTRA)
    return nav_dt, r, tpy, exc


def _get_fund_active(shared, n_years, dates_dt):
    """Reconstruct fund_active from shared mask (T+5 lag)."""
    mask = np.asarray(shared["mask"], float)
    n = len(mask)
    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]
    return fund_active


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _cagr_seg_annualized(r, n_years_frac):
    """After-tax CAGR over a segment of length n_years_frac years."""
    r = np.asarray(r, float)
    n = len(r)
    if n == 0 or n_years_frac <= 0:
        return np.nan
    nav_end = float(np.prod(1.0 + np.clip(r, -0.999, None)))
    if nav_end <= 0:
        return -1.0
    return nav_end ** (1.0 / n_years_frac) - 1.0


def _boundary_metrics(r, dates_dt, oos_start_str, is_end_str):
    """
    Compute CAGR_IS, CAGR_OOS, min9 (after-tax) and IS-OOS gap for a given
    OOS boundary, without re-running any NAV simulation.

    Parameters
    ----------
    r          : daily returns array (full history, aligned with dates_dt)
    dates_dt   : full DatetimeIndex
    oos_start_str : first day of OOS (ISO string)
    is_end_str    : last day of IS  (ISO string)

    Returns dict with keys: CAGR_IS_at, CAGR_OOS_at, min9_at, gap_pp.
    """
    oos_start = pd.Timestamp(oos_start_str)
    is_end    = pd.Timestamp(is_end_str)

    is_mask  = np.asarray(dates_dt <= is_end, dtype=bool)
    oos_mask = np.asarray(dates_dt >= oos_start, dtype=bool)

    n_is  = int(is_mask.sum())
    n_oos = int(oos_mask.sum())

    if n_is < 2 or n_oos < 2:
        return {"CAGR_IS_at": np.nan, "CAGR_OOS_at": np.nan,
                "min9_at": np.nan, "gap_pp": np.nan,
                "n_is": n_is, "n_oos": n_oos}

    r_is  = r[is_mask]
    r_oos = r[oos_mask]

    yrs_is  = n_is  / float(TRADING_DAYS)
    yrs_oos = n_oos / float(TRADING_DAYS)

    cagr_is  = _cagr_seg_annualized(r_is,  yrs_is)  * AFTER_TAX
    cagr_oos = _cagr_seg_annualized(r_oos, yrs_oos) * AFTER_TAX

    min9 = min(cagr_is, cagr_oos)
    gap  = (cagr_is - cagr_oos) * 100.0  if (np.isfinite(cagr_is) and np.isfinite(cagr_oos)) else np.nan

    return {
        "CAGR_IS_at":  round(cagr_is * 100, 4),
        "CAGR_OOS_at": round(cagr_oos * 100, 4),
        "min9_at":     round(min9 * 100, 4),
        "gap_pp":      round(gap, 4) if np.isfinite(gap) else np.nan,
        "n_is":        n_is,
        "n_oos":       n_oos,
    }


def _worst5y_at(nav_dt):
    """Worst 5-year rolling CAGR (after-tax)."""
    try:
        from calculate_p10_5y import compute_worst5y as cw5y
        w5 = cw5y(nav_dt.values)
        return round(float(w5) * AFTER_TAX * 100, 4)
    except Exception:
        return np.nan


def _worst5y_from_returns(r):
    """Worst 5-year rolling CAGR from daily returns (after-tax)."""
    # Reconstruct NAV from returns, then compute worst5y
    r = np.asarray(r, float)
    nav = np.cumprod(1.0 + np.clip(r, -0.999, None))
    try:
        from calculate_p10_5y import compute_worst5y as cw5y
        w5 = cw5y(nav)
        return round(float(w5) * AFTER_TAX * 100, 4)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("OOS BOUNDARY SENSITIVITY  2026-06-16")
    print("Purpose: Quantify how much levert-extension min9 advantage depends on")
    print("  the specific 2021-05 OOS boundary (strong-bull start).")
    print("OOS boundaries: 2019-05 / 2020-05 / 2021-05(canonical) / 2022-05 / 2023-05")
    print("Configs: B3a / scale125 / scale135 / combo_e060_sc125")
    print("NAV is built ONCE per config (full history); boundary shifts only change")
    print("  which segment is labelled IS vs OOS for CAGR computation (no lookahead).")
    print("=" * 120)

    # ---- Load shared data (default enter=0.70) ----
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)

    # Canonical IS/OOS masks (for sanity + regime)
    is_mask_canonical  = np.asarray(dates_dt <= IS_END)
    oos_mask_canonical = np.asarray(dates_dt >= OOS_START)

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

    # ---- Regime labels (for Task 2) ----
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)

    # =========================================================================
    # SANITY GATE: canonical boundary, B3a
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: B3a at canonical boundary (2021-05-08)")
    print("  Expected: min9 +20.98%% +/-0.05pp  MaxDD -38.20%% +/-0.10pp")
    print("=" * 120)

    san_nav, san_r, san_tpy, san_exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3A_MAP_DEFAULT, lev_scale=1.15,
        excess_extra=EXCESS_EXTRA)

    san_pre = compute_10metrics(san_nav, san_tpy)
    san_aft = _apply_aftertax(san_pre)
    san_min9  = min(san_aft["CAGR_IS"], san_aft["CAGR_OOS"])
    san_maxdd = san_pre["MaxDD_FULL"]

    ok_min9  = abs(san_min9  - B3A_KNOWN_MIN9)  <= SANITY_TOL_MIN9
    ok_maxdd = abs(san_maxdd - B3A_KNOWN_MAXDD) <= SANITY_TOL_MAXDD

    print("  min9:  got %+.4f%%  expect ~%+.4f%%  diff=%.4fpp  -> %s"
          % (san_min9 * 100, B3A_KNOWN_MIN9 * 100,
             (san_min9 - B3A_KNOWN_MIN9) * 100,
             "OK" if ok_min9 else "FAIL"))
    print("  MaxDD: got %+.4f%%  expect ~%+.4f%%  diff=%.4fpp  -> %s"
          % (san_maxdd * 100, B3A_KNOWN_MAXDD * 100,
             (san_maxdd - B3A_KNOWN_MAXDD) * 100,
             "OK" if ok_maxdd else "FAIL"))

    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED -- halting.")
        sys.exit(1)
    print("  SANITY PASSED.\n")

    # =========================================================================
    # BUILD NAVs for all 4 configs (once, full history)
    # =========================================================================
    print("=" * 120)
    print("Building NAVs for all 4 configs (full history -- one-time computation)")
    print("=" * 120)

    nav_cache = {}
    r_cache   = {}
    tpy_cache = {}

    for cfg in CONFIGS:
        lbl = cfg["label"]
        enter_thr = cfg["enter_thr"]
        lev_scale = cfg["lev_scale"]
        v7_map    = cfg["v7_map"]
        print("  [%s] enter=%.2f scale=%.2f map=%s ..."
              % (lbl, enter_thr, lev_scale,
                 "default" if v7_map == B3A_MAP_DEFAULT else "strong"))

        if abs(enter_thr - 0.70) < 1e-9:
            shared_cfg = shared
            fa_cfg = fund_active
        else:
            shared_cfg = _build_shared_for_enter(enter_thr)
            fa_cfg = _get_fund_active(shared_cfg, n_years, dates_dt)

        nav_dt, r, tpy, exc = _build_full_c1(
            shared_cfg, dates_dt, n_years,
            ret_gold, ret_bond, fa_cfg, wg, wb, bond_on, sofr_arr,
            v7_map=v7_map, lev_scale=lev_scale,
            excess_extra=EXCESS_EXTRA)

        nav_cache[lbl] = nav_dt
        r_cache[lbl]   = r
        tpy_cache[lbl] = tpy
        print("    built: %d days, tpy=%.1f" % (len(r), tpy))

    # =========================================================================
    # TASK 1: OOS BOUNDARY SENSITIVITY
    # =========================================================================
    print("\n" + "=" * 120)
    print("TASK 1: OOS BOUNDARY SENSITIVITY")
    print("5 boundary points x 4 configs = 20 evaluations")
    print("=" * 120)

    boundary_rows = []

    for oos_str, is_str in OOS_BOUNDARIES:
        oos_start = pd.Timestamp(oos_str)
        is_end_ts = pd.Timestamp(is_str)
        oos_frac = float(np.sum(dates_dt >= oos_start)) / float(TRADING_DAYS)
        is_frac  = float(np.sum(dates_dt <= is_end_ts)) / float(TRADING_DAYS)
        canon = (oos_str == "2021-05-08")
        print("\n  OOS_START=%s  IS_END=%s  (IS=%.1fyr OOS=%.1fyr)%s"
              % (oos_str, is_str, is_frac, oos_frac,
                 "  [CANONICAL]" if canon else ""))

        for cfg in CONFIGS:
            lbl = cfg["label"]
            r   = r_cache[lbl]
            m   = _boundary_metrics(r, dates_dt, oos_str, is_str)
            row = {
                "config":      lbl,
                "oos_start":   oos_str,
                "is_end":      is_str,
                "canonical":   int(canon),
                "n_is":        m["n_is"],
                "n_oos":       m["n_oos"],
                "CAGR_IS_at":  m["CAGR_IS_at"],
                "CAGR_OOS_at": m["CAGR_OOS_at"],
                "min9_at":     m["min9_at"],
                "gap_pp":      m["gap_pp"],
            }
            boundary_rows.append(row)
            print("    %-24s  IS=%+6.2f%%  OOS=%+6.2f%%  min9=%+6.2f%%  gap=%+5.2fpp"
                  % (lbl, m["CAGR_IS_at"], m["CAGR_OOS_at"],
                     m["min9_at"], m["gap_pp"] if np.isfinite(m["gap_pp"]) else 0))

    # ---- Compute Delta (config - B3a) at each boundary ----
    print("\n" + "-" * 120)
    print("DELTA TABLE (config - B3a) per boundary point [pp]")
    print("-" * 120)

    # Index by (oos_start, config)
    brow_idx = {(r["oos_start"], r["config"]): r for r in boundary_rows}

    config_labels = [c["label"] for c in CONFIGS[1:]]  # skip B3a itself
    oos_labels    = [t[0] for t in OOS_BOUNDARIES]

    hdr = "%-12s" % "OOS_START"
    for lbl in config_labels:
        hdr += "  %+12s" % (lbl + "_delta")
    print(hdr)

    delta_store = {lbl: [] for lbl in config_labels}

    for oos_str, _ in OOS_BOUNDARIES:
        b3a_min9 = brow_idx.get((oos_str, "B3a"), {}).get("min9_at", np.nan)
        row_str = "%-12s" % oos_str
        for lbl in config_labels:
            cfg_min9 = brow_idx.get((oos_str, lbl), {}).get("min9_at", np.nan)
            if np.isfinite(b3a_min9) and np.isfinite(cfg_min9):
                delta = cfg_min9 - b3a_min9
            else:
                delta = np.nan
            delta_store[lbl].append(delta)
            row_str += "  %+12.2f" % (delta if np.isfinite(delta) else 0.0)
        print(row_str)

    # ---- Delta stability stats ----
    print("\nDelta stability across 5 boundary points:")
    print("%-24s  %8s  %8s  %8s  %8s  %8s  %8s"
          % ("config", "mean_pp", "std_pp", "range_pp", "min_pp", "max_pp", "2021_delta"))

    for lbl in config_labels:
        deltas = np.asarray([d for d in delta_store[lbl] if np.isfinite(d)])
        canon_idx = [i for i, (oos, _) in enumerate(OOS_BOUNDARIES) if oos == "2021-05-08"]
        canon_delta = delta_store[lbl][canon_idx[0]] if canon_idx else np.nan
        if len(deltas) >= 2:
            print("%-24s  %+7.2f  %7.2f  %7.2f  %+7.2f  %+7.2f  %+7.2f"
                  % (lbl, deltas.mean(), deltas.std(), deltas.max() - deltas.min(),
                     deltas.min(), deltas.max(), canon_delta))
        else:
            print("%-24s  (insufficient valid deltas)" % lbl)

    # ---- Verdict ----
    print("\n--- VERDICT: Is the 2021-05 boundary delta larger than at other boundaries? ---")
    for lbl in config_labels:
        deltas = delta_store[lbl]
        canon_idx_list = [i for i, (oos, _) in enumerate(OOS_BOUNDARIES) if oos == "2021-05-08"]
        if not canon_idx_list:
            continue
        ci = canon_idx_list[0]
        cd = deltas[ci]
        others = [d for i, d in enumerate(deltas) if i != ci and np.isfinite(d)]
        if len(others) == 0 or not np.isfinite(cd):
            print("  %-24s  (insufficient data)" % lbl)
            continue
        max_other = max(others)
        stable = abs(cd - max_other) < 2.0  # within 2pp of max non-canonical
        verdict = ("ROBUST (2021-05 advantage not uniquely large; max_other_delta=%+.2f%%)"
                   % max_other) if stable else (
                   "SUSPECT (2021-05 delta=%+.2f%% >> max_other=%+.2f%%)" % (cd, max_other))
        print("  %-24s  %s" % (lbl, verdict))

    # =========================================================================
    # TASK 2: REGIME STRATIFICATION + WORST5Y  (canonical boundary only)
    # =========================================================================
    print("\n" + "=" * 120)
    print("TASK 2: REGIME STRATIFICATION + WORST5Y (canonical 2021-05-08 boundary)")
    print("=" * 120)

    regime_rows = []

    for cfg in CONFIGS:
        lbl = cfg["label"]
        r   = r_cache[lbl]
        nav_dt = nav_cache[lbl]

        # Regime-stratified CAGR
        reg, regime_min = _regime_cagr(r, regimes)

        # trend:bear CAGR (key metric)
        bear_cagr = reg.get("trend:bear", np.nan)

        # Worst5Y (after-tax; compute_worst5y expects NAV values array)
        w5y = _worst5y_at(nav_dt)

        # MaxDD (pretax)
        maxdd = _maxdd_from_returns(r)

        # Canonical min9 (for reference)
        can_brow = brow_idx.get(("2021-05-08", lbl), {})
        can_min9 = can_brow.get("min9_at", np.nan)
        can_is   = can_brow.get("CAGR_IS_at", np.nan)
        can_oos  = can_brow.get("CAGR_OOS_at", np.nan)

        row = {
            "config":          lbl,
            "min9_at_canon_pp":  can_min9,
            "CAGR_IS_canon_pp":  can_is,
            "CAGR_OOS_canon_pp": can_oos,
            "MaxDD_pretax_pp":   round(maxdd * 100, 2),
            "bear_CAGR_at_pp":   round(bear_cagr * 100, 2) if np.isfinite(bear_cagr) else np.nan,
            "bull_CAGR_at_pp":   round(reg.get("trend:bull", np.nan) * 100, 2),
            "highvol_CAGR_at_pp":round(reg.get("vol:highvol", np.nan) * 100, 2),
            "calm_CAGR_at_pp":   round(reg.get("vol:calm",    np.nan) * 100, 2),
            "rateup_CAGR_at_pp": round(reg.get("rate:rate_up", np.nan) * 100, 2),
            "ratedn_CAGR_at_pp": round(reg.get("rate:rate_down", np.nan) * 100, 2),
            "regime_min_at_pp":  round(regime_min * 100, 2) if np.isfinite(regime_min) else np.nan,
            "Worst5Y_at_pp":     w5y,
            "Worst5Y_negative":  int(w5y < 0) if np.isfinite(w5y) else -1,
        }
        regime_rows.append(row)

    # ---- Print regime table ----
    print("\n%s" % ("-" * 140))
    print("%-24s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s"
          % ("config", "min9%", "IS%", "OOS%", "MaxDD%",
             "bear%", "bull%", "highvol%", "RegMin%", "Worst5Y%"))
    print("-" * 140)
    for rr in regime_rows:
        def _fmt(v):
            return ("%+7.2f" % v) if (v is not None and np.isfinite(float(v))) else "   n/a "
        print("%-24s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s"
              % (rr["config"],
                 _fmt(rr["min9_at_canon_pp"]), _fmt(rr["CAGR_IS_canon_pp"]),
                 _fmt(rr["CAGR_OOS_canon_pp"]), _fmt(rr["MaxDD_pretax_pp"]),
                 _fmt(rr["bear_CAGR_at_pp"]), _fmt(rr["bull_CAGR_at_pp"]),
                 _fmt(rr["highvol_CAGR_at_pp"]), _fmt(rr["regime_min_at_pp"]),
                 _fmt(rr["Worst5Y_at_pp"])))

    print("\nWorst5Y negative (after-tax):")
    for rr in regime_rows:
        flag = "NEGATIVE" if rr["Worst5Y_negative"] == 1 else ("n/a" if rr["Worst5Y_negative"] < 0 else "positive")
        print("  %-24s  Worst5Y=%s%%  %s"
              % (rr["config"],
                 ("%+.2f" % rr["Worst5Y_at_pp"]) if np.isfinite(rr["Worst5Y_at_pp"]) else "n/a",
                 flag))

    # =========================================================================
    # WRITE CSV
    # =========================================================================
    print("\n" + "=" * 120)
    print("Writing CSV: %s" % OUT_CSV)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Section 1: boundary sensitivity
    s1_cols = ["config", "oos_start", "is_end", "canonical",
               "n_is", "n_oos", "CAGR_IS_at", "CAGR_OOS_at", "min9_at", "gap_pp"]
    # Section 2: regime + Worst5Y
    s2_cols = [
        "config", "min9_at_canon_pp", "CAGR_IS_canon_pp", "CAGR_OOS_canon_pp",
        "MaxDD_pretax_pp", "bear_CAGR_at_pp", "bull_CAGR_at_pp",
        "highvol_CAGR_at_pp", "calm_CAGR_at_pp",
        "rateup_CAGR_at_pp", "ratedn_CAGR_at_pp",
        "regime_min_at_pp", "Worst5Y_at_pp", "Worst5Y_negative",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        # Section 1
        f.write("# SECTION1: OOS_BOUNDARY_SENSITIVITY\n")
        writer = csv.DictWriter(f, fieldnames=s1_cols, extrasaction="ignore")
        writer.writeheader()
        for row in boundary_rows:
            writer.writerow(row)
        # Section 2
        f.write("\n# SECTION2: REGIME_WORST5Y (canonical boundary)\n")
        writer2 = csv.DictWriter(f, fieldnames=s2_cols, extrasaction="ignore")
        writer2.writeheader()
        for row in regime_rows:
            writer2.writerow(row)

    print("CSV written: %d boundary rows + %d regime rows." % (len(boundary_rows), len(regime_rows)))

    # =========================================================================
    # RETURN BLOCK
    # =========================================================================
    sanity_pass = ok_min9 and ok_maxdd

    # Summarize delta stability
    delta_summary = {}
    for lbl in config_labels:
        deltas = np.asarray([d for d in delta_store[lbl] if np.isfinite(d)])
        if len(deltas) > 0:
            delta_summary[lbl] = {
                "mean_pp":  round(float(deltas.mean()), 2),
                "std_pp":   round(float(deltas.std()),  2),
                "range_pp": round(float(deltas.max() - deltas.min()), 2),
                "min_pp":   round(float(deltas.min()), 2),
                "max_pp":   round(float(deltas.max()), 2),
            }
            ci2 = [i for i, (oos, _) in enumerate(OOS_BOUNDARIES) if oos == "2021-05-08"]
            if ci2:
                delta_summary[lbl]["canonical_2021_delta_pp"] = round(delta_store[lbl][ci2[0]], 2)

    regime_summary = {}
    for rr in regime_rows:
        regime_summary[rr["config"]] = {
            "bear_CAGR_at_pp":  rr["bear_CAGR_at_pp"],
            "regime_min_at_pp": rr["regime_min_at_pp"],
            "Worst5Y_at_pp":    rr["Worst5Y_at_pp"],
            "Worst5Y_negative": rr["Worst5Y_negative"],
        }

    # Ensure JSON serialisability (numpy bools -> int, floats -> float)
    def _to_json_safe(obj):
        if isinstance(obj, dict):
            return {k: _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json_safe(x) for x in obj]
        if isinstance(obj, bool):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return int(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if (v != v) else v  # NaN -> None
        return obj

    return_block = {
        "script": "oos_boundary_sensitivity_20260616.py",
        "run_date": "2026-06-16",
        "sanity_pass": int(sanity_pass),
        "sanity_min9_got_pct": round(float(san_min9) * 100, 4),
        "sanity_maxdd_got_pct": round(float(san_maxdd) * 100, 4),
        "delta_stability": _to_json_safe(delta_summary),
        "regime_worst5y": _to_json_safe(regime_summary),
        "output_csv": str(OUT_CSV),
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK:")
    print(json.dumps(return_block, indent=2, ensure_ascii=False))
    print("=" * 120)
    print("DONE.")

    return return_block


if __name__ == "__main__":
    main()
