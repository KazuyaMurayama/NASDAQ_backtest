"""
src/audit/a7dd_annual_20260621.py
==================================
A7/DD-reduction ANNUAL (calendar-year) AFTER-TAX returns: A0/A7/B1/B2/B3/B4
overlays on P09_C1 + NASDAQ 1x B&H benchmark column.

Builds the same 6 brake variants as a7dd_stage1_20260621.py (byte-identical
brake params) but instead of running the full Stage-1 gate battery, it:
  1. Extracts CALENDAR-YEAR returns from each strategy's daily NAV.
  2. Applies the repo empirical after-tax multiplier (x0.8273).
  3. Writes audit_results/a7dd_annual_20260621.csv (year x 6 strats + NASDAQ BH).

This feeds the report's annual table and tests the hypothesis that B1
(downside-dev brake) avoids the 1999 upside-vol mis-brake that A7 suffered
(expected: B1 1999 > A7 1999).

BRAKE PARAMS (identical to Stage-1):
  A0: r_strat (unbraked P09_C1 base)
  A7: apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.5)
  B1: apply_downside_dev_brake(r_strat, fund_active, sofr_arr, 0.20, 63, 0.5)
  B2: apply_dd_throttle(r_strat, fund_active, sofr_arr, tiers=((0.15,0.25),(0.25,0.50)))
  B3: apply_asym_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.5, 5)
  B4: apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.25, 63, 0.5)  [B4_VOL025_CAP50]

NASDAQ 1x B&H: built from a["ret"] (same as p09c1_alloc_annual_20260620.py).
Cross-checked vs audit_results/p09c1_alloc_annual_20260620.csv NASDAQ_1x_BH_pct
for years 2000/2008/2020 (must match within 0.1pp -- both are after-tax).

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
Output: audit_results/a7dd_annual_20260621.csv  (utf-8-sig / BOM)
"""

from __future__ import annotations

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
from src.audit.unified_metrics import IS_END, OOS_START

# C1 cost-model builders
from src.audit.k365_recost_20260612 import (
    _build_full_c1, _build_tqqq_base_param,
    EXCESS_EXTRA_K365_CENTRE,
)

# Cost/NAV helpers + calendar-year + after-tax constant
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _count_fund_transitions,
    _calendar_year_returns,
    LAG_DAYS, TRADING_DAYS,
)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY

# OUT-fill variant machinery
from src.audit.out_fill_variants_20260620 import (
    _build_out_fill_variant, alloc_base,
)

# DD-reduction overlays (identical params to Stage-1)
from src.audit.dd_reduction_overlays_20260621 import (
    apply_downside_dev_brake, apply_dd_throttle, apply_asym_vol_brake,
    apply_param_vol_brake,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Canonical P09_C1 wiring: default V7_MAP (v7_map=None), lev_scale=1.0, and
# NO >3x excess charge. excess_extra=0.0 reproduces cfd_excess=False exactly.
P09C1_V7_MAP       = None
P09C1_LEV_SCALE    = 1.0
P09C1_EXCESS_EXTRA = 0.0

# repo empirical effective after-tax multiplier (NOT 1-0.20315)
AFTER_TAX = 0.8273  # repo empirical effective after-tax multiplier

# A0 sanity vs direct _build_full_c1
SANITY_TOL_TIGHT = 1e-6

# Prior-project A0 2008 reference (from p09c1_alloc_annual_20260620.csv)
# A0_P09_C1_BASE 2008 = 20.34%  (after-tax)
PRIOR_A0_2008_AT_PCT = 20.34

# Cross-check reference from p09c1_alloc_annual_20260620.csv NASDAQ_1x_BH_pct
# (already after-tax in that file: check heading comment confirms x0.8273)
XCHECK_BH = {2000: -32.50, 2008: -33.54, 2020: 36.10}
XCHECK_TOL_PP = 0.1


def main():
    print("=" * 120)
    print("A7/DD-REDUCTION ANNUAL AFTER-TAX RETURNS  2026-06-21")
    print("6 brakes: A0 / A7 / B1(downside-dev) / B2(dd-throttle) / B3(asym) / B4(vol025_cap50)")
    print("+ NASDAQ 1x B&H benchmark column")
    print("AFTER_TAX multiplier = %.4f" % AFTER_TAX)
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
    # P09_C1 IN-leg base return (verbatim from Stage-1)
    # =========================================================================
    print("\nBuilding P09_C1 IN-leg base (V7_MAP default, scale=1.0, excess=0.0) ...")
    _, r_base_in, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)

    # Build base P09_C1 daily return (== A0 unbraked)
    nav0_arr, r_strat, eff0 = _build_out_fill_variant(
        r_base_in, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    tpy0 = tpy_base + _count_fund_transitions(eff0) / n_years

    # =========================================================================
    # SANITY GATE: A0 must reproduce canonical P09_C1 (direct _build_full_c1)
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: A0 (base r_strat) vs direct _build_full_c1 P09_C1 (tol 1e-6 on CAGR_OOS)")
    print("=" * 120)

    from src.audit.unified_metrics import compute_10metrics

    canon_nav, canon_r, canon_tpy, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    canon_aft = _apply_aftertax(compute_10metrics(canon_nav, canon_tpy))
    canon_cagr_oos = canon_aft["CAGR_OOS"]

    a0_nav_dt = pd.Series(np.cumprod(1.0 + r_strat), index=dates_dt)
    a0_aft = _apply_aftertax(compute_10metrics(a0_nav_dt, tpy0))
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
    # Build braked returns (identical params to Stage-1)
    # =========================================================================
    # A0: r_strat (unbraked)
    # A7: apply_param_vol_brake(target=0.30, window=63, cap=0.5)
    # B1: apply_downside_dev_brake(target_dvol=0.20, window=63, cap=0.5)
    # B2: apply_dd_throttle(tiers=((0.15,0.25),(0.25,0.50)))
    # B3: apply_asym_vol_brake(target=0.30, window=63, cap=0.5, release=5)
    # B4: apply_param_vol_brake(target=0.25, window=63, cap=0.5)  [B4_VOL025_CAP50]

    STRATEGIES = [
        ("A0_P09_C1_BASE",  r_strat),
        ("A7_REPRODUCE",    apply_param_vol_brake(r_strat, fund_active, sofr_arr,
                                                  0.30, 63, 0.5)),
        ("B1_DOWNSIDE_DEV", apply_downside_dev_brake(r_strat, fund_active, sofr_arr,
                                                     0.20, 63, 0.5)),
        ("B2_DD_THROTTLE",  apply_dd_throttle(r_strat, fund_active, sofr_arr,
                                              tiers=((0.15, 0.25), (0.25, 0.50)))),
        ("B3_ASYM_BRAKE",   apply_asym_vol_brake(r_strat, fund_active, sofr_arr,
                                                  0.30, 63, 0.5, 5)),
        ("B4_VOL025_CAP50", apply_param_vol_brake(r_strat, fund_active, sofr_arr,
                                                   0.25, 63, 0.5)),
    ]

    # =========================================================================
    # Calendar-year AFTER-TAX returns for each strategy
    # =========================================================================
    print("=" * 120)
    print("Building calendar-year after-tax returns for 6 brake strategies ...")
    print("=" * 120)

    strat_annual = {}
    for label, r_braked in STRATEGIES:
        print("  [%s] ..." % label)
        nav_dt = pd.Series(np.cumprod(1.0 + r_braked), index=dates_dt)
        cy = _calendar_year_returns(nav_dt)            # pd.Series indexed by year (fraction)
        strat_annual[label] = {int(y): float(frac) * AFTER_TAX * 100.0
                               for y, frac in cy.items()}

    # =========================================================================
    # NASDAQ 1x B&H benchmark (after-tax)
    # =========================================================================
    print("\nBuilding NASDAQ 1x B&H benchmark from a['ret'] ...")
    ret_1x = np.asarray(a["ret"], float)
    bh_nav = pd.Series(np.cumprod(1.0 + ret_1x), index=dates_dt)
    cy_bh = _calendar_year_returns(bh_nav)
    bh_annual_at = {int(y): float(frac) * AFTER_TAX * 100.0
                    for y, frac in cy_bh.items()}

    # =========================================================================
    # CROSS-CHECK: NASDAQ_1x_BH vs p09c1_alloc_annual_20260620.csv (within 0.1pp)
    # =========================================================================
    print("\n" + "=" * 120)
    print("CROSS-CHECK: NASDAQ_1x_BH_pct vs p09c1_alloc_annual_20260620.csv (tol %.1fpp)" % XCHECK_TOL_PP)
    print("(p09c1_alloc_annual_20260620.csv NASDAQ_1x_BH_pct is AFTER-TAX -- should match within 0.1pp)")
    print("=" * 120)
    xcheck_ok = True
    for yr, ref_pct in sorted(XCHECK_BH.items()):
        our_pct = bh_annual_at.get(yr, float("nan"))
        diff_pp = abs(our_pct - ref_pct)
        status = "OK" if diff_pp <= XCHECK_TOL_PP else "FAIL"
        if status == "FAIL":
            xcheck_ok = False
        print("  %d: ours=%+.2f%%  ref=%+.2f%%  |diff|=%.2fpp  [%s]"
              % (yr, our_pct, ref_pct, diff_pp, status))
    if not xcheck_ok:
        print("\n  WARN: NASDAQ_1x_BH cross-check FAIL on one or more years.")
        print("  Possible cause: a['ret'] mismatch or different date range.")
    else:
        print("  CROSS-CHECK PASSED (all 3 years within %.1fpp)." % XCHECK_TOL_PP)

    # =========================================================================
    # BUILD OUTPUT TABLE
    # =========================================================================
    all_years = sorted(set().union(*[set(v.keys()) for v in strat_annual.values()],
                                   set(bh_annual_at.keys())))

    rows = []
    for yr in all_years:
        row = {"year": yr}
        for label, _ in STRATEGIES:
            row[label] = strat_annual[label].get(yr, float("nan"))
        row["NASDAQ_1x_BH_pct"] = bh_annual_at.get(yr, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    col_order = ["year"] + [label for label, _ in STRATEGIES] + ["NASDAQ_1x_BH_pct"]
    df = df[col_order]

    # =========================================================================
    # KEY HYPOTHESIS SPOT-CHECKS
    # =========================================================================
    print("\n" + "=" * 120)
    print("KEY HYPOTHESIS SPOT-CHECKS")
    print("=" * 120)

    def _get(label_or_col, year):
        if label_or_col == "NASDAQ_1x_BH_pct":
            return bh_annual_at.get(year, float("nan"))
        return strat_annual[label_or_col].get(year, float("nan"))

    # 1. A0 2008 after-tax: expect ~+20.3% (OUT defense), must be > 0
    a0_2008 = _get("A0_P09_C1_BASE", 2008)
    print("\n1. A0 2008 after-tax = %+.2f%%  (expect ~+%.1f%%, must be > 0)"
          % (a0_2008, PRIOR_A0_2008_AT_PCT))
    assert a0_2008 > 0, "FAIL: A0 2008 should be positive (OUT defense, market was -38%)"
    print("   -> PASS (A0 2008 > 0)")

    # 2. NASDAQ B&H 2008: expect ~-33.5%, must be < -25
    bh_2008 = _get("NASDAQ_1x_BH_pct", 2008)
    print("\n2. NASDAQ B&H 2008 after-tax = %+.2f%%  (expect ~-33.5%%, must be < -25%%)"
          % bh_2008)
    assert bh_2008 < -25, "FAIL: B&H 2008 should be deeply negative (NASDAQ crashed ~-40%%)"
    print("   -> PASS (B&H 2008 < -25)")

    # 3. CORE HYPOTHESIS: B1 1999 > A7 1999  (downside-dev avoids upside-vol mis-brake)
    a0_1999 = _get("A0_P09_C1_BASE", 1999)
    a7_1999 = _get("A7_REPRODUCE",   1999)
    b1_1999 = _get("B1_DOWNSIDE_DEV", 1999)
    a0_2000 = _get("A0_P09_C1_BASE", 2000)
    a7_2000 = _get("A7_REPRODUCE",   2000)
    b1_2000 = _get("B1_DOWNSIDE_DEV", 2000)
    print("\n3. CORE HYPOTHESIS: B1 1999 > A7 1999?")
    print("   (downside-dev brake should NOT penalize the 1999 up-market,")
    print("    unlike A7 which braked on total vol including the upside)")
    print("")
    print("   Year 1999:")
    print("     A0_P09_C1_BASE : %+.2f%%" % a0_1999)
    print("     A7_REPRODUCE   : %+.2f%%" % a7_1999)
    print("     B1_DOWNSIDE_DEV: %+.2f%%" % b1_1999)
    hypothesis_pass = b1_1999 > a7_1999
    print("   B1 1999 > A7 1999? -> %s (B1=%+.2f%%, A7=%+.2f%%, gap=%+.2fpp)"
          % ("YES (hypothesis CONFIRMED)" if hypothesis_pass else "NO (hypothesis NOT confirmed)",
             b1_1999, a7_1999, b1_1999 - a7_1999))
    print("")
    print("   Year 2000 (follow-through: brakes should matter most in the crash):")
    print("     A0_P09_C1_BASE : %+.2f%%" % a0_2000)
    print("     A7_REPRODUCE   : %+.2f%%" % a7_2000)
    print("     B1_DOWNSIDE_DEV: %+.2f%%" % b1_2000)

    # 4. NASDAQ_1x_BH cross-check already printed above
    print("\n4. NASDAQ_1x_BH cross-check 2000/2008/2020 (see above).")

    # =========================================================================
    # PRINT FULL ANNUAL TABLE
    # =========================================================================
    print("\n" + "=" * 120)
    print("ANNUAL AFTER-TAX RETURN TABLE  (pct, after-tax x%.4f)" % AFTER_TAX)
    print("=" * 120)
    col_names = [label for label, _ in STRATEGIES] + ["NASDAQ_1x_BH_pct"]
    hdr = "%-6s" % "year"
    for c in col_names:
        hdr += "  %+14s" % c[:14]
    print(hdr)
    print("-" * (6 + 16 * len(col_names)))
    for _, row in df.iterrows():
        line = "%6d" % int(row["year"])
        for c in col_names:
            v = row[c]
            if np.isfinite(v):
                line += "  %+14.2f" % v
            else:
                line += "  %14s" % "nan"
        print(line)

    # =========================================================================
    # WRITE CSV
    # =========================================================================
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "a7dd_annual_20260621.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\nSaved CSV: %s  (%d rows, %d columns)" % (csv_path, len(df), len(df.columns)))
    print("Columns: %s" % list(df.columns))

    print("\nDone.")
    return df


if __name__ == "__main__":
    main()
