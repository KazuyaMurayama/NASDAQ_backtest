"""
src/audit/p09c1_alloc_annual_20260620.py
========================================
Annual (calendar-year) AFTER-TAX returns for the 8 P09_C1 OUT-sleeve allocation
variations (A0-A7), plus a NASDAQ 1x buy&hold benchmark column. This is the
companion of `p09c1_alloc_stage1_20260620.py` (the Stage-1 gate runner): it
re-uses the *byte-identical* data-prep + variation-building logic but, instead of
running the full Stage-1 battery, it extracts calendar-year returns from each
variation's daily NAV, applies the repo after-tax multiplier (x0.8273), and
writes one row per calendar year to feed the report's annual table.

The 8 variations (OUT-sleeve allocation only; instruments + IN-leg leverage map
are IDENTICAL to canonical P09_C1):
  A0_P09_C1_BASE   legacy C1 fill (baseline; == canonical P09_C1)
  A1_INVVOL_W126   inverse-vol gold/bond weights, 126d window, weekly cadence
  A2_INVVOL_DAILY  inverse-vol weights, 63d window, daily cadence
  A3_BOND_HYST     bond gate with hysteresis (+/-5%)
  A4_RISK_BUDGET   vol-target the OUT sleeve to 10% annualized
  A5_CONVICTION    route out_strength*0.5 of OUT sleeve to SOFR cash  (LAGGED knob)
  A6_GOLD_TILT     raise Gold to >=0.75 on highvol days  (CAUSAL highvol mask)
  A7_IN_VOL_BRAKE  base OUT fill + IN-leg vol brake (target 30%)

CAUSALITY (must match Stage-1 at HEAD, I-1 / I-2 fixes):
  * A5's out_strength uses pd.Series(lev_mod_raw).shift(_LEVMOD_DELAY=2) before
    deriving the cash-sizing knob (same delay the canonical NAV builders apply to
    lev_mod_065). Reading the same-day value would be a ~2-day look-ahead.
  * A6's high-vol ACTION signal uses a CAUSAL point-in-time expanding median:
    rv = a["ret"].rolling(63).std(ddof=1)*sqrt(252), threshold =
    rv.expanding(min_periods=252).median().shift(1). The ex-post full-sample
    median from the regime labeler must NOT drive an allocation.

NASDAQ 1x B&H benchmark:
  Built from a["ret"] (= close.pct_change().fillna(0), the 1x NASDAQ daily return)
  as nav = np.cumprod(1 + a["ret"]); calendar-year returns x AFTER_TAX. NOTE the
  existing `audit_results/annual_returns_sc180_200_20260620.csv` NASDAQ_1x_BH_pct
  column is PRE-TAX (no x0.8273), so this AFTER-TAX column will differ from it by
  exactly the 0.8273 factor each year. The cross-check below therefore compares
  the UNDERLYING raw (pre-tax) series and asserts a <=0.5pp/yr match.

ASCII-only prints (Windows cp932). Does NOT commit here. No temp files.
Output: audit_results/p09c1_alloc_annual_20260620.csv  (utf-8-sig / BOM)
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
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START

# C1 cost-model builders
from src.audit.k365_recost_20260612 import _build_full_c1, _build_tqqq_base_param

# Cost/NAV helpers + calendar-year + after-tax constant
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _count_fund_transitions,
    _calendar_year_returns,
    LAG_DAYS, TRADING_DAYS,
)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY

# OUT-fill variant machinery (Task 8) -- same factories Stage-1 uses
from src.audit.out_fill_variants_20260620 import (
    _build_out_fill_variant, alloc_base,
    inverse_vol_weights_cadence, bond_gate_hysteresis,
    make_alloc_vol_target, make_alloc_conviction_cash, make_alloc_gold_tilt,
    apply_in_leg_vol_brake,
)

# ---------------------------------------------------------------------------
# Constants  (copied byte-for-byte from p09c1_alloc_stage1_20260620.py)
# ---------------------------------------------------------------------------
# Canonical P09_C1 wiring: default V7_MAP (v7_map=None), lev_scale=1.0, and NO
# >3x excess charge. excess_extra=0.0 reproduces cfd_excess=False exactly.
P09C1_V7_MAP       = None   # -> default V7_MAP {0:1.20,1:1.10,2:1.00,3:1.00}
P09C1_LEV_SCALE    = 1.0
P09C1_EXCESS_EXTRA = 0.0    # NO excess penalty (== cfd_excess=False)

# repo empirical effective after-tax multiplier (NOT 1-0.20315)
AFTER_TAX = 0.8273

# --- Causality constants for action signals (I-1, I-2 fixes) -----------------
_LEVMOD_DELAY     = 2        # == strategy_runners._DELAY applied to lev_mod_065
_EXIT_THR         = 0.3      # DH-W1 W1 exit threshold (EXIT_THR_W1)
_A6_VOL_WIN       = 63       # = regime_labeler VOL_WIN (match labeler rv exactly)
_A6_EXPMED_MINPER = 252      # min periods before the expanding median is defined

# A0 sanity (vs direct _build_full_c1 P09_C1)
SANITY_TOL_TIGHT = 1e-6

# canonical P09_C1 OOS after-tax reference (book value; rounding tolerance)
P09C1_BOOK_CAGR_OOS_AT = 0.177672


def main():
    print("=" * 120)
    print("P09_C1 ALLOCATION-VARIATION ANNUAL (CALENDAR-YEAR) RETURNS  2026-06-20")
    print("8 OUT-sleeve variations A0-A7 + NASDAQ 1x B&H, AFTER-TAX x%.4f" % AFTER_TAX)
    print("=" * 120)

    # ---- Load shared data (identical to Stage-1) ----
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)

    # ---- Gold/Bond auxiliary series (same as Stage-1) ----
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

    # out_strength (I-1 FIX): lag lev_mod_065 by _LEVMOD_DELAY (==2) BEFORE
    # deriving the cash-sizing knob. _EXIT_THR (0.3) is the DH-W1 W1 exit thr.
    lev_mod_raw = np.nan_to_num(np.asarray(a["lev_mod_065"], float), nan=0.0)
    lev_mod = pd.Series(lev_mod_raw).shift(_LEVMOD_DELAY).fillna(0.0).values
    out_strength = np.clip((_EXIT_THR - lev_mod) / _EXIT_THR, 0.0, 1.0)

    # ---- IN-leg base return for the OUT-fill variant builder ----
    print("\nBuilding P09_C1 IN-leg base (V7_MAP default, scale=1.0, excess=0.0) ...")
    _, r_base_in, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)

    def build_variant(wg_, wb_, bond_on_, alloc_fn, extra_ctx=None, in_brake=False):
        nav_arr, r, eff, w_gold, w_bond, w_cash = _build_out_fill_variant(
            r_base_in, ret_gold, ret_bond, fund_active, wg_, wb_, bond_on_, sofr_arr,
            alloc_fn=alloc_fn, return_weights=True, **(extra_ctx or {}))
        if in_brake:
            r = apply_in_leg_vol_brake(r, fund_active, sofr_arr,
                                       target_vol=0.30, window=63)
            nav_arr = np.cumprod(1.0 + r)
        nav_dt = pd.Series(nav_arr, index=dates_dt)
        tpy = tpy_base + _count_fund_transitions(eff) / n_years
        return nav_dt, r, tpy

    # =========================================================================
    # SANITY GATE: A0 must reproduce canonical P09_C1 (direct _build_full_c1)
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: A0_P09_C1_BASE vs direct _build_full_c1 P09_C1 (tol 1e-6 on CAGR_OOS)")
    print("=" * 120)

    canon_nav, canon_r, canon_tpy, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    canon_aft = _apply_aftertax(compute_10metrics(canon_nav, canon_tpy))
    canon_cagr_oos = canon_aft["CAGR_OOS"]

    a0_nav, a0_r, a0_tpy = build_variant(wg, wb, bond_on, alloc_base)
    a0_aft = _apply_aftertax(compute_10metrics(a0_nav, a0_tpy))
    a0_cagr_oos = a0_aft["CAGR_OOS"]

    diff_a0_direct = abs(a0_cagr_oos - canon_cagr_oos)
    print("  A0 CAGR_OOS(at)            : %+.8f%%" % (a0_cagr_oos * 100))
    print("  direct _build_full_c1 OOS : %+.8f%%" % (canon_cagr_oos * 100))
    print("  CURRENT_BEST P09_C1 OOS   : %+.8f%%" % (P09C1_BOOK_CAGR_OOS_AT * 100))
    print("  |A0 - direct|             : %.3e   (tol %.0e)"
          % (diff_a0_direct, SANITY_TOL_TIGHT))
    if diff_a0_direct > SANITY_TOL_TIGHT:
        print("\nSANITY FAILED -- A0 does not match direct _build_full_c1 P09_C1 to 1e-6.")
        print("The OUT-fill wiring is wrong. Halting.")
        sys.exit(1)
    print("  SANITY PASSED (A0 == direct P09_C1 to 1e-6). Proceeding.\n")

    # =========================================================================
    # CAUSAL high-vol ACTION signal for A6 (I-2 fix; point-in-time)
    # =========================================================================
    rv_a6 = (pd.Series(np.asarray(a["ret"], float))
             .rolling(_A6_VOL_WIN, min_periods=_A6_VOL_WIN).std(ddof=1)
             * np.sqrt(TRADING_DAYS))
    causal_med = rv_a6.expanding(min_periods=_A6_EXPMED_MINPER).median().shift(1)
    highvol_mask_causal = (rv_a6 > causal_med).fillna(False).values
    print("  highvol days (causal, A6 action signal): %d of %d (%.1f%%)"
          % (int(highvol_mask_causal.sum()), n,
             100.0 * highvol_mask_causal.sum() / n))

    # =========================================================================
    # Alternate weights / gates for variations (identical to Stage-1)
    # =========================================================================
    wg63, wb63 = wg, wb
    wg126, wb126       = inverse_vol_weights_cadence(ret_gold, ret_bond, 126, 5)
    wg_daily, wb_daily = inverse_vol_weights_cadence(ret_gold, ret_bond, 63, 1)
    bond_on_hyst = bond_gate_hysteresis(bond_m252_raw)

    VARIATIONS = [
        ("A0_P09_C1_BASE",  lambda: build_variant(wg63, wb63, bond_on, alloc_base)),
        ("A1_INVVOL_W126",  lambda: build_variant(wg126, wb126, bond_on, alloc_base)),
        ("A2_INVVOL_DAILY", lambda: build_variant(wg_daily, wb_daily, bond_on, alloc_base)),
        ("A3_BOND_HYST",    lambda: build_variant(wg63, wb63, bond_on_hyst, alloc_base)),
        ("A4_RISK_BUDGET",  lambda: build_variant(wg63, wb63, bond_on,
                                                  make_alloc_vol_target(0.10, 63))),
        ("A5_CONVICTION",   lambda: build_variant(wg63, wb63, bond_on,
                                                  make_alloc_conviction_cash(0.5),
                                                  extra_ctx={"out_strength": out_strength})),
        ("A6_GOLD_TILT",    lambda: build_variant(wg63, wb63, bond_on,
                                                  make_alloc_gold_tilt(0.75),
                                                  extra_ctx={"highvol_mask": highvol_mask_causal})),
        ("A7_IN_VOL_BRAKE", lambda: build_variant(wg63, wb63, bond_on, alloc_base,
                                                  in_brake=True)),
    ]

    # =========================================================================
    # Calendar-year AFTER-TAX returns for each variation
    # =========================================================================
    print("\n" + "=" * 120)
    print("Building calendar-year AFTER-TAX returns for 8 variations ...")
    print("=" * 120)

    # {label: {year: pct_at}}
    var_annual = {}
    for label, builder in VARIATIONS:
        print("  [%s] building NAV ..." % label)
        nav_dt, _r, _tpy = builder()
        cy = _calendar_year_returns(nav_dt)          # pd.Series indexed by year
        var_annual[label] = {int(y): float(frac) * AFTER_TAX * 100.0
                             for y, frac in cy.items()}

    # =========================================================================
    # NASDAQ 1x B&H benchmark (after-tax)
    # =========================================================================
    # a["ret"] == close.pct_change().fillna(0) (the 1x NASDAQ daily return,
    # confirmed in g14_wfa_sbi_cfd.load_shared_assets line 303). Build the B&H
    # NAV and take calendar-year returns, then x AFTER_TAX.
    print("\nBuilding NASDAQ 1x B&H benchmark from a['ret'] ...")
    ret_1x = np.asarray(a["ret"], float)
    bh_nav = pd.Series(np.cumprod(1.0 + ret_1x), index=dates_dt)
    cy_bh = _calendar_year_returns(bh_nav)
    bh_annual_raw = {int(y): float(frac) for y, frac in cy_bh.items()}        # pre-tax frac
    bh_annual_at  = {y: f * AFTER_TAX * 100.0 for y, f in bh_annual_raw.items()}  # after-tax pct

    # =========================================================================
    # CROSS-CHECK vs existing (PRE-TAX) NASDAQ_1x_BH_pct column
    # =========================================================================
    # The existing annual_returns_sc180_200_20260620.csv NASDAQ_1x_BH_pct column
    # is PRE-TAX (no x0.8273 -- it comes straight from _load_nasdaq_bh's raw
    # pct_change). So we cross-check the UNDERLYING raw (pre-tax) series, NOT the
    # after-tax column. They must match within 0.5pp/yr (same close, same year
    # boundaries).
    existing_csv = os.path.join(_REPO_DIR, "audit_results",
                                "annual_returns_sc180_200_20260620.csv")
    xcheck_ok = True
    xcheck_note = "existing CSV not found -- skipped"
    sample_rows = {}
    if os.path.exists(existing_csv):
        df_ex = pd.read_csv(existing_csv, encoding="utf-8-sig")
        ex_bh = {int(r["year"]): float(r["NASDAQ_1x_BH_pct"]) for _, r in df_ex.iterrows()}
        max_abs = 0.0
        worst_year = None
        for y, ex_pct in ex_bh.items():
            if y in bh_annual_raw:
                mine_raw_pct = bh_annual_raw[y] * 100.0   # pre-tax pct, comparable to existing
                d = abs(mine_raw_pct - ex_pct)
                if d > max_abs:
                    max_abs, worst_year = d, y
        xcheck_ok = max_abs <= 0.5
        xcheck_note = ("max |mine_raw - existing| = %.3fpp @ %s (tol 0.5pp/yr)"
                       % (max_abs, worst_year))
        for sy in (2000, 2008, 2020):
            sample_rows[sy] = {
                "mine_raw_pct": round(bh_annual_raw.get(sy, float("nan")) * 100.0, 2),
                "existing_pretax_pct": round(ex_bh.get(sy, float("nan")), 2),
                "mine_aftertax_pct": round(bh_annual_at.get(sy, float("nan")), 2),
            }

    # =========================================================================
    # Assemble the wide CSV: year x 8 variations + NASDAQ_1x_BH_pct
    # =========================================================================
    var_labels = [lbl for lbl, _ in VARIATIONS]
    all_years = set(bh_annual_at.keys())
    for lbl in var_labels:
        all_years.update(var_annual[lbl].keys())
    years_sorted = sorted(all_years)

    rows = []
    for y in years_sorted:
        row = {"year": y}
        for lbl in var_labels:
            v = var_annual[lbl].get(y, float("nan"))
            row[lbl] = round(v, 2) if v == v else float("nan")
        bv = bh_annual_at.get(y, float("nan"))
        row["NASDAQ_1x_BH_pct"] = round(bv, 2) if bv == bv else float("nan")
        rows.append(row)

    df_out = pd.DataFrame(rows)
    col_order = ["year"] + var_labels + ["NASDAQ_1x_BH_pct"]
    df_out = df_out[col_order]

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "p09c1_alloc_annual_20260620.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.2f", encoding="utf-8-sig")

    # =========================================================================
    # PRINT TABLE
    # =========================================================================
    print("\n" + "=" * 150)
    print("ANNUAL AFTER-TAX RETURNS (%)  --  8 P09_C1 allocation variations + NASDAQ 1x B&H")
    print("=" * 150)
    hdr = "%-6s" % "year"
    for lbl in var_labels:
        hdr += " | %10s" % lbl[:10]
    hdr += " | %10s" % "NDX_1x_BH"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df_out.iterrows():
        line = "%-6d" % int(r["year"])
        for lbl in var_labels:
            v = r[lbl]
            line += " | %+9.2f%%" % v if v == v else " | %10s" % "N/A"
        bv = r["NASDAQ_1x_BH_pct"]
        line += " | %+9.2f%%" % bv if bv == bv else " | %10s" % "N/A"
        print(line)

    # =========================================================================
    # SPOT-CHECKS  (print + assert)
    # =========================================================================
    print("\n" + "=" * 120)
    print("SPOT-CHECKS")
    print("=" * 120)
    a0_2008 = var_annual["A0_P09_C1_BASE"].get(2008, float("nan"))
    bh_2008 = bh_annual_at.get(2008, float("nan"))
    print("  A0_P09_C1_BASE[2008] after-tax : %+.2f%%   (expect POSITIVE, OUT-defense ~ +20%%)"
          % a0_2008)
    print("  NASDAQ_1x_BH[2008]  after-tax  : %+.2f%%   (expect deeply NEGATIVE ~ -33%% to -41%%)"
          % bh_2008)
    print("\n  Cross-check vs existing PRE-TAX NASDAQ_1x_BH_pct:")
    print("    %s" % xcheck_note)
    for sy in (2000, 2008, 2020):
        if sy in sample_rows:
            sr_ = sample_rows[sy]
            print("    %d: mine_raw=%+.2f%%  existing_pretax=%+.2f%%  (mine_aftertax=%+.2f%%)"
                  % (sy, sr_["mine_raw_pct"], sr_["existing_pretax_pct"],
                     sr_["mine_aftertax_pct"]))

    print("\n  Asserting spot-checks ...")
    assert a0_2008 == a0_2008 and a0_2008 > 0.0, \
        "A0_P09_C1_BASE[2008] after-tax must be POSITIVE, got %.4f" % a0_2008
    assert bh_2008 == bh_2008 and bh_2008 < 0.0, \
        "NASDAQ_1x_BH[2008] after-tax must be NEGATIVE, got %.4f" % bh_2008
    assert -41.0 <= bh_2008 <= -30.0, \
        "NASDAQ_1x_BH[2008] after-tax out of expected [-41,-30] band: %.4f" % bh_2008
    if os.path.exists(existing_csv):
        assert xcheck_ok, \
            "B&H raw series mismatch vs existing pre-tax column: %s" % xcheck_note
    print("  All spot-checks PASSED.")

    print("\nSaved CSV: %s  (%d year-rows, %d columns)"
          % (csv_path, len(df_out), df_out.shape[1]))
    print("Columns: %s" % ", ".join(col_order))
    print("\nDone.")

    return {
        "csv": csv_path,
        "n_year_rows": len(df_out),
        "columns": col_order,
        "year_range": [int(years_sorted[0]), int(years_sorted[-1])],
        "A0_2008_at_pct": round(a0_2008, 4),
        "BH_2008_at_pct": round(bh_2008, 4),
        "xcheck_note": xcheck_note,
        "xcheck_ok": bool(xcheck_ok),
        "sample_years": sample_rows,
        "after_tax": AFTER_TAX,
    }


if __name__ == "__main__":
    main()
