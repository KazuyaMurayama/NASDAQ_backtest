"""
src/audit/p09c1_alloc_stage1_20260620.py
========================================
Stage-1 full gate over 8 P09_C1 OUT-sleeve allocation variations (A0-A7).

The variations change ONLY the OUT-sleeve allocation (Gold/Bond/Cash split on
OUT days) -- and A7 additionally applies an IN-leg cash brake. Instruments and
the IN-leg leverage map are IDENTICAL to canonical P09_C1:

  P09_C1 = V7-TQQQ base (default mom63 V7_MAP {0:1.20,1:1.10,2:1.00,3:1.00},
           lev_scale=1.0, NO >3x excess charge) + P09 OUT fill + C1 SOFR cash
           on bond-OFF OUT days.

NOTE on excess_extra:
  Canonical P09_C1 (leverup_b1c1_20260612.py #3) is built with
  _build_tqqq_base(..., cfd_excess=False), i.e. NO excess penalty on the >3x
  NASDAQ sleeve. _build_tqqq_base_param / _build_full_c1 always apply the
  excess penalty, scaled by `excess_extra`. To reproduce canonical P09_C1
  byte-for-byte we therefore pass excess_extra=0.0 (penalty = w * excess * 0 = 0,
  identical to the cfd_excess=False code path). This is the authoritative
  P09_C1 wiring and is asserted to 1e-6 against a direct _build_full_c1 build.

Each variation runs the standard 10-metric battery + Stage-1 full gate:
  WFA (canonical 49 windows) + CPCV + regime-stratified CAGR + stress windows
  + multi-metric block bootstrap vs V7_TQQQ and vs B3a_k365.

Hard veto (Stage-1), all 4 axes:
  MaxDD < -50%            -> VETO
  WFE   > 1.5             -> VETO
  Worst10Y* < 0           -> VETO
  Regime_min(bear) < -10% -> VETO

Variations:
  A0_P09_C1_BASE   legacy C1 fill (baseline; must == canonical P09_C1)
  A1_INVVOL_W126   inverse-vol gold/bond weights, 126d window, weekly cadence
  A2_INVVOL_DAILY  inverse-vol weights, 63d window, daily cadence
  A3_BOND_HYST     bond gate with hysteresis (+/-5%) instead of >0 binary
  A4_RISK_BUDGET   vol-target the OUT sleeve to 10% annualized, rest -> SOFR cash
  A5_CONVICTION    route out_strength*0.5 of OUT sleeve to SOFR cash
  A6_GOLD_TILT     raise Gold to >=0.75 on highvol days (from Bond then Cash)
  A7_IN_VOL_BRAKE  base OUT fill + IN-leg vol brake (target 30%, blend to cash)

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
Output: audit_results/p09c1_alloc_stage1_20260620.csv
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

# C1 cost-model builders
from src.audit.k365_recost_20260612 import (
    _build_full_c1, _build_tqqq_base_param,
    EXCESS_EXTRA_K365_CENTRE,
)

# Full Stage-1 gate
from src.audit.extended_eval_20260611 import _eval_one

# Multi-metric bootstrap
from src.audit.multimetric_bootstrap_20260615 import _block_bootstrap_multimetric

# Cost/NAV helpers + WFA/bootstrap constants
from src.audit.run_p09_tqqq_validate_20260611 import _run_wfa, N_BOOT, BLOCK, SEED
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _count_fund_transitions,
    LAG_DAYS, TRADING_DAYS,
)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base

# OUT-fill variant machinery (Task 8)
from src.audit.out_fill_variants_20260620 import (
    _build_out_fill_variant, alloc_base,
    inverse_vol_weights_cadence, bond_gate_hysteresis,
    make_alloc_vol_target, make_alloc_conviction_cash, make_alloc_gold_tilt,
    apply_in_leg_vol_brake,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Canonical P09_C1 wiring: default V7_MAP (v7_map=None), lev_scale=1.0, and
# NO >3x excess charge. excess_extra=0.0 reproduces cfd_excess=False exactly.
P09C1_V7_MAP      = None   # -> _build_v7_mult default V7_MAP {0:1.20,1:1.10,2:1.00,3:1.00}
P09C1_LEV_SCALE   = 1.0
P09C1_EXCESS_EXTRA = 0.0   # NO excess penalty (== cfd_excess=False)

# B3a_k365 reference (default map, scale 1.15, k365 centre excess) for bootstrap
B3A_MAP_DEFAULT    = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE      = 1.15
B3A_EXCESS_EXTRA   = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# Hard veto thresholds
HARD_VETO_MAXDD  = -0.50
HARD_VETO_WFE    = 1.5
HARD_VETO_W10Y   = 0.0
HARD_VETO_REGIME = -0.10

# --- Causality constants for action signals (I-1, I-2 fixes) -----------------
# Every canonical NAV builder lags lev_mod_065 by _DELAY business days before
# using it (src/audit/strategy_runners.py: _DELAY = 2, applied via .shift(_DELAY)
# in _build_nav_*_realistic). A5's out_strength is derived from lev_mod_065 and
# MUST be lagged by the SAME delay, else the cash-sizing knob look-aheads ~2 days.
_LEVMOD_DELAY = 2          # == strategy_runners._DELAY applied to lev_mod_065
_EXIT_THR     = 0.3        # DH-W1 W1 exit threshold (EXIT_THR_W1); out_strength
                           # is 1.0 at deepest OUT (lev_mod=0) and 0.0 at exit.
# A6 causal high-vol action signal: realized-vol window matches the regime
# labeler's "vol" axis (regime_labeler_20260611.VOL_WIN=63, annualized by
# sqrt(252)). The labeler's threshold is a FULL-SAMPLE median (ex-post, for
# stratification only -- see its docstring lines 17-19). For an ACTION signal we
# replace that with a point-in-time expanding median, shifted by 1 (past-only).
_A6_VOL_WIN          = 63   # = regime_labeler VOL_WIN (match labeler rv exactly)
_A6_EXPMED_MINPER    = 252  # min periods before the expanding median is defined

# Canonical P09_C1 after-tax reference (from leverup_b1c1_20260612.csv / CURRENT_BEST §v8)
P09C1_CANON = {
    "CAGR_IS_at":  0.198838,   # +19.88%
    "CAGR_OOS_at": 0.177672,   # +17.77%
    "Sharpe_FULL_pretax_OOS": 0.911513,  # Sharpe_OOS pre-tax (CURRENT_BEST: 0.912)
    "MaxDD": -0.349879,        # -34.99%
}
SANITY_TOL_TIGHT = 1e-6        # A0 vs direct _build_full_c1 P09_C1


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _stage1_full_gate(label, nav_dt, r, tpy, regimes, stress,
                      is_mask, oos_mask, r_v7, r_b3a):
    """Full Stage-1 gate: standard-10 + WFA + CPCV + regime + stress + bootstrap."""
    ev = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                   baseline_r=r_v7)
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)

    wfe     = float(ev["wfa_WFE"])
    reg_min = float(ev["regime_min_at"])
    w10y    = aft["Worst10Y_star"]
    maxdd   = pre["MaxDD_FULL"]

    v_maxdd  = maxdd < HARD_VETO_MAXDD
    v_wfe    = wfe   > HARD_VETO_WFE
    v_w10y   = w10y  < HARD_VETO_W10Y
    v_reg    = reg_min < HARD_VETO_REGIME
    veto_s1  = v_maxdd or v_wfe or v_w10y or v_reg

    print("      bootstrap vs V7 ...")
    boot_v7  = _block_bootstrap_multimetric(r, r_v7, is_mask, oos_mask,
                                            n_boot=N_BOOT, block=BLOCK, seed=SEED)
    print("      bootstrap vs B3a ...")
    boot_b3a = _block_bootstrap_multimetric(r, r_b3a, is_mask, oos_mask,
                                            n_boot=N_BOOT, block=BLOCK, seed=SEED)

    return {
        "CAGR_IS_at":    aft["CAGR_IS"],
        "CAGR_OOS_at":   aft["CAGR_OOS"],
        "min9_at":       _min_at(aft),
        "Sharpe_FULL":   pre["Sharpe_FULL"],
        "Sharpe_OOS":    pre["Sharpe_OOS"],
        "MaxDD_FULL":    maxdd,
        "Worst10Y_at":   w10y,
        "P10_5Y_at":     aft["P10_5Y"],
        "Worst5Y_at":    aft["Worst5Y"],
        "Trades_yr":     aft["Trades_yr"],
        "Worst1D":       pre["Worst1D"],
        "Worst1D_date":  pre["Worst1D_date"],
        "IS_OOS_gap_pp": aft["IS_OOS_gap_pp"],
        # WFA
        "wfa_WFE":       wfe,
        "wfa_CI95_lo":   float(ev["wfa_CI95_lo"]),
        "wfa_t_p":       float(ev["wfa_t_p"]),
        # CPCV
        "cpcv_p10_at":   float(ev["cpcv_p10_at"]),
        "cpcv_worst_at": float(ev["cpcv_worst_at"]),
        "cpcv_med_at":   float(ev["cpcv_med_at"]),
        # Regime
        "regime_min_at": reg_min,
        "regime":        ev["regime"],
        "stress":        ev["stress"],
        # Bootstrap vs V7
        "mm_v7_P_min":     boot_v7["P_min_better"],
        "mm_v7_CI95_min":  boot_v7["CI95_lo_min_pp"],
        "mm_v7_P_maxdd":   boot_v7["P_maxdd_better"],
        "mm_v7_CI95_dd":   boot_v7["CI95_lo_dd_pp"],
        "mm_v7_P_w10y":    boot_v7["P_worst10y_better"],
        "mm_v7_CI95_w10y": boot_v7["CI95_lo_w10y_pp"],
        "mm_v7_P_sharpe":  boot_v7["P_sharpe_better"],
        "mm_v7_CI95_shp":  boot_v7["CI95_lo_sharpe"],
        # Bootstrap vs B3a
        "mm_b3a_P_min":     boot_b3a["P_min_better"],
        "mm_b3a_CI95_min":  boot_b3a["CI95_lo_min_pp"],
        "mm_b3a_P_maxdd":   boot_b3a["P_maxdd_better"],
        "mm_b3a_CI95_dd":   boot_b3a["CI95_lo_dd_pp"],
        "mm_b3a_P_w10y":    boot_b3a["P_worst10y_better"],
        "mm_b3a_CI95_w10y": boot_b3a["CI95_lo_w10y_pp"],
        "mm_b3a_P_sharpe":  boot_b3a["P_sharpe_better"],
        "mm_b3a_CI95_shp":  boot_b3a["CI95_lo_sharpe"],
        # Veto flags
        "s1_veto_maxdd": int(v_maxdd),
        "s1_veto_wfe":   int(v_wfe),
        "s1_veto_w10y":  int(v_w10y),
        "s1_veto_reg":   int(v_reg),
        "VETO_s1":       int(veto_s1),
    }


def main():
    print("=" * 120)
    print("P09_C1 ALLOCATION-VARIATION STAGE-1 RUNNER  2026-06-20")
    print("8 OUT-sleeve variations A0-A7 (A0 == canonical P09_C1)")
    print("Stage-1 gates: WFA (49w) + CPCV + Regime + Bootstrap vs V7/B3a")
    print("Hard veto: MaxDD<-50%% / WFE>1.5 / Worst10Y*<0 / Regime_min(bear)<-10%%")
    print("=" * 120)

    # ---- Load shared data ----
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

    # ---- Gold/Bond auxiliary series (same as template) ----
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

    # out_strength: 1.0 at deepest OUT (lev_mod=0), 0.0 near the W1 exit threshold.
    # I-1 FIX: lag lev_mod_065 by _LEVMOD_DELAY (== strategy_runners._DELAY=2)
    # BEFORE deriving the cash-sizing knob. The canonical NAV builders all apply
    # .shift(_DELAY) to lev_mod_065; reading the same-day value here would be a
    # ~2-day look-ahead. _EXIT_THR (0.3) is the DH-W1 W1 exit threshold.
    lev_mod_raw = np.nan_to_num(np.asarray(a["lev_mod_065"], float), nan=0.0)
    lev_mod = pd.Series(lev_mod_raw).shift(_LEVMOD_DELAY).fillna(0.0).values
    out_strength = np.clip((_EXIT_THR - lev_mod) / _EXIT_THR, 0.0, 1.0)

    # =========================================================================
    # IN-leg base return for the OUT-fill variant builder
    # =========================================================================
    print("\nBuilding P09_C1 IN-leg base (V7_MAP default, scale=1.0, excess=0.0) ...")
    _, r_base_in, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)

    def build_variant(wg_, wb_, bond_on_, alloc_fn, extra_ctx=None, in_brake=False):
        # return_weights=True so we can surface the intra-sleeve (Gold/Bond
        # re-weighting) turnover that _count_fund_transitions (OUT<->IN flips
        # only) cannot see. The (w_gold, w_bond, w_cash) returned are the SAME
        # arrays used to build r -- no re-running of alloc_fn (I-3 diagnostic).
        nav_arr, r, eff, w_gold, w_bond, w_cash = _build_out_fill_variant(
            r_base_in, ret_gold, ret_bond, fund_active, wg_, wb_, bond_on_, sofr_arr,
            alloc_fn=alloc_fn, return_weights=True, **(extra_ctx or {}))
        if in_brake:
            r = apply_in_leg_vol_brake(r, fund_active, sofr_arr,
                                       target_vol=0.30, window=63)
            nav_arr = np.cumprod(1.0 + r)
        nav_dt = pd.Series(nav_arr, index=dates_dt)
        tpy = tpy_base + _count_fund_transitions(eff) / n_years

        # --- I-3: OUT-sleeve turnover diagnostic (UNCHARGED, reported only) ---
        # sum of |Δw_gold| + |Δw_bond| over active (OUT) days / n_years. Exposes
        # the daily re-weighting churn of A2/A4/A5/A7 that the OUT<->IN flip count
        # misses. NOTE: for A7 (in_brake) the extra churn is on the IN leg, NOT
        # this OUT sleeve; its OUT sleeve == base, so A7's sleeve_turnover_yr ~=
        # A0's. A7's IN-leg churn is a separate (unmeasured) cost.
        dwg = np.abs(np.diff(w_gold, prepend=w_gold[0]))
        dwb = np.abs(np.diff(w_bond, prepend=w_bond[0]))
        active = fund_active.astype(float)
        sleeve_turnover_yr = float(np.sum((dwg + dwb) * active) / n_years)
        return nav_dt, r, tpy, sleeve_turnover_yr

    # =========================================================================
    # SANITY GATE: A0 must reproduce canonical P09_C1 (direct _build_full_c1)
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: A0_P09_C1_BASE vs direct _build_full_c1 P09_C1 (tol 1e-6 on CAGR_OOS)")
    print("=" * 120)

    # Direct canonical P09_C1 NAV (the authoritative wiring)
    canon_nav, canon_r, canon_tpy, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    canon_pre = compute_10metrics(canon_nav, canon_tpy)
    canon_aft = _apply_aftertax(canon_pre)
    canon_cagr_oos = canon_aft["CAGR_OOS"]
    canon_cagr_is  = canon_aft["CAGR_IS"]
    canon_maxdd    = canon_pre["MaxDD_FULL"]
    canon_sharpe   = canon_pre["Sharpe_FULL"]

    # A0 via the OUT-fill builder (4th element = sleeve-turnover diagnostic)
    a0_nav, a0_r, a0_tpy, _a0_sto = build_variant(wg, wb, bond_on, alloc_base)
    a0_pre = compute_10metrics(a0_nav, a0_tpy)
    a0_aft = _apply_aftertax(a0_pre)
    a0_cagr_oos = a0_aft["CAGR_OOS"]

    diff_a0_direct   = abs(a0_cagr_oos - canon_cagr_oos)
    diff_direct_book = abs(canon_cagr_oos - P09C1_CANON["CAGR_OOS_at"])

    print("  A0 CAGR_OOS(at)            : %+.8f%%" % (a0_cagr_oos * 100))
    print("  direct _build_full_c1 OOS : %+.8f%%" % (canon_cagr_oos * 100))
    print("  CURRENT_BEST P09_C1 OOS   : %+.8f%%" % (P09C1_CANON["CAGR_OOS_at"] * 100))
    print("  |A0 - direct|             : %.3e   (tol %.0e)" % (diff_a0_direct, SANITY_TOL_TIGHT))
    print("  |direct - CURRENT_BEST|   : %.3e   (book value, ~rounding)" % diff_direct_book)
    print("  A0 CAGR_IS=%+.4f%%  direct CAGR_IS=%+.4f%%  book CAGR_IS=%+.4f%%"
          % (a0_aft["CAGR_IS"] * 100, canon_cagr_is * 100, P09C1_CANON["CAGR_IS_at"] * 100))
    print("  A0 MaxDD=%+.4f%%  direct MaxDD=%+.4f%%  book MaxDD=%+.4f%%"
          % (a0_pre["MaxDD_FULL"] * 100, canon_maxdd * 100, P09C1_CANON["MaxDD"] * 100))
    print("  A0 Sharpe_FULL=%.4f  direct Sharpe_FULL=%.4f  (book Sharpe_OOS pretax=%.4f)"
          % (a0_pre["Sharpe_FULL"], canon_sharpe, P09C1_CANON["Sharpe_FULL_pretax_OOS"]))

    if diff_a0_direct > SANITY_TOL_TIGHT:
        print("\nSANITY FAILED -- A0 does not match direct _build_full_c1 P09_C1 to 1e-6.")
        print("The OUT-fill wiring is wrong. Halting.")
        sys.exit(1)
    # soft check vs book value (rounding in published table)
    if diff_direct_book > 0.0005:
        print("\n  WARN: direct P09_C1 CAGR_OOS differs from CURRENT_BEST by >0.05pp.")
    print("  SANITY PASSED (A0 == direct P09_C1 to 1e-6). Proceeding to Stage-1.\n")

    # =========================================================================
    # Baselines for bootstrap: V7_TQQQ and B3a_k365
    # =========================================================================
    print("Building V7_TQQQ baseline (bootstrap baseline) ...")
    _, r_v7, _, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    print("Building B3a_k365 reference (default map, scale=1.15, k365 excess) ...")
    _, r_b3a, _, _ = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=B3A_MAP_DEFAULT, lev_scale=B3A_LEV_SCALE,
        excess_extra=B3A_EXCESS_EXTRA)

    # =========================================================================
    # Regime labels and stress masks
    # =========================================================================
    print("Building regime labels and stress masks ...")
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress  = stress_masks(dates_dt)

    # EX-POST high-vol mask (full-sample median threshold). LEGITIMATE for
    # _eval_one stratification of ALL rows (descriptive, not a trading signal),
    # so it stays the `regimes` object. It must NOT drive any allocation.
    highvol_mask_expost = (regimes["vol"].values == "highvol")
    print("  highvol days (ex-post, stratification): %d of %d (%.1f%%)"
          % (int(highvol_mask_expost.sum()), n,
             100.0 * highvol_mask_expost.sum() / n))

    # I-2 FIX: CAUSAL high-vol ACTION signal for A6's gold tilt (point-in-time).
    # The regime labeler computes rv = close.pct_change rolling(63).std(ddof=1)
    # *sqrt(252), then labels highvol via a FULL-SAMPLE median -- ex-post, only
    # valid for stratification (labeler docstring lines 17-19). A6 TRADES on the
    # high-vol flag, so a full-sample median = look-ahead. We replicate the
    # labeler's rv EXACTLY (a["ret"] == close.pct_change().fillna(0), VOL_WIN=63,
    # *sqrt(252)) but replace the threshold with an EXPANDING median shifted by 1
    # so today's tilt decision uses only past data.
    rv_a6 = (pd.Series(np.asarray(a["ret"], float))
             .rolling(_A6_VOL_WIN, min_periods=_A6_VOL_WIN).std(ddof=1)
             * np.sqrt(TRADING_DAYS))
    causal_med = rv_a6.expanding(min_periods=_A6_EXPMED_MINPER).median().shift(1)
    highvol_mask_causal = (rv_a6 > causal_med).fillna(False).values
    print("  highvol days (causal, A6 action signal): %d of %d (%.1f%%)"
          % (int(highvol_mask_causal.sum()), n,
             100.0 * highvol_mask_causal.sum() / n))

    # =========================================================================
    # Build alternate weights / gates for variations
    # =========================================================================
    wg63, wb63 = wg, wb
    wg126, wb126     = inverse_vol_weights_cadence(ret_gold, ret_bond, 126, 5)
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
        # A6 TRADES on the high-vol flag -> use the CAUSAL (point-in-time)
        # mask, NOT the ex-post regimes["vol"] median (I-2 fix).
        ("A6_GOLD_TILT",    lambda: build_variant(wg63, wb63, bond_on,
                                                  make_alloc_gold_tilt(0.75),
                                                  extra_ctx={"highvol_mask": highvol_mask_causal})),
        ("A7_IN_VOL_BRAKE", lambda: build_variant(wg63, wb63, bond_on, alloc_base,
                                                  in_brake=True)),
    ]

    # =========================================================================
    # Stage-1 gate for each variation
    # =========================================================================
    print("\n" + "=" * 120)
    print("STAGE 1: Full gate for 8 allocation variations")
    print("=" * 120)

    results = []
    for label, builder in VARIATIONS:
        print("\n  [%s] building NAV ..." % label)
        nav_dt, r, tpy, sleeve_turnover_yr = builder()
        print("    Running WFA + CPCV + regime + stress + bootstrap ...")
        s1 = _stage1_full_gate(label, nav_dt, r, tpy, regimes, stress,
                               is_mask, oos_mask, r_v7, r_b3a)
        s1["sleeve_turnover_yr"] = round(sleeve_turnover_yr, 4)

        veto_reasons = []
        if s1["s1_veto_maxdd"]: veto_reasons.append("MaxDD<-50%%")
        if s1["s1_veto_wfe"]:   veto_reasons.append("WFE>1.5")
        if s1["s1_veto_w10y"]:  veto_reasons.append("W10Y*<0")
        if s1["s1_veto_reg"]:   veto_reasons.append("Regime_min<-10%%")

        print("    CAGR_IS=%+.2f%%  CAGR_OOS=%+.2f%%  Sharpe=%.4f  MaxDD=%+.2f%%  Trd/yr=%.1f"
              % (s1["CAGR_IS_at"] * 100, s1["CAGR_OOS_at"] * 100,
                 s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan"),
                 s1["MaxDD_FULL"] * 100, s1["Trades_yr"]))
        print("    WFE=%.4f  CI95_lo=%+.2f%%  Regime_min=%+.2f%%  W10Y*=%+.2f%%  [VETO=%s %s]"
              % (s1["wfa_WFE"], s1["wfa_CI95_lo"] * 100, s1["regime_min_at"] * 100,
                 s1["Worst10Y_at"] * 100, "YES" if s1["VETO_s1"] else "no",
                 (", ".join(veto_reasons)) if veto_reasons else ""))
        print("    boot vs V7:  P_min=%.3f CI95_min=%+.2f%% P_maxdd=%.3f P_w10y=%.3f P_sharpe=%.3f"
              % (s1["mm_v7_P_min"], s1["mm_v7_CI95_min"], s1["mm_v7_P_maxdd"],
                 s1["mm_v7_P_w10y"], s1["mm_v7_P_sharpe"]))
        print("    sleeve_turnover_yr (OUT-sleeve |dw_g|+|dw_b|, uncharged diag): %.3f"
              % s1["sleeve_turnover_yr"])

        results.append({"label": label, "s1": s1})

    # =========================================================================
    # STAGE-1 GATE TABLE
    # =========================================================================
    print("\n" + "=" * 150)
    print("STAGE 1 FULL GATE TABLE -- 8 P09_C1 allocation variations")
    print("%-18s | %9s | %9s | %8s | %8s | %7s | %8s | %7s | %8s | %4s"
          % ("label", "CAGR_IS%", "CAGR_OOS%", "ShpFULL", "MaxDD%", "CI95%",
             "Reg_min%", "WFE", "W10Y*%", "VETO"))
    print("-" * 150)
    for entry in results:
        s1 = entry["s1"]
        veto_str = "VETO" if s1["VETO_s1"] else "PASS"
        shp = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        print("%-18s | %+8.2f%% | %+8.2f%% | %7.4f | %+6.2f%% | %+6.2f%% | %+7.2f%% | %7.4f | %+6.2f%% | %-4s"
              % (entry["label"][:18],
                 s1["CAGR_IS_at"] * 100, s1["CAGR_OOS_at"] * 100, shp,
                 s1["MaxDD_FULL"] * 100, s1["wfa_CI95_lo"] * 100,
                 s1["regime_min_at"] * 100, s1["wfa_WFE"],
                 s1["Worst10Y_at"] * 100, veto_str))

    # =========================================================================
    # CSV OUTPUT
    # =========================================================================
    print("\nBuilding CSV ...")
    axes_order_keys = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                       "rate:rate_up", "rate:rate_down"]
    rows = []
    for entry in results:
        s1 = entry["s1"]
        shp_full = s1["Sharpe_FULL"] if s1["Sharpe_FULL"] is not None else float("nan")
        row = {
            "label":          entry["label"],
            "CAGR_IS_at":     round(s1["CAGR_IS_at"], 6),
            "CAGR_OOS_at":    round(s1["CAGR_OOS_at"], 6),
            "min9_at":        round(s1["min9_at"], 6),
            "IS_OOS_gap_pp":  round(s1["IS_OOS_gap_pp"], 4),
            "Sharpe_FULL":    round(shp_full, 4),
            "Sharpe_OOS":     round(s1["Sharpe_OOS"], 4),
            "MaxDD_FULL":     round(s1["MaxDD_FULL"], 6),
            "Worst10Y_at":    round(s1["Worst10Y_at"], 6),
            "P10_5Y_at":      round(s1["P10_5Y_at"], 6),
            "Worst5Y_at":     round(s1["Worst5Y_at"], 6),
            "Trades_yr":      round(s1["Trades_yr"], 2),
            "sleeve_turnover_yr": round(s1["sleeve_turnover_yr"], 4),
            "Worst1D":        round(s1["Worst1D"] if s1["Worst1D"] is not None else float("nan"), 6),
            "Worst1D_date":   s1["Worst1D_date"] if s1["Worst1D_date"] else "",
            "wfa_WFE":        round(s1["wfa_WFE"], 6),
            "wfa_CI95_lo":    round(s1["wfa_CI95_lo"], 6),
            "wfa_t_p":        round(s1["wfa_t_p"], 10),
            "cpcv_p10_at":    round(s1["cpcv_p10_at"], 6),
            "cpcv_worst_at":  round(s1["cpcv_worst_at"], 6),
            "cpcv_med_at":    round(s1["cpcv_med_at"], 6),
            "regime_min_at":  round(s1["regime_min_at"], 6),
            "mm_v7_P_min":     round(float(s1["mm_v7_P_min"]), 4),
            "mm_v7_CI95_min":  round(float(s1["mm_v7_CI95_min"]), 4),
            "mm_v7_P_maxdd":   round(float(s1["mm_v7_P_maxdd"]), 4),
            "mm_v7_CI95_dd":   round(float(s1["mm_v7_CI95_dd"]), 4),
            "mm_v7_P_w10y":    round(float(s1["mm_v7_P_w10y"]), 4),
            "mm_v7_P_sharpe":  round(float(s1["mm_v7_P_sharpe"]), 4),
            "mm_b3a_P_min":    round(float(s1["mm_b3a_P_min"]), 4),
            "mm_b3a_CI95_min": round(float(s1["mm_b3a_CI95_min"]), 4),
            "mm_b3a_P_maxdd":  round(float(s1["mm_b3a_P_maxdd"]), 4),
            "mm_b3a_CI95_dd":  round(float(s1["mm_b3a_CI95_dd"]), 4),
            "mm_b3a_P_w10y":   round(float(s1["mm_b3a_P_w10y"]), 4),
            "mm_b3a_P_sharpe": round(float(s1["mm_b3a_P_sharpe"]), 4),
            "s1_veto_maxdd":   s1["s1_veto_maxdd"],
            "s1_veto_wfe":     s1["s1_veto_wfe"],
            "s1_veto_w10y":    s1["s1_veto_w10y"],
            "s1_veto_reg":     s1["s1_veto_reg"],
            "VETO_s1":         s1["VETO_s1"],
        }
        reg = s1["regime"]
        for ax in axes_order_keys:
            row["regime_" + ax.replace(":", "_")] = round(reg.get(ax, float("nan")), 6)
        rows.append(row)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "p09c1_alloc_stage1_20260620.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f",
                              encoding="utf-8-sig")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # RETURN_BLOCK
    # =========================================================================
    return_block = {
        "script": "p09c1_alloc_stage1_20260620.py",
        "date": "2026-06-20",
        "p09c1_wiring": {
            "v7_map": "default V7_MAP {0:1.20,1:1.10,2:1.00,3:1.00}",
            "lev_scale": P09C1_LEV_SCALE,
            "excess_extra": P09C1_EXCESS_EXTRA,
            "note": "excess_extra=0.0 reproduces canonical cfd_excess=False",
        },
        "sanity_gate": {
            "A0_CAGR_OOS_at_pct":      round(a0_cagr_oos * 100, 6),
            "direct_P09C1_CAGR_OOS_at_pct": round(canon_cagr_oos * 100, 6),
            "book_P09C1_CAGR_OOS_at_pct":   round(P09C1_CANON["CAGR_OOS_at"] * 100, 6),
            "abs_diff_A0_vs_direct":   float(diff_a0_direct),
            "abs_diff_direct_vs_book": float(diff_direct_book),
            "PASS": bool(diff_a0_direct <= SANITY_TOL_TIGHT),
        },
        "variations": [
            {
                "label": e["label"],
                "CAGR_IS_at_pct":  round(e["s1"]["CAGR_IS_at"] * 100, 4),
                "CAGR_OOS_at_pct": round(e["s1"]["CAGR_OOS_at"] * 100, 4),
                "Sharpe_FULL":     round(e["s1"]["Sharpe_FULL"], 4)
                                   if e["s1"]["Sharpe_FULL"] is not None else None,
                "MaxDD_pct":       round(e["s1"]["MaxDD_FULL"] * 100, 4),
                "Worst10Y_at_pct": round(e["s1"]["Worst10Y_at"] * 100, 4),
                "Trades_yr":       round(e["s1"]["Trades_yr"], 2),
                "sleeve_turnover_yr": round(e["s1"]["sleeve_turnover_yr"], 4),
                "wfa_WFE":         round(e["s1"]["wfa_WFE"], 4),
                "wfa_CI95_lo_pct": round(e["s1"]["wfa_CI95_lo"] * 100, 4),
                "regime_min_at_pct": round(e["s1"]["regime_min_at"] * 100, 4),
                "VETO_s1":         e["s1"]["VETO_s1"],
            }
            for e in results
        ],
        "csv": csv_path,
        "n_boot": N_BOOT, "block": BLOCK, "seed": SEED,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))

    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
