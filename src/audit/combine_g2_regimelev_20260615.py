"""
src/audit/combine_g2_regimelev_20260615.py
==========================================
G2: Regime-linked boost/scale attenuation natively integrated into B3a_k365.

Purpose (MULTISTRATEGY_COMBINE_PLAN_20260615.md, G2):
  Reduce V7 boost multiplier and/or uniform scale on defensive regime days.
  Regime labels (trend:bear / vol:highvol) from build_regime_labels are used
  as a *daily trading signal* here -- NOT post-hoc stratification labels.
  This requires causal treatment (see "Causality ruling" below).

Causality ruling:
  trend: close > MA200 (trailing, fully causal at every day).
  vol:   rv63 vs vol_median.
    - vol_median in regime_labeler_20260611 uses FULL-SAMPLE median
      => lookahead when used as a trading signal.
    - Fix applied here: vol_median is computed on IS data only
      (dates <= IS_END = 2021-05-07, approximately 3/4 of full data).
      IS-only median is a conservative but causal approximation
      (OOS regime classification uses the same IS-estimated threshold).
    - This IS-estimated threshold is reported in sanity output.
  rate:  not used in G2.

G2 Design:
  On IN-regime days (inside DH-W1 signal active), apply attenuation s_def < 1.
  ON defensive-trigger days:
    effective_mult = V7_MULT * s_def   (reduces leveraged upside)
    where V7_MULT already encodes the v7_map lookup for the day.
  Attenuation conditions (2 variants):
    (a) bear_only   : trend:bear
    (b) bear_or_hv  : trend:bear OR vol:highvol
  Scale levels s_def in {0.85, 0.90}.
  => 4 configurations total: (a+0.85), (a+0.90), (b+0.85), (b+0.90)
  + 1 sanity anchor: s_def=1.0 (no attenuation) -> must reproduce B3a_k365 exactly.

B3a_k365 base (fixed):
  v7_map = {Q0:1.40, Q1:1.40, Q2:1.05, Q3:1.00}
  lev_scale = 1.15
  excess_extra = EXCESS_EXTRA_K365_CENTRE = 0.0025  (k365 0.25%/yr)
  OUT-fill = P09 C1 (Gold/Bond mix + SOFR on bond-OFF days)

Sanity anchor (mandatory):
  s_def=1.0 must reproduce B3a_k365 known values:
    min(IS,OOS) aftertax >= 20.93%  (tolerance +-0.05pp of 20.98%)
    MaxDD ~ -38.20% (+-1.0pp)
    Sharpe_OOS ~ 0.904 (+-0.02)
    Regime_min ~ -2.88% (+-0.5pp)
    2008-window CAGR ~ +16.78% (+-1.5pp)
  Failure => halt and report.

Stage 0 metrics (each config):
  Standard 10 metrics (CAGR_IS/OOS, min9, gap, Sharpe, MaxDD,
  Worst10Y*, P10_5Y, Worst5Y, Trades/yr) + after-tax (x0.8273)
  + worst calendar year
  + Regime_min (min over trend:bull/bear, vol:calm/highvol CAGR)
  + 2008 stress window CAGR (gfc_2008: 2007-10-09 to 2009-03-09)
  + delta vs B3a_k365 for all of above

Survival gate:
  No hard veto (MaxDD<-50% / W10Y*<0 / Regime_min<-10%)
  AND (Regime_min improvement >= +1pp
       OR 2008-window improvement >= +1pp
       OR MaxDD improvement >= +2pp)
  AND min9 degradation <= 1.0pp

Outputs:
  src/audit/combine_g2_regimelev_20260615.py   (this file)
  audit_results/combine_g2_regimelev_20260615.csv
  RETURN_BLOCK printed as JSON to stdout

Constraints:
  ASCII-only prints (Windows cp932).
  No git operations. No temp files.
  Post-hoc multiplication forbidden -- attenuation is baked into mult_v7 array.
  Timeout budget: 600s (Stage 0 only; no WFA/CPCV/bootstrap here).
"""

from __future__ import annotations

import json
import os
import sys
import types

# ---- multitasking stub (yfinance dependency) ---------------------------------
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

# ---- Re-use from k365_recost (single source of truth for B3a build) ----------
from src.audit.k365_recost_20260612 import (
    _build_nav_v7_tqqq_param,
    _build_p09_on_base_c1,
    EXCESS_EXTRA_K365_CENTRE,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom, LU2_SCALE,
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
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    _build_v7_mult, DELAY as V7_DELAY, DH_PER_UNIT, NAV_FLOOR,
)
from src.audit.lu_cfd_recost_20260611 import (
    AFTER_TAX, SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)

# ============================================================================
# Constants
# ============================================================================

# B3a_k365 fixed configuration
B3A_V7_MAP    = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE = 1.15
B3A_EXCESS    = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# After-tax multiplier (annual forced-reset CFD taxation model)
AFTER_TAX_MULT = AFTER_TAX  # 0.8273

# Known B3a_k365 anchor values (from LEVERUP_SWEEP_RESULTS_20260612.md)
ANCHOR_MIN9    = 0.2098   # +20.98%
ANCHOR_MAXDD   = -0.3820  # -38.20%
ANCHOR_SHARPE  = 0.904
ANCHOR_REGMIN  = -0.0288  # -2.88%
ANCHOR_2008    = 0.1678   # +16.78% (gfc_2008 window)

TOL_MIN9   = 0.0005  # 0.05pp
TOL_MAXDD  = 0.010   # 1.0pp
TOL_SHARPE = 0.020
TOL_REGMIN = 0.005   # 0.5pp
TOL_2008   = 0.015   # 1.5pp

# G2 sweep parameters
S_DEF_VALUES = [0.85, 0.90]
CONDITIONS   = ["bear_only", "bear_or_hv"]

# Hard-veto thresholds
VETO_MAXDD   = -0.50
VETO_W10Y    = 0.00
VETO_REGMIN  = -0.10

# Survival gate thresholds vs B3a
SURV_REGMIN_IMPROVE  = 0.010   # +1pp
SURV_STRESS08_IMPROVE = 0.010  # +1pp
SURV_MAXDD_IMPROVE   = 0.020   # +2pp (absolute, improve=reduce DD)
SURV_MIN9_DEGRADE    = -0.010  # min9 must not degrade more than 1pp

# GFC 2008 stress window key (matches STRESS_WINDOWS in regime_labeler)
STRESS_2008_KEY = "gfc_2008"


# ============================================================================
# Causal vol-regime builder
# ============================================================================

def _build_causal_regime_flags(
    close: pd.Series,
    sofr_daily,
    dates_dt: pd.DatetimeIndex,
) -> tuple:
    """Return (bear_flag, highvol_flag) as boolean numpy arrays, causal.

    trend:bear  => close < MA200 (trailing 200d, fully causal).
    vol:highvol => rv63 > IS_vol_median
                   IS_vol_median computed on IS subsample (dates <= IS_END)
                   to avoid lookahead beyond IS period.

    Returns:
        bear_flag:   bool array, length=len(dates_dt)
        highvol_flag: bool array, length=len(dates_dt)
        vol_median_is: float, the IS-estimated vol threshold (printed for audit)
    """
    close_arr = np.asarray(close, float)
    n = len(dates_dt)

    # --- trend: close < MA200 ---
    close_s = pd.Series(close_arr, index=dates_dt)
    ma200 = close_s.rolling(200, min_periods=200).mean()
    bear_flag = np.where(np.isnan(ma200.values), False, close_arr < ma200.values)

    # --- vol: rv63 vs IS-estimated median ---
    ret = close_s.pct_change().fillna(0.0)
    rv63 = ret.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(TRADING_DAYS)
    is_mask_vol = dates_dt <= IS_END
    rv_is = rv63.values[is_mask_vol & ~np.isnan(rv63.values)]
    vol_median_is = float(np.nanmedian(rv_is)) if len(rv_is) > 0 else 0.20
    highvol_flag = np.where(np.isnan(rv63.values), False,
                            rv63.values > vol_median_is)

    return bear_flag.astype(bool), highvol_flag.astype(bool), vol_median_is


# ============================================================================
# G2 NAV builder -- native attenuation baked into mult_v7
# ============================================================================

def _build_g2_nav(
    shared,
    dates_dt: pd.DatetimeIndex,
    n_years: float,
    ret_gold, ret_bond,
    fund_active, wg, wb, bond_on,
    sofr_arr,
    bear_flag: np.ndarray,
    highvol_flag: np.ndarray,
    condition: str,
    s_def: float,
):
    """Build G2 NAV with regime-linked attenuation of V7 multiplier.

    Native integration: mult_v7 array is modified element-wise before
    passing to _build_nav_v7_tqqq_param.  No post-hoc multiplication.

    condition: 'bear_only' => attenuate when trend:bear
               'bear_or_hv' => attenuate when (trend:bear OR vol:highvol)
    s_def:     scale factor applied to mult_v7 on defensive days.

    The uniform scale (B3A_LEV_SCALE=1.15) and V7 map are baked into
    mult_v7 via _build_v7_mult_custom.  Regime attenuation multiplies
    the resulting mult_v7 by s_def on defensive days, effectively:
      effective_v7_scale_day = (v7_map[Q] * B3A_LEV_SCALE) * s_def
    This is equivalent to choosing a smaller uniform scale on those days.

    OUT-fill: C1 (SOFR on bond-OFF days), same as B3a_k365.

    Returns: (nav_dt, r, tpy, excess_days)
    """
    a = shared["assets"]

    # Build base V7 multiplier array (v7_map x lev_scale)
    mult_v7_base = _build_v7_mult_custom(dates_dt, B3A_V7_MAP)
    mult_v7_base = mult_v7_base * float(B3A_LEV_SCALE)

    # Determine defensive days
    if condition == "bear_only":
        def_flag = bear_flag
    elif condition == "bear_or_hv":
        def_flag = bear_flag | highvol_flag
    else:
        raise ValueError("Unknown condition: %s" % condition)

    # Apply attenuation natively to mult_v7
    mult_v7_g2 = mult_v7_base.copy()
    mult_v7_g2 = np.where(def_flag, mult_v7_g2 * float(s_def), mult_v7_g2)

    # Build TQQQ-side NAV using parameterised builder from k365_recost
    # We pass mult_v7_g2 directly as the "pre-scaled" multiplier.
    # _build_nav_v7_tqqq_param accepts mult_v7 as already full scale,
    # and internally does: lev_mod = lev_raw_masked * mult_v7; L = lev_mod * 3
    # We pass mult_v7_g2 as mult_v7, with lev_scale=1.0 embedded already.
    close       = a["close"]
    dates_raw   = a["dates"]
    gold_2x     = a["gold_2x"]
    bond_3x     = a["bond_3x"]
    sofr_daily  = np.asarray(a["sofr"], float)
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn_arr = np.asarray(shared["wn"], float)
    wg_arr = np.asarray(shared["wg"], float)
    wb_arr = np.asarray(shared["wb"], float)

    nav_base, tpy_base, excess_days = _build_nav_v7_tqqq_param(
        close, dates_raw, gold_2x, bond_3x, sofr_daily,
        lev_raw_masked, wn_arr, wg_arr, wb_arr,
        mult_v7_g2,
        excess_extra=B3A_EXCESS,
    )
    r_base = nav_base.pct_change().fillna(0.0).values

    # Apply C1 OUT fill
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        sofr_arr, dates_dt, tpy_base, n_years,
    )
    return nav_dt, r, tpy, excess_days


# ============================================================================
# Metrics helpers
# ============================================================================

def _min9(aft: dict) -> float:
    return min(float(aft["CAGR_IS"]), float(aft["CAGR_OOS"]))


def _stress_2008_ret(nav_dt: pd.Series) -> float:
    """Cumulative total return of NAV over the gfc_2008 window.

    Window: 2007-10-09 to 2009-03-09 (matches STRESS_WINDOWS in regime_labeler).
    Returns total cumulative return (not annualised), matching the
    'stress_gfc_2008_ret' column in k365_recost_20260612.csv and
    extended_eval_20260611._eval_one stress dict ('ret' key = _cum_ret).

    _cum_ret in extended_eval is: np.prod(1+r_sub) - 1
    (i.e. total cumulative return, not CAGR).
    """
    lo = pd.Timestamp("2007-10-09")
    hi = pd.Timestamp("2009-03-09")
    sub = nav_dt[(nav_dt.index >= lo) & (nav_dt.index <= hi)]
    if len(sub) < 2:
        return float("nan")
    return float(sub.iloc[-1] / sub.iloc[0]) - 1.0


def _regime_min_from_labels(nav_dt: pd.Series, regimes: pd.DataFrame) -> float:
    """Compute min CAGR across regime-axis segments (trend x vol x rate).

    Uses _regime_cagr from extended_eval_20260611 which already applies AFTER_TAX.
    Returns the minimum aftertax CAGR segment.
    """
    from src.audit.extended_eval_20260611 import _regime_cagr
    r = nav_dt.pct_change().fillna(0.0).values
    # _regime_cagr(r, regimes) -> (cagr_map_dict, regime_min_float)
    _, regime_min = _regime_cagr(r, regimes)
    return float(regime_min)


def _compute_stage0(
    label: str,
    nav_dt: pd.Series,
    tpy: float,
    excess_days: int,
    regimes: pd.DataFrame,
    dates_dt: pd.DatetimeIndex,
) -> dict:
    """Compute Stage 0 metrics for one configuration.

    Returns dict with all required fields.
    """
    pre  = compute_10metrics(nav_dt, tpy)
    aft  = _apply_aftertax(pre)
    cy   = _calendar_year_returns(nav_dt)

    min9    = _min9(aft)
    regime_min = _regime_min_from_labels(nav_dt, regimes)
    # stress_2008: pretax cumulative return over gfc_2008 window.
    # Matches 'stress_gfc_2008_ret' in k365_recost CSV (cumulative, not CAGR, not aftertax).
    # B3a_k365 known value: +16.78% pretax cumulative.
    stress_2008_at = _stress_2008_ret(nav_dt)  # pretax; "_at" kept for column-name consistency
    worst_cy  = float(cy.min())
    worst_cy_year = int(cy.idxmin())

    # Hard veto
    v_maxdd  = float(pre["MaxDD_FULL"]) < VETO_MAXDD
    v_w10y   = float(aft["Worst10Y_star"]) < VETO_W10Y
    v_regmin = regime_min < VETO_REGMIN if not np.isnan(regime_min) else False
    veto     = v_maxdd or v_w10y or v_regmin

    return {
        "label": label,
        # Standard 10 (aftertax where indicated)
        "CAGR_IS_at":      float(aft["CAGR_IS"]),
        "CAGR_OOS_at":     float(aft["CAGR_OOS"]),
        "min9_at":         min9,
        "gap_pp":          float(aft["IS_OOS_gap_pp"]),
        "Sharpe_OOS":      float(pre["Sharpe_OOS"]),
        "MaxDD_FULL":      float(pre["MaxDD_FULL"]),
        "Worst10Y_star_at":float(aft["Worst10Y_star"]),
        "P10_5Y_at":       float(aft["P10_5Y"]),
        "Worst5Y_at":      float(aft["Worst5Y"]),
        "Trades_yr":       float(aft["Trades_yr"]),
        "excess_days":     excess_days,
        # Additional
        "worst_cy":        worst_cy,
        "worst_cy_year":   worst_cy_year,
        "Regime_min_at":   regime_min,
        "stress_2008_at":  stress_2008_at,
        # Veto flags
        "veto_maxdd": int(v_maxdd),
        "veto_w10y":  int(v_w10y),
        "veto_regmin":int(v_regmin),
        "VETO":       int(veto),
    }


def _survival_gate(row: dict, anchor: dict) -> dict:
    """Apply survival gate vs anchor (B3a_k365 row).

    Returns {"SURVIVE": bool, "reasons": list_of_strings}.
    """
    reasons = []
    veto = bool(row["VETO"])

    # Improvements vs anchor
    regmin_imp = float(row["Regime_min_at"]) - float(anchor["Regime_min_at"])
    stress_imp = float(row["stress_2008_at"]) - float(anchor["stress_2008_at"])
    maxdd_imp  = float(anchor["MaxDD_FULL"]) - float(row["MaxDD_FULL"])  # positive = better
    min9_delta = float(row["min9_at"]) - float(anchor["min9_at"])

    if veto:
        reasons.append("HARD_VETO")
    if min9_delta < SURV_MIN9_DEGRADE:
        reasons.append("min9_degrade %.2fpp (limit -1.0pp)" % (min9_delta * 100))

    defensive_ok = (
        regmin_imp >= SURV_REGMIN_IMPROVE
        or stress_imp >= SURV_STRESS08_IMPROVE
        or maxdd_imp >= SURV_MAXDD_IMPROVE
    )
    if not defensive_ok:
        reasons.append(
            "no_defensive_improvement (regmin_delta=%.2fpp, stress08_delta=%.2fpp, maxdd_imp=%.2fpp)"
            % (regmin_imp * 100, stress_imp * 100, maxdd_imp * 100)
        )

    survive = (not veto) and (min9_delta >= SURV_MIN9_DEGRADE) and defensive_ok
    return {
        "SURVIVE": survive,
        "reasons": reasons,
        "regmin_imp_pp": round(regmin_imp * 100, 3),
        "stress08_imp_pp": round(stress_imp * 100, 3),
        "maxdd_imp_pp":  round(maxdd_imp * 100, 3),
        "min9_delta_pp": round(min9_delta * 100, 3),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 100)
    print("COMBINE G2 REGIME-LINKED ATTENUATION  2026-06-15/16")
    print("B3a_k365 base + regime-linked mult_v7 attenuation (native integration)")
    print("Stage 0 Screen: standard-10 + Regime_min + 2008-stress window")
    print("=" * 100)

    # ---- Load shared assets --------------------------------------------------
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a      = shared["assets"]
    mask   = np.asarray(shared["mask"], dtype=float)
    dates  = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    # ---- Gold/Bond 1x legs --------------------------------------------------
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x  = np.asarray(prepare_gold_local(dates), float)
    bond_1x  = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0),
        float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    # ---- P09 OUT-fill setup -------------------------------------------------
    out_mask_arr = (mask < 0.5)
    fund_active  = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on   = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr  = np.asarray(a["sofr"], float)

    out_bond_off = fund_active & (~bond_on.astype(bool))
    n_out_bondoff = int(out_bond_off.sum())
    print("\nOUT-and-bondOFF days: %d of %d (%.1f%%)" % (n_out_bondoff, n, 100.0 * n_out_bondoff / n))

    # ---- Regime labels (post-hoc stratification, full-sample vol_median) ----
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)

    # ---- Causal regime flags (IS-estimated vol_median) ---------------------
    bear_flag, highvol_flag, vol_med_is = _build_causal_regime_flags(
        a["close"], a["sofr"], dates_dt)

    full_vol_med = regimes.attrs.get("vol_median", float("nan"))
    print("\nCausality note:")
    print("  vol_median (full-sample, regime_labeler) = %.4f (%.1f%% ann.)"
          % (full_vol_med, full_vol_med * 100))
    print("  vol_median (IS-only causal threshold)    = %.4f (%.1f%% ann.)"
          % (vol_med_is, vol_med_is * 100))
    print("  G2 uses IS-only threshold for highvol_flag to avoid lookahead.")

    # Defensive day counts
    bear_days = int(bear_flag.sum())
    hv_days   = int(highvol_flag.sum())
    bear_or_hv = (bear_flag | highvol_flag)
    print("\nDefensive day counts:")
    print("  bear_only   : %d of %d (%.1f%%)" % (bear_days, n, 100.0 * bear_days / n))
    print("  highvol_only: %d of %d (%.1f%%)" % (hv_days, n, 100.0 * hv_days / n))
    print("  bear_or_hv  : %d of %d (%.1f%%)" % (int(bear_or_hv.sum()), n,
                                                   100.0 * bear_or_hv.sum() / n))

    # =========================================================================
    # SANITY ANCHOR: s_def=1.0 must reproduce B3a_k365
    # =========================================================================
    print("\n" + "=" * 100)
    print("SANITY ANCHOR: s_def=1.0 (no attenuation) -- must reproduce B3a_k365")
    print("  Expected: min9~+20.98% (tol+-0.05pp), MaxDD~-38.20% (+-1.0pp),")
    print("            Sharpe~0.904 (+-0.02), Regime_min~-2.88% (+-0.5pp),")
    print("            2008-window~+16.78% (+-1.5pp, pretax cumulative return)")
    print("=" * 100)

    anchor_nav, anchor_r, anchor_tpy, anchor_exc = _build_g2_nav(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        bear_flag, highvol_flag,
        condition="bear_only", s_def=1.0,
    )
    anchor_row = _compute_stage0(
        "B3a_k365_anchor", anchor_nav, anchor_tpy, anchor_exc, regimes, dates_dt)

    sanity_ok = True
    def _chk(name, got, expected, tol):
        global sanity_ok
        diff = abs(got - expected)
        ok = diff <= tol
        mark = "OK" if ok else "FAIL"
        print("  %-25s got=%+.4f%% expected=~%+.4f%% diff=%.4fpp tol=%.2fpp [%s]"
              % (name, got * 100, expected * 100, diff * 100, tol * 100, mark))
        if not ok:
            sanity_ok = False

    _chk("min9_at",       anchor_row["min9_at"],        ANCHOR_MIN9,    TOL_MIN9)
    _chk("MaxDD_FULL",    anchor_row["MaxDD_FULL"],      ANCHOR_MAXDD,   TOL_MAXDD)
    _chk("Sharpe_OOS",    anchor_row["Sharpe_OOS"],      ANCHOR_SHARPE,  TOL_SHARPE)
    _chk("Regime_min_at", anchor_row["Regime_min_at"],   ANCHOR_REGMIN,  TOL_REGMIN)
    _chk("stress_2008_at",anchor_row["stress_2008_at"],  ANCHOR_2008,    TOL_2008)

    if not sanity_ok:
        print("\n[HALT] SANITY ANCHOR FAILED -- regime insertion bug suspected.")
        print("       Check _build_g2_nav s_def=1.0 path and mult_v7 wiring.")
        sys.exit(2)

    print("\n  SANITY ANCHOR PASSED. Proceeding to G2 configurations.")

    # =========================================================================
    # Stage 0: 4 G2 configurations
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 0: G2 configurations (4 configs)")
    print("  Conditions: bear_only / bear_or_hv")
    print("  s_def: 0.85 / 0.90")
    print("=" * 100)

    configs = []
    for cond in CONDITIONS:
        for s_def in S_DEF_VALUES:
            configs.append({"condition": cond, "s_def": s_def})

    results = {}
    for cfg in configs:
        cond  = cfg["condition"]
        s_def = cfg["s_def"]
        label = "G2_%s_sdef%.2f" % (cond, s_def)
        label = label.replace(".", "")  # remove dot for CSV compatibility
        label = "G2_%s_s%d" % (cond, int(s_def * 100))
        print("\n  Building %s (condition=%s, s_def=%.2f) ..." % (label, cond, s_def))

        nav_dt, r, tpy, exc = _build_g2_nav(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            bear_flag, highvol_flag,
            condition=cond, s_def=s_def,
        )
        row = _compute_stage0(label, nav_dt, tpy, exc, regimes, dates_dt)
        surv = _survival_gate(row, anchor_row)
        row["SURVIVE"] = int(surv["SURVIVE"])
        row["surv_regmin_imp_pp"]  = surv["regmin_imp_pp"]
        row["surv_stress08_imp_pp"]= surv["stress08_imp_pp"]
        row["surv_maxdd_imp_pp"]   = surv["maxdd_imp_pp"]
        row["surv_min9_delta_pp"]  = surv["min9_delta_pp"]
        row["surv_reasons"]        = "; ".join(surv["reasons"]) if surv["reasons"] else ""
        row["condition"]  = cond
        row["s_def"]      = s_def
        results[label]    = row

        print("    min9=%+.2f%% MaxDD=%+.2f%% Sharpe=%.3f Regime_min=%+.2f%% 2008=%+.2f%% VETO=%s SURVIVE=%s"
              % (row["min9_at"] * 100, row["MaxDD_FULL"] * 100, row["Sharpe_OOS"],
                 row["Regime_min_at"] * 100, row["stress_2008_at"] * 100,
                 "YES" if row["VETO"] else "no", "YES" if row["SURVIVE"] else "no"))
        print("    vs B3a: Regime_min %+.2fpp  2008 %+.2fpp  MaxDD %+.2fpp  min9 %+.2fpp"
              % (surv["regmin_imp_pp"], surv["stress08_imp_pp"],
                 surv["maxdd_imp_pp"], surv["min9_delta_pp"]))

    # =========================================================================
    # Stage 0 summary table
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 0 SUMMARY TABLE")
    hdr = ("%-28s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %6s | %8s"
           % ("label", "min9_at%", "MaxDD%", "Sharpe", "W10Y*%",
              "Regmin%", "2008%", "RegImp", "08Imp", "VETO", "SURVIVE"))
    print(hdr)
    print("-" * len(hdr))

    # Anchor row first
    print("%-28s | %+7.2f%% | %+7.2f%% | %7.3f | %+7.2f%% | %+7.2f%% | %+7.2f%% | %7s | %7s | %6s | %8s"
          % ("B3a_k365 (anchor)",
             anchor_row["min9_at"] * 100, anchor_row["MaxDD_FULL"] * 100,
             anchor_row["Sharpe_OOS"], anchor_row["Worst10Y_star_at"] * 100,
             anchor_row["Regime_min_at"] * 100, anchor_row["stress_2008_at"] * 100,
             "---", "---", "---", "---"))

    for label, row in results.items():
        print("%-28s | %+7.2f%% | %+7.2f%% | %7.3f | %+7.2f%% | %+7.2f%% | %+7.2f%% | %+6.2fpp | %+6.2fpp | %6s | %8s"
              % (label,
                 row["min9_at"] * 100, row["MaxDD_FULL"] * 100,
                 row["Sharpe_OOS"], row["Worst10Y_star_at"] * 100,
                 row["Regime_min_at"] * 100, row["stress_2008_at"] * 100,
                 row["surv_regmin_imp_pp"], row["surv_stress08_imp_pp"],
                 "VETO" if row["VETO"] else "no",
                 "SURVIVE" if row["SURVIVE"] else "NO"))

    # Survival summary
    survivors = [lbl for lbl, row in results.items() if row["SURVIVE"]]
    print("\nSURVIVORS: %d of %d" % (len(survivors), len(results)))
    for lbl in survivors:
        row = results[lbl]
        print("  %-28s min9=%+.2f%% Regmin_imp=%+.2fpp 2008_imp=%+.2fpp"
              % (lbl, row["min9_at"] * 100,
                 row["surv_regmin_imp_pp"], row["surv_stress08_imp_pp"]))
    if not survivors:
        print("  (none -- all configurations failed survival gate)")

    # =========================================================================
    # CSV output
    # =========================================================================
    all_rows = []

    # Anchor row
    anchor_out = dict(anchor_row)
    anchor_out["condition"] = "anchor"
    anchor_out["s_def"]     = 1.0
    anchor_out["SURVIVE"]   = -1  # marker
    anchor_out["surv_regmin_imp_pp"]   = 0.0
    anchor_out["surv_stress08_imp_pp"] = 0.0
    anchor_out["surv_maxdd_imp_pp"]    = 0.0
    anchor_out["surv_min9_delta_pp"]   = 0.0
    anchor_out["surv_reasons"]         = ""
    all_rows.append(anchor_out)

    for label, row in results.items():
        all_rows.append(row)

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combine_g2_regimelev_20260615.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("\nSaved CSV: %s  (%d rows)" % (csv_path, len(all_rows)))

    # =========================================================================
    # RETURN BLOCK
    # =========================================================================
    def _rb(row):
        return {
            "min9_at":          round(float(row["min9_at"]),          6),
            "CAGR_IS_at":       round(float(row["CAGR_IS_at"]),       6),
            "CAGR_OOS_at":      round(float(row["CAGR_OOS_at"]),      6),
            "gap_pp":           round(float(row["gap_pp"]),            4),
            "Sharpe_OOS":       round(float(row["Sharpe_OOS"]),        4),
            "MaxDD_FULL":       round(float(row["MaxDD_FULL"]),        6),
            "Worst10Y_star_at": round(float(row["Worst10Y_star_at"]),  6),
            "P10_5Y_at":        round(float(row["P10_5Y_at"]),         6),
            "Worst5Y_at":       round(float(row["Worst5Y_at"]),        6),
            "Trades_yr":        round(float(row["Trades_yr"]),         2),
            "Regime_min_at":    round(float(row["Regime_min_at"]),     6),
            "stress_2008_at":   round(float(row["stress_2008_at"]),    6),
            "VETO":             int(row["VETO"]),
            "SURVIVE":          int(row["SURVIVE"]) if "SURVIVE" in row else -1,
            "surv_regmin_imp_pp":   float(row.get("surv_regmin_imp_pp", 0)),
            "surv_stress08_imp_pp": float(row.get("surv_stress08_imp_pp", 0)),
            "surv_maxdd_imp_pp":    float(row.get("surv_maxdd_imp_pp", 0)),
            "surv_min9_delta_pp":   float(row.get("surv_min9_delta_pp", 0)),
        }

    block = {
        "meta": {
            "B3a_k365_v7_map":   B3A_V7_MAP,
            "B3a_lev_scale":     B3A_LEV_SCALE,
            "B3a_excess_extra":  B3A_EXCESS,
            "vol_median_full_sample": round(full_vol_med, 6),
            "vol_median_is_only":     round(vol_med_is,  6),
            "bear_days_pct":          round(100.0 * bear_days / n, 2),
            "highvol_days_pct":       round(100.0 * hv_days / n,   2),
            "bear_or_hv_days_pct":    round(100.0 * bear_or_hv.sum() / n, 2),
            "n_out_bondoff": n_out_bondoff,
        },
        "sanity_anchor": _rb(anchor_row),
        "sanity_passed":  sanity_ok,
        "g2_configs": {lbl: _rb(row) for lbl, row in results.items()},
        "survivors": survivors,
    }

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
