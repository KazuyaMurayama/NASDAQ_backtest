# -*- coding: utf-8 -*-
"""
src/audit/qc_g5_independent_20260615.py
=========================================
Independent QC verification of B3a+G5_vix_hard metrics.

PURPOSE
-------
Independently reproduce the key metrics for B3a+G5_vix_hard reported in
combine_g5_defoverlay_20260615.csv / combine_phase2_fullgate_20260615.csv,
WITHOUT importing or calling any function from the following modules:
  - combine_g5_defoverlay_20260615.py
  - combine_phase2_fullgate_20260615.py
  - scorecard_g5_20260615.py

Independence boundary: the G5 overlay logic (vix_mom21 4-quantile bucketing,
publication lag, defensive multiplier application to lev_mod) is re-implemented
from scratch based solely on:
  1. The docstrings and published target values in the result files above.
  2. The signal module source (src/signals/quantize.py, src/signals/timing.py)
     which are infrastructure, not the audit being verified.
  3. The B3a base builders (k365_recost_20260612.py / leverup_b1c1_20260612.py)
     which are the FOUNDATION under the G5 layer -- they are NOT part of G5.

DESIGN CHOICES
--------------
Quantile boundary method: IS-ONLY (causal / expanding alternative)
  The combine_g5 docstring says "full-sample cut is acceptable for a slowly-varying
  feature; consistent with how G3 native evaluation was done." This QC script uses
  IS-only quantiles (data up to IS_END = 2021-05-07) to determine the quartile
  boundaries, then applies those fixed boundaries throughout the full sample.

  Rationale for IS-only: The strategy is evaluated on OOS data (2021-05-08 onwards).
  Using OOS data to define quantile cuts is a mild forward-looking bias. IS-only
  boundaries are causally sound and more conservative. This is the KEY INDEPENDENT
  DESIGN CHOICE -- any systematic difference from combine_g5 results will reveal
  whether that module used full-sample or IS-only quantiles.

  Expected: if combine_g5 used full-sample quantiles -> IS-only QC will show
  slightly different results; direction of difference is informative.

Publication lag: daily -> shift by +1 business day (same as signals/timing.py).

Defensive map (hard): {Q0:1.00, Q1:1.00, Q2:0.92, Q3:0.85}
  Q0 = lowest VIX momentum (most risk-off) -> no trim (1.00)
  Q3 = highest VIX momentum (most risk-on/overheating) -> strongest trim (0.85)

Multiplier application: lev_mod_g5 = lev_raw_masked * mult_v7 * def_mult
  Applied BEFORE the V7_DELAY shift (same as combine_g5 design).

SANITY CHECK
------------
B3a素地サニティ: neutral map (all 1.0) must reproduce:
  min(IS,OOS) after-tax ~ +20.98%  (tol +/-0.05pp)
  MaxDD_FULL ~ -38.20%            (tol +/-0.10pp)

TARGET VALUES (B3a+G5_vix_hard, after-tax 9-metric standard)
-------------------------------------------------------------
  min9           : +20.73%
  CAGR_IS        : +22.20%  (from combine_g5 CSV: CAGR_IS_at=0.222015)
  Sharpe_OOS     :  0.928
  MaxDD_FULL     : -35.93%
  Worst10Y_star  : +13.74%
  P10_5Y         :  +8.26%
  Worst5Y        :  -0.07%
  Trades_yr      :  52.9

CONVERGENCE THRESHOLDS
-----------------------
  CAGR-like    : 0.3pp
  MaxDD        : 1.0pp
  Sharpe       : 0.02
  Trades/yr    : 1.0

OUTPUTS
-------
  src/audit/qc_g5_independent_20260615.py  (this file)
  audit_results/qc_g5_independent_20260615.csv

Authors: QC sub-agent (independent verifier)  2026-06-16
ASCII prints / cp932 compatible.
"""

from __future__ import annotations

import json
import os
import sys
import types

# ---- multitasking stub (yfinance dependency) --------------------------------
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

# ---- Allowed imports: infrastructure / base builders (NOT G5 modules) -------
import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START

# B3a base -- allowed (foundation under G5, not the G5 overlay being verified)
from src.audit.k365_recost_20260612 import (
    EXCESS_EXTRA_K365_CENTRE, EXCESS_EXTRA_STORE,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_on_base_c1,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax,
    _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
    FEE_GOLD, FEE_BOND,
)
from src.audit.lu_cfd_recost_20260611 import (
    SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    LU2_SCALE, _build_v7_mult_custom,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    DELAY as V7_DELAY, DH_PER_UNIT, NAV_FLOOR,
)

# ---- Signal infrastructure (allowed: base library, not G5 module) -----------
# We reimplement quantization and lag INDEPENDENTLY below (do NOT use signals modules)
# But we use the same logic as signals/quantize.py and signals/timing.py,
# re-coded here from specification.

# ============================================================================
# Constants
# ============================================================================

B3A_V7_MAP   = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE = 1.15
B3A_EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# Known B3a_k365 ground-truth
B3A_KNOWN_MIN9   = 0.2098   # +20.98%
B3A_KNOWN_MAXDD  = -0.3820  # -38.20%
B3A_KNOWN_SHARPE = 0.904

SANITY_TOL_MIN9  = 0.0005   # +-0.05pp
SANITY_TOL_MAXDD = 0.0010   # +-0.10pp

# G5 vix_hard target values (from combine_g5_defoverlay_20260615.csv)
TARGET_MIN9      = 0.20658  # min9_at = OOS = +20.66% (reported 20.73% in scorecard)
TARGET_CAGR_IS   = 0.222015
TARGET_CAGR_OOS  = 0.206581  # from combine_g5 CSV
TARGET_SHARPE    = 0.927984
TARGET_MAXDD     = -0.359318
TARGET_W10Y      = 0.137352
TARGET_P10_5Y    = 0.082592
TARGET_WORST5Y   = -0.000712
TARGET_TRADES    = 52.948895

# Threshold for convergence
THR_CAGR   = 0.003   # 0.3pp
THR_MAXDD  = 0.010   # 1.0pp
THR_SHARPE = 0.02
THR_TRADES = 1.0

# G5 HARD defensive map: {Q0:1.00, Q1:1.00, Q2:0.92, Q3:0.85}
HARD_MAP = {0: 1.00, 1: 1.00, 2: 0.92, 3: 0.85}
NEUTRAL_MAP = {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00}

MACRO_FEATURES_PATH = os.path.join(_REPO_DIR, "data", "macro_features.csv")

# ============================================================================
# INDEPENDENT re-implementation of G5 signal processing
# ============================================================================

def _qc_quantile_cut_is_only(signal: pd.Series, is_end: pd.Timestamp) -> pd.Series:
    """Independent IS-only 4-quantile bucketing.

    Design choice (documented in module docstring):
    - Compute quartile boundaries ONLY from IS data (<= is_end)
    - Apply fixed IS boundaries to the full sample (IS + OOS)
    - This is causally sound: OOS evaluation never uses OOS data for binning
    - Returns Series of int (0,1,2,3) aligned to signal.index, NaN where signal is NaN

    Uses pandas.qcut on IS data to get bin edges, then pd.cut on full sample.
    This matches the semantics of quantile_cut(window=None) EXCEPT we restrict
    quantile edge computation to IS period only.
    """
    is_mask = signal.index <= is_end
    is_data = signal[is_mask].dropna()

    if len(is_data) < 8:
        raise ValueError("Insufficient IS data for quartile computation: N=%d" % len(is_data))

    # Get quartile bin edges from IS period
    _, bins = pd.qcut(is_data, 4, retbins=True, duplicates='drop')
    # Extend edges to -inf / +inf to capture all OOS values
    bins[0]  = -np.inf
    bins[-1] = +np.inf

    if len(bins) < 5:
        # Degenerate: fewer than 4 unique quantile boundaries
        print("  WARNING: IS-only quantile produced only %d bins (degenerate distribution)" % (len(bins)-1))
        # Fall back to rank-based mapping
        rank_pct = signal.rank(pct=True)
        out = (rank_pct * 4).clip(0, 3.9999).astype('Int8')
        return out.where(signal.notna())

    # Apply fixed IS boundaries to full sample
    out = pd.cut(signal, bins=bins, labels=False, include_lowest=True)
    # Convert to Int8
    out_int = pd.Series(pd.NA, index=signal.index, dtype='Int8')
    valid = out.notna()
    if valid.any():
        out_int.loc[valid] = out[valid].astype('int64').astype('Int8')
    return out_int.where(signal.notna())


def _qc_apply_daily_lag(s: pd.Series) -> pd.Series:
    """Independent daily publication lag: shift index by +1 business day."""
    from pandas.tseries.offsets import BusinessDay
    shifted_idx = s.index + BusinessDay(1)
    return pd.Series(s.values, index=shifted_idx)


def _qc_build_def_mult_arr_is_only(
    signal_col: str,
    def_map: dict,
    dates_dt: pd.DatetimeIndex,
    macro_df: pd.DataFrame,
) -> np.ndarray:
    """Independently build defensive multiplier array using IS-only quantile boundaries.

    Steps:
      1. Load vix_mom21 from macro_df
      2. IS-only 4-quantile bucketing (bin edges from IS data <= IS_END)
      3. Apply daily publication lag (+1 BD)
      4. Align/ffill to strategy dates
      5. Map Q0/Q1/Q2/Q3 -> def_map multiplier

    This is the INDEPENDENT re-implementation. It does NOT call _build_def_mult_arr
    from combine_g5_defoverlay_20260615.py.
    """
    raw_signal = macro_df[signal_col].dropna()
    raw_signal.index = pd.DatetimeIndex(pd.to_datetime(raw_signal.index))

    # Step 2: IS-only quantile bucketing
    sig_q = _qc_quantile_cut_is_only(raw_signal, IS_END)
    sig_q = sig_q.dropna()

    # Step 3: Daily publication lag
    sig_lagged = _qc_apply_daily_lag(sig_q)
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]

    # Step 4: Align to strategy dates, forward-fill
    aligned = sig_lagged.reindex(dates_dt).ffill()

    # Step 5: Map to multiplier
    def _to_mult(s):
        if pd.isna(s):
            return 1.0
        return def_map.get(int(s), 1.0)

    mult = aligned.map(_to_mult)
    arr = np.asarray(mult.fillna(1.0).values, dtype=float)
    arr = np.clip(arr, 0.0, 2.0)
    return arr


# ============================================================================
# INDEPENDENT NAV builder for B3a + G5 overlay
# ============================================================================

def _qc_build_nav_b3a_g5(
    close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
    lev_raw_masked, wn, wg, wb, mult_v7,
    def_mult_arr,
    excess_extra=B3A_EXCESS_EXTRA,
):
    """Independent reimplementation of B3a TQQQ base NAV with G5 overlay.

    Key difference from combine_g5_defoverlay._build_nav_b3a_g5:
    - We reimplement the NAV computation from the k365_recost blueprint,
      but do NOT import it from combine_g5.
    - def_mult_arr is applied natively as:
        lev_mod = lev_raw_masked * mult_v7 * def_mult_arr
    """
    from src.audit.strategy_runners import (
        _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
        _compute_dhw1_trades_per_year,
    )
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

    idx = dates.index
    n   = len(close)
    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3  = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    # --- G5 native integration ---
    lev_raw_arr = np.asarray(lev_raw_masked, float)
    mult_v7_arr = np.asarray(mult_v7, float)
    def_arr     = np.asarray(def_mult_arr, float)
    lev_mod = lev_raw_arr * mult_v7_arr * def_arr   # G5 applied here

    L    = pd.Series(lev_mod * 3.0, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(V7_DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(V7_DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(V7_DELAY).fillna(0).values
    sofr_np = np.asarray(sofr_daily, float)

    # TQQQ financing leg
    borrow  = np.maximum(L - 1.0, 0.0) * (sofr_np + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # >3x excess penalty (k365 centre)
    excess_lev = np.maximum(L - LEV_CAP, 0.0)
    penalty    = wn_s * excess_lev * float(excess_extra) / TRADING_DAYS
    daily      = daily - penalty
    excess_days = int(np.sum(excess_lev > 1e-9))

    # Turnover drag
    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily   = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    ter_drag = (np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
                + np.asarray(wb, float) * _TER_TMF_EXTRA_DAILY)
    tpy      = _compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / 252.0
    r_sim    = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj    = r_sim - ter_drag - etf_daily
    nav_adj  = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )
    return nav_adj, tpy, excess_days


def _qc_build_tqqq_base_g5(shared, date_index, def_mult_arr,
                             v7_map=B3A_V7_MAP, lev_scale=B3A_LEV_SCALE,
                             excess_extra=B3A_EXCESS_EXTRA):
    """Build B3a TQQQ base NAV with G5 overlay (independent reimplementation)."""
    a   = shared["assets"]
    close     = a["close"]
    dates     = a["dates"]
    sofr      = np.asarray(a["sofr"], float)
    gold_2x   = a["gold_2x"]
    bond_3x   = a["bond_3x"]
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn = np.asarray(shared["wn"], float)
    wg = np.asarray(shared["wg"], float)
    wb = np.asarray(shared["wb"], float)

    mult_v7 = _build_v7_mult_custom(date_index, v7_map)
    mult_v7 = mult_v7 * float(lev_scale)

    nav_dt, tpy, excess_days = _qc_build_nav_b3a_g5(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7,
        def_mult_arr, excess_extra=excess_extra)
    r_base = nav_dt.pct_change().fillna(0).values
    return nav_dt, r_base, tpy, excess_days


def _qc_build_full_b3a_g5(shared, dates_dt, n_years,
                            ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
                            bond_on, sofr_arr, def_mult_arr):
    """Build full B3a_k365 + G5 NAV (TQQQ base + C1 OUT fill)."""
    base_nav, r_base, tpy_b, exc = _qc_build_tqqq_base_g5(
        shared, dates_dt, def_mult_arr)
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on,
        sofr_arr, dates_dt, tpy_b, n_years)
    return nav_dt, r, tpy, exc


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 100)
    print("QC G5 INDEPENDENT VERIFICATION  2026-06-16")
    print("Target: B3a+G5_vix_hard")
    print("Independence: G5 overlay re-implemented WITHOUT importing combine_g5/phase2/scorecard_g5")
    print("Quantile method: IS-ONLY (boundaries from IS data <= 2021-05-07)")
    print("Publication lag: daily (+1 BD)")
    print("Defensive map (hard): {Q0:1.00, Q1:1.00, Q2:0.92, Q3:0.85}")
    print("=" * 100)

    # ========================================================================
    # Step 1: Load shared DH-W1 assets
    # ========================================================================
    print("\n[Step 1] Loading DH-W1 shared assets ...")
    sr._load_dhw1_shared()
    shared   = sr._DHW1_SHARED
    a        = shared["assets"]
    mask     = np.asarray(shared["mask"], dtype=float)
    dates    = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)
    print("  Total days: %d  (%.1f years)" % (n, n_years))

    # ========================================================================
    # Step 2: Gold/Bond 1x legs
    # ========================================================================
    print("\n[Step 2] Building Gold/Bond 1x legs ...")
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
    bond_m252    = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on      = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr     = np.asarray(a["sofr"], float)
    print("  fund_active days: %d  bond_on days: %d" % (int(fund_active.sum()), int(bond_on.sum())))

    # ========================================================================
    # Step 3: Load macro_features.csv and inspect vix_mom21
    # ========================================================================
    print("\n[Step 3] Loading macro_features.csv ...")
    macro_df = pd.read_csv(MACRO_FEATURES_PATH, index_col=0, parse_dates=True)
    print("  Columns: %d  vix_mom21 present: %s" % (
        len(macro_df.columns), "YES" if "vix_mom21" in macro_df.columns else "NO"))
    if "vix_mom21" not in macro_df.columns:
        print("  ABORT: vix_mom21 not in macro_features.csv")
        sys.exit(1)
    vix = macro_df["vix_mom21"].dropna()
    vix.index = pd.DatetimeIndex(pd.to_datetime(vix.index))
    vix_is = vix[vix.index <= IS_END]
    print("  vix_mom21 total: N=%d  IS: N=%d  OOS: N=%d" % (
        len(vix), len(vix_is), len(vix[vix.index > IS_END])))
    print("  vix_mom21 IS stats: mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (
        float(vix_is.mean()), float(vix_is.std()),
        float(vix_is.min()), float(vix_is.max())))

    # Show IS-only quartile boundaries
    _, is_bins = pd.qcut(vix_is, 4, retbins=True, duplicates='drop')
    print("  IS-only quartile boundaries: [%.4f, %.4f, %.4f, %.4f, %.4f]" % tuple(is_bins))

    # Also compute full-sample quartile boundaries for comparison
    _, fs_bins = pd.qcut(vix, 4, retbins=True, duplicates='drop')
    print("  Full-sample quartile boundaries: [%.4f, %.4f, %.4f, %.4f, %.4f]" % tuple(fs_bins))
    print("  Q2 boundary difference (IS vs full-sample): %.4f pp" % (is_bins[2] - fs_bins[2]))
    print("  -> If IS != full-sample: combine_g5 full-sample choice vs QC IS-only will diverge")

    # ========================================================================
    # Step 4: Sanity check -- neutral map must reproduce B3a素地
    # ========================================================================
    print("\n" + "=" * 100)
    print("SANITY CHECK: neutral map (all 1.0) must reproduce B3a_k365")
    print("Expected: min9 ~ +%.2f%%  MaxDD ~ %.2f%%  (tol: min9 +/-%.2fpp  MaxDD +/-%.2fpp)" % (
        B3A_KNOWN_MIN9 * 100, B3A_KNOWN_MAXDD * 100,
        SANITY_TOL_MIN9 * 100, SANITY_TOL_MAXDD * 100))
    print("=" * 100)

    neutral_def = np.ones(n, dtype=float)
    nav_neutral, r_neutral, tpy_neutral, exc_neutral = _qc_build_full_b3a_g5(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr, def_mult_arr=neutral_def)

    pre_neutral = compute_10metrics(nav_neutral, tpy_neutral)
    aft_neutral = _apply_aftertax(pre_neutral)
    min9_neutral  = min(aft_neutral["CAGR_IS"], aft_neutral["CAGR_OOS"])
    maxdd_neutral = pre_neutral["MaxDD_FULL"]
    sharpe_neutral = pre_neutral["Sharpe_OOS"]

    print("  QC neutral: min9=%+.4f%%  MaxDD=%+.4f%%  Sharpe=%.4f" % (
        min9_neutral * 100, maxdd_neutral * 100, sharpe_neutral))

    ok_min9  = abs(min9_neutral  - B3A_KNOWN_MIN9)  <= SANITY_TOL_MIN9
    ok_maxdd = abs(maxdd_neutral - B3A_KNOWN_MAXDD) <= SANITY_TOL_MAXDD

    print("  min9  delta: %+.4fpp -> %s" % (
        (min9_neutral - B3A_KNOWN_MIN9) * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD delta: %+.4fpp -> %s" % (
        (maxdd_neutral - B3A_KNOWN_MAXDD) * 100, "OK" if ok_maxdd else "FAIL"))

    sanity_passed = ok_min9 and ok_maxdd
    if not sanity_passed:
        print("\n[HALT] SANITY CHECK FAILED -- independent B3a base builder has a bug.")
        print("  Cannot proceed to G5 verification.")
        sys.exit(1)
    print("  SANITY CHECK PASSED. Proceeding to G5 verification.")

    # ========================================================================
    # Step 5: Build G5_vix_hard defensive multiplier (INDEPENDENT)
    # ========================================================================
    print("\n" + "=" * 100)
    print("STEP 5: Building G5_vix_hard defensive multiplier (IS-ONLY quantile boundaries)")
    print("=" * 100)

    def_arr_isonly = _qc_build_def_mult_arr_is_only(
        "vix_mom21", HARD_MAP, dates_dt, macro_df)

    # Distribution of def_mult values
    unique_vals, counts = np.unique(def_arr_isonly, return_counts=True)
    print("  def_mult distribution (IS-only boundaries):")
    for v, c in zip(unique_vals, counts):
        print("    mult=%.2f  days=%d  (%.1f%%)" % (v, c, 100.0 * c / n))

    # Count days in each quartile tier
    q0_days = int(np.sum(def_arr_isonly == 1.00))
    q2_days = int(np.sum(def_arr_isonly == 0.92))
    q3_days = int(np.sum(def_arr_isonly == 0.85))
    print("  Q0+Q1 (1.00) days: %d  |  Q2 (0.92) days: %d  |  Q3 (0.85) days: %d" % (
        q0_days, q2_days, q3_days))

    # ========================================================================
    # Step 6: Build G5_vix_hard NAV and compute metrics
    # ========================================================================
    print("\n[Step 6] Building B3a+G5_vix_hard NAV (IS-only quantile) ...")
    nav_g5_is, r_g5_is, tpy_g5_is, exc_g5_is = _qc_build_full_b3a_g5(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr, def_mult_arr=def_arr_isonly)

    pre_g5_is = compute_10metrics(nav_g5_is, tpy_g5_is)
    aft_g5_is = _apply_aftertax(pre_g5_is)
    cy_g5_is  = _calendar_year_returns(nav_g5_is)
    min9_g5_is = min(aft_g5_is["CAGR_IS"], aft_g5_is["CAGR_OOS"])

    print("  B3a+G5_vix_hard (IS-only Q):")
    print("    min9       = %+.4f%%" % (min9_g5_is * 100))
    print("    CAGR_IS    = %+.4f%%" % (aft_g5_is["CAGR_IS"] * 100))
    print("    CAGR_OOS   = %+.4f%%" % (aft_g5_is["CAGR_OOS"] * 100))
    print("    Sharpe_OOS = %.4f"    % pre_g5_is["Sharpe_OOS"])
    print("    MaxDD_FULL = %+.4f%%" % (pre_g5_is["MaxDD_FULL"] * 100))
    print("    Worst10Y*  = %+.4f%%" % (aft_g5_is["Worst10Y_star"] * 100))
    print("    P10_5Y     = %+.4f%%" % (aft_g5_is["P10_5Y"] * 100))
    print("    Worst5Y    = %+.4f%%" % (aft_g5_is["Worst5Y"] * 100))
    print("    Trades/yr  = %.2f"    % aft_g5_is["Trades_yr"])
    print("    Excess days = %d"     % exc_g5_is)

    # ========================================================================
    # Step 7: ALSO build with full-sample quantiles for explicit comparison
    # ========================================================================
    print("\n[Step 7] Building G5_vix_hard with FULL-SAMPLE quantile boundaries for comparison ...")

    def _build_def_mult_full_sample(signal_col, def_map, dates_dt, macro_df):
        """Full-sample quantile for comparison only."""
        raw_signal = macro_df[signal_col].dropna()
        raw_signal.index = pd.DatetimeIndex(pd.to_datetime(raw_signal.index))
        try:
            sig_q = pd.qcut(raw_signal, 4, labels=False, duplicates='drop').astype('Int8')
        except Exception:
            return np.ones(len(dates_dt), dtype=float)
        sig_q = sig_q.where(raw_signal.notna())
        from pandas.tseries.offsets import BusinessDay
        shifted_idx = raw_signal.index + BusinessDay(1)
        sig_lagged = pd.Series(sig_q.values, index=shifted_idx)
        sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
        aligned = sig_lagged.reindex(dates_dt).ffill()
        def _to_mult(s):
            if pd.isna(s): return 1.0
            return def_map.get(int(s), 1.0)
        mult = aligned.map(_to_mult)
        arr = np.asarray(mult.fillna(1.0).values, dtype=float)
        return np.clip(arr, 0.0, 2.0)

    def_arr_fs = _build_def_mult_full_sample("vix_mom21", HARD_MAP, dates_dt, macro_df)
    nav_g5_fs, r_g5_fs, tpy_g5_fs, exc_g5_fs = _qc_build_full_b3a_g5(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr, def_mult_arr=def_arr_fs)

    pre_g5_fs = compute_10metrics(nav_g5_fs, tpy_g5_fs)
    aft_g5_fs = _apply_aftertax(pre_g5_fs)
    min9_g5_fs = min(aft_g5_fs["CAGR_IS"], aft_g5_fs["CAGR_OOS"])

    print("  B3a+G5_vix_hard (FULL-SAMPLE Q):")
    print("    min9       = %+.4f%%" % (min9_g5_fs * 100))
    print("    CAGR_OOS   = %+.4f%%" % (aft_g5_fs["CAGR_OOS"] * 100))
    print("    Sharpe_OOS = %.4f"    % pre_g5_fs["Sharpe_OOS"])
    print("    MaxDD_FULL = %+.4f%%" % (pre_g5_fs["MaxDD_FULL"] * 100))
    print("    Trades/yr  = %.2f"    % aft_g5_fs["Trades_yr"])

    # ========================================================================
    # Step 8: Comparison table
    # ========================================================================
    print("\n" + "=" * 100)
    print("METRIC COMPARISON TABLE")
    print("=" * 100)
    hdr = "%-20s | %12s | %12s | %12s | %12s | %8s | %8s"
    print(hdr % ("metric", "target", "QC (IS-only)", "delta_ISonly", "QC (FS)", "delta_FS", "threshold"))
    print("-" * 100)

    targets = {
        "CAGR_OOS_at": (TARGET_CAGR_OOS, aft_g5_is["CAGR_OOS"], aft_g5_fs["CAGR_OOS"], THR_CAGR),
        "CAGR_IS_at":  (TARGET_CAGR_IS,  aft_g5_is["CAGR_IS"],  aft_g5_fs["CAGR_IS"],  THR_CAGR),
        "min9":        (TARGET_CAGR_OOS,  min9_g5_is,             min9_g5_fs,             THR_CAGR),
        "Sharpe_OOS":  (TARGET_SHARPE,    pre_g5_is["Sharpe_OOS"],pre_g5_fs["Sharpe_OOS"],THR_SHARPE),
        "MaxDD_FULL":  (TARGET_MAXDD,     pre_g5_is["MaxDD_FULL"],pre_g5_fs["MaxDD_FULL"],THR_MAXDD),
        "Worst10Y*":   (TARGET_W10Y,      aft_g5_is["Worst10Y_star"],aft_g5_fs["Worst10Y_star"],THR_CAGR),
        "P10_5Y":      (TARGET_P10_5Y,    aft_g5_is["P10_5Y"],   aft_g5_fs["P10_5Y"],    THR_CAGR),
        "Worst5Y":     (TARGET_WORST5Y,   aft_g5_is["Worst5Y"],  aft_g5_fs["Worst5Y"],   THR_CAGR),
        "Trades_yr":   (TARGET_TRADES,    aft_g5_is["Trades_yr"],aft_g5_fs["Trades_yr"], THR_TRADES),
    }

    rows_csv = []
    any_fail_is  = False
    any_fail_fs  = False

    for metric, (tgt, v_is, v_fs, thr) in targets.items():
        d_is = v_is - tgt
        d_fs = v_fs - tgt
        flag_is = "OK" if abs(d_is) <= thr else "MISS"
        flag_fs = "OK" if abs(d_fs) <= thr else "MISS"
        if flag_is == "MISS": any_fail_is = True
        if flag_fs == "MISS": any_fail_fs = True
        fmt = "%-20s | %12.5f | %12.5f | %+12.5f | %12.5f | %+8.5f | %8.5f  %s / %s"
        print(fmt % (metric, tgt, v_is, d_is, v_fs, d_fs, thr, flag_is, flag_fs))
        rows_csv.append({
            "metric": metric,
            "target":       round(float(tgt), 8),
            "qc_is_only":   round(float(v_is), 8),
            "delta_is_only":round(float(d_is), 8),
            "flag_is_only": flag_is,
            "qc_fs":        round(float(v_fs), 8),
            "delta_fs":     round(float(d_fs), 8),
            "flag_fs":      flag_fs,
            "threshold":    round(float(thr), 8),
        })

    # ========================================================================
    # Step 9: Divergence analysis (IS-only vs full-sample vs target)
    # ========================================================================
    print("\n" + "=" * 100)
    print("DIVERGENCE ANALYSIS: IS-only vs Full-Sample vs Target")
    print("=" * 100)

    print("\n  Target comes from combine_g5 (which uses full-sample quantiles per its docstring)")
    print("  This QC uses IS-only quantiles (causal / forward-looking-free)")
    print()
    print("  min9   : IS-only=%+.4f%%  FS=%+.4f%%  target=%+.4f%%  (FS closer to target: %s)" % (
        min9_g5_is * 100, min9_g5_fs * 100, TARGET_CAGR_OOS * 100,
        "YES" if abs(min9_g5_fs - TARGET_CAGR_OOS) < abs(min9_g5_is - TARGET_CAGR_OOS) else "NO"))
    print("  MaxDD  : IS-only=%+.4f%%  FS=%+.4f%%  target=%+.4f%%  (FS closer to target: %s)" % (
        pre_g5_is["MaxDD_FULL"] * 100, pre_g5_fs["MaxDD_FULL"] * 100, TARGET_MAXDD * 100,
        "YES" if abs(pre_g5_fs["MaxDD_FULL"] - TARGET_MAXDD) < abs(pre_g5_is["MaxDD_FULL"] - TARGET_MAXDD) else "NO"))
    print("  Trades : IS-only=%.2f  FS=%.2f  target=%.2f  (FS closer to target: %s)" % (
        aft_g5_is["Trades_yr"], aft_g5_fs["Trades_yr"], TARGET_TRADES,
        "YES" if abs(aft_g5_fs["Trades_yr"] - TARGET_TRADES) < abs(aft_g5_is["Trades_yr"] - TARGET_TRADES) else "NO"))

    delta_min9_is_vs_fs = (min9_g5_is - min9_g5_fs) * 100
    delta_maxdd_is_vs_fs = (pre_g5_is["MaxDD_FULL"] - pre_g5_fs["MaxDD_FULL"]) * 100
    print("\n  IS-only vs FS delta: min9 %+.3fpp  MaxDD %+.3fpp  Trades %.2f" % (
        delta_min9_is_vs_fs, delta_maxdd_is_vs_fs,
        aft_g5_is["Trades_yr"] - aft_g5_fs["Trades_yr"]))

    if abs(delta_min9_is_vs_fs) < 0.2 and abs(delta_maxdd_is_vs_fs) < 0.5:
        print("  -> Quantile method (IS vs FS) makes minimal difference (<0.2pp min9 / <0.5pp MaxDD)")
        print("  -> combine_g5 quantile choice does NOT materially affect results")
        quantile_bias_flag = "NEGLIGIBLE"
    elif abs(delta_min9_is_vs_fs) < 1.0:
        print("  -> Moderate quantile boundary sensitivity. Check if FS-quantile provides lookahead advantage.")
        quantile_bias_flag = "MODERATE"
    else:
        print("  -> LARGE divergence between IS-only and FS quantiles.")
        print("  -> Full-sample quantile boundary provides material look-ahead advantage in OOS period.")
        print("  -> FLAG: potential look-ahead contamination in combine_g5 results.")
        quantile_bias_flag = "LARGE_LOOKAHEAD_RISK"

    # ========================================================================
    # Step 10: Final judgment
    # ========================================================================
    print("\n" + "=" * 100)
    print("FINAL JUDGMENT")
    print("=" * 100)

    fs_match = not any_fail_fs
    is_match = not any_fail_is

    print("\n  Full-sample (replicates combine_g5 design): all metrics within threshold: %s" % (
        "YES" if fs_match else "NO"))
    print("  IS-only (causal independent): all metrics within threshold: %s" % (
        "YES" if is_match else "NO"))
    print("  Quantile boundary sensitivity: %s" % quantile_bias_flag)

    if fs_match:
        print("\n  RESULT: combine_g5 G5_vix_hard metrics independently CONFIRMED (full-sample replication).")
        print("  The full-sample quantile choice is documented in combine_g5 docstring and is")
        print("  explicitly acknowledged; no hidden lookahead.")
        verdict = "CONFIRMED"
    elif is_match:
        print("\n  RESULT: G5_vix_hard metrics confirmed within threshold using IS-only quantiles.")
        print("  combine_g5 may use slightly different (full-sample) boundaries; difference is within thr.")
        verdict = "CONFIRMED_ISONLY"
    else:
        print("\n  RESULT: METRIC MISMATCH -- at least one metric exceeds convergence threshold.")
        print("  Potential integration bug in combine_g5.")
        verdict = "MISMATCH_FLAG"

    print("\n  Sanity (B3a neutral): %s" % ("PASS" if sanity_passed else "FAIL"))

    # ========================================================================
    # CSV output
    # ========================================================================
    print("\n[Saving CSV] ...")
    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "qc_g5_independent_20260615.csv")
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False, float_format="%.8f")
    print("Saved: %s  (%d rows)" % (csv_path, len(rows_csv)))

    # ========================================================================
    # RETURN_BLOCK
    # ========================================================================
    block = {
        "meta": {
            "script": "qc_g5_independent_20260615.py",
            "date": "2026-06-16",
            "quantile_method_primary": "IS-ONLY (boundaries from IS data <= 2021-05-07)",
            "quantile_method_secondary": "FULL-SAMPLE (for comparison only)",
            "publication_lag": "daily (+1 BD)",
            "defensive_map": "hard {Q0:1.00, Q1:1.00, Q2:0.92, Q3:0.85}",
            "sanity_passed": int(sanity_passed),
            "B3a_neutral_min9_pct": round(min9_neutral * 100, 4),
            "B3a_neutral_maxdd_pct": round(maxdd_neutral * 100, 4),
            "IS_quartile_bins": [round(float(b), 4) for b in is_bins],
            "FS_quartile_bins": [round(float(b), 4) for b in fs_bins],
            "quantile_bias_flag": quantile_bias_flag,
            "verdict": verdict,
        },
        "is_only_metrics": {
            "CAGR_IS_at":    round(float(aft_g5_is["CAGR_IS"]), 6),
            "CAGR_OOS_at":   round(float(aft_g5_is["CAGR_OOS"]), 6),
            "min9":          round(float(min9_g5_is), 6),
            "Sharpe_OOS":    round(float(pre_g5_is["Sharpe_OOS"]), 6),
            "MaxDD_FULL":    round(float(pre_g5_is["MaxDD_FULL"]), 6),
            "Worst10Y_star": round(float(aft_g5_is["Worst10Y_star"]), 6),
            "P10_5Y":        round(float(aft_g5_is["P10_5Y"]), 6),
            "Worst5Y":       round(float(aft_g5_is["Worst5Y"]), 6),
            "Trades_yr":     round(float(aft_g5_is["Trades_yr"]), 2),
            "excess_days":   exc_g5_is,
        },
        "fs_metrics": {
            "CAGR_IS_at":    round(float(aft_g5_fs["CAGR_IS"]), 6),
            "CAGR_OOS_at":   round(float(aft_g5_fs["CAGR_OOS"]), 6),
            "min9":          round(float(min9_g5_fs), 6),
            "Sharpe_OOS":    round(float(pre_g5_fs["Sharpe_OOS"]), 6),
            "MaxDD_FULL":    round(float(pre_g5_fs["MaxDD_FULL"]), 6),
            "Worst10Y_star": round(float(aft_g5_fs["Worst10Y_star"]), 6),
            "P10_5Y":        round(float(aft_g5_fs["P10_5Y"]), 6),
            "Worst5Y":       round(float(aft_g5_fs["Worst5Y"]), 6),
            "Trades_yr":     round(float(aft_g5_fs["Trades_yr"]), 2),
            "excess_days":   exc_g5_fs,
        },
        "targets": {
            "CAGR_OOS_at": TARGET_CAGR_OOS,
            "CAGR_IS_at":  TARGET_CAGR_IS,
            "Sharpe_OOS":  TARGET_SHARPE,
            "MaxDD_FULL":  TARGET_MAXDD,
            "Worst10Y_star": TARGET_W10Y,
            "P10_5Y":      TARGET_P10_5Y,
            "Worst5Y":     TARGET_WORST5Y,
            "Trades_yr":   TARGET_TRADES,
        },
        "divergence": {
            "delta_min9_IS_vs_FS_pp": round(delta_min9_is_vs_fs, 4),
            "delta_maxdd_IS_vs_FS_pp": round(delta_maxdd_is_vs_fs, 4),
            "delta_trades_IS_vs_FS": round(float(aft_g5_is["Trades_yr"] - aft_g5_fs["Trades_yr"]), 2),
            "FS_closer_to_target_min9": int(abs(min9_g5_fs - TARGET_CAGR_OOS) < abs(min9_g5_is - TARGET_CAGR_OOS)),
            "FS_closer_to_target_MaxDD": int(abs(pre_g5_fs["MaxDD_FULL"] - TARGET_MAXDD) < abs(pre_g5_is["MaxDD_FULL"] - TARGET_MAXDD)),
        },
    }

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=False))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
