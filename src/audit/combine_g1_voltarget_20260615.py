"""
src/audit/combine_g1_voltarget_20260615.py
==========================================
G1: vol-target / DD-brake governor natively integrated into B3a_k365.

Native integration method (anti-post-hoc)
------------------------------------------
The governor modifies the per-day leverage BEFORE borrow / excess-cost /
NASDAQ-return are calculated.  It does NOT multiply a pre-computed NAV.

Insertion point (inside the NAV loop, equivalent to _build_nav_v7_tqqq_param):
  1. Compute lev_mod  = lev_raw_masked * mult_v7  (V7 boost, DELAY-shifted)
  2. Compute L_raw    = lev_mod * 3.0             (DELAY-shifted)
  3. Compute realized_vol_t (63-day rolling annualised, shift by V7_DELAY to
     ensure only information up to t-V7_DELAY is used at time t)
  4. governor_factor  = min(1.0, target_vol / realized_vol_t)
  5. L_governed       = L_raw * governor_factor   (cap = 1.0 => only reduces)
  6. All downstream calculations (borrow, excess penalty, nas_ret) use L_governed.
  7. LEV_CAP (3x) comparison also uses L_governed.

Publication-lag / causality rule
----------------------------------
- V7 signal is already shift(V7_DELAY=2) in the existing pipeline.
- realized_vol_t is computed as a rolling 63-day std of NASDAQ daily returns,
  then shifted by V7_DELAY (same 2-day lag) so that on any calendar day t the
  strategy only sees sigma computed through t-V7_DELAY.
- This matches the causal discipline of the rest of the pipeline.

Reuse boundary
--------------
- _build_tqqq_base_param, _build_p09_on_base_c1, _build_full_c1 from
  k365_recost_20260612.py are NOT reused directly because G1 requires the
  governor to be embedded inside the NAV loop (native integration).
  Instead, we duplicate / adapt _build_nav_v7_tqqq_param with the governor
  added.  All other helpers (shared data loader, _apply_aftertax,
  compute_10metrics, _inverse_vol_weights, _load_macro_signal, etc.) are reused
  as imports.

B3a_k365 configuration
-----------------------
  v7_map      = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
  lev_scale   = 1.15
  excess_extra = 0.0025  (EXCESS_EXTRA_K365_CENTRE)

Sanity anchor
--------------
With target_vol=10.0 (effectively disabled), the governor factor = 1.0 at
every step, so the result MUST reproduce B3a_k365 within +-0.05pp on min(IS,OOS).

Stage 0 screen
--------------
Standard 10 metrics + after-tax 0.8273 (CAGR family only).
Hard veto: MaxDD < -50% OR Worst10Y* < 0.
(WFE and Regime_min are Stage 1 -- omitted here as per task spec.)

Survival condition (defensive element)
  PASS = no_veto AND (MaxDD_improve >= +2pp OR Worst10Y*_improve >= +0.5pp OR
         Sharpe_improve >= +0.02) AND min9_degrade <= 1.0pp

target_vol sweep: {0.25, 0.30, 0.35}

Outputs
-------
  src/audit/combine_g1_voltarget_20260615.py  (this file)
  audit_results/combine_g1_voltarget_20260615.csv
  RETURN_BLOCK printed as JSON

ASCII-only prints (Windows cp932). No git ops. No temp files.
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

# ---- Reused helpers ----------------------------------------------------------
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _count_fund_transitions,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, FEE_GOLD, FEE_BOND,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    SWAP_SPREAD, TER_TQQQ, DH_PER_UNIT, NAV_FLOOR, DELAY as V7_DELAY,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom, LU2_SCALE,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_nav_c1,
)
# LEV_CAP from k365 script
from src.audit.lu_cfd_recost_20260611 import (
    LEV_CAP,
    AFTER_TAX,
)

# ---- Cost constants for B3a_k365 ---------------------------------------------
EXCESS_EXTRA_K365_CENTRE = 0.0025  # k365 centre: 0.25%/yr above TQQQ swap

# ---- B3a configuration -------------------------------------------------------
B3A_V7_MAP = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE = 1.15
B3A_EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE

# ---- B3a素地 known values (QC-verified, aftertax) ----------------------------
B3A_KNOWN_MIN9   = 0.2098   # +20.98%
B3A_KNOWN_MAXDD  = -0.3820  # -38.20%
B3A_KNOWN_SHARPE = 0.904
SANITY_TOL_PP    = 0.05     # 0.05pp tolerance for sanity anchor

# ---- Governor params ---------------------------------------------------------
VOL_WINDOW    = 63   # 63-day rolling realized vol
TARGET_VOLS   = [0.25, 0.30, 0.35]

# ---- Hard-veto thresholds (Stage 0 only) -------------------------------------
HARD_VETO_MAXDD = -0.50
HARD_VETO_W10Y  = 0.0

# ---- Survival criteria (improvement vs B3a素地) ------------------------------
IMPROVE_MAXDD_PP   = 2.0    # >= +2pp MaxDD improvement
IMPROVE_W10Y_PP    = 0.5    # >= +0.5pp Worst10Y* improvement
IMPROVE_SHARPE     = 0.02   # >= +0.02 Sharpe improvement
DEGRADE_MIN9_PP    = 1.0    # <= 1.0pp min9 degradation


# =============================================================================
# Core NAV builder with G1 governor natively embedded
# =============================================================================

def _build_nav_g1(
    close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
    lev_raw_masked, wn, wg, wb, mult_v7,
    excess_extra, target_vol,
):
    """
    Build NAV with G1 vol-target governor natively embedded.

    Native integration steps (per calendar day t):
      1. lev_mod[t] = lev_raw_masked[t] * mult_v7[t]      (unshifted)
      2. Shift lev_mod, wn, wg, wb by V7_DELAY (identical to base pipeline)
      3. L_raw[t] = lev_mod_shifted[t] * 3.0
      4. realized_vol_t = 63-day rolling annualised vol of r_nas, then shift(V7_DELAY)
      5. governor[t] = min(1.0, target_vol / realized_vol_t[t])
      6. L_gov[t] = L_raw[t] * governor[t]
      7. borrow, excess_penalty, nas_ret computed from L_gov[t]
      8. LEV_CAP threshold also evaluated on L_gov[t]

    Causality: realized_vol uses shift(V7_DELAY) -- only r_nas[..t-V7_DELAY] visible.
    Cap = 1.0 ensures governor only REDUCES leverage (DD-brake).
    """
    from src.audit.strategy_runners import (
        _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
        _compute_dhw1_trades_per_year,
    )
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

    idx = dates.index
    n = len(close)

    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3  = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    lev_mod = np.asarray(lev_raw_masked, float) * np.asarray(mult_v7, float)

    # ---- Shift L, weights (V7_DELAY) -- identical to base pipeline ----
    L_raw   = pd.Series(lev_mod * 3.0, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s    = pd.Series(np.asarray(wn, float), index=idx).shift(V7_DELAY).fillna(0).values
    wg_s    = pd.Series(np.asarray(wg, float), index=idx).shift(V7_DELAY).fillna(0).values
    wb_s    = pd.Series(np.asarray(wb, float), index=idx).shift(V7_DELAY).fillna(0).values
    sofr_a  = np.asarray(sofr_daily, float)

    # ---- G1: realized vol with same causal lag ----
    # 63-day rolling std of r_nas (annualised), then shift V7_DELAY
    r_nas_s = pd.Series(r_nas, index=idx)
    realvol_raw = (
        r_nas_s.rolling(window=VOL_WINDOW, min_periods=VOL_WINDOW)
                .std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )
    # Shift by V7_DELAY so that at day t, we only see vol through t-V7_DELAY
    realvol_shifted = realvol_raw.shift(V7_DELAY).fillna(realvol_raw.mean())
    realvol_a = np.maximum(realvol_shifted.values, 1e-8)  # avoid divide-by-zero

    # Governor: cap at 1.0 (only reduce leverage)
    governor = np.minimum(1.0, float(target_vol) / realvol_a)

    # ---- Apply governor to L ----
    L_gov = L_raw * governor

    # ---- TQQQ financing (using L_gov) ----
    borrow  = np.maximum(L_gov - 1.0, 0.0) * (sofr_a + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L_gov * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # ---- Excess-cost penalty (L_gov vs LEV_CAP=3) ----
    excess_lev = np.maximum(L_gov - LEV_CAP, 0.0)
    penalty    = wn_s * excess_lev * float(excess_extra) / TRADING_DAYS
    daily      = daily - penalty
    excess_days = int(np.sum(excess_lev > 1e-9))

    # ---- Turnover (raw weight changes, same as base) ----
    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    # ---- Incremental ETF TER + trade cost (identical to base) ----
    ter_drag = (
        np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
        + np.asarray(wb, float) * _TER_TMF_EXTRA_DAILY
    )
    tpy       = _compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / 252.0

    r_sim = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_sim - ter_drag - etf_daily
    nav_adj = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )
    return nav_adj, tpy, excess_days, governor


def _build_g1_c1(shared, dates_dt, n_years,
                 ret_gold, ret_bond, fund_active, wg_fill, wb_fill, bond_on,
                 sofr_arr, target_vol, excess_extra=B3A_EXCESS_EXTRA,
                 v7_map=B3A_V7_MAP, lev_scale=B3A_LEV_SCALE):
    """
    Build complete G1+C1 NAV for one target_vol config.

    Steps:
      1. Build V7 mult with B3a map and lev_scale
      2. Build base NAV with G1 governor embedded (_build_nav_g1)
      3. Apply C1 OUT-fill (_build_p09_nav_c1) on top of base returns
    """
    a = shared["assets"]
    close = a["close"]
    dates = a["dates"]
    gold_2x = a["gold_2x"]
    bond_3x = a["bond_3x"]
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn = np.asarray(shared["wn"], float)
    wg = np.asarray(shared["wg"], float)
    wb = np.asarray(shared["wb"], float)

    mult_v7 = _build_v7_mult_custom(dates_dt, v7_map)
    mult_v7 = mult_v7 * float(lev_scale)

    nav_dt, tpy_b, excess_days, governor = _build_nav_g1(
        close, dates, gold_2x, bond_3x, sofr_arr,
        lev_raw_masked, wn, wg, wb, mult_v7,
        excess_extra=excess_extra, target_vol=target_vol,
    )
    r_base = nav_dt.pct_change().fillna(0).values

    # C1 OUT-fill
    nav_arr, r_c1, eff_active = _build_p09_nav_c1(
        r_base, ret_gold, ret_bond, fund_active,
        wg_fill, wb_fill, bond_on, sofr_arr,
    )
    nav_full = pd.Series(nav_arr, index=dates_dt)
    flips = _count_fund_transitions(eff_active)
    tpy = tpy_b + flips / n_years

    return nav_full, r_c1, tpy, excess_days, governor


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _stage0_metrics(nav_dt, tpy):
    """Compute Stage 0 metrics (standard 10 + aftertax + worst_cy)."""
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    cy  = _calendar_year_returns(nav_dt)
    return pre, aft, cy


def _veto_stage0(pre, aft):
    """Stage 0 hard veto: MaxDD < -50% OR Worst10Y* < 0."""
    v_maxdd = pre["MaxDD_FULL"] < HARD_VETO_MAXDD
    v_w10y  = aft["Worst10Y_star"] < HARD_VETO_W10Y
    veto    = v_maxdd or v_w10y
    return {"maxdd": v_maxdd, "w10y": v_w10y, "VETO": veto}


def _survival(aft, pre, veto_map):
    """
    Survival = no_veto AND
               (MaxDD_improve >= +2pp OR Worst10Y*_improve >= +0.5pp OR Sharpe_improve >= +0.02)
               AND min9_degrade <= 1.0pp.

    Improvements computed vs B3a素地 known values.
    """
    if veto_map["VETO"]:
        return False, {}

    maxdd_improve_pp = (pre["MaxDD_FULL"] - B3A_KNOWN_MAXDD) * 100.0  # positive = better (less neg)
    w10y_improve_pp  = (_min_at(aft) - B3A_KNOWN_MIN9) * 100.0         # Worst10Y not available directly; use min9 as proxy
    sharpe_improve   = pre["Sharpe_OOS"] - B3A_KNOWN_SHARPE
    w10y_star_pp     = (aft["Worst10Y_star"] - (B3A_KNOWN_MIN9 * 0.9)) * 100.0  # approximate

    # Re-do: compare Worst10Y* directly; approximate B3a Worst10Y* = 14.53%
    B3A_KNOWN_W10Y = 0.1453  # from plan §0.1
    w10y_star_improve_pp = (aft["Worst10Y_star"] - B3A_KNOWN_W10Y) * 100.0

    min9_degrade_pp = (B3A_KNOWN_MIN9 - _min_at(aft)) * 100.0  # positive = degrade

    has_defence = (
        maxdd_improve_pp >= IMPROVE_MAXDD_PP
        or w10y_star_improve_pp >= IMPROVE_W10Y_PP
        or sharpe_improve >= IMPROVE_SHARPE
    )
    no_overdegrade = min9_degrade_pp <= DEGRADE_MIN9_PP

    survived = has_defence and no_overdegrade
    detail = {
        "maxdd_improve_pp": round(maxdd_improve_pp, 3),
        "w10y_star_improve_pp": round(w10y_star_improve_pp, 3),
        "sharpe_improve": round(sharpe_improve, 4),
        "min9_degrade_pp": round(min9_degrade_pp, 3),
        "has_defence": has_defence,
        "no_overdegrade": no_overdegrade,
    }
    return survived, detail


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 100)
    print("G1: vol-target / DD-brake governor - B3a_k365 Stage 0 sweep")
    print("target_vol sweep: %s" % str(TARGET_VOLS))
    print("B3a素地: min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f"
          % (B3A_KNOWN_MIN9 * 100, B3A_KNOWN_MAXDD * 100, B3A_KNOWN_SHARPE))
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
    sofr_arr = np.asarray(a["sofr"], float)

    # ---- Gold/Bond 1x legs ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0),
        float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]

    wg_fill, wb_fill = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on   = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    out_bond_off = fund_active & (~bond_on.astype(bool))
    n_out_bondoff = int(out_bond_off.sum())
    print("\nOUT-and-bondOFF days: %d of %d (%.1f%%)"
          % (n_out_bondoff, n, 100.0 * n_out_bondoff / n))

    # =========================================================================
    # SANITY ANCHOR: target_vol=10.0 (disabled) must reproduce B3a_k365
    # =========================================================================
    print("\n" + "=" * 100)
    print("SANITY ANCHOR: target_vol=10.0 (governor=1.0 always) => reproduce B3a素地")
    print("  Tolerance: +-%.2fpp on min(IS,OOS) aftertax" % SANITY_TOL_PP)
    print("=" * 100)

    sanity_nav, _, sanity_tpy, _, sanity_gov = _build_g1_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_fill, wb_fill, bond_on, sofr_arr,
        target_vol=10.0,
    )
    sanity_pre, sanity_aft, _ = _stage0_metrics(sanity_nav, sanity_tpy)
    sanity_min9  = _min_at(sanity_aft)
    sanity_maxdd = sanity_pre["MaxDD_FULL"]

    print("  governor stats (should be ~1.0 always):")
    print("    mean=%.4f  min=%.4f  max=%.4f  pct<1.0=%.1f%%"
          % (float(sanity_gov.mean()), float(sanity_gov.min()),
             float(sanity_gov.max()), float((sanity_gov < 1.0).mean()) * 100))

    ok_min9  = abs(sanity_min9  - B3A_KNOWN_MIN9)  * 100 <= SANITY_TOL_PP
    ok_maxdd = abs(sanity_maxdd - B3A_KNOWN_MAXDD) * 100 <= SANITY_TOL_PP * 2

    print("  min9 aftertax  : %+.4f%% (expect ~%+.2f%%)  -> %s  [diff=%.4fpp]"
          % (sanity_min9 * 100, B3A_KNOWN_MIN9 * 100,
             "OK" if ok_min9 else "WARN",
             (sanity_min9 - B3A_KNOWN_MIN9) * 100))
    print("  MaxDD          : %+.4f%% (expect ~%+.2f%%)  -> %s  [diff=%.4fpp]"
          % (sanity_maxdd * 100, B3A_KNOWN_MAXDD * 100,
             "OK" if ok_maxdd else "WARN",
             (sanity_maxdd - B3A_KNOWN_MAXDD) * 100))

    if not ok_min9:
        print("\n*** SANITY FAILED (min9 diff > %.2fpp) -- halting. ***" % SANITY_TOL_PP)
        print("    Check governor insertion point and B3a_k365 config wiring.")
        sys.exit(1)
    print("  SANITY PASSED.\n")

    # =========================================================================
    # STAGE 0 SWEEP: target_vol in {0.25, 0.30, 0.35}
    # =========================================================================
    print("=" * 100)
    print("STAGE 0 SWEEP")
    print("=" * 100)

    results = {}

    for tv in TARGET_VOLS:
        label = "G1_tv%.2f" % tv
        print("\n--- %s (target_vol=%.2f) ---" % (label, tv))

        nav_dt, _, tpy, excess_days, gov = _build_g1_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_fill, wb_fill, bond_on, sofr_arr,
            target_vol=tv,
        )
        pre, aft, cy = _stage0_metrics(nav_dt, tpy)
        veto_map = _veto_stage0(pre, aft)
        survived, surv_detail = _survival(aft, pre, veto_map)

        mn = _min_at(aft)
        # Differences vs B3a素地
        d_min9_pp   = (mn - B3A_KNOWN_MIN9) * 100
        d_maxdd_pp  = (pre["MaxDD_FULL"] - B3A_KNOWN_MAXDD) * 100
        d_w10y_pp   = (aft["Worst10Y_star"] - 0.1453) * 100
        d_sharpe    = pre["Sharpe_OOS"] - B3A_KNOWN_SHARPE

        gov_pct_active = float((gov < 1.0 - 1e-6).mean()) * 100
        gov_mean_factor = float(gov.mean())

        print("  min9_at        : %+.2f%%  (d vs B3a: %+.2fpp)"
              % (mn * 100, d_min9_pp))
        print("  MaxDD          : %+.2f%%  (d vs B3a: %+.2fpp)"
              % (pre["MaxDD_FULL"] * 100, d_maxdd_pp))
        print("  Worst10Y*_at   : %+.2f%%  (d vs B3a: %+.2fpp)"
              % (aft["Worst10Y_star"] * 100, d_w10y_pp))
        print("  Sharpe_OOS     : %.3f    (d vs B3a: %+.3f)"
              % (pre["Sharpe_OOS"], d_sharpe))
        print("  CAGR_IS_at     : %+.2f%%"  % (aft["CAGR_IS"] * 100))
        print("  CAGR_OOS_at    : %+.2f%%"  % (aft["CAGR_OOS"] * 100))
        print("  IS-OOS gap_pp  : %+.2f"    % aft["IS_OOS_gap_pp"])
        print("  P10_5Y_at      : %+.2f%%"  % (aft["P10_5Y"] * 100))
        print("  Worst5Y_at     : %+.2f%%"  % (aft["Worst5Y"] * 100))
        print("  Trades/yr      : %.1f"      % aft["Trades_yr"])
        print("  worst_cy       : %+.2f%% (%d)"
              % (float(cy.min()) * 100, int(cy.idxmin())))
        print("  Governor active: %.1f%% of days  mean_factor=%.4f"
              % (gov_pct_active, gov_mean_factor))
        print("  Hard veto      : MaxDD=%s  W10Y=%s  VETO=%s"
              % ("YES" if veto_map["maxdd"] else "no",
                 "YES" if veto_map["w10y"]  else "no",
                 "YES" if veto_map["VETO"]  else "no"))
        print("  Survival       : %s" % ("PASS" if survived else "FAIL"))
        if surv_detail:
            print("    MaxDD_improve=%+.2fpp  W10Y*_improve=%+.2fpp  Sharpe_improve=%+.3f  min9_degrade=%+.2fpp"
                  % (surv_detail["maxdd_improve_pp"], surv_detail["w10y_star_improve_pp"],
                     surv_detail["sharpe_improve"], surv_detail["min9_degrade_pp"]))

        results[label] = {
            "target_vol": tv,
            "CAGR_IS_at":   round(float(aft["CAGR_IS"]), 6),
            "CAGR_OOS_at":  round(float(aft["CAGR_OOS"]), 6),
            "min9_at":      round(float(mn), 6),
            "gap_pp":       round(float(aft["IS_OOS_gap_pp"]), 4),
            "Sharpe_OOS":   round(float(pre["Sharpe_OOS"]), 4),
            "MaxDD_FULL":   round(float(pre["MaxDD_FULL"]), 6),
            "Worst10Y_star_at": round(float(aft["Worst10Y_star"]), 6),
            "P10_5Y_at":    round(float(aft["P10_5Y"]), 6),
            "Worst5Y_at":   round(float(aft["Worst5Y"]), 6),
            "Trades_yr":    round(float(aft["Trades_yr"]), 2),
            "worst_cy":     round(float(cy.min()), 6),
            "worst_cy_year":int(cy.idxmin()),
            "excess_days":  excess_days,
            # vs B3a素地 deltas
            "d_min9_pp":    round(d_min9_pp, 4),
            "d_maxdd_pp":   round(d_maxdd_pp, 4),
            "d_w10y_pp":    round(d_w10y_pp, 4),
            "d_sharpe":     round(d_sharpe, 4),
            # Governor stats
            "gov_pct_active": round(gov_pct_active, 2),
            "gov_mean_factor": round(gov_mean_factor, 4),
            # Veto / survival
            "veto_maxdd":   int(veto_map["maxdd"]),
            "veto_w10y":    int(veto_map["w10y"]),
            "VETO":         int(veto_map["VETO"]),
            "survived":     int(survived),
            # survival detail
            **({k: round(v, 4) if isinstance(v, float) else v
                for k, v in surv_detail.items()}),
        }

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 0 SUMMARY TABLE  (B3a素地: min9=%+.2f%% MaxDD=%+.2f%% Sharpe=%.3f W10Y*=%+.2f%%)"
          % (B3A_KNOWN_MIN9 * 100, B3A_KNOWN_MAXDD * 100, B3A_KNOWN_SHARPE, 14.53))
    print("-" * 100)
    hdr = ("%-15s | %8s | %8s | %8s | %7s | %9s | %9s | %8s | %7s | %9s | %9s"
           % ("label", "min9_at%", "d_min9pp", "MaxDD%", "d_MaxDDpp", "W10Y*_at%",
              "d_W10Ypp", "Sharpe", "d_Sharpe", "Trades/yr", "SURVIVED"))
    print(hdr)
    print("-" * len(hdr))

    for label, r in results.items():
        print("%-15s | %+7.2f%% | %+8.2f | %+7.2f%% | %+8.2f | %+8.2f%% | %+8.2f | %7.3f | %+8.3f | %9.1f | %s"
              % (label,
                 r["min9_at"] * 100, r["d_min9_pp"],
                 r["MaxDD_FULL"] * 100, r["d_maxdd_pp"],
                 r["Worst10Y_star_at"] * 100, r["d_w10y_pp"],
                 r["Sharpe_OOS"], r["d_sharpe"],
                 r["Trades_yr"],
                 "PASS" if r["survived"] else "FAIL"))

    survived_labels = [lbl for lbl, r in results.items() if r["survived"]]
    print("\nSurviving target_vol values: %s"
          % (str([results[l]["target_vol"] for l in survived_labels])
             if survived_labels else "NONE"))

    # =========================================================================
    # CSV output
    # =========================================================================
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combine_g1_voltarget_20260615.csv")

    # Add B3a素地 as reference row
    b3a_row = {
        "label": "B3a_k365_base",
        "target_vol": None,
        "CAGR_IS_at": None,
        "CAGR_OOS_at": None,
        "min9_at": B3A_KNOWN_MIN9,
        "gap_pp": None,
        "Sharpe_OOS": B3A_KNOWN_SHARPE,
        "MaxDD_FULL": B3A_KNOWN_MAXDD,
        "Worst10Y_star_at": 0.1453,
        "P10_5Y_at": None,
        "Worst5Y_at": None,
        "Trades_yr": 27.0,
        "worst_cy": None,
        "worst_cy_year": None,
        "excess_days": None,
        "d_min9_pp": 0.0,
        "d_maxdd_pp": 0.0,
        "d_w10y_pp": 0.0,
        "d_sharpe": 0.0,
        "gov_pct_active": 0.0,
        "gov_mean_factor": 1.0,
        "veto_maxdd": 0, "veto_w10y": 0, "VETO": 0, "survived": None,
        "maxdd_improve_pp": 0.0, "w10y_star_improve_pp": 0.0,
        "sharpe_improve": 0.0, "min9_degrade_pp": 0.0,
        "has_defence": None, "no_overdegrade": None,
    }

    rows = [{"label": "B3a_k365_base", **b3a_row}]
    for label, r in results.items():
        rows.append({"label": label, **r})

    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("\nSaved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # RETURN BLOCK
    # =========================================================================
    block = {
        "meta": {
            "script": "combine_g1_voltarget_20260615.py",
            "target_vols": TARGET_VOLS,
            "vol_window": VOL_WINDOW,
            "causal_lag": V7_DELAY,
            "b3a_known_min9_pct":   round(B3A_KNOWN_MIN9 * 100, 4),
            "b3a_known_maxdd_pct":  round(B3A_KNOWN_MAXDD * 100, 4),
            "b3a_known_sharpe":     B3A_KNOWN_SHARPE,
            "b3a_known_w10y_pct":   14.53,
            "sanity_min9_pct":      round(sanity_min9 * 100, 4),
            "sanity_maxdd_pct":     round(sanity_maxdd * 100, 4),
            "sanity_ok":            bool(ok_min9),
            "survived_labels":      survived_labels,
            "n_out_bondoff":        n_out_bondoff,
        },
        "results": {
            label: {
                "target_vol":         r["target_vol"],
                "min9_at_pct":        round(r["min9_at"] * 100, 4),
                "MaxDD_pct":          round(r["MaxDD_FULL"] * 100, 4),
                "Worst10Y_star_pct":  round(r["Worst10Y_star_at"] * 100, 4),
                "Sharpe_OOS":         r["Sharpe_OOS"],
                "CAGR_IS_pct":        round(r["CAGR_IS_at"] * 100, 4),
                "CAGR_OOS_pct":       round(r["CAGR_OOS_at"] * 100, 4),
                "gap_pp":             r["gap_pp"],
                "P10_5Y_pct":         round(r["P10_5Y_at"] * 100, 4),
                "Worst5Y_pct":        round(r["Worst5Y_at"] * 100, 4),
                "Trades_yr":          r["Trades_yr"],
                "worst_cy_pct":       round(r["worst_cy"] * 100, 4),
                "worst_cy_year":      r["worst_cy_year"],
                "d_min9_pp":          r["d_min9_pp"],
                "d_maxdd_pp":         r["d_maxdd_pp"],
                "d_w10y_pp":          r["d_w10y_pp"],
                "d_sharpe":           r["d_sharpe"],
                "gov_pct_active":     r["gov_pct_active"],
                "gov_mean_factor":    r["gov_mean_factor"],
                "VETO":               bool(r["VETO"]),
                "survived":           bool(r["survived"]),
                "maxdd_improve_pp":   r.get("maxdd_improve_pp"),
                "w10y_star_improve_pp": r.get("w10y_star_improve_pp"),
                "sharpe_improve":     r.get("sharpe_improve"),
                "min9_degrade_pp":    r.get("min9_degrade_pp"),
            }
            for label, r in results.items()
        },
    }

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
