"""
src/audit/leverext_high_20260618.py
=====================================
High-scale leverage extension sweep -- scale 1.35 (sanity), 1.40, 1.50, 1.60
v7_map = STRONG ONLY: {0:1.60, 1:1.50, 2:1.10, 3:1.00}

Purpose:
  Extend beyond the previously tested range (1.20-1.35) to detect:
  - CAGR frontier peak / inflection (volatility drag)
  - MaxDD -50% veto boundary
  - Worst5Y sign flip
  - M6 forced-liquidation count escalation (m=8%, m=12%)
  - Worst-case hypothesis AUM loss (Black Monday -11.3% single day)

Sanity gate (required to proceed):
  scale=1.35 strong map must reproduce:
    min9 +23.83% +/-0.10pp
    MaxDD -45.04% +/-0.10pp
    max_L 6.48x (not a hard halt, soft warning if differs > 0.05x)

VETO definition (4-axis):
  v_maxdd : MaxDD < -50%
  v_w10y  : Worst10Y* < 0
  v_wfe   : WFE > 1.5  (Stage-1 only, not Stage-0)
  v_reg   : Regime_min_bear < -10%  (Stage-1 only)

M6 forced-liquidation (path-dependent, inline implementation):
  Margin scenarios: m=8.0%, m=12.0%
  intraday_add = 0.0 (conservative/real)
  Reports: n_liq / max_loss_pct_AUM / CAGR_gap_pp / min9_gap_pp / MaxDD_gap_pp
  Crisis presence (1987/2000/2008/COVID/2022): liquidation count per crisis

Worst-case hypothesis:
  Black Monday peak: NASDAQ single-day = -11.3%
  For each scale: max excess_notional_fraction * 0.113 = AUM loss %

ASCII-only prints (Windows cp932). Does NOT commit. No temp files in repo root.
Outputs:
  audit_results/leverext_high_20260618.csv
  RETURN_BLOCK printed to stdout (json.dumps)
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
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    DELAY as V7_DELAY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B3A_MAP_STRONG = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXCESS_EXTRA   = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# Sanity gate (Bext_str_sc1.35)
SANITY_MIN9_EXPECT  = 0.2383  # +23.83%
SANITY_MAXDD_EXPECT = -0.4504  # -45.04%
SANITY_MAX_L_EXPECT = 6.48    # approx
SANITY_TOL = 0.0010
SANITY_MAX_L_TOL = 0.10

# Hard veto thresholds
HARD_VETO_MAXDD  = -0.50
HARD_VETO_W10Y   = 0.0
HARD_VETO_WFE    = 1.5      # Stage-1 only
HARD_VETO_REGIME = -0.10    # Stage-1 only

# Sweep: sanity scale + 3 high scales
SWEEP_SCALES = [1.35, 1.40, 1.50, 1.60]

# M6 margin scenarios for the cross-scale sweep
M6_MARGIN_RATES = [0.08, 0.12]
M6_INTRADAY_ADD = 0.0

# Crisis periods
CRISIS_PERIODS = {
    "BlackMonday1987": ("1987-10-15", "1987-10-20"),
    "Dotcom2000":      ("2000-03-10", "2002-10-10"),
    "PostLehman2008":  ("2008-09-01", "2008-11-30"),
    "COVID2020":       ("2020-02-19", "2020-03-23"),
    "RateHike2022":    ("2022-01-01", "2022-12-31"),
}

# Black Monday worst-case
BM_SINGLE_DAY_DROP = 0.113  # -11.3%

AUM_JPY = 30_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _build_leverage_series(shared, dates_dt, v7_map, lev_scale):
    """Reconstruct L_s, wn_s, wg_s, wb_s, in_mask, fund_active, excess_n_pct."""
    a = shared["assets"]
    dates = a["dates"]
    idx   = dates.index
    mask  = np.asarray(shared["mask"], dtype=float)

    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn = np.asarray(shared["wn"], float)
    wg = np.asarray(shared["wg"], float)
    wb = np.asarray(shared["wb"], float)

    mult_v7 = _build_v7_mult_custom(dates_dt, v7_map) * float(lev_scale)
    lev_mod = lev_raw_masked * mult_v7
    L_unshifted = lev_mod * 3.0

    L_s  = pd.Series(L_unshifted, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s = pd.Series(wn, index=idx).shift(V7_DELAY).fillna(0.0).values
    wg_s = pd.Series(wg, index=idx).shift(V7_DELAY).fillna(0.0).values
    wb_s = pd.Series(wb, index=idx).shift(V7_DELAY).fillna(0.0).values

    out_mask_arr = (mask < 0.5)
    fund_active  = np.zeros(len(dates_dt), dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]
    in_mask = ~fund_active

    excess_n_pct = wn_s * np.maximum(L_s - 3.0, 0.0)
    return L_s, wn_s, wg_s, wb_s, in_mask, fund_active, excess_n_pct


def _compute_m6_path(r_full, r_nasdaq, dates_dt, L_s, wn_s, in_mask, fund_active,
                     margin_rate, intraday_add=0.0):
    """
    Path-dependent daily margin tracking (M6 model).
    Returns summary dict + liq_events list.
    """
    n = len(dates_dt)
    excess_n = wn_s * np.maximum(L_s - 3.0, 0.0)

    nav_path    = np.ones(n, dtype=float)
    k365_equity = 0.0
    k365_sus    = False
    saw_out_liq = False
    liq_events  = []

    for t in range(1, n):
        r_n_t    = float(r_nasdaq[t])
        r_full_t = float(r_full[t])
        excess_t = float(excess_n[t])
        in_day   = bool(in_mask[t])

        # Re-entry logic
        if k365_sus and fund_active[t]:
            saw_out_liq = True
        if k365_sus and saw_out_liq and not fund_active[t]:
            k365_sus    = False
            saw_out_liq = False
            if excess_t > 1e-6:
                k365_equity = margin_rate * excess_t

        liq_today = False
        if (not k365_sus) and in_day and (excess_t > 1e-6):
            if k365_equity < 1e-10:
                k365_equity = margin_rate * excess_t

            required_t = margin_rate * excess_t
            eff_r_n = r_n_t * (1.0 + intraday_add) if r_n_t < 0 else r_n_t
            pnl_t   = eff_r_n * excess_t
            k365_equity += pnl_t

            if k365_equity < required_t:
                single_day_trig = (-r_n_t * (1.0 + intraday_add)) >= margin_rate
                liq_events.append({
                    "date":            str(dates_dt[t].date()),
                    "L":               round(float(L_s[t]), 4),
                    "excess_n_pct":    round(excess_t * 100, 4),
                    "nasdaq_ret_pct":  round(r_n_t * 100, 4),
                    "liq_loss_pct_AUM": round(k365_equity * 100, 4),
                    "pure_path_effect": not single_day_trig,
                })
                k365_sus    = True
                saw_out_liq = False
                k365_equity = 0.0
                liq_today   = True

        # NAV
        if k365_sus and in_day and not liq_today:
            k365_contrib = wn_s[t] * max(L_s[t] - 3.0, 0.0) * r_nasdaq[t]
            r_adj = r_full_t - k365_contrib
        else:
            r_adj = r_full_t
        nav_path[t] = nav_path[t - 1] * (1.0 + r_adj)

    n_years    = n / float(TRADING_DAYS)
    nav_base   = np.cumprod(1.0 + r_full)
    is_mask_g  = np.array(dates_dt <= IS_END)
    oos_mask_g = np.array(dates_dt >= OOS_START)

    def _cagr(nv, ny): return float(nv[-1]) ** (1.0 / ny) - 1.0
    def _mdd(nv):
        rm = np.maximum.accumulate(nv)
        return float((nv / rm - 1.0).min())
    def _seg(nv, m):
        s = nv[m]
        if len(s) < 2: return float("nan")
        ny = m.sum() / float(TRADING_DAYS)
        return float(s[-1] / s[0]) ** (1.0 / ny) - 1.0

    cagr_base  = _cagr(nav_base, n_years)
    cagr_path  = _cagr(nav_path, n_years)
    mdd_base   = _mdd(nav_base)
    mdd_path   = _mdd(nav_path)
    min9_base  = min(_seg(nav_base, is_mask_g), _seg(nav_base, oos_mask_g))
    min9_path  = min(_seg(nav_path, is_mask_g), _seg(nav_path, oos_mask_g))

    max_liq_loss = max((e["liq_loss_pct_AUM"] for e in liq_events), default=0.0)

    return {
        "margin_rate_pct": round(margin_rate * 100, 2),
        "n_liq":           len(liq_events),
        "n_pure_path_liq": sum(1 for e in liq_events if e["pure_path_effect"]),
        "max_liq_loss_pct_AUM": round(max_liq_loss, 4),
        "cagr_base_pct":   round(cagr_base * 100, 4),
        "cagr_path_pct":   round(cagr_path * 100, 4),
        "cagr_gap_pp":     round((cagr_path - cagr_base) * 100, 4),
        "min9_base_pct":   round(min9_base * 100, 4),
        "min9_path_pct":   round(min9_path * 100, 4),
        "min9_gap_pp":     round((min9_path - min9_base) * 100, 4),
        "mdd_base_pct":    round(mdd_base * 100, 4),
        "mdd_path_pct":    round(mdd_path * 100, 4),
        "mdd_gap_pp":      round((mdd_path - mdd_base) * 100, 4),
        "liq_events":      liq_events,
    }


def _crisis_liq_count(liq_events, crisis_name, crisis_start, crisis_end):
    """Count liquidations falling in a crisis window."""
    s = pd.Timestamp(crisis_start)
    e = pd.Timestamp(crisis_end)
    count = 0
    for ev in liq_events:
        d = pd.Timestamp(ev["date"])
        if s <= d <= e:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Stage-0 row builder
# ---------------------------------------------------------------------------

def _stage0_row(label, nav_dt, tpy, exc, n_total, lev_scale):
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    cy  = _calendar_year_returns(nav_dt)
    mn  = _min_at(aft)

    excess_ratio = float(exc) / float(n_total) if n_total > 0 else 0.0
    v_maxdd = pre["MaxDD_FULL"] < HARD_VETO_MAXDD
    v_w10y  = aft["Worst10Y_star"] < HARD_VETO_W10Y
    veto    = v_maxdd or v_w10y

    return {
        "label":             label,
        "lev_scale":         lev_scale,
        "CAGR_IS_at":        round(aft["CAGR_IS"],        6),
        "CAGR_OOS_at":       round(aft["CAGR_OOS"],       6),
        "min9_at":           round(mn,                    6),
        "IS_OOS_gap_pp":     round(aft["IS_OOS_gap_pp"],  4),
        "Sharpe_OOS":        round(pre["Sharpe_OOS"],     4),
        "MaxDD_FULL":        round(pre["MaxDD_FULL"],      6),
        "Worst10Y_star_at":  round(aft["Worst10Y_star"],  6),
        "P10_5Y_at":         round(aft["P10_5Y"],         6),
        "Worst5Y_at":        round(aft["Worst5Y"],        6),
        "Trades_yr":         round(aft["Trades_yr"],      2),
        "excess_days":       int(exc),
        "excess_ratio_pct":  round(excess_ratio * 100,    2),
        "worst_cy":          round(float(cy.min()),       6),
        "worst_cy_year":     int(cy.idxmin()),
        "veto_maxdd":        int(v_maxdd),
        "veto_w10y":         int(v_w10y),
        "VETO_s0":           int(veto),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("LEVEREXT HIGH SCALE SWEEP  2026-06-18")
    print("v7_map STRONG: {0:1.60, 1:1.50, 2:1.10, 3:1.00}")
    print("lev_scale in {1.35 (sanity), 1.40, 1.50, 1.60}")
    print("Purpose: CAGR frontier upper bound, MaxDD-50%% veto detection, M6 tail-risk scaling")
    print("Sanity: scale=1.35 must give min9 +23.83%%+/-0.10pp / MaxDD -45.04%%+/-0.10pp")
    print("Hard veto (Stage-0): MaxDD<-50%%, Worst10Y*<0")
    print("=" * 120)

    # ---- Load shared data ----
    sr._load_dhw1_shared()
    shared   = sr._DHW1_SHARED
    a        = shared["assets"]
    mask     = np.asarray(shared["mask"], dtype=float)
    dates    = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    # ---- Gold/Bond ----
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
    close_arr    = np.asarray(a["close"], float)
    r_nasdaq     = np.concatenate([[0.0], np.diff(close_arr) / close_arr[:-1]])

    # =========================================================================
    # SANITY GATE: Reproduce Bext_str_sc1.35
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: Bext_str_sc1.35")
    print("  Expected: min9 +23.83%% +/-0.10pp  MaxDD -45.04%% +/-0.10pp  max_L ~6.48x")
    print("=" * 120)

    san_nav, san_r, san_tpy, san_exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        v7_map=B3A_MAP_STRONG, lev_scale=1.35, excess_extra=EXCESS_EXTRA)

    san_pre  = compute_10metrics(san_nav, san_tpy)
    san_aft  = _apply_aftertax(san_pre)
    san_min9 = _min_at(san_aft)
    san_mdd  = san_pre["MaxDD_FULL"]

    # Max L
    L_san, wn_san, _, _, in_mask_san, _, _ = _build_leverage_series(
        shared, dates_dt, B3A_MAP_STRONG, 1.35)
    L_in_san = L_san[in_mask_san]
    san_max_L = float(np.max(L_in_san)) if len(L_in_san) > 0 else 0.0

    ok_min9  = abs(san_min9 - SANITY_MIN9_EXPECT)  <= SANITY_TOL
    ok_maxdd = abs(san_mdd  - SANITY_MAXDD_EXPECT) <= SANITY_TOL
    ok_maxL  = abs(san_max_L - SANITY_MAX_L_EXPECT) <= SANITY_MAX_L_TOL

    print("  min9:  got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_min9 * 100, SANITY_MIN9_EXPECT * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD: got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (san_mdd * 100,  SANITY_MAXDD_EXPECT * 100, "OK" if ok_maxdd else "FAIL"))
    print("  max_L: got %.4fx   expect ~%.4fx      -> %s"
          % (san_max_L, SANITY_MAX_L_EXPECT, "OK" if ok_maxL else "WARN (soft)"))

    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED -- halting. Check _build_full_c1 / EXCESS_EXTRA wiring.")
        sys.exit(1)
    print("  SANITY PASSED. Proceeding to high-scale sweep.\n")

    # =========================================================================
    # STAGE 0: Build all 4 scales
    # =========================================================================
    print("=" * 120)
    print("STAGE 0: Building NAVs for scale {1.35, 1.40, 1.50, 1.60} (strong map)")
    print("=" * 120)

    s0_rows   = []
    nav_cache = {}  # scale -> (nav_dt, r, tpy, exc, L_s, wn_s, in_mask, fund_active)

    for sc in SWEEP_SCALES:
        lbl = "Bext_str_sc%.2f" % sc
        print("\n  Building %s (scale=%.2f) ..." % (lbl, sc))
        nav_dt, r, tpy, exc = _build_full_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
            v7_map=B3A_MAP_STRONG, lev_scale=sc, excess_extra=EXCESS_EXTRA)

        L_s, wn_s, wg_s, wb_s, in_mask_sc, fa_sc, exc_n_pct = \
            _build_leverage_series(shared, dates_dt, B3A_MAP_STRONG, sc)
        L_in  = L_s[in_mask_sc]
        max_L = float(np.max(L_in)) if len(L_in) > 0 else 0.0
        n_gt3 = int((L_s > 3.0).sum())
        ratio_gt3 = 100.0 * n_gt3 / n

        nav_cache[sc] = (nav_dt, r, tpy, exc, L_s, wn_s, in_mask_sc, fa_sc,
                         exc_n_pct, max_L, n_gt3, ratio_gt3)

        row = _stage0_row(lbl, nav_dt, tpy, exc, n, sc)
        s0_rows.append(row)

        print("    min9=%+.2f%%  MaxDD=%+.2f%%  W10Y*=%+.2f%%  Worst5Y=%+.2f%%  Sharpe=%.4f  VETO=%s"
              % (row["min9_at"] * 100, row["MaxDD_FULL"] * 100,
                 row["Worst10Y_star_at"] * 100, row["Worst5Y_at"] * 100,
                 row["Sharpe_OOS"], "YES" if row["VETO_s0"] else "no"))
        print("    CAGR_IS=%+.2f%%  CAGR_OOS=%+.2f%%  IS-OOS gap=%.4f pp"
              % (row["CAGR_IS_at"] * 100, row["CAGR_OOS_at"] * 100, row["IS_OOS_gap_pp"]))
        print("    max_L=%.4fx  >3x days=%d (%.2f%%)"
              % (max_L, n_gt3, ratio_gt3))

    # ---- Print Stage-0 table ----
    print("\n" + "=" * 140)
    print("STAGE 0 RESULTS -- STANDARD 10 METRICS (after-tax; Sharpe/MaxDD pretax)")
    hdr = ("%-22s | %5s | %8s | %9s | %6s | %6s | %7s | %7s | %7s | %7s | %7s | %7s | %6s | %5s"
           % ("label", "scale", "CAGR_IS%", "CAGR_OOS%",
              "min9%", "gap_pp", "Sharpe", "MaxDD%",
              "W10Y*%", "P10_5Y%", "W5Y%", "Trd/yr", ">3x%", "VETO"))
    print(hdr)
    print("-" * 140)
    for row in s0_rows:
        sc     = row["lev_scale"]
        cache  = nav_cache[sc]
        ratio_ = cache[11]
        veto_s = "VETO" if row["VETO_s0"] else "pass"
        print("%-22s | %5.2f | %+7.2f%% | %+8.2f%% | %+5.2f%% | %+5.3f | %6.4f | %+6.2f%% | %+6.2f%% | %+6.2f%% | %+6.2f%% | %6.1f | %5.1f%% | %-4s"
              % (row["label"][:22], sc,
                 row["CAGR_IS_at"] * 100, row["CAGR_OOS_at"] * 100,
                 row["min9_at"] * 100, row["IS_OOS_gap_pp"],
                 row["Sharpe_OOS"], row["MaxDD_FULL"] * 100,
                 row["Worst10Y_star_at"] * 100, row["P10_5Y_at"] * 100,
                 row["Worst5Y_at"] * 100, row["Trades_yr"],
                 ratio_, veto_s))

    # ---- Veto analysis ----
    print("\n--- VETO ANALYSIS ---")
    any_maxdd_veto = False
    any_w10y_veto  = False
    for row in s0_rows:
        if row["veto_maxdd"]:
            any_maxdd_veto = True
            print("  MaxDD-50%% VETO: %s  scale=%.2f  MaxDD=%+.2f%%"
                  % (row["label"], row["lev_scale"], row["MaxDD_FULL"] * 100))
        if row["veto_w10y"]:
            any_w10y_veto = True
            print("  W10Y*<0 VETO:   %s  scale=%.2f  W10Y*=%+.2f%%"
                  % (row["label"], row["lev_scale"], row["Worst10Y_star_at"] * 100))
    if not any_maxdd_veto:
        print("  MaxDD-50%% veto NOT triggered through scale=1.60")
    if not any_w10y_veto:
        print("  W10Y*<0 veto NOT triggered through scale=1.60")

    # ---- Worst5Y sign flip ----
    print("\n--- WORST5Y ANALYSIS ---")
    for row in s0_rows:
        sign_flag = "NEGATIVE" if row["Worst5Y_at"] < 0 else "positive"
        print("  scale=%.2f  Worst5Y=%+.4f%%  [%s]"
              % (row["lev_scale"], row["Worst5Y_at"] * 100, sign_flag))

    # ---- Max L and >3x ratio by scale ----
    print("\n--- MAX EFFECTIVE LEVERAGE AND >3x DAY RATIO ---")
    for sc in SWEEP_SCALES:
        cache = nav_cache[sc]
        print("  scale=%.2f  max_L=%.4fx  >3x days=%d (%.2f%%)"
              % (sc, cache[9], cache[10], cache[11]))

    # =========================================================================
    # MONOTONICITY CHECK
    # =========================================================================
    print("\n--- MONOTONICITY CHECK (min9 vs scale) ---")
    prev_m9 = None
    monotone = True
    for row in sorted(s0_rows, key=lambda x: x["lev_scale"]):
        mono_note = ""
        if prev_m9 is not None and row["min9_at"] < prev_m9:
            monotone = False
            mono_note = " [NON-MONOTONE -- potential volatility drag peak CROSSED]"
        print("  scale=%.2f  min9=%+.4f%%  MaxDD=%+.4f%%%s"
              % (row["lev_scale"], row["min9_at"] * 100,
                 row["MaxDD_FULL"] * 100, mono_note))
        prev_m9 = row["min9_at"]
    if monotone:
        print("  => Monotone (CAGR peak NOT yet reached through scale=1.60)")
    else:
        print("  => NON-MONOTONE: volatility drag peak IDENTIFIED above")

    # =========================================================================
    # M6 PATH-DEPENDENT FORCED LIQUIDATION SWEEP (m=8%, m=12%)
    # =========================================================================
    print("\n" + "=" * 120)
    print("M6 PATH-DEPENDENT FORCED LIQUIDATION SWEEP")
    print("Margin scenarios: m=8%% and m=12%%  intraday_add=0%%")
    print("Replicates M6 logic from margin_path_chain_20260617.py, extended to all 4 scales")
    print("=" * 120)

    m6_results = {}  # (scale, margin_rate) -> result_dict

    for sc in SWEEP_SCALES:
        cache   = nav_cache[sc]
        nav_dt  = cache[0]
        r_arr   = np.asarray(cache[1], float)
        L_s     = cache[4]
        wn_s    = cache[5]
        in_mask_sc = cache[6]
        fa_sc   = cache[7]

        for mar_rate in M6_MARGIN_RATES:
            key = (sc, mar_rate)
            print("\n  [M6] scale=%.2f  mar=%.0f%%  ..." % (sc, mar_rate * 100))
            res = _compute_m6_path(r_arr, r_nasdaq, dates_dt,
                                   L_s, wn_s, in_mask_sc, fa_sc,
                                   margin_rate=mar_rate,
                                   intraday_add=M6_INTRADAY_ADD)
            m6_results[key] = res

            print("    n_liq=%d  (path-only=%d)  max_loss=%.4f%% AUM"
                  % (res["n_liq"], res["n_pure_path_liq"], res["max_liq_loss_pct_AUM"]))
            print("    CAGR: base=%+.4f%%  path=%+.4f%%  gap=%+.4f pp"
                  % (res["cagr_base_pct"], res["cagr_path_pct"], res["cagr_gap_pp"]))
            print("    min9: base=%+.4f%%  path=%+.4f%%  gap=%+.4f pp"
                  % (res["min9_base_pct"], res["min9_path_pct"], res["min9_gap_pp"]))
            print("    MaxDD: base=%+.4f%%  path=%+.4f%%  gap=%+.4f pp"
                  % (res["mdd_base_pct"], res["mdd_path_pct"], res["mdd_gap_pp"]))

            # Crisis presence
            crisis_counts = {}
            for cname, (cs, ce) in CRISIS_PERIODS.items():
                cnt = _crisis_liq_count(res["liq_events"], cname, cs, ce)
                crisis_counts[cname] = cnt
            m6_results[key]["crisis_liq_counts"] = crisis_counts

            crisis_str = "  ".join(
                "%s=%d" % (k[:6], v) for k, v in crisis_counts.items())
            print("    Crisis liq counts: %s" % crisis_str)

    # =========================================================================
    # M6 CROSS-SCALE SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 120)
    print("M6 CROSS-SCALE SUMMARY TABLE")
    print("=" * 120)
    for mar_rate in M6_MARGIN_RATES:
        print("\n  margin=%.0f%%:" % (mar_rate * 100))
        print("  %-22s | %5s | %5s | %6s | %9s | %9s | %9s | %9s | %s"
              % ("label", "scale", "n_liq", "maxloss", "CAGR_gap", "min9_gap",
                 "MaxDD_gap", "min9_path", "crisis(BM/DC/GFC/COV/22)"))
        print("  " + "-" * 120)
        for sc in SWEEP_SCALES:
            key = (sc, mar_rate)
            res = m6_results[key]
            cc  = res["crisis_liq_counts"]
            crisis_str = "%d/%d/%d/%d/%d" % (
                cc.get("BlackMonday1987", 0), cc.get("Dotcom2000", 0),
                cc.get("PostLehman2008", 0), cc.get("COVID2020", 0),
                cc.get("RateHike2022", 0))
            lbl = "Bext_str_sc%.2f" % sc
            print("  %-22s | %5.2f | %5d | %5.3f%% | %+8.4fpp | %+8.4fpp | %+8.4fpp | %+8.4f%% | %s"
                  % (lbl, sc, res["n_liq"],
                     res["max_liq_loss_pct_AUM"],
                     res["cagr_gap_pp"], res["min9_gap_pp"], res["mdd_gap_pp"],
                     res["min9_path_pct"],
                     crisis_str))

    # =========================================================================
    # WORST-CASE HYPOTHESIS: Black Monday -11.3% single day
    # =========================================================================
    print("\n" + "=" * 120)
    print("WORST-CASE HYPOTHESIS: Black Monday-grade single-day drop (NASDAQ -11.3%%)")
    print("AUM loss %% = max_excess_notional_fraction x 0.113")
    print("(Note: this is purely from k365 excess position, not the full portfolio)")
    print("=" * 120)
    print("  %-22s | %5s | %8s | %12s | %12s | %12s"
          % ("label", "scale", "max_exc%", "excAUM_pct", "BM_loss%AUM", "BM_JPY"))
    print("  " + "-" * 90)
    for sc in SWEEP_SCALES:
        cache      = nav_cache[sc]
        exc_n_pct  = cache[8]   # excess_notional_pct (fraction of AUM)
        in_mask_sc = cache[6]
        k365_act   = (exc_n_pct > 1e-6) & in_mask_sc
        max_exc    = float(np.max(exc_n_pct[k365_act])) if k365_act.sum() > 0 else 0.0
        bm_loss    = max_exc * BM_SINGLE_DAY_DROP * 100  # % of AUM
        bm_jpy     = max_exc * BM_SINGLE_DAY_DROP * AUM_JPY
        lbl = "Bext_str_sc%.2f" % sc
        print("  %-22s | %5.2f | %7.4f%% | %11.4f%% | %11.4f%% | JPY %9.0f"
              % (lbl, sc, max_exc * 100, max_exc * 100, bm_loss, bm_jpy))

    # =========================================================================
    # CONCLUSIONS (questions 1-5)
    # =========================================================================
    print("\n" + "=" * 120)
    print("CONCLUSIONS (Answering Questions 1-5)")
    print("=" * 120)

    # Q1: min9 monotonicity
    print("\nQ1: min9 trend 1.35 -> 1.40 -> 1.50 -> 1.60:")
    prev_m9 = None
    peaked = False
    for row in sorted(s0_rows, key=lambda x: x["lev_scale"]):
        sc = row["lev_scale"]
        m9 = row["min9_at"] * 100
        if prev_m9 is not None:
            delta = m9 - prev_m9
            flag = "(VOL DRAG REVERSAL -- PEAK CROSSED)" if delta < 0 else ""
            if delta < 0:
                peaked = True
            print("  scale=%.2f  min9=%+.2f%%  delta=%+.3f pp  %s"
                  % (sc, m9, delta, flag))
        else:
            print("  scale=%.2f  min9=%+.2f%%  (reference)" % (sc, m9))
        prev_m9 = m9
    if not peaked:
        print("  => min9 increases monotonically through scale=1.60.")
        print("     USER HYPOTHESIS 'vol drag peak not yet reached' = CONFIRMED (provisionally).")
    else:
        print("  => min9 peaked before scale=1.60. Vol drag dominates above peak.")

    # Q2: MaxDD -50% veto
    print("\nQ2: MaxDD boundary (hard veto = -50%%):")
    veto_first_sc = None
    for row in sorted(s0_rows, key=lambda x: x["lev_scale"]):
        tag = "VETO" if row["MaxDD_FULL"] < HARD_VETO_MAXDD else "pass"
        if row["MaxDD_FULL"] < HARD_VETO_MAXDD and veto_first_sc is None:
            veto_first_sc = row["lev_scale"]
        print("  scale=%.2f  MaxDD=%+.2f%%  [%s]"
              % (row["lev_scale"], row["MaxDD_FULL"] * 100, tag))
    if veto_first_sc is None:
        print("  => MaxDD does NOT reach -50%% through scale=1.60. Veto boundary > 1.60.")
    else:
        print("  => MaxDD-50%% veto first hit at scale=%.2f." % veto_first_sc)

    # Q3: Worst5Y sign flip
    print("\nQ3: Worst5Y sign flip:")
    flipped_sc = None
    for row in sorted(s0_rows, key=lambda x: x["lev_scale"]):
        sign = "NEGATIVE" if row["Worst5Y_at"] < 0 else "positive"
        if row["Worst5Y_at"] < 0 and flipped_sc is None:
            flipped_sc = row["lev_scale"]
        print("  scale=%.2f  Worst5Y=%+.4f%%  [%s]"
              % (row["lev_scale"], row["Worst5Y_at"] * 100, sign))
    if flipped_sc is None:
        print("  => Worst5Y stays positive through scale=1.60.")
        print("     Note: Worst5Y at 1.35 was +0.33%%; close to zero at high scales.")
    else:
        print("  => Worst5Y turns NEGATIVE at scale=%.2f." % flipped_sc)

    # Q4: M6 liquidation scaling
    print("\nQ4: M6 forced-liquidation vs scale (mar=8%% and 12%%):")
    print("  %-22s | %5s | %10s | %10s | %10s | %10s"
          % ("label", "scale", "m8_nliq", "m8_maxloss%", "m12_nliq", "m12_maxloss%"))
    print("  " + "-" * 80)
    for sc in SWEEP_SCALES:
        res8  = m6_results[(sc, 0.08)]
        res12 = m6_results[(sc, 0.12)]
        lbl   = "Bext_str_sc%.2f" % sc
        print("  %-22s | %5.2f | %10d | %10.4f%% | %10d | %10.4f%%"
              % (lbl, sc, res8["n_liq"], res8["max_liq_loss_pct_AUM"],
                 res12["n_liq"], res12["max_liq_loss_pct_AUM"]))

    # Q5: Max L / >3x / worst hypothesis by scale
    print("\nQ5: Max effective leverage, >3x ratio, Black Monday AUM loss hypothesis:")
    print("  %-22s | %5s | %7s | %7s | %11s"
          % ("label", "scale", "max_L", ">3x%", "BM_AUM_loss%"))
    print("  " + "-" * 65)
    for sc in SWEEP_SCALES:
        cache     = nav_cache[sc]
        exc_n_pct = cache[8]
        in_mask_sc= cache[6]
        k365_act  = (exc_n_pct > 1e-6) & in_mask_sc
        max_exc   = float(np.max(exc_n_pct[k365_act])) if k365_act.sum() > 0 else 0.0
        bm_loss   = max_exc * BM_SINGLE_DAY_DROP * 100
        lbl = "Bext_str_sc%.2f" % sc
        print("  %-22s | %5.2f | %6.4fx | %6.2f%% | %10.4f%%"
              % (lbl, sc, cache[9], cache[11], bm_loss))

    # One-line verdict
    print("\n" + "=" * 120)
    print("ONE-LINE VERDICT")
    print("=" * 120)

    # Identify recommended ceiling based on veto + Worst5Y
    last_pass_sc  = None
    first_veto_sc = None
    for row in sorted(s0_rows, key=lambda x: x["lev_scale"]):
        if row["VETO_s0"] == 0 and row["Worst5Y_at"] >= 0:
            last_pass_sc = row["lev_scale"]
        if row["VETO_s0"] == 1 and first_veto_sc is None:
            first_veto_sc = row["lev_scale"]

    if last_pass_sc is not None:
        print("  User pref (CAGR priority / MaxDD<=50%% / 5y+10y positive):")
        print("  Recommended ceiling = scale %.2f (highest scale with VETO=0 and Worst5Y>=0)" % last_pass_sc)
    else:
        print("  All scales triggered some veto condition.")

    # Check if scale 1.60 still passes
    row_160 = next((r for r in s0_rows if abs(r["lev_scale"] - 1.60) < 0.001), None)
    if row_160 and row_160["VETO_s0"] == 0:
        print("  NOTE: scale=1.60 passes Stage-0 veto. Further sweep (1.65+) may find boundary.")
    elif row_160 and row_160["VETO_s0"] == 1:
        print("  NOTE: scale=1.60 already vetoed. Boundary is between 1.50 and 1.60.")

    print()

    # =========================================================================
    # CSV OUTPUT
    # =========================================================================
    print("Building CSV ...")
    csv_rows = []
    for row in s0_rows:
        sc    = row["lev_scale"]
        cache = nav_cache[sc]
        r_out = dict(row)
        r_out["max_L"]        = round(cache[9],  4)
        r_out["n_gt3_days"]   = cache[10]
        r_out["gt3_ratio_pct"]= round(cache[11], 2)
        # M6 columns
        for mar_rate in M6_MARGIN_RATES:
            key = (sc, mar_rate)
            res = m6_results[key]
            tag = "m%d" % int(mar_rate * 100)
            r_out["%s_n_liq" % tag]           = res["n_liq"]
            r_out["%s_n_pure_path" % tag]     = res["n_pure_path_liq"]
            r_out["%s_max_liq_loss_pct" % tag]= res["max_liq_loss_pct_AUM"]
            r_out["%s_cagr_gap_pp" % tag]     = res["cagr_gap_pp"]
            r_out["%s_min9_gap_pp" % tag]     = res["min9_gap_pp"]
            r_out["%s_mdd_gap_pp" % tag]      = res["mdd_gap_pp"]
            r_out["%s_min9_path_pct" % tag]   = res["min9_path_pct"]
            r_out["%s_mdd_path_pct" % tag]    = res["mdd_path_pct"]
            for cname in CRISIS_PERIODS:
                r_out["%s_crisis_%s" % (tag, cname)] = res["crisis_liq_counts"].get(cname, 0)
        # BM hypothesis
        exc_n_pct  = cache[8]
        in_mask_sc = cache[6]
        k365_act   = (exc_n_pct > 1e-6) & in_mask_sc
        max_exc    = float(np.max(exc_n_pct[k365_act])) if k365_act.sum() > 0 else 0.0
        r_out["bm_hypothesis_aum_loss_pct"] = round(max_exc * BM_SINGLE_DAY_DROP * 100, 4)
        r_out["bm_hypothesis_aum_loss_jpy"] = round(max_exc * BM_SINGLE_DAY_DROP * AUM_JPY, 0)
        csv_rows.append(r_out)

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "leverext_high_20260618.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f",
                                  encoding="utf-8-sig")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(csv_rows)))

    # =========================================================================
    # RETURN_BLOCK JSON
    # =========================================================================
    rb_s0 = []
    for row in s0_rows:
        sc    = row["lev_scale"]
        cache = nav_cache[sc]
        exc_n_pct  = cache[8]
        in_mask_sc = cache[6]
        k365_act   = (exc_n_pct > 1e-6) & in_mask_sc
        max_exc    = float(np.max(exc_n_pct[k365_act])) if k365_act.sum() > 0 else 0.0

        rb_row = {
            "label":            row["label"],
            "lev_scale":        sc,
            "CAGR_IS_at_pct":   round(row["CAGR_IS_at"] * 100, 4),
            "CAGR_OOS_at_pct":  round(row["CAGR_OOS_at"] * 100, 4),
            "min9_at_pct":      round(row["min9_at"] * 100, 4),
            "IS_OOS_gap_pp":    round(row["IS_OOS_gap_pp"], 4),
            "Sharpe_OOS":       round(row["Sharpe_OOS"], 4),
            "MaxDD_FULL_pct":   round(row["MaxDD_FULL"] * 100, 4),
            "Worst10Y_at_pct":  round(row["Worst10Y_star_at"] * 100, 4),
            "P10_5Y_at_pct":    round(row["P10_5Y_at"] * 100, 4),
            "Worst5Y_at_pct":   round(row["Worst5Y_at"] * 100, 4),
            "Trades_yr":        round(row["Trades_yr"], 2),
            "excess_ratio_pct": round(row["excess_ratio_pct"], 2),
            "worst_cy_pct":     round(row["worst_cy"] * 100, 4),
            "worst_cy_year":    row["worst_cy_year"],
            "max_L":            round(cache[9], 4),
            "gt3_ratio_pct":    round(cache[11], 2),
            "VETO_s0":          row["VETO_s0"],
            "veto_maxdd":       row["veto_maxdd"],
            "veto_w10y":        row["veto_w10y"],
            "bm_hypothesis_aum_loss_pct": round(max_exc * BM_SINGLE_DAY_DROP * 100, 4),
            "m6_results": {},
        }
        for mar_rate in M6_MARGIN_RATES:
            key = (sc, mar_rate)
            res = m6_results[key]
            tag = "m%d" % int(mar_rate * 100)
            rb_row["m6_results"][tag] = {
                "margin_rate_pct":       int(mar_rate * 100),
                "n_liq":                 res["n_liq"],
                "n_pure_path_liq":       res["n_pure_path_liq"],
                "max_liq_loss_pct_AUM":  res["max_liq_loss_pct_AUM"],
                "cagr_gap_pp":           res["cagr_gap_pp"],
                "min9_gap_pp":           res["min9_gap_pp"],
                "mdd_gap_pp":            res["mdd_gap_pp"],
                "min9_path_pct":         res["min9_path_pct"],
                "crisis_liq_counts":     res["crisis_liq_counts"],
            }
        rb_s0.append(rb_row)

    return_block = {
        "script":   "leverext_high_20260618.py",
        "date":     "2026-06-18",
        "sanity": {
            "min9_got_pct":   round(san_min9 * 100, 4),
            "maxdd_got_pct":  round(san_mdd  * 100, 4),
            "max_L":          round(san_max_L, 4),
            "ok_min9":        bool(ok_min9),
            "ok_maxdd":       bool(ok_maxdd),
            "ok_maxL_soft":   bool(ok_maxL),
            "SANITY_PASS":    bool(ok_min9 and ok_maxdd),
        },
        "stage0": rb_s0,
        "csv_path": csv_path,
        "bm_drop_assumed_pct": BM_SINGLE_DAY_DROP * 100,
        "aum_jpy":             AUM_JPY,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
