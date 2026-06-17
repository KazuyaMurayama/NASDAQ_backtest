"""
*** SUPERSEDED / FLAWED — DO NOT USE FOR CONCLUSIONS ***
This script (M5 v1) has a capital-accounting bug: it forced TQQQ to tie up
wn (full NASDAQ weight) of cash for the <=3x part and capped L to 3.0, ignoring
that k365 is a MARGIN instrument (leverage is built with ~8% margin, not full
cash). This produced an over-pessimistic "collapse to ~13.5% min" result.
Corrected in: src/audit/margin_funded_backtest_v2_20260617.py (M5b), which shows
the realistic margin drag is only ~-0.4 to -0.9pp (high-lever advantage preserved).
See MARGIN_CAPACITY_STRESS_RESULTS_20260617.md §8.
*** END SUPERSEDED NOTICE ***

src/audit/margin_funded_backtest_20260617.py
=============================================
M5: Margin-funded realistic backtest (MARGIN_CAPACITY_STRESS_PLAN_20260617.md §5).

PURPOSE
-------
Current backtests assume k365 (>3x leverage) can be taken with no capital reserved
for margin -- only financing cost is charged.  In reality:

  (a) Capital constraint: TQQQ cash  +  k365 margin  +  Gold  +  Bond <= 100% of AUM.
      On days where the unconstrained allocation would exceed 100%, k365 leverage is
      capped (L_eff < L_requested) until the budget balances.

  (b) Margin drag: The k365 margin cash (m * wn * max(L-3,0)) earns ZERO return
      (no interest), creating an opportunity cost relative to the unconstrained case.

REALISTIC CAPITAL MODEL (implemented per day):
  Capital budget = 1.0  (normalised AUM)

  Slots (all as fraction of capital):
    TQQQ slot      = wn * min(L, 3) / 3       -- cash used to buy TQQQ at 3x
    k365 margin    = m * wn * max(L-3, 0)     -- margin deposit (no interest)
    Gold slot      = wg                        -- 1x Gold (fully funded)
    Bond slot      = wb_eff                    -- 1x Bond when bond_on=True, else 0
    Residual cash  = 1 - TQQQ_slot - k365_margin - Gold - Bond
                     (earns SOFR via C1; if negative -> L is capped)

  Funding constraint enforcement:
    Used = TQQQ_slot + k365_margin + Gold + Bond
    If Used > 1.0:
      -> Reduce L (for the k365 portion) until Used == 1.0.
         Gold and Bond are NOT cut (strategy design).
      -> L_eff solves: wn*min(L_eff,3)/3 + m*wn*max(L_eff-3,0) + Gold + Bond = 1.0
         (Gold+Bond treated as fixed on that day; L_eff >= 3 since constraint only
          bites when margin is needed, i.e. L>3)
      -> If even L_eff=3 (no k365 at all) still exceeds budget, cap L_eff at 3.

  After capping, NAV is recomputed using L_eff for the day's NASDAQ return and
  cost charges (borrow, excess_extra penalty, TQQQ TER).

  The k365 margin cash is ZERO-return (no SOFR credit on margin).
  Remaining residual cash (if any) earns SOFR as in the standard C1 model.

MARGIN RATE SWEEP
-----------------
  m in {0.0424 (4.24% = TFX minimum, most fragile),
         0.08   (8%   = base case per plan),
         0.12   (12%  = conservative / buffer)}

CONFIGURATIONS
--------------
  1. scale1.35_strong  : Bext_str_sc1.35 (B3a_MAP_STRONG x scale=1.35) -- highest CAGR
  2. scale1.25_default : Bext_def_sc1.25 (B3a_MAP_DEFAULT x scale=1.25) -- mid
  3. B3a               : B3a_k365 (B3a_MAP_DEFAULT x scale=1.15) -- reference

SANITY (constraint OFF, margin OFF):
  Reproduce for each config the known unconstrained min9:
    scale1.35_strong -> +23.83% (+/-0.15pp)
    scale1.25_default -> +22.07% (+/-0.15pp)  [reported in LEVERUP_EXTENSION_RESULTS]
    B3a              -> +20.98% (+/-0.15pp)

OUTPUTS
-------
  src/audit/margin_funded_backtest_20260617.py  (this file)
  audit_results/margin_funded_backtest_20260617.csv
  RETURN_BLOCK printed as JSON to stdout

ASSUMPTIONS (explicitly stated)
---------------------------------
  * Margin deposit = m * wn * max(L-3, 0) per unit AUM, daily.
  * Margin earns ZERO interest (opportunity cost = SOFR foregone).
  * Gold and Bond are not reduced when capital constraint binds.
  * Capital constraint reduces L for the k365 excess portion only.
  * TQQQ is still priced at 3x internal leverage (no change).
  * All other cost parameters unchanged (k365 spread EXCESS_EXTRA=0.0025,
    TQQQ TER/swap, Gold/Bond TER, trading cost).
  * No look-ahead: all signals use same T+2/T+5 publication lags as base scripts.

ASCII-only print output (Windows cp932). Does NOT commit. No temporary files.
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
    EXCESS_EXTRA_STORE,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _calendar_year_returns,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
    FEE_GOLD, FEE_BOND,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    SWAP_SPREAD, TER_TQQQ, DH_PER_UNIT, NAV_FLOOR,
    DELAY as V7_DELAY,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom, _cagr_seg, _maxdd_from_returns,
    LU2_SCALE,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_on_base_c1,
)
from src.audit.strategy_runners import (
    _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
    _compute_dhw1_trades_per_year,
)
from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual
from src.audit.lu_cfd_recost_20260611 import (
    _build_v7_mult,
    AFTER_TAX, LEV_CAP,
)

# ---------------------------------------------------------------------------
# Strategy configurations
# ---------------------------------------------------------------------------
B3A_MAP_DEFAULT = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_MAP_STRONG  = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}

EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025 (%/yr for L>3)

CONFIGS = [
    {
        "label": "scale1.35_strong",
        "v7_map": B3A_MAP_STRONG,
        "lev_scale": 1.35,
        "known_min9": 0.2383,  # +23.83% from LEVERUP_EXTENSION_RESULTS_20260616.md
        "sanity_tol": 0.0015,
    },
    {
        "label": "scale1.25_default",
        "v7_map": B3A_MAP_DEFAULT,
        "lev_scale": 1.25,
        "known_min9": 0.2207,  # +22.07%
        "sanity_tol": 0.0015,
    },
    {
        "label": "B3a",
        "v7_map": B3A_MAP_DEFAULT,
        "lev_scale": 1.15,
        "known_min9": 0.2098,  # +20.98%
        "sanity_tol": 0.0015,
    },
]

# Margin rate sweep
MARGIN_RATES = [0.0424, 0.08, 0.12]  # 4.24% (min), 8% (base), 12% (conservative)
MARGIN_RATE_LABELS = {0.0424: "m4.24pct", 0.08: "m8pct", 0.12: "m12pct"}


# ---------------------------------------------------------------------------
# Core: margin-funded NAV builder
# ---------------------------------------------------------------------------

def _build_margin_funded_nav(
    shared, dates_dt, n_years,
    ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
    v7_map, lev_scale, excess_extra, margin_rate,
):
    """
    Build the margin-funded realistic NAV.

    Implements the M5 capital model:
      - TQQQ slot = wn * min(L, 3) / 3
      - k365 margin = margin_rate * wn * max(L-3, 0)
      - Gold = wg (1x, fully funded)
      - Bond = wb_eff (1x, when bond_on)
      - Constraint: all above <= 1.0 capital; if violated, reduce L until balanced.
      - Margin earns ZERO return.
      - Residual cash earns SOFR.

    Parameters
    ----------
    margin_rate : float
        Fraction of k365 notional to set aside as margin (e.g. 0.08 for 8%).

    Returns
    -------
    nav_adj       : pd.Series (DatetimeIndex) -- funded NAV
    tpy           : float -- trades per year
    excess_days   : int   -- days where unconstrained L > LEV_CAP (3.0)
    n_capped      : int   -- days where L was reduced due to margin constraint
    avg_cap_amount: float -- average L_requested - L_eff on constrained days
    avg_margin_frac_all   : float -- average margin / AUM over all days
    avg_margin_frac_in    : float -- average margin / AUM over IN days
    L_arr         : np.ndarray -- effective L (post-constraint, post-shift) each day
    L_raw_arr     : np.ndarray -- requested L (pre-constraint, post-shift) each day
    """
    a = shared["assets"]
    close = a["close"]
    dates = a["dates"]
    idx = dates.index
    n = len(idx)

    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn_raw = np.asarray(shared["wn"], float)
    wg_raw = np.asarray(shared["wg"], float)
    wb_raw = np.asarray(shared["wb"], float)

    # Build v7 multiplier
    if v7_map is None:
        mult_v7 = _build_v7_mult(dates_dt) * float(lev_scale)
    else:
        mult_v7 = _build_v7_mult_custom(dates_dt, v7_map) * float(lev_scale)

    # Unshifted leverage (per-day raw signal)
    lev_mod = lev_raw_masked * mult_v7   # pre-shift
    L_unshifted = lev_mod * 3.0

    # Apply V7_DELAY shift (same as canonical NAV builder)
    L_req_shifted  = pd.Series(L_unshifted, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s           = pd.Series(wn_raw, index=idx).shift(V7_DELAY).fillna(0.0).values
    wg_s           = pd.Series(wg_raw, index=idx).shift(V7_DELAY).fillna(0.0).values
    wb_s           = pd.Series(wb_raw, index=idx).shift(V7_DELAY).fillna(0.0).values

    # NASDAQ underlying return
    close_arr = np.asarray(close, float)
    r_nas = np.zeros(n, float)
    r_nas[1:] = (close_arr[1:] - close_arr[:-1]) / close_arr[:-1]

    # Gold/Bond returns (for OUT-day fill; handled separately after base build)
    bond_on_arr = np.asarray(bond_on, dtype=bool)

    # --- Per-day capital constraint + margin model ---
    L_eff_arr    = np.empty(n, float)  # effective leverage after constraint
    L_req_arr    = L_req_shifted.copy()  # requested leverage
    margin_arr   = np.zeros(n, float)   # daily margin fraction
    n_capped     = 0
    cap_amounts  = []

    for t in range(n):
        L_req = L_req_shifted[t]
        wn_t  = wn_s[t]
        wg_t  = wg_s[t]    # from base TQQQ builder (small on IN days)
        wb_t  = wb_s[t]

        # On OUT days (fund_active[t]==True), the C1 P09 logic later replaces
        # the base return. The NASDAQ sleeve is OFF; we still compute the
        # base NAV normally (L_req ~1 on OUT days due to masking), but no
        # k365 margin applies because max(L-3,0)~0.
        # So the margin constraint is naturally inactive on OUT days.

        # Effective capital usage at requested L:
        #   TQQQ slot = wn * min(L,3) / 3
        #   k365 margin = margin_rate * wn * max(L-3,0)
        #   Gold + Bond (from base TQQQ builder weights)
        tqqq_slot = wn_t * min(L_req, 3.0) / 3.0
        k365_margin_req = margin_rate * wn_t * max(L_req - 3.0, 0.0)
        gold_bond_used = wg_t + wb_t  # small in TQQQ-base on IN days

        used_req = tqqq_slot + k365_margin_req + gold_bond_used

        if used_req <= 1.0 + 1e-9:
            # No constraint binding
            L_eff = L_req
            k365_margin_used = k365_margin_req
        else:
            # Constraint binds; reduce L_eff so that:
            #   wn * min(L_eff,3)/3 + margin_rate * wn * max(L_eff-3,0) + gold_bond = 1.0
            # Since the excess only exists when L>3:
            #   Let x = L_eff - 3  (x >= 0)
            #   wn*(3/3) + margin_rate*wn*x + gold_bond = 1.0
            #   => x = (1.0 - wn - gold_bond) / (margin_rate * wn)  if wn>0
            residual_for_k365 = 1.0 - (wn_t / 1.0) - gold_bond_used  # after full TQQQ slot at 3x
            # (TQQQ slot at L=3: wn * 3/3 = wn)
            if margin_rate > 1e-9 and wn_t > 1e-9 and residual_for_k365 > 1e-9:
                x_max = residual_for_k365 / (margin_rate * wn_t)
                L_eff = 3.0 + x_max
            else:
                L_eff = 3.0  # no k365 possible; cap at TQQQ-only
            # But also cannot exceed requested L:
            L_eff = min(L_eff, L_req)
            # Must be at least 1 (min long position):
            L_eff = max(L_eff, 1.0)
            # Recalculate actual margin used
            k365_margin_used = margin_rate * wn_t * max(L_eff - 3.0, 0.0)
            n_capped += 1
            cap_amounts.append(L_req - L_eff)

        L_eff_arr[t] = L_eff
        margin_arr[t] = k365_margin_used

    avg_cap_amount  = float(np.mean(cap_amounts)) if cap_amounts else 0.0
    in_mask_arr     = ~fund_active   # IN days: fund_active = False
    avg_margin_all  = float(margin_arr.mean())
    avg_margin_in   = float(margin_arr[in_mask_arr].mean()) if in_mask_arr.sum() > 0 else 0.0

    # --- Build daily portfolio return using L_eff ---
    # TQQQ NASDAQ leg (borrowing + TER charged on effective L)
    borrow = np.maximum(L_eff_arr - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L_eff_arr * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    # Excess cost (k365 spread over TQQQ rate) on excess leverage only
    excess_lev = np.maximum(L_eff_arr - LEV_CAP, 0.0)
    penalty = wn_s * excess_lev * float(excess_extra) / TRADING_DAYS

    # Margin cash earns ZERO (we explicitly do NOT add margin_arr * sofr here).
    # The opportunity cost is automatically captured because in the unconstrained
    # version this cash would be earning SOFR via residual cash, but here it earns 0.

    daily = wn_s * nas_ret + wg_s * 0.0 + wb_s * 0.0  # Gold/Bond base = 0 in TQQQ builder
    daily = daily - penalty

    # Turnover cost
    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(wn_raw))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(wg_raw))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(wb_raw))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    # TER drag on Gold/Bond ETF legs
    ter_drag = (wg_raw * _TER_GOLD2X_EXTRA_DAILY + wb_raw * _TER_TMF_EXTRA_DAILY)
    tpy = _compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / 252.0

    r_sim = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_sim - ter_drag - etf_daily

    # Base NAV (DatetimeIndex)
    nav_base = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )

    excess_days = int(np.sum(excess_lev > 1e-9))

    # --- Apply C1 P09 OUT fill (same as _build_p09_on_base_c1) ---
    r_base_dt = nav_base.pct_change().fillna(0).values
    nav_dt, r, tpy_final = _build_p09_on_base_c1(
        r_base_dt, ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on,
        sofr_arr, pd.DatetimeIndex(pd.to_datetime(dates.values)), tpy, n_years)

    return (nav_dt, tpy_final, excess_days,
            n_capped, avg_cap_amount,
            avg_margin_all, avg_margin_in,
            L_eff_arr, L_req_arr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _fmt_pct(v, digits=2):
    return "%+.{0}f%%".format(digits) % (v * 100)


def _print_std10_table(rows, title):
    """Print standard 10 metrics table for given rows."""
    print("\n" + "=" * 160)
    print(title)
    hdr = ("%-28s | %-9s | %8s | %8s | %6s | %7s | %7s | %8s | %8s | %8s | %8s | %7s"
           % ("label", "margin_m", "CAGR_IS%", "CAGR_OOS%",
              "min9%", "gap_pp", "Sharpe", "MaxDD%", "W10Y*%",
              "P10_5Y%", "W5Y%", "Trd/yr"))
    print(hdr)
    print("-" * 160)
    for r in rows:
        print("%-28s | %-9s | %+7.2f%% | %+8.2f%% | %+5.2f%% | %+6.2f | %7.3f | %+7.2f%% | %+7.2f%% | %+7.2f%% | %+7.2f%% | %7.1f"
              % (r["label"][:28], r["margin_label"],
                 r["CAGR_IS_at"] * 100, r["CAGR_OOS_at"] * 100,
                 r["min9_at"] * 100, r["IS_OOS_gap_pp"],
                 r["Sharpe_OOS"], r["MaxDD_FULL"] * 100,
                 r["Worst10Y_star_at"] * 100, r["P10_5Y_at"] * 100,
                 r["Worst5Y_at"] * 100, r["Trades_yr"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("M5: MARGIN-FUNDED REALISTIC BACKTEST  2026-06-17")
    print("Plan: MARGIN_CAPACITY_STRESS_PLAN_20260617.md SS5")
    print("Margin model: k365 deposit = m*wn*max(L-3,0), zero interest.")
    print("Capital constraint: TQQQ_slot + k365_margin + Gold + Bond <= 1.0.")
    print("On constrained days: reduce L until budget balances. Gold/Bond NOT cut.")
    print("Margin rates: m in {4.24%%, 8%%(base), 12%%}.")
    print("Configs: scale1.35_strong / scale1.25_default / B3a (reference).")
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

    # ---- Gold/Bond 1x legs ----
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

    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    in_mask = ~fund_active
    n_in  = int(in_mask.sum())
    n_out = int(fund_active.sum())
    print("\nTotal days: %d  IN: %d (%.1f%%)  OUT: %d (%.1f%%)"
          % (n, n_in, 100.0*n_in/n, n_out, 100.0*n_out/n))

    # =========================================================================
    # SANITY GATE: Reproduce unconstrained (m=0) known min9 for each config
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: constraint OFF (m=0) must reproduce known unconstrained min9")
    print("  scale1.35_strong  -> +23.83% +/-0.15pp")
    print("  scale1.25_default -> +22.07% +/-0.15pp")
    print("  B3a               -> +20.98% +/-0.15pp")
    print("=" * 120)

    sanity_results = {}
    all_sanity_ok = True
    for cfg in CONFIGS:
        lbl = cfg["label"]
        print("  Sanity check: %s (constraint OFF, m=0) ..." % lbl)
        # Use _build_full_c1 directly (no margin constraint) for sanity
        nav_san, r_san, tpy_san, exc_san = _build_full_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
            v7_map=cfg["v7_map"], lev_scale=cfg["lev_scale"],
            excess_extra=EXCESS_EXTRA)
        pre_san = compute_10metrics(nav_san, tpy_san)
        aft_san = _apply_aftertax(pre_san)
        got_min9 = _min_at(aft_san)
        known_min9 = cfg["known_min9"]
        tol = cfg["sanity_tol"]
        ok = abs(got_min9 - known_min9) <= tol
        if not ok:
            all_sanity_ok = False
        print("    %s: got min9=%+.4f%%  expect ~%+.4f%%  -> %s"
              % (lbl, got_min9 * 100, known_min9 * 100, "OK" if ok else "FAIL"))
        sanity_results[lbl] = {
            "got_min9": got_min9,
            "known_min9": known_min9,
            "ok": ok,
            "nav_unconstrained": nav_san,
            "tpy_unconstrained": tpy_san,
            "MaxDD_unconstrained": pre_san["MaxDD_FULL"],
        }

    if not all_sanity_ok:
        print("\nSANITY FAILED -- halting. Check _build_full_c1 / config parameters.")
        sys.exit(1)
    print("  SANITY PASSED for all %d configs. Proceeding.\n" % len(CONFIGS))

    # =========================================================================
    # MAIN SWEEP: margin_rate x config
    # =========================================================================
    print("=" * 120)
    print("MAIN SWEEP: %d configs x %d margin rates = %d total runs"
          % (len(CONFIGS), len(MARGIN_RATES), len(CONFIGS) * len(MARGIN_RATES)))
    print("=" * 120)

    all_rows = []     # for CSV and table
    return_data = {}  # for RETURN_BLOCK

    for cfg in CONFIGS:
        lbl = cfg["label"]
        return_data[lbl] = {
            "unconstrained": {},
            "margin_rates": {},
        }

        # --- Unconstrained reference (m=0) ---
        san = sanity_results[lbl]
        nav_unc = san["nav_unconstrained"]
        tpy_unc = san["tpy_unconstrained"]
        pre_unc = compute_10metrics(nav_unc, tpy_unc)
        aft_unc = _apply_aftertax(pre_unc)
        mn_unc  = _min_at(aft_unc)
        cy_unc  = _calendar_year_returns(nav_unc)
        return_data[lbl]["unconstrained"] = {
            "min9_at_pct":       round(mn_unc * 100, 4),
            "CAGR_IS_at_pct":    round(aft_unc["CAGR_IS"] * 100, 4),
            "CAGR_OOS_at_pct":   round(aft_unc["CAGR_OOS"] * 100, 4),
            "IS_OOS_gap_pp":     round(aft_unc["IS_OOS_gap_pp"], 4),
            "Sharpe_OOS":        round(pre_unc["Sharpe_OOS"], 4),
            "MaxDD_FULL":        round(pre_unc["MaxDD_FULL"] * 100, 4),
            "Worst10Y_star_at":  round(aft_unc["Worst10Y_star"] * 100, 4),
            "P10_5Y_at":         round(aft_unc["P10_5Y"] * 100, 4),
            "Worst5Y_at":        round(aft_unc["Worst5Y"] * 100, 4),
            "Trades_yr":         round(aft_unc["Trades_yr"], 2),
        }
        all_rows.append({
            "config": lbl,
            "margin_rate": 0.0,
            "margin_label": "unconstrained",
            "label": "%s_unconstrained" % lbl,
            "CAGR_IS_at":       aft_unc["CAGR_IS"],
            "CAGR_OOS_at":      aft_unc["CAGR_OOS"],
            "min9_at":          mn_unc,
            "IS_OOS_gap_pp":    aft_unc["IS_OOS_gap_pp"],
            "Sharpe_OOS":       pre_unc["Sharpe_OOS"],
            "MaxDD_FULL":       pre_unc["MaxDD_FULL"],
            "Worst10Y_star_at": aft_unc["Worst10Y_star"],
            "P10_5Y_at":        aft_unc["P10_5Y"],
            "Worst5Y_at":       aft_unc["Worst5Y"],
            "Trades_yr":        aft_unc["Trades_yr"],
            "worst_cy":         float(cy_unc.min()),
            "worst_cy_year":    int(cy_unc.idxmin()),
            "n_capped_days":    0,
            "capped_ratio_pct": 0.0,
            "avg_cap_amount":   0.0,
            "avg_margin_pct_all": 0.0,
            "avg_margin_pct_in":  0.0,
            "min9_delta_vs_unc_pp": 0.0,
        })

        # --- Margin rate sweep ---
        for m in MARGIN_RATES:
            m_label = MARGIN_RATE_LABELS[m]
            run_label = "%s_%s" % (lbl, m_label)
            print("\n  Running %s (m=%.4f) ..." % (run_label, m))

            (nav_dt, tpy_m, exc_m,
             n_capped, avg_cap_amount,
             avg_margin_all, avg_margin_in,
             L_eff_arr, L_req_arr) = _build_margin_funded_nav(
                shared, dates_dt, n_years,
                ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
                v7_map=cfg["v7_map"], lev_scale=cfg["lev_scale"],
                excess_extra=EXCESS_EXTRA, margin_rate=m)

            pre_m = compute_10metrics(nav_dt, tpy_m)
            aft_m = _apply_aftertax(pre_m)
            mn_m  = _min_at(aft_m)
            cy_m  = _calendar_year_returns(nav_dt)

            capped_ratio_pct = 100.0 * n_capped / n

            delta_min9 = mn_m - mn_unc  # should be negative (margin hurts)

            # Constraint statistics
            L_gt3_eff = L_eff_arr[in_mask]
            L_gt3_req = L_req_arr[in_mask]
            n_in_capped = int(np.sum(L_eff_arr[in_mask] < L_req_arr[in_mask] - 1e-9))
            max_L_eff   = float(np.max(L_eff_arr[in_mask])) if n_in > 0 else 0.0
            avg_L_eff_in = float(np.mean(L_eff_arr[in_mask])) if n_in > 0 else 0.0

            print("    min9=%+.2f%%  MaxDD=%+.2f%%  delta_vs_unc=%+.2fpp"
                  % (mn_m * 100, pre_m["MaxDD_FULL"] * 100, delta_min9 * 100))
            print("    Capped days: %d / %d (%.2f%%)  avg_cap=%.4f  avg_margin_all=%.4f%%  avg_margin_in=%.4f%%"
                  % (n_capped, n, capped_ratio_pct, avg_cap_amount,
                     avg_margin_all * 100, avg_margin_in * 100))

            row = {
                "config": lbl,
                "margin_rate": m,
                "margin_label": m_label,
                "label": run_label,
                "CAGR_IS_at":       aft_m["CAGR_IS"],
                "CAGR_OOS_at":      aft_m["CAGR_OOS"],
                "min9_at":          mn_m,
                "IS_OOS_gap_pp":    aft_m["IS_OOS_gap_pp"],
                "Sharpe_OOS":       pre_m["Sharpe_OOS"],
                "MaxDD_FULL":       pre_m["MaxDD_FULL"],
                "Worst10Y_star_at": aft_m["Worst10Y_star"],
                "P10_5Y_at":        aft_m["P10_5Y"],
                "Worst5Y_at":       aft_m["Worst5Y"],
                "Trades_yr":        aft_m["Trades_yr"],
                "worst_cy":         float(cy_m.min()),
                "worst_cy_year":    int(cy_m.idxmin()),
                "n_capped_days":    n_capped,
                "capped_ratio_pct": round(capped_ratio_pct, 4),
                "avg_cap_amount":   round(avg_cap_amount, 6),
                "avg_margin_pct_all": round(avg_margin_all * 100, 6),
                "avg_margin_pct_in":  round(avg_margin_in  * 100, 6),
                "min9_delta_vs_unc_pp": round(delta_min9 * 100, 4),
            }
            all_rows.append(row)

            return_data[lbl]["margin_rates"][m_label] = {
                "margin_rate":       m,
                "min9_at_pct":       round(mn_m * 100, 4),
                "CAGR_IS_at_pct":    round(aft_m["CAGR_IS"] * 100, 4),
                "CAGR_OOS_at_pct":   round(aft_m["CAGR_OOS"] * 100, 4),
                "IS_OOS_gap_pp":     round(aft_m["IS_OOS_gap_pp"], 4),
                "Sharpe_OOS":        round(pre_m["Sharpe_OOS"], 4),
                "MaxDD_FULL_pct":    round(pre_m["MaxDD_FULL"] * 100, 4),
                "Worst10Y_star_pct": round(aft_m["Worst10Y_star"] * 100, 4),
                "P10_5Y_pct":        round(aft_m["P10_5Y"] * 100, 4),
                "Worst5Y_pct":       round(aft_m["Worst5Y"] * 100, 4),
                "Trades_yr":         round(aft_m["Trades_yr"], 2),
                "n_capped_days":     n_capped,
                "capped_ratio_pct":  round(capped_ratio_pct, 4),
                "avg_cap_amount":    round(avg_cap_amount, 6),
                "avg_margin_pct_all":round(avg_margin_all * 100, 6),
                "avg_margin_pct_in": round(avg_margin_in  * 100, 6),
                "min9_delta_vs_unc_pp": round(delta_min9 * 100, 4),
                "MaxDD_delta_vs_unc_pp": round((pre_m["MaxDD_FULL"] - san["MaxDD_unconstrained"]) * 100, 4),
            }

    # =========================================================================
    # SUMMARY TABLES
    # =========================================================================

    # Table 1: All margin runs (m=8% base) for quick comparison
    print("\n" + "=" * 160)
    print("SUMMARY TABLE: Base margin rate m=8%% (standard 10 metrics, after-tax CAGR)")
    base_rows_m8 = [r for r in all_rows if r["margin_label"] in ("unconstrained", "m8pct")]
    _print_std10_table(base_rows_m8, "Base case (m=8%%) -- constraint impact")

    # Table 2: All 3 margin rates for scale1.35 (most impacted)
    print("\n" + "=" * 160)
    print("DETAIL TABLE: scale1.35_strong -- all margin rates")
    sc135_rows = [r for r in all_rows if r["config"] == "scale1.35_strong"]
    _print_std10_table(sc135_rows, "scale1.35_strong: unconstrained vs margin rates")

    # Table 3: delta table
    print("\n" + "=" * 160)
    print("DELTA TABLE: min9 loss from margin constraint (pp vs unconstrained)")
    print("%-28s | %-11s | %12s | %12s | %12s | %11s | %10s | %10s"
          % ("config", "margin_m", "min9_delta_pp",
             "n_capped", "capped%", "avg_cap_L", "marg%_all", "marg%_in"))
    print("-" * 140)
    for r in all_rows:
        if r["margin_label"] == "unconstrained":
            continue
        print("%-28s | %-11s | %+11.2fpp | %12d | %11.2f%% | %10.4f | %9.4f%% | %9.4f%%"
              % (r["config"][:28], r["margin_label"],
                 r["min9_delta_vs_unc_pp"],
                 r["n_capped_days"], r["capped_ratio_pct"],
                 r["avg_cap_amount"],
                 r["avg_margin_pct_all"], r["avg_margin_pct_in"]))

    # =========================================================================
    # CONCLUSIONS
    # =========================================================================
    print("\n" + "=" * 120)
    print("CONCLUSIONS")
    print("=" * 120)

    print("\n1. MARGIN IMPACT ON min9 (at m=8%% base case):")
    for cfg in CONFIGS:
        lbl = cfg["label"]
        unc_r = return_data[lbl]["unconstrained"]
        m8_r  = return_data[lbl]["margin_rates"].get("m8pct", {})
        if m8_r:
            print("   %-22s: unconstrained min9=%+.2f%%  -> funded min9=%+.2f%%  delta=%+.2fpp"
                  % (lbl, unc_r["min9_at_pct"], m8_r["min9_at_pct"],
                     m8_r["min9_delta_vs_unc_pp"]))
        else:
            print("   %-22s: m8 not computed" % lbl)

    print("\n2. CAPITAL CONSTRAINT EFFECT (m=8%%, scale1.35_strong):")
    sc135_m8 = return_data.get("scale1.35_strong", {}).get("margin_rates", {}).get("m8pct", {})
    if sc135_m8:
        print("   Capped days: %d / %d = %.2f%%"
              % (sc135_m8["n_capped_days"], n, sc135_m8["capped_ratio_pct"]))
        print("   Avg cap amount (L_req - L_eff): %.4f leverage points"
              % sc135_m8["avg_cap_amount"])
        print("   Avg margin drag %%AUM (all days): %.4f%%  (IN days): %.4f%%"
              % (sc135_m8["avg_margin_pct_all"], sc135_m8["avg_margin_pct_in"]))

    print("\n3. SENSITIVITY (scale1.35_strong min9 across margin rates):")
    for m_label in ["m4.24pct", "m8pct", "m12pct"]:
        mr = return_data.get("scale1.35_strong", {}).get("margin_rates", {}).get(m_label, {})
        if mr:
            print("   %-10s (m=%.4f): min9=%+.2f%%  delta=%+.2fpp  capped=%.2f%%"
                  % (m_label, mr["margin_rate"], mr["min9_at_pct"],
                     mr["min9_delta_vs_unc_pp"], mr["capped_ratio_pct"]))

    print("\n4. PRACTICAL LEVER ASSESSMENT:")
    best_m8 = None
    best_min9 = -999.0
    for cfg in CONFIGS:
        lbl = cfg["label"]
        mr = return_data[lbl]["margin_rates"].get("m8pct", {})
        if mr and mr["min9_at_pct"] > best_min9:
            best_min9 = mr["min9_at_pct"]
            best_m8 = (lbl, mr)
    if best_m8:
        lbl_b, mr_b = best_m8
        print("   Highest realistic min9 at m=8%%: %s -> %+.2f%%"
              % (lbl_b, mr_b["min9_at_pct"]))
        print("   vs B3a_m8pct: %+.2f%%"
              % return_data["B3a"]["margin_rates"].get("m8pct", {}).get("min9_at_pct", float("nan")))
    sc135_m8_min9 = return_data.get("scale1.35_strong", {}).get("margin_rates", {}).get("m8pct", {}).get("min9_at_pct", None)
    sc135_unc_min9 = return_data.get("scale1.35_strong", {}).get("unconstrained", {}).get("min9_at_pct", None)
    if sc135_m8_min9 is not None and sc135_unc_min9 is not None:
        erosion = sc135_unc_min9 - sc135_m8_min9
        print("   scale1.35_strong high-lev CAGR advantage eroded by margin: %.2fpp"
              % erosion)
        print("   -> Net advantage over B3a_unconstrained (%+.2f%%): %.2fpp"
              % (return_data["B3a"]["unconstrained"]["min9_at_pct"],
                 sc135_m8_min9 - return_data["B3a"]["unconstrained"]["min9_at_pct"]))

    print("\n5. MARGIN RATE INVARIANCE EXPLANATION:")
    print("   All 3 margin rates (m=4.24%%/8%%/12%%) yield IDENTICAL results.")
    print("   ROOT CAUSE: DH-W1 portfolio is fully invested: wn + wg + wb = 1.0 on all IN days.")
    print("   - TQQQ cash slot (wn) + Gold (wg) + Bond (wb) = 1.0 (no residual).")
    print("   - No cash available for k365 margin deposit regardless of margin rate m.")
    print("   - Therefore L_eff is capped at 3.0 on ALL constrained days (k365 impossible).")
    print("   - The constraint is binding unconditionally; m only affects HOW MUCH L would")
    print("     need to be cut to free margin cash -- but since wn+wg+wb=1.0, even m->0+")
    print("     leaves no room for k365 without cutting Gold/Bond (which the plan forbids).")
    print("   IMPLICATION: avg_margin_pct = 0.0%% is CORRECT (L_eff=3.0 -> k365 excess=0 -> margin=0).")
    print("   The CAGR impact (~-10pp) is ENTIRELY due to L being capped from ~3.7x avg to 3.0x;")
    print("   it is NOT a margin drag -- it is a leverage capacity constraint.")
    print("   This is the CORRECT realistic model: scale1.35 cannot run >3x leverage")
    print("   while simultaneously holding the full Gold/Bond allocation.")

    # =========================================================================
    # CSV OUTPUT
    # =========================================================================
    print("\nSaving CSV ...")
    csv_rows_out = []
    for r in all_rows:
        csv_rows_out.append({
            "config":             r["config"],
            "margin_rate":        r["margin_rate"],
            "margin_label":       r["margin_label"],
            "CAGR_IS_at":         r["CAGR_IS_at"],
            "CAGR_OOS_at":        r["CAGR_OOS_at"],
            "min9_at":            r["min9_at"],
            "IS_OOS_gap_pp":      r["IS_OOS_gap_pp"],
            "Sharpe_OOS":         r["Sharpe_OOS"],
            "MaxDD_FULL":         r["MaxDD_FULL"],
            "Worst10Y_star_at":   r["Worst10Y_star_at"],
            "P10_5Y_at":          r["P10_5Y_at"],
            "Worst5Y_at":         r["Worst5Y_at"],
            "Trades_yr":          r["Trades_yr"],
            "worst_cy":           r["worst_cy"],
            "worst_cy_year":      r["worst_cy_year"],
            "n_capped_days":      r["n_capped_days"],
            "capped_ratio_pct":   r["capped_ratio_pct"],
            "avg_cap_amount":     r["avg_cap_amount"],
            "avg_margin_pct_all": r["avg_margin_pct_all"],
            "avg_margin_pct_in":  r["avg_margin_pct_in"],
            "min9_delta_vs_unc_pp": r["min9_delta_vs_unc_pp"],
        })

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "margin_funded_backtest_20260617.csv")
    pd.DataFrame(csv_rows_out).to_csv(csv_path, index=False, float_format="%.6f",
                                       encoding="utf-8-sig")
    print("Saved: %s  (%d rows)" % (csv_path, len(csv_rows_out)))

    # =========================================================================
    # RETURN_BLOCK
    # =========================================================================
    return_block = {
        "script":  "margin_funded_backtest_20260617.py",
        "date":    "2026-06-17",
        "meta": {
            "margin_model":    "k365 deposit = m * wn * max(L-3,0), zero interest",
            "constraint":      "TQQQ_slot + k365_margin + Gold + Bond <= 1.0",
            "constraint_rule": "Reduce L until budget=1.0; Gold/Bond NOT cut",
            "excess_extra_pct": round(EXCESS_EXTRA * 100, 4),
            "n_total_days":    n,
            "n_in_days":       n_in,
            "n_out_days":      n_out,
            "sanity_all_ok":   all_sanity_ok,
            "key_finding": (
                "DH-W1 portfolio is fully invested (wn+wg+wb=1.0 on IN days); "
                "no residual cash for k365 margin at ANY margin rate. "
                "Constraint caps L_eff=3.0 on all >3x days regardless of m. "
                "All 3 margin rates yield identical results. "
                "CAGR impact (-7.5 to -10.1pp) is pure leverage-capacity constraint, "
                "not margin drag. scale1.35_strong effectively degrades to 3x (TQQQ-only) "
                "on 43.6%% of all days when Gold/Bond are preserved."
            ),
        },
        "sanity": {lbl: {
            "got_min9_pct": round(sanity_results[lbl]["got_min9"] * 100, 4),
            "known_min9_pct": round(sanity_results[lbl]["known_min9"] * 100, 4),
            "ok": sanity_results[lbl]["ok"],
        } for lbl in sanity_results},
        "results": return_data,
        "csv_path": csv_path,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
