# -*- coding: cp932 -*-
"""
src/audit/combine_g4_lt2_20260615.py
=====================================
G4: LT2-N750 long-cycle contrarian signal native-integrated into B3a_k365.

Purpose (MULTISTRATEGY_COMBINE_PLAN_20260615.md §1 G4):
  Transplant LT2-N750 contrarian leverage multiplier from CFD/S2 base to
  the B3a_k365 TQQQ base.  Lesson E requires independent re-evaluation --
  do NOT assume CFD-base performance carries over.

Integration method:
  B3a_k365 IN-period effective leverage =
      lev_raw_masked * mult_v7 * lev_scale
  where:
      mult_v7   = V7 map {Q0:1.40, Q1:1.40, Q2:1.05, Q3:1.00}  (per quartile)
      lev_scale = 1.15

  G4 adds a third factor: LT2 contrarian multiplier (Mode A):
      lt_mult = (1 - k_lt * lt_sig).clip(0.5, 1.5)    [Mode A: signal_to_mult]

  Final IN-period leverage:
      L_eff = (lev_raw_masked * mult_v7 * lev_scale * lt_mult) * 3.0

  Contrarian direction: high long-run momentum (lt_sig > 0) -> lt_mult < 1
  -> reduce NASDAQ exposure.  Low momentum -> lt_mult > 1 -> increase exposure.
  This matches the original LT2 modeA intent (long-cycle contrarian rebalancing).

  Causality: LT2 signal at t uses close[t-N] / close[t] momentum (N=750),
  plus a 2N rolling mu/sigma.  All inputs are strictly backward-looking.
  The v7/lev_scale shift by V7_DELAY=2 is preserved (same as B3a parent).

Sanity anchor:
  When k_lt=0.0 (lt_mult=1.0 always), the result must reproduce B3a_k365:
    min9 +20.98%, MaxDD -38.20%, Sharpe 0.904  (tolerance +-0.05pp on min9)

k_lt sweep:
  Two levels tested: k_lt in {0.30, 0.20}
    - 0.30 = moderate contrarian weight (default from long_cycle_signal range)
    - 0.20 = weaker contrarian weight

Early-close condition (plan §6):
  If min9 < B3a_k365_min9 - 1.0pp for ALL k_lt values, report EARLY_CLOSE
  with "Lesson E: LT2 degraded on TQQQ base" and stop.

Outputs:
  src/audit/combine_g4_lt2_20260615.py     -- this script
  audit_results/combine_g4_lt2_20260615.csv
  RETURN_BLOCK JSON printed to stdout

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.

Reuses (imports only):
  k365_recost_20260612  : _build_tqqq_base_param, _build_full_c1,
                          _build_p09_on_base_c1, EXCESS_EXTRA_K365_CENTRE
  lu_cfd_recost_20260611: LU1_MAP, AFTER_TAX
  leverup_b1c1_20260612 : _build_p09_nav_c1
  unified_metrics       : compute_10metrics, IS_END, OOS_START
  long_cycle_signal     : compute_lt2, signal_to_mult
  run_p01_backtest_20260611 : _apply_aftertax, _calendar_year_returns,
                              _ret_from_nav_level, _inverse_vol_weights,
                              LAG_DAYS, TRADING_DAYS
  run_p02_p09_backtest_20260611 : GATE_DELAY, _load_macro_signal
  strategy_runners      : _load_dhw1_shared, _DHW1_SHARED
  cost_model_cfd_vs_tqqq_20260611 : _build_v7_mult, DELAY as V7_DELAY,
                                     DH_PER_UNIT, NAV_FLOOR
  run_p09_tqqq_validate_20260611 : _build_v7_mult_custom, LU2_SCALE
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
_SRC_DIR  = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

# ---- Strategy infrastructure imports ----------------------------------------
import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.lu_cfd_recost_20260611 import (
    LU1_MAP, AFTER_TAX, SWAP_SPREAD, TER_TQQQ, LEV_CAP,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_nav_c1, _build_p09_on_base_c1,
    LAG_DAYS, TRADING_DAYS,
)
from src.audit.k365_recost_20260612 import (
    EXCESS_EXTRA_K365_CENTRE, _build_full_c1, _min_at,
)
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax, _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
    FEE_GOLD, FEE_BOND,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    _build_v7_mult, DELAY as V7_DELAY, DH_PER_UNIT, NAV_FLOOR,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom, LU2_SCALE,
)
from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

# ---- LT2 signal functions ---------------------------------------------------
from long_cycle_signal import compute_lt2, signal_to_mult

# ---- TER daily drag constants (same as strategy_runners) --------------------
_TER_GOLD2X_EXTRA_DAILY = 0.0
_TER_TMF_EXTRA_DAILY    = 0.0
try:
    from src.audit.strategy_runners import _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY
except ImportError:
    pass  # will use 0.0 fallback

# ---------------------------------------------------------------------------
# B3a_k365 config (fixed - do not change)
# ---------------------------------------------------------------------------
B3A_V7_MAP   = {0: 1.40, 1: 1.40, 2: 1.05, 3: 1.00}
B3A_LEV_SCALE = 1.15
B3A_EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE   # 0.0025 /yr on (L-3)+

# LT2 signal parameters
LT2_N = 750

# Known B3a_k365 reference values (from LEVERUP_SWEEP_RESULTS_20260612.md)
B3A_MIN9_REF  = 0.2098   # +20.98% (min(IS,OOS) after-tax)
B3A_MAXDD_REF = -0.3820  # -38.20%
B3A_SHARPE_REF = 0.904

# Sanity tolerance
SANITY_TOL_MIN9  = 0.0005   # 0.05pp on min9
EARLY_CLOSE_TOL  = 0.010    # 1.0pp degradation threshold


# ---------------------------------------------------------------------------
# Parameterised TQQQ-base NAV with V7-map, lev_scale, excess_extra,
# AND an optional LT2 multiplier applied to lev_raw_masked BEFORE V7 scaling.
# ---------------------------------------------------------------------------

def _build_nav_v7_tqqq_g4(close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
                            lev_raw_masked, wn, wg, wb, mult_v7,
                            lt2_mult=None,
                            excess_extra=B3A_EXCESS_EXTRA):
    """Build TQQQ-base NAV with optional LT2 Mode-A multiplier.

    Integration design:
      lev_mod = lev_raw_masked * mult_v7 * lt2_mult   (element-wise)
      L       = lev_mod * 3.0 (shifted by V7_DELAY)

    When lt2_mult is None or all-ones, this reproduces B3a_k365 exactly.

    Parameters
    ----------
    lt2_mult : array-like or None
        LT2 contrarian multiplier series (same length as lev_raw_masked).
        Must be pre-computed from compute_lt2 + signal_to_mult.
        None or ones-array -> neutral (sanity anchor mode).
    """
    idx = dates.index
    n   = len(close)

    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3  = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    lev_base = np.asarray(lev_raw_masked, float) * np.asarray(mult_v7, float)
    if lt2_mult is not None:
        lt2_arr = np.asarray(lt2_mult, float)
        # Clip to [0.5, 1.5] guard (already done in signal_to_mult but be safe)
        lt2_arr = np.clip(lt2_arr, 0.5, 1.5)
        lev_mod = lev_base * lt2_arr
    else:
        lev_mod = lev_base

    # Apply shift (publication lag)
    L     = pd.Series(lev_mod * 3.0,            index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s  = pd.Series(np.asarray(wn, float),   index=idx).shift(V7_DELAY).fillna(0).values
    wg_s  = pd.Series(np.asarray(wg, float),   index=idx).shift(V7_DELAY).fillna(0).values
    wb_s  = pd.Series(np.asarray(wb, float),   index=idx).shift(V7_DELAY).fillna(0).values
    sofr_arr = np.asarray(sofr_daily, float)

    # TQQQ-cost NASDAQ leg
    borrow  = np.maximum(L - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # k365 excess penalty (L > 3x)
    excess_lev  = np.maximum(L - LEV_CAP, 0.0)
    penalty     = wn_s * excess_lev * float(excess_extra) / TRADING_DAYS
    daily       = daily - penalty
    excess_days = int(np.sum(excess_lev > 1e-9))

    # Turnover-based DH per-unit cost
    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn  = np.nan_to_num(dwn + dwg + dwb, nan=0.0)
    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    # Additional TER drag (Gold2x / TMF extra over TQQQ TER)
    ter_drag  = (np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
                 + np.asarray(wb, float) * _TER_TMF_EXTRA_DAILY)
    tpy       = sr._compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / 252.0
    r_sim     = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj     = r_sim - ter_drag - etf_daily

    nav_adj = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )
    r_base = nav_adj.pct_change().fillna(0).values
    return nav_adj, r_base, tpy, excess_days


def _build_tqqq_base_g4(shared, dates_dt, v7_map=None, lev_scale=1.0,
                         excess_extra=B3A_EXCESS_EXTRA, lt2_mult=None):
    """Thin wrapper: extract arrays from shared and call _build_nav_v7_tqqq_g4.

    Returns (nav_dt, r_base, tpy, excess_days) -- 4-tuple matching k365_recost pattern.
    """
    a               = shared["assets"]
    close           = a["close"]
    dates           = a["dates"]
    sofr            = np.asarray(a["sofr"], float)
    gold_2x         = a["gold_2x"]
    bond_3x         = a["bond_3x"]
    lev_raw_masked  = np.asarray(shared["lev_raw_masked"], float)
    wn              = np.asarray(shared["wn"], float)
    wg              = np.asarray(shared["wg"], float)
    wb              = np.asarray(shared["wb"], float)

    if v7_map is None:
        mult_v7 = _build_v7_mult(dates_dt)
    else:
        mult_v7 = _build_v7_mult_custom(dates_dt, v7_map)
    mult_v7 = mult_v7 * float(lev_scale)

    return _build_nav_v7_tqqq_g4(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7,
        lt2_mult=lt2_mult,
        excess_extra=excess_extra,
    )


def _build_b3a_g4(shared, dates_dt, n_years,
                  ret_gold, ret_bond, fund_active, wg_inv, wb_inv,
                  bond_on, sofr_arr, lt2_mult=None):
    """Build complete B3a_k365 (+G4 LT2) NAV: TQQQ base -> P09 OUT-fill -> C1."""
    base_nav, r_base, tpy_b, exc = _build_tqqq_base_g4(
        shared, dates_dt,
        v7_map=B3A_V7_MAP, lev_scale=B3A_LEV_SCALE,
        excess_extra=B3A_EXCESS_EXTRA, lt2_mult=lt2_mult,
    )
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg_inv, wb_inv, bond_on,
        sofr_arr, dates_dt, tpy_b, n_years,
    )
    return nav_dt, r, tpy, exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 96)
    print("G4: LT2-N750 native integration into B3a_k365  [combine_g4_lt2_20260615]")
    print("Lesson E: TQQQ-base re-evaluation required (CFD/S2 base result not assumed).")
    print("=" * 96)

    # ---- Load shared DH-W1 assets -------------------------------------------
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

    # ---- P09 OUT-fill machinery (same as k365_recost_20260612) --------------
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected

    gold_1x  = np.asarray(prepare_gold_local(dates), float)
    bond_1x  = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0),
        float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active  = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg_inv, wb_inv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on   = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr  = np.asarray(a["sofr"], float)

    # ---- Compute LT2-N750 signal on NASDAQ close ----------------------------
    print("\nComputing LT2-N750 signal ...")
    close   = a["close"]
    lt2_sig = compute_lt2(close, N=LT2_N)  # pd.Series, z-score

    # Sanity: LT2 direction check (high positive momentum -> contrarian down-weight)
    # contrarian: high z (strong past momentum) -> mult < 1.0 (reduce exposure)
    high_z  = lt2_sig.quantile(0.90)
    low_z   = lt2_sig.quantile(0.10)
    mult_hi = float((1.0 - 0.30 * high_z).clip(0.5, 1.5))
    mult_lo = float((1.0 - 0.30 * low_z ).clip(0.5, 1.5))
    print("  LT2 contrarian sign check (k_lt=0.30):")
    print("    P90 z=%.3f -> mult=%.3f (should be < 1.0, high-momentum = reduce exposure)" % (high_z, mult_hi))
    print("    P10 z=%.3f -> mult=%.3f (should be > 1.0, low-momentum  = add exposure)"   % (low_z,  mult_lo))
    if mult_hi >= 1.0:
        print("  WARNING: P90 mult >= 1.0 -- LT2 direction may be inverted. Proceed with caution.")
    else:
        print("  OK: LT2 direction consistent with contrarian design.")

    # ---- Build LT2 multiplier arrays for k_lt sweep -------------------------
    K_LT_VALUES = [0.0, 0.30, 0.20]   # 0.0 = sanity anchor (neutral)
    lt2_mults = {}
    for k in K_LT_VALUES:
        if k == 0.0:
            lt2_mults[k] = np.ones(len(lt2_sig))
        else:
            m = signal_to_mult(lt2_sig, k_lt=k)   # pd.Series, clip [0.5, 1.5]
            lt2_mults[k] = m.values

    # ---- Build NAV series for each config -----------------------------------
    print("\nBuilding NAV series ...")
    labels  = []
    nav_map = {}
    r_map   = {}
    tpy_map = {}
    exc_map = {}

    config_map = {
        0.00: "B3a_k365_sanity",
        0.30: "B3a_G4_k030",
        0.20: "B3a_G4_k020",
    }

    for k in K_LT_VALUES:
        label = config_map[k]
        labels.append(label)
        print("  Building %s (k_lt=%.2f) ..." % (label, k))
        nav_dt, r, tpy, exc = _build_b3a_g4(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg_inv, wb_inv,
            bond_on, sofr_arr, lt2_mult=lt2_mults[k],
        )
        nav_map[label] = nav_dt
        r_map[label]   = r
        tpy_map[label] = tpy
        exc_map[label] = exc
        print("    Done. tpy=%.1f, excess_days=%d" % (tpy, exc))

    # ---- Sanity anchor check ------------------------------------------------
    print("\n" + "=" * 96)
    print("SANITY ANCHOR CHECK: k_lt=0.0 must reproduce B3a_k365")
    print("  Expected: min9=%+.2f%%, MaxDD=%+.2f%%, Sharpe=%.3f"
          % (B3A_MIN9_REF * 100, B3A_MAXDD_REF * 100, B3A_SHARPE_REF))
    print("=" * 96)

    sanity_label = "B3a_k365_sanity"
    sanity_nav   = nav_map[sanity_label]
    sanity_pre   = compute_10metrics(sanity_nav, tpy_map[sanity_label])
    sanity_aft   = _apply_aftertax(sanity_pre)
    sanity_min9  = _min_at(sanity_aft)
    sanity_maxdd = sanity_pre["MaxDD_FULL"]
    sanity_sh    = sanity_pre["Sharpe_OOS"]

    diff_min9    = abs(sanity_min9 - B3A_MIN9_REF)
    diff_maxdd   = abs(sanity_maxdd - B3A_MAXDD_REF)

    print("  Computed: min9=%+.2f%%  MaxDD=%+.2f%%  Sharpe=%.3f"
          % (sanity_min9 * 100, sanity_maxdd * 100, sanity_sh))
    print("  diff_min9=%.4fpp  diff_maxdd=%.4fpp" % (diff_min9 * 100, diff_maxdd * 100))

    sanity_ok = (diff_min9 <= SANITY_TOL_MIN9)
    if sanity_ok:
        print("  [SANITY OK] min9 within tolerance (%.2fpp <= %.2fpp)"
              % (diff_min9 * 100, SANITY_TOL_MIN9 * 100))
    else:
        print("  [SANITY FAIL] min9 diff=%.4fpp exceeds tol=%.2fpp -- integration bug!"
              % (diff_min9 * 100, SANITY_TOL_MIN9 * 100))
        print("  Halting. Check _build_nav_v7_tqqq_g4 / lt2_mult=ones path.")
        sys.exit(1)

    # ---- Compute metrics for all configs ------------------------------------
    print("\n" + "=" * 96)
    print("STAGE 0 METRICS (after-tax, standard 10 + worst calendar year)")
    print("=" * 96)

    packs = {}
    for label in labels:
        pre = compute_10metrics(nav_map[label], tpy_map[label])
        aft = _apply_aftertax(pre)
        cy  = _calendar_year_returns(nav_map[label])
        packs[label] = {"pre": pre, "aft": aft, "cy": cy}

    # ---- Hard-veto evaluation -----------------------------------------------
    print("\nHard-veto check (MaxDD<-50% | W10Y_at<0 | Regime constraints at Stage1):")
    VETO_MAXDD = -0.50
    VETO_W10Y  = 0.0

    def _veto_check(label):
        pre = packs[label]["pre"]
        aft = packs[label]["aft"]
        v_maxdd = pre["MaxDD_FULL"] < VETO_MAXDD
        v_w10y  = aft["Worst10Y_star"] < VETO_W10Y
        veto    = v_maxdd or v_w10y
        return {"maxdd_veto": v_maxdd, "w10y_veto": v_w10y, "VETO": veto}

    veto_map = {lbl: _veto_check(lbl) for lbl in labels}
    for lbl in labels:
        v = veto_map[lbl]
        print("  %-22s | MaxDD_veto=%-5s | W10Y_veto=%-5s | VETO=%s"
              % (lbl,
                 "YES" if v["maxdd_veto"] else "no",
                 "YES" if v["w10y_veto"] else "no",
                 "**VETO**" if v["VETO"] else "PASS"))

    # ---- Print Stage 0 table ------------------------------------------------
    hdr = ("%-22s | %9s | %10s | %8s | %8s | %8s | %10s | %8s | %8s | %7s"
           % ("label", "CAGR_IS%", "CAGR_OOS%", "min9%",
              "Sharpe", "MaxDD%", "Worst10Y%", "P10_5Y%",
              "gap_pp", "Trd/yr"))
    sep = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)

    for lbl in labels:
        aft = packs[lbl]["aft"]
        pre = packs[lbl]["pre"]
        mn  = _min_at(aft)
        print("%-22s | %+8.2f%% | %+9.2f%% | %+7.2f%% | %7.3f | %+7.2f%% | %+9.2f%% | %+7.2f%% | %+7.2f | %7.1f"
              % (lbl,
                 100 * aft["CAGR_IS"], 100 * aft["CAGR_OOS"], 100 * mn,
                 pre["Sharpe_OOS"], 100 * pre["MaxDD_FULL"],
                 100 * aft["Worst10Y_star"], 100 * aft["P10_5Y"],
                 aft["IS_OOS_gap_pp"], aft["Trades_yr"]))
    print(sep)

    # ---- B3a diff table (for G4 configs only) -------------------------------
    base_min9   = _min_at(packs[sanity_label]["aft"])
    base_maxdd  = packs[sanity_label]["pre"]["MaxDD_FULL"]
    base_sharpe = packs[sanity_label]["pre"]["Sharpe_OOS"]

    print("\nDiff vs B3a_k365 sanity anchor:")
    diff_hdr = ("%-22s | %9s | %9s | %9s | %9s"
                % ("label", "d_min9pp", "d_MaxDD pp", "d_Sharpe", "VETO"))
    print(diff_hdr)
    print("-" * len(diff_hdr))
    for lbl in labels:
        if lbl == sanity_label:
            continue
        mn  = _min_at(packs[lbl]["aft"])
        dd  = packs[lbl]["pre"]["MaxDD_FULL"]
        sh  = packs[lbl]["pre"]["Sharpe_OOS"]
        print("%-22s | %+8.3fpp | %+8.3fpp | %+8.3f | %s"
              % (lbl,
                 (mn  - base_min9)  * 100,
                 (dd  - base_maxdd) * 100,
                 sh - base_sharpe,
                 "VETO" if veto_map[lbl]["VETO"] else "pass"))

    # ---- Worst calendar year printout ---------------------------------------
    print("\nWorst calendar year by config:")
    for lbl in labels:
        cy = packs[lbl]["cy"]
        worst_yr = int(cy.idxmin())
        worst_r  = float(cy.min())
        print("  %-22s : %d  %+.2f%%" % (lbl, worst_yr, worst_r * 100))

    # ---- Survival / Early-close judgment ------------------------------------
    print("\n" + "=" * 96)
    print("SURVIVAL JUDGMENT (plan §2 Stage 0 + §6 early-close)")
    print("=" * 96)
    print("Attack criterion: min9 >= B3a素地 - 0.3pp = %+.2fpp" % ((B3A_MIN9_REF - 0.003) * 100))
    print("MaxDD criterion:  MaxDD degradation <= +2.0pp (B3a MaxDD ref=%+.2f%%)" % (B3A_MAXDD_REF * 100))
    print("Early-close:      min9 < B3a素地 - 1.0pp for ALL k_lt -> report Lesson E degradation")

    g4_labels  = [lbl for lbl in labels if lbl != sanity_label]
    survive    = []
    early_flag = True  # assume early-close until proven otherwise

    for lbl in g4_labels:
        mn      = _min_at(packs[lbl]["aft"])
        dd      = packs[lbl]["pre"]["MaxDD_FULL"]
        no_veto = not veto_map[lbl]["VETO"]
        ok_min9 = (mn >= B3A_MIN9_REF - 0.003)       # within -0.3pp
        ok_dd   = (dd - base_maxdd) <= 0.02           # degradation <= +2pp
        alive   = no_veto and ok_min9 and ok_dd
        survive.append(alive)
        # Early-close: if ANY config is above early-close threshold, not early-close
        if mn >= (B3A_MIN9_REF - EARLY_CLOSE_TOL):
            early_flag = False
        print("  %-22s : no_veto=%s ok_min9=%s ok_dd=%s -> %s"
              % (lbl,
                 no_veto, ok_min9, ok_dd,
                 "SURVIVE -> Stage1" if alive else "CLOSE (Stage0 fail)"))

    if early_flag:
        print("\n[EARLY CLOSE] All G4 configs show min9 degradation > 1.0pp vs B3a.")
        print("  Lesson E confirmed: LT2 contrarian signal degrades on TQQQ/DH-W1 base.")
        print("  G4 closed at Stage 0. Do not proceed to Stage 1.")
        judgment = "EARLY_CLOSE_LESSON_E"
    else:
        any_survive = any(survive)
        if any_survive:
            survivors = [g4_labels[i] for i, s in enumerate(survive) if s]
            print("\n[STAGE 0 PASS] Survivors: %s -> proceed to Stage 1 (WFA/CPCV)." % survivors)
            judgment = "STAGE0_PASS"
        else:
            print("\n[STAGE 0 FAIL] No G4 config passed Stage 0 criteria.")
            print("  min9 improvement was present but MaxDD or VETO blocked survival.")
            judgment = "STAGE0_FAIL"

    # ---- CSV output ---------------------------------------------------------
    print("\nBuilding CSV ...")
    rows = []
    for lbl in labels:
        aft = packs[lbl]["aft"]
        pre = packs[lbl]["pre"]
        cy  = packs[lbl]["cy"]
        mn  = _min_at(aft)
        vm  = veto_map[lbl]
        k   = [k for k, v in config_map.items() if v == lbl][0]
        rows.append({
            "label":            lbl,
            "k_lt":             k,
            "CAGR_IS_at":       float(aft["CAGR_IS"]),
            "CAGR_OOS_at":      float(aft["CAGR_OOS"]),
            "min_IS_OOS_at":    mn,
            "IS_OOS_gap_pp":    float(aft["IS_OOS_gap_pp"]),
            "Sharpe_OOS":       float(pre["Sharpe_OOS"]),
            "MaxDD_FULL":       float(pre["MaxDD_FULL"]),
            "Worst10Y_star_at": float(aft["Worst10Y_star"]),
            "P10_5Y_at":        float(aft["P10_5Y"]),
            "Worst5Y_at":       float(aft["Worst5Y"]),
            "Trades_yr":        float(aft["Trades_yr"]),
            "worst_cy":         float(cy.min()),
            "worst_cy_year":    int(cy.idxmin()),
            "excess_days":      exc_map[lbl],
            "veto_maxdd":       int(vm["maxdd_veto"]),
            "veto_w10y":        int(vm["w10y_veto"]),
            "VETO":             int(vm["VETO"]),
            "d_min9_pp":        (mn - base_min9) * 100,
            "d_MaxDD_pp":       (float(pre["MaxDD_FULL"]) - base_maxdd) * 100,
            "d_Sharpe":         float(pre["Sharpe_OOS"]) - base_sharpe,
            "stage0_judgment":  judgment if lbl != sanity_label else "SANITY_ANCHOR",
        })

    out_dir  = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combine_g4_lt2_20260615.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("Saved CSV: %s" % csv_path)

    # ---- RETURN_BLOCK -------------------------------------------------------
    def _rb(lbl):
        aft = packs[lbl]["aft"]
        pre = packs[lbl]["pre"]
        vm  = veto_map[lbl]
        mn  = _min_at(aft)
        return {
            "CAGR_IS_at":       round(float(aft["CAGR_IS"]), 6),
            "CAGR_OOS_at":      round(float(aft["CAGR_OOS"]), 6),
            "min_at":           round(mn, 6),
            "gap_pp":           round(float(aft["IS_OOS_gap_pp"]), 4),
            "Sharpe":           round(float(pre["Sharpe_OOS"]), 4),
            "MaxDD":            round(float(pre["MaxDD_FULL"]), 6),
            "Worst10Y_at":      round(float(aft["Worst10Y_star"]), 6),
            "P10_5Y_at":        round(float(aft["P10_5Y"]), 6),
            "Trades_yr":        round(float(aft["Trades_yr"]), 2),
            "d_min9_pp":        round((mn - base_min9) * 100, 4),
            "d_MaxDD_pp":       round((float(pre["MaxDD_FULL"]) - base_maxdd) * 100, 4),
            "VETO":             vm["VETO"],
            "excess_days":      exc_map[lbl],
        }

    block = {
        "meta": {
            "script":          "combine_g4_lt2_20260615.py",
            "B3a_min9_ref_pct":  round(B3A_MIN9_REF * 100, 4),
            "B3a_MaxDD_ref_pct": round(B3A_MAXDD_REF * 100, 4),
            "B3a_Sharpe_ref":    round(B3A_SHARPE_REF, 4),
            "LT2_N":             LT2_N,
            "sanity_ok":         sanity_ok,
            "sanity_diff_min9_pp": round(diff_min9 * 100, 4),
            "lt2_direction_ok":  (mult_hi < 1.0),
        },
        "B3a_k365_sanity": _rb("B3a_k365_sanity"),
        "B3a_G4_k030":     _rb("B3a_G4_k030"),
        "B3a_G4_k020":     _rb("B3a_G4_k020"),
        "judgment":         judgment,
        "stage0_survivors": [g4_labels[i] for i, s in enumerate(survive) if s]
                             if judgment != "EARLY_CLOSE_LESSON_E" else [],
        "early_close_reason": (
            "All G4 configs min9 < B3a_k365 - 1.0pp: Lesson E TQQQ-base degradation confirmed."
            if early_flag else ""
        ),
    }

    print("\n" + "=" * 96)
    print("RETURN_BLOCK")
    print("=" * 96)
    print(json.dumps(block, indent=2, ensure_ascii=False))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
