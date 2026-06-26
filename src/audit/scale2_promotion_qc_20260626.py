"""
src/audit/scale2_promotion_qc_20260626.py
=========================================
P09_STR strong-map leverage-scale BEST-strategy promotion eligibility QC.

History:
  - First pass evaluated Scale2.0 as the promotion target (user asked "make Scale2.0
    the best strategy"). QC concluded MaxDD -61.63% / gap +7.57pp / Regime -11.41% /
    max eff lev 9.6x; CAGR highest (+29.11%) but Sharpe/WFE lose to scale1.4.
  - 2026-06-26: after reviewing the QC, the user chose to promote **Scale1.6** (the
    middle point: CAGR_OOS +26.21% / MaxDD -51.95% / gap +4.90pp / Regime -7.71% /
    max eff lev 7.68x). scale1.6 was added to SCALES/ANCHORS and re-verified here.

The canon (P09_STR_SCALE_DIAL_20260623.md, fixed 2026-06-23, FRONTIER 4-digit sanity
PASS) provides the anchors. This script does NOT re-litigate cost premises
(EVALUATION_STANDARD §1.5 v1.9: margin is collateral, no CAGR drag). It re-derives the
decision-critical premises from first principles for all scales (1.0/1.4/1.6/2.0):

  STAGE 1   reproduce the canon's fixed values independently (anchor match, tol 0.15pp)
  STAGE 2A  MaxDD -61.63% path-robustness: direct-NAV drawdown path + path-AGGREGATE measures
            (time-under-water, avg-DD, CVaR-of-DD) + untouched crisis-window losses.
            R-STAT-1: NO block=21 bootstrap on path-dependent extrema.
  STAGE 2B  IS-OOS gap +7.57pp: is it overfit, or the geometric necessity of leverage that
            only amplifies bull years? Decompose IS vs OOS growth; confirm OUT-year scale
            invariance; compare to an equal-mean-exposure uniform-de-lever twin (R-STAT-3).
  STAGE 2C  best-criteria reconciliation: what thresholds does promoting Scale2.0 relax?
            scale1.4 alternative shown alongside.

Reuses validated builders only (no reimplementation). ASCII-only prints. No temp files,
no commit. Writes audit_results/scale2_promotion_qc_20260626.csv.
"""

from __future__ import annotations

import os
import sys
import types

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
from src.audit.k365_recost_20260612 import _build_full_c1, EXCESS_EXTRA_K365_CENTRE
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base
from src.audit.extended_eval_20260611 import _eval_one
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import GATE_DELAY, _load_macro_signal

AFTER_TAX = 0.8273
STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025 k365 centre, always-on -> matches FRONTIER

# Scales we evaluate. 1.6 is the promotion TARGET (2026-06-26); 1.0/1.4/2.0 are
# reference/alternative anchors.
SCALES = [1.0, 1.4, 1.6, 2.0]

# Canon anchors (P09_STR_SCALE_DIAL_20260623.md / p09_strongmap_scale_dial_20260623.py).
ANCHORS = {
    1.4: {"CAGR_IS": 0.274877, "CAGR_OOS": 0.243414, "MaxDD": -0.464781},
    1.6: {"CAGR_IS": 0.302649, "CAGR_OOS": 0.262127, "MaxDD": -0.519521},
    2.0: {"CAGR_IS": 0.353755, "CAGR_OOS": 0.291102, "MaxDD": -0.616342},
}
ANCHOR_TOL = 0.0015  # +/-0.15pp

# Best-criteria thresholds the canon uses (for STAGE 2C reconciliation; reported as raw
# measured values, NO pass/fail labels in the deliverable per project rule).
CRIT = {"MaxDD": -0.50, "gap_pp": 5.0, "Regime_min": -0.10}

# Crisis windows (untouched-path losses, R-STAT-2). [label, start, end]
CRISIS = [
    ("dotcom 2000-2002", "2000-03-10", "2002-10-09"),
    ("GFC 2007-2009", "2007-10-09", "2009-03-09"),
    ("2022 hike", "2021-11-19", "2022-12-28"),
]


# --------------------------------------------------------------------------- builders
def _setup():
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    is_mask = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    ctx = dict(shared=shared, a=a, dates=dates, dates_dt=dates_dt, n=n, n_years=n_years,
               is_mask=is_mask, oos_mask=oos_mask, ret_gold=ret_gold, ret_bond=ret_bond,
               fund_active=fund_active, wg=wg, wb=wb, bond_on=bond_on, sofr_arr=sofr_arr,
               regimes=regimes, stress=stress, r_v7=r_v7)
    return ctx


def _build(ctx, sc):
    """Return (nav_dt as date-indexed Series, r daily array, tpy, excess_days)."""
    nav_dt, r, tpy, exc = _build_full_c1(
        ctx["shared"], ctx["dates_dt"], ctx["n_years"],
        ctx["ret_gold"], ctx["ret_bond"], ctx["fund_active"],
        ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"],
        v7_map=STRONG_MAP, lev_scale=sc, excess_extra=EXCESS_EXTRA)
    return nav_dt, r, tpy, exc


# --------------------------------------------------------------------------- path measures
def _nav_array(nav_dt):
    return np.asarray(nav_dt.values if hasattr(nav_dt, "values") else nav_dt, float)


def _drawdown_path(nav):
    """Return (maxdd, peak_idx, trough_idx, recover_idx_or_-1)."""
    nav = np.asarray(nav, float)
    run_max = np.maximum.accumulate(nav)
    dd = nav / run_max - 1.0
    trough = int(np.argmin(dd))
    maxdd = float(dd[trough])
    peak = int(np.argmax(nav[:trough + 1])) if trough > 0 else 0
    peak_val = nav[peak]
    recover = -1
    for k in range(trough, len(nav)):
        if nav[k] >= peak_val:
            recover = k
            break
    return maxdd, peak, trough, recover


def _path_aggregates(nav):
    """time-under-water (frac of days below prior peak), average drawdown,
    CVaR(5%) of daily drawdown. All path-AGGREGATE -> valid scale-vs-scale (R-STAT-2)."""
    nav = np.asarray(nav, float)
    run_max = np.maximum.accumulate(nav)
    dd = nav / run_max - 1.0  # <=0
    tuw_frac = float(np.mean(dd < -1e-9))
    avg_dd = float(np.mean(dd))
    q = np.quantile(dd, 0.05)  # 5th pct (deep) of dd
    cvar_dd = float(np.mean(dd[dd <= q])) if np.any(dd <= q) else float(q)
    return tuw_frac, avg_dd, cvar_dd


def _window_loss(nav_dt, dates_dt, start, end):
    """Untouched-path cumulative return inside [start,end]."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    m = (dates_dt >= s) & (dates_dt <= e)
    nav = _nav_array(nav_dt)[np.asarray(m)]
    if len(nav) < 2:
        return np.nan, 0
    return float(nav[-1] / nav[0] - 1.0), int(len(nav))


# --------------------------------------------------------------------------- main
def main():
    print("=" * 100)
    print("SCALE2.0 PROMOTION ELIGIBILITY QC  2026-06-26")
    print("Base: STRONG_MAP=%s x lev_scale ; _build_full_c1 + C1 ; EXCESS_EXTRA=%.4f"
          % (STRONG_MAP, EXCESS_EXTRA))
    print("=" * 100)

    ctx = _setup()
    dates_dt = ctx["dates_dt"]

    built = {}
    for sc in SCALES:
        nav_dt, r, tpy, exc = _build(ctx, sc)
        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)
        ev = _eval_one("P09_STR_sc%.2f" % sc, nav_dt, r, ctx["regimes"], ctx["stress"],
                       ctx["is_mask"], ctx["oos_mask"], baseline_r=ctx["r_v7"])
        built[sc] = dict(nav_dt=nav_dt, r=r, tpy=tpy, exc=exc, pre=pre, aft=aft, ev=ev)

    # ===================================================================== STAGE 1
    print("\n--- STAGE 1: independent reproduction vs canon anchors (tol %.2fpp) ---"
          % (ANCHOR_TOL * 100))
    stage1_ok = True
    for sc in (1.4, 1.6, 2.0):
        aft = built[sc]["aft"]
        dd = built[sc]["pre"]["MaxDD_FULL"]
        exp = ANCHORS[sc]
        d_is = abs(aft["CAGR_IS"] - exp["CAGR_IS"])
        d_oos = abs(aft["CAGR_OOS"] - exp["CAGR_OOS"])
        d_dd = abs(dd - exp["MaxDD"])
        ok = (d_is <= ANCHOR_TOL and d_oos <= ANCHOR_TOL and d_dd <= ANCHOR_TOL)
        stage1_ok = stage1_ok and ok
        print("  sc%.1f: IS %+.4f%%(exp %+.4f) OOS %+.4f%%(exp %+.4f) MaxDD %+.4f%%(exp %+.4f) -> %s"
              % (sc, aft["CAGR_IS"] * 100, exp["CAGR_IS"] * 100,
                 aft["CAGR_OOS"] * 100, exp["CAGR_OOS"] * 100,
                 dd * 100, exp["MaxDD"] * 100, "MATCH" if ok else "MISMATCH"))
    print("  STAGE1 = %s" % ("REPRODUCED (canon values still hold)" if stage1_ok
                             else "DRIFT -- halt promotion decision"))

    # ===================================================================== STAGE 2A
    print("\n--- STAGE 2A: MaxDD path-robustness (direct-NAV, R-STAT-1 no block=21) ---")
    s2a = {}
    for sc in SCALES:
        nav = _nav_array(built[sc]["nav_dt"])
        maxdd, peak, trough, recover = _drawdown_path(nav)
        tuw, avg_dd, cvar_dd = _path_aggregates(nav)
        rec_days = (recover - peak) if recover > 0 else -1
        s2a[sc] = dict(maxdd=maxdd, peak_d=str(dates_dt[peak].date()),
                       trough_d=str(dates_dt[trough].date()),
                       recover_d=(str(dates_dt[recover].date()) if recover > 0 else "not-recovered"),
                       dd_len=trough - peak, rec_len=rec_days,
                       tuw=tuw, avg_dd=avg_dd, cvar_dd=cvar_dd)
        print("  sc%.1f MaxDD=%+.2f%% peak=%s trough=%s recover=%s | drop=%dd recover=%sd | "
              "TUW=%.1f%% avgDD=%+.2f%% CVaR5%%DD=%+.2f%%"
              % (sc, maxdd * 100, s2a[sc]["peak_d"], s2a[sc]["trough_d"], s2a[sc]["recover_d"],
                 s2a[sc]["dd_len"], (str(rec_days) if rec_days > 0 else "n/a"),
                 tuw * 100, avg_dd * 100, cvar_dd * 100))
    print("  -- untouched crisis-window cumulative returns (R-STAT-2) --")
    s2a_win = {}
    for sc in SCALES:
        row = {}
        for label, st, en in CRISIS:
            lo, ln = _window_loss(built[sc]["nav_dt"], dates_dt, st, en)
            row[label] = (lo, ln)
        s2a_win[sc] = row
    for label, st, en in CRISIS:
        print("    %-18s | " % label
              + " ".join("sc%.1f=%+.2f%%(%dd)" % (sc, s2a_win[sc][label][0] * 100,
                                                  s2a_win[sc][label][1]) for sc in SCALES))

    # ===================================================================== STAGE 2B
    print("\n--- STAGE 2B: IS-OOS gap decomposition + OUT-year scale invariance ---")
    print("  scale   CAGR_IS    CAGR_OOS   gap(pp)")
    for sc in SCALES:
        aft = built[sc]["aft"]
        print("  sc%.1f   %+7.2f%%   %+7.2f%%   %+6.2f"
              % (sc, aft["CAGR_IS"] * 100, aft["CAGR_OOS"] * 100, aft["IS_OOS_gap_pp"]))
    # OUT-year scale invariance: calendar-year returns for the crash/OUT years
    out_years = [2000, 2001, 2002, 2008, 2022]
    cy = {sc: _calendar_year_returns(built[sc]["nav_dt"]) for sc in SCALES}
    print("  -- OUT/crash year calendar returns (after-tax x0.8273); scale-invariant if equal --")
    for yr in out_years:
        vals = []
        for sc in SCALES:
            v = cy[sc].loc[yr] * AFTER_TAX * 100 if yr in cy[sc].index else float("nan")
            vals.append(v)
        invariant = (np.nanmax(vals) - np.nanmin(vals)) < 0.05  # within 0.05pp
        print("    %d: " % yr + " ".join("sc%.1f=%+.2f%%" % (SCALES[i], vals[i])
                                         for i in range(len(SCALES)))
              + ("  [scale-invariant]" if invariant else "  [scale-varying]"))
    # bull-year amplification: a strongly positive year
    for yr in (2003, 2020):
        vals = [cy[sc].loc[yr] * AFTER_TAX * 100 if yr in cy[sc].index else float("nan")
                for sc in SCALES]
        print("    %d (bull): " % yr + " ".join("sc%.1f=%+.2f%%" % (SCALES[i], vals[i])
                                                for i in range(len(SCALES))))

    # ===================================================================== STAGE 2C
    print("\n--- STAGE 2C: best-criteria reconciliation (raw values; promoting relaxes) ---")
    b = built[2.0]
    a14 = built[1.4]
    print("  criterion        canon-threshold   Scale2.0        scale1.4(alt)")
    print("  MaxDD            <%-6.0f%% (vetoes)  %+8.2f%%      %+8.2f%%"
          % (CRIT["MaxDD"] * 100, b["pre"]["MaxDD_FULL"] * 100, a14["pre"]["MaxDD_FULL"] * 100))
    print("  IS-OOS gap(pp)   >=%-5.1f (warns)    %+8.2f       %+8.2f"
          % (CRIT["gap_pp"], b["aft"]["IS_OOS_gap_pp"], a14["aft"]["IS_OOS_gap_pp"]))
    print("  Regime_min       <%-6.0f%% (vetoes)  %+8.2f%%      %+8.2f%%"
          % (CRIT["Regime_min"] * 100, b["ev"]["regime_min_at"] * 100,
             a14["ev"]["regime_min_at"] * 100))
    print("  CAGR_OOS(min)    (higher better)   %+8.2f%%      %+8.2f%%"
          % (min(b["aft"]["CAGR_IS"], b["aft"]["CAGR_OOS"]) * 100,
             min(a14["aft"]["CAGR_IS"], a14["aft"]["CAGR_OOS"]) * 100))
    print("  Sharpe_FULL      (higher better)   %+8.3f       %+8.3f"
          % (b["pre"]["Sharpe_FULL"], a14["pre"]["Sharpe_FULL"]))
    print("  WFE              (>=~1 generalizes)%+8.3f       %+8.3f"
          % (b["ev"]["wfa_WFE"], a14["ev"]["wfa_WFE"]))
    max_eff_2 = max(STRONG_MAP.values()) * 2.0 * 3.0
    max_eff_14 = max(STRONG_MAP.values()) * 1.4 * 3.0
    print("  max_eff_lev      (margin/capacity) %8.1fx      %8.1fx" % (max_eff_2, max_eff_14))

    # ===================================================================== CSV
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sc in SCALES:
        aft = built[sc]["aft"]
        pre = built[sc]["pre"]
        ev = built[sc]["ev"]
        rows.append({
            "scale": sc, "boost_map": "strong", "excess_extra": EXCESS_EXTRA,
            "CAGR_IS_at": round(aft["CAGR_IS"], 6), "CAGR_OOS_at": round(aft["CAGR_OOS"], 6),
            "min9_at": round(min(aft["CAGR_IS"], aft["CAGR_OOS"]), 6),
            "IS_OOS_gap_pp": round(aft["IS_OOS_gap_pp"], 4),
            "Sharpe_FULL": round(pre["Sharpe_FULL"], 4),
            "MaxDD_FULL": round(pre["MaxDD_FULL"], 6),
            "Worst10Y_at": round(aft["Worst10Y_star"], 6),
            "regime_min_at": round(ev["regime_min_at"], 6),
            "wfa_WFE": round(ev["wfa_WFE"], 6), "wfa_CI95_lo": round(ev["wfa_CI95_lo"], 6),
            "max_eff_lev": round(max(STRONG_MAP.values()) * sc * 3.0, 3),
            "maxdd_peak": s2a[sc]["peak_d"], "maxdd_trough": s2a[sc]["trough_d"],
            "maxdd_recover": s2a[sc]["recover_d"],
            "dd_drop_days": s2a[sc]["dd_len"], "dd_recover_days": s2a[sc]["rec_len"],
            "time_under_water_frac": round(s2a[sc]["tuw"], 6),
            "avg_drawdown": round(s2a[sc]["avg_dd"], 6),
            "cvar5_drawdown": round(s2a[sc]["cvar_dd"], 6),
            "dotcom_loss": round(s2a_win[sc]["dotcom 2000-2002"][0], 6),
            "gfc_loss": round(s2a_win[sc]["GFC 2007-2009"][0], 6),
            "hike2022_loss": round(s2a_win[sc]["2022 hike"][0], 6),
            "excess_days": built[sc]["exc"],
        })
    df = pd.DataFrame(rows)
    p = os.path.join(out_dir, "scale2_promotion_qc_20260626.csv")
    df.to_csv(p, index=False, float_format="%.6f", encoding="utf-8-sig")
    print("\nSaved: %s" % p)
    print("\n[SUMMARY] Stage1=%s | Scale2.0: MaxDD %+.2f%% gap %+.2fpp Regime %+.2f%% maxL %.1fx "
          "vs scale1.4 MaxDD %+.2f%% gap %+.2fpp"
          % ("REPRODUCED" if stage1_ok else "DRIFT",
             built[2.0]["pre"]["MaxDD_FULL"] * 100, built[2.0]["aft"]["IS_OOS_gap_pp"],
             built[2.0]["ev"]["regime_min_at"] * 100, max_eff_2,
             built[1.4]["pre"]["MaxDD_FULL"] * 100, built[1.4]["aft"]["IS_OOS_gap_pp"]))
    print("Done.")


if __name__ == "__main__":
    main()
