"""
src/audit/labor_zero_harness_v2_20260627.py
===========================================
Round-2 optimization harness for labor-backfill-zero (assets 40M, spend 7.2M/yr,
strict every-year withdrawal). Extends the prior labor_zero_allocation_sweep with
UNTRIED levers A-G that the prior (strategy x static-split x topup) sweep never
explored:

  A reserve_mode  : the RESERVE sleeve is invested (cash|nasdaq|gold|bond|sofr)
                    instead of idle cash. (Bond returns +6.7%/+15.3%/+26.3% in
                    1988/2000/2008 -- grows exactly when the run sleeve crashes.)
  B glide         : start at a lower-leverage strategy for the first K years
                    (protect the early sequence), then switch to high leverage.
  C mix           : hold a weighted blend of two strategies (same-year return mix).
  D init_bucket   : pre-hold init_bucket_years of spend as cash, drawn FIRST
                    (covers the sequence-risk window without selling the run).
  E draw_order    : on down years draw spend from reserve/bucket first (preserve
                    the compounding run sleeve).
  F (grid)        : extended strategy set incl. scale 2.4/2.6 (built via
                    dd_reduction harness, same after-tax convention).
  G regime_scale  : (round 2) regime-driven run fraction -- handled by passing a
                    per-year multiplier; checked for de-lever-in-disguise.

LABOR YEAR = a year where, after any top-up, total wealth (run + reserve + bucket)
cannot fund the 7.2M spend -> external earning required. Goal: ZERO over 31 starts.

SELF-TEST reproduces the prior best (sc2.2, run20M/res20M cash, thr20M, amt=ALL)
=> labor 12 years, only the 1988 start fails. This guarantees v2 is backward
compatible before any new lever is trusted.

ASCII-only prints. No commit, no temp files.
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

AR_DIR = os.path.join(_REPO_DIR, "audit_results")
SCALE_CSV = os.path.join(AR_DIR, "p09_strongmap_scale_dial_annual_20260623.csv")
X4N4_CSV = os.path.join(AR_DIR, "retirement_survival_x4n4_6M_backfill_annual_20260627.csv")
X4EQ_CSV = os.path.join(AR_DIR, "scale_dial_x4equiv_annual_20260627.csv")

M = 1_000_000.0
INIT_TOTAL = 40 * M
SPEND = 7.2 * M
HORIZON = 20
START_YEARS = list(range(1975, 2006))
AFTER_TAX = 0.8273

# cache for the (expensive) 1x sleeve series + extended scales
_SLEEVE_CACHE = None
_EXT_CACHE = None


def load_returns():
    """dict[strategy] -> Series(year->frac after-tax). Same set as prior sweep
    plus extended scales (2.4/2.6) appended lazily by load_extended()."""
    sc = pd.read_csv(SCALE_CSV).set_index("year")
    xn = pd.read_csv(X4N4_CSV).set_index("year")
    xe = pd.read_csv(X4EQ_CSV).set_index("year")
    out = {}
    for s in (1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4):
        col = "sc%.1f_strong_aftertax_pct" % s
        if col in sc.columns:
            out["sc%.1f" % s] = sc[col] / 100.0
    out["N4"] = xn["N4_aftertax_pct"] / 100.0
    out["X4"] = xn["X4_aftertax_pct"] / 100.0
    for s in (1.4, 1.6, 1.8, 2.2):
        out["X4eq_sc%.1f" % s] = xe["X4eq_sc%.1f_pct" % s] / 100.0
    out["NASDAQ_1x"] = sc["NASDAQ_1x_BH_aftertax_pct"] / 100.0
    for k in list(out):
        s = out[k].dropna()
        out[k] = s[(s.index >= 1975) & (s.index <= 2025)]
    return out


def load_sleeve_returns():
    """dict[mode] -> Series(year->frac after-tax) for the RESERVE sleeve.
    cash=0 always; nasdaq/gold/bond/sofr from dd_reduction harness daily->calendar."""
    global _SLEEVE_CACHE
    if _SLEEVE_CACHE is not None:
        return _SLEEVE_CACHE
    import src.audit.dd_reduction_harness_20260626 as H
    from src.audit.run_p01_backtest_20260611 import _calendar_year_returns
    ctx = H.setup()
    dates = ctx["dates_dt"]

    def _cy(r):
        nav = np.cumprod(1.0 + np.asarray(r, float))
        cy = _calendar_year_returns(pd.Series(nav, index=dates))
        cy = cy[(cy.index >= 1975) & (cy.index <= 2025)] * AFTER_TAX
        return cy

    years = list(range(1975, 2026))
    out = {
        "cash": pd.Series(0.0, index=years),
        "nasdaq": _cy(ctx["ret_ndx"]),
        "gold": _cy(ctx["ret_gold"]),
        "bond": _cy(ctx["ret_bond"]),
        "sofr": _cy(ctx["sofr_arr"]),
    }
    _SLEEVE_CACHE = out
    return out


def load_extended():
    """dict for extended leverage scales 2.4/2.6 (strong-map x scale), after-tax
    calendar-year, same builder/convention as the scale dial. Cached."""
    global _EXT_CACHE
    if _EXT_CACHE is not None:
        return _EXT_CACHE
    import src.audit.dd_reduction_harness_20260626 as H
    from src.audit.run_p01_backtest_20260611 import _calendar_year_returns
    ctx = H.setup()
    dates = ctx["dates_dt"]
    out = {}
    for s in (2.4, 2.6):
        nav, r, tpy, exc = H.build(ctx, scale=s)
        cy = _calendar_year_returns(nav)
        cy = cy[(cy.index >= 1975) & (cy.index <= 2025)] * AFTER_TAX
        out["sc%.1f" % s] = cy
    _EXT_CACHE = out
    return out


def _strat_year_return(rets, mix, glide, single, yr, k):
    """Return the strategy sleeve's return for calendar year yr (elapsed k)."""
    if mix is not None:
        return sum(w * float(rets[s].loc[yr]) for s, w in mix.items())
    if glide is not None:
        # glide = list of (elapsed_threshold, strat_key); pick the key whose
        # threshold is the largest <= k.
        key = glide[0][1]
        for thr_k, sk in glide:
            if k >= thr_k:
                key = sk
        return float(rets[key].loc[yr])
    return float(rets[single].loc[yr])


def simulate_v2(rets, sleeves, start, *, single=None, run0, reserve0,
                reserve_mode="cash", glide=None, mix=None,
                init_bucket_years=0, draw_order="run_first",
                topup_thr=20 * M, topup_amt=None, spend=SPEND, horizon=HORIZON,
                regime_scale=None):
    """Simulate one start. Strict spend every year. Returns labor_years etc.

    Order each year k (calendar yr = start+k):
      1. TOP-UP: if run < topup_thr and reserve > 0 -> move min(reserve, amt) into
         run (amt=None means ALL remaining reserve).
      2. SPEND (strict 7.2M): drawn from bucket first (init_bucket cash), then per
         draw_order from run/reserve. If total (run+reserve+bucket) < spend -> LABOR.
      3. GROWTH: run sleeve gets the strategy return (mix/glide/single), optionally
         scaled by regime_scale[k] around cash; reserve gets reserve_mode return;
         bucket is cash (no growth).
    """
    run = float(run0)
    reserve = float(reserve0)
    bucket = float(init_bucket_years) * spend   # extra cash sleeve, drawn first
    sleeve_ret = sleeves[reserve_mode]
    labor_years = 0
    topups = 0
    min_total = run + reserve + bucket
    for k in range(horizon):
        yr = start + k
        # 1. top-up (reserve -> run)
        if run < topup_thr and reserve > 1e-6:
            move = reserve if topup_amt is None else min(topup_amt, reserve)
            run += move
            reserve -= move
            topups += 1
        # 2. strict spend
        total = run + reserve + bucket
        if total + 1e-6 < spend:
            labor_years += 1
            run = reserve = bucket = 0.0
        else:
            need = spend
            # bucket first
            take = min(bucket, need); bucket -= take; need -= take
            if need > 1e-9:
                r_this = _strat_year_return(rets, mix, glide, single, yr, k)
                down = r_this < 0
                if draw_order == "reserve_first_on_down" and down:
                    take = min(reserve, need); reserve -= take; need -= take
                    if need > 1e-9:
                        take = min(run, need); run -= take; need -= take
                else:
                    take = min(run, need); run -= take; need -= take
                    if need > 1e-9:
                        take = min(reserve, need); reserve -= take; need -= take
        # 3. growth
        r_this = _strat_year_return(rets, mix, glide, single, yr, k)
        if regime_scale is not None:
            s = float(np.clip(regime_scale[k], 0.0, 2.0))
            cash_r = float(sleeves["sofr"].loc[yr])
            r_this = cash_r + s * (r_this - cash_r)
        run *= (1.0 + r_this)
        reserve *= (1.0 + float(sleeve_ret.loc[yr]))
        total = run + reserve + bucket
        if total < min_total:
            min_total = total
    ruin = (run + reserve + bucket) <= 1e-6
    return dict(labor_years=labor_years, topups=topups, ruin=int(ruin),
                terminal=run + reserve + bucket, min_total=min_total)


def run_all_starts(rets, sleeves, **kw):
    """Aggregate one config over all 31 starts. Returns summary dict."""
    labor = 0
    topup = 0
    ruin = 0
    terms = []
    floors = []
    fails = []
    for sy in START_YEARS:
        res = simulate_v2(rets, sleeves, sy, **kw)
        labor += res["labor_years"]
        topup += res["topups"]
        ruin += res["ruin"]
        if res["labor_years"] > 0:
            fails.append(sy)
        if res["ruin"] == 0:
            terms.append(res["terminal"])
        floors.append(res["min_total"])
    return dict(labor_years_total=labor, starts_with_labor=len(fails),
                fails=fails, saved_1988=(1988 not in fails),
                topup_events_total=topup, ruin_total=ruin,
                terminal_median_M=(float(np.median(terms)) / M if terms else 0.0),
                terminal_min_M=(float(np.min(terms)) / M if terms else 0.0),
                min_total_floor_M=float(np.min(floors)) / M)


def _self_test():
    sys.stdout.reconfigure(encoding="utf-8")
    rets = load_returns()
    sleeves = load_sleeve_returns()
    r = run_all_starts(rets, sleeves, single="sc2.2", run0=20 * M, reserve0=20 * M,
                       reserve_mode="cash", topup_thr=20 * M, topup_amt=None)
    print("SELF-TEST v2 (sc2.2 run20M/res20M cash thr20M amt=ALL spend7.2M):")
    print("  labor_years_total=%d  fails=%s  ruin=%d  termMed=%.1fM"
          % (r["labor_years_total"], r["fails"], r["ruin_total"], r["terminal_median_M"]))
    ok = (r["labor_years_total"] == 12 and r["fails"] == [1988])
    print("  -> %s" % ("MATCH prior best (12 labor, 1988 only)" if ok
                       else "MISMATCH -- FIX before trusting new levers"))
    return ok


if __name__ == "__main__":
    _self_test()
