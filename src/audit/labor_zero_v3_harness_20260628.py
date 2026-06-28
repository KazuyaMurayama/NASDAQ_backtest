"""
src/audit/labor_zero_v3_harness_20260628.py
===========================================
v3 harness for labor-backfill-zero robustness. Fixes the v2 over-fit found in
LABOR_ZERO_V2_HOLDOUT_20260628.md (v2 fails 2012/2013/2014 starts because its
single all-in reserve bullet assumes the worst crash arrives first).

New levers over v2:
  - SPLIT-TRANCHE refill: reserve is deployed in N tranches at descending run-
    balance thresholds, each tranche investing a fraction of the THEN-REMAINING
    reserve. Keeps dry powder for a crash that arrives AFTER the first refill.
      tranches = [(thr_0, frac_0), (thr_1, frac_1), ...]  thr descending.
      v2 == [(14e6, 1.0)]  (single threshold, all-in).
  - MIXED reserve: reserve return = weighted blend of bond/gold/cash/sofr
    (load_mixed_reserve), so a "stocks AND bonds both down" year (e.g. 2015) can
    be partly cushioned by cash/gold.
  - EXTENDED start set: evaluate on ALL 46 starts 1975..2020 (not just IS 31),
    horizon capped at min(H, data_end-start+1).

Goal (user, strict): labor_years == 0 across ALL 46 starts x data-max horizon.

SELF-TEST reproduces v2 as the special case (single tranche, bond-only reserve):
  (a) IS 1975-2005 x20y      -> labor 0, floor ~18.12M
  (b) holdout 2006-2020 max  -> 3 starts fail (2012/2013/2014)
If either fails to reproduce, HALT (the v3 foundation is wrong).

ASCII-only prints. No commit, no temp files.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "h2", os.path.join(_THIS_DIR, "labor_zero_harness_v2_20260627.py"))
h2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h2)

M = 1_000_000.0
SPEND = 7.2 * M
DATA_END = 2025
ALL_STARTS = list(range(1975, 2021))      # 46 starts (1975..2020)
IS_STARTS = list(range(1975, 2006))       # v2 in-sample
HOLDOUT_STARTS = list(range(2006, 2021))  # v2 holdout

_RETS_CACHE = None
_SLEEVE_CACHE = None


def load_rets():
    global _RETS_CACHE
    if _RETS_CACHE is None:
        r = h2.load_returns()
        r.update(h2.load_extended())
        _RETS_CACHE = r
    return _RETS_CACHE


def load_sleeves():
    global _SLEEVE_CACHE
    if _SLEEVE_CACHE is None:
        _SLEEVE_CACHE = h2.load_sleeve_returns()
    return _SLEEVE_CACHE


def load_mixed_reserve(weights):
    """weights: dict over {bond,gold,cash,sofr,nasdaq}; values sum to 1.
    Returns Series(year->frac after-tax) = weighted blend of the sleeve modes."""
    sl = load_sleeves()
    tot = sum(weights.values())
    assert abs(tot - 1.0) < 1e-9, "reserve weights must sum to 1 (got %.4f)" % tot
    years = sl["cash"].index
    out = pd.Series(0.0, index=years)
    for mode, w in weights.items():
        if w == 0:
            continue
        out = out.add(sl[mode] * w, fill_value=0.0)
    return out


def simulate_v3(rets, reserve_series, start, horizon, *, strat, run0, reserve0,
                tranches, spend=SPEND, data_end=DATA_END):
    """One start. tranches = [(threshold, frac_of_remaining_reserve), ...] with
    thresholds DESCENDING. Each tranche fires at most once, when run<threshold and
    that tranche not yet fired, moving frac*reserve into run. Strict spend.
    Returns dict(labor_years, floor_M, floor_yr, terminal_M, n_years, tranches_fired).
    """
    run, res = float(run0), float(reserve0)
    s = rets[strat]
    n = min(horizon, data_end - start + 1)
    fired = [False] * len(tranches)
    labor = 0
    floor = run + res
    floor_yr = start
    nfire = 0
    for k in range(n):
        yr = start + k
        # 1. split-tranche top-up: check each tranche in order (high thr first)
        for i, (thr, frac) in enumerate(tranches):
            if (not fired[i]) and run < thr and res > 1e-6:
                move = res * frac
                run += move
                res -= move
                fired[i] = True
                nfire += 1
        # 2. strict spend (run first, then reserve)
        total = run + res
        if total + 1e-6 < spend:
            labor += 1
            run = res = 0.0
        else:
            if run >= spend:
                run -= spend
            else:
                res -= (spend - run)
                run = 0.0
        # 3. growth
        run *= (1.0 + float(s.loc[yr]))
        res *= (1.0 + float(reserve_series.loc[yr]))
        tot = run + res
        if tot < floor:
            floor = tot
            floor_yr = yr
    return dict(labor_years=labor, floor_M=round(floor / M, 2), floor_yr=floor_yr,
                terminal_M=round((run + res) / M, 1), n_years=n, tranches_fired=nfire)


def run_all_starts_v3(rets, reserve_series, *, strat, run0, reserve0, tranches,
                      starts=ALL_STARTS, spend=SPEND, data_end=DATA_END,
                      horizon=20):
    """Aggregate over a start set at data-max horizon. Returns summary dict."""
    labor = 0
    fails = []
    floors = []
    terms = []
    for sy in starts:
        r = simulate_v3(rets, reserve_series, sy, horizon, strat=strat, run0=run0,
                        reserve0=reserve0, tranches=tranches, spend=spend,
                        data_end=data_end)
        labor += r["labor_years"]
        if r["labor_years"] > 0:
            fails.append(sy)
        floors.append(r["floor_M"])
        if r["terminal_M"] > 1e-3:
            terms.append(r["terminal_M"])
    terms_sorted = sorted(terms)
    med = terms_sorted[len(terms_sorted) // 2] if terms_sorted else 0.0
    return dict(labor_years_total=labor, starts_fail=fails, n_fail=len(fails),
                n_starts=len(starts), min_floor_M=float(np.min(floors)),
                term_median_M=med)


def _self_test():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 90)
    print("v3 HARNESS SELF-TEST -- reproduce v2 as the special case")
    print("  (single tranche [(14M,1.0)], reserve = bond-only)")
    print("=" * 90)
    rets = load_rets()
    bond_only = load_mixed_reserve({"bond": 1.0})
    v2cfg = dict(strat="sc2.6", run0=14 * M, reserve0=26 * M,
                 tranches=[(14 * M, 1.0)])

    # (a) IS 1975-2005 x20y
    a = run_all_starts_v3(rets, bond_only, starts=IS_STARTS, horizon=20, **v2cfg)
    print("\n(a) IS 1975-2005 x20y:")
    print("    labor=%d (exp 0)  floor=%.2fM (exp ~18.12)  fails=%s"
          % (a["labor_years_total"], a["min_floor_M"], a["starts_fail"]))
    ok_a = (a["labor_years_total"] == 0 and abs(a["min_floor_M"] - 18.12) < 0.2)

    # (b) holdout 2006-2020 data-max
    b = run_all_starts_v3(rets, bond_only, starts=HOLDOUT_STARTS, horizon=20, **v2cfg)
    print("\n(b) holdout 2006-2020 (data-max):")
    print("    labor=%d  fails=%s (exp [2012, 2013, 2014])  floor=%.2fM"
          % (b["labor_years_total"], b["starts_fail"], b["min_floor_M"]))
    ok_b = (b["starts_fail"] == [2012, 2013, 2014])

    # (c) all 46 starts (just report; v2 is NOT labor-zero here)
    c = run_all_starts_v3(rets, bond_only, starts=ALL_STARTS, horizon=20, **v2cfg)
    print("\n(c) ALL 46 starts 1975-2020 (v2 baseline -- expected to FAIL strict):")
    print("    labor=%d  n_fail=%d/46  fails=%s  floor=%.2fM"
          % (c["labor_years_total"], c["n_fail"], c["starts_fail"], c["min_floor_M"]))

    ok = ok_a and ok_b
    print("\n-> SELF-TEST %s" % ("PASS: v2 reproduced (IS labor=0 floor=18.12; "
                                 "holdout fails=[2012,2013,2014])" if ok
                                 else "FAIL -- FIX before trusting v3 levers"))
    return ok


if __name__ == "__main__":
    _self_test()
