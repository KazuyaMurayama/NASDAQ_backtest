"""
src/audit/dd_reduction_finalqc_20260626.py
==========================================
Final path-robust QC for the top DD-reduction candidates (N4 / X4 / X3) vs sc2.0
and vs their equal-MaxDD uniform twin. 2026-06-26.

Tests (R-STAT compliant -- no block=21 bootstrap on path extrema):
  1. UNTOUCHED-WINDOW losses (R-STAT-2): cumulative return inside the sc2.0 MaxDD
     windows (2004-01-20..2005-04-05 and 2014-07..2016-06) for sc2.0, each
     candidate, and the candidate's equal-MaxDD uniform twin. If the candidate
     beats BOTH sc2.0 AND its twin inside these windows, the frontier gap is a
     genuine reshaping of the MaxDD legs, not whole-sample dilution.
  2. PATH-AGGREGATE measures (TUW / avgDD / CVaR5%DD) vs sc2.0 (block-free).
  3. CALENDAR-YEAR returns (after-tax) for the candidates vs sc2.0, to see WHERE
     the gold/bond IN blend helps (the bleed years) vs hurts (strong bull years).

Twin scales come from dd_reduction_verify (bisection equal-MaxDD): N4~1.88, X4~1.92, X3~1.88.
Writes audit_results/dd_reduction_finalqc_20260626.csv (year x series).
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import src.audit.dd_reduction_harness_20260626 as H
from src.audit.run_p01_backtest_20260611 import _calendar_year_returns

AFTER_TAX = 0.8273

TOP = [
    ("N4_g28b07_s2.85", dict(scale=2.85, in_gold_w=0.28, in_bond_w=0.07), 1.88),
    ("X4_g32b05_s3.0",  dict(scale=3.0, in_gold_w=0.32, in_bond_w=0.05), 1.92),
    ("X3_g28b07_cf05",  dict(scale=2.85, in_gold_w=0.28, in_bond_w=0.07, cash_floor=0.05), 1.88),
]

# sc2.0 MaxDD windows (peak -> recovery), from the path decomposition.
WINDOWS = [
    ("2004-05 grind", "2004-01-20", "2006-08-30"),
    ("2015-16 corr",  "2014-07-01", "2017-04-30"),
    ("1990 bear",     "1989-06-01", "1990-12-31"),
]


def _nav(ctx, **kw):
    nav_dt, r, tpy, exc = H.build(ctx, **kw)
    return np.asarray(nav_dt.values, float), nav_dt


def _win_loss(nav, dates_dt, start, end):
    m = (dates_dt >= pd.Timestamp(start)) & (dates_dt <= pd.Timestamp(end))
    seg = nav[np.asarray(m)]
    if len(seg) < 2:
        return np.nan
    # worst peak-to-trough drawdown WITHIN the window (path-robust, untouched)
    run = np.maximum.accumulate(seg)
    return float((seg / run - 1.0).min())


def _aggs(nav):
    run = np.maximum.accumulate(nav)
    dd = nav / run - 1.0
    tuw = float(np.mean(dd < -1e-9))
    avg = float(np.mean(dd))
    q = np.quantile(dd, 0.05)
    cvar = float(np.mean(dd[dd <= q])) if np.any(dd <= q) else float(q)
    return tuw, avg, cvar


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("DD-REDUCTION FINAL PATH-ROBUST QC (top candidates vs sc2.0 vs equal-MaxDD twin) 2026-06-26")
    print("=" * 100)
    ctx = H.setup()
    dates_dt = ctx["dates_dt"]

    nav_sc20, _ = _nav(ctx, scale=2.0)

    print("\n--- untouched MaxDD-window worst drawdown (R-STAT-2; candidate must beat sc2.0 AND twin) ---")
    print("%-18s | %-12s | %-12s | %-12s" % ("window", "sc2.0", "candidate", "twin(=MaxDD)"))
    for name, kw, twin_s in TOP:
        nav_c, _ = _nav(ctx, **kw)
        nav_t, _ = _nav(ctx, scale=twin_s)
        print("CANDIDATE %s (twin scale=%.2f):" % (name, twin_s))
        for wl, st, en in WINDOWS:
            l0 = _win_loss(nav_sc20, dates_dt, st, en)
            lc = _win_loss(nav_c, dates_dt, st, en)
            lt = _win_loss(nav_t, dates_dt, st, en)
            beats = (lc > l0) and (lc > lt)
            print("  %-16s | %+8.2f%%   | %+8.2f%%   | %+8.2f%%   %s"
                  % (wl, l0*100, lc*100, lt*100, "[cand reshapes window]" if beats else ""))

    print("\n--- path-aggregate measures vs sc2.0 (block-free; lower magnitude = better) ---")
    t0, a0, c0 = _aggs(nav_sc20)
    print("  %-18s TUW=%.1f%% avgDD=%+6.2f%% CVaR5%%DD=%+6.2f%%" % ("sc2.0", t0*100, a0*100, c0*100))
    for name, kw, _ in TOP:
        nav_c, _ = _nav(ctx, **kw)
        t, a, c = _aggs(nav_c)
        print("  %-18s TUW=%.1f%% avgDD=%+6.2f%% CVaR5%%DD=%+6.2f%%  (dAvgDD %+.2fpp dCVaR %+.2fpp)"
              % (name, t*100, a*100, c*100, (a-a0)*100, (c-c0)*100))

    print("\n--- calendar-year returns (after-tax) where blend helps/hurts vs sc2.0 ---")
    cy_sc20 = _calendar_year_returns(pd.Series(nav_sc20, index=dates_dt)) * AFTER_TAX
    cy_map = {"sc2.0": cy_sc20}
    for name, kw, _ in TOP:
        _, nav_dt = _nav(ctx, **kw)
        cy_map[name] = _calendar_year_returns(nav_dt) * AFTER_TAX
    years = sorted(cy_sc20.index)
    # show the bleed years (where gold/bond IN blend should help) and bull years (where it costs)
    focus = [1977, 1990, 2002, 2004, 2008, 2011, 2015, 2018, 2022,  # bleed/bear
             1995, 1999, 2003, 2013, 2020, 2023]                     # strong bull
    print("  year | " + " | ".join("%-10s" % k for k in cy_map))
    rows = []
    for y in years:
        row = {"year": y}
        for k, cy in cy_map.items():
            row[k] = round(float(cy.loc[y])*100, 2) if y in cy.index else np.nan
        rows.append(row)
        if y in focus:
            print("  %4d | " % y + " | ".join("%+9.1f%%" % row[k] if not np.isnan(row[k]) else "    n/a   " for k in cy_map))

    df = pd.DataFrame(rows)
    out = os.path.join(_REPO_DIR, "audit_results", "dd_reduction_finalqc_20260626.csv")
    df.to_csv(out, index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\nSaved: %s" % out)
    print("Done.")


if __name__ == "__main__":
    main()
