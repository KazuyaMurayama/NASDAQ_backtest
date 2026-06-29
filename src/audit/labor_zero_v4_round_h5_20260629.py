"""
src/audit/labor_zero_v4_round_h5_20260629.py
============================================
ROUND H5 -- target the ACTUAL binding constraint found by the bisection: under
inflation the floor is bound by the LONGEST full-horizon start (1975), because the
indexed floor compounds by (1+g)^19 over 20 years -- NOT by the 2012-14 crash window.

Lever: the "retirement smile" / real-floor glide. Empirically, real spending in
retirement declines with age (go-go -> slow-go -> no-go). So we let the GUARANTEED
real floor drift DOWN by d/yr after a hinge year (default after year 10). This
directly relieves the late-horizon indexation burden that binds 1975, and should
lift the EARLY-years real floor (the years a 60-something actually spends most).

We report, per inflation g:
  - flat-floor DELIVER ceiling (H1/bisect baseline, glide d=0)
  - smile-floor: max EARLY (years 0..hinge) real floor with labor0/ruin0, where the
    floor glides down at d/yr afterwards. This is the honest "how much can I guarantee
    in my active years" number.

Compares d in {0, -0.5%, -1.0%, -1.5%/yr after year 10}.
ASCII-only. Writes audit_results/labor_zero_v4_roundH5_smile_20260629.csv. No commit.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (os.path.dirname(_THIS), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_spec = importlib.util.spec_from_file_location(
    "v4", os.path.join(_THIS, "labor_zero_v4_inflation_20260629.py"))
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)
M = v4.M
v3 = v4.v3
AR = os.path.join(_REPO, "audit_results")
STARTS = v4.ALL_STARTS
FULL_REAL = v4.FULL_REAL
DATA_END = v4.DATA_END


def sim_smile(rets, bond, sy, *, runfrac, top_wr, g, early_floor_real, glide, hinge=10):
    """Floor real value = early_floor for k<hinge, then *(1+glide)^(k-hinge) after.
    glide<=0 (decline). Spending each year = max(smile floor indexed, opportunistic full).
    Returns (labor, ruin, worst_real_spend over years, worst_EARLY_real_spend)."""
    run0 = runfrac * 40 * M
    res0 = 40 * M - run0
    thr = min(14 * M, run0)
    run, res = float(run0), float(res0)
    s = rets["sc2.6"]
    n = min(20, DATA_END - sy + 1)
    fired = False
    labor = 0
    reals = []
    early_reals = []
    for k in range(n):
        yr = sy + k
        infl = (1.0 + g) ** k
        r_this = float(s.loc[yr])
        # smile real floor (in yr1 purchasing power)
        if k < hinge:
            floor_real = early_floor_real
        else:
            floor_real = early_floor_real * ((1.0 + glide) ** (k - hinge))
        if (not fired) and run < thr and res > 1e-6 and r_this >= 0:
            run += res
            res = 0.0
            fired = True
        total = run + res
        floor_nom = floor_real * infl
        full_nom = FULL_REAL * infl
        spend = floor_nom
        if total > 1e-9 and full_nom / total <= top_wr:
            spend = full_nom
        if total + 1e-6 < spend:
            labor += 1
            spend = total
        rr = spend / infl
        reals.append(rr)
        if k < hinge:
            early_reals.append(rr)
        if run >= spend:
            run -= spend
        else:
            res -= (spend - run)
            run = 0.0
        run *= (1.0 + r_this)
        res *= (1.0 + float(bond[yr]))
    ruin = int((run + res) <= 1e-6)
    return labor, ruin, (min(reals) if reals else 0.0), (min(early_reals) if early_reals else 0.0)


def max_early_floor(rets, bond, *, runfrac, top_wr, g, glide, hinge=10, lo=4.0, hi=8.0, tol=0.01):
    def feasible(fM):
        tot_labor = tot_ruin = 0
        for sy in STARTS:
            lab, ru, _, _ = sim_smile(rets, bond, sy, runfrac=runfrac, top_wr=top_wr,
                                      g=g, early_floor_real=fM * M, glide=glide, hinge=hinge)
            tot_labor += lab
            tot_ruin += ru
        return tot_labor == 0 and tot_ruin == 0
    if not feasible(lo):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return round(lo, 3)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    print("=" * 95)
    print("ROUND H5: retirement-smile floor glide -- relieve the 1975 long-horizon indexation wall")
    print("=" * 95)
    print("  early_floor = guaranteed REAL spend in active years (k<10); declines at glide/yr after.")
    rows = []
    for g in [0.0, 0.02, 0.03]:
        line = []
        for glide in [0.0, -0.005, -0.010, -0.015]:
            best = None
            for runfrac in [0.45, 0.50, 0.55, 0.60]:
                for top_wr in [0.16, 0.18, 0.20]:
                    f = max_early_floor(rets, bond, runfrac=runfrac, top_wr=top_wr,
                                        g=g, glide=glide)
                    if f is not None and (best is None or f > best["early_floor"]):
                        best = dict(g=g, glide=glide, runfrac=runfrac, top_wr=top_wr,
                                    early_floor=f)
            rows.append(best)
            line.append((glide, best["early_floor"], best["runfrac"], best["top_wr"]))
        print("\n  g=%.1f%%:" % (g * 100))
        for glide, ef, rf, tw in line:
            tag = "flat " if glide == 0 else ("%.1f%%/yr" % (glide * 100))
            print("    glide %-8s -> active-years REAL floor = %.3fM  (run%.2f top%.0f%%)"
                  % (tag, ef, rf, tw * 100))
    pd.DataFrame(rows).to_csv(os.path.join(AR, "labor_zero_v4_roundH5_smile_20260629.csv"),
                              index=False, encoding="utf-8-sig")
    print("\nDone (Round H5 smile).")


if __name__ == "__main__":
    main()
