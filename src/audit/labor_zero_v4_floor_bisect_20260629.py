"""
src/audit/labor_zero_v4_floor_bisect_20260629.py
================================================
Pin the EXACT max guaranteed REAL floor per inflation g (the H1/H234 grid jumped
5.5 -> 6.0, so the true 2-3% ceiling lies inside [5.5, 6.0)). Bisection on
floor_real with the v4 Round-G sim, bond reserve, run-fraction chosen per g.
Also identify the BINDING start year (which start's worst-real-spend == the floor)
to confirm it is the 2012-2014 window (the same structural wall as v3).

ASCII-only. Writes audit_results/labor_zero_v4_floor_bisect_20260629.csv. No commit.
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


def worst_and_binding(rets, bond, *, runfrac, top_wr, g, floor_real):
    """Return (min worst-real-spend over starts, ruin_count, binding_start)."""
    run0 = runfrac * 40 * M
    res0 = 40 * M - run0
    thr = min(14 * M, run0)
    worst = 1e18
    binding = None
    ruin = 0
    labor = 0
    for sy in STARTS:
        r = v4.sim_v4(rets, bond, sy, 20, strat="sc2.6", run0=run0, reserve0=res0,
                      thr=thr, floor_real=floor_real, top_wr=top_wr, g=g)
        labor += r["labor_years"]
        ruin += r["ruin"]
        if r["worst_real_spend_M"] < worst:
            worst = r["worst_real_spend_M"]
            binding = sy
    return worst, ruin, labor, binding


def max_floor(rets, bond, *, runfrac, top_wr, g, mode, lo=3.0, hi=8.0, tol=0.01):
    """Bisection: largest floor_real (M) feasible under `mode`.
      mode='survive'  : labor=0 AND ruin=0 (TRUE survival floor; the 2012-14 wall).
                        worst realised real spend may dip below floor in a thin
                        terminal year, but nobody is forced to external work / ruin.
      mode='deliver'  : labor=0 AND ruin=0 AND realised worst real spend >= floor
                        (the floor is honoured EVERY year, incl indexed terminals)."""
    def feasible(fM):
        w, ruin, labor, _ = worst_and_binding(rets, bond, runfrac=runfrac,
                                               top_wr=top_wr, g=g, floor_real=fM * M)
        if mode == "survive":
            return labor == 0 and ruin == 0
        return labor == 0 and ruin == 0 and w >= fM - 1e-6
    if not feasible(lo):
        return None, None, None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    w, _, _, binding = worst_and_binding(rets, bond, runfrac=runfrac, top_wr=top_wr,
                                         g=g, floor_real=lo * M)
    return round(lo, 3), binding, round(w, 3)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    print("=" * 95)
    print("EXACT max guaranteed REAL floor per inflation g (bisection, bond reserve, best runfrac/top)")
    print("=" * 95)
    rows = []
    # for each g, scan a few (runfrac, top_wr) and take the max achievable floor,
    # under BOTH definitions: 'survive' (no labor/ruin) and 'deliver' (floor honoured every yr).
    for g in [0.0, 0.01, 0.02, 0.025, 0.03, 0.035]:
        best = {"survive": None, "deliver": None}
        for mode in ("survive", "deliver"):
            for runfrac in [0.45, 0.50, 0.55, 0.60]:
                for top_wr in [0.16, 0.18, 0.20, 0.22]:
                    f, binding, w = max_floor(rets, bond, runfrac=runfrac,
                                              top_wr=top_wr, g=g, mode=mode)
                    if f is not None and (best[mode] is None or f > best[mode]["floor"]):
                        best[mode] = dict(g=g, mode=mode, runfrac=runfrac, top_wr=top_wr,
                                          floor=f, binding_start=binding, worst_real=w)
        for mode in ("survive", "deliver"):
            rows.append(best[mode])
        s, d = best["survive"], best["deliver"]
        print("  g=%.1f%%: SURVIVE floor=%.3fM (run%.2f top%.0f%% bind=%s worstReal=%.2f) | DELIVER floor=%.3fM (run%.2f top%.0f%% bind=%s)"
              % (g * 100, s["floor"], s["runfrac"], s["top_wr"] * 100, s["binding_start"],
                 s["worst_real"], d["floor"], d["runfrac"], d["top_wr"] * 100, d["binding_start"]))
    pd.DataFrame(rows).to_csv(os.path.join(AR, "labor_zero_v4_floor_bisect_20260629.csv"),
                              index=False, encoding="utf-8-sig")
    print("\nDone (exact real-floor frontier).")


if __name__ == "__main__":
    main()
