"""
src/audit/labor_zero_v4_qc_trace_20260629.py
============================================
INDEPENDENT QC of the v4 headline "2% real floor = 5.84M". Re-implements the
Round-G inflation sim from scratch (NOT importing sim_v4) and traces the binding
starts year-by-year, to confirm: labor=0 everywhere, and the worst ACTIVE-year
real spend at the claimed floor. Also dumps which start binds.

Config under test: sc2.6, run-fraction 0.45 (run 18M / reserve 22M bond),
hold-if-crash refill at run<14M (or run<18M? thr=min(14M,run0)=14M), guaranteed
REAL floor 5.84M indexed at g, top to full(7.2M indexed) when full/total<=0.16.

ASCII-only. Prints to stdout; writes nothing. No commit.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (os.path.dirname(_THIS), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_THIS, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
M = 1_000_000.0
DATA_END = 2025
STARTS = list(range(1975, 2021))


def trace(rets, bond, sy, *, floor_realM, g, runfrac=0.45, top_wr=0.16, verbose=False):
    run0 = runfrac * 40 * M
    res0 = 40 * M - run0
    thr = min(14 * M, run0)
    run, res = run0, res0
    s = rets["sc2.6"]
    n = min(20, DATA_END - sy + 1)
    fired = False
    labor = 0
    worst_active = 1e18
    worst_all = 1e18
    if verbose:
        print("  yr   r_run%%  refill  total(M)  spend_nom(M) spend_real(M) note")
    for k in range(n):
        yr = sy + k
        infl = (1.0 + g) ** k
        r = float(s.loc[yr])
        did_refill = ""
        if (not fired) and run < thr and res > 1e-6 and r >= 0:
            run += res; res = 0.0; fired = True; did_refill = "REFILL"
        total = run + res
        floor_nom = floor_realM * M * infl
        full_nom = 7.2 * M * infl
        spend = floor_nom
        note = "floor"
        if total > 1e-9 and full_nom / total <= top_wr:
            spend = full_nom; note = "FULL"
        if total + 1e-6 < spend:
            labor += 1; spend = total; note = "LABOR!"
        real = spend / infl
        worst_all = min(worst_all, real)
        if k < 10:
            worst_active = min(worst_active, real)
        if verbose:
            print("  %d  %+6.1f  %-6s  %8.2f  %10.2f   %10.2f   %s"
                  % (yr, r * 100, did_refill, total / M, spend / M, real / M, note))
        if run >= spend:
            run -= spend
        else:
            res -= (spend - run); run = 0.0
        run *= (1.0 + r)
        res *= (1.0 + float(bond[yr]))
    return labor, round(worst_active / M, 3), round(worst_all / M, 3), round((run + res) / M, 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    g = 0.02
    fM = 5.84
    print("=" * 95)
    print("QC: independent re-impl, g=2.0%%, real floor %.2fM, sc2.6 run0.45 top16%%, hold-if-crash" % fM)
    print("=" * 95)
    tot_labor = 0
    worst_active_all = 1e18
    worst_active_start = None
    for sy in STARTS:
        lab, wa, wall, term = trace(rets, bond, sy, floor_realM=fM, g=g)
        tot_labor += lab
        if wa < worst_active_all:
            worst_active_all = wa
            worst_active_start = sy
    print("  total labor years across 46 starts = %d (expect 0)" % tot_labor)
    print("  worst ACTIVE-year real spend across all starts = %.3fM at start %d"
          % (worst_active_all, worst_active_start))
    print("\n  --- trace of the binding start %d (active years should hold >= %.2fM) ---" % (worst_active_start, fM))
    trace(rets, bond, worst_active_start, floor_realM=fM, g=g, verbose=True)
    print("\n  --- trace of 1975 (longest horizon; check terminal indexation) ---")
    trace(rets, bond, 1975, floor_realM=fM, g=g, verbose=True)
    # cross-check: bump floor to 5.95 -> expect some start to break labor=0
    print("\n  CROSS-CHECK: raise real floor to 5.95M -> expect labor>0 somewhere")
    tl = 0
    for sy in STARTS:
        lab, _, _, _ = trace(rets, bond, sy, floor_realM=5.95, g=g)
        tl += lab
    print("  total labor at floor 5.95M = %d (expect >0, confirming 5.84M near the ceiling)" % tl)
    print("\nDone (QC trace).")


if __name__ == "__main__":
    main()
