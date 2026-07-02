"""
src/audit/labor_zero_v6_worst3y_front_20260702.py
=================================================
Task 6 addendum of the v6 critical-verification campaign: worst contiguous
3-YEAR run-sleeve window forced to the path front.

Motivation: the plan's worst-10-year variant turned out non-discriminating --
the sc2.6 run sleeve has NO negative decade in 1975-2025 (worst 10y window
2007-2016 still cumulates +663.9%), so every config scores P=1.0 there. The
binding sequence risk for this series is a short crash CLUSTER at retirement,
not a lost decade. This harness finds the worst contiguous 3-year window by
cumulative run return (no wrap) and conditions on it hitting years 0-2 of the
retirement, with the remaining 17 years block-bootstrapped (block=5).

NOTE: the resulting P values are CONDITIONAL on a crash-first start -- they are
crash-robustness measures, not unconditional probabilities.

Parent-verified result (2026-07-02): worst 3y window = 2014-2016 (cum -38.0%).
  lookahead: FA1 0.9707 / FA2 0.7829 / NC1 0.2655 / NB0 0.0000
  lagged   : FA1 0.8210 / FA2 0.7829 / NC1 0.0000 / NB0 0.0000
-> nominal-fixed collapses under a crash-first start regardless of refill rule
   (noHOLD's average superiority hides catastrophic sequence fragility); only
   the floor variant survives, and keeping the reserve (thr20) beats day-0
   all-in (thr26) in this conditional stress under BOTH conventions.

ASCII-only prints.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "la", os.path.join(_THIS, "labor_zero_v6_lookahead_20260702.py"))
la = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(la)
v6 = la.v6

M = 1e6
SEEDS = [20260629 + i for i in range(5)]

CFGS = {
    "FA1 floor3.6 thr20": dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                               floor=3.6 * M, top_wr=0.16),
    "FA2 floor3.6 thr26": dict(run0=20 * M, reserve0=20 * M, thr=26 * M,
                               floor=3.6 * M, top_wr=0.16),
    "NC1 nominal thr20 hold": dict(run0=20 * M, reserve0=20 * M, thr=20 * M),
    "NB0 nominal thr20 noHOLD": dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                                     hold_if_crash=False),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    run_h, res_h = v6.get_paired_history()
    yrs = list(range(1975, 2026))
    best = None
    for i in range(len(run_h) - 2):
        cum = float(np.prod(1.0 + run_h[i:i + 3]) - 1.0)
        if best is None or cum < best[1]:
            best = (i, cum)
    i0, cum = best
    print("worst 3y window: %d-%d  cum=%.1f%%  yearly=%s"
          % (yrs[i0], yrs[i0 + 2], cum * 100,
             ["%.1f%%" % (x * 100) for x in run_h[i0:i0 + 3]]))
    fixed_r = run_h[i0:i0 + 3]
    fixed_s = res_h[i0:i0 + 3]
    for conv in ("lookahead", "lagged"):
        print("\n[%s] worst-3y-front, tail 17y block=5 bootstrap, N=2000 x %d seeds"
              % (conv, len(SEEDS)))
        for name, cfg in CFGS.items():
            ps = []
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                lab = np.empty(2000, int)
                for p in range(2000):
                    rp, sp = v6.make_block_path(rng, run_h, res_h, n=17, block=5)
                    lab[p] = la.sim_conv(np.concatenate([fixed_r, rp]),
                                         np.concatenate([fixed_s, sp]),
                                         convention=conv, **cfg)["labor_years"]
                ps.append(float((lab == 0).mean()))
            ps = np.array(ps)
            print("  %-26s P=%.4f [%.4f-%.4f]" % (name, ps.mean(), ps.min(), ps.max()))
    print("\nDone (worst-3y-front conditional stress).")


if __name__ == "__main__":
    main()
