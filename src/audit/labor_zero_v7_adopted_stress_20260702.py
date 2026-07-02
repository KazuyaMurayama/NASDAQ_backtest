"""
src/audit/labor_zero_v7_adopted_stress_20260702.py
==================================================
Stress numbers for the ADOPTED v7 operating rule (user decision 2026-07-02):
floor 3.6M + top_wr 0.12 + MONTHLY refill (all-in when run sleeve < 12M,
no signal gate), run 20M / reserve 20M, sc2.6 / 1x bond.

Fills the gap that the campaign's stress harness (labor_zero_v6_stress) only
covered the OLD rule shape (annual refill, top_wr 0.16):
  [1] uniform return shifts (run -2pp / -5pp / run-2pp+bond-1pp) -> P
  [2] max floor with all-seed P>=0.999 under the -2pp world
  [3] worst-3y-window (2014-2016) forced to the retirement start (conditional)

Shifted monthly returns keep each year's monthly SHAPE and are geometrically
adjusted so the 12-month compound equals the shifted annual anchor
(same construction as labor_zero_v6_monthly_20260702.build_monthly).

ASCII-only prints. Appends nothing; CSV -> audit_results/labor_zero_v7_adopted_stress_20260702.csv
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, _REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "fu", os.path.join(_THIS, "labor_zero_v6_floorup_20260702.py"))
fu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fu)
lm, la, v6, lz = fu.lm, fu.la, fu.v6, fu.lzstats

M = 1e6
SEEDS = [20260629 + i for i in range(5)]
N = 2000
ADOPT = dict(run0=20 * M, reserve0=20 * M, thr=12 * M, rule="M-noH",
             top_wr=0.12)                      # floor passed per call


def shift_monthly(mat, ann, d):
    """Re-anchor monthly matrix so each year's compound = (1+ann+d)."""
    out = np.empty_like(mat)
    for yi in range(mat.shape[0]):
        pre = float(np.prod(1.0 + mat[yi]))
        g = (1.0 + float(ann[yi]) + d) / pre
        out[yi] = (1.0 + mat[yi]) * g ** (1.0 / 12.0) - 1.0
    return out


def pmc(run_m, res_m, floor, seeds=SEEDS, force_front=None):
    ps = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        lab = np.empty(N, int)
        for p in range(N):
            if force_front is None:
                yidx = lm.sample_year_indices(rng)
            else:
                tail = lm.sample_year_indices(rng, n_years=20 - len(force_front))
                yidx = np.concatenate([force_front, tail])
            lab[p] = fu.sim_m2(yidx, run_m, res_m, floor=floor, **ADOPT)["labor_years"]
        ps.append(float((lab == 0).mean()))
    return np.array(ps)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    run_h, res_h = v6.get_paired_history()
    run_m, res_m = lm.build_monthly()
    print("adopted rule: floor3.6M top0.12 M-noH12 (monthly all-in below 12M), 20:20")

    print("\n[1] uniform return shift -> P(labor0), floor=3.6M")
    scen = {
        "S0_none": (run_m, res_m),
        "Sa_run-2pp": (shift_monthly(run_m, run_h, -0.02), res_m),
        "Sb_run-5pp": (shift_monthly(run_m, run_h, -0.05), res_m),
        "Sc_run-2_bond-1": (shift_monthly(run_m, run_h, -0.02),
                            shift_monthly(res_m, res_h, -0.01)),
    }
    rows = []
    for name, (rm, sm) in scen.items():
        ps = pmc(rm, sm, 3.6 * M)
        print("  %-16s P=%.4f [%.4f-%.4f]" % (name, ps.mean(), ps.min(), ps.max()))
        rows.append(dict(section="shift", scenario=name, floor=3.6,
                         p_mean=ps.mean(), p_min=ps.min(), p_max=ps.max()))

    print("\n[2] max floor with all-seed P>=0.999 under Sa (-2pp)")
    rm, sm = scen["Sa_run-2pp"]
    best = None
    for fl in (2.0, 2.4, 2.8, 3.2, 3.6):
        ps = pmc(rm, sm, fl * M)
        print("  floor=%.1fM: P=%.4f [minseed %.4f]" % (fl, ps.mean(), ps.min()))
        rows.append(dict(section="maxfloor_-2pp", scenario="Sa", floor=fl,
                         p_mean=ps.mean(), p_min=ps.min(), p_max=ps.max()))
        if ps.min() >= 0.999:
            best = fl
    print("  -> max floor P>=0.999 (all seeds) under -2pp: %s"
          % (("%.1fM" % best) if best else "none<=3.6M"))

    print("\n[3] worst-3y window (2014-2016) forced to retirement start (conditional)")
    yrs = list(range(1975, 2026))
    front = np.array([yrs.index(2014), yrs.index(2015), yrs.index(2016)])
    ps = pmc(run_m, res_m, 3.6 * M, force_front=front)
    print("  adopted rule: P=%.4f [%.4f-%.4f]" % (ps.mean(), ps.min(), ps.max()))
    rows.append(dict(section="worst3y_front", scenario="2014-2016", floor=3.6,
                     p_mean=ps.mean(), p_min=ps.min(), p_max=ps.max()))

    out = os.path.join(_REPO, "audit_results",
                       "labor_zero_v7_adopted_stress_20260702.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (v7 adopted-rule stress).")


if __name__ == "__main__":
    main()
