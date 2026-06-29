"""
src/audit/labor_zero_v5_mc_inflation_20260629.py
===============================================
v5 -- STOCHASTIC inflation (Monte Carlo). v4 used a deterministic constant g; here
the inflation PATH varies year-to-year. Goal: how does the GUARANTEED REAL floor
(the v4 headline) scatter when inflation is random rather than fixed at the mean?

Model (Japan-realistic, mean-reverting AR(1) on annual CPI):
  pi_0 = mu
  pi_{t} = mu + rho*(pi_{t-1} - mu) + eps_t,   eps_t ~ N(0, sigma), pi clipped to [floor_pi, cap_pi]
  mu = long-run mean (BoJ target 2%), rho = persistence (0.5), sigma = annual shock (1.0%).
Spending floor and full target are indexed by the CUMULATIVE realised CPI:
  cum_t = prod_{j<=t}(1+pi_j).   floor_nom_t = floor_real * cum_t, full_nom_t = 7.2M * cum_t.
Realised real spend deflated by cum_t. Strategy = v3/v4 Round-G (sc2.6, run-fraction,
hold-if-crash, bond reserve).

For each MC inflation path we measure, across ALL 46 starts at data-max horizon, the
max REAL floor that keeps labor=0 AND ruin=0 (bisection). Distribution over N paths:
median / p5 / p1 / worst. Compares mu in {0%(deflation), 2%(BoJ), 3%(recent)}.

Reproducibility note: Math.random/Date are fine in plain Python (this is a script, not
a workflow). We pass a fixed seed via numpy default_rng(seed) for determinism.

ASCII-only. Writes audit_results/labor_zero_v5_mc_inflation_20260629.csv. No commit.
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
    "v3", os.path.join(_THIS, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
M = 1_000_000.0
FULL_REAL = 7.2 * M
DATA_END = 2025
STARTS = list(range(1975, 2021))


def make_path(rng, mu, rho=0.5, sigma=0.010, n=20, floor_pi=-0.02, cap_pi=0.08):
    """AR(1) annual CPI path, length n, mean-reverting to mu."""
    pis = np.empty(n)
    prev = mu
    for t in range(n):
        eps = rng.normal(0.0, sigma)
        pi = mu + rho * (prev - mu) + eps
        pi = min(cap_pi, max(floor_pi, pi))
        pis[t] = pi
        prev = pi
    return pis


def sim_path(rets, bond, sy, *, runfrac, top_wr, floor_realM, pis):
    """Round-G sim with a given inflation PATH pis (per-year CPI). Returns (labor, ruin)."""
    run0 = runfrac * 40 * M
    res0 = 40 * M - run0
    thr = min(14 * M, run0)
    run, res = run0, res0
    s = rets["sc2.6"]
    n = min(20, DATA_END - sy + 1)
    fired = False
    labor = 0
    cum = 1.0
    for k in range(n):
        yr = sy + k
        cum *= (1.0 + pis[k])
        r = float(s.loc[yr])
        if (not fired) and run < thr and res > 1e-6 and r >= 0:
            run += res; res = 0.0; fired = True
        total = run + res
        floor_nom = floor_realM * M * cum
        full_nom = FULL_REAL * cum
        spend = floor_nom
        if total > 1e-9 and full_nom / total <= top_wr:
            spend = full_nom
        if total + 1e-6 < spend:
            labor += 1
            spend = total
        if run >= spend:
            run -= spend
        else:
            res -= (spend - run); run = 0.0
        run *= (1.0 + r)
        res *= (1.0 + float(bond[yr]))
    return labor, int((run + res) <= 1e-6)


def max_floor_for_path(rets, bond, pis, *, runfrac, top_wr, lo=3.0, hi=8.0, tol=0.02):
    """Largest REAL floor (M) s.t. ALL 46 starts have labor=0 AND ruin=0 on this path."""
    def feasible(fM):
        for sy in STARTS:
            lab, ru = sim_path(rets, bond, sy, runfrac=runfrac, top_wr=top_wr,
                               floor_realM=fM, pis=pis)
            if lab > 0 or ru > 0:
                return False
        return True
    if not feasible(lo):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    # seed must be passed in (no argless rng); fixed for reproducibility
    rng = np.random.default_rng(20260629)
    N = 400          # MC paths per mean scenario
    runfrac, top_wr = 0.45, 0.16   # the v4 frontier-winning geometry
    print("=" * 95)
    print("v5 MONTE CARLO: stochastic AR(1) inflation -> distribution of guaranteed REAL floor")
    print("  (N=%d paths/scenario, AR(1) rho=0.5 sigma=1.0%%, sc2.6 run0.45 top16%%, bond reserve)" % N)
    print("=" * 95)
    rows = []
    for mu in [0.0, 0.02, 0.03]:
        floors = []
        for _ in range(N):
            pis = make_path(rng, mu)
            f = max_floor_for_path(rets, bond, pis, runfrac=runfrac, top_wr=top_wr)
            if f is not None:
                floors.append(f)
        arr = np.array(floors)
        # also realised average inflation across paths (sanity)
        rec = dict(mu=mu, n_paths=len(arr),
                   floor_median=round(float(np.median(arr)), 2),
                   floor_mean=round(float(arr.mean()), 2),
                   floor_p25=round(float(np.percentile(arr, 25)), 2),
                   floor_p05=round(float(np.percentile(arr, 5)), 2),
                   floor_p01=round(float(np.percentile(arr, 1)), 2),
                   floor_worst=round(float(arr.min()), 2),
                   floor_best=round(float(arr.max()), 2))
        rows.append(rec)
        print("\n  mu=%.1f%%:  median floor %.2fM | mean %.2f | p25 %.2f | p5 %.2f | p1 %.2f | worst %.2f | best %.2f  (n=%d)"
              % (mu * 100, rec["floor_median"], rec["floor_mean"], rec["floor_p25"],
                 rec["floor_p05"], rec["floor_p01"], rec["floor_worst"], rec["floor_best"], rec["n_paths"]))
    pd.DataFrame(rows).to_csv(os.path.join(_REPO, "audit_results",
                              "labor_zero_v5_mc_inflation_20260629.csv"),
                             index=False, encoding="utf-8-sig")
    print("\n  read: v4 deterministic floor (2%) was 5.83M. With STOCHASTIC inflation, the floor")
    print("  scatters; p5/p1 show the conservative guarantee under bad inflation draws.")
    print("\nDone (v5 MC).")


if __name__ == "__main__":
    main()
