"""
src/audit/labor_zero_v6_lookahead_20260702.py
=============================================
Task 3 of the v6 critical-verification campaign (plan: docs/superpowers/plans/
2026-07-02-labor-zero-v6-critical-verification.md): LOOK-AHEAD audit of the
refill decision.

Suspicion D1: v6 `sim_one` decides the refill AT THE START of year k using that
year's FULL-YEAR return r_run[k] (hold_if_crash skips refill when r_run[k] < 0),
and the injected money then earns the full year-k return. In reality the annual
return is only observable at year-end, so this convention uses one year of
future information. The claimed benefits of "early injection" (thr up, C1) and
the V6-A/V6-C asymmetry (C3) could be artifacts of this convention.

Three information conventions, identical in every other respect:
  (a) lookahead : the original convention (must reproduce v6.sim_one EXACTLY
                  path-by-path -- self-test).
  (b) lagged    : hold_if_crash consults the PREVIOUS year's realized run return
                  r_run[k-1] (k=0 consults the last pre-retirement year, treated
                  as non-negative -> refill allowed). Injection still happens at
                  the start of the year (implementable: you know last year's
                  return on Jan 1). Injected money earns year-k run return.
  (c) yearend   : spend/pay/grow first; at YEAR-END, after observing r_run[k],
                  move the reserve if (run < thr) and (r_run[k] >= 0). Injected
                  money earned the BOND return during year k and starts earning
                  the run return from year k+1. thr is compared on the
                  post-growth balance. Fully implementable.

For each convention x config we run the SAME bootstrap paths (same seed, same
rng consumption as v6.mc_prob) so differences are paired at the path level
(McNemar via stats helpers).

ASCII-only prints. CSV -> audit_results/labor_zero_v6_lookahead_20260702.csv.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "v6", os.path.join(_THIS, "labor_zero_v6_return_mc_20260629.py"))
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)

_spec2 = importlib.util.spec_from_file_location(
    "lzstats", os.path.join(_THIS, "labor_zero_v6_stats_helpers_20260702.py"))
lzstats = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(lzstats)

M = 1_000_000.0
SPEND = 7.2 * M
BLOCK = 5
NPATHS = 2000
SEEDS = [20260629 + i for i in range(5)]
INFLATION = 0.02


# --------------------------------------------------------------------- sim core
def sim_conv(run_path, res_path, *, run0, reserve0, thr, convention,
             spend=SPEND, hold_if_crash=True, floor=0.0, top_wr=None):
    """One retirement under a given information convention.
    Single all-in tranche (threshold=thr, frac=1.0), matching the v6 headline
    configs. Returns dict(labor_years, ruin, terminal, cut_years)."""
    run, res = float(run0), float(reserve0)
    n = len(run_path)
    fired = False
    labor = 0
    cut = 0
    for k in range(n):
        r_run = run_path[k]
        # --- start-of-year refill (conventions a, b) ---
        if convention in ("lookahead", "lagged"):
            if convention == "lookahead":
                signal = r_run                       # this year's future return
            else:                                    # lagged
                signal = run_path[k - 1] if k > 0 else 0.0
            if not (hold_if_crash and signal < 0):
                if (not fired) and run < thr and res > 1e-6:
                    run += res
                    res = 0.0
                    fired = True
        # --- spend decision ---
        total = run + res
        if floor > 0.0:
            want = floor
            if top_wr is not None and total > 1e-9 and (spend / total) <= top_wr:
                want = spend
            spend_k = want
        else:
            spend_k = spend
        if total + 1e-6 < spend_k:
            labor += 1
            spend_k = total
        if floor > 0.0 and spend_k < spend - 1e-6:
            cut += 1
        # --- pay (run first, then reserve) ---
        if run >= spend_k:
            run -= spend_k
        else:
            res -= (spend_k - run)
            run = 0.0
        if res < 0:
            res = 0.0
        # --- growth ---
        run *= (1.0 + r_run)
        res *= (1.0 + res_path[k])
        # --- year-end refill (convention c): return is now observed ---
        if convention == "yearend":
            if not (hold_if_crash and r_run < 0):
                if (not fired) and run < thr and res > 1e-6:
                    run += res
                    res = 0.0
                    fired = True
    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6),
                terminal=run + res, cut_years=cut)


# ------------------------------------------------------------------- MC driver
def gen_paths(run_h, res_h, seed, n_paths=NPATHS, block=BLOCK, horizon=20):
    """Same rng consumption as v6.mc_prob: fresh rng per seed, paths in order."""
    rng = np.random.default_rng(seed)
    return [v6.make_block_path(rng, run_h, res_h, n=horizon, block=block)
            for _ in range(n_paths)]


def run_config(paths, convention, cfg):
    labors = np.empty(len(paths), int)
    ruins = np.empty(len(paths), int)
    terms = np.empty(len(paths))
    cuts = np.empty(len(paths), int)
    for i, (rp, sp) in enumerate(paths):
        r = sim_conv(rp, sp, convention=convention, **cfg)
        labors[i] = r["labor_years"]
        ruins[i] = r["ruin"]
        terms[i] = r["terminal"]
        cuts[i] = r["cut_years"]
    return labors, ruins, terms, cuts


CONFIGS = {
    # nominal-fixed (V6-C world)
    "N_20_20_thr20":   dict(run0=20 * M, reserve0=20 * M, thr=20 * M),
    "N_20_20_thr26":   dict(run0=20 * M, reserve0=20 * M, thr=26 * M),
    "N_10_30_thr14":   dict(run0=10 * M, reserve0=30 * M, thr=14 * M),
    "N_10_30_thr26":   dict(run0=10 * M, reserve0=30 * M, thr=26 * M),
    "N_20_20_thr20_noHOLD": dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                                 hold_if_crash=False),
    # floor variant (V6-A world)
    "F_20_20_thr20":   dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                            floor=3.6 * M, top_wr=0.16),
    "F_20_20_thr26":   dict(run0=20 * M, reserve0=20 * M, thr=26 * M,
                            floor=3.6 * M, top_wr=0.16),
    "F_10_30_thr14":   dict(run0=10 * M, reserve0=30 * M, thr=14 * M,
                            floor=3.6 * M, top_wr=0.16),
}

# paired comparisons evaluated WITHIN each convention (claim -> (A, B); the claim
# is "P(A) > P(B)" for C1 rows and "P(A) >= P(B)" for the C3 row)
PAIRS = [
    ("C1 thr-up      N_20_20_thr26 vs N_20_20_thr20", "N_20_20_thr26", "N_20_20_thr20"),
    ("C1 res-heavy   N_10_30_thr14 vs N_20_20_thr20", "N_10_30_thr14", "N_20_20_thr20"),
    ("hold-if-crash  N_20_20_thr20 vs noHOLD",        "N_20_20_thr20", "N_20_20_thr20_noHOLD"),
    ("C3 V6-A cur    F_20_20_thr20 vs F_20_20_thr26", "F_20_20_thr20", "F_20_20_thr26"),
    ("C3 V6-A cur    F_20_20_thr20 vs F_10_30_thr14", "F_20_20_thr20", "F_10_30_thr14"),
]


def _self_test(run_h, res_h):
    """Convention (a) must equal v6.sim_one path-by-path for every config."""
    paths = gen_paths(run_h, res_h, SEEDS[0], n_paths=300)
    for name, cfg in CONFIGS.items():
        hold = cfg.get("hold_if_crash", True)
        v6cfg = dict(run0=cfg["run0"], reserve0=cfg["reserve0"],
                     tranches=[(cfg["thr"], 1.0)], hold_if_crash=hold,
                     floor=cfg.get("floor", 0.0), top_wr=cfg.get("top_wr"))
        for rp, sp in paths:
            a = sim_conv(rp, sp, convention="lookahead", **cfg)
            b = v6.sim_one(rp, sp, **v6cfg)
            if (a["labor_years"] != b["labor_years"] or a["ruin"] != b["ruin"]
                    or abs(a["terminal"] - b["terminal"]) > 1e-3
                    or a["cut_years"] != b["cut_years"]):
                print("SELF-TEST FAIL at config %s" % name)
                return False
    print("SELF-TEST PASS: convention (a) == v6.sim_one path-by-path, all %d configs x 300 paths"
          % len(CONFIGS))
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("Task 3: LOOK-AHEAD AUDIT of the refill decision (conventions a/b/c)")
    print("  block=%d N=%d seeds=%s" % (BLOCK, NPATHS, SEEDS))
    print("=" * 100)
    run_h, res_h = v6.get_paired_history()
    if not _self_test(run_h, res_h):
        print("HALT: self-test failed.")
        return

    conventions = ("lookahead", "lagged", "yearend")
    deflate = (1.0 + INFLATION) ** 20
    rows = []
    # labors[seed][conv][config] for paired tests
    store = {s: {c: {} for c in conventions} for s in SEEDS}
    for seed in SEEDS:
        paths = gen_paths(run_h, res_h, seed)
        for conv in conventions:
            for name, cfg in CONFIGS.items():
                labors, ruins, terms, cuts = run_config(paths, conv, cfg)
                store[seed][conv][name] = labors
                rows.append(dict(seed=seed, convention=conv, config=name,
                                 p_labor0=float((labors == 0).mean()),
                                 p_ruin0=float((ruins == 0).mean()),
                                 labor_mean=float(labors.mean()),
                                 cut_mean=float(cuts.mean()),
                                 term_real_med_M=round(float(np.median(terms)) / deflate / M, 1)))
    df = pd.DataFrame(rows)

    print("\n[1] P(labor0) by convention x config  (mean over %d seeds [min-max])" % len(SEEDS))
    print("    %-22s" % "config" + "".join("%-26s" % c for c in conventions))
    for name in CONFIGS:
        line = "    %-22s" % name
        for conv in conventions:
            sub = df[(df.convention == conv) & (df.config == name)].p_labor0
            line += "%-26s" % ("%.4f [%.4f-%.4f]" % (sub.mean(), sub.min(), sub.max()))
        print(line)
    print("\n    cut_mean (floor configs only):")
    for name in [n for n in CONFIGS if n.startswith("F_")]:
        line = "    %-22s" % name
        for conv in conventions:
            sub = df[(df.convention == conv) & (df.config == name)].cut_mean
            line += "%-26s" % ("%.3f [%.3f-%.3f]" % (sub.mean(), sub.min(), sub.max()))
        print(line)

    print("\n[2] Paired tests WITHIN each convention (pooled over seeds; A_only = paths where"
          "\n    first config reaches labor0 and second does not)")
    for label, a, b in PAIRS:
        for conv in conventions:
            la = np.concatenate([store[s][conv][a] for s in SEEDS])
            lb = np.concatenate([store[s][conv][b] for s in SEEDS])
            r = lzstats.paired_diff(la, lb)
            print("    %-48s %-9s dP=%+.4f  A_only=%4d B_only=%4d  p=%.2e"
                  % (label, conv, r["p_a"] - r["p_b"], r["a_only"], r["b_only"],
                     r["mcnemar_p"]))

    print("\n[3] Artifact size: same config, lookahead vs lagged / yearend (pooled)")
    for name in CONFIGS:
        la = np.concatenate([store[s]["lookahead"][name] for s in SEEDS])
        for conv in ("lagged", "yearend"):
            lc = np.concatenate([store[s][conv][name] for s in SEEDS])
            r = lzstats.paired_diff(la, lc)
            print("    %-22s lookahead vs %-9s dP=%+.4f  A_only=%4d B_only=%4d  p=%.2e"
                  % (name, conv, r["p_a"] - r["p_b"], r["a_only"], r["b_only"],
                     r["mcnemar_p"]))

    out = os.path.join(_REPO, "audit_results", "labor_zero_v6_lookahead_20260702.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("\nDone (Task 3 look-ahead audit).")


if __name__ == "__main__":
    main()
