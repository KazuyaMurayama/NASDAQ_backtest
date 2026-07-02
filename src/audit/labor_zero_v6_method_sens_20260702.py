"""
src/audit/labor_zero_v6_method_sens_20260702.py
================================================
labor-zero v6 critical verification -- Task 2 (T1): method-sensitivity attack on
claims C1-C5 of LABOR_ZERO_V6_RETURN_MC_20260629.md.

Plan: docs/superpowers/plans/2026-07-02-labor-zero-v6-critical-verification.md, Task 2.

Question: do the RANK ORDERS (not absolute P values) that C1-C5 rely on survive
when we perturb the bootstrap methodology -- block length, stationary (geometric)
bootstrap, partial-history sampling windows, no-wrap block sampling, horizon, and
choice of primary metric?  We do NOT attach pass/fail labels; we report raw numbers,
ranks, and paired-path tests, and separately narrate which claims are preserved.

Anchor configs (all hold_if_crash=True, spend=7.2M nominal fixed, N=2000, block=5,
seed=20260629, sc2.6 run sleeve / 1x bond reserve unless noted):
  NC1 : run20M/res20M, thr20M                         -> anchor P=0.8745
  NC2 : run20M/res20M, thr26M                         -> anchor P=0.8965
  NC3 : run10M/res30M, thr14M                         -> anchor P=0.9170
  FA1 : run20M/res20M, thr20M, floor=3.6M, top_wr=0.16 -> anchor P=0.9995
  FA2 : run20M/res20M, thr26M, floor=3.6M, top_wr=0.16 -> anchor P=0.9945
  SA49: start=49M, run:res=1:1 (24.5M/24.5M), thr=run0 -> anchor P=0.9570
  SA60: start=60M, run:res=1:1 (30M/30M),     thr=run0 -> anchor P=0.9810

Rank claims under test (label only, no verdict -- verdict is Task 7's job):
  O1 (C1): P(NC3) > P(NC2) > P(NC1)                      [thr/refill-mix dial]
  O2 (C3): P(FA1) >= P(FA2)                               [floor variant: thr20 >= thr26]
  O3 (C5): P(FA1 @ start=40M) > P(SA60 @ start=60M, +20M)  [floor beats a +20M asset increase]
           (SA49 reported alongside as a secondary reference point only -- the source
           report does NOT claim SA49 sits between FA1 and SA60 in a rank chain; SA49
           < SA60 is expected on its own, since more starting assets -> higher P at a
           fixed nominal spend with no floor)
  O4     : P(FA1) >> P(NC1)                               [floor is the main lever]

Sensitivity axes (schemes):
  S-BLOCK  : block in {3,4,5,6,8,10}, v6.make_block_path (wrap-around, i.i.d. block starts)
  S-STAT   : stationary bootstrap (Politis-Romano, geometric block length mean=5, wrap)
  S-HIST75 : source years restricted to 1975-2000 (26y), block=5, wrap within window
  S-HIST00 : source years restricted to 2000-2025 (26y), block=5, wrap within window
  S-NOWRAP : block=5, NO wrap-around (block must fit inside [0, H-block])
  S-HZ15/25/30 : horizon in {15,25,30}, block=5, full 51y history (deflate/discount
                 rescaled to horizon; SA/FA configs unaffected by horizon def besides N)

Pairing discipline: for a fixed scheme, ALL configs are evaluated on the SAME path
list (generated once per (scheme, seed) and reused for every config), so paired_diff
is valid (McNemar on same-path discordance). We do NOT use v6.mc_prob (it reformats
the rng draw internally per call) -- instead we build a local path-cache MC driver
that reproduces mc_prob's rng consumption exactly for the S-BLOCK self-test to match
existing anchors bit for bit.

Seeds: 20260629 + i for i in range(5) (5 seeds per scheme; P reported as seed-mean
with [min-max] range, 4-decimal).

ASCII-only prints. Writes audit_results/labor_zero_v6_method_sens_20260702.csv.
No commit (caller commits).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, os.path.dirname(_THIS), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "v6", os.path.join(_THIS, "labor_zero_v6_return_mc_20260629.py"))
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)

_spec2 = importlib.util.spec_from_file_location(
    "stats_helpers", os.path.join(_THIS, "labor_zero_v6_stats_helpers_20260702.py"))
sh = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(sh)

M = 1_000_000.0
SPEND = 7.2 * M
INFLATION = 0.02
SEEDS = [20260629 + i for i in range(5)]

# ----------------------------------------------------------------- representative configs
# Each config: dict(run0, reserve0, tranches, hold_if_crash, floor, top_wr) fed straight
# to v6.sim_one. "horizon_ok": True means it is meaningful across the S-HZxx axis
# (SA/FA/NC all are -- horizon just changes N years simulated).
CONFIGS = {
    "NC1":  dict(run0=20 * M, reserve0=20 * M, tranches=[(20 * M, 1.0)],
                 hold_if_crash=True, floor=0.0, top_wr=None),
    "NC2":  dict(run0=20 * M, reserve0=20 * M, tranches=[(26 * M, 1.0)],
                 hold_if_crash=True, floor=0.0, top_wr=None),
    "NC3":  dict(run0=10 * M, reserve0=30 * M, tranches=[(14 * M, 1.0)],
                 hold_if_crash=True, floor=0.0, top_wr=None),
    "FA1":  dict(run0=20 * M, reserve0=20 * M, tranches=[(20 * M, 1.0)],
                 hold_if_crash=True, floor=3.6 * M, top_wr=0.16),
    "FA2":  dict(run0=20 * M, reserve0=20 * M, tranches=[(26 * M, 1.0)],
                 hold_if_crash=True, floor=3.6 * M, top_wr=0.16),
    "SA49": dict(run0=24.5 * M, reserve0=24.5 * M, tranches=[(24.5 * M, 1.0)],
                 hold_if_crash=True, floor=0.0, top_wr=None),
    "SA60": dict(run0=30 * M, reserve0=30 * M, tranches=[(30 * M, 1.0)],
                 hold_if_crash=True, floor=0.0, top_wr=None),
}
CONFIG_ORDER = ["NC1", "NC2", "NC3", "FA1", "FA2", "SA49", "SA60"]

ANCHORS = {"NC1": 0.8745, "NC2": 0.8965, "NC3": 0.9170,
           "FA1": 0.9995, "FA2": 0.9945, "SA49": 0.9570, "SA60": 0.9810}

# pairs to run paired_diff on, per scheme (plan-specified 4 pairs + FA1-SA60, the
# pair that actually instantiates the C5 headline claim "FA1(40M) > SA60(60M,+20M)")
PAIRS = [("NC2", "NC1"), ("NC3", "NC1"), ("FA1", "FA2"), ("FA1", "SA49"), ("FA1", "SA60")]


# ----------------------------------------------------------------------- path generators
def gen_paths_block(rng, run_h, res_h, n_paths, horizon, block):
    """v6.make_block_path, wrap-around, called n_paths times in sequence (matches
    v6.mc_prob's rng consumption exactly)."""
    return [v6.make_block_path(rng, run_h, res_h, n=horizon, block=block)
            for _ in range(n_paths)]


def make_stationary_path(rng, run_h, res_h, n, mean_block=5.0):
    """Stationary bootstrap (Politis-Romano): geometric block lengths, wrap-around."""
    H = len(run_h)
    p_new = 1.0 / mean_block
    rp = np.empty(n)
    sp = np.empty(n)
    idx = int(rng.integers(0, H))
    for k in range(n):
        rp[k] = run_h[idx]
        sp[k] = res_h[idx]
        if rng.random() < p_new:
            idx = int(rng.integers(0, H))
        else:
            idx = (idx + 1) % H
    return rp, sp


def gen_paths_stationary(rng, run_h, res_h, n_paths, horizon, mean_block=5.0):
    return [make_stationary_path(rng, run_h, res_h, n=horizon, mean_block=mean_block)
            for _ in range(n_paths)]


def make_block_path_nowrap(rng, run_h, res_h, n, block):
    """Like v6.make_block_path but blocks may not cross the array boundary
    (start in [0, H-block] only) -- isolates the wrap-around (1975<-2025 splice)
    effect from ordinary block bootstrap."""
    H = len(run_h)
    assert block <= H
    rp = np.empty(n)
    sp = np.empty(n)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, H - block + 1))
        take = min(block, n - filled)
        for j in range(take):
            idx = start + j
            rp[filled] = run_h[idx]
            sp[filled] = res_h[idx]
            filled += 1
    return rp, sp


def gen_paths_nowrap(rng, run_h, res_h, n_paths, horizon, block):
    return [make_block_path_nowrap(rng, run_h, res_h, n=horizon, block=block)
            for _ in range(n_paths)]


# ------------------------------------------------------------------------- MC over cache
def run_configs_on_paths(paths, configs=CONFIG_ORDER, horizon=20):
    """Run every config in `configs` over the SAME path list. Returns
    dict(config -> int array of labor_years, length len(paths))."""
    out = {c: np.empty(len(paths), int) for c in configs}
    for pi, (rp, sp) in enumerate(paths):
        for c in configs:
            cfg = CONFIGS[c]
            r = v6.sim_one(rp, sp, run0=cfg["run0"], reserve0=cfg["reserve0"],
                            tranches=cfg["tranches"], hold_if_crash=cfg["hold_if_crash"],
                            floor=cfg["floor"], top_wr=cfg["top_wr"])
            out[c][pi] = r["labor_years"]
    return out


def p_labor0(labors):
    return float((np.asarray(labors) == 0).mean())


def seed_mean_range(per_seed_p):
    arr = np.asarray(per_seed_p)
    return float(arr.mean()), float(arr.min()), float(arr.max())


# --------------------------------------------------------------------------- self-test
def self_test(run_h, res_h):
    """block=5 scheme must reproduce the anchor values to 4 decimals (single seed
    20260629) -- this both validates the local path-cache driver against v6.mc_prob
    and gives S-BLOCK block=5 a free correctness check."""
    print("SELF-TEST: block=5 seed=20260629 must reproduce anchors to 4 decimals")
    rng = np.random.default_rng(20260629)
    paths = gen_paths_block(rng, run_h, res_h, n_paths=2000, horizon=20, block=5)
    labors = run_configs_on_paths(paths, CONFIG_ORDER, horizon=20)
    ok = True
    for c in CONFIG_ORDER:
        p = p_labor0(labors[c])
        exp = ANCHORS[c]
        match = abs(p - exp) < 5e-5
        ok = ok and match
        print("  %-5s got=%.4f  anchor=%.4f  %s" % (c, p, exp, "OK" if match else "MISMATCH"))
    print("  -> SELF-TEST %s" % ("PASS" if ok else "FAIL -- HALT"))
    return ok, labors


# --------------------------------------------------------------------------- scheme runner
def run_scheme(name, run_h, res_h, path_fn, horizon=20, configs=CONFIG_ORDER):
    """path_fn(rng) -> list[(rp,sp)] of length 2000. Runs SEEDS, returns:
      results: dict(config -> (mean_p, lo, hi, per_seed_p_list))
      pair_rows: list of dict for paired_diff on PAIRS, using seed[0] path/labor set
                 AND separately averaged over all 5 seeds (a_only/b_only summed).
      labor_cache: dict(config -> labor array) for seed[0] (for downstream reuse / trace)
    """
    per_seed_p = {c: [] for c in configs}
    pair_accum = {p: dict(a_only=0, b_only=0, n=0) for p in PAIRS}
    seed0_labors = None
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        paths = path_fn(rng)
        labors = run_configs_on_paths(paths, configs, horizon=horizon)
        for c in configs:
            per_seed_p[c].append(p_labor0(labors[c]))
        for (a, b) in PAIRS:
            if a in labors and b in labors:
                d = sh.paired_diff(labors[a], labors[b])
                pair_accum[(a, b)]["a_only"] += d["a_only"]
                pair_accum[(a, b)]["b_only"] += d["b_only"]
                pair_accum[(a, b)]["n"] += d["n"]
        if si == 0:
            seed0_labors = labors
    results = {}
    for c in configs:
        mean_p, lo, hi = seed_mean_range(per_seed_p[c])
        results[c] = dict(mean_p=mean_p, lo=lo, hi=hi, per_seed=per_seed_p[c])
    pair_rows = []
    for (a, b) in PAIRS:
        acc = pair_accum[(a, b)]
        m = acc["a_only"] + acc["b_only"]
        if m == 0:
            mcp = 1.0
        else:
            from scipy.stats import binomtest
            mcp = float(binomtest(min(acc["a_only"], acc["b_only"]), m, 0.5).pvalue)
        pair_rows.append(dict(scheme=name, a=a, b=b, a_only=acc["a_only"],
                               b_only=acc["b_only"], n_pooled=acc["n"], mcnemar_p=mcp))
    return results, pair_rows, seed0_labors


def print_scheme_table(name, results, configs=CONFIG_ORDER):
    print("  [%s]" % name)
    print("    %-6s %10s %20s" % ("cfg", "P(mean)", "seed range"))
    for c in configs:
        r = results[c]
        print("    %-6s %10.4f  [%.4f-%.4f]" % (c, r["mean_p"], r["lo"], r["hi"]))


def print_pair_table(pair_rows):
    print("    %-10s %10s %8s %8s %10s %12s"
          % ("pair(a-b)", "n_pooled", "a_only", "b_only", "diff_sign", "mcnemar_p"))
    for row in pair_rows:
        diff = row["a_only"] - row["b_only"]
        sign = "a>b" if diff > 0 else ("a<b" if diff < 0 else "tie")
        print("    %-10s %10d %8d %8d %10s %12.4f"
              % ("%s-%s" % (row["a"], row["b"]), row["n_pooled"], row["a_only"],
                 row["b_only"], sign, row["mcnemar_p"]))


def rank_claims(results):
    """Return dict of claim -> (holds: bool, detail str) for O1-O4 using mean_p."""
    p = {c: results[c]["mean_p"] for c in results if c in results}
    out = {}
    if all(k in p for k in ("NC1", "NC2", "NC3")):
        o1 = (p["NC3"] > p["NC2"] > p["NC1"])
        out["O1"] = (o1, "NC3=%.4f NC2=%.4f NC1=%.4f" % (p["NC3"], p["NC2"], p["NC1"]))
    if all(k in p for k in ("FA1", "FA2")):
        o2 = (p["FA1"] >= p["FA2"])
        out["O2"] = (o2, "FA1=%.4f FA2=%.4f" % (p["FA1"], p["FA2"]))
    if all(k in p for k in ("FA1", "SA49", "SA60")):
        o3 = (p["FA1"] > p["SA60"])
        sa_monotone = (p["SA49"] < p["SA60"])   # sanity check, not part of the claim
        out["O3"] = (o3, "FA1=%.4f SA60(+20M)=%.4f SA49(ref)=%.4f  SA49<SA60(monotone-in-assets)=%s"
                     % (p["FA1"], p["SA60"], p["SA49"], sa_monotone))
    if all(k in p for k in ("FA1", "NC1")):
        o4 = (p["FA1"] - p["NC1"]) > 0.05   # ">>": treat >5pp as "much greater"
        out["O4"] = (o4, "FA1=%.4f NC1=%.4f diff=%.4f" % (p["FA1"], p["NC1"], p["FA1"] - p["NC1"]))
    return out


def print_rank_claims(name, claims):
    labels = {"O1": "O1(C1) NC3>NC2>NC1", "O2": "O2(C3) FA1>=FA2",
              "O3": "O3(C5) FA1(40M)>SA60(60M)", "O4": "O4 FA1>>NC1"}
    for k in ("O1", "O2", "O3", "O4"):
        if k in claims:
            holds, detail = claims[k]
            print("    %-22s %-9s %s" % (labels[k], "PRESERVED" if holds else "BROKEN", detail))


# ------------------------------------------------------------------------- R-STAT memo
def crash_cluster_lengths(run_h):
    """Identify contiguous negative-return-year clusters in the historical run
    sleeve (HIST_YEARS order) -- for the R-STAT-1/2/3 memo (Step 2-5)."""
    neg = run_h < 0
    clusters = []
    cur = 0
    for x in neg:
        if x:
            cur += 1
        else:
            if cur > 0:
                clusters.append(cur)
            cur = 0
    if cur > 0:
        clusters.append(cur)
    return clusters


# --------------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("labor-zero v6 critical verification -- Task 2: method sensitivity (block / stationary")
    print("bootstrap / partial-history / no-wrap / horizon / metric definition)")
    print("=" * 100)

    run_h, res_h = v6.get_paired_history(strat="sc2.6")
    print("history years: %d  (HIST_YEARS[0]=%d .. [-1]=%d)"
          % (len(run_h), v6.HIST_YEARS[0], v6.HIST_YEARS[-1]))

    ok, seed0_block5_labors = self_test(run_h, res_h)
    if not ok:
        print("\nHALT: self-test failed -- fix before trusting downstream schemes.")
        return

    all_rows = []          # for CSV: scheme, config, mean_p, lo, hi
    all_pair_rows = []      # for CSV: scheme, a, b, a_only, b_only, n_pooled, mcnemar_p
    all_claims = {}         # scheme -> claims dict

    # ---------------------------------------------------------------- S-BLOCK
    print("\n" + "-" * 100)
    print("[S-BLOCK] block in {3,4,5,6,8,10}, wrap-around block bootstrap, horizon=20")
    print("-" * 100)
    for block in (3, 4, 5, 6, 8, 10):
        name = "S-BLOCK_b%d" % block
        results, pair_rows, _ = run_scheme(
            name, run_h, res_h,
            path_fn=lambda rng, b=block: gen_paths_block(rng, run_h, res_h, 2000, 20, b))
        print_scheme_table(name, results)
        print_pair_table(pair_rows)
        claims = rank_claims(results)
        print_rank_claims(name, claims)
        all_claims[name] = claims
        for c, r in results.items():
            all_rows.append(dict(scheme=name, config=c, mean_p=r["mean_p"],
                                  lo=r["lo"], hi=r["hi"]))
        all_pair_rows.extend(pair_rows)

    # ---------------------------------------------------------------- S-STAT
    print("\n" + "-" * 100)
    print("[S-STAT] stationary bootstrap (Politis-Romano, geometric block mean=5), horizon=20")
    print("-" * 100)
    name = "S-STAT_mb5"
    results, pair_rows, _ = run_scheme(
        name, run_h, res_h,
        path_fn=lambda rng: gen_paths_stationary(rng, run_h, res_h, 2000, 20, 5.0))
    print_scheme_table(name, results)
    print_pair_table(pair_rows)
    claims = rank_claims(results)
    print_rank_claims(name, claims)
    all_claims[name] = claims
    for c, r in results.items():
        all_rows.append(dict(scheme=name, config=c, mean_p=r["mean_p"], lo=r["lo"], hi=r["hi"]))
    all_pair_rows.extend(pair_rows)

    # ---------------------------------------------------------------- S-HIST75 / S-HIST00
    print("\n" + "-" * 100)
    print("[S-HIST75 / S-HIST00] source-year window restriction, block=5, horizon=20")
    print("-" * 100)
    run_75 = run_h[:26]; res_75 = res_h[:26]     # 1975-2000 inclusive (26 yrs)
    run_00 = run_h[25:]; res_00 = res_h[25:]     # 2000-2025 inclusive (26 yrs)
    print("  S-HIST75 window years: %d-%d (n=%d)"
          % (v6.HIST_YEARS[0], v6.HIST_YEARS[25], len(run_75)))
    print("  S-HIST00 window years: %d-%d (n=%d)"
          % (v6.HIST_YEARS[25], v6.HIST_YEARS[-1], len(run_00)))
    for name, rh, sh_ in (("S-HIST75", run_75, res_75), ("S-HIST00", run_00, res_00)):
        results, pair_rows, _ = run_scheme(
            name, run_h, res_h,
            path_fn=lambda rng, rh=rh, sh_=sh_: gen_paths_block(rng, rh, sh_, 2000, 20, 5))
        print_scheme_table(name, results)
        print_pair_table(pair_rows)
        claims = rank_claims(results)
        print_rank_claims(name, claims)
        all_claims[name] = claims
        for c, r in results.items():
            all_rows.append(dict(scheme=name, config=c, mean_p=r["mean_p"], lo=r["lo"], hi=r["hi"]))
        all_pair_rows.extend(pair_rows)

    # ---------------------------------------------------------------- S-NOWRAP
    print("\n" + "-" * 100)
    print("[S-NOWRAP] block=5, NO wrap-around (isolates the 2025->1975 splice effect)")
    print("-" * 100)
    name = "S-NOWRAP_b5"
    results, pair_rows, _ = run_scheme(
        name, run_h, res_h,
        path_fn=lambda rng: gen_paths_nowrap(rng, run_h, res_h, 2000, 20, 5))
    print_scheme_table(name, results)
    print_pair_table(pair_rows)
    claims = rank_claims(results)
    print_rank_claims(name, claims)
    all_claims[name] = claims
    for c, r in results.items():
        all_rows.append(dict(scheme=name, config=c, mean_p=r["mean_p"], lo=r["lo"], hi=r["hi"]))
    all_pair_rows.extend(pair_rows)

    # ---------------------------------------------------------------- S-HZxx
    print("\n" + "-" * 100)
    print("[S-HZ15/25/30] horizon in {15,25,30}, block=5, full 51y history")
    print("-" * 100)
    for hz in (15, 25, 30):
        name = "S-HZ%d" % hz
        results, pair_rows, _ = run_scheme(
            name, run_h, res_h,
            path_fn=lambda rng, hz=hz: gen_paths_block(rng, run_h, res_h, 2000, hz, 5),
            horizon=hz)
        print_scheme_table(name, results)
        print_pair_table(pair_rows)
        claims = rank_claims(results)
        print_rank_claims(name, claims)
        all_claims[name] = claims
        for c, r in results.items():
            all_rows.append(dict(scheme=name, config=c, mean_p=r["mean_p"], lo=r["lo"], hi=r["hi"]))
        all_pair_rows.extend(pair_rows)

    # ---------------------------------------------------------------- metric sensitivity
    print("\n" + "-" * 100)
    print("[METRIC] cut_mean as primary metric for the V6-A FA1 vs FA2 comparison (C3)")
    print("(does the FA1 vs FA2 ranking reverse if we rank by cut_mean -- fewer floor-cut")
    print(" years is 'better' -- instead of P(labor=0)? lower cut_mean = better)")
    print("-" * 100)
    rng = np.random.default_rng(SEEDS[0])
    paths_b5 = gen_paths_block(rng, run_h, res_h, 2000, 20, 5)
    cut_cache = {}
    for c in ("FA1", "FA2"):
        cfg = CONFIGS[c]
        cuts = np.empty(len(paths_b5), int)
        labs = np.empty(len(paths_b5), int)
        for pi, (rp, sp) in enumerate(paths_b5):
            r = v6.sim_one(rp, sp, run0=cfg["run0"], reserve0=cfg["reserve0"],
                            tranches=cfg["tranches"], hold_if_crash=cfg["hold_if_crash"],
                            floor=cfg["floor"], top_wr=cfg["top_wr"])
            cuts[pi] = r["cut_years"]
            labs[pi] = r["labor_years"]
        cut_cache[c] = dict(cut_mean=float(cuts.mean()), p_labor0=p_labor0(labs))
    print("    FA1: P(labor0)=%.4f  cut_mean=%.4f" % (cut_cache["FA1"]["p_labor0"], cut_cache["FA1"]["cut_mean"]))
    print("    FA2: P(labor0)=%.4f  cut_mean=%.4f" % (cut_cache["FA2"]["p_labor0"], cut_cache["FA2"]["cut_mean"]))
    p_rank = "FA1>=FA2" if cut_cache["FA1"]["p_labor0"] >= cut_cache["FA2"]["p_labor0"] else "FA2>FA1"
    cut_rank = "FA1<=FA2 (fewer cuts=better)" if cut_cache["FA1"]["cut_mean"] <= cut_cache["FA2"]["cut_mean"] \
        else "FA2<FA1 (fewer cuts=better)"
    p_better = "FA1" if cut_cache["FA1"]["p_labor0"] >= cut_cache["FA2"]["p_labor0"] else "FA2"
    cut_better = "FA1" if cut_cache["FA1"]["cut_mean"] <= cut_cache["FA2"]["cut_mean"] else "FA2"
    reversal = (p_better != cut_better)
    print("    P(labor0)-primary winner: %s   |  cut_mean-primary winner: %s   |  reversal=%s"
          % (p_better, cut_better, reversal))

    # ---------------------------------------------------------------- R-STAT-1/2/3 memo
    print("\n" + "-" * 100)
    print("[R-STAT MEMO] crash-cluster length vs block length (Step 2-5)")
    print("-" * 100)
    clusters = crash_cluster_lengths(run_h)
    print("  negative-return-year clusters in sc2.6 run sleeve, 1975-2025 (contiguous run length):")
    print("  clusters=%s" % clusters)
    print("  max cluster length=%d  (block=5 memo claims 2000-02 crash cluster ~3y: %s)"
          % (max(clusters) if clusters else 0,
             "consistent" if clusters and max(clusters) >= 3 else "check"))
    print("  R-STAT-1 says: block << crisis length invalidates path-dependent-extremum")
    print("  bootstraps. Here block=5 is >= the empirical max crash-cluster length (%d),"
          % (max(clusters) if clusters else 0))
    print("  so P(labor=0) block bootstrap is NOT the R-STAT-1 violation case (that case is")
    print("  path-dependent MAXIMUM DRAWDOWN with block << crisis length; P(labor=0) here is a")
    print("  terminal/cumulative survival statistic over the WHOLE horizon, not an extremum")
    print("  statistic localized to one crisis window -- the mechanism differs from R-STAT-1's")
    print("  target case). We test this empirically below: does P move monotonically/smoothly")
    print("  as block shrinks toward and below the crash-cluster length (block=3), or does it")
    print("  jump discontinuously (which would indicate crash-cluster-splitting artifacts)?")
    print("  (see S-BLOCK table above for block=3..10 trend per config)")

    # ---------------------------------------------------------------- CSV + summary
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(_REPO, "audit_results", "labor_zero_v6_method_sens_20260702.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n  wrote %s (%d rows)" % (out_csv, len(df)))

    pair_df = pd.DataFrame(all_pair_rows)
    out_csv2 = os.path.join(_REPO, "audit_results", "labor_zero_v6_method_sens_pairs_20260702.csv")
    pair_df.to_csv(out_csv2, index=False, encoding="utf-8-sig")
    print("  wrote %s (%d rows)" % (out_csv2, len(pair_df)))

    # ---------------------------------------------------------------- claim preservation summary
    print("\n" + "=" * 100)
    print("SUMMARY: claim preservation across all schemes")
    print("=" * 100)
    scheme_names = list(all_claims.keys())
    for claim_key, label in (("O1", "O1(C1) NC3>NC2>NC1"), ("O2", "O2(C3) FA1>=FA2"),
                              ("O3", "O3(C5) FA1(40M)>SA60(60M)"), ("O4", "O4 FA1>>NC1")):
        preserved = [s for s in scheme_names if claim_key in all_claims[s] and all_claims[s][claim_key][0]]
        broken = [s for s in scheme_names if claim_key in all_claims[s] and not all_claims[s][claim_key][0]]
        print("  %-22s preserved in %d/%d schemes" % (label, len(preserved), len(preserved) + len(broken)))
        if broken:
            print("      BROKEN in: %s" % ", ".join(broken))

    print("\nDone (Task 2 method sensitivity).")


if __name__ == "__main__":
    main()
