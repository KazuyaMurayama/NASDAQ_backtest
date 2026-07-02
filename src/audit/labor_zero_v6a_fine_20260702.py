"""
src/audit/labor_zero_v6a_fine_20260702.py
==========================================
Task 5 of the v6 critical-verification campaign (plan: docs/superpowers/plans/
2026-07-02-labor-zero-v6-critical-verification.md): V6-A (floor variant) FINE
GRID + top_wr sweep.

Per Task 3's confirmed finding (docs/superpowers/plans/... Task 3 result), the
refill-decision information convention (lookahead vs lagged vs yearend) MOVES
the thr ranking within V6-A: lookahead favors thr20, lagged favors thr26
(slightly), yearend ties. This module therefore runs every grid under BOTH
"lookahead" (the v6 headline convention) and "lagged" (the most defensible
non-look-ahead convention) so the fine-grid conclusions are not convention-
specific artifacts. ("yearend" is intentionally left to Task 3's own scope;
re-including it here would triple runtime for a convention Task 3 already
showed ties/tracks lagged closely.)

Reuses:
  - labor_zero_v6_lookahead_20260702.py -> sim_conv, gen_paths, v6 module
    (imported via importlib so this file does NOT re-derive the simulation
    core -- only the grid/sweep/report logic is new).
  - labor_zero_v6_stats_helpers_20260702.py -> wilson_ci, paired_diff

Self-test (must 4-digit match before any grid runs):
  lookahead convention, seed=20260629 alone, run20/res20, floor=3.6M, top_wr=0.16:
    thr20M -> P=0.9995
    thr26M -> P=0.9945

Sections:
  [1] Fine grid: thr in {12..30}M step2 x alloc in {10:30,14:26,20:20,24:16,30:10}
      x floor=3.6M x top_wr=0.16 x 20 seeds x 2 conventions.
      P mean [min-max] per cell + pooled Wilson CI (n=40,000 per cell, i.e.
      20 seeds x 2000 paths). Peak unimodality / noise check on the 20:20 row.
  [2] top_wr sweep: top_wr in {0.12,0.14,0.16,0.18,0.20} x floor in
      {3.0,3.6,4.43}M x thr in {20,26}M (alloc 20:20 fixed) x 20 seeds x 2 conv.
      Reports P, cut_mean, real terminal median (both /1.02^20 already, via v6
      convention) -- what top_wr actually moves.
  [3] V6-A' frontier: thr in {22,24,26,28,30}M x top_wr in {0.14,0.16,0.18}
      (floor=3.6M, alloc=20:20), 5 seeds. P + real terminal median/p5 -- is the
      "P-0.5pp <-> terminal doubles" trade specific to thr26M or a smooth
      neighborhood effect?
  [4] Paired tests (pooled 20 seeds): thr20 vs thr26; thr20 vs each neighbor;
      20:20 vs 10:30 @ thr20 -- for each convention.

ASCII-only prints. CSV -> audit_results/labor_zero_v6a_fine_20260702.csv
(one row per [section, convention, config, seed] cell; sections 1/2 only,
section 3 is small enough to fold into stdout + its own csv suffix).
No commit (per plan instructions).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "la", os.path.join(_THIS, "labor_zero_v6_lookahead_20260702.py"))
la = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(la)   # brings in la.v6, la.sim_conv, la.gen_paths

_spec2 = importlib.util.spec_from_file_location(
    "lzstats", os.path.join(_THIS, "labor_zero_v6_stats_helpers_20260702.py"))
lzstats = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(lzstats)

M = 1_000_000.0
BLOCK = 5
NPATHS = 2000
SEEDS20 = [20260629 + i for i in range(20)]
SEEDS5 = SEEDS20[:5]
INFLATION = 0.02
DEFLATE20 = (1.0 + INFLATION) ** 20

CONVENTIONS = ("lookahead", "lagged")   # Task 3 finding: ranking is convention-dependent

ALLOCS = [(10, 30), (14, 26), (20, 20), (24, 16), (30, 10)]
THRS_FINE = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
FLOOR_BASE = 3.6
TOPWR_BASE = 0.16

TOPWR_SWEEP = [0.12, 0.14, 0.16, 0.18, 0.20]
FLOOR_SWEEP = [3.0, 3.6, 4.43]
THRS_SWEEP = [20, 26]

THRS_FRONTIER = [22, 24, 26, 28, 30]
TOPWR_FRONTIER = [0.14, 0.16, 0.18]


# --------------------------------------------------------------------- self-test
def self_test(run_h, res_h):
    paths = la.gen_paths(run_h, res_h, 20260629)
    ok = True
    for thr, expect in ((20, 0.9995), (26, 0.9945)):
        labors, ruins, terms, cuts = la.run_config(
            paths, "lookahead",
            dict(run0=20 * M, reserve0=20 * M, thr=thr * M, floor=3.6 * M, top_wr=0.16))
        p = float((labors == 0).mean())
        match = abs(p - expect) < 5e-5
        print("  self-test lookahead thr%dM seed=20260629: P=%.4f (expect %.4f) %s"
              % (thr, p, expect, "OK" if match else "MISMATCH"))
        ok = ok and match
    return ok


# ------------------------------------------------------------------ path cache
_PATH_CACHE = {}


def paths_for(seed):
    """Generate paths once per seed, reuse across all configs (per plan Step 5
    performance note: generate the path list once per seed, reuse for every
    config -- do NOT regenerate per-config)."""
    if seed not in _PATH_CACHE:
        run_h, res_h = la.v6.get_paired_history()
        _PATH_CACHE[seed] = la.gen_paths(run_h, res_h, seed, n_paths=NPATHS, block=BLOCK)
    return _PATH_CACHE[seed]


def run_cell(seed, convention, cfg):
    paths = paths_for(seed)
    labors, ruins, terms, cuts = la.run_config(paths, convention, cfg)
    return labors, ruins, terms, cuts


# ------------------------------------------------------------------------ [1]
def section1(rows_out):
    print("\n" + "=" * 100)
    print("[1] FINE GRID: thr x alloc  (floor=%.1fM top_wr=%.2f)  %d seeds x %d conventions"
          % (FLOOR_BASE, TOPWR_BASE, len(SEEDS20), len(CONVENTIONS)))
    print("=" * 100)
    # store[convention][alloc][thr] -> concatenated labors across 20 seeds (for pooling)
    store = {c: {a: {} for a in ALLOCS} for c in CONVENTIONS}
    cell_p = {c: {a: {} for a in ALLOCS} for c in CONVENTIONS}
    for conv in CONVENTIONS:
        for (r0, s0) in ALLOCS:
            for thr in THRS_FINE:
                cfg = dict(run0=r0 * M, reserve0=s0 * M, thr=thr * M,
                           floor=FLOOR_BASE * M, top_wr=TOPWR_BASE)
                ps = []
                all_labors = []
                for seed in SEEDS20:
                    labors, ruins, terms, cuts = run_cell(seed, conv, cfg)
                    p = float((labors == 0).mean())
                    ps.append(p)
                    all_labors.append(labors)
                    rows_out.append(dict(section=1, convention=conv, alloc="%d:%d" % (r0, s0),
                                         thr=thr, seed=seed, p_labor0=p,
                                         cut_mean=float(cuts.mean())))
                pooled = np.concatenate(all_labors)
                k = int((pooled == 0).sum())
                n = len(pooled)
                lo, hi = lzstats.wilson_ci(k, n)
                store[conv][(r0, s0)][thr] = pooled
                cell_p[conv][(r0, s0)][thr] = dict(mean=float(np.mean(ps)), min=float(np.min(ps)),
                                                   max=float(np.max(ps)), pooled=k / n,
                                                   ci_lo=lo, ci_hi=hi)

    for conv in CONVENTIONS:
        print("\n  convention=%s" % conv)
        hdr = "    %-9s" % "alloc" + "".join("  thr%dM" % t for t in THRS_FINE)
        print(hdr)
        for (r0, s0) in ALLOCS:
            line = "    %-9s" % ("%d:%d" % (r0, s0))
            for thr in THRS_FINE:
                d = cell_p[conv][(r0, s0)][thr]
                line += "  %.4f" % d["mean"]
            print(line)
        print("    (mean P over %d seeds; ranges/CI below for the 20:20 row)" % len(SEEDS20))
        print("\n    20:20 row detail (mean[min-max] pooled_CI95):")
        for thr in THRS_FINE:
            d = cell_p[conv][(20, 20)][thr]
            print("      thr%2dM  P=%.4f [%.4f-%.4f]  pooled=%.4f CI95=[%.4f,%.4f]"
                  % (thr, d["mean"], d["min"], d["max"], d["pooled"], d["ci_lo"], d["ci_hi"]))

    return store, cell_p


# ------------------------------------------------------------------------ [2]
def section2(rows_out):
    print("\n" + "=" * 100)
    print("[2] top_wr SWEEP: top_wr x floor x thr(20,26)  alloc=20:20  %d seeds x %d conventions"
          % (len(SEEDS20), len(CONVENTIONS)))
    print("=" * 100)
    results = {c: {} for c in CONVENTIONS}
    for conv in CONVENTIONS:
        print("\n  convention=%s" % conv)
        for thr in THRS_SWEEP:
            print("    thr=%dM" % thr)
            for floor in FLOOR_SWEEP:
                line = "      floor=%.2fM  " % floor
                for topwr in TOPWR_SWEEP:
                    cfg = dict(run0=20 * M, reserve0=20 * M, thr=thr * M,
                               floor=floor * M, top_wr=topwr)
                    ps, cuts_l, terms_l = [], [], []
                    for seed in SEEDS20:
                        labors, ruins, terms, cuts = run_cell(seed, conv, cfg)
                        p = float((labors == 0).mean())
                        ps.append(p)
                        cuts_l.append(float(cuts.mean()))
                        terms_l.append(np.median(terms) / DEFLATE20 / M)
                        rows_out.append(dict(section=2, convention=conv, thr=thr, floor=floor,
                                             top_wr=topwr, seed=seed, p_labor0=p,
                                             cut_mean=float(cuts.mean()),
                                             term_real_med_M=float(np.median(terms) / DEFLATE20 / M)))
                    results[conv][(thr, floor, topwr)] = dict(
                        p_mean=float(np.mean(ps)), cut_mean=float(np.mean(cuts_l)),
                        term_med=float(np.mean(terms_l)))
                    line += " tw%.2f:P%.4f/cut%.2f/T%dM" % (
                        topwr, np.mean(ps), np.mean(cuts_l), round(np.mean(terms_l)))
                print(line)
    return results


# ------------------------------------------------------------------------ [3]
def section3(rows_out):
    print("\n" + "=" * 100)
    print("[3] V6-A' FRONTIER: thr x top_wr  (floor=%.1fM alloc=20:20)  %d seeds x %d conventions"
          % (FLOOR_BASE, len(SEEDS5), len(CONVENTIONS)))
    print("=" * 100)
    results = {c: {} for c in CONVENTIONS}
    for conv in CONVENTIONS:
        print("\n  convention=%s" % conv)
        hdr = "    %-10s" % "thr\\top_wr" + "".join("  tw=%.2f" % tw for tw in TOPWR_FRONTIER)
        print(hdr)
        for thr in THRS_FRONTIER:
            line_p = "    thr%2dM P  " % thr
            line_t = "    thr%2dM Tmed" % thr
            line_p5 = "    thr%2dM Tp5 " % thr
            for topwr in TOPWR_FRONTIER:
                cfg = dict(run0=20 * M, reserve0=20 * M, thr=thr * M,
                           floor=FLOOR_BASE * M, top_wr=topwr)
                ps, tmeds, tp5s = [], [], []
                for seed in SEEDS5:
                    labors, ruins, terms, cuts = run_cell(seed, conv, cfg)
                    ps.append(float((labors == 0).mean()))
                    tmeds.append(np.median(terms) / DEFLATE20 / M)
                    tp5s.append(np.percentile(terms, 5) / DEFLATE20 / M)
                    rows_out.append(dict(section=3, convention=conv, thr=thr, top_wr=topwr,
                                         seed=seed, p_labor0=float((labors == 0).mean()),
                                         term_real_med_M=float(np.median(terms) / DEFLATE20 / M),
                                         term_real_p5_M=float(np.percentile(terms, 5) / DEFLATE20 / M)))
                results[conv][(thr, topwr)] = dict(p_mean=float(np.mean(ps)),
                                                   term_med=float(np.mean(tmeds)),
                                                   term_p5=float(np.mean(tp5s)))
                line_p += "  %.4f" % np.mean(ps)
                line_t += "  %5dM" % round(np.mean(tmeds))
                line_p5 += "  %5dM" % round(np.mean(tp5s))
            print(line_p)
            print(line_t)
            print(line_p5)
    return results


# ------------------------------------------------------------------------ [4]
def section4(store):
    print("\n" + "=" * 100)
    print("[4] PAIRED TESTS (pooled %d seeds, alloc=20:20 unless noted)" % len(SEEDS20))
    print("=" * 100)
    for conv in CONVENTIONS:
        print("\n  convention=%s" % conv)
        base_alloc = (20, 20)
        # thr20 vs thr26
        la_ = store[conv][base_alloc][20]
        lb_ = store[conv][base_alloc][26]
        r = lzstats.paired_diff(la_, lb_)
        print("    thr20 vs thr26            dP=%+.4f  a_only=%5d b_only=%5d  p=%.2e"
              % (r["p_a"] - r["p_b"], r["a_only"], r["b_only"], r["mcnemar_p"]))
        # thr20 vs each neighbor
        for thr in THRS_FINE:
            if thr == 20:
                continue
            lb2 = store[conv][base_alloc][thr]
            r = lzstats.paired_diff(la_, lb2)
            print("    thr20 vs thr%-2d            dP=%+.4f  a_only=%5d b_only=%5d  p=%.2e"
                  % (thr, r["p_a"] - r["p_b"], r["a_only"], r["b_only"], r["mcnemar_p"]))
        # 20:20 vs 10:30 @ thr20
        la3 = store[conv][(20, 20)][20]
        lb3 = store[conv][(10, 30)][20]
        r = lzstats.paired_diff(la3, lb3)
        print("    20:20 vs 10:30 @thr20     dP=%+.4f  a_only=%5d b_only=%5d  p=%.2e"
              % (r["p_a"] - r["p_b"], r["a_only"], r["b_only"], r["mcnemar_p"]))


# ------------------------------------------------------------------------ [5] max guaranteed floor
def section5():
    print("\n" + "=" * 100)
    print("[5] Re-check: max guaranteed floor holding P>=0.999 (thr/alloc/top_wr swept)")
    print("=" * 100)
    floor_grid = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.43, 4.6, 4.8, 5.0]
    best = {}
    for conv in CONVENTIONS:
        overall_max = None
        overall_cfg = None
        for (r0, s0) in ALLOCS:
            for thr in THRS_FINE:
                maxfl = None
                for fl in floor_grid:
                    cfg = dict(run0=r0 * M, reserve0=s0 * M, thr=thr * M,
                               floor=fl * M, top_wr=TOPWR_BASE)
                    ps = []
                    for seed in SEEDS5:
                        labors, ruins, terms, cuts = run_cell(seed, conv, cfg)
                        ps.append(float((labors == 0).mean()))
                    if np.mean(ps) >= 0.999:
                        maxfl = fl
                if maxfl is not None and (overall_max is None or maxfl > overall_max):
                    overall_max = maxfl
                    overall_cfg = (r0, s0, thr)
        best[conv] = (overall_max, overall_cfg)
        print("  convention=%-10s max guaranteed floor (P>=0.999, 5 seeds) = %s  at alloc/thr=%s"
              % (conv, overall_max, overall_cfg))
    return best


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t_start = time.time()
    print("=" * 100)
    print("Task 5: V6-A FINE GRID + top_wr SWEEP  (block=%d N=%d, %d seeds fine / %d seeds frontier)"
          % (BLOCK, NPATHS, len(SEEDS20), len(SEEDS5)))
    print("conventions=%s" % (CONVENTIONS,))
    print("=" * 100)

    run_h, res_h = la.v6.get_paired_history()
    if not self_test(run_h, res_h):
        print("HALT: self-test failed (4-digit mismatch against known anchors).")
        return
    print("Self-test PASS. Proceeding.\n")

    rows = []
    store, cell_p = section1(rows)
    t1 = time.time()
    print("\n  [1] elapsed: %.1fs" % (t1 - t_start))

    results2 = section2(rows)
    t2 = time.time()
    print("\n  [2] elapsed: %.1fs" % (t2 - t1))

    results3 = section3(rows)
    t3 = time.time()
    print("\n  [3] elapsed: %.1fs" % (t3 - t2))

    section4(store)
    t4 = time.time()
    print("\n  [4] elapsed: %.1fs" % (t4 - t3))

    best5 = section5()
    t5 = time.time()
    print("\n  [5] elapsed: %.1fs" % (t5 - t4))

    df = pd.DataFrame(rows)
    out_dir = os.path.join(_REPO, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "labor_zero_v6a_fine_20260702.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s (%d rows)" % (out, len(df)))
    print("\nTotal elapsed: %.1fs" % (time.time() - t_start))
    print("Done (Task 5 fine grid + top_wr sweep).")


if __name__ == "__main__":
    main()
