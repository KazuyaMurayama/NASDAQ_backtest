"""
src/audit/labor_zero_v6_qc_20260629.py
======================================
Independent QC for v6. Re-implements the block-bootstrap + retirement sim FROM SCRATCH
(does NOT import v6.sim_one / v6.make_block_path) and reproduces two headline numbers:
  (1) baseline run20/res20 + hold-if-crash, nominal-fixed 7.2M -> P(labor0) ~ 0.875
  (2) floor=3.6M asset-linked variant -> P(labor0) = 1.000
Only the return data loaders (v3.load_rets / load_mixed_reserve) are shared -- those are
the already-trusted v3 foundation. The sim and the bootstrap are re-coded independently.

If both reproduce to within MC noise (same seed -> EXACT), the v6 headline is trustworthy.
ASCII-only. No commit, no file writes.
"""
from __future__ import annotations
import importlib.util, os, sys
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (os.path.dirname(_THIS), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_spec = importlib.util.spec_from_file_location("v3", os.path.join(_THIS, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)
M = 1e6
YEARS = list(range(1975, 2026))


def hist_pairs():
    rets = v3.load_rets(); res = v3.load_mixed_reserve({"bond": 1.0})
    rr = np.array([float(rets["sc2.6"][y]) for y in YEARS])
    sr = np.array([float(res[y]) for y in YEARS])
    return rr, sr


def qc_sim(rng, rr, sr, *, run0, res0, thr, floor, top_wr, hold, n=20, block=5):
    """Independent re-impl: build a bootstrap path inline, simulate one retirement."""
    H = len(rr)
    # build path
    rpath = []; spath = []
    while len(rpath) < n:
        st = int(rng.integers(0, H))
        for j in range(block):
            if len(rpath) >= n:
                break
            k = (st + j) % H
            rpath.append(rr[k]); spath.append(sr[k])
    run, res, fired, labor = run0, res0, False, 0
    for k in range(n):
        rg = rpath[k]
        # refill (single all-in tranche at thr), held on crash years
        if not (hold and rg < 0):
            if (not fired) and run < thr and res > 1e-6:
                run += res; res = 0.0; fired = True
        total = run + res
        if floor > 0:
            want = floor
            if top_wr is not None and total > 0 and (7.2 * M / total) <= top_wr:
                want = 7.2 * M
        else:
            want = 7.2 * M
        if total + 1e-6 < want:
            labor += 1; want = total
        if run >= want:
            run -= want
        else:
            res -= (want - run); run = 0.0
        if res < 0:
            res = 0.0
        run *= (1.0 + rg); res *= (1.0 + spath[k])
    return labor


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rr, sr = hist_pairs()
    N = 2000
    print("v6 INDEPENDENT QC (re-implemented sim+bootstrap, same seed=20260629)")
    # (1) baseline run20/res20 hold, nominal-fixed
    rng = np.random.default_rng(20260629)
    lab = [qc_sim(rng, rr, sr, run0=20 * M, res0=20 * M, thr=20 * M, floor=0.0,
                  top_wr=None, hold=True) for _ in range(N)]
    p1 = np.mean([x == 0 for x in lab])
    print("  (1) run20/res20 HOLD nominal-fixed:  P(labor0)=%.3f   (v6 reported 0.875)" % p1)
    # (2) floor=3.6 asset-linked
    rng = np.random.default_rng(20260629)
    lab = [qc_sim(rng, rr, sr, run0=20 * M, res0=20 * M, thr=20 * M, floor=3.6 * M,
                  top_wr=0.16, hold=True) for _ in range(N)]
    p2 = np.mean([x == 0 for x in lab])
    print("  (2) floor=3.6M top16 asset-linked:   P(labor0)=%.3f   (v6 reported 1.000)" % p2)
    ok = abs(p1 - 0.875) < 0.01 and abs(p2 - 1.000) < 0.005
    print("  -> QC %s" % ("PASS (headline reproduced independently)" if ok else "FAIL -- investigate"))


if __name__ == "__main__":
    main()
