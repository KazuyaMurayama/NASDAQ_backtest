"""
独立QC: V6-A グリッドの主要ヘッドラインを v6/grid を import せず別実装で再現。
  H1: run20/res20 thr20M floor360 -> 0.9995 cut~2.44 (現V6-A・最良P)
  H2: run20/res20 thr26M floor360 -> 0.9945 cut~2.26 term倍増 (終端改善オプション)
  H3: run10/res30 thr10M floor360 -> 0.9930 (補填厚は逆効果=V6-Cと非対称)
データのみ v3 から。sim/bootstrap は独立実装(floor+top_wr対応)。ASCII. No commit.
"""
from __future__ import annotations
import importlib.util, os, sys
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_THIS, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)

M = 1_000_000.0
SPEND = 7_200_000.0
H = 20
YEARS = list(range(1975, 2026))
INFL = 0.02


def load():
    rets = v3.load_rets(); bond = v3.load_mixed_reserve({"bond": 1.0})
    run = np.array([float(rets["sc2.6"].loc[y]) for y in YEARS])
    res = np.array([float(bond.loc[y]) for y in YEARS])
    return run, res


def bpath(rng, rh, sh, n, block):
    L = len(rh); orr = []; osr = []
    while len(orr) < n:
        s = int(rng.integers(0, L))
        for j in range(block):
            if len(orr) >= n:
                break
            k = (s + j) % L; orr.append(rh[k]); osr.append(sh[k])
    return np.array(orr[:n]), np.array(osr[:n])


def sim(rp, sp, run0, res0, thr, floor, top_wr):
    run, res, fired = run0, res0, False
    labor = cut = 0
    for k in range(len(rp)):
        rr = rp[k]
        if (not fired) and run < thr and res > 1e-6 and rr >= 0:
            run += res; res = 0.0; fired = True
        tot = run + res
        want = floor
        if top_wr is not None and tot > 1e-9 and (SPEND / tot) <= top_wr:
            want = SPEND
        if tot + 1e-6 < want:
            labor += 1; want = tot
        if want < SPEND - 1e-6:
            cut += 1
        if run >= want:
            run -= want
        else:
            res -= (want - run); run = 0.0
        if res < 0:
            res = 0.0
        run *= (1 + rr); res *= (1 + sp[k])
    return labor, cut, run + res


def prob(rh, sh, run0, res0, thr, floor, top_wr=0.16, seed=20260629, n=2000, block=5):
    rng = np.random.default_rng(seed)
    z = c = 0; terms = []
    for _ in range(n):
        rp, sp = bpath(rng, rh, sh, H, block)
        la, cu, tm = sim(rp, sp, run0, res0, thr, floor, top_wr)
        if la == 0:
            z += 1
        c += cu; terms.append(tm)
    defl = (1 + INFL) ** H
    return z / n, c / n, np.median(terms) / defl / M


def main():
    rh, sh = load()
    print("=" * 68)
    print("独立QC: V6-A グリッド ヘッドライン (別実装・seed20260629)")
    print("=" * 68)
    cases = [
        ("H1 run20res20 thr20 floor360 (現V6-A)", 20*M, 20*M, 20*M, 3.6*M, 0.9995, 2.44),
        ("H2 run20res20 thr26 floor360 (終端改善)", 20*M, 20*M, 26*M, 3.6*M, 0.9945, 2.26),
        ("H3 run10res30 thr10 floor360 (補填厚=逆)", 10*M, 30*M, 10*M, 3.6*M, 0.9930, 3.46),
    ]
    allok = True
    for label, r0, s0, t, fl, ep, ecut in cases:
        p, cu, tm = prob(rh, sh, r0, s0, t, fl)
        ok = abs(p - ep) < 0.005 and abs(cu - ecut) < 0.10
        allok = allok and ok
        print(f"  {label:<40}")
        print(f"     QC: P={p:.4f}(exp{ep})  cut={cu:.2f}(exp{ecut})  term={tm:.0f}oku "
              f"-> {'OK' if ok else 'MISMATCH'}")
    print()
    print(f"総合: {'ALL PASS' if allok else 'FAIL -- 再確認'}")


if __name__ == "__main__":
    main()
