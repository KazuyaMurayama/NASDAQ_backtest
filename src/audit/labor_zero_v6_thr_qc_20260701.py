"""
src/audit/labor_zero_v6_thr_qc_20260701.py
==========================================
独立QC: thr_dial の主要ヘッドライン (10:30 -> 0.917, run20res20 thr26 -> 0.8965)
を、v6 / thr_dial を一切 import せず、素の numpy で完全に再実装して再現する。
一致すれば発見は信頼できる。ブロックブートストラップも独立実装。

再現対象:
  H1: run10/res30, thr=14M, hold=True  -> 0.9170 (seed20260629)
  H2: run20/res20, thr=26M, hold=True  -> 0.8965
  H3: run20/res20, thr=20M, hold=True  -> 0.8745 (=v6本体 R1 baseline)

データ (sc2.6 run / bond reserve after-tax 年次) だけ v3 から借りる (数値の一次ソース)。
sim とブートストラップは独立実装。ASCII. No commit.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# データのみ v3 から (numbers are the primary source; logic is independent)
_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_THIS, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

M = 1_000_000.0
SPEND = 7_200_000.0
H = 20
YEARS = list(range(1975, 2026))


def load():
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    run = np.array([float(rets["sc2.6"].loc[y]) for y in YEARS])
    res = np.array([float(bond.loc[y]) for y in YEARS])
    return run, res


def block_path(rng, run_h, res_h, n, block):
    """独立実装のブロックブートストラップ (wrap-around)."""
    L = len(run_h)
    out_r = []
    out_s = []
    while len(out_r) < n:
        s = int(rng.integers(0, L))
        for j in range(block):
            if len(out_r) >= n:
                break
            idx = (s + j) % L
            out_r.append(run_h[idx])
            out_s.append(res_h[idx])
    return np.array(out_r[:n]), np.array(out_s[:n])


def one_sim(rp, sp, run0, res0, thr):
    """独立実装: 名目固定・単発全額投入・hold_if_crash=True 固定."""
    run = run0
    res = res0
    fired = False
    labor = 0
    for k in range(len(rp)):
        rr = rp[k]
        # refill: run<thr, res残, 非マイナス年, 未投入
        if (not fired) and (run < thr) and (res > 1e-6) and (rr >= 0):
            run = run + res
            res = 0.0
            fired = True
        tot = run + res
        need = SPEND
        if tot + 1e-6 < need:
            labor += 1
            need = tot
        if run >= need:
            run -= need
        else:
            res -= (need - run)
            run = 0.0
        if res < 0:
            res = 0.0
        run *= (1.0 + rr)
        res *= (1.0 + sp[k])
    return labor


def prob(run_h, res_h, run0, res0, thr, seed=20260629, n=2000, block=5):
    rng = np.random.default_rng(seed)
    z = 0
    for _ in range(n):
        rp, sp = block_path(rng, run_h, res_h, H, block)
        if one_sim(rp, sp, run0, res0, thr) == 0:
            z += 1
    return z / n


def main():
    run_h, res_h = load()
    print("=" * 66)
    print("独立QC: thr_dial ヘッドライン再現 (別実装・seed20260629)")
    print("=" * 66)
    cases = [
        ("H3 run20/res20 thr20M (=本体baseline)", 20*M, 20*M, 20*M, 0.8745),
        ("H2 run20/res20 thr26M",                20*M, 20*M, 26*M, 0.8965),
        ("H1 run10/res30 thr14M (新天井)",        10*M, 30*M, 14*M, 0.9170),
    ]
    allok = True
    for label, r0, s0, t, expect in cases:
        p = prob(run_h, res_h, r0, s0, t)
        ok = abs(p - expect) < 0.005
        allok = allok and ok
        print(f"  {label:<40} QC={p:.4f} 期待={expect:.4f} "
              f"-> {'OK' if ok else 'MISMATCH'}")
    print()
    print(f"総合: {'ALL PASS -- 発見は信頼できる' if allok else 'FAIL -- 再確認せよ'}")


if __name__ == "__main__":
    main()
