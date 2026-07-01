"""
src/audit/labor_zero_v6_thr_dial_20260701.py
============================================
投入閾値ダイヤル (thr) の精密探索 + run/res 配分との交差。
前段 (labor_zero_v6_refill_timing_20260701.py) で、投入タイミング系ダイヤル
(max_wait / repeatable / postcrash / partial / dd-trigger) は全滅と確定。
唯一効いたのが thr (投入判定の残高水準) で 0.8745 -> 0.8965 (+2.2pp)。

ここで thr を主軸に:
  1. thr 精密掃引 (14M..38M) @ run20/res20, hold=True
  2. thr x run/res 配分 交差 (run が変わると res も変わる=投入原資が変わる)
  3. 最良 thr のシード安定性 (5seed) と労働年分布
  4. thr が「早期投入」に等価か検証: thr>=run0 は「引退直後から補填を寄せる」動きになる
     -> 極端形として「開始時点で run:res を寄せた配分」と比較 (thr の正体を切り分け)

重要な切り分け (QC): thr を上げる効果が
  (a) 真に「動的な早期投入ルール」由来なのか、
  (b) 単に「実効的な初期 run 配分を厚くしているだけ」なのか
を、静的配分 (run 厚め・投入なし) と対照して判定する。もし (b) なら
「thr ダイヤルの発見」でなく「配分の再確認」にすぎない -> 誠実に報告する。

V6-C (名目固定720万・フロアなし) 基準。block=5, N=2000, seeds=5. ASCII. No commit.
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

_spec = importlib.util.spec_from_file_location(
    "v6", os.path.join(_THIS, "labor_zero_v6_return_mc_20260629.py"))
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)

M = 1_000_000.0
SPEND = 7.2 * M
HORIZON = 20
INFLATION = 0.02
SEEDS = [20260629, 42, 12345, 99999, 777777]


def sim(run_path, res_path, *, run0, reserve0, thr, hold_if_crash=True, spend=SPEND):
    """名目固定支出・単発全額投入 (thr 未満 & 非マイナス年で1回だけ)。"""
    run, res = float(run0), float(reserve0)
    n = len(run_path)
    labor = 0
    fired = False
    for k in range(n):
        r_run = run_path[k]
        if (not fired) and run < thr and res > 1e-6 and ((not hold_if_crash) or r_run >= 0):
            run += res
            res = 0.0
            fired = True
        total = run + res
        spend_k = spend
        if total + 1e-6 < spend_k:
            labor += 1
            spend_k = total
        if run >= spend_k:
            run -= spend_k
        else:
            res -= (spend_k - run)
            run = 0.0
        if res < 0:
            res = 0.0
        run *= (1.0 + r_run)
        res *= (1.0 + res_path[k])
    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6), terminal=run + res)


def mc(run_h, res_h, *, run0, reserve0, thr, hold_if_crash=True,
       n_paths=2000, block=5, seed=20260629):
    rng = np.random.default_rng(seed)
    labors = np.empty(n_paths, int)
    terms = np.empty(n_paths)
    for p in range(n_paths):
        rp, sp = v6.make_block_path(rng, run_h, res_h, n=HORIZON, block=block)
        r = sim(rp, sp, run0=run0, reserve0=reserve0, thr=thr, hold_if_crash=hold_if_crash)
        labors[p] = r["labor_years"]
        terms[p] = r["terminal"]
    deflate = (1.0 + INFLATION) ** HORIZON
    return dict(p_labor0=float((labors == 0).mean()),
                labor_mean=float(labors.mean()),
                labor_arr=labors,
                term_med=round(float(np.median(terms)) / deflate / M, 1))


def mc5(run_h, res_h, **kw):
    ps = []
    base = None
    for sd in SEEDS:
        r = mc(run_h, res_h, seed=sd, **kw)
        ps.append(r["p_labor0"])
        if sd == SEEDS[0]:
            base = r
    base["p_range"] = (min(ps), max(ps))
    return base


def main():
    run_h, res_h = v6.get_paired_history(strat="sc2.6")
    print("=" * 80)
    print("投入閾値ダイヤル thr 精密探索 (V6-C 名目固定720万・フロアなし)")
    print("=" * 80)

    # セルフテスト: thr=run0=20M, hold=True で v6本体 R1 を再現
    ref = v6.mc_prob(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                     run0=20*M, reserve0=20*M, tranches=[(20*M, 1.0)], hold_if_crash=True)
    mine = mc(run_h, res_h, run0=20*M, reserve0=20*M, thr=20*M)
    ok = abs(ref["p_labor0"] - mine["p_labor0"]) < 1e-9
    print(f"SELF-TEST: v6本体={ref['p_labor0']:.4f} 拡張={mine['p_labor0']:.4f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("HALT")
        return
    print()

    # 1. thr 精密掃引 @ run20/res20
    print("[1] thr 精密掃引 @ run20M/res20M, hold=True")
    print(f"    {'thr':>6} {'P(labor0)':>11} {'seed範囲':>18} {'labor_mean':>11} {'term_med':>10}")
    for t in [14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 38]:
        r = mc5(run_h, res_h, run0=20*M, reserve0=20*M, thr=t*M)
        lo, hi = r["p_range"]
        print(f"    {t:>4}M {r['p_labor0']:>11.4f}  [{lo:.4f}-{hi:.4f}] "
              f"{r['labor_mean']:>11.2f} {r['term_med']:>9.0f}oku")
    print()

    # 2. thr x run/res 配分 交差
    print("[2] thr x run/res 配分 交差 (hold=True)")
    allocs = [(10, 30), (14, 26), (20, 20), (24, 16), (30, 10)]
    thrs = [14, 20, 26, 30]
    header = "    " + f"{'run:res':>10}" + "".join(f"  thr={t}M" for t in thrs)
    print(header)
    for run0, res0 in allocs:
        row = f"    {run0:>4}:{res0:<4}  "
        for t in thrs:
            r = mc(run_h, res_h, run0=run0*M, reserve0=res0*M, thr=t*M)
            row += f"  {r['p_labor0']:.4f} "
        print(row)
    print("    (単一シード20260629・傾向把握用)")
    print()

    # 3. QC切り分け: thr高 == 早期投入 の効果か、単なる初期配分厚めか
    print("[3] QC: thr の正体切り分け -- 動的投入 vs 静的初期配分")
    print("    対照A: run20/res20 + thr高 (動的に早期投入)")
    print("    対照B: 初期配分を run 厚めにして thr=0 相当 (投入させない=静的)")
    print("    もし A の改善が B で再現するなら thr は『配分の言い換え』にすぎない")
    print()
    # A: run20/res20, thr=30 (ほぼ即・全額を運用へ)
    rA = mc5(run_h, res_h, run0=20*M, reserve0=20*M, thr=30*M)
    print(f"    A run20/res20 thr=30M(早期投入):   P={rA['p_labor0']:.4f} "
          f"[{rA['p_range'][0]:.4f}-{rA['p_range'][1]:.4f}] term_med={rA['term_med']:.0f}oku")
    # B: run40/res0 (最初から全部運用=投入する原資なし=静的)
    rB = mc5(run_h, res_h, run0=40*M, reserve0=0.0, thr=1*M)
    print(f"    B run40/res0 (静的・投入なし):       P={rB['p_labor0']:.4f} "
          f"[{rB['p_range'][0]:.4f}-{rB['p_range'][1]:.4f}] term_med={rB['term_med']:.0f}oku")
    # B2: run34/res6 中間
    rB2 = mc5(run_h, res_h, run0=34*M, reserve0=6*M, thr=1*M)
    print(f"    B2 run34/res6 (静的・ほぼ投入なし):  P={rB2['p_labor0']:.4f} "
          f"[{rB2['p_range'][0]:.4f}-{rB2['p_range'][1]:.4f}] term_med={rB2['term_med']:.0f}oku")
    print()

    # 4. 最良構成の労働年分布
    print("[4] 最良構成 (run20/res20, thr=26M) 労働年分布 (seed=20260629)")
    rbest = mc(run_h, res_h, run0=20*M, reserve0=20*M, thr=26*M)
    la = rbest["labor_arr"]
    print(f"    P(labor0)={ (la==0).mean():.4f}  labor>0={(la>0).sum()}/2000")
    vals, cnts = np.unique(la, return_counts=True)
    for v, c in zip(vals, cnts):
        if v <= 3 or c >= 10:
            print(f"      {v:>2}yr: {c:>4} ({c/2000*100:.1f}%)")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
