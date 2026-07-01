"""
src/audit/labor_zero_v6a_grid_20260701.py
=========================================
V6-A (フロア版) の thr × 初期配分 グリッドサーチ。
§3.2b で V6-C(名目固定) は thr↑＋補填厚めで天井 0.88->0.917 と判明したが、
V6-A(保証フロア360万＋好機720万) は floor/top_wr だけ振って
run20/res20・thr=run0 固定のままだった＝同じ穴。

ここで V6-A について:
  1. thr × 配分 グリッド @ floor=360万 で P(labor0) と cut_mean(節約年数) が改善するか
  2. floor を下げずに P を上げられるか / 同じ P でより高いフロアを保証できるか
     = 「保証支出とP のフロンティア」が thr/配分で外側にシフトするか
  3. 最良構成のシード安定性

V6-A では labor は「総資産<floor の年」にのみ発生。thr/配分が効くのは
「早めに補填を寄せて枯渇を防ぐ」効果が floor 割れ確率を下げるか、という経路。
ただし V6-C と違い、フロアで既に labor がほぼ消えている(0.9985)ので、
改善余地は「cut_mean(720->360に落とす年数)を減らす」or「同Pでフロアを上げる」側に出るはず。

v6本体 mc_prob をそのまま使用(floor/top_wr/tranches 対応済)。block=5,N=2000,seeds=5.
ASCII. No commit.
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
SEEDS = [20260629, 42, 12345, 99999, 777777]
TOP_WR = 0.16   # 好機トップアップの取り崩し率ゲート (V6-A 既定)


def one(run_h, res_h, run0, res0, thr, floorM, seed=20260629, top_wr=TOP_WR):
    return v6.mc_prob(run_h, res_h, n_paths=2000, block=5, seed=seed,
                      run0=run0*M, reserve0=res0*M, tranches=[(thr*M, 1.0)],
                      hold_if_crash=True, floor=floorM*M, top_wr=top_wr)


def five(run_h, res_h, **kw):
    ps, cuts = [], []
    base = None
    for sd in SEEDS:
        r = one(run_h, res_h, seed=sd, **kw)
        ps.append(r["p_labor0"]); cuts.append(r["cut_mean"])
        if sd == SEEDS[0]:
            base = r
    base["p_range"] = (min(ps), max(ps))
    base["cut_range"] = (min(cuts), max(cuts))
    return base


def main():
    run_h, res_h = v6.get_paired_history(strat="sc2.6")
    print("=" * 82)
    print("V6-A (フロア版) thr × 配分 グリッドサーチ  [floor=保証支出, top_wr=0.16]")
    print("=" * 82)

    # セルフテスト: v6本体 ROUND3 の floor3.6 run20res20 thr20 を再現
    ref = v6.mc_prob(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                     run0=20*M, reserve0=20*M, tranches=[(20*M, 1.0)],
                     hold_if_crash=True, floor=3.6*M, top_wr=0.16)
    print(f"SELF-TEST: v6本体 floor3.6 run20res20 thr20 P(labor0)={ref['p_labor0']:.4f} "
          f"cut_mean={ref['cut_mean']:.2f}  (期待 P=0.9995 cut~2.44)")
    print()

    # ---- [1] floor=360万 固定で thr×配分グリッド ----
    print("[1] floor=360万 固定: thr × 配分 グリッド  (P(labor0) / cut_mean)")
    allocs = [(10, 30), (14, 26), (20, 20), (24, 16)]
    thrs = [10, 14, 20, 26, 30]
    hdr = f"    {'run:res':>9}" + "".join(f"   thr{t}M" for t in thrs)
    print(hdr)
    for r0, s0 in allocs:
        row = f"    {r0:>3}:{s0:<4} "
        for t in thrs:
            r = one(run_h, res_h, r0, s0, t, 3.6)
            row += f"  {r['p_labor0']:.3f}/{r['cut_mean']:.1f}"
        print(row)
    print("    (単一シード20260629・P/cut。cut=720万を360万に下げた年数の平均)")
    print()

    # ---- [2] 保証支出フロンティア: 各配分×thr で、P>=0.999 を保つ最大フロア ----
    print("[2] 保証支出フロンティア: P(labor0)>=0.999 を保てる最大フロア (万円)")
    print("    (各構成で floor を 360->上げていき P>=0.999 の上限を二分探索的に走査)")
    floor_grid = [3.6, 4.0, 4.43, 5.0, 5.5, 6.0, 6.5, 7.2]
    configs = [
        ("run10:res30 thr10M(補填厚・早期)", 10, 30, 10),
        ("run20:res20 thr20M(現V6-A)",       20, 20, 20),
        ("run20:res20 thr26M(早期投入)",     20, 20, 26),
        ("run14:res26 thr14M",               14, 26, 14),
    ]
    for label, r0, s0, t in configs:
        best_floor = None
        prow = []
        for fl in floor_grid:
            r = five(run_h, res_h, run0=r0, res0=s0, thr=t, floorM=fl)
            prow.append((fl, r["p_labor0"]))
            if r["p_labor0"] >= 0.999:
                best_floor = fl
        maxfl = max((fl for fl, p in prow if p >= 0.999), default=None)
        detail = " ".join(f"{fl:.1f}:{p:.3f}" for fl, p in prow)
        print(f"    {label:<30} maxFloor(P>=0.999)={maxfl}万 | {detail}")
    print()

    # ---- [3] 最良構成のシード安定性 & 主要指標 ----
    print("[3] 注目構成のシード安定性 (floor=360万)")
    for label, r0, s0, t in [
        ("run10:res30 thr10M", 10, 30, 10),
        ("run20:res20 thr20M (現)", 20, 20, 20),
        ("run20:res20 thr26M", 20, 20, 26),
    ]:
        r = five(run_h, res_h, run0=r0, res0=s0, thr=t, floorM=3.6)
        lo, hi = r["p_range"]; clo, chi = r["cut_range"]
        print(f"    {label:<26} P={r['p_labor0']:.4f}[{lo:.4f}-{hi:.4f}] "
              f"cut={r['cut_mean']:.2f}[{clo:.2f}-{chi:.2f}] "
              f"term_med={r['term_real_med_M']:.0f}oku p5={r['term_real_p5_M']:.0f}oku")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
