"""
src/audit/labor_zero_v6_startasset_sweep_20260629.py
====================================================
v6 補遺 -- 開始資産を変えたときの P(労働=0)（名目固定720万・フロアなし）。
「固定720万を貫くには開始資産はいくら必要か」を直接計算する。§3.5 の一次根拠。

run:res = 1:1（開始資産に比例配分）、tranche閾値=run0で全額1発投入、hold_if_crash=True、
block=5、N=2000、シード5本。v6 の sim_one / mc_prob をそのまま使う（再実装しない）。

結論: フロアなし固定720万では、6000万でも P=0.981 止まり。テールに「引退直後の暴落
クラスタ→固定支出が枯らす」14年労働経路が残るため、資産を積んでも100%に漸近しきらない。
4000万のまま「不況時のみ360万に下げる」（§3.4 フロア360万）が P=0.9985 で確実かつ低コスト。

ASCII-only prints. No commit, no temp files.
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
START_ASSETS = [40, 45, 49, 50, 55, 60]   # million-yen units
SEEDS = [20260629, 42, 12345, 99999, 777777]


def run():
    run_h, res_h = v6.get_paired_history(strat="sc2.6")

    print("=" * 72)
    print("Nominal-fixed 7.2M (NO floor) -- P(labor=0) by starting asset")
    print("split run:res = 1:1 / tranche=run0 all-in / hold_if_crash=True / block=5 / N=2000")
    print("=" * 72)
    print(f"{'start':>7} {'run:res':>10} {'P(labor0)':>11} {'seed-range':>18} "
          f"{'labor_mean':>11} {'term_med':>10}")
    print("-" * 72)
    for total in START_ASSETS:
        run0 = total / 2 * M
        res0 = total / 2 * M
        cfg = dict(run0=run0, reserve0=res0, tranches=[(run0, 1.0)], hold_if_crash=True)
        ps = []
        base = None
        for sd in SEEDS:
            r = v6.mc_prob(run_h, res_h, n_paths=2000, block=5, seed=sd, **cfg)
            ps.append(r["p_labor0"])
            if sd == SEEDS[0]:
                base = r
        print(f"{total:>5}M  {int(total/2):>4}:{int(total/2):<4} {base['p_labor0']:>11.4f}"
              f"  [{min(ps):.4f}-{max(ps):.4f}] {base['labor_mean']:>11.2f}"
              f"  {base['term_real_med_M']:>8.1f}oku")

    print()
    print("withdrawal-rate reference (7.2M / start):")
    for total in START_ASSETS:
        print(f"  {total}M -> WR={7.2/total*100:.1f}%")

    # 49M detail: labor-year distribution
    print()
    print("=" * 72)
    print("49M start detail (seed=20260629) -- labor-year distribution")
    print("=" * 72)
    rng = np.random.default_rng(20260629)
    labors = []
    for _ in range(2000):
        rp, sp = v6.make_block_path(rng, run_h, res_h, n=20, block=5)
        r = v6.sim_one(rp, sp, run0=24.5*M, reserve0=24.5*M,
                       tranches=[(24.5*M, 1.0)], hold_if_crash=True)
        labors.append(r["labor_years"])
    labors = np.array(labors)
    print(f"  P(labor=0)={ (labors==0).mean():.4f}  labor>0 paths={(labors>0).sum()}/2000")
    vals, cnts = np.unique(labors, return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"    {v:>2}yr: {c:>4} ({c/2000*100:.1f}%)")
    print()
    print("DONE.")


if __name__ == "__main__":
    run()
