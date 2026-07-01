"""
src/audit/labor_zero_v6_refill_timing_20260701.py
=================================================
穴埋め検証: 補填投入の「タイミング・ダイヤル」の網羅探索。
v6本体 (sim_one) は hold_if_crash を「マイナス年はスキップ・非マイナス年に投入」の
待ち続ける版しか持たず、以下が一度も試されていなかった:

  A. max_wait: マイナスが N 年続いたら強制投入（待機上限） <- ユーザー提案の核心
  B. 反復投入 (repeatable refill): 一度投入後も run<thr で再度投入（capあり）
  C. postcrash_thr: 暴落後に投入閾値を引き上げ/引き下げ
  D. 部分投入 (partial): 全額でなく一定割合ずつ（v3で逆効果と出たが再確認）
  E. 閾値 thr: 2000万でなく別水準で投入判定
  F. drawdown-trigger: run のドローダウン%で投入（残高でなく下落率）

すべて V6-C (名目固定720万・フロアなし・run20/res20) を基準に、P(労働=0) の変化を測る。
sim を独立に再実装（v6 sim_one を拡張）し、v6本体とセルフテストで一致確認してから探索する。

block=5, N=2000, seeds=5. ASCII prints. No commit.
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


def sim_flex(run_path, res_path, *, run0, reserve0, thr, spend=SPEND,
             hold_if_crash=True, max_wait=None, repeatable=False, cap=None,
             postcrash_thr=None, partial_frac=1.0, dd_trigger=None):
    """
    拡張投入シミュレータ (名目固定支出・フロアなし)。
    - thr: 運用残高がこれ未満で投入判定
    - hold_if_crash: マイナス年は投入を見送る
    - max_wait: hold中の待機が max_wait 年に達したら「強制投入」(マイナスでも入れる) <- A
    - repeatable: True なら1回でなく複数回投入 (cap まで) <- B
    - cap: 反復投入時の1回あたり上限額
    - postcrash_thr: 一度暴落を経験したら thr をこの値に切り替え <- C
    - partial_frac: 1回の投入で移す割合 (1.0=全額) <- D
    - dd_trigger: run のピークからの下落率がこれを超えたら投入 (残高thrの代わり) <- F
    Returns dict(labor_years, ruin, terminal).
    """
    run, res = float(run0), float(reserve0)
    n = len(run_path)
    labor = 0
    wait = 0                 # hold により連続で見送った年数
    fired_once = False
    run_peak = run
    cur_thr = thr

    for k in range(n):
        r_run = run_path[k]
        run_peak = max(run_peak, run)

        # ---- 投入判定 ----
        # トリガー: 残高 < thr  or  (dd_trigger 指定時) ピークからの下落率
        if dd_trigger is not None:
            dd = (run_peak - run) / run_peak if run_peak > 0 else 0.0
            trig = dd >= dd_trigger
        else:
            trig = run < cur_thr

        can_invest = trig and res > 1e-6 and (repeatable or not fired_once)

        if can_invest:
            crash_year = (r_run < 0)
            hold_block = hold_if_crash and crash_year
            # max_wait: 待機が上限に達したら強制投入 (hold_block を無視)
            force = (max_wait is not None) and (wait >= max_wait)
            if hold_block and not force:
                wait += 1        # 見送り、待機カウント +1
            else:
                # 投入実行
                move = res if (cap is None) else min(res, cap)
                move *= partial_frac
                run += move
                res -= move
                wait = 0
                fired_once = True
                if postcrash_thr is not None and crash_year:
                    cur_thr = postcrash_thr
        else:
            # 投入できない年は待機カウントを維持しない (トリガー外なら待機リセット)
            if not trig:
                wait = 0

        # ---- 支出 (名目固定) ----
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

        # ---- 成長 ----
        run *= (1.0 + r_run)
        res *= (1.0 + res_path[k])

    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6),
                terminal=run + res)


def mc(run_h, res_h, *, run0, reserve0, thr, n_paths=2000, block=5, seed=20260629,
       **kw):
    rng = np.random.default_rng(seed)
    labors = np.empty(n_paths, int)
    terms = np.empty(n_paths)
    for p in range(n_paths):
        rp, sp = v6.make_block_path(rng, run_h, res_h, n=HORIZON, block=block)
        r = sim_flex(rp, sp, run0=run0, reserve0=reserve0, thr=thr, **kw)
        labors[p] = r["labor_years"]
        terms[p] = r["terminal"]
    deflate = (1.0 + INFLATION) ** HORIZON
    return dict(
        p_labor0=float((labors == 0).mean()),
        labor_mean=float(labors.mean()),
        labor_max=int(labors.max()),
        term_med=round(float(np.median(terms)) / deflate / M, 1),
    )


def mc_seeds(run_h, res_h, **kw):
    ps = []
    base = None
    for sd in SEEDS:
        r = mc(run_h, res_h, seed=sd, **kw)
        ps.append(r["p_labor0"])
        if sd == SEEDS[0]:
            base = r
    base["p_range"] = (min(ps), max(ps))
    return base


def _selftest(run_h, res_h):
    """max_wait=None, repeatable=False, partial=1.0 で v6本体 R1(run20res20 HOLD) を再現。"""
    ref = v6.mc_prob(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                     run0=20*M, reserve0=20*M, tranches=[(20*M, 1.0)], hold_if_crash=True)
    mine = mc(run_h, res_h, run0=20*M, reserve0=20*M, thr=20*M, hold_if_crash=True,
              n_paths=2000, block=5, seed=20260629)
    print("SELF-TEST (run20res20 HOLD nominal-fixed):")
    print(f"  v6本体 P(labor0)={ref['p_labor0']:.4f}  |  拡張版 P(labor0)={mine['p_labor0']:.4f}")
    ok = abs(ref['p_labor0'] - mine['p_labor0']) < 1e-9
    print(f"  -> {'PASS (完全一致)' if ok else 'FAIL -- 拡張版が本体と不一致。探索前に修正せよ'}")
    return ok


def main():
    run_h, res_h = v6.get_paired_history(strat="sc2.6")
    print("=" * 78)
    print("補填投入タイミング・ダイヤルの網羅探索 (V6-C 名目固定720万・run20/res20)")
    print("=" * 78)
    if not _selftest(run_h, res_h):
        print("HALT")
        return
    print()

    base = dict(run0=20*M, reserve0=20*M, thr=20*M, n_paths=2000, block=5)

    def show(label, **kw):
        r = mc_seeds(run_h, res_h, **{**base, **kw})
        lo, hi = r["p_range"]
        print(f"  {label:<44} P(labor0)={r['p_labor0']:.4f} [{lo:.4f}-{hi:.4f}]"
              f"  labor_mean={r['labor_mean']:.2f} term_med={r['term_med']:.0f}oku")

    print("[基準] 待ち続ける版 (max_wait=None) = V6-C そのもの")
    show("hold=True, wait-forever (=V6-C)", hold_if_crash=True)
    print()

    print("[A] max_wait: マイナスが N 年続いたら強制投入 <- ユーザー提案")
    for mw in [1, 2, 3, 5]:
        show(f"hold=True, max_wait={mw}", hold_if_crash=True, max_wait=mw)
    show("hold=False (即投入・待たない)", hold_if_crash=False)
    print()

    print("[B] 反復投入 (repeatable, capあり) + hold")
    for cap in [8, 12, 16, 20]:
        show(f"repeatable cap={cap}M, hold=True", hold_if_crash=True,
             repeatable=True, cap=cap*M)
    print()

    print("[C] postcrash_thr: 暴落後に投入閾値を変える + hold")
    for pc in [14, 16, 24, 28]:
        show(f"postcrash_thr={pc}M, hold=True", hold_if_crash=True, postcrash_thr=pc*M)
    print()

    print("[D] 部分投入 (partial_frac) + hold  [v3で逆効果・再確認]")
    for pf in [0.5, 0.75, 1.0]:
        show(f"partial_frac={pf}, hold=True", hold_if_crash=True, partial_frac=pf)
    print()

    print("[E] 投入閾値 thr の水準 + hold")
    for t in [14, 20, 26, 30]:
        show(f"thr={t}M, hold=True", thr=t*M, hold_if_crash=True)
    print()

    print("[F] ドローダウン%トリガー (残高でなく下落率で投入) + hold")
    for dd in [0.20, 0.35, 0.50]:
        show(f"dd_trigger={dd:.0%}, hold=True", hold_if_crash=True, dd_trigger=dd)
    print()

    print("[A×max_wait を最良閾値で] max_wait=2 を thr掃引と組合せ")
    for t in [16, 20, 26]:
        show(f"thr={t}M, max_wait=2, hold=True", thr=t*M, hold_if_crash=True, max_wait=2)
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
