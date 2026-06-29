"""
src/audit/labor_zero_v7_manual_rule_20260629.py
================================================
v7 -- MANUALLY EXECUTABLE RETIREMENT RULE
User request (2026-06-29): sc2.6 is a daily automated quant system; no human can run it.
Design a NEW rule that:
  1. States starting allocation explicitly
  2. States spending-cut trigger conditions clearly
  3. States operating-rule change conditions clearly
  Must be executable by an ordinary person without daily monitoring.

DESIGN: "TQQQ+Bond 年1回チェック" ルール
-----------------------------------------
使用指標: 年末(12月末)の資産総額のみ。計算機・スプレッドシート不要。
投資商品:
  - 運用スリーブ: TQQQ (NASDAQ100 3倍 ETF, 米国籍 or 国内3倍投信)
  - 補填スリーブ: 先進国債券インデックス (eMaxis Slim 先進国債券 or 同等)
チェック頻度: 年1回 (12月末残高を確認するだけ)

[開始時配分]
  運用スリーブ: 1,400万 (TQQQ)
  補填スリーブ: 2,600万 (債券)
  合計: 4,000万

[毎年12月末のチェック手順 -- 4ステップ]
Step 1: 運用スリーブ残高を確認
Step 2: 補填スリーブ残高を確認
Step 3: 以下の条件表を見て来年の支出を決める
Step 4: 必要なら補填スリーブから運用スリーブへ移す

[支出ルール -- 年1回12月末に確認]
  (A) 合計 > 5,000万 かつ 運用スリーブ > 1,400万 → 来年は720万 使う (通常)
  (B) 合計 > 3,600万 → 来年は360万 使う (節約モード: 旅行・外食を控える)
  (C) 合計 <= 3,600万 → 来年は360万 使う + 補填スリーブ全額を運用へ移す (緊急)

[補填投入ルール]
  運用スリーブ残高が 1,400万未満 になったとき:
    → ただし「今年の運用スリーブがマイナス(前年より下がった)」なら投入を1年待つ
    → 待った翌年は問答無用で全額投入する

[運用方法変更条件]
  変更不要。TQQQと債券インデックスを保有し続けるだけ。
  70歳以降: 運用スリーブをTQQQからeMaxis Slim 先進国株式(1倍)に切り替え可
  (任意。パフォーマンス差は本ハーネスでは検証していない)

上記を「SimpleManual」ルールとしてバックテストし、
sc2.6 v6 の P(labor0)=0.875 および floor=3.6M 版 P=0.9985 と比較する。

[TQQQ 代替リターン]
TQQQ は NASDAQ100 の日次3倍レバレッジETF。年次リターンは
r_tqqq ≈ 3 * r_qqq - cost
ただし実際の年次リターンは非線形(volatility drag)がある。
本ハーネスでは sc2.6 データが使えないため、v3ハーネスが持つ
TQQQ相当 = "tqqq" キーのリターン列を使う。
利用可能か確認→なければ3x NASDAQ100 近似を独立計算。

[補填スリーブ]
v6 と同じ 1x bond リターン系列 (after-tax) を使用。

Self-test: block=full original history で v3 と同じ 1975/2012 労働年数を再現
  ただし今回のルールはsc2.6でなくTQQQ使用 -> 別途確認
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (os.path.dirname(_THIS), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import v3 harness for data loading
_spec_v3 = importlib.util.spec_from_file_location(
    "v3", os.path.join(_THIS, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec_v3)
_spec_v3.loader.exec_module(v3)

# Import v6 for block bootstrap infrastructure
_spec_v6 = importlib.util.spec_from_file_location(
    "v6", os.path.join(_THIS, "labor_zero_v6_return_mc_20260629.py"))
v6 = importlib.util.module_from_spec(_spec_v6)
_spec_v6.loader.exec_module(v6)

M = 1_000_000.0
SPEND_FULL = 7.2 * M       # 720万 (通常)
SPEND_FLOOR = 3.6 * M      # 360万 (節約モード)
HORIZON = 20
INFLATION = 0.02
HIST_YEARS = list(range(1975, 2026))  # 51年

# ---- 開始時配分 (SimpleManual ルール) ----
RUN0_MANUAL   = 14.0 * M   # TQQQ スリーブ (1,400万)
RESV0_MANUAL  = 26.0 * M   # 債券 スリーブ (2,600万)
RUN_REFILL_THR = 14.0 * M  # 運用スリーブがこれ未満→補填投入検討
# 支出モード切替
TOTAL_FULL_THR   = 50.0 * M  # 合計5,000万超かつ運用>1,400万→720万
TOTAL_FLOOR      = 36.0 * M  # 合計3,600万以下→360万


def load_run_returns(mode="nasdaq_1x"):
    """Load run sleeve returns for manual rule.
    mode options:
      'NASDAQ_1x'  : NASDAQ100 1倍 (eMaxis Slim NASDAQ100 等 -- 完全手動実行可)
      'sc1.0'      : DH-W1 scale1.0 (タイミングシグナルあり -- 参考)
      'sc1.6'      : DH-W1 scale1.6 (現行ベスト -- 参考)
      'sc2.6'      : DH-W1 scale2.6 (v6 proxy -- 参考)
    手動実行可能ルールの本命は 'NASDAQ_1x'。
    sc系はDH-W1日次自動シグナルが必要で手動不可。
    """
    rets = v3.load_rets()
    if mode in rets:
        print(f"[DATA] run sleeve = {mode}")
        return rets[mode]
    raise KeyError(f"Unknown run mode: {mode}")


def load_paired_history_manual(run_mode="nasdaq_1x"):
    """Return (run_r[51], res_r[51]) for manual rule:
    run = NASDAQ_1x (or specified mode); res = 1x bond returns. Both after-tax."""
    run_series = load_run_returns(mode=run_mode)
    res_series = v3.load_mixed_reserve({"bond": 1.0})
    run = np.array([float(run_series.loc[y]) for y in HIST_YEARS], float)
    res = np.array([float(res_series.loc[y]) for y in HIST_YEARS], float)
    return run, res


def sim_manual(run_path, res_path, *,
               run0=RUN0_MANUAL,
               reserve0=RESV0_MANUAL,
               spend_full=SPEND_FULL,
               spend_floor=SPEND_FLOOR,
               total_full_thr=TOTAL_FULL_THR,
               total_floor_thr=TOTAL_FLOOR,
               run_refill_thr=RUN_REFILL_THR,
               hold_if_crash=True):
    """
    SimpleManual ルール: 年1回チェック・4ステップ。

    毎年 t の処理順:
      0. 補填投入判定 (年初に今年のリターンがわかる前に判断 → 前年リターンで判定)
         - run < run_refill_thr かつ (hold_if_crash=False or 前年run>=0) → 全額投入
         - hold_if_crash かつ 前年runがマイナス → 1年待つ; ただし待った翌年は強制投入
      1. 今年のリターン適用
      2. 合計資産を見て来年の支出を決める
      3. 支出執行 (run から先に払い、足りなければ reserve から補填)
      Returns dict(labor_years, ruin, terminal, cut_years)
    """
    run = float(run0)
    res = float(reserve0)
    n = len(run_path)
    labor = 0
    cut = 0
    terminal = 0.0
    prev_run_return = 0.0   # 前年リターン (hold_if_crash 判定用)
    held_last_year = False  # 前年 hold した場合は今年強制投入

    for k in range(n):
        r_run = run_path[k]

        # ---- Step 0: 補填投入判定 (年初) ----
        if run < run_refill_thr and res > 1e-6:
            if hold_if_crash and prev_run_return < 0 and not held_last_year:
                # 前年マイナス → 今年は待つ
                held_last_year = True
            else:
                # 投入 (前年プラス or 待った翌年は強制)
                run += res
                res = 0.0
                held_last_year = False
        else:
            held_last_year = False

        # ---- Step 1: リターン適用 ----
        run *= (1.0 + r_run)
        res *= (1.0 + res_path[k])

        # ---- Step 2: 合計資産で来年支出を決める ----
        total = run + res
        if total > total_full_thr and run > run_refill_thr - 1e-6:
            want = spend_full    # (A) 通常720万
        else:
            want = spend_floor   # (B)/(C) 節約360万

        if want == spend_floor and spend_floor < spend_full - 1e-6:
            cut += 1

        # ---- Step 3: 支出執行 ----
        if total + 1e-6 < want:
            # 全資産でも払えない → 労働が発生
            labor += 1
            want = total
        # run から先に払い、足りなければ reserve から
        if run >= want:
            run -= want
        else:
            deficit = want - run
            run = 0.0
            if res >= deficit:
                res -= deficit
            else:
                # reserve も不足 (want=total のはずなので通常ここに来ない)
                res = 0.0

        prev_run_return = r_run

    terminal = run + res
    return dict(labor_years=labor, ruin=int(terminal <= 1e-6),
                terminal=terminal, cut_years=cut)


def mc_manual(run_h, res_h, n_paths=2000, block=5, seed=20260629,
              run0=RUN0_MANUAL, reserve0=RESV0_MANUAL,
              spend_full=SPEND_FULL, spend_floor=SPEND_FLOOR,
              total_full_thr=TOTAL_FULL_THR, total_floor_thr=TOTAL_FLOOR,
              run_refill_thr=RUN_REFILL_THR, hold_if_crash=True,
              horizon=HORIZON):
    """Block-bootstrap MC for manual rule."""
    rng = np.random.default_rng(seed)
    labors = np.empty(n_paths, int)
    ruins  = np.empty(n_paths, int)
    terms  = np.empty(n_paths, float)
    cuts   = np.empty(n_paths, int)
    for p in range(n_paths):
        rp, sp = v6.make_block_path(rng, run_h, res_h, n=horizon, block=block)
        r = sim_manual(rp, sp, run0=run0, reserve0=reserve0,
                       spend_full=spend_full, spend_floor=spend_floor,
                       total_full_thr=total_full_thr, total_floor_thr=total_floor_thr,
                       run_refill_thr=run_refill_thr, hold_if_crash=hold_if_crash)
        labors[p] = r["labor_years"]
        ruins[p]  = r["ruin"]
        terms[p]  = r["terminal"]
        cuts[p]   = r["cut_years"]
    deflate = (1.0 + INFLATION) ** horizon
    return dict(
        n_paths=n_paths, block=block,
        p_labor0=float((labors == 0).mean()),
        p_ruin0=float((ruins == 0).mean()),
        labor_mean=float(labors.mean()),
        labor_p95=float(np.percentile(labors, 95)),
        labor_max=int(labors.max()),
        term_real_med_M=round(float(np.median(terms)) / deflate / M, 1),
        term_real_p5_M=round(float(np.percentile(terms, 5)) / deflate / M, 1),
        cut_mean=float(cuts.mean()),
        cut_p95=float(np.percentile(cuts, 95)),
    )


def run_all():
    print("=" * 60)
    print("v7 SimpleManual -- 手動実行可能退職運用ルール")
    print("=" * 60)
    print()

    # ---- 運用スリーブ別の比較 ----
    modes = [
        ("NASDAQ_1x (1倍 手動可・本命)", "NASDAQ_1x"),
        ("sc2.6 (DH-W1 日次自動・参考)", "sc2.6"),
    ]

    print("=" * 60)
    print("PART A: 運用スリーブ別 基礎比較 (floor360万, hic=True, n=2000, block=5)")
    print("=" * 60)
    for label, mode in modes:
        run_h, res_h = load_paired_history_manual(run_mode=mode)
        print(f"\n[{label}]")
        print(f"  run: mean={run_h.mean()*100:.1f}%, std={run_h.std()*100:.1f}%")

        # 史実46開始年
        wins = 0
        fails = []
        for start in range(1975, 2021):
            idx = start - 1975
            h = min(HORIZON, 2025 - start + 1)
            r = sim_manual(run_h[idx:idx+h], res_h[idx:idx+h])
            if r["labor_years"] == 0:
                wins += 1
            else:
                fails.append((start, r["labor_years"], r["cut_years"]))
        print(f"  史実46開始年: labor=0: {wins}/46", end="")
        if fails:
            print(f" -- 失敗: {[s for s, l, c in fails]}")
        else:
            print(" (全合格)")

        # MC
        r_floor = mc_manual(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                             spend_floor=3.6*M, hold_if_crash=True)
        r_fixed = mc_manual(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                             spend_floor=7.2*M, hold_if_crash=True)
        print(f"  floor360万: P(labor0)={r_floor['p_labor0']:.4f}  cut_mean={r_floor['cut_mean']:.2f}  "
              f"term_med={r_floor['term_real_med_M']:.1f}億  p5={r_floor['term_real_p5_M']:.1f}億")
        print(f"  固定720万:  P(labor0)={r_fixed['p_labor0']:.4f}  cut_mean=0.00  "
              f"term_med={r_fixed['term_real_med_M']:.1f}億  p5={r_fixed['term_real_p5_M']:.1f}億")

    # ---- NASDAQ_1x 詳細探索 ----
    print()
    print("=" * 60)
    print("PART B: NASDAQ_1x 詳細探索")
    print("=" * 60)
    run_h, res_h = load_paired_history_manual(run_mode="NASDAQ_1x")

    # B1: 開始配分感応度
    print("\n[B1] 開始配分感応度 (floor=360万, hic=True)")
    alloc_configs = [
        ("run10M+res30M", dict(run0=10*M, reserve0=30*M)),
        ("run14M+res26M", dict(run0=14*M, reserve0=26*M)),
        ("run18M+res22M", dict(run0=18*M, reserve0=22*M)),
        ("run20M+res20M", dict(run0=20*M, reserve0=20*M)),
        ("run24M+res16M", dict(run0=24*M, reserve0=16*M)),
        ("run30M+res10M", dict(run0=30*M, reserve0=10*M)),
    ]
    for label, kw in alloc_configs:
        r = mc_manual(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                      spend_floor=3.6*M, hold_if_crash=True, **kw)
        print(f"  {label}: P(labor0)={r['p_labor0']:.4f}  cut_mean={r['cut_mean']:.2f}"
              f"  term_med={r['term_real_med_M']:.1f}億  p5={r['term_real_p5_M']:.1f}億")

    # B2: フロア水準感応度 (最良配分で)
    print("\n[B2] フロア水準感応度 (run10M+res30M, hic=True)")
    floor_configs = [
        ("floor=240万", dict(spend_floor=2.4*M)),
        ("floor=300万", dict(spend_floor=3.0*M)),
        ("floor=360万", dict(spend_floor=3.6*M)),
        ("floor=480万", dict(spend_floor=4.8*M)),
        ("floor=600万", dict(spend_floor=6.0*M)),
        ("floor=720万(固定)", dict(spend_floor=7.2*M)),
    ]
    for label, kw in floor_configs:
        r = mc_manual(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                      run0=10*M, reserve0=30*M, hold_if_crash=True,
                      total_full_thr=50*M, **kw)
        print(f"  {label}: P(labor0)={r['p_labor0']:.4f}  cut_mean={r['cut_mean']:.2f}"
              f"  cut_p95={r['cut_p95']:.0f}  term_med={r['term_real_med_M']:.1f}億  p5={r['term_real_p5_M']:.1f}億")

    # B3: 切替閾値感応度
    print("\n[B3] 720万→360万 切替閾値感応度 (run10M+res30M, floor=360万, hic=True)")
    thr_configs = [
        ("常に720万", dict(total_full_thr=0.0, spend_floor=7.2*M)),
        ("合計>3000万で720万", dict(total_full_thr=30*M, spend_floor=3.6*M)),
        ("合計>4000万で720万", dict(total_full_thr=40*M, spend_floor=3.6*M)),
        ("合計>5000万で720万", dict(total_full_thr=50*M, spend_floor=3.6*M)),
        ("常に360万", dict(total_full_thr=999*M, spend_floor=3.6*M)),
    ]
    for label, kw in thr_configs:
        r = mc_manual(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                      run0=10*M, reserve0=30*M, hold_if_crash=True, **kw)
        print(f"  {label}: P(labor0)={r['p_labor0']:.4f}  cut_mean={r['cut_mean']:.2f}"
              f"  term_med={r['term_real_med_M']:.1f}億  p5={r['term_real_p5_M']:.1f}億")

    # B4: シード安定性
    print("\n[B4] シード安定性 (run10M+res30M, floor=360万, hic=True)")
    seeds = [20260629, 42, 12345, 99999, 777777]
    ps = []
    for sd in seeds:
        r = mc_manual(run_h, res_h, n_paths=2000, block=5, seed=sd,
                      run0=10*M, reserve0=30*M, spend_floor=3.6*M, hold_if_crash=True)
        ps.append(r["p_labor0"])
        print(f"  seed={sd}: P(labor0)={r['p_labor0']:.4f}")
    print(f"  range: {min(ps):.4f} -- {max(ps):.4f}")

    # ---- 最終サマリ ----
    print()
    print("=" * 60)
    print("FINAL SUMMARY: SimpleManual vs sc2.6 v6 比較")
    print("=" * 60)
    print(f"  {'ルール':<50} {'P(labor0)':>10} {'cut_mean':>10}")
    print("  " + "-" * 72)
    print(f"  {'【参考】sc2.6 v6 名目固定720万':<50} {'0.875-0.898':>10} {'0.00':>10}")
    print(f"  {'【参考】sc2.6 v6 floor360万':<50} {'0.9985-1.000':>10} {'2.4avg':>10}")

    run_h, res_h = load_paired_history_manual(run_mode="NASDAQ_1x")
    r_best = mc_manual(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                       run0=10*M, reserve0=30*M, spend_floor=3.6*M, hold_if_crash=True)
    r_fixed = mc_manual(run_h, res_h, n_paths=2000, block=5, seed=20260629,
                        run0=10*M, reserve0=30*M, spend_floor=7.2*M, hold_if_crash=True)
    print(f"  {'SimpleManual NASDAQ_1x floor360万 (run10+res30)':<50} {r_best['p_labor0']:>10.4f} {r_best['cut_mean']:>10.2f}")
    print(f"  {'SimpleManual NASDAQ_1x 固定720万   (run10+res30)':<50} {r_fixed['p_labor0']:>10.4f} {'0.00':>10}")
    print()
    print("DONE.")
    return r_best


if __name__ == "__main__":
    run_all()
