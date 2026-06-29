# 労働補填ゼロ v4【インフレ織り込み】Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** v3 Round-G の推奨案（sc2.6・運用比レバー・hold-if-crash・保証フロア＋好機トップ）を、日本のインフレ実態に合わせて織り込み、インフレ下でも「保証される実質支出」が最大の運用設計を数ラウンドで探索する。

**Architecture:** v3 ハーネスを土台に、支出（フロアと満額目標の両方）を毎年 (1+g)^k で実質維持するインフレ・ポートを作る。g は日本実態シナリオ（0/1/2/3/3.5%）。レバーを数ラウンド（H1 bond baseline → H2 gold-blend → H3 run-fraction → H4 top_wr glide → H5 retirement-smile）で試し、各 g の最大保証実質フロアを二分探索で確定。binding 開始年を同定して弱点を分析。

**Tech Stack:** Python, pandas/numpy, 既存 v3 harness（labor_zero_v3_harness_20260628.py）。税後×0.8273・暦年リターン。

---

## Task 1: インフレ・ポート（v4 sim）+ self-test
**Files:** Create `src/audit/labor_zero_v4_inflation_20260629.py`
- [x] Step 1: `sim_v4`（floor と full_nom を (1+g)^k で indexed・realised spend を deflate して実質計測）
- [x] Step 2: self-test = g=0 で v3 Round-G の保証フロア6.0M（sc2.6/run0.50/top0.20）を再現。再現失敗で HALT
- [x] Step 3: Round H1（bond reserve・run-fraction・hold-if-crash）で g×runfrac×floor×top をスイープ→各 g の frontier

## Task 2: 反インフレ・レバー（H2/H3/H4）
**Files:** Modify `src/audit/labor_zero_v4_inflation_20260629.py`
- [x] Step 1: `_agg_mix`（reserve を bond/gold ブレンドに・top_wr を後年グライド可能に）
- [x] Step 2: Round H234 = mix(bond100/b80g20/b60g40/b50g50) × runfrac × floor × top × glide
- [x] Step 3: 各 g で「最良保証実質フロア vs bond-only」を比較出力

## Task 3: 正確な実質フロア天井（二分探索）+ binding 同定
**Files:** Create `src/audit/labor_zero_v4_floor_bisect_20260629.py`
- [x] Step 1: `max_floor` を2定義で（survive=labor0/ruin0、deliver=フロア毎年honored）
- [x] Step 2: g×(runfrac,top_wr) で最大フロアを二分探索・binding 開始年を同定

## Task 4: 弱点直撃レバー（H5 retirement-smile）
**Files:** Create `src/audit/labor_zero_v4_round_h5_20260629.py`
- [x] Step 1: フロアが後年 d/yr で逓減する smile を実装・active年（k<10）の保証実質フロアを最大化
- [x] Step 2: glide∈{0,-0.5,-1.0,-1.5%/yr} で各 g の active-floor を比較

## Task 5: レポート + QC + commit
**Files:** Create `LABOR_ZERO_V4_INFLATION_20260629.md`; update memory
- [ ] Step 1: 結論（保証実質フロア: 0%=6.27M / 2%=5.84M / 3%=5.54M）・弱点分析・5ラウンド表・QC
- [ ] Step 2: 成果物を main に commit・URL を 200 検証・3列表で報告
