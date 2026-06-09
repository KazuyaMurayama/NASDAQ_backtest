# 2軸最適化（配分×レバレッジ）でベンチマーク超え 実装計画

作成日: 2026-06-09
最終更新日: 2026-06-09

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` または `superpowers:executing-plans`。各ステップは `- [ ]` 追跡。既存テスト済みモジュール（`leverage_eval`/`bond_signals`/`strategy_layers`/`walkforward`/`nine_metric_eval`/`report_format`）を最大限再利用（DRY）。

**Goal:** NASDAQ/GOLD/BOND/**CASH** の4資産配分 と レバレッジ比率 を**2軸で同時最適化**し、ETF環境の既存ベスト **DH-W1（min税後CAGR +18.10%, MaxDD −34.57%）を超える**戦略（min税後CAGR > +18.10% かつ 3軸保守基準でDH-W1超え）を確立する。

**Architecture:** 各資産を「確定タイミング信号 × 純コスト後・税後リターン（レバETF or 1倍投信）」のスリーブとして構築し、4-way配分（CASH含む）×レバレッジを決定変数として、min(IS,OOS)税後CAGRを目的・risk（MaxDD/Worst10Y/P10）を制約に**結合探索**。静的グリッド→動的配分（リスクパリティ/ボラターゲット/レジーム）→WFA/bootstrap確定。

**Tech Stack:** 既存 `src/multi_asset/*` + `src/integration/nine_metric_eval.py` + `src/product_costs.py`。評価は**標準10指標**（CAGR IS/OOS min・gap・Sharpe・MaxDD・Worst10Y・P10_5Y・Trades/yr・WFE・CI95_lo）。

---

## ベンチマーク（これを超える・標準10指標・税後⓽）

| 戦略 | CAGR min(IS,OOS) | gap | Sharpe | MaxDD | Worst10Y | P10_5Y | Trades/yr | WFE | CI95_lo |
|---|---|---|---|---|---|---|---|---|---|
| **DH-W1（ETF環境 現行ベスト）** | **+18.10%** | −0.81pp | 0.845 | −34.57% | +10.37% | +4.82% | 68.7 | 1.023 | +13.61% |

**採用条件（house 3軸保守基準）**: min(IS,OOS)CAGR・Worst10Y・P10_5Y を**同時に** DH-W1 超え。WFA: CI95_lo>0 ∧ 0.5≤WFE≤2.0。

## 設計の要：なぜ4資産×レバでベンチマークを超えられるか
DH-W1 は NASDAQ単独（ETF・cash退避）で +18.10% / MaxDD −34.57%。**Gold/Bond は NASDAQ と低相関**のため、(a) 同じMaxDDで NASDAQレバを増やせる→高CAGR、(b) 同じCAGRでMaxDD圧縮、のどちらかでフロンティアを上へ押す。**CASH軸**はレジーム悪化時の明示的退避でWorst10Y/P10を底上げ。2軸同時最適化でこの利得を取り切る。

## ファイル構成（新規最小）
- Modify: `src/multi_asset/leverage_eval.py` — `min_is_oos_cagr_aftertax()` と「10指標一括」`ten_metrics()` を追加（IS/OOS分割は canonical 2021-05-08）。
- Create: `src/multi_asset/portfolio_engine.py` — 4資産（N/G/B/CASH）×per-assetレバの**ポートフォリオ純コスト後・税後リターン**生成。
- Create: `src/multi_asset/optimize_2axis.py` — 配分×レバの**結合探索**（静的グリッド＋動的配分variant）。
- Create: `src/multi_asset/run_2axis.py` — 実行・10指標レポート（太字）・正典更新。
- Test: `tests/multi_asset/test_portfolio_engine.py`, `test_optimize_2axis.py`, `test_ten_metrics.py`。
- Update: `MULTIASSET_CURRENT_BEST.md`, `STRATEGY_REGISTRY.md`, `tasks.md`。

---

## Phase 0: ベンチマーク固定＋10指標評価器

### Task 0.1: DH-W1 ベンチマーク値の確定（参照のみ）
**Files:** Read `CURRENT_BEST_STRATEGY.md`（§v4.5 ETF行）
- [ ] DH-W1 の標準10指標を定数化（`BENCH_DHW1` dict）。これを全比較の baseline とする。

### Task 0.2: 10指標一括評価器（min(IS,OOS) 込み）
**Files:** Modify `src/multi_asset/leverage_eval.py`; Test `tests/multi_asset/test_ten_metrics.py`
- [ ] **Step1:** 失敗テスト：既知のNAV/日次リターンに対し `ten_metrics(daily_ret)` が `{cagr_is, cagr_oos, cagr_min, is_oos_gap, sharpe_oos, maxdd, worst10y, p10_5y, trades_yr, wfe, ci95_lo}` を返す（split=2021-05-08）。`cagr_*` は**税後**。
- [ ] **Step2:** FAIL確認。
- [ ] **Step3:** 実装：`nine_metric_eval` のprimitive＋`after_tax_cagr`＋`walkforward` を合成。`cagr_min=min(cagr_is,cagr_oos)`。Trades/yrは**実建玉ベース**（`strategy_layers.trades_per_year`）。
- [ ] **Step4:** PASS確認。 **Step5:** commit。

---

## Phase 1: 4資産ポートフォリオ・エンジン（CASH＋per-assetレバ）

### Task 1.1: ポートフォリオ純コスト後・税後リターン生成
**Files:** Create `src/multi_asset/portfolio_engine.py`; Test `tests/multi_asset/test_portfolio_engine.py`
- [ ] **Step1:** 失敗テスト：`portfolio_returns(sleeves, weights)` が `Σ w_i·sleeve_i + w_cash·cash` を返し、`weights` は {N,G,B,CASH} で和=1。CASH脚は T-bill(DTB3)。
- [ ] **Step2:** FAIL確認。
- [ ] **Step3:** 実装：各 sleeve は `leverage_eval.strategy_net_returns(asset_ret, signal_pos, cash, sofr, k_i, product_i, exec_lag_i)`（N=TQQQ/k≤3, G=2036 or 1倍投信/k≤2, B=TMF/k≤3）。weights は静的 or 時変DataFrame。月次リバランス。
- [ ] **Step4:** PASS確認。 **Step5:** commit。

### Task 1.2: 確定シグナルの結線（NASDAQは要競争力）
**Files:** Modify `portfolio_engine.py`; reuse `bond_signals`/`strategy_layers`
- [ ] Gold=`m252_tv0.10_z0.75_mo`, Bond=`m252_tv0.05_z1.0_wk` を流用。
- [ ] NASDAQ は DH-W1 同等以上を担保するため、**mom252×VT×VZ に加えレジーム/ヒステリシス**を含む強化スリーブを構築し、**単独10指標が DH-W1 に劣後しないこと**を確認（劣後なら信号改良）。

---

## Phase 2: 2軸 結合最適化（配分×レバ）

### Task 2.1: 静的グリッド探索
**Files:** Create `src/multi_asset/optimize_2axis.py`; Test `tests/multi_asset/test_optimize_2axis.py`
- [ ] **Step1:** 失敗テスト：`grid_search(sleeves, w_grid, k_grid)` が各 (配分, レバ) の `ten_metrics` を返し、`cagr_min` 降順で並ぶこと。
- [ ] **Step2:** FAIL確認。
- [ ] **Step3:** 実装：配分グリッド（例 w_N∈{0..0.7}, w_G∈{0..0.4}, w_B∈{0..0.4}, w_C=残り, 0.1刻み・和=1）× レバグリッド（k_N∈{1,2,3}, k_G∈{1,2}, k_B∈{1}）を総当たり。各点 `ten_metrics`。
- [ ] **Step4:** PASS確認。 **Step5:** commit。

### Task 2.2: DH-W1 超えフィルタ＋フロンティア抽出
**Files:** Modify `optimize_2axis.py`
- [ ] `BENCH_DHW1` に対し**3軸（min CAGR・Worst10Y・P10）同時超え**の候補のみ抽出。
- [ ] min税後CAGR最大・MaxDD最小のパレートフロンティアを返す。

### Task 2.3: 動的配分 variant（静的を超える上積み）
**Files:** Modify `optimize_2axis.py`; reuse `allocator`
- [ ] リスクパリティ／ボラターゲット（ポート vol 目標→実効レバ）／レジーム条件（risk-off時 CASH↑）の3 variant を評価し、静的最良と比較。

---

## Phase 3: WFA/bootstrap 確定

### Task 3.1: 上位候補の正式検証
**Files:** reuse `walkforward`; Create `outputs`相当 レポート断片
- [ ] フロンティア上位を `wfa_stats`＋`paired_block_bootstrap`（vs DH-W1相当 baseline）。CI95_lo>0 ∧ WFE∈[0.5,2.0] ∧ P(>bench)>0.90 を満たすものを finalist。

---

## Phase 4: 10指標レポート＋正典化

### Task 4.1: 標準10指標レポート（太字）
**Files:** Create `src/multi_asset/run_2axis.py`; 出力 `MULTIASSET_2AXIS_RESULT_20260609.md`
- [ ] `report_format.fmt_metric_table` で**標準10指標**表を生成（**各列最良値＋最良戦略＋ベンチ行を太字**、表構造は自己検証）。
- [ ] DH-W1 ベンチ行を必ず併記し、超過分（Δ）を明示。

### Task 4.2: 正典・台帳更新
**Files:** Update `MULTIASSET_CURRENT_BEST.md`, `STRATEGY_REGISTRY.md`, `tasks.md`
- [ ] ベンチ超え最良構成（配分×レバ）を Active 候補として登録。全成果物リンク報告。

---

## 報告形式（ユーザー指定・厳守）
1. **標準10指標**（CAGR min(IS,OOS)・gap・Sharpe・MaxDD・Worst10Y・P10_5Y・Trades/yr・WFE・CI95_lo）。全て税後⓽。
2. **表形式**。
3. **優れた値・優れた戦略を太字**（`report_format` の best-per-column＋recommended、ベンチ行併記）。

## 自己レビュー（計画→spec整合）
- ✅ 2軸（配分N/G/B/**CASH** × レバ）を Task 2.1 で明示的に探索。
- ✅ ベンチDH-W1（+18.10%）超えを Task 2.2 のフィルタで強制。
- ✅ 標準10指標＋表＋太字を Phase 4 で担保。
- ✅ 既存テスト済みモジュール再利用（engine/optimizeのみ新規）。
- ⚠ NASDAQスリーブの競争力（Task 1.2）が未達なら信号改良が追加で要る（リスク）。
