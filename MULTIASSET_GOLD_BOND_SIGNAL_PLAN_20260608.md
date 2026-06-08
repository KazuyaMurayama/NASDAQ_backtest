# マルチアセット（NASDAQ / Gold / Bond）タイミングシグナル検証 実装計画

作成日: 2026-06-08
最終更新日: 2026-06-08

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`（推奨）または `superpowers:executing-plans` でタスク単位に実行。各ステップは `- [ ]` で追跡。**大原則は既存基盤の再利用**。新規モジュールは原則作らず `src/signals/` `src/integration/` を拡張する。

**Goal:** NASDAQで確立した「9指標標準＋WFA＋台帳」による体系的シグナル探索パイプライン（P1→P5→採否→WFA→integration）を **Gold / Bond にもスタンドアロンで**適用し、3資産の保有/キャッシュ判断・配分ロジック・レバレッジ採否・運用商品を、根拠あるロジックとして決定できる状態にする。

**Architecture:** 既存の単資産シグナルパイプラインを資産パラメータ化して Gold/Bond に流用 →（①シグナル層）。3資産NAVを統合する配分層を `src/multi_asset/` に新設 →（②配分層）。`product_costs.py` を拡張しレバレッジ純便益を判定 →（③レバレッジ層）。deep-research の調査＋`g12/g13` コスト基盤で商品をマッピング →（④商品層）。

**Tech Stack:** 既存 Python backtest 基盤、`src/signals/*`、`src/integration/nine_metric_eval.py`・`phase_d_{metrics,bootstrap,wfa}.py`、`src/_sweep_format.py`、`src/product_costs.py`、`tests/signals/*`。

---

## 現況：既に在るもの／真の空白（2026-06-08 確認）

### データ（**取得フェーズ不要**。長期データ完備）
- **NASDAQ:** `NASDAQ_extended.csv`（1974〜）
- **Gold:** `data/lbma_gold_daily.csv`（**1968〜**）、`data/dgp_daily.csv`（2x Gold）
- **Bond:** `data/dgs10_daily.csv`（**1962〜**）, `data/dgs30_daily.csv`（1977〜）, `data/dgs2_daily.csv`, `data/ief_daily.csv`, `data/tlt_daily.csv`
- **派生マクロ（FRED, 取得済み）:** BAA-AAAクレジットスプレッド, 30y-10yタームプレミアム, DFF変化/DFF-10y, CPI YoYサプライズ, DGP(2x金)・DRN(3xREIT)対数リターン

### 既存のGold/Bond資産（**再利用・出発点**。ただし用途が限定的）
| 既存 | 性格 | 限界（＝今回埋める空白） |
|---|---|---|
| `e2_bond_sweep.py` / `E2_BOND_SWEEP` / `bond_variant_sweep.py` | Bond配分・変種スイープ | NASDAQ戦略の**スリーブ**として。スタンドアロンのhold/cashシグナル探索ではない |
| `f5_bond_regime.py` / `F5_BOND_REGIME` | Bondレジーム | 単体9指標＋WFAの体系評価が未整備 |
| `g5_real_yield.py` | 実質金利シグナル | Goldにも効くが横断適用が未整理 |
| `h5_gold_dyn.py` / `b9_s2lt2_goldfrac_sweep.py` | 動的Gold配分 | NASDAQ内の**金フラクション**最適化。Gold単独timingではない |
| `research_gold_bond_timing.py` / `research_step2_bond_timing.py` | 探索初期 | アドホック。P1→P5標準パイプライン未通過 |
| `a3_regime_asset_tilt.py` / `f8_regime_tilt.py` / `e4_regime_klt.py` | レジーム傾斜 | 配分傾斜であり、資産別hold/cash判断とは別物 |

### 真の空白（本計画のスコープ）
1. **Gold/Bond それぞれの「保有 vs キャッシュ」を、NASDAQと同じ体系的シグナル探索（screening→stability→SPA→採否→WFA）で確立する**
2. **3資産を統合する配分ロジック**（リターン/シャープ最適化）が未整備
3. **レバレッジ採否を純コスト後で判定する横断ロジック**が資産別に未確立
4. **シグナルの回転率に整合した運用商品選定（ETF/CFD/1倍投信）**の枠組みが未整備

---

## 前提・既定値（私の判断で確定。違う場合のみ指摘ください）

| 論点 | 既定値（推奨） | 理由 |
|---|---|---|
| Bondの代理系列 | **長期：DGS10/DGS30利回り→価格換算（1962/1977〜）、実運用整合：IEF/TLT** | 長期検証はデュレーション×利回り変化、商品はETF |
| Goldの基準系列 | **LBMA USD（1968〜）、商品整合：DGP/GLD** | 長期データ |
| キャッシュ基準 | 短期金利（DFF/DTB3） | 「持つ/キャッシュ」の対照 |
| 評価指標 | NASDAQ同一の **9指標標準**（`nine_metric_eval.py`） | リポ整合・横比較 |
| 検証順序 | **シグナル→配分→レバレッジ→商品（直列）** | 後段は配分確定後でないと判断不能 |
| WFAゲート | NASDAQ同一（CI95_lo / WFE、`phase_d_wfa`） | 過学習・レジーム運検出 |

---

## ファイル構成（新規は最小限）

- Create: `config/assets.yaml` — 資産定義（系列・ベンチ・コスト参照キー）
- Create: `src/multi_asset/single_asset_sweep.py` — 資産パラメータ化スイープ・ランナー
- Create: `src/multi_asset/allocator.py` — 配分ロジック群
- Create: `src/multi_asset/leverage_eval.py` — 純コスト後レバレッジ採否判定
- Create: `src/multi_asset/product_selector.py` — 商品マッピング・比較表
- Create: `tests/multi_asset/test_allocator.py`, `test_leverage_eval.py`
- Modify: `src/product_costs.py` — Bond(TMF/TLT/IEF/CFD/1倍投信)・Gold(DGP/GLD/CFD/1倍投信) を追記
- Update: `STRATEGY_REGISTRY.md`（各検証1件）、`tasks.md`、`FILE_INDEX.md`、新規 `MULTIASSET_CURRENT_BEST.md`

---

## Phase 1: シグナル棚卸しと体系化

### Task 1.1: シグナルタクソノミー（資産横断）
**Files:** Create `docs/multiasset/SIGNAL_TAXONOMY.md`
- [ ] 既存族（トレンド/モメンタム、ボラ・レジームVZ、長期トレンドLT、E4レジーム）の Gold/Bond 適用可否を整理
- [ ] 資産固有族と仮説を1行記述（戦略検証プロトコルStep3）
  - **Bond:** イールドカーブ(2s10s/30s10s)、実質金利、タームプレミアム(30y-10y, 取得済)、クレジットスプレッド(BAA-AAA, 取得済)、Fed funds レジーム(DFF, 取得済)、デュレーション・モメンタム、MOVEボラ
  - **Gold:** 実質金利(`g5_real_yield`再利用)、DXY、モメンタム、インフレ(CPIサプライズ, 取得済)、対株式相関レジーム
- [ ] 既存 `e2_bond_sweep`/`f5_bond_regime`/`h5_gold_dyn` の知見を出発点として明記

### Task 1.2: シグナル関数の資産パラメータ化
**Files:** Modify `src/signals/timing.py` ほか（必要時）; Test 追加
- [ ] 価格/マクロ系列を引数化し Gold/Bond で再利用可能に（NASDAQ既存出力の回帰テストで非破壊を担保）

---

## Phase 2: 単資産シグナル検証（保有 vs キャッシュ）

### Task 2.1: 単資産スイープ・ランナー
**Files:** Create `src/multi_asset/single_asset_sweep.py`（ヘッダは `_sweep_format.py`）
- [ ] 資産×シグナル族×パラメータ総当たりを `nine_metric_eval` で評価（CAGR_OOS中心、Trades/yr・Overfit(WFE)列自動算出）
- [ ] ベースライン（B&H・常時キャッシュ）を必ず併記

### Task 2.2: Bond 単独検証 → 候補抽出
**Files:** `outputs/<DATE>_bond-single-asset-sweep.md`
- [ ] スイープ→Shortlist化、3軸ベースライン（min(IS,OOS)・Worst10Y・P10_5Y）超過を採用条件、棄却理由を台帳へ
- [ ] 既存 `e2_bond_sweep`/`f5_bond_regime` 結果との整合・差分を確認

### Task 2.3: Gold 単独検証 → 候補抽出
**Files:** `outputs/<DATE>_gold-single-asset-sweep.md`
- [ ] 同上。実質金利・DXY・インフレ寄与を分離評価、WFE高はレジーム運フラグ
- [ ] 既存 `h5_gold_dyn`/`research_gold_bond_timing` 結果との整合確認

### Task 2.4: 単資産WFAロバスト検証
**Files:** `src/integration/phase_d_wfa.py`・`phase_d_bootstrap.py` 流用
- [ ] Shortlist候補に Walk-Forward 実行、CI95_lo / WFE ゲートで合否。PASS のみ Phase 3 へ

---

## Phase 3: ポートフォリオ配分ロジック（要件①）

### Task 3.1: 配分ロジック群（TDD）
**Files:** Create `src/multi_asset/allocator.py`; Test `tests/multi_asset/test_allocator.py`
- [ ] 「アクティブ集合（シグナルON資産）→ウェイト」I/F
- [ ] 候補ロジック実装: ①等加重 ②インバースボラ/リスクパリティ ③シグナル強度加重 ④平均分散・最大シャープ（共分散シュリンク） ⑤ボラターゲット・オーバーレイ（グロス調整＝レバレッジ層への橋渡し）
- [ ] 既知共分散→期待ウェイトの単体テスト
- [ ] 既存 `f1_alloc_sweep`/`a3_regime_asset_tilt` の知見を反映

### Task 3.2: 配分バックテスト・比較レポート
**Files:** `outputs/<DATE>_multiasset-allocation-comparison.md`（統合レポート標準 §0'/§1'/§5/§6）
- [ ] 配分ロジック別に統合9指標比較、分散効果（NASDAQ-Gold-Bondの相関・寄与分解）提示、最良リターン/シャープ配分を選定、ベンチ併記、台帳登録

---

## Phase 4: レバレッジ採否判定（要件②）

### Task 4.1: コストモデル拡張
**Files:** Modify `src/product_costs.py`
- [ ] `ProductCost` に Bond（TMF=3x米国債／TLT・IEF=1x／Bond-CFD／1倍投信）、Gold（DGP=2x／GLD=1x／Gold-CFD／1倍投信）、税区分差を追記
- [ ] deep-research（gold-bond-3x / sbi-gold-bond-high-leverage 等）＋既存 `g12/g13` を数値根拠に

### Task 4.2: レバレッジ純便益エンジン（TDD）
**Files:** Create `src/multi_asset/leverage_eval.py`; Test `tests/multi_asset/test_leverage_eval.py`
- [ ] 判定基準: **純コスト後シャープ＞無レバ** ∧ **MaxDD/Worst10Y許容内** ∧ **ボラ減衰織込み後も期待リターン増＞コスト増**
- [ ] 倍率グリッド（1x/2x/3x…）で資産別・配分後の双方を評価
- [ ] 「採用すべきか」Yes/No＋根拠指標を出力（既存 `leverage_bin_analysis`/`dyn_lev_backtest` の知見を反映）

### Task 4.3: レバレッジ判定レポート
**Files:** `outputs/<DATE>_leverage-decision-report.md`
- [ ] 資産別・配分後の採否結論、台帳登録

---

## Phase 5: 運用商品の選定（要件③）

### Task 5.1: 商品候補マトリクス
**Files:** Create `src/multi_asset/product_selector.py`; `outputs/<DATE>_product-selection-matrix.md`
- [ ] 3系統で整理: (a) レバレッジETF（NASDAQ=TQQQ／Bond=TMF／Gold=DGP・2036等） (b) CFD (c) 1倍投信
- [ ] 選定軸: コスト総額・取得可能レバレッジ・税区分（特定口座/雑所得/投信）・トラッキング・流動性・最低資金・リバランス摩擦・**シグナルの保有期間/Trades-yr整合**

### Task 5.2: シグナル×商品の最適マッチング
- [ ] 高回転→ETF/CFD、低回転・1倍→投信、を Phase2 の Trades/yr と突合し各資産の推奨商品＋次善案を確定、台帳登録

---

## Phase 6: 統合・最終判定・正典化

### Task 6.1: 統合検証レポート
**Files:** `outputs/<DATE>_multiasset-integrated-report.md`（§0'/§1'/§5/§6 厳守、`_sweep_format.py`）
- [ ] シグナル→配分→レバレッジ→商品の最終ロジックを一気通貫提示、全候補WFA PASS確認、最新スクリプト出力由来を確認

### Task 6.2: 正典化・台帳整合
**Files:** Create `MULTIASSET_CURRENT_BEST.md`; Update `STRATEGY_REGISTRY.md`, `tasks.md`, `FILE_INDEX.md`
- [ ] マルチアセット版「単一の真実」を確立、全成果物をリンク報告

---

## 改訂履歴・設計判断

- **2026-06-08:** 当初「ポンド(GBP)」と誤解して作成した `MULTIASSET_GOLD_GBP_SIGNAL_PLAN_20260607.md` を破棄し、本「Gold/Bond」版へ全面改訂。GBP用に追加したローダー/データ取得コード・テストも撤去済み。
- **データ取得フェーズを削除:** Gold(1968〜)・Bond(1962/1977〜)は長期データ完備のため不要（GBP版に在った Phase 0 取得タスクは廃止）。
- **既存Gold/Bond資産への接地:** `e2_bond_sweep`/`f5_bond_regime`/`g5_real_yield`/`h5_gold_dyn`/`research_gold_bond_timing` を出発点に、不足する「スタンドアロン体系シグナル探索＋3資産統合の配分/レバレッジ/商品」のみを新規スコープとした（DRY）。
- **検証順序の直列依存:** 配分確定前にレバレッジ/商品は判断不能という依存を前提に固定。
- **ボラ減衰/パス依存をレバレッジ判定基準に明記:** 「期待リターン＞コスト」だけでは日次リバランス型ETFの減衰を見落とすため3条件に厳格化。
