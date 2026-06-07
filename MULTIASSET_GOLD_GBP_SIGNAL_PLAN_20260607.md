# マルチアセット（NASDAQ / Gold / GBP）タイミングシグナル検証 実装計画

作成日: 2026-06-07
最終更新日: 2026-06-07

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`（推奨）または `superpowers:executing-plans` でタスク単位に実行。各ステップは `- [ ]` で追跡。本計画は **既存基盤の再利用** が大原則。新規モジュールは原則作らず `src/signals/` `src/integration/` `src/data_loaders/` を拡張する。

**Goal:** NASDAQで確立した「9指標標準＋WFA＋台帳」検証フレーム（P1→P5→G→integration）を Gold/GBP に横展開し、3資産の保有/キャッシュ判断・配分ロジック・レバレッジ採否・運用商品を、根拠あるロジックとして決定できる状態にする。

**Architecture:** 既存の単資産パイプラインを資産パラメータ化して Gold/GBP に流用 →（①シグナル層）。3資産NAVを統合する配分層を `src/multi_asset/` に新設 →（②配分層）。`product_costs.py` を拡張しレバレッジ純便益を判定 →（③レバレッジ層）。deep-researchリポの調査4本＋`g12/g13` コスト基盤で商品をマッピング →（④商品層）。

**Tech Stack:** 既存 Python backtest 基盤（pandas/numpy/yfinance）、`src/signals/*`、`src/integration/nine_metric_eval.py`・`phase_d_{metrics,bootstrap,wfa}.py`、`src/_sweep_format.py`、`src/product_costs.py`、`tests/signals/*`。

---

## 既存資産の棚卸し（再利用先・このまま使う）

| 機能 | 既存モジュール | マルチアセットでの扱い |
|---|---|---|
| シグナル生成（タイミング/IC/分位/合成/組合せ/安定性/SPA/採否/オーバーレイ） | `src/signals/{timing,ic,quantize,composite,combinations,stability,spa_test,adoption,overlay,screening,standalone,multiplicity,forward_returns,hit_rate}.py` | **資産非依存。価格系列を差し替えて流用** |
| 9指標評価 | `src/integration/nine_metric_eval.py` | そのまま流用（NAV入力） |
| 正式WFA/ブートストラップ/メトリクス | `src/integration/phase_d_{wfa,bootstrap,metrics}.py` | そのまま流用 |
| データローダ | `src/data_loaders/signals/{yahoo,fred,cboe,cftc,manual}.py` | **GBP/Gold系ティッカーを辞書追加して拡張** |
| コスト単一の真実 | `src/product_costs.py`（`ProductCost` dataclass） | **GBP-CFD/Gold商品/1倍投信を追記** |
| レポート標準フォーマッタ | `src/_sweep_format.py` | そのまま流用（手書きヘッダ禁止） |
| CFDコスト比較基盤 | `src/g12_sbi_cfd_cost_comparison.py`, `src/g13_realistic_cost_full_comparison.py` | 商品選定の数値根拠に流用 |

## データ現況（確認済み 2026-06-07）

- **Gold: 有り** — `data/lbma_gold_daily.csv`, `data/dgp_daily.csv`（DGP=2x Gold）、Yahoo `GC=F`。既存 `src/h5_gold_dyn.py`・`src/research_gold_bond_timing.py` あり（※NASDAQポートフォリオの一部としてのみ。**単独タイミング体系検証は未実施**）。
- **GBP: 無し（NONE FOUND）** ← 本計画最大のギャップ。取得が Phase 0 の必須前提。
- マクロ素地: `src/fetch_fred_data.py`・`src/data_loaders/signals/fred.py` で金利差/実質金利/DXY取得可。

---

## 前提・既定値（私の判断で確定。違う場合のみ指摘ください）

| 論点 | 既定値（推奨） | 理由 |
|---|---|---|
| GBP通貨ペア | **GBP/JPY 主・GBP/USD 従** | JPY建て投資家。キャリー（英日金利差）が効く |
| Gold基準系列 | **XAU/USD（`GC=F`/LBMA）＋円換算系列** | 長期データ・国際比較。円ヘッジ有無を検証軸に |
| キャッシュ基準 | **円キャッシュ（短期金利）** | 「持つ/キャッシュ」の対照 |
| 評価指標 | NASDAQ同一の **9指標標準**（`nine_metric_eval.py`） | リポ整合・横比較 |
| 検証順序 | **シグナル→配分→レバレッジ→商品（直列）** | 後段は配分確定後でないと判断不能 |
| WFAゲート | NASDAQ同一（CI95_lo / WFE、`phase_d_wfa`） | 過学習・レジーム運検出 |

> **重要な非対称性（前出し）:** Gold/NASDAQ はレバレッジETFが存在するが、**GBPはレバレッジETFがほぼ無くCFD/FX証拠金が事実上唯一**。Phase 4・6 の判断軸に最初から組み込む。

---

## ファイル構成（新規は最小限）

- Create: `config/assets.yaml` — 資産定義（系列・通貨・ベンチ・コスト参照）
- Modify: `src/data_loaders/signals/yahoo.py` — `GBPJPY=X`, `GBPUSD=X`, `GC=F`(明示), 関連ティッカーを `_SIGNAL_TO_TICKER` 追加
- Create: `src/data/fetch_gbp.py` — GBP系列取得・整形・整合性チェック（yfinance/FRED）
- Modify: `src/product_costs.py` — `GBP_CFD`, `GOLD_2X(DGP)`, `GOLD_CFD`, `MF_1X_NASDAQ/GOLD`（1倍投信）等の `ProductCost` 追記
- Create: `src/multi_asset/allocator.py` — 配分ロジック群
- Create: `src/multi_asset/leverage_eval.py` — 純コスト後レバレッジ採否判定
- Create: `src/multi_asset/product_selector.py` — 商品マッピング・比較表
- Create: `tests/multi_asset/test_allocator.py`, `tests/multi_asset/test_leverage_eval.py`
- Update: `STRATEGY_REGISTRY.md`（各検証1件）、`tasks.md`、`FILE_INDEX.md`、新規 `MULTIASSET_CURRENT_BEST.md`

---

## Phase 0: 基盤・データ整備（GBPギャップ解消が主目的）

### Task 0.1: マルチアセット検証プロトコル定義
**Files:** Create `docs/rules/10_multiasset-protocol.md`
- [ ] 4層（シグナル→配分→レバレッジ→商品）の入出力I/Fを定義
- [ ] 9指標標準・WFAゲート・台帳登録をマルチアセットに適用すると明記
- [ ] 保有/キャッシュ2状態（将来0–1スケール拡張可）を定義

### Task 0.2: 資産定義ファイル
**Files:** Create `config/assets.yaml`
- [ ] NASDAQ/Gold/GBP の系列ソース・通貨・ベンチ・`product_costs` 参照キーを記述

### Task 0.3: GBPデータ取得（TDD）
**Files:** Modify `src/data_loaders/signals/yahoo.py`; Create `src/data/fetch_gbp.py`; Test `tests/data/test_fetch_gbp.py`
- [ ] **Step1:** `yfinance.Ticker.history` をモックした失敗テストを書く（`GBPJPY=X` の Adj Close Series を返す期待）
- [ ] **Step2:** テスト実行→FAIL確認
- [ ] **Step3:** `_SIGNAL_TO_TICKER` に GBP系ティッカー追加＋`fetch_gbp.py` で日次取得・tz除去・共通カレンダー整合
- [ ] **Step4:** テスト実行→PASS確認、`data/gbpjpy_daily.csv`/`data/gbpusd_daily.csv` 生成
- [ ] **Step5:** commit

### Task 0.4: 共通データセット整合・品質監査
**Files:** 既存 `src/build_base_dataset.py` を拡張; 品質監査
- [ ] NDX/Gold/GBP/円キャッシュを共通カレンダーへ整合（欠損/通貨換算/サバイバーシップ検査）
- [ ] 既知歴史局面でのサニティテスト

---

## Phase 1: シグナル棚卸し（既存族＋資産固有族）

### Task 1.1: シグナルタクソノミー
**Files:** Create `docs/multiasset/SIGNAL_TAXONOMY.md`
- [ ] 既存族（トレンド/モメンタム、ボラ・レジームVZ、長期トレンドLT、レジームE4）を Gold/GBP に適用可能性付きで整理
- [ ] 資産固有族を追加し各仮説を1行記述（戦略検証プロトコルStep3）
  - **Gold:** 実質金利/DXY（`fred.py`）、インフレ連動、リスクオフ需要、対株式相関レジーム
  - **GBP:** 英日/英米金利差キャリー、PPP乖離、リスクオン/オフ、対JPYボラ

### Task 1.2: シグナル関数の資産パラメータ化
**Files:** Modify `src/signals/timing.py` ほか（必要時）; Test 追加
- [ ] 価格/マクロ系列を引数化し Gold/GBP で再利用可能に（NASDAQ既存出力の回帰テストで非破壊を担保）

---

## Phase 2: 単資産シグナル検証（保有 vs キャッシュ）

### Task 2.1: 単資産スイープ・ランナー
**Files:** Create `src/multi_asset/single_asset_sweep.py`（ヘッダは `_sweep_format.py`）
- [ ] 資産×シグナル族×パラメータ総当たりを `nine_metric_eval` で評価（CAGR_OOS中心、Trades/yr・Overfit(WFE)列自動算出）
- [ ] ベースライン（B&H・常時キャッシュ）を必ず併記

### Task 2.2: Gold 単独検証 → 候補抽出
**Files:** `outputs/<DATE>_gold-single-asset-sweep.md`
- [ ] スイープ→Shortlist化、3軸ベースライン（min(IS,OOS)・Worst10Y・P10_5Y）超過を採用条件、棄却理由を台帳へ

### Task 2.3: GBP 単独検証 → 候補抽出
**Files:** `outputs/<DATE>_gbp-single-asset-sweep.md`
- [ ] 同上。GBP/JPY vs GBP/USD 差・キャリー寄与を分離評価、WFE高はレジーム運フラグ

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

### Task 3.2: 配分バックテスト・比較レポート
**Files:** `outputs/<DATE>_multiasset-allocation-comparison.md`（統合レポート標準 §0'/§1'/§5/§6）
- [ ] 配分ロジック別に統合9指標比較、分散効果（相関・寄与分解）提示、最良リターン/シャープ配分を選定、ベンチ併記、台帳登録

---

## Phase 4: レバレッジ採否判定（要件②）

### Task 4.1: コストモデル拡張
**Files:** Modify `src/product_costs.py`
- [ ] `ProductCost` に GBP-CFD（スプレッド＋オーバーナイト金利）、Gold 2x(DGP)/Gold-CFD、1倍投信（信託報酬）、税区分差を追記
- [ ] deep-research 4レポート（nasdaq-4x / gold-bond-3x / sbi-cfd-nasdaq-8x-9x / sbi-gold-bond-high-leverage）＋`g12/g13` を数値根拠に

### Task 4.2: レバレッジ純便益エンジン（TDD）
**Files:** Create `src/multi_asset/leverage_eval.py`; Test `tests/multi_asset/test_leverage_eval.py`
- [ ] 判定基準: **純コスト後シャープ＞無レバ** ∧ **MaxDD/Worst10Y許容内** ∧ **ボラ減衰織込み後も期待リターン増＞コスト増**
- [ ] 倍率グリッド（1x/2x/3x…、GBPは制約反映）で資産別・配分後の双方を評価
- [ ] 「採用すべきか」Yes/No＋根拠指標を出力

### Task 4.3: レバレッジ判定レポート
**Files:** `outputs/<DATE>_leverage-decision-report.md`
- [ ] 資産別・配分後の採否結論、GBPのETF不在制約を明示、台帳登録

---

## Phase 5: 運用商品の選定（要件③）

### Task 5.1: 商品候補マトリクス
**Files:** Create `src/multi_asset/product_selector.py`; `outputs/<DATE>_product-selection-matrix.md`
- [ ] 3系統で整理: (a) レバレッジETF（TQQQ/2036/TMF等） (b) CFD (c) 1倍投信
- [ ] 選定軸: コスト総額・取得可能レバレッジ・税区分（特定口座/雑所得/投信）・トラッキング・流動性・最低資金・リバランス摩擦・**シグナルの保有期間/Trades-yr整合**
- [ ] GBPは実質CFD一択になる点を結論に反映

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

## 自己レビューで改善した点（v1 提示版 → v2 リポ接地版）

1. **新規ファイル乱立を撤回し既存基盤に接地**: `src/signals/*`・`nine_metric_eval`・`phase_d_*`・`_sweep_format`・`product_costs` の再利用に置換（DRY、重複規約遵守）。当初案の「`src/signals/signal_library.py` 新規作成」は既存 `src/signals/` と重複するため削除。
2. **GBPデータ欠如を最優先ブロッカーとして特定**: 実リポ確認で GBP 系列が皆無と判明 → Phase 0 Task 0.3 を必須前提に格上げ。
3. **Goldは「データ有・単独体系検証は未実施」と正確化**: 既存 `h5_gold_dyn`/`research_gold_bond_timing` はNASDAQポートフォリオ内利用に留まる事実を反映。
4. **検証順序の直列依存を明示**: 配分確定前にレバレッジ/商品は判断不能という依存を前提に固定。
5. **GBPレバレッジETF不在を主要制約として前出し**: Phase 4/5 の判断軸に最初から組込み。
6. **ボラ減衰/パス依存をレバレッジ判定基準に明記**: 「期待リターン＞コスト」だけでは日次リバランス型ETFの減衰を見落とすため3条件に厳格化。
7. **既存CFDコスト基盤（g12/g13）＋deep-research 4レポートを数値根拠に接続**: コスト定数をゼロから作らず再利用。
8. **回転率(Trades/yr)を商品選定に直結**: コスト単独でなく保有期間×商品摩擦の整合を必須軸に。
