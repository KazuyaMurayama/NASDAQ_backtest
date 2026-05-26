# nasdaq_backtest — 運用ルール入口

NASDAQ 3倍レバレッジ戦略のバックテスト研究リポジトリ。**main 単一ブランチ運用**。

## セッション開始時の参照順序
1. **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)** — 現行ベスト戦略の正典 (Single Source of Truth)
2. **FILE_INDEX.md** — 全ファイルの所在・優先度
3. **tasks.md** — 未完了タスク・進捗
4. このCLAUDE.md — ルール入口

## 🎯 ベスト戦略参照プロトコル（最重要・再発防止）

「ベスト戦略は？」「現在の推奨は？」「最終結果は？」と問われた時、**必ず以下の順で確認**:

1. **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を最優先で読む** — このファイルが現行の正典
2. tasks.md の最新 ✅ Completed エントリと整合性チェック
3. **本 CLAUDE.md 内の「現行ベスト戦略:」記述は二次資料**として扱う（更新漏れがありうる）
4. CSV ファイル (`R4_results.csv` 等) は実験ログであり、結論ではない

### 絶対にやってはいけないこと
- ❌ `FINAL_RESULTS_2026-02-06.md` を「最終結果」として読む（SUPERSEDED 済み）
- ❌ ファイル名に `FINAL` を含むからといって最新と判定する（`FINAL_` プレフィックスは廃止）
- ❌ CSV を Sharpe 降順で並べて「トップ」と回答する

## 📊 評価指標ルール（厳守・スキップ禁止）

戦略評価・比較・WFA の**すべての場面**で以下を遵守すること:

1. **9指標のみ使用** ([docs/rules/08_evaluation-metrics.md](docs/rules/08_evaluation-metrics.md) 参照)
   - 標準7: CAGR(IS/OOS/FULL), Sharpe_OOS, MaxDD, Worst10Y★, P10_5Y▷, IS-OOS gap, Trades/yr
   - WFA補助2: WFA_CI95_lo (§3.9), WFA_WFE (§3.10)
2. **禁止指標** (ユーザー明示指示なしに評価・ランキングで使用禁止):
   Stable_Sharpe / WinRate_yr / WorstK5_mean_CAGR / IR_vs_BH
3. **根拠**: [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) v1.1 §3

## 🔬 新検証アイデア着手前プロトコル（必須・スキップ禁止）

新規戦略・改善アイデア・パラメータ探索を着手する前に **4ステップを順次実施**（重複チェック→評価基準確認→差分仮説→登録）。詳細・例外は [docs/rules/06_strategy-verification.md](docs/rules/06_strategy-verification.md)。**スキップ時は重複研究扱いで結果は採用候補から除外。**

## 📛 ドキュメント命名規則・日付ルール
- `FINAL_` プレフィックス禁止 / `<TOPIC>_YYYY-MM-DD.md` 形式 / SUPERSEDED ヘッダで旧版置換
- 詳細は [docs/rules/07_doc-naming-and-dates.md](docs/rules/07_doc-naming-and-dates.md)

## 運用ルール（詳細はスキルファイル）

| # | ファイル | 内容 |
|---|---------|------|
| 01 | [docs/rules/01_response-basics.md](docs/rules/01_response-basics.md) | 回答の基本ルール |
| 02 | [docs/rules/02_task-management.md](docs/rules/02_task-management.md) | タスク管理 |
| 03 | [docs/rules/03_file-index.md](docs/rules/03_file-index.md) | ファイルインデックス管理 |
| 04 | [docs/rules/04_deliverables-and-models.md](docs/rules/04_deliverables-and-models.md) | 成果物・モデル・出力フォーマット |
| 05 | [docs/rules/05_git-and-execution.md](docs/rules/05_git-and-execution.md) | Git操作・実行計画 |
| 06 | [docs/rules/06_strategy-verification.md](docs/rules/06_strategy-verification.md) | 新検証アイデア着手前プロトコル |
| 07 | [docs/rules/07_doc-naming-and-dates.md](docs/rules/07_doc-naming-and-dates.md) | ドキュメント命名・日付ルール |
| 08 | [docs/rules/08_evaluation-metrics.md](docs/rules/08_evaluation-metrics.md) | 評価指標ルール（9指標厳守） |

## プロジェクト概要

52年間（1974-2026）のNASDAQ Composite を対象に、3倍レバレッジ日次リバランス戦略を研究。

### 現行ベスト戦略
具体的な戦略名・指標は **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を必ず参照**。本ファイル（CLAUDE.md）には数値をハードコードしない（更新漏れによる誤情報を防ぐため）。一次根拠ファイルは CURRENT_BEST_STRATEGY.md 内の「一次根拠ファイル」表を参照。

### 実運用リポジトリ
https://github.com/KazuyaMurayama/nasdaq-strategy-gas

## 開発者情報・命名ルール

このリポジトリの開発者・所有者は **男座員也（Kazuya Oza / おざ かずや）** です。

- ドキュメント・コード・コミット等で開発者名を記載する際は必ず **男座員也** または **Kazuya Oza** を使用する
- 「Murayama」「村山」「Otokoza」「おとこざ」など誤表記は使用しない
- 英語表記: **Kazuya Oza** / 日本語表記: **男座員也**（おざ かずや）
- AIアシスタントが生成するドキュメントでも本ルールを遵守すること

### 開発者の作業環境
- **OS:** Windows 11（Macではない）。シェルは PowerShell 5.1 / Bash（WSL/Git Bash）。`brew` / `Cmd+` / Mac専用コマンドは使用不可。パッケージ管理は `winget` / `scoop`。
- **スマートフォン:** iPhone（iOS）。Android固有の手順・adb・Play Store等は不要。
- コマンド例はPowerShell構文（`;` 連結、`$env:VAR`）で提示。macOS専用ツールを回答に含めない。

## ビジュアルルール（レポートMD生成時）
- レポート・成果物MDの新規作成／更新時は `.claude/visual-rules.md` を読み、図の種類判定（§2）と Mermaid 最適化（§3）を毎回適用する。
- 適用対象: `## ` 見出しが2つ以上ある構造化MD（調査結果・戦略レポート・設計書・PR説明など）。

---

## 🔍 戦略監査チェックリスト（毎回必須・スキップ禁止）

「戦略監査」「品質チェック」「過学習検定」「実現性レビュー」「OOS 評価」等の依頼を受けた瞬間、
以下の Phase 1〜3 を**必ずすべてのバリアントに対して**機械的に実行する。
詳細は `audit_results/AUDIT_FRAMEWORK.md` を参照。

### 🔴 鉄則（違反 = レポート全体を再提出対象）

1. **「コードを読めば自明」は実施扱いにしない**
   検証は必ず `python src/audit/check_*.py` を実行し `audit_results/*.md` + `*.yaml` を生成する。
   ファイルが存在しない検証は「未実施」と明記する。

2. **「旧戦略値 + オフセット推定」は実証扱いにしない**
   新戦略バリアント（E4→F→G…）追加時、Phase 1〜3 の**全スクリプトをそのバリアント名で再実行**する。
   古いキャッシュは `force_rebuild=True` で再生成すること。

3. **✅ 判定の右に「実行コマンド + 出力ファイルパス」を必ず併記する**
   出力ファイルが存在しない ✅ は虚偽報告として扱う。

4. **「推奨実施項目」をコードスニペットで提示して実行しない行為を禁止する**
   コードスニペットを書いたなら、そのまま実行してファイルを生成すること。

### Phase 1: ロジック正しさ検証（Phase 2/3 より先に必須）

新バリアント導入・シグナル計算式変更・`build_nav_strategy()` 拡張時に必須。

- [ ] **DELAY=2 コンプライアンス**: NAV 手計算 vs `build_nav_strategy()` を 10 日分突き合わせ → `check_logic_delay_and_causal.py`
- [ ] **因果性**: `rolling(window=N)` 系シグナル（vz, lt_sig）が t 日時点で過去のみ参照を assert 検証
- [ ] **レジーム切替境界条件**: `k_dyn` 等の np.where 分岐を境界値テスト（7 ケース）
- [ ] **ゼロポジション境界**: `lev_A=0` 時の演算整合性（クリップ含む）
- [ ] **サニティ**: 期待 Sharpe_OOS との差 ≤ 0.005

### Phase 2: 実現性検証（バリアントごとに再実行）

- [ ] **ブローカー別 CAGR/Worst10Y★**: 6 シナリオ全て正確計算（推定値禁止）→ `check_e4_broker_matrix.py`
- [ ] **証拠金ダイナミクス**: 5 シナリオ → `check_sim_margin_dynamics.py`（E4 版は `lev_mod_e4` を使って再実行）
- [ ] **コスト根拠台帳**: `product_costs.py` との整合確認 → `check_realworld_report_corpus.py`

### Phase 3: 過学習検出（バリアントごとに再実行）

- [ ] **Block Bootstrap** (B=5000, L=20/63/126): CI95_lo > 0 ∧ p < 0.05 → `check_overfitting_e4_bootstrap.py`
- [ ] **Permutation (d) 同時置換** (B=1000, block=63): p < 0.05 → `check_overfitting_e4_permutation.py`
- [ ] **WFA**: CI95_lo > 0 ∧ 0.5 ≤ WFE ≤ 2.0 → `src/g3_wfa_e4.py`
- [ ] **パラメータ感度** (±20%): 主要 6 パラメータ全て → `check_e4_parameter_sensitivity.py`

### 新戦略バリアント追加プロトコル（F/G 系追加時に必須）

1. `src/audit/_audit_strategy.py` に `build_<NEW>_strategy_assets()` を追加
2. 専用キャッシュ `_cache/<NEW>_nav_cache.pkl` + サニティ定数 `<NEW>_SHARPE_OOS_EXPECTED` を追加
3. Phase 1〜3 の全スクリプトを `<NEW>` 版として複製・実行
4. `audit_results/<NEW>_QUALITY_REPORT_YYYYMMDD.md` を生成（実施証跡表必須）
5. `CURRENT_BEST_STRATEGY.md` に Phase 1/2/3 の判定を追記

### 実施証跡表テンプレ（品質レポート末尾必須）

```markdown
## 実施証跡

| Phase | チェック項目 | スクリプト | 出力ファイル | 判定 |
|---|---|---|---|---|
| 1.1 | DELAY=2 コンプライアンス | `src/audit/check_logic_delay_and_causal.py` | `audit_results/LOGIC_CHECK_YYYYMMDD.md` | ✅ PASS |
| 1.2 | vz/lt_sig 因果性 | （同上） | （同上） | ✅ PASS |
| 1.3 | k_dyn 境界条件 | （同上） | （同上） | ✅ PASS |
| 2.1 | ブローカー別 CAGR | `src/audit/check_e4_broker_matrix.py` | `audit_results/E4_BROKER_MATRIX_YYYYMMDD.md` | ✅ PASS |
| 3.1 | Bootstrap CI95_lo | `src/audit/check_overfitting_e4_bootstrap.py` | `audit_results/E4_BOOTSTRAP_YYYYMMDD.md` | ✅ PASS |
| 3.4 | パラメータ感度 | `src/audit/check_e4_parameter_sensitivity.py` | `audit_results/E4_PARAM_SENSITIVITY_YYYYMMDD.md` | ✅ PASS |
```

未実施項目は「未実施」と明示。「✅ 文章で言及」は禁止。
