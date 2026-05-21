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
