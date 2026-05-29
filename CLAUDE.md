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

## ビジュアルルール（レポートMD生成時）
- レポート・成果物MDの新規作成／更新時は `.claude/visual-rules.md` を読み、図の種類判定（§2）と Mermaid 最適化（§3）を毎回適用する。
- 適用対象: `## ` 見出しが2つ以上ある構造化MD（調査結果・戦略レポート・設計書・PR説明など）。

---

> 「監査」「品質チェック」「Phase 1/2/3」「WFAを回す」等を検出したら `.claude/rules/audit-protocol.md` を読んで実行する

## ファイル保存ルール
- 成果物・スクリプトは本リポジトリ内のみに保存。`C:\\Users\\user\\Desktop` への出力禁止（ユーザー明示指定時を除く）。

<!-- SKILLS_RULES_START -->
## Skill 起動ルール（v2.1 / 2026-05-29）
以下のスキルは **必須・スキップ禁止**。該当シーンでは SKILL.md を読んでから作業を開始すること。

- **時系列・トレンド分析を行う時は必ず** `.claude/skills/time-series-analysis/SKILL.md` を読み、手順に従って分析を実行する
- **A/B テスト・戦略比較の統計検定を行う時は必ず** `.claude/skills/ab-test-analysis/SKILL.md` を読み、手順に従って検定を実行する
- **新戦略・新指標の先行研究調査が必要な時は必ず** `.claude/skills/research-deep/SKILL.md` を読んでから並列 Web リサーチを実行する
- **大規模 sweep/grid 計画を立てる時は必ず** `.claude/skills/sp-writing-plans/SKILL.md` を読んでフェーズ分割計画を作成し、`.claude/skills/sp-executing-plans/SKILL.md` の手順で実行する
- **比較レポートに図表が必要な時は必ず** `.claude/skills/mermaid-agents365/SKILL.md` を読んでからダイアグラムを作成する
- **戦略評価・比較レポートの品質チェック（QC）・レビュー・ステークホルダー共有前は必ず** `.claude/skills/analysis-qa-checklist/SKILL.md` を読んでチェックリストを実施する
- **成果物の納品・コミット前、または品質チェック（QC）・レビューフェーズに入る時は必ず** `.claude/skills/sp-verification-before-completion/SKILL.md` のチェックリストを実行する
- **データ品質・整合性の確認が必要な時は必ず** `.claude/skills/data-quality-audit/SKILL.md` を読んで監査を実行する
<!-- SKILLS_RULES_END -->
