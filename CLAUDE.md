# nasdaq_backtest — Claude Code 運用ルール（入口）

NASDAQ 3倍レバレッジ戦略のバックテスト研究リポジトリ。**main 単一ブランチ運用**。

> **本ファイルは VSCode版 / Web版 Claude Code（claude.ai）の両方で本リポジトリの起点となる単独完結ガイド**。
> Web版はグローバル `~/.claude/CLAUDE.md` を参照しない前提で、本リポの運用に必要な全ルールをここに集約している。

---

## 0. セッション開始時の参照順序（必ず守る）
1. **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)** — 現行ベスト戦略の正典 (Single Source of Truth)
2. **FILE_INDEX.md** — 全ファイルの所在・優先度
3. **tasks.md** — 未完了タスク・進捗
4. このCLAUDE.md — ルール入口

---

## 1. 🎯 ベスト戦略参照プロトコル（最重要・再発防止）

「ベスト戦略は？」「現在の推奨は？」「最終結果は？」と問われた時、**必ず以下の順で確認**:

1. **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を最優先で読む** — このファイルが現行の正典
2. tasks.md の最新 ✅ Completed エントリと整合性チェック
3. **本 CLAUDE.md 内の「現行ベスト戦略:」記述は二次資料**として扱う（更新漏れがありうる）
4. CSV ファイル (`R4_results.csv` 等) は実験ログであり、結論ではない

### 絶対にやってはいけないこと
- ❌ `FINAL_RESULTS_2026-02-06.md` を「最終結果」として読む（SUPERSEDED 済み）
- ❌ ファイル名に `FINAL` を含むからといって最新と判定する（`FINAL_` プレフィックスは廃止）
- ❌ CSV を Sharpe 降順で並べて「トップ」と回答する

---

## 2. 📊 評価指標ルール（厳守・スキップ禁止）

戦略評価・比較・WFA の**すべての場面**で以下を遵守すること:

1. **9指標のみ使用** ([docs/rules/08_evaluation-metrics.md](docs/rules/08_evaluation-metrics.md) 参照)
   - 標準7: CAGR(IS/OOS/FULL), Sharpe_OOS, MaxDD, Worst10Y★, P10_5Y▷, IS-OOS gap, Trades/yr
   - WFA補助2: WFA_CI95_lo (§3.9), WFA_WFE (§3.10)
2. **禁止指標** (ユーザー明示指示なしに評価・ランキングで使用禁止):
   Stable_Sharpe / WinRate_yr / WorstK5_mean_CAGR / IR_vs_BH
3. **根拠**: [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) v1.1 §3

---

## 3. 📋 sweep / grid / 戦略比較レポート生成時のルール（必須・違反は再提出）

1. `src/` 配下に新規 sweep / grid / **戦略比較**スクリプトを作成・更新する場合、または `STRATEGY_COMPARISON_*` 系 MD を手書きで作成・更新する場合は `EVALUATION_STANDARD.md §3.12` の **9指標標準（Strategy/Param + 9指標 = 10列ヘッダ。`Trades_yr`・`Overfit(WFE)`・`WFA_CI95_lo` 必須）**、MD は `<br>` 折り返しヘッダを**必ず**満たす。
2. **MD ヘッダは `src/_sweep_format.py` の `MD_HEADER_1P` / `MD_HEADER_2P` / `MD_HEADER_STRAT` を必ず import して使用する。手書きヘッダ禁止**。
3. **`CAGR_IS` / `CAGR_FULL` を MD テーブルヘッダに含めた時点で v1.1 違反**（CSV 保存は OK、本文中の言及も OK、比較表ヘッダには CAGR_OOS のみ）。
4. `Trades_yr` 欠落・`Overfit(WFE)` 欠落・手書きヘッダ・`CAGR_IS/FULL` の MD 掲載のいずれも v1.3 違反として**再提出対象**。
5. **Overfit(WFE) 列**は `WFA_WFE` から自動算出（`fmt_row_*` 内で `_ovfit_wfe()` が呼ばれる）。列値は2行表示（例: `✅ LOW<br>(1.1)`）。判定: ✅LOW=WFE∈[0.5,2.0] / ⚠MED=WFE>2.0 / ❌HIGH=WFE<0.5。WFE値は小数点1桁。WFE別列は廃止（Overfit(WFE)に統合済み）。
6. **MD レポートをサブエージェントに委託する場合**: タスク prompt に必ず明記「`src/_sweep_format.py` の `MD_HEADER_*` を import 必須、手書きヘッダ禁止、`CAGR_IS/FULL` を MD ヘッダに含めない、`Overfit(WFE)` 列必須（10列 / 9指標標準）」

---

## 4. 🔬 新検証アイデア着手前プロトコル（必須・スキップ禁止）

新規戦略・改善アイデア・パラメータ探索を着手する前に **4ステップを順次実施**（重複チェック→評価基準確認→差分仮説→登録）。詳細・例外は [docs/rules/06_strategy-verification.md](docs/rules/06_strategy-verification.md)。**スキップ時は重複研究扱いで結果は採用候補から除外。**

---

## 5. 📛 ドキュメント命名規則・日付ルール
- `FINAL_` プレフィックス禁止 / `<TOPIC>_YYYY-MM-DD.md` 形式 / SUPERSEDED ヘッダで旧版置換
- レポート系 .md 新規作成時は H1直下に `作成日: YYYY-MM-DD` / `最終更新日: YYYY-MM-DD` を必須記載。更新時は最終更新日のみ書き換え（作成日は固定）
- 除外: README / CLAUDE.md / FILE_INDEX / tasks.md / CHANGELOG / LICENSE
- 詳細は [docs/rules/07_doc-naming-and-dates.md](docs/rules/07_doc-naming-and-dates.md)

---

## 6. 運用ルール（詳細はdocs/rules/）

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

> 「監査」「品質チェック」「Phase 1/2/3」「WFAを回す」等を検出したら `.claude/rules/audit-protocol.md` を読んで実行する

---

## 7. プロジェクト概要

52年間（1974-2026）のNASDAQ Composite を対象に、3倍レバレッジ日次リバランス戦略を研究。

### 現行ベスト戦略
具体的な戦略名・指標は **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を必ず参照**。本ファイル（CLAUDE.md）には数値をハードコードしない（更新漏れによる誤情報を防ぐため）。一次根拠ファイルは CURRENT_BEST_STRATEGY.md 内の「一次根拠ファイル」表を参照。

### 実運用リポジトリ
https://github.com/KazuyaMurayama/nasdaq-strategy-gas — Googleスプレッドシート連携の本番運用版

---

## 8. 開発者情報・命名ルール

| 種別 | 表記 | 用途 |
|---|---|---|
| **システム識別子（変更不可）** | `KazuyaMurayama` | GitHub ユーザー名 / URL / `@KazuyaMurayama` |
| **システム識別子（変更不可）** | `kazuya.murayama.21@gmail.com` | git `user.email` / 連絡先 |
| **表記名（人間として記載する場合）** | **男座員也（Kazuya Oza / おざ かずや）** | ドキュメント本文の著者名 / プロフィール / コミット message 中の自己言及 |

- ドキュメント・コード・コミットメッセージ本文等で開発者名を**人間として**記載する際は **男座員也 / Kazuya Oza** を使用
- GitHub URL や `@KazuyaMurayama` 等のシステム識別子は**そのまま使う**（変更しない）
- 「Murayama」「村山」「Otokoza」「おとこざ」を表記名として誤用しない（システム識別子としての `KazuyaMurayama` 出現は許容）
- AIアシスタントが生成するドキュメントでも本ルールを遵守

---

## 9. ビジュアルルール（レポートMD生成時）
- レポート・成果物MDの新規作成／更新時は `.claude/visual-rules.md` を読み、図の種類判定（§2）と Mermaid 最適化（§3）を毎回適用する。
- 適用対象: `## ` 見出しが2つ以上ある構造化MD（調査結果・戦略レポート・設計書・PR説明など）。

---

## 10. ツール実行・Shell・Git・ファイル保存（運用クリティカル）

### ツール実行ポリシー
- 確認不要・即実行（「Should I...?」等の事前確認文を出力しない）
- ファイル操作は Edit/Write/Read/Grep/Glob を直接使用
- 例外（事前確認必須）: main への `git push --force`、`gh repo delete`

### Shell（VSCode版: Windows 11 + PowerShell 5.1 想定 / Web版: Linux サンドボックス）
- PowerShell 5.1: `&&` 不可 → `;` + `if ($?)`。Bash tool併用可
- GitHub操作は REST API 直接呼び出しを優先（gh CLI より高速）

### ブランチ管理（絶対厳守）
- **デフォルト: mainへ直接コミット**。ブランチ作成はユーザーが明示的に指示した場合のみ
- 作業前: `git branch --show-current` でブランチ確認 → main以外なら `git checkout main && git pull` してから開始
- ブランチを作成した場合、必ず `main` へマージ → ブランチ削除 → push を完了してから作業完了とする
- 「完了 = mainにマージ済み＆push済み」。ブランチにファイルを置いたまま回答を完了することを禁止
- Web版が自動生成したブランチ（`claude/xxx`）も同様。セッション終了前に必ずmainへマージする

### ファイル保存ルール
- 成果物・スクリプトは本リポジトリ内のみに保存。`C:\Users\user\Desktop` への出力禁止（ユーザー明示指定時を除く）
- 一時スクリプトも本リポ内に作成し、作業後に削除またはコミット

---

## 11. 成果物報告ルール（最重要・毎回必須）

ファイルを1つでも作成・更新・pushしたら、**すべての**成果物を以下の形式で報告：

| 成果物 | 説明 | リンク |
|---|---|---|
| file.md | 1行説明 | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/path/to/file.md) |

### 厳守事項
1. Markdownリンク `[表示名](URL)` 形式必須（plain text URL禁止）
2. `/blob/<実ブランチ>/<実パス>` 形式（リポジトリトップURL禁止）
3. **報告前にURL存在確認必須**：`gh api repos/KazuyaMurayama/NASDAQ_backtest/contents/PATH?ref=BRANCH` または `Invoke-WebRequest -Uri https://api.github.com/repos/KazuyaMurayama/NASDAQ_backtest/contents/PATH?ref=BRANCH -UseBasicParsing` でステータス200確認
4. ブランチ名は推測禁止：`git rev-parse --abbrev-ref HEAD` で実値取得
5. push完了後のみURL生成
6. 404を出したら即訂正＆原因1行報告

---

## 12. Skill 起動ルール（必須・スキップ禁止 / v2.3 / 2026-06-02）

該当シーンでは `.claude/skills/<name>/SKILL.md` を読んでから作業を開始する。
（Web版は本リポの `.claude/skills/` を参照。VSCode版は `~/.claude/skills/` も参照可）

| トリガー | スキル |
|---|---|
| 時系列・トレンド分析 | `.claude/skills/time-series-analysis/SKILL.md` |
| A/B テスト・戦略比較の統計検定 | `.claude/skills/ab-test-analysis/SKILL.md` |
| 新戦略・新指標の先行研究調査 | `.claude/skills/research-deep/SKILL.md` |
| 大規模 sweep/grid 計画 | `.claude/skills/sp-writing-plans/SKILL.md` + `sp-executing-plans/SKILL.md` |
| 比較レポートに図表必要 | `.claude/skills/mermaid-agents365/SKILL.md` |
| 戦略評価・比較レポートのQC・レビュー・共有前 | `.claude/skills/analysis-qa-checklist/SKILL.md` |
| 成果物の納品・コミット前 | `.claude/skills/sp-verification-before-completion/SKILL.md` |
| データ品質・整合性確認 | `.claude/skills/data-quality-audit/SKILL.md` |
| アイデア出し・選択肢の洗い出し | `.claude/skills/sp-brainstorming/SKILL.md` |
| バグ・エラーの体系的調査 | `.claude/skills/sp-systematic-debugging/SKILL.md` |
| ドキュメント命名・日付ルール再確認 | `docs/rules/07_doc-naming-and-dates.md` |

---

## 13. プロジェクト記憶（VSCode版限定で参照）
VSCode版で実行されている場合は、追加で以下を参照可：
- `~/.claude/CLAUDE.md` — グローバル運用ルール
- `~/.claude/projects/C--Users-user/memory/MEMORY.md` — 自動メモリ（プロジェクト記憶）
- `~/.claude/projects/C--Users-user/memory/nasdaq_best_strategy.md` — ベスト戦略確認プロトコル詳細
- `~/.claude/projects/C--Users-user/memory/project_nasdaq_9metric_standard.md` — 9指標標準の詳細

Web版ではこれらにアクセスできないため、本リポ内 `docs/rules/` および `CURRENT_BEST_STRATEGY.md` のみで完結すること。
