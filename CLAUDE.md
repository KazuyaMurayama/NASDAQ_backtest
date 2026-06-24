# NASDAQ_backtest — Claude Code 運用ルール（入口）

NASDAQ 3倍レバレッジ戦略のバックテスト研究リポジトリ。**main 単一ブランチ運用**。

> **本ファイルは VSCode版 / Web版 Claude Code（claude.ai）の両方で本リポジトリの起点となる単独完結ガイドです。**
> Web版はグローバル `~/.claude/CLAUDE.md` を参照しない前提で、本リポの運用に必要な全ルールをこの1ファイルに集約しています（他リポ・グローバルとの重複は完結性のため許容）。

---

## 1. セッション開始時の参照順序（毎回・最初に実行）

セッション開始時、次の順で必ず読み込んでから作業を始める：

1. **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)** — 現行ベスト戦略の正典（Single Source of Truth）
2. **FILE_INDEX.md** — 全ファイルの所在・優先度
3. **tasks.md** — 未完了タスク・進捗
4. **本ファイル（CLAUDE.md）** — ルール入口

> 編集前に、ユーザー発話のキーワードを `FILE_INDEX.md` と照合してから対象ファイルを特定する（推測で着手しない）。

### 直近セッション要約 (2026-06-07 時点 / v4.5〜v4.8)

新セッション開始時、以下が既に検証・整理済であることを認識せよ:

**v4.5 (2026-06-05)**: **min(IS, OOS) CAGR を保守的期待リターン指標として標準化** (詳細 §3 / [EVALUATION_STANDARD.md §3.13](EVALUATION_STANDARD.md))。WFE>1.5 は regime luck 警告。当初は「3 軸必須」ルールだったが 2026-06-08 改訂で削除、min(IS, OOS) のみ標準化された保守的指標として残存。Worst10Y/P10_5Y は 9 指標の一部として参照するが強制条件ではない。

**v4.4-v4.5 (2026-06-03〜05)**: vz=0.65+l7+F10ε への非対称機構移植 3 候補 (AH/AT/HL) → **全て棄却** (min ルール下で 3 軸全敗、WFE>1.5 regime luck)。詳細 [STRATEGY_REGISTRY §3 Rejected](STRATEGY_REGISTRY.md)。

**v4.6 (2026-06-05)**: vz=0.65+F10ε の lmax sweep (l5/l5.5/l7) 実装、l5.5 / l5 を §2 Shortlisted 追加。

**v4.7 (2026-06-05)**: ユーザー判断で **CFD 環境 Active 候補を vz=0.65+l7+F10ε → vz=0.65+l5+F10ε に置換** (防御指標優位)。

**v4.8 (2026-06-07)**: **投信環境 Active 候補 4 戦略** を §2 Shortlisted 新規追加 ([STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §6-6](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md)): DH_W1_CashSleeve_P2_GOLD100 (攻め) / DH_W1_CashSleeve_P7_GOLD75BOND25 (中庸推奨) / DH_W1_CashSleeve_P5_GOLD50BOND50 (守り)。詳細は [analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md) 参照。**t_p/bootstrap 未実施で正式 §1 Active 昇格は保留**。

**環境別 Active 候補制度** (v4.5 以降確立、3 環境):
| 環境 | 主候補 | 副候補 |
|---|---|---|
| CFD 利用可 | **vz=0.65+l5+F10ε** (v4.7) | vz=0.65+l7+F10ε (攻め) |
| ETF only (NISA等) | **DH-W1** (Asymm+Hyst) (v4.3) | DH Dyn 2x3x [A] (baseline) |
| 投信環境 | **DH_W1_CashSleeve_P7_GOLD75BOND25** (v4.8) | P2 GOLD100 (攻め) / P5 GOLD50BOND50 (守り) |

**命名規則**: vz=0.65+l7+F10ε を「NEW」「NEW CANDIDATE」と呼ぶことは廃止 (v4.4 以降)。パラメータ表記または Registry ID で参照。

---

## 2. ベスト戦略参照プロトコル（最重要・再発防止）

「ベスト戦略は？」「現在の推奨は？」「最終結果は？」と問われた時、**必ず以下の順で確認**:

1. **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を最優先で読む** — このファイルが現行の正典
2. tasks.md の最新 Completed エントリと整合性チェック
3. **本 CLAUDE.md 内の記述は二次資料**として扱う（更新漏れがありうる）
4. CSV ファイル (`R4_results.csv` 等) は実験ログであり、結論ではない

### 絶対にやってはいけないこと
- `FINAL_RESULTS_2026-02-06.md` を「最終結果」として読む（SUPERSEDED 済み）
- ファイル名に `FINAL` を含むからといって最新と判定する（`FINAL_` プレフィックスは廃止）
- CSV を Sharpe 降順で並べて「トップ」と回答する

---

## 3. 評価指標ルール（厳守・スキップ禁止）

戦略評価・比較・WFA のすべての場面で以下を遵守すること:

1. **9指標のみ使用** ([docs/rules/08_evaluation-metrics.md](docs/rules/08_evaluation-metrics.md) 参照)
   - 標準7: CAGR(IS/OOS/FULL), Sharpe_OOS, MaxDD, Worst10Y★, P10_5Y▷, IS-OOS gap, Trades/yr
   - WFA補助2: WFA_CI95_lo (§3.9), WFA_WFE (§3.10)
2. **禁止指標** (ユーザー明示指示なしに評価・ランキングで使用禁止): Stable_Sharpe / WinRate_yr / WorstK5_mean_CAGR / IR_vs_BH
3. **根拠**: [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) v1.1 §3

### v4.5 追加: 標準化された保守的 CAGR 指標 — min(IS, OOS) CAGR

戦略評価において **min(IS, OOS) CAGR** を保守的な期待リターン推定値として**標準化**する (2026-06-05 以降の確定ルール)。

**定義**: `min(cum_CAGR_IS, cum_CAGR_OOS)` — `cum_CAGR_IS` = IS 値 (1977-2020 暦年複利), `cum_CAGR_OOS` = OOS 値 (2021-2026 暦年複利)

**WFE 補助判定** (regime luck 警告): WFE ≤ 1.2: OK / 1.2 < WFE ≤ 1.5: 注意 / WFE > 1.5: regime luck 強疑い

> v4.5 当初は「3 軸 (min + Worst10Y + P10_5Y) すべて baseline 以上」を必須条件としていたが、2026-06-08 改訂で**この強制条件は削除**。Worst10Y / P10_5Y は標準 9 指標として参照するが、強制的な必須条件ではない。総合判断はユーザー裁量。

詳細: [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §7-2](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md), [EVALUATION_STANDARD.md §3.13](EVALUATION_STANDARD.md)

---

## 4. sweep / grid / 戦略比較レポート生成時のルール（必須・違反は再提出）

1. `src/` 配下に新規 sweep / grid / 戦略比較スクリプトを作成・更新する場合、または `STRATEGY_COMPARISON_*` 系 MD を手書きで作成・更新する場合は `EVALUATION_STANDARD.md §3.12` の **9指標標準（Strategy/Param + 9指標 = 10列ヘッダ。`Trades_yr`・`Overfit(WFE)`・`WFA_CI95_lo` 必須）**、MD は `<br>` 折り返しヘッダを**必ず**満たす。
2. **MD ヘッダは `src/_sweep_format.py` の `MD_HEADER_1P` / `MD_HEADER_2P` / `MD_HEADER_STRAT` を必ず import して使用する。手書きヘッダ禁止**。
3. **`CAGR_IS` / `CAGR_FULL` を MD テーブルヘッダに含めた時点で v1.1 違反**（CSV 保存は OK、本文中の言及も OK、比較表ヘッダには CAGR_OOS のみ）。
4. `Trades_yr` 欠落・`Overfit(WFE)` 欠落・手書きヘッダ・`CAGR_IS/FULL` の MD 掲載のいずれも v1.3 違反として**再提出対象**。
5. **Overfit(WFE) 列**は `WFA_WFE` から自動算出。判定: LOW=WFE∈[0.5,2.0] / MED=WFE>2.0 / HIGH=WFE<0.5。
6. **MD レポートをサブエージェントに委託する場合**: タスク prompt に必ず明記「`src/_sweep_format.py` の `MD_HEADER_*` を import 必須、手書きヘッダ禁止、`CAGR_IS/FULL` を MD ヘッダに含めない、`Overfit(WFE)` 列必須（10列 / 9指標標準）」
7. **戦略検証・採用判断レポート（毎回・例外なし）**: 以下を行うときは次の4セクションをすべて含めること:
   - **適用トリガー**: 新戦略テスト・パラメータ検証・OOS評価・採用判断・候補比較・WFA再実行・OOS延長後の再評価
   - **§0'** 候補戦略 統合比較表（`MD_HEADER_INTEGRATED` + `fmt_row_integrated`）
   - **§1'** コスト・税金 調整前提（14ステップ / 参照元: `STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md §1'`）
   - **§5** 年次リターン表（1977-2026 / `fmt_annual_table`）
   - **§6** 統計サマリ（1974-2026 / `fmt_stats_table`）
   - 詳細: [docs/rules/09_integrated-report-standard.md](docs/rules/09_integrated-report-standard.md)

---

## 5. 新検証アイデア着手前プロトコル（必須・スキップ禁止）

新規戦略・改善アイデア・パラメータ探索を着手する前に **4ステップを順次実施**（重複チェック→評価基準確認→差分仮説→登録）。詳細・例外は [docs/rules/06_strategy-verification.md](docs/rules/06_strategy-verification.md)。**スキップ時は重複研究扱いで結果は採用候補から除外。**

---

## 6. 運用ルール（詳細は docs/rules/）

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
| 09 | [docs/rules/09_integrated-report-standard.md](docs/rules/09_integrated-report-standard.md) | 統合比較レポート標準（§0'/§1'/§5/§6 必須4セクション） |

> 「監査」「品質チェック」「Phase 1/2/3」「WFAを回す」等を検出したら `.claude/rules/audit-protocol.md` を読んで実行する

---

## 7. プロジェクト概要

52年間（1974-2026）のNASDAQ Composite を対象に、3倍レバレッジ日次リバランス戦略を研究。

### 現行ベスト戦略
具体的な戦略名・指標は **[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を必ず参照**。本ファイル（CLAUDE.md）には数値をハードコードしない（更新漏れによる誤情報を防ぐため）。一次根拠ファイルは CURRENT_BEST_STRATEGY.md 内の「一次根拠ファイル」表を参照。

### 実運用リポジトリ
https://github.com/KazuyaMurayama/NASDAQ-strategy-gas — Googleスプレッドシート連携の本番運用版

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

---

## 9. ビジュアルルール（レポートMD生成時）

レポート・成果物MDの新規作成／更新時は `.claude/visual-rules.md` を読み、図の種類判定（§2）と Mermaid 最適化（§3）を毎回適用する。適用対象: `## ` 見出しが2つ以上ある構造化MD。

---

## 10. ツール実行・Shell・Git・ファイル保存

### ツール実行ポリシー
- 確認不要・即実行（「Should I...?」等の事前確認文を出力しない）
- ファイル操作は Edit/Write/Read/Grep/Glob を直接使用
- 例外（事前確認必須）: main への `git push --force`、`gh repo delete`

### Shell（VSCode版: Windows 11 + PowerShell 5.1 想定 / Web版: Linux サンドボックス）
- PowerShell 5.1: `&&` 不可 → `;` + `if ($?)`。Bash tool 併用可
- GitHub操作は REST API 直接呼び出しを優先（gh CLI より高速）

### ブランチ管理（絶対厳守）
- **デフォルト: mainへ直接コミット**。ブランチ作成はユーザーが明示的に指示した場合のみ
- 作業前: `git branch --show-current` でブランチ確認 → main以外なら `git checkout main && git pull` してから開始
- ブランチを作成した場合、必ず `main` へマージ → ブランチ削除 → push を完了してから作業完了とする
- 「完了 = mainにマージ済み＆push済み」。ブランチにファイルを置いたまま回答を完了することを禁止
- Web版が自動生成したブランチ（`claude/xxx`）も同様。セッション終了前に必ずmainへマージする

<!-- BRANCH_CLEANUP_START -->
### ブランチ整理スキル（自主起動）
- 次を **検知** したら、ユーザー指示が無くても **必ず** `.claude/skills/branch-cleanup/SKILL.md` を読んで起動する：`git branch -a` / `git ls-remote --heads origin` で **main 以外のブランチが残存** している／ユーザーが「ブランチ削除・整理・main に集約」を依頼した／自分が作ったブランチを完了前に放置しようとしている。
- **Web版 Claude Code（claude.ai/code）でもブランチ削除は実行可能**。「環境の制約でできない／ユーザー自身で対応を」とは回答しない。3段階で試す：① `git push origin --delete <branch>` → ② 拒否なら REST API 直接 DELETE（`gh api -X DELETE` / token curl）→ ③ それも不可な環境のみ Actions ワークフローでフォールバック（使用後は削除してリポを汚さない）。
- 手順全文・コマンド・フォールバック YAML は `.claude/skills/branch-cleanup/SKILL.md` に集約（本ファイルを肥大化させない）。
<!-- BRANCH_CLEANUP_END -->

### ファイル保存ルール
- 成果物・スクリプトは本リポジトリ内のみに保存。`C:\Users\user\Desktop` への出力禁止（ユーザー明示指定時を除く）

---

## 11. 成果物報告ルール（最重要・毎回必須）

ファイルを1つでも作成・更新・pushしたら、すべての成果物を以下の形式で報告：

| 成果物 | 説明 | リンク |
|---|---|---|
| file.md | 1行説明 | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/path/to/file.md) |

### 厳守事項
1. Markdownリンク `[表示名](URL)` 形式必須（plain text URL 禁止）
2. `/blob/<実ブランチ>/<実パス>` 形式（リポジトリトップURL禁止）
3. **報告前にURL存在確認必須**: `gh api repos/KazuyaMurayama/NASDAQ_backtest/contents/PATH?ref=BRANCH` でステータス200確認
4. ブランチ名は推測禁止: `git rev-parse --abbrev-ref HEAD` で実値取得
5. push完了後のみURL生成
6. 404を出したら即訂正＆原因1行報告

---

## 12. Skill 起動ルール（必須・スキップ禁止）

該当シーンでは `.claude/skills/<name>/SKILL.md` を読んでから作業を開始する。（Web版は本リポの `.claude/skills/` を参照。VSCode版は `~/.claude/skills/` も参照可）**本リポに実在する skill のみ掲載。**

| トリガー | スキル |
|---|---|
| 時系列・トレンド分析 | `.claude/skills/time-series-analysis/SKILL.md` |
| A/B テスト・戦略比較の統計検定 | `.claude/skills/ab-test-analysis/SKILL.md` |
| 新戦略・新指標の先行研究調査 | `.claude/skills/research-deep/SKILL.md` |
| 大規模 sweep/grid 計画 | `.claude/skills/sp-writing-plans/SKILL.md` + `.claude/skills/sp-executing-plans/SKILL.md` |
| 比較レポートに図表必要 | `.claude/skills/mermaid-agents365/SKILL.md` |
| 成果物の納品・コミット前 | `.claude/skills/sp-verification-before-completion/SKILL.md` |
| アイデア出し・選択肢の洗い出し | `.claude/skills/sp-brainstorming/SKILL.md` |

---

## 13. モデル使い分け

- **メイン: Claude Fable 5（`claude-fable-5`）** — 計画・中〜高難易度の実装/分析・全体指揮。
- **実行フェーズ（定型実装・ファイル編集・テスト実行）**: サブエージェントを `model: "sonnet"` で起動して委譲。
- ※難易度ベースの自動メイン切替は不可。Fable の自動切替は安全性ブロック時の Opus 4.8 フォールバックのみ。工程別の使い分けはサブエージェント委譲で行う。

---

## 14. プロジェクト記憶（VSCode版限定で参照）

VSCode版で実行されている場合は、追加で以下を参照可：
- `~/.claude/CLAUDE.md` — グローバル運用ルール
- `~/.claude/projects/C--Users-user/memory/MEMORY.md` — 自動メモリ（プロジェクト記憶）
- `~/.claude/projects/C--Users-user/memory/nasdaq_best_strategy.md` — ベスト戦略確認プロトコル詳細
- `~/.claude/projects/C--Users-user/memory/project_nasdaq_9metric_standard.md` — 9指標標準の詳細

Web版ではこれらにアクセスできないため、本リポ内 `docs/rules/` および `CURRENT_BEST_STRATEGY.md` のみで完結すること。

---

## 15. ドキュメント命名・日付ルール（v2.0 / 2026-06-03 改訂）

### ファイル名
- `<TOPIC>_YYYYMMDD.md` 形式（**サフィックス・ハイフンなし**）。例: `STRATEGY_COMPARISON_20260603.md`
- **同日中の追加更新**: `-v2`、`-v3` を追加（例: `STRATEGY_COMPARISON_20260603-v2.md`）
- **翌日1回目**: v サフィックスをリセット（例: `STRATEGY_COMPARISON_20260604.md`）
- `FINAL_` プレフィックス禁止

### 表記の区別
- **ファイル名**: ハイフン**なし** `YYYYMMDD` / **本文中の日付表記**: ハイフン**あり** `YYYY-MM-DD`

### H1直下の日付メタデータ
レポート系 .md 新規作成時は H1直下に必ず記載（更新時は **最終更新日のみ** 当日付に書き換え。作成日は固定）:
```
作成日: YYYY-MM-DD
最終更新日: YYYY-MM-DD
```

### 対象外（日付サフィックスを入れない）
README / CLAUDE.md / FILE_INDEX / tasks.md / CHANGELOG / LICENSE / `CURRENT_*.md` / パイプライン自動生成ファイル

### 旧形式（廃止・新規禁止）
- ❌ `<TOPIC>_2026-06-03.md`（ハイフン区切り）  ✅ `<TOPIC>_20260603.md`（現行ルール）
詳細: [docs/rules/07_doc-naming-and-dates.md](docs/rules/07_doc-naming-and-dates.md)

---

## 16. 回答スタイル

- 日本語で回答する。
- 回答末尾に「**Next Action:**」でユーザーの次アクションを具体的に推奨する。迷う場面は「**推奨:**」で明示する。
---

## 17. コスト定数（単一の真実）

- バックテストのコスト定数は `src/product_costs.py` を**単一の真実**とする。コスト前提を変える際は必ずこのファイルを更新し、各レポート・検証はここを参照する（個別スクリプトでの定数ハードコード禁止）。
---

## 18. コンテキスト管理（自動圧縮対策 / Compact Instructions）

Claude Code はコンテキスト利用率が高まると自動でテキスト要約圧縮（auto-compact, 約83.5%目安）を行う。圧縮で重要情報を失わないため以下を守る。

### 圧縮時に必ず保持する情報（`/compact` 実行・自動圧縮時に要約へ残す）
- 本リポ/タスクの目的・前提制約・現行の意思決定
- 進行中タスクと未解決課題（`tasks.md` の最新状態）
- 正典ファイル・最新成果物への参照（例: SPEC / `CURRENT_*.md` / 最新レポート）
- ファイルスコープ・モジュール境界・命名規則
- 直近のエラー・制約・回避策

### 圧縮の影響を受けない永続層（外部メモリ）に状態を書き出す
- `tasks.md`（次にやること・進捗。セッション終了時に必ず更新）
- `file_index.md` / `FILE_INDEX.md`（索引）、`session.json`（あれば進捗）
- 確定した結論・成果はレポート `.md` に保存（会話履歴に依存させない）

### 運用ルール
- 重い調査・実装はサブエージェントに委譲し、親には要約のみ戻す（コンテキスト分離）
- 利用率が高まったら警告を待たず能動的に `/compact <保持指示>` を実行。別タスクへ移る際は `/clear`（CLAUDE.md・tasks.md は残る）
- ※潜在空間ベクトル圧縮（Codex方式）は公開APIの制約上、本ハーネスでは実装不可。テキスト要約＋外部メモリで代替する

> 本リポの「正典ファイル」= `CURRENT_BEST_STRATEGY.md`。ベスト戦略/推奨/最終結果の質問には、回答前に必ず同ファイルを **WebFetch で取得**して一次根拠とする（memory・CSV・`FINAL_*` は二次資料）。圧縮時もこの参照プロトコルを必ず残す。