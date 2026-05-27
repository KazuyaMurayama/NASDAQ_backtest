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

## 🔍 戦略監査 — Claude Code 自動実行プロトコル（毎回必須・ユーザー確認不要）

本セクションは Claude Code 自身への命令である。ユーザーの確認を待たず、トリガー検出と同時に Phase 1→2→3 を Bash ツールで順次実行し、Completion Gate を満たすまで「完了」と報告しない。詳細仕様は `audit_results/AUDIT_FRAMEWORK.md`。

### 🟢 自動実行トリガー（以下いずれかをユーザー発話に検出した瞬間、確認なしで Phase 1 を起動する）

- 「監査」「品質チェック」「品質レポート」「品質確認」
- 「過学習」「過学習検定」「実現性レビュー」「実現性チェック」「OOS 評価」「OOS 検証」
- 「Phase 1」「Phase 2」「Phase 3」「ロジック検証」「Bootstrap」「Permutation」「WFA を回す」「感度分析」
- 「新戦略バリアント追加」「`E5`/`F9`/`G7` を検証」「`build_<NEW>_strategy_assets` を作る」
- 「正式 Active に昇格」「CURRENT_BEST_STRATEGY に登録」

トリガー検出後の最初の応答は「Phase 1 を起動します」とだけ宣言し、即 Bash 実行を開始する。「実行してよいですか」は絶対に問わない。

### 🔴 鉄則（Claude Code 自身がこれを破った場合は自己レポートを再提出する）

1. Claude Code は `python src/audit/check_*.py` を**自分で Bash 実行**する。コードスニペットを提示してユーザーに実行依頼することを禁止する。
2. Claude Code は出力ファイルパスの実在を `Glob` または Bash `ls` で検証する。実在しないまま ✅ を書くことを禁止する。
3. Claude Code は「旧戦略値 + オフセット推定」を品質レポートに書かない。新バリアントは `force_rebuild=True` で再生成する。
4. Claude Code は Phase X 完了確認 → Phase X+1 起動を**ユーザー応答を待たず**に連結実行する。

### Phase 1 自動実行（ロジック正しさ）

1. `python src/audit/check_logic_delay_and_causal.py` を Bash で実行する。
2. `audit_results/LOGIC_CHECK_YYYYMMDD.md` と `audit_results/logic_check.yaml` の生成を `Glob` で確認する。
3. ファイル未生成または FAIL の場合は Phase 2/3 に進まず、エラーを 1 行で報告して修正してから再実行する。
4. PASS を確認したらユーザーに問わず即 Phase 2 を起動する。

### Phase 2 自動実行（実現性）

新バリアントの場合は `check_<new>_broker_matrix.py` を `Write` で先に生成してから実行する。

1. `python src/audit/check_realworld_report_corpus.py` を実行 → `CHECK_CORPUS_YYYYMMDD.md` 実在確認。
2. `python src/audit/check_<variant>_broker_matrix.py` を実行 → `<VARIANT>_BROKER_MATRIX_YYYYMMDD.md` 実在確認。
3. `python src/audit/check_sim_margin_dynamics.py` を実行 → `MARGIN_DYNAMICS_YYYYMMDD.md` 実在確認。
4. 全 3 ファイル実在確認後、即 Phase 3 を起動する。

### Phase 3 自動実行（過学習検出）

1. `python src/audit/check_overfitting_<variant>_bootstrap.py` を実行 → Bootstrap MD 実在確認。
2. `python src/audit/check_overfitting_<variant>_permutation.py` を実行 → Permutation MD 実在確認（(d) 同時置換含むこと）。
3. `python src/g_wfa_<variant>.py` を実行 → WFA YAML 実在確認。
4. `python src/audit/check_<variant>_parameter_sensitivity.py` を実行 → Sensitivity MD 実在確認。
5. `python src/audit/check_overfitting_<variant>_summary.py` を実行 → Summary MD 実在確認。

### 🟡 Completion Gate（全条件を満たすまで「監査完了」と報告しない）

Claude Code 自身が以下を `Glob` / `Read` で機械的に検証する。欠落があれば該当 Phase に戻って再実行する。

- [A] Phase 1〜3 の全出力 `.md` が `audit_results/` 配下に**当日日付で実在する**。
- [B] 各 Phase の `.yaml` 機械可読データが実在する。
- [C] `logic_check.yaml` の `overall_verdict: PASS` が記録されている。
- [D] 新バリアントの場合、`_audit_strategy.py` に `build_<new>_strategy_assets()` が定義済み。
- [E] 品質レポート末尾の実施証跡表に書かれた全パスを `Glob` で検証し、存在しないパスが 0 件。

条件 A〜E のいずれかが欠けたら、欠落項目を 1 行で報告し Claude Code 自身が即再実行する。ユーザーに「実行してください」とは絶対に書かない。

### 🟠 例外処理（Claude Code 自身の判断ルール）

- キャッシュ `force_rebuild=True` で 5 分超を見込む場合は `run_in_background: true` で起動し、「Phase X 再構築開始（推定 N 分）」とだけ報告してから継続待機する。
- スクリプトが例外で停止した場合は、トレースバックを読んで `Edit` で修正し再実行する。修正不能な場合のみユーザーに 1 行で問う。

### 新戦略バリアント追加時（自動実行タスク一覧）

バリアント識別子 `<NEW>` を検出したら Claude Code 自身が順次実行する：

1. `src/audit/_audit_strategy.py` を `Edit` で更新: `<NEW>_PARAMS` / `<NEW>_CACHE_FILE` / `<NEW>_SHARPE_OOS_EXPECTED` / `build_<new>_strategy_assets()` / `build_<new>_strategy_nav_for_scenario()` / `build_<new>_assets_with_override()` を追加する。
2. `check_<new>_broker_matrix.py` / `check_overfitting_<new>_bootstrap.py` / `check_overfitting_<new>_permutation.py` / `check_overfitting_<new>_summary.py` / `check_<new>_parameter_sensitivity.py` / `src/g_wfa_<new>.py` を `Write` で生成する（E4 版をテンプレートに差し替え）。
3. `check_logic_delay_and_causal.py` に `<NEW>` Block を追加する。
4. 上記完了後、Phase 1 自動実行手順に合流する。

### Next Action の 2 分類（報告末尾は必ずこの 2 ブロックで書く・混在禁止）

```
## 🤖 Claude Code 自動継続タスク（次トリガーで確認なし実行）
- 未実施 Phase があれば該当スクリプトを再実行する
- 出力ファイル欠落時は force_rebuild=True で再生成する
- 実施証跡表の存在しないパスを修正する

## 👤 ユーザー判断が必要なタスク（選択肢提示のみ・Claude Code は実行しない）
- 戦略を CURRENT_BEST_STRATEGY.md に正式 Active として昇格させるか
- GitHub に push してよいか（force push など破壊的操作の場合）
- 棄却・採用・追加検証の方針判断
```

「CLAUDE.md を見てください」「次回ご自身でチェックリストに従ってください」は両ブロックとも禁止文。Claude Code 側の継続タスクは次回トリガー検出時に自分で実行する責務である。

<!-- SKILLS_RULES_START -->
## Skill 起動ルール（v1.0 / 2026-05-27）
- **時系列・トレンド分析** → `time-series-analysis`
- **A/B テスト・戦略比較の統計検定** → `ab-test-analysis`
- **新戦略/新指標の先行研究調査** → `research-deep`
- **大規模 sweep/grid 計画時** → `sp-writing-plans` でフェーズ分割
- **比較レポートの図表** → `mermaid-agents365`
- **コード変更後（プロダクション影響あり）** → `code-review` 必須
<!-- SKILLS_RULES_END -->
