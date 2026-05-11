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

## 📛 ドキュメント命名規則（再発防止）

新規レポート作成時:

1. **`FINAL_` プレフィックスは禁止** — 「FINAL」と名乗ったファイルが後で覆されると参照地獄になる
2. **`<TOPIC>_YYYY-MM-DD.md` または `REPORT_YYYY-MM-DD.md` 形式**を使用
3. **旧レポートを置き換える時は、必ず旧レポート冒頭に SUPERSEDED ヘッダを追加**:
   ```markdown
   > ⛔ **このドキュメントは SUPERSEDED (置換済み) です**
   > - 廃止日: YYYY-MM-DD
   > - 後継ファイル: [新レポート名](新レポート.md)
   > - 現行ベスト戦略: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)
   > - 廃止理由: <一行で理由>
   ```
4. **CURRENT_BEST_STRATEGY.md と tasks.md を同時更新**

## 運用ルール（詳細はスキルファイル）

| # | ファイル | 内容 |
|---|---------|------|
| 01 | [docs/rules/01_response-basics.md](docs/rules/01_response-basics.md) | 回答の基本ルール |
| 02 | [docs/rules/02_task-management.md](docs/rules/02_task-management.md) | タスク管理 |
| 03 | [docs/rules/03_file-index.md](docs/rules/03_file-index.md) | ファイルインデックス管理 |
| 04 | [docs/rules/04_deliverables-and-models.md](docs/rules/04_deliverables-and-models.md) | 成果物・モデル・出力フォーマット |
| 05 | [docs/rules/05_git-and-execution.md](docs/rules/05_git-and-execution.md) | Git操作・実行計画 |

## プロジェクト概要

52年間（1974-2026）のNASDAQ Composite を対象に、3倍レバレッジ日次リバランス戦略を研究。

### 現行ベスト戦略 → [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を参照

**DH Dyn 2x3x [A]（Approach A・閾値 0.15）** — CAGR +30.81%, Sharpe 1.298, MaxDD -31.36%, Worst5Y +4.77% (FULL 1974-2026)

一次根拠: [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md) / [YEARLY_RETURNS_REPORT_2026-04-20_v3.md](YEARLY_RETURNS_REPORT_2026-04-20_v3.md)

> ⚠️ 過去版にあった「推奨戦略: Ens2(Asym+Slope)」は 2026-04-21 に廃止されました。詳細は [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) の「廃止された旧推奨」セクションを参照。

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


## 🚀 自動実行ポリシー

### 確認不要で即実行する操作
- 全ファイル操作（作成・編集・削除）
- 全シェルコマンド（PowerShell, Bash, git, gh, npm, node, python）
- Git操作: add / commit / push（featureブランチ）/ pull / fetch / merge / branch -D / reset --hard
- GitHub操作: gh pr create / gh api 全般 / ブランチ削除
- パッケージ操作: npm install / pip install
- Web検索・フェッチ
- バックグラウンドプロセス起動

### 事前確認が必要な操作（例外のみ）
- `git push --force` を main / master ブランチに対して実行する場合
- `gh repo delete` 実行時

### 動作原則
- 計画提示（簡潔）→ 即実行 → 結果報告 のフロー厳守
- 事前確認文（「Should I run...?」等）を出力しない
- エラー時は即再試行 or 別アプローチで対応、判断が必要な場合のみ報告

## ドキュメント日付ルール

レポート・分析・調査系 .md ファイルを新規作成する際は、H1直下に必ず記載:

```
作成日: YYYY-MM-DD
最終更新日: YYYY-MM-DD
```

- 更新時は **最終更新日のみ** を当日付に書き換える（作成日は固定）
- 除外: README / CLAUDE.md / FILE_INDEX / tasks.md / CHANGELOG / LICENSE

## 作業品質ルール

### Git・ブランチ管理
- 作業前: `git branch --show-current` でブランチ確認 → main以外なら `git checkout main && git pull` してから開始。

### ファイル特定（編集前）
- ユーザー発話のキーワード全てをファイル名と照合してから編集。キーワード不完全一致・候補不確かなら必ず確認。

### 成果物報告
- ファイル作成・更新・push後は必ず3列表で報告: `| 成果物 | 説明 | リンク |`
- リンクは `/blob/<実ブランチ>/<パス>` 形式。報告前に `gh api repos/OWNER/REPO/contents/PATH?ref=BRANCH` で存在確認。push前はURL生成しない。

### ドキュメント品質
- UIパス・コマンド・設定名は公式ドキュメントで確認後に記載。確認不可なら「[要確認]」と明記。
- OS/環境制約（例: Windows専用）をタスク開始時に確認。完成後に `brew`/`Cmd`/`macOS` 等をgrepして除去。

<!-- GLOBAL_RULES_START -->
## Global Rules (synced from ~/.claude/CLAUDE.md)

### 成果物報告（最重要・毎回必須）
ファイルを1つでも作成・更新・pushしたら、**すべて**を3列表で報告する。例外なし。

| 成果物 | 説明 | リンク |
|---|---|---|
| ファイル名 | 1行説明 | [開く](https://github.com/OWNER/REPO/blob/BRANCH/PATH) |

厳守事項（違反＝再提出）:
1. Markdownリンク `[表示名](URL)` 形式必須。plain URL禁止。
2. `/blob/<実ブランチ名>/<実パス>` 形式。トップURL禁止。
3. 提示前に `gh api repos/OWNER/REPO/contents/PATH?ref=BRANCH` で存在確認。
4. ブランチ名は `git rev-parse --abbrev-ref HEAD` で取得。推測禁止。
5. push完了後のみURL生成。未pushはローカル絶対パス＋「（ローカル）」明記。
6. 404発生時は即訂正＋原因1行報告。

### ツール実行
- 確認不要・即実行。「Should I...?」等の事前確認文は出力しない。
- 例外（事前確認必須）: main/masterへの `git push --force`、`gh repo delete`。
- 長時間処理は `run_in_background: true` を積極使用。

### モデル・サブエージェント
- メイン Opus / 探索・検索・テスト系サブは Sonnet (`model: "sonnet"`)。
- サブ起動promptに必ず明記:「成果物は3列表報告・URL検証必須・Markdownリンク形式」

### 回答スタイル
- 回答末尾に「**Next Action:**」でユーザーの次アクションを具体推奨。
<!-- GLOBAL_RULES_END -->
