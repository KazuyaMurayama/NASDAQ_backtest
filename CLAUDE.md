# nasdaq_backtest — 運用ルール入口

NASDAQ 3倍レバレッジ戦略のバックテスト研究リポジトリ。**main 単一ブランチ運用**。

## セッション開始時の参照順序
1. **FILE_INDEX.md** — 全ファイルの所在・優先度
2. **tasks.md** — 未完了タスク・進捗
3. このCLAUDE.md — ルール入口

## 運用ルール（詳細はスキルファイル）

| # | ファイル | 内容 |
|---|---------|------|
| 01 | [docs/rules/01_response-basics.md](docs/rules/01_response-basics.md) | 回答の基本ルール |
| 02 | [docs/rules/02_task-management.md](docs/rules/02_task-management.md) | タスク管理 |
| 03 | [docs/rules/03_file-index.md](docs/rules/03_file-index.md) | ファイルインデックス管理 |
| 04 | [docs/rules/04_deliverables-and-models.md](docs/rules/04_deliverables-and-models.md) | 成果物・モデル・出力フォーマット |
| 05 | [docs/rules/05_git-and-execution.md](docs/rules/05_git-and-execution.md) | Git操作・実行計画 |

## プロジェクト概要

47年間（1974-2021）のNASDAQ Composite を対象に、3倍レバレッジ日次リバランス戦略を研究。

### 推奨戦略: Ens2(Asym+Slope)
- Sharpe 1.031、CAGR 28.58%、MaxDD -48.17%、Worst5Y +1.41%
- 構成: DD × AsymEWMA VT × SlopeMult
- 詳細: [FINAL_RESULTS_2026-02-06.md](FINAL_RESULTS_2026-02-06.md)

### 実運用リポジトリ
https://github.com/KazuyaMurayama/nasdaq-strategy-gas
