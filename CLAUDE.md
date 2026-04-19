# NASDAQ Backtest Project — Claude Code ガイド

## 必読ルールファイル（セッション開始時に必ず読み込む）

| ファイル | 内容 |
|---------|------|
| [`.claude/rules/response-rules.md`](.claude/rules/response-rules.md) | 回答フォーマット・成果物報告・名前表記 |
| [`.claude/rules/git-rules.md`](.claude/rules/git-rules.md) | Git操作ルール（ブランチ作成禁止等） |
| [`.claude/rules/workflow-rules.md`](.claude/rules/workflow-rules.md) | タスク管理・モデル使い分け・エージェント構成 |

## セッション開始チェックリスト

1. 上記ルールファイルを読み込む
2. [`FILE_INDEX.md`](./FILE_INDEX.md) でファイル構成を把握する
3. [`tasks.md`](./tasks.md) で未完了タスク・優先度を確認する

## プロフィール

- **名前**: 男座員也（おざ かずや / Kazuya Oza）
- **拠点**: 東京都小平市（花小金井近辺）
- **職種**: データサイエンティスト・生成AIコンサルタント（フリーランス）
- **事業**: AIコンサルタント（月単価300万円×稼働50%以内）／SaaS事業／レバレッジ投資／不動産投資

## GASスプレッドシート

ID: `1YqwZ2EGKVFs36tTvUfup28g0GtXZwReiVOBI6eNzmVI`（閲覧: リンクを知っている全員）

WebFetch URL:
```
https://docs.google.com/spreadsheets/d/1YqwZ2EGKVFs36tTvUfup28g0GtXZwReiVOBI6eNzmVI/gviz/tq?tqx=out:json&sheet={シート名}
```

| シート名 | 内容 |
|---------|------|
| `PriceHistory` | NASDAQ日次終値（date, close） |
| `Log` | 戦略シグナル（raw_leverage, new_leverage, w_nasdaq, w_gold, w_bond, rebalanced 等） |
| `State` | 現在の内部状態 |

## GitHub
https://github.com/KazuyaMurayama/NASDAQ_backtest
