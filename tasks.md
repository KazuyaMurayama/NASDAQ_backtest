# Tasks — nasdaq_backtest

最終更新: 2026-04-19

## 🔴 In Progress
（なし）

## 🟡 Pending
- [ ] 2026年データへの拡張（NASDAQ_extended_to_2026.csv は branches 参照）
- [ ] 新規テスト群の main 統合完了（下記「未集約」参照）
- [ ] Ens2 戦略の OOS 検証（2022-2026）

## 📋 未集約ファイル（branch 参照）
> 大量テキスト/CSV の push API タイムアウトにより、以下ファイルは main に未取込。
> `scripts/migrate_large_files.sh` をローカル実行すれば取込可能。

- src/ 新規Python（42ファイル）— claude/create-file-index-vVbP4
- 中小CSV（~30ファイル）— claude/create-file-index-vVbP4
- ルートMD 8ファイル（CAGR_IMPROVEMENT_PLAN 等）— claude/create-file-index-vVbP4
- XLSX 2ファイル — claude/create-file-index-vVbP4
- project-overview 固有（overfitting/OOS系）— claude/project-overview-s2l8s

## ✅ Completed
- 2026-04-19: 運用ルール整備（CLAUDE.md / docs/rules/ 群 / tasks.md / FILE_INDEX.md）
- 2026-04-19: archive/2026-04-19_* 旧main CLAUDE.md と .gitignore 退避
- 2026-04-19: 大容量CSV除外＋migrate スクリプト追加
- 2026-04-16: create-file-index-vVbP4 ブランチで FILE_INDEX.md 初版作成
- 2026-04-09: 研究ドキュメント日付サフィックス付与
