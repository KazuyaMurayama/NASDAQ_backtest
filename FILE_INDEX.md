# File Index — nasdaq_backtest

最終更新: 2026-04-21

## 📁 Living Docs（毎回参照）
| ファイル | 役割 |
|---------|------|
| [CLAUDE.md](CLAUDE.md) | 運用ルール入口 |
| [tasks.md](tasks.md) | タスク管理 |
| [FILE_INDEX.md](FILE_INDEX.md) | このファイル |

## 📘 運用ルール（docs/rules/）
- [01_response-basics.md](docs/rules/01_response-basics.md)
- [02_task-management.md](docs/rules/02_task-management.md)
- [03_file-index.md](docs/rules/03_file-index.md)
- [04_deliverables-and-models.md](docs/rules/04_deliverables-and-models.md)
- [05_git-and-execution.md](docs/rules/05_git-and-execution.md)

## 📊 戦略研究ドキュメント（優先度順、新しいもの優先）
| ファイル | 内容 |
|---------|------|
| [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md) | **Approach A 内リバランス閾値スイープ**(推奨 0.15) |
| [YEARLY_RETURNS_REPORT_2026-04-20_v2.md](YEARLY_RETURNS_REPORT_2026-04-20_v2.md) | 8戦略年次/月次リターン v2（Approach A/B 併記） |
| [LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md](LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md) | レバビン別 × ベスト戦略 NAV 前方 CAGR V4 rev3 |
| [CAGR_DISCREPANCY_ANALYSIS_2026-04-20.md](CAGR_DISCREPANCY_ANALYSIS_2026-04-20.md) | CAGR乖離 原因分析（Approach A vs B） |
| [APPROACH_A_PROPOSAL_2026-04-20.md](APPROACH_A_PROPOSAL_2026-04-20.md) | **GAS Approach A 切替設計書** |
| [FINAL_RESULTS_2026-02-06.md](FINAL_RESULTS_2026-02-06.md) | 旧推奨戦略 Ens2(Asym+Slope) |
| [LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md](LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md) | レバレッジビン別 前方リターン分析 V3 |
| [3x_NASDAQ_Strategy_Research_Summary_2026-02-06.md](3x_NASDAQ_Strategy_Research_Summary_2026-02-06.md) | 研究総括 |
| [R4_RESULTS_SUMMARY_2026-02-06.md](R4_RESULTS_SUMMARY_2026-02-06.md) | R4 55戦略比較 |
| [plan_2026-02-12.md](plan_2026-02-12.md) | 研究計画 |
| [docs/TQQQ_execution_guide.md](docs/TQQQ_execution_guide.md) | TQQQ/Gold/TMF 実運用ガイド |

## 🔧 src/ コードベース
| ファイル | 役割 |
|---------|------|
| [src/backtest_engine.py](src/backtest_engine.py) | **コアエンジン**（全戦略実装） |
| [src/test_threshold_sweep_A.py](src/test_threshold_sweep_A.py) | **Approach A 閾値スイープ**（5閾値×6期間×7指標） |
| [src/gen_yearly_monthly_v2.py](src/gen_yearly_monthly_v2.py) | 年次/月次リターン生成 v2（Approach A/B 併記） |
| [src/analyze_leverage_bin_v4.py](src/analyze_leverage_bin_v4.py) | レバビン分析 V4 rev3 |
| [src/test_ens2_strategies.py](src/test_ens2_strategies.py) | Ens2 戦略テスト |
| [src/test_realistic_product.py](src/test_realistic_product.py) | 実商品条件テスト |
| [src/run_r4_backtest.py](src/run_r4_backtest.py) | R4 検証ランナー |
| [src/test_delay_robust.py](src/test_delay_robust.py) | 遅延耐性テスト |
| [src/analyze_leverage_vs_return_v3.py](src/analyze_leverage_vs_return_v3.py) | レバレッジ vs リターン分析 V3 |

※ その他 src/ 各種テスト・生成スクリプトは GitHub 上で確認

## 🧰 運用ツール
| ファイル | 役割 |
|---------|------|
| [.github/workflows/migrate-branches-to-main.yml](.github/workflows/migrate-branches-to-main.yml) | 旧ブランチ統合ワークフロー（使用済） |
| [scripts/migrate_large_files.sh](scripts/migrate_large_files.sh) | 旧ブランチ取込スクリプト（使用済） |

## 🗄️ archive/
| ファイル | 役割 |
|---------|------|
| `archive/2026-04-19_CLAUDE_main.md` | 旧 main CLAUDE.md（5510B、詳細ルール入り）歴史的参照用 |
| `archive/2026-04-19_gitignore_main.txt` | 旧 main .gitignore |
