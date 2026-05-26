# FILE_INDEX — NASDAQ_backtest

> ⚠️ このファイルは半自動生成です。「最新追加ファイル」セクション（末尾）は手動更新。
> 🎯 **最優先参照ファイル: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)** — 現行ベスト戦略の正典。「ベスト戦略は？」と問われたら必ずここから読むこと。
> 🧭 **研究文脈: [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md)** — 実験系統図・棄却理由・引き継ぎ読書順序。新セッション開始時のオリエンテーション用。
> 📋 **戦略台帳: [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md)** — 全検証済み戦略（Active/Shortlisted/Rejected）の台帳。新検証着手前に重複チェック必須。
> 📐 **評価基準: [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md)** — コスト・期間・指標計算の標準定義 (v1.1)。全検証はここの前提に従う。

| 項目 | 値 |
|---|---|
| リポジトリ | KazuyaMurayama/NASDAQ_backtest |
| ブランチ | main |
| 総ファイル数 | 約 300 |
| 最終更新 | 2026-05-24 |
| 管理者 | 男座員也（Kazuya Oza） |

---

## 🎯 最優先参照ファイル（このリポジトリで最初に見るべき6つ）

| 優先 | ファイル | 役割 |
|---|---|---|
| 1 | [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) | 現行ベスト戦略の正典 (Single Source of Truth) |
| 2 | [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) | 🧭 研究文脈・実験系統図・棄却理由・引き継ぎ読書順序 |
| 3 | [tasks.md](tasks.md) | 未完了タスク・最新進捗 |
| 4 | [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) | 全検証済み戦略の台帳（Active/Shortlisted/Rejected）。新検証着手前の重複チェック必須 |
| 5 | [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) | 評価基準の正典 v1.1（コスト・期間・9指標標準・参考値判定フロー） |
| 6 | [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md), [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md) | 現行ベスト E4 Regime k_lt の一次根拠（採用判定 + G3 WFA PASS）|
| 7 | [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md) | DH Dyn 2x3x [A] 閾値0.15 採用根拠 |

### 新セッション開始時の推奨読書順序

1. `CURRENT_BEST_STRATEGY.md`（3分・現行ベスト把握）
2. `RESEARCH_CONTEXT.md`（5分・実験文脈・棄却理由把握）
3. `tasks.md`（2分・着手点把握）
4. 必要時のみ: `STRATEGY_REGISTRY.md` / `EVALUATION_STANDARD.md` / 本ファイル

## ⛔ SUPERSEDED ファイル（参照禁止・廃止済み）

これらのファイルは冒頭に SUPERSEDED ヘッダ付き。「ベスト戦略」の根拠としては使用しないこと:

| ファイル | 廃止日 | 後継 |
|---|---|---|
| FINAL_RESULTS_2026-02-06.md | 2026-05-11 | CURRENT_BEST_STRATEGY.md |
| R4_RESULTS_SUMMARY_2026-02-06.md | 2026-05-11 | CURRENT_BEST_STRATEGY.md |
| 3x_NASDAQ_Strategy_Research_Summary_2026-02-06.md | 2026-05-11 | CURRENT_BEST_STRATEGY.md |
| CONTEXT_SUMMARY_2026-02-06.md | 2026-05-11 | CURRENT_BEST_STRATEGY.md / tasks.md |
| THRESHOLD_SWEEP_REPORT_2026-04-20.md | 2026-05-11 | THRESHOLD_SWEEP_A_REPORT_2026-04-21.md |
| YEARLY_RETURNS_REPORT_2026-04-20_v2.md | 2026-05-11 | YEARLY_RETURNS_REPORT_2026-04-20_v3.md |

---

## カテゴリ別サマリー

| カテゴリ | ファイル数 |
|---|---|
| Documentation | 41 |
| Code | 79 |
| Data | 86 |
| Asset | 4 |
| Config | 1 |

---

## ディレクトリ構成

```
.
├── .claude/
│   └── rules/
│       ... (3 items)
├── .github/
│   └── workflows/
│       ... (1 items)
├── archive/
│   ├── 2026-04-19_CLAUDE_main.md
│   └── 2026-04-19_gitignore_main.txt
├── data/
│   ├── dgp_daily.csv
│   ├── drn_daily.csv
│   ├── ief_daily.csv
│   ├── lbma_gold_daily.csv
│   ├── sox_daily.csv
│   ├── sp500_daily.csv
│   ├── tmf_daily.csv
│   ├── upro_daily.csv
│   └── vnq_daily.csv
├── docs/
│   ├── rules/
│   │   ... (5 items)
│   └── TQQQ_execution_guide.md
├── scripts/
│   └── migrate_large_files.sh
├── src/
│   ├── analyze_leverage_bin_v4.py
│   ├── analyze_leverage_vs_return_v2.py
│   ├── analyze_leverage_vs_return_v3.py
│   ├── analyze_leverage_vs_return.py
│   ├── backtest_engine.py
│   ├── compare_7strategies.py
│   ├── debug_sharpe_calc.py
│   ├── extend_data.py
│   ├── gen_full_excel.py
│   ├── gen_yearly_excel.py
│   ├── gen_yearly_md.py
│   ├── gen_yearly_monthly_v2.py
│   ├── gen_yearly_monthly_v3.py
│   ├── generate_final_results.py
│   ├── opt_lev2x3x.py
│   ├── opt_phase1_tier1.py
│   ├── opt_phase1_tier2.py
│   ├── opt_phase1_tier3.py
│   ├── opt_phase1_wf_sensitivity.py
│   ├── opt_phase2_dynhybrid.py
│   ├── quality_check.py
│   ├── regime_analysis.py
│   ├── research_gold_bond_timing.py
│   ├── research_step2_bond_timing.py
│   ├── run_r4_backtest.py
│   ├── step_monthly_returns.py
│   ├── step_update_dyn2x3x.py
│   ├── step_yearly_returns.py
│   ├── step1_worst10y.py
│   ├── step2_cagr25_grid.py
│   ├── test_combined_d_f.py
│   ├── test_delay_robust.py
│   ├── test_delay_sensitivity.py
│   ├── test_direction_d.py
│   ├── test_direction_f_oos.py
│   ├── test_direction_f.py
│   ├── test_discrete_leverage.py
│   ├── test_dynamic_portfolio.py
│   ├── test_ens2_strategies.py
│   ├── test_external_oos.py
│   ├── test_external_signals.py
│   ├── test_final_5strategy.py
│   ├── test_final_6strategy.py
│   ├── test_hybrid_dd_vote.py
│   ├── test_improvements_r2.py
│   ├── test_improvements_r3.py
│   ├── test_improvements_r4.py
│   ├── test_improvements.py
│   ├── test_majority_vote_p1.py
│   ├── test_majority_vote_p2.py
│   ├── test_marugoto_leverage.py
│   ├── test_next_improvements.py
│   ├── test_oos_verification.py
│   ├── test_original_method.py
│   ├── test_overfitting_validation.py
│   ├── test_partial_rebalance.py
│   ├── test_portfolio_diversification.py
│   ├── test_realistic_product.py
│   ├── test_rebalance_frequency.py
│   ├── test_regime_comparison.py
│   ├── test_regime_strategy.py
│   ├── test_soxl_addition.py
│   ├── test_threshold_sweep_A.py
│   ├── test_threshold_sweep.py
│   ├── test_validation.py
│   ├── test_vix_integration.py
│   ├── test_vote_ratio_continuous.py
│   ├── test_walkforward_vix_reentry.py
│   ├── validate_c_shared.py
│   ├── validate_c_summary.py
│   ├── validate_c1_calibration.py
│   ├── validate_c2_oos_decade.py
│   ├── validate_c3_reit_bootstrap.py
│   ├── verify_calculations.py
│   ├── verify_max_lev.py
│   ├── yearly_returns_analysis.py
│   ├── yearly_returns_excel_formatted.py
│   └── yearly_returns_excel_ja.py
├── .gitignore
├── 3x_NASDAQ_Strategy_Research_Summary_2026-02-06.md
├── ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md
├── APPROACH_A_PROPOSAL_2026-04-20.md
├── approach_a_sweep_results.csv
├── CAGR_DISCREPANCY_ANALYSIS_2026-04-20.md
├── CAGR_IMPROVEMENT_PLAN_2026-04-09.md
├── CLAUDE.md
├── CONTEXT_SUMMARY_2026-02-06.md
├── delay_robust_results.csv
├── discrete_leverage_results.csv
├── dynamic_portfolio_results.csv
├── ens2_comparison_results.csv
├── external_is_param_sweep.csv
├── external_oos_results.csv
├── external_signal_results.csv
├── FILE_INDEX.md
├── final_5strategy_yearly_returns.csv
├── final_6strategy_yearly_returns.csv
├── FINAL_RESULTS_2026-02-06.md
├── FINAL_RESULTS.csv
├── FINAL_RESULTS.xlsx
├── grid_search_results.csv
├── hybrid_strategy_results.csv
├── improvement_results_a_vix.csv
├── improvement_results_next.csv
├── improvement_results_r2.csv
├── improvement_results_r3.csv
├── improvement_results_r4.csv
├── improvement_results_wf_vix.csv
├── improvement_results.csv
├── lev2x3x_results.csv
├── LEVERAGE_BIN_ANALYSIS_2026-03-19.md
├── LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md
├── leverage_bin_analysis_v2.csv
├── LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md
├── leverage_bin_analysis_v3.csv
├── LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md
├── leverage_bin_analysis_v4.csv
├── leverage_bin_analysis.csv
├── majority_vote_p2_results.csv
├── majority_vote_signals.csv
├── marugoto_leverage_results.csv
├── monthly_returns_oos_v2.csv
├── monthly_returns_oos.csv
├── NASDAQ_Dairy_since1973.csv
├── NASDAQ_extended_to_2026.csv
├── NASDAQ_extended.csv
├── oos_monthly_returns.csv
├── opt_phase1_tier1_results.csv
├── opt_phase1_tier2_results.csv
├── opt_phase1_tier3_results.csv
├── opt_phase1_wf_results.csv
├── opt_phase2_results.csv
├── opt_phase3_results.csv
├── overfitting_validation_results.csv
├── PARAMETER_OPTIMIZATION_PLAN_2026-03-30.md
├── PARAMETER_OPTIMIZATION_REPORT_2026-03-30.md
├── partial_rebalance_results.csv
├── plan_2026-02-12.md
├── portfolio_diversification_results.csv
├── R4_RESULTS_SUMMARY_2026-02-06.md
├── R4_results.csv
├── README.md
├── realistic_product_results.csv
├── rebalance_frequency_results.csv
├── REGIME_ANALYSIS_REPORT_2026-04-04.md
├── regime_analysis_stats.csv
├── regime_strategy_results.csv
├── regime_vs_r4_comparison.csv
├── research_all_corrected.csv
├── research_bond_signals.csv
├── research_correlation_regimes.csv
├── research_gold_signals.csv
├── research_regime_returns.csv
├── research_summary.csv
├── SESSION_SUMMARY_2026-04-04.md
├── signal_correlation_matrix.csv
├── soxl_addition_results.csv
├── step1_worst10y_results.csv
├── step2_baseline.csv
├── step2_partB_ma5.csv
├── step2_partC_ma5_macd.csv
├── step2_partD_full.csv
├── step2_partE_wf.csv
├── step2_static_cagr25_results.csv
├── step2_summary.csv
├── step3_dynamic_cagr25_results.csv
├── STRATEGY_COMPARISON_2026-03-30.md
├── STRATEGY_COMPARISON_CAGR30per_plus_2026-04-02.md
├── strategy_comparison_results.csv
├── STRATEGY_RESEARCH_PLAN_R4_2026-02-06.md
├── tasks.md
├── THRESHOLD_SWEEP_A_REPORT_2026-04-21.md
├── threshold_sweep_A_results.csv
├── THRESHOLD_SWEEP_REPORT_2026-04-20.md
├── threshold_sweep_results.csv
├── Timeout_Prevention.md
├── validation_crisis.csv
├── validation_full.csv
├── validation_oos.csv
├── VolSpike_Yearly_Returns.xlsx
├── vote_ratio_continuous_results.csv
├── YEARLY_RETURNS_7STRATEGIES_v2.xlsx
├── yearly_returns_7strategies.csv
├── YEARLY_RETURNS_7STRATEGIES.xlsx
├── yearly_returns_8strategies_v2.csv
├── yearly_returns_9strategies_v3.csv
├── YEARLY_RETURNS_REPORT_2026-04-01.md
├── YEARLY_RETURNS_REPORT_2026-04-20_v2.md
└── YEARLY_RETURNS_REPORT_2026-04-20_v3.md
```

---

## ファイル詳細

### Documentation (41件)

| ファイル | サイズ | 説明 |
|---|---|---|
| `.claude/rules/git-rules.md` | 911 B | Claude Code 設定・スキル |
| `.claude/rules/response-rules.md` | 1.4 KB | Claude Code 設定・スキル |
| `.claude/rules/workflow-rules.md` | 2.0 KB | Claude Code 設定・スキル |
| `3x_NASDAQ_Strategy_Research_Summary_2026-02-06.md` | 19.2 KB | ⛔ SUPERSEDED → CURRENT_BEST_STRATEGY.md |
| `ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md` | 5.6 KB | Markdown ドキュメント |
| `APPROACH_A_PROPOSAL_2026-04-20.md` | 7.5 KB | Markdown ドキュメント |
| `archive/2026-04-19_CLAUDE_main.md` | 5.4 KB | Markdown ドキュメント |
| `archive/2026-04-19_gitignore_main.txt` | 257 B | ファイル |
| `CAGR_DISCREPANCY_ANALYSIS_2026-04-20.md` | 4.4 KB | Markdown ドキュメント |
| `CAGR_IMPROVEMENT_PLAN_2026-04-09.md` | 13.8 KB | Markdown ドキュメント |
| `CLAUDE.md` | 2.0 KB | Claude Code プロジェクト設定・命名ルール |
| `CONTEXT_SUMMARY_2026-02-06.md` | 2.9 KB | ⛔ SUPERSEDED → CURRENT_BEST_STRATEGY.md |
| `CURRENT_BEST_STRATEGY.md` | — | 🎯 現行ベスト戦略の正典 (Single Source of Truth) |
| `docs/rules/01_response-basics.md` | 708 B | Markdown ドキュメント |
| `docs/rules/02_task-management.md` | 458 B | Markdown ドキュメント |
| `docs/rules/03_file-index.md` | 487 B | Markdown ドキュメント |
| `docs/rules/04_deliverables-and-models.md` | 557 B | Markdown ドキュメント |
| `docs/rules/05_git-and-execution.md` | 1.1 KB | Markdown ドキュメント |
| `docs/TQQQ_execution_guide.md` | 4.0 KB | Markdown ドキュメント |
| `FILE_INDEX.md` | 3.8 KB | （このファイル）全ファイルインデックス |
| `FINAL_RESULTS_2026-02-06.md` | 2.8 KB | ⛔ SUPERSEDED → CURRENT_BEST_STRATEGY.md |
| `LEVERAGE_BIN_ANALYSIS_2026-03-19.md` | 6.0 KB | Markdown ドキュメント |
| `LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md` | 10.2 KB | Markdown ドキュメント |
| `LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md` | 13.5 KB | Markdown ドキュメント |
| `LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md` | 8.5 KB | Markdown ドキュメント |
| `PARAMETER_OPTIMIZATION_PLAN_2026-03-30.md` | 7.7 KB | Markdown ドキュメント |
| `PARAMETER_OPTIMIZATION_REPORT_2026-03-30.md` | 8.2 KB | Markdown ドキュメント |
| `plan_2026-02-12.md` | 2.0 KB | Markdown ドキュメント |
| `R4_RESULTS_SUMMARY_2026-02-06.md` | 5.5 KB | ⛔ SUPERSEDED → CURRENT_BEST_STRATEGY.md |
| `README.md` | 2.2 KB | リポジトリ概要・セットアップ手順 |
| `REGIME_ANALYSIS_REPORT_2026-04-04.md` | 4.8 KB | Markdown ドキュメント |
| `SESSION_SUMMARY_2026-04-04.md` | 13.6 KB | Markdown ドキュメント |
| `STRATEGY_COMPARISON_2026-03-30.md` | 3.7 KB | Markdown ドキュメント |
| `STRATEGY_COMPARISON_CAGR30per_plus_2026-04-02.md` | 36.7 KB | Markdown ドキュメント |
| `STRATEGY_RESEARCH_PLAN_R4_2026-02-06.md` | 18.6 KB | Markdown ドキュメント |
| `tasks.md` | 1.8 KB | タスク管理・セッション履歴 |
| `THRESHOLD_SWEEP_A_REPORT_2026-04-21.md` | 8.0 KB | 🎯 現行ベスト戦略 (DH Dyn 2x3x [A] 閾値0.15) の一次根拠 |
| `THRESHOLD_SWEEP_REPORT_2026-04-20.md` | 5.3 KB | ⛔ SUPERSEDED → THRESHOLD_SWEEP_A_REPORT_2026-04-21.md |
| `Timeout_Prevention.md` | 4.9 KB | タイムアウト対策ガイド |
| `YEARLY_RETURNS_REPORT_2026-04-01.md` | 34.1 KB | Markdown ドキュメント |
| `YEARLY_RETURNS_REPORT_2026-04-20_v2.md` | 12.9 KB | ⛔ SUPERSEDED → YEARLY_RETURNS_REPORT_2026-04-20_v3.md |
| `YEARLY_RETURNS_REPORT_2026-04-20_v3.md` | 11.6 KB | 🎯 51年分の年次比較・BRK ベンチマーク (現行ベスト戦略の根拠) |

### Code (79件)

<details>
<summary>クリックして展開 (79件)</summary>

| ファイル | サイズ | 説明 |
|---|---|---|
| `scripts/migrate_large_files.sh` | 4.0 KB | シェルスクリプト |
| `src/analyze_leverage_bin_v4.py` | 7.6 KB | Python スクリプト |
| `src/analyze_leverage_vs_return_v2.py` | 10.8 KB | Python スクリプト |
| `src/analyze_leverage_vs_return_v3.py` | 17.5 KB | Python スクリプト |
| `src/analyze_leverage_vs_return.py` | 12.8 KB | Python スクリプト |
| `src/backtest_engine.py` | 31.5 KB | Python スクリプト |
| `src/compare_7strategies.py` | 18.0 KB | Python スクリプト |
| `src/debug_sharpe_calc.py` | 6.5 KB | Python スクリプト |
| `src/extend_data.py` | 1.9 KB | Python スクリプト |
| `src/gen_full_excel.py` | 8.5 KB | Python スクリプト |
| `src/gen_yearly_excel.py` | 6.4 KB | Python スクリプト |
| `src/gen_yearly_md.py` | 3.9 KB | Python スクリプト |
| `src/gen_yearly_monthly_v2.py` | 9.9 KB | Python スクリプト |
| `src/gen_yearly_monthly_v3.py` | 15.8 KB | Python スクリプト |
| `src/generate_final_results.py` | 17.6 KB | Python スクリプト |
| `src/opt_lev2x3x.py` | 10.7 KB | Python スクリプト |
| `src/opt_phase1_tier1.py` | 5.0 KB | Python スクリプト |
| `src/opt_phase1_tier2.py` | 7.2 KB | Python スクリプト |
| `src/opt_phase1_tier3.py` | 5.9 KB | Python スクリプト |
| `src/opt_phase1_wf_sensitivity.py` | 7.0 KB | Python スクリプト |
| `src/opt_phase2_dynhybrid.py` | 9.9 KB | Python スクリプト |
| `src/quality_check.py` | 13.9 KB | Python スクリプト |
| `src/regime_analysis.py` | 13.2 KB | Python スクリプト |
| `src/research_gold_bond_timing.py` | 24.5 KB | Python スクリプト |
| `src/research_step2_bond_timing.py` | 22.2 KB | Python スクリプト |
| `src/run_r4_backtest.py` | 15.6 KB | Python スクリプト |
| `src/step_monthly_returns.py` | 5.5 KB | Python スクリプト |
| `src/step_update_dyn2x3x.py` | 4.7 KB | Python スクリプト |
| `src/step_yearly_returns.py` | 9.2 KB | Python スクリプト |
| `src/step1_worst10y.py` | 8.4 KB | Python スクリプト |
| `src/step2_cagr25_grid.py` | 8.6 KB | Python スクリプト |
| `src/test_combined_d_f.py` | 9.2 KB | Python スクリプト |
| `src/test_delay_robust.py` | 37.2 KB | Python スクリプト |
| `src/test_delay_sensitivity.py` | 11.6 KB | Python スクリプト |
| `src/test_direction_d.py` | 10.5 KB | Python スクリプト |
| `src/test_direction_f_oos.py` | 12.9 KB | Python スクリプト |
| `src/test_direction_f.py` | 9.3 KB | Python スクリプト |
| `src/test_discrete_leverage.py` | 13.8 KB | Python スクリプト |
| `src/test_dynamic_portfolio.py` | 14.3 KB | Python スクリプト |
| `src/test_ens2_strategies.py` | 13.1 KB | Python スクリプト |
| `src/test_external_oos.py` | 34.2 KB | Python スクリプト |
| `src/test_external_signals.py` | 18.4 KB | Python スクリプト |
| `src/test_final_5strategy.py` | 8.8 KB | Python スクリプト |
| `src/test_final_6strategy.py` | 10.5 KB | Python スクリプト |
| `src/test_hybrid_dd_vote.py` | 14.3 KB | Python スクリプト |
| `src/test_improvements_r2.py` | 13.6 KB | Python スクリプト |
| `src/test_improvements_r3.py` | 12.4 KB | Python スクリプト |
| `src/test_improvements_r4.py` | 11.6 KB | Python スクリプト |
| `src/test_improvements.py` | 16.3 KB | Python スクリプト |
| `src/test_majority_vote_p1.py` | 14.4 KB | Python スクリプト |
| `src/test_majority_vote_p2.py` | 14.1 KB | Python スクリプト |
| `src/test_marugoto_leverage.py` | 16.3 KB | Python スクリプト |
| `src/test_next_improvements.py` | 16.9 KB | Python スクリプト |
| `src/test_oos_verification.py` | 11.8 KB | Python スクリプト |
| `src/test_original_method.py` | 6.1 KB | Python スクリプト |
| `src/test_overfitting_validation.py` | 30.0 KB | Python スクリプト |
| `src/test_partial_rebalance.py` | 10.0 KB | Python スクリプト |
| `src/test_portfolio_diversification.py` | 14.1 KB | Python スクリプト |
| `src/test_realistic_product.py` | 13.3 KB | Python スクリプト |
| `src/test_rebalance_frequency.py` | 14.2 KB | Python スクリプト |
| `src/test_regime_comparison.py` | 8.6 KB | Python スクリプト |
| `src/test_regime_strategy.py` | 5.9 KB | Python スクリプト |
| `src/test_soxl_addition.py` | 14.2 KB | Python スクリプト |
| `src/test_threshold_sweep_A.py` | 8.8 KB | Python スクリプト |
| `src/test_threshold_sweep.py` | 14.0 KB | Python スクリプト |
| `src/test_validation.py` | 12.6 KB | Python スクリプト |
| `src/test_vix_integration.py` | 19.0 KB | Python スクリプト |
| `src/test_vote_ratio_continuous.py` | 25.4 KB | Python スクリプト |
| `src/test_walkforward_vix_reentry.py` | 15.0 KB | Python スクリプト |
| `src/validate_c_shared.py` | 10.8 KB | Python スクリプト |
| `src/validate_c_summary.py` | 7.3 KB | Python スクリプト |
| `src/validate_c1_calibration.py` | 8.7 KB | Python スクリプト |
| `src/validate_c2_oos_decade.py` | 7.4 KB | Python スクリプト |
| `src/validate_c3_reit_bootstrap.py` | 10.0 KB | Python スクリプト |
| `src/verify_calculations.py` | 21.1 KB | Python スクリプト |
| `src/verify_max_lev.py` | 3.7 KB | Python スクリプト |
| `src/yearly_returns_analysis.py` | 10.5 KB | Python スクリプト |
| `src/yearly_returns_excel_formatted.py` | 17.9 KB | Python スクリプト |
| `src/yearly_returns_excel_ja.py` | 20.9 KB | Python スクリプト |

</details>

### Data (86件)

<details>
<summary>クリックして展開 (86件)</summary>

| ファイル | サイズ | 説明 |
|---|---|---|
| `.github/workflows/migrate-branches-to-main.yml` | 874 B | GitHub Actions ワークフロー |
| `approach_a_sweep_results.csv` | 2.6 KB | CSV データ |
| `data/dgp_daily.csv` | 122.5 KB | CSV データ |
| `data/drn_daily.csv` | 120.7 KB | CSV データ |
| `data/ief_daily.csv` | 122.5 KB | CSV データ |
| `data/lbma_gold_daily.csv` | 263.2 KB | CSV データ |
| `data/sox_daily.csv` | 224.4 KB | CSV データ |
| `data/sp500_daily.csv` | 373.9 KB | CSV データ |
| `data/tmf_daily.csv` | 122.2 KB | CSV データ |
| `data/upro_daily.csv` | 121.1 KB | CSV データ |
| `data/vnq_daily.csv` | 154.2 KB | CSV データ |
| `delay_robust_results.csv` | 28.8 KB | CSV データ |
| `discrete_leverage_results.csv` | 2.9 KB | CSV データ |
| `dynamic_portfolio_results.csv` | 1.1 KB | CSV データ |
| `ens2_comparison_results.csv` | 1.1 KB | CSV データ |
| `external_is_param_sweep.csv` | 9.2 KB | CSV データ |
| `external_oos_results.csv` | 2.5 KB | CSV データ |
| `external_signal_results.csv` | 6.8 KB | CSV データ |
| `final_5strategy_yearly_returns.csv` | 1.8 KB | CSV データ |
| `final_6strategy_yearly_returns.csv` | 2.1 KB | CSV データ |
| `FINAL_RESULTS.csv` | 467 B | CSV データ |
| `grid_search_results.csv` | 5.1 KB | CSV データ |
| `hybrid_strategy_results.csv` | 12.4 KB | CSV データ |
| `improvement_results_a_vix.csv` | 1.6 KB | CSV データ |
| `improvement_results_next.csv` | 1.7 KB | CSV データ |
| `improvement_results_r2.csv` | 1.2 KB | CSV データ |
| `improvement_results_r3.csv` | 1.4 KB | CSV データ |
| `improvement_results_r4.csv` | 1.4 KB | CSV データ |
| `improvement_results_wf_vix.csv` | 803 B | CSV データ |
| `improvement_results.csv` | 1.2 KB | CSV データ |
| `lev2x3x_results.csv` | 9.8 KB | CSV データ |
| `leverage_bin_analysis_v2.csv` | 3.4 KB | CSV データ |
| `leverage_bin_analysis_v3.csv` | 4.3 KB | CSV データ |
| `leverage_bin_analysis_v4.csv` | 8.5 KB | CSV データ |
| `leverage_bin_analysis.csv` | 746 B | CSV データ |
| `majority_vote_p2_results.csv` | 10.3 KB | CSV データ |
| `majority_vote_signals.csv` | 735.0 KB | CSV データ |
| `marugoto_leverage_results.csv` | 1.1 KB | CSV データ |
| `monthly_returns_oos_v2.csv` | 9.3 KB | CSV データ |
| `monthly_returns_oos.csv` | 8.2 KB | CSV データ |
| `NASDAQ_Dairy_since1973.csv` | 898.3 KB | CSV データ |
| `NASDAQ_extended_to_2026.csv` | 993.1 KB | CSV データ |
| `NASDAQ_extended.csv` | 288.4 KB | CSV データ |
| `oos_monthly_returns.csv` | 1.4 KB | CSV データ |
| `opt_phase1_tier1_results.csv` | 11.7 KB | CSV データ |
| `opt_phase1_tier2_results.csv` | 8.2 KB | CSV データ |
| `opt_phase1_tier3_results.csv` | 4.7 KB | CSV データ |
| `opt_phase1_wf_results.csv` | 1.1 KB | CSV データ |
| `opt_phase2_results.csv` | 10.5 KB | CSV データ |
| `opt_phase3_results.csv` | 5.4 KB | CSV データ |
| `overfitting_validation_results.csv` | 3.3 KB | CSV データ |
| `partial_rebalance_results.csv` | 3.8 KB | CSV データ |
| `portfolio_diversification_results.csv` | 1.2 KB | CSV データ |
| `R4_results.csv` | 3.7 KB | CSV データ |
| `realistic_product_results.csv` | 8.6 KB | CSV データ |
| `rebalance_frequency_results.csv` | 705 B | CSV データ |
| `regime_analysis_stats.csv` | 3.9 KB | CSV データ |
| `regime_strategy_results.csv` | 1.6 KB | CSV データ |
| `regime_vs_r4_comparison.csv` | 1.6 KB | CSV データ |
| `research_all_corrected.csv` | 99.8 KB | CSV データ |
| `research_bond_signals.csv` | 49.7 KB | CSV データ |
| `research_correlation_regimes.csv` | 469 B | CSV データ |
| `research_gold_signals.csv` | 48.9 KB | CSV データ |
| `research_regime_returns.csv` | 253 B | CSV データ |
| `research_summary.csv` | 18.7 KB | CSV データ |
| `signal_correlation_matrix.csv` | 3.1 KB | CSV データ |
| `soxl_addition_results.csv` | 1.7 KB | CSV データ |
| `step1_worst10y_results.csv` | 1.3 KB | CSV データ |
| `step2_baseline.csv` | 371 B | CSV データ |
| `step2_partB_ma5.csv` | 1.7 KB | CSV データ |
| `step2_partC_ma5_macd.csv` | 2.6 KB | CSV データ |
| `step2_partD_full.csv` | 7.8 KB | CSV データ |
| `step2_partE_wf.csv` | 1.6 KB | CSV データ |
| `step2_static_cagr25_results.csv` | 3.0 KB | CSV データ |
| `step2_summary.csv` | 1.6 KB | CSV データ |
| `step3_dynamic_cagr25_results.csv` | 9.3 KB | CSV データ |
| `strategy_comparison_results.csv` | 1.1 KB | CSV データ |
| `threshold_sweep_A_results.csv` | 4.7 KB | CSV データ |
| `threshold_sweep_results.csv` | 4.6 KB | CSV データ |
| `validation_crisis.csv` | 1.7 KB | CSV データ |
| `validation_full.csv` | 1.3 KB | CSV データ |
| `validation_oos.csv` | 1.9 KB | CSV データ |
| `vote_ratio_continuous_results.csv` | 24.2 KB | CSV データ |
| `yearly_returns_7strategies.csv` | 7.0 KB | CSV データ |
| `yearly_returns_8strategies_v2.csv` | 7.9 KB | CSV データ |
| `yearly_returns_9strategies_v3.csv` | 8.3 KB | CSV データ |

</details>

### Asset (4件)

| ファイル | サイズ | 説明 |
|---|---|---|
| `FINAL_RESULTS.xlsx` | 8.4 KB | Excel スプレッドシート |
| `VolSpike_Yearly_Returns.xlsx` | 11.9 KB | Excel スプレッドシート |
| `YEARLY_RETURNS_7STRATEGIES_v2.xlsx` | 17.8 KB | Excel スプレッドシート |
| `YEARLY_RETURNS_7STRATEGIES.xlsx` | 17.9 KB | Excel スプレッドシート |

### Config (1件)

| ファイル | サイズ | 説明 |
|---|---|---|
| `.gitignore` | 471 B | Git 除外設定 |

---

_自動生成: 2026-05-02 | 手動更新: 2026-05-11 (CURRENT_BEST_STRATEGY.md追加・SUPERSEDED マーキング) | 管理者: 男座員也（Kazuya Oza）_

---

## ブランチマージ追加ファイル（2026-05-22）

以下のファイルはブランチ `claude/review-best-strategy-Jcjd5` からコピーされました。

### Turtle 戦略 (src) — 7件

| ファイル | サイズ | 説明 |
|---|---|---|
| `src/turtle_core.py` | 2.6 KB | タートル取引戦略 |
| `src/turtle_costs.py` | 3.0 KB | タートル取引戦略 |
| `src/turtle_data.py` | 5.7 KB | タートル取引戦略 |
| `src/turtle_sim.py` | 26.9 KB | タートル取引戦略 |
| `src/turtle_state.py` | 6.8 KB | タートル取引戦略 |
| `src/turtle_t1_pure_long.py` | 3.6 KB | タートル取引戦略 |
| `src/turtle_t2_long_short.py` | 3.7 KB | タートル取引戦略 |

### CFD/Dynamic Leverage (src) — 7件

| ファイル | サイズ | 説明 |
|---|---|---|
| `src/cfd_leverage_backtest.py` | 41.3 KB | CFD・動的レバレッジ戦略 |
| `src/dyn_lev_backtest.py` | 16.0 KB | CFD・動的レバレッジ戦略 |
| `src/enh_lev_backtest.py` | 17.3 KB | CFD・動的レバレッジ戦略 |
| `src/dynamic_leverage_strategies.py` | 14.8 KB | CFD・動的レバレッジ戦略 |
| `src/gen_cfd_yearly_returns.py` | 10.2 KB | CFD・動的レバレッジ戦略 |
| `src/gen_s2_yearly_returns.py` | 13.0 KB | CFD・動的レバレッジ戦略 |
| `src/sleeves_extended.py` | 5.7 KB | CFD・動的レバレッジ戦略 |

### S2/H1-H5/スイープ系 (src) — 15件

| ファイル | サイズ | 説明 |
|---|---|---|
| `src/s2_dh_integration.py` | 10.8 KB | 検証・スイープスクリプト |
| `src/s2_low_vol_sweep.py` | 7.7 KB | 検証・スイープスクリプト |
| `src/s2_rolling_cv.py` | 13.5 KB | 検証・スイープスクリプト |
| `src/s4_sharpe_sweep.py` | 9.8 KB | 検証・スイープスクリプト |
| `src/run_hypotheses_H1_H5.py` | 13.3 KB | 検証・スイープスクリプト |
| `src/h4_wgwb_sweep.py` | 13.9 KB | 検証・スイープスクリプト |
| `src/c_nasdaq_heavy_sweep.py` | 6.4 KB | 検証・スイープスクリプト |
| `src/c2_nasdaq_heavy_sweep.py` | 7.8 KB | 検証・スイープスクリプト |
| `src/d_h4_extended_sweep.py` | 6.4 KB | 検証・スイープスクリプト |
| `src/compute_worst_best_10y.py` | 7.9 KB | 検証・スイープスクリプト |
| `src/p0_verify_critical.py` | 9.2 KB | 検証・スイープスクリプト |
| `src/p1_fetch_timing_data.py` | 8.2 KB | 検証・スイープスクリプト |
| `src/p2_single_signal_backtest.py` | 22 B | 検証・スイープスクリプト |
| `src/p4_overfitting_check.py` | 31.2 KB | 検証・スイープスクリプト |
| `src/p5_bootstrap_stress.py` | 21.9 KB | 検証・スイープスクリプト |

### テスト (tests) — 2件

| ファイル | サイズ | 説明 |
|---|---|---|
| `tests/__init__.py` | 0 B | ユニットテスト |
| `tests/test_turtle_core.py` | 12.9 KB | ユニットテスト |

### CFD・レバレッジ検証レポート (md) — 9件

| ファイル | サイズ | 説明 |
|---|---|---|
| `CFD_DYNAMIC_LEVERAGE_GUIDE.md` | 6.3 KB | CFDバックテスト結果 |
| `CFD_LEVERAGE_BACKTEST_2026-05-15.md` | 7.8 KB | CFDバックテスト結果 |
| `CFD_LEVERAGE_BACKTEST_6x7x_2026-05-15.md` | 12.4 KB | CFDバックテスト結果 |
| `CFD_LEVERAGE_PROCEDURE_2026-05-15.md` | 13.7 KB | CFDバックテスト結果 |
| `CFD_S2_YEARLY_RETURNS_2026-05-17.md` | 5.1 KB | CFDバックテスト結果 |
| `CFD_YEARLY_RETURNS_2026-05-15.md` | 4.2 KB | CFDバックテスト結果 |
| `DYN_LEVERAGE_BACKTEST_2026-05-15.md` | 5.3 KB | CFDバックテスト結果 |
| `ENH_LEVERAGE_BACKTEST_2026-05-15.md` | 6.3 KB | CFDバックテスト結果 |
| `ENH_LEVERAGE_BACKTEST_2026-05-16.md` | 5.0 KB | CFDバックテスト結果 |

### Turtle・タイミング戦略レポート (md) — 6件

| ファイル | サイズ | 説明 |
|---|---|---|
| `TURTLE_RESEARCH_2026-05-18.md` | 17.1 KB | タートル戦略研究 |
| `TURTLE_RESEARCH_PLAN_2026-05-18.md` | 25.5 KB | タートル戦略研究 |
| `A_DH_STANDALONE_VERIFY_2026-05-18.md` | 4.9 KB | タートル戦略研究 |
| `T1_T2_RESULTS_2026-05-18.md` | 9.7 KB | タートル戦略研究 |
| `TIMING_STRATEGY_RESEARCH_PLAN_2026-05-18.md` | 11.5 KB | タートル戦略研究 |
| `GOLD_BOND_STRATEGY_PLAN_2026-05-17.md` | 14.5 KB | タートル戦略研究 |

### S2/H1-H5/P0-P5 検証レポート (md) — 11件

| ファイル | サイズ | 説明 |
|---|---|---|
| `S2_DH_INTEGRATION_2026-05-17.md` | 3.9 KB | 戦略検証レポート |
| `S2_LOW_VOL_SWEEP_2026-05-17.md` | 2.8 KB | 戦略検証レポート |
| `S2_ROLLING_CV_2026-05-17.md` | 10.5 KB | 戦略検証レポート |
| `S4_SHARPE_SWEEP_2026-05-17.md` | 1.7 KB | 戦略検証レポート |
| `H1_H5_SUMMARY_2026-05-17.md` | 3.7 KB | 戦略検証レポート |
| `H4_WGWB_SWEEP_2026-05-17.md` | 7.4 KB | 戦略検証レポート |
| `P1_DATA_FETCH_RESULTS_2026-05-18.md` | 4.8 KB | 戦略検証レポート |
| `P2_SINGLE_SIGNAL_RESULTS_2026-05-18.md` | 6.7 KB | 戦略検証レポート |
| `P3_COMBINATION_RESULTS_2026-05-18.md` | 8.6 KB | 戦略検証レポート |
| `P4_OVERFITTING_CHECK_2026-05-18.md` | 4.4 KB | 戦略検証レポート |
| `P5_BOOTSTRAP_STRESS_2026-05-18.md` | 3.2 KB | 戦略検証レポート |

### セッションサマリー (md) — 3件

| ファイル | サイズ | 説明 |
|---|---|---|
| `SESSION_SUMMARY_2026-05-16.md` | 7.7 KB | セッション作業記録 |
| `SESSION_SUMMARY_2026-05-18.md` | 11.1 KB | セッション作業記録 |
| `SESSION_SUMMARY_2026-05-19.md` | 6.4 KB | セッション作業記録 |

### スイープ・検証結果 CSV — 11件

| ファイル | サイズ | 説明 |
|---|---|---|
| `C2_NASDAQ_CAP_SWEEP_2026-05-18.csv` | 6.1 KB | 検証データ CSV |
| `C_NASDAQ_HEAVY_SWEEP_2026-05-17.csv` | 3.9 KB | 検証データ CSV |
| `D_H4_EXTENDED_SWEEP_2026-05-17.csv` | 9.2 KB | 検証データ CSV |
| `H4_WGWB_SWEEP_2026-05-17.csv` | 5.5 KB | 検証データ CSV |
| `P4_CV_RESULTS_2026-05-18.csv` | 3.2 KB | 検証データ CSV |
| `P4_DSR_RESULTS_2026-05-18.csv` | 1.3 KB | 検証データ CSV |
| `P5_BOOTSTRAP_SUMMARY_2026-05-18.csv` | 2.3 KB | 検証データ CSV |
| `t1_trade_log.csv` | 658 B | 検証データ CSV |
| `t1_yearly_returns.csv` | 1.3 KB | 検証データ CSV |
| `t2_trade_log.csv` | 682 B | 検証データ CSV |
| `t2_yearly_returns.csv` | 556 B | 検証データ CSV |

---

## 2026-05-21 〜 2026-05-24 セッションで追加されたファイル（手動更新セクション）

> 本セクションは2026-05-21 以降の研究で生成された全ファイル群を追記したもの。
> A/B/C/D/E/F/G/H/P/S 系のスイープ／検証 MD + 実装 + CSV を網羅。

### A 系（CFD パラメータ最適化） — 完了済み

| ファイル | 説明 |
|---|---|
| `A1_NVOL_SWEEP_2026-05-21.md` + `a1_nvol_sweep_results.csv` + `src/a1_nvol_sweep.py` | n_vol スイープ、n=20 採用根拠 |
| `A2_TV_SWEEP_2026-05-21.md` + `a2_tv_sweep_results.csv` + `src/a2_tv_sweep.py` | target_vol スイープ、tv=0.8 採用 |
| `A3_KVZ_SWEEP_2026-05-21.md` + `a3_kvz_sweep_results.csv` + `src/a3_kvz_sweep.py` | k_vz 感度（dead parameter 確認） |
| `A4_GATEMIN_SWEEP_2026-05-21.md` + `a4_gatemin_sweep_results.csv` + `src/a4_gatemin_sweep.py` | gate_min スイープ、gate_min=0.5 維持 |
| `A6_LMAX_SWEEP_2026-05-21.md` + `a6_lmax_sweep_results.csv` + `src/a6_lmax_sweep.py` | l_max スイープ、l_max=7.0 採用 |

### B 系（LT 長期逆張りシグナル探索） — LT2-N750 採用

| ファイル | 説明 |
|---|---|
| `B1_S2_LT2_2026-05-21.md` + `b1_s2_lt2_results.csv` + `src/b1_s2_lt2.py` | **LT2 採用判定（PASS）** |
| `B2_KLT_SWEEP_2026-05-21.md` + `b2_klt_sweep_results.csv` + `src/b2_klt_sweep.py` | k_lt スイープ、k=0.5 採用 |
| `B3_S2_LT4_SWEEP_2026-05-22.md` + `b3_s2_lt4_sweep_results.csv` + `src/b3_s2_lt4_sweep.py` | LT4 派生（最良 N=750/k=0.7、Shortlisted） |
| `B4_S2_LT6_SWEEP_2026-05-22.md` + `b4_s2_lt6_sweep_results.csv` + `src/b4_s2_lt6_sweep.py` | LT6 派生（Shortlisted） |
| `B5_S2_LT7_SWEEP_2026-05-22.md` + `b5_s2_lt7_sweep_results.csv` + `src/b5_s2_lt7_sweep.py` | LT7 dual MA cross（Shortlisted） |
| `B6_S2_LT2_N_SWEEP_2026-05-22.md` + `b6_s2_lt2_N_sweep_results.csv` + `src/b6_s2_lt2_N_sweep.py` | **N スイープ、N=750 / N=1500 候補確定** |
| `B7_S2_LT1_SWEEP_2026-05-22.md` + `b7_s2_lt1_sweep_results.csv` + `src/b7_s2_lt1_sweep.py` | LT1 派生（Rejected） |
| `B8_S2_LT3_SWEEP_2026-05-22.md` + `b8_s2_lt3_sweep_results.csv` + `src/b8_s2_lt3_sweep.py` | LT3 派生（Rejected） |
| `B9_COMPARISON_2026-05-23.md` + `B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md` + `b9_s2lt2_goldfrac_results.csv` + `src/b9_s2lt2_goldfrac_sweep.py` | Gold比率スイープ |
| `B9_YEARLY_RETURNS_2026-05-23.md` + `src/gen_b9_yearly_returns.py` + `src/gen_b9_comparison.py` | B9 年次リターン生成 |
| `B11_LT2_DUALN_2026-05-23.md` + `b11_lt2_dualn_results.csv` + `src/b11_lt2_dualn.py` | デュアル N 構造（Rejected） |

### C 系（外部ゲート） — 全 Rejected

| ファイル | 説明 |
|---|---|
| `C1_HY_GATE_2026-05-21.md` + `c1_hy_gate_results.csv` + `src/c1_hy_gate.py` | HY ゲート（効果なし、Rejected） |
| `C2_NASDAQ_CAP_SWEEP_2026-05-18.md/.csv` + `src/c2_nasdaq_heavy_sweep.py` | nasdaq_cap スイープ |
| `C_NASDAQ_HEAVY_SWEEP_2026-05-17.md/.csv` + `src/c_nasdaq_heavy_sweep.py` | nasdaq_heavy 検証 |

### D 系（ロバストネス検証）

| ファイル | 説明 |
|---|---|
| `D1_OOS_BOUNDARY_2026-05-21.md` + `d1_oos_boundary_results.csv` + `src/d1_oos_boundary.py` | OOS 境界±1 年シフトで S2+LT2 ロバスト最高確認 |
| `D_H4_EXTENDED_SWEEP_2026-05-17.md/.csv` + `src/d_h4_extended_sweep.py` | H4 拡張版 |

### E 系（ボラレジーム条件付き動的パラメータ） — **E4 採用**

| ファイル | 説明 |
|---|---|
| `E1_DALT_SWEEP_2026-05-24.md` + `e1_dalt_results.csv` + `src/e1_dalt_sweep.py` | D-alt パラメータ探索 |
| `E1_ENSEMBLE_2026-05-21.md` + `e1_ensemble_results.csv` + `src/e1_ensemble.py` | アンサンブル S2 系（Rejected） |
| `E2_BOND_SWEEP_2026-05-24.md` + `e2_bond_sweep_results.csv` + `src/e2_bond_sweep.py` | Bond スイープ |
| `E2_HYBRID_70_30_2026-05-22.md` + `e2_hybrid_70_30_results.csv` + `src/e2_hybrid_70_30.py` | S2LT2+P05 70/30 ハイブリッド（Shortlisted・[非標準コスト]） |
| `E3_VZGATE_SWEEP_2026-05-24.md` + `e3_vzgate_results.csv` + `src/e3_vzgate_sweep.py` | VZ ゲート派生 |
| `E4_REGIME_KLT_SWEEP_2026-05-24.md` + `e4_regime_klt_results.csv` + `src/e4_regime_klt.py` | **E4 Regime k_lt 採用根拠（vz レジーム条件付き k_lt）** |
| `e4_yearly_returns.csv` + `src/gen_e4_yearly_returns.py` | E4 年次リターン生成 |

### F 系（Bull-Tilt wn/wb 動的傾斜） — **F8 R5_CALM_BOOST 採用**

| ファイル | 説明 |
|---|---|
| `F1_ALLOC_SWEEP_2026-05-21.md` + `f1_alloc_sweep_results.csv` + `src/f1_alloc_sweep.py` | 配分比率スイープ（Rejected） |
| `F5_BOND_REGIME_2026-05-24.md` + `f5_bond_regime_results.csv` + `src/f5_bond_regime.py` | F5 Bond Regime（効果限定的） |
| `F6_VOL_SCALE_2026-05-24.md` + `f6_vol_scale_results.csv` + `src/f6_vol_scale.py` | F6 Vol Scale（効果限定的） |
| `F7_BULL_TILT_2026-05-24.md` + `f7_bull_tilt_results.csv` + `src/f7_bull_tilt.py` | F7 初期定式（cap 飽和の真因発見） |
| `F7V2_BULL_TILT_2026-05-24.md` + `f7v2_bull_tilt_results.csv` + `src/f7v2_bull_tilt.py` | F7-v2 cap 拡張（tilt が cap に届かず効果なし確認） |
| `F7V3_BULL_TILT_2026-05-24.md` + `f7v3_bull_tilt_results.csv` + `src/f7v3_bull_tilt.py` | **F7-v3 step-func 採用（Shortlisted・G4 WFA PASS）** |
| `f7v3_yearly_returns.csv` + `src/gen_f7v3_yearly_returns.py` | F7v3 年次リターン生成 |
| `F8_REGIME_TILT_2026-05-24.md` + `f8_regime_tilt_results.csv` + `src/f8_regime_tilt.py` | **★ F8 R5_CALM_BOOST 採用（現行 Active）** |
| `F9_THRESHOLD_2026-05-24.md` + `f9_threshold_results.csv` + `src/f9_threshold.py` | F9 THRESHOLD 最適化（THR=0.15 維持確認） |

### G 系（Walk-Forward Analysis）

| ファイル | 説明 |
|---|---|
| `G1_WFA_2026-05-21.md` + `g1_wfa_summary.csv` + `g1_wfa_per_window.csv` + `src/g1_wfa.py` | **G1 WFA エンジン（compute_summary_stats）** |
| `G2_WFA_B9_2026-05-23.md` + `g2_wfa_b9_summary.csv` + `g2_wfa_b9_per_window.csv` + `src/g2_wfa_b9.py` | G2 B9 派生 WFA |
| `G3_WFA_E4_2026-05-24.md` + `g3_wfa_e4_summary.csv` + `g3_wfa_e4_per_window.csv` + `src/g3_wfa_e4.py` | **G3 E4 WFA PASS（CI95_lo=+26.51%, WFE=+1.131）** |
| `G4_WFA_F7V3_2026-05-24.md` + `g4_wfa_f7v3_summary.csv` + `g4_wfa_f7v3_per_window.csv` + `src/g4_wfa_f7v3.py` | **G4 F7v3 WFA PASS（CI95_lo=+27.15%, WFE=+1.203）** |
| `G5_WFA_F8R5_2026-05-24.md` + `g5_wfa_f8r5_summary.csv` + `g5_wfa_f8r5_per_window.csv` + `src/g5_wfa_f8r5.py` | **★ G5 F8R5 WFA PASS（CI95_lo=+27.92%, WFE=+1.208）→ 正式 Active 確定** |
| `G5_REAL_YIELD_2026-05-24.md` + `g5_real_yield_results.csv` + `src/g5_real_yield.py` | Real Yield オーバーレイ検証 |

### H 系（外部シグナル / Gold / Real Yield 等） — 主に Rejected

| ファイル | 説明 |
|---|---|
| `H1_H5_SUMMARY_2026-05-17.md` | H1〜H5 サマリ |
| `H1_S4_PARAM_SWEEP_2026-05-22.md` + `h1_s4_param_sweep_results.csv` + `src/h1_s4_param_sweep.py` | S4 RelVol Gated（36/36 Rejected） |
| `H4_WGWB_SWEEP_2026-05-17.md/.csv` + `src/h4_wgwb_sweep.py` | wg/wb スイープ |
| `H5_GOLD_DYN_2026-05-24.md` + `h5_gold_dyn_results.csv` + `src/h5_gold_dyn.py` | Gold 動的オーバーレイ |

### P 系（外部マクロシグナル・モデル系） — 非標準コスト Shortlisted + 大半 Rejected

| ファイル | 説明 |
|---|---|
| `P1_SOFR_SWEEP_2026-05-22.md` + `p1_sofr_sweep_results.csv` + `src/p1_sofr_sweep.py` | SOFR Adaptive（18/18 Rejected） |
| `P3_MOMENTUM_SWEEP_2026-05-22.md` + `p3_momentum_sweep_results.csv` + `src/p3_momentum_sweep.py` | Momentum Lev（16/16 Rejected） |
| `P4_COMPOSITE_SWEEP_2026-05-21.md` + `p4_composite_sweep_results.csv` + `src/p4_composite_sweep.py` | Composite 三因子乗算（24/24 Rejected） |
| `P5_KELLY_SWEEP_2026-05-22.md` + `p5_kelly_sweep_results.csv` + `src/p5_kelly_sweep.py` | Kelly Sizing（12/12 Rejected） |
| `p10_5y_results.csv` + `p10_5y_log.txt` | P10_5Y▷ 計算 |

### S 系（A2 Conviction直接レバ変換） — 全 Rejected（致命的過剰適合警告）

| ファイル | 説明 |
|---|---|
| `S1_CONVICTION_SWEEP_2026-05-21.md` + `s1_conviction_sweep_results.csv` + `src/s1_conviction_sweep.py` | A2 直接変換（IS-OOS gap +21pp、Rejected） |
| `S3_DECOMPOSED_SWEEP_2026-05-22.md` + `s3_decomposed_sweep_results.csv` + `src/s3_decomposed_sweep.py` | **A2 分解（IS-OOS gap +22.39pp、Rejected hard）— 同類実験を二度行わないこと** |
| `src/s4_sharpe_sweep.py` + `S4_SHARPE_SWEEP_2026-05-17.md` | S4 Sharpe スイープ |

### 戦略比較・年次リターン MD

| ファイル | 説明 |
|---|---|
| `STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md` + `STRATEGY_COMPARISON_INTEGRATED_2026-05-22.md` | 11/12 戦略統合比較表 |
| `STRATEGY_PERFORMANCE_COMPARISON_2026-05-23.md` + `STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md` | パフォーマンス比較表（v1.4 で F7v3+E4 列追加） |
| `src/gen_strategy_comparison.py` | 比較 MD 生成（CSV 優先 + static fallback） |
| `src/_sweep_format.py` | **MD ヘッダ標準（MD_HEADER_1P / 2P / STRAT）必須 import 元** |

### 引き継ぎ・ドキュメント系（2026-05-24 新規）

| ファイル | 説明 |
|---|---|
| `RESEARCH_CONTEXT.md` | **🧭 研究文脈・実験系統図・棄却理由・引き継ぎ読書順序（本セッション新規）** |
| `tasks.md` | Pending / Completed 更新 |
| `CURRENT_BEST_STRATEGY.md` | F8 R5_CALM_BOOST 正式昇格を反映 |
| `STRATEGY_REGISTRY.md` | §1 Active を F8R5 に更新 |

### 追加 CSV データ

| ファイル | 説明 |
|---|---|
| `factcheck_log.txt` + `factcheck_sensitivity_results.csv` | ファクトチェック結果 |
| `bond_model_grid_results.csv` + `bond_model_annual_comparison.csv` + `bond_variant_results.csv` + `src/bond_variant_sweep.py` + `src/verify_bond_model.py` + `src/check_bond_baseline.py` + `src/check_bond_details.py` + `src/check_dgs30_splice.py` | Bond モデル検証 |
| `signal_correlation_matrix.csv` | シグナル相関 |
| `corrected_strategy_results.csv` + `src/corrected_strategy_backtest.py` | **Scenario D シグナル基盤の単一の真実** |
| `research_*.csv` (gold/bond/regime/correlation/summary) | 研究データ |
| `tmf_validation_results.csv` + `src/tmf_validation.py` | TMF 検証 |
| `delay_product_comparison_results.csv` + `src/delay_product_comparison.py` | DELAY 比較 |
| `threshold_tax_sensitivity_results.csv` + `src/threshold_tax_sensitivity.py` | 税率感度 |
| `financing_cost_results.csv` + `src/financing_cost_backtest.py` | Financing コスト |
| `external_*.csv` + `src/p1_fetch_timing_data.py` + `src/fetch_fred_data.py` | 外部信号フェッチ |

### audit_results フォルダ

| ファイル | 説明 |
|---|---|
| `audit_results/*` | 監査結果アーカイブ（詳細は `src/audit/*` 参照） |

### src/ 追加スクリプト一覧（2026-05-21 以降）

- A系: `a1_nvol_sweep.py`, `a2_tv_sweep.py`, `a3_kvz_sweep.py`, `a4_gatemin_sweep.py`, `a6_lmax_sweep.py`
- B系: `b1_s2_lt2.py`, `b2_klt_sweep.py`, `b3_s2_lt4_sweep.py`, `b4_s2_lt6_sweep.py`, `b5_s2_lt7_sweep.py`, `b6_s2_lt2_N_sweep.py`, `b7_s2_lt1_sweep.py`, `b8_s2_lt3_sweep.py`, `b9_s2lt2_goldfrac_sweep.py`, `b11_lt2_dualn.py`
- C系: `c1_hy_gate.py`
- D系: `d1_oos_boundary.py`
- E系: `e1_dalt_sweep.py`, `e1_ensemble.py`, `e2_bond_sweep.py`, `e2_hybrid_70_30.py`, `e3_vzgate_sweep.py`, `e4_regime_klt.py`
- F系: `f1_alloc_sweep.py`, `f5_bond_regime.py`, `f6_vol_scale.py`, `f7_bull_tilt.py`, `f7v2_bull_tilt.py`, `f7v3_bull_tilt.py`, `f8_regime_tilt.py`, `f9_threshold.py`
- G系: `g1_wfa.py`, `g2_wfa_b9.py`, `g3_wfa_e4.py`, `g4_wfa_f7v3.py`, `g5_real_yield.py`, `g5_wfa_f8r5.py`
- H系: `h1_s4_param_sweep.py`, `h5_gold_dyn.py`
- P系: `p1_sofr_sweep.py`, `p3_momentum_sweep.py`, `p4_composite_sweep.py`, `p5_kelly_sweep.py`
- S系: `s1_conviction_sweep.py`, `s3_decomposed_sweep.py`
- 年次リターン生成: `gen_e4_yearly_returns.py`, `gen_f7v3_yearly_returns.py`, `gen_b9_yearly_returns.py`, `gen_b9_comparison.py`, `gen_strategy_comparison.py`
- ユーティリティ: `_sweep_format.py`, `calculate_p10_5y.py`, `compute_cfd_worst10y.py`, `compute_dha_worst10y_only.py`, `long_cycle_signal.py`, `product_costs.py`, `corrected_strategy_backtest.py`
- ML系: `ml_model.py`, `ml_walkforward.py`〜`ml_walkforward_v4.py`, `generate_macro_features.py`, `generate_ml_features.py`
- データフェッチ: `extend_data.py`, `fetch_fred_data.py`, `build_base_dataset.py`, `p1_fetch_timing_data.py`

---

_最終手動更新: 2026-05-26 (G7/G8/G9 WFA + H1 Stress Test + F10+lmax5 全指標 + INTEGRATION_DEBATE 追加)_

---

## 2026-05-26 セッション成果物（手動更新セクション）

> 3候補（F10 ε=0.015 / vz065+lmax5 / F10+lmax5）の WFA 完了と採用候補比較レポート。

### WFA レポート (G7/G8/G9) — 3シリーズ

| ファイル | 説明 |
|---|---|
| `G7_WFA_F10_2026-05-26.md` + `g7_wfa_f10_summary.csv` + `g7_wfa_f10_per_window.csv` + `src/g7_wfa_f10.py` | **G7 F10 ε=0.015 WFA PASS** — CI95_lo=+27.93%（全候補中最高）、WFE=+1.208。採用推奨・ユーザー判断待ち |
| `G8_WFA_F10LMAX5_2026-05-26.md` + `g8_wfa_lmax5_summary.csv` + `g8_wfa_lmax5_per_window.csv` + `src/g8_wfa_lmax5.py` | **G8 F10+lmax5 WFA PASS** — CI95_lo=+25.57%、WFE=+1.278。P10_5Y最高+12.8%。Trades/yr=52（per-window 平均 52.09 で確認） |
| `G9_WFA_VZ065_LMAX5_2026-05-26.md` + `g9_wfa_vz065_summary.csv` + `g9_wfa_vz065_per_window.csv` | **G9 vz065+lmax5 WFA PASS** — CI95_lo=+24.82%、WFE=+1.272。Sharpe=+0.947 (最高)・MaxDD=-51.8% (最良) |

### ストレステスト・全指標計算

| ファイル | 説明 |
|---|---|
| `H1_STRESS_TEST_2026-05-26.md` + `h1_stress_test_results.csv` | **H1 Stress Test** — 4候補の P10_5Y▷ 値算出。E4=+9.8%、F10=+10.3%、vz065+lmax5=+11.9%、F10+lmax5=+12.8% |
| `f10lmax5_fullmetrics.csv` + `src/p10_f10lmax5_fullmetrics.py` | **F10+lmax5 全指標計算** — CAGR_OOS=+33.6%、MaxDD=-54.2%、Worst10Y★=+16.9%、IS-OOS gap=-3.37pp。※Trades_yr=27.12 はバグ値（正しくは 52）|

### 統合比較レポート

| ファイル | 説明 |
|---|---|
| `INTEGRATION_DEBATE_2026-05-26.md` | **4候補採用比較レポート** — E4 vs F10 ε=0.015 vs vz065+lmax5 vs F10+lmax5。IS-OOS gap ランキング・Trades/yr・過学習リスク評価。§2 テーブルは MD_HEADER_STRAT 準拠（commit 7c0b091 で修正済み） |

### 2026-05-26 セッション発見事項（バグ・ルール追加）

| 発見 | 内容 | 対処 |
|---|---|---|
| Trades/yr バグ (p10系) | `p10_f10lmax5_fullmetrics.py` が lev_raw 変化のみカウント（27/yr）。正しくは lev_raw+wn/wb（52/yr） | INTEGRATION_DEBATE 表を 52 に修正。CSV 再生成は Pending |
| MD ヘッダ手書き違反 | サブエージェントが `_sweep_format.py` import せず手書きヘッダを生成（P10 列欠落・列名違反） | INTEGRATION_DEBATE §2 を MD_HEADER_STRAT で修正。CLAUDE.md に再発防止 Item 5 追記 |
