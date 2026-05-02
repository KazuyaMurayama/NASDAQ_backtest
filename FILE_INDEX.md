# FILE_INDEX — NASDAQ_backtest

> ⚠️ このファイルは自動生成です。手動編集は次回更新で上書きされます。

| 項目 | 値 |
|---|---|
| リポジトリ | KazuyaMurayama/NASDAQ_backtest |
| ブランチ | main |
| 総ファイル数 | 211 |
| 最終更新 | 2026-05-02 |
| 管理者 | 男座員也（Kazuya Oza） |

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
| `3x_NASDAQ_Strategy_Research_Summary_2026-02-06.md` | 19.2 KB | Markdown ドキュメント |
| `ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md` | 5.6 KB | Markdown ドキュメント |
| `APPROACH_A_PROPOSAL_2026-04-20.md` | 7.5 KB | Markdown ドキュメント |
| `archive/2026-04-19_CLAUDE_main.md` | 5.4 KB | Markdown ドキュメント |
| `archive/2026-04-19_gitignore_main.txt` | 257 B | ファイル |
| `CAGR_DISCREPANCY_ANALYSIS_2026-04-20.md` | 4.4 KB | Markdown ドキュメント |
| `CAGR_IMPROVEMENT_PLAN_2026-04-09.md` | 13.8 KB | Markdown ドキュメント |
| `CLAUDE.md` | 2.0 KB | Claude Code プロジェクト設定・命名ルール |
| `CONTEXT_SUMMARY_2026-02-06.md` | 2.9 KB | Markdown ドキュメント |
| `docs/rules/01_response-basics.md` | 708 B | Markdown ドキュメント |
| `docs/rules/02_task-management.md` | 458 B | Markdown ドキュメント |
| `docs/rules/03_file-index.md` | 487 B | Markdown ドキュメント |
| `docs/rules/04_deliverables-and-models.md` | 557 B | Markdown ドキュメント |
| `docs/rules/05_git-and-execution.md` | 1.1 KB | Markdown ドキュメント |
| `docs/TQQQ_execution_guide.md` | 4.0 KB | Markdown ドキュメント |
| `FILE_INDEX.md` | 3.8 KB | （このファイル）全ファイルインデックス |
| `FINAL_RESULTS_2026-02-06.md` | 2.8 KB | Markdown ドキュメント |
| `LEVERAGE_BIN_ANALYSIS_2026-03-19.md` | 6.0 KB | Markdown ドキュメント |
| `LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md` | 10.2 KB | Markdown ドキュメント |
| `LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md` | 13.5 KB | Markdown ドキュメント |
| `LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md` | 8.5 KB | Markdown ドキュメント |
| `PARAMETER_OPTIMIZATION_PLAN_2026-03-30.md` | 7.7 KB | Markdown ドキュメント |
| `PARAMETER_OPTIMIZATION_REPORT_2026-03-30.md` | 8.2 KB | Markdown ドキュメント |
| `plan_2026-02-12.md` | 2.0 KB | Markdown ドキュメント |
| `R4_RESULTS_SUMMARY_2026-02-06.md` | 5.5 KB | Markdown ドキュメント |
| `README.md` | 2.2 KB | リポジトリ概要・セットアップ手順 |
| `REGIME_ANALYSIS_REPORT_2026-04-04.md` | 4.8 KB | Markdown ドキュメント |
| `SESSION_SUMMARY_2026-04-04.md` | 13.6 KB | Markdown ドキュメント |
| `STRATEGY_COMPARISON_2026-03-30.md` | 3.7 KB | Markdown ドキュメント |
| `STRATEGY_COMPARISON_CAGR30per_plus_2026-04-02.md` | 36.7 KB | Markdown ドキュメント |
| `STRATEGY_RESEARCH_PLAN_R4_2026-02-06.md` | 18.6 KB | Markdown ドキュメント |
| `tasks.md` | 1.8 KB | タスク管理・セッション履歴 |
| `THRESHOLD_SWEEP_A_REPORT_2026-04-21.md` | 8.0 KB | Markdown ドキュメント |
| `THRESHOLD_SWEEP_REPORT_2026-04-20.md` | 5.3 KB | Markdown ドキュメント |
| `Timeout_Prevention.md` | 4.9 KB | タイムアウト対策ガイド |
| `YEARLY_RETURNS_REPORT_2026-04-01.md` | 34.1 KB | Markdown ドキュメント |
| `YEARLY_RETURNS_REPORT_2026-04-20_v2.md` | 12.9 KB | Markdown ドキュメント |
| `YEARLY_RETURNS_REPORT_2026-04-20_v3.md` | 11.6 KB | Markdown ドキュメント |

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

_自動生成: 2026-05-02 | 管理者: 男座員也（Kazuya Oza）_
