#!/bin/bash
# nasdaq_backtest: 未集約の大容量CSV・テキストファイルをclaude/*ブランチからmainへ取込
# 使用方法: bash scripts/migrate_large_files.sh
#
# 2026-04-19時点で未集約のファイル:
#   - 大容量CSV (2.9MB): NASDAQ_extended*, data/*.csv, research_all_corrected.csv
#   - src/ 新規Python 42ファイル (create-file-index-vVbP4)
#   - src/ project-overview固有 5ファイル
#   - ルートMD 8ファイル (CAGR_IMPROVEMENT_PLAN 等)
#   - XLSX 2ファイル (YEARLY_RETURNS_7STRATEGIES*.xlsx)

set -e

BRANCH_CREATE="claude/create-file-index-vVbP4"
BRANCH_OVERVIEW="claude/project-overview-s2l8s"
BRANCH_INVEST="claude/investment-strategy-tracking-gt8uD"

echo "=== Step 1: $BRANCH_CREATE から大容量CSV ==="
git fetch origin $BRANCH_CREATE
git checkout origin/$BRANCH_CREATE -- \
  NASDAQ_extended_to_2026.csv \
  research_all_corrected.csv \
  data/dgp_daily.csv data/drn_daily.csv data/ief_daily.csv \
  data/lbma_gold_daily.csv data/sox_daily.csv data/sp500_daily.csv \
  data/tmf_daily.csv data/upro_daily.csv data/vnq_daily.csv || true

echo "=== Step 2: $BRANCH_CREATE から研究MD (8ファイル) ==="
git checkout origin/$BRANCH_CREATE -- \
  CAGR_IMPROVEMENT_PLAN_2026-04-09.md \
  PARAMETER_OPTIMIZATION_PLAN_2026-03-30.md \
  PARAMETER_OPTIMIZATION_REPORT_2026-03-30.md \
  REGIME_ANALYSIS_REPORT_2026-04-04.md \
  SESSION_SUMMARY_2026-04-04.md \
  STRATEGY_COMPARISON_2026-03-30.md \
  STRATEGY_COMPARISON_CAGR30per_plus_2026-04-02.md \
  YEARLY_RETURNS_REPORT_2026-04-01.md || true

echo "=== Step 3: $BRANCH_CREATE から XLSX ==="
git checkout origin/$BRANCH_CREATE -- \
  YEARLY_RETURNS_7STRATEGIES.xlsx YEARLY_RETURNS_7STRATEGIES_v2.xlsx || true

echo "=== Step 4: $BRANCH_CREATE から src/新規Python (全ファイル取込) ==="
git checkout origin/$BRANCH_CREATE -- src/ || true

echo "=== Step 5: $BRANCH_CREATE から中小CSV (全ファイル取込) ==="
for f in dynamic_portfolio_results.csv grid_search_results.csv improvement_results.csv \
  improvement_results_a_vix.csv improvement_results_next.csv improvement_results_r2.csv \
  improvement_results_r3.csv improvement_results_r4.csv improvement_results_wf_vix.csv \
  lev2x3x_results.csv marugoto_leverage_results.csv monthly_returns_oos.csv \
  opt_phase1_tier1_results.csv opt_phase1_tier2_results.csv opt_phase1_tier3_results.csv \
  opt_phase1_wf_results.csv opt_phase2_results.csv opt_phase3_results.csv \
  portfolio_diversification_results.csv rebalance_frequency_results.csv \
  regime_analysis_stats.csv research_bond_signals.csv research_correlation_regimes.csv \
  research_gold_signals.csv research_regime_returns.csv research_summary.csv \
  soxl_addition_results.csv step1_worst10y_results.csv step2_baseline.csv \
  step2_partB_ma5.csv step2_partC_ma5_macd.csv step2_partD_full.csv \
  step2_partE_wf.csv step2_static_cagr25_results.csv step2_summary.csv \
  step3_dynamic_cagr25_results.csv strategy_comparison_results.csv \
  validation_crisis.csv validation_full.csv validation_oos.csv \
  yearly_returns_7strategies.csv; do
  git checkout origin/$BRANCH_CREATE -- "$f" 2>/dev/null || true
done

echo "=== Step 6: $BRANCH_OVERVIEW 固有ファイル (overfitting/OOS系) ==="
git fetch origin $BRANCH_OVERVIEW
git checkout origin/$BRANCH_OVERVIEW -- \
  NASDAQ_extended.csv plan_2026-02-12.md \
  external_is_param_sweep.csv external_oos_results.csv external_signal_results.csv \
  overfitting_validation_results.csv || true
git checkout origin/$BRANCH_OVERVIEW -- \
  src/test_external_oos.py src/test_external_signals.py \
  src/test_oos_verification.py src/test_overfitting_validation.py \
  src/verify_calculations.py || true

echo "=== Step 7: $BRANCH_INVEST から最新版 ==="
git fetch origin $BRANCH_INVEST
git checkout origin/$BRANCH_INVEST -- src/test_portfolio_diversification.py || true

echo ""
echo "=== 完了 ==="
echo "git status で確認してから以下でコミット:"
echo "  git add -A"
echo "  git commit -m 'data: import unmerged files from claude/* branches via migrate script'"
echo "  git push origin main"
