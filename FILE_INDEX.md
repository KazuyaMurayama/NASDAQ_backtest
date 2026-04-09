# ファイルインデックス - nasdaq_backtest

> **重要**: ファイル検索やファイル参照を行う際には、まずこの一覧を参照してください。

最終更新: 2026-04-09  
リポジトリ: https://github.com/KazuyaMurayama/NASDAQ_backtest

---

## 最新情報の確認フロー

> **「最新のベスト戦略・研究状況」を知りたい場合:**
>
> - **同種のファイルが複数ある場合は、ファイル名の日付サフィックス（`_YYYY-MM-DD`）が最新のものを最優先で参照すること**
>   - 例: `SESSION_SUMMARY_2026-04-04.md` と `SESSION_SUMMARY_2026-03-01.md` があれば → 2026-04-04 を参照
>   - 例: `STRATEGY_COMPARISON_CAGR30per_plus_2026-04-02.md` → 日付が最新の比較レポートを参照
> - **生成ドキュメントには必ず日付サフィックスを付与すること**（ルール: `FILENAME_YYYY-MM-DD.md`）
> - `CLAUDE.md` / `README.md` / `FILE_INDEX.md` は living docs のため日付なし
>
> ⚠️ `CLAUDE.md` は概要説明用であり、最新の数値は含まれていない場合がある。

---

## ブランチ一覧

| ブランチ名 | 用途 |
|-----------|------|
| `main` | メインブランチ（安定版） |
| `claude/compare-trading-strategies-PPCbl` | 戦略比較・パラメータ最適化・研究拡張 |
| `claude/investment-strategy-tracking-gt8uD` | 投資戦略トラッキング・検証強化 |
| `claude/project-overview-s2l8s` | プロジェクト概要・OOS検証・過学習検証 |

---

## ドキュメント（MDファイル）

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `SESSION_SUMMARY_2026-04-04.md` | 最新セッションサマリー — Dyn 2x3x G0.5が最新ベスト戦略と確定。CAGR 31.40%/Sharpe 1.297/MaxDD -33.4%/Worst5Y +5.2%。Gold日次データ改善・190指標検証・Bond追加は改善なしと判定。 | compare-trading-strategies-PPCbl |
| `STRATEGY_COMPARISON_CAGR30per_plus_2026-04-02.md` | 全7戦略の最終比較表（CAGR/Sharpe/MaxDD/Worst5Y/年次リターン） | compare-trading-strategies-PPCbl |
| `REGIME_ANALYSIS_REPORT_2026-04-04.md` | レジーム分析レポート（rawLev×NASDAQ 3×5ビン、フォワードリターン統計） | compare-trading-strategies-PPCbl |
| `YEARLY_RETURNS_REPORT_2026-04-01.md` | 7戦略の年次・月次リターン詳細（Dyn 2x3x含む最新版） | compare-trading-strategies-PPCbl |
| `ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md` | 追加分析レポート（戦略拡張・改善） | compare-trading-strategies-PPCbl |
| `PARAMETER_OPTIMIZATION_PLAN_2026-03-30.md` | パラメータ最適化計画書 | compare-trading-strategies-PPCbl |
| `PARAMETER_OPTIMIZATION_REPORT_2026-03-30.md` | パラメータ最適化結果レポート | compare-trading-strategies-PPCbl |
| `STRATEGY_COMPARISON_2026-03-30.md` | 7戦略比較レポート（旧版） | compare-trading-strategies-PPCbl |
| `CLAUDE.md` | プロジェクト概要・戦略詳細・GAS実装ガイド（AIへの指示書） | 全ブランチ |
| `FILE_INDEX.md` | このファイル（全ファイルインデックス） | claude/create-file-index-vVbP4以降 |
| `CONTEXT_SUMMARY_2026-02-06.md` | セッション文脈サマリー（研究の経緯・決定事項） | 全ブランチ |
| `3x_NASDAQ_Strategy_Research_Summary_2026-02-06.md` | 3倍レバレッジNASDAQ戦略研究全体サマリー | 全ブランチ |
| `FINAL_RESULTS_2026-02-06.md` | 最終戦略比較結果レポート（旧版、Ens2が主体） | 全ブランチ |
| `R4_RESULTS_SUMMARY_2026-02-06.md` | Round 4バックテスト結果サマリー（55戦略） | 全ブランチ |
| `STRATEGY_RESEARCH_PLAN_R4_2026-02-06.md` | Round 4 研究計画書 | 全ブランチ |
| `LEVERAGE_BIN_ANALYSIS_2026-03-19.md` | レバレッジ区間分析レポート v1 | 全ブランチ |
| `LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md` | レバレッジ区間分析レポート v2 | 全ブランチ |
| `LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md` | レバレッジ区間分析レポート v3 | 全ブランチ |
| `docs/TQQQ_execution_guide.md` | TQQQ実行ガイド（実際の売買手順） | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `plan_2026-02-12.md` | プロジェクト実行計画（旧版） | project-overview-s2l8s |

---

## データファイル

### 価格データ（CSV）

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `NASDAQ_Dairy_since1973.csv` | NASDAQ Composite 日次データ（1974〜2021年, 47年間） | 全ブランチ |
| `NASDAQ_extended_to_2026.csv` | NASDAQ Composite 日次データ（2026年まで延長版） | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `NASDAQ_extended.csv` | NASDAQ Composite 延長データ（旧版） | project-overview-s2l8s |
| `data/lbma_gold_daily.csv` | LBMA 金価格日次データ（ポートフォリオ多様化用） | compare-trading-strategies-PPCbl |

### 戦略結果（CSV）

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `R4_results.csv` | Round 4全戦略バックテスト結果（55戦略） | 全ブランチ |
| `FINAL_RESULTS.csv` | 最終戦略比較結果（主要指標） | 全ブランチ |
| `ens2_comparison_results.csv` | Ens2戦略比較結果 | 全ブランチ |
| `hybrid_strategy_results.csv` | ハイブリッドDD+VT戦略結果 | 全ブランチ |
| `majority_vote_p2_results.csv` | 多数決戦略フェーズ2結果 | 全ブランチ |
| `majority_vote_signals.csv` | 多数決シグナル詳細 | 全ブランチ |
| `partial_rebalance_results.csv` | 部分リバランス戦略結果 | 全ブランチ |
| `realistic_product_results.csv` | 実商品条件（遅延・コスト込み）バックテスト結果 | 全ブランチ |
| `regime_strategy_results.csv` | レジーム戦略結果 | 全ブランチ |
| `regime_vs_r4_comparison.csv` | レジーム戦略 vs R4比較 | 全ブランチ |
| `signal_correlation_matrix.csv` | シグナル相関行列 | 全ブランチ |
| `vote_ratio_continuous_results.csv` | 投票比率連続版結果 | 全ブランチ |
| `delay_robust_results.csv` | 遅延ロバスト性テスト結果 | 全ブランチ |
| `discrete_leverage_results.csv` | 離散レバレッジ結果 | 全ブランチ |
| `final_5strategy_yearly_returns.csv` | 主要5戦略の年次リターン | 全ブランチ |
| `final_6strategy_yearly_returns.csv` | 主要6戦略の年次リターン | 全ブランチ |
| `leverage_bin_analysis.csv` | レバレッジ区間分析 v1 | 全ブランチ |
| `leverage_bin_analysis_v2.csv` | レバレッジ区間分析 v2 | 全ブランチ |
| `leverage_bin_analysis_v3.csv` | レバレッジ区間分析 v3 | 全ブランチ |
| `improvement_results.csv` | 改善戦略テスト結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `improvement_results_r2.csv` | 改善戦略 R2 結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `improvement_results_r3.csv` | 改善戦略 R3 結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `improvement_results_r4.csv` | 改善戦略 R4 結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `improvement_results_next.csv` | 次期改善戦略結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `improvement_results_a_vix.csv` | VIX統合改善戦略結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `improvement_results_wf_vix.csv` | WF VIX改善戦略結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `dynamic_portfolio_results.csv` | 動的ポートフォリオ結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `portfolio_diversification_results.csv` | ポートフォリオ多様化結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `grid_search_results.csv` | グリッドサーチ結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `validation_crisis.csv` | 危機時期検証結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `validation_full.csv` | 全期間検証結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `validation_oos.csv` | OOS（アウトオブサンプル）検証結果 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `monthly_returns_oos.csv` | OOS月次リターン | compare-trading-strategies-PPCbl |
| `lev2x3x_results.csv` | 2x/3xレバレッジ比較結果 | compare-trading-strategies-PPCbl |
| `opt_phase1_tier1_results.csv` | 最適化フェーズ1 Tier1結果 | compare-trading-strategies-PPCbl |
| `opt_phase1_tier2_results.csv` | 最適化フェーズ1 Tier2結果 | compare-trading-strategies-PPCbl |
| `opt_phase1_tier3_results.csv` | 最適化フェーズ1 Tier3結果 | compare-trading-strategies-PPCbl |
| `opt_phase1_wf_results.csv` | 最適化フェーズ1 WF結果 | compare-trading-strategies-PPCbl |
| `opt_phase2_results.csv` | 最適化フェーズ2結果 | compare-trading-strategies-PPCbl |
| `opt_phase3_results.csv` | 最適化フェーズ3結果 | compare-trading-strategies-PPCbl |
| `regime_analysis_stats.csv` | レジーム分析統計 | compare-trading-strategies-PPCbl |
| `strategy_comparison_results.csv` | 戦略比較結果 | compare-trading-strategies-PPCbl |
| `yearly_returns_7strategies.csv` | 7戦略年次リターン | compare-trading-strategies-PPCbl |
| `research_all_corrected.csv` | ゴールド・債券調査結果（修正版） | compare-trading-strategies-PPCbl |
| `research_bond_signals.csv` | 債券シグナル調査結果 | compare-trading-strategies-PPCbl |
| `research_correlation_regimes.csv` | 相関レジーム調査結果 | compare-trading-strategies-PPCbl |
| `research_gold_signals.csv` | ゴールドシグナル調査結果 | compare-trading-strategies-PPCbl |
| `research_regime_returns.csv` | レジーム別リターン調査 | compare-trading-strategies-PPCbl |
| `research_summary.csv` | 調査サマリー | compare-trading-strategies-PPCbl |
| `step1_worst10y_results.csv` | 最悪10年間調査結果 | compare-trading-strategies-PPCbl |
| `step2_baseline.csv` | ステップ2ベースライン | compare-trading-strategies-PPCbl |
| `step2_partB_ma5.csv` | ステップ2-B MA5結果 | compare-trading-strategies-PPCbl |
| `step2_partC_ma5_macd.csv` | ステップ2-C MA5+MACD結果 | compare-trading-strategies-PPCbl |
| `step2_partD_full.csv` | ステップ2-D フル結果 | compare-trading-strategies-PPCbl |
| `step2_partE_wf.csv` | ステップ2-E ウォークフォワード結果 | compare-trading-strategies-PPCbl |
| `step2_static_cagr25_results.csv` | CAGR25%静的戦略結果 | compare-trading-strategies-PPCbl |
| `step2_summary.csv` | ステップ2サマリー | compare-trading-strategies-PPCbl |
| `step3_dynamic_cagr25_results.csv` | CAGR25%動的戦略結果 | compare-trading-strategies-PPCbl |
| `oos_monthly_returns.csv` | OOS月次リターン | compare-trading-strategies-PPCbl |
| `external_is_param_sweep.csv` | 外部シグナル in-sample パラメータスイープ | project-overview-s2l8s |
| `external_oos_results.csv` | 外部シグナル OOS結果 | project-overview-s2l8s |
| `external_signal_results.csv` | 外部シグナル結果 | project-overview-s2l8s |
| `overfitting_validation_results.csv` | 過学習検証結果 | project-overview-s2l8s |

### Excelファイル

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `FINAL_RESULTS.xlsx` | 最終結果 Excel版 | 全ブランチ |
| `VolSpike_Yearly_Returns.xlsx` | VolSpike戦略年次リターン Excel | 全ブランチ |
| `YEARLY_RETURNS_7STRATEGIES.xlsx` | 7戦略年次リターン Excel | compare-trading-strategies-PPCbl |
| `YEARLY_RETURNS_7STRATEGIES_v2.xlsx` | 7戦略年次リターン Excel v2 | compare-trading-strategies-PPCbl |

---

## Pythonソースコード（src/）

### コアエンジン

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `src/backtest_engine.py` | **コアエンジン** - 全戦略実装（DD/EWMA/SlopeMult/Ens2/MomDecel等） | 全ブランチ |
| `src/run_r4_backtest.py` | Round 4 全55戦略バックテスト実行スクリプト | 全ブランチ |
| `src/generate_final_results.py` | 最終結果ファイル生成（主要戦略比較） | 全ブランチ |
| `src/debug_sharpe_calc.py` | Sharpe比計算デバッグ・検証ツール | 全ブランチ |
| `src/verify_max_lev.py` | max_levパラメータ検証ツール | 全ブランチ |
| `src/extend_data.py` | NASDAQデータを2026年まで延長 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |

### 戦略テスト（test_*.py）

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `src/test_ens2_strategies.py` | Ens2（アンサンブル）戦略テスト | 全ブランチ |
| `src/test_final_5strategy.py` | 主要5戦略比較テスト | 全ブランチ |
| `src/test_final_6strategy.py` | 主要6戦略比較テスト | 全ブランチ |
| `src/test_hybrid_dd_vote.py` | DD+多数決ハイブリッド戦略テスト | 全ブランチ |
| `src/test_majority_vote_p1.py` | 多数決戦略フェーズ1テスト | 全ブランチ |
| `src/test_majority_vote_p2.py` | 多数決戦略フェーズ2テスト | 全ブランチ |
| `src/test_partial_rebalance.py` | 部分リバランス戦略テスト | 全ブランチ |
| `src/test_discrete_leverage.py` | 離散レバレッジ戦略テスト | 全ブランチ |
| `src/test_delay_robust.py` | 遅延ロバスト性テスト（5日遅延シミュレーション） | 全ブランチ |
| `src/test_realistic_product.py` | 実商品条件（TQQQ等）バックテスト | 全ブランチ |
| `src/test_regime_comparison.py` | レジーム別戦略比較テスト | 全ブランチ |
| `src/test_regime_strategy.py` | レジーム戦略単体テスト | 全ブランチ |
| `src/test_original_method.py` | オリジナル手法ベースラインテスト | 全ブランチ |
| `src/test_vote_ratio_continuous.py` | 投票比率連続版テスト | 全ブランチ |
| `src/test_improvements.py` | 改善戦略テスト（第1世代） | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_improvements_r2.py` | 改善戦略テスト R2 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_improvements_r3.py` | 改善戦略テスト R3 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_improvements_r4.py` | 改善戦略テスト R4 | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_next_improvements.py` | 次期改善戦略テスト | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_delay_sensitivity.py` | 遅延感度分析テスト | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_dynamic_portfolio.py` | 動的ポートフォリオ（多資産）テスト | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_portfolio_diversification.py` | ポートフォリオ多様化テスト（金・債券） | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_validation.py` | 戦略検証テスト | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_vix_integration.py` | VIX統合テスト | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_walkforward_vix_reentry.py` | ウォークフォワードVIX再エントリーテスト | compare-trading-strategies-PPCbl, investment-strategy-tracking-gt8uD |
| `src/test_external_oos.py` | 外部OOSシグナルテスト | project-overview-s2l8s |
| `src/test_external_signals.py` | 外部シグナルテスト | project-overview-s2l8s |
| `src/test_oos_verification.py` | OOS検証テスト | project-overview-s2l8s |
| `src/test_overfitting_validation.py` | 過学習検証テスト | project-overview-s2l8s |

### 分析・出力スクリプト

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `src/analyze_leverage_vs_return.py` | レバレッジ vs リターン分析 v1 | 全ブランチ |
| `src/analyze_leverage_vs_return_v2.py` | レバレッジ vs リターン分析 v2 | 全ブランチ |
| `src/analyze_leverage_vs_return_v3.py` | レバレッジ vs リターン分析 v3 | 全ブランチ |
| `src/yearly_returns_analysis.py` | 年次リターン分析 | 全ブランチ |
| `src/yearly_returns_excel_formatted.py` | 年次リターンExcel出力（書式あり） | 全ブランチ |
| `src/yearly_returns_excel_ja.py` | 年次リターンExcel出力（日本語） | 全ブランチ |
| `src/verify_calculations.py` | 計算結果検証ツール | project-overview-s2l8s |
| `src/compare_7strategies.py` | 7戦略比較スクリプト | compare-trading-strategies-PPCbl |
| `src/gen_full_excel.py` | フルExcel生成 | compare-trading-strategies-PPCbl |
| `src/gen_yearly_excel.py` | 年次リターンExcel生成 | compare-trading-strategies-PPCbl |
| `src/gen_yearly_md.py` | 年次リターンMarkdown生成 | compare-trading-strategies-PPCbl |
| `src/quality_check.py` | データ品質チェック | compare-trading-strategies-PPCbl |
| `src/regime_analysis.py` | レジーム分析スクリプト | compare-trading-strategies-PPCbl |
| `src/research_gold_bond_timing.py` | ゴールド・債券タイミング研究 | compare-trading-strategies-PPCbl |
| `src/research_step2_bond_timing.py` | 債券タイミング研究ステップ2 | compare-trading-strategies-PPCbl |

### 最適化スクリプト

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `src/opt_lev2x3x.py` | 2x/3x レバレッジ最適化 | compare-trading-strategies-PPCbl |
| `src/opt_phase1_tier1.py` | 最適化フェーズ1 Tier1（DDパラメータ） | compare-trading-strategies-PPCbl |
| `src/opt_phase1_tier2.py` | 最適化フェーズ1 Tier2（VTパラメータ） | compare-trading-strategies-PPCbl |
| `src/opt_phase1_tier3.py` | 最適化フェーズ1 Tier3（SlopeMultパラメータ） | compare-trading-strategies-PPCbl |
| `src/opt_phase1_wf_sensitivity.py` | 最適化フェーズ1 ウォークフォワード感度 | compare-trading-strategies-PPCbl |
| `src/opt_phase2_dynhybrid.py` | 最適化フェーズ2 動的ハイブリッド | compare-trading-strategies-PPCbl |
| `src/step1_worst10y.py` | 最悪10年間調査ステップ1 | compare-trading-strategies-PPCbl |
| `src/step2_cagr25_grid.py` | CAGR25% グリッドサーチ | compare-trading-strategies-PPCbl |
| `src/step_monthly_returns.py` | 月次リターン計算 | compare-trading-strategies-PPCbl |
| `src/step_update_dyn2x3x.py` | Dyn2x3x 戦略更新 | compare-trading-strategies-PPCbl |
| `src/step_yearly_returns.py` | 年次リターン計算 | compare-trading-strategies-PPCbl |

---

## その他

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `.gitignore` | Git除外設定（`__pycache__/`, `*.pyc`, `venv/`等） | 全ブランチ |

---

## 戦略の全体像（参考）

```
最終レバレッジ = DD × VT × SlopeMult [× MomDecel]

Layer 1: DD Control（0 or 1）
  - 200日ローリング高値の82%割れ → CASH（0）
  - 92%回復 → HOLD（1）

Layer 2: VT = min(Target_Vol / AsymEWMA_Vol, max_lev)
  - AsymEWMA: 下落時span=5、上昇時span=20
  - TrendTV: Price/MA150 比率でTarget_Volを動的調整

Layer 3: SlopeMult = clip(0.7 + 0.3 × z_score, 0.3, 1.5)
  - MA200の日次変化率をZ-score化

Layer 4 (推奨追加): MomDecel
  - モメンタム減速検出によるリスク抑制
```

**推奨戦略**: `Dyn 2x3x (B0.55/L0.25/V0.1/G0.5)` — CAGR 31.40% / Sharpe 1.297 / MaxDD -33.4%（2026-04-04確定）
