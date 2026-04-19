# ファイルインデックス - nasdaq_backtest

> **重要**: ファイル検索・参照前に必ずこの一覧を確認してください。

最終更新: 2026-04-16
リポジトリ: https://github.com/KazuyaMurayama/NASDAQ_backtest

---

## セッション開始時の参照順序

1. `tasks.md` — 未完了タスク・優先度を確認
2. `FILE_INDEX.md`（このファイル）— ファイル構成を把握
3. `.claude/rules/` — 操作ルールを確認

## 最新情報の確認フロー

> **同種ファイルが複数ある場合は日付サフィックス（`_YYYY-MM-DD`）が最新のものを最優先で参照すること**
> - 例: `SESSION_SUMMARY_2026-04-04.md` が最新
> - 生成ドキュメントには必ず日付サフィックスを付与（ルール: `FILENAME_YYYY-MM-DD.md`）
> - `CLAUDE.md` / `FILE_INDEX.md` / `tasks.md` は living docs のため日付なし

⚠️ `CLAUDE.md` に数値・戦略詳細は書かない。FILE_INDEX.md 経由で参照する。

---

## ブランチ一覧

| ブランチ名 | 用途 |
|-----------|------|
| `main` | メインブランチ（安定版） |
| `claude/create-file-index-vVbP4` | **現在の作業ブランチ**（実行ガイド・ルール整備・tasks.md） |
| `claude/compare-trading-strategies-PPCbl` | 戦略比較・パラメータ最適化・研究拡張 |
| `claude/investment-strategy-tracking-gt8uD` | 投資戦略トラッキング・検証強化 |
| `claude/project-overview-s2l8s` | プロジェクト概要・OOS検証・過学習検証 |

---

## ドキュメント（MDファイル）

### 運用管理（living docs）

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `tasks.md` | **タスク一覧**（未完了・完了・バックログ） | create-file-index-vVbP4 |
| `CLAUDE.md` | Claude Code 指示書（軽量）— ルール参照先・プロフィール・スプレッドシートアクセス | 全ブランチ |
| `FILE_INDEX.md` | このファイル（全ファイルインデックス） | create-file-index-vVbP4 以降 |
| `.claude/rules/response-rules.md` | 回答フォーマット・成果物報告・名前表記ルール | create-file-index-vVbP4 |
| `.claude/rules/git-rules.md` | Git操作ルール（ブランチ作成禁止等） | create-file-index-vVbP4 |
| `.claude/rules/workflow-rules.md` | タスク管理・モデル使い分け・エージェントルール | create-file-index-vVbP4 |

### 実行ガイド・投資運用

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `docs/TQQQ_execution_guide.md` | **Dyn-Hybrid戦略 実行ガイド v3**（SBI証券）— TQQQ/2036/TMF 売買手順・株数計算・保有残高確認 | **create-file-index-vVbP4** ✅ 最新 |

### 研究サマリー・レポート（優先度: 日付降順）

| ファイル | 説明 | ブランチ |
|---------|------|---------|
| `SESSION_SUMMARY_2026-04-04.md` | ⭐ **最新セッションサマリー** — Dyn 2x3x G0.5 が最新ベスト戦略 | compare-trading-strategies-PPCbl |
| `CAGR_IMPROVEMENT_PLAN_2026-04-09.md` | CAGR改善計画v2 | compare-trading-strategies-PPCbl |
| `REGIME_ANALYSIS_REPORT_2026-04-04.md` | レジーム分析レポート | compare-trading-strategies-PPCbl |
| `STRATEGY_COMPARISON_CAGR30per_plus_2026-04-02.md` | 全7戦略最終比較表 | compare-trading-strategies-PPCbl |
| `YEARLY_RETURNS_REPORT_2026-04-01.md` | 7戦略の年次・月次リターン詳細 | compare-trading-strategies-PPCbl |
| `ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md` | 追加分析レポート | compare-trading-strategies-PPCbl |
| `PARAMETER_OPTIMIZATION_REPORT_2026-03-30.md` | パラメータ最適化結果レポート | compare-trading-strategies-PPCbl |
| `PARAMETER_OPTIMIZATION_PLAN_2026-03-30.md` | パラメータ最適化計画書 | compare-trading-strategies-PPCbl |
| `STRATEGY_COMPARISON_2026-03-30.md` | 7戦略比較レポート（旧版） | compare-trading-strategies-PPCbl |
| `LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md` | レバレッジ区間分析 v3（最新） | 全ブランチ |
| `LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md` | レバレッジ区間分析 v2 | 全ブランチ |
| `LEVERAGE_BIN_ANALYSIS_2026-03-19.md` | レバレッジ区間分析 v1 | 全ブランチ |
| `CONTEXT_SUMMARY_2026-02-06.md` | セッション文脈サマリー | 全ブランチ |
| `3x_NASDAQ_Strategy_Research_Summary_2026-02-06.md` | 3倍レバレッジNASDAQ戦略研究全体サマリー | 全ブランチ |
| `FINAL_RESULTS_2026-02-06.md` | 最終戦略比較結果（旧版） | 全ブランチ |
| `R4_RESULTS_SUMMARY_2026-02-06.md` | Round 4バックテスト結果サマリー | 全ブランチ |
| `STRATEGY_RESEARCH_PLAN_R4_2026-02-06.md` | Round 4 研究計画書 | 全ブランチ |
| `plan_2026-02-12.md` | プロジェクト実行計画（旧版） | project-overview-s2l8s |

---

## 戦略の全体像（参考）

```
最終レバレッジ = DD × VT × SlopeMult [× MomDecel]

Layer 1: DD Control（0 or 1）
Layer 2: VT = min(Target_Vol / AsymEWMA_Vol, max_lev)
Layer 3: SlopeMult = clip(0.7 + 0.3 × z_score, 0.3, 1.5)
Layer 4 (推奨追加): MomDecel
```

**推奨戦略**: `Dyn 2x3x (B0.55/L0.25/V0.1/G0.5)` — CAGR 31.40% / Sharpe 1.297 / MaxDD -33.4%（2026-04-04確定）

> **備考**: Pythonソースコード詳細・データファイル・各CSVについては本リポジトリの `src/` ・ルート直下を参照。
