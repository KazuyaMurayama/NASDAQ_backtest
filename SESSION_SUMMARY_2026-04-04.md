# セッションサマリー: Gold日次データ化・Step1-2検証・レジーム分析

*2026-04-04 作成*

## TL;DR（3行要約）

- **Goldデータを月次補間→LBMA日次**に改善し、ベスト戦略が G0.6 → **G0.5** に変化（CAGR 31.40%, Sharpe 1.297）
- **Step 1-2検証**で「Gold/Bondのタイミング追加」を190指標テスト + 27パターンWF検証し、**改善なし**と統計的に判定 → 現行戦略を維持
- **配分レジーム分析**で戦略の挙動を3×5ビンで可視化、全ビンで中央値の単調性を確認

## 概要

本セッションでは、Dyn 2x3x戦略の **データ品質改善**、**改善余地の統計的検証**、
**戦略の挙動理解** を段階的に進めた。最終的に現行戦略の完成度の高さを確認し、
運用実装フェーズ（別セッション）へ引き継ぐ準備を整えた。

**主要トピック**:
1. Goldデータを月次補間 → LBMA日次（1968年〜）に置き換え
2. ベスト戦略の再最適化（G0.6 → G0.5）
3. Step 1: Gold/Bondタイミングシグナルの統計的予測力検証（190テスト）
4. Step 2: 有望シグナルのDyn 2x3x組み込みバックテスト
5. 配分レジーム分析（3×5ビンのフォワードリターン統計）
6. 運用論点の棚卸し（GAS自動化設計）

---

## 1. 重要な状況・結論

### 1.1 現在のベスト戦略: Dyn 2x3x (B0.55/L0.25/V0.1/G0.5)

| 指標 | 値 |
|------|-----|
| CAGR | **31.40%** |
| Sharpe | **1.297** |
| MaxDD | -33.4% |
| Worst 5Y | +5.2% |
| Worst 10Y | +14.9% |
| OOS CAGR | 22.15% |
| OOS Sharpe | 0.914 |
| Walk-Forward 7窓平均 | 1.117 |

### 1.2 Goldデータ改善の影響

- **旧**: GitHub月次データの日次線形補間（1974-2000のボラティリティ過小評価）
- **新**: LBMA AM Fix 日次価格（14,720行、1968-01-02 〜 2026-03-31）
- 結果: Pre-2000の年率ボラティリティが実勢（約21.6%）に回復
- ベスト戦略のGold/Bond比率が **G0.6 → G0.5** に変化（均等配分が最適化）

### 1.3 Step 1 検証結果（Gold/Bondタイミング予測力）

- **190テスト**（Gold 50指標 + Bond 51指標 × 3ホライズン × 7WF窓）
- **55組合せ**が多重検定補正後（|IC|>0.03, WF安定5/7以上, FDR有意）フィルター通過
- **Gold**: 全て負のIC（平均回帰型）。IC絶対値が小さく実用性は限定的
- **Bond**: 正負両方のICあり。MA5（NQ-Bond相関）とMACDが有望

### 1.4 Step 2 検証結果（Bondタイミング組み込み）

27パターンのバックテスト + 7窓Walk-Forward検証で：
- ベースライン: WF7 1.117, Sharpe 1.055, MaxDD -33.4%
- 全拡張版がベースラインと**同等またはMaxDD悪化**
- **結論: Bondタイミングは改善をもたらさない。現行Dyn 2x3xを維持**

### 1.5 レジーム分析の主要発見

| レジーム | 頻度 | 5日後CAGR年率 | プラス確率 |
|---------|:---:|:---:|:---:|
| 強気フル投資 (70-90% × 85-100%) | **37.1%** | +55.2% | 61% |
| 暴落退避 (50-70% × 0%CASH) | 16.2% | +10.4% | 53% |
| 回復途上 (50-70% × 30-60%) | 14.2% | +16.5% | 56% |
| 上昇基調 (70-90% × 60-85%) | 8.9% | +35.0% | 59% |

- 全10ビンで中央値の単調性が成立（rawLevが高いほどリターンが高い）
- Gold+Bondの防御効果: 暴落退避時の下位10%を -4.69% → -1.80% に緩和

---

## 2. 本セッションで作成・更新したファイル（ハイパーリンク一覧）

### 2.1 レポート類（MD / Excel）

| ファイル | 内容 |
|---------|------|
| [STRATEGY_COMPARISON_CAGR30per_plus.md](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/STRATEGY_COMPARISON_CAGR30per_plus.md) | **最終版** 7戦略比較レポート（複数指標 + 年次・月次リターン） |
| [REGIME_ANALYSIS_REPORT.md](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/REGIME_ANALYSIS_REPORT.md) | **配分レジーム分析**: 3×5ビンのフォワードリターン統計 |
| [YEARLY_RETURNS_REPORT.md](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/YEARLY_RETURNS_REPORT.md) | 7戦略 年次・月次リターン（Dyn 2x3x含む最新版） |
| [YEARLY_RETURNS_7STRATEGIES_v2.xlsx](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/YEARLY_RETURNS_7STRATEGIES_v2.xlsx) | データバー付きExcel版（Dyn 2x3xで更新） |

### 2.2 ソースコード（src/）

| ファイル | 用途 |
|---------|------|
| [src/test_portfolio_diversification.py](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/src/test_portfolio_diversification.py) | `prepare_gold_data()` をLBMA日次データ対応に更新 |
| [src/research_gold_bond_timing.py](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/src/research_gold_bond_timing.py) | **Step 1**: 190タイミング指標の予測力検証 |
| [src/research_step2_bond_timing.py](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/src/research_step2_bond_timing.py) | **Step 2**: Bondタイミング組み込みバックテスト |
| [src/regime_analysis.py](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/src/regime_analysis.py) | 配分レジーム分析スクリプト（3×5ビン） |
| [src/step_update_dyn2x3x.py](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/src/step_update_dyn2x3x.py) | Dyn 2x3xのG0.5パラメータで更新 |
| [src/gen_yearly_md.py](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/src/gen_yearly_md.py) | 新CAGRs (31.40%等) でMD生成 |
| [src/gen_full_excel.py](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/src/gen_full_excel.py) | 新CAGRs (31.40%等) でExcel生成 |

### 2.3 データファイル

| ファイル | 内容 |
|---------|------|
| [data/lbma_gold_daily.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/data/lbma_gold_daily.csv) | **LBMA AM Fix 日次金価格** (14,720行、1968-2026) |

### 2.4 計算結果CSV

| ファイル | 内容 |
|---------|------|
| [lev2x3x_results.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/lev2x3x_results.csv) | LBMA日次データでの49パターン再最適化 |
| [yearly_returns_7strategies.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/yearly_returns_7strategies.csv) | 7戦略の年次リターン |
| [monthly_returns_oos.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/monthly_returns_oos.csv) | 7戦略の月次リターン（OOS） |
| [research_gold_signals.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/research_gold_signals.csv) | Gold 50指標 × 3ホライズン = 150テスト結果 |
| [research_bond_signals.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/research_bond_signals.csv) | Bond 51指標 × 3ホライズン = 153テスト結果 |
| [research_all_corrected.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/research_all_corrected.csv) | 多重検定補正適用済み結果 |
| [research_summary.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/research_summary.csv) | フィルター通過55組合せ |
| [research_correlation_regimes.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/research_correlation_regimes.csv) | 3資産ローリング相関統計 |
| [research_regime_returns.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/research_regime_returns.csv) | DD/Rallyレジーム別リターン |
| [step2_baseline.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/step2_baseline.csv) | Step 2 ベースライン |
| [step2_partB_ma5.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/step2_partB_ma5.csv) | Step 2 MA5のみ感度分析 |
| [step2_partC_ma5_macd.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/step2_partC_ma5_macd.csv) | Step 2 MA5+MACD感度分析 |
| [step2_partD_full.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/step2_partD_full.csv) | Step 2 3シグナル27パターン |
| [step2_partE_wf.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/step2_partE_wf.csv) | Step 2 Walk-Forward 7窓検証 |
| [step2_summary.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/step2_summary.csv) | Step 2 最終サマリー |
| [regime_analysis_stats.csv](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/claude/compare-trading-strategies-PPCbl/regime_analysis_stats.csv) | レジーム分析の生データ |

---

## 3. 進展とClaude Codeの有効活用

### 3.1 主要な進展（時系列）

1. **Goldデータ品質改善**
   - 月次補間の問題点を特定 → LBMA日次データ(1968+)に置き換え
   - ベスト戦略を再最適化（49パターン）、G0.5が最良と判明

2. **Step 1: 統計的予測力検証**
   - 190指標テスト、多重検定補正（Bonferroni-Holm + BH FDR）
   - GoldとBondそれぞれの予測力を統計的に評価

3. **Step 2: 組み込みバックテスト**
   - 有望シグナルをDyn 2x3xに統合、27パターン × 7WF窓で検証
   - **改善なし** と結論、現行戦略を維持

4. **レジーム分析**
   - 戦略の挙動を3×5ビンで可視化、フォワードリターン統計で理解を深化
   - 全ビンで中央値の単調性を確認 → シグナル設計の妥当性を裏付け

5. **運用検討への移行**
   - GAS自動化の設計検討
   - 別セッションで実装開始

### 3.2 Claude Codeの有効利用パターン

| 活用法 | 具体例 |
|-------|--------|
| **並列Agent実行** | データソース調査とリポジトリ状況確認を同時実行（Step 1初期） |
| **Explore Agent** | GASリポジトリの既存コード構造の網羅調査 |
| **Plan Agent** | Step 1の検証計画策定（190指標の設計） |
| **細粒度タスク分割** | タイムアウト対策として Part 0-7を個別実行、各段階でCSV保存 |
| **TodoWrite管理** | 長時間タスクの進捗を可視化、意図しない中断を防止 |
| **中間コミット** | 各パート完了毎にgit commit、作業ロストを回避 |
| **品質チェック自動化** | 単調性チェック、全体CAGR整合性チェックなどをスクリプト内に組み込み |

### 3.3 特に効果的だった取り組み

- **データ品質の疑念を即座に実行に移した点**: 「月次補間で良いか？」の疑問から即座にLBMA日次データへの置き換えを実行し、ベスト戦略の更新までをワンセッションで完遂
- **否定的結果の明確化**: Step 2で「改善なし」と結論付けた点。無理に続けず、統計的にクリアな判定を行った
- **戦略の挙動可視化**: レジーム分析により、「どんな局面で」「どう振る舞い」「どの程度の期待値か」を体系化

### 3.4 学び・教訓

1. **データ品質が戦略選定を左右する**: Gold補間の誤差はGold比率の選択を歪めていた
2. **統計的検証 ≠ 実用改善**: Step 1で有意な指標が見つかっても、Step 2の組み込みで改善しないことがある
3. **オーバーフィット検出の重要性**: Walk-Forward 7窓検証は、見かけの改善を排除する上で不可欠
4. **戦略の「完成度」の見極め**: 無理な改善追求より、現行戦略の理解を深める方が実運用に直結する

---

## 4. 次セッションへの引き継ぎ事項

### 4.1 進行中タスク

- **GAS自動化実装**: 別セッションで開始済み
  - 既存の単一資産NASDAQ用GASを、Dyn 2x3x（3資産）用に拡張
  - 約180行の追加・変更（VIXレイヤー、Gold/Bond取得、配分計算、通知）

### 4.2 未着手の運用論点

1. 商品確定（TQQQ、2036、TMFがSBI証券で買付可能か、NISA対応か）
2. リバランスルール（頻度、閾値、執行タイミング、税金）
3. 為替リスク（ドル建て vs 円建て、ヘッジの要否）
4. 初期投資・資金管理（最低投資額、段階的入金、生活防衛資金）
5. コスト構造（売買手数料、スプレッド、為替手数料、税金）
6. リスク管理（損切り、撤退基準、ブラックスワン対処）
7. 運用開始・移行計画
8. 継続運用（監視、見直し基準）

### 4.3 検討済みだが採用しなかった案

- 第4資産の追加（原油2x、VIX系、インバース）— 改善余地が小さいと判断
- Bondタイミングシグナルの組み込み — Step 2で改善なしと判定
- さらなるA2パラメータ最適化 — 既に高度に最適化済み、改善余地2%未満

---

## 5. セッション全体のファイル統計

- **コミット数**: 10
- **変更ファイル**: 29（新規作成16、更新13）
- **MDレポート**: 3（+更新1）
- **Pythonスクリプト**: 5（新規）
- **データCSV**: 14（新規）

---

*Generated: 2026-04-04*

