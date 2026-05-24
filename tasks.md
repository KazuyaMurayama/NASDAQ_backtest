# Tasks — nasdaq_backtest

最終更新: 2026-05-24

> 🎯 **「ベスト戦略は？」と問われたら、まず [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を読むこと。**

## 🔴 In Progress
（なし）

## 🟡 Pending

### 短期（実装フェーズ）
- [ ] **STRATEGY_PERFORMANCE_COMPARISON に F8-R5 列追加**（gen_f8r5_yearly_returns.py 作成 → gen_strategy_comparison.py 更新 → MD再生成）
- [ ] Approach A への GAS 切替実装 (閾値 0.15 と同時変更, nasdaq-strategy-gas 側)
- [ ] 2026年データへの拡張（継続監視）
- [ ] Ens2 戦略の OOS 検証（2022-2026, 完全性のため）

### 中期（新検証アイデア候補）
> 着手前に必ず [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) §3 Rejected で重複チェック、[docs/rules/06_strategy-verification.md](docs/rules/06_strategy-verification.md) の4ステップを実施。

- [ ] **F10 候補**: F8 R5_CALM_BOOST の cap 値細分（calm=0.15→0.20 / bear-VZ=完全停止 等）
- [ ] **F8 + F7v3 ハイブリッド検討**: 静的 cap と動的 cap の時間軸切替（DD閾値ベース等）
- [ ] **VZ_thr 感度再評価**: 現状 ±0.7 固定。±0.5 / ±0.9 で順位入替えあるかスイープ
- [ ] **LT2 N × F8 cap 交差感度**: B6 (N) と F8 (cap regime) を同時グリッド最適化（交互作用未検証）
- [ ] **Trades/yr 削減策**: F8 で 182 trades/yr 過多。新リバランスバンド設計（過去 Partial/Frequency 棄却済み、別アプローチ必須）
- [ ] **Sharpe with Rf>0 再評価**: Rf=4% でランキング入替えあるか（EVALUATION_STANDARD §3.2 注意事項）
- [ ] **P-series 標準コスト再評価**: P01/P02/P05 を Scenario D 統一前提で再計算 → Shortlisted 残存可否判定
- [ ] **G6 WFA 候補**: H4/H5 等の H系を WFA で再評価（過去は単純 OOS のみで棄却。復活可能性は低いが念のため）

### 長期・体系整備
- [ ] [docs/rules/06_strategy-verification.md](docs/rules/06_strategy-verification.md) の4ステップに従ったテンプレ化（PR チェックリスト統合）
- [ ] RESEARCH_CONTEXT.md の系統図を四半期ごとに更新（採用チェーン §2.2 を最新化）

### ⛔ 着手禁止（過去致命的失敗）
- ❌ アンサンブル系（Ens2 派生）— max_lev=3.0系は MaxDD -77% 超で実運用不可
- ❌ A2 ConvictionScore 直接レバ変換（S系）— S3 で IS-OOS gap +22.39pp 致命的過剰適合確認済み
- ❌ Scenario A 単独評価 — コスト過少推計、Scenario D 必須

## ✅ Completed
- 2026-05-24: **他セッション引き継ぎ改善** — `RESEARCH_CONTEXT.md` 新規作成（実験系統図 A〜H 系列、採用チェーン、棄却サマリ、未着手方向性）。`FILE_INDEX.md` を 2026-05-21〜05-24 で追加された A〜H/P/S系 全ファイル（実装・MD・CSV）で更新。`tasks.md` Pending を中期・長期・着手禁止カテゴリで再構成。[RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md)
- 2026-05-24: **G5 WFA for F8 R5_CALM_BOOST PASS → 正式 Active 昇格** — CI95_lo=+27.92%, WFE=+1.208 (α∩β PASS)。Sharpe_OOS=+0.934、CAGR_OOS=+36.83%。`CURRENT_BEST_STRATEGY.md` / `STRATEGY_REGISTRY.md` 更新済み。[G5_WFA_F8R5_2026-05-24.md](G5_WFA_F8R5_2026-05-24.md)
- 2026-05-24: **比較MD更新 v1.4** — F7v3+E4列追加（gen_f7v3_yearly_returns.py作成, gen_strategy_comparison.py更新, 7戦略構成へ拡張）。[STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md](STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md)
- 2026-05-24: **G4 WFA for F7-v3 A:tilt=2.0 PASS → 正式 Active 昇格** — CI95_lo=+27.15%, WFE=+1.203 (α∩β PASS)。CAGR_OOS=+36.30%, Sharpe_OOS=+0.926。`CURRENT_BEST_STRATEGY.md` / `STRATEGY_REGISTRY.md` 更新済み。[G4_WFA_F7V3_2026-05-24.md](G4_WFA_F7V3_2026-05-24.md)
- 2026-05-24: **F8 レジーム条件 tilt スイープ** — R5_CALM_BOOST が Sharpe=+0.934（+0.005 vs F7V3_BASE）、PASS 4 configs。Trades/yr削減は未達、MaxDD若干悪化。R5_CALM_BOOST WFA→G5 Pending。[F8_REGIME_TILT_2026-05-24.md](F8_REGIME_TILT_2026-05-24.md)
- 2026-05-24: **F9 THRESHOLD 最適化スイープ** — THRESHOLD=0.15 が最良（感度低、Sharpe幅0.021）。PASS 5 configs (T005〜T025)、FAIL T030/T040。現行値 0.15 が最適と確認。[F9_THRESHOLD_2026-05-24.md](F9_THRESHOLD_2026-05-24.md)
- 2026-05-24: **F7-v3 Bull-Tilt 定式再設計スイープ** — 定式 A (Large-Tilt Bell): tilt∈{0.6,1.0,2.0,5.0,10.0}/cap=0.10、定式 B (Linear): tilt×cap=4 combos。PASS 4 configs (A:tilt=1.0/2.0/5.0/10.0)。最良 A:tilt=10 (step func): Sharpe=+0.929, CAGR=+36.52%, MaxDD=-62.04%, Trades/yr=179。WFA 実行→G4 Pending。[F7V3_BULL_TILT_2026-05-24.md](F7V3_BULL_TILT_2026-05-24.md)
- 2026-05-24: **G3 WFA for E4** — 50窓WFA PASS (CI95_lo=+26.51%, WFE=+1.131)。E4 正式 Active 確定。[G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md)
- 2026-05-24: **E4 年次リターン生成** — `src/gen_e4_yearly_returns.py` 新規作成、`e4_yearly_returns.csv` を出力。`STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md` §3/§4 を 6 戦略列（E4 を最左）に拡張、v1.2 として改訂履歴を追加。`src/gen_strategy_comparison.py` にも E4 年次リターン列を統合（CSV 優先 + static fallback）。[STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md](STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md)
- 2026-05-24: **E4 Regime k_lt 暫定 Active 昇格** — vz レジーム条件付き k_lt (k_lo=0.1, k_hi=0.8, vz_thr=0.7) が CAGR_OOS +33.53% / Sharpe_OOS +0.891 / IS-OOS gap −1.81pp を達成。`CURRENT_BEST_STRATEGY.md` / `STRATEGY_REGISTRY.md` / `STRATEGY_PERFORMANCE_COMPARISON` / `src/gen_strategy_comparison.py` を更新。WFA は別タスク (🟡 Pending)。[E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md)
- 2026-05-11: **再発防止プロトコル整備** — `CURRENT_BEST_STRATEGY.md` 作成、旧 `FINAL_*` / `*_2026-02-06.md` / `THRESHOLD_SWEEP_REPORT_2026-04-20.md` / `YEARLY_RETURNS_REPORT_2026-04-20_v2.md` に SUPERSEDED ヘッダ追加。CLAUDE.md にベスト戦略参照プロトコルと命名規則を追記。[CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)
- 2026-04-21: **YEARLY_RETURNS_REPORT v3** — Berkshire Hathaway ベンチマーク列追加。DH [A] 1974-2024 CAGR +30.32% vs BRK +19.47% (+10.85pp優位、32-19で勝ち越し)。[YEARLY_RETURNS_REPORT_2026-04-20_v3.md](YEARLY_RETURNS_REPORT_2026-04-20_v3.md)
- 2026-04-21: **Approach A 内リバランス閾値スイープ再検証** — 閾値0.15を推奨（FULL CAGR +30.81% vs 現行0.20の+30.30%）。ダブルチェック3/3 PASS。**現行ベスト戦略: DH Dyn 2x3x [A] 閾値0.15**。[THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md)
- 2026-04-20: YEARLY_RETURNS_REPORT v2（Approach A/B 両方の CAGR をデータから計算）
- 2026-04-20: LEVERAGE_BIN_ANALYSIS V4 rev3（ベスト戦略NAVで前方リターン算定）
- 2026-04-20: CAGR_DISCREPANCY_ANALYSIS（Approach A/B 別名同一名の混在を特定）
- 2026-04-20: APPROACH_A_PROPOSAL（GAS改修設計書 30-50行規模）
- 2026-04-19: main 単一ブランチ運用確立（他ブランチ全削除）
- 2026-04-19: GitHub Actions workflow で旧ブランチ統合完了（src/ Python 42ファイル + CSV + MD + XLSX + overfitting系）
- 2026-04-19: 運用ルール整備（CLAUDE.md / docs/rules/ 群 / tasks.md / FILE_INDEX.md）
- 2026-04-19: archive/2026-04-19_* 旧main CLAUDE.md と .gitignore 退避
- 2026-04-19: 大容量CSV除外＋migrate スクリプト追加
- 2026-04-16: create-file-index 作業で FILE_INDEX.md 初版作成
- 2026-04-09: 研究ドキュメント日付サフィックス付与
- 2026-02-06: **Ens2(Asym+Slope) 推奨戦略確定**（FINAL_RESULTS_2026-02-06.md）
