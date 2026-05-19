# Tasks — nasdaq_backtest

最終更新: 2026-05-18 (Phase T2 完了 — T1/T2 REJECT、Phase T3 へ)

> 🎯 **「ベスト戦略は？」と問われたら、まず [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を読むこと。**

## 🔴 In Progress

### タートル流投資手法 NASDAQ 3xレバ適用研究 (Phase T3〜T6)
- [x] Phase T1: `src/turtle_core.py` / `turtle_state.py` / `turtle_costs.py` + 42ユニットテスト ✅
- [x] Phase T2: T1/T2 単独バックテスト — **T1 REJECT (CAGR 29.66%, MaxDD -60.85%), T2 BLOW-UP (1981口座破綻)** ✅
- [ ] Phase T3: T3-T6 統合バックテスト (Gate / Sized / Sleeves / Hybrid Stop)
  - 優先順位: T6 (Hybrid Stop) > T4 (Sized) > T3 (Gate) > T5 (Sleeves)
  - 理由: T1の「2022 -0.99% (vs Baseline -30.55%)」が示すクラッシュ防御効果のみ借用が最有望
- [ ] Phase T4: 過学習確認 (DSR + 5-Fold WF-CV) — Phase T3通過案件のみ
- [ ] Phase T5: ブロックブートストラップ B=2000 — Phase T4通過案件のみ
- [ ] Phase T6: 採用判断 → CURRENT_BEST_STRATEGY.md 更新 (採用時)

## 🟡 Pending

### Gold/Bond スリーブ拡張軸（次フェーズ判断課題）
- [ ] **H4 採用可否最終判断**（OOS Sharpe 0.837/Gap 0.3pp は魅力的だがWorst5Y -2.24%）
- [ ] **H2/H3 リスク許容判断**（高Sharpe but MaxDD -79% は実運用に難）
- [ ] Worst5Y ≥ 0% 必須条件を満たす代替仮説の探索（H4派生 wg/wb 比率調整、Bond軽減度の最適化）
- [ ] H4採用時のCURRENT_BEST_STRATEGY.md 更新

### CFD動的レバレッジ軸（優先）
（CFD動的レバレッジの主要タスク完了。以下は次フェーズ課題）
- [ ] S2 DH統合の実運用可否判断（MaxDD -62% は許容範囲か？スリーブ比率調整）

### DH Dyn軸（既存）
- [ ] Approach A への GAS 切替実装 (閾値 0.15 と同時変更)
- [ ] 2026年データへの拡張（継続監視）
- [ ] Ens2 戦略の OOS 検証（2022-2026）

## ✅ Completed
- 2026-05-19: **S2_VZGated 年次リターン表生成 + CFD定義バグ修正** — CFD 3x/7x [固定] を純NASDAQから DH Dynポートフォリオ+固定CFDレバに修正（+9.74%→+23.20%、-39.27%→+41.36%）。サニティチェック4件PASS。[CFD_S2_YEARLY_RETURNS_2026-05-17.md](CFD_S2_YEARLY_RETURNS_2026-05-17.md) / [SESSION_SUMMARY_2026-05-19.md](SESSION_SUMMARY_2026-05-19.md)
- 2026-05-18: **Phase T2: T1/T2 純タートルバックテスト完了** — H1棄却確定。T1 (Long-Only): CAGR_FULL +29.66% (vs 正典 30.81% ❌), Sharpe 0.986 (vs 1.298 ❌), MaxDD -60.85% (vs -31.36% ❌), Worst5Y -14.86% (vs +4.77% ❌)。234トレード、56.8%が2N stop。**ただしクラッシュ年防御は強力**: 2022 -0.99% (+29.6pp 優位), 1981 +29.45% (+61.8pp), 1988 +18.38% (+32.5pp)。T2 (Long/Short): 1981 NASDAQ強反発期のSQQQショート連鎖損失で口座破綻 (95% drawdown floor で停止、Final $5,000)。次は Phase T3 で「2N stop だけ DH Dyn [A] に注入」(T6 Hybrid Stop) を最有望候補として検証。[T1_T2_RESULTS_2026-05-18.md](T1_T2_RESULTS_2026-05-18.md)
- 2026-05-18: **Phase T1: タートルコアモジュール実装** — `src/turtle_core.py` (Wilder SMMA ATR / Donchian / unit_size), `src/turtle_state.py` (TurtleState 状態機械, 4 Unit pyramiding, 2N stop, S1 skip flag), `src/turtle_costs.py` (slippage 0.30%/side 確定, TQQQ daily holding cost), `tests/test_turtle_core.py` (42 ユニットテスト全件グリーン)。論点5: B (T1/T2 max 4 Unit、T5 各3スリーブ独立 4×3=12 Unit), 論点G: B (0.30%/side) で確定。
- 2026-05-18: **タートル流投資手法 調査・適用計画完成 (Opus設計)** — Curtis Faith「Way of the Turtle」原典に基づく純正ルール仕様調査 (System 1/2, ATR Wilder SMMA, Unit sizing, Pyramiding 0.5N×4, 2N stop, 4/6/10/12 risk limits) と、NASDAQ 3xレバ環境への適用バックテスト計画 (H1/H2/H3 仮説, T1-T7 変種, Phase T1-T6 ロードマップ)。事前確率の所見: P1-P5 経験から T3 (ゲート) は通過確率低、T6 (2N stop追加) と T4 (サイジング借用) に重点配分。[TURTLE_RESEARCH_2026-05-18.md](TURTLE_RESEARCH_2026-05-18.md) / [TURTLE_RESEARCH_PLAN_2026-05-18.md](TURTLE_RESEARCH_PLAN_2026-05-18.md)
- 2026-05-18: **P5: ブロックブートストラップ・ストレステスト完了 (B=2000, L=21)** — ADOPT確定ゼロ。Dyn系3コンボはMARGINAL → **GRAY維持** (絶対水準は良好だが C8: ΔSharpe 5%ile > 0 を満たず、p_value 0.088-0.091で C9をわずかに通過)。HY/MA系は**REJECT確定** (P06はFRAGILE)。**結論: タイミングゲートはBaseline比優位性が統計的に確立できず、現行DH Dyn [A]維持を最終推奨。** [P5_BOOTSTRAP_STRESS_2026-05-18.md](P5_BOOTSTRAP_STRESS_2026-05-18.md)
- 2026-05-18: **P4: 過学習確認完了 (DSR + 5-Fold WF-CV)** — ADOPT候補ゼロ。Dyn系3コンボ (P01/P02/P03) はGRAY (PSR 0.92〜0.93、N_eff=8)、HY/MA組合せ系はREJECT。CV median CAGRが15%閾値未達 (最高P01=10.8%)。**結論: 現行ベースライン (DH Dyn [A]) 維持を推奨。タイミングゲート戦略は統計的有意性が不十分。** [P4_OVERFITTING_CHECK_2026-05-18.md](P4_OVERFITTING_CHECK_2026-05-18.md)
- 2026-05-18: **P3: シグナル組合せ バックテスト完了 (34コンボ)** — 全コンボが CAGR_OOS < 20% (primary tier 未達)。Secondary best: HY×CPI (OOS=+15.65%, Worst5Y=+6.04%, 2022=-31.31%)。最良2022防御: Dyn(w60,m0.1)×CPI (2022=-14.33%, OOS=+19.23%, Worst5Y=+0.12%)。**コア発見: Dyn_Corr単独で2022を保護するが Worst5Y をベースライン以下に押し下げる。NAS-gate追加は2022防御を損なわずWorst5Yをわずか改善するが依然ベースライン未達。**[P3_COMBINATION_RESULTS_2026-05-18.md](P3_COMBINATION_RESULTS_2026-05-18.md)
- 2026-05-18: **P2: Top5シグナル単独バックテスト完了 (38コンボ)** — Dyn_Correlation のみが2022を実質的に削減 (-30.55%→-15.97%)。HY/CPI/YC/MA は2022に効果なし。採用基準 (OOS≥20%) 達成ゼロ。[P2_SINGLE_SIGNAL_RESULTS_2026-05-18.md](P2_SINGLE_SIGNAL_RESULTS_2026-05-18.md)
- 2026-05-18: **P1: タイミング戦略データ取得完了** — 7 FRED系列取得・3段階HYスプライス・CPI 15営業日ラグ補正。timing_signals_raw.csv (13,169行×5列) 生成。[P1_DATA_FETCH_RESULTS_2026-05-18.md](P1_DATA_FETCH_RESULTS_2026-05-18.md)
- 2026-05-18: **タイミング戦略調査計画 完成 (Opus設計)** — CFD7/S2の大幅ドローダウン問題（1981/1988/1994/2015/2022）に対応する未試験シグナル Top5 を特定。HY Credit Spread / YC+FF / 動的相関 / CPI Momentum / MA200+CPPI。8週間実装ロードマップ（P1データ→P6 DH Dyn [B]確立）。CAGR ≥ 22%・Worst5Y -2〜+1%・Sharpe 1.15〜1.25 を目標。[TIMING_STRATEGY_RESEARCH_PLAN_2026-05-18.md](TIMING_STRATEGY_RESEARCH_PLAN_2026-05-18.md)
- 2026-05-18: **方針A: DH Dyn [A] IS/OOS スタンドアローン検証** — 10コンボ全滅 (0/10 pass)。CAGR_IS構造的天井 ≈ 23%（TQQQ 3x + SOFR financing 約-8%/yr）を確認。正典 (gold=2x, wg=0.50): CAGR_IS=+23.13% ❌ / Sh_IS=1.030 ✅ / Worst5Y=+0.66% ✅。CAGR ≥ 25% は TQQQ 3x ベースでは構造的に不可能。[A_DH_STANDALONE_VERIFY_2026-05-18.md](A_DH_STANDALONE_VERIFY_2026-05-18.md)
- 2026-05-18: **C2スイープ: wn_max 実効化 (0.40-0.70) で W5Y 改善を検証** — 48コンボ全滅 (0/48 pass)。W5Y ベスト = -4.12% (基準 -3% に 1.12pp 不足)。wn_max 引き下げが逆にW5Yを悪化させる逆説を発見。2022 Triple Bear (NASDAQ+Gold+Bond 同時下落) が根本原因。[C2_NASDAQ_CAP_SWEEP_2026-05-18.md](C2_NASDAQ_CAP_SWEEP_2026-05-18.md)
- 2026-05-17: **H1〜H5 Gold/Bond仮説 バックテスト完了**
- 2026-05-17: **S2_VZGated Rolling Window CV 検証**
- 2026-05-16: **CFD動的レバレッジ S1/S2/S3/S4 バックテスト完了**
- 2026-05-15: **CFD固定レバレッジ & 動的レバレッジ P1〜P5 バックテスト**
