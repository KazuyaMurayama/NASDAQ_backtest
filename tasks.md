# Tasks — nasdaq_backtest

最終更新: 2026-05-18 (P5完了)

> 🎯 **「ベスト戦略は？」と問われたら、まず [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を読むこと。**

## 🔴 In Progress
（なし）

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
