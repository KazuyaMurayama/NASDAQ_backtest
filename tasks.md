# Tasks — nasdaq_backtest

最終更新: 2026-05-17

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
- 2026-05-17: **H1〜H5 Gold/Bond仮説 バックテスト完了** — 5仮説全実行。**H2 (S2+Gold5x): Sharpe 0.881** だが MaxDD -79.5%。**H4 (TOCOM+Bond軽減): Sharpe 0.837, Gap 0.3pp** で最ロバスト。⚠️ **Worst5Y≥0%必須基準は全仮説で未達** (最良はH5 -0.17%)。BL_S2超えは H2/H4 の2件。[H1_H5_SUMMARY_2026-05-17.md](H1_H5_SUMMARY_2026-05-17.md)
- 2026-05-17: **Gold/Bond 商品選定とNASDAQ統合戦略 計画書** — 5仮説（H1〜H5）立案完了。Gold候補3本（1540+信用2x / SBI金CFD / TOCOM先物）、Bond候補（TMF維持）、コストモデル統一、採用基準明文化、感度分析計画。[GOLD_BOND_STRATEGY_PLAN_2026-05-17.md](GOLD_BOND_STRATEGY_PLAN_2026-05-17.md)
- 2026-05-17: **S2 DH Dyn統合シナリオ試算** — S2 CFD+DH: CAGR +32.35% (+9.85pp vs TQQQ)、OOS Sharpe 0.769 (+0.123)、IS-OOS Gap 5.4pp。ただしMaxDD -62.36% (-17.27pp)、Worst5Y -4.73% (-5.60pp)。高リターン・高リスクのトレードオフ確認。[S2_DH_INTEGRATION_2026-05-17.md](S2_DH_INTEGRATION_2026-05-17.md)
- 2026-05-17: **S4_RelVol Sharpe改善グリッドサーチ** — 192コンボ中採用基準パス0件。最大Sharpe 0.716 (閾値0.757未達)。S4不採用確定。[S4_SHARPE_SWEEP_2026-05-17.md](S4_SHARPE_SWEEP_2026-05-17.md)
- 2026-05-17: **P2*/S2* 低target_vol グリッドサーチ** — tv∈{0.10〜0.25}は全て機能的(clip_rate=0%)だが最大Sharpe 0.681（採用S2の0.769未達）。採用S2(tv=0.80)が最良。[S2_LOW_VOL_SWEEP_2026-05-17.md](S2_LOW_VOL_SWEEP_2026-05-17.md)
- 2026-05-17: **S2_VZGated Rolling Window CV 検証** — 46窓（1980〜2025, 5yr IS/1yr OOS）で検証。MaxDD勝率63%（S2>P2）、Sharpe勝率41%（1年窓誤差範囲内）、CAGR>0率72%（P2と同等）。危機年（1987/1990/2008/2011/2022）での優位確認。採用維持確定。[S2_ROLLING_CV_2026-05-17.md](S2_ROLLING_CV_2026-05-17.md)
- 2026-05-16: **CFD動的レバレッジ S1/S2/S3/S4 バックテスト完了** — S2_VZGated 採用確定（OOS Sharpe 0.769, Worst5Y -4.75%, IS-OOS Gap 5.4pp）。P0検証3件（SOFR単位✅ / target_vol99.7%クリップ⚠️ / Worst5Y定義✅）。S1/S3/S4 は不採用。[SESSION_SUMMARY_2026-05-16.md](SESSION_SUMMARY_2026-05-16.md) / [CFD_DYNAMIC_LEVERAGE_GUIDE.md](CFD_DYNAMIC_LEVERAGE_GUIDE.md) / [ENH_LEVERAGE_BACKTEST_2026-05-16.md](ENH_LEVERAGE_BACKTEST_2026-05-16.md)
- 2026-05-15: **CFD固定レバレッジ & 動的レバレッジ P1〜P5 バックテスト** — P2(vol-targeting, target_vol=0.8) が OOS Sharpe 0.757 でベースライン。[DYN_LEVERAGE_BACKTEST_2026-05-15.md](DYN_LEVERAGE_BACKTEST_2026-05-15.md) / [CFD_LEVERAGE_BACKTEST_6x7x_2026-05-15.md](CFD_LEVERAGE_BACKTEST_6x7x_2026-05-15.md)
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
