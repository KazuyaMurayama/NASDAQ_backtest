# Tasks — nasdaq_backtest

最終更新: 2026-05-17

> 🎯 **「ベスト戦略は？」と問われたら、まず [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を読むこと。**

## 🔴 In Progress
（なし）

## 🟡 Pending

### CFD動的レバレッジ軸（優先）
- [ ] P2*/S2* 再設計: target_vol ∈ {0.10, 0.13, 0.16, 0.20, 0.25} で再グリッドサーチ（死パラメータ問題の根本対処）
- [ ] S4_RelVol の Sharpe 改善試案（l_base=5以下・IS-OOS gap 14.8pp 縮小）
- [ ] S2 を DH Dynポートフォリオ正式統合シナリオ試算

### DH Dyn軸（既存）
- [ ] Approach A への GAS 切替実装 (閾値 0.15 と同時変更)
- [ ] 2026年データへの拡張（継続監視）
- [ ] Ens2 戦略の OOS 検証（2022-2026）

## ✅ Completed
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
