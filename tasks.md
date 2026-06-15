# Tasks — nasdaq_backtest

最終更新: 2026-06-12

> 🎯 **「ベスト戦略は？」と問われたら、まず [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) を読むこと。**

## 🔴 In Progress

### IS高CAGR再発掘×レバ強化（2026-06-12 検証完了・昇格審議待ち）
> 正典: [REEVALUATION_AND_LEVERUP_PLAN_20260612.md](REEVALUATION_AND_LEVERUP_PLAN_20260612.md) / 結果: [REEVALUATION_RESULTS_20260612.md](REEVALUATION_RESULTS_20260612.md)（Phase A）+ [LEVERUP_SWEEP_RESULTS_20260612.md](LEVERUP_SWEEP_RESULTS_20260612.md)（Phase B/C・6次元採点）
- [x] A1: F10ε/F8-R5/F7v3/vz065_l7 realistic再計算 → **全CLOSE**（IS⓽16.6〜16.8%・MaxDD−67〜−69%）。P01_w63もCLOSE
- [x] B1: LU2フルゲート初通過（ベト無し）。C1（bondOFF日SOFR計上）= 全戦略+0.25pp
- [x] B2-B4: 44構成スイープ → min⓽≥20%が12構成・ベト抵触ゼロ。B4(post-hoc条件付きブースト)はTop8に残らずクローズ
- [x] C4: >3x超過分をくりっく株365へ（コスト2.5%→0.25%/yr）→ B3系+0.3〜0.4pp
- [x] **QC v2/v3（2026-06-13〜15）**: 3エージェント独立レビュー＋独立再実装QC。数値・コードPASS（§6.1値80セル中79を別実装で再現）。採点はバランス重視＝P09_C1首位／CAGR重視＝B3a首位（飽和採点の限界を是正）。サインオフ=[LEVERUP_QC_SIGNOFF_20260613.md](LEVERUP_QC_SIGNOFF_20260613.md)
- [x] **multi-metric bootstrap（2026-06-15・教訓C遵守・留保6解消）**: MaxDD/Worst10Y★/Sharpe を対V7・対P09で取得。B3aの交換条件確定（CAGR/10年テール取得・Sharpe中立・MaxDD悪化）
- [x] **ベスト戦略確定（2026-06-15・ユーザー決定）**: **ベスト戦略=B3a_k365（CAGR重視・min⓽+20.98%）／ベスト戦略候補=P09_C1（バランス重視・採点首位）**。CURRENT_BEST_STRATEGY.md §🏆 v8 ＋ STRATEGY_REGISTRY §1/§2 に反映済。B3c_k365（DD約1pp削減版）も §2 登録
- [ ] **本番切替（7月上旬判断）**: E4→P09 切替判断と一体で B3a/P09_C1 本番化を審議。GAS CONFIG（BOOST_MAP/STRATEGY）切替で追従。C1/C4は採否と独立に実運用ガイド化推奨
- [ ] **改善検討（QC由来・[§10](LEVERUP_SWEEP_RESULTS_20260612.md)）**: I1 防御オーバーレイ併用（既ADOPT mom63×M6 def を B3基盤で再検証＝MaxDD−5pp級・最有力）／I2 vol-target／I3 C2 bondOFF防御資産化

### P09_TQQQ GAS並走運用（2026-06-11〜・NASDAQ-strategy-gas リポと連携）
> 正典手順: NASDAQ-strategy-gas/docs/P09_GAS_MIGRATION_PLAN_20260611.md
- [x] 凍結spec/golden エクスポート（`export_p09_live_spec_20260611.py`）・Python/JS パリティ証明（7チェック PASS）
- [x] 実機 installP09 デプロイ（PARITY ALL PASS）→ 並走開始（2026-06-11）
- [x] 通知フォワードリターンを **P09実戦略ビン別表**へ置換（[P09_FORWARD_RETURN_REPORT_20260612.md](P09_FORWARD_RETURN_REPORT_20260612.md)。最強ブーストIN_Q0_hi=+121%/年・勝率69%）
- [ ] 並走2〜4週監視（〜2026-07上旬目安）→ Phase 4.4 切替判断（E4→P09、ユーザー承認）
- [ ] **年1回（次回2027-06目安）**: mom63凍結四分位境界＋フォワードリターン表の再生成（`p09_forward_return_20260612.py` 再実行→GAS定数更新→変更レポート）

### マルチアセット2軸最適化（NASDAQ/Gold/Bond/Cash × レバレッジ）2026-06-10〜
> 引き継ぎ必読: [MULTIASSET_SESSION_HANDOFF_20260610.md](MULTIASSET_SESSION_HANDOFF_20260610.md) / 計画: [MULTIASSET_2AXIS_OPTIMIZATION_PLAN_20260609.md](MULTIASSET_2AXIS_OPTIMIZATION_PLAN_20260609.md)
- [ ] **2軸最適化で DH-W1（min税後CAGR +18.10%, MaxDD −34.57%）を超える**。配分軸に **CASH を含む**。報告は標準10指標・表・太字。
  - [ ] Phase 0: `leverage_eval.ten_metrics()`（min(IS,OOS)税後CAGR込, split=2021-05-08）追加（TDD）
  - [ ] Phase 1: `portfolio_engine.py`（4資産×per-assetレバ純コスト後・税後NAV、NASDAQスリーブはDH-W1同等以上）
  - [ ] Phase 2: `optimize_2axis.py`（配分×レバ結合探索→DH-W1 3軸同時超えフィルタ→動的配分上積み）
  - [ ] Phase 3-4: WFA/bootstrap確定→10指標太字表→CURRENT_BEST/REGISTRY/tasks更新
- 確定済シグナル: Gold=`m252_tv0.10_z0.75_mo` / Bond=`m252_tv0.05_z1.0_wk`（WFA+bootstrap PASS）。Bondは分散役（cash超えず）。
- ⚠ 旧 `MULTIASSET_INTEGRATED_20260608.md` / `MULTIASSET_CURRENT_BEST.md` の配分・レバ結論は **ベンチ未達(+8.9%)で破棄**。構築モジュール・確定シグナルのみ再利用。

## 🟡 Pending

### 短期（実装フェーズ）
- [ ] **DH-W1 Cash-Sleeve P2/P7 の正式 Active 昇格判定**: P2 GOLD100 / P7 GOLD75BOND25(⭐推奨) の **block bootstrap + permutation(t_p) 統計検証**を実施（WFA 50窓 α∩β は PASS 済み、t_p/bootstrap が未実施）。PASS なら投信環境 Active として CURRENT_BEST_STRATEGY.md §1 相当へ昇格。`g23e_dh_w1_bootstrap.py` 相当を流用。
- [ ] **CURRENT_BEST_STRATEGY.md §1 Active 昇格判断 (v4.5 以降)**: vz=0.65+l7+F10ε を CFD 環境 Active として §1 正式昇格させるか、E4 RegimeKLT を維持するか。実運用変更 (Googleスプレッドシート同期等) を伴うためユーザー承認必須
- [ ] **DH-W1 を ETF only 環境向け副 Active として CURRENT_BEST_STRATEGY.md に正式登録**: 環境別 Active 制度の運用ルール確立
- [ ] **CURRENT_BEST_STRATEGY.md 更新判断 (v4 以前)**: vz065+lmax5 vs F10+lmax5 vs E4 の採用変更を最終決定（INTEGRATION_DEBATE_2026-05-26.md 参照、v4.5 で部分的に整理済）
- [ ] **STRATEGY_PERFORMANCE_COMPARISON v1.7**: F10 ε=0.015 / vz065+lmax5 / F10+lmax5 の3候補列を追加
- [ ] **p10_f10lmax5_fullmetrics.py Trades/yr バグ修正**: 現状は lev_raw 変化のみカウント（=27/yr）→ lev_raw+wn/wb 変化を合算（=52/yr）に修正。CSV も再生成
- [ ] Approach A への GAS 切替実装 (閾値 0.15 と同時変更, nasdaq-strategy-gas 側)
- [ ] 2026年データへの拡張（継続監視）
- [ ] Ens2 戦略の OOS 検証（2022-2026, 完全性のため）

### 中期（新検証アイデア候補）
- [ ] **G10 WFA for F10+lmax5**: 現行 G8 CI95_lo=+25.57% を +27%+ 到達可能か別設定で調査（窓幅・OLS推定など）
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
- **2026-06-08 (v4.9 ルール簡素化)**: 保守的採用基準を簡素化 ─ v4.5 で導入した「3 軸 (min + Worst10Y + P10_5Y) すべて baseline 以上」必須条件は過度に restrictive と判断され撤回。**min(IS, OOS) CAGR の標準化のみ残存**、Worst10Y/P10_5Y は §3.12 9 指標として参照するが強制条件ではない。CLAUDE.md §2 / EVALUATION_STANDARD.md §3.13 (v1.5→v1.6) / STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md (v4.8→v4.9) §7-2 / CURRENT_BEST_STRATEGY.md v4.5 セクションを同期更新。過去判定 (AH/AT/HL 棄却等) は変更しない。
- **2026-06-07 (DH-W1 Cash-Sleeve 4戦略)**: DH-W1 (ETF only) の OUT(キャッシュ 46.9%/6,171日)期を 1 倍投信で運用置換するシミュレーションを実装・評価 (`analysis_cash_sleeve/cash_sleeve_sim.py`)。**投信環境 Active 候補 4 戦略**を確定: DH-W1 baseline / **P2 GOLD100**(攻め, OOS +16.44% 最高/Sharpe +0.875/gap +0.97/WFE 1.229) / **P7 GOLD75BOND25** ⭐(中庸推奨, OOS +14.90%/MaxDD −48.23%/WFE 1.043) / **P5 GOLD50BOND50**(守り, MaxDD −35.97% 最良/CI95_lo +17.23%/gap +5.50)。全 WFA 50窓 α∩β PASS。検証済み: OUT 期資産挙動(NASDAQ −7.63%/Gold +4.07%/Bond +6.51%)、全商品コスト表(SOFR/TER/スワップ/売買/税 20.315%×0.8273)、執行ラグ(レバ脚 TQQQ/TMF/2036 は DELAY=2/T+2、投信スリーブ 5BD)、年次リターン表(1974-2026 税後)。STRATEGY_REGISTRY §2 / CURRENT_BEST_STRATEGY §投信環境 / RESEARCH_CONTEXT §3.3 を同期更新。**t_p/bootstrap は未実施**(上記 Pending)。一次根拠: [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md)。
- **2026-06-05 (v4.7)**: **CFD 環境 Active 候補を vz=0.65+l7+F10ε → vz=0.65+l5+F10ε に置換** (ユーザー判断)。理由: l5 は min CAGR -1.30pp と引換に Worst10Y +2.71pp / P10_5Y +4.70pp 大幅改善 / MaxDD -9.23pp 浅化 / Sharpe +0.012 / Trades -18% 低コスト。STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md (v4.6→v4.7)、CURRENT_BEST_STRATEGY.md、CLAUDE.md §2、EVALUATION_STANDARD.md §3.13、STRATEGY_REGISTRY.md §2 を同期更新。
- **2026-06-05 (v4.6)**: §6-5 新設 vz=0.65+F10ε lmax sweep (l5/l5.5/l7) — l5/l5.5 を初実装。l5.5 をバランス候補、l5 を防御候補として §2 Shortlisted 追加。
- **2026-06-05 (v4.5)**: **保守的採用基準 min(IS, OOS) CAGR + Worst10Y + P10_5Y 3 軸導入** + **AH 棄却** + **環境別 Active 候補確定**
  - 論拠: サンプルサイズ非対称 (IS=44y vs OOS=6y)、戦略選択バイアス補正、regime drift リスク、WFE>1.5 で regime luck 警告
  - **CFD 環境 Active 候補**: vz=0.65+l7+F10ε (min CAGR=+20.23%、5 戦略中 1 位)
  - **ETF 環境 Active 候補**: DH-W1 (Asymm+Hysteresis、ETF 制約下で唯一 DH 改善 +4.10pp)
  - **棄却**: vz=0.65+l7+F10ε-AH/AT/HL (v4.4 採用→v4.5 棄却、3 軸全敗 + WFE>1.5 regime luck)
  - STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md を v4.4→v4.5 (§0 主要変更点、§0' 5 行に戻す、§6-4 AH 棄却判定、§7-2 新設、§10 履歴)
  - STRATEGY_REGISTRY.md: §2 に vz065_l7_F10eps015 / DH_W1_AsymmHyst を v4.5 推奨追加、§3 に AH/AT/HL を v4.5 棄却追加、DH_Z2 を Superseded マーク
  - CURRENT_BEST_STRATEGY.md: 冒頭に v4.5 環境別 Active 候補 + 命名規則 + min ルールセクション新設
  - CLAUDE.md §2: v4.5 保守的採用基準 3 軸 + WFE 補助判定 + 環境別 Active + 命名規則を追記
- 2026-06-03 (v4.4): **vz=0.65+l7+F10ε に DH-W1 の非対称機構を移植 (g26)** — 3 候補 (AH/AT/HL) を実装、AH が CAGR_OOS +5.04pp/Trades 半減 で v4.4 採用と判定 (v4.5 で min ルールにより棄却)。「NEW」呼称全廃。
- 2026-06-03 (v4.3): **DH-Z2 → DH-W1 置換** (Asymm+Hysteresis on lev_mod_065, Enter 0.7/Exit 0.3) + g24 5 戦略 canonical 再検証 + DH gap +10.46pp 整合修正。WFA W1 WFE=0.997 完全汎化、Bootstrap median +3.65pp、threshold sweep 24 組合せ全 OOS +12〜16.7% でロバスト。
- 2026-06-03 (v4.2): §0' 表に「累積 CAGR ⓽ OOS/IS」列追加 (IS 値を gap 経由でなく直接読める)。
- 2026-06-03 (v4.1): §4 8戦略総合比較表削除、§5/§6/§6-2 を §0' 5 戦略に絞込み。
- **2026-06-03 (v4)**: **DH 改善 v4 — 配分 × タイミング 2 軸変動 DH-Z シリーズ完了**。v3 (DH-T4) は `lev_mod` で TQQQ position 連続スケールしたため ETF 「保有/非保有」制約に違反し**全面破棄**。仕切り直しで `src/g22a〜g22f` 7 本実装、5 変種 (Z1〜Z5) で配分パターン (DH base / F10 ε tilt / 固定 bull / regime preset) × timing (binary vz / 保守 composite / 常時 IN) を網羅検証。**peak leverage ≤ 3.0x を assert で機械検証**。最良 **DH-Z2 (F10 ε tilt + binary vz_gate)** は IS-OOS gap +10.46→+2.17pp (-8.29pp ✅) / **WFE 0.662→1.058 (完全汎化)** / **OOS CAGR +12.26% (REF +9.56% を +2.70pp 上回る) / OOS 累積 ×2.00 (REF ×1.73)**。Worst10Y/P10_5Y は防御性能低下 (trade-off)。`STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md` を v3→v4、§0'/§5/§6/§6-2 の 4 箇所で DH-T4→DH-Z2 完全置換。STRATEGY_REGISTRY: DH_T4_Improved を §3 Rejected 降格、DH_Z2_AllocTiming を §2 Shortlisted 新規追加。
- 2026-06-03 (v3, 取消): DH-T4 (vz=0.65+lmax=5.5+F10ε) を 4 箇所追加 → **v4 で破棄** (lev_mod による TQQQ position 連続スケール = ETF レバ操作違反)。
- 2026-05-26: **Trades/yr バグ特定（F10+lmax5）** — `p10_f10lmax5_fullmetrics.py` が lev_raw 変化のみカウント（27/yr）で wn/wb 変化を未計上。正しくは lev_raw+wn/wb で 52/yr（G8 WFA per-window 平均 52.09 で確認）。INTEGRATION_DEBATE §2 表を修正済み。CSV は未再生成。
- 2026-05-26: **INTEGRATION_DEBATE §2 テーブルヘッダ標準化** — 手書きヘッダ → `src/_sweep_format.py` の `MD_HEADER_STRAT` + 過学習リスク列 に修正（commit 7c0b091）。P10 5Y▷ 列も追加。根本原因: サブエージェントが `_sweep_format.py` import 未実施。再発防止: CLAUDE.md §NASDAQ sweep ルール に Item 5 追記。
- 2026-05-26: **F10+lmax5 全指標計算** — `src/p10_f10lmax5_fullmetrics.py` 新規作成・実行。CAGR_OOS=+33.56%、MaxDD=-54.22%、Worst10Y★=+16.90%、IS-OOS gap=-3.37pp（4候補中2位）、P10_5Y=+12.8%（4候補中1位）。`f10lmax5_fullmetrics.csv` 生成（commit d56fcdd）。
- 2026-05-26: **G9 WFA for vz065+lmax5 PASS** — CI95_lo=+24.82%（α PASS）、WFE=+1.272（β PASS）。Shortlisted 確定。MaxDD=-51.8% は全候補中最良。[G9_WFA_VZ065_LMAX5_2026-05-26.md](G9_WFA_VZ065_LMAX5_2026-05-26.md)
- 2026-05-26: **G8 WFA for F10+lmax5 PASS** — CI95_lo=+25.57%（α PASS）、WFE=+1.278（β PASS）。Shortlisted 確定。P10_5Y=+12.8% は全候補中最高。[G8_WFA_F10LMAX5_2026-05-26.md](G8_WFA_F10LMAX5_2026-05-26.md)
- 2026-05-26: **G7 WFA for F10 ε=0.015 PASS** — CI95_lo=+27.93%（α PASS・全候補中最高）、WFE=+1.208（β PASS）。採用推奨・ユーザー判断待ち。IS-OOS gap=-4.31pp⚠ 汎化性懸念あり。[G7_WFA_F10_2026-05-26.md](G7_WFA_F10_2026-05-26.md)
- 2026-05-26: **H1 Stress Test P10_5Y▷ 計算（4候補）** — E4=+9.8%、F10 ε=0.015=+10.3%、vz065+lmax5=+11.9%、F10+lmax5=+12.8%。P10_5Y は F10+lmax5 が最高。[INTEGRATION_DEBATE_2026-05-26.md](INTEGRATION_DEBATE_2026-05-26.md)
- 2026-05-24: **比較MD更新 v1.6** — F8-R5・F7v3+E4 列を削除（Trades/yr過多・採用不採用確定）。6戦略統一評価フレームワークへ縮小。`src/gen_strategy_comparison.py` 更新・`STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md` 再生成。
- 2026-05-24: **比較MD更新 v1.5 + E4 Active 復帰** — `src/gen_f8r5_yearly_returns.py` 新規作成・`f8r5_yearly_returns.csv` 生成。`src/gen_strategy_comparison.py` を8戦略に拡張（F8-R5 ✅列追加・E4 ◆復帰・F7v3+E4 ✅降格）。Trades/yr 182-183回(E4比7倍)・OOS偶然性疑い・IS-OOS gap拡大を棄却理由として CURRENT_BEST_STRATEGY.md・STRATEGY_REGISTRY.md 更新。[STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md](STRATEGY_PERFORMANCE_COMPARISON_2026-05-24.md)
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
