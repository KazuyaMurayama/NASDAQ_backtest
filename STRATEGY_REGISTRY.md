# STRATEGY REGISTRY — 戦略台帳（Active / Shortlisted / Rejected / Deferred）

作成日: 2026-05-21
最終更新日: 2026-05-26 (F10 ε=0.015 / vz065+lmax5 / F10+lmax5 WFA PASS → Shortlisted 追加・採用変更検討中)
管理者: 男座員也（Kazuya Oza）

---

## 目的

本ファイルは **NASDAQ_backtest リポジトリで検証された全戦略の最終ステータスを一元管理する台帳** である。
過去の研究履歴を「採用 / 候補 / 廃止 / 保留」の4ステータスに整理し、

- 過去に検討した戦略を二度実装し直さない（重複研究の防止）
- 廃止理由・代替戦略を即座に参照可能にする
- テーマ別逆引きで「VolSpike系は試したか？」のような質問に1秒で答えられる
- `CURRENT_BEST_STRATEGY.md` と本ファイルの2点さえ見れば現状を把握できる

を達成する。

> **「ベスト戦略は？」という問いには本ファイルではなく [`CURRENT_BEST_STRATEGY.md`](CURRENT_BEST_STRATEGY.md) を一次根拠とすること。本ファイルは"過去全戦略の台帳"であり、現行ベストの正典ではない。**

---

## 凡例 / 使い方

### Status の定義

| Status | 意味 | 取り扱い |
|---|---|---|
| **Active** | 現行ベスト（採用中、本番運用候補） | `CURRENT_BEST_STRATEGY.md` と完全同期 |
| **Shortlisted** | 最終候補。Active には届かなかったが廃止もしていない | 再評価・派生検証の対象 |
| **Rejected** | 廃止・棄却。明確な理由あり | 同じ実験を再度行わない。理由を必ず引用 |
| **Deferred** | 再評価保留。データ追加・前提変更で再検討する可能性あり | 解放条件を明記。条件未充足の間は手を付けない |

### Cost Scenario の定義

| Scenario | 内容 |
|---|---|
| **D** | 標準。SOFR financing drag + TER 反映 + swap spread 補正済み（`src/corrected_strategy_backtest.py` Scenario D, 2026-05-12 以降の標準） |
| **A** | 旧コストモデル。SOFR 補正なし。2026-05-12 以前の値は基本このシナリオ |
| **N/A** | コストモデル不明 / 旧実験ログのみ。再現性なし |
| **[非標準コスト]** | P-series（P01〜P05系）。`data/timing_signals_raw.csv` の HY/CPI シグナルを用いた独自コストモデル。他系統と直接比較不可（§17 参照） |

### Evidence 列について

- 一次根拠ファイル（Markdown レポート、CSV、Python 実装）を **`[ファイル名](相対パス)`** 形式で記載
- 複数ある場合はカンマ区切り
- ファイルが存在しない（実験ログのみ）場合は `N/A` または該当 CSV 名のみ

### Verified Date の意味

- 当該指標が「最後に正規バックテストで確認された日付」
- 廃止戦略の場合は「廃止判定が確定した日付」
- 旧実験で日付不明な場合は `N/A`

### 列定義（全テーブル共通）

```
| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
```

> 具体指標が不明な実験は `N/A` で埋め、ファイル名のみ Evidence 列に記載する。

---

## §1 Active — 採用中（現行ベスト）

> 1行のみ。`CURRENT_BEST_STRATEGY.md` と必ず同期。差異があれば本ファイルではなく `CURRENT_BEST_STRATEGY.md` 側を一次根拠とする。

| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **S2_VZGated+LT2_N750_E4RegimeKLT_v2** | S2_VZGated + LT2-N750 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, vz_thr=0.7) — tiltなし | CFDLeverage + LongCycleSignal + VolRegime | 2026-05-24 | D | **+33.53%** | **0.891** | −60.01% | **Active (WFA PASS 確定)** | E4 sweep 64 config 中 Worst10Y★ +18.67% / MaxDD −60.01% を同時最良化。IS-OOS gap −1.81pp は本プロジェクト最高水準の汎化性。Trades/yr 約27回（月2.3回）でコスト優位。tilt系 (F7v3/F8) は WFA PASS するも Trades/yr 182回超・OOS偶然性疑いにより採用せず本戦略に復帰。**G3 WFA PASS (2026-05-24): CI95_lo=+26.51%, WFE=+1.131** → 正式 Active 確定（復帰）。 | [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md), [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md), [g3_wfa_e4_summary.csv](g3_wfa_e4_summary.csv), [src/e4_regime_klt.py](src/e4_regime_klt.py), [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |

---

## §2 Shortlisted — 最終候補（未採用）

> 11戦略統合比較表（[STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md)）に掲載され、最終候補に残ったが Active に届かなかった戦略群。
> Active が崩壊した際の次善候補。新パラメータ・新シグナルでの派生検証の出発点としても利用。

| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **S2_VZGated+LT2_N750_E4_F10eps015** | S2_VZGated + LT2-N750 + E4 + F10 ε=0.015 (F8 R5_CALM_BOOST + ε-deadband 0.015, l_max=7.0) | CFDLeverage + LongCycleSignal + VolRegime + RegimeCapBullTilt + EpsilonDeadband | 2026-05-26 | D | +36.8% | 0.934 | −63.1% | Shortlisted (WFA G7 PASS, **採用推奨・ユーザー判断待ち**) | G7 WFA PASS（CI95_lo=+27.93%・全候補中最高, WFE=+1.208）。Sharpe_OOS最高+0.934・CAGR_OOS最高+36.8%。IS-OOS gap=-4.31pp⚠ は4候補中最大（汎化性懸念）。Trades/yr=52（E4比約2倍）。採用変更はユーザー判断待ち。 | [INTEGRATION_DEBATE_2026-05-26.md](INTEGRATION_DEBATE_2026-05-26.md), [G7_WFA_F10_2026-05-26.md](G7_WFA_F10_2026-05-26.md), [src/g7_wfa_f10.py](src/g7_wfa_f10.py) |
| **S2_VZGated+LT2_N750_vz065_lmax5** | S2_VZGated + LT2-N750 + E4 (vz_thr=0.65, l_max=5.0) | CFDLeverage + LongCycleSignal + VolRegime | 2026-05-26 | D | +33.5% | 0.947 | −51.8% | Shortlisted (WFA G9 PASS, Sharpe最高・MaxDD最良) | G9 WFA PASS（CI95_lo=+24.82%, WFE=+1.272）。Sharpe_OOS=+0.947 は全候補中最高。MaxDD=-51.8% は4候補中最良（E4比+8.2pp 改善）。IS-OOS gap=-3.91pp（E4比-2.1pp 悪化）。Trades/yr=27 でE4同等コスト。 | [INTEGRATION_DEBATE_2026-05-26.md](INTEGRATION_DEBATE_2026-05-26.md), [G9_WFA_VZ065_LMAX5_2026-05-26.md](G9_WFA_VZ065_LMAX5_2026-05-26.md) |
| **S2_VZGated+LT2_N750_F10lmax5** | S2_VZGated + LT2-N750 + F10 ε=0.015 (l_max=5.0) | CFDLeverage + LongCycleSignal + VolRegime + RegimeCapBullTilt + EpsilonDeadband | 2026-05-26 | D | +33.6% | 0.943 | −54.2% | Shortlisted (WFA G8 PASS, P10_5Y最高) | G8 WFA PASS（CI95_lo=+25.57%, WFE=+1.278）。P10_5Y=+12.8% は4候補中最高。IS-OOS gap=-3.37pp は4候補中2位（E4に次いで良好）。Worst10Y★=+16.9%（E4比-1.8pp 劣後）。Trades/yr=52（lev_raw+wn/wb 正計上）。 | [INTEGRATION_DEBATE_2026-05-26.md](INTEGRATION_DEBATE_2026-05-26.md), [G8_WFA_F10LMAX5_2026-05-26.md](G8_WFA_F10LMAX5_2026-05-26.md), [f10lmax5_fullmetrics.csv](f10lmax5_fullmetrics.csv), [src/g8_wfa_lmax5.py](src/g8_wfa_lmax5.py) |
| **S2_VZGated+LT2_N750_E4_F8R5** | S2_VZGated + LT2-N750 + E4 + F8 R5_CALM_BOOST (tilt=10, cap=calm0.15/bullVZ0.10/bearVZ0.05) | CFDLeverage + LongCycleSignal + VolRegime + RegimeCapBullTilt | 2026-05-24 | D | +36.83% | 0.934 | −63.07% | Shortlisted | G5 WFA PASS（CI95_lo=+27.92%, WFE=+1.208）。Sharpe/CAGR/WFAはE4より優位だが Trades/yr 182回（E4比7倍）のコスト負担と OOS 偶然性疑いにより採用見送り。Active に戻す条件: Trades/yr 削減策実装 or コスト込み実証（実取引ログ）。 | [F8_REGIME_TILT_2026-05-24.md](F8_REGIME_TILT_2026-05-24.md), [G5_WFA_F8R5_2026-05-24.md](G5_WFA_F8R5_2026-05-24.md), [g5_wfa_f8r5_summary.csv](g5_wfa_f8r5_summary.csv) |
| **S2_VZGated+LT2_N750_E4_F7v3** | S2_VZGated + LT2-N750 + E4 + F7v3 Bull-Tilt (A:tilt=2.0, flat cap=0.10) | CFDLeverage + LongCycleSignal + VolRegime + BullTilt | 2026-05-24 | D | +36.30% | 0.926 | −61.96% | Shortlisted | G4 WFA PASS（CI95_lo=+27.15%, WFE=+1.203）。Trades/yr 183回（E4比7倍）の同一理由で採用見送り。F8-R5よりMaxDD良好（−61.96%）。Trades削減策実装後の再評価候補。 | [F7V3_BULL_TILT_2026-05-24.md](F7V3_BULL_TILT_2026-05-24.md), [G4_WFA_F7V3_2026-05-24.md](G4_WFA_F7V3_2026-05-24.md) |
| **S2_VZGated+LT2_N750_E4RegimeKLT** | S2_VZGated + LT2-N750 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, vz_thr=0.7) / tiltなし | CFDLeverage + LongCycleSignal + VolRegime | 2026-05-24 | D | +33.53% | 0.891 | −60.01% | Shortlisted | 旧 Active（〜2026-05-24）。F7v3 Bull-Tilt が Sharpe +0.926 で上回り降格。WFA G3 PASS済み（CI95_lo=+26.51%, WFE=+1.131）。MaxDD −60.01% は本プロジェクト最良水準。最保守的な fallback 候補。 | [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md), [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md) |
| **S2_VZGated+LT2_N750** | S2_VZGated + LT2-N750-k0.5-modeB（固定 k_lt=0.5 / 旧◆ Active） | CFDLeverage + LongCycleSignal | 2026-05-24 | D | +31.16% | 0.858 | −59.45% | Shortlisted | 旧 Active（〜2026-05-24）。E4 Regime k_lt に CAGR_OOS −2.37pp / Sharpe_OOS −0.033 / Worst10Y★ −0.57pp / IS-OOS gap +1.99pp で劣後し降格。WFA 完了済み（CI95_lo=+25.7%, WFE=1.145 PASS α∩β）のため代替候補として強い残置価値あり。E4 の WFA 未通過時の fallback 第二候補。 | [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md), [b1_s2_lt2_results.csv](b1_s2_lt2_results.csv), [g1_wfa_summary.csv](g1_wfa_summary.csv) |
| **S2_VZGated** | Vol-Zone ゲート型 CFD レバレッジ（tv=0.8, k_vz=0.3, gate_min=0.5） | CFDLeverage | 2026-05-21 | D | +27.57% | 0.769 | −62.4% | Shortlisted | 旧Active。S2+LT2 に全7指標（CAGR_OOS/Sharpe_OOS/IS-OOS gap/Worst10Y★/MaxDD/P10_5Y/Worst5Y）で敗北し降格。LT2 を外せばこの構成。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md), [src/cfd_leverage_backtest.py](src/cfd_leverage_backtest.py), [src/dynamic_leverage_strategies.py](src/dynamic_leverage_strategies.py) |
| **P2_VolTarget** | P2 best（vol-target, tv=0.8） | CFDLeverage / VolTarget | 2026-05-21 | D | +27.13% | 0.757 | −60.5% | Shortlisted | S2_VZGated とほぼ同水準。Worst10Y★ +19.09% は S2 より僅かに優位だが Sharpe_OOS 0.757 < 0.769 で総合敗北。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **S4_RelVol** | RelVol ベース CFD レバレッジ（l_base=7, k_rel=2.0） | CFDLeverage | 2026-05-21 | D | +26.19% | 0.697 | −66.1% | Shortlisted | CAGR_IS +40.98% と高いが OOS で +26.19% へ大幅低下（IS-OOS gap 約 14.8pp）。過剰適合疑い。MaxDD −66.1% も大きい。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md), [src/dynamic_leverage_strategies.py](src/dynamic_leverage_strategies.py) |
| **CFD_7x_Fixed** | DH Dyn + 7x 固定レバレッジ CFD | CFDLeverage | 2026-05-21 | D | +24.44% | 0.670 | −65.0% | Shortlisted | CAGR_IS +43.35% / OOS +24.44%（IS-OOS gap ~19pp）と過剰適合大。固定 7x レバはリスク管理面で実運用不可。Worst10Y★ +25.37% のみ最高。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **DH_Dyn_2x3x_A** | DH Dyn 2x3x [A] Approach A（TQQQ, threshold=0.15）Scenario D | DH_Dyn / ETF配分 | 2026-05-21 | D | +14.88% | 0.646 | −45.08% | Shortlisted | 旧Active（〜2026-05-21）。CFD系の登場で CAGR_OOS で大幅劣後。本番運用候補としては最も保守的かつ NISA 適合可能性あり（CFD不要）。MaxDD −45% は CFD 系より良好。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md), [src/corrected_strategy_backtest.py](src/corrected_strategy_backtest.py), [corrected_strategy_results.csv](corrected_strategy_results.csv) |
| **DH_Z2_AllocTiming** | DH Dyn 2x3x [A] + F10 ε tilt 配分 + binary HOLD/OUT (vz_gate `lev_mod_065 ≥ 0.5`)。商品は TQQQ+TMF+2036 維持、保有比率と保有タイミングのみ NEW CANDIDATE/D5 から移植。レバ操作なし、peak lev 2.85x≤3.0x assert で機械検証 | DH_Dyn + VolRegime + EpsilonDeadband + BinaryRegimeGate | 2026-06-03 | D (moderate 0.10%) | +12.26% | 0.837 | −38.92% | Shortlisted (要追加検証 / Active 昇格判断ユーザー待ち) | **DH 過学習 (gap +10.46pp→+2.17pp, -8.29pp) と WFE (0.662→1.058) を完全改善**。WFA 50窓 PASS (CI95_lo=+12.03%, **WFE=1.058 (完全汎化)**, p=0.0000)。Bootstrap (Z2 vs REF) CI95=[-9.78, +16.62pp] / median +3.80pp positive / P(diff>0)=71%。Permutation: 実 θ=0.5 NULL 中央 (52%) で θ ロバスト (過適化なし)。OOS 6 年 diff sum +7.56pp / 2022 +21.56pp (binary OUT が bear 救済) / 2025 -16.6pp (missed rally)。**OOS 累積 ×2.00 (REF ×1.73)**。Worst10Y +6.36% は REF +12.57% に劣後、P10_5Y +5.33% も劣後 → 防御性能と OOS 拡大の trade-off。Min -17.21% は REF -24.12% より良い (新最良)。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §0'/§5/§6/§6-2 §9-v4, [src/g22a_dh_alloc_timing_variants.py](src/g22a_dh_alloc_timing_variants.py), [g22b_dh_alloc_timing_9metrics.csv](g22b_dh_alloc_timing_9metrics.csv), [g22c_dh_alloc_timing_wfa.csv](g22c_dh_alloc_timing_wfa.csv), [g22d_dh_alloc_timing_bootstrap_results.csv](g22d_dh_alloc_timing_bootstrap_results.csv), [g22e_dh_alloc_timing_permutation_summary.csv](g22e_dh_alloc_timing_permutation_summary.csv), [g22f_dh_z2_yearly_returns_aftertax.csv](g22f_dh_z2_yearly_returns_aftertax.csv) |
| **BH_1x** | Buy & Hold NASDAQ 1x（素のNASDAQ） | Benchmark | 2026-05-21 | N/A | +10.11% | 0.540 | −77.9% | Shortlisted | ベンチマーク。常に比較対象として残置。MaxDD −77.9% / Worst10Y★ −5.67% で長期保有リスク大。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **P02_DynCPI** | P02_Dyn × CPI [mult]（CPI ゲート） | TimingGate / DH_Dyn派生 | 2026-05-21 | [非標準コスト] | +19.43% | 0.833‡ | −46.37% | Shortlisted | OOS Sharpe 0.833 と Active 候補級だが **[非標準コスト]**（P-series 独自コストモデル, §17）。直接比較不可。再評価には標準コスト下での再計算が必要。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **P05_HYCPI** | P05_HY × CPI [mult]（HY + CPI ゲート） | TimingGate / DH_Dyn派生 | 2026-05-21 | [非標準コスト] | +15.65% | 0.667‡ | −44.98% | Shortlisted | Worst5Y +6.04% で Worst5Y 観点では最強。ただし **[非標準コスト]** につき他系統と直接比較不可。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **P01_DynHY** | P01_Dyn × HY [mult]（HY スプレッドゲート） | TimingGate / DH_Dyn派生 | 2026-05-21 | [非標準コスト] | +19.92% | 0.829‡ | −42.85% | Shortlisted | DSR 観点で候補級だが **[非標準コスト]**（§17）。MaxDD −42.85% は本表で最小。標準コスト下での再評価で残存可否を判定する。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **DH_Dyn_2x3x_A_LT2** | DH Dyn 2x3x [A+LT2]（LT2-N750-k0.5-modeB, TQQQ参照） | DH_Dyn + LongCycleSignal | 2026-05-21 | D | +18.87% | 0.777 | −44.76% | Shortlisted | Active (S2+LT2) の CFD 抜きバージョン。CFD 不可環境（NISA, 法人税効率優先など）の代替候補。Sharpe_OOS 0.777 は DH Dyn 単体 0.646 から大幅改善。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md), [src/b1_s2_lt2.py](src/b1_s2_lt2.py), [src/long_cycle_signal.py](src/long_cycle_signal.py) |
| **S2_VZGated+LT2_N1500** | S2_VZGated + LT2-N1500-k0.5-modeB（超長期逆張りフィルタ） | CFDLeverage + LongCycleSignal | 2026-05-22 | D | +30.84% | 0.885 | −63.37% | Shortlisted | Sharpe_OOS +0.885（本プロジェクト最高値）/ IS-OOS gap −0.05pp（OOS が IS 超え）は最強の汎化性。CAGR_OOS −0.32pp / MaxDD −3.92pp / Worst10Y★ −1.50pp でリスク指標が N=750 に劣後し、コミット 8503200 で Active から N=750 に差し戻し。代替シナリオとして Shortlisted 保持。 | [b6_s2_lt2_N_sweep_results.csv](b6_s2_lt2_N_sweep_results.csv), [B6_S2_LT2_N_SWEEP_2026-05-22.md](B6_S2_LT2_N_SWEEP_2026-05-22.md), [src/b1_s2_lt2.py](src/b1_s2_lt2.py) |
| **S2_VZGated+LT4_N750_k0.7** | S2_VZGated + LT4-N=750, k_lt=0.7（LT4 系最良） | CFDLeverage + LongCycleSignal | 2026-05-22 | D | +30.50% | 0.850 | N/A | Shortlisted | B3 LT4 N×k_lt スイープ 9/9 が S2 単体（+0.770）超え、最良が CAGR_OOS +30.50% / Sharpe +0.850 / IS-OOS gap +0.60pp と Active 級。LT2-N1500 比で Sharpe −0.035, gap +0.65pp と総合僅差敗北。 | [b3_s2_lt4_sweep_results.csv](b3_s2_lt4_sweep_results.csv) |
| **S2_VZGated+LT6_N500_k0.7** | S2_VZGated + LT6-N=500, k_lt=0.7（LT6 系最良） | CFDLeverage + LongCycleSignal | 2026-05-22 | D | +27.35% | 0.820 | −54.38% | Shortlisted | B4 LT6 N×k_lt スイープ 9/9 が S2 単体超え。最良値は MaxDD −54.38% / Worst10Y★ +17.06% と健闘するが IS-OOS gap +5.73pp で過剰適合傾向。N=750/k=0.7 変種（Sharpe +0.801, gap +4.13pp）のほうがロバスト性で勝る。 | [b4_s2_lt6_sweep_results.csv](b4_s2_lt6_sweep_results.csv) |
| **S2_VZGated+LT7_k0.5** | S2_VZGated + LT7-k_lt=0.5（dual MA cross, N_short=750, N_long=1250 固定） | CFDLeverage + LongCycleSignal | 2026-05-22 | D | +29.42% | 0.814 | −60.39% | Shortlisted | B5 LT7 k_lt スイープ 3/3 が S2 単体超え。dual MA cross 構造を導入したが Sharpe では LT2/LT4 に劣後、N=1500 比で +0.071 差。複雑度に見合う優位性なし、ロバスト性検証ためのスイープ規模（3 config）も小さく要注意。 | [b5_s2_lt7_sweep_results.csv](b5_s2_lt7_sweep_results.csv) |
| **Hybrid_S2LT2_P05_70_30** | S2+LT2-N1500（70%）+ P05_HY×CPI（30%）固定ウェイトブレンド | Hybrid / Multi-Strategy | 2026-05-22 | [非標準コスト] | +30.51%‡ | 0.881‡ | −62.1% | Shortlisted | E2 検証（3基準）で 2/3 改善（MaxDD +1.3pp, IS-OOS gap 維持）。Sharpe_OOS 0.881 < 0.885 で基準(i)不合格。§1.3 により Active 昇格不可（Shortlisted 上限）。S2+LT2-N1500 単独継続を推奨。 | [E2_HYBRID_70_30_2026-05-22.md](E2_HYBRID_70_30_2026-05-22.md), [e2_hybrid_70_30_results.csv](e2_hybrid_70_30_results.csv), [src/e2_hybrid_70_30.py](src/e2_hybrid_70_30.py) |

> ‡ FULL期間 Sharpe は未計算。P4系 (P01/P02/P05) の OOS 期間 (2021/5〜2026/3) Sharpe を記載。Hybrid_S2LT2_P05_70_30 の ‡ は §1.3 非標準コスト継承による参考値扱い。

---

## §3 Rejected — 廃止・棄却ログ

> 「ベスト」「採用候補」と提示してはいけない戦略群。明確な廃止理由とともに記録。
> 同じ実験を再度行わないための備忘録としての性質を持つ。

### §3.1 旧ベスト戦略（CURRENT_BEST_STRATEGY.md ブラックリスト由来）

| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **DH_Dyn_2x3x_A_ScA_v1** | DH Dyn 2x3x [A] Scenario A 旧推奨（CAGR +30.81%） | DH_Dyn / ETF配分 | 2026-05-12 | A | N/A | N/A | N/A | Rejected | Scenario A はコスト過少推計（SOFR financing drag 未反映 等）。Scenario D 補正で CAGR 30.81% → 22.50%, Sharpe 1.299 → 0.993, MaxDD −31.40% → −45.08% に下方修正。Scenario A 単体での値は採用しない。 | [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md), [DELAY_PRODUCT_COMPARISON_REPORT_2026-05-12.md](DELAY_PRODUCT_COMPARISON_REPORT_2026-05-12.md) |
| **DH_Dyn_2x3x_A_ScD_v1** | DH Dyn 2x3x [A] Scenario D 旧推奨（CAGR 22.50% / Sharpe 0.993） | DH_Dyn / ETF配分 | 2026-05-21 | D | +14.88% | 0.993 (FULL) | −45.08% | Rejected | 2026-05-12〜2026-05-21 の暫定ベスト。S2_VZGated が CAGR_OOS +27.57% / Sharpe_OOS 0.769 で上回ることを確認し降格。なお派生形（DH_Dyn_2x3x_A）として Shortlisted には残存（§2）。 | [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md), [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **DH_Dyn_2x3x_A_th020** | DH Dyn 2x3x [A] threshold=0.20 | DH_Dyn / ETF配分 | 2026-04-21 | A | N/A | N/A | N/A | Rejected | 閾値 0.15 が全指標で優位と確認。閾値スイープで決着済み。 | [THRESHOLD_SWEEP_REPORT_2026-04-20.md](THRESHOLD_SWEEP_REPORT_2026-04-20.md), [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md), [threshold_sweep_results.csv](threshold_sweep_results.csv) |
| **DH_T4_Improved** ❌ | DH-T4 (vz=0.65+lmax=5.5+F10ε) — v3 で push 済、v4 で破棄 | DH_Dyn + VolRegime + RegimeCapBullTilt + EpsilonDeadband + lev_mod scaling | 2026-06-03 | D (moderate) | +9.44% | 0.671 | −37.63% | **Rejected (2026-06-03)** | **却下理由**: `lev_mod_065` を `lev_raw × 3` に乗算して TQQQ position を 0〜100% に連続スケール = **ETF (TQQQ/TMF/2036) の物理的「保有/非保有」制約に違反**。「部分 TQQQ 保有 (実効レバを 0〜3x に内部 scaling)」は CFD/SPOT 操作であり ETF 商品で禁止。`L_s2_lmax5p5` cap (CFD-style 動的レバ上限) も ETF では概念違反。代替として DH-Z2 (binary HOLD/OUT + F10 配分) を §2 Shortlisted に採用。一次根拠 (g21*) は履歴保持のため push 残置するが新規参照不可。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §0 v4 主要変更点 A, §9-v3 (破棄マーク), [src/g21a_dh_improved_variants.py](src/g21a_dh_improved_variants.py), [g21b_dh_improved_9metrics.csv](g21b_dh_improved_9metrics.csv) |

### §3.2 アンサンブル系実験（Ens2 / Voting）

| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ens2_AsymSlope** | Ens2(Asym+Slope) max_lev=1.0 | EnsembleVoting | 2026-04-21 | A | +10.52% | 0.479 | −48.99% | Rejected | 2026-04-21 に `DH Dyn 2x3x [A] 閾値0.15` に置換され廃止。Scenario A 値であり Scenario D 補正後はさらに悪化見込み。 | [strategy_comparison_results.csv](strategy_comparison_results.csv), [ens2_comparison_results.csv](ens2_comparison_results.csv), [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |
| **Ens2_AsymSlope_lev30** | Ens2(Asym+Slope) max_lev=3.0（旧 CAGR 54.94% / Sharpe 1.049） | EnsembleVoting / 高レバ | N/A | A（未補正） | N/A | N/A | −77.7% | Rejected | 旧未補正 CAGR 54.94% / Worst5Y −14.4% は Scenario A 未補正の過剰適合値。IS-OOS gap 大確実で Scenario D 補正後は実用水準に達さない。MaxDD −77% も実運用不可水準。 | [ens2_comparison_results.csv](ens2_comparison_results.csv) |
| **Ens2_SlopeTrendTV_lev30** | Ens2(Slope+TrendTV) max_lev=3.0（旧 CAGR 59.20%） | EnsembleVoting / 高レバ | N/A | A（未補正） | N/A | N/A | −78.5% | Rejected | 同上。Scenario A 未補正の値であり、Scenario D 補正で実用水準に達しないことが類似戦略から推定可能。 | [ens2_comparison_results.csv](ens2_comparison_results.csv) |
| **Ens2_SlopeTrendTV** | Ens2(Slope+TrendTV) max_lev=1.0 | EnsembleVoting | 2026-04-21 | A | N/A | N/A | N/A | Rejected | 2026-04-21 に `DH Dyn 2x3x [A] 閾値0.15` に置換され廃止。 | [ens2_comparison_results.csv](ens2_comparison_results.csv), [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |
| **MajorityVote_P2** | Majority Vote (P2 シグナル統合) | EnsembleVoting | N/A | N/A | N/A | N/A | N/A | Rejected | 実験ログのみ。最終結論に組み込まれず採用されなかった。 | [majority_vote_p2_results.csv](majority_vote_p2_results.csv), [majority_vote_signals.csv](majority_vote_signals.csv), [vote_ratio_continuous_results.csv](vote_ratio_continuous_results.csv) |

### §3.3 CFD動的レバレッジ派生系（S2/S4 等の落選候補）

| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **CFD_LMAX_Sweep_Rejects** | l_max スイープで残らなかったパラメータ群 | CFDLeverage / Sweep | 2026-05-21 | D | N/A | N/A | N/A | Rejected | l_max=7.0 が最適と確認。それ以外の l_max（5.0, 6.0, 8.0 等）は同 Sharpe / 高 MaxDD で劣後。 | [A6_LMAX_SWEEP_2026-05-21.md](A6_LMAX_SWEEP_2026-05-21.md), [a6_lmax_sweep_results.csv](a6_lmax_sweep_results.csv) |
| **CFD_NVOL_Sweep_Rejects** | n_vol スイープで残らなかったパラメータ群 | CFDLeverage / Sweep | 2026-05-21 | D | N/A | N/A | N/A | Rejected | n=20 が最適と確認。n=10, 30, 60 は Sharpe で劣後。 | [A1_NVOL_SWEEP_2026-05-21.md](A1_NVOL_SWEEP_2026-05-21.md), [a1_nvol_sweep_results.csv](a1_nvol_sweep_results.csv) |
| **B2_KLT_Sweep_Rejects** | LT2 k スイープで残らなかったパラメータ群 | LongCycleSignal / Sweep | 2026-05-21 | D | N/A | N/A | N/A | Rejected | k=0.5（modeB）が最適と確認。k=0.25, 0.75, 1.0 は IS-OOS gap または Sharpe で劣後。 | [B2_KLT_SWEEP_2026-05-21.md](B2_KLT_SWEEP_2026-05-21.md), [b2_klt_sweep_results.csv](b2_klt_sweep_results.csv) |
| **C1_HY_Gate_Rejected** | HY ゲート単体組み込み | TimingGate / 信号 | 2026-05-21 | D | N/A | N/A | N/A | Rejected | HY ゲートは S2+LT2 に上乗せしても改善せず。標準コストでは効果無し。 | [C1_HY_GATE_2026-05-21.md](C1_HY_GATE_2026-05-21.md), [c1_hy_gate_results.csv](c1_hy_gate_results.csv) |
| **F1_Alloc_Sweep_Rejects** | 配分比率（wn/wg/wb）スイープ落選 | 配分最適化 | 2026-05-21 | D | N/A | N/A | N/A | Rejected | 既存 DH Dyn [A] 配分が最適。他配分は CAGR or Sharpe で劣後。 | [F1_ALLOC_SWEEP_2026-05-21.md](F1_ALLOC_SWEEP_2026-05-21.md), [f1_alloc_sweep_results.csv](f1_alloc_sweep_results.csv) |
| **E1_Ensemble_S2_Rejects** | S2 系アンサンブル（S2+S4 等） | EnsembleVoting / CFD | 2026-05-21 | D | N/A | N/A | N/A | Rejected | S2+LT2 単体に対する優位性なし。アンサンブル化のメリット観測されず。 | [E1_ENSEMBLE_2026-05-21.md](E1_ENSEMBLE_2026-05-21.md), [e1_ensemble_results.csv](e1_ensemble_results.csv) |
| **D1_OOS_Boundary_Variants** | OOS 境界変動テストで脱落した変種 | 検証 / Robustness | 2026-05-21 | D | N/A | N/A | N/A | Rejected | OOS 境界 ±1 年シフトでも S2+LT2 がロバスト最高。他候補は境界変動に対して脆弱。 | [D1_OOS_BOUNDARY_2026-05-21.md](D1_OOS_BOUNDARY_2026-05-21.md), [d1_oos_boundary_results.csv](d1_oos_boundary_results.csv) |
| **Discrete_Leverage_Rejects** | 離散レバレッジ（2x/3x/5x ステップ）変種 | CFDLeverage / 離散化 | N/A | A | N/A | N/A | N/A | Rejected | 連続レバ（step=0.5）が同等以上の性能。離散化のメリット観測されず。 | [discrete_leverage_results.csv](discrete_leverage_results.csv) |
| **Marugoto_Leverage** | "まるごとレバレッジ"（単純倍率） | CFDLeverage / 単純 | N/A | A | N/A | N/A | N/A | Rejected | 単純レバ倍はベンチマーク以下。DH Dyn シグナル統合が必須。 | [marugoto_leverage_results.csv](marugoto_leverage_results.csv) |
| **Lev2x3x_Baseline** | 単純 2x/3x ベースライン | DH_Dyn / 単純 | N/A | A | N/A | N/A | N/A | Rejected | DH Dyn シグナルなしの単純 2x/3x ローテーション。CAGR / Sharpe ともに DH Dyn [A] に劣後。 | [lev2x3x_results.csv](lev2x3x_results.csv) |
| **P4_Composite** | 三因子乗算型レバレッジ（SOFR×ボラ×モメンタム） | CFDLeverage / MultiFactorScore | 2026-05-21 | D | +13.69% (best) | +0.647 (best) | −47.74% (best) | Rejected | 24 config 全て S2_VZGated ベースライン（Sharpe +0.770）を下回る。三因子乗算の論理 AND 効果が過剰デレバをもたらしリターン低下。IS-OOS gap +8〜15 pp と大きく過剰適合傾向。dynamic_leverage_strategies.py で「本命」と記載されていたが実証的に棄却。 | [P4_COMPOSITE_SWEEP_2026-05-21.md](P4_COMPOSITE_SWEEP_2026-05-21.md), [p4_composite_sweep_results.csv](p4_composite_sweep_results.csv) |
| **S1_Conviction** | A2 確信度スコア直接レバ変換 | CFDLeverage / A2Conviction | 2026-05-21 | D | +22.43% (best) | +0.645 (best) | −64.52% (best) | Rejected | 16 config 全て S2_VZGated ベースラインを下回る。IS-OOS gap +21 pp と致命的な過剰適合。target_vol パラメータは NASDAQ 通常ボラレンジ（σ≈13.6%）で全値飽和（dead parameter）。A2 raw スコア自体が IS 最適化バイアスを持つため直接レバ変換は不適。 | [S1_CONVICTION_SWEEP_2026-05-21.md](S1_CONVICTION_SWEEP_2026-05-21.md), [s1_conviction_sweep_results.csv](s1_conviction_sweep_results.csv) |
| **S2_TV_Sweep_Rejects** | S2_VZGated target_vol スイープ落選パラメータ | CFDLeverage / Sweep | 2026-05-21 | D | N/A | N/A | N/A | Rejected | tv=0.80 が Sharpe_OOS 最高（+0.770）と確認。tv=1.00 は CAGR_OOS +28.42% と高いが IS-OOS gap +8.84 pp で過剰適合。tv<0.40 は vol-targeting 機能するが CAGR_OOS が半減。 | [A2_TV_SWEEP_2026-05-21.md](A2_TV_SWEEP_2026-05-21.md), [a2_tv_sweep_results.csv](a2_tv_sweep_results.csv) |
| **S2_KVZ_Sweep_Rejects** | S2_VZGated k_vz スイープ（VIX ゲート感度） | CFDLeverage / Sweep | 2026-05-21 | D | N/A | N/A | N/A | Rejected | k_vz=0.10〜0.70 で Sharpe_OOS は +0.769〜+0.775 と極めてフラット。VIX ゲートの感度係数は dead parameter に近い。現行 k_vz=0.30 は安定中心値として妥当。 | [A3_KVZ_SWEEP_2026-05-21.md](A3_KVZ_SWEEP_2026-05-21.md), [a3_kvz_sweep_results.csv](a3_kvz_sweep_results.csv) |
| **S2_GATEMIN_Sweep_Result** | S2_VZGated gate_min スイープ（VIX 下限） | CFDLeverage / Sweep | 2026-05-21 | D | N/A | N/A | N/A | Rejected | gate_min=0.20/0.35 が Sharpe_OOS +0.775（現行 0.50 の +0.770 を僅かに上回る）。VIX ゲートなし（gate_min=1.00）は Sharpe +0.759 と明確に劣後し VIX ゲートの有効性を確認。差分 +0.005 は微小のため現行設定を維持。S2+LT2 への影響は別途要確認。 | [A4_GATEMIN_SWEEP_2026-05-21.md](A4_GATEMIN_SWEEP_2026-05-21.md), [a4_gatemin_sweep_results.csv](a4_gatemin_sweep_results.csv) |
| **P1_SOFR_Adaptive** | SOFR 水準に応じた l_max 動的調整（18 config） | CFDLeverage / SOFR適応 | 2026-05-22 | D | +19.48% (best) | +0.691 (best) | N/A | Rejected | 18/18 が S2 単体ベースライン（Sharpe +0.770）を下回る。最良 (sofr_high=0.10, l_max=6) でも IS-OOS gap +9.93pp。SOFR 適応化はゼロ金利期で過剰デレバ、リターン低下を招く。 | [p1_sofr_sweep_results.csv](p1_sofr_sweep_results.csv) |
| **P3_Momentum_Lev** | モメンタム派生レバレッジ（m × k スイープ、16 config） | CFDLeverage / Momentum | 2026-05-22 | D | +18.87% (best) | +0.649 (best) | N/A | Rejected | 16/16 が S2 単体を下回る。最良 (m=60, k=0.5) で IS-OOS gap +14.66pp と過剰適合大。DH Dyn シグナルが既にモメンタム要素を内包しており、モメンタム重畳は冗長。 | [p3_momentum_sweep_results.csv](p3_momentum_sweep_results.csv) |
| **P5_Kelly_Sizing** | ローリング μ/σ から Kelly 推定レバ（safety × mu_w × sig_w、12 config） | CFDLeverage / Kelly | 2026-05-22 | D | +21.08% (best) | +0.642 (best) | N/A | Rejected | 12/12 が S2 単体を下回る。最良 (safety=1.0, mu_w=252, sig_w=30) で IS-OOS gap +5.40pp。Kelly は分布安定仮定が NASDAQ 株式 OOS では破綻、CFD 動的レバとして不適。 | [p5_kelly_sweep_results.csv](p5_kelly_sweep_results.csv) |
| **S3_Decomposed_A2** | A2 構成要素を直接 L_t に rewire したレバレッジ（5 config） | CFDLeverage / A2Decomposition | 2026-05-22 | D | +10.13% (best) | +0.443 (best) | N/A | Rejected (hard) | 全 config が S2 単体を大幅下回る。最良で IS-OOS gap **+22.39pp** と致命的過剰適合。A2 raw 構成要素を直接レバ生成に流すと IS 最適化バイアスが直撃するため、構造的に再現不可。**同類実験を二度行わないこと**。 | [s3_decomposed_sweep_results.csv](s3_decomposed_sweep_results.csv) |
| **S4_RelVol_Gated** | RelVol 比率ゲート CFD レバ（l_base × k_rel × rel_th × HL、36 config） | CFDLeverage / RelVol | 2026-05-22 | D | +27.69% (best) | +0.750 (best) | N/A | Rejected | 36/36 が S2 単体（+0.770）を下回る。最良 (l_base=6, k_rel=3.0, rel_th=1.0, HL=10/60) でも IS-OOS gap +9.26pp。RelVol ゲートは過剰デレバ、Sharpe 改善寄与なし。H1 と同等。 | [h1_s4_param_sweep_results.csv](h1_s4_param_sweep_results.csv) |
| **B6_LT2_N_Sweep_Rejects** | B6 LT2 N スイープ落選パラメータ（N=500, 600, 750, 1000, 1250） | LongCycleSignal / Sweep | 2026-05-22 | D | N/A | N/A | N/A | Rejected | N=1500 が Sharpe_OOS +0.885 / IS-OOS gap −0.05pp で最良と確定（Active 昇格）。N=750 は Shortlisted に残置、N=500/600/1000/1250 は Sharpe で劣後し棄却。N=500 (Sharpe +0.830, gap +7.41pp), N=600 (+0.803, +6.79pp), N=1000 (+0.791, +1.63pp), N=1250 (+0.771, +3.72pp)。 | [b6_s2_lt2_N_sweep_results.csv](b6_s2_lt2_N_sweep_results.csv) |
| **B3_LT4_Sweep_Rejects** | B3 LT4 N×k_lt スイープ落選（N=750/k=0.7 以外の 8 config） | LongCycleSignal / Sweep | 2026-05-22 | D | N/A | N/A | N/A | Rejected | N=750/k=0.7 が最良で Shortlisted へ昇格。他 8 config（N=500/1000 × k=0.3/0.5/0.7 + N=750 × k=0.3/0.5）は Sharpe で劣後。 | [b3_s2_lt4_sweep_results.csv](b3_s2_lt4_sweep_results.csv) |
| **B4_LT6_Sweep_Rejects** | B4 LT6 N×k_lt スイープ落選（N=500/k=0.7 以外の 8 config） | LongCycleSignal / Sweep | 2026-05-22 | D | N/A | N/A | N/A | Rejected | N=500/k=0.7 が CAGR_OOS 最大で Shortlisted。他 8 config は劣後（N=750/k=0.7 は Sharpe +0.801 / gap +4.13pp で要注意系として参考）。 | [b4_s2_lt6_sweep_results.csv](b4_s2_lt6_sweep_results.csv) |
| **B5_LT7_Sweep_Rejects** | B5 LT7 k_lt スイープ落選（k_lt=0.5 以外の 2 config） | LongCycleSignal / Sweep | 2026-05-22 | D | N/A | N/A | N/A | Rejected | k_lt=0.5 が最良で Shortlisted。他 2 config は Sharpe で劣後、スイープ規模小（3 config）のためロバスト性証拠は限定的。 | [b5_s2_lt7_sweep_results.csv](b5_s2_lt7_sweep_results.csv) |

### §3.4 シグナル系実験（VIX / Regime / SOXL / DD等）

| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **A2_VIX_MD60** | A2（VIX + MD60 シグナル）FULL Scenario A | VIX / シグナル | 2026-04-21 | A | +14.85% | 0.998 (FULL) | −44.09% | Rejected | Scenario A の値であり Scenario D 補正で大幅劣化見込み。DH Dyn [A] Approach A に統合済み（VIX_MR コンポーネントとして）。単体は不要。 | [strategy_comparison_results.csv](strategy_comparison_results.csv) |
| **DD_Only_18_92** | DD(-18/92) Only（ドローダウンルール単体） | DD / シグナル | 2026-04-21 | A | +14.98% | 0.748 (FULL) | −72.88% | Rejected | DD ルール単体では MaxDD −72.88% で実運用不可。DH Dyn の構成要素として統合済み。 | [strategy_comparison_results.csv](strategy_comparison_results.csv) |
| **DD_VT_VolSpike_R4_Top1** | DD+VT+VolSpike(1.5x)（R4 トップ1） | VolSpike / DD / VT | 2026-04-21 | A | N/A | 0.902 (FULL) | −58.4% | Rejected | R4 個別実験でトップだが最終結論ではない（CURRENT_BEST_STRATEGY.md ブラックリスト記載）。Scenario A 値であり Scenario D 補正後の実力未検証。 | [R4_results.csv](R4_results.csv), [R4_RESULTS_SUMMARY_2026-02-06.md](R4_RESULTS_SUMMARY_2026-02-06.md), [ens2_comparison_results.csv](ens2_comparison_results.csv), [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |
| **DD_VT_Baseline** | DD(-18/92)+VT(25%) Baseline | DD / VT | N/A | A | N/A | 0.861 (FULL) | −61.9% | Rejected | R4 のベースライン。後続 DH Dyn 系に統合・置換済み。 | [ens2_comparison_results.csv](ens2_comparison_results.csv) |
| **DynHybrid_Static_50_25_25** | Dyn-Hybrid Static (50/25/25)（静的配分） | 静的配分 / DH_Dyn派生 | 2026-04-21 | A | +13.27% | 0.867 (FULL) | −22.92% | Rejected | 静的配分のため動的環境（金利急騰時等）に弱い。Sharpe 1.198 は高いが CAGR が低い。動的版（DH Dyn [A]）に置換済み。 | [strategy_comparison_results.csv](strategy_comparison_results.csv) |
| **DynHybrid_LevVix** | Dyn-Hybrid (lev+vix) | VIX / DH_Dyn派生 | 2026-04-21 | A | +15.76% | 0.769 (FULL) | −29.42% | Rejected | 中間版。DH Dyn [A] Approach A に統合・置換済み。 | [strategy_comparison_results.csv](strategy_comparison_results.csv) |
| **Regime_Strategy** | レジーム判定戦略（HMM / 状態推定） | Regime | N/A | N/A | N/A | N/A | N/A | Rejected | レジーム推定戦略は実験段階で R4 を超えず、後続 DH Dyn に吸収されることなく終了。実装も維持されていない。 | [regime_strategy_results.csv](regime_strategy_results.csv), [regime_analysis_stats.csv](regime_analysis_stats.csv), [regime_vs_r4_comparison.csv](regime_vs_r4_comparison.csv), [REGIME_ANALYSIS_REPORT_2026-04-04.md](REGIME_ANALYSIS_REPORT_2026-04-04.md) |
| **SOXL_Addition** | SOXL ポートフォリオ追加実験 | ポートフォリオ追加 / SOXL | N/A | N/A | N/A | N/A | N/A | Rejected | SOXL 追加は分散効果なく、ボラ増加のデメリットが上回り棄却。 | [soxl_addition_results.csv](soxl_addition_results.csv) |
| **External_Signal_Sweep** | 外部シグナル（HY, CPI 等）IS パラメータスイープ | TimingGate / 外部信号 | N/A | [非標準コスト] | N/A | N/A | N/A | Rejected | P-series（P01〜P05）に発展し Shortlisted に残存したが、当該スイープ単体としては最終採用なし。 | [external_signal_results.csv](external_signal_results.csv), [external_is_param_sweep.csv](external_is_param_sweep.csv), [external_oos_results.csv](external_oos_results.csv) |
| **Bond_Variant_Tests** | Bond 系変種（TMF パラメータスイープ等） | Bond / 派生 | N/A | A | N/A | N/A | N/A | Rejected | TMF 既定パラメータが最適と確認。他変種は採用に至らず。 | [bond_variant_results.csv](bond_variant_results.csv), [bond_model_annual_comparison.csv](bond_model_annual_comparison.csv), [bond_model_grid_results.csv](bond_model_grid_results.csv), [tmf_validation_results.csv](tmf_validation_results.csv) |
| **Gold_Signal_Research** | Gold シグナル（金関連トリガー）研究 | Gold / シグナル | N/A | N/A | N/A | N/A | N/A | Rejected | Gold シグナル単独では有意な改善観測されず。Gold は配分要素として残置するがシグナル化しない。 | [research_gold_signals.csv](research_gold_signals.csv) |

### §3.5 ポートフォリオ構造系実験（リバランス頻度・部分リバランス・分散化等）

| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **Rebalance_Frequency_Sweep** | リバランス頻度スイープ（週次/月次/四半期等） | リバランス頻度 | N/A | A | N/A | N/A | N/A | Rejected | DH Dyn シグナルに基づく動的リバランス（年27回）が最適と確認。固定頻度は劣後。 | [rebalance_frequency_results.csv](rebalance_frequency_results.csv) |
| **Partial_Rebalance** | 部分リバランス（許容バンド方式） | リバランス頻度 / 部分 | N/A | A | N/A | N/A | N/A | Rejected | 取引コスト削減効果はあるが、シグナル追従が遅延しリターン低下。最終的に維持されず。 | [partial_rebalance_results.csv](partial_rebalance_results.csv) |
| **Portfolio_Diversification** | 分散化実験（多資産追加） | ポートフォリオ追加 / 分散 | N/A | N/A | N/A | N/A | N/A | Rejected | 多資産追加（コモディティ・REIT 等）で分散効果限定的。CAGR 低下と相殺。 | [portfolio_diversification_results.csv](portfolio_diversification_results.csv) |
| **Overfitting_Validation** | 過剰適合検証（複数パラメータ組合せ） | 検証 / Robustness | N/A | N/A | N/A | N/A | N/A | Rejected | 検証実験。戦略採否そのものは別途決定。本実験単独での戦略は存在せず参照用ログ。 | [overfitting_validation_results.csv](overfitting_validation_results.csv) |
| **Hybrid_Strategy_Drafts** | Hybrid 戦略草案（早期段階複数試行） | DH_Dyn 草案 | N/A | A | N/A | N/A | N/A | Rejected | DH Dyn [A] 確立前の試行錯誤ログ。確定版（[A] 閾値 0.15）に吸収済み。 | [hybrid_strategy_results.csv](hybrid_strategy_results.csv), [dynamic_portfolio_results.csv](dynamic_portfolio_results.csv) |
| **Improvement_Iterations** | 改善イテレーション群（R1〜R4 派生） | 反復改善 | N/A | A | N/A | N/A | N/A | Rejected | R1〜R4 各ラウンドの試行ログ。R4 経由で DH Dyn [A] に収束済み。 | [improvement_results.csv](improvement_results.csv), [improvement_results_r2.csv](improvement_results_r2.csv), [improvement_results_r3.csv](improvement_results_r3.csv), [improvement_results_r4.csv](improvement_results_r4.csv), [improvement_results_a_vix.csv](improvement_results_a_vix.csv), [improvement_results_wf_vix.csv](improvement_results_wf_vix.csv), [improvement_results_next.csv](improvement_results_next.csv) |
| **Step2_Variants** | Step2 検証群（baseline / partB ma5 / partC macd / partD full / partE wf） | 検証 / 過去フェーズ | N/A | A | N/A | N/A | N/A | Rejected | Step1〜Step3 検証パイプラインの中間生成物。最終結論には組み込まれず。 | [step1_worst10y_results.csv](step1_worst10y_results.csv), [step2_baseline.csv](step2_baseline.csv), [step2_partB_ma5.csv](step2_partB_ma5.csv), [step2_partC_ma5_macd.csv](step2_partC_ma5_macd.csv), [step2_partD_full.csv](step2_partD_full.csv), [step2_partE_wf.csv](step2_partE_wf.csv), [step2_static_cagr25_results.csv](step2_static_cagr25_results.csv), [step3_dynamic_cagr25_results.csv](step3_dynamic_cagr25_results.csv) |
| **Realistic_Product_Variants** | 現実プロダクト変種（DELAY違い等） | 検証 / プロダクト現実性 | N/A | D | N/A | N/A | N/A | Rejected | DELAY=2 営業日が最適と確認。他 DELAY は採用見送り。 | [realistic_product_results.csv](realistic_product_results.csv), [delay_product_comparison_results.csv](delay_product_comparison_results.csv), [delay_robust_results.csv](delay_robust_results.csv), [DELAY_PRODUCT_COMPARISON_REPORT_2026-05-12.md](DELAY_PRODUCT_COMPARISON_REPORT_2026-05-12.md) |
| **Threshold_Tax_Sensitivity** | 閾値×税率感度分析 | 税務感度 | 2026-05-12 | D | N/A | N/A | N/A | Rejected | 感度分析であり戦略採否ではない。閾値 0.15 / 税率 20.315% 想定で本表完結。 | [THRESHOLD_TAX_SENSITIVITY_REPORT_2026-05-12.md](THRESHOLD_TAX_SENSITIVITY_REPORT_2026-05-12.md), [threshold_tax_sensitivity_results.csv](threshold_tax_sensitivity_results.csv) |
| **Financing_Cost_Variants** | Financing コスト感度（SOFR 倍率変動） | コスト感度 | N/A | D | N/A | N/A | N/A | Rejected | コスト感度実験。最終的に Scenario D が標準。 | [financing_cost_results.csv](financing_cost_results.csv) |
| **Approach_A_Sweep** | Approach A 内部パラメータスイープ | DH_Dyn / Sweep | N/A | A | N/A | N/A | N/A | Rejected | Approach A 確定版（閾値0.15）に至る過程の探索ログ。 | [approach_a_sweep_results.csv](approach_a_sweep_results.csv), [APPROACH_A_PROPOSAL_2026-04-20.md](APPROACH_A_PROPOSAL_2026-04-20.md) |
| **Phase1_Opt_Tier1_3** | Phase1 Tier1〜Tier3 最適化ログ | 最適化 / 過去フェーズ | N/A | A | N/A | N/A | N/A | Rejected | 最適化過程のログ。最終結論に吸収済み。 | [opt_phase1_tier1_results.csv](opt_phase1_tier1_results.csv), [opt_phase1_tier2_results.csv](opt_phase1_tier2_results.csv), [opt_phase1_tier3_results.csv](opt_phase1_tier3_results.csv), [opt_phase1_wf_results.csv](opt_phase1_wf_results.csv), [opt_phase2_results.csv](opt_phase2_results.csv), [opt_phase3_results.csv](opt_phase3_results.csv) |
| **LT_Sweep_Variants** | LT 系（lt_sweep / lt_extended / lt_combined）変種 | LongCycleSignal / Sweep | N/A | D | N/A | N/A | N/A | Rejected | LT2-N750-k0.5-modeB が最終確定（B1）。他 LT 変種（N=500, 1000 / modeA, modeC 等）は劣後。 | [lt_sweep_results.csv](lt_sweep_results.csv), [lt_extended_results.csv](lt_extended_results.csv), [lt_combined_results.csv](lt_combined_results.csv) |
| **Grid_Search_Legacy** | Grid Search レガシー実験 | 最適化 / 過去 | N/A | A | N/A | N/A | N/A | Rejected | 旧グリッド最適化。R4 / Phase 系に置換済み。 | [grid_search_results.csv](grid_search_results.csv) |
| **Factcheck_Sensitivity** | ファクトチェック感度分析 | 検証 / 監査 | N/A | N/A | N/A | N/A | N/A | Rejected | 過去結果の再現性検証ログ。戦略採否ではない。 | [factcheck_sensitivity_results.csv](factcheck_sensitivity_results.csv), [factcheck_log.txt](factcheck_log.txt) |
| **TQQQ_Verification** | TQQQ 再現性検証 | 検証 / プロダクト | N/A | D | N/A | N/A | N/A | Rejected | TQQQ シミュレータの一致確認用。戦略採否ではない。 | [tqqq_verification_results.csv](tqqq_verification_results.csv) |
| **Validation_Crisis_Full_OOS** | 暴落期 / FULL / OOS 個別検証 | 検証 / 期間別 | N/A | D | N/A | N/A | N/A | Rejected | 期間別検証ログ。Active 戦略の補強根拠として参照、独立戦略ではない。 | [validation_crisis.csv](validation_crisis.csv), [validation_full.csv](validation_full.csv), [validation_oos.csv](validation_oos.csv) |
| **Leverage_Bin_Analysis_V1_V4** | レバレッジビン分析 V1〜V4 | レバレッジ分析 | 2026-04-20 | A/D | N/A | N/A | N/A | Rejected | 分析レポート。CFD レバレッジ最適化の根拠資料として参照されるが、独立戦略ではない。 | [LEVERAGE_BIN_ANALYSIS_2026-03-19.md](LEVERAGE_BIN_ANALYSIS_2026-03-19.md), [LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md](LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md), [LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md](LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md), [LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md](LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md) |

---

## §4 Deferred — 再評価保留

> 「明確な棄却理由はないが、現時点では着手しない」戦略。前提条件（データ追加 / 標準コスト下での再計算 / 新規プロダクト登場等）が満たされ次第、Shortlisted への昇格を再評価する。

### 解放条件 (Promotion Rules)

| 条件 | 詳細 |
|---|---|
| **データ条件** | 必要な追加データ（例: OOS の延長、Real-time フィード接続）が揃った時点 |
| **コスト条件** | Scenario D 相当の標準コストモデルでの再バックテストが完了した時点 |
| **プロダクト条件** | 利用予定の現実プロダクト（CFD ブローカー / レバ ETF 等）が変更・追加された時点 |
| **再評価必須項目** | CAGR_OOS / Sharpe_OOS / IS-OOS gap / Worst10Y★ / MaxDD / P10_5Y / Worst5Y の 7 指標すべてを再計算し、Shortlisted §2 の閾値（CAGR_OOS ≥ 15% かつ Sharpe_OOS ≥ 0.65 を目安）を上回るか判定 |

### 現状エントリ

| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR_OOS | Sharpe_OOS | MaxDD | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| _(なし)_ | — | — | — | — | — | — | — | — | — | — |

> 現時点で Deferred に分類される戦略は無し。新規候補が現れた際にここへ追加。

---

## §5 テーマ別逆引きインデックス

> 「VolSpike 系は試したか？」「VIX シグナル系は？」のような問いに即答するための逆引き。
> 各テーマ末尾の `→ §X.Y` でテーブルへジャンプ。

### CFDLeverage（CFD 動的レバレッジ）
- **Active**: S2_VZGated+LT2_N750_E4RegimeKLT → §1
- **Shortlisted (2026-05-26追加)**: S2_VZGated+LT2_N750_E4_F10eps015（WFA G7 PASS 採用推奨待ち）, S2_VZGated+LT2_N750_vz065_lmax5（WFA G9 PASS Sharpe最高/MaxDD最良）, S2_VZGated+LT2_N750_F10lmax5（WFA G8 PASS P10_5Y最高） → §2
- **Shortlisted (既存)**: S2_VZGated+LT2_N750（旧Active, WFA完了済み fallback 第一候補）, S2_VZGated+LT4_N750_k0.7, S2_VZGated+LT6_N500_k0.7, S2_VZGated+LT7_k0.5, S2_VZGated, P2_VolTarget, S4_RelVol, CFD_7x_Fixed → §2
- **Rejected**: CFD_LMAX_Sweep_Rejects, CFD_NVOL_Sweep_Rejects, Discrete_Leverage_Rejects, Marugoto_Leverage, Lev2x3x_Baseline, E1_Ensemble_S2_Rejects, F1_Alloc_Sweep_Rejects, **P1_SOFR_Adaptive** (18/18 S2未満), **P3_Momentum_Lev** (16/16 S2未満, gap+14.66pp), **P4_Composite** (三因子乗算、全24config S2未満), **P5_Kelly_Sizing** (12/12 S2未満), **S1_Conviction** (A2直接変換、IS-OOSgap+21pp), **S3_Decomposed_A2** (gap+22.39pp 致命的過剰適合), **S4_RelVol_Gated** (36/36 S2未満), S2_TV/KVZ/GATEMIN_Sweep_Rejects → §3.3
- **関連分析**: Leverage_Bin_Analysis_V1_V4 → §3.5

### LongCycleSignal（長期サイクル / モメンタム逆張り = LT2 系）
- **Active**: S2_VZGated+LT2_N750_E4RegimeKLT（LT2-N750 + E4 Regime k_lt） → §1
- **Shortlisted**: S2_VZGated+LT2_N750（旧Active, 固定k=0.5, WFA完了済み）, S2_VZGated+LT4_N750_k0.7, S2_VZGated+LT6_N500_k0.7, S2_VZGated+LT7_k0.5, DH_Dyn_2x3x_A_LT2 → §2

### VolRegime（ボラレジーム動的パラメータ）
- **Active**: S2_VZGated+LT2_N750_E4RegimeKLT（vz による LT2 k_lt 動的調整） → §1
- **Rejected**: （該当なし — 初回検証）
- 結論: vz による LT2 バイアス強度の動的調整が S2+LT2 で +2.37pp CAGR_OOS / −1.99pp gap 改善を実現。
- **Rejected**: B2_KLT_Sweep_Rejects, B6_LT2_N_Sweep_Rejects, B3_LT4_Sweep_Rejects, B4_LT6_Sweep_Rejects, B5_LT7_Sweep_Rejects, LT_Sweep_Variants → §3.3, §3.5

### EnsembleVoting（アンサンブル / 投票統合）
- **Rejected**: Ens2_AsymSlope, Ens2_AsymSlope_lev30, Ens2_SlopeTrendTV, Ens2_SlopeTrendTV_lev30, MajorityVote_P2, E1_Ensemble_S2_Rejects → §3.2, §3.3
- 結論: max_lev=1.0 系は DH Dyn [A] に劣後、max_lev=3.0 系は IS-OOS gap・MaxDD で実用不可。**アンサンブルは現時点で全棄却**。

### VIX（VIX シグナル系）
- **統合済み**: DH Dyn [A] Approach A の VIX_MR コンポーネントとして組み込み済み → §1, §2
- **Rejected**: A2_VIX_MD60, DynHybrid_LevVix → §3.4
- 結論: VIX 単体ロジックは独立戦略として残らず。DH Dyn [A] への統合で完結。

### VolSpike（ボラスパイク系）
- **Rejected**: DD_VT_VolSpike_R4_Top1 → §3.4
- 結論: R4 トップだが最終結論ではない（CURRENT_BEST_STRATEGY.md ブラックリスト記載済み）。**VolSpike 単体での復活見込みなし**。
- 関連ログ: [VolSpike_Yearly_Returns.xlsx](VolSpike_Yearly_Returns.xlsx), [VolSpike_年次リターン.xlsx](VolSpike_年次リターン.xlsx), [VolSpike_年次リターン_v2.xlsx](VolSpike_年次リターン_v2.xlsx)

### SOXL（半導体レバ ETF 追加）
- **Rejected**: SOXL_Addition → §3.4
- 結論: 分散効果なくボラ増のみ。**棄却済み・再評価不要**。

### Regime（レジーム判定）
- **Rejected**: Regime_Strategy → §3.4
- 結論: HMM 等の状態推定は R4 を超えず、DH Dyn 系に置換済み。

### TimingGate（外部シグナルゲート = HY / CPI 等）
- **Shortlisted**: P02_DynCPI, P05_HYCPI, P01_DynHY（**いずれも [非標準コスト]**） → §2
- **Rejected**: C1_HY_Gate_Rejected, External_Signal_Sweep → §3.3, §3.4
- 結論: P-series は非標準コストにつき直接比較不可。標準コスト下での再評価が課題。

### DD（ドローダウンルール）
- **統合済み**: DH Dyn [A] Approach A の DD コンポーネントとして組み込み済み → §1, §2
- **Rejected**: DD_Only_18_92, DD_VT_Baseline → §3.4

### DH_Dyn / ETF配分（動的 ETF 配分系）
- **Shortlisted**: DH_Dyn_2x3x_A, DH_Dyn_2x3x_A_LT2 → §2
- **Rejected**: DH_Dyn_2x3x_A_ScA_v1, DH_Dyn_2x3x_A_ScD_v1, DH_Dyn_2x3x_A_th020 → §3.1
- **草案・反復**: Hybrid_Strategy_Drafts, Improvement_Iterations, Step2_Variants → §3.5

### Benchmark（ベンチマーク）
- **Shortlisted**: BH_1x（NASDAQ 素） → §2

### Bond / Gold（債券 / 金 個別研究）
- **Rejected**: Bond_Variant_Tests, Gold_Signal_Research → §3.4
- 結論: 個別最適化済み。Gold 2x / TMF 既定パラメータが Active 構成に統合。

### リバランス頻度 / 部分リバランス / 分散化
- **Rejected**: Rebalance_Frequency_Sweep, Partial_Rebalance, Portfolio_Diversification → §3.5

### 検証 / Robustness（最終戦略の補強実験）
- **Rejected**（独立戦略ではない補強実験）: D1_OOS_Boundary_Variants, Overfitting_Validation, Threshold_Tax_Sensitivity, Financing_Cost_Variants, Realistic_Product_Variants, Factcheck_Sensitivity, TQQQ_Verification, Validation_Crisis_Full_OOS → §3.3, §3.5

### Hybrid / Multi-Strategy（複数戦略の固定ウェイトブレンド）
- **Shortlisted**: Hybrid_S2LT2_P05_70_30（**[非標準コスト]**） → §2
- 結論: S2+LT2-N1500 の 70/30 ブレンドは MaxDD・IS-OOS gap を小改善するが Sharpe_OOS が僅かに低下。§1.3 制約で Active 昇格不可。**S2+LT2-N1500 単独継続が最優先**。

### Sweep / Optimization（パラメータ最適化過去ログ）
- **Rejected**: Approach_A_Sweep, Phase1_Opt_Tier1_3, Grid_Search_Legacy, A6_LMAX, A1_NVOL, B2_KLT, F1_Alloc, E1_Ensemble → §3.3, §3.5

---

## 記入ルール・更新プロトコル

### 戦略を **Active** に昇格する手順

1. `CURRENT_BEST_STRATEGY.md` を更新（一次根拠）
2. 本ファイル §1 を1行のみで上書き
3. 旧 Active を §2 Shortlisted（残置価値あり）または §3.1 Rejected（廃止）へ移動
4. `tasks.md` に1行ログ追記
5. `MEMORY.md` の "現行ベスト" セクション更新

### 戦略を **Rejected** に降格する手順

1. 廃止理由を明確に1行記載（例: 「S2+LT2 が全7指標で上回ったため」）
2. Verified Date は廃止判定日を記入
3. 該当 Evidence ファイルがある場合は必ず Markdown リンクで添付
4. **同様の実験を再度行わないこと** を確約（再評価したい場合は §4 Deferred へ移動して条件明記）

### 戦略を **Deferred** から **Shortlisted** に昇格する手順

1. §4 の解放条件（データ / コスト / プロダクト）が満たされたことを確認
2. Scenario D 標準コストで 7 指標（CAGR_OOS / Sharpe_OOS / IS-OOS gap / Worst10Y★ / MaxDD / P10_5Y / Worst5Y）を再計算
3. Shortlisted §2 の閾値（目安: CAGR_OOS ≥ 15% かつ Sharpe_OOS ≥ 0.65）を満たせば §2 へ移動
4. 満たさなければ §3 Rejected に降格、解放条件が再現不可能と判明した場合のみ

### **新規戦略を追加** する手順

1. **必ず Scenario D 標準コスト** でバックテスト実行
2. 7 指標を計算（不明項目は `N/A`）
3. Evidence（Markdown レポートまたは CSV + 実装スクリプト）を必ず添付
4. テーマを §5 逆引きから選択（無ければ新テーマを追加）
5. Shortlisted §2 に追加し、本表に新規行を追記
6. Active より優位な場合は上記「Active 昇格手順」を実行

### 記入時の禁則事項

- **❌ Scenario A 単独の値で Active / Shortlisted 判定を行わない**（Scenario D 補正必須）
- **❌ FULL Sharpe を OOS Sharpe として記載しない**（FULL の場合は `(FULL)` を明記）
- **❌ Evidence ファイル名のみの記載は不可**（必ず Markdown リンク `[name](path)` 形式）
- **❌ 「ベスト」表現を §2 以下で使用しない**（Active は §1 のみ）
- **❌ P-series の指標を他系統と直接比較しない**（[非標準コスト] フラグ必須）

### 自己整合性チェック（更新後に必ず実施）

1. §1 Active が `CURRENT_BEST_STRATEGY.md` 冒頭と一致するか
2. §2〜§4 の Strategy ID 重複がないか
3. §5 逆引きインデックスから §1〜§4 への参照が全て生存しているか
4. Evidence の Markdown リンクが死んでいないか（該当ファイルが repo に存在するか）

---

## 一次根拠ファイル（本ファイル更新時に参照する正典）

| ファイル | 役割 |
|---|---|
| [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) | Active の単一の真実 |
| [STRATEGY_COMPARISON_INTEGRATED_2026-05-22.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-22.md) | 12戦略統合比較表（統一9指標、v1.1準拠） |
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | Active 採用判定（PASS） |
| [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) | 評価基準の標準定義 |
| [FILE_INDEX.md](FILE_INDEX.md) | リポジトリ全体のファイル索引 |
| [tasks.md](tasks.md) | タスクログ |
| [src/corrected_strategy_backtest.py](src/corrected_strategy_backtest.py) | Scenario D 標準コストの単一の真実 |
| [src/product_costs.py](src/product_costs.py) | コスト定数の単一の真実 |

---

## 変更履歴

- **2026-06-03 (v4)**: **DH_T4_Improved を §2 Shortlisted → §3 Rejected に降格** (`lev_mod` で TQQQ position 連続スケールは ETF 「保有/非保有」制約違反)。代替として **DH_Z2_AllocTiming を §2 Shortlisted に新規追加** (商品は TQQQ+TMF+2036 維持、F10 ε tilt 配分 + binary HOLD/OUT vz_gate、レバ操作なし、peak lev 2.85x≤3.0x assert で機械検証)。**IS-OOS gap +10.46pp→+2.17pp (-8.29pp)・WFE 0.662→1.058 (完全汎化)・CAGR_OOS +12.26% (REF +9.56% を 2.70pp 上回る)・OOS 累積 ×2.00 (REF ×1.73)**。Worst10Y/P10_5Y は防御性能低下 (trade-off)。STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md は v3→v4、§0'/§5/§6/§6-2 4 箇所で DH-T4→DH-Z2 完全置換。
- 2026-06-03 (v3, 取消): §2 Shortlisted に DH_T4_Improved 追加。**↑ v4 で破棄。**
- 2026-05-26: §2 Shortlisted に F10 ε=0.015 (G7 WFA PASS, CI95_lo=+27.93%) / vz065+lmax5 (G9 PASS, CI95_lo=+24.82%) / F10+lmax5 (G8 PASS, CI95_lo=+25.57%) の3戦略を追加。採用変更はユーザー判断待ち（INTEGRATION_DEBATE_2026-05-26.md 参照）。§5 逆引きの CFDLeverage / VolRegime を更新。
- 2026-05-24 (G3 WFA): E4 Regime k_lt の WFA が α∩β PASS (CI95_lo=+26.51%, t_p=0.0000, WFE=+1.131)。暫定 Active → **正式 Active 確定**。REF-N750 サニティ CI95_lo +25.73% / WFE +1.145 は G1 参照値と完全一致（diff -0.00pp / +0.000）。§1 Active 行から「暫定」表記を削除、CI95_lo / WFE 実測値を記入。Evidence に G3 関連ファイル 3点を追記。
- 2026-05-24: Active を `S2_VZGated+LT2_N750` → `S2_VZGated+LT2_N750_E4RegimeKLT` に暫定昇格（E4 sweep 64 config / 12 PASS / 採用 config: k_lo=0.1, k_hi=0.8, vz_thr=0.7）。旧 Active は §2 Shortlisted へ移動（WFA PASS 済みで fallback 価値あり）。§5 逆引きに新テーマ `VolRegime` 追加。E4 WFA 完了次第、暫定→正式 Active へ確定予定。
- 2026-05-23: Active を `S2_VZGated+LT2_N1500` → `S2_VZGated+LT2_N750` に差し戻し（コミット 8503200）。CAGR_OOS +31.16%・MaxDD −59.45%・Worst10Y★ +18.10% のリスク主要軸で N=750 が優位と再確認。N=1500 は Sharpe_OOS 最高値（+0.885）の実績で Shortlisted に移動。
- 2026-05-22 (E2): §2 Shortlisted に `Hybrid_S2LT2_P05_70_30` を追加（E2 検証 2/3 基準改善・Shortlisted 判定）。§5 逆引きに「Hybrid / Multi-Strategy」新テーマ追加。一次根拠ファイルを 2026-05-22 版に更新。
- 2026-05-22: Active を `S2_VZGated+LT2_N750` → `S2_VZGated+LT2_N1500` に昇格（B6 N-sweep で Sharpe_OOS +0.885 / IS-OOS gap −0.05pp が最良と確認）。§2 Shortlisted に B3/B4/B5 最良 config + 旧 Active N=750 を追加（計5件）。§3.3 Rejected に P1/P3/P5/S3/S4 と B3〜B6 落選 config を追加。逆引きインデックス（CFDLeverage / LongCycleSignal）を更新。
- 2026-05-21: 初版作成。Active = `S2_VZGated+LT2`。Shortlisted 10件、Rejected カテゴリ別整理、Deferred 空、逆引きインデックス整備。

---

*管理者: 男座員也（Kazuya Oza）*
*本ファイルは [tasks.md](tasks.md) および [FILE_INDEX.md](FILE_INDEX.md) に「戦略台帳」として登録予定。*
