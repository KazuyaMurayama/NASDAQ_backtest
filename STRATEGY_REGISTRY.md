# STRATEGY REGISTRY — 戦略台帳（Active / Shortlisted / Rejected / Deferred）

作成日: 2026-05-21
最終更新日: 2026-06-07 (信号拡張プロジェクト Phase A→D + Sessions 1-5 完了 — DH-W1 baseline の canonical split 9+1 指標を §2 Shortlisted に追補、S3 + nasdaq_mom63 × M6 defensive overlay を §2 新規追加。同日早朝に DH-W1 キャッシュ・スリーブ 1倍投信置換 4戦略 P2/P5/P7 を §2 Shortlisted に追加済み。**同日後半**: S3 overlay tuning sweep (10 variants) から **V7 pure_boost {1.20,1.10,1.00,1.00}** を Shortlisted 追加 — min(CAGR_IS, CAGR_OOS) > 18% target 達成の唯一 "MaxDD 据置" 候補 (V0 defensive と相補)。全 Shortlisted エントリに税後 CAGR (NISA 内非課税 / 課税口座 ×0.8273) 情報を併記)
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

| Scenario | 内容 | 指標表記 |
|---|---|---|
| **D** | 標準コスト（SOFR financing drag + TER + swap spread 補正済み、`src/corrected_strategy_backtest.py`）。**税引前**。2026-05-12 以降の標準。 | ⓒ（税引前） |
| **D+tax** | Scenario D + **§3-A 税モデル**（`CAGR_net=(CAGR−0.66%)×0.8273` 逐年複利）+ moderate spread（CFD: 0.05% / ETF: 0.10%）適用。integrated report §0' の値はこの基準。 | ⓽（税後） |
| **A** | 旧コストモデル（SOFR 補正なし）。参考値のみ。2026-05-12 以前の値は基本このシナリオ。 | ⓒ（税引前） |
| **N/A** | コストモデル不明 / 旧実験ログのみ。再現性なし。 | — |
| **[非標準コスト]** | P-series（P01〜P05系）独自コスト。他系統と直接比較不可（§17 参照）。 | ‡ |

### 記号凡例（コスト・税の表記）

| 記号 | 意味 |
|---|---|
| **ⓒ** | コスト後・**税引前**（Scenario D 補正済み） |
| **⓽** | 税後・手取り（D+tax: §3-A 逐年複利 + moderate spread 適用後） |
| **ⓞ** | 原値（コスト・税で変化しない：取引回数・WFE など） |
| **ⓡ** | 再計算値（WFA 実測値 + §3-A 税調整後） |
| **‡** | 非標準コスト（P-series 独自、他系統と直接比較不可） |

> ⚠ **ミスリード防止**: `D`ⓒ（税引前）と `D+tax`⓽（税後）は**直接数値比較不可**。税後換算は §3-A モデルを適用すること。IS-OOS gap ⚠⚠ 判定基準: **≥+5pp は過学習警戒**。

### Evidence 列について

- 一次根拠ファイル（Markdown レポート、CSV、Python 実装）を **`[ファイル名](相対パス)`** 形式で記載
- 複数ある場合はカンマ区切り
- ファイルが存在しない（実験ログのみ）場合は `N/A` または該当 CSV 名のみ

### Verified Date の意味

- 当該指標が「最後に正規バックテストで確認された日付」
- 廃止戦略の場合は「廃止判定が確定した日付」
- 旧実験で日付不明な場合は `N/A`

### 列定義（全テーブル共通）— **標準10指標**

```
| Strategy ID | Name | Theme | Verified Date | Cost Scenario | CAGR ⓒ IS/OOS | IS-OOS gap ⓒ | Sharpe ⓒ_OOS | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ /yr | WFE ⓞ | CI95 ⓡ_lo | Status | Decision Reason | Evidence |
```

**CAGR ⓒ IS/OOS 列の表記**: `IS値 / OOS値` の2値を併記し、**下段に min(IS,OOS)を太字**で示す。
`D`ⓒ = 税引前 Scenario D 値 / `D+tax`⓽ = §3-A 税後換算値（†マーク付き行）

> §1 Active / §2 Shortlisted は全10指標必須。§3 Rejected / §4 Deferred は不明項目 `N/A` 可。

---

## §1 Active — 採用中（現行ベスト）

> 1行のみ。`CURRENT_BEST_STRATEGY.md` と必ず同期。差異があれば本ファイルではなく `CURRENT_BEST_STRATEGY.md` 側を一次根拠とする。

| 戦略（ID / 構成） | Date | Cost | CAGR ⓒ IS / OOS（min） | IS-OOS gap ⓒ | Sharpe ⓒ | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ | WFE ⓞ | CI95 ⓡ | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E4 RegimeKLT** ◆Active<br>S2+LT2-N750+E4 k_lt<br>k_lo=0.1, k_hi=0.8,<br>vz_thr=0.7（tiltなし）<br>CFDLev+LongCycle+VR | 2026-05-24 | D（ⓒ 税引前） | +31.72% / +33.53%（**min +31.72%**） | **−1.81pp** | **0.891** | −60.01% | **+18.67%** | **+9.78%** | **27** | **✅ 1.131** | **+26.51%** | **Active (WFA PASS 確定)** | E4 sweep 64 config 中 Worst10Y★ +18.67% / MaxDD −60.01% を同時最良化。IS-OOS gap −1.81pp（OOS が IS を +1.81pp 上回る）は本プロジェクト最高水準の汎化性。Trades/yr 約27回（月2.3回）でコスト優位。tilt系 (F7v3/F8) は WFA PASS するも Trades/yr 182回超・OOS偶然性疑いにより採用せず本戦略に復帰。**G3 WFA PASS (2026-05-24): CI95_lo=+26.51%, WFE=+1.131** → 正式 Active 確定（復帰）。⚠ 上記CAGR値はⓒ税引前。税後⓽換算: §3-A 適用で概算 CAGR_OOS ≈ +27.2%（単純推計）/ 保守値 ≈ +24.6%（コスト過少 −66bps 補正後）。 | [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md), [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md), [g3_wfa_e4_summary.csv](g3_wfa_e4_summary.csv), [src/e4_regime_klt.py](src/e4_regime_klt.py), [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |

---

## §2 Shortlisted — 最終候補（未採用）

> 11戦略統合比較表（[STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md)）に掲載され、最終候補に残ったが Active に届かなかった戦略群。
> Active が崩壊した際の次善候補。新パラメータ・新シグナルでの派生検証の出発点としても利用。

| 戦略（ID / 構成） | Date | Cost | CAGR ⓒ IS / OOS（min） | IS-OOS gap ⓒ | Sharpe ⓒ | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ | WFE ⓞ | CI95 ⓡ | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F10ε+E4** (G7 PASS)<br>S2+LT2-N750<br>F10ε=0.015, lmax=7<br>CFDLev+LT+VR+Tilt | 2026-05-26 | D（ⓒ 税引前） | +32.49% / +36.8%（**min +32.49%**） | ⚠ −4.31pp | 0.934 | −63.1% | N/A | N/A | 52 | ✅ 1.208 | +27.93% | Shortlisted (WFA G7 PASS, **採用推奨・ユーザー判断待ち**) | G7 WFA PASS（CI95_lo=+27.93%・全候補中最高, WFE=+1.208）。Sharpe_OOS最高+0.934・CAGR_OOS最高+36.8%。IS-OOS gap=-4.31pp⚠ は4候補中最大（汎化性懸念）。Trades/yr=52（E4比約2倍）。採用変更はユーザー判断待ち。 | [INTEGRATION_DEBATE_2026-05-26.md](INTEGRATION_DEBATE_2026-05-26.md), [G7_WFA_F10_2026-05-26.md](G7_WFA_F10_2026-05-26.md), [src/g7_wfa_f10.py](src/g7_wfa_f10.py) |
| **vz065+l5** (G9 PASS)<br>S2+LT2+E4<br>vz_thr=0.65, lmax=5.0<br>CFDLev+LT+VolRegime | 2026-05-26 | D（ⓒ 税引前） | +29.59% / +33.5%（**min +29.59%**） | −3.91pp | 0.947 | −51.8% | N/A | N/A | 27 | ✅ 1.272 | +24.82% | Shortlisted (WFA G9 PASS, Sharpe最高・MaxDD最良) | G9 WFA PASS（CI95_lo=+24.82%, WFE=+1.272）。Sharpe_OOS=+0.947 は全候補中最高。MaxDD=-51.8% は4候補中最良（E4比+8.2pp 改善）。IS-OOS gap=-3.91pp（E4比-2.1pp 悪化）。Trades/yr=27 でE4同等コスト。 | [INTEGRATION_DEBATE_2026-05-26.md](INTEGRATION_DEBATE_2026-05-26.md), [G9_WFA_VZ065_LMAX5_2026-05-26.md](G9_WFA_VZ065_LMAX5_2026-05-26.md) |
| **F10ε+lmax5** (G8 PASS)<br>S2+LT2, F10ε=0.015<br>lmax=5.0<br>CFDLev+LT+VR+Tilt | 2026-05-26 | D（ⓒ 税引前） | +30.23% / +33.6%（**min +30.23%**） | −3.37pp | 0.943 | −54.2% | +16.9% | +12.8% | 52 | ✅ 1.278 | +25.57% | Shortlisted (WFA G8 PASS, P10_5Y最高) | G8 WFA PASS（CI95_lo=+25.57%, WFE=+1.278）。P10_5Y=+12.8% は4候補中最高。IS-OOS gap=-3.37pp は4候補中2位（E4に次いで良好）。Worst10Y★=+16.9%（E4比-1.8pp 劣後）。Trades/yr=52（lev_raw+wn/wb 正計上）。 | [INTEGRATION_DEBATE_2026-05-26.md](INTEGRATION_DEBATE_2026-05-26.md), [G8_WFA_F10LMAX5_2026-05-26.md](G8_WFA_F10LMAX5_2026-05-26.md), [f10lmax5_fullmetrics.csv](f10lmax5_fullmetrics.csv), [src/g8_wfa_lmax5.py](src/g8_wfa_lmax5.py) |
| **F8-R5** (G5 PASS)<br>S2+LT2+E4+F8 tilt=10<br>cap=calm0.15<br>CFDLev+LT+VR+Tilt | 2026-05-24 | D（ⓒ 税引前） | N/A / +36.83%（min N/A） | N/A | 0.934 | −63.07% | N/A | N/A | 182 | ✅ 1.208 | +27.92% | Shortlisted | G5 WFA PASS（CI95_lo=+27.92%, WFE=+1.208）。Sharpe/CAGR/WFAはE4より優位だが Trades/yr 182回（E4比7倍）のコスト負担と OOS 偶然性疑いにより採用見送り。Active に戻す条件: Trades/yr 削減策実装 or コスト込み実証（実取引ログ）。 | [F8_REGIME_TILT_2026-05-24.md](F8_REGIME_TILT_2026-05-24.md), [G5_WFA_F8R5_2026-05-24.md](G5_WFA_F8R5_2026-05-24.md), [g5_wfa_f8r5_summary.csv](g5_wfa_f8r5_summary.csv) |
| **F7v3+E4** (G4 PASS)<br>S2+LT2+E4 tilt=2.0<br>flat cap=0.10<br>CFDLev+LT+VR+Tilt | 2026-05-24 | D（ⓒ 税引前） | N/A / +36.30%（min N/A） | N/A | 0.926 | −61.96% | N/A | N/A | 183 | ✅ 1.203 | +27.15% | Shortlisted | G4 WFA PASS（CI95_lo=+27.15%, WFE=+1.203）。Trades/yr 183回（E4比7倍）の同一理由で採用見送り。F8-R5よりMaxDD良好（−61.96%）。Trades削減策実装後の再評価候補。 | [F7V3_BULL_TILT_2026-05-24.md](F7V3_BULL_TILT_2026-05-24.md), [G4_WFA_F7V3_2026-05-24.md](G4_WFA_F7V3_2026-05-24.md) |
| **E4 RegimeKLT**<br>S2+LT2-N750+E4<br>k_lo=0.1, vz=0.7<br>(旧Active fallback) | 2026-05-24 | D（ⓒ 税引前） | +31.72% / +33.53%（**min +31.72%**） | −1.81pp | 0.891 | −60.01% | +18.67% | N/A | 27 | ✅ 1.131 | +26.51% | Shortlisted | 旧 Active（〜2026-05-24）。F7v3 Bull-Tilt が Sharpe +0.926 で上回り降格。WFA G3 PASS済み（CI95_lo=+26.51%, WFE=+1.131）。MaxDD −60.01% は本プロジェクト最良水準。最保守的な fallback 候補。 | [E4_REGIME_KLT_SWEEP_2026-05-24.md](E4_REGIME_KLT_SWEEP_2026-05-24.md), [G3_WFA_E4_2026-05-24.md](G3_WFA_E4_2026-05-24.md) |
| **S2+LT2-N750**<br>k=0.5 固定 modeB<br>（旧◆Active G1 PASS）<br>CFDLev+LT | 2026-05-24 | D（ⓒ 税引前） | +31.34% / +31.16%（**min +31.16%**） | +0.18pp | 0.858 | −59.45% | +18.10% | N/A | 27 | ✅ 1.145 | +25.7% | Shortlisted | 旧 Active（〜2026-05-24）。E4 Regime k_lt に CAGR_OOS −2.37pp / Sharpe_OOS −0.033 / Worst10Y★ −0.57pp / IS-OOS gap +1.99pp で劣後し降格。WFA 完了済み（CI95_lo=+25.7%, WFE=1.145 PASS α∩β）のため代替候補として強い残置価値あり。E4 の WFA 未通過時の fallback 第二候補。 | [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md), [b1_s2_lt2_results.csv](b1_s2_lt2_results.csv), [g1_wfa_summary.csv](g1_wfa_summary.csv) |
| **S2_VZGated**<br>tv=0.8, k_vz=0.3<br>gate_min=0.5<br>CFDLev | 2026-05-21 | D（ⓒ 税引前） | N/A / +27.57%（min N/A） | N/A | 0.769 | −62.4% | N/A | N/A | N/A | N/A | N/A | Shortlisted | 旧Active。S2+LT2 に全7指標（CAGR_OOS/Sharpe_OOS/IS-OOS gap/Worst10Y★/MaxDD/P10_5Y/Worst5Y）で敗北し降格。LT2 を外せばこの構成。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md), [src/cfd_leverage_backtest.py](src/cfd_leverage_backtest.py), [src/dynamic_leverage_strategies.py](src/dynamic_leverage_strategies.py) |
| **P2 VolTarget**<br>tv=0.8 (旧Active)<br>CFDLev+VolTarget | 2026-05-21 | D（ⓒ 税引前） | N/A / +27.13%（min N/A） | N/A | 0.757 | −60.5% | +19.09% | N/A | N/A | N/A | N/A | Shortlisted | S2_VZGated とほぼ同水準。Worst10Y★ +19.09% は S2 より僅かに優位だが Sharpe_OOS 0.757 < 0.769 で総合敗北。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **S4 RelVol**<br>l_base=7, k_rel=2.0<br>CFDLev | 2026-05-21 | D（ⓒ 税引前） | N/A / +26.19%（min N/A） | ⚠⚠ ~+14.8pp | 0.697 | −66.1% | N/A | N/A | N/A | N/A | N/A | Shortlisted | CAGR_IS +40.98% と高いが OOS で +26.19% へ大幅低下（IS-OOS gap 約 14.8pp）。過剰適合疑い。MaxDD −66.1% も大きい。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md), [src/dynamic_leverage_strategies.py](src/dynamic_leverage_strategies.py) |
| **CFD 7x 固定**<br>DH Dyn+7x レバ<br>CFDLev | 2026-05-21 | D（ⓒ 税引前） | N/A / +24.44%（min N/A） | ⚠⚠ ~+19pp | 0.670 | −65.0% | +25.37% | N/A | N/A | N/A | N/A | Shortlisted | CAGR_IS +43.35% / OOS +24.44%（IS-OOS gap ~19pp）と過剰適合大。固定 7x レバはリスク管理面で実運用不可。Worst10Y★ +25.37% のみ最高。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **DH Dyn 2x3x [A]**<br>TQQQ, thr=0.15<br>（旧Active ETF）<br>DH_Dyn+ETF | 2026-05-21 | D（ⓒ 税引前） | N/A / +14.88%（min N/A） | N/A | 0.646 | −45.08% | N/A | N/A | N/A | N/A | N/A | Shortlisted | 旧Active（〜2026-05-21）。CFD系の登場で CAGR_OOS で大幅劣後。本番運用候補としては最も保守的かつ NISA 適合可能性あり（CFD不要）。MaxDD −45% は CFD 系より良好。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md), [src/corrected_strategy_backtest.py](src/corrected_strategy_backtest.py), [corrected_strategy_results.csv](corrected_strategy_results.csv) |
| **vz065+l5+F10ε** ⭐⭐<br>v4.7 CFD Active候補<br>vz=0.65, lmax=5.0<br>F10ε=0.015<br>CFDLev+LT+VR+Tilt | 2026-06-05 | D+tax (CFD 0.05%)（⓽ 税後） | +20.16%† / +18.93%†（**min +18.93%**） | +1.23pp | 0.841 | −56.72% | ~+12.67% | ~+8.75% | 86 | ✅ 1.389 | N/A | **🟢 CFD 環境 Active 候補 (v4.7 確定、ユーザー判断で l7 から置換)** | v4.6 §6-5 lmax sweep で防御指標 (MaxDD/Worst10Y/P10_5Y) すべて最良。v4.7 でユーザー判断により l7 (旧 REF) から **CFD 環境 Active 候補に昇格**。**旧 REF (l7) 比で**: min CAGR -1.30pp の trade-off と引換に、Worst10Y +2.71pp / **P10_5Y +4.70pp 大幅改善** / MaxDD -9.23pp 浅化 / Sharpe +0.012 改善 / Trades 86 (l7 比 18% 低コスト)。gap +1.23pp で過学習なし、WFE 1.389。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §0'/§6-5 (v4.6/v4.7), [src/g27_vz065_lmax_sweep.py](src/g27_vz065_lmax_sweep.py) |
| **vz065+l5.5+F10ε** ⭐<br>v4.6 バランス候補<br>vz=0.65, lmax=5.5<br>CFDLev+LT+VR+Tilt | 2026-06-05 | D+tax (CFD 0.05%)（⓽ 税後） | +20.38%† / +19.65%†（**min +19.65%**） | +0.73pp | 0.832 | −59.87% | ~+12.46% | ~+7.92% | N/A | N/A | N/A | **Shortlisted (v4.6 バランス候補、l5 と l7 の中間)** | v4.6 §6-5 lmax sweep で「攻めと守りのバランス最良」と判定。l7 (旧 REF) 比で min CAGR -0.58pp / Worst10Y +2.50pp 改善 / P10_5Y +3.87pp 改善 / MaxDD -6.08pp 浅化 / gap +0.73pp (最小、過学習なし)。l5 (v4.7 Active) ほど防御に振らないが、l7 ほど攻めも犠牲にしない中間。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §6-5 (v4.6), [src/g27_vz065_lmax_sweep.py](src/g27_vz065_lmax_sweep.py), [g27_vz065_lmax_sweep_metrics.csv](g27_vz065_lmax_sweep_metrics.csv) |
| **vz065+l7+F10ε**<br>v4.5 REF → v4.7 副候補<br>vz=0.65, lmax=7.0<br>CFDLev+LT+VR+Tilt | 2026-06-05 | D+tax (CFD 0.05%)（⓽ 税後） | +20.23%† / +21.49%†（**min +20.23%**） | −1.26pp | 0.829 | −65.95% | +9.96% | +4.05% | ~105 | N/A | N/A | Shortlisted (CFD 攻め型副候補、v4.7 で l5 に主候補譲渡) | v4.5 で min(IS, OOS) ルール下 5 戦略中 1 位 (min=+20.23%) と判定され CFD Active 候補に昇格したが、v4.7 でユーザー判断により l5 (より防御指標優位) に主候補譲渡。**攻め重視ユーザーには依然有力候補**: OOS 累積 ×3.38、Worst10Y +9.96%、P10_5Y +4.05% (l5 比で防御弱)、MaxDD -65.95% (l5 比 9.23pp 深)。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §0'/§7-2 (v4.5/v4.7), [src/g26_new_asymmetric_variants.py](src/g26_new_asymmetric_variants.py), [g24_5strategies_verification.csv](g24_5strategies_verification.csv) |
| **DH-W1** ⭐v4.5<br>ETF Active候補<br>Asymm Hyst<br>Enter≥0.7, Exit≤0.3<br>DH_Dyn+VR+Hyst | 2026-06-07 (canonical 再計算) | D+tax (ETF 0.10%)（⓽ 税後） | +18.10%† / +18.91%†（**min +18.10%**） | −0.81pp | 0.845 | −34.57% | +10.37% | +4.82% | 68.7 | ✅ 1.023 | +13.61% | **Shortlisted → ETF 環境 Active 候補 (v4.5 §7-2 min ルール下、ETF 制約下で最良)** | DH-W1 は ETF only 環境 (CFD 不可、NISA 等) で DH (min +9.56%) を +4.10pp 上回る唯一の改善案。CFD 環境では vz065_l7_F10eps015 (min +20.23%) に大きく劣後 (-6.57pp)。WFA 50窓 PASS (CI95_lo=+13.95%, **WFE=0.997 完全汎化**, p=0.0000, Trades 17.8/yr)。Bootstrap (W1 vs DH-REF) median +3.65pp positive, P(diff>0)=72%。Threshold (enter, exit) 24 組合せ sweep 全て OOS +12〜16.7% でロバスト。**ETF 制約環境の第一推奨**。 **2026-06-07 canonical split (2021-05-08) 再計算追補**: CAGR_IS=+18.10%, CAGR_OOS=+18.91%, Sharpe_OOS=0.845, MaxDD=-34.57%, Worst10Y★=+10.37%, P10_5Y▷=+4.82%, Trades=68.7/yr(NAV proxy), IS-OOS gap=-0.81pp, WFA_CI95_lo=+13.61%, WFE(50w Sharpe)=1.023。旧値 (+13.66%) は v4.5 §7-2 min ルール (IS, OOS の最小値) 表示で、本行 CAGR_OOS は OOS 単独値に統一。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §0'/§6-2.5/§6-3/§7-2 (v4.3/v4.5), [STRATEGY_DH_REFINEMENT_20260603.md](STRATEGY_DH_REFINEMENT_20260603.md), [SIGNAL_EXPANSION_FINAL_DECISION_20260607.md](SIGNAL_EXPANSION_FINAL_DECISION_20260607.md) §3.1 (canonical 9+1), [src/g23a_dh_refinement_variants.py](src/g23a_dh_refinement_variants.py), [g23d_dh_w1_wfa.csv](g23d_dh_w1_wfa.csv), [g23e_dh_w1_bootstrap_results.csv](g23e_dh_w1_bootstrap_results.csv), [g23f_dh_w1_threshold_sweep.csv](g23f_dh_w1_threshold_sweep.csv), [data/signals/expansion/decision_9metrics_20260607.csv](data/signals/expansion/decision_9metrics_20260607.csv) |
| **DH-W1+mom63 V0** ⭐NEW<br>nasdaq_mom63 overlay<br>M6 def / Q4 mult<br>{1.1,1.0,0.9,0.8}<br>**defensive (MaxDD改善優先)**<br>DH_Dyn+Signal | 2026-06-07 | D+tax (ETF 0.10%)（⓽ 税後） | +16.69%† / +18.06%†（**min +16.69%**） | −1.37pp | **0.892** | **−28.74%** | +10.75% | +5.21% | N/A | ✅ 1.005 | +13.00% | **Shortlisted (ETF 環境 overlay 採用可、Phase D Hard Gate 4/4 全 PASS、S3 限定)** | 信号拡張プロジェクト (Phase A→D + Sessions 1-5, 76 信号 × 306+ patterns 検証) の **唯一の正式 ADOPT 候補**。DH-W1 baseline 比で **MaxDD -34.57% → -28.74% (+5.83pp 改善)** が headline。Sharpe_OOS +0.845→+0.892 (+0.047 改善)、Worst10Y★ +10.37%→+10.75%、P10_5Y▷ +4.82%→+5.21%。CAGR_OOS は -0.86pp の trade-off (-18.91%→+18.06%)。Phase D Hard Gate: WFE(50w Sharpe)=1.005 PASS, CI95_lo=+13.00% PASS, **Bootstrap P(Sharpe>base)=0.930 PASS, P(MaxDD better)=0.988 PASS** ⭐。Session 5 cross-strategy transfer test で S2(D5) WFE=0.963, E4(Active) WFE=0.958 と FAIL → **S3 (ETF only) 限定**で運用、CFD 系には転用不可。canonical split 再計算 (2021-05-08): WFA_CI95_lo=+12.65%, WFE(calendar)=0.782。**ETF only 環境 (NISA 等) ユーザーの第一推奨 overlay**。t_p/permutation 未実施のため §1 Active 昇格は保留、ユーザー判断で採用可。**税後 CAGR (NISA 内非課税 = pretax と同値、課税口座は ×0.8273)**: calendar-year ベース CAGR_OOS = pretax +14.33% / aftertax +11.97% ([aftertax_cagr_20260607.csv](data/signals/expansion/aftertax_cagr_20260607.csv))。 | [SIGNAL_EXPANSION_FINAL_DECISION_20260607.md](SIGNAL_EXPANSION_FINAL_DECISION_20260607.md) §1.2/§3.1/§3.2/§3.6 (v2 canonical), [LESSONS_LEARNED_20260607.md](LESSONS_LEARNED_20260607.md), [data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md), [data/signals/expansion/decision_9metrics_20260607.csv](data/signals/expansion/decision_9metrics_20260607.csv), [data/signals/expansion/session5_transfer_report_20260605.md](data/signals/expansion/session5_transfer_report_20260605.md), [data/signals/expansion/aftertax_cagr_20260607.csv](data/signals/expansion/aftertax_cagr_20260607.csv), [src/integration/build_strategy_with_signal.py](src/integration/build_strategy_with_signal.py) |
| **DH-W1+mom63 V7** ⭐NEW (2026-06-07)<br>nasdaq_mom63 overlay<br>M6 def / Q4 mult<br>{1.20,1.10,1.00,1.00}<br>**pure_boost (CAGR死守 / MaxDD baseline据置)**<br>DH_Dyn+Signal | 2026-06-07 | D+tax (ETF 0.10%)（⓽ 税後） | +18.61%† / +19.18%†（**min +18.61%**） | **−0.57pp** | 0.841 | −34.57% | +11.02% | +5.22% | 26.5 | ✅ 1.029 | +14.06% | **Shortlisted (ETF 環境 overlay、CAGR 死守シナリオ候補、Phase D Bootstrap 未実施)** | V0 (defensive) と相補的な mapping variant。S3 + nasdaq_mom63 × M6 で q3 cut を 0.80 → 1.00 に緩和、q0 boost を 1.10 → 1.20 に強化することで「下位 quartile では brush up、上位 quartile では baseline 据置」の boost-only 構成。**S3 overlay tuning 10 variants 中、min(CAGR_IS, CAGR_OOS) > 18% target を達成する唯一の "MaxDD 据置" 案**。canonical split (2021-05-08): CAGR_IS=+18.61%, CAGR_OOS=+19.18%, Sharpe_OOS=0.841, MaxDD=−34.57% (DH-W1 baseline と同値、改善なし), Worst10Y★=+11.02%, P10_5Y▷=+5.22%, IS-OOS gap=−0.57pp (全 overlay 候補中最良), Trades/yr=26.5 (NAV proxy), WFA_CI95_lo=+14.06% (annual CAGR), WFE(50w Sharpe)=1.029。V0 比: CAGR 攻撃力 +1.12〜1.92pp 強い / IS-OOS gap +0.80pp 改善 / 一方 MaxDD 改善は失われる (V0 −28.74% / V7 −34.57%)。**Phase D Bootstrap audit (P_MaxDD, P_Sharpe, P_CAGR) 未実施**のため正式 ADOPT 化保留、ユーザー判断で V0 (MaxDD改善) と V7 (CAGR死守) を運用方針に合わせ選択可。**税後 CAGR (NISA 内非課税 = pretax と同値、課税口座は ×0.8273)**: calendar-year ベース CAGR_OOS = pretax +15.26% / aftertax +12.76% ([aftertax_cagr_20260607.csv](data/signals/expansion/aftertax_cagr_20260607.csv))。 | [S3_OVERLAY_TUNING_REPORT_20260607.md](S3_OVERLAY_TUNING_REPORT_20260607.md) §6.2, [data/signals/expansion/s3_overlay_tuning_20260607.csv](data/signals/expansion/s3_overlay_tuning_20260607.csv), [data/signals/expansion/aftertax_cagr_20260607.csv](data/signals/expansion/aftertax_cagr_20260607.csv), [SIGNAL_EXPANSION_FINAL_DECISION_20260607.md](SIGNAL_EXPANSION_FINAL_DECISION_20260607.md) §3.6 (税後 CAGR), [scripts/tune_s3_overlay_20260607.py](scripts/tune_s3_overlay_20260607.py), [src/integration/build_strategy_with_signal.py](src/integration/build_strategy_with_signal.py) |
| **DH-W1 P2**<br>GOLD100% スリーブ<br>（攻め型投信）<br>DH_W1+CashSleeve | 2026-06-07 | D+tax (ETF 0.10%)（⓽ 税後） | +17.41%† / +16.44%†（**min +16.44%**） | +0.97pp | **+0.875** | −58.53% | N/A | N/A | N/A | ✅ 1.229 | +16.04% | **Shortlisted (投信環境 Active 候補・攻め型, WFA 50窓 PASS)** | DH-W1 baseline (+13.66%) を **+2.78pp 上回る OOS 最高**。Sharpe_OOS +0.875・IS-OOS gap +0.97pp(4戦略中最汎化)。WFA 50窓 CI95_lo=+16.04% (>0 α), WFE=1.229 (β PASS)。代償は **MaxDD −58.53%**(Gold 単独の宿命)。OUT 期資産分析で Gold 年率 +4.07%/Vol 21.5%。**t_p(permutation)/bootstrap 未実施のため正式 Active 昇格は保留**。 | [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md), [cash_sleeve_4strategies_metrics.csv](analysis_cash_sleeve/cash_sleeve_4strategies_metrics.csv), [cash_sleeve_sim.py](analysis_cash_sleeve/cash_sleeve_sim.py) |
| **DH-W1 P7** ⭐推奨<br>GOLD75/BOND25 スリーブ<br>（中庸推奨）<br>DH_W1+CashSleeve | 2026-06-07 | D+tax (ETF 0.10%)（⓽ 税後） | +18.18%† / +14.90%†（**min +14.90%**） | +3.28pp | +0.827 | −48.23% | +9.92% | +8.05% | N/A | ✅ 1.043 | +16.74% | **Shortlisted (投信環境 Active 候補・中庸推奨, WFA 50窓 PASS)** | **P2 と P5 の中間最適**。OOS +14.90% (P5 +13.28 < P7 < P2 +16.44) を確保しつつ Bond 混で MaxDD を Gold 単独(−58.5%)より **−48.23% に緩和**。gap +3.28pp・WFE 1.043(汎化良好)・CI95_lo=+16.74%。Worst10Y★ +9.92%・P10_5Y +8.05%。「Gold 単独ほど DD 悪化させず Gold/Bond 半々より高リターン」の中庸解で**投信環境の第一推奨候補**。t_p/bootstrap 未実施で正式昇格保留。 | [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md), [cash_sleeve_4strategies_yearly.csv](analysis_cash_sleeve/cash_sleeve_4strategies_yearly.csv), [cash_sleeve_sim.py](analysis_cash_sleeve/cash_sleeve_sim.py) |
| **DH-W1 P5**<br>GOLD50/BOND50 スリーブ<br>（守り型投信）<br>DH_W1+CashSleeve | 2026-06-07 | D+tax (ETF 0.10%)（⓽ 税後） | +18.78%† / +13.28%†（**min +13.28%**） | ⚠ +5.50pp | +0.758 | **−35.97%** | +10.08% | +8.09% | N/A | N/A | +17.23% | **Shortlisted (投信環境 Active 候補・守り型, WFA 50窓 PASS)** | **MaxDD をベース DH-W1 同等(−35.97%)に維持**しつつ OUT 期を運用。CI95_lo=+17.23%(4戦略中最高)・P10_5Y +8.09%(最高)・Worst10Y★ +10.08%(最高)。ただし IS-OOS gap +5.50pp(4戦略中最大、汎化やや弱)・Sharpe +0.758・OOS +13.28%(baseline 並)。**守り最優先・DD を増やしたくない向け**。t_p/bootstrap 未実施で正式昇格保留。 | [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md), [cash_sleeve_4strategies_metrics.csv](analysis_cash_sleeve/cash_sleeve_4strategies_metrics.csv), [cash_sleeve_sim.py](analysis_cash_sleeve/cash_sleeve_sim.py) |
| ~~**DH-Z2 AllocTiming**~~<br>~~F10ε+HOLD/OUT~~<br>~~→W1に置換済み~~<br>DH_Dyn+VR+εDB | 2026-06-03 | D（ⓒ 税引前） | N/A / +12.26%（min N/A） | +2.17pp | 0.837 | −38.92% | N/A | N/A | N/A | ✅ 1.058 | N/A | ~~Shortlisted~~ → **Superseded by DH_W1_AsymmHyst (v4.3)** | v4 で採用したが v4.3 で DH-W1 (Asymm Hysteresis) に置換。Z2 (Enter=Exit=0.5 binary) より W1 (Enter 0.7/Exit 0.3 hysteresis) が CAGR +1.40pp / MaxDD -4.35pp / Trades 152→18 で優位。本エントリは履歴保持のため残置、新規参照不可。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §9-v4 (履歴), [src/g22a_dh_alloc_timing_variants.py](src/g22a_dh_alloc_timing_variants.py) |
| **NDX 1x B&H**<br>ベンチマーク<br>Benchmark | 2026-05-21 | N/A | N/A / +10.11%（min N/A） | N/A | 0.540 | −77.9% | −5.67% | N/A | ~0 | N/A | N/A | Shortlisted | ベンチマーク。常に比較対象として残置。MaxDD −77.9% / Worst10Y★ −5.67% で長期保有リスク大。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **P02 DynCPI** ‡<br>CPI ゲート<br>[非標準コスト] | 2026-05-21 | [非標準コスト] | N/A / +19.43%‡（min N/A） | N/A | 0.833‡ | −46.37% | N/A | N/A | N/A | N/A | N/A | Shortlisted | OOS Sharpe 0.833 と Active 候補級だが **[非標準コスト]**（P-series 独自コストモデル, §17）。直接比較不可。再評価には標準コスト下での再計算が必要。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **P05 HYCPI** ‡<br>HY+CPI ゲート<br>[非標準コスト] | 2026-05-21 | [非標準コスト] | N/A / +15.65%‡（min N/A） | N/A | 0.667‡ | −44.98% | N/A | N/A | N/A | N/A | N/A | Shortlisted | Worst5Y +6.04% で Worst5Y 観点では最強。ただし **[非標準コスト]** につき他系統と直接比較不可。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **P01 DynHY** ‡<br>HY スプレッドゲート<br>[非標準コスト] | 2026-05-21 | [非標準コスト] | N/A / +19.92%‡（min N/A） | N/A | 0.829‡ | −42.85% | N/A | N/A | N/A | N/A | N/A | Shortlisted | DSR 観点で候補級だが **[非標準コスト]**（§17）。MaxDD −42.85% は本表で最小。標準コスト下での再評価で残存可否を判定する。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **DH_Dyn+LT2-N750**<br>ETF版 k=0.5<br>（CFD不可環境代替）<br>DH_Dyn+LT | 2026-05-21 | D（ⓒ 税引前） | N/A / +18.87%（min N/A） | N/A | 0.777 | −44.76% | N/A | N/A | N/A | N/A | N/A | Shortlisted | Active (S2+LT2) の CFD 抜きバージョン。CFD 不可環境（NISA, 法人税効率優先など）の代替候補。Sharpe_OOS 0.777 は DH Dyn 単体 0.646 から大幅改善。 | [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md), [src/b1_s2_lt2.py](src/b1_s2_lt2.py), [src/long_cycle_signal.py](src/long_cycle_signal.py) |
| **S2+LT2-N1500**<br>k=0.5, Sharpe★ 0.885<br>gap −0.05pp<br>CFDLev+LT | 2026-05-22 | D（ⓒ 税引前） | +30.79% / +30.84%（**min +30.79%**） | −0.05pp | 0.885 | −63.37% | +16.60% | N/A | N/A | N/A | N/A | Shortlisted | Sharpe_OOS +0.885（本プロジェクト最高値）/ IS-OOS gap −0.05pp（OOS が IS 超え）は最強の汎化性。CAGR_OOS −0.32pp / MaxDD −3.92pp / Worst10Y★ −1.50pp でリスク指標が N=750 に劣後し、コミット 8503200 で Active から N=750 に差し戻し。代替シナリオとして Shortlisted 保持。 | [b6_s2_lt2_N_sweep_results.csv](b6_s2_lt2_N_sweep_results.csv), [B6_S2_LT2_N_SWEEP_2026-05-22.md](B6_S2_LT2_N_SWEEP_2026-05-22.md), [src/b1_s2_lt2.py](src/b1_s2_lt2.py) |
| **S2+LT4-N750 k0.7**<br>B3 最良 config<br>CFDLev+LT | 2026-05-22 | D（ⓒ 税引前） | +31.10% / +30.50%（**min +30.50%**） | +0.60pp | 0.850 | N/A | N/A | N/A | N/A | N/A | N/A | Shortlisted | B3 LT4 N×k_lt スイープ 9/9 が S2 単体（+0.770）超え、最良が CAGR_OOS +30.50% / Sharpe +0.850 / IS-OOS gap +0.60pp と Active 級。LT2-N1500 比で Sharpe −0.035, gap +0.65pp と総合僅差敗北。 | [b3_s2_lt4_sweep_results.csv](b3_s2_lt4_sweep_results.csv) |
| **S2+LT6-N500 k0.7**<br>B4 最良（gap⚠⚠）<br>CFDLev+LT | 2026-05-22 | D（ⓒ 税引前） | +33.08% / +27.35%（**min +27.35%**） | ⚠⚠ +5.73pp | 0.820 | −54.38% | +17.06% | N/A | N/A | N/A | N/A | Shortlisted | B4 LT6 N×k_lt スイープ 9/9 が S2 単体超え。最良値は MaxDD −54.38% / Worst10Y★ +17.06% と健闘するが IS-OOS gap +5.73pp で過剰適合傾向。N=750/k=0.7 変種（Sharpe +0.801, gap +4.13pp）のほうがロバスト性で勝る。 | [b4_s2_lt6_sweep_results.csv](b4_s2_lt6_sweep_results.csv) |
| **S2+LT7 k0.5**<br>dual MA cross<br>N_s=750, N_l=1250<br>CFDLev+LT | 2026-05-22 | D（ⓒ 税引前） | N/A / +29.42%（min N/A） | N/A | 0.814 | −60.39% | N/A | N/A | N/A | N/A | N/A | Shortlisted | B5 LT7 k_lt スイープ 3/3 が S2 単体超え。dual MA cross 構造を導入したが Sharpe では LT2/LT4 に劣後、N=1500 比で +0.071 差。複雑度に見合う優位性なし、ロバスト性検証ためのスイープ規模（3 config）も小さく要注意。 | [b5_s2_lt7_sweep_results.csv](b5_s2_lt7_sweep_results.csv) |
| **Hybrid S2+P05** ‡<br>S2+LT2(70%)+P05(30%)<br>[非標準コスト]<br>Hybrid | 2026-05-22 | [非標準コスト] | N/A / +30.51%‡（min N/A） | N/A | 0.881‡ | −62.1% | N/A | N/A | N/A | N/A | N/A | Shortlisted | E2 検証（3基準）で 2/3 改善（MaxDD +1.3pp, IS-OOS gap 維持）。Sharpe_OOS 0.881 < 0.885 で基準(i)不合格。§1.3 により Active 昇格不可（Shortlisted 上限）。S2+LT2-N1500 単独継続を推奨。 | [E2_HYBRID_70_30_2026-05-22.md](E2_HYBRID_70_30_2026-05-22.md), [e2_hybrid_70_30_results.csv](e2_hybrid_70_30_results.csv), [src/e2_hybrid_70_30.py](src/e2_hybrid_70_30.py) |

> ‡ FULL期間 Sharpe は未計算。P4系 (P01/P02/P05) の OOS 期間 (2021/5〜2026/3) Sharpe を記載。Hybrid_S2LT2_P05_70_30 の ‡ は §1.3 非標準コスト継承による参考値扱い。

---

## §3 Rejected — 廃止・棄却ログ

> 「ベスト」「採用候補」と提示してはいけない戦略群。明確な廃止理由とともに記録。
> 同じ実験を再度行わないための備忘録としての性質を持つ。

### §3.1 旧ベスト戦略（CURRENT_BEST_STRATEGY.md ブラックリスト由来）

| 戦略（ID / 構成） | Date | Cost | CAGR ⓒ IS / OOS（min） | IS-OOS gap ⓒ | Sharpe ⓒ | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ | WFE ⓞ | CI95 ⓡ | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DH Dyn ScA_v1**<br>Scenario A旧推奨<br>DH_Dyn | 2026-05-12 | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | Scenario A はコスト過少推計（SOFR financing drag 未反映 等）。Scenario D 補正で CAGR 30.81% → 22.50%, Sharpe 1.299 → 0.993, MaxDD −31.40% → −45.08% に下方修正。Scenario A 単体での値は採用しない。 | [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md), [DELAY_PRODUCT_COMPARISON_REPORT_2026-05-12.md](DELAY_PRODUCT_COMPARISON_REPORT_2026-05-12.md) |
| **DH Dyn ScD_v1**<br>Scenario D旧推奨<br>DH_Dyn | 2026-05-21 | D | +14.88% | 0.993 (FULL) | −45.08% | Rejected | 2026-05-12〜2026-05-21 の暫定ベスト。S2_VZGated が CAGR_OOS +27.57% / Sharpe_OOS 0.769 で上回ることを確認し降格。なお派生形（DH_Dyn_2x3x_A）として Shortlisted には残存（§2）。 | [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md), [STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) |
| **DH Dyn th020**<br>threshold=0.20<br>DH_Dyn | 2026-04-21 | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 閾値 0.15 が全指標で優位と確認。閾値スイープで決着済み。 | [THRESHOLD_SWEEP_REPORT_2026-04-20.md](THRESHOLD_SWEEP_REPORT_2026-04-20.md), [THRESHOLD_SWEEP_A_REPORT_2026-04-21.md](THRESHOLD_SWEEP_A_REPORT_2026-04-21.md), [threshold_sweep_results.csv](threshold_sweep_results.csv) |
| **DH-T4** ❌<br>vz065+lmax5.5+F10ε<br>ETF違反・v4で破棄 | 2026-06-03 | D (moderate) | +9.44% | 0.671 | −37.63% | **Rejected (2026-06-03)** | **却下理由**: `lev_mod_065` を `lev_raw × 3` に乗算して TQQQ position を 0〜100% に連続スケール = **ETF (TQQQ/TMF/2036) の物理的「保有/非保有」制約に違反**。「部分 TQQQ 保有 (実効レバを 0〜3x に内部 scaling)」は CFD/SPOT 操作であり ETF 商品で禁止。`L_s2_lmax5p5` cap (CFD-style 動的レバ上限) も ETF では概念違反。代替として DH-Z2 (binary HOLD/OUT + F10 配分) を §2 Shortlisted に採用。一次根拠 (g21*) は履歴保持のため push 残置するが新規参照不可。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §0 v4 主要変更点 A, §9-v3 (破棄マーク), [src/g21a_dh_improved_variants.py](src/g21a_dh_improved_variants.py), [g21b_dh_improved_9metrics.csv](g21b_dh_improved_9metrics.csv) |
| **vz065+l7+AH** ❌<br>AsymmHyst Enter≥0.7<br>min 3軸全敗 WFE1.554 | 2026-06-05 | D (moderate 0.05%) | +26.53% | 0.954 | −64.89% | **Rejected (v4.5, 2026-06-05)** | v4.4 で「採用」と判定したが v4.5 で min(IS, OOS) ルール導入により **3 軸全敗で棄却**: min CAGR +18.90% < REF +20.23% (-1.33pp)、Worst10Y +8.45% < REF +9.96% (-1.51pp)、P10_5Y +1.73% < REF +4.05% (-2.32pp)。**WFE 1.554 (上限 2.0 近接) は regime luck 強疑い**。Trades 半減 (105→56) のコスト優位はあるが CAGR/防御指標劣化を補えない。OOS 単独評価では魅力的だが「OOS 期間固有 fit」の典型例。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §6-4 (v4.4 検証 + v4.5 棄却判定), §7-2 min ルール, [src/g26_new_asymmetric_variants.py](src/g26_new_asymmetric_variants.py), [g26_new_asymmetric_metrics.csv](g26_new_asymmetric_metrics.csv) |
| **vz065+l7+AT** ❌<br>ContinuousTilt<br>WFE≈2.0 regime luck | 2026-06-05 | D (moderate 0.05%) | +27.91% | 1.066 | −63.89% | **Rejected (v4.5)** | min CAGR +16.38% (IS) で REF +20.23% より -3.85pp 大幅劣後。Worst10Y +6.62%、P10_5Y +1.58% も防御性能大幅劣化。**WFE 1.988 ≈ 2.0 上限** で regime luck が顕著。連続 scaling で smooth だが過剰 reduction の罠。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §6-4, [src/g26_new_asymmetric_variants.py](src/g26_new_asymmetric_variants.py) |
| **vz065+l7+HL** ❌<br>HystLite OUT30%残し<br>AHより劣後 | 2026-06-05 | D (moderate 0.05%) | +25.42% | 0.923 | −64.81% | **Rejected (v4.5)** | CAGR_OOS +25.42% で AH に劣る、Trades 106 同等 (cost 削減効果なし)、AH に対する明確な優位なし。min CAGR +19.86% (IS) でも REF +20.23% を僅か下回る。 | [STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md](STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md) §6-4, [src/g26_new_asymmetric_variants.py](src/g26_new_asymmetric_variants.py) |

### §3.2 アンサンブル系実験（Ens2 / Voting）

| 戦略（ID / 構成） | Date | Cost | CAGR ⓒ IS / OOS（min） | IS-OOS gap ⓒ | Sharpe ⓒ | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ | WFE ⓞ | CI95 ⓡ | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ens2 AsymSlope**<br>max_lev=1.0<br>EnsembleVoting | 2026-04-21 | A | +10.52% | 0.479 | −48.99% | Rejected | 2026-04-21 に `DH Dyn 2x3x [A] 閾値0.15` に置換され廃止。Scenario A 値であり Scenario D 補正後はさらに悪化見込み。 | [strategy_comparison_results.csv](strategy_comparison_results.csv), [ens2_comparison_results.csv](ens2_comparison_results.csv), [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |
| **Ens2 AsymSlope lev30**<br>max_lev=3.0 高レバ<br>EnsembleVoting | N/A | A（未補正） | N/A | N/A | −77.7% | Rejected | 旧未補正 CAGR 54.94% / Worst5Y −14.4% は Scenario A 未補正の過剰適合値。IS-OOS gap 大確実で Scenario D 補正後は実用水準に達さない。MaxDD −77% も実運用不可水準。 | [ens2_comparison_results.csv](ens2_comparison_results.csv) |
| **Ens2 SlopeTV lev30**<br>max_lev=3.0 高レバ<br>EnsembleVoting | N/A | A（未補正） | N/A | N/A | −78.5% | Rejected | 同上。Scenario A 未補正の値であり、Scenario D 補正で実用水準に達しないことが類似戦略から推定可能。 | [ens2_comparison_results.csv](ens2_comparison_results.csv) |
| **Ens2 SlopeTV**<br>max_lev=1.0<br>EnsembleVoting | 2026-04-21 | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 2026-04-21 に `DH Dyn 2x3x [A] 閾値0.15` に置換され廃止。 | [ens2_comparison_results.csv](ens2_comparison_results.csv), [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |
| **MajorityVote P2**<br>P2 シグナル統合<br>EnsembleVoting | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 実験ログのみ。最終結論に組み込まれず採用されなかった。 | [majority_vote_p2_results.csv](majority_vote_p2_results.csv), [majority_vote_signals.csv](majority_vote_signals.csv), [vote_ratio_continuous_results.csv](vote_ratio_continuous_results.csv) |

### §3.3 CFD動的レバレッジ派生系（S2/S4 等の落選候補）

| 戦略（ID / 構成） | Date | Cost | CAGR ⓒ IS / OOS（min） | IS-OOS gap ⓒ | Sharpe ⓒ | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ | WFE ⓞ | CI95 ⓡ | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CFD lmax Sweep** ❌<br>l_max 落選群<br>Sweep | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | l_max=7.0 が最適と確認。それ以外の l_max（5.0, 6.0, 8.0 等）は同 Sharpe / 高 MaxDD で劣後。 | [A6_LMAX_SWEEP_2026-05-21.md](A6_LMAX_SWEEP_2026-05-21.md), [a6_lmax_sweep_results.csv](a6_lmax_sweep_results.csv) |
| **CFD nvol Sweep** ❌<br>n_vol 落選群<br>Sweep | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | n=20 が最適と確認。n=10, 30, 60 は Sharpe で劣後。 | [A1_NVOL_SWEEP_2026-05-21.md](A1_NVOL_SWEEP_2026-05-21.md), [a1_nvol_sweep_results.csv](a1_nvol_sweep_results.csv) |
| **B2 KLT Sweep** ❌<br>LT2 k 落選群<br>LT / Sweep | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | k=0.5（modeB）が最適と確認。k=0.25, 0.75, 1.0 は IS-OOS gap または Sharpe で劣後。 | [B2_KLT_SWEEP_2026-05-21.md](B2_KLT_SWEEP_2026-05-21.md), [b2_klt_sweep_results.csv](b2_klt_sweep_results.csv) |
| **C1 HY Gate** ❌<br>HY ゲート単体<br>TimingGate | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | HY ゲートは S2+LT2 に上乗せしても改善せず。標準コストでは効果無し。 | [C1_HY_GATE_2026-05-21.md](C1_HY_GATE_2026-05-21.md), [c1_hy_gate_results.csv](c1_hy_gate_results.csv) |
| **F1 Alloc Sweep** ❌<br>配分比率落選<br>AllocOpt | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 既存 DH Dyn [A] 配分が最適。他配分は CAGR or Sharpe で劣後。 | [F1_ALLOC_SWEEP_2026-05-21.md](F1_ALLOC_SWEEP_2026-05-21.md), [f1_alloc_sweep_results.csv](f1_alloc_sweep_results.csv) |
| **E1 Ens S2** ❌<br>S2+S4 アンサンブル<br>Ensemble/CFD | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | S2+LT2 単体に対する優位性なし。アンサンブル化のメリット観測されず。 | [E1_ENSEMBLE_2026-05-21.md](E1_ENSEMBLE_2026-05-21.md), [e1_ensemble_results.csv](e1_ensemble_results.csv) |
| **D1 OOS Boundary** ❌<br>境界変動テスト落選<br>Robustness | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | OOS 境界 ±1 年シフトでも S2+LT2 がロバスト最高。他候補は境界変動に対して脆弱。 | [D1_OOS_BOUNDARY_2026-05-21.md](D1_OOS_BOUNDARY_2026-05-21.md), [d1_oos_boundary_results.csv](d1_oos_boundary_results.csv) |
| **Discrete Lev** ❌<br>2x/3x/5x ステップ<br>CFDLev/離散 | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 連続レバ（step=0.5）が同等以上の性能。離散化のメリット観測されず。 | [discrete_leverage_results.csv](discrete_leverage_results.csv) |
| **まるごとLev** ❌<br>単純倍率<br>CFDLev | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 単純レバ倍はベンチマーク以下。DH Dyn シグナル統合が必須。 | [marugoto_leverage_results.csv](marugoto_leverage_results.csv) |
| **Lev 2x/3x Baseline** ❌<br>DH Dynなし単純<br>DH_Dyn/単純 | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | DH Dyn シグナルなしの単純 2x/3x ローテーション。CAGR / Sharpe ともに DH Dyn [A] に劣後。 | [lev2x3x_results.csv](lev2x3x_results.csv) |
| **P4 Composite** ❌<br>SOFR×Vol×Mom<br>CFDLev/MultiFactor | 2026-05-21 | D | +13.69% (best) | +0.647 (best) | −47.74% (best) | Rejected | 24 config 全て S2_VZGated ベースライン（Sharpe +0.770）を下回る。三因子乗算の論理 AND 効果が過剰デレバをもたらしリターン低下。IS-OOS gap +8〜15 pp と大きく過剰適合傾向。dynamic_leverage_strategies.py で「本命」と記載されていたが実証的に棄却。 | [P4_COMPOSITE_SWEEP_2026-05-21.md](P4_COMPOSITE_SWEEP_2026-05-21.md), [p4_composite_sweep_results.csv](p4_composite_sweep_results.csv) |
| **S1 Conviction** ❌<br>A2スコア直接Lev変換<br>CFDLev/A2Conv | 2026-05-21 | D | +22.43% (best) | +0.645 (best) | −64.52% (best) | Rejected | 16 config 全て S2_VZGated ベースラインを下回る。IS-OOS gap +21 pp と致命的な過剰適合。target_vol パラメータは NASDAQ 通常ボラレンジ（σ≈13.6%）で全値飽和（dead parameter）。A2 raw スコア自体が IS 最適化バイアスを持つため直接レバ変換は不適。 | [S1_CONVICTION_SWEEP_2026-05-21.md](S1_CONVICTION_SWEEP_2026-05-21.md), [s1_conviction_sweep_results.csv](s1_conviction_sweep_results.csv) |
| **S2 tv Sweep** ❌<br>target_vol落選<br>Sweep | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | tv=0.80 が Sharpe_OOS 最高（+0.770）と確認。tv=1.00 は CAGR_OOS +28.42% と高いが IS-OOS gap +8.84 pp で過剰適合。tv<0.40 は vol-targeting 機能するが CAGR_OOS が半減。 | [A2_TV_SWEEP_2026-05-21.md](A2_TV_SWEEP_2026-05-21.md), [a2_tv_sweep_results.csv](a2_tv_sweep_results.csv) |
| **S2 kvz Sweep** ❌<br>k_vz dead param<br>Sweep | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | k_vz=0.10〜0.70 で Sharpe_OOS は +0.769〜+0.775 と極めてフラット。VIX ゲートの感度係数は dead parameter に近い。現行 k_vz=0.30 は安定中心値として妥当。 | [A3_KVZ_SWEEP_2026-05-21.md](A3_KVZ_SWEEP_2026-05-21.md), [a3_kvz_sweep_results.csv](a3_kvz_sweep_results.csv) |
| **S2 gatemin Sweep**<br>VIX下限スイープ<br>Sweep | 2026-05-21 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | gate_min=0.20/0.35 が Sharpe_OOS +0.775（現行 0.50 の +0.770 を僅かに上回る）。VIX ゲートなし（gate_min=1.00）は Sharpe +0.759 と明確に劣後し VIX ゲートの有効性を確認。差分 +0.005 は微小のため現行設定を維持。S2+LT2 への影響は別途要確認。 | [A4_GATEMIN_SWEEP_2026-05-21.md](A4_GATEMIN_SWEEP_2026-05-21.md), [a4_gatemin_sweep_results.csv](a4_gatemin_sweep_results.csv) |
| **P1 SOFR Adaptive** ❌<br>SOFR→lmax動的調整<br>CFDLev/SOFR | 2026-05-22 | D | +19.48% (best) | +0.691 (best) | N/A | Rejected | 18/18 が S2 単体ベースライン（Sharpe +0.770）を下回る。最良 (sofr_high=0.10, l_max=6) でも IS-OOS gap +9.93pp。SOFR 適応化はゼロ金利期で過剰デレバ、リターン低下を招く。 | [p1_sofr_sweep_results.csv](p1_sofr_sweep_results.csv) |
| **P3 Momentum Lev** ❌<br>m×k スイープ<br>CFDLev/Mom | 2026-05-22 | D | +18.87% (best) | +0.649 (best) | N/A | Rejected | 16/16 が S2 単体を下回る。最良 (m=60, k=0.5) で IS-OOS gap +14.66pp と過剰適合大。DH Dyn シグナルが既にモメンタム要素を内包しており、モメンタム重畳は冗長。 | [p3_momentum_sweep_results.csv](p3_momentum_sweep_results.csv) |
| **P5 Kelly Sizing** ❌<br>μ/σ→Kelly Lev<br>CFDLev/Kelly | 2026-05-22 | D | +21.08% (best) | +0.642 (best) | N/A | Rejected | 12/12 が S2 単体を下回る。最良 (safety=1.0, mu_w=252, sig_w=30) で IS-OOS gap +5.40pp。Kelly は分布安定仮定が NASDAQ 株式 OOS では破綻、CFD 動的レバとして不適。 | [p5_kelly_sweep_results.csv](p5_kelly_sweep_results.csv) |
| **S3 Decomposed A2** ❌<br>A2→L_t rewire<br>致命的gap+22pp | 2026-05-22 | D | +10.13% (best) | +0.443 (best) | N/A | Rejected (hard) | 全 config が S2 単体を大幅下回る。最良で IS-OOS gap **+22.39pp** と致命的過剰適合。A2 raw 構成要素を直接レバ生成に流すと IS 最適化バイアスが直撃するため、構造的に再現不可。**同類実験を二度行わないこと**。 | [s3_decomposed_sweep_results.csv](s3_decomposed_sweep_results.csv) |
| **S4 RelVol Gated** ❌<br>36 config全てS2未満<br>CFDLev/RelVol | 2026-05-22 | D | +27.69% (best) | +0.750 (best) | N/A | Rejected | 36/36 が S2 単体（+0.770）を下回る。最良 (l_base=6, k_rel=3.0, rel_th=1.0, HL=10/60) でも IS-OOS gap +9.26pp。RelVol ゲートは過剰デレバ、Sharpe 改善寄与なし。H1 と同等。 | [h1_s4_param_sweep_results.csv](h1_s4_param_sweep_results.csv) |
| **B6 LT2 N Sweep** ❌<br>N=500-1250落選<br>LT/Sweep | 2026-05-22 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | N=1500 が Sharpe_OOS +0.885 / IS-OOS gap −0.05pp で最良と確定（Active 昇格）。N=750 は Shortlisted に残置、N=500/600/1000/1250 は Sharpe で劣後し棄却。N=500 (Sharpe +0.830, gap +7.41pp), N=600 (+0.803, +6.79pp), N=1000 (+0.791, +1.63pp), N=1250 (+0.771, +3.72pp)。 | [b6_s2_lt2_N_sweep_results.csv](b6_s2_lt2_N_sweep_results.csv) |
| **B3 LT4 Sweep** ❌<br>8 config落選<br>LT/Sweep | 2026-05-22 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | N=750/k=0.7 が最良で Shortlisted へ昇格。他 8 config（N=500/1000 × k=0.3/0.5/0.7 + N=750 × k=0.3/0.5）は Sharpe で劣後。 | [b3_s2_lt4_sweep_results.csv](b3_s2_lt4_sweep_results.csv) |
| **B4 LT6 Sweep** ❌<br>8 config落選<br>LT/Sweep | 2026-05-22 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | N=500/k=0.7 が CAGR_OOS 最大で Shortlisted。他 8 config は劣後（N=750/k=0.7 は Sharpe +0.801 / gap +4.13pp で要注意系として参考）。 | [b4_s2_lt6_sweep_results.csv](b4_s2_lt6_sweep_results.csv) |
| **B5 LT7 Sweep** ❌<br>2 config落選<br>LT/Sweep | 2026-05-22 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | k_lt=0.5 が最良で Shortlisted。他 2 config は Sharpe で劣後、スイープ規模小（3 config）のためロバスト性証拠は限定的。 | [b5_s2_lt7_sweep_results.csv](b5_s2_lt7_sweep_results.csv) |

### §3.4 シグナル系実験（VIX / Regime / SOXL / DD等）

| 戦略（ID / 構成） | Date | Cost | CAGR ⓒ IS / OOS（min） | IS-OOS gap ⓒ | Sharpe ⓒ | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ | WFE ⓞ | CI95 ⓡ | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A2 VIX MD60** ❌<br>VIX+MD60 ScA<br>VIX/Signal | 2026-04-21 | A | +14.85% | 0.998 (FULL) | −44.09% | Rejected | Scenario A の値であり Scenario D 補正で大幅劣化見込み。DH Dyn [A] Approach A に統合済み（VIX_MR コンポーネントとして）。単体は不要。 | [strategy_comparison_results.csv](strategy_comparison_results.csv) |
| **DD Only -18/92** ❌<br>DDルール単体<br>DD/Signal | 2026-04-21 | A | +14.98% | 0.748 (FULL) | −72.88% | Rejected | DD ルール単体では MaxDD −72.88% で実運用不可。DH Dyn の構成要素として統合済み。 | [strategy_comparison_results.csv](strategy_comparison_results.csv) |
| **DD+VT+VolSpike** ❌<br>R4トップ1(ScA)<br>VolSpike/DD | 2026-04-21 | A | N/A | 0.902 (FULL) | −58.4% | Rejected | R4 個別実験でトップだが最終結論ではない（CURRENT_BEST_STRATEGY.md ブラックリスト記載）。Scenario A 値であり Scenario D 補正後の実力未検証。 | [R4_results.csv](R4_results.csv), [R4_RESULTS_SUMMARY_2026-02-06.md](R4_RESULTS_SUMMARY_2026-02-06.md), [ens2_comparison_results.csv](ens2_comparison_results.csv), [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) |
| **DD+VT Baseline** ❌<br>DD+VT(25%) R4base<br>DD/VT | N/A | A | N/A | 0.861 (FULL) | −61.9% | Rejected | R4 のベースライン。後続 DH Dyn 系に統合・置換済み。 | [ens2_comparison_results.csv](ens2_comparison_results.csv) |
| **DynHybrid Static**<br>50/25/25 静的配分<br>DH_Dyn派生 | 2026-04-21 | A | +13.27% | 0.867 (FULL) | −22.92% | Rejected | 静的配分のため動的環境（金利急騰時等）に弱い。Sharpe 1.198 は高いが CAGR が低い。動的版（DH Dyn [A]）に置換済み。 | [strategy_comparison_results.csv](strategy_comparison_results.csv) |
| **DynHybrid LevVix**<br>lev+vix 中間版<br>VIX/DH_Dyn | 2026-04-21 | A | +15.76% | 0.769 (FULL) | −29.42% | Rejected | 中間版。DH Dyn [A] Approach A に統合・置換済み。 | [strategy_comparison_results.csv](strategy_comparison_results.csv) |
| **Regime Strategy** ❌<br>HMM/状態推定<br>Regime | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | レジーム推定戦略は実験段階で R4 を超えず、後続 DH Dyn に吸収されることなく終了。実装も維持されていない。 | [regime_strategy_results.csv](regime_strategy_results.csv), [regime_analysis_stats.csv](regime_analysis_stats.csv), [regime_vs_r4_comparison.csv](regime_vs_r4_comparison.csv), [REGIME_ANALYSIS_REPORT_2026-04-04.md](REGIME_ANALYSIS_REPORT_2026-04-04.md) |
| **SOXL Addition** ❌<br>SOXL追加実験<br>Portfolio/SOXL | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | SOXL 追加は分散効果なく、ボラ増加のデメリットが上回り棄却。 | [soxl_addition_results.csv](soxl_addition_results.csv) |
| **External Signal Sweep**<br>HY/CPI ISスイープ<br>TimingGate | N/A | [非標準コスト] | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | P-series（P01〜P05）に発展し Shortlisted に残存したが、当該スイープ単体としては最終採用なし。 | [external_signal_results.csv](external_signal_results.csv), [external_is_param_sweep.csv](external_is_param_sweep.csv), [external_oos_results.csv](external_oos_results.csv) |
| **Bond Variant Tests** ❌<br>TMF パラムスイープ<br>Bond/派生 | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | TMF 既定パラメータが最適と確認。他変種は採用に至らず。 | [bond_variant_results.csv](bond_variant_results.csv), [bond_model_annual_comparison.csv](bond_model_annual_comparison.csv), [bond_model_grid_results.csv](bond_model_grid_results.csv), [tmf_validation_results.csv](tmf_validation_results.csv) |
| **Gold Signal Research** ❌<br>金シグナル研究<br>Gold/Signal | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | Gold シグナル単独では有意な改善観測されず。Gold は配分要素として残置するがシグナル化しない。 | [research_gold_signals.csv](research_gold_signals.csv) |

### §3.5 ポートフォリオ構造系実験（リバランス頻度・部分リバランス・分散化等）

| 戦略（ID / 構成） | Date | Cost | CAGR ⓒ IS / OOS（min） | IS-OOS gap ⓒ | Sharpe ⓒ | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ | WFE ⓞ | CI95 ⓡ | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Rebalance Freq Sweep** ❌<br>週/月/四半期<br>頻度スイープ | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | DH Dyn シグナルに基づく動的リバランス（年27回）が最適と確認。固定頻度は劣後。 | [rebalance_frequency_results.csv](rebalance_frequency_results.csv) |
| **Partial Rebalance** ❌<br>許容バンド方式<br>頻度/部分 | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 取引コスト削減効果はあるが、シグナル追従が遅延しリターン低下。最終的に維持されず。 | [partial_rebalance_results.csv](partial_rebalance_results.csv) |
| **Portfolio Divers.** ❌<br>多資産追加実験<br>Portfolio/分散 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 多資産追加（コモディティ・REIT 等）で分散効果限定的。CAGR 低下と相殺。 | [portfolio_diversification_results.csv](portfolio_diversification_results.csv) |
| **Overfitting Valid.**<br>過剰適合検証<br>Robustness | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 検証実験。戦略採否そのものは別途決定。本実験単独での戦略は存在せず参照用ログ。 | [overfitting_validation_results.csv](overfitting_validation_results.csv) |
| **Hybrid Drafts**<br>早期試行ログ<br>DH_Dyn 草案 | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | DH Dyn [A] 確立前の試行錯誤ログ。確定版（[A] 閾値 0.15）に吸収済み。 | [hybrid_strategy_results.csv](hybrid_strategy_results.csv), [dynamic_portfolio_results.csv](dynamic_portfolio_results.csv) |
| **Improvement R1-R4**<br>改善イテレーション<br>反復改善ログ | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | R1〜R4 各ラウンドの試行ログ。R4 経由で DH Dyn [A] に収束済み。 | [improvement_results.csv](improvement_results.csv), [improvement_results_r2.csv](improvement_results_r2.csv), [improvement_results_r3.csv](improvement_results_r3.csv), [improvement_results_r4.csv](improvement_results_r4.csv), [improvement_results_a_vix.csv](improvement_results_a_vix.csv), [improvement_results_wf_vix.csv](improvement_results_wf_vix.csv), [improvement_results_next.csv](improvement_results_next.csv) |
| **Step2 Variants**<br>baseline/ma5/macd<br>/full/wf 検証群 | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | Step1〜Step3 検証パイプラインの中間生成物。最終結論には組み込まれず。 | [step1_worst10y_results.csv](step1_worst10y_results.csv), [step2_baseline.csv](step2_baseline.csv), [step2_partB_ma5.csv](step2_partB_ma5.csv), [step2_partC_ma5_macd.csv](step2_partC_ma5_macd.csv), [step2_partD_full.csv](step2_partD_full.csv), [step2_partE_wf.csv](step2_partE_wf.csv), [step2_static_cagr25_results.csv](step2_static_cagr25_results.csv), [step3_dynamic_cagr25_results.csv](step3_dynamic_cagr25_results.csv) |
| **Realistic Product**<br>DELAY違い等<br>プロダクト検証 | N/A | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | DELAY=2 営業日が最適と確認。他 DELAY は採用見送り。 | [realistic_product_results.csv](realistic_product_results.csv), [delay_product_comparison_results.csv](delay_product_comparison_results.csv), [delay_robust_results.csv](delay_robust_results.csv), [DELAY_PRODUCT_COMPARISON_REPORT_2026-05-12.md](DELAY_PRODUCT_COMPARISON_REPORT_2026-05-12.md) |
| **Threshold Tax Sens.**<br>閾値×税率感度<br>税務感度分析 | 2026-05-12 | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 感度分析であり戦略採否ではない。閾値 0.15 / 税率 20.315% 想定で本表完結。 | [THRESHOLD_TAX_SENSITIVITY_REPORT_2026-05-12.md](THRESHOLD_TAX_SENSITIVITY_REPORT_2026-05-12.md), [threshold_tax_sensitivity_results.csv](threshold_tax_sensitivity_results.csv) |
| **Financing Cost Vars.**<br>SOFR倍率変動<br>コスト感度 | N/A | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | コスト感度実験。最終的に Scenario D が標準。 | [financing_cost_results.csv](financing_cost_results.csv) |
| **Approach A Sweep**<br>A内部パラメータ<br>DH_Dyn/Sweep | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | Approach A 確定版（閾値0.15）に至る過程の探索ログ。 | [approach_a_sweep_results.csv](approach_a_sweep_results.csv), [APPROACH_A_PROPOSAL_2026-04-20.md](APPROACH_A_PROPOSAL_2026-04-20.md) |
| **Phase1 Opt T1-T3**<br>Tier1-3最適化ログ<br>最適化/過去 | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 最適化過程のログ。最終結論に吸収済み。 | [opt_phase1_tier1_results.csv](opt_phase1_tier1_results.csv), [opt_phase1_tier2_results.csv](opt_phase1_tier2_results.csv), [opt_phase1_tier3_results.csv](opt_phase1_tier3_results.csv), [opt_phase1_wf_results.csv](opt_phase1_wf_results.csv), [opt_phase2_results.csv](opt_phase2_results.csv), [opt_phase3_results.csv](opt_phase3_results.csv) |
| **LT Sweep Variants** ❌<br>N/mode 落選変種<br>LT/Sweep | N/A | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | LT2-N750-k0.5-modeB が最終確定（B1）。他 LT 変種（N=500, 1000 / modeA, modeC 等）は劣後。 | [lt_sweep_results.csv](lt_sweep_results.csv), [lt_extended_results.csv](lt_extended_results.csv), [lt_combined_results.csv](lt_combined_results.csv) |
| **Grid Search Legacy**<br>旧グリッド最適化<br>最適化/過去 | N/A | A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 旧グリッド最適化。R4 / Phase 系に置換済み。 | [grid_search_results.csv](grid_search_results.csv) |
| **Factcheck Sens.**<br>再現性検証ログ<br>検証/監査 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 過去結果の再現性検証ログ。戦略採否ではない。 | [factcheck_sensitivity_results.csv](factcheck_sensitivity_results.csv), [factcheck_log.txt](factcheck_log.txt) |
| **TQQQ Verification**<br>TQQQ再現性確認<br>検証/Product | N/A | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | TQQQ シミュレータの一致確認用。戦略採否ではない。 | [tqqq_verification_results.csv](tqqq_verification_results.csv) |
| **Validation Crisis/Full/OOS**<br>期間別検証ログ<br>検証/期間 | N/A | D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 期間別検証ログ。Active 戦略の補強根拠として参照、独立戦略ではない。 | [validation_crisis.csv](validation_crisis.csv), [validation_full.csv](validation_full.csv), [validation_oos.csv](validation_oos.csv) |
| **Lev Bin Analysis V1-V4**<br>CFDLev最適化根拠<br>Lev分析 | 2026-04-20 | A/D | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Rejected | 分析レポート。CFD レバレッジ最適化の根拠資料として参照されるが、独立戦略ではない。 | [LEVERAGE_BIN_ANALYSIS_2026-03-19.md](LEVERAGE_BIN_ANALYSIS_2026-03-19.md), [LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md](LEVERAGE_BIN_ANALYSIS_V2_2026-03-19.md), [LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md](LEVERAGE_BIN_ANALYSIS_V3_2026-03-24.md), [LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md](LEVERAGE_BIN_ANALYSIS_V4_2026-04-20.md) |

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

| 戦略（ID / 構成） | Date | Cost | CAGR ⓒ IS / OOS（min） | IS-OOS gap ⓒ | Sharpe ⓒ | MaxDD ⓒ | Worst10Y★ ⓒ | P10 ⓒ 5Y | Trade ⓞ | WFE ⓞ | CI95 ⓡ | Status | Decision Reason | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _(なし)_ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

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

### SignalOverlay / DH_W1派生（信号注入による defensive レバ調整）
- **Shortlisted (2026-06-07 追加, ETF 環境 overlay 採用可)**: DH_W1_AsymmHyst + nasdaq_mom63 × M6 defensive overlay ⭐ → §2
- 着想: DH-W1 (Asymm+Hyst) の `lev_raw` 段階に `nasdaq_mom63` を quantile_cut(levels=4) + M6 defensive で乗算。Mapping: signal_q {0,1,2,3} → multiplier {1.1, 1.0, 0.9, 0.8}。**MaxDD -34.57% → -28.74% (+5.83pp)** が headline、Sharpe +0.047 改善、CAGR -0.86pp の trade-off。
- 検証: 信号拡張プロジェクト (Phase A→D + Sessions 1-5, 2026-06-03〜06-07) の **唯一の正式 ADOPT 候補**。76 信号 × 306+ patterns 検証で発見。Phase D Hard Gate 4/4 全 PASS (WFE=1.005, CI95_lo=+13.00%, P_Sharpe=0.930, **P_MaxDD=0.988 ⭐**)。
- 制約: S3 (ETF only) 限定。CFD 系 (S2_D5 WFE=0.963, E4_Active WFE=0.958) への転用は FAIL。
- 学習: [LESSONS_LEARNED_20260607.md](LESSONS_LEARNED_20260607.md) 参照。**新セッションで信号探索を再開する場合は本書を必ず先読み**。

### CashSleeve / DH_W1派生（OUT 期キャッシュを1倍投信で置換）
- **Shortlisted (2026-06-07 追加, 投信環境 Active 候補)**: DH_W1_CashSleeve_P2_GOLD100（攻め型, OOS +16.44% 最高）, DH_W1_CashSleeve_P7_GOLD75BOND25 ⭐（中庸推奨, MaxDD −48.23%）, DH_W1_CashSleeve_P5_GOLD50BOND50（守り型, MaxDD −35.97% 最良 / CI95_lo +17.23% 最高） → §2
- **ベース**: DH_W1_AsymmHyst（OUT=キャッシュ 0% の現状） → §2
- 着想: DH-W1 は全営業日の **46.9%(6,171日)をキャッシュ 0% で待機**。OUT 期資産分析で Bond 1x が年率 +6.51%/Vol 11.9%(最良)、Gold +4.07%、NASDAQ −7.63%(risk-off で大損)。ゆえに OUT を **Gold/Bond 1倍投信**で埋めるのが正解。NASDAQ 100% スリーブ(P1)等は §2 非掲載(part1 で棄却)、全7パターンは [cash_sleeve_7patterns_metrics.csv](analysis_cash_sleeve/cash_sleeve_7patterns_metrics.csv) 参照。
- コスト前提: 1倍スリーブは **SOFR/スワップなし・信託報酬<0.2%・5営業日ラグ**（レバ脚は DELAY=2 / SOFR/スワップ込み）。税 20.315%(×0.8273)。SBI 商品: NASDAQ100 0.1958% / サクっと純金 0.1838% / iシェアーズ米国債20年超(2255) 0.154%。
- 補足: NASDAQ 脚はコストエンジン上 SBI 店頭 CFD(スプレッド3.0%/年)モデル。Gold(2036)/Bond(TMF) は ETF コスト(TER+SOFR)モデル（[CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md) §5 に全コスト表）。

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

- **2026-06-07 (DH-W1 Cash-Sleeve)**: **§2 Shortlisted に「投信環境 Active 候補」3 件を新規追加** — DH-W1 (ETF only) の OUT(キャッシュ 46.9%)期を 1 倍投信で運用置換する派生系。**DH_W1_CashSleeve_P2_GOLD100** (攻め, OOS +16.44% 最高/Sharpe +0.875/gap +0.97pp/WFE 1.229), **DH_W1_CashSleeve_P7_GOLD75BOND25** ⭐ (中庸推奨, OOS +14.90%/MaxDD −48.23%/WFE 1.043), **DH_W1_CashSleeve_P5_GOLD50BOND50** (守り, MaxDD −35.97% 最良/CI95_lo +17.23% 最高/gap +5.50pp)。いずれも WFA 50窓で CI95_lo>0 (α) ∩ 0.5≤WFE≤2.0 (β) PASS。検証済み事項: OUT 期資産 (NASDAQ −7.63% / Gold +4.07% / Bond +6.51%)、全商品コスト表 (SOFR/TER/スワップ/売買/税 20.315%)、執行ラグ (レバ脚 DELAY=2 / 投信スリーブ 5BD)。**t_p(permutation)/bootstrap 未実施のため正式 Active 昇格は保留**。§5 逆引きに新テーマ「CashSleeve / DH_W1派生」追加。一次根拠: [CASH_SLEEVE_REPORT_20260607.md](analysis_cash_sleeve/CASH_SLEEVE_REPORT_20260607.md)。
- **2026-06-05 (v4.7)**: **CFD 環境 Active 候補を vz065_l7_F10eps015 → vz065_l5_F10eps015 に置換** (ユーザー判断、防御指標優位を優先)。l7 (旧 REF) は副候補に降格、l5.5 はバランス候補として継続 Shortlisted。
- **2026-06-05 (v4.6)**: **§2 Shortlisted に v4.6 lmax sweep 2 件追加**。
  - **vz065_l5p5_F10eps015** ⭐: CFD バランス推奨 (l5.5、min -0.58pp vs REF と引換に W10Y +2.50pp / P10_5Y +3.87pp / MaxDD -6.08pp 改善)
  - **vz065_l5_F10eps015** ⭐: CFD 防御最優先 (l5、防御指標すべて最良、P10_5Y +4.70pp 大幅改善)
  - REF (l7) は CFD 攻め型として継続。risk appetite 別の選択肢を整備
- **2026-06-05 (v4.5)**: **保守的採用基準 min(IS, OOS) CAGR + Worst10Y + P10_5Y の 3 軸を §7-2 で導入**。STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md (v4.5) と整合させ §2 / §3 を更新:
  - §2 Shortlisted 新規追加: **vz065_l7_F10eps015** (CFD 環境 Active 候補, min CAGR=+20.23% で 5 戦略中 1 位) ⭐
  - §2 Shortlisted 新規追加: **DH_W1_AsymmHyst** (ETF 環境 Active 候補, ETF 制約下で DH +9.56→+13.66% の唯一の改善 case) ⭐
  - §2 既存 DH_Z2_AllocTiming → **Superseded by DH_W1 (v4.3)** で取消線マーク
  - §3 Rejected 新規追加: **vz065_l7_F10eps015_AH / _AT / _HL** (v4.4 で AH 採用と判定したが v4.5 min ルール下で 3 軸全敗・WFE>1.5 regime luck 疑いで棄却)
  - 含意: v4.x の OOS-only 評価による改善提案 (Z2/W1/AH 等) は min ルールで再評価すると REF が依然首位 ─ ETF 制約環境での DH-W1 が唯一の改善 case
- **2026-06-03 (v4)**: **DH_T4_Improved を §2 Shortlisted → §3 Rejected に降格** (`lev_mod` で TQQQ position 連続スケールは ETF 「保有/非保有」制約違反)。代替として **DH_Z2_AllocTiming を §2 Shortlisted に新規追加** (商品は TQQQ+TMF+2036 維持、F10 ε tilt 配分 + binary HOLD/OUT vz_gate、レバ操作なし、peak lev 2.85x≤3.0x assert で機械検証)。**IS-OOS gap +10.46pp→+2.17pp (-8.29pp)・WFE 0.662→1.058 (完全汎化)・CAGR_OOS +12.26% (REF +9.56% を 2.70pp 上回る)・OOS 累積 ×2.00 (REF ×1.73)**。Worst10Y/P10_5Y は防御性能低下 (trade-off)。STRATEGY_PERFORMANCE_INTEGRATED_20260603-v2.md は v3→v4、§0'/§5/§6/§6-2 4 箇所で DH-T4→DH-Z2 完全置換。v4.3 で DH_W1 に再置換 (§2 新規エントリ参照)。
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
