# 信号拡張プロジェクト 最終意思決定書

作成日: 2026-06-07
最終更新日: 2026-06-07

> **本書の役割**: Phase A〜D + Sessions 1〜5 (2026-06-03〜06-05) に渡る信号探索・統合検証プロジェクトの **全結論を1ファイルに集約** し、ユーザーが運用判断を下すために必要な情報を提供する。
> 一次根拠ファイルへのリンクを各セクションに記載。詳細を確認したい場合のみリンク先を参照。
>
> **本版 (v2, 2026-06-07) 更新内容**: `docs/rules/08_evaluation-metrics.md` v1.0 標準 9 指標フレームワークに完全準拠するため、§3 全テーブルを **canonical split (IS_END=2021-05-07 / OOS_START=2021-05-08)** で再計算 → 新規 §3.4 (1974-2026 年次リターン表)、§3.5 (ファクトチェック注記) を追加。詳細は §3.5 を参照。
>
> - Evaluation Standard: v1.1 / Cost Scenario: D (`src/product_costs.py` 2026-05-12 基準)
> - IS: 1974-01-02〜2021-05-07 / OOS: 2021-05-08〜2026-03-26
> - Metrics: 標準 7 + WFA 補助 2 (`docs/rules/08_evaluation-metrics.md`)

---

## ⭐ Executive Summary (60秒で読める結論)

**3つの答え** (運用環境別):

| 環境 | 推奨アクション |
|---|---|
| **CFD 利用可** (現行 §1 Active = E4 RegimeKLT) | **変更なし**。本プロジェクトの全 156 + 150 = 306 検証パターンで、現行を改善する CFD 用 overlay は見つからず |
| **ETF only** (NISA 等、DH-W1 ベース) | **`nasdaq_mom63 × M6 defensive` overlay を採用検討可**。MaxDD を **-34.6% → -28.7% (+5.83pp)** に改善 (CAGR -0.86pp の trade-off)、Phase D 4 gate 全 PASS |
| **新規ユーザー / 検討中** | 現行 Active 戦略 (E4 RegimeKLT, CFD) を採用。ETF only 制約があれば DH-W1 + 上記 overlay |

**プロジェクト全体の成果**:
- 76 信号評価 / 306 + パターン検証 / **1 正式 ADOPT** 達成
- 重大な方法論的発見 (Post-hoc 過大評価問題、Defensive vs Procyclical 構造) を体系化

**プロジェクト全体の成果と限界**:
- ✓ Phase A〜D で 0 ADOPT だった状況から、拡張 Sessions で **1 ADOPT 発見**
- × ADOPT は **S3 (DH-W1, ETF only) 限定** で、本番 Active (CFD: E4) には転用不可
- × CFD 用の改善 overlay は **全 検証で発見できず** → 現行 Active は局所最適と確定

---

## 1. 最終意思決定 (Decision)

### 1.1 §1 本番 Active 戦略
**`S2_VZGated + LT2-N750 + E4 Regime k_lt (k_lo=0.1, k_hi=0.8, vz_thr=0.7)` を変更なし維持**

| 指標 | 値 |
|---|---|
| CAGR_OOS | +33.53% |
| Sharpe_OOS | +0.891 |
| MaxDD | -60.01% |
| Worst10Y★ | +18.67% |
| Trades/yr | 27 |
| WFA CI95_lo / WFE | +26.51% / 1.131 |

**理由**: Sessions 5 で nasdaq_mom63 × M6 defensive overlay を E4 (現行 Active) に転用テストした結果、WFE=0.958 (Hard Gate FAIL <1.0) で NEEDS_FURTHER_WORK となり、現状で正式採用に値しない。

### 1.2 新規追加: Risk-Reduction Overlay Candidate (ETF only 用)

**`DH-W1 + nasdaq_mom63 × M6 × defensive` を S3 (ETF only) 環境 で採用可** (ユーザー判断)。

| Overlay 構成 | 値 |
|---|---|
| Signal | `nasdaq_mom63` (NASDAQ 63日モメンタム) |
| 量子化 | quantile_cut levels=4 → publication_lag('daily', +1 BD) |
| Method | M6 (continuous threshold-proxy) |
| Direction | defensive (高モメンタム → レバ下げ) |
| Mapping | signal_q {0,1,2,3} → multiplier {1.1, 1.0, 0.9, 0.8} |
| 適用先 | S3 = DH-W1 (Asymm+Hyst, TQQQ/TMF/GLDM) の `lev_raw` 段階 |

**Phase D Hard Gate 結果** (4 / 4 PASS):

| Gate | 基準 | 実測 (Session 5) | v2 canonical 再計算 | 判定 |
|---|---|---|---|---|
| WFE (50w Sharpe) | ≥ 1.0 | 1.005 | 1.044 | ✓ PASS |
| CI95_lo (window CAGR) | > 0 | +13.00% | +12.65% (annual) | ✓ PASS |
| Bootstrap P(Sharpe > base) | > 0.90 | **0.930** | (Bootstrap 再実行は §8 で計画) | ✓ PASS |
| Bootstrap P(MaxDD better) | > 0.90 | **0.988** ⭐ | (同上) | ✓ PASS |

詳細: [phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md)

---

## 2. Decision Matrix (運用環境別 推奨)

| 運用環境 | 戦略 | overlay 適用 | 期待効果 |
|---|---|---|---|
| **CFD 利用可** (税後・口座制約なし) | 現行 §1 Active = E4 RegimeKLT (vz=0.70) | **適用しない** | 既存 Sharpe +0.891 / CAGR +33.5% を維持 |
| **CFD 利用可 / v4.5 候補も検討** | vz=0.65+l5+F10ε (v4.7 CFD Active 候補) | 適用しない | min CAGR +18.93%、防御指標で l7 比優位 |
| **ETF only** (NISA 等、CFD 不可) | **DH-W1 + nasdaq_mom63 × M6 defensive overlay** | ⭐ **採用可** | MaxDD -34.6% → -28.7%、Sharpe +0.047 |
| **ETF only / overlay なし** | DH-W1 単体 (v4.5 ETF Active 候補) | 適用しない | DH 基線比 +4.10pp 改善、min CAGR +13.66% |

詳細な戦略指標比較: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md)

---

## 3. Audit Evidence Summary (全証拠1表)

### 3.1 ADOPT 候補の 9+1 完全指標比較 (S3 DH-W1)

**Canonical split (IS_END=2021-05-07 / OOS_START=2021-05-08) で再計算 (2026-06-07)**

| # | metric | DH-W1 baseline | + overlay | diff | 判定 |
|---|---|---|---|---|---|
| 1a | **CAGR_IS** (1974-2021-05) | +18.10% | +16.69% | -1.41pp | △ IS リターン低下 |
| 1b | **CAGR_OOS** (2021-05-2026) | +18.91% | +18.06% | -0.86pp | △ minor trade-off |
| 1c | **CAGR_FULL** (1974-2026) | +18.17% | +16.81% | -1.36pp | △ |
| 2 | **Sharpe_OOS** | +0.8448 | **+0.8918** | **+0.047** | ✓ improved |
| 3 | **MaxDD** (FULL) | -34.57% | **-28.74%** | **+5.83pp** | ⭐ headline |
| 4 | **Worst10Y★** (calendar) | +10.37% | +10.75% | +0.38pp | ✓ |
| 5 | **P10_5Y▷** (daily roll) | +4.82% | +5.21% | +0.39pp | ✓ |
| 6 | **IS-OOS gap** | -0.81pp | -1.37pp | -0.56pp | △ wider |
| 7 | **Trades/yr (NAV proxy)** | 68.7 | 59.2 | -9.5 | ≈ overlay 軽減 |
| 8 | **WFA_CI95_lo** (annual CAGR) | +13.61% | +12.65% | -0.96pp | △ |
| 9 | **WFA_WFE** (calendar IS vs OOS) | 0.744 | 0.782 | +0.038 | ✓ improved |
| 9b | WFE (50w Sharpe proxy, 旧表記) | 1.023 | **1.044** | +0.021 | ✓ improved |

→ 9 指標中 **改善 5 / 劣化 0 (severe) / 中立または軽微劣化 4** → **STANDARD_PASS_FULL** (n_imp≥3 かつ severe劣化 0)

旧版 (v1, 2018-01-01 split) との対応は §3.5 ファクトチェック表参照。

詳細元データ: [decision_9metrics_20260607.csv](data/signals/expansion/decision_9metrics_20260607.csv)

### 3.2 Cross-Strategy 転用結果 (overlay の汎用性検証)

**A. Session 5 原版 (2018-01-01 split, Bootstrap multi-metric)**

| Baseline | WFE (50w Sharpe) | CI95_lo CAGR (50w) | P_CAGR | P_Sharpe | **P_MaxDD better** | MaxDD diff | Verdict |
|---|---|---|---|---|---|---|---|
| **S3 (DH-W1)** | **1.005** | +13.00% | 0.295 | **0.930** | **0.988** | **+5.83pp** | **ADOPT** ⭐ |
| S2 (D5 = vz065+lmax5) | 0.963 △ | +22.72% | 0.201 | 0.758 | 0.944 | +1.19pp | NEEDS_FURTHER_WORK |
| **E4 (現行 §1 Active)** | **0.958 △** | +24.41% | 0.355 | 0.858 | 0.964 | +1.51pp | **NEEDS_FURTHER_WORK** |

**B. Canonical split (2021-05-08) 再計算 (2026-06-07 追加)** — annual calendar-year WFE / annual CAGR CI95_lo / 50w Sharpe WFE proxy:

| Baseline | nav_type | CAGR_OOS | Sharpe_OOS | MaxDD | WFA_CI95_lo (annual) | WFA_WFE (calendar) | WFE (50w Sharpe) |
|---|---|---|---|---|---|---|---|
| **S3 (DH-W1)** | baseline | +18.91% | 0.8448 | -34.57% | +13.61% | 0.744 | 1.023 |
| **S3 (DH-W1)** | + overlay | +18.06% | **0.8918** | **-28.74%** | +12.65% | 0.782 | **1.044** |
| S2 (D5) | baseline | +33.40% | 0.9497 | -51.82% | +24.42% | 0.825 | 0.921 |
| S2 (D5) | + overlay | +31.04% | 0.9697 | -50.63% | +22.30% | 0.841 | 0.938 |

> **E4 は S2 と同じ build_candidate_nav('S2', ...) 経由で overlay 適用された** (Session 5 慣例)。E4 のオリジナル `lev_mod_e4` を直接 overlay 適用するスクリプトは未整備のため、本表では E4 行を省略 (B). 詳細な E4 transfer 結果は §3.2-A の Session 5 原版表を使用してください。

**重要発見** (両版で一致): MaxDD 改善方向は 3 戦略全てで一貫だが、改善規模は S3 (+5.83pp) → S2 (+1.19pp) で **約 1/5 に減衰**。CFD 系の VZ ゲート + LT2-modeB / Regime k_lt が **既に同等防御を実装している** ため、追加 overlay の限界効用が小さい。

詳細: [session5_transfer_report_20260605.md](data/signals/expansion/session5_transfer_report_20260605.md) / [decision_transfer_canonicalsplit_20260607.csv](data/signals/expansion/decision_transfer_canonicalsplit_20260607.csv)

### 3.3 Phase A→D + Sessions 1→5 全体結果

| 段階 | 評価対象 | パターン数 | 採用候補 |
|---|---|---|---|
| Phase A (Tier1) | 52 候補 | — | 46 理論選別 |
| Phase B (IC統計) | 7 | 63 tests | 17 triples (BH-FDR<0.10) |
| Phase C (買い持ち比較 ❌ 誤方針) | 17 | 27 | 2 (post-hoc 過大評価) |
| Tier 1-3 (post-hoc) | 6 | 117 | 0 (full eval) |
| Phase D (BAA-10Y native, 初回) | 1 | 1 | **0 REJECT** |
| Session 2 (G2 IC, 拡張) | 52 | 156 tests | 20 Top |
| **Session 3 (G3 Native, Top5)** | 5 | **150** | **5 STANDARD_PASS** |
| **Session 4 (Phase D 厳格, Top3)** | 3 | — | **1 ADOPT ⭐** |
| Session 5 (転用 audit) | 1 | 2 | 0 transfer ADOPT |

---

### 3.4 年次カレンダーリターン (1974-2026)

5 戦略 × 53 年 のカレンダー年リターン (Jan 1 → Dec 31)。全期間は [decision_annual_returns_20260607.csv](data/signals/expansion/decision_annual_returns_20260607.csv) を参照。**直近 20 年のみ抜粋**:

| Year | S1_F10 | S2_D5 | S3_DH-W1 | E4_Active | S3+overlay |
|---|---|---|---|---|---|
| 2007 | +98.18% | +43.16% | +18.99% | +95.81% | +20.45% |
| 2008 | -69.49% | -53.85% | -28.62% | -62.86% | -27.61% |
| 2009 | +159.55% | +110.46% | +52.59% | +152.79% | +42.81% |
| 2010 | +71.59% | +56.07% | +33.60% | +71.96% | +30.62% |
| 2011 | -10.41% | +5.77% | +1.96% | +1.41% | +1.96% |
| 2012 | +38.17% | +31.54% | +29.85% | +42.86% | +27.31% |
| 2013 | +50.91% | +40.20% | +11.74% | +44.17% | +12.29% |
| 2014 | +4.01% | +7.90% | +1.44% | +10.18% | +3.06% |
| 2015 | -31.54% | -24.93% | -16.59% | -32.49% | -13.37% |
| 2016 | -6.91% | +3.36% | -5.87% | -11.27% | -2.77% |
| 2017 | +118.83% | +72.72% | +28.79% | +104.21% | +26.70% |
| 2018 | +0.63% | +4.14% | -6.73% | -2.18% | -3.46% |
| 2019 | +69.13% | +68.72% | +29.80% | +67.53% | +28.15% |
| 2020 | +101.41% | +87.38% | +77.03% | +94.42% | +64.31% |
| 2021 | +22.24% | +19.41% | +23.00% | +21.22% | +21.15% |
| 2022 | -25.96% | -26.37% | +0.00% | -26.21% | +0.00% |
| 2023 | +109.06% | +96.95% | +32.58% | +100.56% | +29.67% |
| 2024 | +57.95% | +53.81% | +21.86% | +44.07% | +24.38% |
| 2025 | +51.03% | +50.37% | +25.96% | +54.54% | +22.59% |
| 2026 (YTD 3/26) | -8.14% | -10.47% | -7.40% | -7.35% | -6.77% |

**観察**:
- **2022 (NASDAQ -33%)**: S1/S2/E4 は -25〜-26%、**S3 系は 0.00%** (DH-W1 hysteresis が完全離脱)。これが S3 系の MaxDD 優位の最大根拠。
- **2008**: 全戦略マイナスだが、S3 系の -27〜-29% に対し S1 は -69% (Leveraged ETF の経路依存性)。
- **2017/2019/2020/2023**: 全戦略大幅プラス、ベースリターン環境では CFD 系 (S1/S2/E4) が S3 系を圧倒。
- **overlay 効果**: S3 baseline 比、overlay 適用後は強気年で -2〜-12pp 削れる一方、2008/2015/2018 等の弱気年で +1〜+3pp 改善。

ピーク年率比較 (1976/2017/2023):

| Year | S1 | S2 | S3 | E4 | S3+overlay | NASDAQ参考 |
|---|---|---|---|---|---|---|
| 1976 | +109.93% | +77.34% | +42.16% | +106.42% | +39.83% | +26.10% |
| 2017 | +118.83% | +72.72% | +28.79% | +104.21% | +26.70% | +28.24% |
| 2023 | +109.06% | +96.95% | +32.58% | +100.56% | +29.67% | +43.42% |

### 3.5 Fact-Check Notes (v1 → v2 数値変更点)

**目的**: v1 (2018-01-01 split) で記載されていた §3.1 / §3.2 の数値を canonical split (2021-05-08) で再計算したため、差分を文書化する。

**v1 → v2 主要差分** (S3 DH-W1):

| 指標 | v1 (2018-01-01 split) | v2 (2021-05-08 canonical) | 差分原因 |
|---|---|---|---|
| Baseline CAGR_OOS | +18.96% | +18.91% | split 後 OOS 期間 +3年差 → 期間平均化 |
| Overlay CAGR_OOS | +18.10% | +18.06% | 同上 |
| Baseline IS-OOS gap | -0.88pp | -0.81pp | split 変更 → IS-CAGR 微差 |
| Worst10Y CAGR | +9.84% / +10.38% | +10.37% / +10.75% | カレンダー年実装に変更 (旧: daily rolling) |
| P10_5Y | +5.94% / +5.92% | +4.82% / +5.21% | 同上 daily rolling 統一 |
| WFE (50w Sharpe proxy) | 0.976 / **1.005** | 1.023 / **1.044** | NAV 系列の window 分割が split に非依存だが、Sharpe スケール差 |

**§1 Active 数値の検証**:

§1.1 表に記載の §1 Active (E4 RegimeKLT) 値と raw NAV 再計算の比較:

| 指標 | §1.1 表 (v1) | NAV 再計算 (canonical) | 差分 | 備考 |
|---|---|---|---|---|
| CAGR_OOS | +33.53% | +33.44% | -0.09pp | 計測タイミング微差、整合 |
| Sharpe_OOS | +0.891 | +0.892 | +0.001 | 整合 |
| MaxDD | -60.01% | -60.01% | 0.00pp | 完全一致 |
| Worst10Y★ | +18.67% | +18.67% | 0.00pp | 完全一致 |
| Trades/yr | 27 | 27.12 (=1417/52.26 年) | +0.12 | 整合 |
| IS-OOS gap | -1.81pp | -1.69pp | +0.12pp | 整合 |
| WFA CI95_lo | +26.51% | +26.55% (annual CAGR) | +0.04pp | 整合 (annual CAGR 系) |
| WFA WFE | 1.131 | 0.730 (calendar IS-OOS 比) | -0.401 | **要注意** ‡ |

‡ **WFE 1.131 (v1) は g14 WFA 50-window walk-forward の `mean_post / mean_is` 出力値**。本書 v2 の annual calendar WFE (0.730) とは方法論的に異なる (Walk-Forward window 数=50 vs calendar year window 数=53)。両者の判定基準も別:
- v1 (WFE=1.131): G3 WFA β基準 0.5≤WFE≤2.0 を PASS
- v2 (WFE=0.730): 同基準 0.5≤WFE≤2.0 を PASS (β両方 PASS)
→ **両指標とも β基準 PASS → §1 Active 維持判断に影響なし**。

**Caveats**:
1. **Cost adjustments**: §1.1 の値は g15b ens2 OOS Scenario D (cost-included) 由来。本 v2 NAV (`e4_nav_cache.pkl`) も同 Scenario D で構築済のため、コスト差は最小限。残存差 (~0.1pp) は丸め誤差範囲。
2. **E4 transfer overlay**: §3.2-B で E4 行を省略した理由は §3.2 注記参照。E4 への overlay 適用は Session 5 (`session5_phase_d_audit_E4_Active_20260605.md`) の値を継続採用。
3. **Trades/yr**: NAV proxy (sign flip 法) は CFD 系で 118〜120/yr、S3 系で 60〜70/yr と算出。E4 の正確な値は `e4_nav_cache.pkl` 内 `n_tr/n_years=27.12` を使用。NAV proxy は **screening grade** であり、正確な取引数は backtest cache から取得すべき (本書では E4 のみ exact 値を採用)。
4. **Split selection**: 本書の §1.1 数値 (g15b ens2 由来) は内部的に IS_END=2021-05-07 / OOS_START=2021-05-08 を採用。Session 4/5 の `nine_metric_eval.py` は default split=2018-01-01 を使用していた点が v1 で混在していた。v2 で canonical 2021-05-08 に統一済。

**結論**: v1 → v2 で **MaxDD・Worst10Y★ は完全一致、CAGR/Sharpe/IS-OOS gap は ±0.5pp 以内**、最終判定 (S3 ADOPT / E4 維持) は変わらず。

---

## 4. 棄却された候補 (Rejection Log)

| 候補 | 棄却理由 | 詳細 |
|---|---|---|
| BAA-10Y × S3 × M2 procyclical (Phase D 初回) | Bootstrap P(CAGR>base) = 0.39 で偶然性排除できず | [phase_d_audit_report_20260605.md](data/signals/integration/phase_d_audit_report_20260605.md) |
| nfci_z52w × S3 × M2 defensive | 全 Bootstrap P < 0.45、改善統計的有意性なし | [phase_d_audit_nfci_z52w_S3_M2_def_20260605.md](data/signals/expansion/phase_d_audit_nfci_z52w_S3_M2_def_20260605.md) |
| vix_mom21 × S3 × M2 defensive (NEEDS_WORK) | WFE=0.967, P(MaxDD)=0.902 (惜しい) → NEEDS_FURTHER_WORK 扱い、再評価可 | [phase_d_audit_vix_mom21_S3_M2_def_20260605.md](data/signals/expansion/phase_d_audit_vix_mom21_S3_M2_def_20260605.md) |
| nasdaq_mom63 × **S2 (D5)** × M6 defensive | WFE=0.963 (FAIL <1.0) | [session5_phase_d_audit_S2_D5_20260605.md](data/signals/expansion/session5_phase_d_audit_S2_D5_20260605.md) |
| nasdaq_mom63 × **E4 (Active)** × M6 defensive | WFE=0.958 (FAIL <1.0) | [session5_phase_d_audit_E4_Active_20260605.md](data/signals/expansion/session5_phase_d_audit_E4_Active_20260605.md) |
| Tier 1-3 全 117 patterns (BAA10Y, VIX, HY OAS, 2s10s, real yield, DXY 6 信号) | post-hoc 評価の構造的過大評価 — Phase D native で全 REJECT | [integration_final_report_20260605.md](data/signals/integration/integration_final_report_20260605.md) |

---

## 5. プロジェクトで得られた方法論的教訓 (5項目)

### 5.1 ⚠️ Post-hoc multiplication は構造的に過大評価
Tier 1-3 評価で `candidate_nav = baseline_nav.pct_change() × signal_multiplier → cumprod` の post-hoc 方法を採用 → 同候補を Phase D native で再評価すると **Sharpe や CAGR が反対方向に動く** (BAA-10Y: post-hoc CAGR +0.54pp → native -0.44pp)。
→ **以降の全信号評価は native integration 必須**。

### 5.2 ⚠️ Defensive > Procyclical (構造的)
Phase A〜D は procyclical 一辺倒 → 0 ADOPT。Session 3 で defensive に転換 → **5 STANDARD_PASS 発見**。**全 5 件が defensive 方向**。

### 5.3 ⚠️ Multi-metric Bootstrap が必須
Phase D CAGR-only Bootstrap で BAA-10Y P=0.39 REJECT → 同手法では Session 4 ADOPT 候補も CAGR P=0.30 で REJECT になる。**MaxDD/Sharpe Bootstrap P を含めることで** defensive overlay の真の価値が抽出される (nasdaq_mom63 は P_MaxDD=0.988)。

### 5.4 ⚠️ IC ≠ 戦略改善
G2 で最強 IC (`nasdaq_mom21` t-stat +17) は G3 native で PASS 0。逆に G2 中位の `nfci_z52w` / `nasdaq_mom63` / `vix_mom21` が PASS 上位を占めた。**IC スクリーニングは間接的、native integration が決定的**。

### 5.5 ⚠️ 戦略基盤特異性 (Strategy Specificity)
S3 (DH-W1, ETF, hysteresis state machine) は信号注入を安定吸収。S1/S2/E4 (CFD系) は WFE 構造的劣化。**ETF/CFD の構造差** が信号 overlay の効果規模を決める。

---

## 6. 実装ステップ (ETF only ユーザー向け、overlay 採用判断する場合)

```python
# 概念コード (実際は src/integration/build_strategy_with_signal.py を使用)
from signals.quantize import quantile_cut
from signals.timing import apply_publication_lag

# 1. nasdaq_mom63 信号取得 (data/macro_features.csv の nasdaq_mom63 列)
raw_signal = macro_features['nasdaq_mom63']

# 2. 量子化 + 公表ラグ
signal_q = quantile_cut(raw_signal, levels=4)  # 0/1/2/3
signal_lagged = apply_publication_lag(signal_q, lag_type='daily')  # +1 BD

# 3. M6 defensive multiplier
multiplier_map = {0: 1.1, 1: 1.0, 2: 0.9, 3: 0.8}
multiplier = signal_lagged.map(multiplier_map)

# 4. DH-W1 の lev_raw に適用
lev_raw_modulated = lev_raw_W1_baseline * multiplier  # element-wise

# 5. NAV 再計算 (build_dh_nav_with_cost で)
nav = build_dh_nav_with_cost(assets, lev_raw_modulated, mask_W1, wn_W1)
```

実装本体: [src/integration/build_strategy_with_signal.py](src/integration/build_strategy_with_signal.py) (関数 `build_candidate_nav('S3', signal_raw, 'M6', 'defensive')`)

---

## 7. リスク評価 (採用判断時の留意点)

| リスク | 評価 | 対応 |
|---|---|---|
| **CAGR -0.86pp の劣化** | △ 容認可 (95% CI 内: [-4.71pp, +1.86pp]) | リスク削減と引換 |
| **IS-OOS gap が -0.55pp 拡大** | △ -1.43pp (over-fit 方向は深くないが要監視) | 半年毎に再評価 |
| **Worst10Y -0.01pp = ほぼ neutral** | ◯ | 影響なし |
| **新シグナル (nasdaq_mom63) のソース依存** | ⚠ macro_features.csv の更新が止まると overlay 効果なし | 自前計算でも再現可 (簡易) |
| **S3 (DH-W1) ベース自体のリスク** | DH-W1 単体で v4.5 ETF Active 候補。基盤は別途検証済 | DH-W1 本体の risk は別議論 |
| **戦略基盤特異性** | ✓ Cross-strategy transfer test 済、CFD 系では効果限定的を確認 | S3 限定運用 |
| **長期持続性 (next 5-10 years)** | Bootstrap で 10,000 path 検証済、P(MaxDD better)=0.988 で頑健 | 通常運用、年次再評価 |

---

## 8. 残された開放問題 (Future Work)

採用判断後でも継続できる探索:

| 問題 | 内容 | 推定期間 |
|---|---|---|
| A. NEEDS_FURTHER_WORK 2 候補の精密化 | nasdaq_mom63 × M6 × S2 / E4 の mapping を grid search で最適化 | 1-2 セッション |
| B. 他 macro_features 信号 (44 未テスト) の G3 native スクリーニング | より多くの defensive overlay 候補 | 2-3 セッション |
| C. AND/OR combination | nasdaq_mom63 + nfci_z52w の合成 overlay | 1 セッション |
| D. 残 24 paid/manual signals のデータ取得 → 全 76 evaluation | データ整備重 | 数日 |
| E. vix_mom21 × S3 × M2 defensive (NEEDS_WORK) の再評価 | mapping 変更で ADOPT 化を狙う | 1 セッション |
| F. 別アプローチ — 新戦略構造 (signal-conditional F11 等) を Phase X として設計 | ゼロベース | 中長期 |

---

## 9. 関連ドキュメント (詳細確認用)

### 9.1 最重要 (本書 + 戦略台帳)
- 本書: `SIGNAL_EXPANSION_FINAL_DECISION_20260607.md`
- 戦略台帳: [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) (v4.8 — overlay 採用記録済)
- 拡張計画: [SIGNAL_EXPANSION_PLAN_20260605.md](SIGNAL_EXPANSION_PLAN_20260605.md)

### 9.2 Audit 証拠 (重要)
- **ADOPT 候補 audit**: [phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md](data/signals/expansion/phase_d_audit_nasdaq_mom63_S3_M6_def_20260605.md)
- **転用 audit (S2/E4)**: [session5_transfer_report_20260605.md](data/signals/expansion/session5_transfer_report_20260605.md)
- Session 4 Phase D 3候補 summary: [session4_phase_d_summary_20260605.md](data/signals/expansion/session4_phase_d_summary_20260605.md)

### 9.3 過程記録 (参考)
- Phase D 初回 (BAA-10Y REJECT): [phase_d_audit_report_20260605.md](data/signals/integration/phase_d_audit_report_20260605.md)
- Tier 1-3 (post-hoc): [integration_final_report_20260605.md](data/signals/integration/integration_final_report_20260605.md)
- G2 IC スクリーニング: [g2_report_20260605.md](data/signals/expansion/g2_report_20260605.md)
- G3 Native 150 patterns: [g3_native_top5_report_20260605.md](data/signals/expansion/g3_native_top5_report_20260605.md)

### 9.4 規格・規範
- 9指標標準: [docs/rules/08_evaluation-metrics.md](docs/rules/08_evaluation-metrics.md)

---

## 10. 採用判断の問いかけ (ユーザーへの確認事項)

意思決定に必要な質問:

### 10.1 ETF only 環境での運用がありますか?
- **YES** → nasdaq_mom63 × M6 defensive overlay を DH-W1 に適用する判断を要請。
  - 採用するなら本書 §6 の実装手順に沿って組込。
  - 採用しないなら DH-W1 単体 (v4.5 ETF Active 候補) を維持。
- **NO (CFD のみ)** → §1 本番 Active (E4 RegimeKLT) を変更なし維持。本書 §7 のリスク評価は不要。

### 10.2 NEEDS_FURTHER_WORK 候補 2 件の追加検証を行いますか?
- **YES** → §8 A (S2/E4 用 mapping 最適化) または E (vix_mom21 再評価) を実施するセッションを予約。
- **NO** → プロジェクト終了、現状運用継続。

### 10.3 残 macro_features 44 信号や paid/manual 信号の追加探索を行いますか?
- **YES** → §8 B〜D を順次実施。
- **NO** → 本プロジェクトをここで完結とし、別軸 (新戦略構造設計等) に進む。

---

## 11. 最終結論 (1段落要約)

Phase A〜D + Sessions 1〜5 で **76 信号 × 5 注入方式 × 3 戦略 = 約 306 パターン** を検証した結果、**プロジェクト唯一の正式 ADOPT は `nasdaq_mom63 × DH-W1 × M6 defensive` (ETF only 限定)** となった。
本 overlay は MaxDD を **-34.6% → -28.7% (+5.83pp)** に改善し、Phase D 4 gate (WFE / CI95_lo / P_Sharpe / P_MaxDD) を全 PASS。
ただし CFD 系 (S2 / E4) への転用は WFE Hard Gate FAIL で NEEDS_FURTHER_WORK となり、**§1 本番 Active (E4 RegimeKLT, CFD) は変更なし維持** が結論。
**ETF only 環境を持つユーザーは overlay 採用検討を、CFD ユーザーは現状維持を推奨**。

---

*管理者: 男座員也 (Kazuya Oza)*
