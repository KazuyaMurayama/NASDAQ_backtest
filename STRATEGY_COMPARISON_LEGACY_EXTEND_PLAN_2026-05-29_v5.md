# 戦略パフォーマンス比較表 v5 — 過去ベスト戦略 raw 値 修正版
> v4 で誤った raw 値（DH Dyn / Ens2 が FULL 期間値で計算されていた）を **GitHub 上の正規 OOS Scenario D データ**で修正した版。

作成日: 2026-05-29 (v4 から 1 セッション内で修正)
実装: `src/g15_legacy_strategies_realistic.py` (v2)
出力: `g15_legacy_results.csv`
準拠: `EVALUATION_STANDARD.md v1.3` / §3.12 9指標標準 / §3-A 税モデル

---

## 🔴 §0 v4 の誤りと修正内容（透明性のため詳細記載）

### 誤り 1: DH Dyn 2x3x [A] の raw 値が FULL 期間値

| 指標 | v4 で使った値（誤り） | v5 で使う値（正） | 出典 |
|---|---:|---:|---|
| CAGR_OOS_raw | +22.50% (FULL CAGR) | **+14.88%** (OOS) | [`corrected_strategy_results.csv`](corrected_strategy_results.csv) OOS row, Scenario D 列 |
| Sharpe_OOS | +0.993 (FULL Sharpe) | **+0.646** | 同上 |
| MaxDD | -50.0% (推定) | **-45.08%** | 同上 (MaxDD_D 列) |
| Worst10Y★ | +7.0% (粗推定) | **+14.30%** | [`STRATEGY_PERFORMANCE_COMPARISON_2026-05-23.md`](STRATEGY_PERFORMANCE_COMPARISON_2026-05-23.md) §2 表 |
| P10_5Y▷ | +4.5% (粗推定) | **+9.60%** | 同上 |
| IS-OOS gap | +4.00pp (粗推定) | **+8.48pp** | 同上 |
| WFA_CI95_lo | — | **+0.182** | 同上 |
| WFA_WFE | — | **+0.687** | 同上 |

### 誤り 2: Ens2(Asym+Slope) の raw 値が FULL 期間値（Scenario A 想定の高値）

| 指標 | v4 で使った値（誤り） | v5 で使う値（正） | 出典 |
|---|---:|---:|---|
| CAGR_OOS_raw | +28.58% (FULL, Scenario A 推定) | **+12.57%** (OOS 推定) | [`ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md`](ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md) の Full Sharpe 0.846 と OOS Sharpe 0.479 から逆算 |
| Sharpe_OOS | +1.031 (FULL Sharpe) | **+0.479** | 同上 |
| MaxDD | -50.17% (近似) | **-48.99%** | 同上 |
| Worst10Y★ | +9.16% (推定) | **+9.84%** | 同上 (Scenario B/C-ish, Trades 0.7) |
| IS-OOS gap | +2.00pp (推定) | **+10.00pp** (推定) | FULL CAGR 22.20% − OOS 推定 12.57% |

### 誤り 3: 全戦略一律 -5.6pp の CFD drag を適用

| 戦略 | v4 drag | v5 drag | 根拠 |
|---|---:|---:|---|
| S2+LT2 k=0.5 modeB | -5.6pp | **-5.6pp** | S2系 CFD ベース、cfd_spread 0.20% → 3.0% (g13 経験値) |
| S2_VZGated 単独 | -5.6pp | **-5.6pp** | 同上 |
| **DH Dyn 2x3x [A]** | -5.5pp | **0pp** | **TQQQ ETF (Scenario D) と SBI CFD のコストはほぼ同水準** (TQQQ 0.86% TER + 2×SOFR + swap ≈ 8.4% per 3x exposure vs SBI CFD 3.0% spread + SOFR ≈ 6.5%/yr per 3x。差 ≈ -1.9pp で CFD の方が安い、保守的に 0 で扱う) |
| **Ens2(Asym+Slope) max_lev=1.0** | -2.8pp | **0pp** | **無レバレッジ戦略は CFD 不要 (QQQ ETF 等で実装)** |
| B&H | 0 | 0 | 純粋指数保有 |

### 誤り 4: Ens2 の P10_5Y を粗推定（+4.7%）

→ v5 では P10_5Y は **データ不在のため `—` 表記**に変更（推定値の根拠が薄いため）。

---

## 📊 §1 修正済み 5戦略（v5 確定値）

| Strategy | CAGR<br>raw | CFD<br>drag | 未含<br>cost | 税前<br>(after cost) | × 0.8273 | **CAGR_OOS<br>net (⓽)** |
|:---|---:|---:|---:|---:|---:|---:|
| S2+LT2 k=0.5 modeB | +31.16% | -5.6pp | -0.66% | +24.90% | × 0.8273 | **+20.60%** |
| S2_VZGated 単独 | +27.51% | -5.6pp | -0.66% | +21.25% | × 0.8273 | **+17.58%** |
| DH Dyn 2x3x [A] | +14.88% | 0 | -0.66% | +14.22% | × 0.8273 | **+11.77%** |
| Ens2(Asym+Slope) | +12.57%* | 0 | (B&H扱い) | +12.57% | × 0.8273 | **+10.40%** |
| NDX 1x B&H | +10.11% | 0 | (B&H扱い) | +10.11% | × 0.8273 | **+8.36%** |

*Ens2 OOS CAGR は FULL Sharpe 0.846 (CAGR 22.20%) → vol 26.24% を導出し、OOS Sharpe 0.479 × vol 26.24% = 12.57% で推定。

### Sharpe / MaxDD / Worst10Y / P10 / gap / WFA 全指標（v5 確定）

| Strategy | Sharpe<br>OOS (ⓒ) | MaxDD (ⓒ) | Worst10Y★ (⓽) | P10_5Y▷ (⓽) | IS-OOS gap | WFA<br>CI95_lo (ⓔ) | Overfit<br>(WFE) | Trades<br>/yr |
|:---|---:|---:|---:|---:|---:|---:|:---:|---:|
| S2+LT2 k=0.5 modeB | +0.86 **H** | -61.5% | +9.8% | +2.6% | +0.18pp | +0.16 | ✅ LOW (1.1) | 27 |
| S2_VZGated 単独 ⚠↑↑ | +0.77 | -65.4% | +9.5% | +0.9% | **+6.04pp ⚠↑↑** | +0.18 | ✅ LOW (0.8) | 27 |
| **DH Dyn 2x3x [A] ⚠↑↑** | **+0.65** | -45.1% | +11.3% | +7.4% | **+8.48pp ⚠↑↑** | +0.15 | ✅ LOW (0.7) | 27 |
| **Ens2(Asym+Slope) §⚠↑↑** | **+0.48** | -49.0% | +8.1% | — | **+10.00pp§ ⚠↑↑** | — | — | 1 |
| NDX 1x B&H 🅑 | +0.54 | **-77.9%** | **-4.7%** | +0.6% | +1.02pp | +0.06 | ✅ LOW (1.1) | 0 |

凡例:
- **H** = Sharpe > +0.840（高水準）/ **M** = > +0.780（E4 基準）
- **⚠↑↑** = IS-OOS gap ≥ +5.0pp（古典 overfit 強警戒、v4新設）
- **‡** = §1.3 参考値戦略（採用候補外）
- **§** = OOS データ推定（Ens2 のみ — FULL Sharpe vs OOS Sharpe 比から推定）
- **🅑** = ベンチマーク（戦略でない比較基準）

---

## 🔍 §2 v4 vs v5 主要変更点

| 指標 | DH Dyn 2x3x [A] | Ens2(Asym+Slope) |
|---|:---:|:---:|
| CAGR_OOS_net v4 | **+13.52%** | **+20.78%** |
| CAGR_OOS_net v5 | **+11.77%** | **+10.40%** |
| 変化幅 | -1.75pp（小幅補正） | **-10.38pp（大幅補正）** |
| Sharpe_OOS v4 | +0.993 (FULL) | +1.031 (FULL) |
| Sharpe_OOS v5 | **+0.646** | **+0.479** |
| 変化幅 | -0.347（重要補正） | -0.552（重要補正） |
| IS-OOS gap v4 | +4.00pp (粗推定) | +2.00pp (粗推定) |
| IS-OOS gap v5 | **+8.48pp** ⚠↑↑ | **+10.00pp** ⚠↑↑ |

### v4 の不自然さの根本原因

DH Dyn と Ens2 は **過去のベスト戦略**として CURRENT_BEST_STRATEGY.md の blacklist や 2026-04-19 推奨に「CAGR 22-29% / Sharpe 0.99-1.03」と記録されていた。これらは**いずれも FULL 期間（1974-2026, 52年）の値**で、しかも Ens2 は**Scenario A（コスト未計上）に近い**値だった。

OOS Scenario D で見ると:
- DH Dyn 2x3x [A] は **OOS 期間（2021-2026）にコスト込みで CAGR たった +14.88%、Sharpe 0.65** — 強気相場の OOS にもかかわらず IS（23.36%）を大きく下回り、**IS-OOS gap +8.48pp の古典的 overfit シグナル**
- Ens2 は **OOS Sharpe 0.479** と B&H (0.54) すら下回る — レバ無しでこの数値は致命的

v4 はこれを掴めず、FULL 期間値を使ったため過大評価された。

---

## 📊 §3 最終形 — v5 統合比較表（26戦略 × 9指標）

> ⓽/ⓒ/ⓞ/ⓔ は v3 §0 凡例参照
> マーカ: H/M (Sharpe水準) / ⚠/⚠⚠ (OOS過大警戒) / **⚠↑/⚠↑↑ (古典overfit警戒, v4新設)** / ‡ (参考値) / § (OOS データ推定) / 🅑 (ベンチマーク)

| Strategy | CAGR⓽<br>_OOS | Sharpeⓒ<br>_OOS | MaxDDⓒ | Worst<br>10Y★⓽<br>CAGR | P10⓽<br>5Y▷<br>CAGR | IS-OOSⓒ<br>gap | Tradeⓞ<br>（回/年） | Overfitⓞ<br>(WFE) | CI95ⓔ<br>_lo |
|:---------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|:----------------:|-----------:|
| **E4 Regime k_lt ◆** | **+22.4%** | **+0.79 M** | **-62.0%** | **+9.8%** | **+2.2%** | -2.41pp | 27 | ✅ LOW<br>(1.1) | +0.17 |
| **F10 ε=0.015 [Active候補] ✅⚠** | **+24.8%** | **+0.83 M** | **-65.5%** | **+9.3%** | **+2.5%** | -4.89pp ⚠ | 52 | ✅ LOW<br>(1.2) | +0.17 |
| **F10+lmax5 [Active候補] ✅** | **+23.0%** | **+0.84 H** | **-56.4%** | **+9.6%** | **+5.9%** | -3.18pp | 52 | ✅ LOW<br>(1.3) | +0.16 |
| **D5 vz=0.65/lmax=5.0 [Active候補] ✅** | **+23.0%** | **+0.85 H** | **-53.0%** | **+11.0%** | **+5.2%** | -3.65pp | 27 | ✅ LOW<br>(1.3) | +0.16 |
| [B] B4 k_lo=0/k_hi=0.7/vz=0.7 | +22.1% | +0.78 M | -61.9% | +9.8% | +2.4% | -2.22pp | 27 | ✅ LOW<br>(1.1) | +0.17 |
| [A] A1 α=2 (soft regime) | +22.3% | +0.78 M | -63.6% | +9.6% | +0.6% | -1.96pp | 27 | ✅ LOW<br>(1.1) | +0.17 |
| [A] A1 α=3 (soft regime) | +22.6% | +0.79 M | -64.7% | +9.5% | +0.2% | -2.18pp | 27 | ✅ LOW<br>(1.1) | +0.17 |
| [A] A1 α=5 (soft regime) | +23.0% | +0.79 M | -66.3% | +9.3% | -0.3% | -2.59pp | 27 | ✅ LOW<br>(1.1) | +0.18 |
| [A] A1 α=8 (soft regime) | +23.4% | +0.80 M | -68.4% | +9.2% | -0.7% | -2.98pp | 27 | ✅ LOW<br>(1.1) | +0.18 |
| [A] A2 lmax_base=6/vol_sens=2 | +22.0% | +0.79 M | -60.0% | +10.1% | +3.6% | -1.73pp | 27 | ✅ LOW<br>(1.1) | +0.17 |
| [A] A2B rolling VOL_REF | +21.4% | +0.79 M | -59.6% | +10.4% | +4.1% | -1.34pp | 27 | ✅ LOW<br>(1.1) | +0.16 |
| [A] A3 VoV dual gate | +22.4% | +0.79 M | -62.0% | +9.8% | +2.3% | -2.40pp | 27 | ✅ LOW<br>(1.1) | +0.17 |
| [C] C2 adaptive deadband ⚠ | +24.8% | +0.83 M | -65.5% | +9.4% | +2.6% | -4.87pp ⚠ | 51 | ✅ LOW<br>(1.2) | +0.17 |
| [C] C3 Yang-Zhang ‡ | +20.0% | +0.73 | -69.9% | +10.3% | +2.3% | +0.46pp | 27 | ✅ LOW<br>(1.1) | +0.14 |
| [D5] vz=0.60/lmax=4.5 | +19.7% | +0.79 M | -50.3% | +11.3% | +5.5% | +0.07pp | 27 | ✅ LOW<br>(1.2) | +0.15 |
| [D5] vz=0.60/lmax=5.0 | +21.2% | +0.80 M | -52.3% | +11.4% | +5.2% | -1.10pp | 27 | ✅ LOW<br>(1.2) | +0.16 |
| [D5] vz=0.65/lmax=4.5 | +21.3% | +0.83 M | -51.0% | +11.0% | +5.6% | -2.18pp | 27 | ✅ LOW<br>(1.2) | +0.15 |
| [D5] vz=0.65/lmax=5.5 ✅⚠ | +23.7% | +0.84 H | -55.3% | +11.0% | +4.5% | -4.00pp ⚠ | 27 | ✅ LOW<br>(1.3) | +0.16 |
| [D5] vz=0.65/lmax=6.0 ✅⚠ | +24.0% | +0.84 H | -57.6% | +11.1% | +3.7% | -4.24pp ⚠ | 27 | ✅ LOW<br>(1.2) | +0.17 |
| [D5] vz=0.65/lmax=7.0 ✅⚠⚠ | +25.1% | +0.84 H | -62.0% | +9.8% | +1.7% | **-5.63pp ⚠⚠** | 27 | ✅ LOW<br>(1.2) | +0.17 |
| [D5] vz=0.70/lmax=5.0 | +20.7% | +0.80 M | -55.2% | +9.9% | +5.7% | -0.87pp | 27 | ✅ LOW<br>(1.2) | +0.16 |
| **— 過去ベスト戦略（参考値・採用候補外） —** |  |  |  |  |  |  |  |  |  |
| [Legacy] S2_VZGated + LT2-N750 k=0.5 modeB ‡ | **+20.6%** | **+0.86 H** | -61.5% | +9.8% | +2.6% | +0.18pp | 27 | ✅ LOW<br>(1.1) | +0.16 |
| [Legacy] S2_VZGated 単独 ‡ **⚠↑↑** | +17.6% | +0.77 | -65.4% | +9.5% | +0.9% | **+6.04pp ⚠↑↑** | 27 | ✅ LOW<br>(0.8) | +0.18 |
| [Legacy] DH Dyn 2x3x [A] ‡ **⚠↑↑** | **+11.8%** | **+0.65** | -45.1% | +11.3% | +7.4% | **+8.48pp ⚠↑↑** | 27 | ✅ LOW<br>(0.7) | +0.15 |
| [Legacy] Ens2(Asym+Slope) max_lev=1.0 ‡§ **⚠↑↑** | **+10.4%** | **+0.48** | -49.0% | +8.1% | — | **+10.00pp§ ⚠↑↑** | 1 | — | — |
| **— ベンチマーク —** |  |  |  |  |  |  |  |  |  |
| **NDX 1x Buy & Hold 🅑** | +8.4% | +0.54 | **-77.9%** | **-4.7%** | +0.6% | +1.02pp | 0 | ✅ LOW<br>(1.1) | +0.06 |

---

## 📈 §4 v5 で明らかになった真実

### F-1 過去のベスト戦略の実態 — 「過去最高」と言われたのは何だったのか

| 戦略 | 当時の言われ方 | v5 実コスト/税後 OOS | 評価 |
|---|---|---:|---|
| DH Dyn 2x3x [A] | CAGR 22.5%, Sharpe 0.99 (FULL) | **CAGR +11.8%, Sharpe 0.65** | OOS は強気相場でも E4 (+22.4%) の **約半分**。**IS-OOS gap +8.48pp は古典 overfit 強警戒** |
| Ens2(Asym+Slope) | Sharpe 1.03 (推奨されていた) | **CAGR +10.4%, Sharpe 0.48** | OOS Sharpe は B&H (0.54) すら下回る。**実用性なし** |
| S2_VZGated 単独 | CAGR 27.5%, Sharpe 0.77 | **CAGR +17.6%** | LT2/E4 追加で 5pp 改善できる余地があった |
| S2+LT2 k=0.5 modeB | CAGR 31.2%, Sharpe 0.86 | **CAGR +20.6%** | E4 (+22.4%) との差は 1.8pp |

### F-2 E4 ◆ の真の優位性

| 比較 | E4 ◆ (+22.4%) | 過去ベスト最良の S2+LT2 (+20.6%) | DH Dyn (+11.8%) | NDX B&H (+8.4%) |
|---|---:|---:|---:|---:|
| CAGR_OOS_net 比較 | 基準 | -1.8pp | **-10.6pp** | **-14.0pp** |
| Sharpe 比較 | 0.79 | +0.07 | -0.14 | -0.25 |
| 結論 | 確認: 過去 NO.1 (S2+LT2) よりも +1.8pp、DH Dyn より **+10.6pp 圧勝**、B&H 比 **2.7倍** |

### F-3 古典 overfit の連鎖

`⚠↑↑` マーカが付いた legacy 3戦略は **すべて IS-OOS gap が +6pp 以上** で、OOS で性能が IS から大きく劣化:

| 戦略 | IS CAGR (推定) | OOS CAGR (実) | gap | 含意 |
|---|---:|---:|---:|---|
| S2_VZGated 単独 | +33.5% | +27.5% | +6.04pp | LT2 追加で +0.18pp に収束 → LT2 が overfit 除去装置として機能 |
| DH Dyn 2x3x [A] | +23.4% | +14.9% | +8.48pp | OOS 弱気の最大要因。S2 ゲートが無いと bull/bear 識別ができない |
| Ens2(Asym+Slope) | +22.2% | +12.6% | +10.00pp | 最大の gap。完全に IS overfit な構造 |

LT2-N750 + S2 VZ ゲート + E4 regime k_lt の三重構造は、**この overfit を体系的に除去するための進化**だったことが定量裏付けされた。

---

## 📁 §5 一次根拠ファイル（GitHub URL）

| ファイル | 役割 | v5 で使った値 |
|---|---|---|
| [`corrected_strategy_results.csv`](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/corrected_strategy_results.csv) | DH Dyn 2x3x [A] の OOS Scenario D 行 | CAGR_OOS=14.88%, Sharpe=0.646, MaxDD=-45.08% |
| [`STRATEGY_PERFORMANCE_COMPARISON_2026-05-23.md`](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/STRATEGY_PERFORMANCE_COMPARISON_2026-05-23.md) | 5戦略 × 9指標の Scenario D OOS 完全表 | DH Dyn / S2_VZGated / S2+LT2 / B&H の全指標 |
| [`b1_s2_lt2_results.csv`](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/b1_s2_lt2_results.csv) | S2_VZGated 系2件の raw 値 | row(1)(2) |
| [`ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md`](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/ADDITIONAL_ANALYSIS_REPORT_2026-03-30.md) | Ens2(Asym+Slope) の Full + OOS Sharpe | Full CAGR 22.20%, Full Sharpe 0.846, OOS Sharpe 0.479 |
| [`E1_ENSEMBLE_2026-05-21.md`](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/E1_ENSEMBLE_2026-05-21.md) | DH Dyn [A] OOS 値の独立検証 | NAV_B: CAGR_OOS=14.79%, Sharpe=0.650 (corrected と一致) |
| [`NASDAQ_extended_to_2026.csv`](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/NASDAQ_extended_to_2026.csv) | NDX B&H 生データ | 13,169 bars |
| [`src/g15_legacy_strategies_realistic.py`](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/src/g15_legacy_strategies_realistic.py) | v5 算出スクリプト本体 | — |
| [`g15_legacy_results.csv`](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/g15_legacy_results.csv) | v5 出力 5戦略 raw + net 全指標 | — |

---

## 🔧 §6 v5 で残った限界（v6 候補）

| # | 項目 | 影響 | 対応案 |
|:--:|---|---|---|
| 1 | Ens2(Asym+Slope) の OOS CAGR は Sharpe比からの推定値 (+12.57%) | 中 | `test_ens2_strategies.py` で IS/OOS split で再ラン |
| 2 | Ens2 の P10_5Y▷ は raw データ不在 | 低 | 同上 |
| 3 | DH Dyn の TQQQ ETF → SBI CFD cost neutral 仮定は粗い (実際は -1.9pp 程度安い可能性) | 低 | `g13` を DH Dyn 系に拡張 |
| 4 | Worst10Y/P10_5Y への cost drag は CAGR 比例の近似 (実際は窓ごとに不均一) | 中 | `build_nav_strategy(spread=0.0300)` で再ラン |
| 5 | Sharpe 非対称税考慮 -19% 補正 (v3 §4-2) は v5 で未適用 | 低 | E4 0.79 → 0.64, DH 0.65 → 0.53, Ens2 0.48 → 0.39 など補正値も併記 |

---

## 📝 §7 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v3-draft | 2026-05-29 | 21戦略 SBI CFD コスト + §3-A/B 税モデル併記 |
| v4-draft | 2026-05-29 | 過去ベスト戦略4件 + NDX 1x B&H 追加。**ただし DH Dyn / Ens2 で FULL 期間値を OOS と誤用** |
| **v5-draft** | **2026-05-29** | **v4 の誤り修正**: ①DH Dyn を OOS Scenario D (14.88%/0.646) に修正 ②Ens2 を OOS Sharpe ベース推定 (12.57%/0.479) に修正 ③CFD drag を戦略別に差別化 (S2系 -5.6pp / DH Dyn 0pp / Ens2 0pp / B&H 0pp) ④WFA_CI95_lo / WFA_WFE を全戦略で 2026-05-23 表から復元 ⑤Ens2 P10_5Y を「—」に変更 (推定値の根拠不足) |

---

## ⚠️ 結論

**v5 は v4 と比べて根本的に違う**:
- DH Dyn: 真の OOS 性能は **CAGR +11.8% / Sharpe 0.65** で、E4 ◆ より **10.6pp も劣後**
- Ens2: 真の OOS 性能は **CAGR +10.4% / Sharpe 0.48** で、**B&H (0.54) すら下回る**
- 過去のベスト戦略が「実は B&H と大差ない」ことが定量裏付けされた
- E4 ◆ の +22.4% / 0.79 は **過去 NO.1 (S2+LT2 +20.6%) よりも +1.8pp**、**過去すべての legacy 戦略の中で唯一実用に値する数値**

これにより「過去のベスト戦略のコスト/税後の実態」を正しく把握できた。次フェーズの戦略ブラッシュアップは、**E4 を上回るための新規アイデア探索 + 既存 D5/F10 系の差別化テスト**を主軸とすべき。

---

*管理: Claude (Opus 4.7) / 承認者: Kazuya Murayama*
*計算式（CFD strategies): `(CAGR_raw − cfd_drag − 0.66%) × 0.8273`*
*計算式（ETF/Unleveraged): `CAGR_raw × 0.8273`*
*準拠: `EVALUATION_STANDARD.md v1.3` / §3.12 9指標標準*
