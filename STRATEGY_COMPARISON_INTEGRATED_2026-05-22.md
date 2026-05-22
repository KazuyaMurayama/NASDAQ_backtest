# 12戦略 統合比較レポート — 統一9指標フレームワーク

作成日: 2026-05-22
最終更新日: 2026-05-22
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**
**▶ このレポートは `STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md` を置換します（SUPERSEDED）**

---

## §1 期間・コスト前提

| 項目 | 定義 |
|------|------|
| **IS** | 1974-01-02 〜 2021-05-07 (47.3年, ~11,916 bars) |
| **OOS** | 2021-05-08 〜 2026-03-26 (4.9年, ~1,253 bars) |
| **FULL** | 1974-01-02 〜 2026-03-26 (52.26年, 13,169 bars) |
| **コスト** | Scenario D (TER + sofr_multiplier×SOFR + swap_spread) |
| **コスト参照** | `src/product_costs.py` 2026-05-12 基準 |
| **DELAY** | 2営業日 (look-ahead bias 対策) |
| **Sharpe Rf** | 0 (高金利期の過大評価注意) |
| **評価指標** | 統一9指標 (`docs/rules/08_evaluation-metrics.md` §1) |

---

## §2 統合比較表 (12戦略 × 9指標)

> **単位**: CAGR / Worst10Y★ / P10▷ / MaxDD = %、IS-OOS gap = pp (CAGR_IS − CAGR_OOS)、Tr = 回/年
> CAGR_FULL = `(1+CAGR_IS)^(11916/13169) × (1+CAGR_OOS)^(1253/13169) − 1` より算出
> ◆ = 現行ベスト | ✅ = Shortlisted | ‡ = P-series §1.3 参考値（異コストモデル・直接比較不可）

| # | 戦略 | CAGR<br>IS | CAGR<br>OOS | CAGR<br>FULL | Sh<br>OOS | Max<br>DD | W10<br>★ | P10<br>▷ | IS-OOS<br>gap | Tr |
|---|------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **S2+LT2-N1500 ◆** | +30.8 | **+30.8** | +30.8 | **0.885** | -63.4 | +16.6 | +9.9 | **-0.1** | 27 |
| 2 | S2+LT2-N750 ✅ | +31.3 | +31.2 | +31.3 | 0.858 | -59.5 | +18.1 | +9.4 | +0.2 | 27 |
| — | — | — | — | — | — | — | — | — | — | — |
| 3 | S2_VZGated | +32.9 | +27.6 | +32.4 | 0.769 | -62.4 | +17.7 | +7.3 | +5.4 | 27 |
| 4 | P2_volTgt | +34.6 | +27.1 | +33.9 | 0.757 | -60.5 | +19.1 | +8.5 | +7.5 | 27 |
| 5 | S4_RelVol | +41.0 | +26.2 | +39.5 | 0.697 | -66.1 | +19.5 | +11.5 | +14.8 | 27 |
| 6 | CFD-7x | +43.4 | +24.4 | +41.4 | 0.670 | -65.0 | +25.4 | +12.7 | +18.9 | 27 |
| 7 | DH[A+LT2] | +22.1 | +18.9 | +21.8 | 0.777 | -44.8 | +13.4 | +9.7 | +3.3 | 27 |
| 8 | DH[A] | +23.4 | +14.9 | +22.5 | 0.646 | -45.1 | +14.3 | +9.6 | +8.5 | 27 |
| — | — | — | — | — | — | — | — | — | — | — |
| 9 | BH-1x | +11.1 | +10.1 | +11.0 | 0.540 | -77.9 | -5.7 | +0.7 | +1.0 | 0 |
| — | — | — | — | — | — | — | — | — | — | — |
| 10 | P02-CPI ‡ | +22.2 | +19.4 | +21.9 | 0.833 | -46.4 | +7.5 | +8.5 | +2.8 | 27 |
| 11 | P01-HY ‡ | +22.2 | +19.9 | +22.0 | 0.829 | -42.9 | +8.5 | +8.2 | +2.3 | 27 |
| 12 | P05-HY ‡ | +25.9 | +15.7 | +24.9 | 0.667 | -45.0 | +11.2 | +11.1 | +10.3 | 27 |

---

## §3 戦略名・記号 凡例

| 省略名 | 正式名 | 出典 |
|--------|--------|------|
| S2+LT2-N1500 ◆ | S2_VZGated + LT2-N1500-k0.5-modeB | B6 N-sweep (2026-05-22) |
| S2+LT2-N750 ✅ | S2_VZGated + LT2-N750-k0.5-modeB | B1 (2026-05-21) |
| S2_VZGated | S2_VZGated (tv=0.8, k=0.3, gate=0.5, l_max=7) | A1-A6 sweep 最適値 |
| P2_volTgt | P2 vol-target (tv=0.8) | A2 sweep 最適値 |
| S4_RelVol | S4_RelVol (l_base=7, k_rel=2.0) | S1 sweep 最適値 |
| CFD-7x | CFD 7x固定 (DH Dyn+7x) | 固定レバレッジ参照 |
| DH[A+LT2] | DH Dyn 2x3x [A+LT2] (LT2-N750-k0.5-modeB) | B1 (2026-05-21) |
| DH[A] | DH Dyn 2x3x [A] (TQQQ, Scenario D, th=0.15) | corrected_strategy_backtest.py |
| BH-1x | BH 1x (NASDAQ 買持ち) | ベンチマーク |
| P02-CPI ‡ | P02_Dyn×CPI [mult] (Best 2022防御) | P4系・参考値 |
| P01-HY ‡ | P01_Dyn×HY [mult] (Best DSR候補) | P4系・参考値 |
| P05-HY ‡ | P05_HY×CPI [mult] (Best Worst5Y secondary) | P4系・参考値 |

> **‡ P-series の参考値扱い根拠** (`EVALUATION_STANDARD.md §1.3`):
> P01/P02/P05 は `timing_signals_raw.csv` の HY スプレッド / CPI 前年比シグナルを使用する
> 異コストモデル戦略。Scenario D と直接比較不可。CAGR_OOS の比較は参考値として参照のみ。

---

## §4 現行ベスト vs 旧ベスト — N1500 vs N750 詳細比較

> 2026-05-22 B6 N-sweep にて N=1500 を採用。根拠: `CURRENT_BEST_STRATEGY.md`

| 評価軸 | S2+LT2-N1500 ◆ | S2+LT2-N750 ✅ | 優位 |
|--------|---------------:|---------------:|:----:|
| Sharpe_OOS | **0.885** | 0.858 | N1500 ◎ |
| IS-OOS gap | **−0.1 pp** | +0.2 pp | N1500 ◎◎ |
| CAGR_OOS | +30.8% | **+31.2%** | N750（差 −0.4pp は誤差範囲） |
| MaxDD | -63.4% | **-59.5%** | N750（差 −3.9pp） |
| Worst10Y★ | +16.6% | **+18.1%** | N750（差 −1.5pp） |
| P10_5Y▷ | +9.9% | +9.4% | N1500（差 +0.5pp） |

> **採用判断**: Sharpe_OOS で全戦略中最高値 (+0.885)。IS-OOS gap が -0.1pp (OOS が IS を上回る)
> は本プロジェクト史上最高の汎化性。MaxDD / Worst10Y★ の劣後は guardrail 内
> (MaxDD > -70%, Worst10Y★ > +15% をともに満たす)。

---

## §5 9指標準拠確認 (EVALUATION_STANDARD v1.1)

本表で使用した指標と EVALUATION_STANDARD §3 の対応:

| §参照 | 指標 | 本表の列 | 充足 |
|-------|------|----------|------|
| §3.1 | CAGR_IS / CAGR_OOS / CAGR_FULL | CAGR IS / OOS / FULL | ✅ |
| §3.2 | Sharpe_OOS (Rf=0) | Sh OOS | ✅ |
| §3.3 | MaxDD (FULL) | Max DD | ✅ |
| §3.5 | Worst10Y★ (カレンダー年) | W10 ★ | ✅ |
| §3.6 | P10_5Y▷ | P10 ▷ | ✅ |
| §3.7 | Trades/yr | Tr | ✅ |
| §3.8 | IS-OOS gap | IS-OOS gap | ✅ |
| §3.9 | WFA_CI95_lo | — (戦略比較表対象外、G1_WFA参照) | N/A |
| §3.10 | WFA_WFE | — (戦略比較表対象外、G1_WFA参照) | N/A |

**禁止指標**: Stable_Sharpe / WinRate_yr / WorstK5_mean_CAGR / IR_vs_BH いずれも使用なし ✅
**Worst5Y(FULL)**: 旧レポート (2026-05-19) 記載。9指標標準外のため本表から除外。

---

## §6 戦略カテゴリ別整理

| カテゴリ | 戦略 | 採用状況 |
|----------|------|----------|
| **◆ 現行ベスト** | S2+LT2-N1500 | Active (2026-05-22採用) |
| **✅ Shortlisted (代替候補)** | S2+LT2-N750 | Shortlisted (CAGR/MaxDD で優位、代替保持) |
| **CFD軸 (動的CFDレバレッジ)** | S2_VZGated, P2_volTgt, S4_RelVol, CFD-7x | 構成要素・比較対象 |
| **DH Dyn軸 (ETF 動的配分)** | DH[A], DH[A+LT2] | 旧推奨・比較対象 |
| **ベンチマーク** | BH-1x | 基準値 |
| **参考値 (§1.3)** | P02-CPI, P01-HY, P05-HY | 参考値のみ・採用候補外 |

---

## §7 一次根拠ファイル

| ファイル | 役割 |
|----------|------|
| [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) | 現行ベスト単一の真実 |
| [B6_S2_LT2_N_SWEEP_2026-05-22.md](B6_S2_LT2_N_SWEEP_2026-05-22.md) | N1500 採用根拠 (N-sweep 6 config) |
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | S2+LT2 検証・N750 参照値 |
| [G1_WFA_2026-05-21.md](G1_WFA_2026-05-21.md) | WFA 補助2指標 (CI95_lo / WFE) |
| [EVALUATION_STANDARD.md](EVALUATION_STANDARD.md) | 評価基準 正典 (v1.1) |
| [docs/rules/08_evaluation-metrics.md](docs/rules/08_evaluation-metrics.md) | 9指標ルール |
| [b6_s2_lt2_N_sweep_results.csv](b6_s2_lt2_N_sweep_results.csv) | N1500 数値ソース |
| [b1_s2_lt2_results.csv](b1_s2_lt2_results.csv) | S2/S2+LT2/DH[A+LT2] 数値ソース |
| [src/b1_s2_lt2.py](src/b1_s2_lt2.py) | S2+LT2 実装 |
| [src/product_costs.py](src/product_costs.py) | コスト定数 単一の真実 |

---

## §8 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|----------|
| v2.0 | 2026-05-22 | **統一9指標フレームワーク適用** (EVALUATION_STANDARD v1.1)。N1500を現行ベストとして追加。Worst5Y除外、CAGR_FULL・IS-OOS gap追加。12戦略に拡張。旧 2026-05-19 版を SUPERSEDED。 |
| v1.x | 2026-05-19 | 初版 (10戦略、旧指標構成)。[STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) を参照。 |

---

*管理者: Kazuya Murayama*
*準拠: `EVALUATION_STANDARD.md v1.1` / `docs/rules/08_evaluation-metrics.md v1.0`*
