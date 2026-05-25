# 14戦略 統合比較レポート — 統一9指標フレームワーク

作成日: 2026-05-22
最終更新日: 2026-05-23 (G2 WFA実測値反映)
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

## §2 統合比較表 (14戦略 × 9指標)

> **単位**: CAGR_OOS / Worst10Y★ / P10▷ / MaxDD = %、IS-OOS gap = pp、Tr = 回/年
> §3.12 統一9指標標準: MDテーブルは CAGR_OOS の1列のみ。CAGR_IS / CAGR_FULL は CSV 専用。
> ◆ = 現行ベスト | ✅ = Shortlisted | ‡ = P-series §1.3 参考値（異コストモデル・直接比較不可）

| # | 戦略 | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|--:|:-----|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
|  1 | **S2+LT2-N750 ◆** | **+31.2%** | **+0.858** | -59.5% | +18.1% |  **+9.4%** | +0.20pp | 27 | +25.7% | +1.145 |
|  2 | S2+LT2-N1500 ✅ | +30.8% | +0.885 | -63.4% | +16.6% |  +9.9% | -0.10pp | 27 |   —   |   —   |
|  3 | B9-Winner (gf=0.65) ✅⚠ | +35.9% | +0.944 | -63.3% | +16.1% |  +8.1% | -5.05pp | 27 | +25.8% | +1.304 |
|  4 | B9-Stable (gf=0.60) ✅ | +34.2% | +0.914 | -62.1% | +16.9% |  +8.6% | -3.22pp | 27 | +25.7% | +1.247 |
|  — | — | — | — | — | — | — | — | — | — | — |
|  5 | S2_VZGated | +27.6% | +0.769 | -62.4% | +17.7% |  +7.3% | +5.40pp | 27 |   —   |   —   |
|  6 | P2_volTgt | +27.1% | +0.757 | -60.5% | +19.1% |  +8.5% | +7.50pp | 27 |   —   |   —   |
|  7 | S4_RelVol | +26.2% | +0.697 | -66.1% | +19.5% | +11.5% | +14.80pp | 27 |   —   |   —   |
|  8 | CFD-7x | +24.4% | +0.670 | -65.0% | +25.4% | +12.7% | +18.90pp | 27 |   —   |   —   |
|  9 | DH[A+LT2] | +18.9% | +0.777 | -44.8% | +13.4% |  +9.7% | +3.30pp | 27 |   —   |   —   |
| 10 | DH[A] | +14.9% | +0.646 | -45.1% | +14.3% |  +9.6% | +8.50pp | 27 |   —   |   —   |
|  — | — | — | — | — | — | — | — | — | — | — |
| 11 | BH-1x | +10.1% | +0.540 | -77.9% |  -5.7% |  +0.7% | +1.00pp |  0 |   —   |   —   |
|  — | — | — | — | — | — | — | — | — | — | — |
| 12 | P02-CPI ‡ | +19.4% | +0.833‡ | -46.4%‡ |  +7.5% |  +8.5% | +2.80pp | 27 |   —   |   —   |
| 13 | P01-HY ‡ | +19.9% | +0.829‡ | -42.9%‡ |  +8.5% |  +8.2% | +2.30pp | 27 |   —   |   —   |
| 14 | P05-HY ‡ | +15.7% | +0.667‡ | -45.0%‡ | +11.2% | +11.1% | +10.30pp | 27 |   —   |   —   |


*◎ Sharpe_OOS > +0.770（S2ベースライン超過）/ ★ > +0.885（現行ベスト超過）。進格: CI95_lo > 0 かつ 0.5 ≤ WFE ≤ 2.0。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカの ◎/★ とは別意味）。*
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
| B9-Winner ✅⚠ | S2+LT2-N750 + alloc(gf=0.65, wn_min=0.20) | B9 (2026-05-23) — Gold-overfit 疑い |
| B9-Stable ✅ | S2+LT2-N750 + alloc(gf=0.60, wn_min=0.30) | B9 (2026-05-23) — F1-BEST と同設定 |

> **⚠ 注**: IS-OOS gap < -3pp は Gold 2021-2026 強気エクスポージャ偏重の可能性。`B9_COMPARISON_2026-05-23.md §4-(a)` 参照。

> **‡ P-series の参考値扱い根拠** (`EVALUATION_STANDARD.md §1.3`):
> P01/P02/P05 は `timing_signals_raw.csv` の HY スプレッド / CPI 前年比シグナルを使用する
> 異コストモデル戦略。Scenario D と直接比較不可。CAGR_OOS の比較は参考値として参照のみ。

---

## §4 現行ベスト vs B9候補 — Sharpe vs リスクのトレードオフ

> 2026-05-23 B9 2D sweep にて gold_frac=0.65, wn_min=0.20 を最有力候補として特定。根拠: `B9_COMPARISON_2026-05-23.md`

| 評価軸 | S2+LT2-N750 ◆ | B9-Winner (gf=0.65) ✅⚠ | 優位 |
|--------|---------------:|---------------:|:----:|
| Sharpe_OOS | 0.858 | **0.944** | B9勝者 ◎ |
| IS-OOS gap | +0.20pp | -5.05pp | B9勝者（ただし Gold OOS バイアス疑い） |
| CAGR_OOS | +31.2% | **+35.9%** | B9勝者 ◎ |
| MaxDD | **-59.5%** | -63.3% | N750（差 -3.8pp） |
| Worst10Y★ | **+18.1%** | +16.1% | N750（差 -2.0pp） |
| P10_5Y▷ | **+9.4%** | +8.1% | N750（差 -1.3pp） |

> **採用判断**: B9勝者は Sharpe_OOS/CAGR_OOS で大幅改善を示すが、IS-OOS gap=-5.05pp は F1a sweep の gold_frac 単調パターンと整合し、OOS期間（2021-2026）のGold強気に依存した偏り（Gold-overfit）の疑いがある（`B9_COMPARISON_2026-05-23.md §4` 参照）。MaxDD・Worst10Y★・P10_5Yが同時悪化。WFA完了まで REF（N750）を維持。

---

## §5 9指標準拠確認 (EVALUATION_STANDARD v1.1)

本表で使用した指標と EVALUATION_STANDARD §3 の対応:

| §参照 | 指標 | 本表の列 | 充足 |
|-------|------|----------|------|
| §3.1 | CAGR_OOS（MD専用1列） | CAGR_OOS | ✅ |
| §3.1' | CAGR_IS / CAGR_FULL（CSV専用） | — (b6_s2_lt2_N_sweep_results.csv 等) | ✅ |
| §3.2 | Sharpe_OOS (Rf=0) | Sharpe_OOS | ✅ |
| §3.3 | MaxDD (FULL) | MaxDD | ✅ |
| §3.5 | Worst10Y★ (カレンダー年) | Worst10Y★ CAGR | ✅ |
| §3.6 | P10_5Y▷ | P10 5Y▷ CAGR | ✅ |
| §3.7 | Trades/yr | Trade（回/年） | ✅ |
| §3.8 | IS-OOS gap | IS-OOS gap | ✅ |
| §3.9 | WFA_CI95_lo | CI95_lo（G1: N750/N1500 実測済み、G2: B9-Winner/B9-Stable 実測済み） | ✅ |
| §3.10 | WFA_WFE | WFE（G1: N750/N1500 実測済み、G2: B9-Winner/B9-Stable 実測済み） | ✅ |

**禁止指標**: Stable_Sharpe / WinRate_yr / WorstK5_mean_CAGR / IR_vs_BH いずれも使用なし ✅
**Worst5Y(FULL)**: 旧レポート (2026-05-19) 記載。9指標標準外のため本表から除外。

---

## §6 戦略カテゴリ別整理

| カテゴリ | 戦略 | 採用状況 |
|----------|------|----------|
| **◆ 現行ベスト** | S2+LT2-N750 | Active (2026-05-22 復元) |
| **✅ Shortlisted (WFA 待ち)** | S2+LT2-N1500, B9-Winner, B9-Stable | WFA CI95_lo/WFE 計算後に再評価 |
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
| [B9_COMPARISON_2026-05-23.md](B9_COMPARISON_2026-05-23.md) | B9 vs 現行ベスト 3戦略詳細比較 |
| [B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md](B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md) | B9 2D sweep 28 configs |

---

## §8 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|----------|
| v2.3 | 2026-05-23 | G2 WFA 実測値を §2 テーブルに反映。B9-Winner CI95_lo=+25.8%/WFE=+1.304, B9-Stable CI95_lo=+25.7%/WFE=+1.247, N750 REF CI95_lo=+25.7%/WFE=+1.145 (全3戦略 PASS)。§5 準拠確認も更新。 |
| v2.2 | 2026-05-23 | B9 2D sweep 結果反映。B9-Winner(gf=0.65)/B9-Stable(gf=0.60) をShortlistedとして§2に追加(合計14戦略)。◆をN750に戻しN1500を✅に降格（CURRENT_BEST_STRATEGY.md 整合）。B9勝者はGold OOSバイアス疑いのためWFA完了まで保留。 |
| v2.1 | 2026-05-22 | **§3.12 9指標標準準拠化**: §2 統合比較表から CAGR_IS / CAGR_FULL 列を削除（MDは CAGR_OOS 1列のみ）。WFA_CI95_lo / WFE 列を追加（全戦略 `—`）。列順を `_sweep_format.MD_HEADER_STRAT` に統一。§5 9指標準拠確認も更新。 |
| v2.0 | 2026-05-22 | **統一9指標フレームワーク適用** (EVALUATION_STANDARD v1.1)。N1500を現行ベストとして追加。Worst5Y除外、CAGR_FULL・IS-OOS gap追加。12戦略に拡張。旧 2026-05-19 版を SUPERSEDED。 |
| v1.x | 2026-05-19 | 初版 (10戦略、旧指標構成)。[STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md](STRATEGY_COMPARISON_INTEGRATED_2026-05-19.md) を参照。 |

---

*管理者: 男座員也（Kazuya Oza）*
*準拠: `EVALUATION_STANDARD.md v1.1` / `docs/rules/08_evaluation-metrics.md v1.0`*
