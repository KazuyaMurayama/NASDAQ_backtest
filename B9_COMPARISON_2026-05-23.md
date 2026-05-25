# B9勝者 vs 現行ベスト 比較レポート — 統一9指標フレームワーク

作成日: 2026-05-23
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

---

## §1 比較前提

| 項目 | 定義 |
|------|------|
| **IS** | 1974-01-02 〜 2021-05-07 (47.3年, ~11,916 bars) |
| **OOS** | 2021-05-08 〜 2026-03-26 (4.9年, ~1,253 bars) |
| **FULL** | 1974-01-02 〜 2026-03-26 (52.26年, 13,169 bars) |
| **コスト** | Scenario D (`src/product_costs.py` 2026-05-12 基準) |
| **DELAY** | 2営業日 |
| **CURRENT_BEST 整合** | YES (REF=S2+LT2-N750-k0.5-modeB) |

---

## §2 9指標比較表 (3戦略 × 9指標)

> **単位**: CAGR_OOS / Worst10Y★ / P10▷ / MaxDD = %、IS-OOS gap = pp、Tr = 回/年
> ★ = Sharpe_OOS > +0.885 / ◎ = > +0.770
> ◆ = 現行ベスト | ✅ = Shortlisted（WFA待ち）| ⚠ = OOS期間バイアス警戒

| Strategy | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:---------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| **B9-Winner (gf=0.65, wn_min=0.20) ✅⚠** | +35.9% | +0.94 ★ | -63.3% | +16.1% |  +8.1% | -5.05pp |  27 | +25.8% | +1.304 |
| **B9-Stable (gf=0.60, wn_min=0.30) ✅** | +34.2% | +0.91 ★ | -62.1% | +16.9% |  +8.6% | -3.22pp |  27 | +25.7% | +1.247 |
| **S2+LT2-N750 ◆** | +31.2% | +0.86 ◎ | -59.5% | +18.1% |  +9.4% | +0.18pp |  27 | +25.7% | +1.145 |

*◎ Sharpe_OOS > +0.770（S2ベースライン超過）/ ★ > +0.885（現行ベスト超過）。進格: CI95_lo > 0 かつ 0.5 ≤ WFE ≤ 2.0。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカの ◎/★ とは別意味）。*

*CI95_lo / WFE: G2 WFA 実測値（`src/g2_wfa_b9.py`, 2026-05-23）。B9-Winner/B9-Stable ともに α+β PASS。*

---

## §3 各指標 有利/不利 判定

| 指標 | REF (N750) | B9勝者 | 差 | 評価 |
|---|---:|---:|---:|---|
| CAGR_OOS | +31.16% | +35.93% | +4.78pp | B9勝者 ◎ |
| Sharpe_OOS | +0.858 | +0.944 | +0.086 | B9勝者 ◎ PASS基準超過 |
| MaxDD | -59.45% | -63.33% | -3.88pp | REF 優位（guardrail -64.45% は満たす） |
| Worst10Y★ | +18.10% | +16.12% | -1.97pp | REF 優位（guardrail +15% は満たすが余裕縮小） |
| P10_5Y▷ | +9.36% | +8.09% | -1.27pp | REF 優位 |
| IS-OOS gap | +0.18pp | -5.05pp | -5.23pp | **要解釈**（Gold OOS期間バイアス疑い、§4参照） |
| Trades/yr | 27 | 27 | 0 | 同等 |
| WFA_CI95_lo | +25.73% | +25.78% | +25.70% | G2 WFA 実測（2026-05-23） |
| WFA_WFE | +1.145 | +1.304 | +1.247 | G2 WFA 実測（2026-05-23） |

---

## §4 採用判断の論拠（WFA完了前）

### (a) IS-OOS gap -5.05pp の解釈
OOS期間（2021-2026）はGold ETFが累積+60%超の強気相場。F1a sweep（`F1_ALLOC_SWEEP_2026-05-21.md`）では、gold_frac を 0.20→0.80 と単調に変化させると IS-OOS gap が +9.86pp→ -10.23pp と単調減少する。これは戦略の汎化性改善ではなく**OOS期間限定のGold強気エクスポージャ**によるものと判断される。

### (b) リスク3指標の同時悪化
- MaxDD: -59.45% → -63.33%（-3.88pp）— guardrail -64.45% まで余裕 +1.12pp
- Worst10Y★: +18.10% → +16.12%（-1.97pp）— guardrail +15.0% まで余裕 +1.12pp
- P10_5Y▷: +9.36% → +8.09%（-1.27pp）— 実運用の「悪い5年」期待値後退

N1500→N750 復元の判断（2026-05-22）と同一の論理で却下対象。

### (c) 総合判定: REF 維持・Shortlisted 登録
WFA（Walk-Forward Analysis）完了まで**REF=S2+LT2-N750 を維持**。
B9-Winner / B9-Stable は Shortlisted（WFA待ち）として保留。
WFA で CI95_lo > 0 かつ 0.5 ≤ WFE ≤ 2.0 を満たした場合のみ昇格を再検討。

---

## §5 推奨アクション

1. **CURRENT_BEST_STRATEGY.md**: 更新不要（REF=N750維持）。§変更履歴に以下を追記:
   > 2026-05-23: B9 (gold_frac×wn_min 2D sweep) で gf=0.65 が Sharpe_OOS +0.944 を記録(PASS)。ただし IS-OOS gap=-5.05ppはGold 2021-2026強気エクスポージャ偏重の疑いあり、リスク3指標が同時悪化のためWFA完了まで Shortlisted 保留・REF維持。
2. **STRATEGY_COMPARISON_INTEGRATED**: B9-Winner / B9-Stable を Shortlisted 行として §2 に追加（別タスクで実施）。
3. **WFA キュー**: B9-Winner / B9-Stable を `g2_wfa_shortlist.py` に追加（次フェーズ）。

---

## §6 一次根拠ファイル

| ファイル | 役割 |
|----------|------|
| [B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md](B9_S2LT2_GOLDFRAC_SWEEP_2026-05-23.md) | B9 2Dスイープ結果（28 configs） |
| [B1_S2_LT2_2026-05-21.md](B1_S2_LT2_2026-05-21.md) | REF(N750)確定値 |
| [F1_ALLOC_SWEEP_2026-05-21.md](F1_ALLOC_SWEEP_2026-05-21.md) | gold_frac 1D感度（OOSバイアス参照） |
| [CURRENT_BEST_STRATEGY.md](CURRENT_BEST_STRATEGY.md) | 現行ベスト単一の真実 |
| [b9_s2lt2_goldfrac_results.csv](b9_s2lt2_goldfrac_results.csv) | B9数値ソース |
| [b1_s2_lt2_results.csv](b1_s2_lt2_results.csv) | REF数値ソース |

---

## §7 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v1.1 | 2026-05-23 | G2 WFA 実測値を §2/§3 テーブルに反映。B9-Winner CI95_lo=+25.8%, WFE=+1.304 (PASS)。B9-Stable CI95_lo=+25.7%, WFE=+1.247 (PASS)。 |
| v1.0 | 2026-05-23 | 初版。B9勝者・B9安定をShortlisted保留、REF=N750維持を決定。 |

---

*管理者: 男座員也（Kazuya Oza）*
*準拠: `EVALUATION_STANDARD.md v1.1` / `src/_sweep_format.py MD_HEADER_STRAT`*
