# Signal × Strategy Integration — Final Report (Session S3)

作成日: 2026-06-05
最終更新日: 2026-06-05

セッション: S3 — 9+1 指標完全評価 + Tier 1 再判定 + 集中 Tier 2/3
コミット起点: `6711bd6` (Tier 1 緩和判定)
プラン典拠: `SIGNAL_INTEGRATION_PLAN_20260604.md`
評価フレーム: `docs/rules/08_evaluation-metrics.md` 9+1 指標 (Trades/yr, WFE, CI95_lo は NAV プロキシ)

---

## 1. サマリー

| 項目 | 値 |
|---|---|
| 総評価パターン数 | **117** |
| 内訳 (Tier 1 全列挙) | 72 |
| 内訳 (Tier 2 集中) | 27 |
| 内訳 (Tier 3 集中) | 18 |
| 対象戦略 | S1 (LT2-N750), S2 (S2_VZGated), S3 (DH_W1) |
| 対象シグナル (Phase B 通過 6 本) | #6 VIX, #21 HY OAS, #23 BAA-10Y, #26 2s10s, #28 10Y real, #41 DXY |
| 評価指標 | 9+1 (CAGR_OOS, IS-OOS gap, Sharpe, MaxDD, Worst10Y, P10_5Y, Trades/yr, WFE, CI95_lo) |

### 結論先出し

- **9+1 完全フレームで STRONG / STANDARD PASS は 0 件**。全 117 パターン中 105 が `MARGINAL_FULL`、12 が `FAIL_FULL`。
- 緩和判定 (relaxed, 6 軸) では **2 件のみ STANDARD_PASS_RELAXED** (どちらも S3 × BAA-10Y × M2_procyclical)。
- 「シグナル後注入によって既存 S1/S2/S3 を堅牢に強化する設計」は、本セッションのスクリーニングでは **採用閾値に到達せず**。
- 全パターンで Worst10Y_CAGR が劣化 (102/117 件)、すなわち**最悪 10 年区間を悪化させずに改善する方法は見つかっていない**。

---

## 2. Tier 別 9+1 完全フレーム判定

| Tier | パターン数 | STRONG_PASS_FULL | STANDARD_PASS_FULL | MARGINAL_FULL | FAIL_FULL |
|---|---:|---:|---:|---:|---:|
| Tier 1 (全 72) | 72 | 0 | 0 | 60 | 12 |
| Tier 2 (集中 27) | 27 | 0 | 0 | 27 | 0 |
| Tier 3 (集中 18) | 18 | 0 | 0 | 18 | 0 |
| **合計** | **117** | **0** | **0** | **105** | **12** |

### 戦略別ブレークダウン (judgment_full)

| Strategy | STRONG | STANDARD | MARGINAL | FAIL |
|---|---:|---:|---:|---:|
| S1 | 0 | 0 | 21 | 3 |
| S2 | 0 | 0 | 21 | 3 |
| S3 | 0 | 0 | 18 | 6 |

(Tier 1 のみカウント; Tier 2/3 集中分は全 MARGINAL_FULL のため列省略)

### Relaxed 判定 (緩い severe 閾値) との比較

| Judgment (relaxed, 6 軸) | 件数 |
|---|---:|
| STANDARD_PASS_RELAXED | **2** |
| MARGINAL_RELAXED | 99 |
| FAIL_RELAXED | 16 |

緩和フレームでさえ STANDARD_PASS は 2 件のみ。完全フレームに 3 軸 (Trades/yr 上限、WFE、CI95_lo) を追加した結果、その 2 件も Worst10Y 等で打ち消され `MARGINAL_FULL` に落ちた。

---

## 3. 集中 Tier 2 結果 (27 パターン)

設計:
- シグナル 3: #21 HY OAS, #23 BAA-10Y, #41 DXY
- 戦略 3: S1, S2, S3
- 手法 3: M2_procyclical (連続レバ tilt), M4_vol_adj (vol target), M5_filter_entry (高シグナル時のみ in)
- 計 27 = 3 × 3 × 3

| Judgment | 件数 |
|---|---:|
| MARGINAL_FULL | 27 |
| STRONG / STANDARD / FAIL | 0 |

**Top 3 (n_improved_full 降順、CAGR_OOS_diff タイブレーク)**:

| pattern_id | strategy | signal | method | n_imp | n_deg | CAGR_OOS diff | Sharpe diff | MaxDD diff | judgment_full |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| T2F_M5_41_S1_filter_entry | S1 | DXY | M5_filter_entry | 4 | 4 | +0.026 | +0.056 | +0.219 | MARGINAL_FULL |
| T2F_M5_41_S2_filter_entry | S2 | DXY | M5_filter_entry | 4 | 4 | +0.023 | +0.058 | +0.121 | MARGINAL_FULL |
| T2F_M5_41_S3_filter_entry | S3 | DXY | M5_filter_entry | 4 | 4 | +0.010 | +0.046 | +0.082 | MARGINAL_FULL |

観察:
- **M5_filter_entry × DXY が 3 戦略全てで Top 入り** (CAGR_OOS, Sharpe, MaxDD, WFE が同時改善)。
- ただし n_deg=4 (Worst10Y, P10_5Y, IS-OOS gap, CI95_lo が劣化)。**IS-OOS gap が +0.30 以上拡大**しており、OOS で偶然強かった可能性が高い。
- 集中 Tier 2 は M2_vol_adj/M4 でも改善幅が限定的で、Tier 1 を超えるブレイクスルーは確認できなかった。

---

## 4. 集中 Tier 3 結果 (18 パターン)

設計:
- ペア 3: (#21,#23), (#21,#41), (#23,#41)
- 演算子 2: AND, OR
- 戦略 3: S1, S2, S3
- 手法 1: M1_procyclical (2 値シグナルに直接対応)
- 計 18 = 3 × 2 × 3

| Judgment | 件数 |
|---|---:|
| MARGINAL_FULL | 18 |
| STRONG / STANDARD / FAIL | 0 |

観察:
- AND/OR いずれの組み合わせも、単一シグナル (Tier 1) を上回る改善を生まなかった。
- Tier 3 集中の WFE 中央値は 0.92 で **Tier 1 より低下**。複合シグナルは均質化効果ではなく、ノイズ増幅に近い挙動。
- CI95_lo_diff の中央値はほぼ 0 で、複合化による下方信頼区間の改善は確認されず。

---

## 5. クロス Tier 採用候補 Top 10

n_improved_full 降順、CAGR_OOS_diff タイブレーク。

| 順 | pattern_id | strategy | signal | method | dir | n_imp | n_deg | CAGR_OOS Δ | Sharpe Δ | MaxDD Δ | Trades/yr | WFE | CI95_lo Δ | judgment_full |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | T2F_M5_41_S1_filter_entry | S1 | DXY | M5 | filter_entry | 4 | 4 | +0.026 | +0.056 | +0.219 | 97.2 | 1.44 | -0.612 | MARGINAL_FULL |
| 2 | T2F_M5_41_S2_filter_entry | S2 | DXY | M5 | filter_entry | 4 | 4 | +0.023 | +0.058 | +0.121 | 97.7 | 1.51 | -0.598 | MARGINAL_FULL |
| 3 | T1_M2_6_S3_procyclical | S3 | VIX | M2 | procyclical | 4 | 1 | +0.019 | +0.049 | -0.071 | 72.2 | 1.08 | +0.039 | MARGINAL_FULL |
| 4 | T1_M1_41_S1_procyclical | S1 | DXY | M1 | procyclical | 4 | 3 | +0.015 | +0.033 | +0.177 | 98.5 | 0.99 | -0.006 | MARGINAL_FULL |
| 5 | T1_M1_41_S2_procyclical | S2 | DXY | M1 | procyclical | 4 | 3 | +0.012 | +0.034 | +0.121 | 99.2 | 1.00 | -0.004 | MARGINAL_FULL |
| 6 | T2F_M5_41_S3_filter_entry | S3 | DXY | M5 | filter_entry | 4 | 4 | +0.010 | +0.046 | +0.082 | 53.5 | 1.80 | -0.443 | MARGINAL_FULL |
| 7 | T1_M1_26_S1_procyclical | S1 | 2s10s | M1 | procyclical | 4 | 2 | -0.148 | +0.007 | +0.081 | 65.3 | 0.98 | +0.009 | MARGINAL_FULL |
| 8 | T1_M2_41_S2_procyclical | S2 | DXY | M2 | procyclical | 3 | 2 | +0.074 | +0.007 | +0.044 | 126.8 | 0.98 | +0.003 | MARGINAL_FULL |
| 9 | T2F_M2_41_S2_procyclical | S2 | DXY | M2 | procyclical | 3 | 2 | +0.074 | +0.007 | +0.044 | 126.8 | 0.98 | +0.003 | MARGINAL_FULL |
| 10 | T1_M2_41_S1_procyclical | S1 | DXY | M2 | procyclical | 3 | 2 | +0.074 | +0.015 | +0.081 | 126.4 | 0.97 | +0.004 | MARGINAL_FULL |

**観察**:
- Top 10 中 **8 件が DXY (#41) 関連** — DXY は 3 戦略すべてに改善寄与あり (procyclical 方向)。
- **#3 (S3 × VIX × M2_procyclical) のみ n_deg=1** で最も「副作用が少ない改善」候補。
- M5_filter_entry の上位 3 件は n_deg=4 で副作用過多。IS-OOS gap 拡大 (+0.30 超) は OOS 過適合の強い兆候。

---

## 6. 戦略別ベスト signal-augmented variant

各既存戦略 (S1/S2/S3) に対し、**シグナル増強で最良の単一改造**は以下。

| Strategy | 推奨 pattern | signal | method × direction | n_imp/n_deg | CAGR_OOS Δ | Sharpe Δ | MaxDD Δ | Trades/yr | コメント |
|---|---|---|---|---|---|---|---|---|---|
| **S1** (LT2-N750) | T2F_M5_41_S1_filter_entry | DXY | M5_filter_entry | 4/4 | +0.026 | +0.056 | +0.219 | 97.2 | n_deg=4。**副作用大、要 Phase D 厳格再評価** |
| 代替 (副作用小) | T1_M1_41_S1_procyclical | DXY | M1_procyclical | 4/3 | +0.015 | +0.033 | +0.177 | 98.5 | n_deg=3、より穏当 |
| **S2** (S2_VZGated) | T2F_M5_41_S2_filter_entry | DXY | M5_filter_entry | 4/4 | +0.023 | +0.058 | +0.121 | 97.7 | n_deg=4、IS-OOS gap +0.30 拡大 |
| 代替 (副作用小) | T1_M1_41_S2_procyclical | DXY | M1_procyclical | 4/3 | +0.012 | +0.034 | +0.121 | 99.2 | n_deg=3 |
| **S3** (DH_W1) | T1_M2_6_S3_procyclical | VIX | M2_procyclical | **4/1** | +0.019 | +0.049 | -0.071 | 72.2 | **n_deg=1 (本セッション最良の改善/副作用比)** |
| 代替 | T1_M2_23_S3_procyclical | BAA-10Y | M2_procyclical | 2/0 | +0.005 | +0.067 | -0.050 | 72.0 | **n_deg=0 唯一**、ただし n_imp=2 (STANDARD_PASS_RELAXED 該当) |

**最も信頼性が高い改善候補**:
- **S3 × BAA-10Y × M2_procyclical** — **n_deg=0** (唯一の劣化軸ゼロ)、緩和フレームで STANDARD_PASS_RELAXED。
- **S3 × VIX × M2_procyclical** — n_imp=4, n_deg=1 (IS-OOS gap のみ)、CAGR/Sharpe/P10_5Y/WFE 同時改善。

---

## 7. Top 5 の 9+1 軸完全ブレークダウン

### #1 T2F_M5_41_S1_filter_entry (S1 × DXY × M5_filter_entry)

| 軸 | cand | base | diff | 評価 |
|---|---:|---:|---:|---|
| CAGR_OOS | +0.3951 | +0.3690 | +0.0262 | improved |
| Sharpe_OOS | 1.000 | 0.944 | +0.056 | improved |
| MaxDD | -0.412 | -0.631 | +0.219 | improved |
| Worst10Y | -0.0450 | +0.1601 | -0.2051 | **degraded** |
| P10_5Y | 0.0000 | +0.1027 | -0.1027 | **degraded** |
| IS-OOS gap | -0.3833 | -0.0494 | +0.3339 | **degraded (gap 大幅拡大)** |
| Trades/yr | 97.2 | 118.6 | -21.5 | (cap OK) |
| WFE | 1.437 | 0.922 | +0.514 | improved |
| CI95_lo | -0.086 | +0.526 | -0.612 | **degraded** |

判定: improved 4 / degraded 4 → **MARGINAL_FULL**。
要注意: IS-OOS gap が +0.33 拡大、CI95_lo が +0.5 → -0.09 に転落。**OOS 偶然性 / Phase D 必須**。

### #2 T2F_M5_41_S2_filter_entry (S2 × DXY × M5_filter_entry)

| 軸 | cand | base | diff |
|---|---:|---:|---:|
| CAGR_OOS | +0.3650 | +0.3423 | +0.0226 |
| Sharpe | 1.048 | 0.990 | +0.058 |
| MaxDD | -0.397 | -0.518 | +0.121 |
| Worst10Y | -0.0268 | +0.1593 | -0.186 |
| P10_5Y | 0.0000 | +0.1187 | -0.119 |
| IS-OOS gap | -0.355 | -0.053 | +0.302 |
| Trades/yr | 97.7 | 119.4 | -21.8 |
| WFE | 1.508 | 0.921 | +0.587 |
| CI95_lo | -0.001 | +0.598 | -0.598 |

判定: 同上構造 (improved 4 / degraded 4) → **MARGINAL_FULL**。

### #3 T1_M2_6_S3_procyclical (S3 × VIX × M2_procyclical) — **本セッション最良の改善/副作用バランス**

| 軸 | cand | base | diff |
|---|---:|---:|---:|
| CAGR_OOS | +0.2302 | +0.2111 | +0.019 |
| Sharpe | 0.952 | 0.903 | +0.049 |
| MaxDD | -0.417 | -0.346 | -0.071 |
| Worst10Y | +0.0767 | +0.0900 | -0.013 |
| P10_5Y | +0.0533 | +0.0482 | +0.005 |
| IS-OOS gap | -0.0691 | -0.0355 | +0.034 (only degraded) |
| Trades/yr | 72.2 | 68.7 | +3.5 |
| WFE | 1.080 | 1.023 | +0.057 |
| CI95_lo | +0.709 | +0.670 | +0.039 |

判定: improved 4 / degraded 1 → **MARGINAL_FULL** (n_imp=4 だが完全フレーム STANDARD は ≥3 + no severe; degraded=1 が IS-OOS gap で severe 扱い)。
**Phase D 候補としては最有力**。MaxDD のみ -7pp 悪化なので、その許容判断次第。

### #4 T1_M1_41_S1_procyclical (S1 × DXY × M1_procyclical)

| 軸 | cand | base | diff |
|---|---:|---:|---:|
| CAGR_OOS | +0.3836 | +0.3690 | +0.015 |
| Sharpe | 0.978 | 0.944 | +0.033 |
| MaxDD | -0.454 | -0.631 | +0.177 |
| Worst10Y | +0.0863 | +0.1601 | -0.074 |
| P10_5Y | +0.0793 | +0.1027 | -0.023 |
| IS-OOS gap | -0.2092 | -0.0494 | +0.160 |
| Trades/yr | 98.5 | 118.6 | -20.1 |
| WFE | 0.994 | 0.922 | +0.072 |
| CI95_lo | +0.520 | +0.526 | -0.006 |

improved 4 / degraded 3 → MARGINAL_FULL。IS-OOS gap +0.16 拡大は注意。

### #5 T1_M1_41_S2_procyclical (S2 × DXY × M1_procyclical)

| 軸 | cand | base | diff |
|---|---:|---:|---:|
| CAGR_OOS | +0.3545 | +0.3423 | +0.012 |
| Sharpe | 1.024 | 0.990 | +0.034 |
| MaxDD | -0.397 | -0.518 | +0.121 |
| Worst10Y | +0.0866 | +0.1593 | -0.073 |
| P10_5Y | +0.0814 | +0.1187 | -0.037 |
| IS-OOS gap | -0.2011 | -0.0526 | +0.149 |
| Trades/yr | 99.2 | 119.4 | -20.2 |
| WFE | 1.004 | 0.921 | +0.083 |
| CI95_lo | +0.594 | +0.598 | -0.004 |

improved 4 / degraded 3 → MARGINAL_FULL。

---

## 8. Trades/yr, WFE, CI95_lo の判定への影響

| 指標 | 観察 |
|---|---|
| **Trades/yr (cap=200/yr)** | 117 件中 **超過 0 件**。NAV プロキシでは sign-flip ベース推定 (中央値 72-122/yr)。本シグナル注入は約定数を大きく増やさず、Trades 上限による FAIL は発生せず。**判定に影響なし**。 |
| **WFE (50 窓 Sharpe 比)** | T1 中央値 0.974, T2 集中 1.039, T3 集中 0.922。**Top 5 は全て WFE>=0.99**、改善側に寄与。ただし WFE=1.4-1.8 は「区間 Sharpe が full Sharpe の倍近く」=過適合か区間集中。**Top 2 (M5_filter_entry) の WFE=1.4-1.5 は危険信号**。 |
| **CI95_lo (50 窓 Sharpe 95% CI 下限)** | T1 中央値 -0.01, T2 +0.007, T3 0.000。**M5_filter_entry の CI95_lo は base +0.5 → cand -0.09 と大幅低下** (#1#2#6)。下方信頼区間の悪化は強い警告。逆に #3 (S3 × VIX × M2) は CI95_lo が +0.67 → +0.71 と僅かに改善。 |

**結論**: Trades/yr cap は判定を変えなかったが、**WFE と CI95_lo が「上位だが採用すべきでない」を識別する重要軸として機能**した (特に M5_filter_entry の OOS 偶然性検出)。

---

## 9. 採用判定

### 9+1 完全フレーム

| カテゴリ | 件数 | 採用可否 |
|---|---:|---|
| STRONG_PASS_FULL | 0 | — |
| STANDARD_PASS_FULL | 0 | — |
| MARGINAL_FULL | 105 | **既存ベスト (CURRENT_BEST_STRATEGY) を置換する根拠なし** |
| FAIL_FULL | 12 | 棄却 |

**結論**: 本セッションのスクリーニングでは、**6 つの Phase B 通過シグナルを 5 手法で注入しても、9+1 完全フレームの STRONG/STANDARD PASS は得られなかった**。既存の S2_VZGated + LT2-N750 + E4 (CURRENT_BEST_STRATEGY) は **据え置き** が妥当。

### Hard requirements (G7 / SPA) の別途再評価が必要

本セッションの 9+1 評価は**スクリーニング目的の NAV プロキシ**であり、以下は別途厳格再評価が必須:
- **WFE / CI95_lo**: 真値は WFA (g20/g30 系) で再計算する必要あり。本レポートの値は 50 等分窓近似。
- **Trades/yr**: 本実装は (cand_ret - base_ret) の sign-flip カウントで、実約定数とは異なる。
- **SPA test**: 多重比較補正は本セッションでは未実施。117 パターン中 4 件のみが n_imp≥4 という事実は、SPA 補正後にほぼ消失する可能性が高い。
- **G7 6 gate**: CI95_lo 下限 / WFE 1.0±0.2 / Trades 上限などの WFA gate は別途要実行。

---

## 10. Phase D (g20/g30 厳格 audit) 推奨

以下の 3 候補のみ Phase D 厳格再評価を推奨。それ以外は本セッションで棄却。

| 優先 | pattern | 推奨理由 |
|---:|---|---|
| ★ 1 | **S3 × BAA-10Y × M2_procyclical** | n_deg=0 (本セッション唯一)、STANDARD_PASS_RELAXED、Sharpe +0.067 / CAGR_OOS +0.005 / MaxDD -0.05。控えめだが副作用が極小 |
| ★ 2 | **S3 × VIX × M2_procyclical** | n_imp=4, n_deg=1 (IS-OOS gap)、CAGR +0.019 / Sharpe +0.049 / WFE +0.06 / P10_5Y +0.005。最も均衡が取れた改善 |
| 3 | S1/S2 × DXY × M1_procyclical | n_imp=4, n_deg=3。CAGR_OOS Δ は +0.01-0.015 と限定的だが MaxDD 改善幅 (+0.12-0.18) が大きい。要 Worst10Y 劣化の精査 |

**棄却 (Phase D 不要)**:
- M5_filter_entry × DXY × S1/S2/S3 (top 1/2/6): IS-OOS gap +0.30 拡大、CI95_lo -0.6 劣化。**強い OOS 過適合兆候**。
- Tier 3 (AND/OR 組合せ): 全件 n_imp が単一シグナル下限を超えず。複合化のメリット確認できず。
- Tier 1 で n_deg≥4 のもの (M1_defensive 系の多くで該当)。

---

## 11. 参照ファイル / 成果物

| 成果物 | 説明 | リンク |
|---|---|---|
| `src/integration/nine_metric_eval.py` | 9+1 指標 + judge_improvement_full 実装 | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/src/integration/nine_metric_eval.py) |
| `scripts/rerun_tier1_full.py` | Tier 1 (72 パターン) 完全フレーム再実行 | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/scripts/rerun_tier1_full.py) |
| `scripts/run_focused_integration.py` | 集中 Tier 2 (27) + Tier 3 (18) 実行 | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/scripts/run_focused_integration.py) |
| `data/signals/integration/tier1_results_full_20260605.csv` | Tier 1 完全再評価結果 (72 行 × 29 指標) | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/data/signals/integration/tier1_results_full_20260605.csv) |
| `data/signals/integration/tier2_focused_results_20260605.csv` | 集中 Tier 2 結果 (27 行) | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/data/signals/integration/tier2_focused_results_20260605.csv) |
| `data/signals/integration/tier3_focused_results_20260605.csv` | 集中 Tier 3 結果 (18 行) | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/data/signals/integration/tier3_focused_results_20260605.csv) |
| `data/signals/integration/integration_final_report_20260605.md` | 本レポート | [開く](https://github.com/KazuyaMurayama/NASDAQ_backtest/blob/main/data/signals/integration/integration_final_report_20260605.md) |

---

## 12. Next Action

**推奨**: 上記 ★1 (S3 × BAA-10Y × M2_procyclical) を Phase D (g20/g30 audit) で厳格再評価する。理由:
1. n_deg=0 で副作用が最も少ない
2. Sharpe +0.067 は実用的改善幅
3. S3 (DH_W1) は現行ベスト後継候補で、シグナル増強の検証価値が高い

それ以外のパターン (M5_filter_entry 系、Tier 3 組合せ) は本セッションで棄却し、シグナル研究は新規シグナル発掘 (例: Phase A 未通過の改良版、新規外部データ) に方針転換することを推奨。

CURRENT_BEST_STRATEGY.md の **更新は不要** (本セッションは候補発掘に終わり、置換採用には至らず)。
