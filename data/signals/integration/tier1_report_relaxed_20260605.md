# Tier 1 統合検証 緩和判定レポート

作成日: 2026-06-05
最終更新日: 2026-06-05

## 1. 目的

厳格判定（`judge_improvement`）で **0 STRONG / 0 STANDARD / 58 MARGINAL / 14 FAIL** という結果になった
Tier 1 (72 patterns) を、severe-degradation 閾値を **2 倍緩めた** `judge_improvement_relaxed` で再評価し、
Tier 2/3/4 着手前に「採用に値する候補が緩和でも発生するか」を確認する。

改善判定（IMP_THR）は変更せず、「何をもって改善とするか」の honesty は維持する。

## 2. 閾値比較

| metric | 改善 (変更なし) | 厳格・重大悪化 | 緩和・重大悪化 |
|---|---|---|---|
| cagr_oos diff | ≥ +0.005 | < -0.010 | **< -0.020** |
| sharpe diff   | ≥ +0.030 | < -0.030 | **< -0.050** |
| maxdd diff    | ≥ +0.020 | < -0.050 | **< -0.100** |
| worst10y diff | ≥ +0.005 | < -0.010 | **< -0.020** |
| p10_5y diff   | ≥ +0.005 | < -0.010 | **< -0.020** |
| is_oos_gap diff | ≤ -0.005 | > +0.015 | **> +0.030** |

判定ラベル: `STRONG_PASS_RELAXED` (改善≥4, 重大悪化なし) / `STANDARD_PASS_RELAXED` (改善≥2, 重大悪化なし) / `MARGINAL_RELAXED` (改善≥1) / `FAIL_RELAXED`。

## 3. 結果比較

### 3.1 全体

| 区分 | STRONG | STANDARD | MARGINAL | FAIL | 合計 |
|---|---|---|---|---|---|
| 厳格 | 0 | 0 | 58 | 14 | 72 |
| 緩和 | 0 | **1** | 57 | 14 | 72 |

緩和でも **STRONG は 0**、STANDARD は **1 件のみ**。アップグレード件数は **1** (`T1_M2_23_S3_procyclical`)。

### 3.2 戦略別

| Strategy | STRONG_RELAXED | STANDARD_RELAXED | MARGINAL_RELAXED | FAIL_RELAXED |
|---|---|---|---|---|
| S1 (LT2-N750 単体) | 0 | 0 | 20 | 4 |
| S2 (VZGated + LT2 + E4 — 現行 Active) | 0 | 0 | 20 | 4 |
| S3 (DH-W1 完全分散) | 0 | **1** | 17 | 6 |

→ 全戦略でほぼ同じ傾向。**緩和でも S2 / S1 は 0 PASS**、S3 で唯一 1 件のみ発生。

## 4. 緩和判定 STANDARD_PASS の詳細

| 項目 | 値 |
|---|---|
| pattern_id | `T1_M2_23_S3_procyclical` |
| strategy | S3 (DH-W1 完全分散) |
| method | M2 (overlay) |
| direction | procyclical (信号 ON 時のみベース戦略実行) |
| signal | #23 BAA-10Y credit spread |
| 改善軸 | cagr_oos / sharpe |
| 悪化軸 (緩和基準で) | なし |

### フルメトリクス

| metric | candidate | baseline | diff | 改善判定 |
|---|---|---|---|---|
| CAGR_OOS | +21.66% | +21.11% | **+0.54pp** | imp (≥0.5pp) |
| Sharpe_OOS | +0.9697 | +0.9027 | **+0.067** | imp (≥0.03) |
| MaxDD | -39.62% | -34.57% | -5.05pp | 厳格 deg / 緩和 NOT deg (−10pp 内) |
| Worst10Y | +7.26% | +9.00% | -1.74pp | 厳格 deg / 緩和 NOT deg (−2pp 内) |
| P10_5Y | +4.84% | +4.82% | +0.03pp | 中立 |
| IS-OOS gap | -5.64% | -3.55% | +2.09pp | 厳格 deg / 緩和 NOT deg (≤+3pp) |

注意: **MaxDD は -5.05pp 悪化、IS-OOS gap も +2.09pp 拡大しており、厳格基準で deg 扱い**。緩和で「重大悪化なし」と判定された結果。**実質的には Sharpe・CAGR の僅か (~0.5pp) な改善とのトレードオフ**。WFA 通過および G7 系の CI95_lo / WFE での確認が不可欠。

## 5. 採用候補トップ（緩和 PASS + 緩和に近い MARGINAL）

緩和 PASS が 1 件しかないため、**MARGINAL のうち「改善 ≥ 2 軸 かつ severe-deg が緩和基準ぎりぎり」の上位**も並べて検討する。

### 5.1 Sharpe 改善トップ 10 (全 72 中)

| pattern_id | signal | strat | meth | dir | cagr | sharpe | maxdd | w10 | p10 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| T1_M2_23_S3_procyclical | BAA-10Y | S3 | M2 | pro | +0.005 | **+0.067** | -0.050 | -0.017 | +0.000 | **STANDARD_RELAXED** |
| T1_M2_23_S2_procyclical | BAA-10Y | S2 | M2 | pro | -0.005 | +0.053 | +0.031 | -0.022 | -0.001 | MARG_RELAXED (worst10y deg) |
| T1_M2_23_S1_procyclical | BAA-10Y | S1 | M2 | pro | -0.003 | +0.053 | -0.003 | -0.020 | +0.000 | MARG_RELAXED (worst10y deg ≈) |
| T1_M2_6_S3_procyclical | VIX | S3 | M2 | pro | +0.019 | +0.049 | -0.071 | -0.013 | +0.005 | MARG_RELAXED (is_oos_gap) |
| T1_M2_6_S1_procyclical | VIX | S1 | M2 | pro | +0.012 | +0.041 | -0.031 | -0.075 | +0.004 | MARG_RELAXED (worst10y, is_oos_gap) |
| T1_M2_21_S1_procyclical | HY OAS | S1 | M2 | pro | -0.010 | +0.051 | +0.140 | -0.036 | -0.014 | MARG_RELAXED (worst10y, is_oos_gap) |
| T1_M2_21_S3_procyclical | HY OAS | S3 | M2 | pro | -0.016 | +0.053 | +0.014 | -0.023 | -0.011 | MARG_RELAXED |
| T1_M2_21_S2_procyclical | HY OAS | S2 | M2 | pro | -0.017 | +0.038 | +0.059 | -0.042 | -0.026 | MARG_RELAXED (worst10y, p10_5y, is_oos_gap) |

### 5.2 CAGR 改善トップ (positive のみ)

| pattern_id | signal | strat | meth | dir | cagr | sharpe | maxdd | 判定 |
|---|---|---|---|---|---|---|---|---|
| T1_M2_41_S2_procyclical | DXY | S2 | M2 | pro | **+0.074** | +0.007 | +0.044 | MARG_RELAXED |
| T1_M2_41_S1_procyclical | DXY | S1 | M2 | pro | **+0.074** | +0.015 | +0.081 | MARG_RELAXED |
| T1_M2_41_S3_procyclical | DXY | S3 | M2 | pro | **+0.051** | +0.019 | +0.014 | MARG_RELAXED |
| T1_M2_28_S3_defensive | 10Y real yield | S3 | M2 | def | +0.022 | -0.007 | +0.023 | MARG_RELAXED |
| T1_M1_41_S1_procyclical | DXY | S1 | M1 | pro | +0.015 | +0.033 | +0.177 | MARG_RELAXED |
| T1_M1_41_S2_procyclical | DXY | S2 | M1 | pro | +0.012 | +0.034 | +0.121 | MARG_RELAXED |

## 6. 観察と所見

### 6.1 PASS が極めて少ない理由

1. **Baseline がすでに強い** — S2 = Sharpe+0.94 / CAGR+33.5% / MaxDD-51.8% という現行ベスト。これに対し「diff ≥ +0.5pp CAGR」「Sharpe ≥ +0.03」を同時に満たすのは構造的に困難。
2. **Worst10Y と P10_5Y の悪化が頻発** — overlay/defensive 系は信号 OFF 期にキャッシュ化されるため、長期 rolling 系の指標が削れやすい (構造的副作用)。
3. **DXY (#41)** は CAGR を大きく改善 (+5–7pp) するが Sharpe 改善は僅少。「リターン特化」型。
4. **HY OAS (#21) / BAA-10Y (#23)** は Sharpe+5pp 帯で安定。クレジット系信号は **リスク調整リターン改善** に強い (procyclical=ON のみ実行)。
5. **defensive (信号 ON 時のみ DD 抑制) は概ね劣化** — ベース戦略の trades/yr=27 と低頻度のため、defensive で更に減らすと OOS でリターン取りこぼし。

### 6.2 緩和でもこの結果が意味するもの

- **「改善ぽい」のは S3 + クレジット系 procyclical** に偏在。S2 単体改善は確認できない。
- M1 (overlay) より **M2 (gate)** が一貫して有利 (Sharpe 改善トップ群は全て M2)。
- 信号別に見ると **#21 HY OAS + #23 BAA-10Y** が安定して Sharpe 改善寄与。**#26 (2s10s)、#28 (10Y real yield)** は overall に弱い。

## 7. 戦略別ベストメソッド (改善ベクトル別)

| 戦略 | リスク調整リターン特化 (Sharpe 改善 max) | リターン特化 (CAGR 改善 max) |
|---|---|---|
| **S1** (LT2-N750) | M2 × #23 BAA-10Y procyclical (Sharpe +0.053, worst10y近接で deg) | M2 × #41 DXY procyclical (CAGR +7.4pp, Sharpe +0.015) |
| **S2** (現行 Active) | M2 × #23 BAA-10Y procyclical (Sharpe +0.053, worst10y deg) | M2 × #41 DXY procyclical (CAGR +7.4pp) |
| **S3** (DH-W1) | **M2 × #23 BAA-10Y procyclical (Sharpe +0.067, 緩和 PASS)** | M2 × #41 DXY procyclical (CAGR +5.1pp) |

## 8. Tier 2/3/4 着手判断

### 結論: **条件付き Go**

| 観点 | 判断 |
|---|---|
| 緩和でも採用候補が大量に出てきたか？ | **No** (1 件のみ・しかも S3 限定) |
| Tier 2/3/4 で更に強い組み合わせが期待できるか？ | **Yes** — Tier 2 (AND/OR 組み合わせ) で credit (#21/#23) + DXY (#41) のような直交した改善ベクトルの統合が有望 |
| 現状 0 STANDARD のまま Tier 2/3/4 を回す価値はあるか？ | **限定的に Yes** — ただし *全 72 パターン全数展開ではなく、優先信号を絞る* |

### 推奨 Tier 2/3/4 着手スコープ

**Priority 1 — 即実行価値あり (S2 改善寄与あり信号 × M2 procyclical)**
- 信号: **#23 BAA-10Y** ・ **#21 HY OAS** ・ **#41 DXY**
- 方式: M2 (gate) procyclical 中心
- Tier 2 (AND/OR): credit (#21 / #23) ∧ DXY (#41) で「Sharpe 改善 + CAGR 改善」両立を狙う

**Priority 2 — 検討余地あり**
- 信号: **#6 VIX** (S2/S3 procyclical で Sharpe +0.04~0.07、ただし worst10y deg ≈)
- AND 条件 (HY OAS + VIX) で worst10y deg の打ち消しが期待できる

**Skip 推奨 — Tier 2 以降で深追いしない**
- 信号: **#26 2s10s 、 #28 10Y real yield** (どの軸でも改善寄与が小さい)
- 全 defensive 系 (構造的にリターン取りこぼし)

### Tier 2/3/4 の前提条件
1. 全数 72 → 優先 3 信号 × M2 procyclical × 3 戦略 = **9 base × (AND/OR/STACK) ≈ 27 patterns** に絞ること
2. **WFA / G7 系 (CI95_lo, WFE) を Tier 2 から組み込む** — 6 metric だけでは S2 baseline 超えの統計有意性は判定不能
3. 緩和判定でも「Sharpe ≥ +0.03 かつ CAGR diff ≥ 0」を満たす組合せのみ Tier 3 / Tier 4 (採用候補) に進める

## 9. 次工程候補 (ユーザー判断要)

| Option | 内容 |
|---|---|
| **A. 推奨 — Tier 2 を絞り込み実行** | 信号 #21/#23/#41 × M2 procyclical × 3 戦略 で AND/OR 組み合わせ生成、WFA 込みで再評価 |
| B. 閾値再調整 | 緩和でも実質 deg ばかり残るため、Sharpe/CAGR 同時改善のみで PASS とする alt 判定を試す |
| C. Tier 1 拡張 | 信号集合を Phase B 通過全部 (現行 6 → 全) に拡張して Tier 1 全数再走 |
| D. 現行 S2 維持 | 統合信号採用は見送り、CURRENT_BEST_STRATEGY (S2_VZGated + LT2-N750 + E4) のまま運用継続 |

**推奨: A**。理由 = Tier 1 単一信号では構造的に S2 baseline 超えが困難だが、Tier 2 で **直交ベクトル統合** (Sharpe ベクトル + CAGR ベクトル) が機能する余地は十分残っている。
