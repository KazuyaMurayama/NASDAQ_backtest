# G5: Real-yield LT Overlay スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **base** | E4 (k_lo=0.1, k_hi=0.8, vz_thr=0.7) + LT2-N750-modeB + S2_VZGated |
| **macro signal** | 実質10年金利 z-score (DGS10 − 5yr CPI inflation, 10yr rolling z) |
| **lag** | 2 営業日 (lookahead 防止) |
| **k_macro グリッド** | [0.3, 0.5, 0.7] |
| **alpha グリッド** | [0.25, 0.5] (= macro 寄与の重み) |
| **合計 configs** | 6 + REF |
| **IS** | 1974-01-02 〜 2021-05-07 |
| **OOS** | 2021-05-08 〜 |

**合成式**:
```
ry        = DGS10_t − (log(CPI_t / CPI_{t−1260}) / 5) × 100
ry_z      = (ry − rolling_mean(ry, 2520)) / rolling_std(ry, 2520)
lt_bias_macro    = (−k_macro × ry_z × 0.5).clip(−0.5, 0.5)
lt_bias_E4       = (−k_dyn(vz) × lt_sig × 0.5).clip(−0.5, 0.5)   # 現行ベスト
lt_bias_combined = (1 − alpha) × lt_bias_E4 + alpha × lt_bias_macro
```

**サニティ**: REF CAGR_OOS=+33.53% (diff vs E4 best +0.00pp)

---

## §2 9指標テーブル（REF + 全 6 configs, Sharpe 降順）

| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| REF (E4 only) | +33.5% | +0.89 ★ | -60.0% | +18.7% |  +9.8% | -1.81pp |  27 |    —    |    —    |
| k=0.7/a=0.25 | +33.7% | +0.88 ◎ | -56.8% | +20.9% |  +9.8% | +0.37pp |  27 |    —    |    —    |
| k=0.5/a=0.25 | +33.5% | +0.88 ◎ | -56.6% | +20.8% |  +9.8% | +0.41pp |  27 |    —    |    —    |
| k=0.3/a=0.25 | +33.0% | +0.87 ◎ | -57.9% | +20.0% |  +9.6% | +0.59pp |  27 |    —    |    —    |
| k=0.5/a=0.50 | +32.6% | +0.85 ◎ | -58.8% | +21.9% |  +9.9% | +2.62pp |  27 |    —    |    —    |
| k=0.7/a=0.50 | +32.6% | +0.85 ◎ | -58.6% | +21.7% |  +9.8% | +2.38pp |  27 |    —    |    —    |
| k=0.3/a=0.50 | +32.2% | +0.85 ◎ | -58.5% | +21.4% |  +9.7% | +2.67pp |  27 |    —    |    —    |

*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+0.020 (≥0.911) |
| (ii) CAGR_OOS | ≥ 29.5% |
| (iii) IS-OOS gap | ≤ 6.0pp |
| (iv) MaxDD | > -65.01% |
| (v) Worst10Y★ | ≥ 15.0% |

- **PASS configs**: 0 / 6
- **最高 Sharpe**: k_macro=0.7, alpha=0.25 → Sharpe=+0.881
- **総合判定: FAIL**

---

## §4 考察

実質金利 z-score は LT2 (価格モメンタム) や vz (ボラ z-score) と独立な
マクロシグナルとして overlay を加算する設計:
- alpha=0.25 は macro を弱く混ぜる (E4 を保ち補助)
- alpha=0.50 は均等 (LT2 と macro を等重)
- contrarian: 実質金利高騰局面 → リスク資産抑制（NASDAQ 配分↓）
- 注意点: 初期 ~7 年 (1962-1969) は CPI/DGS10 ローリング窓不足のため ry_z=0 (neutral)。
  OOS は 1980 以降に位置するため判定への影響は限定的。
- 同方向のシグナル重複に注意: VIX レジーム / vz_gated と相関する可能性。

---

*生成スクリプト: `src/g5_real_yield.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`*
