# A1: Soft-Regime k_lt (sigmoid連続化) スイープ

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

A1: Soft-Regime k_lt (sigmoid連続化) - 境界ジャンプ除去によるSharpe改善

| 項目 | 定義 |
|------|------|
| **K_LO** | 0.1（E4採用値固定: 低ボラ端） |
| **K_HI** | 0.8（E4採用値固定: 高ボラ端） |
| **ALPHA_GRID** | [2.0, 3.0, 5.0, 8.0]（sigmoid 勾配） |
| **REF** | α=100 ≈ E4離散版 (k_lo=0.1, k_hi=0.8, vz_thr=0.7) |
| **合計 configs** | 4 + REF |
| **IS** | 1974-01-02 〜 2021-05-07 |
| **OOS** | 2021-05-08 〜 |

**sigmoid 連続 k_lt 計算式**:
```
k(vz) = K_LO + (K_HI - K_LO) * sigmoid(alpha * vz)
sigmoid(x) = 1 / (1 + exp(-x))
```
`lt_bias = (-k × lt_sig × 0.5).clip(-0.5, 0.5)`

**alpha → ∞ の極限**: sigmoid が step 関数に収束 → E4 離散版（vz_thr=0 相当）に近似
**alpha → 0 の極限**: k が一定 = (K_LO + K_HI) / 2 = 0.45（レジーム無効化）

**サニティ**: REF(α=100) CAGR_OOS=+36.34% (E4比 +2.81pp) → WARN

---

## §2 9指標テーブル（Sharpe 降順 + REF）

| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| α=100 (REF≈E4) | +36.3% | +0.92 ★ | -69.9% | +18.6% |  +6.1% | -3.58pp |  27 |    —    |    —    |
| α=8.0 | +35.1% | +0.90 ★ | -67.1% | +18.2% |  +6.4% | -2.48pp |  27 |    —    |    —    |
| α=5.0 | +34.6% | +0.90 ★ | -65.0% | +18.3% |  +6.9% | -2.06pp |  27 |    —    |    —    |
| α=3.0 | +33.9% | +0.89 ★ | -62.9% | +18.4% |  +7.4% | -1.59pp |  27 |    —    |    —    |
| α=2.0 | +33.5% | +0.89 ★ | -61.6% | +18.6% |  +7.8% | -1.33pp |  27 |    —    |    —    |

*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ E4_REF+0.020 (≥0.911) |
| (ii) CAGR_OOS | ≥ 29.2% |
| (iii) IS-OOS gap | ≤ 3.0pp |
| (iv) MaxDD | > -64.5% |
| (v) Worst10Y★ | ≥ 15.0% |

- **PASS configs**: 0 / 4
- **最高 Sharpe**: α=8.0 → Sharpe=+0.904, CAGR_OOS=+35.10%
- **E4 REF (α=100)**: Sharpe=+0.923, CAGR_OOS=+36.34%
- **総合判定: WARN**

---

## §4 考察

sigmoid 連続化により境界でのホイップソー損失を排除:
- alpha が小さい（2〜3）: 緩やかな連続遷移 → 過剰反応を抑制するが、regime 信号の強度も低下
- alpha が大きい（8〜100）: ほぼ離散判定に近づく → E4 との差異が縮小
- IS-OOS gap の変化に注目: 連続化でギャップ縮小なら「境界ホイップソーが gap 原因」と示唆
- PASS なら WFA (G3) 実施を推奨

---

*生成スクリプト: `src/a1_soft_regime_klt.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`*
