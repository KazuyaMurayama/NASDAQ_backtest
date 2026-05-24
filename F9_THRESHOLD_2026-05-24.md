# F9: Bull-Tilt THRESHOLD 最適化スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

### 背景
F7-v3（2026-05-24）で「E4 + step-func bull-tilt (tilt=10.0, cap=0.10)」が
PASS 基準 (Sharpe_OOS ≥ 0.911, CAGR_OOS ≥ 29.5%,
gap ≤ 6.0pp, MaxDD > -65.0%, W10Y★ ≥ 15.0%) を達成。
ただし THRESHOLD は signal 構築の bull/bear 判定値 (BASE_THRESHOLD=0.15) を
継承して固定していた。

### F9 の動機
「tilt を適用する bull 日」の閾値を変えると Sharpe / CAGR / Trades/yr に
どう影響するかを系統的に検証する。
- 低 THRESHOLD（例 0.05）→ ほぼ全 bull 日に tilt（積極的）
- 高 THRESHOLD（例 0.40）→ 強確信日のみ tilt（選択的）

### Tilt 定式（F7-v3 step-func を継承）
```
bull_mask    = raw_a2 > THRESHOLD       # ← F9 で可変
tilt_amount  = TILT_STEP × (raw_a2 - THRESHOLD) × (1 - raw_a2)
tilt_amount  = clip(tilt_amount, 0, TILT_CAP)
wn_tilted    = wn_A + tilt_amount
wb_tilted    = clip(wb_A - tilt_amount, 0, wb_A)
```
- TILT_STEP = 10.0（極端値 → 実質ステップ関数）
- TILT_CAP  = 0.10（cap 固定）

### 共通設定
| 項目 | 定義 |
|------|------|
| **Base config** | E4 採用: k_lo=0.1, k_hi=0.8, vz_thr=0.7, LT2-N750, mode B |
| **wn/wg/wb 構築** | simulate_rebalance_A の THRESHOLD は BASE_THRESHOLD=0.15 固定 |
| **F9 可変要素** | tilt 適用 bull_mask の閾値のみ |
| **IS** | 1974-01-02 〜 2021-05-07 |
| **OOS** | 2021-05-08 〜 |

**サニティ**: REF CAGR_OOS=+33.53% (diff vs E4 best CAGR=+0.00pp)

### グリッド
| config_id | THRESHOLD | 解釈 |
|:---|---:|:---|
| REF | — | E4 (tilt なし) |
| T005 | 0.05 | ほぼ全 bull 日に tilt（積極的） |
| T010 | 0.10 | 中程度の積極度 |
| T015 | 0.15 | F7-v3 A:tilt=10 の再現（ベースライン） |
| T020 | 0.20 | やや選択的 |
| T025 | 0.25 | 選択的 |
| T030 | 0.30 | 高確信日のみ |
| T040 | 0.40 | 非常に選択的 |

---

## §2 9指標テーブル

| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| REF (no tilt) | +33.5% | +0.89 ★ | -60.0% | +18.7% |  +9.8% | -1.81pp |  27 |    —    |    —    |
| T005 (thr=0.05) | +36.4% | +0.93 ★ | -61.4% | +17.9% |  +9.8% | -4.56pp | 178 |    —    |    —    |
| T010 (thr=0.10) | +36.4% | +0.93 ★ | -61.6% | +18.2% | +10.0% | -4.47pp | 179 |    —    |    —    |
| T015 (thr=0.15) ←baseline | +36.5% | +0.93 ★ | -62.0% | +18.1% |  +9.8% | -4.52pp | 179 |    —    |    —    |
| T020 (thr=0.20) | +36.4% | +0.93 ★ | -61.8% | +18.2% |  +9.8% | -4.46pp | 179 |    —    |    —    |
| T025 (thr=0.25) | +35.9% | +0.92 ★ | -61.8% | +18.1% |  +9.8% | -3.85pp | 179 |    —    |    —    |
| T030 (thr=0.30) | +35.4% | +0.91 ★ | -60.7% | +18.5% | +10.1% | -3.09pp | 179 |    —    |    —    |
| T040 (thr=0.40) | +35.2% | +0.91 ★ | -60.6% | +18.4% | +10.6% | -2.78pp | 179 |    —    |    —    |

*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

### Bull 日数 / Trades_yr 詳細
| config_id | THR | bull_days | bull_ratio | Trades_yr |
|:---|---:|---:|---:|---:|
| T005 | 0.05 | 10,148 | 77.1% | 178.0 |
| T010 | 0.10 | 9,858 | 74.9% | 178.6 |
| T015 | 0.15 | 9,565 | 72.6% | 179.2 |
| T020 | 0.20 | 9,243 | 70.2% | 179.1 |
| T025 | 0.25 | 8,923 | 67.8% | 179.1 |
| T030 | 0.30 | 8,579 | 65.1% | 179.0 |
| T040 | 0.40 | 7,841 | 59.5% | 178.8 |

---

## §3 考察

### パターン
**観察パターン**: 内側ピーク型（最良は T015: thr=0.15）

### 最良 vs ベースライン (T015)
- **最良 config**: T015 (thr=0.15)
  - Sharpe_OOS = +0.9293
  - CAGR_OOS = +36.52%
  - MaxDD = -62.04%
  - Trades_yr = 179.2
- **T015 (baseline)**: Sharpe = +0.9293, CAGR_OOS = +36.52%, Trades_yr = 179.2
- **Sharpe 差**: 最良 − T015 = +0.0000

### Bull 日数 vs Sharpe トレードオフ
THRESHOLD を上げると tilt 適用日数が減り、Trades_yr も減少する。
- 最低 Trades_yr: 178.0 (T005)
- 最高 Trades_yr: 179.2 (T015)

### Sharpe レンジ
- 最高 Sharpe: +0.9293 (T015)
- 最低 Sharpe: +0.9081 (T040)
- レンジ幅: 0.0213

---

## §4 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+0.020 (≥ 0.911) |
| (ii) CAGR_OOS | ≥ 29.5% |
| (iii) IS-OOS gap | ≤ 6.0pp |
| (iv) MaxDD | > -65.0% |
| (v) Worst10Y★ | ≥ 15.0% |

- **PASS configs**: 5 / 7
- **PASS リスト**: T005, T010, T015, T020, T025
- **最高 Sharpe**: T015 (thr=0.15) → Sharpe=+0.929, CAGR_OOS=+36.52%, Tr/yr=179.2
- **総合判定: PASS**

---

*生成スクリプト: `src/f9_threshold.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/f7v3_bull_tilt.py`, `src/e4_regime_klt.py`*
