# F8: Regime-Conditional Bull-Tilt スイープ

作成日: 2026-05-24
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

### 問題意識（F7-v3 からの継承）
F7-v3 で「定式 A: step-function tilt (tilt=10.0, cap=0.10)」が PASS 候補となった。
- F7V3_BASE: CAGR_OOS=+36.52%, Sharpe=+0.929,
  MaxDD=-62.04%, Worst10Y★=+18.11%,
  Trades/yr=179.2

しかし副作用として:
1. **Trades/yr が REF 比 6.6× に急増**（27 → 179）→ 取引コスト感応度悪化
2. **MaxDD がやや拡大**（-60.0% → -62.0%）
3. **IS-OOS gap が拡大**（-1.8pp → -4.5pp）

### F8 設計の動機
tilt が**全 bull 日**に発火するのが過剰。VZ レジームで絞ることで以下を狙う:
- **calm 限定**: 高ボラ局面の上振せを止める → MaxDD 改善
- **bull-VZ 限定**: tilt 発火日を激減 → Trades/yr 削減
- **scaled cap**: 段階的縮小で滑らかな移行
- **calm boost**: calm 時のみ cap を引き上げる「逆張り」設計

### 共通設定
| 項目 | 定義 |
|------|------|
| Base config | E4 採用: k_lo=0.1, k_hi=0.8, vz_thr=0.7, k_mid=0.5, LT2-N750, mode B |
| Tilt 定式 | step-function: clip(tilt × (raw_a2-0.15) × (1-raw_a2), 0, cap), tilt=10.0 |
| TILT_CAP (base) | 0.1 |
| VZ_REG 閾値 | ±0.7 （calm / stressed の境界） |
| THRESHOLD | 0.15 (raw_a2 bull 判定) |
| wn 調整 | wn_tilted = wn_A + tilt_amount |
| wb 調整 | wb_tilted = max(wb_A - tilt_amount, 0) |
| IS  | 1974-01-02 〜 2021-05-07 |
| OOS | 2021-05-08 〜 |

**サニティ**: REF CAGR_OOS=+33.53% (diff vs E4 best CAGR=+0.00pp → OK)

### config 一覧

| config_id | tilt 適用条件 | 直感 |
|:----------|:--------------|:-----|
| REF | tilt なし | E4 採用 config そのまま |
| F7V3_BASE | raw_a2 > 0.15 | F7-v3 A:tilt=10（step function）と完全同一 |
| R1_CALM | raw_a2 > 0.15 AND \|vz\| < 0.7 | calm regime 限定。stressed 時の上振せ抑制 |
| R2_NO_BEAR_VZ | raw_a2 > 0.15 AND vz > -0.7 | bear VZ 除外（高ボラ売り局面では tilt しない） |
| R3_BULL_VZ | raw_a2 > 0.15 AND vz > +0.7 | bull VZ 限定（最も保守的・極端ケース） |
| R4_VZ_SCALED | cap_eff = 0.10 * max(0, 1 - \|vz\|) | VZ 大きいほど cap 縮小 |
| R5_CALM_BOOST | \|vz\|<0.7→cap=0.15 / vz>+0.7→cap=0.10 / vz<-0.7→cap=0.05 | calm 時 +50% 増量 |
| R6_DOUBLE | raw_a2 > 0.15 AND vz > 0.0 | vz > 0 = ボラ平均超え |

### Bull-day VZ 分布（diagnostic）
- Bull days total (raw_a2 > 0.15): **9,565** / 13,169 (72.6%)
- ∩ calm (|vz|<0.7): 5,369 (56.1% of bull)
- ∩ no-bear-vz (vz>-0.7): 6,586 (68.9% of bull)
- ∩ bull-vz only (vz>+0.7): 1,217 (12.7% of bull)
- ∩ vz>0: 3,007 (31.4% of bull)

---

## §2 9指標テーブル

| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| REF (E4, tilt=0) | +33.5% | +0.89 ★ | -60.0% | +18.7% |  +9.8% | -1.81pp |  27 |    —    |    —    |
| F7V3_BASE | +36.5% | +0.93 ★ | -62.0% | +18.1% |  +9.8% | -4.52pp | 179 |    —    |    —    |
| R1_CALM | +35.4% | +0.92 ★ | -61.1% | +18.7% | +10.1% | -2.99pp | 178 |    —    |    —    |
| R2_NO_BEAR_VZ | +35.6% | +0.92 ★ | -61.9% | +19.0% | +10.3% | -3.18pp | 179 |    —    |    —    |
| R3_BULL_VZ | +33.7% | +0.89 ★ | -60.5% | +18.8% | +10.1% | -2.00pp | 177 |    —    |    —    |
| R4_VZ_SCALED | +34.8% | +0.91 ★ | -60.2% | +18.8% |  +9.8% | -2.60pp | 191 |    —    |    —    |
| R5_CALM_BOOST | +36.8% | +0.93 ★ | -63.1% | +18.6% | +10.3% | -4.28pp | 182 |    —    |    —    |
| R6_DOUBLE | +33.7% | +0.89 ★ | -61.9% | +18.7% |  +9.7% | -2.16pp | 178 |    —    |    —    |

*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

---

## §3 考察

### F7V3_BASE 比 差分（同 base, レジーム条件のみ追加）

| config | ΔSharpe | ΔCAGR_OOS | ΔMaxDD | ΔTrades/yr |
|:-------|--------:|----------:|-------:|-----------:|
| R1 calm only | -0.011 | -1.11pp | +0.93pp | -1.4 |
| R2 no-bear-vz | -0.009 | -0.92pp | +0.16pp | +0.1 |
| R3 bull-vz only | -0.035 | -2.79pp | +1.52pp | -2.1 |
| R4 vz-scaled cap | -0.021 | -1.76pp | +1.83pp | +11.7 |
| R5 calm boost | +0.005 | +0.31pp | -1.03pp | +2.4 |
| R6 vz>0 | -0.038 | -2.82pp | +0.18pp | -0.7 |

### 最良 config
**R5_CALM_BOOST** が Sharpe_OOS=+0.934 で最大。
- vs REF: ΔSharpe=+0.043, ΔCAGR=+3.30pp,
  ΔMaxDD=-3.06pp, ΔTr/yr=+154.4
- vs F7V3_BASE: ΔSharpe=+0.005,
  ΔCAGR=+0.31pp,
  ΔMaxDD=-1.03pp,
  ΔTr/yr=+2.4

### レジーム条件の効果（観察ポイント）
- **Trades/yr 削減**: tilt 発火日が減るほど Trades/yr が落ちる。
  R3_BULL_VZ（最少発火）と R1_CALM, R6_DOUBLE で比較。
- **MaxDD 改善**: bear-vz 局面で tilt しない R2/R3 で MaxDD 改善が期待。
- **Sharpe 維持**: F7V3_BASE の Sharpe=+0.929 に対し、レジーム
  限定で Sharpe を維持/向上できるかが採用判定の鍵。
- **R4_VZ_SCALED / R5_CALM_BOOST**: 段階的設計が単純マスクより優位かをチェック。

---

## §4 判定

| 基準 | 条件 |
|------|------|
| (i)   Sharpe_OOS  | ≥ REF + 0.020 (≥ 0.911) |
| (ii)  CAGR_OOS    | ≥ 29.5% |
| (iii) IS-OOS gap  | ≤ 6.0pp |
| (iv)  MaxDD       | > -65.01% |
| (v)   Worst10Y★   | ≥ 15.0% |

- PASS configs: **4 / 7** ['F7V3_BASE', 'R1_CALM', 'R2_NO_BEAR_VZ', 'R5_CALM_BOOST']
- WARN configs: 2 ['R3_BULL_VZ', 'R4_VZ_SCALED']
- FAIL configs: 1 ['R6_DOUBLE']
- 最高 Sharpe: **R5_CALM_BOOST** (Sharpe=+0.934, CAGR_OOS=+36.83%, Tr/yr=181.6)

### 総合判定: PASS

**PASS**: 次の config が全基準クリア — F7V3_BASE, R1_CALM, R2_NO_BEAR_VZ, R5_CALM_BOOST

PASS 基準（§3.12 v1.1 準拠）: Sharpe_OOS ≥ +0.9115、CAGR_OOS ≥ +29.5%、IS-OOS gap ≤ +6.0pp、MaxDD > -65.01%、Worst10Y★ ≥ +15.0%

---

*生成スクリプト: `src/f8_regime_tilt.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/f7v3_bull_tilt.py`, `src/e4_regime_klt.py`*
