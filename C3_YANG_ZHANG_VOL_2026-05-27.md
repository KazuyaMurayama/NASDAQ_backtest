# C3: Yang-Zhang Vol推定（ボラ推定量置換）スイープ

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

| 項目 | 定義 |
|------|------|
| **YZ_lookback グリッド** | [10, 20, 30]（Yang-Zhang推定窓サイズ） |
| **vz_thr グリッド** | [0.5, 0.7, 1.0]（レジーム切替え閾値） |
| **k_lo** | 0.1 固定（E4採用値） |
| **k_hi** | 0.8 固定（E4採用値） |
| **k_mid** | 0.5（閾値外の中間域） |
| **合計 configs** | 9 + REF |
| **IS** | 1974-01-02 〜 2021-05-07 |
| **OOS** | 2021-05-08 〜 |

**Yang-Zhang式**:
```
σ²_YZ = σ²_overnight + k×σ²_cc_open + (1-k)×σ²_rs
σ²_overnight = mean((log(Open/Close_prev))²)
σ²_rs (Rogers-Satchell) = mean(log(H/O)×log(H/C) + log(L/O)×log(L/C))
σ²_cc_open = mean((log(Close/Open))²)
k = 0.34 / (1.34 + (n+1)/(n-1))
```

**注記: 1974-1984のOHLCは合成データ（O=H=L=C）のため、
この期間のYZ推定量はclose-to-closeと同等になる。**

**レジーム割り当て（k_lo/k_hi固定）**:
```
vz_yz > +vz_thr  → k = 0.8  (高YZボラ: 強い防御バイアス)
vz_yz < -vz_thr  → k = 0.1  (低YZボラ: 弱い防御バイアス)
otherwise        → k = 0.5  (中間域)
```
`lt_bias = (-k × lt_sig × 0.5).clip(-0.5, 0.5)`

**サニティ**: CC-REF CAGR_OOS=+33.53% (diff +0.00pp)

---

## §2 9指標テーブル（全9 configs + CC-REF）

> CC-REF = close-to-close vz を使ったE4ベスト（k_lo=0.1/k_hi=0.8/vz=0.7）
> YZ configs は Sharpe_OOS 降順

| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| CC-REF (E4 k_lo=0.1/k_hi=0.8/vz=0.7) | +33.5% | +0.89 ★ | -60.0% | +18.7% |  +9.8% | -1.81pp |  27 |    —    |    —    |
| YZ_n=10/vz=0.7 | +33.3% | +0.89 ★ | -58.4% | +19.4% | +10.6% | -2.05pp |  27 |    —    |    —    |
| YZ_n=10/vz=0.5 | +30.3% | +0.82 ◎ | -60.0% | +21.1% | +10.1% | +3.02pp |  27 |    —    |    —    |
| YZ_n=10/vz=1.0 | +29.1% | +0.82 ◎ | -59.1% | +17.2% | +10.1% | +1.76pp |  27 |    —    |    —    |
| YZ_n=20/vz=0.5 | +30.2% | +0.82 ◎ | -61.3% | +17.3% |  +8.3% | +0.77pp |  27 |    —    |    —    |
| YZ_n=20/vz=1.0 | +27.6% | +0.78 ◎ | -58.9% | +17.4% |  +9.7% | +2.34pp |  27 |    —    |    —    |
| YZ_n=20/vz=0.7 | +27.9% | +0.78 ◎ | -58.9% | +18.5% |  +9.5% | +3.08pp |  27 |    —    |    —    |
| YZ_n=30/vz=0.5 | +25.6% | +0.75 | -61.7% | +16.5% |  +7.3% | +3.92pp |  27 |    —    |    —    |
| YZ_n=30/vz=0.7 | +23.3% | +0.70 | -61.6% | +17.0% |  +8.4% | +6.01pp |  27 |    —    |    —    |
| YZ_n=30/vz=1.0 | +22.6% | +0.70 | -61.1% | +15.2% |  +8.5% | +6.15pp |  27 |    —    |    —    |

*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ CC-REF+0.020 (≥0.911) |
| (ii) CAGR_OOS | ≥ 29.2% |
| (iii) IS-OOS gap | ≤ 3.0pp |
| (iv) MaxDD | > -64.5% |
| (v) Worst10Y★ | ≥ 15.0% |

- **PASS configs**: 0 / 9
- **最高 Sharpe**: YZ_n=10, vz_thr=0.7 → Sharpe=+0.888
- **総合判定: FAIL**

---

## §4 考察

YZ推定量はOHLC情報を活用してclose-to-closeより効率的なボラ推定を行う:
- 1974-1984の合成OHLC期間はclose-to-closeと同等（YZの優位性は実データ期間のみ有効）
- YZ Z-scoreの分布はcc Z-scoreと異なる可能性があるため、vz_thr=0.7だけでなく0.5/1.0も探索
- 改善が見られない場合: YZボラの「精度向上」がvzのZ-score計算で相殺される可能性がある
  （分母の252日rolling stdも変化するため）
- 次の実験候補: YZボラを直接パーセンタイルでゾーン分割する（Z-scoreを経由しない）

---

*生成スクリプト: `src/c3_yang_zhang_vol.py`*
*参照: `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`*
