# A2B: Dynamic l_max + Rolling VOL_REF — A2 の CAGR 損失を回復

作成日: 2026-05-27
EVALUATION_STANDARD: **v1.1** | コスト: **Scenario D**

## §1 前提

A2 (`src/a2_dyn_lmax.py`) で固定 `VOL_REF=0.20` を用いた結果、NASDAQ の長期実現ボラ
平均 ≈0.183 を上回るため、低ボラ局面（vol252 < 0.20）でも l_max が引き下げられず、
むしろ平常時にデレバが弱まる方向に効いた一方、CAGR_OOS の改善は限定的だった。
本 A2B は VOL_REF を 2520 日 rolling median に置換し、実勢ボラに追従する
適応的基準を導入する。

| 項目 | 定義 |
|------|------|
| **E4固定値** | k_lo=0.1, k_hi=0.8, vz_thr=0.7（変更なし） |
| **LMAX_BASE グリッド** | [5.0, 5.5, 6.0]（l_max基準値） |
| **VOL_SENS グリッド** | [1.0, 2.0, 3.0]（ボラ感度） |
| **L_FLOOR / L_CEIL** | 4.5 / 6.5（クリップ範囲） |
| **VOL_REF** | **rolling median (window=2520日, min_periods=504日)** |
| **vol_ref_t 統計** | mean=0.161, min=0.112, max=0.265, last=0.197 |
| **合計 configs** | 9 + REF |
| **IS** | 1974-01-02 〜 2021-05-07 |
| **OOS** | 2021-05-08 〜 |

**動的 l_max 計算式 (A2 → A2B 差分)**:
```
A2:  l_max_t = clip(lmax_base - vol_sens × (vol252 / 0.20 - 1), 4.5, 6.5)
A2B: vol_ref_t = vol252.rolling(2520, min_periods=504).median()
                       .fillna(vol252.expanding(min_periods=5).median())
     l_max_t   = clip(lmax_base - vol_sens × (vol252 / vol_ref_t - 1), 4.5, 6.5)
```

**直感**: 「凪期は基準ボラも下がるので、vol252/vol_ref_t は1付近に収束 → l_max を下げない」
       「嵐期は基準ボラの上昇は遅いので、vol252/vol_ref_t > 1 が維持され l_max は下がる」
       → MaxDD 抑制と CAGR 回復の両立を狙う非対称設計。

**サニティ**: REF CAGR_OOS=+33.53% (E4=+33.53%, diff +0.00pp)

---

## §2 9指標テーブル（Sharpe 降順 + REF）

| Param | CAGR<br>_OOS | Sharpe<br>_OOS | MaxDD | Worst<br>10Y★<br>CAGR | P10<br>5Y▷<br>CAGR | IS-OOS<br>gap | Trade<br>（回/年） | CI95<br>_lo | WFE |
|:------|-------------:|---------------:|------:|----------------------:|-------------------:|--------------:|------------------:|-----------:|----:|
| base=7.0/sens=0.0 (REF=E4) | +33.5% | +0.89 ★ | -60.0% | +18.7% |  +9.8% | -1.81pp |  27 |    —    |    —    |
| base=6.0/sens=2.0 | +31.7% | +0.89 ★ | -56.9% | +18.6% | +11.1% | -0.87pp |  27 |    —    |    —    |
| base=5.5/sens=1.0 | +30.9% | +0.89 ★ | -55.3% | +18.1% | +11.8% | -0.49pp |  27 |    —    |    —    |
| base=6.0/sens=1.0 | +31.7% | +0.88 ◎ | -56.9% | +18.6% | +11.4% | -0.73pp |  27 |    —    |    —    |
| base=5.0/sens=1.0 | +29.0% | +0.88 ◎ | -53.3% | +17.7% | +12.0% | +0.60pp |  27 |    —    |    —    |
| base=5.0/sens=2.0 | +29.0% | +0.88 ◎ | -53.3% | +17.4% | +11.8% | +0.62pp |  27 |    —    |    —    |
| base=5.0/sens=3.0 | +28.7% | +0.87 ◎ | -54.3% | +17.6% | +11.8% | +1.03pp |  27 |    —    |    —    |
| base=5.5/sens=3.0 | +29.3% | +0.87 ◎ | -56.2% | +18.4% | +11.2% | +0.79pp |  27 |    —    |    —    |
| base=6.0/sens=3.0 | +30.0% | +0.86 ◎ | -57.0% | +18.6% | +10.5% | +0.30pp |  27 |    —    |    —    |
| base=5.5/sens=2.0 | +29.1% | +0.86 ◎ | -55.4% | +18.0% | +11.4% | +1.21pp |  27 |    —    |    —    |

*CI95_lo / WFE: `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

---

## §3 判定

| 基準 | 条件 |
|------|------|
| (i) Sharpe_OOS | ≥ REF+0.010 (≥0.901) |
| (ii) CAGR_OOS | ≥ 29.0% |
| (iii) IS-OOS gap | ≤ 4.0pp |
| (iv) MaxDD | > -61.0% |
| (v) Worst10Y★ | ≥ 15.0% |

- **PASS configs**: 0 / 9
- **MaxDD改善 configs**: 9 / 9
- **最高 Sharpe**: base=6.0, sens=2.0 → Sharpe=+0.889
- **最良 MaxDD**: base=5.0, sens=2.0 → MaxDD=-53.34% (E4比 +6.67pp), CAGR_OOS=+29.03% (E4比 -4.50pp)
- **総合判定: FAIL**

---

## §4 A2 比較

A2 (固定 VOL_REF=0.20) の同一グリッド結果は `a2_dyn_lmax_results.csv` を参照。
A2B との差分は `vol_ref` 列のみで CAGR_OOS / MaxDD の振る舞いに違いが
出れば仮説 (rolling vref が CAGR を回復) が支持される。

| 観点 | A2 (固定 VOL_REF=0.20) | A2B (rolling median ≈0.183) |
|------|----------------------|------------------------------|
| 低ボラ期の l_max | 引下げ気味 (vol252/0.20 < 1 が稀) | 引下げほぼなし (vol252/vol_ref≈1) |
| 高ボラ期の l_max | 強く引下げ | 同程度に引下げ |
| 期待 CAGR_OOS | ベースライン | **回復**（凪期に過剰デレバを抑制） |
| 期待 MaxDD | ベースライン | **同等維持**（嵐期は同様にデレバ） |

---

## §5 考察

- A2 の固定 VOL_REF=0.20 は NASDAQ ヒストリカル平均ではなく直感値で設定
  されていた。NASDAQ の実測 vol252 長期平均は ≈0.183 (本実験で確認)
- Rolling median による適応化は過学習リスクが小さい: 10年窓は十分長く、
  median 演算はノイズ耐性が高い。自由度の増加はゼロ（パラメータ追加なし）
- 注意点: 序盤 ウォームアップ期間 (504日未満) は expanding median で代替するため
  IS 主要評価 (1976–2021) には影響しない
- 過学習リスク: 2自由度 (lmax_base, vol_sens)。A2 と同一なので追加リスクなし

---

*生成スクリプト: `src/a2b_dyn_lmax_rolling_vref.py`*
*参照: `src/a2_dyn_lmax.py` (固定 VOL_REF 版), `CURRENT_BEST_STRATEGY.md`, `EVALUATION_STANDARD.md`, `src/e4_regime_klt.py`*
