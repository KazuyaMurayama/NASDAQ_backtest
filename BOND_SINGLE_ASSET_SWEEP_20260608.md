# Bond 単独タイミングシグナル スイープ（保有 vs キャッシュ）

作成日: 2026-06-08
最終更新日: 2026-06-08

> `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` Phase 2.2 の成果物。
> Bondリターン: `base_dataset.csv`（synth duration 1974-2009 + IEF 2009+）、キャッシュ: DTB3（3M T-bill）。OOS分割: 2021-05-08。
> **Trades/WFE/CI95 は NAV代理の screening 値**（正式は Phase 2.4 WFA）。

## 9指標スイープ結果（CAGR_OOS降順）

| Strategy | CAGR<br>⓽<br>_<br>OOS | IS-OOS<br>gap<br>CAGR | Sharpe<br>ⓒ<br>_OOS | MaxDD<br>ⓒ | Worst<br>10Y★<br>⓽<br>CAGR | P10<br>⓽<br>5Y▷<br>CAGR | Trade<br>ⓞ<br>(回/<br>年) | Overfit<br>ⓞ<br>(WFE) | CI95<br>ⓡ<br>_lo |
|:---------|-------------:|-------------:|---------------:|------:|----------------------:|-------------------:|------------------:|:----------------:|-----------:|
| ALL_CASH |  +3.5% | +1.03pp | +29.34 ★ |  -0.0% |  +0.3% |  +0.3% | 125 | ⚠ MED<br>(9.4) | +135.968 |
| bond_mom252 |  +2.2% | +3.23pp | +0.60 | -15.0% |  +0.9% |  +1.2% |  57 | ⚠ MED<br>(19.6) | +3.158 |
| fed_easy |  +1.8% | +0.62pp | +0.36 | -28.7% |  -0.6% |  +0.0% |  73 | ⚠ MED<br>(46.3) | -10.940 |
| bond_ma100 |  +1.8% | +3.07pp | +0.38 | -16.9% |  +0.8% |  +1.3% |  64 | ✅ LOW<br>(1.1) | +0.525 |
| credit_spread_hi |  +0.4% | +4.30pp | +0.10 | -23.2% |  -0.0% |  +0.9% |  87 | ⚠ MED<br>(64.5) | +12.600 |
| termprem_lo |  +0.0% | +3.18pp | +0.04 | -31.7% |  -1.0% |  -0.2% |  75 | ⚠ MED<br>(34.7) | +2.915 |
| bond_mom126 |  -0.2% | +5.30pp | -0.02 | -18.3% |  +0.5% |  +1.2% |  59 | ⚠ MED<br>(3.6) | -1.241 |
| bond_ma200 |  -0.8% | +5.65pp | -0.14 | -13.9% |  +0.2% |  +0.7% |  58 | ⚠ MED<br>(3.8) | -1.306 |
| BUY_AND_HOLD |  -1.1% | +2.83pp | -0.11 | -50.1% |  -4.8% |  -3.2% |   0 | ✅ LOW<br>(1.1) | -0.118 |

*CI95_lo / Overfit(WFE): `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

*Sharpe ◎/★ マーカは NASDAQ ベースライン基準（参考）。*

## シグナル仮説

- **ALL_CASH** — baseline: always in T-bill cash  (判定: STRONG_PASS)
- **bond_mom252** — momentum: hold when 12m return > 0  (判定: STRONG_PASS)
- **fed_easy** — hold when DFF-10y z <= 0 (easy/inverted)  (判定: STRONG_PASS)
- **bond_ma100** — trend: hold when price > 100d MA  (判定: STRONG_PASS)
- **credit_spread_hi** — flight-to-quality: hold when BAA-AAA z >= 0.5  (判定: STRONG_PASS)
- **termprem_lo** — hold when 30y-10y term premium z <= 0  (判定: STRONG_PASS)
- **bond_mom126** — momentum: hold when 6m return > 0  (判定: MARGINAL)
- **bond_ma200** — trend: hold when price > 200d MA  (判定: MARGINAL)
- **BUY_AND_HOLD** — baseline: always invested  (判定: FAIL)

## 所見（暫定・要 Phase 2.4 WFA 確認）

- OOS(2021-05-08〜)はBond弱気相場。**常時保有(B&H) CAGR_OOS=-1.14%・MaxDD≈-50%** と劣悪 → 「vs B&H」の判定は楽観的に出る点に注意。
- **本質的ベンチマークはキャッシュ**: ALL_CASH CAGR_OOS=**+3.53%**。
- アクティブ最良は **bond_mom252**（CAGR_OOS=+2.16%, MaxDD=-15.0%）。
- ⚠ **どのBondシグナルもOOSでキャッシュを上回れなかった**（この局面はキャッシュ保有が正解。タイミングはDD圧縮には寄与: 例 bond_mom252 MaxDD≈-15% vs B&H -50%）。
- ALL_CASH の Sharpe は無リスク近似による退化値のため参考外。
