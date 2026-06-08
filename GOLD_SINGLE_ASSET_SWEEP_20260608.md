# Gold 単独タイミングシグナル スイープ（保有 vs キャッシュ）

作成日: 2026-06-08
最終更新日: 2026-06-08

> `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` Phase 2 の成果物。Goldリターン: `base_dataset.csv`（LBMA 1974+）、キャッシュ: DTB3、実質金利: DGS10−CPI YoY、DXY: 2006+（以前はキャッシュ扱い）。
> **全期間(1974-2026)を主・単一OOS分割(2021-05-08)を従**として併記（統合レポート§6=全期間 と WFA=WFE/CI95 で手法整合）。

## ① 全期間(1974-2026) 9指標（CAGR_full 降順・主表）
> CAGR/Sharpe列=全期間。MaxDD/Worst10Y/P10/WFE/CI95も全期間。

| Strategy | CAGR<br>⓽<br>_<br>OOS | IS-OOS<br>gap<br>CAGR | Sharpe<br>ⓒ<br>_OOS | MaxDD<br>ⓒ | Worst<br>10Y★<br>⓽<br>CAGR | P10<br>⓽<br>5Y▷<br>CAGR | Trade<br>ⓞ<br>(回/<br>年) | Overfit<br>ⓞ<br>(WFE) | CI95<br>ⓡ<br>_lo |
|:---------|-------------:|-------------:|---------------:|------:|----------------------:|-------------------:|------------------:|:----------------:|-----------:|
| gold_realyield_lo |  +9.0% | +2.82pp | +0.68 | -29.3% |  +0.5% |  +0.1% |  71 | ⚠ MED<br>(14.0) | -3.023 |
| gold_mom126 |  +8.9% | -7.40pp | +0.62 | -46.7% |  -0.7% |  +0.5% |  66 | ⚠ MED<br>(34.4) | -9.891 |
| gold_ma200 |  +7.8% | -7.15pp | +0.56 | -48.4% |  -1.8% |  -0.7% |  67 | ⚠ MED<br>(41.7) | -8.730 |
| gold_mom252 |  +7.3% | -5.50pp | +0.53 | -46.7% |  -1.3% |  -1.4% |  66 | ⚠ MED<br>(48.4) | -6.407 |
| gold_dxy_lo |  +6.6% | -2.50pp | +0.92 ★ | -23.9% |  -0.2% |  +2.2% | 117 | ⚠ MED<br>(164.6) | +96.644 |
| gold_ma100 |  +6.5% | -10.23pp | +0.49 | -47.2% |  -1.7% |  -1.3% |  74 | ✅ LOW<br>(0.8) | +0.167 |
| gold_infl_hi |  +6.5% | +4.59pp | +0.55 | -46.7% |  -0.7% |  -0.7% |  87 | ⚠ MED<br>(38.6) | +4.410 |
| BUY_AND_HOLD |  +5.1% | -14.24pp | +0.35 | -79.7% | -10.0% |  -6.8% |   0 | ✅ LOW<br>(0.9) | +0.030 |
| ALL_CASH |  +4.5% | +1.03pp | +20.25 ★ |  -0.0% |  +0.3% |  +0.3% | 136 | ⚠ MED<br>(9.5) | +136.101 |

## ② 参考: OOS(2021-05-08) 単一分割 CAGR/Sharpe（従表）

| Strategy | CAGR<br>⓽<br>_<br>OOS | IS-OOS<br>gap<br>CAGR | Sharpe<br>ⓒ<br>_OOS | MaxDD<br>ⓒ | Worst<br>10Y★<br>⓽<br>CAGR | P10<br>⓽<br>5Y▷<br>CAGR | Trade<br>ⓞ<br>(回/<br>年) | Overfit<br>ⓞ<br>(WFE) | CI95<br>ⓡ<br>_lo |
|:---------|-------------:|-------------:|---------------:|------:|----------------------:|-------------------:|------------------:|:----------------:|-----------:|
| BUY_AND_HOLD | +18.1% | -14.24pp | +1.04 ★ | -79.7% | -10.0% |  -6.8% |   0 | ✅ LOW<br>(0.9) | +0.030 |
| gold_ma100 | +15.8% | -10.23pp | +1.00 ★ | -47.2% |  -1.7% |  -1.3% |  74 | ✅ LOW<br>(0.8) | +0.167 |
| gold_mom126 | +15.6% | -7.40pp | +0.98 ★ | -46.7% |  -0.7% |  +0.5% |  66 | ⚠ MED<br>(34.4) | -9.891 |
| gold_ma200 | +14.3% | -7.15pp | +0.91 ★ | -48.4% |  -1.8% |  -0.7% |  67 | ⚠ MED<br>(41.7) | -8.730 |
| gold_mom252 | +12.2% | -5.50pp | +0.80 ◎ | -46.7% |  -1.3% |  -1.4% |  66 | ⚠ MED<br>(48.4) | -6.407 |
| gold_dxy_lo |  +8.9% | -2.50pp | +0.70 | -23.9% |  -0.2% |  +2.2% | 117 | ⚠ MED<br>(164.6) | +96.644 |
| gold_realyield_lo |  +6.5% | +2.82pp | +0.55 | -29.3% |  +0.5% |  +0.1% |  71 | ⚠ MED<br>(14.0) | -3.023 |
| ALL_CASH |  +3.5% | +1.03pp | +29.34 ★ |  -0.0% |  +0.3% |  +0.3% | 136 | ⚠ MED<br>(9.5) | +136.101 |
| gold_infl_hi |  +2.3% | +4.59pp | +0.27 | -46.7% |  -0.7% |  -0.7% |  87 | ⚠ MED<br>(38.6) | +4.410 |

*CI95_lo / Overfit(WFE): `—` は WFA 未計算（`§3.12` 促進ゲート待ち）。計算後は `src/g2_wfa_shortlist.py` で自動補完予定。*  
*進格条件: **CI95_lo > 0**（期待リターンがプラス）/ **0.5 ≤ WFE ≤ 2.0**（過学習なし）。Sharpe マーカ: ◎ = Sharpe_OOS > +0.770 / ★ = > +0.885。列名 `Worst10Y★` の ★ は最悪10年CAGRの重要指標記号（Sharpe マーカとは別意味）。*

*Sharpe ◎/★ マーカは NASDAQ ベースライン基準（参考）。*

## 所見（全期間ベース・暫定／要 Phase 2.4 WFA・bootstrap）

- 全期間(1974-2026) 常時保有(B&H) CAGR=**+5.09%** / 常時キャッシュ CAGR=**+4.46%**。
- アクティブ最良 **gold_realyield_lo**: CAGR=+9.05%, Sharpe_full=+0.683, MaxDD=-29.3%, Worst10Y=+0.53%, WFE=14.05, CI95_lo=-3.023。
- **全期間でキャッシュ超のシグナル**: gold_realyield_lo, gold_mom126, gold_ma200, gold_mom252, gold_dxy_lo, gold_ma100, gold_infl_hi。
- ⚠ Trades/WFE/CI95 は NAV代理の screening 値（正式は Phase 2.4）。ALL_CASH の Sharpe は無リスク近似による退化値で参考外。

## シグナル仮説

- **gold_realyield_lo** — hold when real 10y yield z <= 0 (low real yield favors gold) (判定vs B&H: MARGINAL)
- **gold_mom126** — momentum: hold when 6m return > 0 (判定vs B&H: MARGINAL)
- **gold_ma200** — trend: hold when price > 200d MA (判定vs B&H: MARGINAL)
- **gold_mom252** — momentum: hold when 12m return > 0 (判定vs B&H: MARGINAL)
- **gold_dxy_lo** — hold when DXY z <= 0 (weak dollar favors gold; DXY 2006+) (判定vs B&H: MARGINAL)
- **gold_ma100** — trend: hold when price > 100d MA (判定vs B&H: MARGINAL)
- **gold_infl_hi** — hold when CPI YoY z >= 0.5 (accelerating inflation) (判定vs B&H: MARGINAL)
- **BUY_AND_HOLD** — baseline: always invested (判定vs B&H: FAIL)
- **ALL_CASH** — baseline: always in T-bill cash (判定vs B&H: MARGINAL)
