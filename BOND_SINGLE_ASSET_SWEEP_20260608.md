# Bond 単独タイミングシグナル スイープ（保有 vs キャッシュ）

作成日: 2026-06-08
最終更新日: 2026-06-08

> `MULTIASSET_GOLD_BOND_SIGNAL_PLAN_20260608.md` Phase 2 の成果物。Bondリターン: `base_dataset.csv`（synth duration 1974-2009 + IEF 2009+）、キャッシュ: DTB3（3M T-bill）。
> **全期間(1974-2026)を主・単一OOS分割(2021-05-08)を従**として併記（統合レポート§6=全期間 と WFA=WFE/CI95 で手法整合）。

## ① 全期間(1974-2026) 9指標（CAGR_full 降順・主表）
> CAGR/Sharpe列=全期間。MaxDD/Worst10Y/P10/WFE/CI95も全期間。

| Strategy | CAGR<br>⓽<br>_<br>OOS | IS-OOS<br>gap<br>CAGR | Sharpe<br>ⓒ<br>_OOS | MaxDD<br>ⓒ | Worst<br>10Y★<br>⓽<br>CAGR | P10<br>⓽<br>5Y▷<br>CAGR | Trade<br>ⓞ<br>(回/<br>年) | Overfit<br>ⓞ<br>(WFE) | CI95<br>ⓡ<br>_lo |
|:---------|-------------:|-------------:|---------------:|------:|----------------------:|-------------------:|------------------:|:----------------:|-----------:|
| bond_mom252 |  +5.1% | +3.23pp | +0.88 ◎ | -15.0% |  +0.9% |  +1.2% |  57 | ⚠ MED<br>(19.6) | +3.158 |
| bond_mom126 |  +4.6% | +5.30pp | +0.76 | -18.3% |  +0.5% |  +1.2% |  59 | ⚠ MED<br>(3.6) | -1.241 |
| bond_ma100 |  +4.6% | +3.07pp | +0.77 ◎ | -16.9% |  +0.8% |  +1.3% |  64 | ✅ LOW<br>(1.1) | +0.525 |
| ALL_CASH |  +4.5% | +1.03pp | +20.25 ★ |  -0.0% |  +0.3% |  +0.3% | 125 | ⚠ MED<br>(9.4) | +135.968 |
| credit_spread_hi |  +4.3% | +4.30pp | +0.77 | -23.2% |  -0.0% |  +0.9% |  87 | ⚠ MED<br>(64.5) | +12.600 |
| bond_ma200 |  +4.3% | +5.65pp | +0.72 | -13.9% |  +0.2% |  +0.7% |  58 | ⚠ MED<br>(3.8) | -1.306 |
| termprem_lo |  +2.9% | +3.18pp | +0.51 | -31.7% |  -1.0% |  -0.2% |  75 | ⚠ MED<br>(34.7) | +2.915 |
| fed_easy |  +2.4% | +0.62pp | +0.40 | -28.7% |  -0.6% |  +0.0% |  73 | ⚠ MED<br>(46.3) | -10.940 |
| BUY_AND_HOLD |  +1.4% | +2.83pp | +0.21 | -50.1% |  -4.8% |  -3.2% |   0 | ✅ LOW<br>(1.1) | -0.118 |

## ② 参考: OOS(2021-05-08) 単一分割 CAGR/Sharpe（従表）

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

## 所見（全期間ベース・暫定／要 Phase 2.4 WFA・bootstrap）

- 全期間(1974-2026) 常時保有(B&H) CAGR=**+1.42%** / 常時キャッシュ CAGR=**+4.46%**。
- アクティブ最良 **bond_mom252**: CAGR=+5.08%, Sharpe_full=+0.878, MaxDD=-15.0%, Worst10Y=+0.92%, WFE=19.59, CI95_lo=+3.158。
- **全期間でキャッシュ超のシグナル**: bond_mom252, bond_mom126, bond_ma100。
- ⚠ Trades/WFE/CI95 は NAV代理の screening 値（正式は Phase 2.4）。ALL_CASH の Sharpe は無リスク近似による退化値で参考外。

## シグナル仮説

- **bond_mom252** — momentum: hold when 12m return > 0 (判定vs B&H: STRONG_PASS)
- **bond_mom126** — momentum: hold when 6m return > 0 (判定vs B&H: MARGINAL)
- **bond_ma100** — trend: hold when price > 100d MA (判定vs B&H: STRONG_PASS)
- **ALL_CASH** — baseline: always in T-bill cash (判定vs B&H: STRONG_PASS)
- **credit_spread_hi** — flight-to-quality: hold when BAA-AAA z >= 0.5 (判定vs B&H: STRONG_PASS)
- **bond_ma200** — trend: hold when price > 200d MA (判定vs B&H: MARGINAL)
- **termprem_lo** — hold when 30y-10y term premium z <= 0 (判定vs B&H: STRONG_PASS)
- **fed_easy** — hold when DFF-10y z <= 0 (easy/inverted) (判定vs B&H: STRONG_PASS)
- **BUY_AND_HOLD** — baseline: always invested (判定vs B&H: FAIL)
