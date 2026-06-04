# Phase C 検証結果

作成日: 2026-06-04
最終更新日: 2026-06-04

## サマリ

- 評価バリアント数: 27 (overlay + standalone + AND/OR combos)
- **Pareto PASS**: **2** バリアント
- SPA (best-of-K honest test): p_consistent = 0.0640, best = combo_23_OR_41

## 採用候補 (Pareto PASS)

| mode | signal | asset | CAGR (cand vs base) | Sharpe (cand vs base) | MaxDD (cand vs base) | improved | degraded |
|---|---|---|---|---|---|---|---|
| overlay | 10Y real yield | IEF | +1.65% vs +2.18% | 0.41 vs 0.36 | -11.55% vs -23.92% | sharpe|maxdd |  |
| overlay | DXY | GLD | +11.94% vs +9.93% | 0.74 vs 0.66 | -32.37% vs -44.44% | cagr|sharpe|maxdd |  |

## 近接候補トップ5 (improved_axes - degraded_axes でランク)

| rank | mode | signal | asset | CAGR (cand vs base) | Sharpe (cand vs base) | MaxDD (cand vs base) | improved | degraded | score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | overlay | DXY | GLD | +11.94% vs +9.93% | 0.74 vs 0.66 | -32.37% vs -44.44% | cagr|sharpe|maxdd |  | 3 |
| 2 | overlay | 10Y real yield | IEF | +1.65% vs +2.18% | 0.41 vs 0.36 | -11.55% vs -23.92% | sharpe|maxdd |  | 2 |
| 3 | combo_OR_overlay_NDX | (BAA-10Y credit spread (HY-IG proxy)) OR (DXY) | NDX | +23.20% vs +16.11% | 0.82 vs 0.83 | -49.77% vs -36.40% | cagr | maxdd | 0 |
| 4 | combo_OR_overlay_NDX | (VIX level) OR (BAA-10Y credit spread (HY-IG proxy)) | NDX | +21.97% vs +16.11% | 0.81 vs 0.83 | -49.02% vs -36.40% | cagr | maxdd | 0 |
| 5 | combo_OR_overlay_NDX | (VIX level) OR (DXY) | NDX | +21.35% vs +16.11% | 0.78 vs 0.83 | -49.77% vs -36.40% | cagr | maxdd | 0 |

## 評価方法

**Overlay モード**: 信号値を倍率マップ {0:0.0, 1:0.5, 2:1.0, 3:1.5} で leverage 修正。対象資産の買い持ちと比較。
**Standalone モード**: 信号値を 3資産配分マップ (NDX/IEF/GLD) で配分。等加重ベースラインと比較。
**AND/OR Combo**: 上位4信号の組合せ。NDX overlay として評価。

**Pareto 判定**: CAGR +2pp 以上 OR Sharpe +0.05 OR MaxDD -5pp 改善が **2軸以上**、悪化なし。
**SPA**: Hansen 多重比較補正後の best-of-K p値。

## 次工程

Pareto PASS 2 件採用候補が確認できた。SPA p_consistent = 0.0640 (α=0.10 で有意)。

**推奨アクション:**
1. **採用候補の G3/G7/G9 厳格再評価**: 現状は run_g_series 簡易版 (50 windows, 500 boots)。本番採用前に audit cache (g20/g30) で正典 WFA を回す。
2. **動的 overlay マップ最適化**: 固定マップ {0:0.0, 1:0.5, 2:1.0, 3:1.5} を信号別に最適化すれば改善余地大。
3. **既存 NEW CANDIDATE への信号注入**: 採用候補の信号 (DXY, 10Y real yield 等) を Dyn2x3x ベースラインに乗せて再検証。
4. **SPA 改善**: K=27 → 採用 2 で p=0.0640。閾値 IC を上げて K を絞れば SPA も鋭くなる。
