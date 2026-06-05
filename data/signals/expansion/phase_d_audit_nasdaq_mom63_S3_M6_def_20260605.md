# Phase D Audit — nasdaq_mom63 × S3 × M6 defensive

作成日: 2026-06-05
最終更新日: 2026-06-05

## Candidate
- Signal: `nasdaq_mom63`
- Strategy: S3 (DH-W1 baseline)
- Method × Direction: M6 × defensive
- G3 Result: STANDARD_PASS_FULL (Session 3, 2026-06-05)

## 9+1 Metrics (Native Audit)

| metric | baseline | candidate | diff | better if |
|---|---|---|---|---|
| CAGR_OOS_pct | +18.9610 | +18.1044 | -0.8566 | positive |
| IS_OOS_gap_pp | -0.8820 | -1.4338 | -0.5518 | negative |
| Sharpe_OOS | +0.8445 | +0.8914 | +0.0470 | positive |
| MaxDD_full_pct | -34.5727 | -28.7403 | +5.8324 | positive (less neg) |
| Worst10Y_pct | +9.8385 | +10.3833 | +0.5448 | positive |
| P10_5Y_pct | +5.9362 | +5.9217 | -0.0145 | positive |
| Trades_per_yr | +17.6345 | +17.6345 | +0.0000 | lower / equal |
| WFE_full | +0.9757 | +1.0053 | +0.0296 | ≥ 1.0 |
| CI95_lo_window_CAGR_pct | +13.9487 | +12.9966 | -0.9521 | > 0 |
| +1 composite (n_imp / n_deg) | — | 6 / 3 | — | n_deg = 0 |

## WFA 50窓

- baseline full Sharpe : +0.938
- candidate full Sharpe : +0.975
- baseline WFE          : 0.976
- candidate WFE         : 1.005  (PASS ≥ 1.0: YES)
- baseline OOS Sharpe   : +1.046
- candidate OOS Sharpe  : +1.104
- candidate CI95 CAGR   : [+13.00%, +25.76%]  (PASS > 0: YES)
- candidate CI95 Sharpe : [+0.652, +1.308]

## Block Bootstrap 10,000 — Multi-Metric

| metric | actual diff | boot median | 95% CI of diff | P(cand better) |
|---|---|---|---|---|
| CAGR (pp) | -0.857 | -0.826 | [-4.714, +1.858] | 0.2949 |
| Sharpe | +0.0470 | +0.0532 | [-0.0172, +0.1317] | 0.9305 |
| MaxDD (pp, +=better) | +1.774 | +2.913 | [+0.297, +5.943] | 0.9882 |

OOS days = 1226, years = 4.87

## Verdict

- WFA WFE ≥ 1.0      : PASS  (actual = 1.005)
- WFA CI95_lo > 0    : PASS  (actual = +13.00%)
- Bootstrap P > 0.90 : PASS  (max P = 0.988)

**Final: ADOPT**

## Comparison vs Phase D BAA-10Y (procyclical)

| | BAA-10Y procyclical | This candidate (defensive) |
|---|---|---|
| Bootstrap P(CAGR) | 0.391 | 0.295 |
| Bootstrap CAGR diff median pp | -0.75 | -0.826 |
| Direction | procyclical (CAGR push) | defensive (DD reduction) |
| Verdict | REJECT | ADOPT |
