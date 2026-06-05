# Phase D Audit — vix_mom21 × S3 × M2 defensive

作成日: 2026-06-05
最終更新日: 2026-06-05

## Candidate
- Signal: `vix_mom21`
- Strategy: S3 (DH-W1 baseline)
- Method × Direction: M2 × defensive
- G3 Result: STANDARD_PASS_FULL (Session 3, 2026-06-05)

## 9+1 Metrics (Native Audit)

| metric | baseline | candidate | diff | better if |
|---|---|---|---|---|
| CAGR_OOS_pct | +18.9610 | +19.5464 | +0.5854 | positive |
| IS_OOS_gap_pp | -0.8820 | -3.6504 | -2.7684 | negative |
| Sharpe_OOS | +0.8445 | +0.9399 | +0.0955 | positive |
| MaxDD_full_pct | -34.5727 | -31.1173 | +3.4553 | positive (less neg) |
| Worst10Y_pct | +9.8385 | +8.0806 | -1.7579 | positive |
| P10_5Y_pct | +5.9362 | +6.3883 | +0.4520 | positive |
| Trades_per_yr | +17.6345 | +17.6345 | +0.0000 | lower / equal |
| WFE_full | +0.9757 | +0.9673 | -0.0084 | ≥ 1.0 |
| CI95_lo_window_CAGR_pct | +13.9487 | +11.9991 | -1.9496 | > 0 |
| +1 composite (n_imp / n_deg) | — | 6 / 3 | — | n_deg = 0 |

## WFA 50窓

- baseline full Sharpe : +0.938
- candidate full Sharpe : +0.921
- baseline WFE          : 0.976
- candidate WFE         : 0.967  (PASS ≥ 1.0: NO)
- baseline OOS Sharpe   : +1.046
- candidate OOS Sharpe  : +0.954
- candidate CI95 CAGR   : [+12.00%, +25.93%]  (PASS > 0: YES)
- candidate CI95 Sharpe : [+0.555, +1.226]

## Block Bootstrap 10,000 — Multi-Metric

| metric | actual diff | boot median | 95% CI of diff | P(cand better) |
|---|---|---|---|---|
| CAGR (pp) | +0.585 | -0.146 | [-9.045, +7.005] | 0.4851 |
| Sharpe | +0.0955 | +0.0611 | [-0.2046, +0.3477] | 0.6687 |
| MaxDD (pp, +=better) | +3.129 | +4.016 | [-1.920, +11.786] | 0.9021 |

OOS days = 1226, years = 4.87

## Verdict

- WFA WFE ≥ 1.0      : FAIL  (actual = 0.967)
- WFA CI95_lo > 0    : PASS  (actual = +12.00%)
- Bootstrap P > 0.90 : PASS  (max P = 0.902)

**Final: NEEDS_FURTHER_WORK**

## Comparison vs Phase D BAA-10Y (procyclical)

| | BAA-10Y procyclical | This candidate (defensive) |
|---|---|---|
| Bootstrap P(CAGR) | 0.391 | 0.485 |
| Bootstrap CAGR diff median pp | -0.75 | -0.146 |
| Direction | procyclical (CAGR push) | defensive (DD reduction) |
| Verdict | REJECT | NEEDS_FURTHER_WORK |
