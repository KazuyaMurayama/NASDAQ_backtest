# Phase D Audit — nfci_z52w × S3 × M2 defensive

作成日: 2026-06-05
最終更新日: 2026-06-05

## Candidate
- Signal: `nfci_z52w`
- Strategy: S3 (DH-W1 baseline)
- Method × Direction: M2 × defensive
- G3 Result: STANDARD_PASS_FULL (Session 3, 2026-06-05)

## 9+1 Metrics (Native Audit)

| metric | baseline | candidate | diff | better if |
|---|---|---|---|---|
| CAGR_OOS_pct | +18.9610 | +16.1330 | -2.8280 | positive |
| IS_OOS_gap_pp | -0.8820 | +3.2226 | +4.1046 | negative |
| Sharpe_OOS | +0.8445 | +0.7649 | -0.0795 | positive |
| MaxDD_full_pct | -34.5727 | -30.2494 | +4.3233 | positive (less neg) |
| Worst10Y_pct | +9.8385 | +9.9024 | +0.0639 | positive |
| P10_5Y_pct | +5.9362 | +7.2230 | +1.2868 | positive |
| Trades_per_yr | +17.6345 | +17.6345 | +0.0000 | lower / equal |
| WFE_full | +0.9757 | +1.0033 | +0.0276 | ≥ 1.0 |
| CI95_lo_window_CAGR_pct | +13.9487 | +14.7629 | +0.8143 | > 0 |
| +1 composite (n_imp / n_deg) | — | 6 / 3 | — | n_deg = 0 |

## WFA 50窓

- baseline full Sharpe : +0.938
- candidate full Sharpe : +1.009
- baseline WFE          : 0.976
- candidate WFE         : 1.003  (PASS ≥ 1.0: YES)
- baseline OOS Sharpe   : +1.046
- candidate OOS Sharpe  : +0.982
- candidate CI95 CAGR   : [+14.76%, +29.68%]  (PASS > 0: YES)
- candidate CI95 Sharpe : [+0.683, +1.342]

## Block Bootstrap 10,000 — Multi-Metric

| metric | actual diff | boot median | 95% CI of diff | P(cand better) |
|---|---|---|---|---|
| CAGR (pp) | -2.828 | -2.784 | [-10.008, +4.099] | 0.2102 |
| Sharpe | -0.0795 | -0.0719 | [-0.2961, +0.1743] | 0.2712 |
| MaxDD (pp, +=better) | +1.361 | -0.437 | [-6.544, +7.320] | 0.4425 |

OOS days = 1226, years = 4.87

## Verdict

- WFA WFE ≥ 1.0      : PASS  (actual = 1.003)
- WFA CI95_lo > 0    : PASS  (actual = +14.76%)
- Bootstrap P > 0.90 : FAIL  (max P = 0.443)

**Final: REJECT**

## Comparison vs Phase D BAA-10Y (procyclical)

| | BAA-10Y procyclical | This candidate (defensive) |
|---|---|---|
| Bootstrap P(CAGR) | 0.391 | 0.210 |
| Bootstrap CAGR diff median pp | -0.75 | -2.784 |
| Direction | procyclical (CAGR push) | defensive (DD reduction) |
| Verdict | REJECT | REJECT |
