# Session 5 Transfer Audit — nasdaq_mom63 x M6 defensive on S2_D5

作成日: 2026-06-05
最終更新日: 2026-06-05

## Target Baseline
- ID: `S2_D5`
- Label: S2 (D5 vz=0.65, lmax=5)
- Cache: `audit_results/_cache/vz065lmax5_nav_cache.pkl`

## Overlay (transferred from Session 4 ADOPT)
- Signal: `nasdaq_mom63` (NASDAQ 63-day momentum, daily lag)
- Method: M6 (threshold-proxy continuous tilt)
- Direction: defensive (high momentum -> reduce leverage)
- Multiplier: signal_q {0,1,2,3} -> {1.1, 1.0, 0.9, 0.8}

## 9+1 Metrics (Native Audit)

| metric | baseline | candidate | diff | better if |
|---|---|---|---|---|
| CAGR_OOS_pct | +33.4871 | +31.1247 | -2.3624 | positive |
| IS_OOS_gap_pp | -3.9117 | -3.7349 | +0.1769 | negative |
| Sharpe_OOS | +0.9368 | +0.9552 | +0.0184 | positive |
| MaxDD_full_pct | -51.8185 | -50.6312 | +1.1873 | positive (less neg) |
| Worst10Y_pct | +18.5687 | +15.2042 | -3.3646 | positive |
| P10_5Y_pct | +14.6325 | +12.0969 | -2.5357 | positive |
| Trades_per_yr | +76.0523 | +90.9679 | +14.9156 | lower / equal |
| WFE_full | +0.9343 | +0.9626 | +0.0283 | >= 1.0 |
| CI95_lo_window_CAGR_pct | +24.8187 | +22.7181 | -2.1006 | > 0 |
| +1 composite (n_imp / n_deg) | - | 3 / 6 | - | n_deg = 0 |

## WFA 50 windows
- baseline full Sharpe : +1.005
- candidate full Sharpe : +1.017
- baseline WFE          : 0.934
- candidate WFE         : 0.963  (PASS >= 1.0: NO)
- baseline mean OOS Sharpe : +0.708
- candidate mean OOS Sharpe : +0.755
- candidate CI95 CAGR   : [+22.72%, +44.82%]  (PASS > 0: YES)

## Block Bootstrap 10,000 — Multi-Metric

| metric | actual diff | boot median | 95% CI of diff | P(cand better) |
|---|---|---|---|---|
| CAGR (pp) | -2.351 | -2.250 | [-10.571, +2.384] | 0.2010 |
| Sharpe | +0.0184 | +0.0273 | [-0.0488, +0.1078] | 0.7580 |
| MaxDD (pp, +=better) | +1.424 | +3.204 | [-0.660, +6.965] | 0.9438 |

OOS days = 1226, years = 4.87

## Verdict

- WFA WFE >= 1.0      : FAIL  (actual = 0.963)
- WFA CI95_lo > 0    : PASS  (actual = +22.72%)
- Bootstrap P > 0.90 : PASS  (max P = 0.9438)

**Final: NEEDS_FURTHER_WORK**
