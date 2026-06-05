# Session 5 Transfer Audit — nasdaq_mom63 x M6 defensive on E4_Active

作成日: 2026-06-05
最終更新日: 2026-06-05

## Target Baseline
- ID: `E4_Active`
- Label: E4 RegimeKLT (current §1 Active)
- Cache: `audit_results/_cache/e4_nav_cache.pkl`

## Overlay (transferred from Session 4 ADOPT)
- Signal: `nasdaq_mom63` (NASDAQ 63-day momentum, daily lag)
- Method: M6 (threshold-proxy continuous tilt)
- Direction: defensive (high momentum -> reduce leverage)
- Multiplier: signal_q {0,1,2,3} -> {1.1, 1.0, 0.9, 0.8}

## 9+1 Metrics (Native Audit)

| metric | baseline | candidate | diff | better if |
|---|---|---|---|---|
| CAGR_OOS_pct | +33.5301 | +32.1596 | -1.3705 | positive |
| IS_OOS_gap_pp | -1.8127 | -2.5484 | -0.7358 | negative |
| Sharpe_OOS | +0.8793 | +0.9142 | +0.0349 | positive |
| MaxDD_full_pct | -60.0102 | -58.4971 | +1.5131 | positive (less neg) |
| Worst10Y_pct | +17.5191 | +15.2693 | -2.2498 | positive |
| P10_5Y_pct | +13.4368 | +10.6363 | -2.8005 | positive |
| Trades_per_yr | +75.8225 | +90.8147 | +14.9922 | lower / equal |
| WFE_full | +0.9281 | +0.9581 | +0.0300 | >= 1.0 |
| CI95_lo_window_CAGR_pct | +26.5093 | +24.4094 | -2.0999 | > 0 |
| +1 composite (n_imp / n_deg) | - | 4 / 5 | - | n_deg = 0 |

## WFA 50 windows
- baseline full Sharpe : +0.934
- candidate full Sharpe : +0.949
- baseline WFE          : 0.928
- candidate WFE         : 0.958  (PASS >= 1.0: NO)
- baseline mean OOS Sharpe : +0.664
- candidate mean OOS Sharpe : +0.712
- candidate CI95 CAGR   : [+24.41%, +51.78%]  (PASS > 0: YES)

## Block Bootstrap 10,000 — Multi-Metric

| metric | actual diff | boot median | 95% CI of diff | P(cand better) |
|---|---|---|---|---|
| CAGR (pp) | -1.363 | -1.161 | [-9.920, +3.747] | 0.3548 |
| Sharpe | +0.0349 | +0.0432 | [-0.0349, +0.1239] | 0.8577 |
| MaxDD (pp, +=better) | +1.534 | +4.048 | [-0.354, +8.223] | 0.9643 |

OOS days = 1226, years = 4.87

## Verdict

- WFA WFE >= 1.0      : FAIL  (actual = 0.958)
- WFA CI95_lo > 0    : PASS  (actual = +24.41%)
- Bootstrap P > 0.90 : PASS  (max P = 0.9643)

**Final: NEEDS_FURTHER_WORK**
