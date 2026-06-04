# Manual signal CSV format

Each signal CSV must have:
- `Date` column: ISO date string (YYYY-MM-DD)
- `value` column: numeric value

Sample template: `_template.csv`

## Signal → file mapping

| Signal ID | File | Update frequency | Source |
|---|---|---|---|
| 13 | aaii_weekly.csv | weekly | https://www.aaii.com/sentimentsurvey/sent_results |
| 14 | naaim_weekly.csv | weekly | https://www.naaim.org/programs/naaim-exposure-index/ |
| 20 | finra_margin_debt_monthly.csv | monthly | https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics |
| 32 | gdpnow_atlanta.csv | weekly | https://www.atlantafed.org/cqer/research/gdpnow |
| 33 | nyfed_nowcast.csv | weekly | https://www.newyorkfed.org/research/policy/nowcast |
| 34 | citi_surprise_usmi.csv | weekly | scrape from Citigroup (low_paid tier) |
| 35 | cleveland_inflation_nowcast.csv | monthly | https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting |
| 37 | ndx_eps_revision_4wk.csv | weekly | EODHD / Finnhub (low_paid) |
| 38 | equity_risk_premium.csv | daily | computed (NDX fwd yld − 10Y TIPS) |
| 39 | ndx_forward_pe_zscore.csv | weekly | EODHD / Finnhub |
| 40 | mag7_eps_revision.csv | weekly | aggregated from Finnhub |
| 46 | fomc_blackout.csv | event | Fed schedule (manual) |
| 47 | mag7_earnings_dates.csv | event | Earnings calendar |
| 48 | triple_witching.csv | event | 3rd Friday quarterly |
| 50 | fed_minutes_nlp.csv | event | LLM scoring of FOMC minutes |
| 51 | news_riskoff_composite.csv | daily | News API + LLM scoring |

## Update workflow

1. Acquire fresh data from source
2. Format as `Date,value` CSV
3. Save to `data/signals/manual/<filename>.csv`
4. Optionally run `ldr.get(signal_id, force=True)` to refresh parquet cache
