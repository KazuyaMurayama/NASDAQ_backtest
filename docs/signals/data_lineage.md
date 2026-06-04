# Signal Data Lineage

Generated from `data/signals/tier1_selection_20260603.csv`. Single source of truth for signal metadata.

| ID | Name | Category | Asset | Source | Lag | Earliest | Cost | Priority |
|---|---|---|---|---|---|---|---|---|
| 1 | NDX 200DMA breadth | A_Breadth | `N` | yahoo | daily | 2003-01-01 | free | A |
| 2 | McClellan Oscillator NDX | A_Breadth | `N` | yahoo | daily | 2003-01-01 | free | A |
| 3 | NDX New Highs minus Lows 52W | A_Breadth | `N` | yahoo | daily | 2003-01-01 | free | A |
| 4 | AD Line price divergence | A_Breadth | `N` | yahoo | daily | 2003-01-01 | free | B |
| 5 | NYSE TICK terminal | A_Breadth | `N` | yahoo | daily | 2000-01-01 | free | C |
| 6 | VIX level | B_Vol | `N` | yahoo | daily | 1990-01-01 | free | A |
| 7 | VIX9D over VIX ratio | B_Vol | `N` | yahoo | daily | 2011-01-01 | free | A |
| 8 | VIX term structure VIX1 VIX2 VIX3 | B_Vol | `N` | cboe | daily | 2008-01-01 | free | A |
| 9 | VVIX | B_Vol | `N` | yahoo | daily | 2007-03-01 | free | A |
| 10 | MOVE index bond vol | B_Vol | B/N | yahoo | daily | 2002-04-01 | free | A |
| 11 | GVZ gold vol | B_Vol | `G` | yahoo | daily | 2008-06-01 | free | B |
| 12 | CBOE PutCall equity | C_Sentiment | `N` | cboe | daily | 2003-10-01 | free | A |
| 13 | AAII Bull Bear spread | C_Sentiment | `N` | manual | weekly | 1987-07-31 | free | A |
| 14 | NAAIM Exposure Index | C_Sentiment | `N` | manual | weekly | 2006-07-19 | free | B |
| 15 | CFTC CoT NQ noncommercial net | C_Sentiment | `N` | cftc | weekly | 2010-01-01 | free | A |
| 16 | CFTC CoT GC net gold | C_Sentiment | `G` | cftc | weekly | 1995-01-01 | free | A |
| 17 | CFTC CoT ZB ZN net bond | C_Sentiment | `B` | cftc | weekly | 2000-01-01 | free | A |
| 18 | QQQ daily net creation redemption | C_Sentiment | `N` | yahoo | daily | 2005-01-01 | low_paid | B |
| 19 | GLD TLT net flows | C_Sentiment | G/B | yahoo | daily | 2005-01-01 | low_paid | B |
| 20 | Margin Debt YoY FINRA monthly | C_Sentiment | `N` | manual | monthly | 1997-01-01 | free | C |
| 21 | ICE BofA HY OAS | D_Credit | `N` | fred | daily | 1996-12-31 | free | A |
| 22 | ICE BofA IG OAS | D_Credit | `N` | fred | daily | 1996-12-31 | free | B |
| 23 | HY minus IG spread | D_Credit | `N` | fred | daily | 1996-12-31 | free | A |
| 24 | SOFR minus IORB spread | D_Credit | `B` | fred | daily | 2018-04-02 | free | B |
| 25 | 3M Treasury minus SOFR | D_Credit | B/N | fred | daily | 2018-04-02 | free | C |
| 26 | 2s10s spread | E_YieldCurve | B/N | fred | daily | 1976-06-01 | free | A |
| 27 | 3M10Y spread | E_YieldCurve | B/N | fred | daily | 1982-01-04 | free | A |
| 28 | 10Y TIPS real yield | E_YieldCurve | G/N | fred | daily | 2003-01-02 | free | A |
| 29 | 5Y5Y forward inflation | E_YieldCurve | G/B | fred | daily | 2003-01-02 | free | A |
| 30 | CME FedWatch 25bp cut prob 3M | E_YieldCurve | B/N | fedwatch | daily | 2015-01-01 | low_paid | A |
| 31 | 10Y minus 2Y real yield | E_YieldCurve | `G` | fred | daily | 2004-01-01 | free | B |
| 32 | Atlanta Fed GDPNow | F_MacroNowcast | N/B | manual | weekly | 2011-07-15 | free | A |
| 33 | NY Fed Nowcast | F_MacroNowcast | N/B | manual | weekly | 2016-04-15 | free | B |
| 34 | Citi Economic Surprise USMI | F_MacroNowcast | N/B/G | manual | weekly | 2003-01-01 | low_paid | A |
| 35 | Cleveland Fed Inflation Nowcast | F_MacroNowcast | G/B | manual | monthly | 2001-01-01 | free | B |
| 36 | Chicago Fed NFCI | F_MacroNowcast | `N` | fred | weekly | 1973-01-08 | free | A |
| 37 | NDX Forward EPS Revision 4wk | G_Earnings | `N` | manual | weekly | 2010-01-01 | low_paid | A |
| 38 | Equity Risk Premium fwd yld minus 10Y real | G_Earnings | `N` | manual | daily | 2003-01-02 | low_paid | A |
| 39 | NDX Forward PE zscore | G_Earnings | `N` | manual | weekly | 2003-01-01 | low_paid | B |
| 40 | Mag7 EPS Revision composite | G_Earnings | `N` | manual | weekly | 2014-01-01 | low_paid | A |
| 41 | DXY weekly change | H_CrossAsset | G/N | yahoo | daily | 1971-01-04 | free | A |
| 42 | Copper over Gold ratio | H_CrossAsset | N/B | yahoo | daily | 1989-04-01 | free | A |
| 43 | Silver over Gold ratio | H_CrossAsset | `G` | yahoo | daily | 1989-04-01 | free | B |
| 44 | Oil WTI 5d change | H_CrossAsset | N/B | yahoo | daily | 1986-01-02 | free | B |
| 45 | BTC over QQQ correlation | H_CrossAsset | `N` | yahoo | daily | 2014-09-17 | free | C |
| 46 | FOMC blackout window | I_Calendar | N/B | manual | event | 1980-01-01 | free | A |
| 47 | NDX earnings season Mag7 | I_Calendar | `N` | manual | event | 2014-01-01 | free | B |
| 48 | Triple Witching Friday | I_Calendar | `N` | manual | event | 1980-01-01 | free | C |
| 49 | Google Trends recession 90d Z | J_NLP | N/B/G | google_trends | daily | 2004-01-04 | free | A |
| 50 | Fed minutes hawkish dovish NLP | J_NLP | B/N | manual | event | 2010-01-01 | low_paid | A |
| 51 | Headline News risk off composite NLP | J_NLP | N/B/G | manual | daily | 2015-01-01 | low_paid | B |
| 52 | Google Trends TQQQ QQQ search | J_NLP | `N` | google_trends | daily | 2010-01-01 | free | B |
