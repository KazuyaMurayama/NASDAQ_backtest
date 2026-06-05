# Global 16-Asset USD CAGR Implementation Plan

> **For agentic workers:** Execute this plan task-by-task. The Python script is fully specified below — create it verbatim and run it. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fetch long-history daily data for 16 global assets, convert to USD, and compute common-window CAGR to find assets whose CAGR exceeds NASDAQ and Gold — for portfolio diversification screening.

**Architecture:** Single self-contained Python script. Source = yfinance only (FRED/stooq are network-blocked in this environment). Gold long-history from the repo's `data/lbma_gold_daily.csv` (USD, since 1968). Two common windows reported: Window A (all 16, USD, ~2006–2026 ~20yr) and Window B (long subset, USD, ~1996.10–2026 ~30yr).

**Tech Stack:** Python 3.13, yfinance 1.3.0, pandas 2.3.2, numpy.

---

## Confirmed design decisions

- **Currency:** USD-unified. FX direction (verified live):
  - Divide local index by FX `XXX=X` (foreign per USD): `JPY=X`(N225), `HKD=X`(HSI), `INR=X`(Sensex), `BRL=X`(Bovespa)
  - Multiply local index by FX (USD per unit): `GBPUSD=X`(FTSE), `EURUSD=X`(DAX, CAC)
  - No conversion (already USD): ^IXIC, ^GSPC, ^DJI, ^RUT, GLD, TLT, VNQ, EEM, DBC
- **ETFs (GLD/TLT/VNQ/EEM/DBC):** use max available history; they only matter in Window A (~20yr).
- **Comparison:** common window only (two tiers A & B).
- **Caveats to surface in report:** ^GDAXI (DAX) is a *total-return* index (includes dividends); ^GSPC/^IXIC/^DJI/^RUT/^FTSE/^HSI/^FCHI/^BSESN/^BVSP are *price* indices (exclude dividends) — so DAX is biased upward and the US/foreign indices are biased downward vs a true total-return basis. ETFs via `auto_adjust=True` are total-return (incl. dividends). This inconsistency is unavoidable with free data; rank with the caveat noted.

## Asset registry

| Ticker | Name | Local ccy | FX ticker | FX op |
|---|---|---|---|---|
| ^IXIC | NASDAQ Composite | USD | — | none |
| ^GSPC | S&P 500 | USD | — | none |
| ^DJI | Dow Jones | USD | — | none |
| ^RUT | Russell 2000 | USD | — | none |
| ^N225 | Nikkei 225 | JPY | JPY=X | divide |
| ^FTSE | FTSE 100 | GBP | GBPUSD=X | multiply |
| ^GDAXI | DAX (TR) | EUR | EURUSD=X | multiply |
| ^FCHI | CAC 40 | EUR | EURUSD=X | multiply |
| ^HSI | Hang Seng | HKD | HKD=X | divide |
| ^BSESN | BSE Sensex | INR | INR=X | divide |
| ^BVSP | Bovespa | BRL | BRL=X | divide |
| GLD | Gold ETF | USD | — | none |
| TLT | UST 20Y+ ETF | USD | — | none |
| VNQ | US REIT ETF | USD | — | none |
| EEM | EM ETF | USD | — | none |
| DBC | Commodity ETF | USD | — | none |
| (LBMA) | Gold spot (repo CSV) | USD | — | none (Window B benchmark) |

---

### Task 1: Create and run the analysis script

**Files:**
- Create: `analysis_global_cagr/compute_global_cagr.py`
- Output: `analysis_global_cagr/global_cagr_results.csv`, `analysis_global_cagr/GLOBAL_ASSETS_CAGR_REPORT_20260605.md`

- [ ] **Step 1: Write the script** (`analysis_global_cagr/compute_global_cagr.py`)

```python
"""Fetch 16 global assets (+LBMA gold), USD-convert, compute common-window CAGR.
Source: yfinance only (FRED/stooq blocked in this env). Gold long-history from repo LBMA CSV.
"""
import warnings, io, urllib.request
import numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")

ASSETS = [
    ("^IXIC", "NASDAQ Composite", "USD", None, None),
    ("^GSPC", "S&P 500",          "USD", None, None),
    ("^DJI",  "Dow Jones",        "USD", None, None),
    ("^RUT",  "Russell 2000",     "USD", None, None),
    ("^N225", "Nikkei 225",       "JPY", "JPY=X",    "div"),
    ("^FTSE", "FTSE 100",         "GBP", "GBPUSD=X", "mul"),
    ("^GDAXI","DAX (Total Return)","EUR","EURUSD=X", "mul"),
    ("^FCHI", "CAC 40",           "EUR", "EURUSD=X", "mul"),
    ("^HSI",  "Hang Seng",        "HKD", "HKD=X",    "div"),
    ("^BSESN","BSE Sensex",       "INR", "INR=X",    "div"),
    ("^BVSP", "Bovespa",          "BRL", "BRL=X",    "div"),
    ("GLD",   "Gold ETF (GLD)",   "USD", None, None),
    ("TLT",   "UST 20Y+ ETF (TLT)","USD",None, None),
    ("VNQ",   "US REIT ETF (VNQ)","USD", None, None),
    ("EEM",   "EM ETF (EEM)",     "USD", None, None),
    ("DBC",   "Commodity ETF (DBC)","USD",None, None),
]

def close_series(ticker, start="1970-01-01"):
    d = yf.download(ticker, start=start, interval="1d", progress=False, auto_adjust=True)
    if d is None or len(d) == 0:
        return None
    c = d["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna()

# --- fetch FX once (cache) ---
fx_cache = {}
def get_fx(fx):
    if fx not in fx_cache:
        fx_cache[fx] = close_series(fx)
    return fx_cache[fx]

def to_usd(price, fx_ticker, op):
    if fx_ticker is None:
        return price
    fx = get_fx(fx_ticker).reindex(price.index).ffill()
    out = price * fx if op == "mul" else price / fx
    return out.dropna()

# --- LBMA gold (USD) from repo for Window B 30yr benchmark ---
def lbma_gold():
    url = "https://raw.githubusercontent.com/KazuyaMurayama/NASDAQ_backtest/main/data/lbma_gold_daily.csv"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"}), timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["USD"].dropna()

# --- build USD series for all assets ---
usd = {}
meta = {}
for tk, name, ccy, fx, op in ASSETS:
    s = close_series(tk)
    if s is None:
        print(f"WARN no data {tk}"); continue
    u = to_usd(s, fx, op)
    usd[tk] = u
    meta[tk] = name
    print(f"{tk:8s} USD {u.index[0].date()} -> {u.index[-1].date()} n={len(u)}")

gold_lbma = lbma_gold()
print(f"LBMA     USD {gold_lbma.index[0].date()} -> {gold_lbma.index[-1].date()} n={len(gold_lbma)}")

def cagr(s, start, end):
    s = s[(s.index >= start) & (s.index <= end)].dropna()
    if len(s) < 2:
        return np.nan, np.nan, None, None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    c = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    return c, yrs, s.index[0].date(), s.index[-1].date()

# ---------- Window A: all 16, common window ----------
common_start_A = max(u.index[0] for u in usd.values())
common_end     = min(u.index[-1] for u in usd.values())
print(f"\nWindow A common: {common_start_A.date()} -> {common_end.date()}")

rowsA = []
for tk, name in meta.items():
    c, yrs, s0, s1 = cagr(usd[tk], common_start_A, common_end)
    rowsA.append((tk, name, round(c*100, 2), round(yrs, 1), str(s0), str(s1)))
dfA = pd.DataFrame(rowsA, columns=["ticker","name","CAGR_pct","years","start","end"]).sort_values("CAGR_pct", ascending=False)

# ---------- Window B: long subset, USD from ~1996 ----------
LONG = ["^IXIC", "^GSPC", "^DJI", "^RUT", "^N225"]
long_series = {tk: usd[tk] for tk in LONG if tk in usd}
long_series["LBMA_GOLD"] = gold_lbma
meta_long = {**{tk: meta[tk] for tk in LONG if tk in meta}, "LBMA_GOLD": "Gold spot (LBMA, USD)"}
common_start_B = max(s.index[0] for s in long_series.values())
common_end_B   = min(s.index[-1] for s in long_series.values())
print(f"Window B common: {common_start_B.date()} -> {common_end_B.date()}")

rowsB = []
for tk, name in meta_long.items():
    c, yrs, s0, s1 = cagr(long_series[tk], common_start_B, common_end_B)
    rowsB.append((tk, name, round(c*100, 2), round(yrs, 1), str(s0), str(s1)))
dfB = pd.DataFrame(rowsB, columns=["ticker","name","CAGR_pct","years","start","end"]).sort_values("CAGR_pct", ascending=False)

# ---------- save ----------
dfA.insert(0, "window", "A_all16_~20yr")
dfB.insert(0, "window", "B_long_~30yr")
out = pd.concat([dfA, dfB], ignore_index=True)
out.to_csv("analysis_global_cagr/global_cagr_results.csv", index=False)
print("\n=== WINDOW A (all 16, USD) ==="); print(dfA.to_string(index=False))
print("\n=== WINDOW B (long subset, USD) ==="); print(dfB.to_string(index=False))

nas_A = dfA.loc[dfA.ticker=="^IXIC","CAGR_pct"].iloc[0]
gld_A = dfA.loc[dfA.ticker=="GLD","CAGR_pct"].iloc[0]
print(f"\nWindow A: NASDAQ={nas_A}%  Gold(GLD)={gld_A}%")
print("Beats NASDAQ (A):", list(dfA[dfA.CAGR_pct>nas_A].ticker))
print("Beats Gold   (A):", list(dfA[dfA.CAGR_pct>gld_A].ticker))
```

- [ ] **Step 2: Run the script** (network sandbox disabled)

Run: `python analysis_global_cagr/compute_global_cagr.py`
Expected: prints per-asset USD ranges, both windows' common periods, two ranked tables, and CSV written. No exceptions.

- [ ] **Step 3: Sanity-check output**

- Each USD series start/end printed and plausible (e.g. ^N225 USD starts ~1996-10 due to JPY=X start).
- NASDAQ CAGR Window A roughly mid-teens %; Gold Window A roughly low-double-digit %.
- Window B common start ~1996-10, years ~29-30.
- Flag any asset with CAGR > NASDAQ and > Gold — these are the diversification candidates.

- [ ] **Step 4: Write the markdown report** `analysis_global_cagr/GLOBAL_ASSETS_CAGR_REPORT_20260605.md`

Include H1 + `作成日: 2026-06-05` / `最終更新日: 2026-06-05`, both ranked tables, the data-source limitation note (Yahoo-only; FRED/stooq blocked), the price-vs-total-return caveat (DAX is TR), the JPY-FX-limits-N225-to-1996 note, and a conclusion answering: *which assets beat NASDAQ and Gold, and are they viable diversifiers*.

---

## Self-Review

- Spec coverage: 16 assets ✓, 30yr where feasible (Window B) ✓, USD ✓, common window ✓, CAGR ✓, find > NASDAQ/Gold ✓.
- No placeholders: full script included ✓.
- Type consistency: `cagr()` returns 4-tuple, used consistently ✓; FX op strings "div"/"mul" match registry ✓.
