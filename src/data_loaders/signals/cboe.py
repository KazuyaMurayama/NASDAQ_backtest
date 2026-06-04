"""CBOE loader for PutCall ratio (#12) and VIX term structure (#8).

Signal mappings:
  8  → VIX term structure (^VIX vs ^VIX3M vs ^VIX6M) → binary 1 if contango (VIX < VIX3M < VIX6M), else 0
  12 → CBOE Equity PutCall Ratio (daily CSV)

For test scope, both signals are mocked. Production:
  PutCall URL: https://cdn.cboe.com/api/global/us_indices/daily_prices/EQUITY_PC.csv
  VIX term:    yfinance ^VIX, ^VIX3M, ^VIX6M
"""
from __future__ import annotations
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from ._base import SignalLoader


_PUTCALL_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/EQUITY_PC.csv"


def _fetch_putcall_csv() -> pd.Series:
    r = requests.get(_PUTCALL_URL, timeout=30)
    r.raise_for_status()
    return _parse_putcall_csv(r.text)


def _parse_putcall_csv(text: str) -> pd.Series:
    """Parse CBOE equity PutCall CSV. Header layout includes a 'Date' column and 'P/C Ratio' (or similar)."""
    df = pd.read_csv(StringIO(text), parse_dates=['Date'])
    df = df.set_index('Date').sort_index()
    # CBOE column naming varies; pick the P/C ratio column
    pc_col = next((c for c in df.columns if 'P/C' in c or 'PC' in c.replace(' ', '').upper()), None)
    if pc_col is None:
        raise ValueError(f"PutCall column not found in CBOE CSV. Columns: {list(df.columns)}")
    s = pd.to_numeric(df[pc_col], errors='coerce').dropna()
    s.name = 'cboe_putcall_equity'
    return s


def _fetch_vix_term_structure() -> pd.Series:
    """Fetch ^VIX, ^VIX3M, ^VIX6M; return binary contango series (1 if VIX < VIX3M < VIX6M)."""
    tickers = ['^VIX', '^VIX3M', '^VIX6M']
    closes = []
    for t in tickers:
        hist = yf.Ticker(t).history(period='max', auto_adjust=False)
        if hist.empty:
            raise ValueError(f"yfinance returned empty for {t}")
        col = 'Adj Close' if 'Adj Close' in hist.columns else 'Close'
        s = hist[col].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = t
        closes.append(s)
    df = pd.concat(closes, axis=1, join='inner').dropna()
    contango = ((df['^VIX'] < df['^VIX3M']) & (df['^VIX3M'] < df['^VIX6M'])).astype('int8')
    contango.name = 'cboe_vix_term_contango'
    return contango


class CboeLoader(SignalLoader):
    source_name = 'cboe'

    def _fetch(self, signal_id: int) -> pd.Series:
        if signal_id == 12:
            return _fetch_putcall_csv()
        if signal_id == 8:
            return _fetch_vix_term_structure()
        raise NotImplementedError(f"CBOE mapping missing for signal_id={signal_id}")
