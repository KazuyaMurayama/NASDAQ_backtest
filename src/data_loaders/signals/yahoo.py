"""Yahoo Finance loader.

Single-ticker signals: returns Adj Close series.
Ratio signals (#42 Cu/Au, #43 Ag/Au): returns numerator_close / denominator_close.

Signal ID mappings (subset; spec §4.2):
  6  → ^VIX
  7  → ^VIX9D
  9  → ^VVIX
  10 → ^MOVE
  11 → ^GVZ
  41 → DX-Y.NYB (DXY)
  42 → HG=F / GC=F (Copper/Gold)
  43 → SI=F / GC=F (Silver/Gold)
  44 → CL=F (Oil WTI)

Not yet supported (require special handling):
  #1,2,3   NDX breadth (needs constituent data, not in yfinance)
  #18,19   ETF flows (requires creation/redemption, not in yfinance)

Uses yfinance under the hood. Tests mock yfinance.Ticker.history.
"""
from __future__ import annotations
from typing import Union, Tuple
import pandas as pd
import yfinance as yf
from ._base import SignalLoader


_SIGNAL_TO_TICKER: dict[int, Union[str, Tuple[str, str]]] = {
    6: '^VIX',
    7: '^VIX9D',
    9: '^VVIX',
    10: '^MOVE',
    11: '^GVZ',
    41: 'DX-Y.NYB',
    42: ('HG=F', 'GC=F'),
    43: ('SI=F', 'GC=F'),
    44: 'CL=F',
}


def _close_series(ticker: str) -> pd.Series:
    """Fetch Adj Close (or Close fallback) as a Series indexed by date."""
    t = yf.Ticker(ticker)
    df = t.history(period='max', auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    s = df[col].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = f"yahoo_{ticker}"
    return s


class YahooLoader(SignalLoader):
    source_name = 'yahoo'

    def _fetch(self, signal_id: int) -> pd.Series:
        if signal_id not in _SIGNAL_TO_TICKER:
            raise NotImplementedError(f"Yahoo mapping missing for signal_id={signal_id}")
        spec = _SIGNAL_TO_TICKER[signal_id]
        if isinstance(spec, tuple):
            num_ticker, den_ticker = spec
            num = _close_series(num_ticker)
            den = _close_series(den_ticker)
            ratio = (num / den).dropna()
            ratio.name = f"yahoo_{num_ticker}_over_{den_ticker}"
            return ratio
        return _close_series(spec)
