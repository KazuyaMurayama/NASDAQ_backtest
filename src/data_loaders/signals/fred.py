"""FRED data loader.

Signal ID → FRED series ID mapping (subset; spec §4.2):
  21 → BAMLH0A0HYM2 (ICE BofA HY OAS)
  22 → BAMLC0A0CM   (ICE BofA IG OAS)
  26 → T10Y2Y
  27 → T10Y3M
  28 → DFII10
  29 → T5YIFR
  36 → NFCI
"""
from __future__ import annotations
from io import StringIO
import requests
import pandas as pd
from ._base import SignalLoader


_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

_SIGNAL_TO_SERIES = {
    21: 'BAMLH0A0HYM2',
    22: 'BAMLC0A0CM',
    26: 'T10Y2Y',
    27: 'T10Y3M',
    28: 'DFII10',
    29: 'T5YIFR',
    36: 'NFCI',
}


class FredLoader(SignalLoader):
    source_name = 'fred'

    def _fetch(self, signal_id: int) -> pd.Series:
        if signal_id not in _SIGNAL_TO_SERIES:
            raise NotImplementedError(f"FRED mapping missing for signal_id={signal_id}")
        series_id = _SIGNAL_TO_SERIES[signal_id]
        url = f"{_FRED_BASE}{series_id}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), parse_dates=['observation_date'])
        df = df.set_index('observation_date').sort_index()
        s = pd.to_numeric(df[series_id], errors='coerce').dropna()
        s.name = f"fred_{series_id}"
        return s
