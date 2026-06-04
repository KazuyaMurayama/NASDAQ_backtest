"""CFTC Commitments of Traders (CoT) loader.

Signal ID mapping (futures-only legacy COT, weekly):
  15 → NASDAQ-100 e-mini (209742)
  16 → Gold (088691)
  17 → 30Y Bond + 10Y Note combined (020601 + 043602)

Output: rolling 52-week z-score of (NonComm Long - NonComm Short).

For A8 test scope, real HTTP fetching is stubbed via the SignalLoader.get cache path;
tests mock requests.get with the CSV fixture.

Phase D will add: live URL fetch, fixed-width fallback if CSV unavailable,
historical yearly archive concatenation.
"""
from __future__ import annotations
import requests
import pandas as pd
from io import StringIO
from typing import Union, List, Tuple
from ._base import SignalLoader


_COT_BASE = "https://www.cftc.gov/dea/newcot/deafut.txt"  # placeholder URL

_SIGNAL_TO_CONTRACT: dict[int, Union[str, Tuple[str, ...]]] = {
    15: '209742',                  # NQ
    16: '088691',                  # GC
    17: ('020601', '043602'),      # ZB + ZN
}


def _zscore_52w(net: pd.Series) -> pd.Series:
    """52-week rolling z-score; first 51 obs NaN."""
    mu = net.rolling(window=52, min_periods=52).mean()
    sd = net.rolling(window=52, min_periods=52).std(ddof=0)
    return (net - mu) / sd


def _parse_cot_csv(csv_text: str, contract_code: str) -> pd.Series:
    """Parse a COT-format CSV, filter to contract_code, return Net (long-short) by date."""
    df = pd.read_csv(StringIO(csv_text), parse_dates=['Report_Date_as_YYYY-MM-DD'])
    # CFTC publishes contract codes as int (no leading zero). Strip+pad here.
    df['CFTC_Contract_Market_Code'] = df['CFTC_Contract_Market_Code'].astype(str).str.zfill(6)
    sub = df[df['CFTC_Contract_Market_Code'] == contract_code].copy()
    if sub.empty:
        return pd.Series(dtype='float64')
    sub = sub.sort_values('Report_Date_as_YYYY-MM-DD')
    sub.set_index('Report_Date_as_YYYY-MM-DD', inplace=True)
    return (sub['NonComm_Positions_Long_All'] - sub['NonComm_Positions_Short_All']).astype('float64')


class CftcLoader(SignalLoader):
    source_name = 'cftc'

    def _fetch_csv(self) -> str:
        r = requests.get(_COT_BASE, timeout=60)
        r.raise_for_status()
        return r.text

    def _fetch(self, signal_id: int) -> pd.Series:
        if signal_id not in _SIGNAL_TO_CONTRACT:
            raise NotImplementedError(f"CFTC mapping missing for signal_id={signal_id}")
        csv_text = self._fetch_csv()
        spec = _SIGNAL_TO_CONTRACT[signal_id]
        if isinstance(spec, tuple):
            nets: List[pd.Series] = [_parse_cot_csv(csv_text, c) for c in spec]
            net = sum(nets) if nets else pd.Series(dtype='float64')
        else:
            net = _parse_cot_csv(csv_text, spec)
        if net.empty:
            raise ValueError(f"No CoT data found for signal_id={signal_id}")
        s = _zscore_52w(net)
        s.name = f"cftc_{signal_id}"
        return s
