"""Fetch GBP price series for multi-asset signal research.

GBP/JPY is the primary pair (JPY-resident carry relevance); GBP/USD is
secondary. Mirrors the house convention of src/fetch_fred_data.py.

See MULTIASSET_GOLD_GBP_SIGNAL_PLAN_20260607.md, Phase 0 Task 0.3.

CLI:
    python -m fetch_gbp_data            # writes data/gbpjpy_daily.csv, data/gbpusd_daily.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from data_loaders.signals.yahoo import YahooLoader, _SIGNAL_TO_TICKER

# Signal IDs for the GBP price series (see yahoo.py mapping).
SIGNAL_GBPJPY = 60  # primary
SIGNAL_GBPUSD = 61  # secondary

# Ordered: primary first.
_PAIRS = (
    ('gbpjpy', SIGNAL_GBPJPY),
    ('gbpusd', SIGNAL_GBPUSD),
)


def fetch_gbp(loader: Optional[object] = None,
              out_dir: str | Path = 'data',
              write: bool = True) -> dict[str, pd.Series]:
    """Fetch GBP/JPY and GBP/USD daily close series.

    Args:
        loader: object with .get(signal_id, force) -> pd.Series. Defaults to
            a real YahooLoader; tests inject a fake.
        out_dir: directory for the CSV outputs.
        write: when True, write <name>_daily.csv per pair.

    Returns:
        {'gbpjpy': Series, 'gbpusd': Series}
    """
    loader = loader or YahooLoader()
    out_path = Path(out_dir)
    if write:
        out_path.mkdir(parents=True, exist_ok=True)

    result: dict[str, pd.Series] = {}
    for name, signal_id in _PAIRS:
        series = loader.get(signal_id, force=True)
        result[name] = series
        if write:
            series.to_frame(name=series.name or name).to_csv(
                out_path / f'{name}_daily.csv')
    return result


if __name__ == '__main__':  # pragma: no cover
    out = fetch_gbp()
    for nm, s in out.items():
        print(f'{nm}: {len(s)} rows, {s.index.min().date()} .. {s.index.max().date()}')
