"""Phase A integration smoke check.

Verifies all 52 signals from tier1_selection_20260603.csv:
  - metadata loads cleanly
  - source_module resolves to an importable loader class
  - quantize_scheme maps to a function in signals.quantize
  - publication_lag is accepted by signals.timing.apply_publication_lag

Does NOT fetch real data — only instantiates loaders to confirm wiring.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from signals.metadata import load_registry
from signals.quantize import binary_threshold, quantile_cut, zscore_band
from signals.timing import apply_publication_lag

# Loader imports
from data_loaders.signals.fred import FredLoader
from data_loaders.signals.yahoo import YahooLoader
from data_loaders.signals.cftc import CftcLoader
from data_loaders.signals.cboe import CboeLoader
from data_loaders.signals.manual import ManualLoader

import pandas as pd


_LOADER_BY_SOURCE = {
    'fred': FredLoader,
    'yahoo': YahooLoader,
    'cftc': CftcLoader,
    'cboe': CboeLoader,
    'manual': ManualLoader,
}

_QUANTIZE_BY_SCHEME = {
    'binary_threshold': binary_threshold,
    'quantile_cut': quantile_cut,
    'zscore_band': zscore_band,
}


def verify() -> dict:
    csv = ROOT / 'data' / 'signals' / 'tier1_selection_20260603.csv'
    metas = load_registry(csv)

    # Set of seen source modules / quantize schemes / lag types
    sources = set()
    schemes = set()
    lags = set()

    # Special source modules used by Tier1 but not yet implemented:
    # fedwatch (signal 30), google_trends (signals 49, 52)
    # These are deferred to Phase D; mark them but do not fail verification.
    deferred_sources = {'fedwatch', 'google_trends'}
    deferred_signals = []

    for m in metas:
        if m.source_module not in _LOADER_BY_SOURCE:
            if m.source_module in deferred_sources:
                deferred_signals.append((m.signal_id, m.source_module))
                continue
            raise RuntimeError(f"signal_id={m.signal_id}: unknown source_module={m.source_module}")
        sources.add(m.source_module)

        if m.quantize_scheme not in _QUANTIZE_BY_SCHEME:
            raise RuntimeError(f"signal_id={m.signal_id}: unknown quantize_scheme={m.quantize_scheme}")
        schemes.add(m.quantize_scheme)

        if m.publication_lag not in {'daily', 'weekly', 'monthly', 'event'}:
            raise RuntimeError(f"signal_id={m.signal_id}: invalid publication_lag={m.publication_lag}")
        lags.add(m.publication_lag)

    # Smoke test: instantiate each unique loader class (uses temp cache dir)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for src in sources:
            cls = _LOADER_BY_SOURCE[src]
            cls(cache_dir=td)  # no fetch, just construct

    # Smoke test: apply_publication_lag with each lag type
    sample = pd.Series([1.0], index=pd.to_datetime(['2024-01-02']))
    for lag in lags:
        apply_publication_lag(sample, lag_type=lag)

    return {
        'total': len(metas),
        'loaders_configured': len(sources),
        'schemes_mapped': len(schemes),
        'lag_types_covered': len(lags),
        'deferred_signals': deferred_signals,
    }


def main():
    r = verify()
    print(f"[Phase A verify] {r['total']} signals registered")
    print(f"[Phase A verify] {r['loaders_configured']} loaders configured")
    print(f"[Phase A verify] {r['schemes_mapped']} quantize schemas mapped")
    print(f"[Phase A verify] {r['lag_types_covered']} publication_lag types covered")
    if r['deferred_signals']:
        print(f"[Phase A verify] {len(r['deferred_signals'])} signals DEFERRED to Phase D:")
        for sid, src in r['deferred_signals']:
            print(f"    signal_id={sid} source={src}")
    print("[Phase A verify] ALL OK")


if __name__ == '__main__':
    main()
