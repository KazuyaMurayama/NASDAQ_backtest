"""Manual update CSV loader.

Reads pre-curated CSV files from data/signals/manual/.
Schema: each CSV has 'Date' (YYYY-MM-DD) + 'value' columns.

Signals listed below require manual update (no automated source available
or restricted access). Phase D may add automation for a subset.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ._base import SignalLoader


class ManualLoader(SignalLoader):
    source_name = 'manual'

    _SIGNAL_TO_FILE: dict[int, str] = {
        13: 'aaii_weekly.csv',
        14: 'naaim_weekly.csv',
        20: 'finra_margin_debt_monthly.csv',
        32: 'gdpnow_atlanta.csv',
        33: 'nyfed_nowcast.csv',
        34: 'citi_surprise_usmi.csv',
        35: 'cleveland_inflation_nowcast.csv',
        37: 'ndx_eps_revision_4wk.csv',
        38: 'equity_risk_premium.csv',
        39: 'ndx_forward_pe_zscore.csv',
        40: 'mag7_eps_revision.csv',
        46: 'fomc_blackout.csv',
        47: 'mag7_earnings_dates.csv',
        48: 'triple_witching.csv',
        50: 'fed_minutes_nlp.csv',
        51: 'news_riskoff_composite.csv',
    }

    # Resolve manual dir from repo root (matches existing src/build_base_dataset.py pattern)
    _BASE_DIR = Path(__file__).resolve().parents[3]
    _MANUAL_DIR = _BASE_DIR / 'data' / 'signals' / 'manual'

    def _fetch(self, signal_id: int) -> pd.Series:
        if signal_id not in self._SIGNAL_TO_FILE:
            raise NotImplementedError(f"Manual mapping missing for signal_id={signal_id}")
        path = self._MANUAL_DIR / self._SIGNAL_TO_FILE[signal_id]
        if not path.exists():
            raise FileNotFoundError(
                f"Manual CSV not found: {path}. "
                f"See data/signals/manual/README.md for format spec."
            )
        df = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
        if 'value' not in df.columns:
            raise ValueError(f"Manual CSV missing 'value' column: {path}")
        s = pd.to_numeric(df['value'], errors='coerce').dropna()
        s.name = f"manual_{signal_id}"
        return s
