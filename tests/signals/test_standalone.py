"""Tests for src/signals/standalone.py (Phase C C2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import numpy as np
import pandas as pd
import pytest

from signals.standalone import signal_driven_allocation


def test_signal_driven_allocation_basic():
    idx = pd.date_range('2024-01-01', periods=4, freq='B')
    sig = pd.Series([0, 1, 2, 3], index=idx)
    mapping = {
        0: [0.0, 0.0, 0.0],
        1: [0.3, 0.4, 0.3],
        2: [0.5, 0.3, 0.2],
        3: [0.7, 0.2, 0.1],
    }
    out = signal_driven_allocation(sig, mapping)
    assert list(out.columns) == ['TQQQ', 'TMF', 'GLD']
    assert out.iloc[0].tolist() == [0.0, 0.0, 0.0]
    assert out.iloc[3].tolist() == [0.7, 0.2, 0.1]


def test_nan_signal_returns_zero_weights():
    idx = pd.date_range('2024-01-01', periods=2, freq='B')
    sig = pd.Series([1, np.nan], index=idx)
    mapping = {1: [0.5, 0.3, 0.2]}
    out = signal_driven_allocation(sig, mapping)
    assert out.iloc[1].tolist() == [0.0, 0.0, 0.0]


def test_invalid_allocation_size_raises():
    idx = pd.date_range('2024-01-01', periods=1, freq='B')
    sig = pd.Series([1], index=idx)
    mapping = {1: [0.5, 0.5]}  # only 2 weights, default expects 3
    with pytest.raises(ValueError, match="weights"):
        signal_driven_allocation(sig, mapping)


def test_custom_asset_universe():
    idx = pd.date_range('2024-01-01', periods=2, freq='B')
    sig = pd.Series([1, 1], index=idx)
    mapping = {1: [0.6, 0.4]}
    out = signal_driven_allocation(sig, mapping, asset_universe=['SPY', 'AGG'])
    assert list(out.columns) == ['SPY', 'AGG']
    assert out.iloc[0].tolist() == [0.6, 0.4]
