"""Tests for src/signals/overlay.py (Phase C C1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import numpy as np
import pandas as pd

from signals.overlay import apply_overlay


def test_apply_overlay_basic():
    idx = pd.date_range('2024-01-01', periods=5, freq='B')
    base = pd.Series([1.0] * 5, index=idx)
    sig = pd.Series([0, 1, 2, 3, 1], index=idx)
    mapping = {0: 0.0, 1: 0.5, 2: 0.8, 3: 1.0}
    out = apply_overlay(base, sig, mapping)
    assert list(out.values) == [0.0, 0.5, 0.8, 1.0, 0.5]


def test_apply_overlay_nan_signal_uses_default():
    idx = pd.date_range('2024-01-01', periods=3, freq='B')
    base = pd.Series([2.0, 2.0, 2.0], index=idx)
    sig = pd.Series([1, np.nan, 1], index=idx)
    mapping = {1: 0.5}
    out = apply_overlay(base, sig, mapping, default=1.0)
    # base*0.5, base*1.0 (default), base*0.5
    assert list(out.values) == [1.0, 2.0, 1.0]


def test_apply_overlay_unknown_signal_uses_default():
    idx = pd.date_range('2024-01-01', periods=3, freq='B')
    base = pd.Series([1.0] * 3, index=idx)
    sig = pd.Series([99, 1, 99], index=idx)
    mapping = {1: 0.5}
    out = apply_overlay(base, sig, mapping, default=1.0)
    assert list(out.values) == [1.0, 0.5, 1.0]


def test_apply_overlay_preserves_base_when_default_one():
    """If default=1.0 and all signal entries are NaN, output == base."""
    idx = pd.date_range('2024-01-01', periods=4, freq='B')
    base = pd.Series([0.5, 1.0, 1.5, 2.0], index=idx)
    sig = pd.Series([np.nan] * 4, index=idx)
    out = apply_overlay(base, sig, {0: 0.0, 1: 0.5}, default=1.0)
    assert list(out.values) == list(base.values)
