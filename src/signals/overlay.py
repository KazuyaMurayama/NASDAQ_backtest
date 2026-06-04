"""Signal-driven overlay: modify base leverage by quantized signal level.

Phase C (C1): Overlay adapter.

Applies a quantized signal as a `lev_mod` multiplier on top of an existing
strategy's leverage series. The base leverage represents the host strategy
(e.g., vz=0.65 + l7 + F10ε-AH); the signal mapping rescales it by regime
or composite signal level.

API:
    apply_overlay(base_lev, signal, mapping, default=1.0) -> pd.Series
"""
from __future__ import annotations

import pandas as pd


def apply_overlay(
    base_lev: pd.Series,
    signal: pd.Series,
    mapping: dict[int, float],
    default: float = 1.0,
) -> pd.Series:
    """Element-wise multiply base leverage by signal-mapped multiplier.

    Args:
        base_lev: existing strategy's leverage per date (e.g., NEW CANDIDATE
            lev_mod_065 output).
        signal: quantized signal series (0/1 or 0/1/2/3).
        mapping: signal_value → multiplier (e.g., {0: 0.0, 1: 0.5, 2: 0.8, 3: 1.0}).
        default: multiplier for dates where signal is NaN or unmapped
            (inherits base unchanged when default=1.0).

    Returns:
        adjusted_lev = base_lev * signal_mapped_multiplier (pd.Series).
    """
    df = pd.concat([base_lev.rename('b'), signal.rename('s')], axis=1)

    def _mult(s_val):
        if pd.isna(s_val):
            return default
        try:
            return mapping.get(int(s_val), default)
        except (ValueError, TypeError):
            return default

    mult = df['s'].apply(_mult)
    return (df['b'] * mult).rename('adjusted_lev')
