"""Signal-driven standalone strategy: signal directly determines allocation.

Phase C (C2): Standalone adapter.

Converts a quantized signal into a direct portfolio allocation across
TQQQ/TMF/GLD (or any 3-asset universe).

API:
    signal_driven_allocation(signal, allocation_map, asset_universe=None) -> pd.DataFrame
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd


def signal_driven_allocation(
    signal: pd.Series,
    allocation_map: dict[int, List[float]],
    asset_universe: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert signal levels to portfolio weights.

    Args:
        signal: quantized signal series (e.g., {0, 1, 2, 3}).
        allocation_map: signal_value → [w_TQQQ, w_TMF, w_GLD] (weights should sum to 1.0,
            but caller may choose cash buffers; not enforced here).
        asset_universe: column names. Default ['TQQQ', 'TMF', 'GLD'].

    Returns:
        DataFrame with one column per asset, weights per row given by mapping.
        NaN or unmapped signal → all-zero (cash).

    Raises:
        ValueError: if any allocation row length does not match `asset_universe`.
    """
    if asset_universe is None:
        asset_universe = ['TQQQ', 'TMF', 'GLD']

    n_assets = len(asset_universe)
    for v, w in allocation_map.items():
        if len(w) != n_assets:
            raise ValueError(
                f"allocation for value={v} has {len(w)} weights, expected {n_assets}"
            )

    rows = []
    for s in signal:
        if pd.isna(s):
            rows.append([0.0] * n_assets)
            continue
        try:
            rows.append(allocation_map.get(int(s), [0.0] * n_assets))
        except (ValueError, TypeError):
            rows.append([0.0] * n_assets)

    return pd.DataFrame(rows, index=signal.index, columns=asset_universe)
