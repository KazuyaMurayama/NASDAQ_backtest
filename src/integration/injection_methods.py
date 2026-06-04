"""Injection methods M1-M5 for signal × baseline strategy integration.

All methods take:
  base_nav: pd.Series (baseline strategy NAV)
  signal:   pd.Series (quantized 0-3, sparse/aligned by date)

And return:
  candidate_nav: pd.Series (signal-augmented NAV, starts at base_nav.iloc[0])

Implementation approach: convert base_nav to daily returns, multiply by
injection-derived lev_mod, recompute candidate NAV. This preserves the
baseline's gross daily return structure while applying the signal overlay.

NOTE: This is the simplified return-level overlay used for Tier 1 screening.
For full asset-level rotation (M3 proper), use Session S2's underlying-asset
NAV reconstruction (deferred).
"""
from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Return / NAV helpers
# ------------------------------------------------------------------

def _nav_to_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().fillna(0.0)


def _returns_to_nav(r: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + r).cumprod()


def _align_signal(signal: pd.Series, index: pd.Index) -> pd.Series:
    """Reindex signal to base index, forward-fill, fillna with 0."""
    s = signal.reindex(index).ffill().fillna(0)
    return s


def _map_signal(signal: pd.Series, mapping: dict, default: float = 1.0) -> pd.Series:
    """Apply integer-key mapping to a signal series."""
    return signal.apply(
        lambda v: mapping.get(int(v), default) if pd.notna(v) else default
    )


# ------------------------------------------------------------------
# M1: Binary leverage mask
# ------------------------------------------------------------------

def m1_lev_mask(
    base_nav: pd.Series,
    signal: pd.Series,
    mask_threshold: int = 2,
    direction: Literal['defensive', 'procyclical'] = 'defensive',
) -> pd.Series:
    """M1: Binary gate on leverage.

    defensive   : signal >= mask_threshold → cash (lev_mod = 0)
    procyclical : signal >= mask_threshold → full (lev_mod = 1), else half
    """
    start = float(base_nav.dropna().iloc[0])
    r = _nav_to_returns(base_nav)
    sig = _align_signal(signal, r.index)
    binary = (sig >= mask_threshold).astype(float)
    if direction == 'defensive':
        lev_mod = 1.0 - binary
    else:  # procyclical
        lev_mod = 0.5 + 0.5 * binary
    return _returns_to_nav(r * lev_mod, start=start)


# ------------------------------------------------------------------
# M2: Continuous leverage tilt
# ------------------------------------------------------------------

def m2_lev_tilt(
    base_nav: pd.Series,
    signal: pd.Series,
    direction: Literal['defensive', 'procyclical'] = 'defensive',
) -> pd.Series:
    """M2: Continuous leverage multiplier from 4-level signal."""
    if direction == 'defensive':
        mult_map = {0: 1.2, 1: 1.0, 2: 0.7, 3: 0.3}
    else:
        mult_map = {0: 0.7, 1: 0.9, 2: 1.1, 3: 1.3}
    start = float(base_nav.dropna().iloc[0])
    r = _nav_to_returns(base_nav)
    sig = _align_signal(signal, r.index)
    mult = _map_signal(sig, mult_map, default=1.0)
    return _returns_to_nav(r * mult, start=start)


# ------------------------------------------------------------------
# M3: Asset tilt (full version requires per-asset NAVs)
# ------------------------------------------------------------------

def m3_asset_tilt(
    base_nav: pd.Series,
    signal: pd.Series,
    direction: Literal['risk_off', 'reverse'] = 'risk_off',
) -> pd.Series:
    """M3: Asset rotation — STUB.

    Full implementation requires per-asset NAV reconstruction (TQQQ/Gold/Bond)
    via build_dh_nav_with_cost from src/g18_daily_trade_cost_wfa.py with
    rotated wn/wg/wb arrays. That belongs to Session S2 follow-up.

    For Tier 1 screening, this returns a return-level approximation using a
    distinct multiplier map from M2.
    """
    if direction == 'risk_off':
        mult_map = {0: 1.1, 1: 0.9, 2: 0.7, 3: 0.5}
    else:
        mult_map = {0: 0.5, 1: 0.7, 2: 0.9, 3: 1.1}
    start = float(base_nav.dropna().iloc[0])
    r = _nav_to_returns(base_nav)
    sig = _align_signal(signal, r.index)
    mult = _map_signal(sig, mult_map, default=1.0)
    return _returns_to_nav(r * mult, start=start)


# ------------------------------------------------------------------
# M4: Vol-target modifier
# ------------------------------------------------------------------

def m4_vol_target_mod(
    base_nav: pd.Series,
    signal: pd.Series,
    direction: Literal['vol_adj'] = 'vol_adj',
) -> pd.Series:
    """M4: Vol target — scale leverage by (target_vol / realized_vol) × signal_mult."""
    start = float(base_nav.dropna().iloc[0])
    r = _nav_to_returns(base_nav)
    realized_vol = r.rolling(60, min_periods=20).std()
    sig = _align_signal(signal, r.index)
    vol_mult_map = {0: 1.5, 1: 1.0, 2: 0.7, 3: 0.5}
    sig_mult = _map_signal(sig, vol_mult_map, default=1.0)
    target_vol = realized_vol.median()
    lev = (
        (target_vol / realized_vol.replace(0, np.nan))
        .clip(0.3, 1.5)
        .fillna(1.0)
        * sig_mult
    )
    return _returns_to_nav(r * lev, start=start)


# ------------------------------------------------------------------
# M5: Hard entry/exit filter
# ------------------------------------------------------------------

def m5_entry_exit_filter(
    base_nav: pd.Series,
    signal: pd.Series,
    direction: Literal['stop_only', 'filter_entry'] = 'stop_only',
) -> pd.Series:
    """M5: Hard gate.

    stop_only    : signal == 3 → cash (in_position = 0); else in
    filter_entry : signal >= 2 → in; else cash
    """
    start = float(base_nav.dropna().iloc[0])
    r = _nav_to_returns(base_nav)
    sig = _align_signal(signal, r.index)
    if direction == 'stop_only':
        in_position = (sig < 3).astype(float)
    else:
        in_position = (sig >= 2).astype(float)
    return _returns_to_nav(r * in_position, start=start)


# ------------------------------------------------------------------
# Registry (used by pattern enumerator → executor)
# ------------------------------------------------------------------

METHOD_REGISTRY = {
    ('M1', 'defensive'):   lambda b, s: m1_lev_mask(b, s, mask_threshold=2, direction='defensive'),
    ('M1', 'procyclical'): lambda b, s: m1_lev_mask(b, s, mask_threshold=2, direction='procyclical'),
    ('M2', 'defensive'):   lambda b, s: m2_lev_tilt(b, s, direction='defensive'),
    ('M2', 'procyclical'): lambda b, s: m2_lev_tilt(b, s, direction='procyclical'),
    ('M3', 'risk_off'):    lambda b, s: m3_asset_tilt(b, s, direction='risk_off'),
    ('M3', 'reverse'):     lambda b, s: m3_asset_tilt(b, s, direction='reverse'),
    ('M4', 'vol_adj'):     lambda b, s: m4_vol_target_mod(b, s, direction='vol_adj'),
    ('M5', 'stop_only'):   lambda b, s: m5_entry_exit_filter(b, s, direction='stop_only'),
    ('M5', 'filter_entry'): lambda b, s: m5_entry_exit_filter(b, s, direction='filter_entry'),
}
