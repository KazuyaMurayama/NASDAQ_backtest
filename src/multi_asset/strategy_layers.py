"""Composable timing-strategy layers ported from the NASDAQ archetypes.

See docs/multiasset/NASDAQ_STRATEGY_ARCHETYPES.md. Every layer returns a
series of position multipliers (gates in {0,1} or scales in [0, max]) and is
**causal** (acts on info through t-1). Layers are combined with `compose`
(elementwise product), then optionally smoothed with `deadband`/`hysteresis`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Volatility targeting (P2_VolTarget / S2 tv / F6_vol_scale)
# --------------------------------------------------------------------------
def vol_target_scale(asset_ret: pd.Series, target_vol: float = 0.10,
                     vol_window: int = 20, max_leverage: float = 1.0) -> pd.Series:
    """Scale = clip(target_vol / realized_vol, 0, max_leverage), causal."""
    rv = asset_ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / rv).clip(lower=0.0, upper=max_leverage)
    scale[(rv.isna()) | (rv <= 0)] = np.nan
    return scale.shift(1)


# --------------------------------------------------------------------------
# Vol-regime / VZ gate (S2_VZGated)
# --------------------------------------------------------------------------
def vol_regime_gate(asset_ret: pd.Series, vol_window: int = 20,
                    z_window: int = 252, z_thresh: float = 1.0,
                    gate_min: float = 0.0) -> pd.Series:
    """Gate = 1 when realized-vol z-score <= z_thresh, else gate_min. Causal."""
    rv = asset_ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    z = (rv - rv.rolling(z_window).mean()) / rv.rolling(z_window).std()
    gate = pd.Series(np.where(z <= z_thresh, 1.0, gate_min), index=z.index)
    gate[z.isna()] = np.nan
    return gate.shift(1)


# --------------------------------------------------------------------------
# Dual-MA cross (LT7)
# --------------------------------------------------------------------------
def dual_ma_position(price: pd.Series, n_short: int, n_long: int) -> pd.Series:
    """Hold when short MA > long MA. Causal."""
    ms = price.rolling(n_short).mean()
    ml = price.rolling(n_long).mean()
    raw = (ms > ml).astype(float)
    raw[ml.isna()] = np.nan
    return raw.shift(1)


# --------------------------------------------------------------------------
# Donchian breakout (turtle: turtle_core / t1 / t2)
# --------------------------------------------------------------------------
def donchian_breakout_position(price: pd.Series, entry_n: int,
                               exit_n: int) -> pd.Series:
    """Turtle: enter (1) on a new entry_n-day high, exit (0) on a new
    exit_n-day low, hold otherwise. Channels use prior bars only (causal)."""
    hi = price.rolling(entry_n).max().shift(1)
    lo = price.rolling(exit_n).min().shift(1)
    pos = pd.Series(np.nan, index=price.index)
    state = 0.0
    for i in range(len(price)):
        h, l, p = hi.iloc[i], lo.iloc[i], price.iloc[i]
        if np.isnan(h) or np.isnan(l):
            pos.iloc[i] = np.nan
            continue
        if p > h:
            state = 1.0
        elif p < l:
            state = 0.0
        pos.iloc[i] = state
    return pos.shift(1)


# --------------------------------------------------------------------------
# Asymmetric hysteresis (AsymmHysteresis)
# --------------------------------------------------------------------------
def hysteresis(raw: pd.Series, enter: float = 0.7, exit: float = 0.3) -> pd.Series:
    """Convert a continuous signal to {0,1}: go 1 when raw>=enter, 0 when
    raw<=exit, hold otherwise. Reduces flip-flop. (raw should be pre-lagged.)"""
    out = pd.Series(0.0, index=raw.index)
    state = 0.0
    for i in range(len(raw)):
        v = raw.iloc[i]
        if not np.isnan(v):
            if v >= enter:
                state = 1.0
            elif v <= exit:
                state = 0.0
        out.iloc[i] = state
    return out


# --------------------------------------------------------------------------
# Deadband (F10 epsilon-deadband)
# --------------------------------------------------------------------------
def deadband(position: pd.Series, eps: float = 0.1) -> pd.Series:
    """Only update the held position when |new - current| > eps."""
    out = pd.Series(0.0, index=position.index)
    current = 0.0
    for i in range(len(position)):
        v = position.iloc[i]
        if np.isnan(v):
            out.iloc[i] = current
            continue
        if abs(v - current) > eps:
            current = v
        out.iloc[i] = current
    return out


# --------------------------------------------------------------------------
# Compose layers (multiplicative)
# --------------------------------------------------------------------------
def compose(*layers: pd.Series) -> pd.Series:
    """Elementwise product of position/gate layers on the first layer's index.
    Missing values are treated as 0 (cash)."""
    base = layers[0]
    out = base.reindex(base.index).fillna(0.0).astype(float).copy()
    for layer in layers[1:]:
        out = out * layer.reindex(base.index).fillna(0.0)
    return out


def ensemble_vote(positions: list, threshold: float = 0.5) -> pd.Series:
    """Majority/average vote across position series (E1_ensemble). Returns the
    mean position; callers may threshold it for a binary gate."""
    idx = positions[0].index
    stacked = pd.concat([p.reindex(idx).fillna(0.0) for p in positions], axis=1)
    return stacked.mean(axis=1)
