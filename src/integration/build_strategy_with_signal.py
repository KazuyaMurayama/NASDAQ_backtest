"""Native signal injection for S1 (F10), S2 (vz065lmax5), S3 (DH-W1).

Session 3 (G3, 2026-06-05): Generic native injector for Top-5 macro signals
× 3 strategies × 5 methods × 2 directions = 150 patterns.

Native means: the signal modulates the **leverage stream BEFORE NAV is
computed** (so daily trade-cost is assessed on the modulated path), not a
post-hoc NAV-multiplication. The injection point per strategy:

  S1 (F10)         : lev_mod_e4   (cached) ×= mult → build_nav_strategy(...)
  S2 (vz065lmax5)  : lev_mod_065  (cached) ×= mult → build_nav_strategy(...)
  S3 (DH-W1)       : lev_raw      (load_shared_assets) ×= mask × mult
                     → build_dh_nav_with_cost(...)   (W1 hysteresis mask applied)

For S3 we pay a one-time ~2-5 min asset-load cost (load_shared_assets),
then reuse the cached dict for all 50 S3 candidates.
"""
from __future__ import annotations
import os
import sys
import types
import pickle
from pathlib import Path

# multitasking stub (matches g23a / build_w1_baa)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from signals.quantize import quantile_cut  # noqa: E402
from signals.timing import apply_publication_lag  # noqa: E402


CACHE_DIR = ROOT / 'audit_results' / '_cache'
CFD_SPREAD_LOW = 0.0020  # 0.20%/yr — matches g14 constant


# ------------------------------------------------------------------
# NAV re-index helper (native builders return RangeIndex-indexed Series)
# ------------------------------------------------------------------

def _reindex_to_dates(nav: pd.Series, dates: pd.Series) -> pd.Series:
    """Replace nav.index (RangeIndex) with a DatetimeIndex from `dates`."""
    date_index = pd.DatetimeIndex(pd.to_datetime(dates.values))
    return pd.Series(nav.values, index=date_index).dropna()


# ------------------------------------------------------------------
# Lazy global for S3 (DH-W1) assets — load_shared_assets is ~2-5 min
# ------------------------------------------------------------------
_DH_ASSETS_CACHE: dict | None = None


def _get_dh_assets() -> dict:
    global _DH_ASSETS_CACHE
    if _DH_ASSETS_CACHE is not None:
        return _DH_ASSETS_CACHE
    print('[S3] Loading DH shared assets (one-time, ~2-5 min)...')
    from g14_wfa_sbi_cfd import load_shared_assets
    _DH_ASSETS_CACHE = load_shared_assets()
    print('[S3] DH assets loaded.')
    return _DH_ASSETS_CACHE


# ------------------------------------------------------------------
# Method-multiplier dispatcher
# ------------------------------------------------------------------

def _get_method_multiplier(
    signal_q: pd.Series,
    method: str,
    direction: str,
    vol_series: pd.Series | None = None,
) -> pd.Series:
    """Per-date multiplier series for (method, direction).

    Signal bucket convention (quantile_cut levels=4):
      0 = lowest quartile, 1 = Q2, 2 = Q3, 3 = highest quartile

    Defensive direction: high-signal-quartile lowers exposure
                         (signal interpreted as risk-on warning).
    Procyclical: high-quartile increases exposure (signal as momentum).
    """
    if method == 'M1':
        # Binary mask via top-half threshold
        binary = (signal_q >= 2).astype(float)
        if direction == 'defensive':
            # High-signal → cash (mult=0)
            return 1.0 - binary
        else:  # procyclical
            # Low-signal → half exposure; high-signal → full
            return 0.5 + 0.5 * binary

    elif method == 'M2':
        if direction == 'defensive':
            mult_map = {0: 1.2, 1: 1.0, 2: 0.7, 3: 0.3}
        else:  # procyclical
            mult_map = {0: 0.7, 1: 0.9, 2: 1.1, 3: 1.3}
        return signal_q.map(
            lambda s: mult_map.get(int(s), 1.0) if pd.notna(s) else 1.0
        )

    elif method == 'M4':
        # Vol-target × signal modifier
        if vol_series is None:
            return pd.Series(1.0, index=signal_q.index)
        if direction == 'vol_adj':
            vol_mult_map = {0: 1.5, 1: 1.0, 2: 0.7, 3: 0.5}
        else:  # reverse
            vol_mult_map = {0: 0.5, 1: 0.7, 2: 1.0, 3: 1.5}
        sig_mult = signal_q.map(
            lambda s: vol_mult_map.get(int(s), 1.0) if pd.notna(s) else 1.0
        )
        target_vol = float(vol_series.median())
        vol_clean = vol_series.replace(0.0, np.nan)
        lev = (target_vol / vol_clean).clip(0.3, 1.5).fillna(1.0) * sig_mult
        return lev

    elif method == 'M5':
        if direction == 'stop_only':
            # Drop exposure only in top quartile
            return (signal_q < 3).astype(float)
        else:  # filter_entry
            # Allow full exposure only in top half
            return (signal_q >= 2).astype(float)

    elif method == 'M6':
        # Internal-threshold proxy (M6 has no direct cache hook; use scaled M2)
        if direction == 'defensive':
            mult_map = {0: 1.1, 1: 1.0, 2: 0.9, 3: 0.8}
        else:  # procyclical
            mult_map = {0: 0.9, 1: 1.0, 2: 1.1, 3: 1.2}
        return signal_q.map(
            lambda s: mult_map.get(int(s), 1.0) if pd.notna(s) else 1.0
        )

    raise ValueError(f"Unknown method: {method!r}")


def _build_mult_array(
    sig_q: pd.Series,
    method: str,
    direction: str,
    target_dates: pd.Series | pd.DatetimeIndex,
    ret_values: np.ndarray | None = None,
) -> np.ndarray:
    """Compute multiplier aligned to target_dates (positional ndarray, NaN→1.0).

    target_dates : Series-of-Timestamps or DatetimeIndex describing the
                   strategy's day-by-day calendar (positional length = NAV length)
    ret_values   : optional ndarray of daily returns for vol-target (M4)
    """
    if isinstance(target_dates, pd.Series):
        date_index = pd.DatetimeIndex(pd.to_datetime(target_dates.values))
    else:
        date_index = pd.DatetimeIndex(pd.to_datetime(target_dates))

    sig_aligned = sig_q.reindex(date_index).ffill()
    if method == 'M4':
        if ret_values is None:
            vol = pd.Series(np.nan, index=date_index)
        else:
            ret_s = pd.Series(np.asarray(ret_values, dtype=float), index=date_index)
            vol = ret_s.rolling(60, min_periods=20).std()
        mult = _get_method_multiplier(sig_aligned, method, direction, vol_series=vol)
    else:
        mult = _get_method_multiplier(sig_aligned, method, direction)
    arr = np.asarray(mult.fillna(1.0).values, dtype=float)
    # Defensive: clip to [0, 3] to guard against runaway multipliers
    arr = np.clip(arr, 0.0, 3.0)
    return arr


# ------------------------------------------------------------------
# S1 (F10) native build
# ------------------------------------------------------------------

def _build_s1_native(sig_q: pd.Series, method: str, direction: str) -> pd.Series:
    from g14_wfa_sbi_cfd import build_nav_strategy

    obj = pickle.load(open(CACHE_DIR / 'f10_nav_cache.pkl', 'rb'))
    dates = obj['dates']
    close = obj['close']
    ret = obj['ret']

    mult_arr = _build_mult_array(
        sig_q, method, direction,
        target_dates=dates,
        ret_values=ret.values if method == 'M4' else None,
    )
    lev_mod_base = np.asarray(obj['lev_mod_e4'], dtype=float)
    lev_mod_mod = lev_mod_base * mult_arr

    nav = build_nav_strategy(
        close, lev_mod_mod,
        obj['wn_tilted'], obj['wg_A'], obj['wb_tilted'], dates,
        obj['gold_2x'], obj['bond_3x'], obj['sofr'],
        nas_mode='CFD',
        cfd_leverage=obj['L_s2'].values if hasattr(obj['L_s2'], 'values') else obj['L_s2'],
        cfd_spread=CFD_SPREAD_LOW,
    )
    return _reindex_to_dates(nav, dates)


# ------------------------------------------------------------------
# S2 (vz065lmax5) native build
# ------------------------------------------------------------------

def _build_s2_native(sig_q: pd.Series, method: str, direction: str) -> pd.Series:
    from g14_wfa_sbi_cfd import build_nav_strategy

    obj = pickle.load(open(CACHE_DIR / 'vz065lmax5_nav_cache.pkl', 'rb'))
    dates = obj['dates']
    close = obj['close']
    ret = obj['ret']

    mult_arr = _build_mult_array(
        sig_q, method, direction,
        target_dates=dates,
        ret_values=ret.values if method == 'M4' else None,
    )
    lev_mod_base = np.asarray(obj['lev_mod_065'], dtype=float)
    lev_mod_mod = lev_mod_base * mult_arr

    nav = build_nav_strategy(
        close, lev_mod_mod,
        obj['wn_A'], obj['wg_A'], obj['wb_A'], dates,
        obj['gold_2x'], obj['bond_3x'], obj['sofr'],
        nas_mode='CFD',
        cfd_leverage=obj['L_s2_lmax5'].values if hasattr(obj['L_s2_lmax5'], 'values') else obj['L_s2_lmax5'],
        cfd_spread=CFD_SPREAD_LOW,
    )
    return _reindex_to_dates(nav, dates)


# ------------------------------------------------------------------
# S3 (DH-W1) native build
# ------------------------------------------------------------------

def _build_s3_native(sig_q: pd.Series, method: str, direction: str) -> pd.Series:
    from g23a_dh_refinement_variants import hold_mask_W1, DH_PER_UNIT
    from g18_daily_trade_cost_wfa import build_dh_nav_with_cost

    a = _get_dh_assets()
    mask = hold_mask_W1(a)
    wn = np.asarray(a['wn_A']) * mask
    wg = np.asarray(a['wg_A']) * mask
    wb = np.asarray(a['wb_A']) * mask
    lev_raw_base = np.asarray(a['lev_raw']) * mask

    # Multiplier aligned to DH calendar (a['dates'])
    ret_vals = a['ret'].values if method == 'M4' else None
    mult_arr = _build_mult_array(
        sig_q, method, direction,
        target_dates=a['dates'],
        ret_values=ret_vals,
    )
    lev_raw_mod = lev_raw_base * mult_arr

    nav, cost = build_dh_nav_with_cost(
        a['close'], lev_raw_mod, wn, wg, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )
    return _reindex_to_dates(nav, a['dates'])


# ------------------------------------------------------------------
# Public entrypoint
# ------------------------------------------------------------------

def build_candidate_nav(
    strategy: str,
    signal_raw: pd.Series,
    method: str,
    direction: str,
) -> pd.Series:
    """Build candidate NAV for (strategy, signal, method, direction).

    Parameters
    ----------
    strategy   : 'S1' | 'S2' | 'S3'
    signal_raw : raw float signal pd.Series with DatetimeIndex
    method     : 'M1' | 'M2' | 'M4' | 'M5' | 'M6'
    direction  : per-method (see _get_method_multiplier)

    Pipeline
    --------
    1. quantile_cut(levels=4) on full sample
    2. apply_publication_lag('daily') → shift +1 BD
    3. dispatch to S1/S2/S3 native builder
    """
    sig_q = quantile_cut(signal_raw.dropna(), levels=4)
    sig_q = sig_q.dropna().astype('int8')
    sig_lagged = apply_publication_lag(sig_q, 'daily')
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep='last')]

    if strategy == 'S1':
        return _build_s1_native(sig_lagged, method, direction)
    elif strategy == 'S2':
        return _build_s2_native(sig_lagged, method, direction)
    elif strategy == 'S3':
        return _build_s3_native(sig_lagged, method, direction)
    raise ValueError(f"Unknown strategy: {strategy!r}")
