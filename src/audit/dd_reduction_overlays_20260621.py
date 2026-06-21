"""A7-based IN-leg drawdown-reduction overlays.

All overlays share A7's safety skeleton: IN days only (~fund_active), retreat
to SOFR cash (not de-lever), shift(1) causal signal, capped retreat fraction.
They differ only in WHAT they threshold (total vol / downside dev / current
drawdown) and the entry/exit symmetry. Used to push MaxDD below A7 while
keeping CAGR ~flat. Every overlay must be benchmarked against a uniform-delever
control (build_uniform_delever) to isolate timing value from plain de-lever
(per the G5 retraction lesson)."""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _blend_to_cash(r, fund_active, sofr_arr, frac_cash):
    """Apply a per-day cash-retreat fraction on IN days only. frac_cash already
    causal/capped. Returns NEW return array."""
    r = np.asarray(r, float).copy()
    fund_active = np.asarray(fund_active, dtype=bool)
    sofr_arr = np.asarray(sofr_arr, float)
    frac = np.clip(np.asarray(frac_cash, float), 0.0, 1.0)
    in_day = ~fund_active
    apply = in_day & (frac > 0)
    r[apply] = (1.0 - frac[apply]) * r[apply] + frac[apply] * sofr_arr[apply]
    return r


def apply_downside_dev_brake(r, fund_active, sofr_arr, target_dvol=0.20,
                             window=63, max_frac_cash=0.5):
    """B1: brake on DOWNSIDE deviation (annualized std of negative daily
    returns only), not total vol. Upside high-vol days do NOT trigger the
    brake. frac_cash = clip(1 - target_dvol/dvol, 0, max_frac_cash). Causal
    via shift(1). IN days only."""
    r_arr = np.asarray(r, float)
    neg = np.where(r_arr < 0.0, r_arr, 0.0)
    dvol = (pd.Series(neg).rolling(window, min_periods=window).std(ddof=1)
            * np.sqrt(TRADING_DAYS)).shift(1).values
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where((np.isfinite(dvol)) & (dvol > target_dvol),
                        1.0 - target_dvol / dvol, 0.0)
    frac = np.clip(frac, 0.0, max_frac_cash)
    return _blend_to_cash(r_arr, fund_active, sofr_arr, frac)


def apply_dd_throttle(r, fund_active, sofr_arr,
                      tiers=((0.15, 0.25), (0.25, 0.50))):
    """B2: throttle by CURRENT drawdown. Build the strategy NAV from r, compute
    running peak and current drawdown dd_t = 1 - nav_t/peak_t, SHIFT by 1 so
    day t uses dd through t-1 (causal). tiers = ascending ((dd_thr, frac_cash),
    ...): the largest tier whose dd_thr <= dd_{t-1} sets frac_cash. IN days only.

    NOTE: dd is computed on the *unbraked* NAV (the strategy's own equity curve
    as it would stand pre-throttle). This is a deliberate causal simplification
    that avoids a feedback loop (we do NOT re-derive dd from the post-brake NAV;
    a fully self-consistent post-throttle dd would need iteration, out of scope).
    The report must disclose this."""
    st = sorted(tiers)
    if any(st[i][1] > st[i + 1][1] for i in range(len(st) - 1)):
        raise ValueError(
            f"tiers must have non-decreasing frac_cash with depth, got {tiers}"
        )
    r_arr = np.asarray(r, float)
    nav = np.cumprod(1.0 + r_arr)
    peak = np.maximum.accumulate(nav)
    dd = 1.0 - nav / peak                     # current drawdown (>=0)
    dd_lag = pd.Series(dd).shift(1).fillna(0.0).values   # causal
    frac = np.zeros_like(r_arr)
    for thr, fc in st:                        # ascending; last (deepest) match wins
        frac = np.where(dd_lag >= thr, fc, frac)
    return _blend_to_cash(r_arr, fund_active, sofr_arr, frac)


def apply_asym_vol_brake(r, fund_active, sofr_arr, target_vol=0.30,
                         window=63, max_frac_cash=0.5, release_days=5):
    """B3: same realized-vol trigger as A7 but ASYMMETRIC entry/exit. The brake
    turns ON the day vol(shift1) exceeds target_vol, and stays ON (holding the
    last engaged frac_cash) until vol has been BELOW target for `release_days`
    consecutive days. Fast to retreat, slow to re-risk. frac magnitude when on =
    clip(1 - target_vol/vol, 0, max_frac_cash) using the latest above-threshold
    vol seen, decaying only on release. IN days only, causal."""
    r_arr = np.asarray(r, float)
    vol = (pd.Series(r_arr).rolling(window, min_periods=window).std(ddof=1)
           * np.sqrt(TRADING_DAYS)).shift(1).values
    n = len(r_arr)
    frac = np.zeros(n)
    state_on = False
    below = 0
    cur = 0.0
    for t in range(n):
        v = vol[t]
        if np.isfinite(v) and v > target_vol:
            state_on = True
            below = 0
            cur = min(max_frac_cash, 1.0 - target_vol / v)
        elif state_on:
            if np.isfinite(v) and v <= target_vol:
                below += 1
                if below >= release_days:
                    state_on = False
                    cur = 0.0
            else:                              # NaN vol while on -> hold
                below = 0
        frac[t] = cur if state_on else 0.0
    return _blend_to_cash(r_arr, fund_active, sofr_arr, frac)
