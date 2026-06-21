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
    frac = np.where((np.isfinite(dvol)) & (dvol > target_dvol),
                    1.0 - target_dvol / dvol, 0.0)
    frac = np.clip(frac, 0.0, max_frac_cash)
    return _blend_to_cash(r_arr, fund_active, sofr_arr, frac)
