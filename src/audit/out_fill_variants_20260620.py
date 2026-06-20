"""P09_C1 OUT-sleeve allocation variations (signal->action only).

Generalizes _build_p09_nav_c1: the Gold/Bond/Cash split on OUT days is supplied
by an alloc_fn callback. Instruments, IN-leg, cost model are UNCHANGED.
alloc_base() reproduces the legacy C1 fill bit-for-bit (proven by test).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.audit.run_p02_p09_backtest_20260611 import FEE_GOLD, FEE_BOND, TRADING_DAYS


def alloc_base(ctx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy C1 split: Gold=w_g always, Bond=w_b iff bond_on else 0,
    Cash(SOFR)=w_b iff not bond_on else 0. Sums to w_g+w_b=1.0."""
    w_g = ctx["w_g"]
    w_b = ctx["w_b"]
    bond_on = ctx["bond_on"]
    w_bond = np.where(bond_on, w_b, 0.0)
    w_cash = np.where(bond_on, 0.0, w_b)
    return w_g.copy(), w_bond, w_cash


def _build_out_fill_variant(r_base, ret_gold, ret_bond, fund_active,
                            w_g, w_b, bond_on, sofr_arr, *, alloc_fn):
    """Same signature/return as _build_p09_nav_c1 but split via alloc_fn.

    alloc_fn(ctx) returns (w_gold, w_bond, w_cash) arrays (len n). Cash earns
    SOFR. Fees charged only on active Gold/Bond legs.
    Sleeve weights need not sum to 1.0; any gap to 1.0 is implicitly cash at 0% (not SOFR).
    """
    bond_on = np.asarray(bond_on, dtype=bool)
    sofr_arr = np.asarray(sofr_arr, float)
    ctx = {
        "ret_gold": np.array(ret_gold, float),
        "ret_bond": np.array(ret_bond, float),
        "w_g": np.array(w_g, float),
        "w_b": np.array(w_b, float),
        "bond_on": np.array(bond_on, dtype=bool),
        "sofr_arr": np.array(sofr_arr, float),
        "fund_active": np.array(fund_active, dtype=bool),
    }
    w_gold, w_bond, w_cash = alloc_fn(ctx)
    fee_daily = (w_gold * FEE_GOLD + w_bond * FEE_BOND) / TRADING_DAYS
    cash_yield = w_cash * sofr_arr
    r_blend = (w_gold * ctx["ret_gold"] + w_bond * ctx["ret_bond"]
               + cash_yield - fee_daily)
    r = np.where(fund_active, r_blend, r_base)
    r = np.clip(r, -0.999, None)
    nav = np.cumprod(1.0 + r)
    eff_active = np.asarray(fund_active, dtype=bool).copy()
    return nav, r, eff_active


def inverse_vol_weights_cadence(ret_gold, ret_bond, window, update_bd,
                                clamp=(0.25, 0.75)):
    """Inverse-vol gold/bond weights, recomputed every `update_bd` business
    days (1=daily, 5=weekly). Mirrors run_p01._inverse_vol_weights logic with a
    parameterized cadence. Returns (w_g, w_b). At update_bd=5, window=63 this
    reduces to the legacy _inverse_vol_weights exactly.

    Legacy details matched:
      - rolling std: ddof=1, annualized * sqrt(252), min_periods=window
      - warm-start: last = 0.5 (before first valid window)
      - clamp: [0.25, 0.75]
      - cadence: t % update_bd == 0 (update_bd=5 => legacy WEIGHT_UPDATE_BD)
      - output array initialized with np.full(n, np.nan) but last is always
        assigned, so no NaNs in output
    """
    if update_bd < 1:
        raise ValueError(f"update_bd must be >= 1, got {update_bd!r}")
    rg = pd.Series(np.asarray(ret_gold, float))
    rb = pd.Series(np.asarray(ret_bond, float))
    sig_g = (rg.rolling(window, min_periods=window).std(ddof=1)
             * np.sqrt(TRADING_DAYS)).values
    sig_b = (rb.rolling(window, min_periods=window).std(ddof=1)
             * np.sqrt(TRADING_DAYS)).values
    n = len(rg)
    lo, hi = clamp
    w_g = np.full(n, np.nan)
    last = 0.5  # warm-start default before first valid window
    for t in range(n):
        if t % update_bd == 0:
            sg, sb = sig_g[t], sig_b[t]
            if np.isfinite(sg) and np.isfinite(sb) and sg > 0 and sb > 0:
                inv_g = 1.0 / sg
                inv_b = 1.0 / sb
                wg = inv_g / (inv_g + inv_b)
                wg = float(np.clip(wg, lo, hi))
                last = wg
            # else: keep previous `last` (warm-start / ffill)
        w_g[t] = last
    return w_g, 1.0 - w_g


def bond_gate_hysteresis(bond_mom252, on_thr=0.05, off_thr=-0.05):
    """Hysteresis bond gate. ON when mom>on_thr, OFF when mom<off_thr, hold
    previous state in between. NaN -> OFF. Returns bool array.

    Replaces the binary `bond_mom252 > 0` gate; reduces flip-flopping near 0."""
    mom = np.asarray(bond_mom252, float)
    n = len(mom)
    on = np.zeros(n, dtype=bool)
    state = False
    for t in range(n):
        m = mom[t]
        if np.isnan(m):
            state = False
        elif m > on_thr:
            state = True
        elif m < off_thr:
            state = False
        # else: hold previous state
        on[t] = state
    return on


def make_alloc_vol_target(target_vol=0.10, window=63, max_scale=1.0):
    """Factory: alloc_fn that scales the Gold+Bond sleeve so the blend's
    realized annualized vol ~= target_vol, routing the de-scaled remainder to
    SOFR cash. The vol estimate uses returns STRICTLY before day t (shift(1))
    to avoid look-ahead. scale capped at max_scale (sleeve never levered).
    Weights sum to exactly 1.0 (cash absorbs de-scale)."""
    def alloc(ctx):
        rg, rb = ctx["ret_gold"], ctx["ret_bond"]
        w_g, w_b, bond_on = ctx["w_g"], ctx["w_b"], ctx["bond_on"]
        w_bond_raw = np.where(bond_on, w_b, 0.0)
        blend = w_g * rg + w_bond_raw * rb
        # rolling realized vol of the blend, then SHIFT(1): day t uses [.. t-1]
        sig = (pd.Series(blend).rolling(window, min_periods=window).std(ddof=1)
               * np.sqrt(TRADING_DAYS)).shift(1).values
        scale = np.where((np.isfinite(sig)) & (sig > 0),
                         target_vol / sig, 1.0)
        scale = np.clip(scale, 0.0, max_scale)
        scale = np.where(np.isnan(sig), 1.0, scale)  # warmup -> full exposure
        w_gold = w_g * scale
        w_bond = w_bond_raw * scale
        w_cash = 1.0 - w_gold - w_bond
        return w_gold, w_bond, w_cash
    return alloc
