"""P09_C1 OUT-sleeve allocation variations (signal->action only).

Generalizes _build_p09_nav_c1: the Gold/Bond/Cash split on OUT days is supplied
by an alloc_fn callback. Instruments, IN-leg, cost model are UNCHANGED.
alloc_base() reproduces the legacy C1 fill bit-for-bit (proven by test).
"""
from __future__ import annotations
import numpy as np

from src.audit.run_p02_p09_backtest_20260611 import FEE_GOLD, FEE_BOND

TRADING_DAYS = 252


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
    SOFR. Fees charged only on active Gold/Bond legs. Sleeve weights need not
    sum to 1.0; any residual is implicitly held flat (0%-yield).
    """
    bond_on = np.asarray(bond_on, dtype=bool)
    sofr_arr = np.asarray(sofr_arr, float)
    ctx = {
        "ret_gold": np.asarray(ret_gold, float),
        "ret_bond": np.asarray(ret_bond, float),
        "w_g": np.asarray(w_g, float),
        "w_b": np.asarray(w_b, float),
        "bond_on": bond_on,
        "sofr_arr": sofr_arr,
        "fund_active": np.asarray(fund_active, dtype=bool),
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
