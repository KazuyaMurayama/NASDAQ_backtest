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
                            w_g, w_b, bond_on, sofr_arr, *, alloc_fn,
                            **extra_ctx):
    """Same signature/return as _build_p09_nav_c1 but split via alloc_fn.

    alloc_fn(ctx) returns (w_gold, w_bond, w_cash) arrays (len n). Cash earns
    SOFR. Fees charged only on active Gold/Bond legs.
    Sleeve weights need not sum to 1.0; any gap to 1.0 is implicitly cash at 0% (not SOFR).

    extra_ctx: any additional keys (e.g. out_strength, highvol_mask) are merged
    into ctx right before alloc_fn is called, so conviction/tilt alloc factories
    can read regime/strength signals supplied by the caller.
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
    ctx.update(extra_ctx)
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


def make_alloc_conviction_cash(max_cash=0.5):
    """Factory: alloc_fn that routes out_strength*max_cash of the OUT sleeve to
    SOFR cash (more cash on strong/deep OUT, less on weak OUT near the exit
    threshold) and scales the base Gold/Bond/base-cash split down to fill the
    rest, preserving their ratios. out_strength in [0,1] read from ctx; if
    absent, returns the unmodified base split."""
    if not (0.0 < max_cash <= 1.0):
        raise ValueError(f"max_cash must be in (0, 1], got {max_cash!r}")
    def alloc(ctx):
        w_g, w_b, bond_on = ctx["w_g"], ctx["w_b"], ctx["bond_on"]
        strength = ctx.get("out_strength")
        w_bond_base = np.where(bond_on, w_b, 0.0)
        w_cash_base = np.where(bond_on, 0.0, w_b)
        if strength is None:
            return w_g.copy(), w_bond_base, w_cash_base
        s = np.clip(np.asarray(strength, float), 0.0, 1.0)
        extra_cash = s * max_cash
        risk_frac = 1.0 - extra_cash
        base_total = w_g + w_bond_base + w_cash_base  # == 1.0
        w_gold = w_g / base_total * risk_frac
        w_bond = w_bond_base / base_total * risk_frac
        w_cash = w_cash_base / base_total * risk_frac + extra_cash
        return w_gold, w_bond, w_cash
    return alloc


def make_alloc_gold_tilt(gold_floor_highvol=0.75):
    """Factory: alloc_fn = base split on calm days; on highvol_mask days raise
    Gold to at least gold_floor_highvol, taking the increment from Bond first
    then Cash. Preserves sum==1.0. If highvol_mask absent from ctx, base split.

    Edge case: if gold_floor_highvol <= w_g (base gold already meets floor),
    need=0 so this is a no-op for those days (correct behavior, sum preserved).
    If floor exceeds available bond+cash, gold ends at w_g+w_bond+w_cash (=1.0)
    which preserves sum==1.0.
    """
    def alloc(ctx):
        w_g, w_b, bond_on = ctx["w_g"], ctx["w_b"], ctx["bond_on"]
        hv = ctx.get("highvol_mask")
        w_gold = w_g.copy()
        w_bond = np.where(bond_on, w_b, 0.0)
        w_cash = np.where(bond_on, 0.0, w_b)
        if hv is None:
            return w_gold, w_bond, w_cash
        hv = np.asarray(hv, dtype=bool)
        need = np.maximum(0.0, gold_floor_highvol - w_gold)
        need = np.where(hv, need, 0.0)
        take_bond = np.minimum(need, w_bond)
        w_bond = w_bond - take_bond
        rem = need - take_bond
        take_cash = np.minimum(rem, w_cash)
        w_cash = w_cash - take_cash
        w_gold = w_gold + take_bond + take_cash
        return w_gold, w_bond, w_cash
    return alloc


def apply_in_leg_vol_brake(r, fund_active, sofr_arr, target_vol=0.30,
                           window=63, max_frac_cash=0.5):
    """Post-process the full-strategy return array: on IN days (~fund_active)
    whose trailing realized vol exceeds target_vol, blend a fraction of the
    day's return into SOFR cash (de-risk by holding cash, not de-levering).
    frac_cash = clip(1 - target_vol/sig, 0, max_frac_cash). The vol estimate
    uses returns STRICTLY before day t (shift(1)) -> no look-ahead. OUT days
    untouched. Returns a NEW return array; recompute NAV downstream."""
    r = np.asarray(r, float).copy()
    fund_active = np.asarray(fund_active, dtype=bool)
    sofr_arr = np.asarray(sofr_arr, float)
    sig = (pd.Series(r).rolling(window, min_periods=window).std(ddof=1)
           * np.sqrt(TRADING_DAYS)).shift(1).values
    frac_cash = np.where((np.isfinite(sig)) & (sig > target_vol),
                         1.0 - target_vol / sig, 0.0)
    frac_cash = np.clip(frac_cash, 0.0, max_frac_cash)
    in_day = ~fund_active
    apply = in_day & (frac_cash > 0)
    r[apply] = ((1.0 - frac_cash[apply]) * r[apply]
                + frac_cash[apply] * sofr_arr[apply])
    return r
