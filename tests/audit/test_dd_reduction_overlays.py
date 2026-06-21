import numpy as np
import pandas as pd
from src.audit.out_fill_variants_20260620 import apply_in_leg_vol_brake
from src.audit.dd_reduction_overlays_20260621 import (
    apply_downside_dev_brake, apply_dd_throttle, apply_asym_vol_brake,
)


def _toy(n=400, seed=5):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, 0.02, n)
    fund_active = np.zeros(n, dtype=bool)        # all IN
    sofr = np.full(n, 0.04 / 252)
    return r, fund_active, sofr


def test_downside_dev_only_penalizes_downside_vol():
    """An all-UP high-vol series must NOT trigger the downside brake (downside
    dev ~ 0), whereas A7 (total vol) WOULD brake it."""
    n = 400
    r_up = np.abs(np.random.default_rng(1).normal(0.0, 0.03, n))  # all >= 0
    fund_active = np.zeros(n, dtype=bool)
    sofr = np.full(n, 0.04 / 252)
    out_b1 = apply_downside_dev_brake(r_up, fund_active, sofr,
                                      target_dvol=0.20, window=63)
    assert np.allclose(out_b1, r_up, atol=1e-12)          # downside dev ~0 -> no brake
    out_a7 = apply_in_leg_vol_brake(r_up, fund_active, sofr,
                                    target_vol=0.20, window=63)
    assert not np.allclose(out_a7, r_up, atol=1e-9)        # A7 brakes the up-vol


def test_downside_dev_brakes_a_drawdown():
    n = 400
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.005, n)
    r[200:230] = -0.04                       # sustained down-run -> high dvol
    fund_active = np.zeros(n, dtype=bool)
    sofr = np.full(n, 0.04 / 252)
    out = apply_downside_dev_brake(r, fund_active, sofr, target_dvol=0.20, window=63)
    braked = ~np.isclose(out[231:260], r[231:260])
    assert braked.any()


def test_downside_dev_no_lookahead():
    n = 300
    rng = np.random.default_rng(3)
    ra = rng.normal(-0.001, 0.02, n)
    rb = ra.copy(); rb[150] -= 0.5
    fund_active = np.zeros(n, dtype=bool)
    sofr = np.full(n, 0.04 / 252)
    oa = apply_downside_dev_brake(ra, fund_active, sofr, 0.20, 63)
    ob = apply_downside_dev_brake(rb, fund_active, sofr, 0.20, 63)
    # implied (1-f) at day 150: only r[150] changed, so if f150 unchanged,
    # (ob[150]-oa[150]) == (1-f150)*(rb[150]-ra[150]).
    implied = (ob[150] - oa[150]) / (rb[150] - ra[150])
    neg = np.where(ra < 0, ra, 0.0)
    dvol150 = (pd.Series(neg).rolling(63, min_periods=63).std(ddof=1)
               * np.sqrt(252)).shift(1).values[150]
    f = 0.0
    if np.isfinite(dvol150) and dvol150 > 0.20:
        f = min(0.5, 1.0 - 0.20 / dvol150)
    assert np.isclose(implied, 1.0 - f, atol=1e-9), "downside brake look-ahead"


def test_dd_throttle_engages_in_drawdown():
    n = 300
    r = np.full(n, 0.001)
    r[100:140] = -0.03                       # build a deep drawdown
    fund_active = np.zeros(n, dtype=bool)
    sofr = np.full(n, 0.04 / 252)
    out = apply_dd_throttle(r, fund_active, sofr,
                            tiers=((0.15, 0.25), (0.25, 0.50)))
    changed = ~np.isclose(out[120:140], r[120:140])
    assert changed.any()                     # deep in drawdown -> brake engages
    assert np.isclose(out[50], r[50])        # before any drawdown -> no brake


def test_dd_throttle_no_lookahead():
    n = 250
    rng = np.random.default_rng(9)
    ra = rng.normal(-0.002, 0.02, n)
    rb = ra.copy(); rb[120] -= 0.4
    fund_active = np.zeros(n, dtype=bool)
    sofr = np.full(n, 0.04 / 252)
    oa = apply_dd_throttle(ra, fund_active, sofr, tiers=((0.15, 0.25), (0.25, 0.50)))
    ob = apply_dd_throttle(rb, fund_active, sofr, tiers=((0.15, 0.25), (0.25, 0.50)))
    implied = (ob[120] - oa[120]) / (rb[120] - ra[120])
    assert np.allclose(oa[:120], ob[:120], atol=1e-12)     # past unaffected
    assert 0.5 - 1e-9 <= implied <= 1.0 + 1e-9             # day120 frac (1-f), f in [0,0.5]


def test_dd_throttle_tiers_monotone():
    """Deeper drawdown -> higher cash fraction (the 0.25 tier engages 0.50).
    Day 110: dd_lag~0.183 -> shallow tier frac=0.25. Day 195: dd_lag~0.853 ->
    deep tier frac=0.50. Both must be in DIFFERENT tiers."""
    n = 400
    r = np.full(n, 0.0005)
    r[100:200] = -0.02                        # progressively deeper DD
    fund_active = np.zeros(n, dtype=bool)
    sofr = np.full(n, 0.04 / 252)
    out = apply_dd_throttle(r, fund_active, sofr, tiers=((0.15, 0.25), (0.25, 0.50)))
    # reconstruct applied frac on a deep day vs a shallow day
    def frac_at(t):
        denom = r[t] - sofr[t]
        return (r[t] - out[t]) / denom if abs(denom) > 1e-12 else 0.0
    # day 110: dd_lag~0.183 -> shallow tier [0.15,0.25) -> frac=0.25
    # day 195: dd_lag~0.853 -> deep tier [0.25,inf) -> frac=0.50
    shallow = frac_at(110)
    deep = frac_at(195)
    assert deep >= shallow - 1e-9
    assert shallow < deep - 1e-9 or (abs(shallow - 0.25) < 1e-6 and abs(deep - 0.50) < 1e-6)


# ---------------------------------------------------------------------------
# Part C: B2 non-monotone tier guard
# ---------------------------------------------------------------------------

def test_dd_throttle_rejects_nonmonotone_tiers():
    import pytest
    with pytest.raises(ValueError):
        apply_dd_throttle(np.zeros(10), np.zeros(10, dtype=bool),
                          np.full(10, 0.04 / 252), tiers=((0.15, 0.50), (0.25, 0.25)))


# ---------------------------------------------------------------------------
# Part A: B3 asymmetric vol brake
# ---------------------------------------------------------------------------

def test_asym_brake_slow_to_release():
    """After vol drops below target, the brake persists for release_days
    consecutive sub-threshold days before releasing. A symmetric A7 releases
    immediately, so asym stays braked LONGER after a spike."""
    n = 200
    r = np.full(n, 0.002)
    r[80:90] = -0.06                          # vol spike window
    fund_active = np.zeros(n, dtype=bool)
    sofr = np.full(n, 0.04 / 252)
    out_asym = apply_asym_vol_brake(r, fund_active, sofr, target_vol=0.30,
                                    window=63, max_frac_cash=0.5, release_days=5)
    out_a7 = apply_in_leg_vol_brake(r, fund_active, sofr, target_vol=0.30,
                                    window=63)
    post = slice(140, 175)
    a7_released = np.isclose(out_a7[post], r[post])
    asym_braked = ~np.isclose(out_asym[post], r[post])
    assert (a7_released & asym_braked).any(), "asym should hold brake longer than A7"


def test_asym_brake_no_lookahead():
    """frac[t] must use vol over returns strictly BEFORE t (shift(1)).
    Perturbing r[t]'s OWN value must NOT change frac[t]. We reconstruct the
    implied (1-frac[t]) from the linear blend: only r[t] changed, so
    (ob[t]-oa[t]) == (1-frac[t])*(rb[t]-ra[t]) IFF frac[t] is unchanged.
    A no-shift impl would let r[t]'s spike enter vol[t] and change frac[t].

    Verified: with seed=4, std=0.025, frac_t0 ≈ 0.2876 (brake ON at t0=130).
    The test is only meaningful when frac_t0 > 0; this configuration ensures it."""
    n = 260
    rng = np.random.default_rng(4)
    ra = rng.normal(0.0, 0.025, n)          # ~40% annualized vol -> brake is active
    rb = ra.copy()
    t0 = 130
    rb[t0] += 0.6                            # perturb ONLY day t0's own return
    fund_active = np.zeros(n, dtype=bool)
    sofr = np.full(n, 0.04 / 252)
    oa = apply_asym_vol_brake(ra, fund_active, sofr, 0.30, 63, 0.5, 5)
    ob = apply_asym_vol_brake(rb, fund_active, sofr, 0.30, 63, 0.5, 5)
    # past days unaffected (sanity)
    assert np.allclose(oa[:t0], ob[:t0], atol=1e-12)
    # implied (1-frac[t0]); only r[t0] differs between a and b
    implied = (ob[t0] - oa[t0]) / (rb[t0] - ra[t0])
    # frac[t0] is in [0, 0.5] so (1-frac) in [0.5, 1.0]; with correct shift it is
    # IDENTICAL in a and b (frac[t0] independent of r[t0]) -> implied == 1-frac_a[t0].
    # Recompute frac_a[t0] independently from PAST-ONLY vol (shift(1)) replicating
    # the stateful loop up to t0 on series `ra`:
    vol = (pd.Series(ra).rolling(63, min_periods=63).std(ddof=1)
           * np.sqrt(252)).shift(1).values
    state_on = False; below = 0; cur = 0.0
    for t in range(t0 + 1):
        v = vol[t]
        if np.isfinite(v) and v > 0.30:
            state_on = True; below = 0; cur = min(0.5, 1.0 - 0.30 / v)
        elif state_on:
            if np.isfinite(v) and v <= 0.30:
                below += 1
                if below >= 5:
                    state_on = False; cur = 0.0
            else:
                below = 0
    frac_t0 = cur if state_on else 0.0
    assert np.isclose(implied, 1.0 - frac_t0, atol=1e-9), \
        "asym brake look-ahead: frac[t0] depends on r[t0]'s own value"
