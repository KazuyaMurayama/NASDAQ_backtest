import numpy as np
import pandas as pd
from src.audit.out_fill_variants_20260620 import apply_in_leg_vol_brake
from src.audit.dd_reduction_overlays_20260621 import apply_downside_dev_brake, apply_dd_throttle


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
    """Deeper drawdown -> higher cash fraction (the 0.25 tier engages 0.50)."""
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
    # find a shallow-DD day (early in the down run) and a deep-DD day (late)
    shallow = frac_at(115)
    deep = frac_at(195)
    assert deep >= shallow - 1e-9
