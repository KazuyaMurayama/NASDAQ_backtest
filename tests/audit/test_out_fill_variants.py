import numpy as np
from src.audit.leverup_b1c1_20260612 import _build_p09_nav_c1
from src.audit.out_fill_variants_20260620 import _build_out_fill_variant, alloc_base


def _toy_inputs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    r_base = rng.normal(0.0005, 0.02, n)
    ret_gold = rng.normal(0.0002, 0.01, n)
    ret_bond = rng.normal(0.0001, 0.006, n)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[100:] = rng.random(n - 100) < 0.5  # ~half OUT
    w_g = np.full(n, 0.6)
    w_b = np.full(n, 0.4)
    bond_on = rng.random(n) < 0.5
    sofr_arr = np.full(n, 0.04 / 252)
    return r_base, ret_gold, ret_bond, fund_active, w_g, w_b, bond_on, sofr_arr


def test_base_variant_matches_legacy_c1():
    args = _toy_inputs()
    nav_legacy, r_legacy, eff_legacy = _build_p09_nav_c1(*args)
    nav_new, r_new, eff_new = _build_out_fill_variant(*args, alloc_fn=alloc_base)
    assert np.allclose(r_new, r_legacy, atol=1e-12, equal_nan=True)
    assert np.allclose(nav_new, nav_legacy, atol=1e-9, equal_nan=True)
    assert np.array_equal(eff_new, eff_legacy)


def test_extra_ctx_reaches_alloc_fn():
    args = _toy_inputs()
    seen = {}
    def spy(ctx):
        seen["k"] = "out_strength" in ctx
        return alloc_base(ctx)
    _build_out_fill_variant(*args, alloc_fn=spy, out_strength=np.zeros(len(args[0])))
    assert seen["k"] is True


from src.audit.out_fill_variants_20260620 import inverse_vol_weights_cadence
from src.audit.run_p01_backtest_20260611 import _inverse_vol_weights


def test_cadence_helper_matches_legacy_at_5bd():
    rng = np.random.default_rng(1)
    rg = rng.normal(0, 0.01, 500)
    rb = rng.normal(0, 0.006, 500)
    wg_legacy, _ = _inverse_vol_weights(rg, rb, 63)
    wg_new, _ = inverse_vol_weights_cadence(rg, rb, 63, update_bd=5)
    assert np.allclose(wg_legacy, wg_new, atol=1e-12)


def test_cadence_daily_differs_from_weekly():
    rng = np.random.default_rng(7)
    rg = rng.normal(0, 0.012, 400)
    rb = rng.normal(0, 0.006, 400)
    wg_daily, _ = inverse_vol_weights_cadence(rg, rb, 63, update_bd=1)
    wg_weekly, _ = inverse_vol_weights_cadence(rg, rb, 63, update_bd=5)
    # daily updates strictly more often -> at least some days differ
    assert not np.allclose(wg_daily, wg_weekly, atol=1e-9)
    # both still within clamp
    assert wg_daily.min() >= 0.25 - 1e-9 and wg_daily.max() <= 0.75 + 1e-9


def test_cadence_rejects_zero_update_bd():
    import pytest
    with pytest.raises(ValueError):
        inverse_vol_weights_cadence(np.zeros(10), np.zeros(10), 5, update_bd=0)


from src.audit.out_fill_variants_20260620 import bond_gate_hysteresis


def test_bond_gate_hysteresis():
    mom = np.array([np.nan, -0.10, 0.02, 0.06, 0.01, -0.02, -0.06, 0.00])
    on = bond_gate_hysteresis(mom, on_thr=0.05, off_thr=-0.05)
    # NaN->off; -0.10 off; 0.02 in-band prev off->off; 0.06->on;
    # 0.01 in-band prev on->on; -0.02 in-band prev on->on; -0.06->off;
    # 0.00 in-band prev off->off
    assert on.tolist() == [False, False, False, True, True, True, False, False]


def test_bond_gate_hysteresis_nan_forces_off():
    mom = np.array([0.10, np.nan, 0.01])  # on, then NaN forces off, then in-band holds off
    on = bond_gate_hysteresis(mom)
    assert on.tolist() == [True, False, False]


from src.audit.out_fill_variants_20260620 import make_alloc_vol_target


def _ctx_highvol(n=400, seed=2):
    rng = np.random.default_rng(seed)
    return {
        "ret_gold": rng.normal(0, 0.025, n),   # ~40% annualized vol (high)
        "ret_bond": rng.normal(0, 0.020, n),
        "w_g": np.full(n, 0.6),
        "w_b": np.full(n, 0.4),
        "bond_on": np.ones(n, dtype=bool),
        "sofr_arr": np.full(n, 0.04 / 252),
        "fund_active": np.ones(n, dtype=bool),
    }


def test_vol_target_scales_down_high_vol():
    ctx = _ctx_highvol()
    alloc = make_alloc_vol_target(target_vol=0.10, window=63)
    w_gold, w_bond, w_cash = alloc(ctx)
    assert w_cash[200] > 0.0
    assert (w_gold[200] + w_bond[200]) < 1.0
    assert np.all(w_gold >= 0) and np.all(w_bond >= 0) and np.all(w_cash >= 0)
    assert np.all(w_gold + w_bond + w_cash <= 1.0 + 1e-9)
    # cash fully absorbs the de-scale -> weights sum to exactly 1.0
    assert np.allclose(w_gold + w_bond + w_cash, 1.0, atol=1e-9)


def test_vol_target_no_lookahead():
    """The scale on day t must not depend on day t's own return.
    Changing ONLY the last day's gold return must not change any weight
    on days < n-1 (because sigma is shifted by 1, day t uses past only)."""
    ctx_a = _ctx_highvol()
    ctx_b = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ctx_a.items()}
    ctx_b["ret_gold"][-1] += 0.5  # perturb only the LAST day's return
    a = make_alloc_vol_target(0.10, 63)
    g_a, b_a, c_a = a(ctx_a)
    g_b, b_b, c_b = a(ctx_b)
    # all weights for days 0..n-2 identical; only the last day MAY differ if it
    # used same-day return (it must NOT, so even last-day weight is unchanged).
    assert np.allclose(g_a[:-1], g_b[:-1], atol=1e-12)
    assert np.allclose(g_a, g_b, atol=1e-12)  # last day too: shift(1) => no same-day use


from src.audit.out_fill_variants_20260620 import make_alloc_conviction_cash, make_alloc_gold_tilt


def test_conviction_cash_more_cash_when_strong():
    n = 50
    ctx = {
        "ret_gold": np.zeros(n), "ret_bond": np.zeros(n),
        "w_g": np.full(n, 0.6), "w_b": np.full(n, 0.4),
        "bond_on": np.ones(n, dtype=bool),
        "sofr_arr": np.full(n, 0.04 / 252),
        "fund_active": np.ones(n, dtype=bool),
        "out_strength": np.linspace(0.0, 1.0, n),  # 0=weak OUT, 1=strong OUT
    }
    alloc = make_alloc_conviction_cash(max_cash=0.5)
    w_gold, w_bond, w_cash = alloc(ctx)
    assert w_cash[-1] > w_cash[0]
    assert abs(w_cash[0]) < 1e-9          # weakest OUT -> no extra cash (bond_on so base-cash=0)
    assert abs(w_cash[-1] - 0.5) < 1e-9   # strongest OUT -> max_cash
    assert np.allclose(w_gold + w_bond + w_cash, 1.0, atol=1e-9)


def test_conviction_cash_fallback_without_strength():
    n = 10
    ctx = {
        "ret_gold": np.zeros(n), "ret_bond": np.zeros(n),
        "w_g": np.full(n, 0.6), "w_b": np.full(n, 0.4),
        "bond_on": np.ones(n, dtype=bool),
        "sofr_arr": np.full(n, 0.04 / 252),
        "fund_active": np.ones(n, dtype=bool),
        # no out_strength key
    }
    alloc = make_alloc_conviction_cash(max_cash=0.5)
    w_gold, w_bond, w_cash = alloc(ctx)
    # falls back to base: gold=0.6, bond=0.4 (bond_on), cash=0
    assert np.allclose(w_gold, 0.6) and np.allclose(w_bond, 0.4)
    assert np.allclose(w_cash, 0.0)


def test_conviction_cash_rejects_bad_max_cash():
    import pytest
    with pytest.raises(ValueError):
        make_alloc_conviction_cash(max_cash=1.5)


def test_gold_tilt_in_highvol():
    n = 20
    ctx = {
        "ret_gold": np.zeros(n), "ret_bond": np.zeros(n),
        "w_g": np.full(n, 0.5), "w_b": np.full(n, 0.5),
        "bond_on": np.ones(n, dtype=bool),
        "sofr_arr": np.full(n, 0.04 / 252),
        "fund_active": np.ones(n, dtype=bool),
        "highvol_mask": np.array([False] * 10 + [True] * 10),
    }
    alloc = make_alloc_gold_tilt(gold_floor_highvol=0.75)
    w_gold, w_bond, w_cash = alloc(ctx)
    assert abs(w_gold[0] - 0.5) < 1e-9      # calm day: base
    assert w_gold[15] >= 0.75 - 1e-9        # highvol day: gold tilted up
    assert np.allclose(w_gold + w_bond + w_cash, 1.0, atol=1e-9)


def test_gold_tilt_fallback_without_mask():
    n = 5
    ctx = {
        "ret_gold": np.zeros(n), "ret_bond": np.zeros(n),
        "w_g": np.full(n, 0.5), "w_b": np.full(n, 0.5),
        "bond_on": np.ones(n, dtype=bool),
        "sofr_arr": np.full(n, 0.04 / 252),
        "fund_active": np.ones(n, dtype=bool),
    }
    alloc = make_alloc_gold_tilt(0.75)
    w_gold, w_bond, w_cash = alloc(ctx)
    assert np.allclose(w_gold, 0.5) and np.allclose(w_bond, 0.5)
    assert np.allclose(w_cash, 0.0)


from src.audit.out_fill_variants_20260620 import apply_in_leg_vol_brake


def test_in_leg_vol_brake_blends_to_cash():
    n = 300
    rng = np.random.default_rng(3)
    r = rng.normal(0.0008, 0.03, n)        # IN-leg full-lev returns, ~48% vol
    fund_active = np.zeros(n, dtype=bool)  # all IN
    sofr = np.full(n, 0.04 / 252)
    r_braked = apply_in_leg_vol_brake(r, fund_active, sofr,
                                      target_vol=0.30, window=63)
    # on a high-vol IN day, braked return is a convex mix of r and (small) sofr,
    # so it is pulled toward sofr: |r_braked - sofr| <= |r - sofr| for that day
    hi = 200
    assert abs(r_braked[hi] - sofr[hi]) <= abs(r[hi] - sofr[hi]) + 1e-12
    # warmup days (no sigma yet) unchanged
    assert np.allclose(r_braked[:63], r[:63], atol=1e-12)


def test_in_leg_vol_brake_skips_out_days():
    n = 200
    rng = np.random.default_rng(11)
    r = rng.normal(0.0008, 0.03, n)
    fund_active = np.ones(n, dtype=bool)   # all OUT -> brake never applies
    sofr = np.full(n, 0.04 / 252)
    r_braked = apply_in_leg_vol_brake(r, fund_active, sofr, 0.30, 63)
    assert np.allclose(r_braked, r, atol=1e-12)


def test_in_leg_vol_brake_no_lookahead():
    """Day t's brake must use sigma over returns strictly BEFORE t. With
    shift(1), sig[t] = std(r[t-window .. t-1]); perturbing r[t] itself must NOT
    change the braked value at day t. A broken (no-shift) impl WOULD change it,
    because sig[t] would then include r[t]."""
    n = 300
    rng = np.random.default_rng(3)
    r_a = rng.normal(0.0008, 0.03, n)
    r_b = r_a.copy()
    r_b[200] += 2.0                       # perturb ONLY day 200's own return
    fund_active = np.zeros(n, dtype=bool)  # all IN
    sofr = np.full(n, 0.04 / 252)
    ra = apply_in_leg_vol_brake(r_a, fund_active, sofr, 0.30, 63)
    rb = apply_in_leg_vol_brake(r_b, fund_active, sofr, 0.30, 63)
    # With shift(1): sig[200] uses r[137:200] (excludes r[200]), so day 200's
    # brake fraction is identical; the only difference in the OUTPUT at index
    # 200 is the direct r value passed through the (unchanged) blend. To isolate
    # the brake-fraction (the thing that must not see the future), compare the
    # frac applied: easiest is to check days AFTER 200 where the perturbation
    # legitimately DOES enter sig, AND day 200's fraction is unchanged.
    #
    # Direct check: reconstruct frac at day 200 from output is hard, so assert
    # the brake FRACTION at 200 is unchanged by checking that the mapping from
    # input to output at 200 is the same linear blend in both runs. Since only
    # r[200] changed, if the fraction f200 is identical, then:
    #   ra[200] = (1-f)*r_a[200] + f*sofr ; rb[200] = (1-f)*r_b[200] + f*sofr
    # => (rb[200]-ra[200]) == (1-f)*(r_b[200]-r_a[200]).
    # A no-shift impl would change f at 200 (since sig[200] would include the
    # +2.0 spike -> larger sig -> larger f), making the implied (1-f) differ.
    implied_one_minus_f = (rb[200] - ra[200]) / (r_b[200] - r_a[200])
    # recompute the correct fraction independently (shift(1), past-only sigma)
    import pandas as pd
    sig200 = (pd.Series(r_a).rolling(63, min_periods=63).std(ddof=1)
              * np.sqrt(252)).shift(1).values[200]
    f_correct = max(0.0, min(0.5, 1.0 - 0.30 / sig200)) if (np.isfinite(sig200) and sig200 > 0.30) else 0.0
    assert np.isclose(implied_one_minus_f, 1.0 - f_correct, atol=1e-9), (
        "shift(1) violated: day 200 brake fraction depends on day 200's own return")
