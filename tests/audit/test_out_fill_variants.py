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
