# -*- coding: utf-8 -*-
"""
src/audit/placebo_audit_g5_20260616.py
=======================================
Adversarial placebo audit for G5_vix_hard MaxDD improvement.

Checks:
  P1 - Signal shuffle: random permutation of vix_mom21 quartile labels
       (block-shuffle to preserve distribution) -> does MaxDD still improve?
  P2 - Time-shift sensitivity: shift vix_mom21 signal by -5BD (older)
       vs +5BD (future leak direction) -> does leak direction inflate results?
  P3 - Uniform de-leverage: same average leverage reduction as G5 but
       without any signal -> how much of MaxDD improvement is pure de-lev?
  P4 - Full-sample vs IS-only quartile boundary: re-run G5 with quartile
       computed on IS-only dates to measure look-ahead bias magnitude.

Each placebo run uses the same NAV machinery as G5.

Output: prints results. No file writes (audit only).
"""

from __future__ import annotations

import os
import sys
import types
import json

# ---- multitasking stub -------------------------------------------------------
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

# ---- Project imports --------------------------------------------------------
import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, FEE_GOLD, FEE_BOND,
)
from src.audit.combine_g5_defoverlay_20260615 import (
    _build_def_mult_arr, _build_full_b3a_g5,
    HARD_MAP, MACRO_FEATURES_PATH,
    B3A_KNOWN_MIN9, B3A_KNOWN_MAXDD,
    SANITY_TOL_MIN9, SANITY_TOL_MAXDD,
)
from src.signals.quantize import quantile_cut
from src.signals.timing import apply_publication_lag

SEED = 42


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _eval_nav(nav_dt, tpy):
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    return pre, aft


def _build_placebo_nav(shared, dates_dt, n_years, ret_gold, ret_bond,
                        fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
                        def_arr):
    nav_dt, r, tpy, exc = _build_full_b3a_g5(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv,
        bond_on, sofr_arr,
        def_mult_arr=def_arr)
    pre, aft = _eval_nav(nav_dt, tpy)
    return pre, aft, nav_dt, r, tpy


def main():
    print("=" * 100)
    print("G5_vix_hard PLACEBO / LOOK-AHEAD AUDIT  2026-06-16")
    print("Adversarially tests whether MaxDD improvement is causal or artifactual")
    print("=" * 100)

    # ---- Load shared DH-W1 assets ----
    print("\n[Step 0] Loading DH-W1 shared assets ...")
    sr._load_dhw1_shared()
    shared   = sr._DHW1_SHARED
    a        = shared["assets"]
    mask     = np.asarray(shared["mask"], dtype=float)
    dates    = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n        = len(dates_dt)
    n_years  = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)

    # ---- Gold/Bond 1x legs ----
    print("[Step 1] Building auxiliary series ...")
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x  = np.asarray(prepare_gold_local(dates), float)
    bond_1x  = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active  = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on   = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr  = np.asarray(a["sofr"], float)

    # ---- Load macro_features.csv ----
    print("[Step 2] Loading macro_features.csv ...")
    macro_df = pd.read_csv(MACRO_FEATURES_PATH, index_col=0, parse_dates=True)

    # =========================================================================
    # BASELINE: B3a_k365 (neutral def_arr = all 1.0)
    # =========================================================================
    print("\n" + "=" * 100)
    print("BASELINE: B3a_k365 (neutral def_arr = all 1.0)")
    neutral_def = np.ones(n, dtype=float)
    pre_base, aft_base, _, r_base, _ = _build_placebo_nav(
        shared, dates_dt, n_years, ret_gold, ret_bond,
        fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        def_arr=neutral_def)
    base_maxdd  = pre_base["MaxDD_FULL"]
    base_min9   = _min_at(aft_base)
    base_sharpe = pre_base["Sharpe_OOS"]
    print("  B3a base: MaxDD=%+.4f%%  min9=%+.4f%%  Sharpe=%.4f"
          % (base_maxdd * 100, base_min9 * 100, base_sharpe))

    ok_min9  = abs(base_min9  - B3A_KNOWN_MIN9)  <= SANITY_TOL_MIN9
    ok_maxdd = abs(base_maxdd - B3A_KNOWN_MAXDD) <= SANITY_TOL_MAXDD
    print("  Sanity: min9 %s  MaxDD %s" % ("OK" if ok_min9 else "FAIL", "OK" if ok_maxdd else "FAIL"))
    if not (ok_min9 and ok_maxdd):
        print("  [HALT] Sanity failed. Aborting.")
        return

    # =========================================================================
    # REFERENCE: G5_vix_hard with FULL-SAMPLE quantile (original implementation)
    # =========================================================================
    print("\n" + "=" * 100)
    print("REFERENCE: G5_vix_hard (FULL-SAMPLE quantile = original implementation)")
    def_arr_g5_fullsample = _build_def_mult_arr(
        "vix_mom21", "daily", HARD_MAP, dates_dt, macro_df)
    avg_lev_g5_full = float(np.mean(def_arr_g5_fullsample))
    print("  def_arr avg=%.4f  min=%.4f  max=%.4f  (HARD_MAP applies)" %
          (avg_lev_g5_full, float(def_arr_g5_fullsample.min()), float(def_arr_g5_fullsample.max())))

    pre_g5_full, aft_g5_full, _, r_g5_full, _ = _build_placebo_nav(
        shared, dates_dt, n_years, ret_gold, ret_bond,
        fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        def_arr=def_arr_g5_fullsample)
    g5_full_maxdd = pre_g5_full["MaxDD_FULL"]
    delta_maxdd_g5_full = (g5_full_maxdd - base_maxdd) * 100
    print("  G5_full: MaxDD=%+.4f%%  delta_MaxDD=%+.4fpp  min9=%+.4f%%  Sharpe=%.4f"
          % (g5_full_maxdd * 100, delta_maxdd_g5_full,
             _min_at(aft_g5_full) * 100, pre_g5_full["Sharpe_OOS"]))
    print("  REFERENCE RESULT: MaxDD improvement = %+.4fpp (from CSV: +2.27pp expected)"
          % delta_maxdd_g5_full)

    # =========================================================================
    # CHECK A4: IS-ONLY quantile boundary (look-ahead bias test)
    # =========================================================================
    print("\n" + "=" * 100)
    print("CHECK A4: IS-ONLY quantile boundary vs FULL-SAMPLE (CRITICAL look-ahead test)")
    print("  IS period ends at: %s" % str(IS_END))

    raw_vix = macro_df["vix_mom21"].dropna()
    raw_vix.index = pd.DatetimeIndex(pd.to_datetime(raw_vix.index))

    # IS-only quantile: compute qcut boundaries from IS dates only
    is_dates_macro = raw_vix.index[raw_vix.index <= IS_END]
    raw_vix_is = raw_vix.loc[is_dates_macro]

    # Compute IS-only quantile boundaries
    try:
        _, is_bins = pd.qcut(raw_vix_is, 4, labels=False, duplicates='drop', retbins=True)
    except ValueError:
        print("  IS-only qcut failed (degenerate). Using full-sample bins.")
        is_bins = None

    if is_bins is not None:
        # Apply IS-only bins to full series
        try:
            sig_q_isonly = pd.cut(raw_vix, bins=is_bins, labels=False,
                                   include_lowest=True)
            # Fill values outside IS bins (OOS values outside IS range)
            # For values above max IS bin -> Q3; below min -> Q0
            sig_q_isonly = sig_q_isonly.astype("float")
            sig_q_isonly[raw_vix < is_bins[0]] = 0.0
            sig_q_isonly[raw_vix > is_bins[-1]] = 3.0
            sig_q_isonly = sig_q_isonly.fillna(3.0)  # NaN -> highest Q (conservative)
            sig_q_isonly = sig_q_isonly.astype("int8")
        except Exception as e:
            print("  IS-only cut failed: %s" % e)
            is_bins = None

    if is_bins is not None:
        sig_q_isonly_lagged = apply_publication_lag(sig_q_isonly, "daily")
        sig_q_isonly_lagged = sig_q_isonly_lagged[~sig_q_isonly_lagged.index.duplicated(keep="last")]
        aligned_isonly = sig_q_isonly_lagged.reindex(dates_dt).ffill()
        def_arr_g5_isonly = aligned_isonly.map(
            lambda s: HARD_MAP.get(int(s), 1.0) if pd.notna(s) else 1.0
        ).fillna(1.0).values
        def_arr_g5_isonly = np.clip(def_arr_g5_isonly.astype(float), 0.0, 2.0)

        print("  IS-only def_arr avg=%.4f  (full-sample avg=%.4f)"
              % (float(np.mean(def_arr_g5_isonly)), avg_lev_g5_full))

        pre_g5_isonly, aft_g5_isonly, _, r_g5_isonly, _ = _build_placebo_nav(
            shared, dates_dt, n_years, ret_gold, ret_bond,
            fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
            def_arr=def_arr_g5_isonly)
        g5_isonly_maxdd  = pre_g5_isonly["MaxDD_FULL"]
        delta_isonly_vs_base = (g5_isonly_maxdd - base_maxdd) * 100
        delta_isonly_vs_full = (g5_isonly_maxdd - g5_full_maxdd) * 100
        print("  G5_IS-only: MaxDD=%+.4f%%  delta_vs_B3a=%+.4fpp  delta_vs_fullsample=%+.4fpp"
              % (g5_isonly_maxdd * 100, delta_isonly_vs_base, delta_isonly_vs_full))
        la_bias_pp = delta_maxdd_g5_full - delta_isonly_vs_base
        print("  LOOK-AHEAD BIAS estimate = %+.4fpp" % la_bias_pp)
        print("  (Positive = full-sample overstates improvement vs IS-only)")
        if abs(la_bias_pp) < 0.5:
            print("  -> VERDICT A4: PASS (look-ahead bias < 0.5pp, negligible)")
        elif abs(la_bias_pp) < 1.0:
            print("  -> VERDICT A4: BORDERLINE (%.2fpp bias)" % la_bias_pp)
        else:
            print("  -> VERDICT A4: FAIL (%.2fpp look-ahead bias exceeds 1pp)" % la_bias_pp)

    # =========================================================================
    # PLACEBO P3: Uniform de-leverage (same avg lev reduction, no signal)
    # =========================================================================
    print("\n" + "=" * 100)
    print("PLACEBO P3: Uniform de-leverage (avg lev reduction = G5, no VIX signal)")
    print("  Q: Is G5 MaxDD improvement entirely explained by lower average leverage?")

    avg_mult_g5 = float(np.mean(def_arr_g5_fullsample))
    print("  G5 average def_mult = %.6f" % avg_mult_g5)
    print("  Applying uniform scalar %.6f to all dates ..." % avg_mult_g5)

    def_arr_uniform = np.full(n, avg_mult_g5, dtype=float)
    pre_uniform, aft_uniform, _, r_uniform, _ = _build_placebo_nav(
        shared, dates_dt, n_years, ret_gold, ret_bond,
        fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        def_arr=def_arr_uniform)
    uniform_maxdd  = pre_uniform["MaxDD_FULL"]
    delta_uniform_vs_base   = (uniform_maxdd - base_maxdd) * 100
    delta_g5_vs_uniform     = (g5_full_maxdd - uniform_maxdd) * 100

    print("  Uniform de-lev: MaxDD=%+.4f%%  delta_vs_B3a=%+.4fpp"
          % (uniform_maxdd * 100, delta_uniform_vs_base))
    print("  G5_full:        MaxDD=%+.4f%%  delta_vs_B3a=%+.4fpp"
          % (g5_full_maxdd * 100, delta_maxdd_g5_full))
    print("  Signal-specific increment (G5 - uniform): %+.4fpp"
          % delta_g5_vs_uniform)
    print("  Fraction of G5 MaxDD improvement from pure de-lev: %.1f%%"
          % (delta_uniform_vs_base / delta_maxdd_g5_full * 100 if abs(delta_maxdd_g5_full) > 1e-6 else float("nan")))
    if abs(delta_maxdd_g5_full) > 1e-6:
        signal_share = delta_g5_vs_uniform / delta_maxdd_g5_full
        print("  Fraction from signal-specific timing: %.1f%%" % (signal_share * 100))
        if signal_share > 0.3:
            print("  -> VERDICT P3: Signal has INCREMENTAL value beyond pure de-leverage")
        elif signal_share > 0:
            print("  -> VERDICT P3: BORDERLINE - signal adds modest increment beyond de-leverage")
        else:
            print("  -> VERDICT P3: FAIL - G5 improvement is dominated by pure de-leverage, not signal")

    # =========================================================================
    # PLACEBO P1: Block-shuffle quartile labels
    # =========================================================================
    print("\n" + "=" * 100)
    print("PLACEBO P1: Block-shuffle quartile labels (N=20 shuffles)")
    print("  Q: If vix_mom21 quartile labels are randomized, does MaxDD still improve?")

    rng = np.random.default_rng(SEED)
    n_shuffle = 20
    shuffle_maxdd_deltas = []

    # Get the original quartile series (before mapping to def_mult)
    raw_signal = macro_df["vix_mom21"].dropna()
    raw_signal.index = pd.DatetimeIndex(pd.to_datetime(raw_signal.index))
    sig_q_original = quantile_cut(raw_signal, levels=4)
    sig_q_original = sig_q_original.dropna().astype("int8")
    sig_q_lagged   = apply_publication_lag(sig_q_original, "daily")
    sig_q_lagged   = sig_q_lagged[~sig_q_lagged.index.duplicated(keep="last")]
    aligned_q      = sig_q_lagged.reindex(dates_dt).ffill()
    valid_q        = aligned_q.dropna()
    q_values       = valid_q.values.astype(int)
    q_index        = valid_q.index

    block_size = 21  # shuffle in 21-day blocks to preserve local structure
    n_q = len(q_values)
    n_blocks_q = int(np.ceil(n_q / block_size))

    for i in range(n_shuffle):
        # Block-shuffle: randomly permute block order
        block_order = rng.permutation(n_blocks_q)
        shuffled_vals = np.concatenate([
            q_values[b * block_size: (b + 1) * block_size]
            for b in block_order
        ])[:n_q]

        # Rebuild aligned def_arr from shuffled quartiles
        shuffled_series = pd.Series(shuffled_vals, index=q_index, dtype=int)
        shuffled_aligned = shuffled_series.reindex(dates_dt).ffill()
        def_arr_shuffled = shuffled_aligned.map(
            lambda s: HARD_MAP.get(int(s), 1.0) if pd.notna(s) else 1.0
        ).fillna(1.0).values.astype(float)
        def_arr_shuffled = np.clip(def_arr_shuffled, 0.0, 2.0)

        pre_shuf, _, _, _, _ = _build_placebo_nav(
            shared, dates_dt, n_years, ret_gold, ret_bond,
            fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
            def_arr=def_arr_shuffled)
        delta = (pre_shuf["MaxDD_FULL"] - base_maxdd) * 100
        shuffle_maxdd_deltas.append(delta)

    arr_deltas = np.array(shuffle_maxdd_deltas)
    print("  Shuffle MaxDD deltas (pp vs B3a base):")
    print("    min=%.3f  median=%.3f  max=%.3f  mean=%.3f  std=%.3f"
          % (arr_deltas.min(), np.median(arr_deltas), arr_deltas.max(),
             arr_deltas.mean(), arr_deltas.std()))
    print("  Reference G5 (original signal): delta=%.3fpp" % delta_maxdd_g5_full)
    print("  Fraction of shuffles beating G5: %.1f%%"
          % (100.0 * np.mean(arr_deltas >= delta_maxdd_g5_full)))

    # Empirical p-value: rank of true G5 among shuffles (one-sided, higher = more signal-like)
    p_shuffle = float(np.mean(arr_deltas >= delta_maxdd_g5_full))
    signal_avg = float(arr_deltas.mean())

    if signal_avg >= delta_maxdd_g5_full * 0.7 and p_shuffle > 0.3:
        print("  -> VERDICT P1: FAIL/CONCERN - shuffled avg close to real G5. "
              "Improvement is largely from map structure (average de-lev), not signal timing.")
    elif p_shuffle < 0.2:
        print("  -> VERDICT P1: PASS - original signal clearly outperforms shuffles.")
    else:
        print("  -> VERDICT P1: BORDERLINE - some share from de-lev, some from timing.")

    # =========================================================================
    # PLACEBO P2: Time-shift sensitivity (+5BD vs -5BD)
    # =========================================================================
    print("\n" + "=" * 100)
    print("PLACEBO P2: Time-shift sensitivity")
    print("  Shift vix_mom21 -5BD (older info, less predictive) vs +5BD (future leak)")

    for extra_shift in [-5, +5]:
        shift_label = "older5BD" if extra_shift < 0 else "future5BD"
        desc = "further into past (OLDER / less predictive)" if extra_shift < 0 else "FUTURE (look-ahead direction)"
        print("\n  -- %s (shift %+dBD) --" % (shift_label, extra_shift))

        # Build shifted series: apply extra lag
        from pandas.tseries.offsets import BusinessDay
        if extra_shift > 0:
            shifted_idx = sig_q_lagged.index + BusinessDay(extra_shift)
        else:
            shifted_idx = sig_q_lagged.index - BusinessDay(-extra_shift)

        sig_q_shifted = pd.Series(sig_q_lagged.values, index=shifted_idx)
        sig_q_shifted = sig_q_shifted[~sig_q_shifted.index.duplicated(keep="last")]
        aligned_shifted = sig_q_shifted.reindex(dates_dt).ffill()
        def_arr_shifted = aligned_shifted.map(
            lambda s: HARD_MAP.get(int(s), 1.0) if pd.notna(s) else 1.0
        ).fillna(1.0).values.astype(float)
        def_arr_shifted = np.clip(def_arr_shifted, 0.0, 2.0)

        print("    def_arr avg=%.4f (should be same as G5 avg=%.4f if only time-shift)"
              % (float(np.mean(def_arr_shifted)), avg_lev_g5_full))

        pre_shifted, aft_shifted, _, _, _ = _build_placebo_nav(
            shared, dates_dt, n_years, ret_gold, ret_bond,
            fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
            def_arr=def_arr_shifted)
        shifted_maxdd  = pre_shifted["MaxDD_FULL"]
        delta_shifted  = (shifted_maxdd - base_maxdd) * 100
        diff_vs_real   = delta_shifted - delta_maxdd_g5_full

        print("    MaxDD=%+.4f%%  delta_vs_B3a=%+.4fpp  diff_vs_G5_real=%+.4fpp  min9=%+.4f%%"
              % (shifted_maxdd * 100, delta_shifted, diff_vs_real,
                 _min_at(aft_shifted) * 100))

        if extra_shift > 0 and diff_vs_real > 0.5:
            print("    -> CONCERN: Future shift IMPROVES MaxDD delta by %.2fpp. Suggests look-ahead leak." % diff_vs_real)
        elif extra_shift > 0 and diff_vs_real <= 0.5:
            print("    -> OK: Future shift does NOT materially inflate MaxDD delta (diff=%.2fpp <= 0.5pp)." % diff_vs_real)
        elif extra_shift < 0 and diff_vs_real < -0.5:
            print("    -> CONCERN: Older signal REDUCES MaxDD delta by %.2fpp. Time proximity matters -> potential timing edge (or look-ahead)." % abs(diff_vs_real))
        elif extra_shift < 0:
            print("    -> OK: Older signal slightly reduces effectiveness (%.2fpp). Normal." % diff_vs_real)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("PLACEBO AUDIT SUMMARY")
    print("=" * 100)
    print("\nG5_vix_hard reference MaxDD improvement vs B3a: %+.4fpp" % delta_maxdd_g5_full)
    print()
    print("A4 (IS-only quantile boundary):")
    if is_bins is not None:
        print("   Full-sample delta: %+.4fpp" % delta_maxdd_g5_full)
        print("   IS-only delta:     %+.4fpp" % delta_isonly_vs_base)
        print("   Look-ahead bias:   %+.4fpp" % la_bias_pp)
    else:
        print("   IS-only quantile test could not be computed.")

    print()
    print("P3 (Uniform de-leverage, same avg mult):")
    print("   Uniform MaxDD delta:   %+.4fpp (%.1f%% of G5 improvement)"
          % (delta_uniform_vs_base,
             delta_uniform_vs_base / delta_maxdd_g5_full * 100 if abs(delta_maxdd_g5_full) > 1e-6 else float("nan")))
    print("   Signal-specific extra: %+.4fpp (%.1f%% of G5 improvement)"
          % (delta_g5_vs_uniform,
             delta_g5_vs_uniform / delta_maxdd_g5_full * 100 if abs(delta_maxdd_g5_full) > 1e-6 else float("nan")))

    print()
    print("P1 (Block-shuffle, %d shuffles):" % n_shuffle)
    print("   Shuffle avg MaxDD delta:  %+.4fpp (vs G5 real: %+.4fpp)"
          % (signal_avg, delta_maxdd_g5_full))
    print("   Fraction of shuffles >= G5: %.1f%%" % (p_shuffle * 100))

    print()
    print("OVERALL VERDICT:")
    print("  The key structural issue is:")
    print("  - Full-sample quantile cut in _build_def_mult_arr (quantize.py line 74-81)")
    print("  - window=None -> pd.qcut over full history -> OOS quartile boundaries")
    print("    are set using OOS data = LOOK-AHEAD BIAS in quartile assignment")
    print("  - The comment in combine_g5_defoverlay_20260615.py line 289 acknowledges")
    print("    this: 'full-sample cut is acceptable for a slowly-varying feature'")
    print("  - Magnitude of bias measured in A4 above.")
    print()
    print("  P3 separation: what fraction of improvement is 'smart timing' vs 'just de-lev'?")
    print("  See above for signal-specific increment.")
    print("  P1 shuffle: what fraction is 'distribution-of-map' vs 'signal timing'?")
    print("  See above for shuffle mean vs real G5.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
