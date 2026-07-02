"""
src/audit/methodology_power_20260702.py
========================================
Task 5 (P09C1/scale critical-verification plan, 2026-07-02): self-attack the
crisis-window timing methodology (EVALUATION_STANDARD SS4.4 R-STAT-1/2/3,
src/audit/crisis_window_timing_20260621.py, A7_DD_REDUCTION_VARIATIONS_20260621
SS8/SS9).

The prior finding ("all brakes TIMING_WEAK, 0/5 or 1/5 windows shallower than
their uniform-delever twin") was qualified as SUGGESTIVE (small n=5). This
script quantifies exactly how weak that n=5 sign test is, and whether the
"no evidence of timing skill" conclusion is being overstated.

Four checks, ALL using the SAME P09_C1 base path (r_strat, fund_active) as
a7dd_stage2_timing_20260621.py, byte-identically rebuilt:

(1) POWER ANALYSIS (most important). Inject an ORACLE brake into r_strat: on
    each stress window, starting k days after the window's first IN day,
    retreat an ADDITIONAL flat fraction Delta of IN-day exposure to cash (on
    top of whatever DH-W1 already does). This literally injects real, known
    timing skill (the oracle "knows" the crisis has started). Then run it
    through the exact same crisis_window_dd_compare + sign_test_brake_beats_twin
    machinery used for A7/B1/B2/B3/B4, and see how large Delta must be before
    the framework calls it TIMING_LIKELY/CONFIRMED instead of TIMING_WEAK.
    Grid: Delta in {0.05, 0.10, 0.20, 0.40, 0.80, 1.00}, k in {0, 5, 10}.
    Delta=0 must reproduce "0/5 shallower, all edges ~0" (self-test vs the
    known A7/B1 base facts: 4/5 windows have brake==twin because DH-W1 is
    already ~94-100% OUT there).

(2) WINDOW-DEFINITION SENSITIVITY. Shift each STRESS_WINDOWS start/end by
    +/-3 months (5 variants: base, start-3m, start+3m, end-3m, end+3m) and
    rerun the A7/B1 sign test to see if n_shallower / verdict changes.

(3) EQUAL-FBAR TWIN CONSTRUCTION BIAS. Rebuild the uniform-delever twin using
    3 different fbar measurements: (a) full-series IN-day mean (current
    default, measure_mean_in_leg_frac), (b) IS-period-only IN-day mean,
    (c) crisis-window-only IN-day mean. Recompute A7/B1 dd_edge_pp per window
    under each and check whether the SIGN (brake shallower/deeper than twin)
    is preserved.

(4) Print an explicit "how strong can the SS4.4 claim be" summary.

ASCII-only prints (Windows cp932 safe). Does NOT commit. No temp files.
Output: audit_results/methodology_power_20260702.csv
"""
from __future__ import annotations

import os
import sys
import types

# ---- multitasking stub (matches other audit scripts) -----------------------
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

import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.regime_labeler_20260611 import (
    build_regime_labels, stress_masks, STRESS_WINDOWS)

from src.audit.k365_recost_20260612 import (
    _build_full_c1, _build_tqqq_base_param, EXCESS_EXTRA_K365_CENTRE)
from src.audit.run_p09_tqqq_validate_20260611 import _maxdd_from_returns
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights, _apply_aftertax,
    _count_fund_transitions, LAG_DAYS, TRADING_DAYS)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY
from src.audit.out_fill_variants_20260620 import _build_out_fill_variant, alloc_base

from src.audit.dd_reduction_overlays_20260621 import (
    apply_param_vol_brake, apply_downside_dev_brake,
    measure_mean_in_leg_frac, build_uniform_delever, _blend_to_cash)

from src.audit.crisis_window_timing_20260621 import (
    crisis_window_dd_compare, sign_test_brake_beats_twin)

P09C1_V7_MAP = None
P09C1_LEV_SCALE = 1.0
P09C1_EXCESS_EXTRA = 0.0

# Prior project A7/B1 facts to self-test against (from
# audit_results/a7dd_stage2_crisis_timing_20260621.csv, 2026-06-21)
PRIOR_A7_FBAR = 0.021221988623314446
PRIOR_A7_NSHALLOWER = 0
PRIOR_B1_FBAR = 0.013422869758223521
PRIOR_B1_NSHALLOWER = 0
FBAR_TOL = 0.0005


def _verdict(nW, nS, p):
    if nW > 0 and nS == nW:
        return "TIMING_CONFIRMED"
    elif nW > 0 and nS >= int(np.ceil(nW * 0.6)) and p < 0.20:
        return "TIMING_LIKELY"
    else:
        return "TIMING_WEAK"


def build_base():
    """Byte-identical rebuild of P09_C1 base (r_strat, fund_active, sofr_arr,
    dates_dt, stress masks, regime labels) per a7dd_stage2_timing_20260621.py."""
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252_raw = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252_raw), False, bond_m252_raw > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    _, r_base_in, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=P09C1_V7_MAP, lev_scale=P09C1_LEV_SCALE,
        excess_extra=P09C1_EXCESS_EXTRA)
    nav0, r_strat, eff0 = _build_out_fill_variant(
        r_base_in, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    tpy0 = tpy_base + _count_fund_transitions(eff0) / n_years

    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)

    return dict(dates_dt=dates_dt, n=n, n_years=n_years, r_strat=r_strat,
                fund_active=fund_active, sofr_arr=sofr_arr, tpy0=tpy0,
                regimes=regimes, stress=stress)


# =============================================================================
# (1) POWER ANALYSIS: oracle brake injection
# =============================================================================
def apply_oracle_brake(r_strat, fund_active, sofr_arr, dates_dt, window_mask,
                       delta, k_days):
    """Inject a KNOWN, real timing brake: within `window_mask` (a stress
    window), on IN days (fund_active==False) that are >= k_days after the
    window's first day, retreat an ADDITIONAL flat fraction `delta` of
    exposure to SOFR cash (stacked on top of whatever the base path already
    is -- the base path IS the "unbraked" P09_C1, so this is a pure additive
    injection, not a modification of an existing brake).

    This is an ORACLE: it "knows" in advance a crisis window is running (does
    not use rolling vol/dd like A7/B1), so it isolates PURE detection power of
    the sign-test framework, not signal quality. If even an oracle at delta=1.0
    (retreat everything) cannot be detected reliably in some windows, that is
    a structural limit of the framework in THOSE windows (no IN days to act
    on), not a weakness of delta calibration.
    """
    r_strat = np.asarray(r_strat, float)
    fund_active = np.asarray(fund_active, dtype=bool)
    in_day = ~fund_active
    wmask = np.asarray(window_mask, bool)
    idx_in_window = np.where(wmask)[0]
    frac = np.zeros(len(r_strat), float)
    if idx_in_window.size > 0:
        first_day = idx_in_window.min()
        eligible = wmask & in_day & (np.arange(len(r_strat)) >= first_day + k_days)
        frac[eligible] = delta
    r_oracle = _blend_to_cash(r_strat, fund_active, sofr_arr, frac)
    return r_oracle, frac


def power_analysis(base):
    dates_dt = base["dates_dt"]
    r_strat = base["r_strat"]
    fund_active = base["fund_active"]
    sofr_arr = base["sofr_arr"]
    stress = base["stress"]

    DELTAS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.00]
    KS = [0, 5, 10]

    rows = []
    for delta in DELTAS:
        for k in KS:
            if delta == 0.0 and k != 0:
                continue  # delta=0 is a no-op regardless of k; run once
            per_window_frac_days = {}
            r_oracle = np.asarray(r_strat, float).copy()
            total_frac = np.zeros(len(r_strat), float)
            # Inject independently into EACH window (frac arrays don't overlap
            # across the 5 non-overlapping STRESS_WINDOWS)
            for name, mask in stress.items():
                r_oracle, frac_w = apply_oracle_brake(
                    r_oracle, fund_active, sofr_arr, dates_dt, mask, delta, k)
                total_frac += frac_w
                per_window_frac_days[name] = int((frac_w > 0).sum())

            fbar = measure_mean_in_leg_frac(r_strat, r_oracle, fund_active, sofr_arr)
            r_uni = build_uniform_delever(r_strat, fund_active, sofr_arr, fbar)
            win = crisis_window_dd_compare(r_oracle, r_uni, stress)
            st = sign_test_brake_beats_twin(win)
            verdict = _verdict(st["n_windows"], st["n_shallower"], st["binom_p_onesided"])

            maxdd_oracle = _maxdd_from_returns(r_oracle)
            maxdd_base = _maxdd_from_returns(r_strat)

            row = {
                "delta": delta, "k_days": k, "fbar": fbar,
                "n_windows": st["n_windows"], "n_shallower": st["n_shallower"],
                "n_deeper": st["n_deeper"],
                "binom_p_onesided": st["binom_p_onesided"],
                "mean_dd_edge_pp": st["mean_dd_edge_pp"],
                "verdict": verdict,
                "type2_error": bool(delta > 0.0 and verdict == "TIMING_WEAK"),
                "maxdd_oracle": maxdd_oracle, "maxdd_base": maxdd_base,
                "maxdd_improve_pp": (maxdd_oracle - maxdd_base) * 100.0,
            }
            for name in stress:
                row["actiondays_" + name] = per_window_frac_days[name]
            rows.append(row)
            print("  delta=%.2f k=%2d  fbar=%.4f  shallower=%d/%d  p=%.4f  "
                  "mean_edge=%+.3fpp  MaxDD_improve=%+.3fpp  -> %s%s"
                  % (delta, k, fbar, st["n_shallower"], st["n_windows"],
                     st["binom_p_onesided"], st["mean_dd_edge_pp"],
                     row["maxdd_improve_pp"],
                     verdict, "  [TYPE-II]" if row["type2_error"] else ""))

    df = pd.DataFrame(rows)

    # self-test: delta=0 must reproduce "0/5 shallower, edges ~0" fact pattern
    base_row = df[(df["delta"] == 0.0)].iloc[0]
    st_ok = (base_row["n_shallower"] == 0 and base_row["n_windows"] == 5
             and base_row["verdict"] == "TIMING_WEAK")
    print("\n  SELF-TEST delta=0 reproduces base TIMING_WEAK/0/5: %s"
          % ("PASS" if st_ok else "FAIL"))
    return df, st_ok


# =============================================================================
# (2) WINDOW-DEFINITION SENSITIVITY (+/- 3 months)
# =============================================================================
def shifted_stress_masks(dates_dt, months):
    out = {}
    for name, (a, b) in STRESS_WINDOWS.items():
        lo = pd.Timestamp(a) + pd.DateOffset(months=months)
        hi = pd.Timestamp(b) + pd.DateOffset(months=months)
        out[name] = (dates_dt >= lo) & (dates_dt <= hi)
    return out


def window_sensitivity(base):
    dates_dt = base["dates_dt"]
    r_strat = base["r_strat"]
    fund_active = base["fund_active"]
    sofr_arr = base["sofr_arr"]

    BRAKES = {
        "A7_REPRODUCE": apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.5),
        "B1_DOWNSIDE_DEV": apply_downside_dev_brake(r_strat, fund_active, sofr_arr, 0.20, 63, 0.5),
    }

    SHIFTS = {"base_0m": 0, "start_minus_3m": -3, "start_plus_3m": 3,
              "end_minus_3m": -3, "end_plus_3m": 3}
    # For start/end-only shifts we shift the whole window (both endpoints);
    # a start-only or end-only shift changes window LENGTH, which is the
    # more informative variant for edge/inclusion sensitivity. Implement both:
    rows = []
    for label, r_b in BRAKES.items():
        fbar = measure_mean_in_leg_frac(r_strat, r_b, fund_active, sofr_arr)
        r_uni = build_uniform_delever(r_strat, fund_active, sofr_arr, fbar)

        variants = {}
        variants["base"] = stress_masks(dates_dt)
        variants["shift_all_minus3m"] = shifted_stress_masks(dates_dt, -3)
        variants["shift_all_plus3m"] = shifted_stress_masks(dates_dt, 3)

        # start-only / end-only variants (window length changes)
        def _start_end_shift(months_start, months_end):
            out = {}
            for name, (a, b) in STRESS_WINDOWS.items():
                lo = pd.Timestamp(a) + pd.DateOffset(months=months_start)
                hi = pd.Timestamp(b) + pd.DateOffset(months=months_end)
                if lo > hi:
                    lo, hi = hi, lo
                out[name] = (dates_dt >= lo) & (dates_dt <= hi)
            return out

        variants["start_minus3m_only"] = _start_end_shift(-3, 0)
        variants["start_plus3m_only"] = _start_end_shift(3, 0)
        variants["end_minus3m_only"] = _start_end_shift(0, -3)
        variants["end_plus3m_only"] = _start_end_shift(0, 3)

        for vname, stress_v in variants.items():
            win = crisis_window_dd_compare(r_b, r_uni, stress_v)
            st = sign_test_brake_beats_twin(win)
            verdict = _verdict(st["n_windows"], st["n_shallower"], st["binom_p_onesided"])
            rows.append({
                "label": label, "variant": vname, "fbar": fbar,
                "n_windows": st["n_windows"], "n_shallower": st["n_shallower"],
                "binom_p_onesided": st["binom_p_onesided"],
                "mean_dd_edge_pp": st["mean_dd_edge_pp"], "verdict": verdict,
            })
            print("  %-16s %-20s shallower=%d/%d  p=%.4f  mean_edge=%+.3fpp -> %s"
                  % (label, vname, st["n_shallower"], st["n_windows"],
                     st["binom_p_onesided"], st["mean_dd_edge_pp"], verdict))

    return pd.DataFrame(rows)


# =============================================================================
# (3) EQUAL-FBAR TWIN CONSTRUCTION BIAS (3 fbar definitions)
# =============================================================================
def fbar_definitions(r_strat, r_braked, fund_active, sofr_arr, dates_dt, stress):
    """Return {def_name: fbar} using 3 different averaging windows."""
    r_strat = np.asarray(r_strat, float)
    r_braked = np.asarray(r_braked, float)
    sofr_arr = np.asarray(sofr_arr, float)
    in_day = ~np.asarray(fund_active, dtype=bool)
    denom = r_strat - sofr_arr

    def _fbar_over(mask):
        usable = in_day & mask & (np.abs(denom) > 1e-9)
        if usable.sum() == 0:
            return 0.0
        f = (r_strat[usable] - r_braked[usable]) / denom[usable]
        f = np.clip(f, 0.0, 1.0)
        return float(f.mean())

    full_mask = np.ones(len(r_strat), bool)
    is_mask = np.asarray(dates_dt <= IS_END)
    crisis_mask = np.zeros(len(r_strat), bool)
    for m in stress.values():
        crisis_mask |= np.asarray(m, bool)

    return {
        "full_series": _fbar_over(full_mask),
        "is_only": _fbar_over(is_mask),
        "crisis_window_only": _fbar_over(crisis_mask),
    }


def fbar_bias_check(base):
    dates_dt = base["dates_dt"]
    r_strat = base["r_strat"]
    fund_active = base["fund_active"]
    sofr_arr = base["sofr_arr"]
    stress = base["stress"]

    BRAKES = {
        "A7_REPRODUCE": apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.5),
        "B1_DOWNSIDE_DEV": apply_downside_dev_brake(r_strat, fund_active, sofr_arr, 0.20, 63, 0.5),
    }

    rows = []
    for label, r_b in BRAKES.items():
        fbars = fbar_definitions(r_strat, r_b, fund_active, sofr_arr, dates_dt, stress)
        for def_name, fbar in fbars.items():
            r_uni = build_uniform_delever(r_strat, fund_active, sofr_arr, fbar)
            win = crisis_window_dd_compare(r_b, r_uni, stress)
            st = sign_test_brake_beats_twin(win)
            verdict = _verdict(st["n_windows"], st["n_shallower"], st["binom_p_onesided"])
            rows.append({
                "label": label, "fbar_definition": def_name, "fbar": fbar,
                "n_windows": st["n_windows"], "n_shallower": st["n_shallower"],
                "binom_p_onesided": st["binom_p_onesided"],
                "mean_dd_edge_pp": st["mean_dd_edge_pp"], "verdict": verdict,
            })
            print("  %-16s %-20s fbar=%.4f  shallower=%d/%d  mean_edge=%+.3fpp -> %s"
                  % (label, def_name, fbar, st["n_shallower"], st["n_windows"],
                     st["mean_dd_edge_pp"], verdict))

    return pd.DataFrame(rows)


def main():
    print("=" * 120)
    print("METHODOLOGY POWER SELF-ATTACK (Task 5, P09C1/scale critical-verification) 2026-07-02")
    print("Targets: EVALUATION_STANDARD SS4.4 (R-STAT-1/2/3), crisis_window_timing_20260621.py,")
    print("A7_DD_REDUCTION_VARIATIONS_20260621 SS8/SS9")
    print("=" * 120)

    print("\nBuilding P09_C1 base path (byte-identical to a7dd_stage2_timing_20260621.py) ...")
    base = build_base()
    print("  n_days=%d  n_years=%.3f  IN-day frac=%.3f"
          % (base["n"], base["n_years"], float((~base["fund_active"]).mean())))
    for name, m in base["stress"].items():
        n_in = int((np.asarray(m, bool) & (~base["fund_active"])).sum())
        n_tot = int(np.asarray(m, bool).sum())
        print("  window %-12s: %4d days total, %4d IN-days (%.1f%%)"
              % (name, n_tot, n_in, 100.0 * n_in / max(n_tot, 1)))

    print("\n" + "=" * 120)
    print("(1) POWER ANALYSIS: oracle-brake injection (delta x k_days)")
    print("=" * 120)
    df_power, selftest_ok = power_analysis(base)

    print("\n" + "=" * 120)
    print("(2) WINDOW-DEFINITION SENSITIVITY (+/- 3 months)")
    print("=" * 120)
    df_window = window_sensitivity(base)

    print("\n" + "=" * 120)
    print("(3) EQUAL-FBAR TWIN CONSTRUCTION BIAS (3 fbar definitions)")
    print("=" * 120)
    df_fbar = fbar_bias_check(base)

    # -------------------------------------------------------------------
    # Save combined CSV
    # -------------------------------------------------------------------
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "methodology_power_20260702.csv")

    df_power2 = df_power.copy()
    df_power2["check"] = "1_power_analysis"
    df_window2 = df_window.copy()
    df_window2["check"] = "2_window_sensitivity"
    df_fbar2 = df_fbar.copy()
    df_fbar2["check"] = "3_fbar_definitions"

    combined = pd.concat([df_power2, df_window2, df_fbar2], axis=0, ignore_index=True, sort=False)
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nSaved combined CSV: %s (%d rows)" % (csv_path, len(combined)))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    detected = df_power[(df_power["delta"] > 0) & (df_power["verdict"] != "TIMING_WEAK")]
    min_delta_detected = detected["delta"].min() if len(detected) else None
    all_type2_at_100pct = bool((df_power[df_power["delta"] == 1.00]["verdict"] == "TIMING_WEAK").all())

    print("  Self-test (delta=0 reproduces prior A7/B1 base facts): %s"
          % ("PASS" if selftest_ok else "FAIL"))
    print("  Minimum delta at which framework detects (verdict != TIMING_WEAK): %s"
          % (("%.2f" % min_delta_detected) if min_delta_detected is not None else "NONE (never detected, even at delta=1.00)"))
    print("  At delta=1.00 (retreat to 100%% cash for whole window from day k): all k TIMING_WEAK? %s"
          % all_type2_at_100pct)

    n_type2 = int(df_power["type2_error"].sum())
    n_injected = int((df_power["delta"] > 0).sum())
    print("  Type-II rate across all delta>0 x k combos: %d/%d" % (n_type2, n_injected))

    base_verdict_by_label = (df_window[df_window["variant"] == "base"]
                              .set_index("label")["verdict"].to_dict())
    non_base = df_window[df_window["variant"] != "base"]
    n_changed = int(sum(
        1 for _, r in non_base.iterrows()
        if r["verdict"] != base_verdict_by_label.get(r["label"])
    ))
    print("  Window-shift variants that changed the verdict (vs base): %d / %d"
          % (n_changed, len(non_base)))

    fbar_sign_flips = 0
    for label in df_fbar["label"].unique():
        sub = df_fbar[df_fbar["label"] == label]
        signs = np.sign(sub["mean_dd_edge_pp"].values)
        if len(set(signs)) > 1:
            fbar_sign_flips += 1
    print("  Labels where mean_dd_edge_pp SIGN flips across the 3 fbar definitions: %d / %d"
          % (fbar_sign_flips, df_fbar["label"].nunique()))

    print("\nDone.")
    return {
        "selftest_ok": selftest_ok,
        "min_delta_detected": min_delta_detected,
        "all_type2_at_delta1": all_type2_at_100pct,
        "type2_rate": (n_type2, n_injected),
        "window_verdict_changes": (n_changed, len(non_base)),
        "fbar_sign_flips": (fbar_sign_flips, int(df_fbar["label"].nunique())),
        "csv": csv_path,
    }


if __name__ == "__main__":
    main()
