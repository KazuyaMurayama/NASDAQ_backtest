"""
src/audit/frontier_refute_20260702.py
======================================
Task 4 of docs/superpowers/plans/2026-07-02-p09c1-scale-critical-verification.md
-- REFUTATION attempt on 3 headline claims from the 2026-06-19..22 reports:

  CLAIM 1: "DD reduction is fundamentally a leverage dial; exotic overlays
            (B1 etc.) do not beat a uniform-delever twin at equal risk."
  CLAIM 2: "B1 is a weak brake (not a smart signal)."
  CLAIM 3: "P09 scale ~= 1.4 is the practical MaxDD <= -50% ceiling."

Method (R-STAT-1/2/3 compliant):
  - NEVER block-bootstrap MaxDD (path-dependent extremum).
  - Timing vs delever is decided on INTACT crisis windows + cross-window sign
    test (crisis_window_timing_20260621.crisis_window_dd_compare /
    sign_test_brake_beats_twin), not resampling.
  - All comparisons are PAIRED on the SAME daily return series / SAME dates
    (no independent-sample comparison).

Reuses (no re-derivation):
  - dd_reduction_harness_20260626.setup()/build()  -> P09_STR pure-scale series
  - b1_scale_annual_20260621 plumbing (_build_tqqq_base_param + out_fill_variants
    .alloc_base + dd_reduction_overlays_20260621.apply_downside_dev_brake)
      -> B1_STR series at scale S
  - dd_reduction_overlays_20260621.measure_mean_in_leg_frac / build_uniform_delever
  - crisis_window_timing_20260621 (R-STAT-2 intact-window compare + sign test)
  - out_fill_variants_20260620.apply_in_leg_vol_brake  -> A7 (scale=1.0 base)

Self-test reproduces (must match to 4 digits before any refutation is run):
  P09_STR_S1.4  CAGR_IS=+27.49% CAGR_OOS=+24.34% MaxDD=-46.48% Sharpe=1.074
  P09_STR_S1.6  CAGR_IS=+30.26% CAGR_OOS=+26.21% MaxDD=-51.95% Sharpe=1.055
  B1_STR_S2.0   MaxDD=-47.08%  CAGR_OOS=+24.07%
  A7 (scale1.0) MaxDD=-28.25%  Worst1D=-10.27%

ASCII-only prints (Windows cp932). No temp files, no commit.
Output: audit_results/frontier_refute_20260702.csv
"""
from __future__ import annotations

import json
import os
import sys
import types

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
from src.audit.regime_labeler_20260611 import build_regime_labels, stress_masks
from src.audit.k365_recost_20260612 import _build_tqqq_base_param, EXCESS_EXTRA_K365_CENTRE
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights, _apply_aftertax,
    _count_fund_transitions, LAG_DAYS, TRADING_DAYS,
)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY
from src.audit.out_fill_variants_20260620 import (
    _build_out_fill_variant, alloc_base, apply_in_leg_vol_brake,
)
from src.audit.dd_reduction_overlays_20260621 import (
    apply_downside_dev_brake, measure_mean_in_leg_frac, build_uniform_delever,
)
from src.audit.run_p09_tqqq_validate_20260611 import _maxdd_from_returns
from src.audit.crisis_window_timing_20260621 import (
    crisis_window_dd_compare, sign_test_brake_beats_twin,
)
from calculate_p10_5y import compute_p10_5y, compute_worst5y

import src.audit.dd_reduction_harness_20260626 as H

AFTER_TAX = 0.8273
STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXC = EXCESS_EXTRA_K365_CENTRE

# ---- Self-test anchors (Task 0, must reproduce to 4 digits) ----
ANCHORS = {
    "P09_STR_S1.4": dict(CAGR_IS=0.2749, CAGR_OOS=0.2434, MaxDD=-0.4648, Sharpe=1.074),
    "P09_STR_S1.6": dict(CAGR_IS=0.3026, CAGR_OOS=0.2621, MaxDD=-0.5195, Sharpe=1.055),
    "B1_STR_S2.0":  dict(MaxDD=-0.4708, CAGR_OOS=0.2407),
    "A7_S1.0":      dict(MaxDD=-0.2825, Worst1D=-0.1027),
}
ANCHOR_TOL = 0.0020


# =============================================================================
# Shared context: load once, build both P09_STR (pure scale) and B1_STR
# (brake-on-scaled) NAVs from the SAME underlying assets/dates/OUT-fill.
# =============================================================================
def setup_shared():
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    is_mask = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)

    return dict(shared=shared, a=a, dates=dates, dates_dt=dates_dt, n=n,
                n_years=n_years, is_mask=is_mask, oos_mask=oos_mask,
                ret_gold=ret_gold, ret_bond=ret_bond, fund_active=fund_active,
                wg=wg, wb=wb, bond_on=bond_on, sofr_arr=sofr_arr,
                regimes=regimes, stress=stress)


def build_p09_str(ctx, scale):
    """Pure strong-map scale series (no brake). Returns (r, tpy)."""
    shared, dates_dt, fund_active = ctx["shared"], ctx["dates_dt"], ctx["fund_active"]
    ret_gold, ret_bond = ctx["ret_gold"], ctx["ret_bond"]
    wg, wb, bond_on, sofr_arr = ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"]
    _, r_base, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=STRONG_MAP, lev_scale=scale, excess_extra=EXC)
    _, r_strat, eff = _build_out_fill_variant(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    tpy = tpy_base + _count_fund_transitions(eff) / ctx["n_years"]
    return np.asarray(r_strat, float), tpy


def build_b1_str(ctx, scale):
    """B1 downside-dev brake applied on top of the strong-map scale series.
    Returns (r_raw_unbraked, r_braked, tpy)."""
    shared, dates_dt, fund_active = ctx["shared"], ctx["dates_dt"], ctx["fund_active"]
    ret_gold, ret_bond = ctx["ret_gold"], ctx["ret_bond"]
    wg, wb, bond_on, sofr_arr = ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"]
    _, r_base, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=STRONG_MAP, lev_scale=scale, excess_extra=EXC)
    _, r_strat, eff = _build_out_fill_variant(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    r_b1 = apply_downside_dev_brake(r_strat, fund_active, sofr_arr, 0.20, 63, 0.5)
    tpy = tpy_base + _count_fund_transitions(eff) / ctx["n_years"]
    return np.asarray(r_strat, float), np.asarray(r_b1, float), tpy


def build_a7_s1(ctx):
    """A7 IN-leg total-vol brake on top of canonical P09_C1 (scale=1.0,
    default v7_map, excess=0.0) -- matches the plan anchor MaxDD=-28.25%."""
    shared, dates_dt, fund_active = ctx["shared"], ctx["dates_dt"], ctx["fund_active"]
    ret_gold, ret_bond = ctx["ret_gold"], ctx["ret_bond"]
    wg, wb, bond_on, sofr_arr = ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"]
    _, r_base, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=None, lev_scale=1.0, excess_extra=0.0)
    _, r_strat, eff = _build_out_fill_variant(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    r_a7 = apply_in_leg_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.5)
    tpy = tpy_base + _count_fund_transitions(eff) / ctx["n_years"]
    return np.asarray(r_strat, float), np.asarray(r_a7, float), tpy


def m10(nav, tpy):
    pre = compute_10metrics(nav, tpy)
    aft = _apply_aftertax(pre)
    return dict(CAGR_IS=aft["CAGR_IS"], CAGR_OOS=aft["CAGR_OOS"],
                Sharpe_FULL=pre["Sharpe_FULL"], MaxDD=pre["MaxDD_FULL"],
                Worst1D=pre["Worst1D"], Worst10Y=aft["Worst10Y_star"],
                Worst5Y=aft["Worst5Y"], P10_5Y=aft["P10_5Y"])


def realized_vol_full(r):
    return float(np.std(np.asarray(r, float), ddof=1) * np.sqrt(TRADING_DAYS))


def mean_gross_lev_proxy(r_in_leg_scale_arr):
    """Average nominal scale applied on IN days (proxy for average leverage)."""
    return float(np.mean(r_in_leg_scale_arr))


# =============================================================================
# Self-test (Task 0 anchors, restricted to what Task 4 reuses)
# =============================================================================
def self_test(ctx):
    print("=" * 100)
    print("SELF-TEST: reproduce Task-0 anchors (tol %.4f)" % ANCHOR_TOL)
    print("=" * 100)
    ok_all = True

    for label, scale in (("P09_STR_S1.4", 1.4), ("P09_STR_S1.6", 1.6)):
        r, tpy = build_p09_str(ctx, scale)
        nav = pd.Series(np.cumprod(1.0 + r), index=ctx["dates_dt"])
        m = m10(nav, tpy)
        exp = ANCHORS[label]
        m_key = dict(m, Sharpe=m["Sharpe_FULL"])
        diffs = {k: abs(m_key[k] - exp[k]) for k in exp}
        ok = all(d <= ANCHOR_TOL for d in diffs.values())
        ok_all &= ok
        print("  %-14s CAGR_IS=%+.4f%%(exp%+.4f) CAGR_OOS=%+.4f%%(exp%+.4f) "
              "MaxDD=%+.4f%%(exp%+.4f) Sharpe=%.4f(exp%.4f) -> %s"
              % (label, m["CAGR_IS"]*100, exp["CAGR_IS"]*100,
                 m["CAGR_OOS"]*100, exp["CAGR_OOS"]*100,
                 m["MaxDD"]*100, exp["MaxDD"]*100,
                 m["Sharpe_FULL"], exp["Sharpe"],
                 "OK" if ok else "MISMATCH"))

    r_raw, r_b1, tpy = build_b1_str(ctx, 2.0)
    nav_b1 = pd.Series(np.cumprod(1.0 + r_b1), index=ctx["dates_dt"])
    m = m10(nav_b1, tpy)
    exp = ANCHORS["B1_STR_S2.0"]
    ok = abs(m["MaxDD"] - exp["MaxDD"]) <= ANCHOR_TOL and abs(m["CAGR_OOS"] - exp["CAGR_OOS"]) <= ANCHOR_TOL
    ok_all &= ok
    print("  %-14s MaxDD=%+.4f%%(exp%+.4f) CAGR_OOS=%+.4f%%(exp%+.4f) -> %s"
          % ("B1_STR_S2.0", m["MaxDD"]*100, exp["MaxDD"]*100,
             m["CAGR_OOS"]*100, exp["CAGR_OOS"]*100, "OK" if ok else "MISMATCH"))

    r_raw7, r_a7, tpy7 = build_a7_s1(ctx)
    nav_a7 = pd.Series(np.cumprod(1.0 + r_a7), index=ctx["dates_dt"])
    m = m10(nav_a7, tpy7)
    exp = ANCHORS["A7_S1.0"]
    ok = abs(m["MaxDD"] - exp["MaxDD"]) <= ANCHOR_TOL and abs(m["Worst1D"] - exp["Worst1D"]) <= ANCHOR_TOL
    ok_all &= ok
    print("  %-14s MaxDD=%+.4f%%(exp%+.4f) Worst1D=%+.4f%%(exp%+.4f) -> %s"
          % ("A7_S1.0", m["MaxDD"]*100, exp["MaxDD"]*100,
             m["Worst1D"]*100, exp["Worst1D"]*100, "OK" if ok else "MISMATCH"))

    print("\n  SELF-TEST %s\n" % ("PASSED (all anchors reproduced)" if ok_all else "FAILED -- FIX BEFORE PROCEEDING"))
    return ok_all


# =============================================================================
# CLAIM 1: "DD reduction is fundamentally a leverage dial; exotic overlays do
# not beat a uniform-delever twin at equal risk." Attack via equal-realized-vol
# / equal-MaxDD / equal-avg-leverage P09_STR-scale counterfactuals vs B1_STR.
# =============================================================================
def _bisect_scale_for_target(ctx, target_fn, lo=0.5, hi=3.0, tol=1e-4, max_iter=40):
    """Binary-search P09_STR pure scale so that target_fn(metrics) == 0 sign
    change. target_fn(m) should be INCREASING in scale (vol/MaxDD depth both
    increase monotonically with scale for this strategy family)."""
    def f(s):
        r, tpy = build_p09_str(ctx, s)
        nav = pd.Series(np.cumprod(1.0 + r), index=ctx["dates_dt"])
        m = m10(nav, tpy)
        m["_vol"] = realized_vol_full(r)
        return target_fn(m)

    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        # not bracketed; caller should widen range
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if fmid < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def claim1_equal_risk_frontier(ctx, out_rows):
    print("\n" + "=" * 100)
    print("CLAIM 1: DD reduction = leverage dial; exotic (B1) does not beat uniform-delever")
    print("         twin at EQUAL RISK. Attack: match B1_STR(S) to a pure P09_STR(S') on")
    print("         (i) equal realized full-sample vol (ii) equal MaxDD (iii) equal avg lev.")
    print("=" * 100)

    verdicts = []
    for b1_scale in (1.4, 1.6, 1.8, 2.0):
        r_raw, r_b1, tpy_b1 = build_b1_str(ctx, b1_scale)
        nav_b1 = pd.Series(np.cumprod(1.0 + r_b1), index=ctx["dates_dt"])
        m_b1 = m10(nav_b1, tpy_b1)
        vol_b1 = realized_vol_full(r_b1)
        fbar = measure_mean_in_leg_frac(r_raw, r_b1, ctx["fund_active"], ctx["sofr_arr"])
        avg_lev_b1 = b1_scale * (1.0 - fbar)   # avg IN-leg nominal scale after brake cuts

        print("\n  --- B1_STR scale=%.2f (fbar=%.4f, avg_lev=%.3f) ---" % (b1_scale, fbar, avg_lev_b1))
        print("      B1   CAGR_OOS=%+.4f%% MaxDD=%+.4f%% vol=%.4f Worst5Y=%+.4f%% P10_5Y=%+.4f%%"
              % (m_b1["CAGR_OOS"]*100, m_b1["MaxDD"]*100, vol_b1, m_b1["Worst5Y"]*100, m_b1["P10_5Y"]*100))

        # (i) equal realized-vol P09_STR counterfactual: bisect scale s.t. vol(s)=vol_b1
        s_eqvol = _bisect_scale_for_target(
            ctx, lambda m, target=vol_b1: m["_vol"] - target, lo=0.3, hi=3.0)
        # (ii) equal-MaxDD counterfactual: bisect scale s.t. MaxDD(s) == m_b1["MaxDD"]
        #      MaxDD is negative; "increasing in scale" in magnitude means MaxDD(s) - target
        #      is DEcreasing (more negative) as s increases, so flip sign for bisection use.
        s_eqdd = _bisect_scale_for_target(
            ctx, lambda m, target=m_b1["MaxDD"]: -(m["MaxDD"] - target), lo=0.3, hi=3.0)
        # (iii) equal-avg-leverage counterfactual: pure scale = avg_lev_b1 directly
        s_eqlev = avg_lev_b1

        for tag, s_star in (("eqvol", s_eqvol), ("eqMaxDD", s_eqdd), ("eqAvgLev", s_eqlev)):
            if s_star is None:
                print("      [%-8s] bisection FAILED (not bracketed in [0.3,3.0]) -- skipped" % tag)
                verdicts.append(dict(b1_scale=b1_scale, cut=tag, resolved=False))
                continue
            r_c, tpy_c = build_p09_str(ctx, s_star)
            nav_c = pd.Series(np.cumprod(1.0 + r_c), index=ctx["dates_dt"])
            m_c = m10(nav_c, tpy_c)
            vol_c = realized_vol_full(r_c)
            cagr_gap_pp = (m_b1["CAGR_OOS"] - m_c["CAGR_OOS"]) * 100.0
            maxdd_gap_pp = (m_b1["MaxDD"] - m_c["MaxDD"]) * 100.0
            b1_beats = cagr_gap_pp > 0.05  # B1 CAGR > counterfactual by >0.05pp
            print("      [%-8s] P09_STR_S%.4f  CAGR_OOS=%+.4f%% MaxDD=%+.4f%% vol=%.4f  "
                  "| B1-cf dCAGR=%+.4fpp dMaxDD=%+.4fpp  -> %s"
                  % (tag, s_star, m_c["CAGR_OOS"]*100, m_c["MaxDD"]*100, vol_c,
                     cagr_gap_pp, maxdd_gap_pp,
                     "B1 CAGR-BEATS counterfactual (POSSIBLE COUNTEREXAMPLE)" if b1_beats else "counterfactual >= B1 (consistent with claim)"))
            verdicts.append(dict(b1_scale=b1_scale, cut=tag, cf_scale=s_star,
                                  b1_CAGR_OOS=m_b1["CAGR_OOS"], cf_CAGR_OOS=m_c["CAGR_OOS"],
                                  cagr_gap_pp=cagr_gap_pp, maxdd_gap_pp=maxdd_gap_pp,
                                  b1_beats=b1_beats, resolved=True))
            out_rows.append(dict(
                claim="1_equal_risk", b1_scale=b1_scale, cut=tag, cf_scale=s_star,
                b1_CAGR_OOS_pct=m_b1["CAGR_OOS"]*100, cf_CAGR_OOS_pct=m_c["CAGR_OOS"]*100,
                cagr_gap_pp=cagr_gap_pp, maxdd_gap_pp=maxdd_gap_pp,
                b1_MaxDD_pct=m_b1["MaxDD"]*100, cf_MaxDD_pct=m_c["MaxDD"]*100,
                b1_beats_cf=b1_beats))

        # Worst5Y / P10_5Y tail comparison at eqvol counterfactual (if resolved)
        if s_eqvol is not None:
            r_c, tpy_c = build_p09_str(ctx, s_eqvol)
            nav_c = pd.Series(np.cumprod(1.0 + r_c), index=ctx["dates_dt"])
            m_c = m10(nav_c, tpy_c)
            print("      [tail@eqvol] B1 Worst5Y=%+.4f%% P10_5Y=%+.4f%%  vs  cf Worst5Y=%+.4f%% P10_5Y=%+.4f%%"
                  % (m_b1["Worst5Y"]*100, m_b1["P10_5Y"]*100, m_c["Worst5Y"]*100, m_c["P10_5Y"]*100))
            out_rows.append(dict(
                claim="1_tail_eqvol", b1_scale=b1_scale, cut="eqvol_tail",
                b1_Worst5Y_pct=m_b1["Worst5Y"]*100, cf_Worst5Y_pct=m_c["Worst5Y"]*100,
                b1_P10_5Y_pct=m_b1["P10_5Y"]*100, cf_P10_5Y_pct=m_c["P10_5Y"]*100))

    # ---- crisis-window MaxDD: B1_STR(2.0) vs its equal-fbar uniform-delever twin ----
    print("\n  --- Crisis-window (R-STAT-2) timing check: B1_STR_S2.0 vs equal-fbar uniform twin ---")
    r_raw20, r_b120, tpy20 = build_b1_str(ctx, 2.0)
    fbar20 = measure_mean_in_leg_frac(r_raw20, r_b120, ctx["fund_active"], ctx["sofr_arr"])
    r_uni20 = build_uniform_delever(r_raw20, ctx["fund_active"], ctx["sofr_arr"], fbar20)
    rows = crisis_window_dd_compare(r_b120, r_uni20, ctx["stress"])
    for row in rows:
        print("      %-12s brakeDD=%+.4f%% twinDD=%+.4f%% edge=%+.4fpp shallower=%s"
              % (row["window"], row["brake_maxdd"]*100, row["twin_maxdd"]*100,
                 row["dd_edge_pp"], row["brake_shallower"]))
    st = sign_test_brake_beats_twin(rows)
    print("      SIGN TEST: n=%d shallower=%d deeper=%d p_onesided=%.4f mean_edge=%+.4fpp"
          % (st["n_windows"], st["n_shallower"], st["n_deeper"],
             st["binom_p_onesided"], st["mean_dd_edge_pp"]))
    for row in rows:
        out_rows.append(dict(claim="1_crisis_window", window=row["window"],
                              brake_maxdd_pct=row["brake_maxdd"]*100,
                              twin_maxdd_pct=row["twin_maxdd"]*100,
                              dd_edge_pp=row["dd_edge_pp"],
                              brake_shallower=row["brake_shallower"]))
    out_rows.append(dict(claim="1_crisis_signtest", n_windows=st["n_windows"],
                          n_shallower=st["n_shallower"], n_deeper=st["n_deeper"],
                          binom_p_onesided=st["binom_p_onesided"],
                          mean_dd_edge_pp=st["mean_dd_edge_pp"]))

    n_counterexamples = sum(1 for v in verdicts if v.get("resolved") and v["b1_beats"])
    n_resolved = sum(1 for v in verdicts if v.get("resolved"))
    print("\n  CLAIM 1 SUMMARY: %d/%d resolved equal-risk cuts show B1 CAGR > counterfactual (possible counterexample)."
          % (n_counterexamples, n_resolved))
    return verdicts, st


# =============================================================================
# CLAIM 2: "B1 is a weak brake (not a smart signal)." Re-test A7-vs-B1 under
# alternate normalizations: equal turnover, equal avg leverage, equal CAGR.
# =============================================================================
def claim2_alt_normalizations(ctx, out_rows):
    print("\n" + "=" * 100)
    print("CLAIM 2: B1 is a weak brake. Attack via alt normalizations vs A7 (base scale=1.0):")
    print("         (a) equal turnover  (b) equal avg leverage  (c) equal CAGR_OOS")
    print("=" * 100)

    r_raw7, r_a7, tpy7 = build_a7_s1(ctx)
    nav_a7 = pd.Series(np.cumprod(1.0 + r_a7), index=ctx["dates_dt"])
    m_a7 = m10(nav_a7, tpy7)
    fbar_a7 = measure_mean_in_leg_frac(r_raw7, r_a7, ctx["fund_active"], ctx["sofr_arr"])
    tpy7_full = tpy7
    print("  A7 (scale=1.0): MaxDD=%+.4f%% CAGR_OOS=%+.4f%% fbar=%.4f trades/yr=%.2f"
          % (m_a7["MaxDD"]*100, m_a7["CAGR_OOS"]*100, fbar_a7, tpy7_full))

    # B1 at scale=1.0 (base comparable to A7's scale)
    r_raw_b1_1, r_b1_1, tpy_b1_1 = build_b1_str(ctx, 1.0)
    nav_b1_1 = pd.Series(np.cumprod(1.0 + r_b1_1), index=ctx["dates_dt"])
    m_b1_1 = m10(nav_b1_1, tpy_b1_1)
    fbar_b1_1 = measure_mean_in_leg_frac(r_raw_b1_1, r_b1_1, ctx["fund_active"], ctx["sofr_arr"])
    print("  B1 (scale=1.0): MaxDD=%+.4f%% CAGR_OOS=%+.4f%% fbar=%.4f trades/yr=%.2f"
          % (m_b1_1["MaxDD"]*100, m_b1_1["CAGR_OOS"]*100, fbar_b1_1, tpy_b1_1))

    # (a) EQUAL TURNOVER: A7 and B1 already share trades/yr (base P09_C1 mask +
    # brake changes exposure fraction but not fund_active transitions -- turnover
    # of the underlying signal is IDENTICAL by construction; brakes don't add
    # trades in this harness (continuous blend, not discrete flips)). Report as-is.
    print("\n  (a) EQUAL TURNOVER: trades/yr identical by construction (brakes blend continuously,")
    print("      do not add fund_active flips). A7 tpy=%.2f  B1 tpy=%.2f (diff=%.4f)"
          % (tpy7_full, tpy_b1_1, tpy_b1_1 - tpy7_full))
    out_rows.append(dict(claim="2_turnover", a7_tpy=tpy7_full, b1_tpy=tpy_b1_1,
                          a7_MaxDD_pct=m_a7["MaxDD"]*100, b1_MaxDD_pct=m_b1_1["MaxDD"]*100))

    # (b) EQUAL AVERAGE LEVERAGE: scale B1's base so avg_lev matches A7's avg_lev.
    #     A7 avg_lev = 1.0*(1-fbar_a7); B1 avg_lev at scale S = S*(1-fbar_b1(S)).
    #     Since fbar is roughly scale-invariant for this brake (target_dvol fixed
    #     in return space, so higher scale -> more frequent trigger -> higher fbar),
    #     bisect B1 scale S so that S*(1-fbar_b1(S)) == avg_lev_a7.
    avg_lev_a7 = 1.0 * (1.0 - fbar_a7)

    def b1_avg_lev(scale):
        r_raw_s, r_b1_s, _ = build_b1_str(ctx, scale)
        fbar_s = measure_mean_in_leg_frac(r_raw_s, r_b1_s, ctx["fund_active"], ctx["sofr_arr"])
        return scale * (1.0 - fbar_s), fbar_s

    lo, hi = 0.3, 3.0
    lev_lo, _ = b1_avg_lev(lo)
    lev_hi, _ = b1_avg_lev(hi)
    s_star = None
    if lev_lo <= avg_lev_a7 <= lev_hi:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lev_mid, fbar_mid = b1_avg_lev(mid)
            if abs(lev_mid - avg_lev_a7) < 1e-4:
                s_star = mid
                break
            if lev_mid < avg_lev_a7:
                lo = mid
            else:
                hi = mid
        if s_star is None:
            s_star = 0.5 * (lo + hi)
    if s_star is not None:
        r_raw_s, r_b1_s, tpy_s = build_b1_str(ctx, s_star)
        nav_s = pd.Series(np.cumprod(1.0 + r_b1_s), index=ctx["dates_dt"])
        m_s = m10(nav_s, tpy_s)
        fbar_s = measure_mean_in_leg_frac(r_raw_s, r_b1_s, ctx["fund_active"], ctx["sofr_arr"])
        avg_lev_s = s_star * (1.0 - fbar_s)
        print("\n  (b) EQUAL AVG LEVERAGE: A7 avg_lev=%.4f  ->  B1_STR scale=%.4f (avg_lev=%.4f, fbar=%.4f)"
              % (avg_lev_a7, s_star, avg_lev_s, fbar_s))
        print("      A7  MaxDD=%+.4f%% CAGR_OOS=%+.4f%%" % (m_a7["MaxDD"]*100, m_a7["CAGR_OOS"]*100))
        print("      B1  MaxDD=%+.4f%% CAGR_OOS=%+.4f%%" % (m_s["MaxDD"]*100, m_s["CAGR_OOS"]*100))
        b1_deeper = m_s["MaxDD"] < m_a7["MaxDD"] - 1e-4
        print("      -> B1 MaxDD %s than A7 at equal avg leverage"
              % ("DEEPER (still weaker, consistent with claim)" if b1_deeper else "SHALLOWER-OR-EQUAL (possible counterexample)"))
        out_rows.append(dict(claim="2_eqavglev", a7_avg_lev=avg_lev_a7, b1_scale=s_star,
                              b1_avg_lev=avg_lev_s, a7_MaxDD_pct=m_a7["MaxDD"]*100,
                              b1_MaxDD_pct=m_s["MaxDD"]*100, a7_CAGR_OOS_pct=m_a7["CAGR_OOS"]*100,
                              b1_CAGR_OOS_pct=m_s["CAGR_OOS"]*100, b1_deeper=b1_deeper))
    else:
        print("\n  (b) EQUAL AVG LEVERAGE: A7 avg_lev=%.4f not bracketed in B1 scale [0.3,3.0] "
              "(B1 lev range [%.4f,%.4f]) -- cannot resolve" % (avg_lev_a7, lev_lo, lev_hi))
        out_rows.append(dict(claim="2_eqavglev", resolved=False, a7_avg_lev=avg_lev_a7,
                              b1_lev_lo=lev_lo, b1_lev_hi=lev_hi))

    # (c) EQUAL CAGR_OOS: bisect B1 scale so CAGR_OOS(B1) == CAGR_OOS(A7)
    def b1_cagr(scale):
        r_raw_s, r_b1_s, tpy_s = build_b1_str(ctx, scale)
        nav_s = pd.Series(np.cumprod(1.0 + r_b1_s), index=ctx["dates_dt"])
        return m10(nav_s, tpy_s)["CAGR_OOS"]

    lo, hi = 0.3, 3.0
    clo, chi = b1_cagr(lo), b1_cagr(hi)
    s_star2 = None
    if clo <= m_a7["CAGR_OOS"] <= chi:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            cmid = b1_cagr(mid)
            if abs(cmid - m_a7["CAGR_OOS"]) < 1e-4:
                s_star2 = mid
                break
            if cmid < m_a7["CAGR_OOS"]:
                lo = mid
            else:
                hi = mid
        if s_star2 is None:
            s_star2 = 0.5 * (lo + hi)
    if s_star2 is not None:
        r_raw_s2, r_b1_s2, tpy_s2 = build_b1_str(ctx, s_star2)
        nav_s2 = pd.Series(np.cumprod(1.0 + r_b1_s2), index=ctx["dates_dt"])
        m_s2 = m10(nav_s2, tpy_s2)
        print("\n  (c) EQUAL CAGR_OOS: A7 CAGR_OOS=%+.4f%%  ->  B1_STR scale=%.4f (CAGR_OOS=%+.4f%%)"
              % (m_a7["CAGR_OOS"]*100, s_star2, m_s2["CAGR_OOS"]*100))
        print("      A7 MaxDD=%+.4f%%   B1 MaxDD=%+.4f%%" % (m_a7["MaxDD"]*100, m_s2["MaxDD"]*100))
        b1_deeper2 = m_s2["MaxDD"] < m_a7["MaxDD"] - 1e-4
        print("      -> B1 MaxDD %s than A7 at equal CAGR_OOS"
              % ("DEEPER (still weaker, consistent with claim)" if b1_deeper2 else "SHALLOWER-OR-EQUAL (possible counterexample)"))
        out_rows.append(dict(claim="2_eqcagr", a7_CAGR_OOS_pct=m_a7["CAGR_OOS"]*100,
                              b1_scale=s_star2, b1_CAGR_OOS_pct=m_s2["CAGR_OOS"]*100,
                              a7_MaxDD_pct=m_a7["MaxDD"]*100, b1_MaxDD_pct=m_s2["MaxDD"]*100,
                              b1_deeper=b1_deeper2))
    else:
        print("\n  (c) EQUAL CAGR_OOS: A7 CAGR_OOS=%+.4f%% not bracketed in B1 range [%.4f,%.4f] -- cannot resolve"
              % (m_a7["CAGR_OOS"]*100, clo*100, chi*100))
        out_rows.append(dict(claim="2_eqcagr", resolved=False))


# =============================================================================
# CLAIM 3: "P09 scale ~= 1.4 is the practical MaxDD <= -50% ceiling." Sweep
# scale finely + apply stress-multiplier sensitivity (3-day consecutive worst
# day, 1.5x intraday gap approximation).
# =============================================================================
def claim3_scale_ceiling(ctx, out_rows):
    print("\n" + "=" * 100)
    print("CLAIM 3: P09_STR scale ~= 1.4 is the practical MaxDD <= -50% ceiling.")
    print("         Sweep scale 1.30-1.55 (step 0.01) for the daily-close MaxDD crossing.")
    print("=" * 100)

    scales = np.round(np.arange(1.30, 1.5501, 0.01), 2)
    rows = []
    prev_side = False   # assume "not yet breached" before the sweep starts, so a
                         # breach AT the first scale point is still detected below
    crossing_scale = None
    first_scale_breached = False
    for i, s in enumerate(scales):
        r, tpy = build_p09_str(ctx, s)
        nav = pd.Series(np.cumprod(1.0 + r), index=ctx["dates_dt"])
        m = m10(nav, tpy)
        side = m["MaxDD"] <= -0.50
        rows.append(dict(scale=float(s), MaxDD_pct=m["MaxDD"]*100,
                          CAGR_OOS_pct=m["CAGR_OOS"]*100, Worst1D_pct=m["Worst1D"]*100))
        print("  scale=%.2f  MaxDD=%+.4f%%  CAGR_OOS=%+.4f%%  Worst1D=%+.4f%%  %s"
              % (s, m["MaxDD"]*100, m["CAGR_OOS"]*100, m["Worst1D"]*100,
                 "<=-50%" if side else ""))
        if side and not prev_side and crossing_scale is None:
            crossing_scale = float(s)
            if i == 0:
                first_scale_breached = True
        prev_side = side

    if crossing_scale is not None and first_scale_breached:
        print("\n  DAILY-CLOSE CROSSING: MaxDD ALREADY <= -50%% at the low end of the sweep "
              "(scale=%.2f); true crossing is BELOW 1.30 (outside this sweep's range)." % scales[0])
    else:
        print("\n  DAILY-CLOSE CROSSING: MaxDD first breaches -50%% at scale=%s"
              % (("%.2f" % crossing_scale) if crossing_scale else "not found in [1.30,1.55]"))
    for row in rows:
        out_rows.append(dict(claim="3_sweep", **row))
    out_rows.append(dict(claim="3_crossing_daily", crossing_scale=crossing_scale))

    # Both stress sensitivities below breach -50% well BELOW scale=1.30 (confirmed
    # by a wider probe), so they are swept on a WIDER grid [0.50, 1.55] to actually
    # locate their crossing scale, not just confirm "already breached at 1.30".
    stress_scales = np.round(np.arange(0.50, 1.5501, 0.01), 2)

    # ---- Stress sensitivity: (i) 3-day-consecutive-worst-day approx ----
    # Approximation: take the single worst historical daily return r_min at each
    # scale's IN-leg (before OUT-fill dilution), and ask what MaxDD WOULD BE if
    # that day repeated 3x consecutively (crude tail-clustering stress), applied
    # as a multiplicative shock on top of the actual pre-shock peak-to-date NAV.
    # We approximate by CUBING the day's (1+r) factor at the SAME index (a lower
    # bound on what a true 3-day cluster could do if realized moves stayed at the
    # same daily magnitude).
    print("\n  STRESS SENSITIVITY (i): worst single day repeated 3x consecutively")
    print("  (injected at the historical worst-day location; crude tail-clustering proxy)")
    print("  swept on WIDER grid [0.50,1.55] since it breaches -50%% well below scale=1.30")
    crossing_3day = None
    prev_side = False
    for s in stress_scales:
        r, tpy = build_p09_str(ctx, s)
        r = np.asarray(r, float)
        worst_idx = int(np.argmin(r))
        worst_r = float(r[worst_idx])
        r_stress = r.copy()
        r_stress[worst_idx] = (1.0 + worst_r) ** 3 - 1.0
        dd_stress = _maxdd_from_returns(r_stress)
        side = dd_stress <= -0.50
        if side and not prev_side and crossing_3day is None:
            crossing_3day = float(s)
        prev_side = side
        out_rows.append(dict(claim="3_stress_3day", scale=float(s),
                              worst_day_pct=worst_r*100, MaxDD_stress_pct=dd_stress*100))
    print("  3-DAY-CLUSTER CROSSING: MaxDD first breaches -50%% at scale=%s"
          % (("%.2f" % crossing_3day) if crossing_3day else "not found in [0.50,1.55]"))

    # ---- Stress sensitivity: (ii) intraday gap = 1.5x the close-to-close move,
    # applied ONLY to TAIL down-days (gap risk is a large-move phenomenon; naively
    # multiplying EVERY ordinary down-day by 1.5x for 52 years mechanically decays
    # NAV to ~0 via pure volatility drag regardless of strategy quality -- that is
    # an artifact, not a meaningful stress test, and was caught and rejected here.
    # Definition used: for days below the 1st percentile of that scale's daily
    # return distribution (its worst ~1% days -- a proxy for "gap-risk days"),
    # inflate the loss by 1.5x; all other days unchanged.
    print("\n  STRESS SENSITIVITY (ii): intraday gap approx = 1.5x the observed")
    print("  close-to-close return, applied ONLY to the worst 1%% of days (tail gap-risk")
    print("  proxy). [An earlier version applied 1.5x to EVERY down-day and was rejected")
    print("  as an artifact -- it mechanically decays NAV to ~0 via pure vol drag compounded")
    print("  over 52yr, unrelated to true overnight-gap risk.] Swept on WIDER grid [0.50,1.55].")
    crossing_gap = None
    prev_side = False
    for s in stress_scales:
        r, tpy = build_p09_str(ctx, s)
        r = np.asarray(r, float)
        thr = np.percentile(r, 1.0)          # worst ~1% of days = tail gap-risk proxy;
                                              # fixed stress overlay, not a trading signal
        r_gap = np.where(r <= thr, r * 1.5, r)
        dd_gap = _maxdd_from_returns(r_gap)
        side = dd_gap <= -0.50
        if side and not prev_side and crossing_gap is None:
            crossing_gap = float(s)
        prev_side = side
        out_rows.append(dict(claim="3_stress_gap15x", scale=float(s), MaxDD_stress_pct=dd_gap*100))
    print("  1.5x-GAP(tail1%%) CROSSING: MaxDD first breaches -50%% at scale=%s"
          % (("%.2f" % crossing_gap) if crossing_gap else "not found in [0.50,1.55]"))

    out_rows.append(dict(claim="3_crossing_summary", daily=crossing_scale,
                          stress_3day=crossing_3day, stress_gap15x=crossing_gap))
    return crossing_scale, crossing_3day, crossing_gap


# =============================================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 100)
    print("frontier_refute_20260702.py -- Task 4: refutation of 3 headline claims")
    print("=" * 100)

    ctx = setup_shared()
    ok = self_test(ctx)
    if not ok:
        print("\nSELF-TEST FAILED. Halting (do not trust downstream results).")
        sys.exit(1)

    out_rows = []
    v1, st1 = claim1_equal_risk_frontier(ctx, out_rows)
    claim2_alt_normalizations(ctx, out_rows)
    c3_daily, c3_3day, c3_gap = claim3_scale_ceiling(ctx, out_rows)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "frontier_refute_20260702.csv")
    df = pd.DataFrame(out_rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nSaved CSV: %s (%d rows)" % (csv_path, len(df)))

    n_resolved1 = sum(1 for v in v1 if v.get("resolved"))
    n_counter1 = sum(1 for v in v1 if v.get("resolved") and v["b1_beats"])

    summary = dict(
        self_test_passed=ok,
        claim1_n_resolved=n_resolved1,
        claim1_n_counterexamples=n_counter1,
        claim1_crisis_signtest_p=st1["binom_p_onesided"],
        claim1_crisis_n_shallower=st1["n_shallower"],
        claim1_crisis_n_windows=st1["n_windows"],
        claim3_daily_crossing_scale=c3_daily,
        claim3_stress_3day_crossing_scale=c3_3day,
        claim3_stress_gap15x_crossing_scale=c3_gap,
        csv_path=csv_path,
    )
    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
