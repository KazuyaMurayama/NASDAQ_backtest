"""
src/audit/frontier_rederive_20260702.py
========================================
P09_C1 x scale critical verification -- Task 3: independent re-derivation of
the decision-critical numbers in the 4 reports dated 2026-06-19..22:
  LEVERUP_SCALE_FRONTIER_20260619.md
  P09C1_ALLOCATION_VARIATIONS_20260620.md
  A7_DD_REDUCTION_VARIATIONS_20260621.md
  B1_SCALE_FRONTIER_20260621.md

Independence requirement: metrics are computed from the raw daily-return /
NAV arrays with a SELF-CONTAINED implementation in this file. We do NOT
import compute_10metrics / calc_7metrics / _apply_aftertax -- only the
validated NAV BUILDERS (dd_reduction_harness_20260626.setup()+build(),
b1_scale_annual_20260621's builder logic re-used via the same underlying
strategy_runners / k365_recost / out_fill_variants / dd_reduction_overlays
modules) are reused, per the plan's instruction ("read and understand the
structure, then re-implement the metric layer").

Definitions re-derived from first principles (matching src/cfd_leverage_backtest.py
calc_7metrics + src/audit/run_p01_backtest_20260611.py _apply_aftertax, but
written independently here):
  - IS window:  1974-01-02 .. 2021-05-07  (inclusive)
  - OOS window: 2021-05-08 .. 2026-12-31  (inclusive; FULL_END)
  - CAGR_period = (NAV_end / NAV_start) ** (1/years) - 1, years = ndays/252,
    where NAV is re-based to 1.0 at the first observation INSIDE the window
    (i.e. period return, not full-history NAV level).
  - Sharpe_FULL = mean(daily_ret_full) * 252 / (std(daily_ret_full, ddof=1) * sqrt(252)), rf=0.
  - MaxDD = min over full history of (NAV_t / running_max(NAV) - 1).
  - Worst1D = min daily return over full history (+ its date).
  - Worst10Y = min over rolling non-overlapping-free calendar-year-chained
    10Y CAGR windows: for each year-end t, (NAV[t]/NAV[t-10y])**(1/10)-1,
    take the min (uses calendar-year NAV snapshots, matching
    compute_cfd_worst10y.rolling_nY_cagr's calendar-year basis).
  - Worst5Y / P10_5Y: rolling 5-year (1260 trading day) DAILY window CAGR
    distribution; Worst5Y = min, P10_5Y = 10th percentile.
  - Trades/yr: passed through from the builder (transition count / n_years).
  - After-tax: CAGR_IS, CAGR_OOS, CAGR_FULL, Worst10Y, Worst5Y, P10_5Y are
    multiplied by AFTER_TAX=0.8273 (metric-level, matching repo convention
    _AFTERTAX_KEYS). Sharpe, MaxDD, Worst1D stay PRE-tax (report convention).

ASCII-only prints (Windows cp932-safe). Writes ONE CSV:
  audit_results/frontier_rederive_20260702.csv
No commit, no temp files outside audit_results/.
"""
from __future__ import annotations

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

# ---- ONLY reuse validated NAV BUILDERS (not the metric layer) ----
import src.audit.strategy_runners as sr
from src.audit.k365_recost_20260612 import (
    _build_tqqq_base_param, EXCESS_EXTRA_K365_CENTRE,
)
from src.audit.run_p01_backtest_20260611 import (
    _ret_from_nav_level, _inverse_vol_weights, _count_fund_transitions,
    LAG_DAYS, TRADING_DAYS,
)
from src.audit.run_p02_p09_backtest_20260611 import _load_macro_signal, GATE_DELAY
from src.audit.out_fill_variants_20260620 import _build_out_fill_variant, alloc_base
from src.audit.dd_reduction_overlays_20260621 import (
    apply_downside_dev_brake, apply_param_vol_brake,
)

# ---------------------------------------------------------------------------
# Independent metric-layer constants (re-derived from first principles;
# cross-referenced to src/cfd_leverage_backtest.py IS_START/IS_END/OOS_START/
# FULL_END and run_p01_backtest_20260611.AFTER_TAX -- values match but the
# CODE below is written independently, not imported).
# ---------------------------------------------------------------------------
IS_START = pd.Timestamp("1974-01-02")
IS_END = pd.Timestamp("2021-05-07")
OOS_START = pd.Timestamp("2021-05-08")
FULL_END = pd.Timestamp("2026-12-31")
AFTER_TAX = 0.8273
TDAYS = 252

B1_TARGET_DVOL = 0.20
B1_WINDOW = 63
B1_MAX_FRAC_CASH = 0.5

STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXC = EXCESS_EXTRA_K365_CENTRE  # 0.0025


# ---------------------------------------------------------------------------
# Independent metric functions
# ---------------------------------------------------------------------------
def period_cagr(nav_dt: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """(NAV_end/NAV_start)^(1/years) - 1 over [start, end], rebased at window start."""
    sub = nav_dt.loc[(nav_dt.index >= start) & (nav_dt.index <= end)]
    if len(sub) < 100:
        return float("nan")
    ns = sub / sub.iloc[0]
    years = len(ns) / float(TDAYS)
    if years <= 0 or ns.iloc[-1] <= 0:
        return float("nan")
    return float(ns.iloc[-1]) ** (1.0 / years) - 1.0


def full_daily_returns(nav_dt: pd.Series) -> pd.Series:
    return nav_dt.pct_change().dropna()


def sharpe_full(nav_dt: pd.Series) -> float:
    r = full_daily_returns(nav_dt)
    if len(r) < 2 or r.std(ddof=1) <= 0:
        return float("nan")
    return float(r.mean() * TDAYS / (r.std(ddof=1) * np.sqrt(TDAYS)))


def max_drawdown(nav_dt: pd.Series) -> float:
    run_max = nav_dt.cummax()
    dd = nav_dt / run_max - 1.0
    return float(dd.min())


def worst_1d(nav_dt: pd.Series):
    r = full_daily_returns(nav_dt)
    if len(r) == 0:
        return float("nan"), None
    idx = r.idxmin()
    return float(r.min()), idx.strftime("%Y-%m-%d")


def worst_10y_calendar(nav_dt: pd.Series) -> float:
    """Min rolling 10-calendar-year CAGR using calendar year-END NAV snapshots."""
    yr_end = nav_dt.groupby(nav_dt.index.year).last()
    yr_end = yr_end.sort_index()
    years = yr_end.index.values
    vals = []
    for i in range(len(years)):
        y0 = years[i] - 10
        if y0 in yr_end.index:
            nav0 = yr_end.loc[y0]
            nav1 = yr_end.loc[years[i]]
            if nav0 > 0:
                vals.append((nav1 / nav0) ** (1.0 / 10.0) - 1.0)
    if not vals:
        return float("nan")
    return float(min(vals))


def rolling_5y_daily_cagr(nav_dt: pd.Series) -> np.ndarray:
    """Rolling 5y (1260 trading-day) DAILY-window CAGR series (daily step)."""
    vals = nav_dt.values
    n = len(vals)
    w = 5 * TDAYS
    if n <= w:
        return np.array([])
    out = np.empty(n - w)
    for i in range(w, n):
        out[i - w] = (vals[i] / vals[i - w]) ** (1.0 / 5.0) - 1.0
    return out


def worst_5y(nav_dt: pd.Series) -> float:
    r5 = rolling_5y_daily_cagr(nav_dt)
    if len(r5) == 0:
        return float("nan")
    return float(np.min(r5))


def p10_5y(nav_dt: pd.Series) -> float:
    r5 = rolling_5y_daily_cagr(nav_dt)
    if len(r5) == 0:
        return float("nan")
    return float(np.percentile(r5, 10))


def calendar_year_returns(nav_dt: pd.Series) -> dict:
    yr_end = nav_dt.groupby(nav_dt.index.year).last()
    yr_start = nav_dt.groupby(nav_dt.index.year).first()
    prev_end = yr_end.shift(1)
    base = prev_end.copy()
    base.iloc[0] = yr_start.iloc[0]
    cy = yr_end / base - 1.0
    return {int(y): float(v) for y, v in cy.items()}


def metrics_independent(nav_dt: pd.Series, tpy: float, label: str) -> dict:
    """Full independent metric set for one NAV series."""
    nav_dt = nav_dt.dropna().sort_index()
    cagr_is = period_cagr(nav_dt, IS_START, IS_END)
    cagr_oos = period_cagr(nav_dt, OOS_START, FULL_END)
    cagr_full = period_cagr(nav_dt, IS_START, FULL_END)
    sh = sharpe_full(nav_dt)
    mdd = max_drawdown(nav_dt)
    w1d, w1d_date = worst_1d(nav_dt)
    w10y = worst_10y_calendar(nav_dt)
    w5y = worst_5y(nav_dt)
    p10 = p10_5y(nav_dt)
    gap_pp_pretax = (cagr_is - cagr_oos) * 100.0  # pre-tax pp (report convention: gap uses pre-tax CAGR diff)

    return dict(
        label=label,
        CAGR_IS_at=cagr_is * AFTER_TAX,
        CAGR_OOS_at=cagr_oos * AFTER_TAX,
        CAGR_FULL_at=cagr_full * AFTER_TAX,
        IS_OOS_gap_pp=gap_pp_pretax,
        Sharpe_FULL=sh,
        MaxDD=mdd,
        Worst1D=w1d,
        Worst1D_date=w1d_date,
        Worst10Y_at=w10y * AFTER_TAX,
        Worst5Y_at=w5y * AFTER_TAX,
        P10_5Y_at=p10 * AFTER_TAX,
        Trades_yr=tpy,
    )


# ---------------------------------------------------------------------------
# Shared context loader (verbatim wiring, independently re-typed)
# ---------------------------------------------------------------------------
def load_ctx():
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

    return dict(shared=shared, a=a, dates=dates, dates_dt=dates_dt, n=n,
                n_years=n_years, ret_gold=ret_gold, ret_bond=ret_bond,
                fund_active=fund_active, wg=wg, wb=wb, bond_on=bond_on,
                sofr_arr=sofr_arr)


def build_p09_str_scale(ctx, scale: float):
    """P09_STR x scale (no brake). LEVERUP_SCALE_FRONTIER / B1 report 'P09 (no brake)' rows."""
    shared, dates_dt, n_years = ctx["shared"], ctx["dates_dt"], ctx["n_years"]
    fund_active = ctx["fund_active"]
    ret_gold, ret_bond = ctx["ret_gold"], ctx["ret_bond"]
    wg, wb, bond_on, sofr_arr = ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"]

    _, r_base, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=STRONG_MAP, lev_scale=scale, excess_extra=EXC)
    nav_s, r_strat, eff = _build_out_fill_variant(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    tpy = tpy_base + _count_fund_transitions(eff) / n_years
    nav_dt = pd.Series(np.cumprod(1.0 + r_strat), index=dates_dt)
    return nav_dt, r_strat, tpy


def build_b1_str_scale(ctx, scale: float):
    """B1_STR x scale (strong map + downside-dev brake). B1_SCALE_FRONTIER report rows."""
    shared, dates_dt, n_years = ctx["shared"], ctx["dates_dt"], ctx["n_years"]
    fund_active = ctx["fund_active"]
    ret_gold, ret_bond = ctx["ret_gold"], ctx["ret_bond"]
    wg, wb, bond_on, sofr_arr = ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"]

    _, r_base, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=STRONG_MAP, lev_scale=scale, excess_extra=EXC)
    nav_s, r_strat, eff = _build_out_fill_variant(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    r_b1 = apply_downside_dev_brake(
        r_strat, fund_active, sofr_arr, B1_TARGET_DVOL, B1_WINDOW, B1_MAX_FRAC_CASH)
    tpy = tpy_base + _count_fund_transitions(eff) / n_years
    nav_dt = pd.Series(np.cumprod(1.0 + r_b1), index=dates_dt)
    return nav_dt, r_b1, tpy


def build_a0_p09c1(ctx):
    """A0 = P09_C1 canonical (v7_map=None strong-implicit default, scale=1.0, excess=0.0)."""
    shared, dates_dt, n_years = ctx["shared"], ctx["dates_dt"], ctx["n_years"]
    fund_active = ctx["fund_active"]
    ret_gold, ret_bond = ctx["ret_gold"], ctx["ret_bond"]
    wg, wb, bond_on, sofr_arr = ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"]

    _, r_base, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=None, lev_scale=1.0, excess_extra=0.0)
    nav_s, r_strat, eff = _build_out_fill_variant(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    tpy = tpy_base + _count_fund_transitions(eff) / n_years
    nav_dt = pd.Series(np.cumprod(1.0 + r_strat), index=dates_dt)
    return nav_dt, r_strat, tpy


def build_a7_p09c1(ctx):
    """A7 = P09_C1 + IN-leg TOTAL realized-vol brake (vol > 30% (w63) -> up to 50% cash).
    Matches a7dd_annual_20260621.py's A7_REPRODUCE: apply_param_vol_brake(...,0.30,63,0.5)."""
    shared, dates_dt, n_years = ctx["shared"], ctx["dates_dt"], ctx["n_years"]
    fund_active = ctx["fund_active"]
    ret_gold, ret_bond = ctx["ret_gold"], ctx["ret_bond"]
    wg, wb, bond_on, sofr_arr = ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"]

    _, r_base, tpy_base, _ = _build_tqqq_base_param(
        shared, dates_dt, v7_map=None, lev_scale=1.0, excess_extra=0.0)
    nav_s, r_strat, eff = _build_out_fill_variant(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        alloc_fn=alloc_base)
    r_a7 = apply_param_vol_brake(r_strat, fund_active, sofr_arr, 0.30, 63, 0.5)
    tpy = tpy_base + _count_fund_transitions(eff) / n_years
    nav_dt = pd.Series(np.cumprod(1.0 + r_a7), index=dates_dt)
    return nav_dt, r_a7, tpy


# ---------------------------------------------------------------------------
# Anchors (4-digit, from the 4 reports; label -> dict of report values)
# ---------------------------------------------------------------------------
ANCHORS = {
    "P09_STR_S1.4": dict(CAGR_IS=27.49, CAGR_OOS=24.34, Sharpe_FULL=1.074, MaxDD=-46.48,
                          Worst1D=-19.45, Worst1D_date="2016-06-24", Worst10Y=17.87,
                          Worst5Y=0.22, P10_5Y=9.05, Trades_yr=35.2),
    "P09_STR_S1.6": dict(CAGR_IS=30.26, CAGR_OOS=26.21, Sharpe_FULL=1.055, MaxDD=-51.95,
                          Worst1D=-22.41, Worst1D_date="2016-06-24", Worst10Y=19.36,
                          Worst5Y=-0.31, P10_5Y=9.31, Trades_yr=35.2),
    "B1_STR_S2.0": dict(CAGR_OOS=24.07, MaxDD=-47.08),
    "B1_STR_S1.4": dict(CAGR_IS=24.69, CAGR_OOS=20.75, Sharpe_FULL=1.085, MaxDD=-39.19,
                         Worst1D=-15.03),
    "A7": dict(MaxDD=-28.25, Worst1D=-10.27),
    "A0_P09C1": dict(MaxDD=-34.99, Worst1D=-13.82),
}

# Frontier claim anchors (report §7 of B1_SCALE_FRONTIER)
FRONTIER_CLAIM = {
    "B1x2.0_vs_P09x1.4": dict(
        b1_maxdd=-47.08, p09_maxdd=-46.48, b1_cagr=24.07, p09_cagr=24.34),
}
DELTA_RANGE_MAXDD_PP = (7.3, 14.6)   # B1 shallower than P09 at same map/scale
DELTA_RANGE_CAGR_PP = (-5.04, -3.60)  # B1 CAGR minus P09 CAGR


def fmt_diff(rep, our):
    if rep is None or our is None or (isinstance(our, float) and np.isnan(our)):
        return "n/a", "n/a"
    d = our - rep
    return "%.4f" % our, "%+.4f" % d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 120)
    print("FRONTIER REDERIVE 2026-07-02 -- Task 3 independent metric re-derivation")
    print("Self-contained metric layer (not importing compute_10metrics/calc_7metrics)")
    print("=" * 120)

    ctx = load_ctx()

    rows = []

    # ---- P09_STR x scale (no brake): 1.4 and 1.6 ----
    for scale in (1.4, 1.6):
        label = "P09_STR_S%.1f" % scale
        nav_dt, r, tpy = build_p09_str_scale(ctx, scale)
        m = metrics_independent(nav_dt, tpy, label)
        rows.append(m)

    # ---- B1_STR x scale (brake): 1.4 and 2.0 ----
    for scale in (1.4, 2.0):
        label = "B1_STR_S%.1f" % scale
        nav_dt, r, tpy = build_b1_str_scale(ctx, scale)
        m = metrics_independent(nav_dt, tpy, label)
        rows.append(m)

    # ---- A0 = P09_C1 canonical ----
    nav_dt, r, tpy = build_a0_p09c1(ctx)
    m = metrics_independent(nav_dt, tpy, "A0_P09C1")
    rows.append(m)

    # ---- A7 = P09_C1 + total-vol brake ----
    try:
        nav_dt, r, tpy = build_a7_p09c1(ctx)
        m = metrics_independent(nav_dt, tpy, "A7")
        rows.append(m)
    except Exception as e:
        print("  A7 build FAILED: %r (apply_vol_brake not found or signature mismatch)" % e)

    df = pd.DataFrame(rows)
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "frontier_rederive_20260702.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nSaved CSV: %s (%d rows)" % (csv_path, len(df)))

    # =========================================================================
    # COMPARISON TABLE: report value vs re-derived value vs diff
    # =========================================================================
    print("\n" + "=" * 120)
    print("COMPARISON: report(4-digit) vs re-derived vs diff (pp for CAGR/MaxDD/Worst*, abs for Sharpe)")
    print("=" * 120)

    field_map = [
        ("CAGR_IS", "CAGR_IS_at", "pp"),
        ("CAGR_OOS", "CAGR_OOS_at", "pp"),
        ("Sharpe_FULL", "Sharpe_FULL", "abs"),
        ("MaxDD", "MaxDD", "pp"),
        ("Worst1D", "Worst1D", "pp"),
        ("Worst10Y", "Worst10Y_at", "pp"),
        ("Worst5Y", "Worst5Y_at", "pp"),
        ("P10_5Y", "P10_5Y_at", "pp"),
        ("Trades_yr", "Trades_yr", "abs"),
    ]

    by_label = {r["label"]: r for r in rows}
    diag_rows = []
    for label, anc in ANCHORS.items():
        our = by_label.get(label)
        if our is None:
            print("\n  [%s] NOT BUILT in this script (see note below)" % label)
            continue
        print("\n  [%s]" % label)
        for rep_key, our_key, kind in field_map:
            if rep_key not in anc:
                continue
            rep_val = anc[rep_key]
            our_val = our.get(our_key, float("nan"))
            if kind == "pp":
                # our metric functions return FRACTIONS (e.g. 0.2734); convert to %
                our_disp = our_val * 100.0
                diff = our_disp - rep_val
            else:
                our_disp = our_val
                diff = our_disp - rep_val
            flag = "  <-- DIFF>0.1" if abs(diff) > 0.1 else ""
            print("    %-14s report=%9.4f  rederived=%9.4f  diff=%+8.4f%s" %
                  (rep_key, rep_val, our_disp, diff, flag))
            diag_rows.append(dict(label=label, field=rep_key, report=rep_val,
                                   rederived=our_disp, diff=diff))

    diag_df = pd.DataFrame(diag_rows)
    diag_path = os.path.join(out_dir, "frontier_rederive_diag_20260702.csv")
    diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")
    print("\nSaved diagnostic diff table: %s (%d rows)" % (diag_path, len(diag_df)))

    # =========================================================================
    # FRONTIER CLAIM CHECK: "B1x2.0 ~= P09x1.4"
    # =========================================================================
    print("\n" + "=" * 120)
    print("FRONTIER CLAIM CHECK: B1_STR_S2.0 (brake) vs P09_STR_S1.4 (no brake)")
    print("Report: MaxDD -47.08 vs -46.48 (near); CAGR_OOS +24.07 vs +24.34 (near)")
    print("=" * 120)
    b1_20 = by_label.get("B1_STR_S2.0")
    p09_14 = by_label.get("P09_STR_S1.4")
    if b1_20 is not None and p09_14 is not None:
        b1_maxdd = b1_20["MaxDD"] * 100.0
        p09_maxdd = p09_14["MaxDD"] * 100.0
        b1_cagr = b1_20["CAGR_OOS_at"] * 100.0
        p09_cagr = p09_14["CAGR_OOS_at"] * 100.0
        print("  Rederived B1_STR_S2.0  : MaxDD=%+.4f%%  CAGR_OOS=%+.4f%%" % (b1_maxdd, b1_cagr))
        print("  Rederived P09_STR_S1.4 : MaxDD=%+.4f%%  CAGR_OOS=%+.4f%%" % (p09_maxdd, p09_cagr))
        print("  Rederived MaxDD gap (B1-P09): %+.4fpp  (report: -47.08-(-46.48)=%.4fpp)"
              % (b1_maxdd - p09_maxdd, -47.08 - (-46.48)))
        print("  Rederived CAGR gap (B1-P09): %+.4fpp  (report: 24.07-24.34=%.4fpp)"
              % (b1_cagr - p09_cagr, 24.07 - 24.34))
    else:
        print("  Cannot check -- one of B1_STR_S2.0 / P09_STR_S1.4 not built.")

    # =========================================================================
    # DELTA RANGE CHECK: "same map/scale B1 is +7.3~+14.6pp shallower MaxDD,
    # -3.60~-5.04pp lower CAGR_OOS than P09" -- check scale=1.4 pair (only
    # scale we independently built on both sides).
    # =========================================================================
    print("\n" + "=" * 120)
    print("DELTA-RANGE CHECK (same map/scale, B1 minus P09) at scale=1.4")
    print("Report range: MaxDD delta +7.3..+14.6pp shallower; CAGR_OOS delta -3.60..-5.04pp")
    print("=" * 120)
    b1_14 = by_label.get("B1_STR_S1.4")
    if b1_14 is not None and p09_14 is not None:
        maxdd_delta = (b1_14["MaxDD"] - p09_14["MaxDD"]) * 100.0  # positive = B1 shallower
        cagr_delta = (b1_14["CAGR_OOS_at"] - p09_14["CAGR_OOS_at"]) * 100.0
        _TOL = 0.05  # pp tolerance for boundary rounding (report ranges are rounded to 2dp)
        in_range_dd = (DELTA_RANGE_MAXDD_PP[0] - _TOL) <= maxdd_delta <= (DELTA_RANGE_MAXDD_PP[1] + _TOL)
        in_range_cagr = (DELTA_RANGE_CAGR_PP[0] - _TOL) <= cagr_delta <= (DELTA_RANGE_CAGR_PP[1] + _TOL)
        print("  Rederived MaxDD delta (B1-P09) at scale1.4: %+.4fpp  in-report-range=%s"
              % (maxdd_delta, in_range_dd))
        print("  Rederived CAGR_OOS delta (B1-P09) at scale1.4: %+.4fpp  in-report-range=%s"
              % (cagr_delta, in_range_cagr))
        print("  (Report scale1.4 row: MaxDD delta +7.28pp, CAGR delta -3.60pp)")
    else:
        print("  Cannot check -- missing series.")

    print("\nDone.")


if __name__ == "__main__":
    main()
