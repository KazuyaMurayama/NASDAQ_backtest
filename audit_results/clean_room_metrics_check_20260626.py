#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLEAN-ROOM independent verifier for standard-10 metrics.

This script imports NOTHING from the project's metric code. Every formula is
implemented here from the spec. It reads ONLY the raw daily-series export CSV.

Author: independent clean-room verifier (2026-06-26)
"""

import csv
import math
from collections import OrderedDict

CSV_PATH = r"C:\Users\user\Desktop\投資・不動産\nasdaq_backtest\audit_results\dd_series_export_20260626.csv"

TRADING_DAYS = 252
AFTER_TAX = 0.8273
IS_END = "2021-05-07"      # IS = dates <= this
OOS_START = "2021-05-08"   # OOS = dates >= this
# Calendar-year IS/OOS assignment for after-tax CAGR: year<=2020 -> IS, year>=2021 -> OOS
IS_YEAR_MAX = 2020

CANDIDATES = ["sc20", "N4", "X4"]

# Reported values to check against (from the pipeline)
REPORTED = {
    "sc20": dict(CAGR_IS=35.38, CAGR_OOS=29.11, Sharpe_FULL=1.0275, MaxDD=-61.63,
                 Worst10Y=20.84, Worst5Y=-1.93, P10_5Y=9.38, IS_OOS_gap=6.27, Trades=35.2),
    "N4":   dict(CAGR_IS=33.94, CAGR_OOS=31.29, Sharpe_FULL=1.0472, MaxDD=-58.82,
                 Worst10Y=20.28, Worst5Y=-1.98, P10_5Y=8.73, Trades=35.2),
    "X4":   dict(CAGR_IS=34.45, CAGR_OOS=32.15, Sharpe_FULL=1.0457, MaxDD=-59.85,
                 Worst10Y=20.40, Worst5Y=-2.31, P10_5Y=8.57, Trades=35.2),
}


# ----------------------------------------------------------------------------
# Load raw series
# ----------------------------------------------------------------------------
def load():
    rows = []
    with open(CSV_PATH, "r", newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    # parse
    dates = [r["date"] for r in rows]
    data = {}
    for name in CANDIDATES:
        ret = [float(r["ret_" + name]) for r in rows]
        nav = [float(r["nav_" + name]) for r in rows]
        tpy = float(rows[0]["tpy_" + name])
        data[name] = dict(ret=ret, nav=nav, tpy=tpy)
    return dates, data


def year_of(d):
    return int(d[:4])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def mean(xs):
    return sum(xs) / len(xs)


def std_sample(xs):
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def percentile_linear(sorted_xs, q):
    """Linear interpolation percentile (numpy default 'linear' / type 7). q in [0,100]."""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = (q / 100.0) * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


# ----------------------------------------------------------------------------
# Calendar-year after-tax machinery
# ----------------------------------------------------------------------------
def calendar_year_last_nav(dates, nav):
    """Return OrderedDict {year: nav_at_last_trading_day_of_year}, and the first nav."""
    last = OrderedDict()
    for d, v in zip(dates, nav):
        last[year_of(d)] = v  # overwrite -> ends at last row of that year
    return last, nav[0]


def after_tax_yearly(dates, nav):
    """
    yearly_return[Y] = nav_last[Y] / nav_last[Y-1] - 1, with first year using
    first nav as base.  after_tax = yearly * AFTER_TAX.
    Returns OrderedDict {year: after_tax_yearly_return}.
    """
    last, first_nav = calendar_year_last_nav(dates, nav)
    years = list(last.keys())
    out = OrderedDict()
    prev = first_nav
    for i, y in enumerate(years):
        cur = last[y]
        yr = cur / prev - 1.0
        out[y] = yr * AFTER_TAX
        prev = cur
    return out


def geo_cagr_from_yearly(aty_dict, years_subset):
    """Geometric mean of (1+aty) over the given years - 1."""
    vals = [aty_dict[y] for y in years_subset]
    if not vals:
        return float("nan")
    prod = 1.0
    for v in vals:
        prod *= (1.0 + v)
    n = len(vals)
    return prod ** (1.0 / n) - 1.0


def build_after_tax_yearly_nav(aty_dict):
    """Cumulative after-tax NAV indexed by year-end. Returns (years, atnav list) with
    atnav[i] = product_{k<=i}(1+aty[year_k]).  Also returns a 'base' of 1.0 before first year."""
    years = list(aty_dict.keys())
    atnav = []
    cum = 1.0
    for y in years:
        cum *= (1.0 + aty_dict[y])
        atnav.append(cum)
    return years, atnav


def rolling_n_calendar_year_geo(aty_dict, n):
    """
    Rolling n-calendar-year geometric-mean CAGR from after-tax yearly returns.
    For a window of n consecutive years [y0..y_{n-1}], CAGR = (prod(1+aty))^(1/n) - 1.
    Returns list of (start_year, end_year, cagr).
    """
    years = list(aty_dict.keys())
    out = []
    for i in range(0, len(years) - n + 1):
        window_years = years[i:i + n]
        prod = 1.0
        for y in window_years:
            prod *= (1.0 + aty_dict[y])
        cagr = prod ** (1.0 / n) - 1.0
        out.append((window_years[0], window_years[-1], cagr))
    return out


# ----------------------------------------------------------------------------
# Pre-tax daily metrics
# ----------------------------------------------------------------------------
def sharpe_full(ret):
    m = mean(ret)
    s = std_sample(ret)
    return m / s * math.sqrt(TRADING_DAYS)


def maxdd_full(nav):
    peak = -float("inf")
    mdd = 0.0
    for v in nav:
        if v > peak:
            peak = v
        dd = v / peak - 1.0
        if dd < mdd:
            mdd = dd
    return mdd


def worst1d(dates, ret):
    idx = min(range(len(ret)), key=lambda i: ret[i])
    return ret[idx], dates[idx]


def daily_rolling_cagr(nav, window_days, years):
    """roll = (nav[t]/nav[t-window])^(1/years) - 1, for t >= window. Returns list of values."""
    out = []
    for t in range(window_days, len(nav)):
        base = nav[t - window_days]
        if base <= 0:
            continue
        out.append((nav[t] / base) ** (1.0 / years) - 1.0)
    return out


def daily_compounded_cagr(nav_sub, n_days):
    """(nav_end/nav_start)^(252/n_days) - 1 over a sub-series."""
    if n_days <= 0 or nav_sub[0] <= 0:
        return float("nan")
    return (nav_sub[-1] / nav_sub[0]) ** (TRADING_DAYS / n_days) - 1.0


# ----------------------------------------------------------------------------
# PIPELINE-CONVENTION reconstructions (diagnosed empirically, NOT imported).
# Rule: compute the PRE-TAX annualized rate, then multiply the RATE by AFTER_TAX.
# ----------------------------------------------------------------------------
def pipe_cagr_split(nav, idx_mask):
    """Daily-compounded CAGR over an exact date-index subset, rate*AFTER_TAX."""
    sub = [nav[i] for i in idx_mask]
    rate = daily_compounded_cagr(sub, len(sub))
    return rate * AFTER_TAX


def pretax_calendar_yearly(dates, nav):
    last, first_nav = calendar_year_last_nav(dates, nav)
    out = OrderedDict()
    prev = first_nav
    for y in last:
        out[y] = last[y] / prev - 1.0
        prev = last[y]
    return out


def pipe_worst10y(dates, nav):
    """Calendar-year 10y rolling PRE-TAX geo rate, min, then *AFTER_TAX."""
    yr = pretax_calendar_yearly(dates, nav)
    years = list(yr.keys())
    best = None
    for i in range(0, len(years) - 10 + 1):
        wy = years[i:i + 10]
        prod = 1.0
        for y in wy:
            prod *= (1.0 + yr[y])
        rate = prod ** (1.0 / 10) - 1.0
        if best is None or rate < best[2]:
            best = (wy[0], wy[-1], rate)
    return best[2] * AFTER_TAX, best


def pipe_worst5y_and_p10(nav):
    """Daily 1260-day rolling PRE-TAX rate * AFTER_TAX -> Worst5Y (min) and P10."""
    rates = daily_rolling_cagr(nav, 1260, 5)
    at = [r * AFTER_TAX for r in rates]
    worst5 = min(at)
    p10 = percentile_linear(sorted(at), 10)
    return worst5, p10


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------
def classify(metric, diff_abs):
    """diff_abs in the metric's native units (pp for pct metrics, raw for sharpe)."""
    if metric == "Sharpe_FULL":
        small, med = 0.003, 0.009
    elif metric == "MaxDD":
        small, med = 0.20, 0.60
    else:  # CAGR / Worst / P10 / gap  -> pp
        small, med = 0.10, 0.30
    if diff_abs <= small:
        return "SMALL"
    if diff_abs <= med:
        return "MEDIUM"
    return "LARGE"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    dates, data = load()
    n = len(dates)
    print(f"Loaded {n} rows  first={dates[0]}  last={dates[-1]}")
    print(f"IS_END={IS_END}  OOS_START={OOS_START}  IS_YEAR_MAX={IS_YEAR_MAX}")
    print("=" * 110)

    results = {}
    extra = {}
    pipe = {}

    for name in CANDIDATES:
        ret = data[name]["ret"]
        nav = data[name]["nav"]
        tpy = data[name]["tpy"]

        # --- after-tax calendar-year CAGR (IS / OOS) ---
        aty = after_tax_yearly(dates, nav)
        is_years = [y for y in aty if y <= IS_YEAR_MAX]
        oos_years = [y for y in aty if y >= IS_YEAR_MAX + 1]
        cagr_is = geo_cagr_from_yearly(aty, is_years)
        cagr_oos = geo_cagr_from_yearly(aty, oos_years)
        is_oos_gap_pp = (cagr_is - cagr_oos) * 100.0

        # cross-check: daily compounded CAGR over exact date split
        is_mask = [i for i, d in enumerate(dates) if d <= IS_END]
        oos_mask = [i for i, d in enumerate(dates) if d >= OOS_START]
        nav_is = [nav[i] for i in is_mask]
        nav_oos = [nav[i] for i in oos_mask]
        cagr_is_daily = daily_compounded_cagr(nav_is, len(nav_is))
        cagr_oos_daily = daily_compounded_cagr(nav_oos, len(nav_oos))

        # --- Sharpe FULL (pre-tax) ---
        sh = sharpe_full(ret)

        # --- MaxDD FULL (pre-tax) ---
        mdd = maxdd_full(nav)

        # --- Worst1D ---
        w1, w1date = worst1d(dates, ret)

        # --- Worst10Y (after-tax, 10 calendar-year rolling geo) ---
        roll10 = rolling_n_calendar_year_geo(aty, 10)
        worst10y = min(r[2] for r in roll10) if roll10 else float("nan")
        worst10y_win = min(roll10, key=lambda r: r[2]) if roll10 else None

        # --- Worst5Y (after-tax 5 calendar-year rolling, AND pre-tax daily 1260) ---
        roll5_cal = rolling_n_calendar_year_geo(aty, 5)
        worst5y_cal = min(r[2] for r in roll5_cal) if roll5_cal else float("nan")
        worst5y_cal_win = min(roll5_cal, key=lambda r: r[2]) if roll5_cal else None
        roll5_daily = daily_rolling_cagr(nav, 1260, 5)
        worst5y_daily = min(roll5_daily) if roll5_daily else float("nan")

        # --- P10_5Y (after-tax 5 calendar-year rolling, AND pre-tax daily 1260) ---
        cal5_vals = sorted(r[2] for r in roll5_cal)
        p10_5y_cal = percentile_linear(cal5_vals, 10)
        daily5_vals = sorted(roll5_daily)
        p10_5y_daily = percentile_linear(daily5_vals, 10)

        results[name] = dict(
            CAGR_IS=cagr_is * 100, CAGR_OOS=cagr_oos * 100,
            Sharpe_FULL=sh, MaxDD=mdd * 100,
            Worst10Y=worst10y * 100, Worst5Y=worst5y_cal * 100,
            P10_5Y=p10_5y_cal * 100, IS_OOS_gap=is_oos_gap_pp, Trades=tpy,
        )

        # ---- PIPELINE-CONVENTION values (rate * AFTER_TAX) ----
        p_cagr_is = pipe_cagr_split(nav, is_mask)
        p_cagr_oos = pipe_cagr_split(nav, oos_mask)
        p_w10, _ = pipe_worst10y(dates, nav)
        p_w5, p_p10 = pipe_worst5y_and_p10(nav)
        pipe[name] = dict(
            CAGR_IS=p_cagr_is * 100, CAGR_OOS=p_cagr_oos * 100,
            Sharpe_FULL=sh, MaxDD=mdd * 100,
            Worst10Y=p_w10 * 100, Worst5Y=p_w5 * 100,
            P10_5Y=p_p10 * 100,
            IS_OOS_gap=(p_cagr_is - p_cagr_oos) * 100, Trades=tpy,
        )
        extra[name] = dict(
            CAGR_IS_daily=cagr_is_daily * 100, CAGR_OOS_daily=cagr_oos_daily * 100,
            Worst1D=w1 * 100, Worst1D_date=w1date,
            Worst5Y_daily=worst5y_daily * 100, P10_5Y_daily=p10_5y_daily * 100,
            n_is_years=len(is_years), n_oos_years=len(oos_years),
            is_years_range=(is_years[0], is_years[-1]),
            oos_years_range=(oos_years[0], oos_years[-1]),
            worst10y_win=worst10y_win, worst5y_cal_win=worst5y_cal_win,
            n_is_days=len(nav_is), n_oos_days=len(nav_oos),
        )

    # ----- Comparison table -----
    metric_order = ["CAGR_IS", "CAGR_OOS", "Sharpe_FULL", "MaxDD",
                    "Worst10Y", "Worst5Y", "P10_5Y", "IS_OOS_gap", "Trades"]
    for name in CANDIDATES:
        print(f"\n### Candidate {name}")
        print(f"{'metric':<12} {'mine':>12} {'reported':>12} {'abs_diff':>11}  class")
        print("-" * 64)
        rep = REPORTED[name]
        for mtr in metric_order:
            if mtr not in rep:
                continue
            myv = results[name][mtr]
            rpv = rep[mtr]
            diff = abs(myv - rpv)
            if mtr == "Trades":
                cls = "SMALL" if diff <= 0.1 else ("MEDIUM" if diff <= 0.3 else "LARGE")
            elif mtr == "IS_OOS_gap":
                cls = classify("CAGR", diff)
            else:
                cls = classify(mtr, diff)
            print(f"{mtr:<12} {myv:>12.4f} {rpv:>12.4f} {diff:>11.4f}  {cls}")

        e = extra[name]
        print(f"  [cross-check] CAGR_IS_daily(exact split)={e['CAGR_IS_daily']:.4f}%  "
              f"CAGR_OOS_daily(exact split)={e['CAGR_OOS_daily']:.4f}%")
        print(f"  [cross-check] Worst1D={e['Worst1D']:.4f}% on {e['Worst1D_date']}  "
              f"Worst5Y_daily(pre-tax 1260)={e['Worst5Y_daily']:.4f}%  "
              f"P10_5Y_daily(pre-tax 1260)={e['P10_5Y_daily']:.4f}%")
        print(f"  [info] IS years={e['is_years_range']} (n={e['n_is_years']}), "
              f"OOS years={e['oos_years_range']} (n={e['n_oos_years']}); "
              f"IS days={e['n_is_days']}, OOS days={e['n_oos_days']}")
        if e['worst10y_win']:
            w = e['worst10y_win']
            print(f"  [info] Worst10Y window: {w[0]}-{w[1]} = {w[2]*100:.4f}%")
        if e['worst5y_cal_win']:
            w = e['worst5y_cal_win']
            print(f"  [info] Worst5Y(cal) window: {w[0]}-{w[1]} = {w[2]*100:.4f}%")

    # ===== SECOND TABLE: pipeline-convention reconstruction vs reported =====
    print("\n\n" + "#" * 64)
    print("# TABLE 2: PIPELINE-CONVENTION reconstruction (rate * AFTER_TAX) vs reported")
    print("# (after-tax = multiply the PRETAX annualized RATE by 0.8273)")
    print("# Worst10Y = calendar-year 10y geo; Worst5Y/P10 = daily 1260-day rolling")
    print("#" * 64)
    for name in CANDIDATES:
        print(f"\n### Candidate {name} [pipeline convention]")
        print(f"{'metric':<12} {'mine_pipe':>12} {'reported':>12} {'abs_diff':>11}  class")
        print("-" * 64)
        rep = REPORTED[name]
        for mtr in metric_order:
            if mtr not in rep:
                continue
            myv = pipe[name][mtr]
            rpv = rep[mtr]
            diff = abs(myv - rpv)
            if mtr == "Trades":
                cls = "SMALL" if diff <= 0.1 else ("MEDIUM" if diff <= 0.3 else "LARGE")
            elif mtr == "IS_OOS_gap":
                cls = classify("CAGR", diff)
            else:
                cls = classify(mtr, diff)
            print(f"{mtr:<12} {myv:>12.4f} {rpv:>12.4f} {diff:>11.4f}  {cls}")

    # ----- X4 vs N4 relationship -----
    print("\n" + "=" * 64)
    print("### X4 vs N4 relationship (my independent values, pipeline convention)")
    for mtr in ["CAGR_OOS", "CAGR_IS", "MaxDD", "Sharpe_FULL", "Worst10Y", "Worst5Y", "P10_5Y"]:
        x = pipe["X4"][mtr]
        nn = pipe["N4"][mtr]
        print(f"  {mtr:<12} X4={x:>10.4f}  N4={nn:>10.4f}  (X4-N4)={x-nn:>+8.4f}")

    print("\n### X4 vs N4 relationship (my independent values, SPEC-LITERAL convention)")
    for mtr in ["CAGR_OOS", "CAGR_IS", "MaxDD", "Sharpe_FULL", "Worst10Y", "Worst5Y", "P10_5Y"]:
        x = results["X4"][mtr]
        nn = results["N4"][mtr]
        print(f"  {mtr:<12} X4={x:>10.4f}  N4={nn:>10.4f}  (X4-N4)={x-nn:>+8.4f}")

    print("\n[reported] X4-N4 deltas:")
    for mtr in ["CAGR_OOS", "CAGR_IS", "MaxDD"]:
        print(f"  {mtr:<12} X4={REPORTED['X4'][mtr]:>10.4f}  N4={REPORTED['N4'][mtr]:>10.4f}  "
              f"(X4-N4)={REPORTED['X4'][mtr]-REPORTED['N4'][mtr]:>+8.4f}")


if __name__ == "__main__":
    main()
