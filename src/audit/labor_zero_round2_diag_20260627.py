"""
src/audit/labor_zero_round2_diag_20260627.py
============================================
Round-2 DIAGNOSTIC for the labor-zero problem. Two jobs:
  1. Print the after-tax annual return matrix (1988-2007 window + full) for every
     strategy + reserve sleeve actually loaded, so the analyst's prompt numbers are
     verified against the live data (not trusted blind).
  2. Trace the 1988 start year-by-year under the prior best (sc2.2, run20/res20 cash,
     thr20, ALL) to confirm the KNOWN MECHANISM (run grinds to 0 by ~1996).
ASCII-only. No commit, no temp files.
"""
from __future__ import annotations
import os, sys, importlib.util
import numpy as np, pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "h2", os.path.join(_THIS, "labor_zero_harness_v2_20260627.py"))
h2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h2)
M = h2.M


def trace_start(rets, sleeves, start, *, single="sc2.2", run0=20 * M, reserve0=20 * M,
                reserve_mode="cash", topup_thr=20 * M, topup_amt=None,
                glide=None, mix=None, init_bucket_years=0, draw_order="run_first",
                spend=h2.SPEND, horizon=h2.HORIZON, regime_scale=None):
    run = float(run0); reserve = float(reserve0); bucket = init_bucket_years * spend
    sret = sleeves[reserve_mode]
    print("  yr  | k | run_pre  topup  spend_from        run_post  reserve  bucket  ret%   | labor")
    labor = 0
    for k in range(horizon):
        yr = start + k
        tu = 0.0
        if run < topup_thr and reserve > 1e-6:
            move = reserve if topup_amt is None else min(topup_amt, reserve)
            run += move; reserve -= move; tu = move
        total = run + reserve + bucket
        src_desc = ""
        lab = 0
        if total + 1e-6 < spend:
            labor += 1; lab = 1; run = reserve = bucket = 0.0
        else:
            need = spend
            take = min(bucket, need); bucket -= take; need -= take
            if take > 0: src_desc += "bkt%.1f " % (take / M)
            if need > 1e-9:
                r_this = h2._strat_year_return(rets, mix, glide, single, yr, k)
                down = r_this < 0
                if draw_order == "reserve_first_on_down" and down:
                    take = min(reserve, need); reserve -= take; need -= take
                    if take > 0: src_desc += "res%.1f " % (take / M)
                    if need > 1e-9:
                        take = min(run, need); run -= take; need -= take
                        if take > 0: src_desc += "run%.1f " % (take / M)
                else:
                    take = min(run, need); run -= take; need -= take
                    if take > 0: src_desc += "run%.1f " % (take / M)
                    if need > 1e-9:
                        take = min(reserve, need); reserve -= take; need -= take
                        if take > 0: src_desc += "res%.1f " % (take / M)
        r_this = h2._strat_year_return(rets, mix, glide, single, yr, k)
        if regime_scale is not None:
            s = float(np.clip(regime_scale[k], 0.0, 2.0))
            cash_r = float(sleeves["sofr"].loc[yr])
            r_this = cash_r + s * (r_this - cash_r)
        run_pre_growth = run
        run *= (1.0 + r_this)
        reserve *= (1.0 + float(sret.loc[yr]))
        print("  %4d | %d | %7.2f  %5.1f  %-16s %7.2f  %6.2f  %5.2f  %+6.1f | %d"
              % (yr, k, (run_pre_growth + (spend if not lab else 0)) / M if False else run_pre_growth / M,
                 tu / M, src_desc.strip(), run / M, reserve / M, bucket / M, r_this * 100, lab))
    print("  => labor=%d terminal=%.1fM" % (labor, (run + reserve + bucket) / M))
    return labor


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rets = h2.load_returns(); sleeves = h2.load_sleeve_returns(); rets.update(h2.load_extended())

    yrs = list(range(1988, 2008))
    print("=" * 110)
    print("AFTER-TAX ANNUAL RETURN MATRIX  (verify prompt table vs live data)")
    print("=" * 110)
    cols = ["sc1.6", "sc2.0", "sc2.2", "sc2.4", "sc2.6", "N4", "X4"]
    scols = ["bond", "gold", "sofr", "nasdaq"]
    hdr = "year | " + " ".join("%7s" % c for c in cols) + " || " + " ".join("%7s" % c for c in scols)
    print(hdr); print("-" * len(hdr))
    for y in yrs:
        line = "%4d |" % y
        for c in cols:
            v = rets[c].loc[y] * 100 if (c in rets and y in rets[c].index) else float("nan")
            line += " %7.1f" % v
        line += " ||"
        for c in scols:
            v = sleeves[c].loc[y] * 100 if y in sleeves[c].index else float("nan")
            line += " %7.1f" % v
        print(line)

    print("\n" + "=" * 110)
    print("TRACE 1988 START -- prior best (sc2.2 run20/res20 cash thr20 ALL)")
    print("=" * 110)
    trace_start(rets, sleeves, 1988)

    # full-range CAGR + worst-year per strategy (context for trade-offs)
    print("\n" + "=" * 110)
    print("STRATEGY STATS 1975-2025 (after-tax annual): CAGR, worst yr, #neg yrs")
    print("=" * 110)
    for c in cols + scols:
        s = (rets[c] if c in rets else sleeves[c]).dropna()
        s = s[(s.index >= 1975) & (s.index <= 2025)]
        cagr = (np.prod(1 + s.values) ** (1 / len(s)) - 1) * 100
        print("  %-8s CAGR=%+6.2f%%  worstYr=%+6.1f%% (%d)  negYrs=%d/%d  mean=%+6.1f%%"
              % (c, cagr, s.min() * 100, int(s.idxmin()), int((s < 0).sum()), len(s), s.mean() * 100))


if __name__ == "__main__":
    main()
