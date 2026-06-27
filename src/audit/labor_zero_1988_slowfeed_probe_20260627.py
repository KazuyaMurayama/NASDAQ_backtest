"""
src/audit/labor_zero_1988_slowfeed_probe_20260627.py
====================================================
Focused probe of the 1988-start bottleneck for the labor-zero problem
(40M assets, strict 7.2M/yr spend). Round-1 found NO single lever saves 1988.

This probe tests the RESERVE-ASSET hypothesis specifically:
  A LARGER, BOND-invested reserve deployed SLOWLY (small fixed top-up chunks +
  lower threshold) may survive 1988 by FEEDING the run sleeve gradually through
  the choppy 1990(-30)/1993(+3.6)/1994(-9.2) grind, instead of dumping all 20M
  into run in year 2 (where it then gets ground to zero by 1996, missing the
  1995-99 boom).

It does three things:
  1. Instruments the baseline 1988 path (sc2.2, run20/res20 CASH, thr20, amt=ALL)
     year-by-year to confirm the ruin mechanism (run -> 0 by ~1996).
  2. Tests a focused grid of slow-feed BOND-reserve configs varying:
       - reserve asset  : bond | sofr | bond/gold mix | bond/sofr mix
       - run0/reserve0  : 12/28, 15/25, 18/22, 20/20
       - topup_thr      : 6,8,10,12 M  (lower = feed only when run is small)
       - topup_amt      : 3,4,5,6 M fixed chunks vs ALL
     evaluated on the 1988 start ALONE (labor years for that start) AND across
     all 31 starts (does it break other starts / what is the cost in growth?).
  3. Prints the 1988 path for the top slow-feed configs so the mechanism is
     visible (does the run sleeve survive the grind and catch 1995-99?).

ASCII-only. No commit, no temp files. Reuses harness v2 (simulate_v2) so the
spend/top-up/growth convention is identical and self-test-verified.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)

spec = importlib.util.spec_from_file_location(
    "h2", os.path.join(_THIS_DIR, "labor_zero_harness_v2_20260627.py"))
h2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h2)
M = h2.M
SPEND = h2.SPEND
AR_DIR = os.path.join(_REPO_DIR, "audit_results")


def _mix_sleeve(sleeves, parts):
    """Build a synthetic blended reserve sleeve Series from {asset: weight}."""
    years = sleeves["bond"].index
    out = pd.Series(0.0, index=years)
    for a, w in parts.items():
        out = out + w * sleeves[a]
    return out


def trace_1988(rets, sleeves, *, single, run0, reserve0, reserve_mode,
               topup_thr, topup_amt, label, sleeve_override=None, horizon=20):
    """Re-implement the simulate_v2 inner loop for one start with full tracing.
    Mirrors harness v2 exactly (run_first draw, top-up before spend, growth after)
    so the printed path is the same engine the grid scores."""
    start = 1988
    run = float(run0)
    reserve = float(reserve0)
    sleeve_ret = sleeve_override if sleeve_override is not None else sleeves[reserve_mode]
    labor = 0
    rows = []
    for k in range(horizon):
        yr = start + k
        topped = 0.0
        if run < topup_thr and reserve > 1e-6:
            move = reserve if topup_amt is None else min(topup_amt, reserve)
            run += move
            reserve -= move
            topped = move
        total = run + reserve
        lab = 0
        if total + 1e-6 < SPEND:
            lab = 1
            labor += 1
            run = reserve = 0.0
        else:
            need = SPEND
            take = min(run, need); run -= take; need -= take
            if need > 1e-9:
                take = min(reserve, need); reserve -= take; need -= take
        r_this = float(rets[single].loc[yr])
        run_pre = run
        run *= (1.0 + r_this)
        reserve *= (1.0 + float(sleeve_ret.loc[yr]))
        rows.append((yr, r_this * 100, topped / M, run_pre / M, run / M,
                     reserve / M, (run + reserve) / M, lab))
    print("\n  1988 PATH -- %s" % label)
    print("  yr   strat%%  topup  run(pre)  run(post)  reserve  total  LABOR")
    for (yr, rp, tp, rpre, rpost, res, tot, lab) in rows:
        print("  %4d %7.1f %6.1f %9.2f %10.2f %8.2f %6.2f %5s"
              % (yr, rp, tp, rpre, rpost, res, tot, "***" if lab else ""))
    print("  -> labor(1988)=%d  terminal=%.1fM" % (labor, run + reserve))
    return labor


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("1988 BOTTLENECK SLOW-FEED PROBE -- bond reserve, gradual top-up (labor-zero)")
    print("=" * 100)
    rets = h2.load_returns()
    sleeves = h2.load_sleeve_returns()
    rets.update(h2.load_extended())

    # synthetic blended reserve sleeves
    sleeves["bond75gold25"] = _mix_sleeve(sleeves, {"bond": 0.75, "gold": 0.25})
    sleeves["bond50gold50"] = _mix_sleeve(sleeves, {"bond": 0.50, "gold": 0.50})
    sleeves["bond50sofr50"] = _mix_sleeve(sleeves, {"bond": 0.50, "sofr": 0.50})

    # ---------- (1) baseline 1988 path (confirm ruin mechanism) ----------
    print("\n" + "-" * 100)
    print("(1) BASELINE 1988 PATH  (sc2.2, run20/res20 CASH, thr20, amt=ALL)")
    print("-" * 100)
    trace_1988(rets, sleeves, single="sc2.2", run0=20 * M, reserve0=20 * M,
               reserve_mode="cash", topup_thr=20 * M, topup_amt=None,
               label="BASELINE sc2.2 cash res, dump-all at thr20")

    # also show: bond reserve but STILL dump-all (lever A best) -- why it still fails
    trace_1988(rets, sleeves, single="sc2.2", run0=20 * M, reserve0=20 * M,
               reserve_mode="bond", topup_thr=20 * M, topup_amt=None,
               label="LEVER-A sc2.2 BOND res, dump-all at thr20 (round1 best A)")

    # ---------- (2) slow-feed grid ----------
    print("\n" + "-" * 100)
    print("(2) SLOW-FEED GRID  (bond/mix reserve, small chunks, low threshold)")
    print("-" * 100)
    rows = []
    SINGLES = ["sc2.2", "sc2.0", "sc1.6"]
    SPLITS = [(12, 28), (15, 25), (18, 22), (20, 20)]
    THRS = [6 * M, 8 * M, 10 * M, 12 * M, 14 * M]
    AMTS = [("3M", 3 * M), ("4M", 4 * M), ("5M", 5 * M), ("6M", 6 * M),
            ("8M", 8 * M), ("ALL", None)]
    MODES = ["bond", "sofr", "bond75gold25", "bond50gold50", "bond50sofr50"]

    for single in SINGLES:
        for (r0, rs0) in SPLITS:
            for mode in MODES:
                ov = sleeves[mode] if mode in ("bond75gold25", "bond50gold50",
                                               "bond50sofr50") else None
                rmode = "bond" if ov is not None else mode  # placeholder; override used
                for thr in THRS:
                    for atag, amt in AMTS:
                        # 1988-only labor via traced engine == simulate_v2; use sim for speed
                        if ov is None:
                            res88 = h2.simulate_v2(rets, sleeves, 1988, single=single,
                                                   run0=r0 * M, reserve0=rs0 * M,
                                                   reserve_mode=mode, topup_thr=thr,
                                                   topup_amt=amt)
                            lab88 = res88["labor_years"]
                            # all starts
                            agg = h2.run_all_starts(rets, sleeves, single=single,
                                                    run0=r0 * M, reserve0=rs0 * M,
                                                    reserve_mode=mode, topup_thr=thr,
                                                    topup_amt=amt)
                        else:
                            # blended reserve: temporarily install override into sleeves dict
                            sleeves["_ov"] = ov
                            res88 = h2.simulate_v2(rets, sleeves, 1988, single=single,
                                                   run0=r0 * M, reserve0=rs0 * M,
                                                   reserve_mode="_ov", topup_thr=thr,
                                                   topup_amt=amt)
                            lab88 = res88["labor_years"]
                            agg = h2.run_all_starts(rets, sleeves, single=single,
                                                    run0=r0 * M, reserve0=rs0 * M,
                                                    reserve_mode="_ov", topup_thr=thr,
                                                    topup_amt=amt)
                        rows.append(dict(
                            single=single, run_M=r0, reserve_M=rs0, reserve_mode=mode,
                            topup_thr_M=thr / M, topup_amt=atag,
                            labor_1988=lab88,
                            labor_total=agg["labor_years_total"],
                            starts_with_labor=agg["starts_with_labor"],
                            saved_1988=int(agg["saved_1988"]),
                            fails=";".join(str(x) for x in agg["fails"]),
                            ruin_total=agg["ruin_total"],
                            terminal_median_M=round(agg["terminal_median_M"], 1),
                            min_total_floor_M=round(agg["min_total_floor_M"], 2)))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(AR_DIR, "labor_zero_1988_slowfeed_probe_20260627.csv")
    df.to_csv(out_csv, index=False, float_format="%.3f", encoding="utf-8-sig")
    print("Saved slow-feed grid: %d configs -> %s" % (len(df), os.path.basename(out_csv)))

    # ---- did ANY config save 1988 (labor_1988 == 0)? ----
    saved = df[df["labor_1988"] == 0].sort_values(
        ["labor_total", "terminal_median_M"], ascending=[True, False])
    print("\n*** configs that SAVE the 1988 start (labor_1988==0): %d ***" % len(saved))
    cols = ["single", "run_M", "reserve_M", "reserve_mode", "topup_thr_M",
            "topup_amt", "labor_1988", "labor_total", "starts_with_labor",
            "fails", "terminal_median_M", "min_total_floor_M"]
    if len(saved):
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(saved[cols].head(25).to_string(index=False))

    # ---- overall best by total labor (even if 1988 not fully saved) ----
    print("\n--- TOP 15 by labor_total (then terminal) across ALL starts ---")
    top = df.sort_values(["labor_total", "terminal_median_M"],
                         ascending=[True, False]).head(15)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(top[cols].to_string(index=False))

    # ---- best 1988-only reduction (min labor_1988) ----
    print("\n--- min labor_1988 achieved (1988 start in isolation) ---")
    print("min labor_1988 = %d  (baseline = 1)" % int(df["labor_1988"].min()))
    best88 = df.sort_values(["labor_1988", "labor_total", "terminal_median_M"],
                            ascending=[True, True, False]).head(8)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(best88[cols].to_string(index=False))

    # ---------- (3) trace the 1988 path for the most promising slow-feed configs ----------
    print("\n" + "-" * 100)
    print("(3) 1988 PATH for promising slow-feed configs (mechanism visibility)")
    print("-" * 100)
    # pick a few representative slow-feed configs to visualize
    probes = [
        dict(single="sc2.2", run0=18 * M, reserve0=22 * M, reserve_mode="bond",
             topup_thr=8 * M, topup_amt=4 * M,
             label="sc2.2 run18/bond22 thr8 chunk4M (slow feed)"),
        dict(single="sc2.2", run0=15 * M, reserve0=25 * M, reserve_mode="bond",
             topup_thr=10 * M, topup_amt=5 * M,
             label="sc2.2 run15/bond25 thr10 chunk5M (slow feed)"),
        dict(single="sc2.2", run0=20 * M, reserve0=20 * M, reserve_mode="bond",
             topup_thr=10 * M, topup_amt=4 * M,
             label="sc2.2 run20/bond20 thr10 chunk4M (slow feed)"),
        dict(single="sc1.6", run0=18 * M, reserve0=22 * M, reserve_mode="bond",
             topup_thr=8 * M, topup_amt=4 * M,
             label="sc1.6 run18/bond22 thr8 chunk4M (lower-lev slow feed)"),
    ]
    for p in probes:
        trace_1988(rets, sleeves, single=p["single"], run0=p["run0"],
                   reserve0=p["reserve0"], reserve_mode=p["reserve_mode"],
                   topup_thr=p["topup_thr"], topup_amt=p["topup_amt"],
                   label=p["label"])

    print("\nDone.")
    return df


if __name__ == "__main__":
    main()
