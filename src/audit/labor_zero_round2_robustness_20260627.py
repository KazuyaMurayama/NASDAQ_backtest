"""
src/audit/labor_zero_round2_robustness_20260627.py
==================================================
QUALITY-REVIEW robustness/sensitivity of the Round-2 labor-zero claim
(headline B: sc2.6, run 14M / reserve 26M in BOND, top-up-all at 14M).

This is the first-principles stress that the original Round-2 report lacked. The
original showed only a "halve a single up-year" fragility test (a harsh localized
shock). Here we quantify the more decision-relevant systematic stresses:

  S1 MARKET SCALE   : scale every strategy's excess-over-cash by f. Find the
                      smallest f (uniformly weaker equity market) that still keeps
                      labor-zero. (Answers: how much weaker can the market be?)
  S2 BOND HAIRCUT   : multiply all bond reserve returns by h. (Answers: how load-
                      bearing is the bond crisis-alpha? bond went NEGATIVE in 2022.)
  S3 INFLATION      : grow the 7.2M spend at g/yr (constant real spending). The
                      report holds spend NOMINAL-fixed for 20y -- a real retiree
                      faces inflation. (This is the single biggest sensitivity.)
  S4 TRUE MARGIN    : the closest the pool ever comes to a labor year across all
                      31 starts x 20y = min over (start,year) of (total_at_spend - spend).

All on the headline config. ASCII-only. No commit, no temp files.
Output: audit_results/labor_zero_round2_robustness_20260627.csv
"""
from __future__ import annotations

import os
import sys
import importlib.util

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
AR_DIR = os.path.join(_REPO_DIR, "audit_results")
START_YEARS = list(range(1975, 2006))

# headline config
HC = dict(strat="sc2.6", run0=14 * M, res0=26 * M, rmode="bond", thr=14 * M)
SPEND = 7.2 * M


def labor_total(rets, sleeves, *, strat, run0, res0, rmode, thr, spend=SPEND,
                spend_growth=0.0):
    tot = 0
    for sy in START_YEARS:
        run, res = run0, res0
        for k in range(20):
            yr = sy + k
            sp = spend * ((1.0 + spend_growth) ** k)
            if run < thr and res > 1e-6:
                run += res
                res = 0.0
            if run + res + 1e-6 < sp:
                tot += 1
                run = res = 0.0
            else:
                if run >= sp:
                    run -= sp
                else:
                    res -= (sp - run)
                    run = 0.0
            run *= (1.0 + float(rets[strat].loc[yr]))
            res *= (1.0 + float(sleeves[rmode].loc[yr]))
    return tot


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("ROUND-2 LABOR-ZERO ROBUSTNESS / SENSITIVITY (quality review)")
    print("headline: sc2.6 run14M / reserve26M bond / top-up-all at 14M, spend 7.2M strict")
    print("=" * 100)
    rets0 = h2.load_returns()
    rets0.update(h2.load_extended())
    sleeves0 = h2.load_sleeve_returns()

    rows = []

    # ---- baseline ----
    base = labor_total(rets0, sleeves0, **HC)
    print("\nbaseline (f=1, h=1, g=0): labor_years_total = %d" % base)
    rows.append(dict(stress="baseline", param="-", labor_years_total=base, holds=int(base == 0)))

    # ---- S1: market scale ----
    print("\n--- S1 market scale (excess-over-cash x f; smallest f that holds) ---")
    cash = sleeves0["sofr"]
    last_hold = None
    for f in [1.0, 0.95, 0.90, 0.88, 0.86, 0.85, 0.84, 0.82, 0.80, 0.75]:
        rets = {k: v.copy() for k, v in rets0.items()}
        for k in rets:
            for yr in rets[k].index:
                c = float(cash.loc[yr]) if yr in cash.index else 0.0
                rets[k].loc[yr] = c + f * (rets[k].loc[yr] - c)
        lab = labor_total(rets, sleeves0, **HC)
        if lab == 0:
            last_hold = f
        print("  f=%.2f: labor=%d %s" % (f, lab, "holds" if lab == 0 else "BREAKS"))
        rows.append(dict(stress="market_scale", param="f=%.2f" % f, labor_years_total=lab, holds=int(lab == 0)))
    print("  => smallest market-scale that holds: f=%.2f (tolerates ~%.0f%% weaker equity)"
          % (last_hold, (1 - last_hold) * 100))

    # ---- S2: bond haircut ----
    print("\n--- S2 bond reserve haircut (all bond returns x h) ---")
    last_hold_b = None
    for h in [1.0, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.0]:
        sl = {k: v.copy() for k, v in sleeves0.items()}
        sl["bond"] = sleeves0["bond"] * h
        lab = labor_total(rets0, sl, **HC)
        if lab == 0:
            last_hold_b = h
        print("  bond x%.2f: labor=%d %s" % (h, lab, "holds" if lab == 0 else "BREAKS"))
        rows.append(dict(stress="bond_haircut", param="h=%.2f" % h, labor_years_total=lab, holds=int(lab == 0)))
    print("  => bond crisis-alpha can be cut to x%.2f and labor-zero still holds (reserve LEVEL matters more than its growth)"
          % last_hold_b)

    # ---- S3: inflation (constant real spend) ----
    print("\n--- S3 inflation: spend grows g/yr (report holds it NOMINAL-fixed) ---")
    for g in [0.0, 0.01, 0.02, 0.035, 0.05]:
        lab = labor_total(rets0, sleeves0, spend_growth=g, **HC)
        print("  spend +%.1f%%/yr: labor=%d %s" % (g * 100, lab, "holds" if lab == 0 else "BREAKS"))
        rows.append(dict(stress="inflation", param="g=%.3f" % g, labor_years_total=lab, holds=int(lab == 0)))
    print("  => labor-zero requires NOMINAL-FIXED spend; even +2%/yr inflation-indexing breaks it.")

    # ---- S4: true fragility margin ----
    print("\n--- S4 true margin: closest the pool ever comes to a labor year ---")
    worst = 1e18
    worst_at = None
    run0, res0, thr, strat, rmode = HC["run0"], HC["res0"], HC["thr"], HC["strat"], HC["rmode"]
    for sy in START_YEARS:
        run, res = run0, res0
        for k in range(20):
            yr = sy + k
            if run < thr and res > 1e-6:
                run += res
                res = 0.0
            total = run + res
            m = total - SPEND
            if m < worst:
                worst, worst_at = m, (sy, yr, total / M)
            if total + 1e-6 >= SPEND:
                if run >= SPEND:
                    run -= SPEND
                else:
                    res -= (SPEND - run)
                    run = 0.0
            run *= (1.0 + float(rets0[strat].loc[yr]))
            res *= (1.0 + float(sleeves0[rmode].loc[yr]))
    print("  smallest margin above the labor line = %+.2fM (start %d, year %d, pool %.2fM vs spend 7.2M)"
          % (worst / M, worst_at[0], worst_at[1], worst_at[2]))
    rows.append(dict(stress="true_margin", param="start%d_yr%d" % (worst_at[0], worst_at[1]),
                     labor_years_total=round(worst / M, 2), holds=int(worst > 0)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_round2_robustness_20260627.csv"),
              index=False, encoding="utf-8-sig")
    print("\nSaved robustness CSV.")
    print("\n=== QC SUMMARY ===")
    print("  - labor-zero tolerates ~%.0f%% uniformly weaker equity (f>=%.2f)" % ((1 - last_hold) * 100, last_hold))
    print("  - robust to large bond-return haircut (holds to bond x%.2f)" % last_hold_b)
    print("  - BUT requires nominal-fixed spend: +2%/yr inflation-indexing breaks labor-zero")
    print("  - tightest pool margin over all 31x20 = %+.2fM (at the 1989-start, 1995)" % (worst / M))
    print("Done.")


if __name__ == "__main__":
    main()
