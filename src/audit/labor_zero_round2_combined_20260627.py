"""
src/audit/labor_zero_round2_combined_20260627.py
================================================
ROUND 2: combined-lever search for labor-backfill-zero (40M assets, strict 7.2M/yr
spend, 31 starts 1975-2005 x 20yr). Round 1 found NO single lever achieves labor-zero
and none even saves the binding 1988 start. This script:

  (1) Traces the 1988 window year-by-year for the prior best and for each candidate,
      exposing WHEN reserve empties vs WHEN the bad years (1988/1990/1994) hit.
  (2) Tests specific COMBINED-lever configs (glide-K + bond-reserve + bucket-N +
      topup params + strategy) most likely to keep total wealth > 7.2M every year of
      the 1988 window.
  (3) Runs a focused combined grid and reports any labor-zero (0 labor, 0 ruin)
      config, ranked by floor wealth in the 1988 window.

Reuses simulate_v2 / run_all_starts from labor_zero_harness_v2_20260627 (self-tested
to reproduce the prior best: sc2.2 run20/res20 cash thr20 ALL -> 12 labor, 1988 only).

ASCII-only. No commit, no temp files.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_THIS); _REPO = os.path.dirname(_SRC)
for _p in (_SRC, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.audit.labor_zero_harness_v2_20260627 as V2

M = V2.M; SPEND = V2.SPEND; HORIZON = V2.HORIZON


def trace_1988(rets, sleeves, *, label, **kw):
    """Year-by-year trace of the 1988 start (the binding window 1988-2007)."""
    single = kw.get("single"); glide = kw.get("glide"); mix = kw.get("mix")
    run = float(kw["run0"]); reserve = float(kw["reserve0"])
    bucket = float(kw.get("init_bucket_years", 0)) * SPEND
    mode = kw.get("reserve_mode", "cash"); sret = sleeves[mode]
    thr = kw.get("topup_thr", 20 * M); amt = kw.get("topup_amt", None)
    draw_order = kw.get("draw_order", "run_first")
    rows = []
    labor = 0
    for k in range(HORIZON):
        yr = 1988 + k
        ev = []
        if run < thr and reserve > 1e-6:
            move = reserve if amt is None else min(amt, reserve)
            run += move; reserve -= move; ev.append("topup+%.0f" % (move / M))
        total = run + reserve + bucket
        r_this = V2._strat_year_return(rets, mix, glide, single, yr, k)
        if total + 1e-6 < SPEND:
            labor += 1; ev.append("LABOR"); run = reserve = bucket = 0.0
        else:
            need = SPEND
            take = min(bucket, need); bucket -= take; need -= take
            if take > 1e-9:
                ev.append("buk-%.1f" % (take / M))
            if need > 1e-9:
                if draw_order == "reserve_first_on_down" and r_this < 0:
                    t = min(reserve, need); reserve -= t; need -= t
                    if t > 1e-9:
                        ev.append("res-%.1f" % (t / M))
                    if need > 1e-9:
                        t = min(run, need); run -= t; need -= t; ev.append("run-%.1f" % (t / M))
                else:
                    t = min(run, need); run -= t; need -= t
                    if need > 1e-9:
                        tr = min(reserve, need); reserve -= tr; need -= tr
                        ev.append("run-%.1f+res-%.1f" % (t / M, tr / M))
        run *= (1.0 + r_this)
        reserve *= (1.0 + float(sret.loc[yr]))
        rows.append((yr, r_this * 100, run / M, reserve / M, bucket / M,
                     (run + reserve + bucket) / M, ";".join(ev)))
    print("\n--- 1988 TRACE: %s ---" % label)
    print(" yr   ret%%   run(M) res(M) buk(M) TOT(M)  events")
    for (yr, r, ru, re, bu, to, ev) in rows:
        print("%4d %7.1f %7.2f %6.2f %6.2f %7.2f  %s" % (yr, r, ru, re, bu, to, ev))
    print("  -> 1988-start labor years = %d" % labor)
    return labor


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("ROUND 2 COMBINED-LEVER SEARCH  (40M, 7.2M/yr strict, 31 starts)")
    print("=" * 100)
    rets = V2.load_returns(); sleeves = V2.load_sleeve_returns()
    ext = V2.load_extended()
    rets.update(ext)  # add sc2.4, sc2.6

    # ---- self-test passthrough ----
    base = V2.run_all_starts(rets, sleeves, single="sc2.2", run0=20 * M,
                             reserve0=20 * M, reserve_mode="cash",
                             topup_thr=20 * M, topup_amt=None)
    print("PRIOR BEST sc2.2 run20/res20 cash thr20 ALL: labor=%d fails=%s termMed=%.0fM"
          % (base["labor_years_total"], base["fails"], base["terminal_median_M"]))

    # ---- trace prior best on 1988 ----
    trace_1988(rets, sleeves, label="PRIOR BEST sc2.2 run20/res20 CASH thr20 ALL",
               single="sc2.2", run0=20 * M, reserve0=20 * M, reserve_mode="cash",
               topup_thr=20 * M, topup_amt=None)
    # ---- trace single bond-reserve (round1 A best) ----
    trace_1988(rets, sleeves, label="A: sc2.2 run20/res20 BOND thr20 ALL",
               single="sc2.2", run0=20 * M, reserve0=20 * M, reserve_mode="bond",
               topup_thr=20 * M, topup_amt=None)

    # ============================================================
    # CANDIDATE COMBINED CONFIGS (the deliverable)
    # ============================================================
    glide_2y = [(0, "sc1.6"), (2, "sc2.2")]      # protect 1988-89, then high lev
    glide_3y = [(0, "sc1.6"), (3, "sc2.2")]
    glide_2y_lowlow = [(0, "sc1.4"), (2, "sc2.2")]
    glide_3y_to24 = [(0, "sc1.6"), (3, "sc2.4")]

    configs = {
      # name: kwargs
      "C1 glide2y(sc1.6->2.2)+BOND res20/run20+thr25+ALL":
        dict(glide=glide_2y, run0=20 * M, reserve0=20 * M, reserve_mode="bond",
             topup_thr=25 * M, topup_amt=None),
      "C2 glide2y+BOND+bucket1y res19/run19/buk7.2 thr20 ALL":
        dict(glide=glide_2y, run0=19 * M, reserve0=19 * M, reserve_mode="bond",
             init_bucket_years=1, topup_thr=20 * M, topup_amt=None),
      "C3 glide3y(sc1.6->2.2)+BOND res22/run18 thr22 ALL":
        dict(glide=glide_3y, run0=18 * M, reserve0=22 * M, reserve_mode="bond",
             topup_thr=22 * M, topup_amt=None),
      "C4 glide2y+BOND+E(resfirst-down) res20/run20 thr25 ALL":
        dict(glide=glide_2y, run0=20 * M, reserve0=20 * M, reserve_mode="bond",
             draw_order="reserve_first_on_down", topup_thr=25 * M, topup_amt=None),
      "C5 glide3y+BOND+bucket2y res18/run16/buk14.4 thr20 ALL":
        dict(glide=glide_3y, run0=16 * M, reserve0=18 * M, reserve_mode="bond",
             init_bucket_years=2, topup_thr=20 * M, topup_amt=None),
      "C6 glide2y_low(sc1.4->2.2)+BOND res20/run20 thr25 ALL":
        dict(glide=glide_2y_lowlow, run0=20 * M, reserve0=20 * M, reserve_mode="bond",
             topup_thr=25 * M, topup_amt=None),
      "C7 sc1.6 first3y->sc2.4 + BOND res22/run18 thr22 ALL (more lev later)":
        dict(glide=glide_3y_to24, run0=18 * M, reserve0=22 * M, reserve_mode="bond",
             topup_thr=22 * M, topup_amt=None),
      "C8 glide3y+BOND+E+bucket1y res20/run18/buk7.2 thr22 ALL":
        dict(glide=glide_3y, run0=18 * M, reserve0=20 * M, reserve_mode="bond",
             init_bucket_years=1, draw_order="reserve_first_on_down",
             topup_thr=22 * M, topup_amt=None),
    }

    print("\n" + "=" * 100)
    print("CANDIDATE COMBINED CONFIGS -- full 31-start eval")
    print("=" * 100)
    print("%-58s | laborY fails(n) saved88 ruin termMed(M) floor(M)" % "config")
    results = {}
    for name, kw in configs.items():
        r = V2.run_all_starts(rets, sleeves, topup_thr=kw.get("topup_thr", 20 * M),
                              topup_amt=kw.get("topup_amt", None),
                              **{k: v for k, v in kw.items()
                                 if k not in ("topup_thr", "topup_amt")})
        results[name] = r
        print("%-58s | %5d %7d %7s %4d %9.0f %8.2f"
              % (name[:58], r["labor_years_total"], r["starts_with_labor"],
                 "YES" if r["saved_1988"] else "no", r["ruin_total"],
                 r["terminal_median_M"], r["min_total_floor_M"]))

    # trace the most promising candidates on 1988
    for name in ["C1 glide2y(sc1.6->2.2)+BOND res20/run20+thr25+ALL",
                 "C3 glide3y(sc1.6->2.2)+BOND res22/run18 thr22 ALL",
                 "C5 glide3y+BOND+bucket2y res18/run16/buk14.4 thr20 ALL",
                 "C8 glide3y+BOND+E+bucket1y res20/run18/buk7.2 thr22 ALL"]:
        kw = configs[name]
        trace_1988(rets, sleeves, label=name, **kw)

    # ============================================================
    # FOCUSED COMBINED GRID -- hunt for labor-zero
    # ============================================================
    print("\n" + "=" * 100)
    print("FOCUSED COMBINED GRID (glide x base x reserve_mode x split x bucket x topup)")
    print("=" * 100)
    grid_rows = []
    glides = {
        "none": None,
        "g2_16_22": [(0, "sc1.6"), (2, "sc2.2")],
        "g3_16_22": [(0, "sc1.6"), (3, "sc2.2")],
        "g2_14_22": [(0, "sc1.4"), (2, "sc2.2")],
        "g3_16_24": [(0, "sc1.6"), (3, "sc2.4")],
        "g4_16_22": [(0, "sc1.6"), (4, "sc2.2")],
    }
    singles = {"none": None, "sc2.2": "sc2.2", "sc2.0": "sc2.0", "sc2.4": "sc2.4"}
    reserve_modes = ["bond", "sofr", "cash"]
    splits = [(20, 20), (18, 22), (16, 24), (22, 18), (19, 21)]  # (run,res) M
    buckets = [0, 1, 2]
    draws = ["run_first", "reserve_first_on_down"]
    thrs = [20 * M, 22 * M, 25 * M]

    best = []
    n = 0
    for gname, g in glides.items():
        for sname, s in singles.items():
            if (g is None) == (s is None):
                continue  # exactly one of glide/single
            for mode in reserve_modes:
                for (run_m, res_m) in splits:
                    for buk in buckets:
                        # bucket comes out of total 40M: run+res = 40 - buk*7.2
                        buk_cash = buk * 7.2
                        run0 = run_m * M; res0 = (40 - run_m - buk_cash) * M
                        if res0 < 0:
                            continue
                        for draw in draws:
                            for thr in thrs:
                                n += 1
                                kw = dict(run0=run0, reserve0=res0,
                                          reserve_mode=mode, init_bucket_years=buk,
                                          draw_order=draw, topup_thr=thr,
                                          topup_amt=None)
                                if g is not None:
                                    kw["glide"] = g
                                else:
                                    kw["single"] = s
                                r = V2.run_all_starts(rets, sleeves, **kw)
                                grid_rows.append(dict(
                                    glide=gname, single=sname, reserve_mode=mode,
                                    run_M=run_m, res_M=40 - run_m - buk_cash,
                                    bucket_y=buk, draw=draw, thr_M=thr / M,
                                    labor=r["labor_years_total"],
                                    starts=r["starts_with_labor"],
                                    saved88=int(r["saved_1988"]),
                                    ruin=r["ruin_total"],
                                    termMed=r["terminal_median_M"],
                                    floor=r["min_total_floor_M"]))
    gdf = pd.DataFrame(grid_rows)
    gdf.to_csv(os.path.join(V2.AR_DIR, "labor_zero_round2_combined_20260627.csv"),
               index=False, float_format="%.3f", encoding="utf-8-sig")
    print("Evaluated %d combined configs." % n)

    zero = gdf[(gdf["labor"] == 0) & (gdf["ruin"] == 0)]
    print("\nLABOR-ZERO combined configs (0 labor, 0 ruin): %d" % len(zero))
    if len(zero):
        for _, r in zero.sort_values("termMed", ascending=False).head(20).iterrows():
            print("  glide=%s single=%s res=%s run%.0f/res%.0f buk%dy draw=%s thr%.0f "
                  "| termMed=%.0fM floor=%.2fM"
                  % (r["glide"], r["single"], r["reserve_mode"], r["run_M"],
                     r["res_M"], r["bucket_y"], r["draw"], r["thr_M"],
                     r["termMed"], r["floor"]))

    # saved-1988 configs (the binding constraint) even if other starts still fail
    s88 = gdf[gdf["saved88"] == 1].sort_values(["labor", "termMed"],
                                               ascending=[True, False])
    print("\nConfigs that SAVE 1988 (saved88=1): %d  (top 20 by fewest labor):" % len(s88))
    for _, r in s88.head(20).iterrows():
        print("  glide=%s single=%s res=%s run%.0f/res%.0f buk%dy draw=%s thr%.0f "
              "| labor=%d starts=%d floor=%.2fM termMed=%.0fM"
              % (r["glide"], r["single"], r["reserve_mode"], r["run_M"], r["res_M"],
                 r["bucket_y"], r["draw"], r["thr_M"], r["labor"], r["starts"],
                 r["floor"], r["termMed"]))

    # overall best by fewest labor
    print("\nOVERALL best combined (fewest labor years), top 15:")
    for _, r in gdf.sort_values(["labor", "termMed"], ascending=[True, False]).head(15).iterrows():
        print("  glide=%s single=%s res=%s run%.0f/res%.0f buk%dy draw=%s thr%.0f "
              "| labor=%d starts=%d saved88=%d floor=%.2fM termMed=%.0fM"
              % (r["glide"], r["single"], r["reserve_mode"], r["run_M"], r["res_M"],
                 r["bucket_y"], r["draw"], r["thr_M"], r["labor"], r["starts"],
                 r["saved88"], r["floor"], r["termMed"]))

    block = {"n_combined": int(n), "n_labor_zero": int(len(zero)),
             "n_saved_1988": int(len(s88)),
             "min_labor_overall": int(gdf["labor"].min()),
             "best_floor_saved88": (float(s88.iloc[0]["floor"]) if len(s88) else None)}
    print("\nRETURN_BLOCK"); print(json.dumps(block, indent=2))
    print("Done.")
    return block


if __name__ == "__main__":
    main()
