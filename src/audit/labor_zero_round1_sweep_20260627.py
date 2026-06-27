"""
src/audit/labor_zero_round1_sweep_20260627.py
=============================================
ROUND 1 of the labor-zero optimization (assets 40M, strict spend 7.2M/yr).
Explores the untried levers A-F via harness v2 and writes a full grid CSV plus a
per-lever summary. The goal each config: labor_years_total == 0 over all 31 starts
(1975-2005). Prior best (cash reserve, sc2.2 50:50) = 12 labor / 1988 fails.

Levers (each a block):
  A reserve_mode  : reserve invested cash|nasdaq|gold|bond|sofr
  B glide         : first K years low-leverage -> then high-leverage
  C mix           : weighted blend of two strategies
  D init_bucket   : pre-hold N years of spend as cash, drawn first
  E draw_order    : down-year draws from reserve first
  F ext_grid      : extended leverage scale 2.4/2.6
Plus PAIR combos that the single-lever probe suggested (A bond x D bucket, etc.)
are deferred to round 2; round 1 maps each lever's standalone effect + a few
obvious 2-lever pairs to find what moves the 1988 bottleneck.

ASCII-only prints. No commit, no temp files.
Outputs:
  audit_results/labor_zero_round1_sweep_20260627.csv
"""
from __future__ import annotations

import json
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

# canonical base allocation knobs reused across levers
RUNS = [20 * M, 25 * M, 30 * M]
THRS = [15 * M, 20 * M, 25 * M]
AMTS = [("ALL", None), ("10M", 10 * M)]
CORE = ["sc2.0", "sc2.2", "N4", "X4"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("LABOR-ZERO ROUND 1 -- untried levers A-F  (assets=40M spend=7.2M, goal labor=0)")
    print("=" * 100)
    rets = h2.load_returns()
    sleeves = h2.load_sleeve_returns()
    # extended scales for lever F
    rets.update(h2.load_extended())

    rows = []

    def rec(lever, label, kw):
        r = h2.run_all_starts(rets, sleeves, **kw)
        rows.append(dict(lever=lever, label=label,
                         run_M=kw.get("run0", 0) / M, reserve_M=kw.get("reserve0", 0) / M,
                         reserve_mode=kw.get("reserve_mode", "cash"),
                         init_bucket_years=kw.get("init_bucket_years", 0),
                         draw_order=kw.get("draw_order", "run_first"),
                         topup_thr_M=kw.get("topup_thr", 0) / M,
                         topup_amt=("ALL" if kw.get("topup_amt") is None else "10M"),
                         labor_years_total=r["labor_years_total"],
                         starts_with_labor=r["starts_with_labor"],
                         saved_1988=int(r["saved_1988"]),
                         fails=";".join(str(x) for x in r["fails"]),
                         topup_events_total=r["topup_events_total"],
                         ruin_total=r["ruin_total"],
                         terminal_median_M=round(r["terminal_median_M"], 1),
                         min_total_floor_M=round(r["min_total_floor_M"], 2)))
        return r

    # ---- A: reserve invested ----
    print("\n[A] reserve_mode (reserve invested) ...")
    for strat in CORE:
        for run in RUNS:
            res = 40 * M - run
            for thr in THRS:
                for atag, amt in AMTS:
                    for mode in ("cash", "bond", "sofr", "gold", "nasdaq"):
                        rec("A", "%s_run%.0f_res%s_thr%.0f_%s" % (strat, run / M, mode, thr / M, atag),
                            dict(single=strat, run0=run, reserve0=res, reserve_mode=mode,
                                 topup_thr=thr, topup_amt=amt))

    # ---- B: glide (low-lev first K years -> high) ----
    print("[B] glide path ...")
    for k0 in (2, 3, 5):
        for lo in ("sc1.0", "sc1.4", "sc1.6"):
            for hi in ("sc2.0", "sc2.2", "X4"):
                for run in RUNS:
                    res = 40 * M - run
                    rec("B", "glide%d_%s_to_%s_run%.0f" % (k0, lo, hi, run / M),
                        dict(single=hi, glide=[(0, lo), (k0, hi)], run0=run, reserve0=res,
                             reserve_mode="cash", topup_thr=20 * M, topup_amt=None))

    # ---- C: mix of two strategies ----
    print("[C] strategy mix ...")
    for a, b in (("sc2.2", "N4"), ("sc2.2", "X4"), ("sc1.6", "sc2.2"), ("N4", "X4"), ("sc2.2", "sc1.6")):
        for w in (0.3, 0.5, 0.7):
            for run in RUNS:
                res = 40 * M - run
                rec("C", "mix_%s%.0f_%s%.0f_run%.0f" % (a, w * 100, b, (1 - w) * 100, run / M),
                    dict(mix={a: w, b: 1 - w}, single=a, run0=run, reserve0=res,
                         reserve_mode="cash", topup_thr=20 * M, topup_amt=None))

    # ---- D: initial bucket (cash drawn first) ----
    print("[D] initial spend bucket ...")
    for strat in CORE:
        for nb in (2, 3, 4, 5):
            # bucket comes out of reserve allocation; run + reserve + bucket = 40M
            for run in (20 * M, 25 * M):
                bucket = nb * h2.SPEND
                res = 40 * M - run - bucket
                if res < 0:
                    continue
                rec("D", "%s_bucket%dy_run%.0f" % (strat, nb, run / M),
                    dict(single=strat, run0=run, reserve0=res, reserve_mode="cash",
                         init_bucket_years=nb, topup_thr=20 * M, topup_amt=None))

    # ---- E: draw order (down-year reserve-first) ----
    print("[E] draw order ...")
    for strat in CORE:
        for mode in ("cash", "bond"):
            for run in RUNS:
                res = 40 * M - run
                rec("E", "%s_reservefirst_res%s_run%.0f" % (strat, mode, run / M),
                    dict(single=strat, run0=run, reserve0=res, reserve_mode=mode,
                         draw_order="reserve_first_on_down", topup_thr=20 * M, topup_amt=None))

    # ---- F: extended leverage scales ----
    print("[F] extended scale 2.4/2.6 ...")
    for strat in ("sc2.4", "sc2.6"):
        for run in RUNS:
            res = 40 * M - run
            for thr in THRS:
                rec("F", "%s_run%.0f_thr%.0f" % (strat, run / M, thr / M),
                    dict(single=strat, run0=run, reserve0=res, reserve_mode="cash",
                         topup_thr=thr, topup_amt=None))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_round1_sweep_20260627.csv"),
              index=False, float_format="%.3f", encoding="utf-8-sig")
    print("\nSaved round1 grid: %d configs" % len(df))

    # ---- per-lever summary ----
    print("\n--- per-lever min labor_years (lower=better; prior best=12) ---")
    print("%-4s | %6s | %s" % ("lev", "minLab", "best config (min labor)"))
    for lev in ("A", "B", "C", "D", "E", "F"):
        sub = df[df["lever"] == lev]
        if not len(sub):
            continue
        b = sub.sort_values(["labor_years_total", "terminal_median_M"],
                            ascending=[True, False]).iloc[0]
        print("%-4s | %6d | %s  (saved1988=%d fails=%s termMed=%.0fM)"
              % (lev, int(b["labor_years_total"]), b["label"],
                 int(b["saved_1988"]), b["fails"], b["terminal_median_M"]))

    nzero = df[(df["labor_years_total"] == 0) & (df["ruin_total"] == 0)]
    print("\nLABOR-ZERO configs in round 1: %d" % len(nzero))
    if len(nzero):
        for _, r in nzero.head(20).iterrows():
            print("  *** %s | %s" % (r["lever"], r["label"]))

    # which levers SAVED 1988 (even if other starts still fail)?
    saved = df[df["saved_1988"] == 1].sort_values("labor_years_total")
    print("\nconfigs that SAVED the 1988 bottleneck: %d" % len(saved))
    for _, r in saved.head(15).iterrows():
        print("  [%s] %s  labor=%d fails=%s" % (r["lever"], r["label"],
                                                int(r["labor_years_total"]), r["fails"]))

    block = {"script": "labor_zero_round1_sweep_20260627.py", "date": "2026-06-27",
             "n_configs": len(df), "n_labor_zero": int(len(nzero)),
             "per_lever_min": {lev: int(df[df["lever"] == lev]["labor_years_total"].min())
                               for lev in ("A", "B", "C", "D", "E", "F") if len(df[df["lever"] == lev])},
             "n_saved_1988": int(len(saved))}
    print("\n" + "=" * 100); print("RETURN_BLOCK"); print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
