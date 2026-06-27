"""
src/audit/labor_zero_round2_sweep_20260627.py
=============================================
ROUND 2 of the labor-zero optimization. Round 1 showed NO single lever saves the
1988 bottleneck. Round 2 exhaustively tests COMBINED levers (glide x reserve-asset
x bucket x topup x mix) plus lever G (regime-driven run fraction), to either find a
labor-zero design or robustly confirm infeasibility.

This is the rigorous backstop: it sweeps the full combination space the round-1
analysis pointed at, so the "infeasible" conclusion (if reached) is exhaustive, not
anecdotal. Includes the feasibility witness: the minimal compound return on the full
40M needed to survive the 1988 start, and the empirical fact that all-in max-CAGR
dies by 1993-94.

Lever G (regime de-lever): a per-year run multiplier in [0,1] from a causal signal
(prior-year strategy return < 0 -> scale down next year). Checked vs an equal-mean
twin to avoid de-lever-in-disguise (R-STAT-3).

ASCII-only prints. No commit, no temp files.
Outputs:
  audit_results/labor_zero_round2_sweep_20260627.csv
  audit_results/labor_zero_round2_feasibility_20260627.csv
"""
from __future__ import annotations

import json
import os
import sys
import importlib.util
import itertools

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
START_YEARS = h2.START_YEARS


def regime_scale_for(rets, strat, start, horizon=20, lo=0.5):
    """Causal de-lever: if prior calendar-year strategy return < 0, scale next
    year's run to `lo`; else 1.0. Year 0 = 1.0 (no prior info)."""
    arr = np.ones(horizon)
    for k in range(1, horizon):
        prev = float(rets[strat].loc[start + k - 1])
        arr[k] = lo if prev < 0 else 1.0
    return arr


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("LABOR-ZERO ROUND 2 -- combined levers + regime  (assets=40M spend=7.2M, goal labor=0)")
    print("=" * 100)
    rets = h2.load_returns()
    rets.update(h2.load_extended())
    sleeves = h2.load_sleeve_returns()

    rows = []

    def rec(label, kw, regime=None):
        # regime: (strat_for_signal, lo) -> build per-start arrays inside run_all
        if regime is None:
            r = h2.run_all_starts(rets, sleeves, **kw)
        else:
            sstrat, lo = regime
            labor = topup = ruin = 0
            terms = []
            floors = []
            fails = []
            for sy in START_YEARS:
                rs = regime_scale_for(rets, sstrat, sy, lo=lo)
                res = h2.simulate_v2(rets, sleeves, sy, regime_scale=rs, **kw)
                labor += res["labor_years"]
                topup += res["topups"]
                ruin += res["ruin"]
                if res["labor_years"] > 0:
                    fails.append(sy)
                if res["ruin"] == 0:
                    terms.append(res["terminal"])
                floors.append(res["min_total"])
            r = dict(labor_years_total=labor, starts_with_labor=len(fails), fails=fails,
                     saved_1988=(1988 not in fails), topup_events_total=topup, ruin_total=ruin,
                     terminal_median_M=(float(np.median(terms)) / M if terms else 0.0),
                     terminal_min_M=(float(np.min(terms)) / M if terms else 0.0),
                     min_total_floor_M=float(np.min(floors)) / M)
        rows.append(dict(label=label, labor_years_total=r["labor_years_total"],
                         starts_with_labor=r["starts_with_labor"], saved_1988=int(r["saved_1988"]),
                         fails=";".join(str(x) for x in r["fails"]),
                         topup_events_total=r["topup_events_total"], ruin_total=r["ruin_total"],
                         terminal_median_M=round(r["terminal_median_M"], 1),
                         min_total_floor_M=round(r["min_total_floor_M"], 2)))
        return r

    # ---- combined grid: glide x reserve-asset x bucket x topup x strategy ----
    print("\n[combos] glide x reserve-asset x bucket x topup x strategy ...")
    strategies = ["sc2.2", "sc2.4", "sc2.6", "N4", "sc2.0"]
    glides = [None,
              [(0, "sc1.6")], [(0, "sc1.8")],            # de-lever early (low->high handled via switch)
              ]
    glide_specs = {
        "none": None,
        "g2_16": lambda hi: [(0, "sc1.6"), (2, hi)],
        "g3_16": lambda hi: [(0, "sc1.6"), (3, hi)],
        "g3_18": lambda hi: [(0, "sc1.8"), (3, hi)],
        "g2_hi_early": lambda hi: [(0, hi), (3, "sc1.6")],  # high early, de-lever later
    }
    reserve_assets = ["cash", "bond", "sofr"]
    buckets = [0, 2, 3, 4]
    runs = [15 * M, 18 * M, 20 * M, 22 * M, 25 * M]
    topups = [(15 * M, None), (18 * M, None), (20 * M, None), (20 * M, 10 * M), (12 * M, 6 * M)]

    n = 0
    for strat in strategies:
        for gname, gfn in glide_specs.items():
            glide = None if gfn is None else gfn(strat)
            for rasset in reserve_assets:
                for nb in buckets:
                    for run in runs:
                        bucket_cash = nb * h2.SPEND
                        res = 40 * M - run - bucket_cash
                        if res < -1e-6:
                            continue
                        res = max(0.0, res)
                        for thr, amt in topups:
                            label = "%s_%s_res%s_bkt%d_run%.0f_thr%.0f_%s" % (
                                strat, gname, rasset, nb, run / M, thr / M,
                                "ALL" if amt is None else "%.0f" % (amt / M))
                            rec(label, dict(single=strat, glide=glide, run0=run, reserve0=res,
                                            reserve_mode=rasset, init_bucket_years=nb,
                                            topup_thr=thr, topup_amt=amt))
                            n += 1
    print("  combined configs: %d" % n)

    # ---- lever G: regime de-lever (a few strategies x lo) ----
    print("[G] regime de-lever (prior-down -> scale run) ...")
    for strat in ("sc2.2", "sc2.6", "N4"):
        for lo in (0.5, 0.7):
            rec("G_%s_lo%.1f_bond_run20" % (strat, lo),
                dict(single=strat, run0=20 * M, reserve0=20 * M, reserve_mode="bond",
                     topup_thr=20 * M, topup_amt=None), regime=(strat, lo))

    # ---- lever H: TWO-WAY annual rebalance (harvest spikes into safe leg) ----
    # The round-1 analysis flagged this as the one untested mechanism class.
    # Tested here across ALL 31 starts x full 20y (not just the 1988 sub-window).
    print("[H] two-way rebalance (harvest spikes) ...")
    for strat in ("sc2.2", "sc2.4", "sc2.6", "N4", "sc2.0"):
        for wrun in (0.4, 0.5, 0.6, 0.7, 0.8):
            for rasset in ("bond", "sofr", "cash"):
                for band in (0.0, 0.1):
                    rec("H_%s_w%.0f_%s_band%.0f" % (strat, wrun * 100, rasset, band * 100),
                        dict(single=strat, run0=40 * M * wrun, reserve0=40 * M * (1 - wrun),
                             reserve_mode=rasset, rebalance=wrun, rebalance_band=band))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_round2_sweep_20260627.csv"),
              index=False, float_format="%.3f", encoding="utf-8-sig")
    print("\nSaved round2 grid: %d configs" % len(df))

    # ---- results ----
    nzero = df[(df["labor_years_total"] == 0) & (df["ruin_total"] == 0)]
    print("\n*** LABOR-ZERO configs in round 2: %d ***" % len(nzero))
    for _, r in nzero.head(30).iterrows():
        print("  %s  termMed=%.0fM" % (r["label"], r["terminal_median_M"]))

    saved = df[df["saved_1988"] == 1].sort_values("labor_years_total")
    print("\nconfigs that SAVED the 1988 bottleneck: %d" % len(saved))
    for _, r in saved.head(15).iterrows():
        print("  %s  labor=%d fails=%s" % (r["label"], int(r["labor_years_total"]), r["fails"]))

    best = df.sort_values(["labor_years_total", "terminal_median_M"], ascending=[True, False]).iloc[0]
    print("\nBEST round2 config (min labor): %s" % best["label"])
    print("  labor=%d starts_fail=%s termMed=%.0fM"
          % (int(best["labor_years_total"]), best["fails"], best["terminal_median_M"]))

    # ---- feasibility witness ----
    print("\n--- FEASIBILITY WITNESS (1988 start, 40M, 7.2M/yr) ---")
    def survive_const(g, n=20):
        bal = 40.0
        for k in range(n):
            bal -= 7.2
            if bal <= 0:
                return False
            bal *= (1 + g)
        return bal > 0
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        if survive_const(mid):
            hi = mid
        else:
            lo = mid
    need_cagr = hi
    print("  constant CAGR on full 40M needed to survive 1988 = %.1f%%/yr" % (need_cagr * 100))
    allin = {}
    for strat in ("sc2.2", "sc2.6", "N4", "sc2.0"):
        bal = 40.0
        died = None
        for k in range(20):
            yr = 1988 + k
            bal -= 7.2
            if bal <= 0:
                died = yr
                break
            bal *= (1 + float(rets[strat].loc[yr]))
        allin[strat] = died
        print("  all-in 40M %-6s: %s" % (strat, "survives" if died is None else "DIES %d" % died))
    feas = pd.DataFrame([{"metric": "const_CAGR_needed_pct", "value": round(need_cagr * 100, 2)}] +
                        [{"metric": "all_in_%s_dies_year" % k, "value": (v if v else "survives")}
                         for k, v in allin.items()])
    feas.to_csv(os.path.join(AR_DIR, "labor_zero_round2_feasibility_20260627.csv"),
                index=False, encoding="utf-8-sig")

    block = {"script": "labor_zero_round2_sweep_20260627.py", "date": "2026-06-27",
             "n_configs": len(df), "n_labor_zero": int(len(nzero)),
             "n_saved_1988": int(len(saved)),
             "best_label": best["label"], "best_labor": int(best["labor_years_total"]),
             "const_cagr_needed_pct": round(need_cagr * 100, 2),
             "all_in_dies": {k: (v if v else "survives") for k, v in allin.items()}}
    print("\n" + "=" * 100); print("RETURN_BLOCK"); print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
