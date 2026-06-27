"""
src/audit/labor_zero_allocation_sweep_20260627.py
=================================================
GOAL (user, 2026-06-27): assets 40,000,000 JPY, annual spend 7,200,000 JPY.
Find the operating design (strategy x run-amount x reserve top-up rule) that needs
ZERO labor backfill across ALL 31 start years (1975-2005) x 20 years -- i.e. the
40M (run sleeve + reserve sleeve) alone funds every year's 7.2M withdrawal, never
forcing external earning.

DEFINITIONS (per user choice "predeposit-included 40M lasts, no labor"):
  - TOTAL wealth = run_balance + reserve. Spend 7,200,000/yr comes out of the run.
  - RESERVE = cash set aside (no market growth), deployed into the run by a TOP-UP
    rule (a self-asset rebalance, NOT labor): when run < TOPUP_THRESHOLD at year
    start, move min(reserve, TOPUP_AMOUNT) from reserve into run.
  - LABOR YEAR (the thing to drive to ZERO): a year where, after any top-up, the
    run balance cannot cover the 7.2M spend AND reserve is empty -> must earn
    externally. (Equivalently: total wealth < 7.2M at the moment of spending.)
  - Market return applies to the run sleeve only (reserve is cash).
  - RUIN = total wealth hits 0.

SWEEP:
  strategies (annual after-tax returns, x0.8273): P09_STR sc1.0/1.4/1.6/1.8/2.0/2.2,
    N4, X4, X4eq_sc1.4/1.6/1.8/2.2, NASDAQ_1x (reference).
  run_amount in {20,25,30,35,40} M  (reserve = 40M - run)
  topup_threshold in {15,20,25} M
  topup_amount in {10M, ALL remaining reserve}
  (when run==40M there is no reserve; topup params are moot -> single all-in case.)

For each (strategy, run, thr, amt): simulate all 31 starts x 20yr; record
  - labor_years_total (over 620 sims) and starts_with_any_labor
  - topup_events_total (reserve->run moves; "手間" even if not labor)
  - ruin count
  - terminal TOTAL wealth (median over surviving starts), min-ever total wealth
The PRIMARY filter is labor_years_total == 0 AND ruin == 0. Among those, rank by
fewest topup_events, then highest terminal median, then highest min-ever wealth.

ASCII-only prints. No commit, no temp files.
Outputs:
  audit_results/labor_zero_allocation_sweep_20260627.csv   (full grid)
  audit_results/labor_zero_winners_20260627.csv            (labor-zero feasible set)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)

AR_DIR = os.path.join(_REPO_DIR, "audit_results")
SCALE_CSV = os.path.join(AR_DIR, "p09_strongmap_scale_dial_annual_20260623.csv")
X4N4_CSV = os.path.join(AR_DIR, "retirement_survival_x4n4_6M_backfill_annual_20260627.csv")
X4EQ_CSV = os.path.join(AR_DIR, "scale_dial_x4equiv_annual_20260627.csv")

M = 1_000_000.0
INIT_TOTAL = 40 * M
SPEND = 7.2 * M
HORIZON = 20
START_YEARS = list(range(1975, 2006))

RUN_AMOUNTS = [20 * M, 25 * M, 30 * M, 35 * M, 40 * M]
TOPUP_THRESHOLDS = [15 * M, 20 * M, 25 * M]
# topup amount: "10M chunk" or "ALL remaining reserve"
TOPUP_AMOUNTS = [("10M", 10 * M), ("ALL", None)]


def load_returns():
    """Return dict[strategy] -> Series(year->frac after-tax)."""
    sc = pd.read_csv(SCALE_CSV).set_index("year")
    xn = pd.read_csv(X4N4_CSV).set_index("year")
    xe = pd.read_csv(X4EQ_CSV).set_index("year")
    out = {}
    for s in (1.0, 1.4, 1.6, 1.8, 2.0, 2.2):
        out["sc%.1f" % s] = (sc["sc%.1f_strong_aftertax_pct" % s] / 100.0)
    out["N4"] = xn["N4_aftertax_pct"] / 100.0
    out["X4"] = xn["X4_aftertax_pct"] / 100.0
    for s in (1.4, 1.6, 1.8, 2.2):
        out["X4eq_sc%.1f" % s] = (xe["X4eq_sc%.1f_pct" % s] / 100.0)
    out["NASDAQ_1x"] = sc["NASDAQ_1x_BH_aftertax_pct"] / 100.0
    # clip to 1975-2025
    for k in out:
        s = out[k].dropna()
        out[k] = s[(s.index >= 1975) & (s.index <= 2025)]
    return out


def simulate(ret, start, run0, reserve0, thr, amt):
    """One 31-start sim. Returns dict with labor_years, topups, ruin, terminal, min_total.
    Order each year:
      1. (year start) TOP-UP: if run < thr and reserve > 0, move min(reserve, amt_eff) into run.
         amt_eff = reserve if amt is None else min(amt, reserve).
      2. SPEND: withdraw 7.2M. Prefer run; if run < spend, take the remainder from reserve.
         If run+reserve < spend -> LABOR year (cover the shortfall externally; do not
         draw wealth negative; that year's run/reserve are depleted to 0 for the spend
         they could cover, shortfall is "earned"). Count labor.
      3. MARKET: apply that year's return to the run sleeve.
    """
    run = float(run0)
    reserve = float(reserve0)
    labor_years = 0
    topups = 0
    min_total = run + reserve
    for k in range(HORIZON):
        yr = start + k
        # 1. top-up (rebalance reserve -> run)
        if run < thr and reserve > 1e-6:
            move = reserve if amt is None else min(amt, reserve)
            run += move
            reserve -= move
            topups += 1
        # 2. spend
        total = run + reserve
        if total + 1e-6 < SPEND:
            # cannot fund the year from own assets -> labor year
            labor_years += 1
            # deplete what we have toward the spend (rest earned externally)
            run = 0.0
            reserve = 0.0
        else:
            if run >= SPEND:
                run -= SPEND
            else:
                short = SPEND - run
                run = 0.0
                reserve -= short  # reserve covers the remainder
        # 3. market on run sleeve
        r = float(ret.loc[yr])
        run *= (1.0 + r)
        total = run + reserve
        if total < min_total:
            min_total = total
    ruin = (run + reserve) <= 1e-6
    return dict(labor_years=labor_years, topups=topups, ruin=int(ruin),
                terminal=run + reserve, min_total=min_total)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 110)
    print("LABOR-ZERO ALLOCATION SWEEP  assets=%.0f spend=%.0f/yr  goal: 0 labor years over 31x20"
          % (INIT_TOTAL, SPEND))
    print("=" * 110)
    rets = load_returns()
    strategies = ["sc1.0", "sc1.4", "sc1.6", "sc1.8", "sc2.0", "sc2.2",
                  "N4", "X4", "X4eq_sc1.4", "X4eq_sc1.6", "X4eq_sc1.8", "X4eq_sc2.2",
                  "NASDAQ_1x"]

    rows = []
    for strat in strategies:
        ret = rets[strat]
        # ensure full coverage
        for run0 in RUN_AMOUNTS:
            reserve0 = INIT_TOTAL - run0
            # if no reserve, topup params are irrelevant -> one canonical case
            param_sets = ([(15 * M, "n/a", None)] if reserve0 <= 1e-6
                          else [(thr, tag, amt) for thr in TOPUP_THRESHOLDS
                                for tag, amt in TOPUP_AMOUNTS])
            for thr, tag, amt in param_sets:
                labor_tot = 0
                topup_tot = 0
                ruin_tot = 0
                terms = []
                min_totals = []
                starts_with_labor = []
                for sy in START_YEARS:
                    if not all(sy + k in ret.index for k in range(HORIZON)):
                        raise RuntimeError("%s missing %d" % (strat, sy))
                    res = simulate(ret, sy, run0, reserve0, thr, amt)
                    labor_tot += res["labor_years"]
                    topup_tot += res["topups"]
                    ruin_tot += res["ruin"]
                    if res["labor_years"] > 0:
                        starts_with_labor.append(sy)
                    if res["ruin"] == 0:
                        terms.append(res["terminal"])
                    min_totals.append(res["min_total"])
                rows.append(dict(
                    strategy=strat, run_M=run0 / M, reserve_M=reserve0 / M,
                    topup_thr_M=(thr / M if reserve0 > 1e-6 else np.nan),
                    topup_amt=(tag if reserve0 > 1e-6 else "n/a"),
                    labor_years_total=labor_tot,
                    starts_with_labor=len(starts_with_labor),
                    starts_with_labor_list=";".join(str(x) for x in starts_with_labor),
                    topup_events_total=topup_tot,
                    ruin_total=ruin_tot,
                    terminal_median_M=(float(np.median(terms)) / M if terms else 0.0),
                    terminal_min_M=(float(np.min(terms)) / M if terms else 0.0),
                    min_total_floor_M=float(np.min(min_totals)) / M,
                ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_allocation_sweep_20260627.csv"),
              index=False, float_format="%.4f", encoding="utf-8-sig")
    print("\nSaved full grid: %d combos" % len(df))

    # ---- labor-zero feasible set ----
    feas = df[(df["labor_years_total"] == 0) & (df["ruin_total"] == 0)].copy()
    feas = feas.sort_values(["topup_events_total", "terminal_median_M", "min_total_floor_M"],
                            ascending=[True, False, False])
    feas.to_csv(os.path.join(AR_DIR, "labor_zero_winners_20260627.csv"),
                index=False, float_format="%.4f", encoding="utf-8-sig")
    print("LABOR-ZERO feasible combos (0 labor years, 0 ruin): %d" % len(feas))

    # ---- summary prints ----
    print("\n--- labor-years by strategy x run_amount (min over topup params) ---")
    print("%-12s | %s" % ("strategy", " ".join("run%2dM" % (r / M) for r in RUN_AMOUNTS)))
    for strat in strategies:
        cells = []
        for run0 in RUN_AMOUNTS:
            sub = df[(df["strategy"] == strat) & (abs(df["run_M"] - run0 / M) < 1e-6)]
            cells.append("%5d" % int(sub["labor_years_total"].min()))
        print("%-12s | %s" % (strat, "  ".join(cells)))

    print("\n--- TOP labor-zero designs (fewest top-ups, then highest terminal median) ---")
    print("%-12s run/res(M) thr/amt | laborY topups ruin | termMed(M) termMin(M) floor(M)" % "strategy")
    for _, r in feas.head(25).iterrows():
        print("%-12s %4.0f/%-4.0f %4s/%-3s | %5d %6d %4d | %9.1f %9.1f %8.2f"
              % (r["strategy"], r["run_M"], r["reserve_M"],
                 ("%.0f" % r["topup_thr_M"]) if r["topup_thr_M"] == r["topup_thr_M"] else "n/a",
                 r["topup_amt"], int(r["labor_years_total"]), int(r["topup_events_total"]),
                 int(r["ruin_total"]), r["terminal_median_M"], r["terminal_min_M"],
                 r["min_total_floor_M"]))

    # current user design for reference: sc2.0, run=30M, reserve=10M, thr=20M, amt=10M
    cur = df[(df["strategy"] == "sc2.0") & (abs(df["run_M"] - 30) < 1e-6)
             & (abs(df["topup_thr_M"] - 20) < 1e-6) & (df["topup_amt"] == "10M")]
    print("\n--- USER CURRENT DESIGN (sc2.0, run30M/res10M, thr20M, amt10M) ---")
    if len(cur):
        r = cur.iloc[0]
        print("  labor_years=%d  topups=%d  ruin=%d  termMed=%.1fM termMin=%.1fM floor=%.2fM  labor-zero? %s"
              % (int(r["labor_years_total"]), int(r["topup_events_total"]), int(r["ruin_total"]),
                 r["terminal_median_M"], r["terminal_min_M"], r["min_total_floor_M"],
                 "YES" if r["labor_years_total"] == 0 and r["ruin_total"] == 0 else "NO"))

    block = {"script": "labor_zero_allocation_sweep_20260627.py", "date": "2026-06-27",
             "init_total_M": INIT_TOTAL / M, "spend_M": SPEND / M, "horizon": HORIZON,
             "n_starts": len(START_YEARS), "n_combos": len(df),
             "n_labor_zero": len(feas),
             "top_labor_zero": feas.head(15).to_dict(orient="records")}
    print("\n" + "=" * 110); print("RETURN_BLOCK"); print("=" * 110)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
