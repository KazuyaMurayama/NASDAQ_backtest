"""
src/audit/labor_zero_v3_rounds_de_20260628.py
=============================================
ROUND D & E -- the method family the earlier rounds pointed to.

WEAK POINT from rounds A-C: a FIXED-yen spend (rigid OR a fixed-yen guardrail
floor) is structurally unfundable once the 2013-start sleeve hits near-zero by
2017. The wall is "a fixed yen demand on a depleted portfolio". The fix is to
make spending SCALE WITH REMAINING WEALTH, so a labor year (external earning)
can never be forced.

ROUND D  PERCENT-OF-PORTFOLIO (endowment rule):
  spend p% of total wealth each year. labor_years is identically 0 by construction
  (you always take a fraction of what remains). The real metric becomes the
  WORST realized annual spend across all 46 starts x 20y (the purchasing-power
  floor), and the median spend. Trades "never work" for "income fluctuates".
  We sweep p and report, for the v2 base (sc2.6 14/26 bond), the distribution.

ROUND E  WEALTH-GUARDRAIL HYBRID (Guyton-Klinger, wealth-based not drawdown-based):
  target spend 7.2M, but cap by a current-withdrawal-rate guardrail: if
  spend/total > upper_wr, cut spend to upper_wr*total (down to a floor fraction
  of 7.2M); optionally raise toward 7.2M when spend/total < lower_wr. A "shortfall
  year" = realized spend < 7.2M (NOT labor; you just spend less). Metric: number
  of shortfall years, worst realized spend, total lifetime spend, AND whether
  total ever hits 0 (ruin). This NEVER forces labor as long as spend scales.
  We report the config that keeps realized spend closest to 7.2M with no ruin.

Outputs:
  audit_results/labor_zero_v3_roundD_pctport_20260628.csv
  audit_results/labor_zero_v3_roundE_wealthguardrail_20260628.csv
ASCII-only. No commit.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_THIS_DIR, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
M = v3.M
AR_DIR = os.path.join(_REPO_DIR, "audit_results")

FULL_SPEND = 7.2 * M
DATA_END = 2025
ALL_STARTS = list(range(1975, 2021))


def sim_pct(rets, reserve_series, start, horizon, *, strat, run0, reserve0, thr,
            pct, floor_yen=0.0, cap=FULL_SPEND, data_end=DATA_END):
    """ROUND D: spend = clamp(pct*total, floor_yen, cap). If floor_yen=0, labor is
    impossible. Returns realized spend path stats + ruin + floor."""
    run, res = float(run0), float(reserve0)
    s = rets[strat]
    n = min(horizon, data_end - start + 1)
    fired = False
    spends = []
    floor_tot = run + res
    labor = 0
    for k in range(n):
        yr = start + k
        if (not fired) and run < thr and res > 1e-6:
            run += res
            res = 0.0
            fired = True
        total = run + res
        spend = min(cap, max(floor_yen, pct * total))
        if total + 1e-6 < spend:    # only possible if floor_yen>0
            labor += 1
            spend = total
        # draw run-first
        if run >= spend:
            run -= spend
        else:
            res -= (spend - run)
            run = 0.0
        spends.append(spend)
        run *= (1.0 + float(s.loc[yr]))
        res *= (1.0 + float(reserve_series.loc[yr]))
        if run + res < floor_tot:
            floor_tot = run + res
    spends = np.array(spends)
    return dict(labor_years=labor, min_spend_M=round(float(spends.min()) / M, 2),
                med_spend_M=round(float(np.median(spends)) / M, 2),
                total_spend_M=round(float(spends.sum()) / M, 1),
                terminal_M=round((run + res) / M, 1),
                ruin=int((run + res) <= 1e-6), n_years=n)


def sim_wealth_guardrail(rets, reserve_series, start, horizon, *, strat, run0,
                         reserve0, thr, upper_wr, floor_frac, data_end=DATA_END):
    """ROUND E: target 7.2M, but if 7.2M/total > upper_wr, cut to max(upper_wr*total,
    floor_frac*7.2M). shortfall year = realized < 7.2M. Returns stats."""
    run, res = float(run0), float(reserve0)
    s = rets[strat]
    n = min(horizon, data_end - start + 1)
    fired = False
    spends = []
    floor_tot = run + res
    labor = 0
    short = 0
    for k in range(n):
        yr = start + k
        if (not fired) and run < thr and res > 1e-6:
            run += res
            res = 0.0
            fired = True
        total = run + res
        spend = FULL_SPEND
        if total <= 1e-9 or FULL_SPEND / total > upper_wr:
            spend = max(upper_wr * total, floor_frac * FULL_SPEND)
        if total + 1e-6 < spend:
            labor += 1
            spend = total
        if spend < FULL_SPEND - 1e-6:
            short += 1
        if run >= spend:
            run -= spend
        else:
            res -= (spend - run)
            run = 0.0
        spends.append(spend)
        run *= (1.0 + float(s.loc[yr]))
        res *= (1.0 + float(reserve_series.loc[yr]))
        if run + res < floor_tot:
            floor_tot = run + res
    spends = np.array(spends)
    return dict(labor_years=labor, shortfall_years=short,
                min_spend_M=round(float(spends.min()) / M, 2),
                med_spend_M=round(float(np.median(spends)) / M, 2),
                total_spend_M=round(float(spends.sum()) / M, 1),
                terminal_M=round((run + res) / M, 1),
                ruin=int((run + res) <= 1e-6), n_years=n)


def round_D(rets, bond):
    print("\n" + "=" * 95)
    print("ROUND D: PERCENT-OF-PORTFOLIO (spend p%% of wealth; labor impossible by construction)")
    print("=" * 95)
    print("  metric = WORST realized annual spend across all 46 starts (purchasing-power floor),")
    print("  median spend, and worst terminal. Base sc2.6 14/26 bond all-in@14M.")
    rows = []
    for pct in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25]:
        worst_min = 1e18
        worst_min_at = None
        meds = []
        tots = []
        terms = []
        ruin = 0
        for sy in ALL_STARTS:
            r = sim_pct(rets, bond, sy, 20, strat="sc2.6", run0=14 * M, reserve0=26 * M,
                        thr=14 * M, pct=pct)
            if r["min_spend_M"] < worst_min:
                worst_min = r["min_spend_M"]
                worst_min_at = sy
            meds.append(r["med_spend_M"])
            tots.append(r["total_spend_M"])
            terms.append(r["terminal_M"])
            ruin += r["ruin"]
        rows.append(dict(pct=pct, worst_min_spend_M=round(worst_min, 2),
                         worst_min_at=worst_min_at,
                         median_of_median_spend_M=round(float(np.median(meds)), 2),
                         median_total_spend_M=round(float(np.median(tots)), 1),
                         worst_terminal_M=round(float(np.min(terms)), 1), ruin_count=ruin))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v3_roundD_pctport_20260628.csv"),
              index=False, encoding="utf-8-sig")
    print("  %-6s %-16s %-9s %-16s %-13s %-13s %s"
          % ("p%", "worstYrSpendM", "@start", "medMedSpendM", "medTotalM", "worstTermM", "ruin"))
    for _, r in df.iterrows():
        print("  %-6.0f %-16.2f %-9d %-16.2f %-13.1f %-13.1f %d"
              % (r["pct"] * 100, r["worst_min_spend_M"], r["worst_min_at"],
                 r["median_of_median_spend_M"], r["median_total_spend_M"],
                 r["worst_terminal_M"], r["ruin_count"]))
    print("\n  read: at p%, you NEVER work (labor=0 always), but in the worst start the lowest single-year")
    print("  income drops to 'worstYrSpendM'. Higher p = more income but lower worst-year floor / ruin risk.")
    return df


def round_E(rets, bond):
    print("\n" + "=" * 95)
    print("ROUND E: WEALTH GUARDRAIL HYBRID (target 7.2M; cut by withdrawal-rate guardrail; never labor)")
    print("=" * 95)
    rows = []
    for upper_wr in [0.14, 0.16, 0.18, 0.20]:
        for floor_frac in [0.5, 0.6, 0.7]:
            short_tot = 0
            worst_min = 1e18
            ruin = 0
            tots = []
            worst_short_start = None
            max_short = 0
            for sy in ALL_STARTS:
                r = sim_wealth_guardrail(rets, bond, sy, 20, strat="sc2.6", run0=14 * M,
                                         reserve0=26 * M, thr=14 * M, upper_wr=upper_wr,
                                         floor_frac=floor_frac)
                short_tot += r["shortfall_years"]
                if r["shortfall_years"] > max_short:
                    max_short = r["shortfall_years"]
                    worst_short_start = sy
                worst_min = min(worst_min, r["min_spend_M"])
                ruin += r["ruin"]
                tots.append(r["total_spend_M"])
            rows.append(dict(upper_wr=upper_wr, floor_frac=floor_frac,
                             shortfall_years_total=short_tot, worst_start_shortfalls=max_short,
                             worst_short_at=worst_short_start,
                             worst_min_spend_M=round(worst_min, 2),
                             median_total_spend_M=round(float(np.median(tots)), 1), ruin=ruin))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v3_roundE_wealthguardrail_20260628.csv"),
              index=False, encoding="utf-8-sig")
    df = df.sort_values(["ruin", "shortfall_years_total", "worst_min_spend_M"],
                        ascending=[True, True, False])
    print("  %-9s %-11s %-13s %-15s %-15s %-13s %s"
          % ("upperWR", "floorFrac", "totShortYrs", "worstStartShort", "worstYrSpendM", "medTotalM", "ruin"))
    for _, r in df.iterrows():
        print("  %-9.2f %-11.1f %-13d %-15d %-15.2f %-13.1f %d"
              % (r["upper_wr"], r["floor_frac"], r["shortfall_years_total"],
                 r["worst_start_shortfalls"], r["worst_min_spend_M"],
                 r["median_total_spend_M"], r["ruin"]))
    best = df.iloc[0]
    print("\n  => best (no ruin, fewest shortfall years, highest worst-year spend):")
    print("     upper WR %.0f%%, floor %.0f%% of 7.2M: total shortfall years=%d (of 46x20=920),"
          % (best["upper_wr"] * 100, best["floor_frac"] * 100, best["shortfall_years_total"]))
    print("     worst single-year spend %.2fM, median lifetime spend %.1fM, ruin=%d."
          % (best["worst_min_spend_M"], best["median_total_spend_M"], best["ruin"]))
    return df


def sim_floor_top(rets, reserve_series, start, horizon, *, strat, run0, reserve0,
                  thr, floor_yen, top_wr, data_end=DATA_END):
    """ROUND F: GUARANTEED floor_yen every year; top up toward 7.2M only when the
    full-spend withdrawal rate is comfortable (7.2M/total <= top_wr). labor only if
    total can't fund floor_yen (with floor_yen<=4.43M this never happens on 46 starts)."""
    run, res = float(run0), float(reserve0)
    s = rets[strat]
    n = min(horizon, data_end - start + 1)
    fired = False
    spends = []
    floor_tot = run + res
    labor = 0
    short = 0
    for k in range(n):
        yr = start + k
        if (not fired) and run < thr and res > 1e-6:
            run += res
            res = 0.0
            fired = True
        total = run + res
        spend = floor_yen
        if total > 1e-9 and FULL_SPEND / total <= top_wr:
            spend = FULL_SPEND
        if total + 1e-6 < spend:
            labor += 1
            spend = total
        if spend < FULL_SPEND - 1e-6:
            short += 1
        if run >= spend:
            run -= spend
        else:
            res -= (spend - run)
            run = 0.0
        spends.append(spend)
        run *= (1.0 + float(s.loc[yr]))
        res *= (1.0 + float(reserve_series.loc[yr]))
        if run + res < floor_tot:
            floor_tot = run + res
    spends = np.array(spends)
    return dict(labor_years=labor, shortfall_years=short,
                min_spend_M=round(float(spends.min()) / M, 2),
                med_spend_M=round(float(np.median(spends)) / M, 2),
                total_spend_M=round(float(spends.sum()) / M, 1),
                terminal_M=round((run + res) / M, 1),
                ruin=int((run + res) <= 1e-6), n_years=n)


def round_F(rets, bond):
    print("\n" + "=" * 95)
    print("ROUND F: GUARANTEED FLOOR + OPPORTUNISTIC TOP (floor always; top to 7.2M when WR<=top_wr)")
    print("=" * 95)
    rows = []
    for floor_M in [3.6, 4.0, 4.43, 4.7, 5.0]:
        for top_wr in [0.12, 0.14, 0.15, 0.16, 0.18]:
            labor = 0
            short = 0
            worst_min = 1e18
            ruin = 0
            tots = []
            for sy in ALL_STARTS:
                r = sim_floor_top(rets, bond, sy, 20, strat="sc2.6", run0=14 * M,
                                  reserve0=26 * M, thr=14 * M, floor_yen=floor_M * M,
                                  top_wr=top_wr)
                labor += r["labor_years"]
                short += r["shortfall_years"]
                worst_min = min(worst_min, r["min_spend_M"])
                ruin += r["ruin"]
                tots.append(r["total_spend_M"])
            rows.append(dict(floor_M=floor_M, top_wr=top_wr, labor_total=labor,
                             shortfall_years_total=short, worst_min_spend_M=round(worst_min, 2),
                             median_total_spend_M=round(float(np.median(tots)), 1), ruin=ruin))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v3_roundF_floortop_20260628.csv"),
              index=False, encoding="utf-8-sig")
    # labor-zero AND no ruin, then maximize median total spend, then fewest shortfalls
    ok = df[(df.labor_total == 0) & (df.ruin == 0)].sort_values(
        ["median_total_spend_M", "shortfall_years_total"], ascending=[False, True])
    print("  labor-zero & ruin-free configs (sorted by lifetime spend, then fewest cut-years): %d" % len(ok))
    print("  %-8s %-8s %-9s %-13s %-15s %-13s %s"
          % ("floorM", "topWR", "labor", "shortYrs", "worstYrSpendM", "medTotalM", "ruin"))
    for _, r in ok.head(12).iterrows():
        print("  %-8.2f %-8.2f %-9d %-13d %-15.2f %-13.1f %d"
              % (r["floor_M"], r["top_wr"], r["labor_total"], r["shortfall_years_total"],
                 r["worst_min_spend_M"], r["median_total_spend_M"], r["ruin"]))
    best = ok.iloc[0] if len(ok) else None
    if best is not None:
        print("\n  => best: guaranteed floor %.2fM/yr + top to 7.2M when WR<=%.0f%%:"
              % (best["floor_M"], best["top_wr"] * 100))
        print("     labor=0 (never work), ruin=0, worst single-year spend %.2fM, median lifetime %.1fM,"
              % (best["worst_min_spend_M"], best["median_total_spend_M"]))
        print("     %d of 920 years spent below 7.2M (the rest full)." % best["shortfall_years_total"])
    return df, best


def sim_G(rets, reserve_series, start, horizon, *, strat, run0, reserve0, thr,
          floor_yen, top_wr, hold_if_crash=True, data_end=DATA_END):
    """ROUND G: Round F PLUS two levers the Workflow surfaced --
      (1) hold-if-crash: skip the all-in reserve refill in a year whose run-strategy
          return < 0 (don't dump powder one year before a crash);
      (2) caller chooses run0/reserve0 (run-fraction), which the v2 geometry fixed at 0.35.
    Guaranteed floor every year; top to 7.2M when FULL/total <= top_wr."""
    run, res = float(run0), float(reserve0)
    s = rets[strat]
    n = min(horizon, data_end - start + 1)
    fired = False
    spends = []
    floor_tot = run + res
    labor = 0
    short = 0
    for k in range(n):
        yr = start + k
        r_this = float(s.loc[yr])
        if (not fired) and run < thr and res > 1e-6 and ((not hold_if_crash) or r_this >= 0):
            run += res
            res = 0.0
            fired = True
        total = run + res
        spend = floor_yen
        if total > 1e-9 and FULL_SPEND / total <= top_wr:
            spend = FULL_SPEND
        if total + 1e-6 < spend:
            labor += 1
            spend = total
        if spend < FULL_SPEND - 1e-6:
            short += 1
        if run >= spend:
            run -= spend
        else:
            res -= (spend - run)
            run = 0.0
        spends.append(spend)
        run *= (1.0 + r_this)
        res *= (1.0 + float(reserve_series.loc[yr]))
        if run + res < floor_tot:
            floor_tot = run + res
    spends = np.array(spends)
    return dict(labor_years=labor, shortfall_years=short,
                min_spend_M=round(float(spends.min()) / M, 2),
                total_spend_M=round(float(spends.sum()) / M, 1),
                terminal_M=round((run + res) / M, 1),
                ruin=int((run + res) <= 1e-6), n_years=n)


def round_G(rets, bond):
    print("\n" + "=" * 95)
    print("ROUND G: F + hold-if-crash refill + run-fraction (Workflow-surfaced levers)")
    print("=" * 95)
    print("  hold-if-crash = don't dump the reserve all-in in a year the run strategy is NEGATIVE")
    print("  (v2 dumped 26M reserve in the weak-positive 2014, one year before the 2015 -38% crash).")
    print("  Also sweep run-fraction (v2 fixed it at 0.35); higher run-fraction raises the safe floor.")
    rows = []
    for strat in ["sc2.6", "sc2.4", "sc2.2", "sc2.0"]:
        for runfrac in [0.35, 0.45, 0.50, 0.60]:
            run0 = runfrac * 40 * M
            res0 = 40 * M - run0
            thr = min(14 * M, run0)
            for floor_M in [5.0, 5.5, 6.0, 6.37]:
                for top_wr in [0.16, 0.18, 0.20]:
                    labor = short = ruin = 0
                    wmin = 1e18
                    tots = []
                    terms = []
                    for sy in ALL_STARTS:
                        r = sim_G(rets, bond, sy, 20, strat=strat, run0=run0, reserve0=res0,
                                  thr=thr, floor_yen=floor_M * M, top_wr=top_wr)
                        labor += r["labor_years"]
                        short += r["shortfall_years"]
                        wmin = min(wmin, r["min_spend_M"])
                        ruin += r["ruin"]
                        tots.append(r["total_spend_M"])
                        terms.append(r["terminal_M"])
                    rows.append(dict(strat=strat, runfrac=runfrac, floor_M=floor_M, top_wr=top_wr,
                                     labor_total=labor, shortfall_years_total=short,
                                     worst_min_spend_M=round(wmin, 2),
                                     median_total_spend_M=round(float(np.median(tots)), 1),
                                     median_terminal_M=round(float(np.median(terms)), 1), ruin=ruin))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AR_DIR, "labor_zero_v3_roundG_holdcrash_20260628.csv"),
              index=False, encoding="utf-8-sig")
    ok = df[(df.labor_total == 0) & (df.ruin == 0)].sort_values(
        ["worst_min_spend_M", "median_total_spend_M"], ascending=[False, False])
    print("\n  labor-zero & ruin-free configs, sorted by GUARANTEED FLOOR (worst single-year spend): %d" % len(ok))
    print("  %-7s %-9s %-8s %-7s %-9s %-13s %-15s %-13s %s"
          % ("strat", "runfrac", "floorM", "topWR", "labor", "shortYrs", "worstYrSpendM",
             "medTotalM", "medTermM"))
    for _, r in ok.head(12).iterrows():
        print("  %-7s %-9.2f %-8.2f %-7.2f %-9d %-13d %-15.2f %-13.1f %.0f"
              % (r["strat"], r["runfrac"], r["floor_M"], r["top_wr"], r["labor_total"],
                 r["shortfall_years_total"], r["worst_min_spend_M"], r["median_total_spend_M"],
                 r["median_terminal_M"]))
    best = ok.iloc[0] if len(ok) else None
    if best is not None:
        print("\n  => BEST (highest guaranteed floor with labor=0/ruin=0):")
        print("     %s runfrac %.2f, floor %.2fM + top to 7.2M when WR<=%.0f%%, hold-if-crash ON:"
              % (best["strat"], best["runfrac"], best["floor_M"], best["top_wr"] * 100))
        print("     worst single-year spend %.2fM (guaranteed), median lifetime %.1fM, median terminal %.0fM,"
              % (best["worst_min_spend_M"], best["median_total_spend_M"], best["median_terminal_M"]))
        print("     %d of 920 years below 7.2M, labor=0, ruin=0." % best["shortfall_years_total"])
    return df, best


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 95)
    print("LABOR-ZERO v3 -- ROUND D/E/F/G: wealth-scaled spending (labor cannot be forced)")
    print("=" * 95)
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    dfD = round_D(rets, bond)
    dfE = round_E(rets, bond)
    dfF, bestF = round_F(rets, bond)
    dfG, bestG = round_G(rets, bond)
    block = {
        "script": "labor_zero_v3_rounds_de_20260628.py", "date": "2026-06-28",
        "roundD": dfD.to_dict("records"),
        "roundE_best": dfE.sort_values(["ruin", "shortfall_years_total"]).iloc[0].to_dict(),
        "roundF_best": (bestF.to_dict() if bestF is not None else None),
        "roundG_best": (bestG.to_dict() if bestG is not None else None),
    }
    print("\n" + "=" * 95)
    print("RETURN_BLOCK")
    print("=" * 95)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True, default=str))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
