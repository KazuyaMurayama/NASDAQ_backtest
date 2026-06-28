"""
src/audit/scale_dial_sc26_addendum_20260628.py
==============================================
Add scale 2.6 to the FULL scale-dial report P09_STR_SCALE_DIAL_20260623.md
(the ladder report: sc1.0/1.4/1.6/1.8/2.0/2.2/2.4 + NASDAQ 1x), and trace the
labor-zero backfill so the user can SEE which calendar year the top-up fires for
each of the 31 starts.

Two deliverables in one run:

  (A) STANDARD-10 (v2.0) + annual after-tax row for sc2.6, on the IDENTICAL footing
      as the existing ladder (same builder H.build(scale=2.6), same metrics10, same
      after-tax x0.8273, same half-up 1dp display rounding as the report). A sanity
      check first reproduces the FRONTIER sc2.0 anchor to 4 digits, so sc2.6 sits on
      the same validated machinery as sc2.0/2.4 before it enters the report.

  (B) BACKFILL TRACE for the labor-zero headline config:
        sc2.6 run 14M / reserve 26M in BOND / top-up-ALL when run < 14M, spend 7.2M.
      For each start year 1975..2005, replay 20 years and record EXACTLY the calendar
      year(s) the run sleeve falls below 14M and the reserve is emptied into it. This
      DIRECTLY answers the user's question and CORRECTS the premise: the top-up is NOT
      triggered by "a year whose return < ~51%". It is triggered when the run sleeve
      BALANCE (cumulative: prior balance, minus that year's 7.2M withdrawal, times that
      year's return) drops under 14M -- a path effect, not a single-year return cut.

ASCII-only prints. Does NOT commit. No temp files written outside audit_results/.
Outputs:
  audit_results/scale_dial_sc26_metrics_20260628.csv     (sc2.6 standard-10)
  audit_results/scale_dial_sc26_annual_20260628.csv      (sc2.6 + sc2.4 annual, after-tax)
  audit_results/labor_zero_backfill_trace_20260628.csv   (per-start backfill year(s))
"""
from __future__ import annotations

import json
import math
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

import src.audit.dd_reduction_harness_20260626 as H
from src.audit.run_p01_backtest_20260611 import _calendar_year_returns, AFTER_TAX as _AT

AFTER_TAX = 0.8273
assert abs(_AT - AFTER_TAX) < 1e-9, "AFTER_TAX drift"

AR_DIR = os.path.join(_REPO_DIR, "audit_results")
EXISTING_ANNUAL = os.path.join(AR_DIR, "p09_strongmap_scale_dial_annual_20260623.csv")

# FRONTIER anchor for the harness sc2.0 sanity (same as the dial report).
SC20_ANCHOR = {"CAGR_IS": 0.353755, "CAGR_OOS": 0.291102, "MaxDD": -0.616342}

STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}

# labor-zero headline config (LABOR_ZERO_ROUND2_OPTIMIZATION_40M_720_20260627.md)
M = 1_000_000.0
RUN0 = 14 * M
RES0 = 26 * M
THR = 14 * M
SPEND = 7.2 * M
HORIZON = 20
START_YEARS = list(range(1975, 2006))


def _round1_halfup(x):
    """1dp, half-UP, to match the ORIGINAL ladder report's displayed values."""
    if x != x:
        return x
    return math.floor(abs(x) * 10 + 0.5) / 10 * (1 if x >= 0 else -1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("SCALE 2.6 ADDENDUM to P09_STR_SCALE_DIAL_20260623.md  (2026-06-28)")
    print("=" * 100)

    ctx = H.setup()

    # ---- SANITY: harness sc2.0 reproduces FRONTIER anchor ----
    nav20, r20, tpy20, _ = H.build(ctx, scale=2.0)
    m20 = H.metrics10(ctx, nav20, r20, tpy20, label="sc2.0", with_wfa=False)
    d_is = abs(m20["CAGR_IS"] - SC20_ANCHOR["CAGR_IS"])
    d_oos = abs(m20["CAGR_OOS"] - SC20_ANCHOR["CAGR_OOS"])
    d_dd = abs(m20["MaxDD"] - SC20_ANCHOR["MaxDD"])
    print("\n--- SANITY harness sc2.0 vs FRONTIER anchor ---")
    print("  IS %+.4f%% (exp %+.4f%%)  OOS %+.4f%% (exp %+.4f%%)  MaxDD %+.4f%% (exp %+.4f%%)"
          % (m20["CAGR_IS"]*100, SC20_ANCHOR["CAGR_IS"]*100,
             m20["CAGR_OOS"]*100, SC20_ANCHOR["CAGR_OOS"]*100,
             m20["MaxDD"]*100, SC20_ANCHOR["MaxDD"]*100))
    sane = (d_is <= 0.0015 and d_oos <= 0.0015 and d_dd <= 0.0015)
    print("  SANITY: %s" % ("PASS" if sane else "FAIL -- HALT"))
    if not sane:
        sys.exit(1)

    # ========== (A) sc2.6 standard-10 + annual ==========
    print("\n=== (A) sc2.6 standard-10 (with WFA) ===")
    nav26, r26, tpy26, exc26 = H.build(ctx, scale=2.6)
    m26 = H.metrics10(ctx, nav26, r26, tpy26, label="sc2.6", with_wfa=True)
    max_eff_lev26 = 2.6 * 3.0 * max(STRONG_MAP.values())  # IN-leg peak proxy (same convention)
    print("  sc2.6  IS=%+.4f%% OOS=%+.4f%% min9=%+.4f%% gap=%+.2fpp Sharpe=%.3f SharpeOOS=%.3f"
          % (m26["CAGR_IS"]*100, m26["CAGR_OOS"]*100, m26["min9"]*100,
             m26["IS_OOS_gap_pp"], m26["Sharpe_FULL"], m26["Sharpe_OOS"]))
    print("         MaxDD=%+.4f%% W1D=%+.2f%%(%s) W10Y=%+.2f%% W5Y=%+.2f%% P10=%+.2f%% Tr=%.1f"
          % (m26["MaxDD"]*100, m26["Worst1D"]*100, str(m26.get("Worst1D_date", "")),
             m26["Worst10Y"]*100, m26["Worst5Y"]*100, m26["P10_5Y"]*100, m26["Trades_yr"]))
    print("         WFE=%.3f CI95lo=%+.2f%% Regime=%+.2f%% maxEffLev~%.2fx"
          % (m26["WFE"], m26["CI95_lo"]*100, m26["Regime_min"]*100, max_eff_lev26))

    std_row = {
        "label": "P09_STR_sc2.6", "scale": 2.6, "kind": "scale",
        "CAGR_IS": m26["CAGR_IS"], "CAGR_OOS": m26["CAGR_OOS"], "min9": m26["min9"],
        "IS_OOS_gap_pp": m26["IS_OOS_gap_pp"], "Sharpe_FULL": m26["Sharpe_FULL"],
        "Sharpe_OOS": m26["Sharpe_OOS"], "MaxDD": m26["MaxDD"], "Worst1D": m26["Worst1D"],
        "Worst1D_date": str(m26.get("Worst1D_date", "")),
        "Worst10Y": m26["Worst10Y"], "Worst5Y": m26["Worst5Y"], "P10_5Y": m26["P10_5Y"],
        "Trades_yr": m26["Trades_yr"], "max_eff_lev": max_eff_lev26,
        "WFE": m26["WFE"], "CI95_lo": m26["CI95_lo"], "Regime_min": m26["Regime_min"],
    }
    pd.DataFrame([std_row]).to_csv(
        os.path.join(AR_DIR, "scale_dial_sc26_metrics_20260628.csv"),
        index=False, float_format="%.6f", encoding="utf-8-sig")

    # annual after-tax for sc2.6 (calendar-year, x0.8273), half-up 1dp like report
    cy26 = _calendar_year_returns(nav26)
    cy26 = cy26[cy26.index <= 2025] * AFTER_TAX
    ex_ann = pd.read_csv(EXISTING_ANNUAL).set_index("year")
    ann_rows = []
    for y in range(1975, 2026):
        sc24 = ex_ann.loc[y, "sc2.4_strong_aftertax_pct"] if y in ex_ann.index else np.nan
        v26 = cy26.get(y, np.nan)
        ann_rows.append({
            "year": y,
            "sc2.4_pct": _round1_halfup(float(sc24)) if sc24 == sc24 else np.nan,
            "sc2.6_pct": _round1_halfup(float(v26) * 100) if v26 == v26 else np.nan,
        })
    ann = pd.DataFrame(ann_rows)
    ann.to_csv(os.path.join(AR_DIR, "scale_dial_sc26_annual_20260628.csv"),
               index=False, float_format="%.1f", encoding="utf-8-sig")

    # annual stats sc2.6 (1975-2025, ddof=1)
    v = np.asarray(ann["sc2.6_pct"].dropna(), float)
    stats26 = dict(mean=round(float(np.mean(v)), 1), median=round(float(np.median(v)), 1),
                   std=round(float(np.std(v, ddof=1)), 1), max=round(float(np.max(v)), 1),
                   min=round(float(np.min(v)), 1), pos=int(np.sum(v > 0)), neg=int(np.sum(v < 0)))
    print("\n  sc2.6 annual stats (after-tax): mean=%+.1f%% med=%+.1f%% std=%.1f%% max=%+.1f%% "
          "min=%+.1f%% pos/neg=%d/%d"
          % (stats26["mean"], stats26["median"], stats26["std"], stats26["max"],
             stats26["min"], stats26["pos"], stats26["neg"]))

    # ---- premise check: "is sc2.6 below ~51% in >=half the years?" ----
    pct = ann["sc2.6_pct"].dropna().astype(float)
    below51 = int((pct < 51.0).sum())
    print("\n--- PREMISE CHECK (user said 'years with sc2.6 net return < ~51%% drain the run') ---")
    print("  sc2.6 annual after-tax return < 51%% in %d of %d years (%.0f%%)."
          % (below51, len(pct), 100.0 * below51 / len(pct)))
    print("  But the 51%% threshold is NOT what triggers the top-up. The top-up fires when the")
    print("  run sleeve BALANCE < 14M after that year's 7.2M withdrawal and growth (a path effect).")

    # ========== (B) labor-zero backfill trace ==========
    print("\n=== (B) labor-zero backfill trace: sc2.6 run14M / res26M bond / topup-ALL @14M ===")
    # build the after-tax calendar-year series the simulator uses (same as harness v2)
    sc26_at = cy26.copy()  # already x0.8273, year->frac
    bond_cy = _calendar_year_returns(
        pd.Series(np.cumprod(1.0 + np.asarray(ctx["ret_bond"], float)), index=ctx["dates_dt"]))
    bond_at = bond_cy[(bond_cy.index >= 1975) & (bond_cy.index <= 2025)] * AFTER_TAX

    trace_rows = []
    topup_year_counter = {}
    total_labor = 0
    for sy in START_YEARS:
        run, res = RUN0, RES0
        fired = []           # calendar years the top-up fires
        run_at_fire = []     # run balance just before top-up
        labor = 0
        min_total = run + res
        for k in range(HORIZON):
            yr = sy + k
            # 1. top-up (one-way, all-in) when run < 14M
            if run < THR and res > 1e-6:
                fired.append(yr)
                run_at_fire.append(round(run / M, 2))
                run += res
                res = 0.0
                topup_year_counter[yr] = topup_year_counter.get(yr, 0) + 1
            # 2. strict spend
            total = run + res
            if total + 1e-6 < SPEND:
                labor += 1
                run = res = 0.0
            else:
                if run >= SPEND:
                    run -= SPEND
                else:
                    res -= (SPEND - run)
                    run = 0.0
            # 3. growth
            run *= (1.0 + float(sc26_at.loc[yr]))
            res *= (1.0 + float(bond_at.loc[yr]))
            if run + res < min_total:
                min_total = run + res
        total_labor += labor
        trace_rows.append({
            "start": sy, "labor_years": labor,
            "n_topups": len(fired),
            "topup_years": ";".join(str(y) for y in fired),
            "run_balance_at_each_topup_M": ";".join("%.2f" % b for b in run_at_fire),
            "terminal_M": round((run + res) / M, 1),
            "min_total_M": round(min_total / M, 2),
        })

    tdf = pd.DataFrame(trace_rows)
    tdf.to_csv(os.path.join(AR_DIR, "labor_zero_backfill_trace_20260628.csv"),
               index=False, encoding="utf-8-sig")

    print("  total labor years over all 31 starts x 20y = %d  (expected 0)" % total_labor)
    print("  floor (min total over all starts x years)  = %.2fM" % tdf["min_total_M"].min())
    n_no_topup = int((tdf["n_topups"] == 0).sum())
    print("  starts that NEVER needed a top-up = %d of 31" % n_no_topup)
    print("  starts that needed >=1 top-up     = %d of 31" % (31 - n_no_topup))

    print("\n  --- per-start backfill (only starts that fired a top-up) ---")
    print("  %-6s %-7s %s" % ("start", "labor", "top-up calendar year(s)  [run balance M just before]"))
    for _, rr in tdf.iterrows():
        if rr["n_topups"] > 0:
            pairs = ", ".join(
                "%s [%.2fM]" % (y, float(b))
                for y, b in zip(rr["topup_years"].split(";"),
                                rr["run_balance_at_each_topup_M"].split(";")))
            print("  %-6d %-7d %s" % (rr["start"], rr["labor_years"], pairs))

    # histogram of WHICH calendar years ever see a top-up (across all starts)
    print("\n  --- calendar years that EVER trigger a top-up (count across the 31 starts) ---")
    for y in sorted(topup_year_counter):
        print("    %d : %d start(s)" % (y, topup_year_counter[y]))

    block = {
        "script": "scale_dial_sc26_addendum_20260628.py", "date": "2026-06-28",
        "sanity_sc2.0_pass": bool(sane),
        "sc26_std10": {k: std_row[k] for k in
                       ("CAGR_IS", "CAGR_OOS", "min9", "IS_OOS_gap_pp", "Sharpe_FULL",
                        "Sharpe_OOS", "MaxDD", "Worst1D", "Worst1D_date", "Worst10Y",
                        "Worst5Y", "P10_5Y", "Trades_yr", "max_eff_lev", "WFE", "CI95_lo",
                        "Regime_min")},
        "sc26_annual_stats": stats26,
        "sc26_below51_years": below51,
        "labor_zero_total_labor": total_labor,
        "labor_zero_floor_M": float(tdf["min_total_M"].min()),
        "starts_no_topup": n_no_topup,
        "topup_year_histogram": topup_year_counter,
    }
    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
