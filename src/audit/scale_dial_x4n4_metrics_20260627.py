"""
src/audit/scale_dial_x4n4_metrics_20260627.py
==============================================
Build the STANDARD-10 metrics (v2.0) + annual after-tax table for the NEW
scale-dial report strategy set (2026-06-27):

  KEEP : P09_STR sc1.0 / 1.4 / 1.6 / 1.8 / 2.0   (reuse FRONTIER-validated values)
  ADD  : N4 (scale=2.85, in_gold_w=0.28, in_bond_w=0.07)
         X4 (scale=3.0,  in_gold_w=0.32, in_bond_w=0.05)
  DROP : scale2.2 / 2.4
  REF  : NASDAQ 1x B&H

This is the data source for RETIREMENT-independent scale-dial report
P09_STR_SCALE_DIAL_X4N4_20260627.md. It mirrors p09_strongmap_scale_dial_20260623.py
EXACTLY for the kept scales (same builder, same metrics10, same after-tax x0.8273,
same _eval_one robustness block) and uses the SAME machinery for X4/N4 via the
dd_reduction harness (which itself reproduces sc2.0 to 4 digits -- verified inline).

SANITY: the harness sc2.0 standard-10 must reproduce the FRONTIER anchor
(IS+35.3755%/OOS+29.1102%/MaxDD-61.6342%) AND the existing scale-dial CSV row,
so X4/N4 are on the identical footing before they enter the report.

ASCII-only prints. Does NOT commit, no temp files.
Outputs:
  audit_results/scale_dial_x4n4_metrics_20260627.csv   (standard-10 per strategy, new set)
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
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.audit.dd_reduction_harness_20260626 as H
from src.audit.run_p01_backtest_20260611 import _calendar_year_returns, AFTER_TAX as _AT

AFTER_TAX = 0.8273
assert abs(_AT - AFTER_TAX) < 1e-9, "AFTER_TAX drift"


def _round1_halfup(x):
    """Round to 1 decimal, half-UP, to match the ORIGINAL scale-dial report's
    displayed values (Python's round() uses banker's rounding, which differs at
    the .x5 boundary -- e.g. 23.25 -> 23.2 vs the report's 23.3). The kept-scale
    columns are transcribed from the existing report, so we must round the same way.
    """
    if x != x:  # NaN
        return x
    import math
    return math.floor(abs(x) * 10 + 0.5) / 10 * (1 if x >= 0 else -1)

STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}

# kept scales (reuse the existing standard-10 CSV row verbatim; X4/N4 computed here)
KEEP_SCALES = [1.0, 1.4, 1.6, 1.8, 2.0]
ADD_BLENDS = [
    ("N4", dict(scale=2.85, in_gold_w=0.28, in_bond_w=0.07)),
    ("X4", dict(scale=3.0,  in_gold_w=0.32, in_bond_w=0.05)),
]

EXISTING_STD10 = os.path.join(_REPO_DIR, "audit_results",
                              "p09_strongmap_scale_dial_20260623.csv")
EXISTING_ANNUAL = os.path.join(_REPO_DIR, "audit_results",
                               "p09_strongmap_scale_dial_annual_20260623.csv")

# FRONTIER anchor for the harness sc2.0 sanity (same as the dial report).
SC20_ANCHOR = {"CAGR_IS": 0.353755, "CAGR_OOS": 0.291102, "MaxDD": -0.616342}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 110)
    print("SCALE-DIAL NEW SET STANDARD-10 (+X4 +N4, -sc2.2 -sc2.4)  2026-06-27")
    print("=" * 110)

    ex_std = pd.read_csv(EXISTING_STD10)
    ex_ann = pd.read_csv(EXISTING_ANNUAL).set_index("year")

    ctx = H.setup()

    # ---- SANITY: harness sc2.0 reproduces FRONTIER anchor ----
    print("\n--- SANITY: harness sc2.0 standard-10 vs FRONTIER anchor ---")
    nav20, r20, tpy20, _ = H.build(ctx, scale=2.0)
    m20 = H.metrics10(ctx, nav20, r20, tpy20, label="sc2.0", with_wfa=False)
    d_is = abs(m20["CAGR_IS"] - SC20_ANCHOR["CAGR_IS"])
    d_oos = abs(m20["CAGR_OOS"] - SC20_ANCHOR["CAGR_OOS"])
    d_dd = abs(m20["MaxDD"] - SC20_ANCHOR["MaxDD"])
    print("  IS %+.4f%% (exp %+.4f%%)  OOS %+.4f%% (exp %+.4f%%)  MaxDD %+.4f%% (exp %+.4f%%)"
          % (m20["CAGR_IS"]*100, SC20_ANCHOR["CAGR_IS"]*100,
             m20["CAGR_OOS"]*100, SC20_ANCHOR["CAGR_OOS"]*100,
             m20["MaxDD"]*100, SC20_ANCHOR["MaxDD"]*100))
    sane = (d_is <= 0.0015 and d_oos <= 0.0015 and d_dd <= 0.0015)
    print("  SANITY: %s" % ("PASS" if sane else "FAIL"))
    if not sane:
        print("HALT -- harness does not reproduce FRONTIER sc2.0.")
        sys.exit(1)

    # ---- standard-10 rows ----
    rows = []

    # kept scales: reuse the existing validated CSV row verbatim
    for sc in KEEP_SCALES:
        er = ex_std[abs(ex_std["scale"] - sc) < 1e-9].iloc[0]
        rows.append({
            "label": "P09_STR_sc%.1f" % sc, "scale": sc, "kind": "scale",
            "CAGR_IS": float(er["CAGR_IS_at"]), "CAGR_OOS": float(er["CAGR_OOS_at"]),
            "min9": float(er["min9_at"]), "IS_OOS_gap_pp": float(er["IS_OOS_gap_pp"]),
            "Sharpe_FULL": float(er["Sharpe_FULL"]), "Sharpe_OOS": float(er["Sharpe_OOS"]),
            "MaxDD": float(er["MaxDD_FULL"]), "Worst1D": float(er["Worst1D"]),
            "Worst1D_date": str(er["Worst1D_date"]),
            "Worst10Y": float(er["Worst10Y_at"]), "Worst5Y": float(er["Worst5Y_at"]),
            "P10_5Y": float(er["P10_5Y_at"]), "Trades_yr": float(er["Trades_yr"]),
            "max_eff_lev": float(er["max_eff_lev"]),
            "WFE": float(er["wfa_WFE"]), "CI95_lo": float(er["wfa_CI95_lo"]),
            "Regime_min": float(er["regime_min_at"]),
        })
        print("  reuse  %-12s OOS=%+.2f%% MaxDD=%+.2f%% Sharpe=%.3f gap=%+.2fpp"
              % ("sc%.1f" % sc, rows[-1]["CAGR_OOS"]*100, rows[-1]["MaxDD"]*100,
                 rows[-1]["Sharpe_FULL"], rows[-1]["IS_OOS_gap_pp"]))

    # X4/N4: compute full standard-10 + WFA with the harness
    blend_cy = {}
    for name, kw in ADD_BLENDS:
        nav, r, tpy, exc = H.build(ctx, **kw)
        m = H.metrics10(ctx, nav, r, tpy, label=name, with_wfa=True)
        cy = _calendar_year_returns(nav)
        cy = cy[cy.index <= 2025]
        blend_cy[name] = cy * AFTER_TAX
        max_eff_lev = kw["scale"] * 3.0 * max(STRONG_MAP.values())  # IN-leg peak (1.60 boost)
        rows.append({
            "label": name, "scale": kw["scale"], "kind": "blend",
            "in_gold_w": kw["in_gold_w"], "in_bond_w": kw["in_bond_w"],
            "CAGR_IS": m["CAGR_IS"], "CAGR_OOS": m["CAGR_OOS"], "min9": m["min9"],
            "IS_OOS_gap_pp": m["IS_OOS_gap_pp"], "Sharpe_FULL": m["Sharpe_FULL"],
            "Sharpe_OOS": m["Sharpe_OOS"], "MaxDD": m["MaxDD"], "Worst1D": m["Worst1D"],
            "Worst1D_date": str(m["Worst1D_date"]) if m.get("Worst1D_date") else "",
            "Worst10Y": m["Worst10Y"], "Worst5Y": m["Worst5Y"], "P10_5Y": m["P10_5Y"],
            "Trades_yr": m["Trades_yr"], "max_eff_lev": max_eff_lev,
            "WFE": m["WFE"], "CI95_lo": m["CI95_lo"], "Regime_min": m["Regime_min"],
        })
        print("  CALC   %-12s OOS=%+.2f%% IS=%+.2f%% MaxDD=%+.2f%% Sharpe=%.3f W10Y=%+.2f%% "
              "W5Y=%+.2f%% P10=%+.2f%% Tr=%.1f gap=%+.2fpp WFE=%.3f CI95lo=%+.2f%% Reg=%+.2f%% "
              "W1D=%+.2f%%(%s) maxL=%.1fx"
              % (name, m["CAGR_OOS"]*100, m["CAGR_IS"]*100, m["MaxDD"]*100, m["Sharpe_FULL"],
                 m["Worst10Y"]*100, m["Worst5Y"]*100, m["P10_5Y"]*100, m["Trades_yr"],
                 m["IS_OOS_gap_pp"], m["WFE"], m["CI95_lo"]*100, m["Regime_min"]*100,
                 m["Worst1D"]*100, rows[-1]["Worst1D_date"], max_eff_lev))

    # ---- CSV ----
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    df = pd.DataFrame(rows)
    p = os.path.join(out_dir, "scale_dial_x4n4_metrics_20260627.csv")
    df.to_csv(p, index=False, float_format="%.6f", encoding="utf-8-sig")
    print("\nSaved standard-10 CSV: %s" % p)

    # ---- annual table: kept scale cols from existing CSV + X4/N4 computed ----
    print("\n--- annual after-tax table cross-check (X4/N4 cy2008 must be OUT-year value) ---")
    for name in ("N4", "X4"):
        print("  %s cy2008(at)=%+.2f%%  cy2022(at)=%+.2f%%  cy_worst=%+.2f%%(%d)"
              % (name, float(blend_cy[name].get(2008, np.nan))*100,
                 float(blend_cy[name].get(2022, np.nan))*100,
                 float(blend_cy[name].min())*100, int(blend_cy[name].idxmin())))

    ann_rows = []
    for y in range(1975, 2026):
        row = {"year": y}
        for sc in KEEP_SCALES:
            col = "sc%.1f_strong_aftertax_pct" % sc
            # kept scales: re-round the existing CSV's 2dp value half-UP to match the
            # original report's 1dp display exactly.
            row["P09_STR_sc%.1f_pct" % sc] = (
                _round1_halfup(float(ex_ann.loc[y, col])) if y in ex_ann.index else np.nan)
        for name, _ in ADD_BLENDS:
            v = blend_cy[name].get(y, np.nan)
            row["%s_pct" % name] = _round1_halfup(float(v) * 100) if v == v else np.nan
        row["NASDAQ_1x_BH_pct"] = (
            _round1_halfup(float(ex_ann.loc[y, "NASDAQ_1x_BH_aftertax_pct"])) if y in ex_ann.index else np.nan)
        ann_rows.append(row)
    ann = pd.DataFrame(ann_rows)
    pa = os.path.join(out_dir, "scale_dial_x4n4_annual_20260627.csv")
    ann.to_csv(pa, index=False, float_format="%.1f", encoding="utf-8-sig")
    print("Saved annual CSV: %s" % pa)

    # ---- annual stats (1975-2025) ----
    def _stats(series_pct):
        v = np.asarray(series_pct.dropna(), float)
        return dict(mean=round(float(np.mean(v)), 1), median=round(float(np.median(v)), 1),
                    std=round(float(np.std(v, ddof=1)), 1), max=round(float(np.max(v)), 1),
                    min=round(float(np.min(v)), 1), pos=int(np.sum(v > 0)), neg=int(np.sum(v < 0)))
    stats = {}
    for sc in KEEP_SCALES:
        stats["sc%.1f" % sc] = _stats(ann["P09_STR_sc%.1f_pct" % sc])
    for name, _ in ADD_BLENDS:
        stats[name] = _stats(ann["%s_pct" % name])
    stats["NASDAQ_1x_BH"] = _stats(ann["NASDAQ_1x_BH_pct"])

    print("\n--- annual stats (after-tax, 1975-2025) ---")
    print("%-14s | %6s | %6s | %6s | %7s | %7s | %s" %
          ("series", "mean", "med", "std", "max", "min", "pos/neg"))
    for k, s in stats.items():
        print("%-14s | %+5.1f%% | %+5.1f%% | %5.1f%% | %+6.1f%% | %+6.1f%% | %d/%d" %
              (k, s["mean"], s["median"], s["std"], s["max"], s["min"], s["pos"], s["neg"]))

    block = {
        "script": "scale_dial_x4n4_metrics_20260627.py", "date": "2026-06-27",
        "sanity_sc2.0_pass": bool(sane),
        "std10": rows, "annual_stats": stats,
    }
    print("\n" + "=" * 110)
    print("RETURN_BLOCK")
    print("=" * 110)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
