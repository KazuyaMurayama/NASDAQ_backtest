"""
src/audit/p09_scale_dial_20260623.py
====================================
GENUINE P09_C1 leverage-scale dial: scale 1.4 / 1.6 / 1.8 / 2.0 / 2.2 / 2.4.

WHY THIS SCRIPT EXISTS (premise correction):
  The existing LEVERUP_SCALE_FRONTIER_20260619.md "scale 1.40-2.00" series is
  built on B3A_MAP_STRONG = {0:1.60,1:1.50,2:1.10,3:1.00} via _build_full_c1
  (B3a/DH-W1 + k365 excess). That is the *B3a strong-map* extension, NOT the
  P09_C1 dial. P09_C1's real base is:
      _build_tqqq_base(v7_map=None [V7_MAP={0:1.20,1:1.10,2:1.00,3:1.00}],
                       lev_scale=1.0, cfd_excess=False)
    + _build_p09_on_base_c1 (Gold/Bond OUT-fill + C1 SOFR cash yield).
  This script extends THAT base with lev_scale in {1.4..2.4}, charging the
  k365 CFD excess (cfd_excess=True) on the (L-3)+ notional that scaling creates.

  Sanity anchor: scale=1.0 with cfd_excess=False must reproduce P09_C1
  (min9 +17.77% +/-0.15pp, MaxDD -34.99% +/-0.20pp).

Standard 10 metrics v2.0 (EVALUATION_STANDARD §3.12) + calendar-year returns
1975-2025 (after-tax x0.8273) + NASDAQ 1x B&H reference.

Reuses validated builders only (no reimplementation). ASCII-only prints. Does
NOT commit, no temp files.
Outputs:
  audit_results/p09_scale_dial_20260623.csv          (standard-10 per scale)
  audit_results/p09_scale_dial_annual_20260623.csv   (calendar-year, after-tax)
"""

from __future__ import annotations

import json
import os
import sys
import types

if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.regime_labeler_20260611 import build_regime_labels, stress_masks

from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base
from src.audit.leverup_b1c1_20260612 import _build_p09_on_base_c1
from src.audit.extended_eval_20260611 import _eval_one
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import V7_MAP

AFTER_TAX = 0.8273
HARD_VETO_MAXDD = -0.50
HARD_VETO_WFE = 1.5
HARD_VETO_W10Y = 0.0
HARD_VETO_REGIME = -0.10

SCALES = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

# P09_C1 known anchor (scale=1.0, cfd_excess=False) from leverup_b1c1_20260612.csv
P09_C1_ANCHOR = {"min9": 0.177672, "MaxDD": -0.349879}
ANCHOR_TOL_MIN9 = 0.0015
ANCHOR_TOL_MAXDD = 0.0020

NASDAQ_CSV_PATH = os.path.join(_REPO_DIR, "NASDAQ_extended_to_2026.csv")


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _load_nasdaq_bh():
    df = pd.read_csv(NASDAQ_CSV_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    close = df["Close"].dropna()
    yearly = close.resample("YE").last()
    ann = yearly.pct_change().dropna()
    ann = ann[ann.index.year <= 2025]
    return pd.Series(ann.values, index=ann.index.year)


def _build_p09_scaled(shared, dates_dt, n_years, ret_gold, ret_bond,
                      fund_active, wg, wb, bond_on, sofr_arr,
                      lev_scale, cfd_excess):
    """Genuine P09_C1 at a given lev_scale (V7 default map x scale)."""
    base_nav, r_base, tpy_base, exc = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=lev_scale, cfd_excess=cfd_excess)
    nav_dt, r_p09c1, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        sofr_arr, dates_dt, tpy_base, n_years)
    return nav_dt, r_p09c1, tpy, exc


def main():
    print("=" * 110)
    print("P09_C1 LEVERAGE-SCALE DIAL  2026-06-23")
    print("Base: V7_MAP=%s  x lev_scale ; P09 OUT-fill + C1 SOFR cash yield" % V7_MAP)
    print("Scales: %s   cfd_excess=True (k365 charge on (L-3)+ for scaled dial)" % SCALES)
    print("Sanity anchor: scale=1.0 cfd_excess=False -> P09_C1 min9 +17.77%% / MaxDD -34.99%%")
    print("=" * 110)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    is_mask = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)

    # ---- baseline V7_TQQQ for bootstrap / WFA baseline ----
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    # =====================================================================
    # SANITY: scale=1.0, cfd_excess=False must reproduce P09_C1
    # =====================================================================
    print("\n--- SANITY: P09_C1 anchor (scale=1.0, cfd_excess=False) ---")
    s_nav, s_r, s_tpy, s_exc = _build_p09_scaled(
        shared, dates_dt, n_years, ret_gold, ret_bond, fund_active, wg, wb,
        bond_on, sofr_arr, lev_scale=1.0, cfd_excess=False)
    s_pre = compute_10metrics(s_nav, s_tpy)
    s_aft = _apply_aftertax(s_pre)
    s_min9 = _min_at(s_aft)
    s_maxdd = s_pre["MaxDD_FULL"]
    ok_min9 = abs(s_min9 - P09_C1_ANCHOR["min9"]) <= ANCHOR_TOL_MIN9
    ok_maxdd = abs(s_maxdd - P09_C1_ANCHOR["MaxDD"]) <= ANCHOR_TOL_MAXDD
    print("  min9:  got %+.4f%%  expect %+.4f%%  -> %s"
          % (s_min9 * 100, P09_C1_ANCHOR["min9"] * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD: got %+.4f%%  expect %+.4f%%  -> %s"
          % (s_maxdd * 100, P09_C1_ANCHOR["MaxDD"] * 100, "OK" if ok_maxdd else "FAIL"))
    print("  excess(L>3) days at scale1.0 = %d (%.2f%%)" % (s_exc, 100.0 * s_exc / n))
    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED -- halting. P09_C1 base wiring mismatch.")
        sys.exit(1)
    print("  SANITY PASSED.\n")

    # Also report scale=1.0 WITH cfd_excess to quantify the honest-cost gap
    s_nav_x, s_r_x, s_tpy_x, s_exc_x = _build_p09_scaled(
        shared, dates_dt, n_years, ret_gold, ret_bond, fund_active, wg, wb,
        bond_on, sofr_arr, lev_scale=1.0, cfd_excess=True)
    s_aft_x = _apply_aftertax(compute_10metrics(s_nav_x, s_tpy_x))
    print("  [info] scale=1.0 WITH cfd_excess: min9=%+.4f%% (vs %.4f%% without) "
          "-> k365 charge costs %+.3fpp at scale1.0"
          % (_min_at(s_aft_x) * 100, s_min9 * 100,
             (_min_at(s_aft_x) - s_min9) * 100))

    # =====================================================================
    # SCALE DIAL: 1.4 .. 2.4 (cfd_excess=True)
    # =====================================================================
    results = []
    cy_map = {}
    print("=" * 110)
    print("SCALE DIAL (cfd_excess=True)")
    print("=" * 110)
    for sc in SCALES:
        print("\n  [P09_C1_sc%.2f] building + full gate ..." % sc)
        nav_dt, r, tpy, exc = _build_p09_scaled(
            shared, dates_dt, n_years, ret_gold, ret_bond, fund_active, wg, wb,
            bond_on, sofr_arr, lev_scale=sc, cfd_excess=True)
        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)
        ev = _eval_one("P09_C1_sc%.2f" % sc, nav_dt, r, regimes, stress,
                       is_mask, oos_mask, baseline_r=r_v7)

        max_eff_lev = max(V7_MAP.values()) * sc * 3.0   # IN-day theoretical max
        wfe = float(ev["wfa_WFE"])
        reg_min = float(ev["regime_min_at"])
        w10y = aft["Worst10Y_star"]
        maxdd = pre["MaxDD_FULL"]
        veto_reasons = []
        if maxdd < HARD_VETO_MAXDD: veto_reasons.append("MaxDD<-50%%")
        if wfe > HARD_VETO_WFE: veto_reasons.append("WFE>1.5")
        if w10y < HARD_VETO_W10Y: veto_reasons.append("W10Y*<0")
        if reg_min < HARD_VETO_REGIME: veto_reasons.append("Regime_min<-10%%")
        veto = len(veto_reasons) > 0

        cy = _calendar_year_returns(nav_dt)
        cy = cy[cy.index <= 2025]
        cy_map[sc] = cy

        rec = {
            "scale": sc,
            "CAGR_IS_at": aft["CAGR_IS"], "CAGR_OOS_at": aft["CAGR_OOS"],
            "min9_at": _min_at(aft), "IS_OOS_gap_pp": aft["IS_OOS_gap_pp"],
            "Sharpe_FULL": pre["Sharpe_FULL"], "Sharpe_OOS": pre["Sharpe_OOS"],
            "MaxDD_FULL": maxdd, "Worst1D": pre["Worst1D"],
            "Worst1D_date": pre["Worst1D_date"],
            "Worst10Y_at": w10y, "Worst5Y_at": aft["Worst5Y"],
            "P10_5Y_at": aft["P10_5Y"], "Trades_yr": aft["Trades_yr"],
            "max_eff_lev": max_eff_lev, "excess_days": exc,
            "excess_ratio": exc / n,
            "wfa_WFE": wfe, "wfa_CI95_lo": float(ev["wfa_CI95_lo"]),
            "wfa_t_p": float(ev["wfa_t_p"]),
            "cpcv_p10_at": float(ev["cpcv_p10_at"]),
            "regime_min_at": reg_min,
            "VETO": int(veto), "veto_reasons": ";".join(veto_reasons),
        }
        results.append(rec)
        print("    min9=%+.2f%%  CAGR_IS=%+.2f%%  CAGR_OOS=%+.2f%%  Sharpe=%.3f  MaxDD=%+.2f%%"
              % (rec["min9_at"] * 100, rec["CAGR_IS_at"] * 100, rec["CAGR_OOS_at"] * 100,
                 rec["Sharpe_FULL"], maxdd * 100))
        print("    W10Y*=%+.2f%%  W5Y=%+.2f%%  P10_5Y=%+.2f%%  Trd/yr=%.1f  maxL=%.1fx  excess=%.1f%%"
              % (w10y * 100, rec["Worst5Y_at"] * 100, rec["P10_5Y_at"] * 100,
                 rec["Trades_yr"], max_eff_lev, 100 * rec["excess_ratio"]))
        print("    WFE=%.4f  CI95_lo=%+.2f%%  CPCV_p10=%+.2f%%  Reg_min=%+.2f%%  gap=%+.2fpp  [%s%s]"
              % (wfe, rec["wfa_CI95_lo"] * 100, rec["cpcv_p10_at"] * 100,
                 reg_min * 100, rec["IS_OOS_gap_pp"],
                 "VETO: " + rec["veto_reasons"] if veto else "PASS",
                 ""))

    # =====================================================================
    # STANDARD-10 TABLE
    # =====================================================================
    print("\n" + "=" * 150)
    print("STANDARD 10 METRICS v2.0 -- P09_C1 scale dial")
    print("%-14s | %8s | %9s | %7s | %7s | %8s | %8s | %7s | %7s | %6s | %4s"
          % ("scale", "CAGR_IS", "CAGR_OOS", "ShpFULL", "MaxDD", "W10Y*", "W5Y",
             "P10_5Y", "Trd/yr", "maxL", "s1"))
    print("-" * 150)
    for r in results:
        print("P09_C1_sc%.2f | %+7.2f%% | %+8.2f%% | %7.4f | %+6.2f%% | %+6.2f%% | %+6.2f%% | %+6.2f%% | %6.1f | %5.1fx | %-4s"
              % (r["scale"], r["CAGR_IS_at"] * 100, r["CAGR_OOS_at"] * 100,
                 r["Sharpe_FULL"], r["MaxDD_FULL"] * 100, r["Worst10Y_at"] * 100,
                 r["Worst5Y_at"] * 100, r["P10_5Y_at"] * 100, r["Trades_yr"],
                 r["max_eff_lev"], "VETO" if r["VETO"] else "PASS"))

    # =====================================================================
    # CALENDAR-YEAR RETURNS (after-tax x0.8273)
    # =====================================================================
    cy_ndx = _load_nasdaq_bh()
    years = sorted(set().union(*[set(cy_map[sc].index.tolist()) for sc in SCALES]))
    print("\n" + "=" * 130)
    print("CALENDAR-YEAR RETURNS (after-tax x%.4f) 1975-2025" % AFTER_TAX)
    hdr = "| Year |" + "".join(" sc%.1f |" % sc for sc in SCALES) + " NDX1x |"
    print(hdr)
    for yr in years:
        line = "| %4d |" % yr
        for sc in SCALES:
            v = cy_map[sc].loc[yr] * AFTER_TAX * 100 if yr in cy_map[sc].index else float("nan")
            line += (" %+5.1f |" % v) if v == v else "  N/A |"
        ndx = cy_ndx.loc[yr] * AFTER_TAX * 100 if yr in cy_ndx.index else float("nan")
        line += (" %+5.1f |" % ndx) if ndx == ndx else "  N/A |"
        print(line)

    # ---- annual stats ----
    print("\n--- ANNUAL STATS (after-tax, 1975-2025, ddof=1) ---")
    target_years = list(range(1975, 2026))
    def _stats(cy):
        vals = np.array([cy.loc[y] * AFTER_TAX * 100 for y in target_years if y in cy.index])
        return (float(np.mean(vals)), float(np.median(vals)), float(np.std(vals, ddof=1)),
                float(np.max(vals)), float(np.min(vals)),
                int(np.sum(vals > 0)), int(np.sum(vals < 0)))
    print("%-12s | %7s | %7s | %7s | %8s | %8s | %4s | %4s"
          % ("series", "mean", "median", "std", "max", "min", "pos", "neg"))
    stat_block = {}
    for sc in SCALES:
        st = _stats(cy_map[sc])
        stat_block["sc%.1f" % sc] = st
        print("sc%.2f       | %+6.1f%% | %+6.1f%% | %6.1f%% | %+7.1f%% | %+7.1f%% | %4d | %4d"
              % (sc, st[0], st[1], st[2], st[3], st[4], st[5], st[6]))
    st_ndx = _stats(cy_ndx)
    stat_block["NDX1x"] = st_ndx
    print("NASDAQ1xB&H  | %+6.1f%% | %+6.1f%% | %6.1f%% | %+7.1f%% | %+7.1f%% | %4d | %4d"
          % (st_ndx[0], st_ndx[1], st_ndx[2], st_ndx[3], st_ndx[4], st_ndx[5], st_ndx[6]))

    # =====================================================================
    # CSV OUTPUT
    # =====================================================================
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)

    df_main = pd.DataFrame([{
        "label": "P09_C1_sc%.2f" % r["scale"], "scale": r["scale"],
        "CAGR_IS_at": round(r["CAGR_IS_at"], 6), "CAGR_OOS_at": round(r["CAGR_OOS_at"], 6),
        "min9_at": round(r["min9_at"], 6), "IS_OOS_gap_pp": round(r["IS_OOS_gap_pp"], 4),
        "Sharpe_FULL": round(r["Sharpe_FULL"], 4), "Sharpe_OOS": round(r["Sharpe_OOS"], 4),
        "MaxDD_FULL": round(r["MaxDD_FULL"], 6),
        "Worst1D": round(r["Worst1D"], 6) if r["Worst1D"] is not None else float("nan"),
        "Worst1D_date": r["Worst1D_date"] or "",
        "Worst10Y_at": round(r["Worst10Y_at"], 6), "Worst5Y_at": round(r["Worst5Y_at"], 6),
        "P10_5Y_at": round(r["P10_5Y_at"], 6), "Trades_yr": round(r["Trades_yr"], 2),
        "max_eff_lev": round(r["max_eff_lev"], 3), "excess_days": r["excess_days"],
        "excess_ratio": round(r["excess_ratio"], 6),
        "wfa_WFE": round(r["wfa_WFE"], 6), "wfa_CI95_lo": round(r["wfa_CI95_lo"], 6),
        "wfa_t_p": round(r["wfa_t_p"], 10), "cpcv_p10_at": round(r["cpcv_p10_at"], 6),
        "regime_min_at": round(r["regime_min_at"], 6),
        "VETO": r["VETO"], "veto_reasons": r["veto_reasons"],
    } for r in results])
    p1 = os.path.join(out_dir, "p09_scale_dial_20260623.csv")
    df_main.to_csv(p1, index=False, float_format="%.6f", encoding="utf-8-sig")
    print("\nSaved: %s" % p1)

    ar_rows = []
    for yr in years:
        row = {"year": yr}
        for sc in SCALES:
            v = cy_map[sc].loc[yr] * AFTER_TAX * 100 if yr in cy_map[sc].index else float("nan")
            row["sc%.1f_aftertax_pct" % sc] = round(v, 2) if v == v else float("nan")
        ndx = cy_ndx.loc[yr] * AFTER_TAX * 100 if yr in cy_ndx.index else float("nan")
        row["NASDAQ_1x_BH_aftertax_pct"] = round(ndx, 2) if ndx == ndx else float("nan")
        ar_rows.append(row)
    p2 = os.path.join(out_dir, "p09_scale_dial_annual_20260623.csv")
    pd.DataFrame(ar_rows).to_csv(p2, index=False, float_format="%.2f", encoding="utf-8-sig")
    print("Saved: %s" % p2)

    block = {
        "script": "p09_scale_dial_20260623.py", "date": "2026-06-23",
        "sanity": {"min9_got_pct": round(s_min9 * 100, 4),
                   "maxdd_got_pct": round(s_maxdd * 100, 4),
                   "PASS": bool(ok_min9 and ok_maxdd),
                   "scale1.0_cfd_excess_min9_pct": round(_min_at(s_aft_x) * 100, 4)},
        "scales": [{
            "scale": r["scale"],
            "min9_at_pct": round(r["min9_at"] * 100, 4),
            "CAGR_IS_at_pct": round(r["CAGR_IS_at"] * 100, 4),
            "CAGR_OOS_at_pct": round(r["CAGR_OOS_at"] * 100, 4),
            "Sharpe_FULL": round(r["Sharpe_FULL"], 4),
            "MaxDD_pct": round(r["MaxDD_FULL"] * 100, 4),
            "Worst1D_pct": round(r["Worst1D"] * 100, 4) if r["Worst1D"] is not None else None,
            "Worst1D_date": r["Worst1D_date"],
            "Worst10Y_at_pct": round(r["Worst10Y_at"] * 100, 4),
            "Worst5Y_at_pct": round(r["Worst5Y_at"] * 100, 4),
            "P10_5Y_at_pct": round(r["P10_5Y_at"] * 100, 4),
            "Trades_yr": round(r["Trades_yr"], 2),
            "max_eff_lev": round(r["max_eff_lev"], 3),
            "excess_ratio_pct": round(r["excess_ratio"] * 100, 2),
            "wfa_WFE": round(r["wfa_WFE"], 4),
            "wfa_CI95_lo_pct": round(r["wfa_CI95_lo"] * 100, 4),
            "cpcv_p10_at_pct": round(r["cpcv_p10_at"] * 100, 4),
            "regime_min_at_pct": round(r["regime_min_at"] * 100, 4),
            "IS_OOS_gap_pp": round(r["IS_OOS_gap_pp"], 4),
            "VETO": r["VETO"], "veto_reasons": r["veto_reasons"],
        } for r in results],
        "annual_stats_aftertax": {k: {
            "mean": round(v[0], 4), "median": round(v[1], 4), "std": round(v[2], 4),
            "max": round(v[3], 4), "min": round(v[4], 4), "pos": v[5], "neg": v[6],
        } for k, v in stat_block.items()},
    }
    print("\n" + "=" * 110)
    print("RETURN_BLOCK")
    print("=" * 110)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
