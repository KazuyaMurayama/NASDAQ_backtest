"""
src/audit/p09_strongmap_scale_dial_20260623.py
==============================================
P09 STRONG-MAP leverage-scale dial: scale 1.4 / 1.6 / 1.8 / 2.0 / 2.2 / 2.4
(+ scale 1.0 reference, + NASDAQ 1x B&H).

WHY THIS SCRIPT EXISTS (correction of the previous default-map dial):
  The previous src/audit/p09_scale_dial_20260623.py used the V7 *default* boost
  map {0:1.20,1:1.10,2:1.00,3:1.00} -> a LOWER-leverage, INFERIOR dial.
  The canonical "P09 leverage-extension frontier" that CURRENT_BEST_STRATEGY.md
  (L72/L270) marks for adoption uses the *strong* boost map
      B3A_MAP_STRONG = {0:1.60,1:1.50,2:1.10,3:1.00}
  built via _build_full_c1 (k365 cost model, EXCESS_EXTRA = 0.0025 always-on).
  This script rebuilds the dial on THAT strong base so the report matches the
  adopted P09_STR_S* series (P09_STR_S2.0 = +35.38% IS / +29.11% OOS).

  SANITY anchors (from leverext_high_stage1_20260618.csv / sc180_200 CSV):
    sc1.40: CAGR_IS +27.4877% / OOS +24.3414% / MaxDD -46.4781%
    sc1.60: CAGR_IS +30.2649% / OOS +26.2127% / MaxDD -51.9521%
    sc1.80: CAGR_IS +32.8989% / OOS +27.8068% / MaxDD -56.9995%
    sc2.00: CAGR_IS +35.3755% / OOS +29.1102% / MaxDD -61.6342%

Standard 10 metrics v2.0 (EVALUATION_STANDARD §3.12) + calendar-year returns
1975-2025 (after-tax x0.8273) + NASDAQ 1x B&H reference.

Reuses validated builders only (no reimplementation). ASCII-only prints. Does
NOT commit, no temp files.
Outputs:
  audit_results/p09_strongmap_scale_dial_20260623.csv          (standard-10 per scale)
  audit_results/p09_strongmap_scale_dial_annual_20260623.csv   (calendar-year, after-tax)
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

# Strong-map builder = same one the FRONTIER uses (k365 cost model + C1 fill)
from src.audit.k365_recost_20260612 import _build_full_c1, EXCESS_EXTRA_K365_CENTRE
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base
from src.audit.extended_eval_20260611 import _eval_one
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)

AFTER_TAX = 0.8273
HARD_VETO_MAXDD = -0.50
HARD_VETO_WFE = 1.5
HARD_VETO_W10Y = 0.0
HARD_VETO_REGIME = -0.10

# Strong boost map (= B3A_MAP_STRONG in leverext_high_*; the adopted P09_STR base)
STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025 (k365 centre), always-on -> matches FRONTIER

SCALES = [1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

# Sanity anchors from the existing FRONTIER CSVs (strong map)
STRONG_ANCHORS = {
    1.4: {"CAGR_IS": 0.274877, "CAGR_OOS": 0.243414, "MaxDD": -0.464781},
    1.6: {"CAGR_IS": 0.302649, "CAGR_OOS": 0.262127, "MaxDD": -0.519521},
    1.8: {"CAGR_IS": 0.328989, "CAGR_OOS": 0.278068, "MaxDD": -0.569995},
    2.0: {"CAGR_IS": 0.353755, "CAGR_OOS": 0.291102, "MaxDD": -0.616342},
}
ANCHOR_TOL = 0.0015   # +/-0.15pp

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


def main():
    print("=" * 110)
    print("P09 STRONG-MAP LEVERAGE-SCALE DIAL  2026-06-23")
    print("Base: STRONG_MAP=%s x lev_scale ; _build_full_c1 + C1 ; EXCESS_EXTRA=%.4f"
          % (STRONG_MAP, EXCESS_EXTRA))
    print("Scales: %s" % SCALES)
    print("Sanity: sc1.4-2.0 must reproduce FRONTIER CSV (sc2.0 OOS +29.11%% MaxDD -61.63%%)")
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

    # baseline V7_TQQQ for WFA/bootstrap baseline (default map, scale 1.0)
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    def build_strong(sc):
        return _build_full_c1(
            shared, dates_dt, n_years, ret_gold, ret_bond, fund_active,
            wg, wb, bond_on, sofr_arr,
            v7_map=STRONG_MAP, lev_scale=sc, excess_extra=EXCESS_EXTRA)

    # =====================================================================
    # SANITY: sc1.4-2.0 reproduce FRONTIER CSV values
    # =====================================================================
    print("\n--- SANITY: strong-map sc1.4-2.0 vs FRONTIER CSV ---")
    sane = True
    for sc in (1.4, 1.6, 1.8, 2.0):
        nav_dt, r, tpy, exc = build_strong(sc)
        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)
        exp = STRONG_ANCHORS[sc]
        d_is = abs(aft["CAGR_IS"] - exp["CAGR_IS"])
        d_oos = abs(aft["CAGR_OOS"] - exp["CAGR_OOS"])
        d_dd = abs(pre["MaxDD_FULL"] - exp["MaxDD"])
        ok = (d_is <= ANCHOR_TOL and d_oos <= ANCHOR_TOL and d_dd <= ANCHOR_TOL)
        sane = sane and ok
        print("  sc%.1f: IS %+.4f%%(exp %+.4f) OOS %+.4f%%(exp %+.4f) MaxDD %+.4f%%(exp %+.4f) -> %s"
              % (sc, aft["CAGR_IS"]*100, exp["CAGR_IS"]*100,
                 aft["CAGR_OOS"]*100, exp["CAGR_OOS"]*100,
                 pre["MaxDD_FULL"]*100, exp["MaxDD"]*100, "OK" if ok else "FAIL"))
    if not sane:
        print("\nSANITY FAILED -- strong-map wiring does not reproduce FRONTIER. Halting.")
        sys.exit(1)
    print("  SANITY PASSED (strong-map dial reproduces FRONTIER sc1.4-2.0).\n")

    # =====================================================================
    # SCALE DIAL: 1.0 .. 2.4
    # =====================================================================
    results = []
    cy_map = {}
    print("=" * 110)
    print("STRONG-MAP SCALE DIAL")
    print("=" * 110)
    for sc in SCALES:
        print("\n  [P09_STR_sc%.2f] building + full gate ..." % sc)
        nav_dt, r, tpy, exc = build_strong(sc)
        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)
        ev = _eval_one("P09_STR_sc%.2f" % sc, nav_dt, r, regimes, stress,
                       is_mask, oos_mask, baseline_r=r_v7)

        max_eff_lev = max(STRONG_MAP.values()) * sc * 3.0
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
              % (rec["min9_at"]*100, rec["CAGR_IS_at"]*100, rec["CAGR_OOS_at"]*100,
                 rec["Sharpe_FULL"], maxdd*100))
        print("    W10Y*=%+.2f%%  W5Y=%+.2f%%  P10_5Y=%+.2f%%  Trd/yr=%.1f  maxL=%.1fx  excess=%.1f%%"
              % (w10y*100, rec["Worst5Y_at"]*100, rec["P10_5Y_at"]*100,
                 rec["Trades_yr"], max_eff_lev, 100*rec["excess_ratio"]))
        print("    WFE=%.4f  CI95_lo=%+.2f%%  Reg_min=%+.2f%%  gap=%+.2fpp  [%s]"
              % (wfe, rec["wfa_CI95_lo"]*100, reg_min*100, rec["IS_OOS_gap_pp"],
                 "VETO: " + rec["veto_reasons"] if veto else "PASS"))

    # =====================================================================
    # CALENDAR-YEAR RETURNS (after-tax)
    # =====================================================================
    cy_ndx = _load_nasdaq_bh()
    years = sorted(set().union(*[set(cy_map[sc].index.tolist()) for sc in SCALES]))

    # =====================================================================
    # CSV OUTPUT
    # =====================================================================
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)

    df_main = pd.DataFrame([{
        "label": "P09_STR_sc%.2f" % r["scale"], "scale": r["scale"],
        "boost_map": "strong", "excess_extra": EXCESS_EXTRA,
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
    p1 = os.path.join(out_dir, "p09_strongmap_scale_dial_20260623.csv")
    df_main.to_csv(p1, index=False, float_format="%.6f", encoding="utf-8-sig")
    print("\nSaved: %s" % p1)

    ar_rows = []
    for yr in years:
        row = {"year": yr}
        for sc in SCALES:
            v = cy_map[sc].loc[yr] * AFTER_TAX * 100 if yr in cy_map[sc].index else float("nan")
            tag = "sc%.1f_strong_aftertax_pct" % sc
            row[tag] = round(v, 2) if v == v else float("nan")
        ndx = cy_ndx.loc[yr] * AFTER_TAX * 100 if yr in cy_ndx.index else float("nan")
        row["NASDAQ_1x_BH_aftertax_pct"] = round(ndx, 2) if ndx == ndx else float("nan")
        ar_rows.append(row)
    p2 = os.path.join(out_dir, "p09_strongmap_scale_dial_annual_20260623.csv")
    pd.DataFrame(ar_rows).to_csv(p2, index=False, float_format="%.2f", encoding="utf-8-sig")
    print("Saved: %s" % p2)

    # ---- annual stats ----
    target_years = list(range(1975, 2026))
    def _stats(cy):
        vals = np.array([cy.loc[y] * AFTER_TAX * 100 for y in target_years if y in cy.index])
        return (float(np.mean(vals)), float(np.median(vals)), float(np.std(vals, ddof=1)),
                float(np.max(vals)), float(np.min(vals)),
                int(np.sum(vals > 0)), int(np.sum(vals < 0)))
    stat_block = {}
    for sc in SCALES:
        stat_block["sc%.1f" % sc] = _stats(cy_map[sc])
    stat_block["NDX1x"] = _stats(cy_ndx)

    block = {
        "script": "p09_strongmap_scale_dial_20260623.py", "date": "2026-06-23",
        "boost_map": STRONG_MAP, "excess_extra": EXCESS_EXTRA,
        "sanity_pass": True,
        "scales": [{
            "scale": r["scale"],
            "min9_at_pct": round(r["min9_at"]*100, 4),
            "CAGR_IS_at_pct": round(r["CAGR_IS_at"]*100, 4),
            "CAGR_OOS_at_pct": round(r["CAGR_OOS_at"]*100, 4),
            "Sharpe_FULL": round(r["Sharpe_FULL"], 4),
            "MaxDD_pct": round(r["MaxDD_FULL"]*100, 4),
            "Worst1D_pct": round(r["Worst1D"]*100, 4) if r["Worst1D"] is not None else None,
            "Worst1D_date": r["Worst1D_date"],
            "Worst10Y_at_pct": round(r["Worst10Y_at"]*100, 4),
            "Worst5Y_at_pct": round(r["Worst5Y_at"]*100, 4),
            "P10_5Y_at_pct": round(r["P10_5Y_at"]*100, 4),
            "Trades_yr": round(r["Trades_yr"], 2),
            "max_eff_lev": round(r["max_eff_lev"], 3),
            "excess_ratio_pct": round(r["excess_ratio"]*100, 2),
            "wfa_WFE": round(r["wfa_WFE"], 4),
            "wfa_CI95_lo_pct": round(r["wfa_CI95_lo"]*100, 4),
            "cpcv_p10_at_pct": round(r["cpcv_p10_at"]*100, 4),
            "regime_min_at_pct": round(r["regime_min_at"]*100, 4),
            "IS_OOS_gap_pp": round(r["IS_OOS_gap_pp"], 4),
            "VETO": r["VETO"], "veto_reasons": r["veto_reasons"],
        } for r in results],
        "annual_stats_aftertax": {k: {
            "mean": round(v[0], 4), "median": round(v[1], 4), "std": round(v[2], 4),
            "max": round(v[3], 4), "min": round(v[4], 4), "pos": v[5], "neg": v[6],
        } for k, v in stat_block.items()},
        "annual_aftertax_pct": {
            ("sc%.1f" % sc): {int(y): round(float(cy_map[sc].loc[y]*AFTER_TAX*100), 2)
                              for y in cy_map[sc].index if 1975 <= y <= 2025}
            for sc in SCALES},
        "ndx_aftertax_pct": {int(y): round(float(cy_ndx.loc[y]*AFTER_TAX*100), 2)
                             for y in cy_ndx.index if 1975 <= y <= 2025},
    }
    print("\n" + "=" * 110)
    print("RETURN_BLOCK")
    print("=" * 110)
    print(json.dumps(block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
