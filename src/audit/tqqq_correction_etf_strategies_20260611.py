"""
src/audit/tqqq_correction_etf_strategies_20260611.py
====================================================
Generalize the validated V7 CFD-vs-TQQQ harness
(cost_model_cfd_vs_tqqq_20260611.py) to ALL FOUR ETF-designated strategies in
the canonical "ETF only" environment:

    DH-W1  = run_dhw1('realistic')            (NASDAQ leg L = lev_raw_masked*3 ; mult=1)
    V0     = run_overlay('V0','realistic')    (mult map {0:1.10,1:1.00,2:0.90,3:0.80})
    V7     = run_overlay('V7','realistic')    (mult map {0:1.20,1:1.10,2:1.00,3:1.00})
    P7     = run_p7('realistic')              (DH-W1 IN legs + GOLD75/BOND25 1x fund
                                               sleeve on OUT days; IN NASDAQ leg = TQQQ-style)

For each strategy we build the NAV two ways, swapping ONLY the NASDAQ-leg
financing/TER formula, everything else identical:

  CFD  (current canonical) : nas_ret = L*r_nas - (L-1)*(sofr + SBI_CFD_SPREAD/252)   [no TER]
  TQQQ (designed product)  : nas_ret = L*r_nas - max(L-1,0)*(sofr + SWAP_SPREAD/252) - TER_TQQQ/252

  L = lev_raw_masked * mult * 3.0  (DELAY=2 shifted).
  Gold-2x / Bond-3x legs and (P7) the 1x fund sleeve are identical between models.

Skeleton: the same baseline-mimic that reproduces run_overlay('V7','realistic') to
0.0000pp (build_nav_v7). DH-W1 and V0 reuse it with mult=1 and the V0 map. P7
reuses the DH-W1 build (mult=1) on HOLD days and grafts the fund sleeve on OUT
days (fund_active, T+5), exactly as run_p7 does.

VALIDATION GATE per strategy: CFD version must reproduce the existing run_* realistic
output within 0.1pp (printed per strategy).

After-tax: CAGR/Worst10Y/P10/Worst5Y *0.8273 ; Sharpe/MaxDD pretax.
ASCII-only prints (Windows cp932 console).
"""

from __future__ import annotations

import os
import sys
import types
import json

# ---------------------------------------------------------------------------
# multitasking stub + sys.path
# ---------------------------------------------------------------------------
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_REPO = os.path.dirname(_SRC_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd

import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics

# ---------------------------------------------------------------------------
# Constants (identical to validated V7 harness)
# ---------------------------------------------------------------------------
TRADING_DAYS = 252.0
DELAY = 2
TAX_FACTOR = 0.8273

SBI_CFD_SPREAD = 0.0300       # CFD leg (current canonical baseline)
TER_TQQQ = 0.0086             # TQQQ designed-product TER
SWAP_SPREAD = 0.0050          # TQQQ swap spread

DH_PER_UNIT = 0.0010
NAV_FLOOR = -0.999

OVERLAY_MAPS = {
    "DH_W1": None,                                   # mult = 1 everywhere
    "V0": {0: 1.10, 1: 1.00, 2: 0.90, 3: 0.80},      # defensive
    "V7": {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00},      # boost
}

_AFTERTAX_KEYS = {"CAGR_IS", "CAGR_OOS", "CAGR_FULL", "Worst10Y_star", "P10_5Y", "Worst5Y"}


def _to_aftertax(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if k in _AFTERTAX_KEYS and v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[k] = v * TAX_FACTOR
        else:
            out[k] = v
    if not (np.isnan(out["CAGR_IS"]) or np.isnan(out["CAGR_OOS"])):
        out["IS_OOS_gap_pp"] = (out["CAGR_IS"] - out["CAGR_OOS"]) * 100.0
    return out


def _build_overlay_mult(date_index: pd.DatetimeIndex, mapping) -> np.ndarray:
    """Per-day overlay multiplier for V0/V7 (matches run_overlay's pipeline).
    mapping=None -> all-ones (DH-W1, P7 HOLD-leg)."""
    if mapping is None:
        return np.ones(len(date_index), dtype=float)
    macro_path = os.path.join(_REPO, "data", "macro_features.csv")
    macro = pd.read_csv(macro_path, parse_dates=["date"], index_col="date").sort_index()
    signal_raw = macro["nasdaq_mom63"].dropna()

    from signals.quantize import quantile_cut as _quantile_cut
    from signals.timing import apply_publication_lag as _apply_lag

    sig_q = _quantile_cut(signal_raw, levels=4).dropna().astype("int8")
    sig_lagged = _apply_lag(sig_q, "daily")
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    sig_aligned = sig_lagged.reindex(date_index).ffill()
    mult_arr = sig_aligned.map(
        lambda s: mapping.get(int(s), 1.0) if pd.notna(s) else 1.0
    ).fillna(1.0).values
    return np.clip(np.asarray(mult_arr, dtype=float), 0.0, 3.0)


def _nas_leg(L, r_nas, sofr_arr, nas_cost_model):
    """NASDAQ-leg daily return; the ONLY thing that differs CFD vs TQQQ."""
    if nas_cost_model == "CFD":
        borrow = (L - 1.0) * (sofr_arr + SBI_CFD_SPREAD / TRADING_DAYS)
        return L * r_nas - borrow                                   # CFD: no TER
    elif nas_cost_model == "TQQQ":
        borrow = np.maximum(L - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
        return L * r_nas - borrow - TER_TQQQ / TRADING_DAYS
    raise ValueError("nas_cost_model must be 'CFD' or 'TQQQ'")


def build_nav_overlay(
    close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
    lev_raw_masked, wn, wg, wb, mult, nas_cost_model,
):
    """DH-W1 / V0 / V7 baseline NAV (validated skeleton). Returns (nav, tpy).
    For DH-W1 mult is all-ones; for V0/V7 the overlay map. Reproduces
    run_dhw1 / run_overlay realistic to 0.0pp on the CFD branch.
    """
    from src.audit.strategy_runners import (
        _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
        _compute_dhw1_trades_per_year,
    )
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

    idx = dates.index
    n = len(close)
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3 = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    lev_mod = np.asarray(lev_raw_masked, float) * np.asarray(mult, float)
    L = pd.Series(lev_mod * 3.0, index=idx).shift(DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(DELAY).fillna(0).values
    sofr_arr = np.asarray(sofr_daily, float)

    nas_ret = _nas_leg(L, r_nas, sofr_arr, nas_cost_model)
    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    ter_drag = (np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
                + np.asarray(wb, float) * _TER_TMF_EXTRA_DAILY)
    tpy = _compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / 252.0
    r_sim = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_sim - ter_drag - etf_daily
    nav_adj = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )
    return nav_adj, tpy


def build_nav_p7(
    close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
    lev_raw_masked, wn, wg, wb, nas_cost_model,
):
    """P7 = DH-W1 (mult=1) realistic returns on HOLD days, GOLD75/BOND25 1x fund
    sleeve on OUT days (fund_active, T+5). Only the DH-W1 NASDAQ leg is swapped
    CFD<->TQQQ; the fund sleeve is identical between models.
    """
    sr._load_p7_shared()
    p7s = sr._P7_SHARED
    fund_active = p7s["fund_active"]
    r_blend = p7s["r_blend"]
    fee_daily = p7s["fee_daily"]

    # DH-W1 realistic NAV under the chosen NASDAQ cost model (mult=1)
    nav_dh, tpy = build_nav_overlay(
        close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
        lev_raw_masked, wn, wg, wb,
        np.ones(len(close), float), nas_cost_model,
    )
    r_dh = np.nan_to_num(nav_dh.pct_change().fillna(0).values, nan=0.0)

    # P7 composite: HOLD -> DH return, OUT+lag -> fund return - daily TER
    r_p7 = np.where(fund_active, r_blend - fee_daily, r_dh)
    nav_p7 = pd.Series(
        np.cumprod(1.0 + r_p7),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )
    return nav_p7, tpy


# ---------------------------------------------------------------------------
# Reference realistic outputs (for the per-strategy repro gate)
# ---------------------------------------------------------------------------
def _reference(strategy: str) -> dict:
    if strategy == "DH_W1":
        return sr.run_dhw1("realistic")
    if strategy == "V0":
        return sr.run_overlay("V0", "realistic")
    if strategy == "V7":
        return sr.run_overlay("V7", "realistic")
    if strategy == "P7":
        return sr.run_p7("realistic")
    raise ValueError(strategy)


def main():
    print("=" * 78)
    print("ETF-STRATEGY COST CORRECTION: CFD vs TQQQ-ETF (NASDAQ leg)  2026-06-11")
    print("DH-W1 / V0 / V7 / P7   (ETF-only environment)")
    print("=" * 78)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    close = a["close"]
    dates = a["dates"]
    sofr = np.asarray(a["sofr"], float)
    gold_2x = a["gold_2x"]
    bond_3x = a["bond_3x"]
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn = np.asarray(shared["wn"], float)
    wg = np.asarray(shared["wg"], float)
    wb = np.asarray(shared["wb"], float)
    date_index = pd.DatetimeIndex(pd.to_datetime(dates.values))

    strategies = ["DH_W1", "V0", "V7", "P7"]
    results = {}
    csv_rows = []

    for strat in strategies:
        mult = _build_overlay_mult(date_index, OVERLAY_MAPS.get(strat))
        if strat == "P7":
            nav_cfd, tpy_cfd = build_nav_p7(
                close, dates, gold_2x, bond_3x, sofr,
                lev_raw_masked, wn, wg, wb, "CFD")
            nav_tqqq, tpy_tqqq = build_nav_p7(
                close, dates, gold_2x, bond_3x, sofr,
                lev_raw_masked, wn, wg, wb, "TQQQ")
        else:
            nav_cfd, tpy_cfd = build_nav_overlay(
                close, dates, gold_2x, bond_3x, sofr,
                lev_raw_masked, wn, wg, wb, mult, "CFD")
            nav_tqqq, tpy_tqqq = build_nav_overlay(
                close, dates, gold_2x, bond_3x, sofr,
                lev_raw_masked, wn, wg, wb, mult, "TQQQ")

        pre_cfd = compute_10metrics(nav_cfd, tpy_cfd)
        pre_tqqq = compute_10metrics(nav_tqqq, tpy_tqqq)
        aft_cfd = _to_aftertax(pre_cfd)
        aft_tqqq = _to_aftertax(pre_tqqq)

        # Repro gate: CFD version vs run_* realistic
        ref = _reference(strat)
        err_is = (pre_cfd["CAGR_IS"] - ref["CAGR_IS"]) * 100.0
        err_oos = (pre_cfd["CAGR_OOS"] - ref["CAGR_OOS"]) * 100.0
        repro_err_pp = max(abs(err_is), abs(err_oos))
        gate = "PASS" if repro_err_pp <= 0.1 else "FAIL"

        min_at_cfd = min(aft_cfd["CAGR_IS"], aft_cfd["CAGR_OOS"])
        min_at_tqqq = min(aft_tqqq["CAGR_IS"], aft_tqqq["CAGR_OOS"])
        delta_min = (min_at_tqqq - min_at_cfd) * 100.0

        results[strat] = dict(
            pre_cfd=pre_cfd, pre_tqqq=pre_tqqq, aft_cfd=aft_cfd, aft_tqqq=aft_tqqq,
            min_at_cfd=min_at_cfd, min_at_tqqq=min_at_tqqq, delta_min=delta_min,
            repro_err_pp=repro_err_pp, gate=gate,
            err_is=err_is, err_oos=err_oos, ref=ref,
        )

        print()
        print("-" * 78)
        print("STRATEGY %s : repro gate (CFD vs run_* realistic)" % strat)
        print("-" * 78)
        print("            run_*           CFD_version     err_pp")
        print("CAGR_IS     %+.4f         %+.4f        %+.4f" % (ref["CAGR_IS"], pre_cfd["CAGR_IS"], err_is))
        print("CAGR_OOS    %+.4f         %+.4f        %+.4f" % (ref["CAGR_OOS"], pre_cfd["CAGR_OOS"], err_oos))
        print("max |err| = %.4f pp  ->  GATE %s" % (repro_err_pp, gate))

        for model, pre, aft in (("CFD", pre_cfd, aft_cfd), ("TQQQ", pre_tqqq, aft_tqqq)):
            for tax in ("pretax", "aftertax"):
                m = pre if tax == "pretax" else aft
                csv_rows.append({
                    "strategy": strat, "model": model, "tax": tax,
                    "CAGR_IS": m["CAGR_IS"], "CAGR_OOS": m["CAGR_OOS"],
                    "CAGR_FULL": m["CAGR_FULL"], "IS_OOS_gap_pp": m["IS_OOS_gap_pp"],
                    "Sharpe_OOS": m["Sharpe_OOS"], "MaxDD_FULL": m["MaxDD_FULL"],
                    "Worst10Y_star": m["Worst10Y_star"], "P10_5Y": m["P10_5Y"],
                    "Worst5Y": m["Worst5Y"], "Trades_yr": m["Trades_yr"],
                })

    out_csv = os.path.join(_REPO, "audit_results", "tqqq_correction_etf_strategies_20260611.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False, float_format="%.6f")
    print()
    print("Saved CSV: %s" % out_csv)

    # ---- ASCII summary table ----
    print()
    print("=" * 90)
    print("RESULT (after-tax CAGR; Sharpe/MaxDD pretax)")
    print("=" * 90)
    hdr = "%-7s | %-5s | %11s | %12s | %9s | %8s | %8s | %9s" % (
        "strat", "model", "CAGR_IS_at", "CAGR_OOS_at", "min_at",
        "Sharpe", "MaxDD", "dmin_pp")
    print(hdr)
    print("-" * len(hdr))
    for strat in strategies:
        r = results[strat]
        for model, aft, pre in (("CFD", r["aft_cfd"], r["pre_cfd"]),
                                 ("TQQQ", r["aft_tqqq"], r["pre_tqqq"])):
            min_at = min(aft["CAGR_IS"], aft["CAGR_OOS"])
            dmin = "" if model == "CFD" else "%+8.4f" % r["delta_min"]
            print("%-7s | %-5s | %+10.4f%% | %+11.4f%% | %+8.4f%% | %+7.3f | %+8.4f | %8s" % (
                strat, model, aft["CAGR_IS"] * 100, aft["CAGR_OOS"] * 100,
                min_at * 100, pre["Sharpe_OOS"], pre["MaxDD_FULL"], dmin))

    # ---- RETURN BLOCK (raw JSON-like) ----
    block = {}
    for strat in strategies:
        r = results[strat]
        ac, at = r["aft_cfd"], r["aft_tqqq"]
        pc, pt = r["pre_cfd"], r["pre_tqqq"]
        block[strat] = {
            "CFD": {
                "CAGR_IS_at": round(ac["CAGR_IS"], 4), "CAGR_OOS_at": round(ac["CAGR_OOS"], 4),
                "min_at": round(r["min_at_cfd"], 4), "Sharpe": round(pc["Sharpe_OOS"], 4),
                "MaxDD": round(pc["MaxDD_FULL"], 4),
                "W10Y_at": round(ac["Worst10Y_star"], 4) if ac["Worst10Y_star"] is not None and not np.isnan(ac["Worst10Y_star"]) else None,
                "P10_at": round(ac["P10_5Y"], 4) if ac["P10_5Y"] is not None and not np.isnan(ac["P10_5Y"]) else None,
                "Trades_yr": round(pc["Trades_yr"], 2),
            },
            "TQQQ": {
                "CAGR_IS_at": round(at["CAGR_IS"], 4), "CAGR_OOS_at": round(at["CAGR_OOS"], 4),
                "min_at": round(r["min_at_tqqq"], 4), "Sharpe": round(pt["Sharpe_OOS"], 4),
                "MaxDD": round(pt["MaxDD_FULL"], 4),
                "W10Y_at": round(at["Worst10Y_star"], 4) if at["Worst10Y_star"] is not None and not np.isnan(at["Worst10Y_star"]) else None,
                "P10_at": round(at["P10_5Y"], 4) if at["P10_5Y"] is not None and not np.isnan(at["P10_5Y"]) else None,
                "Trades_yr": round(pt["Trades_yr"], 2),
            },
            "delta_min_at_pp": round(r["delta_min"], 4),
            "repro_err_pp": round(r["repro_err_pp"], 4),
            "gate": r["gate"],
        }
    print()
    print("=" * 78)
    print("RETURN_BLOCK")
    print("=" * 78)
    print(json.dumps(block, indent=2))
    return block


if __name__ == "__main__":
    main()
