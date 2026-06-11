"""
src/audit/cost_model_cfd_vs_tqqq_20260611.py
=============================================
Quantify the V7 baseline (DH-W1 + nasdaq_mom63 V7 boost overlay) standard 10
metrics under BOTH NASDAQ-leg cost models, holding everything else identical:

  CFD version  (current canonical) :
      nas_ret = L*r_nas - (L-1)*(sofr + SBI_CFD_SPREAD/252)            ; NO TER
      SBI_CFD_SPREAD = 0.0300  (3.0%/yr)
      == run_overlay('V7','realistic') exactly (g18 -> g14 _make_nav ->
         build_nav_strategy nas_mode='CFD', cfd_spread=SBI_CFD_SPREAD)

  TQQQ-ETF version (designed product) :
      nas_ret = L*r_nas - max(L-1,0)*(sofr + SWAP_SPREAD/252) - TER_TQQQ/252
      SWAP_SPREAD = 0.0050 (0.50%/yr) ; TER_TQQQ = 0.0086 (0.86%/yr)
      (matches build_nav_strategy nas_mode='TQQQ' intent at L=3:
         3r - 2(sofr+swap/252) - 0.86%/252, generalized to dynamic L.)

EVERYTHING ELSE identical between the two: weights (wn/wg/wb), per-day NASDAQ
leverage L = lev_raw_masked * mult_v7 * 3.0 (DELAY=2 shifted), V7 multiplier,
Gold-2x leg, Bond-3x leg, DH turnover cost (DH_PER_UNIT), incremental ETF TER
drag on gold/bond legs, US-ETF $22-cap trade cost.

The skeleton is `build_nav_baseline_mimic` from run_p03_backtest_20260611.py,
which reproduces run_overlay('V7','realistic') to 0.0000pp. We swap ONLY the
NASDAQ-leg financing/TER formula (CFD -> TQQQ).

After-tax: CAGR / Worst10Y / P10 / Worst5Y * 0.8273 ; Sharpe & MaxDD pretax.
Validation gate: CFD version must reproduce run_overlay('V7','realistic') within
0.1pp.

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

import numpy as np
import pandas as pd

import src.audit.strategy_runners as sr
from src.audit.unified_metrics import compute_10metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS = 252.0
DELAY = 2
TAX_FACTOR = 0.8273

# CFD leg (current canonical baseline)
SBI_CFD_SPREAD = 0.0300       # g14_wfa_sbi_cfd.SBI_CFD_SPREAD

# TQQQ-ETF leg (designed product)
TER_TQQQ = 0.0086             # product_costs.TQQQ.ter / cfd_leverage_backtest.ANNUAL_COST
SWAP_SPREAD = 0.0050          # product_costs.TQQQ.swap_spread / corrected_strategy.SWAP_SPREAD

# Shared baseline cost layers (identical for both models)
DH_PER_UNIT = 0.0010          # build_dh_nav_with_cost moderate per-unit turnover
NAV_FLOOR = -0.999

V7_MAP = {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00}

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


def _build_v7_mult(date_index: pd.DatetimeIndex) -> np.ndarray:
    """V7 multiplier per day (matches run_overlay's pipeline exactly)."""
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
        lambda s: V7_MAP.get(int(s), 1.0) if pd.notna(s) else 1.0
    ).fillna(1.0).values
    return np.clip(np.asarray(mult_arr, dtype=float), 0.0, 3.0)


def build_nav_v7(
    close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
    lev_raw_masked, wn, wg, wb, mult_v7,
    nas_cost_model: str,
):
    """V7 baseline NAV. `nas_cost_model` in {'CFD','TQQQ'} swaps ONLY the
    NASDAQ-leg financing/TER formula; all other layers identical.

    NASDAQ leg leverage L = lev_raw_masked * mult_v7 * 3.0 (DELAY-shifted).
      CFD : nas_ret = L*r_nas - (L-1)*(sofr + SBI_CFD_SPREAD/252)        ; no TER
      TQQQ: nas_ret = L*r_nas - max(L-1,0)*(sofr + SWAP_SPREAD/252) - TER_TQQQ/252
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

    lev_mod = np.asarray(lev_raw_masked, float) * np.asarray(mult_v7, float)
    L = pd.Series(lev_mod * 3.0, index=idx).shift(DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(DELAY).fillna(0).values
    sofr_arr = np.asarray(sofr_daily, float)

    if nas_cost_model == "CFD":
        borrow = (L - 1.0) * (sofr_arr + SBI_CFD_SPREAD / TRADING_DAYS)
        nas_ret = L * r_nas - borrow  # CFD: no TER
    elif nas_cost_model == "TQQQ":
        borrow = np.maximum(L - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
        nas_ret = L * r_nas - borrow - TER_TQQQ / TRADING_DAYS
    else:
        raise ValueError("nas_cost_model must be 'CFD' or 'TQQQ'")

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # DH turnover on RAW (unshifted) weight changes -- identical for both models
    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    # incremental ETF TER drag on GOLD/BOND legs (identical for both models),
    # plus US-ETF trade cost. NASDAQ TER drag = 0 in both (CFD has no TER; TQQQ
    # TER is already in the leg formula above).
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


def main():
    print("=" * 78)
    print("V7 BASELINE COST MODEL: CFD vs TQQQ-ETF  (NASDAQ leg)  2026-06-11")
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
    mult_v7 = _build_v7_mult(date_index)

    # ---- Build both NAVs ----
    nav_cfd, tpy_cfd = build_nav_v7(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7, "CFD")
    nav_tqqq, tpy_tqqq = build_nav_v7(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7, "TQQQ")

    pre_cfd = compute_10metrics(nav_cfd, tpy_cfd)
    pre_tqqq = compute_10metrics(nav_tqqq, tpy_tqqq)
    aft_cfd = _to_aftertax(pre_cfd)
    aft_tqqq = _to_aftertax(pre_tqqq)

    # ---- Validation gate: CFD version vs run_overlay('V7','realistic') ----
    base = sr.run_overlay("V7", "realistic")
    err_is = (pre_cfd["CAGR_IS"] - base["CAGR_IS"]) * 100.0
    err_oos = (pre_cfd["CAGR_OOS"] - base["CAGR_OOS"]) * 100.0
    repro_err_pp = max(abs(err_is), abs(err_oos))
    gate = "PASS" if repro_err_pp <= 0.1 else "FAIL"

    print()
    print("-" * 78)
    print("VALIDATION GATE: CFD version vs run_overlay('V7','realistic')")
    print("-" * 78)
    print("            run_overlay      CFD_version     err_pp")
    print("CAGR_IS     %+.4f         %+.4f        %+.4f" % (base["CAGR_IS"], pre_cfd["CAGR_IS"], err_is))
    print("CAGR_OOS    %+.4f         %+.4f        %+.4f" % (base["CAGR_OOS"], pre_cfd["CAGR_OOS"], err_oos))
    print("max |err| = %.4f pp  ->  GATE %s (threshold 0.1pp)" % (repro_err_pp, gate))

    # ---- CSV: 2 models x {pretax, aftertax} = 4 rows ----
    rows = []
    for model, pre, aft in (("CFD", pre_cfd, aft_cfd), ("TQQQ", pre_tqqq, aft_tqqq)):
        for tax in ("pretax", "aftertax"):
            m = pre if tax == "pretax" else aft
            rows.append({
                "model": model,
                "tax": tax,
                "CAGR_IS": m["CAGR_IS"],
                "CAGR_OOS": m["CAGR_OOS"],
                "CAGR_FULL": m["CAGR_FULL"],
                "IS_OOS_gap_pp": m["IS_OOS_gap_pp"],
                "Sharpe_OOS": m["Sharpe_OOS"],
                "MaxDD_FULL": m["MaxDD_FULL"],
                "Worst10Y_star": m["Worst10Y_star"],
                "P10_5Y": m["P10_5Y"],
                "Worst5Y": m["Worst5Y"],
                "Trades_yr": m["Trades_yr"],
            })
    out_csv = os.path.join(_REPO, "audit_results", "cost_model_cfd_vs_tqqq_20260611.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
    print()
    print("Saved CSV: %s" % out_csv)

    # ---- ASCII table: model | CAGR_IS_at | CAGR_OOS_at | min_at | Sharpe | MaxDD ----
    print()
    print("=" * 78)
    print("RESULT (after-tax CAGR; Sharpe/MaxDD pretax)")
    print("=" * 78)
    hdr = "%-6s | %11s | %12s | %10s | %8s | %9s" % (
        "model", "CAGR_IS_at", "CAGR_OOS_at", "min_at", "Sharpe", "MaxDD")
    print(hdr)
    print("-" * len(hdr))
    for model, aft, pre in (("CFD", aft_cfd, pre_cfd), ("TQQQ", aft_tqqq, pre_tqqq)):
        min_at = min(aft["CAGR_IS"], aft["CAGR_OOS"])
        print("%-6s | %+10.4f%% | %+11.4f%% | %+9.4f%% | %+7.3f | %+8.4f" % (
            model, aft["CAGR_IS"] * 100, aft["CAGR_OOS"] * 100, min_at * 100,
            pre["Sharpe_OOS"], pre["MaxDD_FULL"]))

    # ---- RETURN BLOCK ----
    min_at_cfd = min(aft_cfd["CAGR_IS"], aft_cfd["CAGR_OOS"])
    min_at_tqqq = min(aft_tqqq["CAGR_IS"], aft_tqqq["CAGR_OOS"])
    delta_min_at_pp = (min_at_tqqq - min_at_cfd) * 100.0

    block = {
        "CFD_version": {
            "CAGR_IS_at": round(aft_cfd["CAGR_IS"], 4),
            "CAGR_OOS_at": round(aft_cfd["CAGR_OOS"], 4),
            "min_at": round(min_at_cfd, 4),
            "Sharpe": round(pre_cfd["Sharpe_OOS"], 4),
            "MaxDD": round(pre_cfd["MaxDD_FULL"], 4),
        },
        "TQQQ_version": {
            "CAGR_IS_at": round(aft_tqqq["CAGR_IS"], 4),
            "CAGR_OOS_at": round(aft_tqqq["CAGR_OOS"], 4),
            "min_at": round(min_at_tqqq, 4),
            "Sharpe": round(pre_tqqq["Sharpe_OOS"], 4),
            "MaxDD": round(pre_tqqq["MaxDD_FULL"], 4),
        },
        "delta_min_at_pp": round(delta_min_at_pp, 4),
        "repro_err_vs_runoverlay_pp": round(repro_err_pp, 4),
    }
    print()
    print("=" * 78)
    print("RETURN_BLOCK")
    print("=" * 78)
    print(json.dumps(block, indent=2))
    return block


if __name__ == "__main__":
    main()
