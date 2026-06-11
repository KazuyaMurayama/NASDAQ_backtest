"""
src/audit/lu_cfd_recost_20260611.py
===================================
Re-cost the lever-up variants LU1 (stronger V7 map) and LU2 (uniform IN NASDAQ
leverage x1.15) charging the >3x leverage EXCESS at the CFD rate (SOFR+3.0%),
instead of the cheap TQQQ swap rate (SOFR+0.5%) used by the existing
run_p09_tqqq_validate script.

Precise correction:
  - The TQQQ-cost NASDAQ leg already charges max(L-1,0)*(sofr + 0.005/252) + TER.
  - Up to 3x is legitimately TQQQ-priced. The (L-3)+ excess must instead be held
    via CFD/margin at SOFR+3.0%, i.e. an EXTRA (0.030 - 0.005) = 2.5%/yr on that
    excess notional.
  - So subtract from the daily portfolio return:
        wn_s * max(L - 3.0, 0.0) * (0.030 - 0.005) / 252
    This leaves <=3x at the TQQQ cost and charges >3x excess the CFD rate.

This penalty is injected into the IN-period NASDAQ leg BEFORE the P09 OUT-fill,
so OUT periods (where L is masked ~1, penalty ~0, and the day is replaced by the
1x Gold/Bond blend anyway) are unaffected -- exactly as intended.

Builds & metrics (full standard 10, pretax + after-tax) for:
  P09_TQQQ (reference), LU1_tqqq, LU1_cfd, LU2_tqqq, LU2_cfd.
WFA (canonical g1_wfa windows) for LU1_cfd and LU2_cfd.

Task 2: consolidate after-tax standard-10 metrics for all comparison rows from
the existing CSVs into standard10_consolidated_20260611.csv.

ASCII-only prints (Windows cp932). Saves into repo; does NOT commit.
"""

from __future__ import annotations

import os
import sys
import types
import json

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

from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax, _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, _build_p09_nav,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    _build_v7_mult, TER_TQQQ, SWAP_SPREAD,
    DELAY as V7_DELAY, DH_PER_UNIT, NAV_FLOOR,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom, _run_wfa, LU1_MAP, LU2_SCALE,
)

AFTER_TAX = 0.8273

# CFD-excess penalty: extra rate on the (L-3)+ notional.
# CFD margin = SOFR + 3.0% ; TQQQ swap (already charged) = SOFR + 0.5%.
CFD_RATE = 0.030
TQQQ_SWAP = SWAP_SPREAD            # 0.0050
EXCESS_EXTRA = CFD_RATE - TQQQ_SWAP  # 0.025 /yr on (L-3)+ * wn_s
LEV_CAP = 3.0


# ---------------------------------------------------------------------------
# Replicate cost_model.build_nav_v7 (TQQQ) with an OPTIONAL CFD-excess penalty
# on the >3x NASDAQ leverage. When cfd_excess=False this is byte-for-byte the
# TQQQ path; when True it additionally subtracts
#   wn_s * max(L-3,0) * EXCESS_EXTRA / 252
# from the daily portfolio return.
# ---------------------------------------------------------------------------
def _build_nav_v7_tqqq(close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
                       lev_raw_masked, wn, wg, wb, mult_v7, cfd_excess=False):
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
    L = pd.Series(lev_mod * 3.0, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(V7_DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(V7_DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(V7_DELAY).fillna(0).values
    sofr_arr = np.asarray(sofr_daily, float)

    # TQQQ leg
    borrow = np.maximum(L - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # CFD-excess penalty on the >3x portion of the NASDAQ sleeve.
    excess_days = 0
    if cfd_excess:
        excess_lev = np.maximum(L - LEV_CAP, 0.0)
        penalty = wn_s * excess_lev * EXCESS_EXTRA / TRADING_DAYS
        daily = daily - penalty
        excess_days = int(np.sum(excess_lev > 1e-9))

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
    return nav_adj, tpy, excess_days


def _build_tqqq_base(shared, date_index, v7_map=None, lev_scale=1.0, cfd_excess=False):
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

    if v7_map is None:
        mult_v7 = _build_v7_mult(date_index)
    else:
        mult_v7 = _build_v7_mult_custom(date_index, v7_map)
    mult_v7 = mult_v7 * float(lev_scale)

    nav_dt, tpy, excess_days = _build_nav_v7_tqqq(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7, cfd_excess=cfd_excess)
    r_base = nav_dt.pct_change().fillna(0).values
    return nav_dt, r_base, tpy, excess_days


def _build_p09_on_base(r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
                       dates_dt, tpy_base, n_years):
    nav_arr, r_p09, eff_active = _build_p09_nav(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on)
    nav_dt = pd.Series(nav_arr, index=dates_dt)
    flips = _count_fund_transitions(eff_active)
    tpy = tpy_base + flips / n_years
    return nav_dt, r_p09, tpy


def _metrics_pack(nav_dt, tpy):
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    cy = _calendar_year_returns(nav_dt)
    worst_cy = float(cy.min())
    worst_cy_year = int(cy.idxmin())
    return pre, aft, worst_cy, worst_cy_year


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


# ---------------------------------------------------------------------------
# Task 2: consolidate standard-10 after-tax metrics from existing CSVs.
# ---------------------------------------------------------------------------
AFTER_TAX_KEYS = {"CAGR_IS", "CAGR_OOS", "CAGR_FULL", "Worst10Y_star", "P10_5Y", "Worst5Y"}


def _row_to_aftertax(row):
    """row is a pandas Series (one CSV line). Returns a normalised dict of the
    standard-10 after-tax values. If the CSV row is already 'aftertax', use as-is;
    if it's 'pretax', multiply the CAGR/Worst10Y/P10/Worst5Y family by AFTER_TAX."""
    tax = str(row.get("tax", "")).strip().lower()
    out = {}

    def g(k):
        v = row.get(k, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    for k in ("CAGR_IS", "CAGR_OOS", "CAGR_FULL", "Worst10Y_star", "P10_5Y",
              "Worst5Y", "IS_OOS_gap_pp", "Sharpe_OOS", "MaxDD_FULL", "Trades_yr"):
        out[k] = g(k)

    if tax == "pretax":
        for k in AFTER_TAX_KEYS:
            if np.isfinite(out.get(k, np.nan)):
                out[k] = out[k] * AFTER_TAX
        # recompute gap after tax
        if np.isfinite(out["CAGR_IS"]) and np.isfinite(out["CAGR_OOS"]):
            out["IS_OOS_gap_pp"] = (out["CAGR_IS"] - out["CAGR_OOS"]) * 100.0
    # min(IS,OOS)
    if np.isfinite(out["CAGR_IS"]) and np.isfinite(out["CAGR_OOS"]):
        out["min_IS_OOS"] = min(out["CAGR_IS"], out["CAGR_OOS"])
    else:
        out["min_IS_OOS"] = np.nan

    # WFE / CI95 passthrough (any of several column names)
    wfe = np.nan
    for k in ("wfa_WFE", "WFE", "WFA_WFE"):
        v = row.get(k, None)
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip() != "":
            try:
                wfe = float(v); break
            except (TypeError, ValueError):
                pass
    ci = np.nan
    for k in ("wfa_CI95_lo", "CI95_lo", "WFA_CI95_lo"):
        v = row.get(k, None)
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip() != "":
            try:
                ci = float(v); break
            except (TypeError, ValueError):
                pass
    out["WFE"] = wfe
    out["CI95_lo"] = ci
    return out


def _pick_aftertax_row(df, cond_col, cond_val, extra_filter=None):
    """From a CSV df, pick the row for cond_val preferring tax=='aftertax'.
    extra_filter: optional dict {col: value} to further narrow (e.g. model)."""
    sub = df[df[cond_col].astype(str).str.strip() == cond_val]
    if extra_filter:
        for c, v in extra_filter.items():
            if c in sub.columns:
                sub = sub[sub[c].astype(str).str.strip() == v]
    if len(sub) == 0:
        return None
    if "tax" in df.columns:
        aft = sub[sub["tax"].astype(str).str.strip().str.lower() == "aftertax"]
        if len(aft) > 0:
            return aft.iloc[0]
    return sub.iloc[0]


def _load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _consolidate(out_csv_path, extra_rows):
    """Build the consolidated standard-10 after-tax table from existing CSVs.
    `extra_rows` = list of dicts already-after-tax (our LU recost rows) to append."""
    ar = os.path.join(_REPO_DIR, "audit_results")
    # Explicit specs: (file, cond_col, csv_condition, display_label, extra_filter)
    # csv_condition uses the REAL value present in the file; display_label is the
    # human label used in the report; extra_filter narrows multi-key files.
    specs = [
        # p03: C0..C3b
        ("p03_backtest_metrics_20260611.csv", "condition", "C0", "C0", None),
        ("p03_backtest_metrics_20260611.csv", "condition", "C1", "C1 (p03)", None),
        ("p03_backtest_metrics_20260611.csv", "condition", "C2", "C2", None),
        ("p03_backtest_metrics_20260611.csv", "condition", "C3", "C3", None),
        ("p03_backtest_metrics_20260611.csv", "condition", "C3b", "C3b", None),
        # p01
        ("p01_backtest_metrics_20260611.csv", "condition", "C1_baseline_V7", "C1 (p01)", None),
        ("p01_backtest_metrics_20260611.csv", "condition", "P13_fixed_50_50", "P13", None),
        ("p01_backtest_metrics_20260611.csv", "condition", "P01_w63", "P01_w63", None),
        ("p01_backtest_metrics_20260611.csv", "condition", "P01_w21", "P01_w21", None),
        ("p01_backtest_metrics_20260611.csv", "condition", "P01_w126", "P01_w126", None),
        # p07
        ("p07_backtest_metrics_20260611.csv", "condition", "C1", "C1 (p07)", None),
        ("p07_backtest_metrics_20260611.csv", "condition", "P07_st20", "P07_st20", None),
        ("p07_backtest_metrics_20260611.csv", "condition", "P07_st15", "P07_st15", None),
        ("p07_backtest_metrics_20260611.csv", "condition", "P07_st25", "P07_st25", None),
        # p02_p09
        ("p02_p09_backtest_metrics_20260611.csv", "condition", "C1_baseline_V7", "C1 (p02)", None),
        ("p02_p09_backtest_metrics_20260611.csv", "condition", "P02a_AND_cash", "P02a", None),
        ("p02_p09_backtest_metrics_20260611.csv", "condition", "P02b_OR_cash", "P02b", None),
        ("p02_p09_backtest_metrics_20260611.csv", "condition", "P02c_AND_half", "P02c", None),
        ("p02_p09_backtest_metrics_20260611.csv", "condition", "P09_bondtiming", "P09", None),
        # p09_tqqq_validate (LU1 = LU1_strongmap, charged at TQQQ-cost = pre-recost)
        ("p09_tqqq_validate_20260611.csv", "condition", "P09_TQQQ", "P09_TQQQ (validate)", None),
        ("p09_tqqq_validate_20260611.csv", "condition", "LU1_strongmap", "LU1 (TQQQ-cost, validate)", None),
        ("p09_tqqq_validate_20260611.csv", "condition", "LU2_lever1p15", "LU2 (TQQQ-cost, validate)", None),
        # cost_model V7 CFD & TQQQ baselines (model col is the condition)
        ("cost_model_cfd_vs_tqqq_20260611.csv", "model", "CFD", "V7 baseline CFD (cost_model)", None),
        ("cost_model_cfd_vs_tqqq_20260611.csv", "model", "TQQQ", "V7 baseline TQQQ (cost_model)", None),
        # tqqq_correction: V7 strategy, CFD & TQQQ
        ("tqqq_correction_etf_strategies_20260611.csv", "strategy", "V7", "V7 CFD (tqqq_correction)", {"model": "CFD"}),
        ("tqqq_correction_etf_strategies_20260611.csv", "strategy", "V7", "V7 TQQQ (tqqq_correction)", {"model": "TQQQ"}),
    ]

    rows = []
    seen_notes = []
    for fname, cond_col, csv_cond, disp, extra_filter in specs:
        path = os.path.join(ar, fname)
        df = _load_csv(path)
        if df is None:
            seen_notes.append("MISSING: %s" % fname)
            continue
        if cond_col not in df.columns:
            seen_notes.append("NO_COND_COL %s in %s (cols=%s)" % (cond_col, fname, list(df.columns)))
            continue
        r = _pick_aftertax_row(df, cond_col, csv_cond, extra_filter=extra_filter)
        if r is None:
            seen_notes.append("ROW_NOT_FOUND %s=%s in %s" % (cond_col, csv_cond, fname))
            continue
        m = _row_to_aftertax(r)
        rows.append({
                "source_csv": fname,
                "condition": disp,
                "CAGR_IS_at": m["CAGR_IS"],
                "CAGR_OOS_at": m["CAGR_OOS"],
                "min_IS_OOS_at": m["min_IS_OOS"],
                "IS_OOS_gap_pp": m["IS_OOS_gap_pp"],
                "Sharpe_OOS": m["Sharpe_OOS"],
                "MaxDD_FULL": m["MaxDD_FULL"],
                "Worst10Y_star_at": m["Worst10Y_star"],
                "P10_5Y_at": m["P10_5Y"],
                "Worst5Y_at": m["Worst5Y"],
                "Trades_yr": m["Trades_yr"],
                "WFE": m["WFE"] if np.isfinite(m["WFE"]) else "N/A",
                "CI95_lo": m["CI95_lo"] if np.isfinite(m["CI95_lo"]) else "N/A",
            })

    # append our freshly-computed LU recost rows (already after-tax dicts)
    for er in extra_rows:
        rows.append(er)

    cons = pd.DataFrame(rows)
    cons.to_csv(out_csv_path, index=False, float_format="%.6f")
    return cons, seen_notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("LU CFD-EXCESS RE-COST : charge (L-3)+ excess at SOFR+3.0%  2026-06-11")
    print("=" * 80)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / TRADING_DAYS

    # Gold/Bond 1x legs + OUT mask + fund lag (same as P09)
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), dtype=float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0),
        dtype=float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]

    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    def _make(v7_map=None, lev_scale=1.0, cfd_excess=False):
        base_nav, r_base, tpy_b, exc = _build_tqqq_base(
            shared, dates_dt, v7_map=v7_map, lev_scale=lev_scale, cfd_excess=cfd_excess)
        nav, r, tpy = _build_p09_on_base(
            r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
            dates_dt, tpy_b, n_years)
        pre, aft, wcy, wcyy = _metrics_pack(nav, tpy)
        return {"nav": nav, "pre": pre, "aft": aft, "wcy": wcy, "wcyy": wcyy,
                "tpy": tpy, "excess_days": exc}

    # ---- Build the five variants ----
    print("\nBuilding variants ...")
    P09 = _make(v7_map=None, lev_scale=1.0, cfd_excess=False)        # reference
    LU1_tqqq = _make(v7_map=LU1_MAP, lev_scale=1.0, cfd_excess=False)
    LU1_cfd  = _make(v7_map=LU1_MAP, lev_scale=1.0, cfd_excess=True)
    LU2_tqqq = _make(v7_map=None, lev_scale=LU2_SCALE, cfd_excess=False)
    LU2_cfd  = _make(v7_map=None, lev_scale=LU2_SCALE, cfd_excess=True)

    print("  Days with L>3x (excess subject to CFD rate):")
    print("    P09_TQQQ=%d  LU1=%d  LU2=%d  (of %d days)"
          % (P09["excess_days"], LU1_cfd["excess_days"], LU2_cfd["excess_days"], n))

    variants = {
        "P09_TQQQ": P09,
        "LU1_tqqq": LU1_tqqq,
        "LU1_cfd": LU1_cfd,
        "LU2_tqqq": LU2_tqqq,
        "LU2_cfd": LU2_cfd,
    }
    order = ["P09_TQQQ", "LU1_tqqq", "LU1_cfd", "LU2_tqqq", "LU2_cfd"]

    # ---- WFA on the CFD-recosted variants (+ reference for sanity) ----
    print("\nWFA (canonical g1_wfa windows) on LU1_cfd, LU2_cfd, P09_TQQQ ...")
    wfa = {}
    wfa["P09_TQQQ"] = _run_wfa(P09["nav"], "P09_TQQQ")
    wfa["LU1_cfd"] = _run_wfa(LU1_cfd["nav"], "LU1_cfd")
    wfa["LU2_cfd"] = _run_wfa(LU2_cfd["nav"], "LU2_cfd")
    for k in ("P09_TQQQ", "LU1_cfd", "LU2_cfd"):
        w = wfa[k]
        print("  %-9s WFE=%.4f  CI95_lo=%+.4f%%  n=%d  t_p=%.4f"
              % (k, w["WFE"], 100 * w["CI95_lo"], w["n_windows"], w["t_p"]))

    # ---- CSV rows for lu_cfd_recost (variant x {pretax,aftertax} x 10 metrics + WFA) ----
    rows = []
    for label in order:
        v = variants[label]
        w = wfa.get(label, {"WFE": "", "CI95_lo": "", "n_windows": "", "t_p": ""})
        for tax_label, m in (("pretax", v["pre"]), ("aftertax", v["aft"])):
            rows.append({
                "condition": label,
                "tax": tax_label,
                "CAGR_IS": m["CAGR_IS"],
                "CAGR_OOS": m["CAGR_OOS"],
                "CAGR_FULL": m["CAGR_FULL"],
                "min_IS_OOS": min(m["CAGR_IS"], m["CAGR_OOS"]),
                "IS_OOS_gap_pp": m["IS_OOS_gap_pp"],
                "Sharpe_OOS": v["pre"]["Sharpe_OOS"],
                "MaxDD_FULL": v["pre"]["MaxDD_FULL"],
                "Worst10Y_star": m["Worst10Y_star"],
                "P10_5Y": m["P10_5Y"],
                "Worst5Y": m["Worst5Y"],
                "Trades_yr": m["Trades_yr"],
                "worst_calendar_year_return": v["wcy"],
                "worst_calendar_year": v["wcyy"],
                "excess_days_L_gt_3": v["excess_days"],
                "wfa_WFE": w["WFE"] if w["WFE"] != "" else "",
                "wfa_CI95_lo": w["CI95_lo"] if w["CI95_lo"] != "" else "",
                "wfa_n_windows": w["n_windows"] if w["n_windows"] != "" else "",
                "wfa_t_p": w["t_p"] if w["t_p"] != "" else "",
            })
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "lu_cfd_recost_20260611.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("\nSaved CSV: %s" % csv_path)

    # ---- Results table ----
    print("\n" + "=" * 118)
    print("RESULTS (CAGR_IS_at/CAGR_OOS_at/min_at AFTER-TAX; Sharpe/MaxDD pretax)")
    print("=" * 118)
    hdr = ("%-10s | %10s | %11s | %9s | %7s | %8s | %10s | %9s | %8s | %7s | %8s"
           % ("condition", "CAGR_IS_at", "CAGR_OOS_at", "min_at", "Sharpe",
              "MaxDD", "Worst10Y*", "P10_5Y", "gap_pp", "Trd/yr", "worstCY"))
    print(hdr)
    print("-" * 118)
    for label in order:
        v = variants[label]
        aft = v["aft"]; pre = v["pre"]
        mn = _min_at(aft)
        print("%-10s | %+9.2f%% | %+10.2f%% | %+8.2f%% | %7.3f | %+7.2f%% | %+9.2f%% | %+8.2f%% | %+7.2f | %7.1f | %+7.2f%%"
              % (label, 100 * aft["CAGR_IS"], 100 * aft["CAGR_OOS"], 100 * mn,
                 pre["Sharpe_OOS"], 100 * pre["MaxDD_FULL"],
                 100 * aft["Worst10Y_star"], 100 * aft["P10_5Y"],
                 aft["IS_OOS_gap_pp"], aft["Trades_yr"], 100 * v["wcy"]))
    print("=" * 118)

    # ---- CFD haircut (tqqq -> cfd) on after-tax min ----
    hc_lu1 = (_min_at(LU1_cfd["aft"]) - _min_at(LU1_tqqq["aft"])) * 100.0
    hc_lu2 = (_min_at(LU2_cfd["aft"]) - _min_at(LU2_tqqq["aft"])) * 100.0
    print("\nCFD re-cost haircut on after-tax min CAGR:")
    print("  LU1: tqqq min_at=%+.2f%% -> cfd min_at=%+.2f%%  haircut=%+.2f pp"
          % (100 * _min_at(LU1_tqqq["aft"]), 100 * _min_at(LU1_cfd["aft"]), hc_lu1))
    print("  LU2: tqqq min_at=%+.2f%% -> cfd min_at=%+.2f%%  haircut=%+.2f pp"
          % (100 * _min_at(LU2_tqqq["aft"]), 100 * _min_at(LU2_cfd["aft"]), hc_lu2))

    # ---- Task 2: consolidated standard-10 ----
    print("\n" + "=" * 80)
    print("TASK 2: consolidate standard-10 after-tax across comparison CSVs")
    print("=" * 80)
    # Append our LU recost rows (after-tax) into the consolidated table too.
    extra = []
    for label in order:
        v = variants[label]
        aft = v["aft"]; pre = v["pre"]
        w = wfa.get(label, None)
        extra.append({
            "source_csv": "lu_cfd_recost_20260611.csv",
            "condition": label,
            "CAGR_IS_at": aft["CAGR_IS"],
            "CAGR_OOS_at": aft["CAGR_OOS"],
            "min_IS_OOS_at": _min_at(aft),
            "IS_OOS_gap_pp": aft["IS_OOS_gap_pp"],
            "Sharpe_OOS": pre["Sharpe_OOS"],
            "MaxDD_FULL": pre["MaxDD_FULL"],
            "Worst10Y_star_at": aft["Worst10Y_star"],
            "P10_5Y_at": aft["P10_5Y"],
            "Worst5Y_at": aft["Worst5Y"],
            "Trades_yr": aft["Trades_yr"],
            "WFE": (w["WFE"] if (w and np.isfinite(w["WFE"])) else "N/A"),
            "CI95_lo": (w["CI95_lo"] if (w and np.isfinite(w["CI95_lo"])) else "N/A"),
        })

    cons_path = os.path.join(out_dir, "standard10_consolidated_20260611.csv")
    cons, notes = _consolidate(cons_path, extra)
    print("Saved CSV: %s  (%d rows)" % (cons_path, len(cons)))
    for nte in notes:
        print("  NOTE: %s" % nte)

    print("\nConsolidated (condition -> Worst10Y*_at / P10_5Y_at / WFE / CI95_lo):")
    print("-" * 100)
    for _, r in cons.iterrows():
        w10 = r["Worst10Y_star_at"]
        p10 = r["P10_5Y_at"]
        wfe = r["WFE"]
        ci = r["CI95_lo"]
        w10s = ("%+.2f%%" % (100 * w10)) if isinstance(w10, (int, float)) and np.isfinite(w10) else "  N/A "
        p10s = ("%+.2f%%" % (100 * p10)) if isinstance(p10, (int, float)) and np.isfinite(p10) else "  N/A "
        wfes = ("%.3f" % wfe) if isinstance(wfe, (int, float)) and np.isfinite(wfe) else "N/A"
        try:
            cif = float(ci)
            cis = "%+.2f%%" % (100 * cif)
        except (TypeError, ValueError):
            cis = "N/A"
        print("  %-32s | W10*=%8s | P10_5Y=%8s | WFE=%5s | CI95_lo=%8s | (%s)"
              % (str(r["condition"])[:32], w10s, p10s, wfes, cis, r["source_csv"]))

    # ---- RETURN BLOCK ----
    def _block(v, w):
        aft = v["aft"]; pre = v["pre"]
        return {
            "CAGR_IS_at": round(aft["CAGR_IS"], 6),
            "CAGR_OOS_at": round(aft["CAGR_OOS"], 6),
            "min_at": round(_min_at(aft), 6),
            "gap_pp": round(aft["IS_OOS_gap_pp"], 4),
            "Sharpe": round(pre["Sharpe_OOS"], 4),
            "MaxDD": round(pre["MaxDD_FULL"], 6),
            "Worst10Y_star_at": round(aft["Worst10Y_star"], 6),
            "P10_5Y_at": round(aft["P10_5Y"], 6),
            "Trades_yr": round(aft["Trades_yr"], 2),
            "WFE": (round(float(w["WFE"]), 4) if w and np.isfinite(w["WFE"]) else None),
            "CI95_lo": (round(float(w["CI95_lo"]), 6) if w and np.isfinite(w["CI95_lo"]) else None),
        }

    block = {
        "P09_TQQQ": _block(P09, wfa["P09_TQQQ"]),
        "LU1_tqqq_min_at": round(_min_at(LU1_tqqq["aft"]), 6),
        "LU1_cfd": _block(LU1_cfd, wfa["LU1_cfd"]),
        "LU1_cfd_haircut_pp": round(hc_lu1, 4),
        "LU2_tqqq_min_at": round(_min_at(LU2_tqqq["aft"]), 6),
        "LU2_cfd": _block(LU2_cfd, wfa["LU2_cfd"]),
        "LU2_cfd_haircut_pp": round(hc_lu2, 4),
    }
    print("\n" + "=" * 80)
    print("RETURN_BLOCK")
    print("=" * 80)
    print(json.dumps(block, indent=2))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
