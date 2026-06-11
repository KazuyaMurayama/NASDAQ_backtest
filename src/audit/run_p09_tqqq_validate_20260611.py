"""
src/audit/run_p09_tqqq_validate_20260611.py
============================================
Combine the two un-combined wins on DH-W1 + V7:
  (A) P09 bond-timing conditional OUT-fill (Gold-always + Bond-when-bond_mom252>0,
      inverse-vol W63, T+5 fund lag, fund TER) -- best lead so far.
  (B) CFD -> TQQQ cost correction on the IN-period NASDAQ leg (the designed product
      cost: nas_ret = L*r_nas - max(L-1,0)*(sofr+SWAP_SPREAD/252) - TER_TQQQ/252).

P09_TQQQ = apply the P09 OUT-fill logic on top of the TQQQ-cost V7 baseline daily
returns, instead of the CFD-cost baseline used in the original P09.

Task 1 : build P09_TQQQ, report 10 metrics (pretax + after-tax) + cy2022 + worstCY.
Task 2 : robustness -- block bootstrap (10,000 x block 21) on daily returns, P(P09_TQQQ
         beats baseline V7_TQQQ) for min(IS,OOS) CAGR and MaxDD, CI95_lo of CAGR gain;
         WFA via unified_wfa.summarize_wfa over canonical g1_wfa windows (WFE, CI95_lo).
Task 3 : lever-up variants (user accepts MaxDD worsening for CAGR):
         LU1 stronger V7 map {0:1.40,1:1.20,2:1.05,3:1.00};
         LU2 uniform IN NASDAQ leverage x1.15.

Premises: canonical split IS<=2021-05-07 / OOS>=2021-05-08, DELAY=2 (NASDAQ leg),
T+5 fund lag, after-tax x0.8273 (CAGR/Worst10Y/P10/Worst5Y), Sharpe/MaxDD pretax,
252-day. compute_10metrics for metrics. The TQQQ-V7 baseline must reproduce the
cost_model harness TQQQ V7 (min_at ~ +16.27%) -- printed check.

ASCII-only prints (Windows cp932 console). Saves into the repo; does NOT commit.
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
from src.audit.unified_wfa import summarize_wfa

# P01/P09 fill machinery (reused, unmodified)
from src.audit.run_p01_backtest_20260611 import (
    FEE_GOLD, FEE_BOND, LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _apply_aftertax, _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, _build_p09_nav,
)
# TQQQ-cost V7 baseline builder (reused, unmodified): build_nav_v7(..., "TQQQ")
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    build_nav_v7, _build_v7_mult, V7_MAP, TER_TQQQ, SWAP_SPREAD,
)

# Canonical WFA windows builder
from src.audit.compute_wfa_realistic_20260610 import build_per_window_df_canonical

AFTER_TAX = 0.8273
N_BOOT = 10000
BLOCK = 21
SEED = 20260611

# Lever-up configs
LU1_MAP = {0: 1.40, 1: 1.20, 2: 1.05, 3: 1.00}   # stronger V7 boost mapping
LU2_SCALE = 1.15                                  # uniform IN NASDAQ leverage scale


# ---------------------------------------------------------------------------
# Build a TQQQ-cost V7 baseline daily-return series with an OPTIONAL alternate
# V7 multiplier map / uniform leverage scale (for lever-up variants).
# ---------------------------------------------------------------------------
def _build_tqqq_base(shared, date_index, v7_map=None, lev_scale=1.0):
    """Return (nav_dt, r_base, tpy) for the TQQQ-cost V7 baseline.

    v7_map   : dict replacing V7_MAP for the IN-period boost (None => default).
    lev_scale: uniform multiplicative scale on the IN NASDAQ leverage (mult_v7).
    Everything else identical to cost_model harness build_nav_v7(..., 'TQQQ').
    """
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

    nav_dt, tpy = build_nav_v7(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7, "TQQQ")
    r_base = nav_dt.pct_change().fillna(0).values
    return nav_dt, r_base, tpy


def _build_v7_mult_custom(date_index, v7_map):
    """Same pipeline as cost_model._build_v7_mult but with a custom quantile->mult map."""
    macro_path = os.path.join(_REPO_DIR, "data", "macro_features.csv")
    macro = pd.read_csv(macro_path, parse_dates=["date"], index_col="date").sort_index()
    signal_raw = macro["nasdaq_mom63"].dropna()
    from signals.quantize import quantile_cut as _quantile_cut
    from signals.timing import apply_publication_lag as _apply_lag
    sig_q = _quantile_cut(signal_raw, levels=4).dropna().astype("int8")
    sig_lagged = _apply_lag(sig_q, "daily")
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    sig_aligned = sig_lagged.reindex(date_index).ffill()
    mult_arr = sig_aligned.map(
        lambda s: v7_map.get(int(s), 1.0) if pd.notna(s) else 1.0
    ).fillna(1.0).values
    return np.clip(np.asarray(mult_arr, dtype=float), 0.0, 3.0)


# ---------------------------------------------------------------------------
# Apply the P09 OUT-fill on top of a given TQQQ-cost baseline return series.
# ---------------------------------------------------------------------------
def _build_p09_on_base(r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
                       dates_dt, tpy_base, n_years):
    nav_arr, r_p09, eff_active = _build_p09_nav(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on)
    nav_dt = pd.Series(nav_arr, index=dates_dt)
    flips = _count_fund_transitions(eff_active)
    tpy = tpy_base + flips / n_years
    return nav_dt, r_p09, tpy


# ---------------------------------------------------------------------------
# Metrics packaging
# ---------------------------------------------------------------------------
def _metrics_pack(nav_dt, tpy):
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    cy = _calendar_year_returns(nav_dt)
    cy2022 = float(cy.get(2022, np.nan))
    worst_cy = float(cy.min())
    worst_cy_year = int(cy.idxmin())
    return pre, aft, cy2022, worst_cy, worst_cy_year


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


# ---------------------------------------------------------------------------
# Block bootstrap on daily returns: compare P09_TQQQ vs baseline V7_TQQQ.
# Resample the SAME block indices for both series (paired) so the comparison
# isolates the strategy difference, not independent path noise.
# ---------------------------------------------------------------------------
def _maxdd_from_returns(r):
    nav = np.cumprod(1.0 + np.clip(r, -0.999, None))
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def _cagr_seg(r):
    n = len(r)
    if n == 0:
        return np.nan
    nav_end = float(np.prod(1.0 + np.clip(r, -0.999, None)))
    if nav_end <= 0:
        return -1.0
    return nav_end ** (TRADING_DAYS / n) - 1.0


def _block_bootstrap_compare(r_strat, r_base, is_mask, oos_mask,
                             n_boot=N_BOOT, block=BLOCK, seed=SEED):
    """Paired stationary-block bootstrap. For each resample we draw block start
    indices, build a resampled day-ordering of length n, then evaluate BOTH series
    on that SAME ordering. Returns dict with P(strat better) for min(IS,OOS) after-tax
    CAGR and for MaxDD, plus CI95_lo of the min-CAGR improvement (after-tax, pp).

    IS/OOS CAGR computed by selecting resampled days whose ORIGINAL index falls in
    the IS / OOS span (mask carried along with the index draw)."""
    rng = np.random.default_rng(seed)
    n = len(r_strat)
    r_strat = np.asarray(r_strat, float)
    r_base = np.asarray(r_base, float)
    is_mask = np.asarray(is_mask, bool)
    oos_mask = np.asarray(oos_mask, bool)

    n_blocks = int(np.ceil(n / block))

    d_min = np.empty(n_boot)        # after-tax min(IS,OOS) improvement (strat - base), fraction
    strat_better_min = 0
    strat_better_dd = 0

    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]

        rs = r_strat[idx]
        rb = r_base[idx]
        im = is_mask[idx]
        om = oos_mask[idx]

        # after-tax IS/OOS CAGR
        s_is = _cagr_seg(rs[im]) * AFTER_TAX
        s_oos = _cagr_seg(rs[om]) * AFTER_TAX
        b_is = _cagr_seg(rb[im]) * AFTER_TAX
        b_oos = _cagr_seg(rb[om]) * AFTER_TAX
        s_min = np.nanmin([s_is, s_oos])
        b_min = np.nanmin([b_is, b_oos])

        d_min[b] = s_min - b_min
        if s_min > b_min:
            strat_better_min += 1

        # MaxDD on the resampled full path (less negative = better)
        s_dd = _maxdd_from_returns(rs)
        b_dd = _maxdd_from_returns(rb)
        if s_dd > b_dd:
            strat_better_dd += 1

    ci95_lo = float(np.percentile(d_min, 2.5))
    return {
        "P_min_better": strat_better_min / n_boot,
        "P_maxdd_better": strat_better_dd / n_boot,
        "CI95_lo_min_pp": ci95_lo * 100.0,
        "mean_min_improve_pp": float(np.mean(d_min)) * 100.0,
        "n_boot": n_boot, "block": block,
    }


def _run_wfa(nav_dt, label):
    per_df = build_per_window_df_canonical(nav_dt, label)
    if per_df is None:
        return {"WFE": np.nan, "CI95_lo": np.nan, "n_windows": 0, "t_p": np.nan}
    res = summarize_wfa(per_df)
    return {
        "WFE": res.get("WFA_WFE", np.nan),
        "CI95_lo": res.get("WFA_CI95_lo", np.nan),
        "n_windows": res.get("n_windows", 0),
        "t_p": res.get("t_pvalue", np.nan),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("P09_TQQQ : P09 bond-timing OUT-fill on the TQQQ-cost V7 IN leg  2026-06-11")
    print("=" * 78)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)  # 1=IN, 0=OUT
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / TRADING_DAYS

    # ---- TQQQ-cost V7 baseline (the new reference) ----
    base_nav_dt, r_base_tqqq, tpy_base = _build_tqqq_base(shared, dates_dt)

    # ---- Validation: reproduce cost_model harness TQQQ V7 (min_at ~ +16.27%) ----
    pre_base, aft_base, cy22_base, wcy_base, wcyy_base = _metrics_pack(base_nav_dt, tpy_base)
    min_at_base = _min_at(aft_base)
    print("")
    print("VALIDATION: TQQQ-V7 baseline vs cost_model harness (expect min_at ~ +16.27%)")
    print("  CAGR_IS_at = %+.4f%%  CAGR_OOS_at = %+.4f%%  min_at = %+.4f%%"
          % (100 * aft_base["CAGR_IS"], 100 * aft_base["CAGR_OOS"], 100 * min_at_base))
    print("  Sharpe_OOS = %.4f  MaxDD = %+.4f%%" % (pre_base["Sharpe_OOS"], 100 * pre_base["MaxDD_FULL"]))
    diff = abs(min_at_base * 100 - 16.27)
    print("  |min_at - 16.27%%| = %.4f pp -> %s" % (diff, "OK" if diff <= 0.15 else "CHECK"))
    print("")

    # ---- Gold/Bond 1x return series + OUT mask + fund lag (same as P09) ----
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

    # ---- TASK 1: P09_TQQQ ----
    p09_nav, r_p09, tpy_p09 = _build_p09_on_base(
        r_base_tqqq, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        dates_dt, tpy_base, n_years)
    pre9, aft9, cy22_9, wcy_9, wcyy_9 = _metrics_pack(p09_nav, tpy_p09)
    min_at_9 = _min_at(aft9)

    # ---- TASK 3: lever-up variants (P09 fill on a stronger IN baseline) ----
    # LU1: stronger V7 map
    lu1_base_nav, r_lu1_base, tpy_lu1b = _build_tqqq_base(shared, dates_dt, v7_map=LU1_MAP)
    lu1_nav, r_lu1, tpy_lu1 = _build_p09_on_base(
        r_lu1_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        dates_dt, tpy_lu1b, n_years)
    pre_lu1, aft_lu1, cy22_lu1, wcy_lu1, wcyy_lu1 = _metrics_pack(lu1_nav, tpy_lu1)
    min_at_lu1 = _min_at(aft_lu1)

    # LU2: uniform IN leverage x1.15
    lu2_base_nav, r_lu2_base, tpy_lu2b = _build_tqqq_base(shared, dates_dt, lev_scale=LU2_SCALE)
    lu2_nav, r_lu2, tpy_lu2 = _build_p09_on_base(
        r_lu2_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        dates_dt, tpy_lu2b, n_years)
    pre_lu2, aft_lu2, cy22_lu2, wcy_lu2, wcyy_lu2 = _metrics_pack(lu2_nav, tpy_lu2)
    min_at_lu2 = _min_at(aft_lu2)

    # ---- TASK 2: robustness on P09_TQQQ vs baseline V7_TQQQ ----
    is_mask = dates_dt <= IS_END
    oos_mask = dates_dt >= OOS_START
    print("Block bootstrap (%d resamples, block %d days), paired P09_TQQQ vs V7_TQQQ ..."
          % (N_BOOT, BLOCK))
    boot = _block_bootstrap_compare(r_p09, r_base_tqqq, is_mask, oos_mask)
    print("  P(min better)=%.4f  P(MaxDD better)=%.4f  CI95_lo(min gain)=%+.4f pp  mean gain=%+.4f pp"
          % (boot["P_min_better"], boot["P_maxdd_better"],
             boot["CI95_lo_min_pp"], boot["mean_min_improve_pp"]))
    print("")

    print("WFA (canonical g1_wfa windows) ...")
    wfa_base = _run_wfa(base_nav_dt, "V7_TQQQ")
    wfa_p09 = _run_wfa(p09_nav, "P09_TQQQ")
    print("  V7_TQQQ : WFE=%.3f  CI95_lo=%+.4f%%  n=%d  t_p=%.4f"
          % (wfa_base["WFE"], 100 * wfa_base["CI95_lo"], wfa_base["n_windows"], wfa_base["t_p"]))
    print("  P09_TQQQ: WFE=%.3f  CI95_lo=%+.4f%%  n=%d  t_p=%.4f"
          % (wfa_p09["WFE"], 100 * wfa_p09["CI95_lo"], wfa_p09["n_windows"], wfa_p09["t_p"]))
    print("")

    # ---- Assemble rows for CSV ----
    packs = {
        "baseline_V7_TQQQ": (pre_base, aft_base, cy22_base, wcy_base, wcyy_base),
        "P09_TQQQ": (pre9, aft9, cy22_9, wcy_9, wcyy_9),
        "LU1_strongmap": (pre_lu1, aft_lu1, cy22_lu1, wcy_lu1, wcyy_lu1),
        "LU2_lever1p15": (pre_lu2, aft_lu2, cy22_lu2, wcy_lu2, wcyy_lu2),
    }
    order = ["baseline_V7_TQQQ", "P09_TQQQ", "LU1_strongmap", "LU2_lever1p15"]

    rows = []
    for label in order:
        pre, aft, cy22, wcy, wcyy = packs[label]
        for tax_label, m in (("pretax", pre), ("aftertax", aft)):
            rows.append({
                "condition": label,
                "tax": tax_label,
                "CAGR_IS": m["CAGR_IS"],
                "CAGR_OOS": m["CAGR_OOS"],
                "CAGR_FULL": m["CAGR_FULL"],
                "min_IS_OOS": min(m["CAGR_IS"], m["CAGR_OOS"]),
                "IS_OOS_gap_pp": m["IS_OOS_gap_pp"],
                "Sharpe_OOS": pre["Sharpe_OOS"],
                "MaxDD_FULL": pre["MaxDD_FULL"],
                "Worst10Y_star": m["Worst10Y_star"],
                "P10_5Y": m["P10_5Y"],
                "Worst5Y": m["Worst5Y"],
                "Trades_yr": m["Trades_yr"],
                "cy2022_return": cy22,
                "worst_calendar_year_return": wcy,
                "worst_calendar_year": wcyy,
                # bootstrap / WFA summary columns (filled on P09_TQQQ row only)
                "boot_P_min_better": boot["P_min_better"] if label == "P09_TQQQ" else "",
                "boot_P_maxdd_better": boot["P_maxdd_better"] if label == "P09_TQQQ" else "",
                "boot_CI95_lo_min_pp": boot["CI95_lo_min_pp"] if label == "P09_TQQQ" else "",
                "wfa_WFE": (wfa_p09["WFE"] if label == "P09_TQQQ"
                            else (wfa_base["WFE"] if label == "baseline_V7_TQQQ" else "")),
                "wfa_CI95_lo": (wfa_p09["CI95_lo"] if label == "P09_TQQQ"
                                else (wfa_base["CI95_lo"] if label == "baseline_V7_TQQQ" else "")),
            })

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "p09_tqqq_validate_20260611.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8", float_format="%.6f")
    print("Saved CSV: %s" % csv_path)
    print("")

    # ---- ASCII table: Task 1 + Task 3 ----
    print("=" * 118)
    print("RESULTS (CAGR_IS_at / CAGR_OOS_at / min_at AFTER-TAX; Sharpe/MaxDD/cy2022 pretax)")
    print("=" * 118)
    hdr = ("%-18s | %10s | %11s | %9s | %7s | %8s | %8s | %8s | %8s"
           % ("condition", "CAGR_IS_at", "CAGR_OOS_at", "min_at",
              "Sharpe", "MaxDD", "cy2022", "worstCY", "Trades/yr"))
    print(hdr)
    print("-" * 118)
    for label in order:
        pre, aft, cy22, wcy, wcyy = packs[label]
        mn = _min_at(aft)
        print("%-18s | %+9.2f%% | %+10.2f%% | %+8.2f%% | %7.3f | %+7.2f%% | %+7.2f%% | %+6.2f%%(%d) | %8.1f"
              % (label, 100 * aft["CAGR_IS"], 100 * aft["CAGR_OOS"], 100 * mn,
                 pre["Sharpe_OOS"], 100 * pre["MaxDD_FULL"], 100 * cy22,
                 100 * wcy, wcyy, aft["Trades_yr"]))
    print("=" * 118)
    print("")

    # ---- Tradeoff: CAGR gain per pp of MaxDD worsening vs P09_TQQQ ----
    dd9 = pre9["MaxDD_FULL"]
    print("Lever-up tradeoff vs P09_TQQQ (min_at=%+.2f%%, MaxDD=%+.2f%%):" % (100 * min_at_9, 100 * dd9))
    print("-" * 78)
    for label, pre, aft in (("LU1_strongmap", pre_lu1, aft_lu1), ("LU2_lever1p15", pre_lu2, aft_lu2)):
        mn = _min_at(aft)
        d_cagr = (mn - min_at_9) * 100.0
        d_dd = (pre["MaxDD_FULL"] - dd9) * 100.0   # negative = worse (more negative DD)
        ratio = (d_cagr / abs(d_dd)) if abs(d_dd) > 1e-9 else float("nan")
        print("  %-16s d_min_at=%+.2fpp  d_MaxDD=%+.2fpp  CAGRgain/DDworsen=%.3f"
              % (label, d_cagr, d_dd, ratio))
    print("-" * 78)
    print("")

    # ---- JSON-like return block ----
    block = {
        "baseline_V7_TQQQ": {
            "min_at": round(min_at_base, 6),
            "Sharpe": round(pre_base["Sharpe_OOS"], 4),
            "MaxDD": round(pre_base["MaxDD_FULL"], 6),
        },
        "P09_TQQQ": {
            "CAGR_IS_at": round(aft9["CAGR_IS"], 6),
            "CAGR_OOS_at": round(aft9["CAGR_OOS"], 6),
            "min_at": round(min_at_9, 6),
            "Sharpe": round(pre9["Sharpe_OOS"], 4),
            "MaxDD": round(pre9["MaxDD_FULL"], 6),
            "cy2022": round(cy22_9, 6),
            "Trades_yr": round(aft9["Trades_yr"], 2),
        },
        "bootstrap": {
            "P_min_better": boot["P_min_better"],
            "CI95_lo_min_pp": round(boot["CI95_lo_min_pp"], 4),
            "P_maxdd_better": boot["P_maxdd_better"],
            "mean_min_improve_pp": round(boot["mean_min_improve_pp"], 4),
        },
        "WFA": {
            "WFE": round(float(wfa_p09["WFE"]), 4),
            "CI95_lo": round(float(wfa_p09["CI95_lo"]), 6),
            "n_windows": int(wfa_p09["n_windows"]),
            "t_p": round(float(wfa_p09["t_p"]), 4),
        },
        "LU1": {"min_at": round(min_at_lu1, 6), "MaxDD": round(pre_lu1["MaxDD_FULL"], 6)},
        "LU2": {"min_at": round(min_at_lu2, 6), "MaxDD": round(pre_lu2["MaxDD_FULL"], 6)},
    }
    print("=" * 78)
    print("RETURN_BLOCK")
    print("=" * 78)
    print(json.dumps(block, indent=2))
    print("")
    print("Done.")
    return block


if __name__ == "__main__":
    main()
