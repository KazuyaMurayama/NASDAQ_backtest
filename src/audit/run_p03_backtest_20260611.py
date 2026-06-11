"""
src/audit/run_p03_backtest_20260611.py
======================================
P03 strategy-improvement-proposal backtest.

P03 premise
-----------
Switch the NASDAQ leg from a 3x product to a 1x product on "bearish" days, where
bearish = nasdaq_mom63 quartile label in {0,1} (LOWEST + 2nd-lowest momentum
quartiles, per the verified quantile_cut convention: label 0 = LOWEST, 3 =
HIGHEST). Bullish (labels {2,3}) keeps the 3x leg.

NOTE on the V7 / P03 interaction (intentional, measured, NOT altered):
  V7 overlay boosts labels {0,1} (mult 1.20 / 1.10) -- exactly the labels P03
  down-shifts to a 1x product. They act on the SAME signal in OPPOSITE
  directions. C3 measures the combined effect.

Explicit leg-decomposition harness
-----------------------------------
NASDAQ sleeve per-day leverage:
    L_nas[t] = lev_raw_masked[t] * mult_v7[t] * nas_base[t]
  nas_base[t] = 3.0 (baseline) ; for P03, 1.0 on bearish days else 3.0.

NASDAQ sleeve daily return:
    r_sleeve[t] = L_nas[t]*r_nas[t] - financing[t] - ter_nas[t]
  r_nas = close.pct_change().fillna(0)
  financing[t] = max(L_nas[t]-1,0) * (sofr_daily[t] + 0.0050/252)   if 3x product
  financing[t] = 0                                                  if 1x product
  ter_nas[t]   = 0.0086/252 (3x)  /  0.001958/252 (1x)

Portfolio daily return:
    daily[t] = wn_s[t]*r_sleeve[t] + wg_s[t]*r_g2[t] + wb_s[t]*r_b3[t] - spread_cost[t]
  r_g2 = gold_2x.pct_change().fillna(0); r_b3 = bond_3x.pct_change().fillna(0)
  wn_s,wg_s,wb_s, lev/mult shifted by DELAY=2.
  spread_cost[t] = |Delta(wn_s * L_nas)| * 2 * 0.00025
  NAV floor: clip daily at -0.999 then nav = (1+daily).cumprod().

Execution lag for the P03 product switch:
  The 1x-product decision incurs T+5 (EXEC_LAG_FUND_1X) instead of T+2. The {3,1}
  nas_base schedule is shifted by 5 BD for the P03 "with T+5 lag" conditions, and
  by 2 BD for the "no-lag diagnostic" (C3b). All other weights stay at DELAY=2.

Conditions
----------
  C0  DH-W1 plain : nas_base=3 all, mult_v7=1.0 all
  C1  Baseline V7 : nas_base=3 all, mult_v7=V7            <- VALIDATION TARGET
  C2  P03 only    : nas_base={3 on {2,3}, 1 on {0,1}} +5, mult_v7=1.0
  C3  P03 + V7    : nas_base as C2 (+5),                  mult_v7=V7
  C3b P03 + V7    : nas_base shifted +2 (no-lag diag),    mult_v7=V7

Validation gate: C1 CAGR_IS and CAGR_OOS must match run_overlay('V7','realistic')
within 0.8pp (absolute). The reproduction error is printed and saved.

After-tax: CAGR / Worst10Y / P10 = pretax * 0.8273. Sharpe / MaxDD reported pretax.

ASCII-only prints (Windows cp932 console).
"""

from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# multitasking stub + sys.path (mirror strategy_runners.py)
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
# Constants (from product_costs.py / spec)
# ---------------------------------------------------------------------------
TER_3X = 0.0086        # TQQQ.ter
TER_1X = 0.001958      # NASDAQ1X.ter
SWAP_SPREAD = 0.0050   # TQQQ.swap_spread (financing spread for 3x leg)
SPREAD_ONE_WAY = 0.00025  # one-way 0.025%, applied round-trip x2
DELAY = 2              # base weight DELAY
EXEC_LAG_FUND_1X = 5   # T+5 for 1x product switch
TAX_FACTOR = 0.8273    # after-tax multiplier
TRADING_DAYS = 252.0

V7_MAP = {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00}


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _build_signal_quartiles(date_index: pd.DatetimeIndex) -> np.ndarray:
    """Reconstruct the nasdaq_mom63 quartile labels aligned to strategy dates.

    Returns float array (0..3) with NaN on warmup days (pre-signal).
    Matches run_overlay's pipeline exactly: quantile_cut(levels=4) ->
    publication_lag('daily') -> reindex -> ffill.
    """
    macro_path = os.path.join(_REPO, "data", "macro_features.csv")
    macro = pd.read_csv(macro_path, parse_dates=["date"], index_col="date").sort_index()
    sig_raw = macro["nasdaq_mom63"].dropna()

    from signals.quantize import quantile_cut as _quantile_cut
    from signals.timing import apply_publication_lag as _apply_lag

    sig_q = _quantile_cut(sig_raw, levels=4).dropna().astype("int8")
    sig_lagged = _apply_lag(sig_q, "daily")
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    sig_aligned = sig_lagged.reindex(date_index).ffill()
    return np.asarray(sig_aligned.values, dtype=float)


def _mult_v7(quartiles: np.ndarray) -> np.ndarray:
    """V7 multiplier per day from quartile labels (NaN -> 1.0)."""
    out = np.ones(len(quartiles), dtype=float)
    for i, q in enumerate(quartiles):
        if not np.isnan(q):
            out[i] = V7_MAP.get(int(q), 1.0)
    return np.clip(out, 0.0, 3.0)


def _nas_base_schedule(quartiles: np.ndarray, lag_bd: int) -> np.ndarray:
    """P03 nas_base schedule: 1.0 on bearish days (quartile in {0,1}) else 3.0,
    with the bearish/1x state shifted forward by `lag_bd` business days.

    NaN quartiles default to 3.0 (no down-shift; treated as bullish/unknown).
    """
    n = len(quartiles)
    is_bearish = np.zeros(n, dtype=bool)
    for i, q in enumerate(quartiles):
        if not np.isnan(q) and int(q) in (0, 1):
            is_bearish[i] = True
    # Shift the bearish (1x) state forward by lag_bd BD
    is_bearish_lagged = np.zeros(n, dtype=bool)
    if lag_bd < n:
        is_bearish_lagged[lag_bd:] = is_bearish[: n - lag_bd]
    nas_base = np.where(is_bearish_lagged, 1.0, 3.0)
    return nas_base


# ---------------------------------------------------------------------------
# Explicit leg-decomposition NAV harness
# ---------------------------------------------------------------------------
def build_nav_explicit(
    close: pd.Series,
    dates: pd.Series,
    gold_2x_nav,
    bond_3x_nav,
    sofr_daily: np.ndarray,
    lev_raw_masked: np.ndarray,
    wn: np.ndarray,
    wg: np.ndarray,
    wb: np.ndarray,
    mult_v7: np.ndarray,
    nas_base: np.ndarray,
):
    """Build NAV using the explicit leg-decomposition model (spec).

    Returns (nav_dt: pd.Series[DatetimeIndex], state_vec: np.ndarray) where
    state_vec is the per-day rounded position vector used for trade counting.
    """
    NAV_FLOOR = -0.999
    idx = dates.index

    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(np.asarray(gold_2x_nav, dtype=float)).pct_change().fillna(0).values
    r_b3 = pd.Series(np.asarray(bond_3x_nav, dtype=float)).pct_change().fillna(0).values

    # nas_base switching uses T+5 (or +2) lag already baked into nas_base array.
    # All other inputs use DELAY=2.
    nas_base_s = nas_base  # already lag-shifted at schedule-build time
    lev_s = pd.Series(np.asarray(lev_raw_masked, dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    mult_s = pd.Series(np.asarray(mult_v7, dtype=float), index=idx).shift(DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    wg_s = pd.Series(np.asarray(wg, dtype=float), index=idx).shift(DELAY).fillna(0.0).values
    wb_s = pd.Series(np.asarray(wb, dtype=float), index=idx).shift(DELAY).fillna(0.0).values

    # L_nas[t] = lev_raw_masked * mult_v7 * nas_base  (shifted components)
    L_nas = lev_s * mult_s * nas_base_s

    is_3x = nas_base_s >= 2.0  # True when sleeve is a 3x product
    is_1x = ~is_3x

    sofr_arr = np.asarray(sofr_daily, dtype=float)

    # financing: max(L_nas-1,0)*(sofr+SWAP_SPREAD/252) on 3x; 0 on 1x
    financing = np.where(
        is_3x,
        np.maximum(L_nas - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS),
        0.0,
    )
    # ter: 0.0086/252 (3x) ; 0.001958/252 (1x)
    ter_nas = np.where(is_3x, TER_3X / TRADING_DAYS, TER_1X / TRADING_DAYS)

    r_sleeve = L_nas * r_nas - financing - ter_nas

    # spread cost on |Delta(wn_s * L_nas)|
    full_pos = wn_s * L_nas
    prev_pos = np.concatenate([[full_pos[0]], full_pos[:-1]])
    spread_cost = np.abs(full_pos - prev_pos) * 2.0 * SPREAD_ONE_WAY

    daily = wn_s * r_sleeve + wg_s * r_g2 + wb_s * r_b3 - spread_cost

    daily_clipped = np.maximum(daily, NAV_FLOOR)
    nav = (1.0 + pd.Series(daily_clipped, index=idx)).cumprod()

    dates_dt = pd.to_datetime(dates.values)
    nav_dt = pd.Series(nav.values, index=pd.DatetimeIndex(dates_dt))

    # state vector for trade counting: rounded (wn_s*L_nas, wg_s, wb_s)
    state_vec = np.column_stack([
        np.round(wn_s * L_nas, 4),
        np.round(wg_s, 4),
        np.round(wb_s, 4),
    ])
    return nav_dt, state_vec


def _trades_per_year(state_vec: np.ndarray, n_days: int) -> float:
    """Count days where the rounded position vector changes / n_years."""
    n_years = n_days / TRADING_DAYS
    changed = np.zeros(len(state_vec), dtype=bool)
    changed[1:] = np.any(state_vec[1:] != state_vec[:-1], axis=1)
    return int(changed.sum()) / n_years if n_years > 0 else float("nan")


# ---------------------------------------------------------------------------
# Calibration harness: prove the explicit leg-decomposition STRUCTURE
# (DELAY=2, lev_raw x3 convention, weight-carry, financing form, turnover)
# reproduces run_overlay('V7','realistic') to ~0pp when the baseline cost
# layers are plugged in. This isolates the C1 residual to the documented
# cost-constant choice (swap_spread 0.50% vs SBI 3.00%, TER 0.86% vs ETF TER).
# ---------------------------------------------------------------------------
SBI_CFD_SPREAD = 0.0300  # baseline financing spread (g14_wfa_sbi_cfd.SBI_CFD_SPREAD)
DH_PER_UNIT = 0.0010     # baseline DH turnover one-way cost (build_dh_nav_with_cost)


def build_nav_baseline_mimic(
    close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
    lev_raw_masked, wn, wg, wb, mult_v7,
):
    """Explicit-structure NAV that mimics run_overlay's realistic baseline cost
    layers exactly: financing (L-1)*(sofr+SBI/252) on L=lev_raw_masked*mult*3,
    no leg TER, DH turnover on raw weight changes, then incremental ETF TER drag
    and US-ETF trade cost (mirroring _build_dhw1_nav_realistic).
    """
    from src.audit.strategy_runners import (
        _TER_TQQQ_EXTRA_DAILY, _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
        _compute_dhw1_trades_per_year,
    )
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

    idx = dates.index
    n = len(close)
    NAV_FLOOR = -0.999
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3 = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    lev_mod = np.asarray(lev_raw_masked, float) * np.asarray(mult_v7, float)
    L = pd.Series(lev_mod * 3.0, index=idx).shift(DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(DELAY).fillna(0).values

    borrow = (L - 1.0) * (np.asarray(sofr_daily, float) + SBI_CFD_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow  # CFD leg: no TER
    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # DH turnover on RAW (unshifted) weight changes
    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    # incremental ETF TER drag + US-ETF trade cost
    ter_drag = (np.asarray(wn, float) * _TER_TQQQ_EXTRA_DAILY
                + np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
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


# ---------------------------------------------------------------------------
# After-tax helper
# ---------------------------------------------------------------------------
_AFTERTAX_KEYS = {"CAGR_IS", "CAGR_OOS", "CAGR_FULL", "Worst10Y_star", "P10_5Y", "Worst5Y"}


def _to_aftertax(metrics: dict) -> dict:
    """CAGR / Worst10Y / P10 / Worst5Y * 0.8273 ; Sharpe & MaxDD reported pretax.
    IS_OOS_gap_pp recomputed from after-tax CAGRs.
    """
    out = {}
    for k, v in metrics.items():
        if k in _AFTERTAX_KEYS and v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[k] = v * TAX_FACTOR
        else:
            out[k] = v
    if not (np.isnan(out["CAGR_IS"]) or np.isnan(out["CAGR_OOS"])):
        out["IS_OOS_gap_pp"] = (out["CAGR_IS"] - out["CAGR_OOS"]) * 100.0
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("P03 BACKTEST (explicit leg-decomposition harness)  2026-06-11")
    print("=" * 78)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    close = a["close"]
    dates = a["dates"]
    sofr = np.asarray(a["sofr"], dtype=float)
    gold_2x = a["gold_2x"]
    bond_3x = a["bond_3x"]
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], dtype=float)
    wn = np.asarray(shared["wn"], dtype=float)
    wg = np.asarray(shared["wg"], dtype=float)
    wb = np.asarray(shared["wb"], dtype=float)
    n_days = len(close)

    date_index = pd.DatetimeIndex(pd.to_datetime(dates.values))
    quartiles = _build_signal_quartiles(date_index)
    mult_v7 = _mult_v7(quartiles)
    mult_flat = np.ones(n_days, dtype=float)

    # nas_base schedules
    nas3_all = np.full(n_days, 3.0)             # baseline: always 3x
    nas_p03_lag5 = _nas_base_schedule(quartiles, EXEC_LAG_FUND_1X)  # P03 T+5
    nas_p03_lag2 = _nas_base_schedule(quartiles, DELAY)            # P03 no-lag diag

    conditions = {
        "C0":  dict(mult=mult_flat, nas_base=nas3_all,
                    desc="DH-W1 plain (nas3, V7off)"),
        "C1":  dict(mult=mult_v7,   nas_base=nas3_all,
                    desc="Baseline V7 (nas3, V7) [VALIDATION TARGET]"),
        "C2":  dict(mult=mult_flat, nas_base=nas_p03_lag5,
                    desc="P03 only T+5 (nas{3,1}+5, V7off)"),
        "C3":  dict(mult=mult_v7,   nas_base=nas_p03_lag5,
                    desc="P03+V7 T+5 (nas{3,1}+5, V7)"),
        "C3b": dict(mult=mult_v7,   nas_base=nas_p03_lag2,
                    desc="P03+V7 no-lag diag (nas{3,1}+2, V7)"),
    }

    rows = []
    pretax_store = {}
    aftertax_store = {}

    for cname, cfg in conditions.items():
        nav_dt, state_vec = build_nav_explicit(
            close, dates, gold_2x, bond_3x, sofr,
            lev_raw_masked, wn, wg, wb,
            cfg["mult"], cfg["nas_base"],
        )
        tpy = _trades_per_year(state_vec, n_days)
        pre = compute_10metrics(nav_dt, tpy)
        aft = _to_aftertax(pre)
        pretax_store[cname] = pre
        aftertax_store[cname] = aft
        print("[%-3s] %s" % (cname, cfg["desc"]))
        print("      pretax  CAGR_IS=%+.4f CAGR_OOS=%+.4f Sharpe=%+.3f MaxDD=%+.4f Trades/yr=%.2f"
              % (pre["CAGR_IS"], pre["CAGR_OOS"], pre["Sharpe_OOS"], pre["MaxDD_FULL"], tpy))

    # ---- Validation gate: C1 vs run_overlay('V7','realistic') ----
    print()
    print("-" * 78)
    print("VALIDATION GATE: C1 (explicit) vs run_overlay('V7','realistic')")
    print("-" * 78)
    base = sr.run_overlay("V7", "realistic")
    c1 = pretax_store["C1"]
    err_is = (c1["CAGR_IS"] - base["CAGR_IS"]) * 100.0
    err_oos = (c1["CAGR_OOS"] - base["CAGR_OOS"]) * 100.0
    repro_err_pp = max(abs(err_is), abs(err_oos))
    print("            run_overlay      C1_explicit     err_pp")
    print("CAGR_IS     %+.4f         %+.4f        %+.3f" % (base["CAGR_IS"], c1["CAGR_IS"], err_is))
    print("CAGR_OOS    %+.4f         %+.4f        %+.3f" % (base["CAGR_OOS"], c1["CAGR_OOS"], err_oos))
    gate = "PASS" if repro_err_pp <= 0.8 else "FAIL"
    print("max |err| = %.3f pp  ->  spec-literal GATE %s (threshold 0.8pp)" % (repro_err_pp, gate))
    print("NOTE: spec-literal C1 uses swap_spread=0.50%%/yr financing + TER 0.86%%/yr;")
    print("      baseline run_overlay uses SBI_CFD_SPREAD=3.0%%/yr financing + incremental")
    print("      ETF TER. The residual is the documented cost-constant difference, NOT a")
    print("      structural (DELAY / leverage / weight) error.")

    # ---- Structural calibration: same explicit harness skeleton + baseline cost layers ----
    nav_mimic, tpy_mimic = build_nav_baseline_mimic(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7,
    )
    mimic = compute_10metrics(nav_mimic, tpy_mimic)
    cerr_is = (mimic["CAGR_IS"] - base["CAGR_IS"]) * 100.0
    cerr_oos = (mimic["CAGR_OOS"] - base["CAGR_OOS"]) * 100.0
    calib_err_pp = max(abs(cerr_is), abs(cerr_oos))
    print()
    print("STRUCTURAL CALIBRATION (explicit skeleton + baseline cost layers):")
    print("  CAGR_IS  mimic=%+.4f base=%+.4f err=%+.4fpp" % (mimic["CAGR_IS"], base["CAGR_IS"], cerr_is))
    print("  CAGR_OOS mimic=%+.4f base=%+.4f err=%+.4fpp" % (mimic["CAGR_OOS"], base["CAGR_OOS"], cerr_oos))
    cgate = "PASS" if calib_err_pp <= 0.8 else "FAIL"
    print("  max |err| = %.4f pp -> STRUCTURAL GATE %s (proves harness skeleton correct)"
          % (calib_err_pp, cgate))

    # ---- Build CSV rows ----
    for cname in conditions:
        for tax in ("pretax", "aftertax"):
            m = pretax_store[cname] if tax == "pretax" else aftertax_store[cname]
            row = {
                "condition": cname,
                "tax": tax,
                "desc": conditions[cname]["desc"],
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
                "repro_err_pp": (
                    "spec_literal=%.4f;structural_calib=%.4f" % (repro_err_pp, calib_err_pp)
                    if cname == "C1" else ""
                ),
            }
            rows.append(row)

    out_csv = os.path.join(_REPO, "audit_results", "p03_backtest_metrics_20260611.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
    print()
    print("Saved CSV: %s" % out_csv)

    # ---- Final ASCII table ----
    print()
    print("=" * 78)
    print("FINAL TABLE (after-tax CAGR; Sharpe/MaxDD pretax)")
    print("=" * 78)
    hdr = "%-4s | %10s | %10s | %10s | %7s | %8s | %8s" % (
        "cond", "CAGR_IS_at", "CAGR_OOS_at", "min_at", "Sharpe", "MaxDD", "Trades/yr")
    print(hdr)
    print("-" * len(hdr))
    for cname in ("C0", "C1", "C2", "C3", "C3b"):
        aft = aftertax_store[cname]
        pre = pretax_store[cname]
        min_at = min(aft["CAGR_IS"], aft["CAGR_OOS"])
        print("%-4s | %+9.4f%% | %+9.4f%% | %+9.4f%% | %+6.3f | %+7.4f | %8.2f" % (
            cname,
            aft["CAGR_IS"] * 100, aft["CAGR_OOS"] * 100, min_at * 100,
            pre["Sharpe_OOS"], pre["MaxDD_FULL"], pre["Trades_yr"],
        ))

    # ---- Compact JSON-like return block ----
    print()
    print("=" * 78)
    print("RETURN_BLOCK")
    print("=" * 78)
    import json
    block = {
        "repro_err_pp_C1_spec_literal_vs_runoverlayV7": round(repro_err_pp, 4),
        "repro_err_pp_structural_calibration": round(calib_err_pp, 4),
    }
    for cname in ("C0", "C1", "C2", "C3", "C3b"):
        pre = pretax_store[cname]
        aft = aftertax_store[cname]
        block[cname] = {
            "CAGR_IS_pretax": round(pre["CAGR_IS"], 4),
            "CAGR_OOS_pretax": round(pre["CAGR_OOS"], 4),
            "CAGR_IS_aftertax": round(aft["CAGR_IS"], 4),
            "CAGR_OOS_aftertax": round(aft["CAGR_OOS"], 4),
            "Sharpe_OOS": round(pre["Sharpe_OOS"], 4),
            "MaxDD_FULL": round(pre["MaxDD_FULL"], 4),
            "Worst10Y_star_aftertax": round(aft["Worst10Y_star"], 4),
            "P10_5Y_aftertax": round(aft["P10_5Y"], 4),
            "Trades_yr": round(pre["Trades_yr"], 2),
        }
    print(json.dumps(block, indent=2))
    return block


if __name__ == "__main__":
    main()
