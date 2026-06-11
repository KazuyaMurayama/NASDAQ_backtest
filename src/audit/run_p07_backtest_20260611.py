"""
src/audit/run_p07_backtest_20260611.py
======================================
P07 strategy-improvement-proposal backtest.

P07 premise (volatility-target leverage cap -- MaxDD-specialist layer, angle 2)
-------------------------------------------------------------------------------
Cap the IN-period NASDAQ effective leverage by realized volatility so the
strategy auto-de-levers in high-vol regimes. This uses a DIFFERENT axis than
V7 / nasdaq_mom63 (no signal collision, unlike the rejected P03). Goal: cut
MaxDD (-34.66% baseline) WITHOUT lowering min(IS,OOS) after-tax CAGR (+15.07%
baseline).

Skeleton reuse (fair A/B)
-------------------------
The canonical baseline run_overlay('V7','realistic') is reproduced to 0.00pp by
the *baseline-mimic* leg-decomposition skeleton in run_p03_backtest_20260611.py
(SBI_CFD_SPREAD=3.00%/yr financing on a CFD NASDAQ leg, no leg TER, DH turnover
on raw weight changes, then incremental ETF TER drag + US-ETF trade cost). NOTE:
the *spec-literal* build_nav_explicit skeleton (swap_spread=0.50%) is OFF by
~1.5pp vs run_overlay -- it does NOT reproduce the canonical baseline. The P07
spec's financing formula `(L-1)*(sofr+0.03/252)` uses 0.03 = SBI_CFD_SPREAD,
i.e. the mimic skeleton. We therefore build P07 on the mimic skeleton so that
C1 reproduces run_overlay('V7','realistic') within 0.1pp (validation gate).

P07 construction
----------------
- NASDAQ realized vol:
      sigma63[t] = rolling std of close.pct_change() over 63 trading days
                   annualized * sqrt(252).  (1974-computable; no VIX proxy.)
- vol_scale_target[t] = clip(sigma_target / sigma63[t], cap_lo, 1.0)
      DE-LEVER ONLY (cap at 1.0 -- never boost). sigma_target IS-fixed (no tune).
      cap_lo = 0.33 (never cut below ~1/3 leverage).
- Turnover control: recompute vol_scale every 21 BD (hold between) + deadband
      (only adopt a new held value if |delta vol_scale| > 0.1). Then DELAY=2 the
      resulting scale series (ETF T+2; no product switch so no T+5).
- Apply to the baseline NASDAQ-leg leverage:
      L_nas_p07[t] = L_baseline_shifted[t] * vol_scale_shifted[t]
  where L_baseline = lev_raw_masked * mult_v7 * 3.0 (the C1 leverage). Gold/bond
  legs unchanged. V7 unchanged. Financing/TER on the NASDAQ leg scale with the
  new (lower) leverage automatically via the same CFD formula.

Conditions (each -> full 10 metrics, pretax + after-tax x0.8273)
----------------------------------------------------------------
  C1        Baseline V7 (mimic skeleton, vol_scale==1 all). VALIDATION TARGET.
  P07_st20  sigma_target=0.20 (main).
  P07_st15  sigma_target=0.15 (more aggressive de-lever -- sensitivity).
  P07_st25  sigma_target=0.25 (milder -- sensitivity).

After-tax: CAGR / Worst10Y / P10 / Worst5Y = pretax * 0.8273. Sharpe / MaxDD
pretax. ASCII-only prints (Windows cp932 console).
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

# Reuse the proven helpers from the P03 harness (signal quartiles, V7 map).
from src.audit.run_p03_backtest_20260611 import (
    _build_signal_quartiles,
    _mult_v7,
    SBI_CFD_SPREAD,
    DH_PER_UNIT,
    DELAY,
    TRADING_DAYS,
    TAX_FACTOR,
)

# ---------------------------------------------------------------------------
# P07 constants
# ---------------------------------------------------------------------------
VOL_WINDOW = 63          # rolling realized-vol window (trading days)
CAP_LO = 0.33            # never cut below ~1/3 leverage
CAP_HI = 1.0             # de-lever only: never boost
REBAL_BD = 21            # recompute vol_scale every 21 BD, hold between
DEADBAND = 0.10          # only adopt change if |delta vol_scale| > 0.10
NAV_FLOOR = -0.999

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


# ---------------------------------------------------------------------------
# Vol-scale construction
# ---------------------------------------------------------------------------
def _build_vol_scale(close: pd.Series, sigma_target: float):
    """Build the P07 vol_scale series (BEFORE DELAY shift).

    Returns (vol_scale: np.ndarray, n_changes: int) where n_changes counts the
    number of monthly rebalance points whose held value passed the deadband
    (i.e. actual changes to the applied scale -- used for added-turnover/Trades).
    """
    n = len(close)
    r = close.pct_change()
    # rolling realized vol, annualized; min_periods=VOL_WINDOW so warmup -> NaN
    sigma63 = r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(TRADING_DAYS)
    sigma63 = sigma63.values

    # raw target scale per day (de-lever only)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_scale = np.clip(sigma_target / sigma63, CAP_LO, CAP_HI)
    # warmup (NaN sigma) -> scale 1.0 (no de-lever before vol is computable)
    raw_scale = np.where(np.isfinite(raw_scale), raw_scale, 1.0)

    # Monthly recompute + hold-between + deadband.
    # We sample raw_scale every REBAL_BD; between samples the applied scale is
    # held. A new sampled value is only adopted if it differs from the currently
    # applied value by more than DEADBAND.
    applied = np.ones(n, dtype=float)
    current = 1.0
    n_changes = 0
    for t in range(n):
        if t % REBAL_BD == 0:
            candidate = raw_scale[t]
            if abs(candidate - current) > DEADBAND:
                current = candidate
                n_changes += 1
        applied[t] = current
    return applied, n_changes


# ---------------------------------------------------------------------------
# P07 NAV harness (built on the baseline-mimic skeleton that reproduces
# run_overlay('V7','realistic') to 0.00pp). vol_scale multiplies the NASDAQ-leg
# leverage L (after DELAY shift). vol_scale==1 everywhere reduces to C1.
# ---------------------------------------------------------------------------
def build_nav_p07(
    close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
    lev_raw_masked, wn, wg, wb, mult_v7,
    vol_scale,
):
    """Returns (nav_dt: pd.Series[DatetimeIndex], base_tpy: float,
    state_vec: np.ndarray).

    base_tpy is the DH/V7 trades-per-year from the unchanged weight schedule
    (mirrors the baseline). P07's added turnover is counted separately from
    vol_scale changes. state_vec is the rounded NASDAQ exposure path for
    diagnostics.
    """
    from src.audit.strategy_runners import (
        _TER_TQQQ_EXTRA_DAILY, _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
        _compute_dhw1_trades_per_year,
    )
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

    idx = dates.index
    n = len(close)
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3 = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    lev_mod = np.asarray(lev_raw_masked, float) * np.asarray(mult_v7, float)
    L_base = pd.Series(lev_mod * 3.0, index=idx).shift(DELAY).fillna(1.0).values
    # DELAY=2 the vol_scale (ETF T+2; no product switch -> no T+5)
    vs_s = pd.Series(np.asarray(vol_scale, float), index=idx).shift(DELAY).fillna(1.0).values

    # P07 effective NASDAQ-leg leverage
    L = L_base * vs_s

    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(DELAY).fillna(0).values

    # CFD NASDAQ leg: financing (L-1)*(sofr+SBI/252), no leg TER. Lower L (from
    # vol_scale) automatically lowers financing & gross exposure.
    borrow = (L - 1.0) * (np.asarray(sofr_daily, float) + SBI_CFD_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow
    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # DH turnover on RAW (unshifted) weight changes (baseline behaviour)
    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    # incremental ETF TER drag + US-ETF trade cost (baseline cost layers)
    ter_drag = (np.asarray(wn, float) * _TER_TQQQ_EXTRA_DAILY
                + np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
                + np.asarray(wb, float) * _TER_TMF_EXTRA_DAILY)
    base_tpy = _compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(base_tpy) / 252.0
    r_sim = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_sim - ter_drag - etf_daily
    nav_adj = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )

    state_vec = np.round(wn_s * L, 4)
    return nav_adj, base_tpy, state_vec


# ---------------------------------------------------------------------------
# Worst calendar year (pretax) for tail-confirmation
# ---------------------------------------------------------------------------
def _worst_calendar_year(nav_dt: pd.Series):
    """Return (year, return_pct) of the worst calendar-year NAV return."""
    nav = nav_dt.dropna().sort_index()
    yr_end = nav.resample("Y").last()
    yr_ret = yr_end.pct_change().dropna()
    # include the first (partial) year from start-of-series to first year end
    if len(yr_end) > 0:
        first_year_ret = yr_end.iloc[0] / nav.iloc[0] - 1.0
        first_idx = yr_end.index[0]
        yr_ret = pd.concat([pd.Series([first_year_ret], index=[first_idx]), yr_ret])
        yr_ret = yr_ret[~yr_ret.index.duplicated(keep="last")].sort_index()
    if len(yr_ret) == 0:
        return (None, np.nan)
    wy = yr_ret.idxmin()
    return (int(wy.year), float(yr_ret.min()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("P07 BACKTEST (vol-target leverage cap; mimic skeleton)  2026-06-11")
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

    vs_ones = np.ones(n_days, dtype=float)

    conditions = {
        "C1": dict(sigma_target=None, desc="Baseline V7 (vol_scale=1) [VALIDATION TARGET]"),
        "P07_st20": dict(sigma_target=0.20, desc="P07 sigma_target=0.20 (main)"),
        "P07_st15": dict(sigma_target=0.15, desc="P07 sigma_target=0.15 (aggressive)"),
        "P07_st25": dict(sigma_target=0.25, desc="P07 sigma_target=0.25 (mild)"),
    }

    pretax_store, aftertax_store, extra = {}, {}, {}

    for cname, cfg in conditions.items():
        if cfg["sigma_target"] is None:
            vs = vs_ones
            n_changes = 0
        else:
            vs, n_changes = _build_vol_scale(close, cfg["sigma_target"])

        nav_dt, base_tpy, _state = build_nav_p07(
            close, dates, gold_2x, bond_3x, sofr,
            lev_raw_masked, wn, wg, wb, mult_v7, vs,
        )
        # Trades/yr = baseline DH/V7 trades + vol-scale changes per year
        n_years = n_days / TRADING_DAYS
        vs_tpy = n_changes / n_years if n_years > 0 else 0.0
        total_tpy = base_tpy + vs_tpy

        pre = compute_10metrics(nav_dt, total_tpy)
        aft = _to_aftertax(pre)
        wy_year, wy_ret = _worst_calendar_year(nav_dt)

        pretax_store[cname] = pre
        aftertax_store[cname] = aft
        extra[cname] = dict(
            base_tpy=base_tpy, vs_tpy=vs_tpy, n_changes=n_changes,
            worst_year=wy_year, worst_year_ret=wy_ret,
            min_vs=float(np.min(vs)), mean_vs=float(np.mean(vs)),
        )
        print("[%-8s] %s" % (cname, cfg["desc"]))
        print("      pretax CAGR_IS=%+.4f CAGR_OOS=%+.4f Sharpe=%+.3f MaxDD=%+.4f Trades/yr=%.2f"
              % (pre["CAGR_IS"], pre["CAGR_OOS"], pre["Sharpe_OOS"], pre["MaxDD_FULL"], total_tpy))
        print("      vol_scale: changes=%d (=%.2f/yr) min=%.3f mean=%.3f | worst_cal_yr=%s (%+.2f%%)"
              % (n_changes, vs_tpy, extra[cname]["min_vs"], extra[cname]["mean_vs"],
                 wy_year, wy_ret * 100.0))

    # ---- Validation gate: C1 vs run_overlay('V7','realistic') ----
    print()
    print("-" * 78)
    print("VALIDATION GATE: C1 (P07 harness, vol_scale=1) vs run_overlay('V7','realistic')")
    print("-" * 78)
    base = sr.run_overlay("V7", "realistic")
    c1 = pretax_store["C1"]
    err_is = (c1["CAGR_IS"] - base["CAGR_IS"]) * 100.0
    err_oos = (c1["CAGR_OOS"] - base["CAGR_OOS"]) * 100.0
    repro_err_pp = max(abs(err_is), abs(err_oos))
    print("            run_overlay      C1_P07harness   err_pp")
    print("CAGR_IS     %+.4f         %+.4f        %+.4f" % (base["CAGR_IS"], c1["CAGR_IS"], err_is))
    print("CAGR_OOS    %+.4f         %+.4f        %+.4f" % (base["CAGR_OOS"], c1["CAGR_OOS"], err_oos))
    gate = "PASS" if repro_err_pp <= 0.1 else "FAIL"
    print("max |err| = %.4f pp  ->  VALIDATION GATE %s (threshold 0.1pp)" % (repro_err_pp, gate))

    c1_min_at = min(aftertax_store["C1"]["CAGR_IS"], aftertax_store["C1"]["CAGR_OOS"])
    print("C1 min after-tax CAGR = %+.4f%% (spec baseline target ~+15.07%%)" % (c1_min_at * 100.0))
    print("C1 MaxDD (pretax) = %+.4f%% (spec baseline target ~-34.66%%)"
          % (pretax_store["C1"]["MaxDD_FULL"] * 100.0))

    # ---- Build CSV rows ----
    rows = []
    for cname in conditions:
        for tax in ("pretax", "aftertax"):
            m = pretax_store[cname] if tax == "pretax" else aftertax_store[cname]
            rows.append({
                "condition": cname,
                "tax": tax,
                "desc": conditions[cname]["desc"],
                "sigma_target": conditions[cname]["sigma_target"],
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
                "worst_calendar_year": extra[cname]["worst_year"],
                "worst_calendar_year_ret": extra[cname]["worst_year_ret"],
                "vol_scale_changes": extra[cname]["n_changes"],
                "vol_scale_min": extra[cname]["min_vs"],
                "vol_scale_mean": extra[cname]["mean_vs"],
                "repro_err_pp": (round(repro_err_pp, 4) if cname == "C1" else ""),
            })

    out_csv = os.path.join(_REPO, "audit_results", "p07_backtest_metrics_20260611.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
    print()
    print("Saved CSV: %s" % out_csv)

    # ---- Final ASCII table ----
    print()
    print("=" * 78)
    print("FINAL TABLE (after-tax CAGR; Sharpe/MaxDD pretax)")
    print("=" * 78)
    hdr = "%-9s | %10s | %10s | %10s | %7s | %8s | %9s" % (
        "cond", "CAGR_IS_at", "CAGR_OOS_at", "min_at", "Sharpe", "MaxDD", "Trades/yr")
    print(hdr)
    print("-" * len(hdr))
    for cname in ("C1", "P07_st20", "P07_st15", "P07_st25"):
        aft = aftertax_store[cname]
        pre = pretax_store[cname]
        min_at = min(aft["CAGR_IS"], aft["CAGR_OOS"])
        print("%-9s | %+9.4f%% | %+9.4f%% | %+9.4f%% | %+6.3f | %+7.4f | %9.2f" % (
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
        "C1_min_aftertax": round(c1_min_at, 4),
        "C1_repro_err_pp_vs_runoverlayV7": round(repro_err_pp, 4),
    }
    for cname in ("C1", "P07_st20", "P07_st15", "P07_st25"):
        pre = pretax_store[cname]
        aft = aftertax_store[cname]
        min_at = min(aft["CAGR_IS"], aft["CAGR_OOS"])
        block[cname] = {
            "CAGR_IS_pretax": round(pre["CAGR_IS"], 4),
            "CAGR_OOS_pretax": round(pre["CAGR_OOS"], 4),
            "CAGR_IS_aftertax": round(aft["CAGR_IS"], 4),
            "CAGR_OOS_aftertax": round(aft["CAGR_OOS"], 4),
            "min_aftertax": round(min_at, 4),
            "Sharpe_OOS": round(pre["Sharpe_OOS"], 4),
            "MaxDD_FULL": round(pre["MaxDD_FULL"], 4),
            "Worst10Y_star_aftertax": round(aft["Worst10Y_star"], 4),
            "P10_5Y_aftertax": round(aft["P10_5Y"], 4),
            "Trades_yr": round(pre["Trades_yr"], 2),
            "worst_calendar_year": extra[cname]["worst_year"],
            "worst_calendar_year_ret_pct": round(extra[cname]["worst_year_ret"] * 100, 2),
        }
    print(json.dumps(block, indent=2))
    return block


if __name__ == "__main__":
    main()
