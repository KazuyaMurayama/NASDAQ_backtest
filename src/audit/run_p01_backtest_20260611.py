"""
src/audit/run_p01_backtest_20260611.py
======================================
P01 / P13 backtest: fill DH-W1 OUT (cash) periods with a 1x Gold/Bond blend.

Idea
----
DH-W1's OUT periods (W1 hold mask == 0, ~47% of days) sit in cash at 0%.
P01 fills that idle cash with a 1x Gold/Bond blend (cheap funds, no leverage
financing). Goal: raise min(IS, OOS) CAGR without worsening MaxDD.

Conditions
----------
  C1        Baseline V7 (= run_overlay('V7','realistic')), unchanged.
  P13       OUT fill fixed 50/50 (w_g=w_b=0.5 always, T+5 lag + TER). Benchmark.
  P01       OUT fill inverse-vol W=63 (main).
  P01_w21   sensitivity (inverse-vol W=21).
  P01_w126  sensitivity (inverse-vol W=126).

Construction (P01 changes OUT days only; IN days = baseline unchanged)
  - out_mask[t] = (mask[t] < 0.5).
  - T+5 fund execution lag: fund_active[5:] = out_mask[:-5].
  - Realized-vol weights (inverse-vol / risk parity):
      sigma_g[t] = rolling std of ret_gold over window W (annualize *sqrt(252)).
      sigma_b[t] likewise for ret_bond. Default W=63 (also W=21, W=126).
      w_g[t] = (1/sigma_g)/((1/sigma_g)+(1/sigma_b)); w_b = 1 - w_g.
      Clamp w_g to [0.25, 0.75]. Update weekly (every 5 BD, hold between).
      Warm-start: forward-fill weights; before first valid window use 0.5/0.5.
  - Blend daily return on fund-active days:
      r_blend[t] = w_g[t]*ret_gold[t] + w_b[t]*ret_bond[t]
                   - (w_g[t]*0.001838 + w_b[t]*0.00154)/252.
  - r_p01[t] = r_blend[t] if fund_active[t] else r_base[t].
  - nav_p01 = (1 + clip(r_p01, -0.999, None)).cumprod().

Premises
  - After-tax = pretax * 0.8273 for CAGR/Worst10Y/P10. Sharpe/MaxDD pretax.
  - Print each condition's calendar-2022 return and its worst single
    calendar-year return.
  - Trades/yr: baseline trades + fund-sleeve OUT<->IN transitions / n_years.

ASCII-only prints (Windows console cp932).
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# multitasking stub + sys.path (mirror strategy_runners.py)
# ---------------------------------------------------------------------------
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
from src.audit.strategy_runners import run_overlay, _load_dhw1_shared
from src.audit.unified_metrics import compute_10metrics

# ---------------------------------------------------------------------------
# Constants (spec)
# ---------------------------------------------------------------------------
AFTER_TAX = 0.8273           # pretax -> aftertax multiplier (CAGR/Worst10Y/P10)
FEE_GOLD = 0.001838          # GOLD1X.ter (per src/product_costs.py)
FEE_BOND = 0.00154           # BOND1X.ter
LAG_DAYS = 5                 # EXEC_LAG_FUND_1X = 5 (T+5)
WG_CLAMP = (0.25, 0.75)      # inverse-vol weight clamp
WEIGHT_UPDATE_BD = 5         # recompute weights every 5 business days (weekly)
TRADING_DAYS = 252

# After-tax-scaled metric keys (CAGR/Worst10Y/P10 family)
_AFTERTAX_KEYS = {"CAGR_IS", "CAGR_OOS", "CAGR_FULL", "Worst10Y_star", "P10_5Y", "Worst5Y"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ret_from_nav_level(level: np.ndarray) -> np.ndarray:
    """diff/prev (nan->0) for a 1x total-return NAV level series."""
    level = np.asarray(level, dtype=float)
    out = np.zeros_like(level)
    out[1:] = np.diff(level) / level[:-1]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_sigma(ret: np.ndarray, window: int) -> np.ndarray:
    """Rolling std (ddof=1) of ret over `window`, annualized * sqrt(252).

    Before the first full window, value is NaN.
    """
    s = pd.Series(ret)
    sig = s.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return sig.values


def _inverse_vol_weights(ret_gold, ret_bond, window):
    """Inverse-vol (risk-parity) weekly weights with clamp + warm-start.

    Returns (w_g, w_b) arrays length n.
      - sigma_g/sigma_b from rolling std (annualized).
      - w_g_raw = (1/sig_g) / ((1/sig_g)+(1/sig_b)); clamp to [0.25, 0.75].
      - Recompute only every WEIGHT_UPDATE_BD days; hold (ffill) between.
      - Warm-start: forward-fill; before first valid window use 0.5/0.5.
    """
    n = len(ret_gold)
    sig_g = _rolling_sigma(ret_gold, window)
    sig_b = _rolling_sigma(ret_bond, window)

    lo, hi = WG_CLAMP
    w_g = np.full(n, np.nan)
    last = 0.5  # warm-start default before first valid window
    for t in range(n):
        if t % WEIGHT_UPDATE_BD == 0:
            sg, sb = sig_g[t], sig_b[t]
            if np.isfinite(sg) and np.isfinite(sb) and sg > 0 and sb > 0:
                inv_g = 1.0 / sg
                inv_b = 1.0 / sb
                wg = inv_g / (inv_g + inv_b)
                wg = float(np.clip(wg, lo, hi))
                last = wg
            # else: keep previous `last` (warm-start / ffill)
        w_g[t] = last
    w_b = 1.0 - w_g
    return w_g, w_b


def _build_p01_nav(r_base, ret_gold, ret_bond, fund_active, w_g, w_b):
    """Construct P01-style NAV.

    r_blend on fund-active days; r_base otherwise. clip(r, -0.999, None).
    """
    fee_daily = (w_g * FEE_GOLD + w_b * FEE_BOND) / TRADING_DAYS
    r_blend = w_g * ret_gold + w_b * ret_bond - fee_daily
    r_p01 = np.where(fund_active, r_blend, r_base)
    r_p01 = np.clip(r_p01, -0.999, None)
    nav = np.cumprod(1.0 + r_p01)
    return nav, r_p01


def _calendar_year_returns(nav_dt: pd.Series) -> pd.Series:
    """Calendar-year total returns from a daily NAV (DatetimeIndex)."""
    yr_end = nav_dt.groupby(nav_dt.index.year).last()
    yr_start = nav_dt.groupby(nav_dt.index.year).first()
    # Use prior year-end as the base for continuity; first year uses its own first obs.
    prev_end = yr_end.shift(1)
    base = prev_end.copy()
    base.iloc[0] = yr_start.iloc[0]
    cy_ret = yr_end / base - 1.0
    return cy_ret


def _apply_aftertax(metrics: dict) -> dict:
    """Return a copy with CAGR/Worst10Y/P10 family * AFTER_TAX; others unchanged."""
    out = {}
    for k, v in metrics.items():
        if k in _AFTERTAX_KEYS and v is not None and np.isfinite(v):
            out[k] = v * AFTER_TAX
        else:
            out[k] = v
    return out


def _count_fund_transitions(fund_active: np.ndarray) -> int:
    """Number of OUT<->IN sleeve transitions (fund_active flips)."""
    fa = fund_active.astype(int)
    return int(np.sum(fa[1:] != fa[:-1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("P01 / P13 backtest: fill DH-W1 OUT periods with 1x Gold/Bond blend")
    print("=" * 78)

    # --- Load shared DH-W1 assets + mask ---
    _load_dhw1_shared()
    a = sr._DHW1_SHARED["assets"]
    mask = np.asarray(sr._DHW1_SHARED["mask"], dtype=float)  # 1.0=IN, 0.0=OUT
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / TRADING_DAYS

    # --- Baseline V7 (realistic) ---
    base = run_overlay("V7", "realistic")
    nav_base = base["nav"]
    r_base = nav_base.pct_change().fillna(0).values
    trades_base = base["trades_per_year"]

    # --- 1x asset return series (Gold / Bond, unhedged total-return NAV) ---
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected

    gold_1x = np.asarray(prepare_gold_local(dates), dtype=float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0),
        dtype=float,
    )
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    # --- OUT mask + T+5 fund execution lag ---
    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    n_fund_days = int(fund_active.sum())
    fund_transitions = _count_fund_transitions(fund_active)
    sleeve_trades_yr = fund_transitions / n_years

    print("")
    print("Setup:")
    print("  n_days            = %d  (%.1f yrs)" % (n, n_years))
    print("  OUT days (mask<0.5)= %d  (%.1f%%)" % (int(out_mask.sum()), 100.0 * out_mask.mean()))
    print("  fund-active days   = %d  (%.1f%%, after T+5 lag)"
          % (n_fund_days, 100.0 * n_fund_days / n))
    print("  baseline trades/yr = %.1f" % trades_base)
    print("  sleeve transitions = %d  (~%.1f /yr added)" % (fund_transitions, sleeve_trades_yr))
    print("")

    # --- Validation: reconstruction of baseline on IN days == baseline ---
    # P01 keeps r_base on non-fund-active days, so nav must match baseline exactly
    # when fund sleeve is disabled. Verify by rebuilding with all-zero fund_active.
    nav_check, _ = _build_p01_nav(
        r_base, ret_gold, ret_bond, np.zeros(n, dtype=bool),
        np.full(n, 0.5), np.full(n, 0.5),
    )
    # NOTE: baseline nav[0] != 1.0 (first bar already carries a return), whereas the
    # reconstruction starts at 1.0 after pct_change().fillna(0). That produces a constant
    # ~1e-5 LEVEL offset which is invariant for every metric (CAGR ratios, MaxDD, returns).
    # Normalize both to their first value to confirm PATH identity (the load-bearing check).
    rebased_check = nav_check / nav_check[0]
    rebased_base = nav_base.values / nav_base.values[0]
    max_rel_err = float(np.max(np.abs(rebased_check - rebased_base) / np.abs(rebased_base)))
    raw_offset = float(np.max(np.abs(nav_check - nav_base.values) / np.abs(nav_base.values)))
    print("Baseline reconstruction check (fund disabled):")
    print("  path max_rel_err (rebased) = %.3e -> %s"
          % (max_rel_err, "PASS (path matches baseline)" if max_rel_err < 1e-9 else "WARN"))
    print("  raw level offset           = %.3e (constant nav[0] convention, metric-invariant)"
          % raw_offset)
    print("")

    # --- Build condition NAVs ---
    conditions = {}

    # C1 Baseline (use run_overlay output directly)
    conditions["C1_baseline_V7"] = nav_base

    # P13 fixed 50/50
    wg13 = np.full(n, 0.5)
    wb13 = np.full(n, 0.5)
    nav_p13, _ = _build_p01_nav(r_base, ret_gold, ret_bond, fund_active, wg13, wb13)
    conditions["P13_fixed_50_50"] = pd.Series(nav_p13, index=dates_dt)

    # P01 inverse-vol windows
    for label, W in (("P01_w63", 63), ("P01_w21", 21), ("P01_w126", 126)):
        wg, wb = _inverse_vol_weights(ret_gold, ret_bond, W)
        nav_p, _ = _build_p01_nav(r_base, ret_gold, ret_bond, fund_active, wg, wb)
        conditions[label] = pd.Series(nav_p, index=dates_dt)

    # --- Trades/yr per condition ---
    trades_yr = {}
    trades_yr["C1_baseline_V7"] = trades_base
    for label in ("P13_fixed_50_50", "P01_w63", "P01_w21", "P01_w126"):
        trades_yr[label] = trades_base + sleeve_trades_yr

    # --- Compute metrics (pretax + aftertax) + calendar-year stats ---
    rows = []
    summary = {}
    for label, nav_dt in conditions.items():
        nav_dt = pd.Series(nav_dt.values, index=dates_dt)  # ensure clean index
        tpy = trades_yr[label]
        m_pre = compute_10metrics(nav_dt, tpy)
        m_aft = _apply_aftertax(m_pre)

        cy = _calendar_year_returns(nav_dt)
        cy2022 = float(cy.get(2022, np.nan))
        worst_cy = float(cy.min())
        worst_cy_year = int(cy.idxmin())

        for tax_label, m in (("pretax", m_pre), ("aftertax", m_aft)):
            row = {
                "condition": label,
                "tax": tax_label,
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
                "cy2022_return": cy2022,
                "worst_calendar_year_return": worst_cy,
                "worst_calendar_year": worst_cy_year,
            }
            rows.append(row)

        summary[label] = {
            "pre": m_pre, "aft": m_aft,
            "cy2022": cy2022, "worst_cy": worst_cy, "worst_cy_year": worst_cy_year,
        }

    # --- Save CSV ---
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "p01_backtest_metrics_20260611.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print("Saved CSV: %s" % csv_path)
    print("")

    # --- ASCII table (after-tax CAGR view) ---
    print("=" * 110)
    print("RESULTS (CAGR_IS_at / CAGR_OOS_at / min_at are AFTER-TAX; Sharpe/MaxDD/cy pretax)")
    print("=" * 110)
    hdr = ("%-16s | %10s | %10s | %10s | %7s | %8s | %8s | %8s | %8s"
           % ("condition", "CAGR_IS_at", "CAGR_OOS_at", "min_at",
              "Sharpe", "MaxDD", "cy2022", "worstCY", "Trades/yr"))
    print(hdr)
    print("-" * 110)
    order = ["C1_baseline_V7", "P13_fixed_50_50", "P01_w63", "P01_w21", "P01_w126"]
    for label in order:
        s = summary[label]
        cagr_is_at = s["aft"]["CAGR_IS"]
        cagr_oos_at = s["aft"]["CAGR_OOS"]
        min_at = min(cagr_is_at, cagr_oos_at)
        print("%-16s | %+9.2f%% | %+9.2f%% | %+9.2f%% | %7.3f | %+7.2f%% | %+7.2f%% | %+7.2f%% (%d) | %8.1f"
              % (label,
                 100 * cagr_is_at, 100 * cagr_oos_at, 100 * min_at,
                 s["pre"]["Sharpe_OOS"], 100 * s["pre"]["MaxDD_FULL"],
                 100 * s["cy2022"], 100 * s["worst_cy"], s["worst_cy_year"],
                 s["aft"]["Trades_yr"]))
    print("=" * 110)
    print("")

    # --- Machine-readable block per condition (final return) ---
    print("JSON-like block per condition (CAGR/Worst10Y/P10 = fractions):")
    base_min_aft = min(summary["C1_baseline_V7"]["aft"]["CAGR_IS"],
                       summary["C1_baseline_V7"]["aft"]["CAGR_OOS"])
    for label in order:
        s = summary[label]
        min_aft = min(s["aft"]["CAGR_IS"], s["aft"]["CAGR_OOS"])
        print("  %s = {" % label)
        print("    CAGR_IS_pretax: %.6f, CAGR_OOS_pretax: %.6f,"
              % (s["pre"]["CAGR_IS"], s["pre"]["CAGR_OOS"]))
        print("    CAGR_IS_aftertax: %.6f, CAGR_OOS_aftertax: %.6f, min_aftertax: %.6f,"
              % (s["aft"]["CAGR_IS"], s["aft"]["CAGR_OOS"], min_aft))
        print("    Sharpe_OOS: %.4f, MaxDD_FULL: %.6f,"
              % (s["pre"]["Sharpe_OOS"], s["pre"]["MaxDD_FULL"]))
        print("    Worst10Y_star_aftertax: %.6f, P10_5Y_aftertax: %.6f,"
              % (s["aft"]["Worst10Y_star"], s["aft"]["P10_5Y"]))
        print("    cy2022_return: %.6f, worst_calendar_year: %d (%.6f), Trades_yr: %.2f"
              % (s["cy2022"], s["worst_cy_year"], s["worst_cy"], s["aft"]["Trades_yr"]))
        print("  }")
    print("")
    print("  baseline_C1_min_aftertax_CAGR = %.6f" % base_min_aft)
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
