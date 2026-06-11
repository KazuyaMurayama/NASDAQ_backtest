"""
src/audit/run_p02_p09_backtest_20260611.py
==========================================
Conditional OUT-fill backtest: does a regime-gated retreat-to-cash rescue P01?

Context
-------
P01 (fill DH-W1's OUT cash with 1x inverse-vol Gold/Bond) was REJECTED:
it RAISED IS CAGR (+15.07 -> +19.15%) but LOWERED OOS to +14.34%, driven
entirely by 2022 (stock+bond+gold all down): baseline OUT cash = -0.27%,
P01 fill = -14%. The idle OUT cash acts as a "triple-down hedge".

Hypothesis
----------
A CONDITIONAL fill that retreats to cash during all-asset-down regimes can
rescue P01: keep the IS fill benefit in normal pullbacks but dodge the 2022
triple-down. We test sign-gated retreats (P02a/b/c) and a bond-timing gate
(P09), and check whether min(IS,OOS) after-tax CAGR clears baseline +15.07%
WITHOUT worsening MaxDD (<= -34.66%) and WITHOUT materially worse 2022.

Conditions
----------
  C1    Baseline V7 (cash on OUT).                       [reference]
  P01   OUT fill inverse-vol W63 (known fail).           [reference]
  P02a  Sign gate AND: on OUT days, if (gold_mom63<0 AND bond_mom63<0)
        -> CASH (r=0); else -> P01 inverse-vol fill.  Gate DELAY=2.
  P02b  Sign gate OR: gate->cash if (gold_mom63<0 OR bond_mom63<0).
  P02c  Partial: on gate days hold 50% fill + 50% cash.
  P09   Bond-timing: OUT fill = Gold1x always; Bond1x only when
        bond_mom252>0 (else that bond portion -> cash). Inverse-vol within
        the active legs. Gate DELAY=2 on bond_mom252.

Gate DELAY=2 = the signal is shifted +2 BD before use (causal; signals are
backward-looking log-momentum, no look-ahead, plus a 2-day execution buffer).

Keeper criterion
----------------
A condition is a KEEPER ONLY IF:
  min(IS,OOS) after-tax CAGR > +15.07% (baseline)  AND
  MaxDD_FULL >= -34.66% (i.e. not worse / less negative)  AND
  cy2022_return not materially worse than ~-3%.

Diagnostics per conditional condition: gate fire % of OUT-days, cy2022,
IS vs OOS after-tax CAGR, min(IS,OOS) vs baseline.

ASCII-only prints (Windows console cp932). Reuses baseline + fill machinery
from run_p01_backtest_20260611.py.
"""

from __future__ import annotations

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
from src.audit.strategy_runners import run_overlay, _load_dhw1_shared
from src.audit.unified_metrics import compute_10metrics

# Reuse the EXACT baseline + fill machinery from the P01 harness.
from src.audit.run_p01_backtest_20260611 import (
    AFTER_TAX, FEE_GOLD, FEE_BOND, LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights, _build_p01_nav,
    _calendar_year_returns, _apply_aftertax, _count_fund_transitions,
)

GATE_DELAY = 2          # business-day shift applied to gate signals (causal buffer)
MACRO_CSV = os.path.join(_REPO_DIR, "data", "macro_features.csv")


# ---------------------------------------------------------------------------
# Signal alignment
# ---------------------------------------------------------------------------
def _load_macro_signal(dates: pd.Series, colname: str, delay: int) -> np.ndarray:
    """Load one macro_features column, align to strategy `dates` (reindex+ffill),
    then shift forward by `delay` business days (causal execution buffer).

    Signals in macro_features.csv are backward-looking (log_p - log_p.shift(N)),
    so they carry no look-ahead; the +delay shift only adds a lag, never removes one.
    Returns a float array aligned 1:1 with `dates`.
    """
    raw = pd.read_csv(MACRO_CSV, usecols=["date", colname])
    raw["date"] = pd.to_datetime(raw["date"])
    s = raw.set_index("date")[colname].sort_index()

    strat_idx = pd.DatetimeIndex(pd.to_datetime(dates.values))
    # reindex onto strategy dates with forward-fill (last known publication value)
    aligned = s.reindex(s.index.union(strat_idx)).ffill().reindex(strat_idx)
    arr = aligned.values.astype(float)
    # shift forward by `delay` BD: value known at t-delay is acted on at t.
    if delay > 0:
        shifted = np.full_like(arr, np.nan)
        shifted[delay:] = arr[:-delay]
        arr = shifted
    return arr


# ---------------------------------------------------------------------------
# Conditional NAV builders
# ---------------------------------------------------------------------------
def _build_conditional_nav(r_base, ret_gold, ret_bond, fund_active,
                           w_g, w_b, gate_to_cash, cash_fraction=1.0):
    """P01-style fill, but on fund-active days where gate_to_cash is True,
    replace `cash_fraction` of the fill with cash (return 0).

    gate_to_cash : bool array (True => retreat). NaN-gate days treated as False
                   (no retreat) so warm-up days behave like plain P01.
    cash_fraction: 1.0 = full cash on gate days; 0.5 = half fill / half cash.

    Returns (nav, r, eff_active) where eff_active marks days the fund sleeve is
    *materially* invested (fund_active AND not a full cash-retreat), used for the
    OUT-day trade count.
    """
    fee_daily = (w_g * FEE_GOLD + w_b * FEE_BOND) / TRADING_DAYS
    r_blend = w_g * ret_gold + w_b * ret_bond - fee_daily

    gate = np.asarray(gate_to_cash, dtype=bool)
    # On gate+fund days: hold (1-cash_fraction) of the blend, rest cash (0).
    r_fill = np.where(fund_active & gate, (1.0 - cash_fraction) * r_blend, r_blend)
    r = np.where(fund_active, r_fill, r_base)
    r = np.clip(r, -0.999, None)
    nav = np.cumprod(1.0 + r)

    # effective active = fund day that is not a FULL cash retreat
    full_retreat = gate & (cash_fraction >= 0.999)
    eff_active = fund_active & (~full_retreat)
    return nav, r, eff_active


def _build_p09_nav(r_base, ret_gold, ret_bond, fund_active, w_g, w_b, bond_on):
    """P09 bond-timing fill.

    OUT fill always holds the Gold leg at its inverse-vol weight w_g. The Bond
    leg (weight w_b) is held only when bond_on[t] (bond_mom252>0); otherwise that
    bond portion sits in cash (return 0). Inverse-vol weights are computed over
    both legs as usual; we simply zero out the bond contribution when bond is off.
    Fees are charged only on the legs actually held.

    Returns (nav, r, eff_active). eff_active = fund_active (Gold always invested).
    """
    bond_on = np.asarray(bond_on, dtype=bool)
    w_b_eff = np.where(bond_on, w_b, 0.0)          # bond weight only when timing on
    fee_daily = (w_g * FEE_GOLD + w_b_eff * FEE_BOND) / TRADING_DAYS
    r_blend = w_g * ret_gold + w_b_eff * ret_bond - fee_daily   # bond-off => cash on w_b
    r = np.where(fund_active, r_blend, r_base)
    r = np.clip(r, -0.999, None)
    nav = np.cumprod(1.0 + r)
    eff_active = fund_active.copy()  # gold always invested on OUT days
    return nav, r, eff_active


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("P02/P09 conditional OUT-fill: does regime-gated cash-retreat rescue P01?")
    print("=" * 78)

    _load_dhw1_shared()
    a = sr._DHW1_SHARED["assets"]
    mask = np.asarray(sr._DHW1_SHARED["mask"], dtype=float)  # 1=IN, 0=OUT
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / TRADING_DAYS

    base = run_overlay("V7", "realistic")
    nav_base = base["nav"]
    r_base = nav_base.pct_change().fillna(0).values
    trades_base = base["trades_per_year"]

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected

    gold_1x = np.asarray(prepare_gold_local(dates), dtype=float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0),
        dtype=float,
    )
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    n_fund_days = int(fund_active.sum())

    # Inverse-vol weights W=63 (the P01 main config)
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)

    # --- Gate signals (causal, +GATE_DELAY shift) ---
    gold_m63 = _load_macro_signal(dates, "gold_mom63", GATE_DELAY)
    bond_m63 = _load_macro_signal(dates, "bond_mom63", GATE_DELAY)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)

    # NaN signal -> treat as "not triggered" (no retreat / bond on by default? )
    # For retreat gates: NaN => do NOT retreat (act like plain P01) -> nan_to False.
    g_neg = np.where(np.isnan(gold_m63), False, gold_m63 < 0)
    b_neg = np.where(np.isnan(bond_m63), False, bond_m63 < 0)
    gate_and = g_neg & b_neg                    # P02a: both down
    gate_or = g_neg | b_neg                     # P02b: either down
    # P09 bond-on: bond_mom252>0. NaN => treat as bond OFF (conservative -> cash).
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    print("")
    print("Setup:")
    print("  n_days             = %d  (%.1f yrs)" % (n, n_years))
    print("  OUT days (mask<0.5) = %d  (%.1f%%)" % (int(out_mask.sum()), 100.0 * out_mask.mean()))
    print("  fund-active days    = %d  (%.1f%%, after T+5 lag)"
          % (n_fund_days, 100.0 * n_fund_days / n))
    print("  baseline trades/yr  = %.1f" % trades_base)
    print("  GATE_DELAY          = %d BD" % GATE_DELAY)
    print("")

    # --- Build conditions ---
    conditions = {}
    eff_active_map = {}

    # C1 baseline
    conditions["C1_baseline_V7"] = nav_base
    eff_active_map["C1_baseline_V7"] = np.zeros(n, dtype=bool)

    # P01 reference (full inverse-vol fill, no gate)
    nav_p01, _, ea01 = _build_conditional_nav(
        r_base, ret_gold, ret_bond, fund_active, wg, wb,
        np.zeros(n, dtype=bool), cash_fraction=1.0)
    conditions["P01_w63"] = pd.Series(nav_p01, index=dates_dt)
    eff_active_map["P01_w63"] = ea01

    # P02a sign gate AND -> full cash
    nav_a, _, ea_a = _build_conditional_nav(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, gate_and, cash_fraction=1.0)
    conditions["P02a_AND_cash"] = pd.Series(nav_a, index=dates_dt)
    eff_active_map["P02a_AND_cash"] = ea_a

    # P02b sign gate OR -> full cash
    nav_b, _, ea_b = _build_conditional_nav(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, gate_or, cash_fraction=1.0)
    conditions["P02b_OR_cash"] = pd.Series(nav_b, index=dates_dt)
    eff_active_map["P02b_OR_cash"] = ea_b

    # P02c sign gate AND -> 50% fill / 50% cash
    nav_c, _, ea_c = _build_conditional_nav(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, gate_and, cash_fraction=0.5)
    conditions["P02c_AND_half"] = pd.Series(nav_c, index=dates_dt)
    eff_active_map["P02c_AND_half"] = ea_c

    # P09 bond-timing
    nav_9, _, ea_9 = _build_p09_nav(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on)
    conditions["P09_bondtiming"] = pd.Series(nav_9, index=dates_dt)
    eff_active_map["P09_bondtiming"] = ea_9

    # --- Gate-fire diagnostics (% of OUT/fund-active days the gate retreats) ---
    gate_map = {
        "P02a_AND_cash": gate_and,
        "P02b_OR_cash": gate_or,
        "P02c_AND_half": gate_and,
        "P09_bondtiming": ~bond_on,   # "retreat" = bond portion in cash
    }
    gate_fire_pct = {}
    for label, gate in gate_map.items():
        fired_on_out = int(np.sum(fund_active & np.asarray(gate, dtype=bool)))
        gate_fire_pct[label] = 100.0 * fired_on_out / max(n_fund_days, 1)
    gate_fire_pct["C1_baseline_V7"] = float("nan")
    gate_fire_pct["P01_w63"] = 0.0

    # --- Trades/yr per condition (baseline + effective OUT<->IN sleeve flips) ---
    trades_yr = {"C1_baseline_V7": trades_base}
    for label, ea in eff_active_map.items():
        if label == "C1_baseline_V7":
            continue
        flips = _count_fund_transitions(ea)
        trades_yr[label] = trades_base + flips / n_years

    # --- Metrics (pretax + aftertax) + calendar-year stats ---
    rows = []
    summary = {}
    for label, nav_dt in conditions.items():
        nav_dt = pd.Series(nav_dt.values, index=dates_dt)
        tpy = trades_yr[label]
        m_pre = compute_10metrics(nav_dt, tpy)
        m_aft = _apply_aftertax(m_pre)

        cy = _calendar_year_returns(nav_dt)
        cy2022 = float(cy.get(2022, np.nan))
        worst_cy = float(cy.min())
        worst_cy_year = int(cy.idxmin())

        for tax_label, m in (("pretax", m_pre), ("aftertax", m_aft)):
            rows.append({
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
                "gate_fire_pct_of_outdays": gate_fire_pct.get(label, float("nan")),
            })

        summary[label] = {
            "pre": m_pre, "aft": m_aft,
            "cy2022": cy2022, "worst_cy": worst_cy, "worst_cy_year": worst_cy_year,
            "gate_pct": gate_fire_pct.get(label, float("nan")),
        }

    # --- Save CSV ---
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "p02_p09_backtest_metrics_20260611.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    print("Saved CSV: %s" % csv_path)
    print("")

    # --- Baseline keeper thresholds ---
    base_min_aft = min(summary["C1_baseline_V7"]["aft"]["CAGR_IS"],
                       summary["C1_baseline_V7"]["aft"]["CAGR_OOS"])
    base_maxdd = summary["C1_baseline_V7"]["pre"]["MaxDD_FULL"]
    THRESH_MIN = 0.1507     # baseline min(IS,OOS) after-tax CAGR
    THRESH_DD = -0.3466     # baseline MaxDD (keeper must be >= this)
    THRESH_CY2022 = -0.03   # ~ -3% tolerance

    order = ["C1_baseline_V7", "P01_w63", "P02a_AND_cash", "P02b_OR_cash",
             "P02c_AND_half", "P09_bondtiming"]

    # --- ASCII table ---
    print("=" * 120)
    print("RESULTS (CAGR_IS_at / CAGR_OOS_at / min_at AFTER-TAX; Sharpe/MaxDD/cy2022 pretax)")
    print("baseline C1: min_at=+15.07%%, MaxDD=-34.66%%, cy2022=-0.27%%")
    print("=" * 120)
    hdr = ("%-16s | %10s | %10s | %9s | %7s | %8s | %8s | %6s | %8s"
           % ("condition", "CAGR_IS_at", "CAGR_OOS_at", "min_at",
              "Sharpe", "MaxDD", "cy2022", "gate%", "Trades/yr"))
    print(hdr)
    print("-" * 120)
    for label in order:
        s = summary[label]
        cis = s["aft"]["CAGR_IS"]
        coos = s["aft"]["CAGR_OOS"]
        mn = min(cis, coos)
        gp = s["gate_pct"]
        gp_s = "  n/a" if (gp != gp) else ("%5.1f" % gp)
        print("%-16s | %+9.2f%% | %+9.2f%% | %+8.2f%% | %7.3f | %+7.2f%% | %+7.2f%% | %s | %8.1f"
              % (label, 100 * cis, 100 * coos, 100 * mn,
                 s["pre"]["Sharpe_OOS"], 100 * s["pre"]["MaxDD_FULL"],
                 100 * s["cy2022"], gp_s, s["aft"]["Trades_yr"]))
    print("=" * 120)
    print("")

    # --- Keeper evaluation ---
    print("KEEPER test: min_at > +15.07%% AND MaxDD >= -34.66%% AND cy2022 >= ~-3%%")
    print("-" * 78)
    any_keeper = False
    for label in order:
        if label == "C1_baseline_V7":
            continue
        s = summary[label]
        mn = min(s["aft"]["CAGR_IS"], s["aft"]["CAGR_OOS"])
        dd = s["pre"]["MaxDD_FULL"]
        cy = s["cy2022"]
        c_min = mn > THRESH_MIN
        c_dd = dd >= THRESH_DD
        c_cy = cy >= THRESH_CY2022
        keeper = c_min and c_dd and c_cy
        any_keeper = any_keeper or keeper
        print("  %-16s min=%+.4f(%s) maxdd=%+.4f(%s) cy2022=%+.4f(%s) -> %s"
              % (label, mn, "OK" if c_min else "x",
                 dd, "OK" if c_dd else "x",
                 cy, "OK" if c_cy else "x",
                 "*** KEEPER ***" if keeper else "reject"))
    print("-" * 78)
    print("  ANY KEEPER: %s" % ("YES" if any_keeper else "NO"))
    print("")

    # --- JSON-like block (final return) ---
    print("JSON-like block per condition:")
    for label in order:
        s = summary[label]
        mn = min(s["aft"]["CAGR_IS"], s["aft"]["CAGR_OOS"])
        gp = s["gate_pct"]
        print("  %s = {" % label)
        print("    CAGR_IS_aftertax: %.6f, CAGR_OOS_aftertax: %.6f, min_aftertax: %.6f,"
              % (s["aft"]["CAGR_IS"], s["aft"]["CAGR_OOS"], mn))
        print("    Sharpe_OOS: %.4f, MaxDD_FULL: %.6f,"
              % (s["pre"]["Sharpe_OOS"], s["pre"]["MaxDD_FULL"]))
        print("    cy2022_return: %.6f, gate_fire_pct_of_outdays: %s, Trades_yr: %.2f"
              % (s["cy2022"], ("nan" if gp != gp else "%.4f" % gp), s["aft"]["Trades_yr"]))
        print("  }")
    print("")
    print("  baseline_C1_min_aftertax_CAGR = %.6f" % base_min_aft)
    print("  baseline_C1_MaxDD_FULL        = %.6f" % base_maxdd)
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
