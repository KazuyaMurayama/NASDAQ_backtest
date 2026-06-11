"""
src/audit/tqqq_wfa_recompute_20260611.py
========================================
GATE step requested by the user before restructuring CURRENT_BEST_STRATEGY.md:
recompute the WFA columns (WFE, CI95_lo, t_p, n_windows) on the **TQQQ cost
basis** for the four ETF-designated strategies whose point metrics were already
TQQQ-corrected in v6 but whose WFA columns are still flagged with "+" (CFD
basis, "TQQQ recompute pending"):

    DH-W1, V0, V7, P7

and report the TQQQ-basis WFA for the three candidates being added to the
canonical table:

    P09_TQQQ, LU1 (strong-map), LU2 (uniform x1.15)

For DH-W1/V0/V7/P7 we build the NAV on BOTH cost bases via the validated
cost_model skeleton (build_nav_overlay / build_nav_p7) and run WFA on each:
  - CFD branch  -> must reproduce the current "+" table values (sanity check),
  - TQQQ branch -> the new value that replaces the "+" entry.

For P09_TQQQ / LU1 / LU2 we reuse the validated TQQQ-basis builders from
run_p09_tqqq_validate (already TQQQ cost) and run WFA directly.

WFA engine: src.audit.unified_wfa.summarize_wfa over canonical g1_wfa windows
(identical to every other strategy in the repo). EVAL_STD gates:
  - beta : WFE in [0.5, 2.0]   (3.10 / 3.13 <=1.2 preferred)
  - alpha: CI95_lo > 0  AND  t_p < 0.05   (3.9)

ASCII-only prints (Windows cp932). Saves CSV; does NOT commit.
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

# ETF cost-correction NAV builders (validated skeleton)
from src.audit.tqqq_correction_etf_strategies_20260611 import (
    build_nav_overlay, build_nav_p7, _build_overlay_mult, OVERLAY_MAPS,
)
# WFA runner + TQQQ-basis candidate builders
from src.audit.run_p09_tqqq_validate_20260611 import (
    _run_wfa, _build_tqqq_base, _build_p09_on_base, _metrics_pack, _min_at,
    LU1_MAP, LU2_SCALE,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS, _ret_from_nav_level, _inverse_vol_weights,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)

# Current "+" (CFD-basis) WFA values in CURRENT_BEST_STRATEGY.md v6, for the
# reproduction sanity check.
CURRENT_PLUS = {
    "DH_W1": {"WFE": 0.996, "CI95_lo": 0.1369},
    "V0":    {"WFE": 1.043, "CI95_lo": 0.1257},
    "V7":    {"WFE": 0.975, "CI95_lo": 0.1409},
    "P7":    {"WFE": 1.042, "CI95_lo": 0.1657},
}

WFE_LO, WFE_HI = 0.5, 2.0


def _verdict(wfe, ci95_lo, t_p):
    beta = (WFE_LO <= wfe <= WFE_HI)
    alpha = (ci95_lo > 0.0) and (t_p < 0.05)
    return beta, alpha


def main():
    print("=" * 90)
    print("TQQQ-basis WFA RECOMPUTE (gate before canonical table restructure)  2026-06-11")
    print("=" * 90)

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
    mask = np.asarray(shared["mask"], float)
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / TRADING_DAYS

    rows = []

    # ---- 1) Four ETF rows: CFD repro + TQQQ recompute ----
    print("\n[A] ETF-designated strategies: CFD (repro check) vs TQQQ (new) WFA")
    print("-" * 90)
    print("%-7s | %-5s | %7s | %9s | %7s | %4s | %-6s | %-6s"
          % ("strat", "basis", "WFE", "CI95_lo", "t_p", "n", "beta", "alpha"))
    print("-" * 90)
    for strat in ["DH_W1", "V0", "V7", "P7"]:
        mult = _build_overlay_mult(dates_dt, OVERLAY_MAPS.get(strat))
        for basis in ("CFD", "TQQQ"):
            if strat == "P7":
                nav, tpy = build_nav_p7(close, dates, gold_2x, bond_3x, sofr,
                                        lev_raw_masked, wn, wg, wb, basis)
            else:
                nav, tpy = build_nav_overlay(close, dates, gold_2x, bond_3x, sofr,
                                             lev_raw_masked, wn, wg, wb, mult, basis)
            w = _run_wfa(nav, "%s_%s" % (strat, basis))
            beta, alpha = _verdict(w["WFE"], w["CI95_lo"], w["t_p"])
            print("%-7s | %-5s | %7.4f | %+8.4f%% | %7.4g | %4d | %-6s | %-6s"
                  % (strat, basis, w["WFE"], 100 * w["CI95_lo"], w["t_p"],
                     w["n_windows"], "PASS" if beta else "FAIL",
                     "PASS" if alpha else "FAIL"))
            rows.append(dict(strategy=strat, basis=basis, WFE=w["WFE"],
                             CI95_lo=w["CI95_lo"], t_p=w["t_p"],
                             n_windows=w["n_windows"], beta_pass=beta, alpha_pass=alpha))
        # repro delta (CFD branch vs current "+" table value)
        cfd_row = rows[-2]
        cur = CURRENT_PLUS[strat]
        d_wfe = cfd_row["WFE"] - cur["WFE"]
        d_ci = (cfd_row["CI95_lo"] - cur["CI95_lo"]) * 100.0
        print("        repro: CFD-branch vs table-'+' : dWFE=%+.4f  dCI95_lo=%+.4fpp  %s"
              % (d_wfe, d_ci, "OK" if (abs(d_wfe) <= 0.03 and abs(d_ci) <= 0.5) else "CHECK"))
        print("-" * 90)

    # ---- 2) Candidates added to the table: P09_TQQQ / LU1 / LU2 (TQQQ basis) ----
    print("\n[B] New candidates (TQQQ basis): P09_TQQQ / LU1 / LU2")
    print("-" * 90)

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), dtype=float)
    bond_1x = np.asarray(build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0), dtype=float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)
    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    candidates = [
        ("P09_TQQQ", None, 1.0),
        ("LU1", LU1_MAP, 1.0),
        ("LU2", None, LU2_SCALE),
    ]
    print("%-9s | %7s | %9s | %7s | %4s | %-6s | %-6s"
          % ("cand", "WFE", "CI95_lo", "t_p", "n", "beta", "alpha"))
    print("-" * 90)
    for label, v7map, scale in candidates:
        base_nav, r_base, tpy_base = _build_tqqq_base(
            shared, dates_dt, v7_map=v7map, lev_scale=scale)
        nav, r_c, tpy = _build_p09_on_base(
            r_base, ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on,
            dates_dt, tpy_base, n_years)
        w = _run_wfa(nav, label)
        beta, alpha = _verdict(w["WFE"], w["CI95_lo"], w["t_p"])
        print("%-9s | %7.4f | %+8.4f%% | %7.4g | %4d | %-6s | %-6s"
              % (label, w["WFE"], 100 * w["CI95_lo"], w["t_p"], w["n_windows"],
                 "PASS" if beta else "FAIL", "PASS" if alpha else "FAIL"))
        rows.append(dict(strategy=label, basis="TQQQ", WFE=w["WFE"],
                         CI95_lo=w["CI95_lo"], t_p=w["t_p"],
                         n_windows=w["n_windows"], beta_pass=beta, alpha_pass=alpha))

    out_csv = os.path.join(_REPO_DIR, "audit_results", "tqqq_wfa_recompute_20260611.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
    print("\nSaved CSV: %s" % out_csv)

    # ---- 3) Gate verdict ----
    tqqq_rows = [r for r in rows if r["basis"] == "TQQQ"]
    all_beta = all(r["beta_pass"] for r in tqqq_rows
                   if r["strategy"] in ("DH_W1", "V0", "V7", "P7"))
    print("\n" + "=" * 90)
    print("GATE: TQQQ-basis WFA for the 4 ETF rows")
    print("  all WFE in [0.5,2.0] (beta) : %s" % ("PASS" if all_beta else "FAIL"))
    print("=" * 90)

    block = {r["strategy"] + "_" + r["basis"]: {
        "WFE": round(float(r["WFE"]), 4),
        "CI95_lo_pct": round(100 * float(r["CI95_lo"]), 4),
        "t_p": round(float(r["t_p"]), 6),
        "n": int(r["n_windows"]),
        "beta": "PASS" if r["beta_pass"] else "FAIL",
        "alpha": "PASS" if r["alpha_pass"] else "FAIL",
    } for r in rows}
    print("\nRETURN_BLOCK")
    print(json.dumps(block, indent=2))
    return block


if __name__ == "__main__":
    main()
