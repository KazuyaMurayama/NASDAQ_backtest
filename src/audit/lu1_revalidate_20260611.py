"""
src/audit/lu1_revalidate_20260611.py
====================================
QC follow-up to run_p09_tqqq_validate_20260611.py.

The original validation ran WFA + block bootstrap on P09_TQQQ (min +17.51%), NOT
on LU1 (= P09_TQQQ on a stronger IN boost map {0:1.40,1:1.20,2:1.05,3:1.00},
min +18.12%). LU1's extra +0.61pp was never validated, and the strong-boost map
was hardcoded with no grid/CV. This script fixes both:

  Task 1 : run the SAME _run_wfa + _block_bootstrap_compare on lu1_nav vs the SAME
           TQQQ-V7 baseline used for P09_TQQQ. Report LU1's OWN robustness.
  Task 2 : sweep a small pre-defined family of monotone boost maps on the SAME
           P09_TQQQ base. Report each map's after-tax min(IS,OOS), CAGR_IS/OOS,
           MaxDD, worst calendar year -> is M_lu1 an isolated peak or one of many?
           Does the edge come from SHAPE (boosting low-mom quartiles) or just
           more LEVERAGE (uniform maps)?

Premises: canonical split IS<=2021-05-07 / OOS>=2021-05-08, DELAY=2, T+5 fund lag,
after-tax x0.8273 (CAGR), Sharpe/MaxDD pretax, 252-day. All machinery reused from
run_p09_tqqq_validate_20260611.py (no re-implementation).

ASCII-only prints (Windows cp932). Saves into the repo; does NOT commit.
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
from src.audit.unified_metrics import IS_END, OOS_START
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)

# Reuse EVERYTHING from the original validation script (no rebuild).
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_tqqq_base, _build_p09_on_base, _metrics_pack, _min_at,
    _block_bootstrap_compare, _run_wfa,
)

# ---- Boost-map family (Task 2) ----
# M_v7   = original V7 = P09_TQQQ reference
# M_lu1  = the claimed LU1
# M_a/M_b= alternative monotone shapes (more / less aggressive)
# M_c/M_d= uniform leverage (no shape) at x1.40 / x1.20
MAP_FAMILY = {
    "M_v7":  {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00},
    "M_lu1": {0: 1.40, 1: 1.20, 2: 1.05, 3: 1.00},
    "M_a":   {0: 1.30, 1: 1.15, 2: 1.05, 3: 1.00},
    "M_b":   {0: 1.50, 1: 1.25, 2: 1.10, 3: 1.00},
    "M_c":   {0: 1.40, 1: 1.40, 2: 1.40, 3: 1.40},
    "M_d":   {0: 1.20, 1: 1.20, 2: 1.20, 3: 1.20},
}
MAP_ORDER = ["M_v7", "M_lu1", "M_a", "M_b", "M_c", "M_d"]


def main():
    print("=" * 78)
    print("LU1 REVALIDATE : LU1's own WFA/bootstrap + boost-map overfit sweep  2026-06-11")
    print("=" * 78)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)  # 1=IN, 0=OUT
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / TRADING_DAYS

    # ---- TQQQ-cost V7 baseline (the SAME reference used for P09_TQQQ) ----
    base_nav_dt, r_base_tqqq, tpy_base = _build_tqqq_base(shared, dates_dt)

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

    is_mask = dates_dt <= IS_END
    oos_mask = dates_dt >= OOS_START

    # =====================================================================
    # Helper: build P09-on-base for a given boost map -> (nav, r_p09, packs)
    # =====================================================================
    def _p09_for_map(v7_map):
        _, r_mbase, tpy_mb = _build_tqqq_base(shared, dates_dt, v7_map=v7_map)
        nav, r_p09, tpy = _build_p09_on_base(
            r_mbase, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
            dates_dt, tpy_mb, n_years)
        pre, aft, cy22, wcy, wcyy = _metrics_pack(nav, tpy)
        return nav, r_p09, (pre, aft, cy22, wcy, wcyy)

    # =====================================================================
    # TASK 1 : LU1's OWN robustness (lu1_nav vs SAME TQQQ-V7 baseline)
    # =====================================================================
    lu1_nav, r_lu1, lu1_pack = _p09_for_map(MAP_FAMILY["M_lu1"])
    pre_lu1, aft_lu1, cy22_lu1, wcy_lu1, wcyy_lu1 = lu1_pack
    min_at_lu1 = _min_at(aft_lu1)

    print("")
    print("TASK 1: LU1 own block bootstrap (paired LU1 vs V7_TQQQ baseline) ...")
    boot_lu1 = _block_bootstrap_compare(r_lu1, r_base_tqqq, is_mask, oos_mask)
    print("  P(min better)=%.4f  P(MaxDD better)=%.4f  CI95_lo(min gain)=%+.4f pp  mean gain=%+.4f pp"
          % (boot_lu1["P_min_better"], boot_lu1["P_maxdd_better"],
             boot_lu1["CI95_lo_min_pp"], boot_lu1["mean_min_improve_pp"]))

    print("TASK 1: LU1 own WFA (canonical g1_wfa windows) ...")
    wfa_lu1 = _run_wfa(lu1_nav, "LU1_strongmap")
    print("  LU1: WFE=%.3f  CI95_lo=%+.4f%%  n=%d  t_p=%.4f"
          % (wfa_lu1["WFE"], 100 * wfa_lu1["CI95_lo"], wfa_lu1["n_windows"], wfa_lu1["t_p"]))
    print("")
    print("  (P09_TQQQ reference, already validated: WFE 1.017, CI95_lo +17.94%, boot P_min=0.80)")
    print("")

    # =====================================================================
    # TASK 2 : boost-map family sweep on the SAME P09 base
    # =====================================================================
    print("TASK 2: boost-map family sweep (P09 OUT-fill on each map's IN baseline) ...")
    sweep = {}
    for name in MAP_ORDER:
        nav, r_p09, (pre, aft, cy22, wcy, wcyy) = _p09_for_map(MAP_FAMILY[name])
        sweep[name] = {
            "map": MAP_FAMILY[name],
            "CAGR_IS_at": aft["CAGR_IS"],
            "CAGR_OOS_at": aft["CAGR_OOS"],
            "min_at": _min_at(aft),
            "MaxDD": pre["MaxDD_FULL"],
            "Sharpe_OOS": pre["Sharpe_OOS"],
            "worstCY": wcy,
            "worstCY_year": wcyy,
            "cy2022": cy22,
        }
    print("  done.")
    print("")

    # =====================================================================
    # Assemble CSV: LU1 WFA/bootstrap row + the 6 map-sweep rows
    # =====================================================================
    rows = []
    rows.append({
        "row_type": "LU1_revalidate",
        "label": "LU1_strongmap",
        "map": json.dumps(MAP_FAMILY["M_lu1"]),
        "CAGR_IS_at": aft_lu1["CAGR_IS"],
        "CAGR_OOS_at": aft_lu1["CAGR_OOS"],
        "min_at": min_at_lu1,
        "MaxDD": pre_lu1["MaxDD_FULL"],
        "Sharpe_OOS": pre_lu1["Sharpe_OOS"],
        "worstCY_return": wcy_lu1,
        "worstCY_year": wcyy_lu1,
        "cy2022": cy22_lu1,
        "wfa_WFE": wfa_lu1["WFE"],
        "wfa_CI95_lo": wfa_lu1["CI95_lo"],
        "wfa_n_windows": wfa_lu1["n_windows"],
        "wfa_t_p": wfa_lu1["t_p"],
        "boot_P_min_better": boot_lu1["P_min_better"],
        "boot_P_maxdd_better": boot_lu1["P_maxdd_better"],
        "boot_CI95_lo_min_pp": boot_lu1["CI95_lo_min_pp"],
        "boot_mean_min_improve_pp": boot_lu1["mean_min_improve_pp"],
    })
    for name in MAP_ORDER:
        s = sweep[name]
        rows.append({
            "row_type": "map_sweep",
            "label": name,
            "map": json.dumps(s["map"]),
            "CAGR_IS_at": s["CAGR_IS_at"],
            "CAGR_OOS_at": s["CAGR_OOS_at"],
            "min_at": s["min_at"],
            "MaxDD": s["MaxDD"],
            "Sharpe_OOS": s["Sharpe_OOS"],
            "worstCY_return": s["worstCY"],
            "worstCY_year": s["worstCY_year"],
            "cy2022": s["cy2022"],
            "wfa_WFE": "", "wfa_CI95_lo": "", "wfa_n_windows": "", "wfa_t_p": "",
            "boot_P_min_better": "", "boot_P_maxdd_better": "",
            "boot_CI95_lo_min_pp": "", "boot_mean_min_improve_pp": "",
        })

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "lu1_revalidate_20260611.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8", float_format="%.6f")
    print("Saved CSV: %s" % csv_path)
    print("")

    # =====================================================================
    # ASCII tables
    # =====================================================================
    print("=" * 92)
    print("TASK 1 -- LU1's OWN robustness (lu1_nav vs SAME TQQQ-V7 baseline)")
    print("=" * 92)
    print("  LU1 min_at=%+.2f%%  MaxDD=%+.2f%%" % (100 * min_at_lu1, 100 * pre_lu1["MaxDD_FULL"]))
    print("  WFA      : WFE=%.3f  CI95_lo=%+.4f%%  n=%d  t_p=%.4f"
          % (wfa_lu1["WFE"], 100 * wfa_lu1["CI95_lo"], wfa_lu1["n_windows"], wfa_lu1["t_p"]))
    print("  Bootstrap: P(min better)=%.4f  CI95_lo(min gain)=%+.4f pp  P(MaxDD better)=%.4f"
          % (boot_lu1["P_min_better"], boot_lu1["CI95_lo_min_pp"], boot_lu1["P_maxdd_better"]))
    print("  Reference P09_TQQQ (validated): WFE=1.017  CI95_lo=+17.94%  boot P_min=0.80")
    print("=" * 92)
    print("")

    print("=" * 104)
    print("TASK 2 -- BOOST-MAP FAMILY SWEEP (after-tax min/CAGR; MaxDD/Sharpe/worstCY pretax)")
    print("=" * 104)
    hdr = ("%-7s | %-26s | %10s | %11s | %9s | %8s | %7s | %12s"
           % ("map", "{q0,q1,q2,q3}", "CAGR_IS_at", "CAGR_OOS_at",
              "min_at", "MaxDD", "Sharpe", "worstCY"))
    print(hdr)
    print("-" * 104)
    for name in MAP_ORDER:
        s = sweep[name]
        m = s["map"]
        mstr = "{%.2f,%.2f,%.2f,%.2f}" % (m[0], m[1], m[2], m[3])
        flag = "  <- LU1" if name == "M_lu1" else ("  (ref V7)" if name == "M_v7" else "")
        print("%-7s | %-26s | %+9.2f%% | %+10.2f%% | %+8.2f%% | %+7.2f%% | %6.3f | %+6.2f%%(%d)%s"
              % (name, mstr, 100 * s["CAGR_IS_at"], 100 * s["CAGR_OOS_at"],
                 100 * s["min_at"], 100 * s["MaxDD"], s["Sharpe_OOS"],
                 100 * s["worstCY"], s["worstCY_year"], flag))
    print("=" * 104)
    print("")

    # Rank by min_at to see if M_lu1 is an isolated peak
    ranked = sorted(MAP_ORDER, key=lambda nm: sweep[nm]["min_at"], reverse=True)
    best = ranked[0]
    lu1_min = sweep["M_lu1"]["min_at"]
    best_min = sweep[best]["min_at"]
    # gap from LU1 to best non-LU1 and to 2nd place
    second = ranked[1] if ranked[0] == "M_lu1" else ranked[0]
    print("Ranking by min_at (desc): %s" % " > ".join(
        ["%s(%+.2f%%)" % (nm, 100 * sweep[nm]["min_at"]) for nm in ranked]))
    print("  Best map = %s (min_at=%+.2f%%). LU1 min_at=%+.2f%%. Gap(LU1 - 2nd)=%+.3fpp"
          % (best, 100 * best_min, 100 * lu1_min,
             100 * (lu1_min - sweep[second]["min_at"])))
    # shape vs leverage diagnostic
    print("  Shape vs leverage: best uniform map min = %+.2f%% (max of M_c,M_d); "
          "best shaped map min = %+.2f%% (max of M_v7,M_lu1,M_a,M_b)"
          % (100 * max(sweep["M_c"]["min_at"], sweep["M_d"]["min_at"]),
             100 * max(sweep["M_v7"]["min_at"], sweep["M_lu1"]["min_at"],
                       sweep["M_a"]["min_at"], sweep["M_b"]["min_at"])))
    print("")

    # =====================================================================
    # JSON-like return block
    # =====================================================================
    out = {
        "LU1_own": {
            "WFE": round(float(wfa_lu1["WFE"]), 4),
            "WFA_CI95_lo": round(float(wfa_lu1["CI95_lo"]), 6),
            "n_windows": int(wfa_lu1["n_windows"]),
            "t_p": round(float(wfa_lu1["t_p"]), 4),
            "boot_P_min": boot_lu1["P_min_better"],
            "boot_CI95_lo_pp": round(boot_lu1["CI95_lo_min_pp"], 4),
            "boot_P_maxdd": boot_lu1["P_maxdd_better"],
            "min_at": round(min_at_lu1, 6),
            "MaxDD": round(pre_lu1["MaxDD_FULL"], 6),
        },
        "map_sweep": {
            name: {
                "min_at": round(sweep[name]["min_at"], 6),
                "MaxDD": round(sweep[name]["MaxDD"], 6),
                "worstCY": round(sweep[name]["worstCY"], 6),
            } for name in MAP_ORDER
        },
    }
    print("=" * 78)
    print("RETURN_BLOCK")
    print("=" * 78)
    print(json.dumps(out, indent=2))
    print("")
    print("Done.")
    return out


if __name__ == "__main__":
    main()
