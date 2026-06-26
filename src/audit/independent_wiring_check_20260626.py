"""
src/audit/independent_wiring_check_20260626.py
==============================================
INDEPENDENT wiring verifier for the SC2 DD-reduction candidates N4 and X4.

GOAL
----
Reproduce CAGR_OOS (after-tax, calendar split via the standard pipeline) and
MaxDD (pre-tax) for sc2.0 / N4 / X4 by building the strategies from the
LOW-LEVEL validated builders directly -- WITHOUT importing or executing
dd_reduction_harness_20260626.py or dd_reduction_verify_20260626.py.

If this script reproduces the harness numbers from an independent assembly of
the lower-level pieces, the candidate "wiring" is independently confirmed.

WIRING (per spec)
-----------------
Base = P09_STR strong-map:
  IN leg uses STRONG_MAP boost {0:1.60,1:1.50,2:1.10,3:1.00} times uniform
  `scale`, built via the TQQQ cost model with k365 excess EXCESS_EXTRA=0.0025.
  OUT days use the P09 C1 fill (Gold always + Bond when bond_mom252>0,
  inverse-vol W63 weights, SOFR cash yield on bond-OFF OUT days).
  This is exactly the sc2.0 family at scale=2.0 (P09_STR_sc2.00).

IN-leg gold/bond BLEND (the new part), on IN/HOLD days only:
  r_in_blend = (1 - gw - bw) * r_strongmap_IN
             + gw * ret_gold_1x + bw * ret_bond_1x
             - (gw*FEE_GOLD + bw*FEE_BOND)/252
where r_strongmap_IN = r_base from _build_tqqq_base_param (strong-map x scale
IN-leg daily return), ret_gold_1x / ret_bond_1x are the same 1x gold/bond series
used by the OUT fill, FEE_GOLD/FEE_BOND from run_p02_p09_backtest_20260611.

Final: r = where(fund_active, r_OUT_fill, r_in_blend)
  fund_active = DH-W1 OUT mask shifted forward by LAG_DAYS (= P09/cash-sleeve).

For sc2.0 (gw=bw=0): r_in_blend == r_strongmap_IN, so r == r_OUT_fill exactly
=> must reproduce P09_STR_sc2.00 (CAGR_OOS +29.11%, MaxDD -61.63%).

Candidates:
  N4 : scale=2.85, gw=0.28, bw=0.07
  X4 : scale=3.00, gw=0.32, bw=0.05

Metrics use the SAME pipeline path as the reported numbers:
  compute_10metrics (calc_7metrics date-split IS<=2021-05-07 / OOS>=2021-05-08;
  MaxDD on FULL window) + _apply_aftertax (CAGR x0.8273; MaxDD pre-tax).

NOTE: this script deliberately does NOT touch the forbidden harness/verify files.
It re-derives r_strongmap_IN, the OUT C1 fill, and the IN blend from the
component builders listed in the task. ASCII-only prints; does NOT commit.
"""

from __future__ import annotations

import json
import os
import sys
import types

# ---- multitasking stub (yfinance dep) ---------------------------------------
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

# Low-level validated builders (allowed) --------------------------------------
# strong-map IN leg (TQQQ cost model + k365 excess), parameterised:
from src.audit.k365_recost_20260612 import (
    _build_tqqq_base_param, EXCESS_EXTRA_K365_CENTRE,
)
# OUT C1 fill (reused for the OUT leg only):
from src.audit.leverup_b1c1_20260612 import _build_p09_nav_c1
# constants + helpers:
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, FEE_GOLD, FEE_BOND,
)

# ---- Strategy definitions (per spec / dd_export_series candidate list) -------
STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

CANDS = {
    "sc2.0": dict(scale=2.0, gw=0.0,  bw=0.0),
    "N4":    dict(scale=2.85, gw=0.28, bw=0.07),
    "X4":    dict(scale=3.00, gw=0.32, bw=0.05),
}

# Reported values to reproduce (from the pipeline / dd_reduction_verify CSV)
REPORTED = {
    "sc2.0": dict(CAGR_OOS=0.291102, MaxDD=-0.616342),
    "N4":    dict(CAGR_OOS=0.312910, MaxDD=-0.588201),
    "X4":    dict(CAGR_OOS=0.321472, MaxDD=-0.598517),
}


def _classify(metric, diff_abs):
    """SMALL/MEDIUM/LARGE per task tolerance.
    CAGR (pp): SMALL <0.1pp, MEDIUM <0.3pp (~3x), else LARGE.
    MaxDD (pp): SMALL <0.2pp, MEDIUM <0.6pp (~3x), else LARGE.
    diff_abs given in fraction; convert to pp.
    """
    pp = diff_abs * 100.0
    if metric == "MaxDD":
        small, med = 0.20, 0.60
    else:  # CAGR
        small, med = 0.10, 0.30
    if pp <= small:
        return "SMALL"
    if pp <= med:
        return "MEDIUM"
    return "LARGE"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 100)
    print("INDEPENDENT WIRING CHECK  2026-06-26  (N4 / X4 / sc2.0)")
    print("Built from low-level builders; harness/verify files NOT imported.")
    print("STRONG_MAP=%s  EXCESS_EXTRA=%.4f" % (STRONG_MAP, EXCESS_EXTRA))
    print("=" * 100)

    # ---- shared DH-W1 assets + mask --------------------------------------
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)  # 1=IN/HOLD, 0=OUT
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)

    # ---- gold / bond 1x legs (same series as OUT fill) -------------------
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    # ---- DH-W1 OUT mask shifted forward by LAG_DAYS (fund_active) --------
    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]

    # ---- inverse-vol W63 weights + bond-timing gate + SOFR ----------------
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    out_bond_off = fund_active & (~bond_on.astype(bool))
    print("\nSetup: n=%d (%.1f yrs)  OUT days=%d (%.1f%%)  fund_active=%d  OUT&bondOFF=%d"
          % (n, n_years, int(out_mask.sum()), 100.0 * out_mask.mean(),
             int(fund_active.sum()), int(out_bond_off.sum())))
    print("LAG_DAYS=%d  GATE_DELAY=%d  FEE_GOLD=%.6f  FEE_BOND=%.6f"
          % (LAG_DAYS, GATE_DELAY, FEE_GOLD, FEE_BOND))

    fee_in_const = None  # informational

    def build_candidate(scale, gw, bw):
        """Build one candidate NAV from low-level pieces.

        Returns (nav_dt, r_final, tpy, exc, r_strong, r_out).
        """
        # (1) strong-map x scale IN-leg daily return (= r_base)
        nav_strong, r_strong, tpy_base, exc = _build_tqqq_base_param(
            shared, dates_dt, v7_map=STRONG_MAP, lev_scale=scale,
            excess_extra=EXCESS_EXTRA)
        r_strong = np.asarray(r_strong, float)

        # (2) OUT C1 fill on top of the strong-map base (OUT days only differ;
        #     on IN days _build_p09_nav_c1 returns r_strong unchanged)
        nav_out, r_out, eff_active = _build_p09_nav_c1(
            r_strong, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr)
        r_out = np.asarray(r_out, float)

        # (3) IN-leg gold/bond blend on IN/HOLD days
        fee_in = (gw * FEE_GOLD + bw * FEE_BOND) / TRADING_DAYS
        r_in_blend = ((1.0 - gw - bw) * r_strong
                      + gw * ret_gold + bw * ret_bond
                      - fee_in)

        # (4) final daily return: OUT days -> C1 fill; IN days -> blend
        r_final = np.where(fund_active, r_out, r_in_blend)
        r_final = np.clip(r_final, -0.999, None)
        nav_arr = np.cumprod(1.0 + r_final)
        nav_dt = pd.Series(nav_arr, index=dates_dt)

        # trades/yr: base (lev change events) + OUT<->IN sleeve flips
        flips = _count_fund_transitions(eff_active)
        tpy = tpy_base + flips / n_years
        return nav_dt, r_final, tpy, exc, r_strong, r_out

    # ---- self-consistency check: sc2.0 blend path == pure C1 path ---------
    # For gw=bw=0, r_in_blend must equal r_strong on IN days, so r_final must
    # equal the pure C1 OUT-fill NAV bit-for-bit.
    nav_sc, r_sc, tpy_sc, exc_sc, r_strong_sc, r_out_sc = build_candidate(2.0, 0.0, 0.0)
    r_out_clipped = np.clip(r_out_sc, -0.999, None)
    max_dev = float(np.max(np.abs(r_sc - r_out_clipped)))
    print("\n[self-check] sc2.0 (gw=bw=0): max |r_blendpath - r_C1path| = %.3e -> %s"
          % (max_dev, "OK (blend collapses to C1)" if max_dev < 1e-12 else "WARN"))

    # ---- evaluate all candidates -----------------------------------------
    print("\n" + "=" * 100)
    print("RESULTS (metrics via compute_10metrics + _apply_aftertax)")
    print("=" * 100)
    print("%-6s | %-10s | %-10s | %-9s || %-10s | %-10s | %-9s"
          % ("cand", "CAGR_OOS", "reported", "diff_pp", "MaxDD", "reported", "diff_pp"))
    print("-" * 100)

    summary = {}
    for name, kw in CANDS.items():
        nav_dt, r_final, tpy, exc, r_strong, r_out = build_candidate(
            kw["scale"], kw["gw"], kw["bw"])
        pre = compute_10metrics(nav_dt, tpy)
        aft = _apply_aftertax(pre)
        cagr_oos = float(aft["CAGR_OOS"])   # after-tax
        cagr_is = float(aft["CAGR_IS"])     # after-tax
        maxdd = float(pre["MaxDD_FULL"])    # pre-tax

        rep = REPORTED[name]
        d_cagr = abs(cagr_oos - rep["CAGR_OOS"])
        d_dd = abs(maxdd - rep["MaxDD"])
        cls_cagr = _classify("CAGR", d_cagr)
        cls_dd = _classify("MaxDD", d_dd)

        summary[name] = dict(
            scale=kw["scale"], gw=kw["gw"], bw=kw["bw"],
            CAGR_OOS_at=cagr_oos, CAGR_IS_at=cagr_is, MaxDD=maxdd,
            Sharpe_OOS=float(pre["Sharpe_OOS"]),
            Sharpe_FULL=float(pre["Sharpe_FULL"]),
            Trades_yr=float(tpy), excess_days=int(exc),
            rep_CAGR_OOS=rep["CAGR_OOS"], rep_MaxDD=rep["MaxDD"],
            diff_CAGR_pp=d_cagr * 100.0, diff_MaxDD_pp=d_dd * 100.0,
            cls_CAGR=cls_cagr, cls_MaxDD=cls_dd,
        )

        print("%-6s | %+9.4f%% | %+9.4f%% | %+8.4f || %+9.4f%% | %+9.4f%% | %+8.4f"
              % (name, cagr_oos * 100, rep["CAGR_OOS"] * 100, (cagr_oos - rep["CAGR_OOS"]) * 100,
                 maxdd * 100, rep["MaxDD"] * 100, (maxdd - rep["MaxDD"]) * 100))

    print("\n" + "=" * 100)
    print("CLASSIFICATION (SMALL <0.1pp CAGR / <0.2pp MaxDD ; MEDIUM ~3x ; LARGE beyond)")
    print("=" * 100)
    for name in CANDS:
        s = summary[name]
        print("  %-6s : CAGR_OOS diff=%+.4fpp [%s]   MaxDD diff=%+.4fpp [%s]"
              % (name, s["diff_CAGR_pp"], s["cls_CAGR"],
                 s["diff_MaxDD_pp"], s["cls_MaxDD"]))

    all_small = all(summary[n]["cls_CAGR"] == "SMALL" and summary[n]["cls_MaxDD"] == "SMALL"
                    for n in CANDS)
    sc20_small = (summary["sc2.0"]["cls_CAGR"] == "SMALL"
                  and summary["sc2.0"]["cls_MaxDD"] == "SMALL")
    print("\n  sc2.0 anchor reproduced within SMALL tolerance: %s" % ("YES" if sc20_small else "NO"))
    print("  ALL candidates reproduced within SMALL tolerance: %s" % ("YES" if all_small else "NO"))
    print("  => WIRING INDEPENDENTLY CONFIRMED: %s"
          % ("YES" if all_small else "NO (see diffs above)"))

    block = {
        "script": "independent_wiring_check_20260626.py",
        "date": "2026-06-26",
        "strong_map": STRONG_MAP,
        "excess_extra": EXCESS_EXTRA,
        "metric_path": "compute_10metrics + _apply_aftertax (pipeline-identical)",
        "self_check_sc20_max_dev": max_dev,
        "candidates": {n: {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in summary[n].items()} for n in CANDS},
        "sc20_within_small": bool(sc20_small),
        "all_within_small": bool(all_small),
        "wiring_confirmed": bool(all_small),
    }
    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2, ensure_ascii=True))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
