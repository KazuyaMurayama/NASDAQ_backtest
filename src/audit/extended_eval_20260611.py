"""
src/audit/extended_eval_20260611.py
===================================
Phase 1-3 driver of the evaluation-methodology upgrade. Evaluate the contested
candidates under the "don't decide on one 5-year OOS" framework and present
min(IS,OOS) and WFA CI95_lo SIDE BY SIDE (co-primary), plus CPCV, regime
stratification and bootstrap-vs-baseline.

Candidates (all after-tax x0.8273 on CAGR; Sharpe/MaxDD pretax):
  V7_TQQQ   baseline (ETF) -- the comparison base
  P09_TQQQ  (ETF, attack/DD-tolerant)
  LU1_cfd   (ETF+CFD>3x recosted, strong boost)
  vz065_l5  (CFD, leverage 5x; the retained CFD candidate)

Metrics per candidate:
  - min(IS,OOS) after-tax CAGR              (current headline)
  - WFA CI95_lo (canonical g1 windows)      (proposed co-headline)
  - CPCV p10 / worst-fold after-tax CAGR    (N=10 blocks, k=2 -> C(10,2)=45 folds, embargo 21)
  - Regime-stratified after-tax CAGR over trend/vol/rate axes + Regime_min
  - Stress-window cumulative return + within-window MaxDD (2000/2008/2020/2022/2015)
  - Paired block bootstrap vs V7_TQQQ baseline: P(min better), CI95_lo of min gain
    (cross-environment for vz065_l5 -> flagged, not a clean ETF-vs-ETF comparison)

Reuses validated NAV builders (lu_cfd_recost / run_p09_tqqq_validate / strategy_runners).
ASCII-only prints. Saves audit_results/extended_eval_20260611.csv; does NOT commit.
"""
from __future__ import annotations

import os
import sys
import types
import json
from itertools import combinations

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
from src.audit.regime_labeler_20260611 import build_regime_labels, stress_masks

# Validated builders (single source: lu_cfd_recost handles cfd_excess uniformly)
from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base, _build_p09_on_base, LU1_MAP,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _run_wfa, _cagr_seg, _maxdd_from_returns, AFTER_TAX,
    _block_bootstrap_compare,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, _ret_from_nav_level, _inverse_vol_weights,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)

CPCV_N_BLOCKS = 10
CPCV_K = 2
CPCV_EMBARGO = 21


def _cum_ret(r):
    r = np.asarray(r, float)
    if len(r) == 0:
        return np.nan
    return float(np.prod(1.0 + np.clip(r, -0.999, None)) - 1.0)


def _cpcv_dist(r, n_blocks=CPCV_N_BLOCKS, k=CPCV_K, embargo=CPCV_EMBARGO):
    """After-tax CAGR over each C(n_blocks,k) combination of contiguous test blocks,
    embargo-trimming the front of each block to damp edge autocorrelation."""
    r = np.asarray(r, float)
    n = len(r)
    bounds = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [(bounds[i], bounds[i + 1]) for i in range(n_blocks)]
    out = []
    for combo in combinations(range(n_blocks), k):
        idx = []
        for bi in combo:
            s, e = blocks[bi]
            s2 = min(s + embargo, e)
            idx.extend(range(s2, e))
        rr = r[np.asarray(idx, int)]
        out.append(_cagr_seg(rr) * AFTER_TAX)
    return np.asarray(out, float)


def _regime_cagr(r, regimes):
    """After-tax CAGR within each regime label of each axis. Returns dict + Regime_min."""
    out = {}
    for ax in ("trend", "vol", "rate"):
        labs = pd.Series(regimes[ax].values)
        for lab in [x for x in labs.unique() if x != "n/a"]:
            mask = (regimes[ax].values == lab)
            out["%s:%s" % (ax, lab)] = _cagr_seg(r[mask]) * AFTER_TAX
    regime_min = float(np.nanmin(list(out.values()))) if out else np.nan
    return out, regime_min


def _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask, baseline_r=None,
              cross_env=False):
    pre = compute_10metrics(nav_dt, np.nan)
    cagr_is = _cagr_seg(r[is_mask]) * AFTER_TAX
    cagr_oos = _cagr_seg(r[oos_mask]) * AFTER_TAX
    min_at = min(cagr_is, cagr_oos)

    w = _run_wfa(nav_dt, label)

    reg, regime_min = _regime_cagr(r, regimes)

    st = {}
    for name, m in stress.items():
        rr = r[np.asarray(m, bool)]
        st[name] = {"ret": _cum_ret(rr), "maxdd": _maxdd_from_returns(rr) if len(rr) else np.nan}

    cpcv = _cpcv_dist(r)
    cpcv_p10 = float(np.percentile(cpcv, 10))
    cpcv_worst = float(np.min(cpcv))
    cpcv_med = float(np.median(cpcv))

    boot = None
    if baseline_r is not None:
        boot = _block_bootstrap_compare(r, baseline_r, is_mask, oos_mask)

    return {
        "label": label,
        "cagr_is_at": cagr_is, "cagr_oos_at": cagr_oos, "min_at": min_at,
        "Sharpe_OOS": pre["Sharpe_OOS"], "MaxDD_FULL": pre["MaxDD_FULL"],
        "wfa_WFE": w["WFE"], "wfa_CI95_lo": w["CI95_lo"], "wfa_t_p": w["t_p"],
        "regime": reg, "regime_min_at": regime_min,
        "cpcv_p10_at": cpcv_p10, "cpcv_worst_at": cpcv_worst, "cpcv_med_at": cpcv_med,
        "stress": st,
        "boot": boot, "cross_env": cross_env,
    }


def main():
    print("=" * 96)
    print("EXTENDED EVALUATION (rolling-WFA + CPCV + regime-stratified + bootstrap)  2026-06-11")
    print("co-primary headline: min(IS,OOS) AND WFA CI95_lo")
    print("=" * 96)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / 252.0
    is_mask = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)

    # ---- Gold/Bond 1x + OUT-fill machinery (shared by P09/LU1) ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)
    mask = np.asarray(shared["mask"], float)
    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)

    # ---- Build NAVs ----
    print("\nBuilding NAVs ...")
    # V7_TQQQ baseline
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(shared, dates_dt, cfd_excess=False)
    # P09_TQQQ
    p09_base_nav, r_p09base, tpy_p09b, _ = _build_tqqq_base(shared, dates_dt, cfd_excess=False)
    p09_nav, r_p09, tpy_p09 = _build_p09_on_base(
        r_p09base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, dates_dt, tpy_p09b, n_years)
    # LU1_cfd
    lu1_base_nav, r_lu1base, tpy_lu1b, exc1 = _build_tqqq_base(
        shared, dates_dt, v7_map=LU1_MAP, cfd_excess=True)
    lu1_nav, r_lu1, tpy_lu1 = _build_p09_on_base(
        r_lu1base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, dates_dt, tpy_lu1b, n_years)
    # vz065_l5 (CFD, realistic)
    vz = sr.run_vz065(5.0, "realistic")
    vz_nav = vz["nav"]
    r_vz = vz_nav.pct_change().fillna(0).values
    if len(vz_nav) != n:
        print("  WARN: vz065_l5 length %d != %d; aligning by reindex" % (len(vz_nav), n))
        vz_nav = vz_nav.reindex(dates_dt).ffill()
        r_vz = vz_nav.pct_change().fillna(0).values

    # ---- Sanity check: V7 2022 calendar return ~ -0.27% (matches canonical) ----
    cy = pd.Series(r_v7, index=dates_dt)
    cy2022 = _cum_ret(cy[(dates_dt.year == 2022)].values)
    print("  Sanity: V7_TQQQ 2022 calendar return = %+.2f%% (canonical ~ -0.27%%)" % (100 * cy2022))

    # ---- Evaluate ----
    res = {}
    res["V7_TQQQ"] = _eval_one("V7_TQQQ", v7_nav, r_v7, regimes, stress, is_mask, oos_mask)
    res["P09_TQQQ"] = _eval_one("P09_TQQQ", p09_nav, r_p09, regimes, stress, is_mask, oos_mask,
                                baseline_r=r_v7)
    res["LU1_cfd"] = _eval_one("LU1_cfd", lu1_nav, r_lu1, regimes, stress, is_mask, oos_mask,
                               baseline_r=r_v7)
    res["vz065_l5"] = _eval_one("vz065_l5", vz_nav, r_vz, regimes, stress, is_mask, oos_mask,
                                baseline_r=r_v7, cross_env=True)

    order = ["V7_TQQQ", "P09_TQQQ", "LU1_cfd", "vz065_l5"]

    # ---- Co-primary headline table ----
    print("\n" + "=" * 96)
    print("CO-PRIMARY HEADLINE  (after-tax; Sharpe/MaxDD pretax)")
    print("=" * 96)
    hdr = ("%-9s | %8s | %9s | %9s | %9s | %9s | %8s | %8s"
           % ("cand", "min_at", "WFA_CI95", "CPCV_p10", "CPCV_wst", "Reg_min", "Sharpe", "MaxDD"))
    print(hdr); print("-" * len(hdr))
    for k in order:
        x = res[k]
        print("%-9s | %+7.2f%% | %+8.2f%% | %+8.2f%% | %+8.2f%% | %+8.2f%% | %7.3f | %+7.2f%%"
              % (k, 100*x["min_at"], 100*x["wfa_CI95_lo"], 100*x["cpcv_p10_at"],
                 100*x["cpcv_worst_at"], 100*x["regime_min_at"], x["Sharpe_OOS"], 100*x["MaxDD_FULL"]))

    # ---- Regime detail ----
    print("\n" + "=" * 96)
    print("REGIME-STRATIFIED after-tax CAGR")
    print("=" * 96)
    axes_order = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol", "rate:rate_up", "rate:rate_down"]
    print("%-9s | %s" % ("cand", " | ".join("%-12s" % a for a in axes_order)))
    print("-" * 110)
    for k in order:
        rg = res[k]["regime"]
        cells = " | ".join("%+11.2f%%" % (100*rg.get(a, np.nan)) for a in axes_order)
        print("%-9s | %s" % (k, cells))

    # ---- Stress windows ----
    print("\n" + "=" * 96)
    print("STRESS WINDOWS: cumulative return (within-window MaxDD)")
    print("=" * 96)
    sw_order = list(stress.keys())
    print("%-9s | %s" % ("cand", " | ".join("%-16s" % s for s in sw_order)))
    print("-" * 120)
    for k in order:
        st = res[k]["stress"]
        cells = " | ".join("%+6.1f%%(%+5.1f%%)" % (100*st[s]["ret"], 100*st[s]["maxdd"]) for s in sw_order)
        print("%-9s | %s" % (k, cells))

    # ---- Bootstrap vs baseline ----
    print("\n" + "=" * 96)
    print("PAIRED BLOCK BOOTSTRAP vs V7_TQQQ baseline (min(IS,OOS) after-tax)")
    print("=" * 96)
    for k in order:
        b = res[k]["boot"]
        if b is None:
            print("  %-9s : baseline" % k); continue
        flag = "  [cross-env: CFD vs ETF, not clean]" if res[k]["cross_env"] else ""
        print("  %-9s : P(better)=%.3f  CI95_lo=%+.2fpp  mean=%+.2fpp%s"
              % (k, b["P_min_better"], b["CI95_lo_min_pp"], b["mean_min_improve_pp"], flag))

    # ---- CSV ----
    rows = []
    for k in order:
        x = res[k]
        row = {
            "candidate": k,
            "min_at": x["min_at"], "wfa_CI95_lo": x["wfa_CI95_lo"], "wfa_WFE": x["wfa_WFE"],
            "cpcv_p10_at": x["cpcv_p10_at"], "cpcv_worst_at": x["cpcv_worst_at"],
            "cpcv_med_at": x["cpcv_med_at"], "regime_min_at": x["regime_min_at"],
            "Sharpe_OOS": x["Sharpe_OOS"], "MaxDD_FULL": x["MaxDD_FULL"],
            "cagr_is_at": x["cagr_is_at"], "cagr_oos_at": x["cagr_oos_at"],
        }
        for a in axes_order:
            row["regime_" + a.replace(":", "_")] = x["regime"].get(a, np.nan)
        for s in sw_order:
            row["stress_%s_ret" % s] = x["stress"][s]["ret"]
            row["stress_%s_maxdd" % s] = x["stress"][s]["maxdd"]
        if x["boot"] is not None:
            row["boot_P_min_better"] = x["boot"]["P_min_better"]
            row["boot_CI95_lo_min_pp"] = x["boot"]["CI95_lo_min_pp"]
            row["boot_mean_min_improve_pp"] = x["boot"]["mean_min_improve_pp"]
        rows.append(row)
    out_csv = os.path.join(_REPO_DIR, "audit_results", "extended_eval_20260611.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
    print("\nSaved CSV: %s" % out_csv)

    # ---- JSON block ----
    block = {k: {
        "min_at": round(res[k]["min_at"], 4),
        "wfa_CI95_lo": round(float(res[k]["wfa_CI95_lo"]), 4),
        "wfa_WFE": round(float(res[k]["wfa_WFE"]), 4),
        "cpcv_p10_at": round(res[k]["cpcv_p10_at"], 4),
        "cpcv_worst_at": round(res[k]["cpcv_worst_at"], 4),
        "regime_min_at": round(res[k]["regime_min_at"], 4),
        "Sharpe": round(res[k]["Sharpe_OOS"], 4),
        "MaxDD": round(res[k]["MaxDD_FULL"], 4),
        "boot_P_min_better": (round(res[k]["boot"]["P_min_better"], 3) if res[k]["boot"] else None),
        "boot_CI95_lo_min_pp": (round(res[k]["boot"]["CI95_lo_min_pp"], 3) if res[k]["boot"] else None),
        "cross_env": res[k]["cross_env"],
    } for k in order}
    print("\nRETURN_BLOCK")
    print(json.dumps(block, indent=2))
    return block


if __name__ == "__main__":
    main()
