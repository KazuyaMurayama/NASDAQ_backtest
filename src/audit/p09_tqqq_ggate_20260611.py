"""
src/audit/p09_tqqq_ggate_20260611.py
====================================
FULL G-gate / Phase-D robustness suite on the validated lead candidate
**P09_TQQQ** (= DH-W1 + mom63 V7 boost {1.20,1.10,1.00,1.00} + Bond-timing
OUT-fill, NASDAQ leg costed as TQQQ ETF) vs the **TQQQ-V7 baseline**.

This script REUSES the constructors from
  src/audit/run_p09_tqqq_validate_20260611.py
(no NAV is rebuilt from scratch) and adds the two gate pieces the validate
script did not run:
  - block-bootstrap P(better)/CI95_lo for Sharpe_OOS (in addition to
    min(IS,OOS) CAGR and MaxDD which the validate script already paired),
  - a permutation test on the OUT-fill (fill-vs-cash) decision: block-shuffle
    the fund_active OUT-day mask and measure how often a shuffled fill yields
    a min(IS,OOS) after-tax CAGR edge over baseline >= the observed edge.

Gate thresholds (the repo's standard multi-metric robustness gate):
  - Phase-D block bootstrap  : P(candidate better) > 0.90   PER metric
    (src/integration/phase_d_bootstrap.py: "PASS gate >90%")
  - WFA alpha (EVAL_STD 3.9) : WFA_CI95_lo > 0  AND  t_p < 0.05
  - WFA beta  (EVAL_STD 3.10): WFA_WFE in [0.5, 2.0]  (3.13: <=1.2 OK)
  - Permutation              : p < 0.05 PASS / <0.10 WARN / >=0.10 FAIL
    (src/audit/check_overfitting_e4_permutation.py: perm_verdict)

Premises: canonical split IS<=2021-05-07 / OOS>=2021-05-08, DELAY=2,
after-tax x0.8273 on CAGR, Sharpe/MaxDD pretax, T+5 fund lag, 252-day.

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
from src.audit.run_p01_backtest_20260611 import (
    TRADING_DAYS, LAG_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax, _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
)

# Reuse the validated P09_TQQQ constructors (do NOT rebuild from scratch).
from src.audit.run_p09_tqqq_validate_20260611 import (
    AFTER_TAX, N_BOOT, BLOCK, SEED,
    _build_tqqq_base, _build_p09_on_base, _build_p09_nav,
    _metrics_pack, _min_at,
    _maxdd_from_returns, _cagr_seg,
    _run_wfa,
)


# ---------------------------------------------------------------------------
# Gate thresholds (single source of truth for this script)
# ---------------------------------------------------------------------------
GATE = {
    "boot_P_better_min": 0.90,   # Phase-D: P(cand better) > 0.90 per metric
    "wfa_ci95_lo_min": 0.0,      # EVAL_STD 3.9: WFA_CI95_lo > 0
    "wfa_tp_max": 0.05,          # EVAL_STD 3.9: t_p < 0.05
    "wfa_wfe_lo": 0.5,           # EVAL_STD 3.10: WFE in [0.5, 2.0]
    "wfa_wfe_hi": 2.0,
    "perm_p_max": 0.05,          # permutation: p < 0.05 PASS
    "perm_p_warn": 0.10,         # 0.05 <= p < 0.10 WARN
}

N_PERM = 2000
PERM_BLOCK = 63   # ~3 months, matches check_overfitting_*_permutation


def _sharpe_oos(r, oos_mask):
    """Annualized Sharpe (Rf=0) over OOS days only."""
    r_o = np.asarray(r, float)[np.asarray(oos_mask, bool)]
    sd = np.std(r_o, ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(r_o) / sd * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# Full paired block bootstrap: P(better) + CI95_lo for THREE metrics:
#   min(IS,OOS) after-tax CAGR, Sharpe_OOS (pretax), MaxDD (pretax FULL).
# Same paired block ordering applied to both series each resample.
# ---------------------------------------------------------------------------
def _block_bootstrap_full(r_strat, r_base, is_mask, oos_mask,
                          n_boot=N_BOOT, block=BLOCK, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(r_strat)
    r_strat = np.asarray(r_strat, float)
    r_base = np.asarray(r_base, float)
    is_mask = np.asarray(is_mask, bool)
    oos_mask = np.asarray(oos_mask, bool)
    n_blocks = int(np.ceil(n / block))

    d_min = np.empty(n_boot)        # after-tax min(IS,OOS) CAGR gain (frac)
    d_sharpe = np.empty(n_boot)     # Sharpe_OOS gain
    d_maxdd = np.empty(n_boot)      # MaxDD gain (strat - base; >0 = less negative = better)
    nb_min = nb_sharpe = nb_maxdd = 0

    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]
        rs = r_strat[idx]
        rb = r_base[idx]
        im = is_mask[idx]
        om = oos_mask[idx]

        # min(IS,OOS) after-tax CAGR
        s_min = np.nanmin([_cagr_seg(rs[im]) * AFTER_TAX, _cagr_seg(rs[om]) * AFTER_TAX])
        b_min = np.nanmin([_cagr_seg(rb[im]) * AFTER_TAX, _cagr_seg(rb[om]) * AFTER_TAX])
        d_min[b] = s_min - b_min
        if s_min > b_min:
            nb_min += 1

        # Sharpe_OOS
        s_sh = _sharpe_oos(rs, om)
        b_sh = _sharpe_oos(rb, om)
        d_sharpe[b] = s_sh - b_sh
        if s_sh > b_sh:
            nb_sharpe += 1

        # MaxDD on resampled full path
        s_dd = _maxdd_from_returns(rs)
        b_dd = _maxdd_from_returns(rb)
        d_maxdd[b] = s_dd - b_dd
        if s_dd > b_dd:
            nb_maxdd += 1

    return {
        "P_min_better": nb_min / n_boot,
        "P_sharpe_better": nb_sharpe / n_boot,
        "P_maxdd_better": nb_maxdd / n_boot,
        "CI95_lo_min_pp": float(np.percentile(d_min, 2.5)) * 100.0,
        "CI95_lo_sharpe": float(np.percentile(d_sharpe, 2.5)),
        "CI95_lo_maxdd_pp": float(np.percentile(d_maxdd, 2.5)) * 100.0,
        "mean_min_pp": float(np.mean(d_min)) * 100.0,
        "mean_sharpe": float(np.mean(d_sharpe)),
        "mean_maxdd_pp": float(np.mean(d_maxdd)) * 100.0,
        "n_boot": n_boot, "block": block,
    }


# ---------------------------------------------------------------------------
# Permutation test on the OUT-fill (fill-vs-cash) decision.
#
# The entire P09 edge over the TQQQ-V7 baseline comes from CHOOSING to fill the
# OUT (cash) days with the Gold(+timed-Bond) sleeve. Under the null "the timing
# of which days are filled carries no information", block-shuffling the
# fund_active mask should produce an edge as large as observed with probability
# >= p. We block-shuffle fund_active (preserving the fraction of filled days and
# local autocorrelation), rebuild P09 with the SAME gold/bond returns & weights,
# and recompute the min(IS,OOS) after-tax CAGR edge vs baseline.
#
# p = fraction of permutations whose edge >= observed edge.
# ---------------------------------------------------------------------------
def _block_shuffle_mask(mask, block_len, rng):
    mask = np.asarray(mask, bool)
    T = len(mask)
    starts = list(range(0, T, block_len))
    perm = rng.permutation(len(starts))
    out = []
    for i in perm:
        s = starts[i]
        out.extend(mask[s:min(s + block_len, T)])
    return np.array(out[:T], dtype=bool)


def _permutation_fill(r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
                      is_mask, oos_mask, observed_edge_frac,
                      n_perm=N_PERM, block_len=PERM_BLOCK, seed=SEED):
    rng = np.random.RandomState(seed)
    b_min = np.nanmin([_cagr_seg(np.asarray(r_base)[is_mask]) * AFTER_TAX,
                       _cagr_seg(np.asarray(r_base)[oos_mask]) * AFTER_TAX])
    edges = np.empty(n_perm)
    ge = 0
    for b in range(n_perm):
        fa_perm = _block_shuffle_mask(fund_active, block_len, rng)
        _, r_perm, _ = _build_p09_nav(r_base, ret_gold, ret_bond, fa_perm, wg, wb, bond_on)
        s_min = np.nanmin([_cagr_seg(r_perm[is_mask]) * AFTER_TAX,
                           _cagr_seg(r_perm[oos_mask]) * AFTER_TAX])
        edge = s_min - b_min
        edges[b] = edge
        if edge >= observed_edge_frac:
            ge += 1
    p = ge / n_perm
    return {
        "p_value": p,
        "n_perm": n_perm, "block_len": block_len,
        "observed_edge_pp": observed_edge_frac * 100.0,
        "perm_mean_edge_pp": float(np.mean(edges)) * 100.0,
        "perm_P95_edge_pp": float(np.percentile(edges, 95)) * 100.0,
    }


def _perm_verdict(p):
    if p < GATE["perm_p_max"]:
        return "PASS"
    if p < GATE["perm_p_warn"]:
        return "WARN"
    return "FAIL"


def main():
    print("=" * 78)
    print("P09_TQQQ FULL G-GATE / Phase-D suite  vs  TQQQ-V7 baseline   2026-06-11")
    print("=" * 78)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / TRADING_DAYS

    # ---- TQQQ-V7 baseline (reused builder) ----
    base_nav, r_base, tpy_base = _build_tqqq_base(shared, dates_dt)
    pre_b, aft_b, _, _, _ = _metrics_pack(base_nav, tpy_base)
    min_at_b = _min_at(aft_b)
    print("\nBaseline TQQQ-V7 : min_at=%+.4f%%  Sharpe_OOS=%.4f  MaxDD=%+.4f%%  (expect min ~16.27%%)"
          % (100 * min_at_b, pre_b["Sharpe_OOS"], 100 * pre_b["MaxDD_FULL"]))

    # ---- Gold/Bond 1x returns + OUT mask + fund lag (same as P09) ----
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

    # ---- P09_TQQQ (reused builder) ----
    p09_nav, r_p09, tpy_p09 = _build_p09_on_base(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        dates_dt, tpy_base, n_years)
    pre9, aft9, cy22_9, wcy_9, wcyy_9 = _metrics_pack(p09_nav, tpy_p09)
    min_at_9 = _min_at(aft9)
    print("P09_TQQQ         : min_at=%+.4f%%  Sharpe_OOS=%.4f  MaxDD=%+.4f%%  Trades/yr=%.1f"
          % (100 * min_at_9, pre9["Sharpe_OOS"], 100 * pre9["MaxDD_FULL"], aft9["Trades_yr"]))

    is_mask = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    # ---- 1) Full block bootstrap (3 metrics) ----
    print("\n[1] Paired block bootstrap (%d resamples, block %d): 3 metrics ..." % (N_BOOT, BLOCK))
    boot = _block_bootstrap_full(r_p09, r_base, is_mask, oos_mask)
    print("    min(IS,OOS) CAGR : P_better=%.4f  CI95_lo=%+.4fpp  mean=%+.4fpp"
          % (boot["P_min_better"], boot["CI95_lo_min_pp"], boot["mean_min_pp"]))
    print("    Sharpe_OOS       : P_better=%.4f  CI95_lo=%+.4f   mean=%+.4f"
          % (boot["P_sharpe_better"], boot["CI95_lo_sharpe"], boot["mean_sharpe"]))
    print("    MaxDD            : P_better=%.4f  CI95_lo=%+.4fpp  mean=%+.4fpp"
          % (boot["P_maxdd_better"], boot["CI95_lo_maxdd_pp"], boot["mean_maxdd_pp"]))

    # ---- 2) WFA (canonical g1 windows, reused) ----
    print("\n[2] WFA (canonical g1_wfa windows) ...")
    wfa = _run_wfa(p09_nav, "P09_TQQQ")
    print("    WFE=%.4f  CI95_lo=%+.4f%%  n=%d  t_p=%.4g"
          % (wfa["WFE"], 100 * wfa["CI95_lo"], wfa["n_windows"], wfa["t_p"]))

    # ---- 3) Permutation test on the OUT-fill decision ----
    print("\n[3] Permutation test on OUT-fill mask (%d perms, block %d) ..." % (N_PERM, PERM_BLOCK))
    observed_edge = min_at_9 - min_at_b   # after-tax min(IS,OOS) edge, fraction
    perm = _permutation_fill(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        is_mask, oos_mask, observed_edge)
    perm_v = _perm_verdict(perm["p_value"])
    print("    observed_edge=%+.4fpp  perm_mean=%+.4fpp  perm_P95=%+.4fpp  p=%.4f [%s]"
          % (perm["observed_edge_pp"], perm["perm_mean_edge_pp"],
             perm["perm_P95_edge_pp"], perm["p_value"], perm_v))

    # ---- 4) Apply gates ----
    def _pf(ok):
        return "PASS" if ok else "FAIL"

    g_min = boot["P_min_better"] > GATE["boot_P_better_min"]
    g_sharpe = boot["P_sharpe_better"] > GATE["boot_P_better_min"]
    g_maxdd = boot["P_maxdd_better"] > GATE["boot_P_better_min"]
    g_wfa_alpha = (wfa["CI95_lo"] > GATE["wfa_ci95_lo_min"]) and (wfa["t_p"] < GATE["wfa_tp_max"])
    g_wfa_beta = GATE["wfa_wfe_lo"] <= wfa["WFE"] <= GATE["wfa_wfe_hi"]
    g_perm = perm["p_value"] < GATE["perm_p_max"]

    gate_results = {
        "boot_min_CAGR_P>0.90": (g_min, boot["P_min_better"]),
        "boot_Sharpe_P>0.90": (g_sharpe, boot["P_sharpe_better"]),
        "boot_MaxDD_P>0.90": (g_maxdd, boot["P_maxdd_better"]),
        "WFA_alpha_CI95lo>0_&_tp<0.05": (g_wfa_alpha, wfa["CI95_lo"]),
        "WFA_beta_WFE_in[0.5,2.0]": (g_wfa_beta, wfa["WFE"]),
        "permutation_p<0.05": (g_perm, perm["p_value"]),
    }
    overall = all(v[0] for v in gate_results.values())

    print("\n" + "=" * 78)
    print("GATE RESULTS")
    print("=" * 78)
    for k, (ok, val) in gate_results.items():
        print("  %-34s value=%+.4f   -> %s" % (k, val, _pf(ok)))
    print("-" * 78)
    print("  OVERALL : %s  (%d/%d gates pass)"
          % (_pf(overall), sum(v[0] for v in gate_results.values()), len(gate_results)))
    print("=" * 78)

    # ---- CSV ----
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "p09_tqqq_ggate_20260611.csv")
    rows = [
        ("baseline_min_at_pct", 100 * min_at_b, "", ""),
        ("P09_min_at_pct", 100 * min_at_9, "", ""),
        ("P09_Sharpe_OOS", pre9["Sharpe_OOS"], "", ""),
        ("P09_MaxDD_pct", 100 * pre9["MaxDD_FULL"], "", ""),
        ("P09_Trades_yr", aft9["Trades_yr"], "", ""),
        ("boot_P_min_better", boot["P_min_better"], GATE["boot_P_better_min"], _pf(g_min)),
        ("boot_CI95_lo_min_pp", boot["CI95_lo_min_pp"], "", ""),
        ("boot_P_sharpe_better", boot["P_sharpe_better"], GATE["boot_P_better_min"], _pf(g_sharpe)),
        ("boot_CI95_lo_sharpe", boot["CI95_lo_sharpe"], "", ""),
        ("boot_P_maxdd_better", boot["P_maxdd_better"], GATE["boot_P_better_min"], _pf(g_maxdd)),
        ("boot_CI95_lo_maxdd_pp", boot["CI95_lo_maxdd_pp"], "", ""),
        ("WFA_WFE", wfa["WFE"], "[0.5,2.0]", _pf(g_wfa_beta)),
        ("WFA_CI95_lo_pct", 100 * wfa["CI95_lo"], ">0", ""),
        ("WFA_t_p", wfa["t_p"], "<0.05", _pf(g_wfa_alpha)),
        ("WFA_n_windows", wfa["n_windows"], "", ""),
        ("permutation_p", perm["p_value"], "<0.05", _pf(g_perm)),
        ("permutation_observed_edge_pp", perm["observed_edge_pp"], "", ""),
        ("permutation_P95_edge_pp", perm["perm_P95_edge_pp"], "", ""),
        ("OVERALL", 1 if overall else 0, "", _pf(overall)),
    ]
    pd.DataFrame(rows, columns=["metric", "value", "threshold", "verdict"]).to_csv(
        csv_path, index=False, encoding="utf-8", float_format="%.6f")
    print("\nSaved CSV: %s" % csv_path)

    # ---- JSON return block ----
    block = {
        "gate_thresholds": GATE,
        "P09_TQQQ_results": {
            "boot_P_min": round(boot["P_min_better"], 4),
            "boot_CI95_lo_min_pp": round(boot["CI95_lo_min_pp"], 4),
            "boot_P_sharpe": round(boot["P_sharpe_better"], 4),
            "boot_CI95_lo_sharpe": round(boot["CI95_lo_sharpe"], 4),
            "boot_P_maxdd": round(boot["P_maxdd_better"], 4),
            "boot_CI95_lo_maxdd_pp": round(boot["CI95_lo_maxdd_pp"], 4),
            "WFE": round(float(wfa["WFE"]), 4),
            "WFA_CI95_lo_pct": round(100 * float(wfa["CI95_lo"]), 4),
            "t_p": round(float(wfa["t_p"]), 6),
            "n_windows": int(wfa["n_windows"]),
            "permutation_p": round(perm["p_value"], 4),
            "permutation_verdict": perm_v,
        },
        "gate_pass": {k: _pf(v[0]) for k, v in gate_results.items()},
        "overall": _pf(overall),
    }
    print("\n" + "=" * 78)
    print("RETURN_BLOCK")
    print("=" * 78)
    print(json.dumps(block, indent=2))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
