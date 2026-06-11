"""
src/audit/export_p09_live_spec_20260611.py
==========================================
Phase 0 of the P09_TQQQ / LU1 GAS notification system
(NASDAQ-strategy-gas/docs/P09_GAS_MIGRATION_PLAN_20260611.md).

Exports two artifacts for the GAS port:

1) audit_results/p09_live_spec_20260611.json
   Frozen constants + formula references (single source for CONFIG.P09):
     - A2 5-layer params (cross-check vs existing GAS CONFIG)
     - LT2-N750 params, vz065 regime (K_LO/K_MID/K_HI, thr 0.65)
     - W1 hysteresis (enter 0.7 / exit 0.3, start OUT)
     - mom63 frozen quartile boundaries (full-sample, freeze + annual refresh)
     - boost maps V7 (P09 default) / LU1, mult clip, DELAY
     - OUT-fill: inverse-vol(63) weekly w/ clamp, bond_mom252 gate, fees, LAG
     - minimum price-history requirements per signal

2) audit_results/p09_golden_vectors_20260611.csv
   Daily intermediates for the FULL chain over the last N_GOLDEN business
   days (default 756 = ~3y), used for 1:1 parity testing of the GAS port:
     t_index (absolute row index since 1974 -- anchors the weekly
     inverse-vol update cadence t%5==0), close, lev_raw, wn_A/wg_A/wb_A,
     vz, lt_sig, lev_mod_065, w1_mask, mom63, mult_p09/mult_lu1,
     L_p09/L_lu1 (DELAY-shifted), fund_active, gold_1x, bond_1x,
     w_g, w_b, bond_on, w_b_eff.

3) audit_results/p09_close_history_20260611.csv
   date,close for the last N_CLOSE_HIST (3,100) business days from the
   research dataset, so the GAS LT2-N750 port can be parity-tested on
   identical input (LT2 needs ~2,250 prior days for a full window).

ASCII-only prints. Saves into repo; no commit.
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
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, FEE_GOLD, FEE_BOND, WG_CLAMP, WEIGHT_UPDATE_BD,
    _ret_from_nav_level, _inverse_vol_weights,
)
from src.audit.run_p02_p09_backtest_20260611 import GATE_DELAY, _load_macro_signal
from src.audit.cost_model_cfd_vs_tqqq_20260611 import V7_MAP, DELAY as V7_DELAY
from src.audit.run_p09_tqqq_validate_20260611 import LU1_MAP, _build_v7_mult_custom
from g14_wfa_sbi_cfd import K_LO, K_MID, K_HI, VZ_THR_065, N_LT2
from g23a_dh_refinement_variants import ENTER_THR_W1, EXIT_THR_W1, hold_mask_W1
from long_cycle_signal import build_lt_signal
from corrected_strategy_backtest import THRESHOLD

N_GOLDEN = 756      # ~3 years of business days
N_CLOSE_HIST = 3100  # close history rows for LT2 parity (2250 warmup + golden)


def main():
    print("=" * 86)
    print("EXPORT P09 LIVE SPEC + GOLDEN VECTORS (Phase 0)   2026-06-11")
    print("=" * 86)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    close = a["close"]
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    idx = close.index

    # ---- frozen mom63 quartile boundaries (replicates _build_v7_mult_custom) ----
    macro_path = os.path.join(_REPO_DIR, "data", "macro_features.csv")
    macro = pd.read_csv(macro_path, parse_dates=["date"], index_col="date").sort_index()
    mom63_raw = macro["nasdaq_mom63"].dropna()
    q_bounds = [float(mom63_raw.quantile(q)) for q in (0.25, 0.50, 0.75)]
    print("mom63 frozen quartile boundaries (q25/q50/q75): "
          + " / ".join("%+.6f" % b for b in q_bounds))

    # mult arrays exactly as the harness builds them (publication lag included)
    mult_p09 = _build_v7_mult_custom(dates_dt, V7_MAP)
    mult_lu1 = _build_v7_mult_custom(dates_dt, LU1_MAP)

    # ---- DH-W1 chain intermediates ----
    lev_raw = np.asarray(a["lev_raw"], float)          # post-rebalance A2 leverage (0..1)
    wn_A = np.asarray(a["wn_A"], float)
    wg_A = np.asarray(a["wg_A"], float)
    wb_A = np.asarray(a["wb_A"], float)
    vz = np.asarray(a["vz_arr"], float)
    lev_mod_065 = np.asarray(a["lev_mod_065"], float)
    mask = np.asarray(shared["mask"], float)           # W1 hysteresis 1=IN, 0=OUT
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)

    # sanity: mask reproduces hold_mask_W1(a)
    assert np.array_equal(mask, hold_mask_W1(a)), "W1 mask mismatch vs g23a"

    # LT2-N750 signal (same as g14 build_lt_signal(close,'LT2',N_LT2))
    lt_sig = build_lt_signal(close, "LT2", N=N_LT2)

    # Effective leverage in x-units, DELAY-shifted as executed (per _build_nav_v7_tqqq)
    def _L(mult):
        lev_mod = lev_raw_masked * np.asarray(mult, float)
        return pd.Series(lev_mod * 3.0, index=idx).shift(V7_DELAY).fillna(1.0).values

    L_p09 = _L(mult_p09)
    L_lu1 = _L(mult_lu1)

    # ---- OUT-fill chain ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)
    out_mask = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask[:-LAG_DAYS]
    w_g, w_b = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    w_b_eff = np.where(bond_on, w_b, 0.0)

    # mom63 aligned to strategy dates (for the golden CSV; raw, no lag)
    mom63_aligned = mom63_raw.reindex(
        mom63_raw.index.union(dates_dt)).ffill().reindex(dates_dt).values

    # quantile labels as actually used (lagged, ffilled) -- recover from mult via map
    inv_p09 = {}
    for k, v in V7_MAP.items():
        inv_p09.setdefault(v, k)

    # ---- golden vectors CSV (last N_GOLDEN rows) ----
    sl = slice(n - N_GOLDEN, n)
    gv = pd.DataFrame({
        "t_index": np.arange(n)[sl],
        "date": dates_dt[sl].strftime("%Y-%m-%d"),
        "close": np.asarray(close.values, float)[sl],
        "lev_raw": lev_raw[sl],
        "wn_A": wn_A[sl], "wg_A": wg_A[sl], "wb_A": wb_A[sl],
        "vz": vz[sl],
        "lt_sig": np.asarray(lt_sig.values, float)[sl],
        "lev_mod_065": lev_mod_065[sl],
        "w1_mask": mask[sl],
        "mom63": mom63_aligned[sl],
        "mult_p09": np.asarray(mult_p09, float)[sl],
        "mult_lu1": np.asarray(mult_lu1, float)[sl],
        "L_p09": L_p09[sl], "L_lu1": L_lu1[sl],
        "fund_active": fund_active[sl].astype(int),
        "gold_1x": gold_1x[sl], "bond_1x": bond_1x[sl],
        "w_g": w_g[sl], "w_b": w_b[sl],
        "bond_on": bond_on[sl].astype(int),
        "w_b_eff": w_b_eff[sl],
    })
    out_csv = os.path.join(_REPO_DIR, "audit_results", "p09_golden_vectors_20260611.csv")
    gv.to_csv(out_csv, index=False, float_format="%.8f")
    print("Saved golden vectors: %s (%d rows, %s .. %s)"
          % (out_csv, len(gv), gv["date"].iloc[0], gv["date"].iloc[-1]))

    # ---- close history CSV (LT2 parity warmup + golden window) ----
    slh = slice(n - N_CLOSE_HIST, n)
    ch = pd.DataFrame({
        "date": dates_dt[slh].strftime("%Y-%m-%d"),
        "close": np.asarray(close.values, float)[slh],
    })
    out_ch = os.path.join(_REPO_DIR, "audit_results", "p09_close_history_20260611.csv")
    ch.to_csv(out_ch, index=False, float_format="%.6f")
    print("Saved close history: %s (%d rows, %s ..)" % (out_ch, len(ch), ch["date"].iloc[0]))

    # ---- spec JSON ----
    spec = {
        "spec_version": "1.0",
        "exported": "2026-06-11",
        "default_strategy": "P09_TQQQ",
        "alt_strategy": "LU1 (boost map swap only)",
        "a2_layers_existing_gas": {
            "note": "Already implemented in NASDAQ-strategy-gas Layers.gs; listed for cross-check only.",
            "dd": {"lookback": 200, "exit": 0.82, "reentry": 0.92},
            "asym_ewma": {"span_down": 10, "span_up": 30},
            "trend_tv": {"ma": 150, "tv_min": 0.10, "tv_max": 0.30,
                         "ratio_low": 0.85, "ratio_high": 1.15},
            "slope": {"ma": 200, "norm_window": 60, "base": 0.9,
                      "sensitivity": 0.35, "min": 0.3, "max": 1.5},
            "mom_decel": {"short": 60, "long": 180, "sensitivity": 0.3,
                          "min": 0.5, "max": 1.3},
            "vix_mr": {"ma_window": 252, "coeff": 0.25, "min": 0.50, "max": 1.15},
            "rebalance_threshold": float(THRESHOLD),
            "allocation": "wn=clip(0.55+0.25*lev-0.10*max(vz,0),0.30,0.90); wg=wb=(1-wn)/2",
            "vz_definition": "z of vix_proxy vs rolling252 mean/std "
                             "(corrected_strategy_backtest.build_a2_signal L224); "
                             "VERIFY equals GAS vix_z via golden vectors (plan task 0.3)",
        },
        "lt2": {
            "name": "LT2-N750", "N": int(N_LT2),
            "formula": "mom_N=close/close.shift(N)-1; z=((mom_N-roll_mean(2N,minp=N))"
                       "/roll_std(2N,minp=N)).clip(-3,3).fillna(0)",
            "source": "src/long_cycle_signal.py:compute_lt2",
        },
        "vz065_regime": {
            "vz_thr": float(VZ_THR_065),
            "k_lo": float(K_LO), "k_mid": float(K_MID), "k_hi": float(K_HI),
            "k_rule": "k = vz>+thr ? k_hi : (vz<-thr ? k_lo : k_mid)",
            "lt_bias": "clip(-k*lt_sig*0.5, -0.5, +0.5)",
            "lev_mod_065": "clip(lev_raw + lt_bias, 0, 1)",
            "source": "src/g14_wfa_sbi_cfd.py L334-339",
        },
        "w1_hysteresis": {
            "enter_thr": float(ENTER_THR_W1), "exit_thr": float(EXIT_THR_W1),
            "initial_state": "OUT",
            "rule": "OUT->IN if lev_mod_065>=enter; IN->OUT if lev_mod_065<=exit; else hold",
            "source": "src/g23a_dh_refinement_variants.py:hold_mask_W1",
        },
        "boost": {
            "signal": "nasdaq_mom63 = log(close)-log(close.shift(63))",
            "quantile_boundaries_frozen": {
                "q25": q_bounds[0], "q50": q_bounds[1], "q75": q_bounds[2],
                "freeze_policy": "full-sample quartiles frozen at export; refresh annually "
                                 "with a change report",
                "bucketing": "q = mom63<=q25 ? 0 : mom63<=q50 ? 1 : mom63<=q75 ? 2 : 3",
            },
            "publication_lag_days": 1,
            "map_p09_V7": {str(k): float(v) for k, v in V7_MAP.items()},
            "map_lu1": {str(k): float(v) for k, v in LU1_MAP.items()},
            "mult_clip": [0.0, 3.0],
            "L_t": "lev_raw_masked * mult * 3.0, shift(DELAY)",
            "delay_days": int(V7_DELAY),
        },
        "out_fill": {
            "fund_exec_lag_days": int(LAG_DAYS),
            "inverse_vol": {"window": 63, "clamp": list(WG_CLAMP),
                            "update_every_bd": int(WEIGHT_UPDATE_BD),
                            "warm_start": "0.5/0.5 before first valid window"},
            "bond_gate": {"signal": "bond_mom252 = log(bond_p)-log(bond_p.shift(252))",
                          "rule": "bond held iff bond_mom252>0; else that weight in CASH",
                          "gate_delay_days": int(GATE_DELAY)},
            "fees_annual": {"gold_1x": float(FEE_GOLD), "bond_1x": float(FEE_BOND)},
            "live_proxies": {"gold": "GC=F", "bond": "TLT",
                             "note": "research series are local-gold / yield-built bond; "
                                     "parity is judged on decisions (IN/OUT, bond_on, weights), "
                                     "not NAV"},
        },
        "min_history_days": {
            "nasdaq_close": int(2 * N_LT2 + N_LT2),   # mom_N lag 750 + rolling 1500
            "nasdaq_close_note": "LT2-N750: mom needs shift(750), z needs rolling(1500,minp=750) "
                                 "on mom => full-window parity from ~2250d; plan Phase1 corrected "
                                 "from 1150d to 2300d",
            "gold_close": 63 + GATE_DELAY + 10,
            "bond_close": 252 + GATE_DELAY + 10,
        },
        "golden_vectors": {
            "file": "audit_results/p09_golden_vectors_20260611.csv",
            "rows": int(N_GOLDEN),
            "parity_criteria": {"w1_mask_match": "100%", "L_abs_diff": "<0.01",
                                "weight_abs_diff": "<0.001"},
        },
    }
    out_json = os.path.join(_REPO_DIR, "audit_results", "p09_live_spec_20260611.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print("Saved spec JSON: %s" % out_json)

    # ---- console summary ----
    print("\n--- SUMMARY ---")
    print("rows total=%d  golden=%d" % (n, N_GOLDEN))
    print("W1 mask: IN days=%.1f%%  OUT days=%.1f%%"
          % (100 * mask.mean(), 100 * (1 - mask.mean())))
    g = gv.iloc[-1]
    print("Latest (%s): w1=%s  L_p09=%.3f  L_lu1=%.3f  fund_active=%s  bond_on=%s  w_g=%.3f"
          % (g["date"], "IN" if g["w1_mask"] > 0.5 else "OUT",
             g["L_p09"], g["L_lu1"], bool(g["fund_active"]), bool(g["bond_on"]), g["w_g"]))
    print("L_p09 >3x days (golden window): %d   L_lu1 >3x days: %d"
          % (int((gv["L_p09"] > 3.0 + 1e-9).sum()), int((gv["L_lu1"] > 3.0 + 1e-9).sum())))
    print("\nDone.")
    return spec


if __name__ == "__main__":
    main()
