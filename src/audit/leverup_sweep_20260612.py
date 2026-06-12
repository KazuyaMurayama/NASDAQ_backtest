"""
src/audit/leverup_sweep_20260612.py
====================================
B2-B4 lever-up grid sweep on DH-W1 + P09 framework with C1 SOFR cash yield.

Stage 1 (--stage 1): Build NAVs for ~44 configs, compute standard-10 metrics +
  MaxDD/W10Y hard veto check only (no WFA/CPCV/bootstrap).
  Output: audit_results/leverup_sweep_stage1_20260612.csv

Stage 2 (--stage 2 [--part N/M]): Read Stage-1 CSV, filter candidates that pass
  Stage-1 gate, run _eval_one full-gate (WFA+CPCV+regime+stress+bootstrap).
  Output: audit_results/leverup_sweep_stage2_20260612.csv

Grid:
  B2 (LU1-style map sweep): v7_map = {0:q0, 1:q1, 2:1.05, 3:1.00},
    q0 in {1.4, 1.6, 1.8, 2.0} x q1 in {1.2, 1.3, 1.4}, lev_scale=1.0
    => 12 configs. (q0=1.4, q1=1.2 should match LU1_C1 sanity)

  B3 (B2 map x scale): same maps x lev_scale in {1.10, 1.15} => 24 configs.

  B4 (conditional boost LU3): mult depends on (q, lev_raw_masked):
    if q==0 and lev_raw_masked_lagged > 0.6 : q0_hi (high-lev IN Q0)
    else: {0: 1.0, 1: 1.20, 2: 1.05, 3: 1.00}[q]
    q0_hi in {1.6, 1.8, 2.0, 2.4} x lev_scale in {1.0, 1.15} => 8 configs.
    *** posthoc_flag=1: condition discovered post-hoc on same data -- WFA/CPCV
    gate results must still be reported but single-OOS adoption NOT permitted.
    Additional validation required before any adoption decision. ***

Causal integrity for B4:
  The condition (q==0 and lev_raw_masked > 0.6) uses SIGNAL-TIME values ONLY:
    - q: from sig_aligned (quantile_cut -> apply_publication_lag -> reindex/ffill)
      identical pipeline to _build_v7_mult_custom, no lookahead.
    - lev_raw_masked: shared["lev_raw_masked"] = lev_raw * hold_mask_W1, a pure
      DH-W1 output without any forward information.
  Both are assembled into mult_cond_arr BEFORE the DELAY=2 pd.Series.shift in
  _build_nav_v7_tqqq.  The shift is applied to the already-computed mult_cond_arr,
  exactly as mult_v7 is shifted in the base implementation.
  => No lookahead introduced at the signal level. The DELAY shift handles
  execution lag identically to all other variants.

All configs use C1 (SOFR cash on bond-OFF OUT days) as standard.

Sanity checkpoints (printed; script halts if violated):
  B2 q0=1.4,q1=1.2,scale=1.0 == LU1_C1: min9 ~+18.30%, MaxDD ~-34.76%
  (tolerance +/-0.3pp on min9, +/-0.8pp on MaxDD)

ASCII-only prints (cp932). Does NOT commit. No temp files created.
"""

from __future__ import annotations

import argparse
import json
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
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.regime_labeler_20260611 import build_regime_labels, stress_masks

from src.audit.lu_cfd_recost_20260611 import (
    _build_tqqq_base, _build_p09_on_base, _metrics_pack, LU1_MAP,
    AFTER_TAX, EXCESS_EXTRA, LEV_CAP,
    _build_nav_v7_tqqq,
    SWAP_SPREAD, TER_TQQQ,
)
from src.audit.extended_eval_20260611 import (
    _eval_one, _cpcv_dist, _regime_cagr,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _run_wfa, _block_bootstrap_compare, LU2_SCALE, _cagr_seg, _maxdd_from_returns,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _calendar_year_returns, _count_fund_transitions,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, _build_p09_nav,
    FEE_GOLD, FEE_BOND,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    _build_v7_mult, DELAY as V7_DELAY, DH_PER_UNIT, NAV_FLOOR,
)
from src.audit.run_p09_tqqq_validate_20260611 import _build_v7_mult_custom

# ---------------------------------------------------------------------------
# C1: SOFR cash on bond-OFF OUT days  (copied from leverup_b1c1_20260612.py)
# ---------------------------------------------------------------------------

def _build_p09_nav_c1(r_base, ret_gold, ret_bond, fund_active, w_g, w_b,
                      bond_on, sofr_arr):
    bond_on = np.asarray(bond_on, dtype=bool)
    sofr_arr = np.asarray(sofr_arr, float)
    w_b_eff = np.where(bond_on, w_b, 0.0)
    fee_daily = (w_g * FEE_GOLD + w_b_eff * FEE_BOND) / TRADING_DAYS
    cash_yield = np.where(bond_on, 0.0, w_b) * sofr_arr
    r_blend = w_g * ret_gold + w_b_eff * ret_bond + cash_yield - fee_daily
    r = np.where(fund_active, r_blend, r_base)
    r = np.clip(r, -0.999, None)
    nav = np.cumprod(1.0 + r)
    return nav, r, fund_active.copy()


def _build_p09_on_base_c1(r_base, ret_gold, ret_bond, fund_active, wg, wb,
                           bond_on, sofr_arr, dates_dt, tpy_base, n_years):
    nav_arr, r_c1, eff_active = _build_p09_nav_c1(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr)
    nav_dt = pd.Series(nav_arr, index=dates_dt)
    flips = _count_fund_transitions(eff_active)
    tpy = tpy_base + flips / n_years
    return nav_dt, r_c1, tpy


# ---------------------------------------------------------------------------
# B4: conditional mult builder (causal-safe)
# ---------------------------------------------------------------------------

def _build_v7_mult_cond(date_index, q0_hi, lev_raw_masked_arr, lev_thr=0.6):
    """Build per-day multiplier array for B4 conditional boost.

    Causal rule:
      - Signal quantile q is computed identically to _build_v7_mult_custom:
          quantile_cut -> apply_publication_lag -> reindex/ffill
        This is a SIGNAL-TIME series with all publication lags already applied.
      - lev_raw_masked_arr is shared["lev_raw_masked"]: DH-W1 output, no forward
        information. It is aligned to date_index without any additional lag --
        it enters the formula at signal time exactly as the quantile signal does.
      - The resulting mult_cond_arr is then passed into _build_nav_v7_tqqq which
        applies pd.Series.shift(V7_DELAY) to produce the execution-lagged L.
        This shift covers BOTH the signal and lev_raw_masked condition, so no
        lookahead is introduced anywhere in the pipeline.

    mult formula (signal-time, pre-shift):
      q==0 and lev_raw_masked > lev_thr  ->  q0_hi   (IN Q0 high-leverage window)
      q==0 and lev_raw_masked <= lev_thr ->  1.0     (IN Q0 low-leverage, no boost)
      q==1                               ->  1.20
      q==2                               ->  1.05
      q==3                               ->  1.00
    """
    macro_path = os.path.join(_REPO_DIR, "data", "macro_features.csv")
    macro = pd.read_csv(macro_path, parse_dates=["date"], index_col="date").sort_index()
    signal_raw = macro["nasdaq_mom63"].dropna()

    from signals.quantize import quantile_cut as _quantile_cut
    from signals.timing import apply_publication_lag as _apply_lag

    sig_q = _quantile_cut(signal_raw, levels=4).dropna().astype("int8")
    sig_lagged = _apply_lag(sig_q, "daily")
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    sig_aligned = sig_lagged.reindex(date_index).ffill()

    # lev_raw_masked as a Series aligned to date_index (signal-time, pre-shift)
    lev_series = pd.Series(np.asarray(lev_raw_masked_arr, float), index=date_index)

    def _cond_mult(i):
        s = sig_aligned.iloc[i]
        lev = lev_series.iloc[i]
        if pd.isna(s):
            return 1.0
        q = int(s)
        if q == 0:
            return float(q0_hi) if lev > lev_thr else 1.0
        elif q == 1:
            return 1.20
        elif q == 2:
            return 1.05
        else:
            return 1.00

    mult_arr = np.array([_cond_mult(i) for i in range(len(date_index))], dtype=float)
    return np.clip(mult_arr, 0.0, 3.0)


# ---------------------------------------------------------------------------
# NAV builder for B2/B3/B4 (routes through lu_cfd_recost._build_nav_v7_tqqq)
# ---------------------------------------------------------------------------

def _build_tqqq_base_custom_mult(shared, date_index, mult_v7, cfd_excess=True):
    """Like _build_tqqq_base but accepts a pre-built mult_v7 array (for B4)."""
    from src.audit.strategy_runners import (
        _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
        _compute_dhw1_trades_per_year,
    )
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

    nav_dt, tpy, excess_days = _build_nav_v7_tqqq(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7, cfd_excess=cfd_excess)
    r_base = nav_dt.pct_change().fillna(0).values
    return nav_dt, r_base, tpy, excess_days


# ---------------------------------------------------------------------------
# Build one config -> (nav_dt, r, tpy, excess_days)  with C1
# ---------------------------------------------------------------------------

def _build_one(shared, date_index, dates_dt, n_years,
               ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
               v7_map=None, lev_scale=1.0, cfd_excess=True,
               mult_v7_prebuilt=None):
    """Build NAV for one sweep config (always C1).

    mult_v7_prebuilt: if provided, use this array directly (for B4).
    Otherwise construct from v7_map / lev_scale.
    """
    if mult_v7_prebuilt is not None:
        mult_v7 = np.asarray(mult_v7_prebuilt, float) * float(lev_scale)
        base_nav, r_base, tpy_b, exc = _build_tqqq_base_custom_mult(
            shared, date_index, mult_v7, cfd_excess=cfd_excess)
    else:
        base_nav, r_base, tpy_b, exc = _build_tqqq_base(
            shared, date_index, v7_map=v7_map, lev_scale=lev_scale,
            cfd_excess=cfd_excess)
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        sofr_arr, dates_dt, tpy_b, n_years)
    return nav_dt, r, tpy, exc


# ---------------------------------------------------------------------------
# Stage 1: metrics + veto only  (no WFA/CPCV/bootstrap)
# ---------------------------------------------------------------------------
STAGE1_PASS_MAXDD = -0.50
STAGE1_PASS_W10Y  = 0.0
STAGE1_PASS_MIN9  = 0.19   # LU2_C1 = 0.1909 => threshold 19.0%


def _stage1_row(label, nav_dt, r, tpy, exc, is_mask, oos_mask,
                group, posthoc_flag, params):
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    min9 = min(aft["CAGR_IS"], aft["CAGR_OOS"])
    maxdd = pre["MaxDD_FULL"]
    w10y = aft["Worst10Y_star"]
    cy = _calendar_year_returns(nav_dt)

    v_maxdd = maxdd < STAGE1_PASS_MAXDD
    v_w10y  = w10y  < STAGE1_PASS_W10Y
    veto = v_maxdd or v_w10y

    row = {
        "label": label,
        "group": group,
        "posthoc_flag": posthoc_flag,
        "CAGR_IS_at": aft["CAGR_IS"],
        "CAGR_OOS_at": aft["CAGR_OOS"],
        "min9_at": min9,
        "gap_pp": aft["IS_OOS_gap_pp"],
        "Sharpe_OOS": pre["Sharpe_OOS"],
        "MaxDD_FULL": maxdd,
        "Worst10Y_star_at": w10y,
        "P10_5Y_at": aft["P10_5Y"],
        "Trades_yr": aft["Trades_yr"],
        "excess_days": exc,
        "worst_cy": float(cy.min()),
        "worst_cy_year": int(cy.idxmin()),
        "veto_maxdd": int(v_maxdd),
        "veto_w10y":  int(v_w10y),
        "VETO": int(veto),
        "stage1_pass": int(not veto and min9 >= STAGE1_PASS_MIN9),
    }
    row.update(params)
    return row


# ---------------------------------------------------------------------------
# Stage 2: full gate via _eval_one
# ---------------------------------------------------------------------------
STAGE2_MAX_CANDIDATES = 8

HARD_VETO_MAXDD   = -0.50
HARD_VETO_WFE     = 1.5
HARD_VETO_W10Y    = 0.0
HARD_VETO_REGIME  = -0.10


def _stage2_row(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                r_v7, group, posthoc_flag, params, s1_row):
    """Run full _eval_one and assemble a Stage-2 row."""
    bl = r_v7  # bootstrap vs V7_TQQQ
    ev = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask,
                   baseline_r=bl)
    pre = compute_10metrics(nav_dt, s1_row["Trades_yr"])  # tpy already in s1

    w10y = s1_row["Worst10Y_star_at"]
    maxdd = s1_row["MaxDD_FULL"]
    wfe = float(ev["wfa_WFE"])
    reg_min = float(ev["regime_min_at"])

    v_maxdd  = maxdd < HARD_VETO_MAXDD
    v_wfe    = wfe   > HARD_VETO_WFE
    v_w10y   = w10y  < HARD_VETO_W10Y
    v_reg    = reg_min < HARD_VETO_REGIME
    veto = v_maxdd or v_wfe or v_w10y or v_reg

    boot = ev.get("boot") or {}
    axes_order = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                  "rate:rate_up", "rate:rate_down"]
    sw_order = list(stress.keys())

    row = dict(s1_row)  # carry Stage-1 cols
    row.update({
        "wfa_WFE":         wfe,
        "wfa_CI95_lo":     float(ev["wfa_CI95_lo"]),
        "wfa_t_p":         float(ev["wfa_t_p"]),
        "cpcv_p10_at":     float(ev["cpcv_p10_at"]),
        "cpcv_worst_at":   float(ev["cpcv_worst_at"]),
        "cpcv_med_at":     float(ev["cpcv_med_at"]),
        "regime_min_at":   reg_min,
        "boot_P_min_better":   boot.get("P_min_better", ""),
        "boot_CI95_lo_min_pp": boot.get("CI95_lo_min_pp", ""),
        "s2_veto_maxdd":   int(v_maxdd),
        "s2_veto_wfe":     int(v_wfe),
        "s2_veto_w10y":    int(v_w10y),
        "s2_veto_reg":     int(v_reg),
        "S2_VETO":         int(veto),
        "posthoc_note": (
            "WARNING: posthoc_flag=1 (IN_Q0_hi criterion discovered on same data). "
            "WFA/CPCV results shown for transparency but single-OOS adoption NOT "
            "permitted. Additional out-of-sample validation required."
        ) if posthoc_flag else "",
    })
    for ax in axes_order:
        row["regime_" + ax.replace(":", "_")] = ev["regime"].get(ax, np.nan)
    for sw in sw_order:
        row["stress_%s_ret" % sw] = ev["stress"][sw]["ret"]
        row["stress_%s_maxdd" % sw] = ev["stress"][sw]["maxdd"]
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def stage1(shared, dates_dt, ret_gold, ret_bond, fund_active, wg, wb,
           bond_on, sofr_arr, is_mask, oos_mask, out_dir):
    print("=" * 96)
    print("STAGE 1: Grid sweep NAV build + standard-10 metrics (no WFA/CPCV)")
    print("=" * 96)

    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    a = shared["assets"]
    dates = a["dates"]
    lev_raw_masked_arr = np.asarray(shared["lev_raw_masked"], float)

    rows = []

    # ------------------------------------------------------------------
    # Reference rows: P09_TQQQ, LU1_C1, LU2_C1 (read from B1C1 CSV)
    # ------------------------------------------------------------------
    b1c1_csv = os.path.join(out_dir, "leverup_b1c1_20260612.csv")
    b1c1_refs = {}
    if os.path.exists(b1c1_csv):
        df_ref = pd.read_csv(b1c1_csv)
        for lbl in ("P09_TQQQ", "LU1_C1", "LU2_C1"):
            row_ref = df_ref[(df_ref["condition"] == lbl) & (df_ref["tax"] == "aftertax")]
            if len(row_ref):
                r0 = row_ref.iloc[0]
                b1c1_refs[lbl] = {
                    "min9_at":        float(r0["min_IS_OOS"]),
                    "MaxDD_FULL":     float(r0["MaxDD_FULL"]),
                    "Worst10Y_star_at": float(r0["Worst10Y_star"]),
                    "wfa_WFE":        float(r0["wfa_WFE"]) if "wfa_WFE" in r0 else np.nan,
                    "wfa_CI95_lo":    float(r0["wfa_CI95_lo"]) if "wfa_CI95_lo" in r0 else np.nan,
                }
        print("\nLoaded B1C1 reference values:")
        for k, v in b1c1_refs.items():
            print("  %-10s  min9=%+.4f%%  MaxDD=%+.2f%%  WFE=%.4f  CI95_lo=%+.4f%%"
                  % (k, v["min9_at"] * 100, v["MaxDD_FULL"] * 100,
                     v["wfa_WFE"] if np.isfinite(v["wfa_WFE"]) else float("nan"),
                     v["wfa_CI95_lo"] * 100 if np.isfinite(v["wfa_CI95_lo"]) else float("nan")))
    else:
        print("\nWARNING: leverup_b1c1_20260612.csv not found; reference rows skipped")

    # ------------------------------------------------------------------
    # Define sweep configs
    # ------------------------------------------------------------------
    configs = []

    # B2: LU1-style map, lev_scale=1.0
    for q0 in (1.4, 1.6, 1.8, 2.0):
        for q1 in (1.2, 1.3, 1.4):
            v7m = {0: q0, 1: q1, 2: 1.05, 3: 1.00}
            lbl = "B2_q0%.1f_q1%.2f" % (q0, q1)
            configs.append({
                "label": lbl, "group": "B2",
                "v7_map": v7m, "lev_scale": 1.0, "posthoc_flag": 0,
                "params": {"q0": q0, "q1": q1, "lev_scale": 1.0,
                           "q0_hi": np.nan, "lev_thr": np.nan},
            })

    # B3: B2 maps x lev_scale in {1.10, 1.15}
    for q0 in (1.4, 1.6, 1.8, 2.0):
        for q1 in (1.2, 1.3, 1.4):
            v7m = {0: q0, 1: q1, 2: 1.05, 3: 1.00}
            for sc in (1.10, 1.15):
                lbl = "B3_q0%.1f_q1%.2f_sc%.2f" % (q0, q1, sc)
                configs.append({
                    "label": lbl, "group": "B3",
                    "v7_map": v7m, "lev_scale": sc, "posthoc_flag": 0,
                    "params": {"q0": q0, "q1": q1, "lev_scale": sc,
                               "q0_hi": np.nan, "lev_thr": np.nan},
                })

    # B4: conditional boost (posthoc_flag=1)
    print("\nBuilding B4 conditional mult arrays (q0_hi candidates) ...")
    b4_mults = {}
    for q0_hi in (1.6, 1.8, 2.0, 2.4):
        print("  Computing B4 mult for q0_hi=%.1f ..." % q0_hi)
        b4_mults[q0_hi] = _build_v7_mult_cond(dates_dt, q0_hi, lev_raw_masked_arr)
    print("  Done building B4 mults.")

    for q0_hi in (1.6, 1.8, 2.0, 2.4):
        for sc in (1.0, 1.15):
            lbl = "B4_qhi%.1f_sc%.2f" % (q0_hi, sc)
            configs.append({
                "label": lbl, "group": "B4",
                "v7_map": None, "lev_scale": sc, "posthoc_flag": 1,
                "b4_mult": b4_mults[q0_hi],
                "params": {"q0": np.nan, "q1": np.nan, "lev_scale": sc,
                           "q0_hi": q0_hi, "lev_thr": 0.6},
            })

    print("\nTotal sweep configs: %d" % len(configs))
    print("  B2: 12  B3: 24  B4: 8")

    # ------------------------------------------------------------------
    # Build NAVs and collect Stage-1 rows
    # ------------------------------------------------------------------
    sanity_ok = True
    sanity_config = None

    for i, cfg in enumerate(configs):
        lbl = cfg["label"]
        print("  [%d/%d] %s ..." % (i + 1, len(configs), lbl))
        b4_mult = cfg.get("b4_mult", None)
        nav_dt, r, tpy, exc = _build_one(
            shared, dates_dt, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            v7_map=cfg["v7_map"],
            lev_scale=cfg["lev_scale"],
            cfd_excess=True,
            mult_v7_prebuilt=b4_mult,
        )
        row = _stage1_row(lbl, nav_dt, r, tpy, exc, is_mask, oos_mask,
                          cfg["group"], cfg["posthoc_flag"], cfg["params"])
        rows.append(row)

        # Sanity: B2 q0=1.4, q1=1.2, scale=1.0 should match LU1_C1
        if cfg["group"] == "B2" and cfg["params"]["q0"] == 1.4 and cfg["params"]["q1"] == 1.2:
            sanity_config = row
            expected_min9  = 0.1830
            expected_maxdd = -0.3476
            got_min9  = row["min9_at"]
            got_maxdd = row["MaxDD_FULL"]
            tol_min9  = 0.003   # 0.3pp
            tol_maxdd = 0.008   # 0.8pp
            ok_min9  = abs(got_min9  - expected_min9)  <= tol_min9
            ok_maxdd = abs(got_maxdd - expected_maxdd) <= tol_maxdd
            print("\n  SANITY CHECK (B2 q0=1.4,q1=1.2,sc=1.0 == LU1_C1):")
            print("    min9_at:  got=%+.4f%%  expect~%+.4f%%  -> %s"
                  % (got_min9 * 100, expected_min9 * 100, "OK" if ok_min9 else "FAIL"))
            print("    MaxDD:    got=%+.4f%%  expect~%+.4f%%  -> %s"
                  % (got_maxdd * 100, expected_maxdd * 100, "OK" if ok_maxdd else "FAIL"))
            if not (ok_min9 and ok_maxdd):
                print("  SANITY FAILED -- stopping. Investigate _build_one / C1 wiring.")
                sanity_ok = False
                break
            print()

    if not sanity_ok:
        print("Stage 1 aborted due to sanity failure.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Save Stage-1 CSV
    # ------------------------------------------------------------------
    out_csv = os.path.join(out_dir, "leverup_sweep_stage1_20260612.csv")
    df1 = pd.DataFrame(rows)
    df1.to_csv(out_csv, index=False, float_format="%.6f")
    print("\nSaved Stage-1 CSV: %s  (%d rows)" % (out_csv, len(df1)))

    # ------------------------------------------------------------------
    # Stage-1 results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("STAGE 1 RESULTS (min9 descending, after-tax)")
    print("%-30s | %6s | %8s | %9s | %9s | %6s | %8s | %4s | %4s | %4s | %s"
          % ("label", "group", "min9_%", "MaxDD_%", "W10Y*_%", "Trd/yr",
             "gap_pp", "ph", "vMax", "vW10", "pass"))
    print("-" * 120)
    df1_sorted = df1.sort_values("min9_at", ascending=False)
    for _, rw in df1_sorted.iterrows():
        print("%-30s | %-6s | %+8.2f%% | %+8.2f%% | %+8.2f%% | %6.1f | %+7.2f | %4d | %4d | %4d | %s"
              % (str(rw["label"])[:30], str(rw["group"]),
                 rw["min9_at"] * 100, rw["MaxDD_FULL"] * 100,
                 rw["Worst10Y_star_at"] * 100, rw["Trades_yr"],
                 rw.get("gap_pp", 0.0),
                 int(rw["posthoc_flag"]),
                 int(rw["veto_maxdd"]), int(rw["veto_w10y"]),
                 "PASS" if rw["stage1_pass"] else ("veto" if rw["VETO"] else "min9<19")))

    # Reference comparison
    if b1c1_refs:
        print("\nREFERENCE (from B1C1 CSV):")
        for k, v in sorted(b1c1_refs.items(), key=lambda x: -x[1]["min9_at"]):
            print("  %-10s  min9=%+.2f%%  MaxDD=%+.2f%%"
                  % (k, v["min9_at"] * 100, v["MaxDD_FULL"] * 100))

    n_pass = int(df1["stage1_pass"].sum())
    print("\nStage-1 pass count (MaxDD>-50%% AND W10Y*>0 AND min9>=19.0%%): %d / %d"
          % (n_pass, len(df1)))

    # RETURN_BLOCK for Stage 1
    block1 = {
        "stage": 1,
        "n_configs": len(df1),
        "n_stage1_pass": n_pass,
        "sanity_ok": sanity_ok,
        "top5": [],
    }
    for _, rw in df1_sorted.head(5).iterrows():
        block1["top5"].append({
            "label": str(rw["label"]),
            "group": str(rw["group"]),
            "min9_at_pct": round(rw["min9_at"] * 100, 4),
            "MaxDD_pct": round(rw["MaxDD_FULL"] * 100, 4),
            "Worst10Y_star_pct": round(rw["Worst10Y_star_at"] * 100, 4),
            "stage1_pass": int(rw["stage1_pass"]),
            "posthoc_flag": int(rw["posthoc_flag"]),
        })
    print("\n" + "=" * 96)
    print("RETURN_BLOCK Stage 1")
    print("=" * 96)
    print(json.dumps(block1, indent=2))
    return df1


def stage2(shared, dates_dt, ret_gold, ret_bond, fund_active, wg, wb,
           bond_on, sofr_arr, is_mask, oos_mask,
           r_v7, regimes, stress, out_dir, part=None):
    print("=" * 96)
    print("STAGE 2: Full gate (WFA + CPCV + regime + stress + bootstrap)")
    print("=" * 96)

    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)

    # Load Stage-1 CSV
    s1_csv = os.path.join(out_dir, "leverup_sweep_stage1_20260612.csv")
    if not os.path.exists(s1_csv):
        print("ERROR: Stage-1 CSV not found: %s" % s1_csv)
        sys.exit(1)
    df1 = pd.read_csv(s1_csv)

    # Filter Stage-1 pass
    candidates = df1[df1["stage1_pass"] == 1].copy()
    print("\nStage-1 pass candidates: %d" % len(candidates))

    if len(candidates) == 0:
        print("No candidates passed Stage-1 gate. Exiting Stage 2.")
        # Still write empty CSV and RETURN_BLOCK
        out_csv = os.path.join(out_dir, "leverup_sweep_stage2_20260612.csv")
        pd.DataFrame().to_csv(out_csv, index=False)
        block2 = {"stage": 2, "n_candidates": 0, "results": []}
        print("\n" + "=" * 96)
        print("RETURN_BLOCK Stage 2")
        print("=" * 96)
        print(json.dumps(block2, indent=2))
        return

    # If more than 8 candidates, keep top-8 by min9_at
    if len(candidates) > STAGE2_MAX_CANDIDATES:
        print("  Trimming to top-%d by min9_at (was %d)"
              % (STAGE2_MAX_CANDIDATES, len(candidates)))
        candidates = candidates.nlargest(STAGE2_MAX_CANDIDATES, "min9_at")

    # Part filter (e.g. --part 1/2)
    if part:
        part_idx, total_parts = [int(x) for x in part.split("/")]
        all_labels = candidates["label"].tolist()
        chunk_size = int(np.ceil(len(all_labels) / total_parts))
        start = (part_idx - 1) * chunk_size
        end = start + chunk_size
        labels_this_part = all_labels[start:end]
        candidates = candidates[candidates["label"].isin(labels_this_part)]
        print("  Part %d/%d: running %d of %d candidates: %s"
              % (part_idx, total_parts, len(candidates), len(all_labels), labels_this_part))

    print("\nRunning full evaluation on %d candidates:" % len(candidates))
    for _, r in candidates.iterrows():
        print("  %s (group=%s, posthoc=%d, min9=%.2f%%)"
              % (r["label"], r["group"], int(r["posthoc_flag"]), r["min9_at"] * 100))

    # Rebuild B4 mults if needed (Stage 2 is a separate process)
    lev_raw_masked_arr = np.asarray(shared["lev_raw_masked"], float)
    b4_groups_needed = candidates[candidates["group"] == "B4"]["q0_hi"].dropna().unique()
    b4_mults_s2 = {}
    for q0_hi in b4_groups_needed:
        print("  Rebuilding B4 mult q0_hi=%.1f ..." % q0_hi)
        b4_mults_s2[q0_hi] = _build_v7_mult_cond(
            dates_dt, q0_hi, lev_raw_masked_arr)

    rows2 = []
    for _, s1_row in candidates.iterrows():
        lbl = str(s1_row["label"])
        group = str(s1_row["group"])
        posthoc = int(s1_row["posthoc_flag"])
        q0   = float(s1_row.get("q0",   np.nan))
        q1   = float(s1_row.get("q1",   np.nan))
        sc   = float(s1_row.get("lev_scale", 1.0))
        q0hi = float(s1_row.get("q0_hi", np.nan))

        print("\n  Evaluating %s ..." % lbl)

        if group in ("B2", "B3"):
            v7m = {0: q0, 1: q1, 2: 1.05, 3: 1.00}
            nav_dt, r, tpy, exc = _build_one(
                shared, dates_dt, dates_dt, n_years,
                ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
                v7_map=v7m, lev_scale=sc, cfd_excess=True)
        elif group == "B4":
            b4_m = b4_mults_s2.get(q0hi)
            if b4_m is None:
                print("    B4 mult not found for q0_hi=%.1f, rebuilding..." % q0hi)
                b4_m = _build_v7_mult_cond(dates_dt, q0hi, lev_raw_masked_arr)
                b4_mults_s2[q0hi] = b4_m
            nav_dt, r, tpy, exc = _build_one(
                shared, dates_dt, dates_dt, n_years,
                ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
                lev_scale=sc, cfd_excess=True, mult_v7_prebuilt=b4_m)
        else:
            print("    Unknown group %s, skipping" % group)
            continue

        params = {k: s1_row.get(k, np.nan)
                  for k in ("q0", "q1", "lev_scale", "q0_hi", "lev_thr")}
        row2 = _stage2_row(
            lbl, nav_dt, r, regimes, stress, is_mask, oos_mask,
            r_v7, group, posthoc, params, dict(s1_row))
        rows2.append(row2)

        ph_tag = " [posthoc WARNING]" if posthoc else ""
        print("    min9=%.2f%%  WFE=%.4f  CI95_lo=%.2f%%  CPCV_p10=%.2f%%  Regime_min=%.2f%%  VETO=%d%s"
              % (row2["min9_at"] * 100,
                 row2["wfa_WFE"],
                 row2["wfa_CI95_lo"] * 100,
                 row2["cpcv_p10_at"] * 100,
                 row2["regime_min_at"] * 100,
                 row2["S2_VETO"], ph_tag))

    # Save Stage-2 CSV
    out_csv = os.path.join(out_dir, "leverup_sweep_stage2_20260612.csv")
    df2 = pd.DataFrame(rows2)
    if os.path.exists(out_csv) and part:
        # Append to existing partial results
        df_existing = pd.read_csv(out_csv)
        df2 = pd.concat([df_existing, df2], ignore_index=True)
    df2.to_csv(out_csv, index=False, float_format="%.6f")
    print("\nSaved Stage-2 CSV: %s  (%d rows)" % (out_csv, len(df2)))

    # Full gate table
    print("\n" + "=" * 130)
    print("STAGE 2 FULL GATE RESULTS")
    print("%-30s | %6s | %8s | %7s | %8s | %8s | %8s | %9s | %4s | %4s | %s"
          % ("label", "group", "min9_%", "WFE", "CI95_%", "CPCV_p10%",
             "Reg_min%", "W10Y*%", "ph", "VETO", "notes"))
    print("-" * 130)
    df2_sorted = df2.sort_values("min9_at", ascending=False) if len(df2) else df2
    for _, rw in df2_sorted.iterrows():
        ph = int(rw.get("posthoc_flag", 0))
        veto = int(rw.get("S2_VETO", 0))
        note = "posthoc" if ph else ("VETO" if veto else "PASS")
        print("%-30s | %-6s | %+7.2f%% | %7.4f | %+7.2f%% | %+8.2f%% | %+7.2f%% | %+8.2f%% | %4d | %4d | %s"
              % (str(rw["label"])[:30], str(rw["group"]),
                 rw["min9_at"] * 100,
                 rw.get("wfa_WFE", float("nan")),
                 rw.get("wfa_CI95_lo", float("nan")) * 100,
                 rw.get("cpcv_p10_at", float("nan")) * 100,
                 rw.get("regime_min_at", float("nan")) * 100,
                 rw.get("Worst10Y_star_at", float("nan")) * 100,
                 ph, veto, note))

    # RETURN_BLOCK Stage 2
    block2 = {
        "stage": 2,
        "n_candidates": len(candidates),
        "n_evaluated": len(rows2),
        "results": [],
    }
    for rw in rows2:
        entry = {
            "label":           str(rw["label"]),
            "group":           str(rw["group"]),
            "posthoc_flag":    int(rw["posthoc_flag"]),
            "min9_at_pct":     round(float(rw["min9_at"]) * 100, 4),
            "MaxDD_pct":       round(float(rw["MaxDD_FULL"]) * 100, 4),
            "Worst10Y_star_pct": round(float(rw["Worst10Y_star_at"]) * 100, 4),
            "wfa_WFE":         round(float(rw["wfa_WFE"]), 4),
            "wfa_CI95_lo_pct": round(float(rw["wfa_CI95_lo"]) * 100, 4),
            "cpcv_p10_at_pct": round(float(rw["cpcv_p10_at"]) * 100, 4),
            "regime_min_at_pct": round(float(rw["regime_min_at"]) * 100, 4),
            "S2_VETO":         int(rw["S2_VETO"]),
            "boot_P_min_better": rw.get("boot_P_min_better", ""),
            "boot_CI95_lo_min_pp": rw.get("boot_CI95_lo_min_pp", ""),
        }
        if rw.get("posthoc_flag"):
            entry["posthoc_warning"] = (
                "POSTHOC: criterion discovered post-hoc on same OOS data. "
                "Single-OOS adoption NOT permitted; WFA/CPCV shown for "
                "information only. Requires additional independent validation."
            )
        block2["results"].append(entry)

    print("\n" + "=" * 96)
    print("RETURN_BLOCK Stage 2")
    print("=" * 96)
    print(json.dumps(block2, indent=2))
    return df2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="B2-B4 lever-up grid sweep (leverup_sweep_20260612)")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="1 = NAV sweep + Stage-1 metrics only; "
                             "2 = full gate on Stage-1 pass candidates")
    parser.add_argument("--part", type=str, default=None,
                        help="Partition for Stage 2 (e.g. '1/2', '2/2')")
    args = parser.parse_args()

    print("=" * 96)
    print("LEVERUP B2-B4 SWEEP  2026-06-12  (stage=%d%s)"
          % (args.stage, "  part=%s" % args.part if args.part else ""))
    print("=" * 96)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)

    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    if args.stage == 1:
        stage1(shared, dates_dt, ret_gold, ret_bond, fund_active, wg, wb,
               bond_on, sofr_arr, is_mask, oos_mask, out_dir)

    elif args.stage == 2:
        # Build V7_TQQQ baseline for bootstrap comparison
        v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
            shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)
        # Regime labels and stress masks
        regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
        stress = stress_masks(dates_dt)

        stage2(shared, dates_dt, ret_gold, ret_bond, fund_active, wg, wb,
               bond_on, sofr_arr, is_mask, oos_mask,
               r_v7, regimes, stress, out_dir, part=args.part)

    print("\nDone.")


if __name__ == "__main__":
    main()
