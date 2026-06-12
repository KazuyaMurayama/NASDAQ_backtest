"""
src/audit/k365_recost_20260612.py
===================================
Re-cost evaluation: replace >3x leverage excess cost with
"くりっく株365 NASDAQ-100 (取引所CFD)" financing rate.

Purpose (§9 of PRODUCT_COST_COMPARISON_2026-06-10.md):
  くりっく株365 金利相当額 ≈ 4.38%/yr (SOFR+0.75pp, 週次公表).
  店頭CFD (SOFR+3.0pp) に比べ上乗せ幅が約1/4.

  Currently, EXCESS_EXTRA = CFD_RATE - TQQQ_SWAP = 3.0% - 0.5% = 2.5%/yr
  charges the (L-3)+ NASDAQ notional beyond TQQQ cost.

  For くりっく株365:
    EXCESS_EXTRA_K365 = k365_spread - TQQQ_SWAP
    = (0.75pp) - 0.5% = 0.25%/yr  [centre case]
    = (1.00pp) - 0.5% = 0.50%/yr  [sensitivity: spread 1.0pp]
    = (1.50pp) - 0.5% = 1.00%/yr  [sensitivity: spread 1.5pp]

  ≤3x leverage continues to use TQQQ pricing (unchanged).
  Trading cost: くりっく株365 片道¥30/枚 ≈ 0.014%, cheaper than current model
  assumption → this direction is already conservative (cost model is safe-side).

Sanity gate:
  EXCESS_EXTRA=2.5% must reproduce LU2_C1 known values from leverup_b1c1_20260612.csv:
    min(IS,OOS) aftertax ~+19.09%, MaxDD ~-38.49%
  If mismatch > tolerance, script halts.

Cautions (document in use):
  - 年次リセット強制課税: the current aftertax model applies ×0.8273 multiplicative
    scaling; this is consistent with annual forced-reset CFD taxation.
  - 板の薄さ: くりっく株365 volume is currently thin relative to NASDAQ-100 daily
    exposure; large position sizes may face adverse fills not modelled here.
  - 利率週次変動: the 0.75pp / 1.0pp / 1.5pp spreads are point-in-time estimates
    from 2026-06 price sheets plus 2024/2025 historical observation; actual rates
    reset weekly and may widen.
  - スプレッド推定は 2026-06 単一時点 + 2024/2025 実績由来; not a forward commitment.

Outputs:
  audit_results/k365_recost_20260612.csv  -- store_cfd reference rows + k365 centre +
                                              sensitivity rows; full-gate columns included
  RETURN_BLOCK printed to stdout (json.dumps)

ASCII-only prints (Windows cp932). Does NOT commit. No temp files.
"""

from __future__ import annotations

import json
import os
import sys
import types

# ---- multitasking stub -------------------------------------------------------
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

# ---- Re-use builders from leverup_b1c1_20260612 and lu_cfd_recost_20260611 --
from src.audit.lu_cfd_recost_20260611 import (
    _build_p09_on_base, _metrics_pack, LU1_MAP,
    AFTER_TAX, SWAP_SPREAD, TER_TQQQ, LEV_CAP,
    _build_nav_v7_tqqq,
)
from src.audit.leverup_b1c1_20260612 import (
    _build_p09_nav_c1, _build_p09_on_base_c1,
)
from src.audit.extended_eval_20260611 import (
    _eval_one, _cpcv_dist, _regime_cagr,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _run_wfa, _block_bootstrap_compare, LU2_SCALE, _cagr_seg, _maxdd_from_returns,
    _build_v7_mult_custom,
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

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------
# Store CFD (現行): SOFR+3.0pp (超過コスト = 3.0% - 0.5% = 2.5%/yr)
CFD_RATE_STORE = 0.030
TQQQ_SWAP = SWAP_SPREAD  # 0.0050
EXCESS_EXTRA_STORE = CFD_RATE_STORE - TQQQ_SWAP  # 0.025

# くりっく株365 spread scenarios (over SOFR)
K365_SPREAD_CENTRE = 0.0075   # 0.75pp -> EXCESS = 0.25%/yr
K365_SPREAD_SEN1   = 0.0100   # 1.00pp -> EXCESS = 0.50%/yr
K365_SPREAD_SEN2   = 0.0150   # 1.50pp -> EXCESS = 1.00%/yr

EXCESS_EXTRA_K365_CENTRE = K365_SPREAD_CENTRE - TQQQ_SWAP  # 0.0025
EXCESS_EXTRA_K365_SEN1   = K365_SPREAD_SEN1   - TQQQ_SWAP  # 0.0050
EXCESS_EXTRA_K365_SEN2   = K365_SPREAD_SEN2   - TQQQ_SWAP  # 0.0100


# ---------------------------------------------------------------------------
# Parameterised NAV builder with configurable EXCESS_EXTRA
# ---------------------------------------------------------------------------

def _build_nav_v7_tqqq_param(close, dates, gold_2x_nav, bond_3x_nav, sofr_daily,
                              lev_raw_masked, wn, wg, wb, mult_v7,
                              excess_extra=EXCESS_EXTRA_STORE):
    """Like _build_nav_v7_tqqq but with parameterisable excess_extra.

    When excess_extra == EXCESS_EXTRA_STORE (0.025) this reproduces the store-CFD result.
    When excess_extra == EXCESS_EXTRA_K365_CENTRE (0.0025) it models くりっく株365.

    excess_extra=0.0 would mean >3x is also at TQQQ cost (lower bound / hypothetical).
    excess_extra must be >= 0.
    """
    from src.audit.strategy_runners import (
        _TER_GOLD2X_EXTRA_DAILY, _TER_TMF_EXTRA_DAILY,
        _compute_dhw1_trades_per_year,
    )
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual

    idx = dates.index
    n = len(close)
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(np.asarray(gold_2x_nav, float)).pct_change().fillna(0).values
    r_b3 = pd.Series(np.asarray(bond_3x_nav, float)).pct_change().fillna(0).values

    lev_mod = np.asarray(lev_raw_masked, float) * np.asarray(mult_v7, float)
    L = pd.Series(lev_mod * 3.0, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(V7_DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(V7_DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(V7_DELAY).fillna(0).values
    sofr_arr = np.asarray(sofr_daily, float)

    # TQQQ leg (identical to original)
    borrow = np.maximum(L - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
    nas_ret = L * r_nas - borrow - TER_TQQQ / TRADING_DAYS

    daily = wn_s * nas_ret + wg_s * r_g2 + wb_s * r_b3

    # Excess penalty (parameterised)
    excess_lev = np.maximum(L - LEV_CAP, 0.0)
    penalty = wn_s * excess_lev * float(excess_extra) / TRADING_DAYS
    daily = daily - penalty
    excess_days = int(np.sum(excess_lev > 1e-9))

    dwn = np.zeros(n); dwn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    dwg = np.zeros(n); dwg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    dwb = np.zeros(n); dwb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = np.nan_to_num(dwn + dwg + dwb, nan=0.0)

    daily = np.maximum(daily, NAV_FLOOR)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    ter_drag = (np.asarray(wg, float) * _TER_GOLD2X_EXTRA_DAILY
                + np.asarray(wb, float) * _TER_TMF_EXTRA_DAILY)
    tpy = _compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / 252.0
    r_sim = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_sim - ter_drag - etf_daily
    nav_adj = pd.Series(
        np.cumprod(1.0 + r_adj),
        index=pd.DatetimeIndex(pd.to_datetime(dates.values)),
    )
    return nav_adj, tpy, excess_days


def _build_tqqq_base_param(shared, date_index, v7_map=None, lev_scale=1.0,
                            excess_extra=EXCESS_EXTRA_STORE):
    """Build TQQQ base NAV with parameterised excess_extra."""
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

    if v7_map is None:
        mult_v7 = _build_v7_mult(date_index)
    else:
        mult_v7 = _build_v7_mult_custom(date_index, v7_map)
    mult_v7 = mult_v7 * float(lev_scale)

    nav_dt, tpy, excess_days = _build_nav_v7_tqqq_param(
        close, dates, gold_2x, bond_3x, sofr,
        lev_raw_masked, wn, wg, wb, mult_v7,
        excess_extra=excess_extra)
    r_base = nav_dt.pct_change().fillna(0).values
    return nav_dt, r_base, tpy, excess_days


def _build_full_c1(shared, dates_dt, n_years,
                   ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
                   v7_map=None, lev_scale=1.0, excess_extra=EXCESS_EXTRA_STORE):
    """Build complete C1 NAV (TQQQ base + C1 OUT fill) for one config."""
    base_nav, r_base, tpy_b, exc = _build_tqqq_base_param(
        shared, dates_dt, v7_map=v7_map, lev_scale=lev_scale,
        excess_extra=excess_extra)
    nav_dt, r, tpy = _build_p09_on_base_c1(
        r_base, ret_gold, ret_bond, fund_active, wg, wb, bond_on,
        sofr_arr, dates_dt, tpy_b, n_years)
    return nav_dt, r, tpy, exc


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


# ---------------------------------------------------------------------------
# Main evaluation configs
# ---------------------------------------------------------------------------

# k365 centre configs (5 candidates, full gate)
K365_CENTRE_CONFIGS = [
    {"label": "LU1_C1_k365",  "v7_map": LU1_MAP,
     "lev_scale": 1.0,  "excess_extra": EXCESS_EXTRA_K365_CENTRE},
    {"label": "LU2_C1_k365",  "v7_map": None,
     "lev_scale": LU2_SCALE, "excess_extra": EXCESS_EXTRA_K365_CENTRE},
    {"label": "B3a_k365",     "v7_map": {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.15, "excess_extra": EXCESS_EXTRA_K365_CENTRE},
    {"label": "B3b_k365",     "v7_map": {0: 2.0, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.15, "excess_extra": EXCESS_EXTRA_K365_CENTRE},
    {"label": "B3c_k365",     "v7_map": {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.10, "excess_extra": EXCESS_EXTRA_K365_CENTRE},
]

# store-CFD reference configs (matching labels for comparison)
STORE_CFD_CONFIGS = [
    {"label": "LU1_C1_store",  "v7_map": LU1_MAP,
     "lev_scale": 1.0,  "excess_extra": EXCESS_EXTRA_STORE},
    {"label": "LU2_C1_store",  "v7_map": None,
     "lev_scale": LU2_SCALE, "excess_extra": EXCESS_EXTRA_STORE},
    {"label": "B3a_store",     "v7_map": {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.15, "excess_extra": EXCESS_EXTRA_STORE},
    {"label": "B3b_store",     "v7_map": {0: 2.0, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.15, "excess_extra": EXCESS_EXTRA_STORE},
    {"label": "B3c_store",     "v7_map": {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.10, "excess_extra": EXCESS_EXTRA_STORE},
]

# sensitivity configs (centre k365 candidates x 2 spread levels; standard-10 only, no WFA/CPCV)
K365_SEN1_CONFIGS = [
    {"label": "LU1_C1_k365_sen1",  "v7_map": LU1_MAP,
     "lev_scale": 1.0,  "excess_extra": EXCESS_EXTRA_K365_SEN1},
    {"label": "LU2_C1_k365_sen1",  "v7_map": None,
     "lev_scale": LU2_SCALE, "excess_extra": EXCESS_EXTRA_K365_SEN1},
    {"label": "B3a_k365_sen1",     "v7_map": {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.15, "excess_extra": EXCESS_EXTRA_K365_SEN1},
    {"label": "B3b_k365_sen1",     "v7_map": {0: 2.0, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.15, "excess_extra": EXCESS_EXTRA_K365_SEN1},
    {"label": "B3c_k365_sen1",     "v7_map": {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.10, "excess_extra": EXCESS_EXTRA_K365_SEN1},
]

K365_SEN2_CONFIGS = [
    {"label": "LU1_C1_k365_sen2",  "v7_map": LU1_MAP,
     "lev_scale": 1.0,  "excess_extra": EXCESS_EXTRA_K365_SEN2},
    {"label": "LU2_C1_k365_sen2",  "v7_map": None,
     "lev_scale": LU2_SCALE, "excess_extra": EXCESS_EXTRA_K365_SEN2},
    {"label": "B3a_k365_sen2",     "v7_map": {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.15, "excess_extra": EXCESS_EXTRA_K365_SEN2},
    {"label": "B3b_k365_sen2",     "v7_map": {0: 2.0, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.15, "excess_extra": EXCESS_EXTRA_K365_SEN2},
    {"label": "B3c_k365_sen2",     "v7_map": {0: 1.4, 1: 1.40, 2: 1.05, 3: 1.00},
     "lev_scale": 1.10, "excess_extra": EXCESS_EXTRA_K365_SEN2},
]

# Hard veto thresholds (same as leverup_b1c1)
HARD_VETO_MAXDD  = -0.50
HARD_VETO_WFE    = 1.5
HARD_VETO_W10Y   = 0.0
HARD_VETO_REGIME = -0.10


def _eval_full_gate(label, nav_dt, r, tpy, regimes, stress, is_mask, oos_mask, r_v7, n_out_bondoff):
    """Run full gate and return (packs, ev, veto_map) dict."""
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    cy = _calendar_year_returns(nav_dt)

    ev = _eval_one(label, nav_dt, r, regimes, stress, is_mask, oos_mask, baseline_r=r_v7)

    maxdd = pre["MaxDD_FULL"]
    wfe = float(ev["wfa_WFE"])
    w10y = aft["Worst10Y_star"]
    reg_min = float(ev["regime_min_at"])

    v_maxdd = maxdd < HARD_VETO_MAXDD
    v_wfe   = wfe > HARD_VETO_WFE
    v_w10y  = w10y < HARD_VETO_W10Y
    v_reg   = reg_min < HARD_VETO_REGIME
    veto = v_maxdd or v_wfe or v_w10y or v_reg

    return {
        "pre": pre,
        "aft": aft,
        "cy": cy,
        "ev": ev,
        "veto": {
            "maxdd": v_maxdd, "wfe": v_wfe, "w10y": v_w10y, "reg": v_reg, "VETO": veto,
        },
    }


def _metrics_only(label, nav_dt, tpy):
    """Standard-10 metrics only (no WFA/CPCV/bootstrap)."""
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    cy = _calendar_year_returns(nav_dt)
    return {"pre": pre, "aft": aft, "cy": cy}


def _to_csv_row(label, packs, ev_result=None, n_out_bondoff=0,
                excess_days=0, spread_pp=None, scenario="centre"):
    """Assemble a CSV row dict from packs + optional full-gate ev_result."""
    aft = packs["aft"]
    pre = packs["pre"]
    cy = packs["cy"]
    mn = _min_at(aft)

    row = {
        "label": label,
        "scenario": scenario,
        "spread_pp": spread_pp,
        "excess_extra_pct": (
            EXCESS_EXTRA_STORE * 100 if scenario == "store_cfd"
            else EXCESS_EXTRA_K365_CENTRE * 100 if scenario == "centre"
            else EXCESS_EXTRA_K365_SEN1 * 100 if scenario == "sen1"
            else EXCESS_EXTRA_K365_SEN2 * 100
        ),
        "CAGR_IS_at": aft["CAGR_IS"],
        "CAGR_OOS_at": aft["CAGR_OOS"],
        "min_IS_OOS_at": mn,
        "IS_OOS_gap_pp": aft["IS_OOS_gap_pp"],
        "Sharpe_OOS": pre["Sharpe_OOS"],
        "MaxDD_FULL": pre["MaxDD_FULL"],
        "Worst10Y_star_at": aft["Worst10Y_star"],
        "P10_5Y_at": aft["P10_5Y"],
        "Worst5Y_at": aft["Worst5Y"],
        "Trades_yr": aft["Trades_yr"],
        "worst_cy": float(cy.min()),
        "worst_cy_year": int(cy.idxmin()),
        "excess_days": excess_days,
        "OUT_bondOFF_days": n_out_bondoff,
        # Full-gate columns (empty for sensitivity rows)
        "wfa_WFE": "",
        "wfa_CI95_lo": "",
        "wfa_t_p": "",
        "cpcv_p10_at": "",
        "cpcv_worst_at": "",
        "cpcv_med_at": "",
        "regime_min_at": "",
        "boot_P_min_better": "",
        "boot_CI95_lo_min_pp": "",
        "veto_maxdd": "",
        "veto_wfe": "",
        "veto_w10y": "",
        "veto_reg": "",
        "VETO": "",
    }

    if ev_result is not None:
        ev = ev_result["ev"]
        vm = ev_result["veto"]
        boot = ev.get("boot") or {}
        row.update({
            "wfa_WFE":         float(ev["wfa_WFE"]),
            "wfa_CI95_lo":     float(ev["wfa_CI95_lo"]),
            "wfa_t_p":         float(ev["wfa_t_p"]),
            "cpcv_p10_at":     float(ev["cpcv_p10_at"]),
            "cpcv_worst_at":   float(ev["cpcv_worst_at"]),
            "cpcv_med_at":     float(ev["cpcv_med_at"]),
            "regime_min_at":   float(ev["regime_min_at"]),
            "boot_P_min_better":   boot.get("P_min_better", ""),
            "boot_CI95_lo_min_pp": boot.get("CI95_lo_min_pp", ""),
            "veto_maxdd": int(vm["maxdd"]),
            "veto_wfe":   int(vm["wfe"]),
            "veto_w10y":  int(vm["w10y"]),
            "veto_reg":   int(vm["reg"]),
            "VETO":       int(vm["VETO"]),
        })

        # Regime breakdowns
        axes_order = ["trend:bull", "trend:bear", "vol:calm", "vol:highvol",
                      "rate:rate_up", "rate:rate_down"]
        for ax in axes_order:
            row["regime_" + ax.replace(":", "_")] = ev["regime"].get(ax, np.nan)

        # Stress windows
        stress_keys = list(ev["stress"].keys())
        for sw in stress_keys:
            row["stress_%s_ret" % sw]   = ev["stress"][sw]["ret"]
            row["stress_%s_maxdd" % sw] = ev["stress"][sw]["maxdd"]

    return row


def main():
    print("=" * 100)
    print("K365 RECOST EVALUATION  2026-06-12")
    print("Replace >3x excess cost: store-CFD 2.5%%/yr -> k365 centre 0.25%%/yr")
    print("EXCESS_EXTRA: store=%.4f  k365_centre=%.4f  sen1=%.4f  sen2=%.4f"
          % (EXCESS_EXTRA_STORE, EXCESS_EXTRA_K365_CENTRE,
             EXCESS_EXTRA_K365_SEN1, EXCESS_EXTRA_K365_SEN2))
    print("=" * 100)

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

    # ---- Gold/Bond 1x legs ----
    from compute_cfd_worst10y import prepare_gold_local
    from corrected_strategy_backtest import build_bond_1x_nav_corrected
    gold_1x = np.asarray(prepare_gold_local(dates), float)
    bond_1x = np.asarray(
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    out_bond_off = fund_active & (~bond_on.astype(bool))
    n_out_bondoff = int(out_bond_off.sum())
    print("\nOUT-and-bondOFF days: %d of %d (%.1f%%)" % (n_out_bondoff, n, 100.0 * n_out_bondoff / n))

    # ---- V7_TQQQ baseline for bootstrap ----
    print("\nBuilding V7_TQQQ baseline ...")
    from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    # ---- Regime labels and stress masks ----
    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)

    # =========================================================================
    # SANITY CHECK
    # Reproduce LU2_C1 known values with EXCESS_EXTRA = 2.5%
    # Known: min(IS,OOS) aftertax ~ +19.09%, MaxDD ~ -38.49%
    # =========================================================================
    print("\n" + "=" * 100)
    print("SANITY CHECK: EXCESS_EXTRA=2.5%% must reproduce LU2_C1 from leverup_b1c1_20260612.csv")
    print("  Expected: min_at~+19.09%, MaxDD~-38.49%")
    print("=" * 100)

    sanity_nav, sanity_r, sanity_tpy, sanity_exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
        v7_map=None, lev_scale=LU2_SCALE, excess_extra=EXCESS_EXTRA_STORE)

    sanity_pre = compute_10metrics(sanity_nav, sanity_tpy)
    sanity_aft = _apply_aftertax(sanity_pre)
    sanity_min = _min_at(sanity_aft)
    sanity_maxdd = sanity_pre["MaxDD_FULL"]

    KNOWN_MIN_AT  = 0.1909   # +19.09%
    KNOWN_MAXDD   = -0.3849  # -38.49%
    TOL_MIN9  = 0.005  # 0.5pp
    TOL_MAXDD = 0.010  # 1.0pp

    ok_min  = abs(sanity_min  - KNOWN_MIN_AT)  <= TOL_MIN9
    ok_maxdd = abs(sanity_maxdd - KNOWN_MAXDD) <= TOL_MAXDD

    print("  LU2_C1 sanity: min_at=%+.4f%% (expect ~+19.09%%) -> %s"
          % (sanity_min * 100, "OK" if ok_min else "FAIL"))
    print("  LU2_C1 sanity: MaxDD=%+.4f%% (expect ~-38.49%%) -> %s"
          % (sanity_maxdd * 100, "OK" if ok_maxdd else "FAIL"))

    if not (ok_min and ok_maxdd):
        print("\nSANITY FAILED -- halting. Check _build_full_c1 / EXCESS_EXTRA wiring.")
        import sys; sys.exit(1)

    print("  SANITY PASSED. Proceeding.\n")

    # =========================================================================
    # PHASE 1: Store-CFD reference rows (full gate)
    # =========================================================================
    print("=" * 100)
    print("PHASE 1: Building store-CFD reference rows (full gate, 5 configs)")
    print("=" * 100)
    store_results = {}
    for cfg in STORE_CFD_CONFIGS:
        lbl = cfg["label"]
        print("  Building %s ..." % lbl)
        nav_dt, r, tpy, exc = _build_full_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            v7_map=cfg["v7_map"], lev_scale=cfg["lev_scale"],
            excess_extra=cfg["excess_extra"])
        print("    Running full gate ...")
        ev_res = _eval_full_gate(lbl, nav_dt, r, tpy, regimes, stress,
                                 is_mask, oos_mask, r_v7, n_out_bondoff)
        store_results[lbl] = {"nav": nav_dt, "r": r, "tpy": tpy, "exc": exc,
                              "packs": {"pre": ev_res["pre"], "aft": ev_res["aft"], "cy": ev_res["cy"]},
                              "ev_res": ev_res}
        mn = _min_at(ev_res["aft"])
        veto = ev_res["veto"]["VETO"]
        print("    min_at=%+.2f%%  MaxDD=%+.2f%%  WFE=%.4f  CI95_lo=%+.2f%%  VETO=%s"
              % (mn * 100, ev_res["pre"]["MaxDD_FULL"] * 100,
                 float(ev_res["ev"]["wfa_WFE"]), float(ev_res["ev"]["wfa_CI95_lo"]) * 100,
                 "YES" if veto else "no"))

    # =========================================================================
    # PHASE 2: k365 centre rows (full gate)
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 2: Building k365 centre rows (full gate, spread=0.75pp, EXCESS_EXTRA=0.25%%/yr)")
    print("=" * 100)
    k365_centre_results = {}
    for cfg in K365_CENTRE_CONFIGS:
        lbl = cfg["label"]
        print("  Building %s ..." % lbl)
        nav_dt, r, tpy, exc = _build_full_c1(
            shared, dates_dt, n_years,
            ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
            v7_map=cfg["v7_map"], lev_scale=cfg["lev_scale"],
            excess_extra=cfg["excess_extra"])
        print("    Running full gate ...")
        ev_res = _eval_full_gate(lbl, nav_dt, r, tpy, regimes, stress,
                                 is_mask, oos_mask, r_v7, n_out_bondoff)
        k365_centre_results[lbl] = {"nav": nav_dt, "r": r, "tpy": tpy, "exc": exc,
                                    "packs": {"pre": ev_res["pre"], "aft": ev_res["aft"], "cy": ev_res["cy"]},
                                    "ev_res": ev_res}
        mn = _min_at(ev_res["aft"])
        veto = ev_res["veto"]["VETO"]
        print("    min_at=%+.2f%%  MaxDD=%+.2f%%  WFE=%.4f  CI95_lo=%+.2f%%  VETO=%s"
              % (mn * 100, ev_res["pre"]["MaxDD_FULL"] * 100,
                 float(ev_res["ev"]["wfa_WFE"]), float(ev_res["ev"]["wfa_CI95_lo"]) * 100,
                 "YES" if veto else "no"))

    # =========================================================================
    # PHASE 3: Sensitivity rows (standard-10 only; no WFA/CPCV)
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 3: Sensitivity rows (standard-10 only, no WFA/CPCV)")
    print("  sen1: spread=1.0pp, EXCESS_EXTRA=0.50%%/yr")
    print("  sen2: spread=1.5pp, EXCESS_EXTRA=1.00%%/yr")
    print("=" * 100)

    sen_results = {}
    for scenario_name, configs_list, xtra in [
        ("sen1", K365_SEN1_CONFIGS, EXCESS_EXTRA_K365_SEN1),
        ("sen2", K365_SEN2_CONFIGS, EXCESS_EXTRA_K365_SEN2),
    ]:
        for cfg in configs_list:
            lbl = cfg["label"]
            print("  Building %s ..." % lbl)
            nav_dt, r, tpy, exc = _build_full_c1(
                shared, dates_dt, n_years,
                ret_gold, ret_bond, fund_active, wg, wb, bond_on, sofr_arr,
                v7_map=cfg["v7_map"], lev_scale=cfg["lev_scale"],
                excess_extra=cfg["excess_extra"])
            packs_only = _metrics_only(lbl, nav_dt, tpy)
            sen_results[lbl] = {"nav": nav_dt, "r": r, "tpy": tpy, "exc": exc,
                                "packs": packs_only, "ev_res": None}
            mn = _min_at(packs_only["aft"])
            print("    min_at=%+.2f%%  MaxDD=%+.2f%%  (no WFA)"
                  % (mn * 100, packs_only["pre"]["MaxDD_FULL"] * 100))

    # =========================================================================
    # Summary tables
    # =========================================================================
    print("\n" + "=" * 100)
    print("CENTRE CASE COMPARISON: store_cfd vs k365 centre")
    print("%-20s | %9s | %8s | %8s | %8s | %8s | %7s | %-8s"
          % ("label", "scenario", "min_at%", "MaxDD%", "CI95_lo%", "WFE", "Trd/yr", "VETO"))
    print("-" * 100)

    CENTRE_PAIRS = [
        ("LU1_C1_store",  "LU1_C1_k365"),
        ("LU2_C1_store",  "LU2_C1_k365"),
        ("B3a_store",     "B3a_k365"),
        ("B3b_store",     "B3b_k365"),
        ("B3c_store",     "B3c_k365"),
    ]

    improvement_pp = {}
    for store_lbl, k365_lbl in CENTRE_PAIRS:
        for lbl, scen, results in [
            (store_lbl, "store_cfd", store_results),
            (k365_lbl,  "k365_ctr",  k365_centre_results),
        ]:
            p = results[lbl]["packs"]
            ev_r = results[lbl]["ev_res"]
            mn = _min_at(p["aft"])
            maxdd = p["pre"]["MaxDD_FULL"]
            wfe = float(ev_r["ev"]["wfa_WFE"])
            ci95 = float(ev_r["ev"]["wfa_CI95_lo"])
            tpy = p["aft"]["Trades_yr"]
            veto = "VETO" if ev_r["veto"]["VETO"] else "pass"
            print("%-20s | %-9s | %+8.2f%% | %+7.2f%% | %+7.2f%% | %7.4f | %7.1f | %-8s"
                  % (lbl, scen, mn * 100, maxdd * 100, ci95 * 100, wfe, tpy, veto))
        # improvement
        mn_store = _min_at(store_results[store_lbl]["packs"]["aft"])
        mn_k365  = _min_at(k365_centre_results[k365_lbl]["packs"]["aft"])
        delta = (mn_k365 - mn_store) * 100
        improvement_pp[k365_lbl] = delta
        print("  => k365 improvement: %+.3f pp (must be positive)" % delta)
        if delta < 0:
            print("  WARNING: negative improvement -- check excess_extra sign")
        print()

    # Sensitivity table
    print("=" * 100)
    print("SENSITIVITY TABLE (standard-10 only; spread 1.0pp / 1.5pp)")
    print("%-25s | %9s | %9s | %9s | %9s | %9s"
          % ("label", "min_at%", "MaxDD%", "W10Y*%", "P10_5Y%", "gap_pp"))
    print("-" * 100)
    for lbl, res in sorted(sen_results.items()):
        p = res["packs"]
        mn = _min_at(p["aft"])
        print("%-25s | %+8.2f%% | %+8.2f%% | %+8.2f%% | %+8.2f%% | %+8.2f"
              % (lbl, mn * 100, p["pre"]["MaxDD_FULL"] * 100,
                 p["aft"]["Worst10Y_star"] * 100, p["aft"]["P10_5Y"] * 100,
                 p["aft"]["IS_OOS_gap_pp"]))

    # =========================================================================
    # CSV output
    # =========================================================================
    print("\nBuilding CSV ...")
    rows = []

    # store-CFD reference rows (full gate)
    for store_lbl, k365_lbl in CENTRE_PAIRS:
        res = store_results[store_lbl]
        row = _to_csv_row(
            store_lbl, res["packs"], ev_result=res["ev_res"],
            n_out_bondoff=n_out_bondoff, excess_days=res["exc"],
            spread_pp=None, scenario="store_cfd")
        rows.append(row)

    # k365 centre rows (full gate)
    for k365_lbl in [c["label"] for c in K365_CENTRE_CONFIGS]:
        res = k365_centre_results[k365_lbl]
        row = _to_csv_row(
            k365_lbl, res["packs"], ev_result=res["ev_res"],
            n_out_bondoff=n_out_bondoff, excess_days=res["exc"],
            spread_pp=0.75, scenario="centre")
        row["k365_improvement_pp"] = improvement_pp.get(k365_lbl, "")
        rows.append(row)

    # sensitivity rows (metrics only)
    for lbl, res in sorted(sen_results.items()):
        spread = 1.0 if "sen1" in lbl else 1.5
        scen = "sen1" if "sen1" in lbl else "sen2"
        row = _to_csv_row(
            lbl, res["packs"], ev_result=None,
            n_out_bondoff=n_out_bondoff, excess_days=res["exc"],
            spread_pp=spread, scenario=scen)
        rows.append(row)

    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "k365_recost_20260612.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    print("Saved CSV: %s  (%d rows)" % (csv_path, len(rows)))

    # =========================================================================
    # RETURN BLOCK
    # =========================================================================
    def _rblock_full(lbl, res):
        p = res["packs"]
        ev_r = res["ev_res"]
        ev = ev_r["ev"]
        vm = ev_r["veto"]
        boot = ev.get("boot") or {}
        aft = p["aft"]; pre = p["pre"]
        return {
            "CAGR_IS_at":   round(float(aft["CAGR_IS"]), 6),
            "CAGR_OOS_at":  round(float(aft["CAGR_OOS"]), 6),
            "min_at":       round(float(_min_at(aft)), 6),
            "gap_pp":       round(float(aft["IS_OOS_gap_pp"]), 4),
            "Sharpe":       round(float(pre["Sharpe_OOS"]), 4),
            "MaxDD":        round(float(pre["MaxDD_FULL"]), 6),
            "Worst10Y_at":  round(float(aft["Worst10Y_star"]), 6),
            "P10_5Y_at":    round(float(aft["P10_5Y"]), 6),
            "Trades_yr":    round(float(aft["Trades_yr"]), 2),
            "wfa_WFE":      round(float(ev["wfa_WFE"]), 4),
            "wfa_CI95_lo":  round(float(ev["wfa_CI95_lo"]), 6),
            "wfa_t_p":      round(float(ev["wfa_t_p"]), 4),
            "cpcv_p10_at":  round(float(ev["cpcv_p10_at"]), 6),
            "cpcv_worst_at": round(float(ev["cpcv_worst_at"]), 6),
            "cpcv_med_at":  round(float(ev["cpcv_med_at"]), 6),
            "regime_min_at": round(float(ev["regime_min_at"]), 6),
            "boot_P_min_better": round(float(boot["P_min_better"]), 4) if boot else None,
            "boot_CI95_lo_min_pp": round(float(boot["CI95_lo_min_pp"]), 4) if boot else None,
            "VETO": vm["VETO"],
            "excess_days": res["exc"],
        }

    def _rblock_metrics(lbl, res):
        p = res["packs"]
        aft = p["aft"]; pre = p["pre"]
        return {
            "CAGR_IS_at":  round(float(aft["CAGR_IS"]), 6),
            "CAGR_OOS_at": round(float(aft["CAGR_OOS"]), 6),
            "min_at":      round(float(_min_at(aft)), 6),
            "gap_pp":      round(float(aft["IS_OOS_gap_pp"]), 4),
            "Sharpe":      round(float(pre["Sharpe_OOS"]), 4),
            "MaxDD":       round(float(pre["MaxDD_FULL"]), 6),
            "Worst10Y_at": round(float(aft["Worst10Y_star"]), 6),
            "P10_5Y_at":   round(float(aft["P10_5Y"]), 6),
            "Trades_yr":   round(float(aft["Trades_yr"]), 2),
            "excess_days": res["exc"],
        }

    block = {
        "meta": {
            "EXCESS_EXTRA_STORE_pct":  round(EXCESS_EXTRA_STORE * 100, 4),
            "EXCESS_EXTRA_K365_CENTRE_pct": round(EXCESS_EXTRA_K365_CENTRE * 100, 4),
            "EXCESS_EXTRA_K365_SEN1_pct":   round(EXCESS_EXTRA_K365_SEN1 * 100, 4),
            "EXCESS_EXTRA_K365_SEN2_pct":   round(EXCESS_EXTRA_K365_SEN2 * 100, 4),
            "sanity_LU2_C1_min_at_pct":  round(sanity_min * 100, 4),
            "sanity_LU2_C1_MaxDD_pct":   round(sanity_maxdd * 100, 4),
            "sanity_ok": bool(ok_min and ok_maxdd),
            "n_out_bondoff": n_out_bondoff,
        },
        "store_cfd": {
            lbl: _rblock_full(lbl, store_results[lbl])
            for lbl in store_results
        },
        "k365_centre": {},
        "k365_sensitivity": {},
        "improvement_pp": {},
    }

    for k365_lbl, store_lbl in [
        ("LU1_C1_k365", "LU1_C1_store"),
        ("LU2_C1_k365", "LU2_C1_store"),
        ("B3a_k365", "B3a_store"),
        ("B3b_k365", "B3b_store"),
        ("B3c_k365", "B3c_store"),
    ]:
        block["k365_centre"][k365_lbl] = _rblock_full(k365_lbl, k365_centre_results[k365_lbl])
        block["improvement_pp"][k365_lbl] = round(improvement_pp.get(k365_lbl, float("nan")), 4)

    for lbl, res in sorted(sen_results.items()):
        block["k365_sensitivity"][lbl] = _rblock_metrics(lbl, res)

    print("\n" + "=" * 100)
    print("RETURN_BLOCK")
    print("=" * 100)
    print(json.dumps(block, indent=2))
    print("\nDone.")
    return block


if __name__ == "__main__":
    main()
