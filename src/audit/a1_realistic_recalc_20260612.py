"""
src/audit/a1_realistic_recalc_20260612.py
==========================================
目的:
  過去 scenarioD (CFD_SPREAD_LOW = 0.20%/yr) で評価された CFD 系高頻度4戦略を
  realistic コスト (時変SOFR+3.0%/yr x フルNotional + 売買スプレッド) で再計算し、
  after-tax IS CAGR >= 18% が残存するか白黒つける。

対象4戦略:
  1. F10eps:    F8-R5 (CALM_BOOST) + eps=0.015 deadband + E4 lev_mod + L_s2_lmax7
  2. F8R5:      F8-R5 (CALM_BOOST, eps=0) + E4 lev_mod + L_s2_lmax7
  3. F7v3_E4:   F7-v3 formula A (tilt=2.0, cap=0.10) + E4 lev_mod + L_s2_lmax7
  4. vz065_l7:  vz065 (lmax=7) + F10 eps=0.015 + vz065 lev_mod (vz_thr=0.65)
                [使用: run_vz065(7.0, basis)]

after-tax 係数: 0.8273 (CAGR_IS/OOS/FULL のみ)
判定: after-tax IS CAGR >= 18% -> SURVIVE / CLOSE

出力:
  - audit_results/a1_realistic_recalc_20260612.csv

使用法 (リポジトリルートから):
  python src/audit/a1_realistic_recalc_20260612.py [--strategy STRAT1 STRAT2 ...]

  STRAT: F10eps | F8R5 | F7v3_E4 | vz065_l7 | ALL (default: ALL)

注意: cp932 端末対応のため ASCII のみ出力。
"""

from __future__ import annotations

import argparse
import os
import sys
import types

# ---------------------------------------------------------------------------
# multitasking スタブ (yfinance 依存回避)
# ---------------------------------------------------------------------------
if "multitasking" not in sys.modules:
    _m = types.ModuleType("multitasking")
    _m.set_max_threads = lambda x: None
    _m.set_engine = lambda x: None
    _m.task = lambda *a, **k: (lambda f: f)
    _m.wait_for_tasks = lambda: None
    sys.modules["multitasking"] = _m

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in [_SRC_DIR, _REPO_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# canonical imports
# ---------------------------------------------------------------------------
from backtest_engine import load_data
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    DATA_PATH,
    TRADING_DAYS,
    THRESHOLD,
)
from cfd_leverage_backtest import (
    build_nav_strategy,
    CFD_SPREAD_LOW,
    NAV_FLOOR,
)
from dynamic_leverage_strategies import compute_L_s2_vz_gated
from compute_cfd_worst10y import prepare_gold_local
from long_cycle_signal import build_lt_signal, apply_lt_mode_b
from e4_regime_klt import signal_to_bias_dynamic
from g14_wfa_sbi_cfd import (
    K_LO, K_HI, K_MID,
    VZ_THR_E4,
    VZ_THR_065,
    VZ_REG,
    TILT_R5,
    TILT_CAP_CALM,
    TILT_CAP_BULL_VZ,
    TILT_CAP_BEAR_VZ,
    EPS_F10,
    EPS_REF,
    S2_LMAX7,
    N_LT2,
    SBI_CFD_SPREAD,
    compute_tilt_with_deadband,
)
from g18_daily_trade_cost_wfa import build_cfd_nav_with_cost
from g19a_f10_eps_extended import build_f10_wn_for_eps

# audit helpers
from src.audit.unified_metrics import compute_10metrics
from src.audit.product_costs_realistic_20260610 import (
    cfd_overnight_daily,
    CFD_SPREAD_ONE_WAY,
)
from src.audit.strategy_runners import run_vz065

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
AFTER_TAX_FACTOR = 0.8273
SURVIVE_THRESHOLD = 0.18
_DELAY = 2

# vz065 spread_rt (g27 scenarioD と同一: moderate=0.050% round-trip)
VZ065_SPREAD_RT = 0.00050

# F7v3 formula A params
F7V3_TILT_A = 2.0
F7V3_TILT_CAP_A = 0.10


# ---------------------------------------------------------------------------
# shared asset cache
# ---------------------------------------------------------------------------
_SHARED: dict | None = None


def _load_shared() -> dict:
    """Load and cache shared assets (E4 base infrastructure)."""
    global _SHARED
    if _SHARED is not None:
        return _SHARED

    df = load_data(DATA_PATH)
    close = df["Close"]
    ret = close.pct_change().fillna(0)
    dates = df["Date"]
    n = len(df)
    n_years = n / TRADING_DAYS

    sofr = load_sofr(dates)
    gold_1x = prepare_gold_local(dates)
    gold_2x = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x = build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0
    )
    bond_3x = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    raw_a2, vz = build_a2_signal(close, ret)
    lev_raw, wn_A, wg_A, wb_A, n_tr = simulate_rebalance_A(raw_a2, vz, THRESHOLD)
    n_trades_yr_ref = n_tr / n_years

    raw_a2_vals = raw_a2.values if hasattr(raw_a2, "values") else np.asarray(raw_a2)
    vz_arr = vz.values if hasattr(vz, "values") else np.asarray(vz)

    # L_s2 with lmax=7 (E4/F10/F8R5/F7v3 base)
    L_s2_lmax7 = compute_L_s2_vz_gated(ret, vz, **S2_LMAX7)

    # E4 regime lev_mod (k_lo=0.1, k_hi=0.8, vz_thr=0.7)
    lt_sig_raw = build_lt_signal(close, "LT2", N_LT2)
    lt_sig_arr = lt_sig_raw.values
    k_dyn_e4 = np.where(
        vz_arr > VZ_THR_E4, K_HI,
        np.where(vz_arr < -VZ_THR_E4, K_LO, K_MID),
    )
    lt_bias_e4 = pd.Series(
        np.clip(-k_dyn_e4 * lt_sig_arr * 0.5, -0.5, 0.5),
        index=lt_sig_raw.index,
    )
    lev_mod_e4 = apply_lt_mode_b(lev_raw, lt_bias_e4, l_min=0.0, l_max=1.0)

    # bull_mask
    bull_mask = raw_a2_vals > THRESHOLD

    # F10 wn/wb (F8-R5 + eps=0.015 deadband)
    wn_f10, wb_f10, _, _ = build_f10_wn_for_eps(
        raw_a2_vals, vz_arr, wn_A, wb_A, np.asarray(lev_raw), bull_mask, EPS_F10
    )
    wg_f10 = np.asarray(wg_A)

    # F8-R5 wn/wb (eps=0, no deadband)
    tilt_r5, _ = compute_tilt_with_deadband(raw_a2_vals, vz_arr, bull_mask, EPS_REF)
    wn_r5 = wn_A + tilt_r5
    wb_r5 = np.clip(wb_A - tilt_r5, 0.0, np.asarray(wb_A))
    wg_r5 = np.asarray(wg_A)

    # F7v3 formula A wn/wb (tilt=2.0, cap=0.10)
    tilt_raw_f7 = F7V3_TILT_A * (raw_a2_vals - THRESHOLD) * (1.0 - raw_a2_vals)
    tilt_f7 = np.where(bull_mask, np.clip(tilt_raw_f7, 0.0, F7V3_TILT_CAP_A), 0.0)
    wn_f7 = wn_A + tilt_f7
    wb_f7 = np.clip(np.asarray(wb_A) - tilt_f7, 0.0, np.asarray(wb_A))
    wg_f7 = np.asarray(wg_A)

    _SHARED = dict(
        close=close,
        dates=dates,
        sofr=sofr,
        gold_2x=gold_2x,
        bond_3x=bond_3x,
        ret=ret,
        vz=vz,
        vz_arr=vz_arr,
        raw_a2_vals=raw_a2_vals,
        bull_mask=bull_mask,
        lev_raw=lev_raw,
        lev_mod_e4=lev_mod_e4,
        wn_A=wn_A,
        wg_A=wg_A,
        wb_A=wb_A,
        L_s2_lmax7=L_s2_lmax7,
        wn_f10=wn_f10,
        wg_f10=wg_f10,
        wb_f10=wb_f10,
        wn_r5=wn_r5,
        wg_r5=wg_r5,
        wb_r5=wb_r5,
        wn_f7=wn_f7,
        wg_f7=wg_f7,
        wb_f7=wb_f7,
        n=n,
        n_years=n_years,
        n_trades_yr_ref=n_trades_yr_ref,
    )
    return _SHARED


def _count_trades_tilt(wn, wb, lev_arr, n_years):
    """Count rebalance events where wn, wb, or lev changes.

    Uses lev_mod_e4 (not lev_raw) to match f8_regime_tilt.count_trades_tilted().
    lev_mod_e4 is the E4 regime-adjusted leverage: applied over lev_raw via apply_lt_mode_b.
    """
    wn_arr = np.asarray(wn, dtype=float)
    wb_arr = np.asarray(wb, dtype=float)
    lev = np.asarray(lev_arr, dtype=float)
    n = len(wn_arr)
    change = np.zeros(n, dtype=bool)
    change[1:] = (
        (wn_arr[1:] != wn_arr[:-1])
        | (wb_arr[1:] != wb_arr[:-1])
        | (lev[1:] != lev[:-1])
    )
    n_tr = int(change.sum())
    return n_tr / n_years if n_years > 0 else float("nan")


def _build_nav_realistic_e4base(
    close, lev_mod, wn, wg, wb, dates, gold_2x, bond_3x, sofr, L_s2
):
    """Build realistic NAV for E4-base strategies (full notional CFD cost).

    Identical to strategy_runners._build_nav_realistic but parameterized.
    overnight = (sofr_daily_t + 3.0%/252) * L_t (full notional)
    spread    = pos_change * 2 * CFD_SPREAD_ONE_WAY
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x).pct_change().fillna(0).values

    idx = dates.index
    lev_s = pd.Series(np.asarray(lev_mod, dtype=float), index=idx).shift(_DELAY).fillna(0).values
    wn_s = pd.Series(np.asarray(wn, dtype=float), index=idx).shift(_DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, dtype=float), index=idx).shift(_DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, dtype=float), index=idx).shift(_DELAY).fillna(0).values

    L_arr = np.asarray(L_s2.values if hasattr(L_s2, "values") else L_s2, dtype=float)
    L_shifted = pd.Series(L_arr, index=idx).shift(_DELAY).fillna(1.0).values

    sofr_arr = np.asarray(sofr, dtype=float)
    overnight = np.vectorize(cfd_overnight_daily)(sofr_arr, L_shifted)

    full_pos = wn_s * lev_s * L_shifted
    prev_pos = np.concatenate([[full_pos[0]], full_pos[:-1]])
    pos_change = np.abs(full_pos - prev_pos)
    spread_cost = pos_change * 2.0 * CFD_SPREAD_ONE_WAY

    nas_ret = L_shifted * r_nas - overnight - spread_cost
    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3
    daily_clipped = np.maximum(daily, NAV_FLOOR)

    nav = (1.0 + pd.Series(daily_clipped, index=idx)).cumprod()
    dates_dt = pd.to_datetime(dates.values)
    nav_dt = pd.Series(nav.values, index=pd.DatetimeIndex(dates_dt))
    return nav_dt


def _build_nav_scenariod_e4base(
    close, lev_mod, wn, wg, wb, dates, gold_2x, bond_3x, sofr, L_s2
):
    """Build scenarioD NAV for E4-base strategies (CFD_SPREAD_LOW=0.20%/yr)."""
    nav = build_nav_strategy(
        close, lev_mod, wn, wg, wb, dates,
        gold_2x, bond_3x, sofr,
        nas_mode="CFD",
        cfd_leverage=np.asarray(L_s2.values if hasattr(L_s2, "values") else L_s2),
        cfd_spread=CFD_SPREAD_LOW,
    )
    dates_dt = pd.to_datetime(dates.values)
    nav_dt = pd.Series(nav.values, index=pd.DatetimeIndex(dates_dt))
    return nav_dt


# ---------------------------------------------------------------------------
# Strategy runners for the 4 CFD strategies
# ---------------------------------------------------------------------------

def run_f10eps(basis: str) -> dict:
    """F10eps: F8-R5 (CALM_BOOST) + eps=0.015 deadband + E4 lev_mod + L_s2_lmax7.

    scenarioD: build_nav_strategy with CFD_SPREAD_LOW.
    realistic: full-notional overnight SOFR+3%/252 + CFD_SPREAD_ONE_WAY.
    Note: Trade count per g7/g14 spec (check_logic_delay_and_causal Block I):
      uses lev_raw (discrete, from simulate_rebalance_A) not lev_mod_e4 (continuous).
      F10 deadband reduces tilt changes => ~52/yr.
    """
    s = _load_shared()
    trades_yr = _count_trades_tilt(s["wn_f10"], s["wb_f10"], s["lev_raw"], s["n_years"])
    if basis == "scenarioD":
        nav = _build_nav_scenariod_e4base(
            s["close"], s["lev_mod_e4"], s["wn_f10"], s["wg_f10"], s["wb_f10"],
            s["dates"], s["gold_2x"], s["bond_3x"], s["sofr"], s["L_s2_lmax7"],
        )
    else:
        nav = _build_nav_realistic_e4base(
            s["close"], s["lev_mod_e4"], s["wn_f10"], s["wg_f10"], s["wb_f10"],
            s["dates"], s["gold_2x"], s["bond_3x"], s["sofr"], s["L_s2_lmax7"],
        )
    metrics = compute_10metrics(nav, trades_yr)
    return {"basis": basis, "strategy": "F10eps", "trades_per_year": trades_yr, **metrics}


def run_f8r5(basis: str) -> dict:
    """F8R5: F8-R5 (CALM_BOOST, eps=0) + E4 lev_mod + L_s2_lmax7.

    This is the F8-R5 without the epsilon deadband (same as F10eps but eps=0).
    scenarioD: build_nav_strategy with CFD_SPREAD_LOW.
    realistic: full-notional overnight SOFR+3%/252 + CFD_SPREAD_ONE_WAY.
    Note: Trade count uses lev_mod_e4 (regime-adjusted) to match f8_regime_tilt methodology.
    """
    s = _load_shared()
    trades_yr = _count_trades_tilt(s["wn_r5"], s["wb_r5"], s["lev_mod_e4"], s["n_years"])
    if basis == "scenarioD":
        nav = _build_nav_scenariod_e4base(
            s["close"], s["lev_mod_e4"], s["wn_r5"], s["wg_r5"], s["wb_r5"],
            s["dates"], s["gold_2x"], s["bond_3x"], s["sofr"], s["L_s2_lmax7"],
        )
    else:
        nav = _build_nav_realistic_e4base(
            s["close"], s["lev_mod_e4"], s["wn_r5"], s["wg_r5"], s["wb_r5"],
            s["dates"], s["gold_2x"], s["bond_3x"], s["sofr"], s["L_s2_lmax7"],
        )
    metrics = compute_10metrics(nav, trades_yr)
    return {"basis": basis, "strategy": "F8R5", "trades_per_year": trades_yr, **metrics}


def run_f7v3_e4(basis: str) -> dict:
    """F7v3+E4: F7-v3 formula A (tilt=2.0, cap=0.10) + E4 lev_mod + L_s2_lmax7.

    scenarioD: build_nav_strategy with CFD_SPREAD_LOW.
    realistic: full-notional overnight SOFR+3%/252 + CFD_SPREAD_ONE_WAY.
    Note: Trade count uses lev_mod_e4 (regime-adjusted) to match f7v3_bull_tilt methodology.
    """
    s = _load_shared()
    trades_yr = _count_trades_tilt(s["wn_f7"], s["wb_f7"], s["lev_mod_e4"], s["n_years"])
    if basis == "scenarioD":
        nav = _build_nav_scenariod_e4base(
            s["close"], s["lev_mod_e4"], s["wn_f7"], s["wg_f7"], s["wb_f7"],
            s["dates"], s["gold_2x"], s["bond_3x"], s["sofr"], s["L_s2_lmax7"],
        )
    else:
        nav = _build_nav_realistic_e4base(
            s["close"], s["lev_mod_e4"], s["wn_f7"], s["wg_f7"], s["wb_f7"],
            s["dates"], s["gold_2x"], s["bond_3x"], s["sofr"], s["L_s2_lmax7"],
        )
    metrics = compute_10metrics(nav, trades_yr)
    return {"basis": basis, "strategy": "F7v3_E4", "trades_per_year": trades_yr, **metrics}


def run_vz065_l7(basis: str) -> dict:
    """vz065_l7+F10eps: vz065 (lmax=7) + F10 eps=0.015 + vz_thr=0.65 lev_mod.

    Uses strategy_runners.run_vz065(7.0, basis) which is already implemented.
    """
    result = run_vz065(7.0, basis)
    result["strategy"] = "vz065_l7"
    result["basis"] = basis
    return result


# ---------------------------------------------------------------------------
# E4 sanity check (IS approx 20.46% realistic, after-tax ~16.93%)
# ---------------------------------------------------------------------------

def run_e4_sanity(basis: str) -> dict:
    """Re-run E4 to verify sanity against known values (should match audit_results/audit_e4_*.csv)."""
    from src.audit.strategy_runners import run_e4
    result = run_e4(basis)
    result["strategy"] = "E4_sanity"
    result["basis"] = basis
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "F10eps":   run_f10eps,
    "F8R5":     run_f8r5,
    "F7v3_E4":  run_f7v3_e4,
    "vz065_l7": run_vz065_l7,
}

# Reference scenarioD numbers from previous evaluations (for sanity check)
# Source: MEMORY.md / INTEGRATION_DEBATE_2026-05-26.md / audit_results/*.csv
SCENARIOD_REF = {
    "F10eps":   {"CAGR_OOS": 0.368, "Sharpe_OOS": 0.934, "IS_OOS_gap_pp": -4.31, "Trades_yr": 52},
    "F8R5":     {"CAGR_OOS": None,  "Sharpe_OOS": 0.934, "IS_OOS_gap_pp": None,  "Trades_yr": 182},
    "F7v3_E4":  {"CAGR_OOS": None,  "Sharpe_OOS": 0.926, "IS_OOS_gap_pp": None,  "Trades_yr": 183},
    # vz065_l7 scenarioD OOS = +31.64% (from audit_vz065_l7_scenarioD.csv).
    # Note: "OOS+22.5%" in MEMORY was the realistic value, not scenarioD.
    "vz065_l7": {"CAGR_OOS": 0.316, "Sharpe_OOS": None,  "IS_OOS_gap_pp": None,  "Trades_yr": None},
}


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   N/A  "
    return f"{v*100:+7.2f}%"


def _fmt_float(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   N/A  "
    return format(v, fmt)


def main():
    parser = argparse.ArgumentParser(description="A1 realistic cost recalculation for 4 CFD strategies")
    parser.add_argument(
        "--strategy", nargs="*", default=["ALL"],
        help="Strategies to run: F10eps F8R5 F7v3_E4 vz065_l7 ALL (default: ALL)"
    )
    args = parser.parse_args()

    selected = list(STRATEGY_MAP.keys()) if "ALL" in args.strategy else args.strategy
    for s in selected:
        if s not in STRATEGY_MAP:
            print(f"ERROR: Unknown strategy {s}. Choose from {list(STRATEGY_MAP.keys())} or ALL")
            sys.exit(1)

    print("=" * 72)
    print("A1 Realistic Cost Recalc - CFD High-Frequency Strategies")
    print("Date: 2026-06-12")
    print(f"Target: {selected}")
    print("=" * 72)

    # -- E4 sanity check --
    print("\n[SANITY] Re-running E4 realistic to verify IS ~20.46%, after-tax ~16.93%...")
    e4_r = run_e4_sanity("realistic")
    e4_is_pt = e4_r.get("CAGR_IS", float("nan"))
    e4_is_at = e4_is_pt * AFTER_TAX_FACTOR
    print(f"  E4 realistic IS pretax: {e4_is_pt*100:+.2f}%  after-tax: {e4_is_at*100:+.2f}%")
    e4_sd = run_e4_sanity("scenarioD")
    e4_sd_oos = e4_sd.get("CAGR_OOS", float("nan"))
    print(f"  E4 scenarioD CAGR_OOS:  {e4_sd_oos*100:+.2f}%  (ref: +33.53%)")
    sanity_ok = abs(e4_is_pt - 0.2046) < 0.005
    print(f"  Sanity: {'OK' if sanity_ok else 'WARN (deviation > 0.5pp)'}")

    all_rows = []
    all_rows.append({
        "strategy": "E4_sanity",
        "basis": "scenarioD",
        **{k: e4_sd.get(k, float("nan")) for k in [
            "CAGR_IS", "CAGR_OOS", "CAGR_FULL", "IS_OOS_gap_pp",
            "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y", "Worst5Y", "Trades_yr"
        ]},
    })
    all_rows.append({
        "strategy": "E4_sanity",
        "basis": "realistic",
        **{k: e4_r.get(k, float("nan")) for k in [
            "CAGR_IS", "CAGR_OOS", "CAGR_FULL", "IS_OOS_gap_pp",
            "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y", "Worst5Y", "Trades_yr"
        ]},
    })

    results_by_strategy: dict[str, dict] = {}

    for strat_id in selected:
        runner = STRATEGY_MAP[strat_id]
        ref = SCENARIOD_REF.get(strat_id, {})

        print(f"\n{'='*60}")
        print(f"Strategy: {strat_id}")
        print(f"{'='*60}")

        # -- scenarioD (for verification against old numbers) --
        print(f"  Running {strat_id} scenarioD...")
        sd = runner("scenarioD")
        sd_oos = sd.get("CAGR_OOS", float("nan"))
        sd_sharpe = sd.get("Sharpe_OOS", float("nan"))
        sd_tyr = sd.get("Trades_yr", float("nan"))
        sd_is = sd.get("CAGR_IS", float("nan"))
        print(f"  scenarioD: IS={sd_is*100:+.2f}%  OOS={sd_oos*100:+.2f}%  "
              f"Sharpe={sd_sharpe:.3f}  Tr/yr={sd_tyr:.1f}")

        # Verify against reference
        ref_oos = ref.get("CAGR_OOS")
        ref_sharpe = ref.get("Sharpe_OOS")
        ref_tyr = ref.get("Trades_yr")
        if ref_oos is not None:
            diff_oos = (sd_oos - ref_oos) * 100
            ok = abs(diff_oos) <= 1.0
            print(f"  VERIFY OOS: got {sd_oos*100:+.2f}%  ref {ref_oos*100:+.2f}%  "
                  f"diff={diff_oos:+.2f}pp -> {'OK' if ok else 'WARN (>1pp)'}")
        if ref_sharpe is not None:
            diff_sharpe = sd_sharpe - ref_sharpe
            ok = abs(diff_sharpe) <= 0.010
            print(f"  VERIFY Sharpe: got {sd_sharpe:.3f}  ref {ref_sharpe:.3f}  "
                  f"diff={diff_sharpe:+.3f} -> {'OK' if ok else 'WARN (>0.010)'}")
        if ref_tyr is not None:
            diff_tyr = sd_tyr - ref_tyr
            ok = abs(diff_tyr) <= 5.0
            print(f"  VERIFY Tr/yr: got {sd_tyr:.1f}  ref {ref_tyr:.1f}  "
                  f"diff={diff_tyr:+.1f} -> {'OK' if ok else 'WARN (>5)'}")

        # -- realistic --
        print(f"  Running {strat_id} realistic...")
        rl = runner("realistic")
        rl_is_pt = rl.get("CAGR_IS", float("nan"))
        rl_oos_pt = rl.get("CAGR_OOS", float("nan"))
        rl_is_at = rl_is_pt * AFTER_TAX_FACTOR
        rl_sharpe = rl.get("Sharpe_OOS", float("nan"))
        rl_maxdd = rl.get("MaxDD_FULL", float("nan"))
        rl_w10y = rl.get("Worst10Y_star", float("nan"))
        rl_p10 = rl.get("P10_5Y", float("nan"))
        rl_gap = rl.get("IS_OOS_gap_pp", float("nan"))
        rl_tyr = rl.get("Trades_yr", float("nan"))

        verdict = "SURVIVE" if rl_is_at >= SURVIVE_THRESHOLD else "CLOSE"

        print(f"  realistic IS pretax: {rl_is_pt*100:+.2f}%  after-tax: {rl_is_at*100:+.2f}%")
        print(f"  realistic OOS pretax:{rl_oos_pt*100:+.2f}%")
        print(f"  realistic Sharpe:    {rl_sharpe:.3f}")
        print(f"  realistic MaxDD:     {rl_maxdd*100:+.2f}%")
        print(f"  realistic Worst10Y:  {rl_w10y*100:+.2f}%")
        print(f"  IS-OOS gap:          {rl_gap:+.2f}pp")
        print(f"  Tr/yr:               {rl_tyr:.1f}")
        print(f"  >>> VERDICT: after-tax IS CAGR = {rl_is_at*100:+.2f}%  "
              f"(>= 18% threshold?)  -> {verdict}")

        results_by_strategy[strat_id] = {"sd": sd, "rl": rl, "verdict": verdict}

        # append to CSV rows
        all_rows.append({
            "strategy": strat_id,
            "basis": "scenarioD",
            **{k: sd.get(k, float("nan")) for k in [
                "CAGR_IS", "CAGR_OOS", "CAGR_FULL", "IS_OOS_gap_pp",
                "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y", "Worst5Y", "Trades_yr"
            ]},
        })
        all_rows.append({
            "strategy": strat_id,
            "basis": "realistic",
            **{k: rl.get(k, float("nan")) for k in [
                "CAGR_IS", "CAGR_OOS", "CAGR_FULL", "IS_OOS_gap_pp",
                "Sharpe_OOS", "MaxDD_FULL", "Worst10Y_star", "P10_5Y", "Worst5Y", "Trades_yr"
            ]},
        })

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY: realistic after-tax IS CAGR vs 18% threshold")
    print("=" * 72)
    print(f"{'Strategy':<14} {'IS_pt%':>8} {'IS_at%':>8} {'OOS_pt%':>9} "
          f"{'Sharpe':>7} {'MaxDD%':>8} {'W10Y%':>7} {'Gap_pp':>7} "
          f"{'Tr/yr':>6} {'VERDICT':>8}")
    print("-" * 90)

    for strat_id in selected:
        if strat_id not in results_by_strategy:
            continue
        rl = results_by_strategy[strat_id]["rl"]
        verdict = results_by_strategy[strat_id]["verdict"]
        is_pt = rl.get("CAGR_IS", float("nan"))
        is_at = is_pt * AFTER_TAX_FACTOR
        oos_pt = rl.get("CAGR_OOS", float("nan"))
        sharpe = rl.get("Sharpe_OOS", float("nan"))
        maxdd = rl.get("MaxDD_FULL", float("nan"))
        w10y = rl.get("Worst10Y_star", float("nan"))
        gap = rl.get("IS_OOS_gap_pp", float("nan"))
        tyr = rl.get("Trades_yr", float("nan"))
        print(f"{strat_id:<14} {is_pt*100:>+8.2f} {is_at*100:>+8.2f} {oos_pt*100:>+9.2f} "
              f"{sharpe:>7.3f} {maxdd*100:>+8.2f} {w10y*100:>+7.2f} {gap:>+7.2f} "
              f"{tyr:>6.1f} {verdict:>8}")

    print("-" * 90)
    print(f"  IS_at% = IS_pt% x {AFTER_TAX_FACTOR} (after-tax, CAGR only)")
    print(f"  Threshold: IS_at >= +18.00% -> SURVIVE  else -> CLOSE")

    # Print E4 reference line
    e4_is_pt2 = all_rows[1].get("CAGR_IS", float("nan"))  # realistic row
    e4_is_at2 = e4_is_pt2 * AFTER_TAX_FACTOR
    print(f"\n  E4 (current Active): IS_at = {e4_is_at2*100:+.2f}%  (ref ~16.93%)")

    # ---------------------------------------------------------------------------
    # Export CSV
    # ---------------------------------------------------------------------------
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "a1_realistic_recalc_20260612.csv")

    df_out = pd.DataFrame(all_rows)

    # Add after-tax columns for CAGR
    for col in ["CAGR_IS", "CAGR_OOS", "CAGR_FULL"]:
        df_out[f"{col}_aftertax"] = df_out[col] * AFTER_TAX_FACTOR

    # Add IS_at >= 18% flag
    df_out["IS_at_survive"] = df_out.apply(
        lambda r: "SURVIVE" if r["basis"] == "realistic" and r["CAGR_IS_aftertax"] >= SURVIVE_THRESHOLD
        else ("CLOSE" if r["basis"] == "realistic" else "N/A"),
        axis=1,
    )

    # Add min(IS, OOS) pretax
    df_out["min_IS_OOS_pretax"] = df_out[["CAGR_IS", "CAGR_OOS"]].min(axis=1)
    df_out["min_IS_OOS_aftertax"] = df_out["min_IS_OOS_pretax"] * AFTER_TAX_FACTOR

    col_order = [
        "strategy", "basis",
        "CAGR_IS", "CAGR_OOS", "CAGR_FULL",
        "CAGR_IS_aftertax", "CAGR_OOS_aftertax", "CAGR_FULL_aftertax",
        "min_IS_OOS_pretax", "min_IS_OOS_aftertax",
        "IS_OOS_gap_pp", "Sharpe_OOS", "MaxDD_FULL",
        "Worst10Y_star", "P10_5Y", "Worst5Y", "Trades_yr",
        "IS_at_survive",
    ]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]
    df_out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nCSV saved: {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
