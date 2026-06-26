"""
src/audit/dd_reduction_harness_20260626.py
==========================================
Shared harness for the "sc2.0-CAGR with lower MaxDD" exploration (2026-06-26).

Goal: find a strategy with CAGR_OOS >= +28.1% (sc2.0 -1pp) AND MaxDD better than
sc2.0's -61.63%. sc2.0 = P09_STR strong-map {1.60,1.50,1.10,1.00} x scale 2.0
with P09 OUT-fill (Gold always + Bond@bond_mom252>0, inverse-vol W63, T+5 lag) + C1.

This module exposes ONE setup() that loads all shared assets once, and a family of
build_*() functions that flip exactly one of the 4 levers, so candidates are
comparable and nobody re-derives the wiring:

  LEVER A  OUT-fill composition  (gold/bond weights, bond-timing gate, cash floor)
  LEVER B  IN-leg asset blend    (mix a fraction of 1x Gold/Bond INTO the IN NASDAQ leg)
  LEVER C  regime-dependent scale (scale the leverage by a regime/vol/drawdown signal)
  LEVER D  entry/exit signal      (DH-W1 thresholds; handled upstream, exposed via mask)

All builds reuse validated builders from k365_recost / lu_cfd_recost / run_p0x.
metrics10() returns the standard-10 (compute_10metrics + after-tax) + WFA + regime.

REJECTED approaches (do NOT propose / re-run -- already QC'd as "deleverage not timing"):
  - uniform de-lever (lowers CAGR proportionally)            [scale dial]
  - A7 DH-W1 IN-leg volatility brake                         [no target: crashes are OUT]
  - G5 vix-hard defensive overlay                            [66% just de-lever, p=0.40]
  - B1 x scale (IN-leg downside-skew brake + leverage)       [dominated by P09 low-scale]

Cost premise (EVALUATION_STANDARD §1.5 v1.9, do NOT re-litigate): >3x margin is
collateral, no CAGR drag; only recurring >3x cost is financing (already modeled).

ASCII-only prints. No temp files, no commit.
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
from src.audit.unified_metrics import compute_10metrics, IS_END, OOS_START
from src.audit.regime_labeler_20260611 import build_regime_labels, stress_masks
from src.audit.k365_recost_20260612 import EXCESS_EXTRA_K365_CENTRE
from src.audit.lu_cfd_recost_20260611 import _build_tqqq_base
from src.audit.extended_eval_20260611 import _eval_one
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights, _apply_aftertax,
    _count_fund_transitions,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal, FEE_GOLD, FEE_BOND,
)
from src.audit.k365_recost_20260612 import _build_tqqq_base_param

AFTER_TAX = 0.8273
STRONG_MAP = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# sc2.0 anchors (must reproduce; from P09_STR_SCALE_DIAL_20260623.md)
SC20_ANCHOR = {"CAGR_IS": 0.353755, "CAGR_OOS": 0.291102, "MaxDD": -0.616342}
TARGET_CAGR_OOS = 0.281   # >= sc2.0 - 1pp
TARGET_MAXDD = -0.616342  # must be better (less negative)


# ---------------------------------------------------------------------------
def setup():
    """Load all shared assets once. Returns a context dict reused by every build."""
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]
    mask = np.asarray(shared["mask"], dtype=float)
    dates = a["dates"]
    dates_dt = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n = len(dates_dt)
    n_years = n / float(TRADING_DAYS)
    is_mask = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)

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
    wg, wb = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    close = np.asarray(a["close"], float)
    ret_ndx = np.nan_to_num(np.concatenate([[0.0], np.diff(close) / close[:-1]]), nan=0.0)

    regimes = build_regime_labels(a["close"], a["sofr"], dates_dt)
    stress = stress_masks(dates_dt)
    v7_nav, r_v7, tpy_v7, _ = _build_tqqq_base(
        shared, dates_dt, v7_map=None, lev_scale=1.0, cfd_excess=False)

    return dict(shared=shared, a=a, dates=dates, dates_dt=dates_dt, n=n, n_years=n_years,
                mask=mask, out_mask=out_mask, is_mask=is_mask, oos_mask=oos_mask,
                ret_gold=ret_gold, ret_bond=ret_bond, fund_active=fund_active,
                wg=wg, wb=wb, bond_on=bond_on, sofr_arr=sofr_arr, ret_ndx=ret_ndx,
                regimes=regimes, stress=stress, r_v7=r_v7)


# ---------------------------------------------------------------------------
# IN-leg builder: strong-map x scale, optional regime-dependent scale, optional
# IN-leg 1x asset blend. Returns r_in (daily) for the IN/HOLD days.
# ---------------------------------------------------------------------------
def _in_leg_return(ctx, scale_arr, v7_map=STRONG_MAP):
    """Strong-map IN-leg with a per-day scale array (LEVER C). scale_arr scalar->broadcast."""
    shared, dates_dt, n_years = ctx["shared"], ctx["dates_dt"], ctx["n_years"]
    scale_arr = np.asarray(scale_arr, float) if np.ndim(scale_arr) else float(scale_arr)
    # _build_tqqq_base_param applies a scalar lev_scale; for per-day scaling we
    # build at scale 1.0 then re-lever the EXCESS-correct return is complex, so for
    # LEVER C we approximate by building at a representative scalar and document it.
    # The clean path: callers pass a scalar scale for the strong-map base, and
    # per-day modulation is done on the realized IN-leg return below.
    base_nav, r_base, tpy_b, exc = _build_tqqq_base_param(
        shared, dates_dt, v7_map=v7_map, lev_scale=1.0,
        excess_extra=EXCESS_EXTRA)
    return r_base, tpy_b, exc


def build(ctx, *, scale=2.0, regime_scale=None, in_gold_w=0.0, in_bond_w=0.0,
          out_gold_w=None, out_bond_w=None, bond_gate=True, cash_floor=0.0,
          v7_map=STRONG_MAP):
    """Unified candidate builder. Returns (nav_dt, r, tpy, exc).

    LEVER A (OUT): out_gold_w/out_bond_w override inverse-vol weights (None=inverse-vol);
                   bond_gate=False removes bond-timing; cash_floor adds a min cash frac.
    LEVER B (IN):  in_gold_w/in_bond_w blend 1x Gold/Bond into the IN NASDAQ leg
                   (NASDAQ weight = 1 - in_gold_w - in_bond_w), reducing effective NDX
                   exposure with low-corr assets (risk-parity-lite IN leg).
    LEVER C (scale): scale = scalar strong-map leverage; regime_scale = optional
                   per-day multiplier array in [0,1.x] applied to the IN-leg return's
                   excess-over-cash (deleverage in bad regimes, NOT uniform).
    """
    shared, dates_dt, n_years = ctx["shared"], ctx["dates_dt"], ctx["n_years"]
    fund_active = ctx["fund_active"]
    ret_gold, ret_bond = ctx["ret_gold"], ctx["ret_bond"]
    wg, wb, bond_on, sofr_arr = ctx["wg"], ctx["wb"], ctx["bond_on"], ctx["sofr_arr"]
    n = ctx["n"]

    # ---- IN leg: strong-map x scale (scalar), excess-correct cost ----
    base_nav, r_in, tpy_b, exc = _build_tqqq_base_param(
        shared, dates_dt, v7_map=v7_map, lev_scale=scale, excess_extra=EXCESS_EXTRA)
    r_in = np.asarray(r_in, float).copy()

    # LEVER C: per-day regime de-lever on the IN leg (modulate return around cash).
    # r_modulated = cash + s*(r_in - cash); s in [0,1] reduces leverage that day.
    if regime_scale is not None:
        s = np.clip(np.asarray(regime_scale, float), 0.0, 2.0)
        cash_day = sofr_arr  # daily cash return
        r_in = cash_day + s * (r_in - cash_day)

    # LEVER B: blend 1x Gold/Bond INTO the IN leg (lower NDX exposure with low-corr).
    if in_gold_w > 0 or in_bond_w > 0:
        w_ndx = max(0.0, 1.0 - in_gold_w - in_bond_w)
        fee_in = (in_gold_w * FEE_GOLD + in_bond_w * FEE_BOND) / TRADING_DAYS
        r_in = w_ndx * r_in + in_gold_w * ret_gold + in_bond_w * ret_bond - fee_in

    # ---- OUT leg: P09 fill (LEVER A) ----
    if out_gold_w is None and out_bond_w is None:
        w_g, w_b = wg, wb            # inverse-vol (default P09)
    else:
        og = 0.0 if out_gold_w is None else out_gold_w
        ob = 0.0 if out_bond_w is None else out_bond_w
        w_g = np.full(n, og, float)
        w_b = np.full(n, ob, float)
    bond_on_eff = bond_on if bond_gate else np.ones(n, dtype=bool)
    w_b_eff = np.where(bond_on_eff, w_b, 0.0)
    # cash_floor: hold at least this fraction in cash (SOFR) on OUT days
    scale_assets = max(0.0, 1.0 - cash_floor)
    fee_out = (w_g * FEE_GOLD + w_b_eff * FEE_BOND) * scale_assets / TRADING_DAYS
    cash_yield = (np.where(bond_on_eff, 0.0, w_b) + cash_floor) * sofr_arr
    r_out = (w_g * ret_gold + w_b_eff * ret_bond) * scale_assets + cash_yield - fee_out

    r = np.where(fund_active, r_out, r_in)
    r = np.clip(r, -0.999, None)
    nav = np.cumprod(1.0 + r)
    nav_dt = pd.Series(nav, index=dates_dt)
    flips = _count_fund_transitions(fund_active)
    tpy = tpy_b + flips / n_years
    return nav_dt, r, tpy, exc


# ---------------------------------------------------------------------------
def metrics10(ctx, nav_dt, r, tpy, label="cand", with_wfa=True):
    """Standard-10 (after-tax CAGR) + WFA/regime. Returns a flat dict."""
    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    out = dict(
        label=label,
        CAGR_IS=aft["CAGR_IS"], CAGR_OOS=aft["CAGR_OOS"],
        min9=min(aft["CAGR_IS"], aft["CAGR_OOS"]),
        IS_OOS_gap_pp=aft["IS_OOS_gap_pp"],
        Sharpe_FULL=pre["Sharpe_FULL"], Sharpe_OOS=pre["Sharpe_OOS"],
        MaxDD=pre["MaxDD_FULL"], Worst10Y=aft["Worst10Y_star"],
        Worst5Y=aft["Worst5Y"], P10_5Y=aft["P10_5Y"],
        Worst1D=pre["Worst1D"], Trades_yr=aft["Trades_yr"],
    )
    if with_wfa:
        ev = _eval_one(label, nav_dt, r, ctx["regimes"], ctx["stress"],
                       ctx["is_mask"], ctx["oos_mask"], baseline_r=ctx["r_v7"])
        out.update(WFE=float(ev["wfa_WFE"]), CI95_lo=float(ev["wfa_CI95_lo"]),
                   Regime_min=float(ev["regime_min_at"]),
                   cpcv_p10=float(ev["cpcv_p10_at"]))
    return out


# ---------------------------------------------------------------------------
# Causal regime-scale signal factory (LEVER C). All inputs are backward-looking
# (trend vs 200d MA, 63d realized vol percentile, drawdown-from-peak) and are
# applied with an execution lag so there is no look-ahead.
# ---------------------------------------------------------------------------
def regime_signals(ctx, lag=2):
    """Return a dict of causal per-day signal arrays for building regime_scale."""
    dates_dt = ctx["dates_dt"]
    reg = ctx["regimes"]
    close = np.asarray(ctx["a"]["close"], float)
    n = ctx["n"]
    rv63 = np.asarray(reg["rv63"].values, float)          # 63d annualized realized vol
    ma200 = np.asarray(reg["ma200"].values, float)
    above_ma = (close > ma200).astype(float)              # 1 if uptrend
    # vol percentile rank over trailing 5y (causal, expanding-min window)
    s = pd.Series(rv63)
    vol_pct = s.rolling(1260, min_periods=252).apply(
        lambda w: (w[-1] > w[:-1]).mean() if len(w) > 1 else np.nan, raw=True).values
    # drawdown from trailing peak of NASDAQ close (causal)
    run_max = np.maximum.accumulate(np.nan_to_num(close, nan=0.0))
    dd_from_peak = np.where(run_max > 0, close / run_max - 1.0, 0.0)

    def _lag(x):
        y = np.asarray(x, float).copy()
        if lag > 0:
            y[lag:] = y[:-lag]
            y[:lag] = np.nan
        return y

    return dict(
        above_ma=_lag(above_ma), vol_pct=_lag(vol_pct),
        rv63=_lag(rv63), dd_from_peak=_lag(dd_from_peak),
    )


def passes(m):
    """Success gate: sc2.0-class CAGR AND better MaxDD."""
    return (m["CAGR_OOS"] >= TARGET_CAGR_OOS) and (m["MaxDD"] > TARGET_MAXDD)


def fmt(m):
    return ("%-22s CAGR_OOS=%+6.2f%% CAGR_IS=%+6.2f%% min=%+6.2f%% MaxDD=%+7.2f%% "
            "Sh=%+.3f W10Y=%+5.2f%% gap=%+5.2f Tr=%.0f%s"
            % (m["label"], m["CAGR_OOS"]*100, m["CAGR_IS"]*100, m["min9"]*100,
               m["MaxDD"]*100, m["Sharpe_FULL"], m["Worst10Y"]*100,
               m["IS_OOS_gap_pp"], m.get("Trades_yr", float("nan")),
               "  [PASS]" if passes(m) else ""))


# ---------------------------------------------------------------------------
def _self_test():
    """Reproduce sc2.0 anchors with the unified builder (scale=2.0, all levers off)."""
    sys.stdout.reconfigure(encoding="utf-8")
    ctx = setup()
    nav_dt, r, tpy, exc = build(ctx, scale=2.0)
    m = metrics10(ctx, nav_dt, r, tpy, label="sc2.0_repro", with_wfa=False)
    ok = (abs(m["CAGR_IS"] - SC20_ANCHOR["CAGR_IS"]) <= 0.0015 and
          abs(m["CAGR_OOS"] - SC20_ANCHOR["CAGR_OOS"]) <= 0.0015 and
          abs(m["MaxDD"] - SC20_ANCHOR["MaxDD"]) <= 0.0015)
    print("SELF-TEST sc2.0 reproduction:")
    print("  CAGR_IS  %+.4f%% (exp %+.4f%%)" % (m["CAGR_IS"]*100, SC20_ANCHOR["CAGR_IS"]*100))
    print("  CAGR_OOS %+.4f%% (exp %+.4f%%)" % (m["CAGR_OOS"]*100, SC20_ANCHOR["CAGR_OOS"]*100))
    print("  MaxDD    %+.4f%% (exp %+.4f%%)" % (m["MaxDD"]*100, SC20_ANCHOR["MaxDD"]*100))
    print("  -> %s" % ("MATCH (harness wired correctly)" if ok else "MISMATCH -- FIX BEFORE USE"))
    return ok


if __name__ == "__main__":
    _self_test()
