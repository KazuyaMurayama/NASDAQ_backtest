"""
src/audit/sc135_weight_leverage_breakdown_20260617.py
======================================================
Weight / leverage breakdown for strategy #1: Bext_str_sc1.35
  = B3a base + v7_map STRONG {Q0:1.60,Q1:1.50,Q2:1.10,Q3:1.00}
    x uniform scale 1.35 + P09 OUT fill + C1 SOFR cash + k365 cost.

Tasks:
  1. Sanity: reproduce min9=+23.83%, MaxDD=-45.04% within +/-0.1pp.
     (These are the values reported in LEVERUP_EXTENSION_RESULTS_20260616.md
      for Bext_str_sc1.35.)
  2. Aggregate1: per-asset time-average capital weight (shift-applied series).
  3. Aggregate2: IN-day effective leverage L distribution + Q0-Q3 breakdown.
  4. Aggregate3: >3x excess statistics.
  5. Aggregate4: leverage bin time-ratio table.

Outputs:
  src/audit/sc135_weight_leverage_breakdown_20260617.py  (this file)
  audit_results/sc135_weight_leverage_breakdown_20260617.csv
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
from src.audit.k365_recost_20260612 import (
    _build_full_c1,
    EXCESS_EXTRA_K365_CENTRE,
)
from src.audit.run_p01_backtest_20260611 import (
    LAG_DAYS, TRADING_DAYS,
    _ret_from_nav_level, _inverse_vol_weights,
    _apply_aftertax,
)
from src.audit.run_p02_p09_backtest_20260611 import (
    GATE_DELAY, _load_macro_signal,
    FEE_GOLD, FEE_BOND,
)
from src.audit.cost_model_cfd_vs_tqqq_20260611 import (
    SWAP_SPREAD, TER_TQQQ, DH_PER_UNIT, NAV_FLOOR,
    DELAY as V7_DELAY,
)
from src.audit.run_p09_tqqq_validate_20260611 import (
    _build_v7_mult_custom,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
B3A_MAP_STRONG = {0: 1.60, 1: 1.50, 2: 1.10, 3: 1.00}
LEV_SCALE = 1.35
EXCESS_EXTRA = EXCESS_EXTRA_K365_CENTRE  # 0.0025

# Sanity targets for Bext_str_sc1.35 (from LEVERUP_EXTENSION_RESULTS_20260616.md)
SANITY_MIN9_EXPECT  = 0.2383   # +23.83%
SANITY_MAXDD_EXPECT = -0.4504  # -45.04%
SANITY_TOL = 0.0010             # +/-0.10pp


def _min_at(aft):
    return min(aft["CAGR_IS"], aft["CAGR_OOS"])


def _build_leverage_series(shared, dates_dt, v7_map, lev_scale):
    """
    Reconstruct the per-day effective leverage L and weight series
    (post-shift = what was actually held) for the NASDAQ sleeve.

    Returns:
        L_shifted      : np.ndarray shape (n,) -- effective leverage after V7_DELAY shift
        wn_shifted     : np.ndarray shape (n,) -- NASDAQ capital weight (shifted)
        wg_shifted     : np.ndarray shape (n,) -- Gold capital weight (shifted)
        wb_shifted     : np.ndarray shape (n,) -- Bond capital weight (shifted)
        mult_v7_raw    : np.ndarray shape (n,) -- mult_v7 before shift (for Q labelling)
        quantile_raw   : np.ndarray shape (n,) -- Q0/Q1/Q2/Q3 integer label (for info)
    """
    a = shared["assets"]
    dates = a["dates"]
    idx = dates.index

    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)
    wn = np.asarray(shared["wn"], float)
    wg = np.asarray(shared["wg"], float)
    wb = np.asarray(shared["wb"], float)

    # Build v7 mult (custom map * scale)
    mult_v7 = _build_v7_mult_custom(dates_dt, v7_map) * float(lev_scale)

    # Unshifted: lev_mod = lev_raw_masked * mult_v7
    lev_mod = lev_raw_masked * mult_v7  # pre-shift

    # Apply V7_DELAY shift (same as _build_nav_v7_tqqq_param line: L = shift(lev_mod*3,2))
    L_unshifted = lev_mod * 3.0
    L_shifted = pd.Series(L_unshifted, index=idx).shift(V7_DELAY).fillna(1.0).values
    wn_shifted = pd.Series(wn, index=idx).shift(V7_DELAY).fillna(0.0).values
    wg_shifted = pd.Series(wg, index=idx).shift(V7_DELAY).fillna(0.0).values
    wb_shifted = pd.Series(wb, index=idx).shift(V7_DELAY).fillna(0.0).values

    # Recover Q label from macro signal (same pipeline as _build_v7_mult_custom)
    macro_path = os.path.join(_REPO_DIR, "data", "macro_features.csv")
    macro = pd.read_csv(macro_path, parse_dates=["date"], index_col="date").sort_index()
    signal_raw = macro["nasdaq_mom63"].dropna()
    from signals.quantize import quantile_cut as _quantile_cut
    from signals.timing import apply_publication_lag as _apply_lag
    sig_q = _quantile_cut(signal_raw, levels=4).dropna().astype("int8")
    sig_lagged = _apply_lag(sig_q, "daily")
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    sig_aligned = sig_lagged.reindex(dates_dt).ffill()
    quantile_raw = np.where(sig_aligned.isna(), -1, sig_aligned.values.astype(int))

    return L_shifted, wn_shifted, wg_shifted, wb_shifted, mult_v7, quantile_raw


def main():
    print("=" * 120)
    print("SC1.35 WEIGHT / LEVERAGE BREAKDOWN  2026-06-17")
    print("Config: Bext_str_sc1.35  v7_map={0:1.60,1:1.50,2:1.10,3:1.00} x scale=1.35")
    print("Cost model: k365 EXCESS_EXTRA=0.0025 (%/yr for L>3). C1 SOFR OUT fill.")
    print("Sanity: min9 +23.83%+/-0.1pp / MaxDD -45.04%+/-0.1pp")
    print("=" * 120)

    # ---- Load shared data ----
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
        build_bond_1x_nav_corrected(dates, use_time_varying_duration=True,
                                    bond_maturity=22.0), float)
    ret_gold = _ret_from_nav_level(gold_1x)
    ret_bond = _ret_from_nav_level(bond_1x)

    out_mask_arr = (mask < 0.5)
    fund_active = np.zeros(n, dtype=bool)  # True = OUT (NASDAQ-sleeve is OFF)
    fund_active[LAG_DAYS:] = out_mask_arr[:-LAG_DAYS]

    wg_iv, wb_iv = _inverse_vol_weights(ret_gold, ret_bond, 63)
    bond_m252 = _load_macro_signal(dates, "bond_mom252", GATE_DELAY)
    bond_on = np.where(np.isnan(bond_m252), False, bond_m252 > 0)
    sofr_arr = np.asarray(a["sofr"], float)

    # =========================================================================
    # SANITY GATE: Bext_str_sc1.35 must reproduce min9 +23.83% / MaxDD -45.04%
    # =========================================================================
    print("\n" + "=" * 120)
    print("SANITY GATE: Reproducing Bext_str_sc1.35")
    print("  Expected: min9 +23.83%+/-0.10pp  MaxDD -45.04%+/-0.10pp")
    print("=" * 120)

    nav_dt, r, tpy, exc = _build_full_c1(
        shared, dates_dt, n_years,
        ret_gold, ret_bond, fund_active, wg_iv, wb_iv, bond_on, sofr_arr,
        v7_map=B3A_MAP_STRONG, lev_scale=LEV_SCALE,
        excess_extra=EXCESS_EXTRA)

    pre = compute_10metrics(nav_dt, tpy)
    aft = _apply_aftertax(pre)
    got_min9  = _min_at(aft)
    got_maxdd = pre["MaxDD_FULL"]

    ok_min9  = abs(got_min9  - SANITY_MIN9_EXPECT)  <= SANITY_TOL
    ok_maxdd = abs(got_maxdd - SANITY_MAXDD_EXPECT) <= SANITY_TOL

    print("  min9:  got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (got_min9 * 100, SANITY_MIN9_EXPECT * 100, "OK" if ok_min9 else "FAIL"))
    print("  MaxDD: got %+.4f%%  expect ~%+.4f%%  -> %s"
          % (got_maxdd * 100, SANITY_MAXDD_EXPECT * 100, "OK" if ok_maxdd else "FAIL"))

    if not (ok_min9 and ok_maxdd):
        print("\nSANITY FAILED. Halting.")
        print("NOTE: tolerances are +/-0.10pp. If values differ by more, check")
        print("  that LEVERUP_EXTENSION_RESULTS_20260616.md values match this config.")
        print("  Actual: min9=%+.6f  MaxDD=%+.6f" % (got_min9, got_maxdd))
        sys.exit(1)
    print("  SANITY PASSED. Proceeding.\n")

    # =========================================================================
    # Reconstruct intermediate series (L, wn_s, wg_s, wb_s, quantile)
    # =========================================================================
    print("Reconstructing leverage and weight series (post-shift) ...")
    L_s, wn_s, wg_s, wb_s, mult_v7_arr, q_raw = _build_leverage_series(
        shared, dates_dt, B3A_MAP_STRONG, LEV_SCALE)

    # IN days: fund_active == False (NASDAQ sleeve active)
    # OUT days: fund_active == True
    in_mask  = ~fund_active   # shape (n,) bool -- NASDAQ-holding days (pre-shift)
    out_mask = fund_active

    # Note: L_s is already shift(V7_DELAY) applied, same as wn_s, wg_s, wb_s.
    # For OUT days the blend replaces the base return; wn_s should be ~1 during OUT
    # but the base already incorporates wn from the DH-W1 signal. We track the
    # unshifted fund_active to identify which days are IN/OUT in the portfolio sense.
    # V7_DELAY=2: shift means that the leverage effective at day t was decided at t-2.
    # We use fund_active (unshifted) to define IN/OUT for the purpose of all aggregates,
    # because this is the actual position state on the trading day.
    #
    # However, the capital weight series wn_s / wg_s / wb_s are the ones used in
    # the NAV computation (after shift). We use these for the weight aggregates.

    # For the C1 fill (OUT days), the actual allocation in the portfolio on OUT days is:
    #   wg_iv * gold + wb_iv_eff * bond + cash (SOFR)
    # where wb_iv_eff = wb_iv * bond_on.
    # These weights replace wn_s on OUT days in the NAV.
    # But wn_s, wg_s, wb_s from the base (TQQQ builder) reflect the DH-W1 weights
    # which are already 0 or 1 for wn (IN signal) and small for wg/wb.
    # For weight aggregation, we use the actual effective weights on each day.

    # Effective weights (accounting for C1 P09 replacement on OUT days)
    # On IN days:  wn_eff = wn_s (shifted DH-W1 weight; ~ 1 when IN)
    #              wg_eff = wg_s, wb_eff = wb_s (small non-zero from DH-W1 baseline)
    #              cash_eff = 0
    # On OUT days (fund_active=True): the P09/C1 logic replaces the base return with
    #              wg_iv*gold + wb_iv_eff*bond + cash_sofr
    #              So: wn_eff=0, wg_eff=wg_iv, wb_eff=wb_iv*bond_on, cash_eff=wb_iv*(1-bond_on)
    bond_on_arr = np.asarray(bond_on, dtype=bool)
    wb_eff_out  = wb_iv * bond_on_arr.astype(float)  # bond weight on OUT days
    cash_eff_out = wb_iv * (~bond_on_arr).astype(float)  # cash (SOFR) weight on OUT days

    # Effective weight per day (combining IN and OUT)
    wn_eff  = np.where(in_mask,  wn_s,     0.0)
    wg_eff  = np.where(in_mask,  wg_s,     wg_iv)
    wb_eff  = np.where(in_mask,  wb_s,     wb_eff_out)
    cash_eff = np.where(in_mask, 0.0,      cash_eff_out)

    # =========================================================================
    # AGGREGATE 1: Per-asset time-average capital weight
    # =========================================================================
    print("\n" + "=" * 120)
    print("AGGREGATE 1: Per-asset time-average capital weight")
    print("=" * 120)

    n_in  = int(in_mask.sum())
    n_out = int(out_mask.sum())
    n_out_bondon  = int((out_mask & bond_on_arr).sum())
    n_out_bondoff = int((out_mask & ~bond_on_arr).sum())

    print("  Total days: %d  IN: %d (%.1f%%)  OUT: %d (%.1f%%)"
          % (n, n_in, 100.0*n_in/n, n_out, 100.0*n_out/n))
    print("  OUT breakdown: bond-ON %d (%.1f%%)  bond-OFF %d (%.1f%%)"
          % (n_out_bondon, 100.0*n_out_bondon/n,
             n_out_bondoff, 100.0*n_out_bondoff/n))

    def _wgt_stats(arr, label, in_m, out_m):
        all_avg  = float(arr.mean())
        in_avg   = float(arr[in_m].mean())  if in_m.sum() > 0  else float("nan")
        out_avg  = float(arr[out_m].mean()) if out_m.sum() > 0 else float("nan")
        return all_avg, in_avg, out_avg

    assets_order = [
        ("NASDAQ (wn_eff)",  wn_eff),
        ("Gold   (wg_eff)",  wg_eff),
        ("Bond   (wb_eff)",  wb_eff),
        ("Cash   (SOFR)",    cash_eff),
    ]

    print("\n  %-22s | %9s | %9s | %9s" % ("Asset", "All-day%", "IN-day%", "OUT-day%"))
    print("  " + "-" * 60)
    wgt_rows = []
    total_all = 0.0
    for lbl, arr in assets_order:
        aa, ia, oa = _wgt_stats(arr, lbl, in_mask, out_mask)
        total_all += aa
        ia_s = "%+.4f%%" % (ia * 100) if not np.isnan(ia) else "  n/a"
        oa_s = "%+.4f%%" % (oa * 100) if not np.isnan(oa) else "  n/a"
        print("  %-22s | %+8.4f%% | %s | %s" % (lbl, aa * 100, ia_s, oa_s))
        wgt_rows.append({
            "asset": lbl.strip(), "all_avg_pct": round(aa*100,4),
            "in_avg_pct": round(ia*100,4) if not np.isnan(ia) else None,
            "out_avg_pct": round(oa*100,4) if not np.isnan(oa) else None,
        })
    print("  %-22s | %+8.4f%%  (should be ~1.0 on IN days; <1 total due to OUT)"
          % ("Sum", total_all * 100))

    # =========================================================================
    # AGGREGATE 2: IN-day effective leverage L distribution + Q0-Q3 breakdown
    # =========================================================================
    print("\n" + "=" * 120)
    print("AGGREGATE 2: IN-day effective leverage L distribution")
    print("=" * 120)

    # L_s is the effective leverage (post-shift). On IN days, this is the NASDAQ
    # sleeve leverage. On OUT days L_s ~ lev_raw_masked*mult*3 but the actual
    # portfolio uses the gold/bond/cash blend (L_s is not applied).
    L_in = L_s[in_mask]
    L_all = L_s  # for reference

    print("  IN-day L statistics:")
    print("  N_in=%d  mean=%.4f  median=%.4f  p10=%.4f  p90=%.4f  max=%.4f"
          % (len(L_in),
             float(np.mean(L_in)), float(np.median(L_in)),
             float(np.percentile(L_in, 10)), float(np.percentile(L_in, 90)),
             float(np.max(L_in))))
    print("  All-day L mean=%.4f (reference)" % float(np.mean(L_all)))

    # Q0-Q3 breakdown of L (using pre-shift quantile label; maps to mult_v7_arr before shift)
    # q_raw is the unshifted Q label at each day.
    print("\n  Q0-Q3 breakdown (IN days only; Q=quartile of nasdaq_mom63):")
    print("  Q label: Q0=lowest momentum (gets highest mult), Q3=highest mom (mult=1.0)")
    print("  %-6s | %8s | %9s | %9s | %9s" % ("Q-label", "N_days", "pct_IN%", "mean_L", "mult_map"))
    q_rows = []
    mult_by_q = {0: B3A_MAP_STRONG[0]*LEV_SCALE, 1: B3A_MAP_STRONG[1]*LEV_SCALE,
                 2: B3A_MAP_STRONG[2]*LEV_SCALE, 3: B3A_MAP_STRONG[3]*LEV_SCALE}
    for q in [0, 1, 2, 3]:
        q_in_mask = in_mask & (q_raw == q)
        n_q = int(q_in_mask.sum())
        pct_in = 100.0 * n_q / n_in if n_in > 0 else 0.0
        L_q = L_s[q_in_mask]
        mean_L_q = float(np.mean(L_q)) if len(L_q) > 0 else float("nan")
        mult_q = mult_by_q.get(q, float("nan"))
        print("  Q%-5d | %8d | %8.2f%% | %9.4f | %.2f -> L_avg=%.2fx (expect ~%.2f)"
              % (q, n_q, pct_in, mean_L_q, mult_q, mean_L_q,
                 mult_q * 3.0))  # expect ~ mult*3 if lev_raw_masked~1 when IN
        q_rows.append({
            "q_label": q, "n_in_days": n_q, "pct_of_in_pct": round(pct_in,2),
            "mean_L": round(mean_L_q,4) if not np.isnan(mean_L_q) else None,
            "mult_map_value": round(mult_q,4),
            "expected_L_if_raw1": round(mult_q*3.0,4),
        })

    # =========================================================================
    # AGGREGATE 3: >3x excess statistics
    # =========================================================================
    print("\n" + "=" * 120)
    print("AGGREGATE 3: >3x excess leverage statistics")
    print("=" * 120)

    excess_all = np.maximum(L_s - 3.0, 0.0)
    excess_in  = np.maximum(L_in - 3.0, 0.0)

    gt3_all = (L_s > 3.0)
    gt3_in  = (L_in > 3.0)

    n_gt3_all  = int(gt3_all.sum())
    n_gt3_in   = int(gt3_in.sum())

    ratio_all_pct = 100.0 * n_gt3_all / n
    ratio_in_pct  = 100.0 * n_gt3_in  / n_in  if n_in > 0 else 0.0

    mean_exc_given_gt3_all = float(excess_all[gt3_all].mean()) if n_gt3_all > 0 else 0.0
    mean_exc_given_gt3_in  = float(excess_in [gt3_in].mean())  if n_gt3_in  > 0 else 0.0
    mean_exc_all = float(excess_all.mean())
    mean_exc_in  = float(excess_in.mean())

    print("  >3x day count (all days):  %d / %d = %.2f%%"
          % (n_gt3_all, n, ratio_all_pct))
    print("  >3x day count (IN days):   %d / %d = %.2f%%"
          % (n_gt3_in, n_in, ratio_in_pct))
    print("  mean(excess | L>3, all):   %.4f" % mean_exc_given_gt3_all)
    print("  mean(excess | L>3, IN):    %.4f" % mean_exc_given_gt3_in)
    print("  mean(excess, all days):    %.4f" % mean_exc_all)
    print("  mean(excess, IN days):     %.4f" % mean_exc_in)
    print("  Note: excess = max(L-3, 0); k365 EXCESS_EXTRA=%.4f%%/yr charged on this."
          % (EXCESS_EXTRA * 100))

    # =========================================================================
    # AGGREGATE 4: Leverage bin time-ratio table
    # =========================================================================
    print("\n" + "=" * 120)
    print("AGGREGATE 4: Leverage bin time-ratio table")
    print("(effective leverage L after V7_DELAY shift; OUT days L reflects base wn signal)")
    print("=" * 120)

    bins = [
        (0.0,   0.5,  "L<0.5  (near-zero/OUT)"),
        (0.5,   1.0,  "0.5-1.0"),
        (1.0,   2.0,  "1.0-2.0"),
        (2.0,   3.0,  "2.0-3.0"),
        (3.0,   4.0,  "3.0-4.0  >3x"),
        (4.0,   5.0,  "4.0-5.0  >3x"),
        (5.0,   6.0,  "5.0-6.0  >3x"),
        (6.0, 999.9,  "6.0+     >3x"),
    ]

    print("  %-24s | %7s | %7s | %7s | %7s | %9s | %9s"
          % ("Bin", "N_all", "all%", "N_IN", "N_OUT", "mean_L_all", "mean_exc"))
    print("  " + "-" * 90)

    bin_rows = []
    total_check = 0
    for lo, hi, blabel in bins:
        if hi >= 999:
            bm = (L_s >= lo)
        else:
            bm = (L_s >= lo) & (L_s < hi)
        n_b     = int(bm.sum())
        n_b_in  = int((bm & in_mask).sum())
        n_b_out = int((bm & out_mask).sum())
        all_pct = 100.0 * n_b / n if n > 0 else 0.0
        L_b = L_s[bm]
        mean_L_b  = float(np.mean(L_b))  if n_b > 0 else float("nan")
        exc_b     = np.maximum(L_b - 3.0, 0.0)
        mean_exc_b = float(np.mean(exc_b)) if n_b > 0 else 0.0
        total_check += n_b
        print("  %-24s | %7d | %6.2f%% | %7d | %7d | %+9.4f | %9.4f"
              % (blabel, n_b, all_pct, n_b_in, n_b_out, mean_L_b, mean_exc_b))
        bin_rows.append({
            "bin_label": blabel.strip(),
            "lo": lo, "hi": min(hi, 999.0),
            "n_all": n_b, "all_pct": round(all_pct, 4),
            "n_in": n_b_in, "n_out": n_b_out,
            "mean_L": round(mean_L_b, 4) if not np.isnan(mean_L_b) else None,
            "mean_excess_above3": round(mean_exc_b, 4),
        })
    print("  %-24s | %7d (check = n=%d)" % ("Total", total_check, n))

    print("\n  NOTE: OUT days appear in L<0.5 or 1.0-2.0 bins because")
    print("  the DH-W1 lev_raw_masked on OUT days is masked to near-0 or 1.")
    print("  The actual portfolio on OUT days holds Gold/Bond/Cash (not NASDAQ).")
    print("  >3x bins ONLY arise from IN days where mult_v7*scale*lev_raw > 1.")

    # =========================================================================
    # CSV output
    # =========================================================================
    out_dir = os.path.join(_REPO_DIR, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sc135_weight_leverage_breakdown_20260617.csv")

    # Build a multi-section CSV: section header rows + data rows
    csv_sections = []

    # Section 1: sanity
    csv_sections.append({"section": "sanity", "key": "min9_got_pct",
                          "value": round(got_min9*100, 4), "note": "expect +23.83%"})
    csv_sections.append({"section": "sanity", "key": "maxdd_got_pct",
                          "value": round(got_maxdd*100, 4), "note": "expect -45.04%"})
    csv_sections.append({"section": "sanity", "key": "ok",
                          "value": int(ok_min9 and ok_maxdd), "note": ""})

    # Section 2: day-count
    csv_sections.append({"section": "day_counts", "key": "n_total",   "value": n,        "note": ""})
    csv_sections.append({"section": "day_counts", "key": "n_in",      "value": n_in,     "note": "NASDAQ-active"})
    csv_sections.append({"section": "day_counts", "key": "n_out",     "value": n_out,    "note": "OUT days"})
    csv_sections.append({"section": "day_counts", "key": "n_out_bondon",  "value": n_out_bondon,  "note": ""})
    csv_sections.append({"section": "day_counts", "key": "n_out_bondoff", "value": n_out_bondoff, "note": ""})

    # Section 3: weight aggregates
    for row in wgt_rows:
        csv_sections.append({"section": "agg1_weight",
                              "key": row["asset"],
                              "value": row["all_avg_pct"],
                              "in_value": row["in_avg_pct"],
                              "out_value": row["out_avg_pct"],
                              "note": "pct of capital"})

    # Section 4: IN-day L stats
    L_in_stats = {
        "N_in": len(L_in),
        "mean": float(np.mean(L_in)),
        "median": float(np.median(L_in)),
        "p10": float(np.percentile(L_in, 10)),
        "p25": float(np.percentile(L_in, 25)),
        "p75": float(np.percentile(L_in, 75)),
        "p90": float(np.percentile(L_in, 90)),
        "max": float(np.max(L_in)),
        "mean_all": float(np.mean(L_all)),
    }
    for k, v in L_in_stats.items():
        csv_sections.append({"section": "agg2_L_stats", "key": k,
                              "value": round(v,4), "note": ""})

    # Q breakdown
    for qr in q_rows:
        csv_sections.append({
            "section": "agg2_Q_breakdown",
            "key": "Q%d" % qr["q_label"],
            "value": qr["n_in_days"],
            "in_value": qr["pct_of_in_pct"],
            "out_value": qr["mean_L"],
            "note": "mult=%.2f -> L_exp~%.2f" % (qr["mult_map_value"], qr["expected_L_if_raw1"]),
        })

    # Section 5: >3x excess
    exc_stats = {
        "n_gt3_all": n_gt3_all, "ratio_all_pct": round(ratio_all_pct,4),
        "n_gt3_in": n_gt3_in,   "ratio_in_pct": round(ratio_in_pct,4),
        "mean_exc_given_gt3_all": round(mean_exc_given_gt3_all,6),
        "mean_exc_given_gt3_in": round(mean_exc_given_gt3_in,6),
        "mean_exc_all": round(mean_exc_all,6),
        "mean_exc_in": round(mean_exc_in,6),
    }
    for k, v in exc_stats.items():
        csv_sections.append({"section": "agg3_excess", "key": k, "value": v, "note": ""})

    # Section 6: bin table
    for br in bin_rows:
        csv_sections.append({
            "section": "agg4_bin_table",
            "key": br["bin_label"],
            "value": br["n_all"],
            "in_value": br["n_in"],
            "out_value": br["n_out"],
            "all_pct": br["all_pct"],
            "mean_L": br["mean_L"],
            "mean_excess_above3": br["mean_excess_above3"],
            "note": "",
        })

    pd.DataFrame(csv_sections).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\nSaved CSV: %s  (%d rows)" % (csv_path, len(csv_sections)))

    # =========================================================================
    # RETURN_BLOCK JSON
    # =========================================================================
    return_block = {
        "script": "sc135_weight_leverage_breakdown_20260617.py",
        "date": "2026-06-17",
        "config": {
            "v7_map": B3A_MAP_STRONG,
            "lev_scale": LEV_SCALE,
            "excess_extra_pct": round(EXCESS_EXTRA*100, 4),
        },
        "sanity": {
            "min9_got_pct": round(got_min9*100, 4),
            "maxdd_got_pct": round(got_maxdd*100, 4),
            "ok_min9": bool(ok_min9), "ok_maxdd": bool(ok_maxdd),
            "SANITY_PASS": bool(ok_min9 and ok_maxdd),
        },
        "day_counts": {
            "n_total": n, "n_in": n_in, "n_out": n_out,
            "n_out_bondon": n_out_bondon, "n_out_bondoff": n_out_bondoff,
            "in_pct": round(100.0*n_in/n, 2), "out_pct": round(100.0*n_out/n, 2),
        },
        "agg1_weights": wgt_rows,
        "agg2_L_stats": L_in_stats,
        "agg2_Q_breakdown": q_rows,
        "agg3_excess": exc_stats,
        "agg4_bins": bin_rows,
        "csv_path": csv_path,
    }

    print("\n" + "=" * 120)
    print("RETURN_BLOCK")
    print("=" * 120)
    print(json.dumps(return_block, indent=2, ensure_ascii=True, allow_nan=True))
    print("\nDone.")
    return return_block


if __name__ == "__main__":
    main()
