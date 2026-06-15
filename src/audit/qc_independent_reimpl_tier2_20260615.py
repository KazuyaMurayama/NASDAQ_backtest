# -*- coding: utf-8 -*-
"""
src/audit/qc_independent_reimpl_tier2_20260615.py
==================================================
QC Tier-2: Zero-shot independent re-implementation of the 5 window/resampling/
regime-stratified metrics in sec.6.1 of LEVERUP_SWEEP_RESULTS_20260612.md.

INDEPENDENCE BOUNDARY (strictly enforced):
  BANNED function calls (import to call is forbidden):
    extended_eval_20260611._run_wfa
    extended_eval_20260611._cpcv_dist
    extended_eval_20260611._regime_cagr
    run_p09_tqqq_validate_20260611._block_bootstrap_compare
    run_p09_tqqq_validate_20260611._run_wfa
    unified_wfa.summarize_wfa
    regime_labeler_20260611 (all functions)
  PERMITTED read-only input:
    audit_results/qc_independent_navs_20260615.csv  -- 5 candidate daily NAVs (Tier1 validated)
    strategy_runners._DHW1_SHARED                   -- raw close/sofr/dates (for regime labels)
    strategy_runners._load_dhw1_shared()
    src/g1_wfa.py                                    -- READ for window boundary spec (no call)
    src/audit/compute_wfa_realistic_20260610.py      -- READ for eval_start/eval_end constants
    src/audit/run_p09_tqqq_validate_20260611.py      -- READ for seed/block/n_boot constants only
  V7_TQQQ BASELINE: independently re-built from raw inputs
    (default map {0:1.20,1:1.10,2:1.00,3:1.00}, scale=1.0, cfd_excess=False,
     no C1 fill, no OUT fill on NASDAQ) -- same logic as Tier1 qc_independent_reimpl_tier1.

INDEPENDENT IMPLEMENTATIONS IN THIS FILE:
  - wfa_windows_canon(): calendar-year anchor windows from g1_wfa spec
  - compute_wfa(): CI95_lo and WFE from per-window CAGRs
  - cpcv_dist(): N=10 blocks, k=2, embargo=21, C(10,2)=45 fold after-tax CAGRs
  - build_regime_labels_ind(): trend/vol/rate labels from close+sofr, independent
  - regime_cagr_ind(): after-tax CAGR per regime label, Regime_min
  - block_bootstrap_vs_v7(): paired block bootstrap, seed=20260611, n_boot=10000, block=21
  - _build_v7_nav_ind(): independent V7 TQQQ baseline NAV

METRICS VALIDATED (sec.6.1 Tier-2 targets):
  WFA CI95_lo / WFE / CPCV p10 / Regime_min / boot P(min better) vs V7

DEVIATION THRESHOLDS (Tier-2 is seed/window-sensitive so tolerances are wider):
  WFA CI95_lo : flag if |delta| >= 1.0 pp
  WFE         : flag if |delta| >= 0.03
  CPCV p10    : flag if |delta| >= 1.0 pp
  Regime_min  : flag if |delta| >= 1.0 pp
  boot P      : flag if |delta| >= 0.05

ASCII-only stdout (Windows cp932). Saves audit_results/qc_independent_tier2_20260615.csv.
Does NOT commit. Does NOT create temp files.
"""
from __future__ import annotations

import json
import os
import sys
import types
from itertools import combinations

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
from scipy import stats

# ===========================================================================
# SECTION 0: CONSTANTS (from codebase reading, not from function calls)
# ===========================================================================
TRADING_DAYS = 252
AFTER_TAX    = 0.8273

# IS/OOS split (from unified_metrics.py / cfd_leverage_backtest.py)
IS_END    = pd.Timestamp("2021-05-07")
OOS_START = pd.Timestamp("2021-05-08")

# WFA canonical window bounds (from g1_wfa.py / compute_wfa_realistic_20260610.py)
WFA_EVAL_START = "1977-01-03"  # LT2 N=750 warmup complete
WFA_EVAL_END   = "2026-03-26"
WFA_WINDOW_MIN_DAYS = 201  # short_flag threshold = 0.8 * 252

# CPCV parameters (from extended_eval_20260611.py)
CPCV_N_BLOCKS = 10
CPCV_K        = 2
CPCV_EMBARGO  = 21  # days

# Bootstrap parameters (from run_p09_tqqq_validate_20260611.py)
N_BOOT = 10000
BLOCK  = 21
SEED   = 20260611

# Regime labeler parameters (from regime_labeler_20260611.py)
MA_TREND       = 200
VOL_WIN        = 63
RATE_LOOKBACK  = 252

# V7 default map (from cost_model_cfd_vs_tqqq_20260611.py)
V7_MAP_DEFAULT = {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00}

# TER / swap from Tier1 / cost_model_cfd_vs_tqqq_20260611.py
TER_TQQQ    = 0.0086
SWAP_SPREAD = 0.0050
DH_PER_UNIT = 0.0010
TER_UGL_REAL = 0.0095; TER_UGL_SIM = 0.0049
TER_TMF_REAL = 0.0106; TER_TMF_SIM = 0.0091
TER_GOLD2X_EXTRA_DAILY = (TER_UGL_REAL - TER_UGL_SIM) / 252.0
TER_TMF_EXTRA_DAILY    = (TER_TMF_REAL - TER_TMF_SIM) / 252.0
LEV_CAP = 3.0


# ===========================================================================
# SECTION 1: LOAD TIER1 NAVs (already validated)
# ===========================================================================

def load_candidate_navs():
    """Load the 5 candidate daily NAVs produced by Tier1."""
    csv_path = os.path.join(_REPO_DIR, "audit_results", "qc_independent_navs_20260615.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    print("Loaded Tier1 NAVs: %d rows, columns=%s" % (len(df), list(df.columns)))
    print("  Date range: %s to %s" % (df.index[0].date(), df.index[-1].date()))
    return df


# ===========================================================================
# SECTION 2: INDEPENDENT V7_TQQQ BASELINE BUILDER
#   - Uses strategy_runners._DHW1_SHARED raw inputs
#   - Default V7_MAP, scale=1.0, cfd_excess=False, no C1, no OUT-fill
#   - No imports from existing NAV builders
# ===========================================================================

def _build_v7_nav_ind():
    """Build V7_TQQQ baseline NAV independently.

    SPEC (from cost_model_cfd_vs_tqqq_20260611.build_nav_v7 with nas_cost_model='TQQQ'):
      INPUTS: close, dates, gold_2x, bond_3x, sofr, lev_raw_masked, wn, wg, wb
              from _DHW1_SHARED (DH-W1 masked weights)
      V7 multiplier: nasdaq_mom63 -> quantile_cut(levels=4) -> pub_lag -> V7_MAP
      lev_mod = lev_raw_masked * mult_v7   (in [0..1] range; the *.values before mult)
      L (actual TQQQ effective leverage) = lev_mod * 3.0, DELAY-shifted by 2 days
      NASDAQ leg: r_nas = L * r_close
                         - max(L-1, 0) * (sofr + SWAP_SPREAD/252)
                         - TER_TQQQ / 252
      DH turnover: applied on RAW (unshifted) wn/wg/wb changes * DH_PER_UNIT
      Portfolio daily = wn_shifted * r_nas + wg_shifted * r_gold + wb_shifted * r_bond
                        - turn * DH_PER_UNIT   (subtracted from daily directly)
      (daily clipped at NAV_FLOOR), nav_sim = cumprod(1+daily)
      Then incremental TER drag on GOLD/BOND legs and US-ETF trade cost applied as:
        r_adj = r_sim - (wg * TER_GOLD2X_EXTRA_DAILY + wb * TER_TMF_EXTRA_DAILY) - etf_daily
      Final NAV = cumprod(1 + r_adj)

    NOTE: The DH-W1 wn/wg/wb are already masked (OUT-period weights set to 0).
    NOTE: lev_raw_masked is in [0..1] range (DH-W1 modulates exposure [0..1]).
    NOTE: NAV starts at 1.0 at the first day.
    """
    import src.audit.strategy_runners as sr
    sr._load_dhw1_shared()
    shared = sr._DHW1_SHARED
    a = shared["assets"]

    close          = a["close"]
    dates          = a["dates"]
    sofr           = np.asarray(a["sofr"], float)       # daily rate
    gold_2x        = np.asarray(a["gold_2x"], float)
    bond_3x        = np.asarray(a["bond_3x"], float)
    lev_raw_masked = np.asarray(shared["lev_raw_masked"], float)  # [0..1] range
    wn             = np.asarray(shared["wn"], float)
    wg             = np.asarray(shared["wg"], float)
    wb             = np.asarray(shared["wb"], float)
    dates_dt       = pd.DatetimeIndex(pd.to_datetime(dates.values))
    n              = len(dates_dt)
    idx            = dates.index

    # ---- V7 multiplier ----
    macro_path = os.path.join(_REPO_DIR, "data", "macro_features.csv")
    macro = pd.read_csv(macro_path, parse_dates=["date"], index_col="date").sort_index()

    from signals.quantize import quantile_cut as _qcut
    from signals.timing import apply_publication_lag as _apply_lag

    signal_raw = macro["nasdaq_mom63"].dropna()
    sig_q = _qcut(signal_raw, levels=4).dropna().astype("int8")
    sig_lagged = _apply_lag(sig_q, "daily")
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep="last")]
    sig_aligned = sig_lagged.reindex(dates_dt).ffill()

    mult_v7 = sig_aligned.map(
        lambda s: V7_MAP_DEFAULT.get(int(s), 1.0) if pd.notna(s) else 1.0
    ).fillna(1.0).values
    mult_v7 = np.clip(np.asarray(mult_v7, float), 0.0, 3.0)

    # ---- lev_mod = lev_raw_masked * mult_v7 ----
    lev_mod = np.asarray(lev_raw_masked, float) * mult_v7

    # ---- L (effective TQQQ leverage) = lev_mod * 3.0, DELAY=2 shifted ----
    DELAY = 2
    L = pd.Series(lev_mod * 3.0, index=idx).shift(DELAY).fillna(1.0).values

    # ---- Shifted weights ----
    wn_s = pd.Series(np.asarray(wn, float), index=idx).shift(DELAY).fillna(0).values
    wg_s = pd.Series(np.asarray(wg, float), index=idx).shift(DELAY).fillna(0).values
    wb_s = pd.Series(np.asarray(wb, float), index=idx).shift(DELAY).fillna(0).values

    # ---- Returns ----
    r_close = np.asarray(close.pct_change().fillna(0).values, float)
    r_gold  = pd.Series(np.asarray(gold_2x, float)).pct_change().fillna(0).values
    r_bond  = pd.Series(np.asarray(bond_3x, float)).pct_change().fillna(0).values

    # ---- NASDAQ leg (TQQQ cost model) ----
    sofr_arr = np.asarray(sofr, float)
    borrow   = np.maximum(L - 1.0, 0.0) * (sofr_arr + SWAP_SPREAD / TRADING_DAYS)
    r_nas    = L * r_close - borrow - TER_TQQQ / TRADING_DAYS

    # ---- DH turnover on RAW (unshifted) wn/wg/wb changes ----
    d_wn = np.zeros(n, float); d_wn[1:] = np.abs(np.diff(np.asarray(wn, float)))
    d_wg = np.zeros(n, float); d_wg[1:] = np.abs(np.diff(np.asarray(wg, float)))
    d_wb = np.zeros(n, float); d_wb[1:] = np.abs(np.diff(np.asarray(wb, float)))
    turn = d_wn + d_wg + d_wb

    # ---- Daily portfolio return (sim level) ----
    daily = wn_s * r_nas + wg_s * r_gold + wb_s * r_bond
    daily = np.maximum(daily, -0.999)
    nav_sim = pd.Series(np.cumprod(1.0 + (daily - turn * DH_PER_UNIT)), index=idx)

    # ---- Incremental ETF TER + US-ETF trade cost ----
    ter_drag = (np.asarray(wg, float) * TER_GOLD2X_EXTRA_DAILY
                + np.asarray(wb, float) * TER_TMF_EXTRA_DAILY)

    # ETF trade cost (from product_costs_realistic_20260610.us_etf_trade_cost_annual)
    from src.audit.product_costs_realistic_20260610 import us_etf_trade_cost_annual
    from src.audit.strategy_runners import _compute_dhw1_trades_per_year
    tpy = _compute_dhw1_trades_per_year(lev_mod, dates)
    etf_daily = us_etf_trade_cost_annual(tpy) / TRADING_DAYS

    r_sim = np.nan_to_num(nav_sim.pct_change().fillna(0).values, nan=0.0)
    r_adj = r_sim - ter_drag - etf_daily
    nav_arr = np.cumprod(1.0 + r_adj)
    nav_dt  = pd.Series(nav_arr, index=dates_dt)

    print("V7_TQQQ baseline: NAV range [%.4f, %.4f], end=%.4f, tpy=%.1f"
          % (nav_arr.min(), nav_arr.max(), nav_arr[-1], tpy))

    return nav_dt, r_adj, dates_dt, shared


# ===========================================================================
# SECTION 3: INDEPENDENT WFA
#   Canonical calendar-year window generation (mirrors g1_wfa.generate_windows)
#   and CI95_lo / WFE computation (mirrors g1_wfa.compute_summary_stats)
# ===========================================================================

def wfa_windows_canon(dates_dt: pd.DatetimeIndex) -> list:
    """Generate canonical calendar-year anchor windows.

    SPEC (from g1_wfa.generate_windows):
      - eval_start = '1977-01-03', eval_end = '2026-03-26'
      - Per year: window_start = max(Jan1, eval_start), window_end = min(Dec31, eval_end)
      - First trading day >= window_start as start_idx
      - Last trading day <= window_end as end_idx
      - short_flag = True if n_days < 201 (0.8 * 252)
    Returns list of dicts with keys: year, start_idx, end_idx, n_days, short_flag
    """
    eval_start_ts = pd.Timestamp(WFA_EVAL_START)
    eval_end_ts   = pd.Timestamp(WFA_EVAL_END)
    first_year    = eval_start_ts.year
    last_year     = eval_end_ts.year
    dates_arr     = np.asarray(dates_dt)  # numpy datetime64 array

    windows = []
    for year in range(first_year, last_year + 1):
        yr_start_ts = max(pd.Timestamp("%d-01-01" % year), eval_start_ts)
        yr_end_ts   = min(pd.Timestamp("%d-12-31" % year), eval_end_ts)

        # First trading day >= yr_start
        mask_s = (dates_dt >= yr_start_ts)
        if not mask_s.any():
            break
        s_idx = int(np.argmax(mask_s))

        # Last trading day <= yr_end
        mask_e = (dates_dt <= yr_end_ts)
        if not mask_e.any():
            break
        e_idx = int(np.where(mask_e)[0][-1])

        if e_idx < s_idx:
            break

        n_days = e_idx - s_idx + 1
        windows.append(dict(
            year=year, start_idx=s_idx, end_idx=e_idx,
            n_days=n_days,
            short_flag=(n_days < WFA_WINDOW_MIN_DAYS),
        ))
    return windows


def compute_wfa(nav_arr: np.ndarray, windows: list) -> dict:
    """Compute WFA CI95_lo and WFE independently.

    SPEC (from g1_wfa.compute_summary_stats):
      - Exclude short_flag=True windows
      - CAGR per window = (nav[e_idx] / nav[s_idx]) ^ (252/n_days) - 1
      - mean / std over valid CAGRs -> t-distribution CI95_lo
      - WFE = mean_CAGR_postIS / mean_CAGR_IS
        where IS: window start < OOS_START (2021-05-08)
              postIS: window start >= OOS_START
    """
    cagrs = []
    is_cagrs = []
    post_cagrs = []

    for w in windows:
        if w["short_flag"]:
            continue
        s, e = w["start_idx"], w["end_idx"]
        seg = nav_arr[s: e + 1]
        if len(seg) < 2 or seg[0] <= 0:
            continue
        n_days = len(seg)
        n_years = n_days / float(TRADING_DAYS)
        cagr = (seg[-1] / seg[0]) ** (1.0 / n_years) - 1.0
        cagrs.append(cagr)

        # IS vs postIS: use window start index to determine
        # The window start date is dates_dt[s_idx]
        w_start = w.get("start_dt")
        if w_start is not None:
            if w_start < OOS_START:
                is_cagrs.append(cagr)
            else:
                post_cagrs.append(cagr)

    if len(cagrs) == 0:
        return dict(CI95_lo=float("nan"), WFE=float("nan"), n_windows=0,
                    mean_CAGR=float("nan"), n_IS=0, n_postIS=0)

    n = len(cagrs)
    mean_c = float(np.mean(cagrs))
    std_c  = float(np.std(cagrs, ddof=1)) if n > 1 else float("nan")
    se     = std_c / np.sqrt(n) if (not np.isnan(std_c) and n > 1) else float("nan")

    if np.isnan(se) or se == 0:
        ci95_lo = float("nan")
    else:
        t_crit  = float(stats.t.ppf(0.975, df=n - 1))
        ci95_lo = mean_c - t_crit * se

    mean_is   = float(np.mean(is_cagrs))   if len(is_cagrs)   > 0 else float("nan")
    mean_post = float(np.mean(post_cagrs)) if len(post_cagrs) > 0 else float("nan")
    if np.isnan(mean_is) or mean_is == 0:
        wfe = float("nan")
    else:
        wfe = mean_post / mean_is

    return dict(
        CI95_lo=ci95_lo,
        WFE=wfe,
        n_windows=n,
        mean_CAGR=mean_c,
        n_IS=len(is_cagrs),
        n_postIS=len(post_cagrs),
    )


# ===========================================================================
# SECTION 4: INDEPENDENT CPCV
#   N=10 blocks, k=2, embargo=21, C(10,2)=45 folds, after-tax CAGR
#   Mirrors extended_eval_20260611._cpcv_dist
# ===========================================================================

def _cagr_seg_ind(r: np.ndarray) -> float:
    """CAGR from daily return array. Returns raw (pre-tax)."""
    n = len(r)
    if n == 0:
        return float("nan")
    nav_end = float(np.prod(1.0 + np.clip(r, -0.999, None)))
    if nav_end <= 0:
        return -1.0
    return nav_end ** (float(TRADING_DAYS) / n) - 1.0


def cpcv_dist_ind(r: np.ndarray,
                  n_blocks: int = CPCV_N_BLOCKS,
                  k: int = CPCV_K,
                  embargo: int = CPCV_EMBARGO) -> np.ndarray:
    """After-tax CAGR over each C(n_blocks,k) combination of contiguous test blocks.

    SPEC (from extended_eval_20260611._cpcv_dist):
      - Divide r into n_blocks equal contiguous blocks using np.linspace
      - For each C(n_blocks, k) combination:
          - Collect all indices in the selected blocks, apply embargo to start of each block
          - Compute after-tax CAGR = _cagr_seg(r[idx]) * AFTER_TAX
      - Return array of 45 values

    INDEPENDENCE NOTE: We re-implement using the same boundary formula:
      bounds = np.linspace(0, n, n_blocks+1).astype(int)
    """
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
        at_cagr = _cagr_seg_ind(rr) * AFTER_TAX
        out.append(at_cagr)

    return np.asarray(out, float)


# ===========================================================================
# SECTION 5: INDEPENDENT REGIME LABELER
#   Mirrors regime_labeler_20260611.build_regime_labels (read for spec).
#   trend: close vs 200d MA. vol: 63d realized vol vs full-sample median.
#   rate: SOFR level today vs 252 trading days ago.
# ===========================================================================

def build_regime_labels_ind(close: pd.Series,
                             sofr_daily: np.ndarray,
                             dates_dt: pd.DatetimeIndex) -> pd.DataFrame:
    """Build per-day regime labels (trend/vol/rate) independently.

    SPEC (from regime_labeler_20260611.build_regime_labels):
      trend: 'bull' if close_t > MA200_t else 'bear'; 'n/a' while MA warming up
      vol  : 63d realized annualized vol vs full-sample median
             'highvol' if rv > median else 'calm'; 'n/a' while warming up
      rate : SOFR_annual_t vs SOFR_annual_{t-252}
             'rate_up' if current > past else 'rate_down'; 'n/a' while past is nan
    NOTE: vol median is full-sample (ex-post), same as canonical.
    SOFR input is per-day (annual/252) from _DHW1_SHARED -> sofr.
    Convert to annual: sofr_annual = sofr_daily * 252
    """
    close_arr = pd.Series(np.asarray(close, float), index=dates_dt)
    ret       = close_arr.pct_change().fillna(0.0)

    # trend: 200d trailing MA
    ma200 = close_arr.rolling(MA_TREND, min_periods=MA_TREND).mean()
    trend_vals = np.where(close_arr.values > ma200.values, "bull", "bear")
    trend_vals = np.where(np.isnan(ma200.values), "n/a", trend_vals)

    # vol: 63d realized vol annualized
    rv63 = ret.rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=1) * np.sqrt(TRADING_DAYS)
    vol_med = float(np.nanmedian(rv63.values))
    vol_vals = np.where(rv63.values > vol_med, "highvol", "calm")
    vol_vals = np.where(np.isnan(rv63.values), "n/a", vol_vals)

    # rate: SOFR level (annualized) today vs 252 days ago
    sofr_annual = pd.Series(np.asarray(sofr_daily, float) * TRADING_DAYS, index=dates_dt)
    sofr_past   = sofr_annual.shift(RATE_LOOKBACK)
    rate_vals   = np.where(sofr_annual.values > sofr_past.values, "rate_up", "rate_down")
    rate_vals   = np.where(np.isnan(sofr_past.values), "n/a", rate_vals)

    df = pd.DataFrame({
        "trend": trend_vals,
        "vol":   vol_vals,
        "rate":  rate_vals,
    }, index=dates_dt)
    df.attrs["vol_median"] = vol_med
    return df


def regime_cagr_ind(r: np.ndarray, regimes: pd.DataFrame) -> tuple:
    """Compute after-tax CAGR within each regime label.

    SPEC (from extended_eval_20260611._regime_cagr):
      For each axis in (trend, vol, rate):
        For each unique label != 'n/a':
          mask = (regimes[axis] == label)
          CAGR = _cagr_seg(r[mask]) * AFTER_TAX
      Regime_min = min over all values
    """
    out = {}
    for ax in ("trend", "vol", "rate"):
        labs = regimes[ax].values
        for lab in set(labs):
            if lab == "n/a":
                continue
            mask = (labs == lab)
            if mask.sum() == 0:
                continue
            out["%s:%s" % (ax, lab)] = _cagr_seg_ind(r[mask]) * AFTER_TAX
    regime_min = float(np.nanmin(list(out.values()))) if out else float("nan")
    return out, regime_min


# ===========================================================================
# SECTION 6: INDEPENDENT BLOCK BOOTSTRAP VS V7
#   Paired stationary-block bootstrap.
#   Mirrors run_p09_tqqq_validate_20260611._block_bootstrap_compare
# ===========================================================================

def _maxdd_from_returns_ind(r: np.ndarray) -> float:
    """MaxDD from daily returns (less negative = better)."""
    nav = np.cumprod(1.0 + np.clip(r, -0.999, None))
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def block_bootstrap_vs_v7(r_strat: np.ndarray,
                           r_base: np.ndarray,
                           is_mask: np.ndarray,
                           oos_mask: np.ndarray,
                           n_boot: int = N_BOOT,
                           block: int = BLOCK,
                           seed: int = SEED) -> dict:
    """Paired stationary-block bootstrap.

    SPEC (from run_p09_tqqq_validate_20260611._block_bootstrap_compare):
      - rng = np.random.default_rng(seed)
      - n_blocks = ceil(n / block)
      - Per boot iteration:
          starts = rng.integers(0, n, size=n_blocks)  [uniform on [0,n))
          idx = (starts[:,None] + arange(block)[None,:]).ravel() % n
          idx = idx[:n]   [trim to exactly n]
          rs = r_strat[idx]; rb = r_base[idx]
          im = is_mask[idx]; om = oos_mask[idx]
          s_is = _cagr_seg(rs[im]) * AFTER_TAX
          s_oos= _cagr_seg(rs[om]) * AFTER_TAX
          b_is = _cagr_seg(rb[im]) * AFTER_TAX
          b_oos= _cagr_seg(rb[om]) * AFTER_TAX
          s_min = nanmin(s_is, s_oos); b_min = nanmin(b_is, b_oos)
          strat_better_min += (s_min > b_min)
      - P_min_better = strat_better_min / n_boot
    """
    rng = np.random.default_rng(seed)
    n   = len(r_strat)
    r_strat   = np.asarray(r_strat, float)
    r_base    = np.asarray(r_base, float)
    is_mask   = np.asarray(is_mask, bool)
    oos_mask  = np.asarray(oos_mask, bool)

    n_blocks = int(np.ceil(n / block))
    strat_better_min = 0
    d_min = np.empty(n_boot, float)

    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]

        rs = r_strat[idx]; rb = r_base[idx]
        im = is_mask[idx]; om = oos_mask[idx]

        s_is  = _cagr_seg_ind(rs[im]) * AFTER_TAX
        s_oos = _cagr_seg_ind(rs[om]) * AFTER_TAX
        b_is  = _cagr_seg_ind(rb[im]) * AFTER_TAX
        b_oos = _cagr_seg_ind(rb[om]) * AFTER_TAX

        s_min = float(np.nanmin([s_is, s_oos]))
        b_min = float(np.nanmin([b_is, b_oos]))

        d_min[b] = s_min - b_min
        if s_min > b_min:
            strat_better_min += 1

    return {
        "P_min_better": strat_better_min / float(n_boot),
        "CI95_lo_pp": float(np.percentile(d_min, 2.5)) * 100.0,
        "mean_improve_pp": float(np.mean(d_min)) * 100.0,
        "n_boot": n_boot,
    }


# ===========================================================================
# SECTION 7: SELF-TESTS
# ===========================================================================

def self_tests(windows: list, dates_dt: pd.DatetimeIndex) -> bool:
    """Verify window counts and CPCV fold counts before evaluation."""
    ok = True

    # WFA window count: expect 50 total (1977..2026), 1 short (2026 is partial)
    n_total = len(windows)
    n_valid = sum(1 for w in windows if not w["short_flag"])
    print("  [SELF-TEST] WFA windows: total=%d, valid(short_flag=False)=%d" % (n_total, n_valid))
    # g1_wfa.py docstring says ~49 windows (2026 excluded as short).
    # We allow 48-50 as valid range.
    if n_valid < 48 or n_valid > 51:
        print("    WARNING: expected ~49 valid windows, got %d" % n_valid)
        ok = False
    else:
        print("    OK: ~49 valid windows")

    # CPCV fold count: C(10,2) = 45
    n_folds = len(list(combinations(range(CPCV_N_BLOCKS), CPCV_K)))
    print("  [SELF-TEST] CPCV C(%d,%d) = %d folds (expected 45)" % (CPCV_N_BLOCKS, CPCV_K, n_folds))
    if n_folds != 45:
        print("    FAIL: expected 45 folds")
        ok = False
    else:
        print("    OK: 45 folds")

    # IS/postIS split sanity: OOS_START = 2021-05-08
    n_is   = sum(1 for w in windows if (not w["short_flag"]) and w.get("start_dt") is not None and w["start_dt"] < OOS_START)
    n_post = sum(1 for w in windows if (not w["short_flag"]) and w.get("start_dt") is not None and w["start_dt"] >= OOS_START)
    print("  [SELF-TEST] WFA IS windows=%d, postIS windows=%d" % (n_is, n_post))
    if n_is < 40 or n_post < 3:
        print("    WARNING: IS/postIS split looks unusual")
        ok = False
    else:
        print("    OK: IS/postIS split")

    return ok


# ===========================================================================
# SECTION 8: MAIN EVALUATION
# ===========================================================================

# sec.6.1 target values
SEC61 = {
    "P09_C1":       {"CI95_lo": 0.1896, "WFE": 0.989, "CPCV_p10": 0.1415, "Regime_min": -0.0008, "boot_P": 0.820},
    "LU1_C1_k365":  {"CI95_lo": 0.1968, "WFE": 0.985, "CPCV_p10": 0.1457, "Regime_min": -0.0066, "boot_P": 0.840},
    "LU2_C1_k365":  {"CI95_lo": 0.2106, "WFE": 0.965, "CPCV_p10": 0.1534, "Regime_min": -0.0115, "boot_P": 0.858},
    "B3a_k365":     {"CI95_lo": 0.2252, "WFE": 0.987, "CPCV_p10": 0.1601, "Regime_min": -0.0288, "boot_P": 0.893},
    "B3c_k365":     {"CI95_lo": 0.2175, "WFE": 0.995, "CPCV_p10": 0.1563, "Regime_min": -0.0245, "boot_P": 0.888},
}

THRESHOLDS = {
    "CI95_lo": 1.0e-2,    # 1.0 pp
    "WFE":     0.03,
    "CPCV_p10": 1.0e-2,   # 1.0 pp
    "Regime_min": 1.0e-2, # 1.0 pp
    "boot_P":  0.05,
}


def main():
    print("=" * 80)
    print("QC TIER-2: Independent re-implementation  2026-06-15")
    print("Metrics: WFA CI95_lo / WFE / CPCV p10 / Regime_min / boot P(min better)")
    print("=" * 80)

    # ---- Load candidate NAVs ----
    print("\n[1] Loading Tier1 NAVs ...")
    cand_df = load_candidate_navs()
    cand_names = [c for c in cand_df.columns]
    dates_dt = pd.DatetimeIndex(cand_df.index)
    n = len(dates_dt)

    # ---- Build V7_TQQQ baseline independently ----
    print("\n[2] Building V7_TQQQ baseline (independent) ...")
    v7_nav_dt, r_v7, dates_dt_v7, shared = _build_v7_nav_ind()

    # Align V7 to candidate date range
    v7_nav_aligned = v7_nav_dt.reindex(dates_dt)
    r_v7_aligned   = v7_nav_aligned.pct_change().fillna(0.0).values

    # Sanity: V7 should have ~16% min(IS,OOS) after-tax
    is_mask  = np.asarray(dates_dt <= IS_END)
    oos_mask = np.asarray(dates_dt >= OOS_START)
    r_v7_vals = r_v7_aligned

    is_cagr_v7  = _cagr_seg_ind(r_v7_vals[is_mask])  * AFTER_TAX
    oos_cagr_v7 = _cagr_seg_ind(r_v7_vals[oos_mask]) * AFTER_TAX
    min_at_v7   = min(is_cagr_v7, oos_cagr_v7)
    print("  V7 sanity: IS_at=%+.2f%%, OOS_at=%+.2f%%, min_at=%+.2f%% (expect ~16.27%%)"
          % (is_cagr_v7 * 100, oos_cagr_v7 * 100, min_at_v7 * 100))
    if abs(min_at_v7 * 100 - 16.27) > 1.0:
        print("  WARNING: V7 min_at deviates from 16.27%% by more than 1pp. Check baseline.")
    else:
        print("  V7 baseline: OK")

    # ---- Build WFA windows (calendar-year anchor) ----
    print("\n[3] Building canonical WFA windows ...")
    windows = wfa_windows_canon(dates_dt)
    # Attach start_dt for IS/postIS classification
    for w in windows:
        w["start_dt"] = dates_dt[w["start_idx"]]
    ok = self_tests(windows, dates_dt)
    if not ok:
        print("  WARN: Self-tests did not all pass. Proceeding with caution.")

    # ---- Load raw inputs for regime labels ----
    print("\n[4] Building regime labels ...")
    a       = shared["assets"]
    close   = a["close"]
    sofr_dh = np.asarray(a["sofr"], float)   # daily rate
    dates_s = a["dates"]
    dates_dt_sr = pd.DatetimeIndex(pd.to_datetime(dates_s.values))

    # Build regime labels on the shared date range, then reindex to candidate dates
    regimes_full = build_regime_labels_ind(close, sofr_dh, dates_dt_sr)
    regimes = regimes_full.reindex(dates_dt)
    print("  Regime label coverage (non-n/a fraction):")
    for ax in ("trend", "vol", "rate"):
        vc = regimes[ax][regimes[ax] != "n/a"].value_counts(normalize=True)
        print("    %-6s : %s" % (ax, dict(vc.round(3))))
    print("  vol_median (annualized) = %.4f" % regimes_full.attrs.get("vol_median", float("nan")))

    # ---- Evaluate each candidate ----
    print("\n[5] Evaluating 5 candidates x 5 metrics ...")
    results = {}

    for cname in cand_names:
        nav_vals = cand_df[cname].values
        nav_s    = pd.Series(nav_vals, index=dates_dt)

        # daily returns
        r = np.zeros(n, float)
        r[1:] = nav_vals[1:] / nav_vals[:-1] - 1.0

        print("\n  --- %s ---" % cname)

        # (A) WFA
        wfa_res = compute_wfa(nav_vals, windows)
        ci95_lo = wfa_res["CI95_lo"]
        wfe     = wfa_res["WFE"]
        print("    WFA: CI95_lo=%+.2f%%  WFE=%.3f  n_windows=%d  n_IS=%d  n_postIS=%d"
              % (ci95_lo * 100, wfe, wfa_res["n_windows"], wfa_res["n_IS"], wfa_res["n_postIS"]))

        # (B) CPCV p10
        cpcv = cpcv_dist_ind(r)
        cpcv_p10 = float(np.percentile(cpcv, 10))
        print("    CPCV: n_folds=%d, p10=%+.2f%%  (min=%+.2f%%, median=%+.2f%%)"
              % (len(cpcv), cpcv_p10 * 100, float(np.min(cpcv)) * 100, float(np.median(cpcv)) * 100))

        # (C) Regime_min
        reg_dict, regime_min = regime_cagr_ind(r, regimes)
        print("    Regime_min = %+.2f%%" % (regime_min * 100))
        for k, v in sorted(reg_dict.items()):
            print("      %s: %+.2f%%" % (k, v * 100))

        # (D) Boot P(min better) vs V7
        boot = block_bootstrap_vs_v7(r, r_v7_aligned, is_mask, oos_mask)
        print("    Boot P(min better) = %.3f  CI95_lo=%+.2fpp  mean=%+.2fpp"
              % (boot["P_min_better"], boot["CI95_lo_pp"], boot["mean_improve_pp"]))

        results[cname] = {
            "CI95_lo":    ci95_lo,
            "WFE":        wfe,
            "CPCV_p10":   cpcv_p10,
            "Regime_min": regime_min,
            "boot_P":     boot["P_min_better"],
            "wfa_n":      wfa_res["n_windows"],
            "wfa_n_IS":   wfa_res["n_IS"],
            "wfa_n_postIS": wfa_res["n_postIS"],
            "regime_detail": reg_dict,
            "boot_ci95_lo_pp": boot["CI95_lo_pp"],
        }

    # ---- Comparison table ----
    print("\n" + "=" * 100)
    print("TIER-2 VERIFICATION TABLE  (5 metrics x 5 candidates)")
    print("=" * 100)
    metric_keys = ["CI95_lo", "WFE", "CPCV_p10", "Regime_min", "boot_P"]
    metric_units = {"CI95_lo": "%", "WFE": "", "CPCV_p10": "%", "Regime_min": "%", "boot_P": ""}

    rows_csv = []
    flag_list = []

    for cname in cand_names:
        res = results[cname]
        ref = SEC61.get(cname, {})
        print("\n  %s" % cname)
        hdr = "  %-12s | %10s | %10s | %10s | %5s" % ("metric", "independent", "sec.6.1", "delta", "FLAG")
        print(hdr)
        print("  " + "-" * 60)
        for mk in metric_keys:
            ind_val = res[mk]
            ref_val = ref.get(mk, float("nan"))
            delta   = ind_val - ref_val
            thr     = THRESHOLDS[mk]
            is_pct  = metric_units[mk] == "%"
            flagged = (abs(delta) >= thr)
            flag    = "FLAG" if flagged else "ok"
            if is_pct:
                print("  %-12s | %+9.2f%% | %+9.2f%% | %+9.2f%% | %s"
                      % (mk, ind_val * 100, ref_val * 100, delta * 100, flag))
            else:
                print("  %-12s | %10.3f | %10.3f | %10.3f | %s"
                      % (mk, ind_val, ref_val, delta, flag))

            rows_csv.append({
                "candidate": cname,
                "metric":    mk,
                "independent": ind_val,
                "sec61":     ref_val,
                "delta":     delta,
                "threshold": thr,
                "flagged":   flagged,
                "flag":      flag,
            })
            if flagged:
                flag_list.append((cname, mk, ind_val, ref_val, delta))

    # ---- Summary of flags ----
    print("\n" + "=" * 80)
    if flag_list:
        print("FLAGGED (|delta| >= threshold):")
        for cname, mk, ind_v, ref_v, dv in flag_list:
            thr = THRESHOLDS[mk]
            is_pct = metric_units.get(mk, "") == "%"
            if is_pct:
                print("  %s / %s : ind=%+.2f%% ref=%+.2f%% delta=%+.2f%% (thr=%.2f%%)"
                      % (cname, mk, ind_v * 100, ref_v * 100, dv * 100, thr * 100))
            else:
                print("  %s / %s : ind=%.3f ref=%.3f delta=%.3f (thr=%.3f)"
                      % (cname, mk, ind_v, ref_v, dv, thr))
    else:
        print("ALL METRICS WITHIN TOLERANCE: No flags raised.")
    print("=" * 80)

    # ---- Save CSV ----
    out_csv = os.path.join(_REPO_DIR, "audit_results", "qc_independent_tier2_20260615.csv")
    pd.DataFrame(rows_csv).to_csv(out_csv, index=False, float_format="%.6f")
    print("\nSaved CSV: %s" % out_csv)

    # ---- RETURN_BLOCK ----
    ret_block = {}
    for cname in cand_names:
        res = results[cname]
        ret_block[cname] = {
            "CI95_lo_pct":    round(res["CI95_lo"] * 100, 4),
            "WFE":            round(res["WFE"], 4),
            "CPCV_p10_pct":   round(res["CPCV_p10"] * 100, 4),
            "Regime_min_pct": round(res["Regime_min"] * 100, 4),
            "boot_P":         round(res["boot_P"], 4),
        }

    ret_block["meta"] = {
        "n_wfa_windows": len(windows),
        "n_valid_windows": sum(1 for w in windows if not w["short_flag"]),
        "cpcv_folds": 45,
        "n_boot": N_BOOT,
        "seed":   SEED,
        "block":  BLOCK,
        "n_flags": len(flag_list),
        "flag_list": ["%s/%s" % (c, m) for c, m, _, _, _ in flag_list],
    }

    print("\nRETURN_BLOCK")
    print(json.dumps(ret_block, indent=2, ensure_ascii=False))

    return ret_block


if __name__ == "__main__":
    main()
