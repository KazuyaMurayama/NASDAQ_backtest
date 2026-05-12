"""
Yearly Returns Report v4 (2026-05-12)
======================================
Applies SOFR financing cost corrections to ALL 9 strategies:

Corrections applied:
  - TQQQ sleeve: 2xSOFR + 0.50% swap spread + 0.86% TER
  - TMF/Bond 3x sleeve: dgs30 + time-varying Dmod + 2xSOFR + 0.50% swap + 0.91% TER
  - Gold 2x sleeve: 1xSOFR + 0.50% swap spread + 0.50% TER
  - SOFR proxy: DTB3 (FRED 3M T-bill)
  - Splice fix: dgs10 -> dgs30 level correction at 1977-02-15

Per-strategy correction rules:
  DH Dyn 2x3x [A]  : Full Scenario D (TQQQ 2xSOFR + TMF 2xSOFR + Gold 1xSOFR + dgs30+dur17+time-varying Dmod + splice fix)
  DH Dyn 2x3x [B]  : Same corrections as [A] but with [B] signal (lev x whole)
  DH Dyn CAGR25+   : 1x Gold/Bond (no leverage financing on gold/bond) - bond model correction only, no SOFR on gold
  A2 Optimized     : Single-asset TQQQ (TQQQ 2xSOFR when in TQQQ, else CASH)
  Ens2(Asym+Slope) : TQQQ-only sleeve - same as A2
  DD Only          : TQQQ <-> CASH - TQQQ 2xSOFR when in TQQQ
  BH 3x            : 3x NASDAQ B&H: r_t = 3*r_nasdaq - (TER + 2*SOFR_daily + swap_spread/252)
  BH 1x            : No correction (benchmark, no leverage)
  Berkshire Hathaway: External annual data - no correction

Previous version (v3): no SOFR corrections applied to any strategy.
"""
import sys, os, types

# Stub out multitasking to avoid import errors
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# Import from existing modules
from backtest_engine import load_data, calc_dd_signal
from test_ens2_strategies import strategy_ens2_asym_slope
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from test_dynamic_portfolio import build_dynamic_portfolio
from opt_lev2x3x import build_lev_navs, calc_asym_ewma

# Import corrected pipeline components from corrected_strategy_backtest.py
from corrected_strategy_backtest import (
    load_sofr,
    build_bond_1x_nav_corrected,
    build_gold_2x,
    build_bond_3x,
    build_a2_signal,
    simulate_rebalance_A,
    TRADING_DAYS,
    SWAP_SPREAD,
    DELAY,
    BASE_LEV,
    THRESHOLD,
    ANNUAL_COST,    # TQQQ TER = 0.0086
    GOLD_2X_COST,   # 0.0050
    BOND_3X_COST,   # 0.0091 (TMF TER)
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

# ============================================================
# BRK-A per-share market value annual returns (unchanged from v3)
# Source: Warren Buffett 2023 Annual Letter (official)
# ============================================================
brk_annual_returns = {
    1974: -0.487, 1975: 0.025, 1976: 1.293, 1977: 0.468, 1978: 0.145,
    1979: 1.025, 1980: 0.328, 1981: 0.318, 1982: 0.384, 1983: 0.690,
    1984: -0.027, 1985: 0.937, 1986: 0.142, 1987: 0.046, 1988: 0.593,
    1989: 0.846, 1990: -0.231, 1991: 0.356, 1992: 0.298, 1993: 0.389,
    1994: 0.250, 1995: 0.574, 1996: 0.062, 1997: 0.349, 1998: 0.522,
    1999: -0.199, 2000: 0.266, 2001: 0.065, 2002: -0.038, 2003: 0.158,
    2004: 0.043, 2005: 0.008, 2006: 0.241, 2007: 0.287, 2008: -0.318,
    2009: 0.027, 2010: 0.214, 2011: -0.047, 2012: 0.168, 2013: 0.327,
    2014: 0.270, 2015: -0.125, 2016: 0.234, 2017: 0.219, 2018: 0.028,
    2019: 0.110, 2020: 0.024, 2021: 0.296, 2022: 0.040, 2023: 0.158,
    2024: 0.255,
}


# ============================================================
# Helper functions
# ============================================================

def yearly_from_nav(nav, dates):
    """Compute annual returns (%) from a NAV array aligned to dates."""
    ndf = pd.DataFrame({'nav': nav, 'date': dates.values})
    ndf['year'] = pd.to_datetime(ndf['date']).dt.year
    yn = ndf.groupby('year')['nav'].last()
    yr = yn.pct_change() * 100
    first = ndf['nav'].iloc[0]
    yr.iloc[0] = (yn.iloc[0] / first - 1) * 100
    return yr


def period_cagr(nav_arr, dates, start, end):
    """Compute CAGR for a given period."""
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if mask.sum() < 20:
        return np.nan
    idx = np.where(mask)[0]
    n_sub = nav_arr[idx[0]:idx[-1]+1]
    n_norm = n_sub / n_sub[0]
    yrs = len(n_norm) / TRADING_DAYS
    return n_norm[-1]**(1/yrs) - 1 if yrs > 0 and n_norm[-1] > 0 else np.nan


def period_metrics(nav_arr, dates, start, end):
    """Full metrics for a period."""
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if mask.sum() < 20:
        return {}
    idx = np.where(mask)[0]
    n = nav_arr[idx[0]:idx[-1]+1].copy()
    n = n / n[0]
    r = np.diff(n, prepend=n[0]) / np.maximum(np.roll(n, 1), 1e-10)
    r[0] = 0
    yrs = len(n) / TRADING_DAYS
    cagr = n[-1]**(1/yrs) - 1 if yrs > 0 and n[-1] > 0 else np.nan
    sh = (r.mean() * TRADING_DAYS) / (r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0
    running_max = np.maximum.accumulate(n)
    maxdd = ((n / running_max) - 1).min()
    return {'CAGR': cagr, 'Sharpe': sh, 'MaxDD': maxdd}


def calc_brk_cagr(start_year, end_year):
    years = [y for y in range(start_year, end_year+1) if y in brk_annual_returns]
    if not years:
        return np.nan
    prod = 1.0
    for y in years:
        prod *= (1 + brk_annual_returns[y])
    return prod**(1/len(years)) - 1


# ============================================================
# Strategy builders
# ============================================================

def build_bh1x(close):
    """BH 1x: no leverage, no correction."""
    return (close / close.iloc[0]).values


def build_bh3x_corrected(close, sofr_daily):
    """
    BH 3x: 3x NASDAQ buy-and-hold with SOFR correction.
    r_daily = 3*r_nasdaq - (TER + 2*SOFR_daily + swap_spread/252)
    = 3*r_nasdaq - ANNUAL_COST/252 - 2*sofr_daily - SWAP_SPREAD/252
    """
    r_nas = close.pct_change().fillna(0).values
    dc = ANNUAL_COST / TRADING_DAYS
    swap_d = SWAP_SPREAD / TRADING_DAYS
    n = len(r_nas)
    nav = np.ones(n)
    for i in range(1, n):
        # 2xSOFR financing (same as TQQQ: sofr_multiplier=2)
        daily_ret = 3.0 * r_nas[i] - dc - 2.0 * sofr_daily[i] - swap_d
        nav[i] = nav[i-1] * (1 + daily_ret)
    return nav


def build_tqqq_only_corrected(close, signal_lev, sofr_daily, delay=DELAY):
    """
    Single-asset TQQQ strategies (A2, Ens2, DD Only).
    When signal > 0 (in TQQQ): r = 3*r_nasdaq - TER/252 - 2*SOFR - SWAP/252
    When signal = 0 (in cash): r = 0
    signal_lev: daily leverage signal (0 or positive fraction)
    """
    r_nas = close.pct_change().fillna(0).values
    dc = ANNUAL_COST / TRADING_DAYS
    swap_d = SWAP_SPREAD / TRADING_DAYS
    n = len(r_nas)

    lev_s = pd.Series(signal_lev, index=close.index).shift(delay).fillna(0).values

    nav = np.ones(n)
    for i in range(1, n):
        lv = lev_s[i]
        if lv > 0:
            # In TQQQ: pay 2xSOFR financing
            r = lv * (3.0 * r_nas[i] - dc - 2.0 * sofr_daily[i] - swap_d)
        else:
            r = 0.0
        nav[i] = nav[i-1] * (1 + r)
    return nav


def build_dh_cagr25_corrected(nav_a2, gold_1x, bond_1x_nav, raw_v, vz_v, n, sofr_daily, dates):
    """
    DH Dyn CAGR25+: 1x Gold/Bond variant.
    Uses corrected 1x bond (dgs30+dur17+time-varying Dmod) but NO leverage financing on gold/bond.
    The TQQQ sleeve is the corrected A2 NAV.
    Gold 1x: no SOFR (1x unleveraged gold futures/spot, TER ~0.25% already in GLD)
    Bond 1x: corrected model (dgs30+dur17+Dmod), no SOFR (not leveraged)
    """
    n_arr = n
    wn = np.zeros(n_arr); wg = np.zeros(n_arr); wb = np.zeros(n_arr)
    for i in range(n_arr):
        lv = raw_v[i]; vzv = max(vz_v[i], 0)
        w = np.clip(0.50 + 0.25*lv - 0.10*vzv, 0.30, 0.90)
        wn[i] = w; wg[i] = (1-w)*0.55; wb[i] = (1-w)*0.45
    # build_dynamic_portfolio expects (tqqq_nav, gold_1x, bond_1x, wn, wg, wb)
    # Uses the corrected A2 NAV as NASDAQ sleeve and corrected bond 1x
    return build_dynamic_portfolio(nav_a2, gold_1x, bond_1x_nav, wn, wg, wb)


def build_dh_a_corrected(close, lev_A, wn_A, wg_A, wb_A, dates,
                           gold_2x_nav, bond_3x_nav, sofr_daily):
    """
    DH Dyn 2x3x [A]: Full Scenario D correction.
    apply_tqqq_sofr=True in build_nav from corrected_strategy_backtest.
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x_nav).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x_nav).pct_change().fillna(0).values
    dc = ANNUAL_COST / TRADING_DAYS
    swap_d = SWAP_SPREAD / TRADING_DAYS

    lev_s = pd.Series(lev_A, index=dates.index).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn_A,  index=dates.index).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg_A,  index=dates.index).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb_A,  index=dates.index).shift(DELAY).fillna(0).values

    # TQQQ sleeve: 2xSOFR
    nas_ret = r_nas * BASE_LEV - 2.0 * (sofr_daily + swap_d) - dc

    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3
    return (1 + pd.Series(daily, index=dates.index)).cumprod().values


def build_dh_b_corrected(close, lev_s_arr, wn_s_arr, wg_s_arr, wb_s_arr, dates,
                           gold_2x_nav, bond_3x_nav, sofr_daily):
    """
    DH Dyn 2x3x [B]: lev x (wn*nas + wg*gold + wb*bond) - same corrections as [A].
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x_nav).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x_nav).pct_change().fillna(0).values
    dc = ANNUAL_COST / TRADING_DAYS
    swap_d = SWAP_SPREAD / TRADING_DAYS

    nas_ret = r_nas * BASE_LEV - 2.0 * (sofr_daily + swap_d) - dc
    daily = lev_s_arr * (wn_s_arr * nas_ret + wg_s_arr * r_g2 + wb_s_arr * r_b3)
    return (1 + pd.Series(daily, index=dates.index)).cumprod().values


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Gen Yearly Returns v4 -- SOFR+Bond Corrected (2026-05-12)")
    print("=" * 80)

    # Load data
    df = load_data(DATA_PATH)
    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']
    n = len(df)
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n} days, {n/TRADING_DAYS:.1f} years)\n")

    # Load SOFR
    print("Loading SOFR (DTB3)...")
    sofr_daily = load_sofr(dates)
    sofr_mean_ann = np.nanmean(sofr_daily) * TRADING_DAYS * 100
    print(f"  Mean SOFR (full period): {sofr_mean_ann:.2f}%/yr\n")

    # Build corrected bond 1x NAV (Scenario D: dgs30, time-varying Dmod, splice fix)
    print("Building corrected bond models...")
    bond_1x_corr = build_bond_1x_nav_corrected(dates,
                                                 use_time_varying_duration=True,
                                                 bond_maturity=22.0)
    bond_3x_corr = build_bond_3x(bond_1x_corr, sofr_daily, apply_sofr=True)
    print("  Bond 1x corrected (dgs30+Dmod_var+splice) -- done")

    # Build corrected gold 2x NAV (1xSOFR)
    gold_1x = prepare_gold_data(dates)
    gold_2x_corr = build_gold_2x(gold_1x, sofr_daily=sofr_daily, apply_sofr=True)
    print("  Gold 2x corrected (1xSOFR) -- done\n")

    # Build A2 signal (same as v3, shared across all signal-based strategies)
    print("Building A2 signal...")
    raw, vz = build_a2_signal(close, returns)
    raw_v = raw.values
    vz_v = vz.fillna(0).values

    # ---- BH 1x (no correction) ----
    print("Building BH 1x (no correction)...")
    nav_bh1x = build_bh1x(close)

    # ---- BH 3x (corrected: 2xSOFR) ----
    print("Building BH 3x (corrected: 2xSOFR)...")
    nav_bh3x = build_bh3x_corrected(close, sofr_daily)

    # ---- DD Only signal ----
    print("Building DD Only signal...")
    dd_sig = calc_dd_signal(close, 0.82, 0.92)
    lev_dd = rebalance_threshold(dd_sig, THRESHOLD)
    nav_dd = build_tqqq_only_corrected(close, lev_dd.values, sofr_daily, delay=DELAY)

    # ---- Ens2(Asym+Slope) ----
    print("Building Ens2(Asym+Slope)...")
    lev_ens, _ = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    lev_ens = rebalance_threshold(lev_ens, THRESHOLD)
    nav_ens2 = build_tqqq_only_corrected(close, lev_ens.values, sofr_daily, delay=DELAY)

    # ---- A2 Optimized ----
    print("Building A2 Optimized...")
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean(); ratio = close/ma150
    ttv = (0.10+0.20*(ratio-0.85)/0.30).clip(0.10,0.30).fillna(0.20)
    vt = (ttv/av).clip(0,1.0)
    ma200 = close.rolling(200).mean(); sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0,0.0001)
    z = (sl-sm)/ss; slope = (0.9+0.35*z).clip(0.3,1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0,0.001)
    vz_ser = (vp-vma)/vs; vm = (1.0-0.25*vz_ser).clip(0.5,1.15)
    raw_a2 = (calc_dd_signal(close, 0.82, 0.92)*vt*slope*mom*vm).clip(0,1.0).fillna(0)
    lev_a2 = rebalance_threshold(raw_a2, THRESHOLD)
    nav_a2 = build_tqqq_only_corrected(close, lev_a2.values, sofr_daily, delay=DELAY)

    # ---- DH Dyn CAGR25+ ----
    # Uses corrected bond 1x (no lever financing on bond), no SOFR on gold
    # The TQQQ-part is effectively the A2 corrected NAV
    print("Building DH Dyn CAGR25+ (corrected bond model, no SOFR on gold/bond legs)...")
    nav_25plus = build_dh_cagr25_corrected(nav_a2, gold_1x, bond_1x_corr,
                                             raw_v, vz_v, n, sofr_daily, dates)

    # ---- DH Dyn 2x3x [A] (Approach A) -- Full Scenario D ----
    print("Building DH Dyn 2x3x [A] (Full Scenario D correction)...")
    lev_A, wn_A, wg_A, wb_A, n_trades_A = simulate_rebalance_A(raw, vz, THRESHOLD)
    nav_dh_a = build_dh_a_corrected(close, lev_A, wn_A, wg_A, wb_A, dates,
                                      gold_2x_corr, bond_3x_corr, sofr_daily)
    print(f"  [A] Trades: {n_trades_A}, {n_trades_A/(n/TRADING_DAYS):.1f}/yr")

    # ---- DH Dyn 2x3x [B] -- same corrections ----
    print("Building DH Dyn 2x3x [B] (same corrections as [A], lev x whole)...")
    lev_s_arr = pd.Series(lev_A, index=dates.index).shift(DELAY).fillna(0).values
    wn_s_arr  = pd.Series(wn_A, index=dates.index).shift(DELAY).fillna(0).values
    wg_s_arr  = pd.Series(wg_A, index=dates.index).shift(DELAY).fillna(0).values
    wb_s_arr  = pd.Series(wb_A, index=dates.index).shift(DELAY).fillna(0).values
    nav_dh_b = build_dh_b_corrected(close, lev_s_arr, wn_s_arr, wg_s_arr, wb_s_arr, dates,
                                      gold_2x_corr, bond_3x_corr, sofr_daily)

    print("\nAll NAVs built successfully.\n")

    # ============================================================
    # Assemble NAVs dict
    # ============================================================
    navs = {
        'DH Dyn 2x3x [A]':    nav_dh_a,
        'DH Dyn 2x3x [B]':    nav_dh_b,
        'DH Dyn CAGR25+':     nav_25plus,
        'A2 Optimized':        nav_a2,
        'Ens2(Asym+Slope)':   nav_ens2,
        'DD Only':             nav_dd,
        'BH 3x':              nav_bh3x,
        'BH 1x':              nav_bh1x,
    }
    order_8 = list(navs.keys())

    # ============================================================
    # Build yearly returns table
    # ============================================================
    yr_df = pd.DataFrame({name: yearly_from_nav(navs[name], dates) for name in order_8})

    # Add BRK (external, no correction)
    brk_series = pd.Series({yr: ret*100 for yr, ret in brk_annual_returns.items()},
                           name='Berkshire Hathaway')
    yr_df['Berkshire Hathaway'] = brk_series
    order_9 = order_8 + ['Berkshire Hathaway']

    # ============================================================
    # Double-check: NAV endpoint CAGR vs compounded yearly returns
    # ============================================================
    print("=" * 80)
    print("DOUBLE CHECK: NAV endpoint CAGR vs compounded yearly returns")
    print("=" * 80)
    check_ok = True
    for name in order_8:
        nav = navs[name]
        yrs = n / TRADING_DAYS
        cagr_nav = nav[-1]**(1/yrs) - 1
        yearly = yr_df[name] / 100
        cagr_yearly = ((1+yearly).prod())**(1/yrs) - 1
        diff = abs(cagr_nav - cagr_yearly)
        status = "OK" if diff < 0.001 else f"FAIL (diff={diff*100:.3f}pp)"
        if diff >= 0.001:
            check_ok = False
        print(f"  {name:<22} NAV:{cagr_nav*100:+6.2f}%  Yearly:{cagr_yearly*100:+6.2f}%  {status}")
    print()

    # ============================================================
    # BRK double-check
    # ============================================================
    print("=" * 80)
    print("BRK DOUBLE CHECK: 1974-2023 CAGR (expect ~19-20%)")
    print("=" * 80)
    brk_cagr_50y = calc_brk_cagr(1974, 2023)
    brk_cagr_51y = calc_brk_cagr(1974, 2024)
    print(f"  BRK 1974-2023 (50y) CAGR: {brk_cagr_50y*100:.2f}%")
    print(f"  BRK 1974-2024 (51y) CAGR: {brk_cagr_51y*100:.2f}%")
    print()

    # ============================================================
    # DH[A] CAGR verification (must be ~22.50% +/- 1%)
    # ============================================================
    print("=" * 80)
    print("DH[A] VERIFICATION (target ~22.50% for Scenario D)")
    print("=" * 80)
    dha_yrs = n / TRADING_DAYS
    dha_cagr = nav_dh_a[-1]**(1/dha_yrs) - 1
    dha_ok = abs(dha_cagr*100 - 22.50) <= 1.0
    print(f"  DH[A] CAGR: {dha_cagr*100:.2f}% | Target: ~22.50% | {'OK' if dha_ok else 'WARNING: OUTSIDE TOLERANCE'}")
    print()

    # ============================================================
    # Period metrics
    # ============================================================
    FULL_START = '1974-01-02'
    FULL_END   = '2026-12-31'
    IS_END     = '2021-05-07'
    OOS_START  = '2021-05-08'

    print("=" * 80)
    print("Full-period, IS, OOS metrics")
    print("=" * 80)
    print(f"{'Strategy':<22} {'Full CAGR':>10} {'Sharpe':>7} {'MaxDD':>8} {'IS CAGR':>9} {'OOS CAGR':>9}")

    full_metrics = {}
    is_metrics = {}
    oos_metrics = {}
    for name in order_8:
        fm = period_metrics(navs[name], dates, FULL_START, FULL_END)
        im = period_metrics(navs[name], dates, FULL_START, IS_END)
        om = period_metrics(navs[name], dates, OOS_START, FULL_END)
        full_metrics[name] = fm
        is_metrics[name] = im
        oos_metrics[name] = om
        print(f"  {name:<20} {fm.get('CAGR',np.nan)*100:>+8.2f}% "
              f"{fm.get('Sharpe',np.nan):>7.3f} "
              f"{fm.get('MaxDD',np.nan)*100:>7.2f}% "
              f"{im.get('CAGR',np.nan)*100:>+7.2f}% "
              f"{om.get('CAGR',np.nan)*100:>+7.2f}%")

    # BRK
    brk_is_cagr   = calc_brk_cagr(1974, 2020)
    brk_oos_cagr  = calc_brk_cagr(2021, 2024)
    brk_full_cagr = calc_brk_cagr(1974, 2024)
    print(f"  {'Berkshire Hathaway':<20} {brk_full_cagr*100:>+8.2f}%     N/A "
          f"     N/A {brk_is_cagr*100:>+7.2f}% {brk_oos_cagr*100:>+7.2f}%")
    print()

    # ============================================================
    # Annual returns summary
    # ============================================================
    print("=" * 80)
    print("Annual Returns Summary (1974-2025)")
    print("=" * 80)
    years_to_print = sorted(yr_df.index)
    col_hdr = f"{'Year':>4} | {'DH_A':>7} {'DH_B':>7} {'DH25+':>7} {'A2':>7} {'Ens2':>7} {'DD':>7} {'BH3x':>7} {'BH1x':>7} {'BRK':>7}"
    print(col_hdr)
    print("-" * len(col_hdr))
    for yr in years_to_print:
        row = yr_df.loc[yr]
        def fmt(v):
            if pd.isna(v): return "    N/A"
            return f"{v:>+7.1f}"
        print(f"{yr:>4} | {fmt(row.get('DH Dyn 2x3x [A]'))} {fmt(row.get('DH Dyn 2x3x [B]'))} "
              f"{fmt(row.get('DH Dyn CAGR25+'))} {fmt(row.get('A2 Optimized'))} "
              f"{fmt(row.get('Ens2(Asym+Slope)'))} {fmt(row.get('DD Only'))} "
              f"{fmt(row.get('BH 3x'))} {fmt(row.get('BH 1x'))} "
              f"{fmt(row.get('Berkshire Hathaway'))}")
    print()

    # ============================================================
    # Per-strategy summary stats
    # ============================================================
    print("=" * 80)
    print("Per-strategy summary stats (over available years)")
    print("=" * 80)
    for name in order_9:
        col = yr_df[name].dropna()
        if len(col) == 0:
            continue
        yrs_cnt = n / TRADING_DAYS
        if name == 'Berkshire Hathaway':
            prod = (1 + col/100).prod()
            yrs_cnt = len(col)
            cagr_pct = (prod**(1/yrs_cnt)-1)*100
        else:
            nav = navs[name]
            cagr_pct = (nav[-1]**(1/yrs_cnt)-1)*100
        pos_yrs = (col > 0).sum()
        neg_yrs = (col <= 0).sum()
        print(f"  {name:<22} CAGR:{cagr_pct:>+7.2f}%  median:{col.median():>+7.1f}%  "
              f"max:{col.max():>+7.1f}%  min:{col.min():>+7.1f}%  "
              f"+{pos_yrs}/-{neg_yrs} yrs")
    print()

    # ============================================================
    # Save CSV
    # ============================================================
    # Rename columns for CSV brevity
    csv_df = yr_df.copy()
    csv_df.index.name = 'year'
    csv_df.columns = ['DH_A', 'DH_B', 'DH_25plus', 'A2', 'Ens2', 'DD_Only', 'BH_3x', 'BH_1x', 'BRK']
    out_csv = os.path.join(BASE, 'yearly_returns_corrected_v4.csv')
    csv_df.to_csv(out_csv, float_format='%.1f')
    print(f"Saved CSV: {out_csv}")

    return {
        'navs': navs,
        'yr_df': yr_df,
        'dates': dates,
        'full_metrics': full_metrics,
        'is_metrics': is_metrics,
        'oos_metrics': oos_metrics,
        'check_ok': check_ok,
        'brk_full_cagr': brk_full_cagr,
        'brk_is_cagr': brk_is_cagr,
        'brk_oos_cagr': brk_oos_cagr,
        'dha_cagr': dha_cagr,
        'dha_ok': dha_ok,
        'sofr_mean_ann': sofr_mean_ann,
    }


if __name__ == '__main__':
    main()
