"""
2022 Bear Market Deep Dive + VIX/Stress Index Integration
=========================================================
D: Why baseline beats all improvements in 2022
A: VIX as leading indicator, financial stress index as regime input
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult
)
from test_improvements import (
    DATA_PATH, DEFAULT_DELAY, ANNUAL_COST,
    strategy_baseline_momdecel_ens2_st, calc_asym_har_vol
)

DEFAULT_THRESHOLD = 0.20


# =============================================================================
# Part D: Diagnose 2022 — what each layer does differently
# =============================================================================
def diagnose_2022(close, dates):
    """Analyze each strategy layer's behavior in 2022 to understand
    why improvements hurt OOS performance."""
    returns = close.pct_change()
    mask_2022 = (dates.dt.year == 2022)
    idx_2022 = dates[mask_2022].index

    print("=" * 90)
    print("DIAGNOSIS: Strategy Layer Behavior in 2022")
    print("=" * 90)

    # DD Control
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    dd_exits = []
    dd_entries = []
    state = 'HOLD'
    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    for i in range(len(close)):
        if state == 'HOLD' and ratio.iloc[i] <= 0.82:
            state = 'CASH'
            if i in idx_2022:
                dd_exits.append(dates.iloc[i].strftime('%Y-%m-%d'))
        elif state == 'CASH' and ratio.iloc[i] >= 0.92:
            state = 'HOLD'
            if i in idx_2022:
                dd_entries.append(dates.iloc[i].strftime('%Y-%m-%d'))

    print(f"\nDD Control in 2022:")
    print(f"  Exits to CASH: {dd_exits if dd_exits else 'None'}")
    print(f"  Re-entries:    {dd_entries if dd_entries else 'None'}")
    dd_in_cash_2022 = (dd_signal[mask_2022] == 0).sum()
    dd_total_2022 = mask_2022.sum()
    print(f"  Days in CASH:  {dd_in_cash_2022}/{dd_total_2022} ({dd_in_cash_2022/dd_total_2022*100:.1f}%)")

    # MomDecel comparison: 40/120 vs 60/180
    md_40_120 = calc_momentum_decel_mult(close, 40, 120, 0.3, 0.5, 1.3)
    md_60_180 = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)

    print(f"\nMomDecel in 2022:")
    print(f"  MD(40/120) mean: {md_40_120[mask_2022].mean():.3f}, min: {md_40_120[mask_2022].min():.3f}")
    print(f"  MD(60/180) mean: {md_60_180[mask_2022].mean():.3f}, min: {md_60_180[mask_2022].min():.3f}")
    print(f"  → MD(60/180) {'higher' if md_60_180[mask_2022].mean() > md_40_120[mask_2022].mean() else 'lower'} average leverage in 2022")

    # AsymEWMA vs AsymHAR
    asym_ewma = calc_asym_ewma_vol(returns, 20, 5)
    asym_har = calc_asym_har_vol(returns)

    print(f"\nVol Estimation in 2022:")
    print(f"  AsymEWMA mean: {asym_ewma[mask_2022].mean():.4f}")
    print(f"  AsymHAR  mean: {asym_har[mask_2022].mean():.4f}")
    print(f"  → AsymHAR {'higher' if asym_har[mask_2022].mean() > asym_ewma[mask_2022].mean() else 'lower'} vol estimate → {'lower' if asym_har[mask_2022].mean() > asym_ewma[mask_2022].mean() else 'higher'} VT leverage")

    # Resulting leverage comparison
    trend_tv = calc_trend_target_vol(close)
    slope_mult = calc_slope_multiplier(close)

    # Baseline leverage
    vt_base = (trend_tv / asym_ewma).clip(0, 1.0)
    lev_baseline = dd_signal * vt_base * slope_mult * md_40_120
    lev_baseline = lev_baseline.clip(0, 1.0).fillna(0)

    # V2 leverage
    vt_v2 = (trend_tv / asym_har).clip(0, 1.0)
    lev_v2 = dd_signal * vt_v2 * slope_mult * md_60_180
    lev_v2 = lev_v2.clip(0, 1.0).fillna(0)

    print(f"\nResulting Leverage in 2022:")
    print(f"  Baseline mean leverage:  {lev_baseline[mask_2022].mean():.4f}")
    print(f"  V2       mean leverage:  {lev_v2[mask_2022].mean():.4f}")
    print(f"  → V2 is {'more' if lev_v2[mask_2022].mean() > lev_baseline[mask_2022].mean() else 'less'} invested in 2022")

    # Monthly breakdown
    print(f"\n  Monthly leverage comparison (2022):")
    monthly_base = pd.Series(lev_baseline.values, index=dates.values)
    monthly_v2 = pd.Series(lev_v2.values, index=dates.values)
    for month in range(1, 13):
        m = (dates.dt.year == 2022) & (dates.dt.month == month)
        if m.any():
            b = lev_baseline[m].mean()
            v = lev_v2[m].mean()
            diff = v - b
            bar = "+" * int(abs(diff) * 50) if diff > 0 else "-" * int(abs(diff) * 50)
            print(f"    2022-{month:02d}: Base={b:.3f}, V2={v:.3f}, Diff={diff:+.3f} {bar}")

    # NASDAQ price in 2022
    close_2022 = close[mask_2022]
    print(f"\n  NASDAQ in 2022: {close_2022.iloc[0]:.0f} → {close_2022.iloc[-1]:.0f} "
          f"({(close_2022.iloc[-1]/close_2022.iloc[0]-1)*100:.1f}%)")

    return {
        'dd_exits': dd_exits, 'dd_entries': dd_entries,
        'md40_mean': md_40_120[mask_2022].mean(),
        'md60_mean': md_60_180[mask_2022].mean(),
    }


# =============================================================================
# Part A: VIX Integration
# Since we don't have VIX data before 1990, we create a VIX proxy
# using realized vol and use actual VIX where available
# =============================================================================

def fetch_vix_data():
    """Fetch VIX data from Yahoo Finance (available from 1990)."""
    try:
        import yfinance as yf
        vix = yf.Ticker('^VIX')
        vix_data = vix.history(start='1990-01-01', end='2026-03-28')
        vix_data = vix_data.reset_index()
        vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.tz_localize(None)
        return vix_data[['Date', 'Close']].rename(columns={'Close': 'VIX'})
    except Exception as e:
        print(f"  VIX fetch failed: {e}")
        return None


def calc_vix_proxy(returns, span_short=10, span_long=63):
    """Create a VIX-like implied vol proxy from realized vol.
    VIX ≈ forward-looking vol, which tends to be higher than realized vol in stress.
    Proxy: short-term vol with upward bias during acceleration."""
    real_vol = calc_ewma_vol(returns, span_short)
    vol_trend = real_vol / calc_ewma_vol(returns, span_long)
    # When vol is rising (trend > 1), proxy is higher than realized
    proxy = real_vol * (0.8 + 0.4 * vol_trend.clip(0.5, 2.0))
    return (proxy * 100).fillna(20)  # Scale to VIX-like (percentage points)


def calc_vol_term_structure(returns, short_span=5, long_span=63):
    """Vol term structure: short-term vs long-term vol ratio.
    Inverted (short > long) = stress. Contango (short < long) = calm.
    This mimics VIX term structure (VIX vs VIX3M)."""
    vol_short = calc_ewma_vol(returns, short_span)
    vol_long = calc_ewma_vol(returns, long_span)
    return (vol_short / vol_long).fillna(1.0)


# =============================================================================
# Strategy A1: Vol Term Structure Filter
# =============================================================================
def strategy_a1_vol_term_structure(close, returns):
    """A1: Vol Term Structure as regime filter.

    When vol term structure is inverted (short > long), market is stressed.
    Reduce leverage proportionally.
    Normal state: short/long ≈ 0.8-1.0 (contango)
    Stress state: short/long > 1.2 (backwardation)
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # Vol term structure filter
    vts = calc_vol_term_structure(returns, 5, 63)
    # Map: vts=0.8 -> 1.0 (full), vts=1.0 -> 0.9, vts=1.5 -> 0.5
    vts_filter = (1.5 - vts).clip(0.4, 1.0)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vts_filter
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Strategy A2: VIX Proxy Mean-Reversion Signal
# =============================================================================
def strategy_a2_vix_mean_reversion(close, returns):
    """A2: VIX proxy z-score as leverage adjuster.

    When VIX proxy is abnormally high (z > 1.5), reduce leverage.
    When VIX proxy is abnormally low (z < -1), boost leverage slightly.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # VIX proxy z-score
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std

    # Map z-score to multiplier
    # z = -2 -> 1.15 (low fear, boost), z = 0 -> 1.0, z = 2 -> 0.6 (high fear, reduce)
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Strategy A3: Realized-Implied Vol Spread (VIX proxy vs realized)
# =============================================================================
def strategy_a3_vol_spread(close, returns):
    """A3: Spread between implied vol proxy and realized vol.

    When implied > realized (fear premium high), market anticipates trouble.
    When implied < realized (rare), market is complacent.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # Vol spread: short-term fast vol vs slow vol
    vol_fast = calc_ewma_vol(returns, 3)  # 3-day (very reactive)
    vol_slow = calc_ewma_vol(returns, 30)  # 30-day (stable)
    vol_spread = vol_fast - vol_slow

    # Normalize
    spread_ma = vol_spread.rolling(252).mean()
    spread_std = vol_spread.rolling(252).std().replace(0, 0.001)
    spread_z = (vol_spread - spread_ma) / spread_std

    # When fast vol >> slow vol (z > 1.5): reduce
    # When fast vol << slow vol (z < -1): conditions improving
    spread_mult = (1.0 - 0.15 * spread_z).clip(0.5, 1.1)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * spread_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Strategy A4: Combined — Vol Term Structure + MomDecel(60/180) + AsymHAR
# =============================================================================
def strategy_a4_vts_md60_asym_har(close, returns):
    """A4: Vol term structure + MD(60/180) + AsymHAR.

    Add vol term structure to V2 (best all-4 winner).
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    vts = calc_vol_term_structure(returns, 5, 63)
    vts_filter = (1.5 - vts).clip(0.5, 1.0)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vts_filter
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Strategy A5: VIX z-score + VolCPPI + MD(60/180) + AsymHAR (kitchen sink)
# =============================================================================
def strategy_a5_full_vix_cppi(close, returns):
    """A5: Best of R4-6 + VIX mean-reversion overlay."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    base_cppi = ((ratio - 0.82) / (0.90 - 0.82)).clip(0.3, 1.0)
    ewma_vol_21 = calc_ewma_vol(returns, 21)
    vol_ma_126 = ewma_vol_21.rolling(126).mean()
    vol_rel = (ewma_vol_21 / vol_ma_126.replace(0, 0.001)).clip(0.5, 2.0).fillna(1.0)
    vol_adj = (1.3 - 0.3 * vol_rel).clip(0.8, 1.2)
    cppi_mult = (base_cppi * vol_adj).clip(0.3, 1.0)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # VIX z-score overlay (gentle)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.15 * vix_z).clip(0.6, 1.1)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Test Runner with OOS
# =============================================================================
def run_full_and_oos(close, dates, strategy_func, name, threshold=DEFAULT_THRESHOLD):
    returns = close.pct_change()
    raw_leverage, dd_signal = strategy_func(close, returns)
    lev = rebalance_threshold(raw_leverage, threshold)
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)
    m_full = calc_metrics(nav, strat_ret, dd_signal, dates)
    rebal = count_rebalances(lev)

    # OOS (2021-05-07 onwards)
    split_date = '2021-05-07'
    split_idx = dates[dates >= split_date].index[0]
    nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
    ret_oos = strat_ret.iloc[split_idx:]
    oos_days = len(nav_oos)
    oos_years = oos_days / 252
    oos_cagr = (nav_oos.iloc[-1] ** (1 / oos_years)) - 1
    oos_sharpe = (ret_oos.mean() * 252) / (ret_oos.std() * np.sqrt(252)) if ret_oos.std() > 0 else 0
    oos_maxdd = (nav_oos / nav_oos.cummax() - 1).min()

    # 2022 yearly return
    mask_2022 = (dates.dt.year == 2022)
    if mask_2022.any():
        nav_2022 = nav[mask_2022]
        ret_2022 = (nav_2022.iloc[-1] / nav_2022.iloc[0] - 1) * 100
    else:
        ret_2022 = np.nan

    return {
        'Strategy': name,
        'Sharpe': m_full['Sharpe'], 'CAGR': m_full['CAGR'],
        'MaxDD': m_full['MaxDD'], 'Worst5Y': m_full['Worst5Y'],
        'OOS_Sharpe': oos_sharpe, 'OOS_CAGR': oos_cagr, 'OOS_MaxDD': oos_maxdd,
        'Y2022': ret_2022, 'Trades': m_full['Trades'], 'Rebalances': rebal,
    }


def main():
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)\n")

    # Part D: Diagnose 2022
    diag = diagnose_2022(close, dates)

    # Part A: Test VIX-based improvements
    print(f"\n{'=' * 120}")
    print("PART A: VIX/VOL STRUCTURE IMPROVEMENTS (Full + OOS + 2022)")
    print("=" * 120)

    strategies = [
        (strategy_baseline_momdecel_ens2_st, "Baseline: MD(40/120)"),
        # Best from previous rounds for reference
        (lambda c, r: strategy_r4_md60_only_wrapper(c, r), "Ref: MD(60/180)"),
        (lambda c, r: strategy_v2_md60_asym_har_wrapper(c, r), "Ref: V2 MD(60)+AsymHAR"),
        # New VIX-based strategies
        (strategy_a1_vol_term_structure, "A1: VolTermStructure+MD(60)"),
        (strategy_a2_vix_mean_reversion, "A2: VIX MeanRevert+MD(60)"),
        (strategy_a3_vol_spread, "A3: VolSpread+MD(60)"),
        (strategy_a4_vts_md60_asym_har, "A4: VTS+MD(60)+AsymHAR"),
        (strategy_a5_full_vix_cppi, "A5: VIX+VolCPPI+MD(60)+AsymHAR"),
    ]

    results = []
    for i, (func, name) in enumerate(strategies, 1):
        print(f"[{i}/{len(strategies)}] {name}...")
        r = run_full_and_oos(close, dates, func, name)
        results.append(r)

    # Print results
    baseline = results[0]
    print(f"\n{'Strategy':<33} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} | {'OOS_Sh':>7} {'OOS_CAGR':>9} {'Y2022':>7}")
    print("-" * 110)
    for r in results:
        flag = ""
        if r != baseline:
            all4 = (r['Sharpe'] > baseline['Sharpe'] and r['CAGR'] > baseline['CAGR']
                    and r['MaxDD'] > baseline['MaxDD'] and r['Worst5Y'] > baseline['Worst5Y'])
            oos_better = r['OOS_Sharpe'] > baseline['OOS_Sharpe']
            if all4 and oos_better:
                flag = " ★★★"
            elif all4:
                flag = " ★★"
            elif oos_better:
                flag = " ★(OOS)"
        print(f"{r['Strategy']:<33} {r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>6.2f}% | "
              f"{r['OOS_Sharpe']:>7.4f} {r['OOS_CAGR']*100:>8.2f}% {r['Y2022']:>6.1f}%{flag}")

    # Save
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.DataFrame(results).to_csv(os.path.join(base_dir, 'improvement_results_a_vix.csv'), index=False)
    print(f"\nSaved to improvement_results_a_vix.csv")


# Wrappers for reference strategies
def strategy_r4_md60_only_wrapper(close, returns):
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal

def strategy_v2_md60_asym_har_wrapper(close, returns):
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


if __name__ == '__main__':
    main()
