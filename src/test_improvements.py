"""
Strategy Improvement Testing Framework
======================================
Tests improvement ideas one by one against the baseline:
MomDecel(40/120) + Ens2(S+T) under realistic conditions (5-day delay, 1.5% cost)

Usage:
    python3 src/test_improvements.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol,
    calc_vt_leverage
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult
)

# =============================================================================
# Constants
# =============================================================================
ANNUAL_COST = 0.015
BASE_LEVERAGE = 3.0
DEFAULT_DELAY = 5
DEFAULT_THRESHOLD = 0.20
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

# =============================================================================
# Baseline: MomDecel(40/120) + Ens2(S+T)
# =============================================================================
def strategy_baseline_momdecel_ens2_st(close, returns):
    """Baseline: MomDecel(40/120) + Ens2(Slope+TrendTV)"""
    # DD Signal
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # Ens2(S+T): AsymEWMA + TrendTV + SlopeMult
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)

    # MomDecel
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Improvement 1: Credit Spread / Yield Curve Leading Indicator (proxy: VIX-like)
# Uses vol acceleration as a leading indicator for DD
# =============================================================================
def strategy_imp1_vol_acceleration_dd(close, returns):
    """Improvement 1: Vol Acceleration as DD Early Warning

    Instead of waiting for price to drop 18%, detect ACCELERATION in volatility
    as an early warning. If vol is accelerating rapidly, tighten DD exit threshold.
    """
    dd_exit = 0.82
    dd_reentry = 0.92

    # Detect vol acceleration (2nd derivative of vol)
    ewma_vol = calc_ewma_vol(returns, 10)
    vol_accel = ewma_vol.diff().diff()  # 2nd derivative
    vol_accel_z = (vol_accel - vol_accel.rolling(120).mean()) / vol_accel.rolling(120).std().replace(0, 0.001)

    # Adaptive DD: tighten exit when vol is accelerating
    adaptive_exit = pd.Series(dd_exit, index=close.index)
    # When vol accelerating rapidly (z > 2), exit earlier at -15% instead of -18%
    adaptive_exit[vol_accel_z > 2.0] = 0.85
    # When vol accelerating moderately (z > 1), exit at -16.5%
    adaptive_exit[(vol_accel_z > 1.0) & (vol_accel_z <= 2.0)] = 0.835

    # DD with adaptive exit
    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak
    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    for i in range(len(close)):
        if state == 'HOLD' and ratio.iloc[i] <= adaptive_exit.iloc[i]:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= dd_reentry:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    # Rest of Ens2(S+T) + MomDecel
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = position * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, position


# =============================================================================
# Improvement 2: HAR-style Multi-timescale Volatility
# =============================================================================
def calc_har_vol(returns, day_span=5, week_span=22, month_span=66):
    """HAR-style volatility: weighted average of daily, weekly, monthly vol"""
    vol_d = calc_ewma_vol(returns, day_span)
    vol_w = calc_ewma_vol(returns, week_span)
    vol_m = calc_ewma_vol(returns, month_span)
    # HAR weights (typically roughly equal)
    har_vol = 0.4 * vol_d + 0.35 * vol_w + 0.25 * vol_m
    return har_vol

def strategy_imp2_har_vol(close, returns):
    """Improvement 2: HAR Multi-timescale Volatility Targeting

    Replace AsymEWMA with HAR-style vol that captures daily, weekly, monthly dynamics.
    Should be more stable and forward-looking.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # HAR vol instead of AsymEWMA
    har_vol = calc_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / har_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Improvement 3: Asymmetric HAR Vol (combines HAR + AsymEWMA ideas)
# =============================================================================
def calc_asym_har_vol(returns, day_dn=3, day_up=10, week_span=22, month_span=66):
    """Asymmetric HAR: fast reaction to downside at daily scale, stable at weekly/monthly"""
    # Asymmetric daily vol
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    alpha_dn = 2 / (day_dn + 1)
    alpha_up = 2 / (day_up + 1)
    for i in range(1, len(returns)):
        ret = returns.iloc[i]
        alpha = alpha_dn if ret < 0 else alpha_up
        variance.iloc[i] = (1 - alpha) * variance.iloc[i-1] + alpha * (ret ** 2)
    vol_d = np.sqrt(variance * 252)

    # Standard weekly and monthly vol
    vol_w = calc_ewma_vol(returns, week_span)
    vol_m = calc_ewma_vol(returns, month_span)

    har_vol = 0.5 * vol_d + 0.3 * vol_w + 0.2 * vol_m
    return har_vol

def strategy_imp3_asym_har_vol(close, returns):
    """Improvement 3: Asymmetric HAR Vol — fast downside reaction + multi-timescale stability"""
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    asym_har = calc_asym_har_vol(returns)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_har).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Improvement 4: Q3/Q4 Seasonal Dampening
# =============================================================================
def strategy_imp4_seasonal(close, returns, dates):
    """Improvement 4: Seasonal Q3/Q4 Dampening

    R4 showed Q3Q4weak(0.8x) improved Sharpe to 0.894 on the simpler DD+VT strategy.
    Apply the same concept to MomDecel+Ens2(S+T).
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    # Seasonal adjustment: reduce leverage in weak months
    months = dates.dt.month
    seasonal_adj = pd.Series(1.0, index=close.index)
    # Q3 weakness (Jul-Sep) + October effect
    for m in [7, 8, 9, 10]:
        seasonal_adj[months == m] = 0.8

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * seasonal_adj
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Improvement 5: CPPI-style Dynamic Floor
# =============================================================================
def strategy_imp5_cppi_floor(close, returns):
    """Improvement 5: CPPI-style Dynamic Floor Protection

    Constant Proportion Portfolio Insurance: as drawdown deepens,
    leverage decreases proportionally. Provides smoother protection
    than binary DD exit.

    Leverage = multiplier * (NAV - Floor) / NAV
    Floor grows at risk-free rate, multiplied by a base.
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # CPPI overlay: gradual reduction starting from -10% drawdown
    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak

    # CPPI multiplier: 1.0 at ratio=1.0, linearly decreasing
    # At ratio=0.90 (10% DD): mult=0.80
    # At ratio=0.82 (18% DD): mult=0.0 (DD kicks in anyway)
    cppi_mult = ((ratio - 0.82) / (1.0 - 0.82)).clip(0, 1.0)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * cppi_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Improvement 6: Hurst Exponent Trend Quality
# =============================================================================
def calc_hurst_approx(close, window=100):
    """Approximate Hurst exponent using rescaled range (R/S) method.
    H > 0.5: trending, H < 0.5: mean-reverting, H ≈ 0.5: random walk
    """
    log_returns = np.log(close / close.shift(1))
    hurst = pd.Series(0.5, index=close.index)

    for i in range(window, len(close)):
        ts = log_returns.iloc[i-window:i].dropna()
        if len(ts) < window * 0.8:
            continue
        mean_r = ts.mean()
        deviate = (ts - mean_r).cumsum()
        R = deviate.max() - deviate.min()
        S = ts.std()
        if S > 0 and R > 0:
            hurst.iloc[i] = np.log(R / S) / np.log(window)

    return hurst.clip(0.1, 0.9)

def strategy_imp6_hurst_trend(close, returns):
    """Improvement 6: Hurst Exponent as Trend Quality Indicator

    Replace SlopeMult with Hurst-based trend quality measure.
    H > 0.5 (trending): boost leverage
    H < 0.5 (mean-reverting): reduce leverage
    """
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)

    # Hurst-based multiplier instead of SlopeMult
    hurst = calc_hurst_approx(close, 100)
    # Map H to multiplier: H=0.3 -> 0.5x, H=0.5 -> 1.0x, H=0.7 -> 1.5x
    hurst_mult = (0.5 + 2.5 * (hurst - 0.3)).clip(0.3, 1.5)

    mom_decel = calc_momentum_decel_mult(close, short=40, long=120,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)

    raw_leverage = dd_signal * vt_lev * hurst_mult * mom_decel
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)
    return raw_leverage, dd_signal


# =============================================================================
# Test Runner
# =============================================================================
def run_test(close, dates, strategy_func, name, needs_dates=False):
    """Run a single strategy test and return metrics."""
    returns = close.pct_change()

    if needs_dates:
        raw_leverage, dd_signal = strategy_func(close, returns, dates)
    else:
        raw_leverage, dd_signal = strategy_func(close, returns)

    # Apply rebalance threshold
    lev = rebalance_threshold(raw_leverage, DEFAULT_THRESHOLD)

    # Apply execution delay
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)

    # Calculate metrics
    metrics = calc_metrics(nav, strat_ret, dd_signal, dates)
    rebal = count_rebalances(lev)
    metrics['Strategy'] = name
    metrics['Rebalances'] = rebal

    return metrics


def print_results(results):
    """Print results as a formatted table."""
    df = pd.DataFrame(results)
    cols = ['Strategy', 'CAGR', 'Sharpe', 'MaxDD', 'Worst5Y', 'Trades', 'Rebalances']
    df = df[cols].copy()

    # Format
    df['CAGR'] = df['CAGR'].apply(lambda x: f"{x*100:.2f}%")
    df['Sharpe'] = df['Sharpe'].apply(lambda x: f"{x:.4f}")
    df['MaxDD'] = df['MaxDD'].apply(lambda x: f"{x*100:.2f}%")
    df['Worst5Y'] = df['Worst5Y'].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A")
    df['Trades'] = df['Trades'].astype(int)
    df['Rebalances'] = df['Rebalances'].astype(int)

    print("\n" + "="*100)
    print("STRATEGY IMPROVEMENT COMPARISON (1974-2026, 5-day delay, 1.5% cost)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)


def main():
    # Load extended data
    print(f"Loading data from {DATA_PATH}")
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"Data range: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)")

    results = []

    # ---- Baseline ----
    print("\n[1/7] Testing Baseline: MomDecel+Ens2(S+T)...")
    r = run_test(close, dates, strategy_baseline_momdecel_ens2_st, "Baseline: MomDecel+Ens2(S+T)")
    results.append(r)
    print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, MaxDD: {r['MaxDD']*100:.2f}%")

    # ---- Improvement 1: Vol Acceleration DD ----
    print("\n[2/7] Testing Imp1: Vol Acceleration DD Early Warning...")
    r = run_test(close, dates, strategy_imp1_vol_acceleration_dd, "Imp1: VolAccel DD Warning")
    results.append(r)
    print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, MaxDD: {r['MaxDD']*100:.2f}%")

    # ---- Improvement 2: HAR Vol ----
    print("\n[3/7] Testing Imp2: HAR Multi-timescale Vol...")
    r = run_test(close, dates, strategy_imp2_har_vol, "Imp2: HAR Vol")
    results.append(r)
    print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, MaxDD: {r['MaxDD']*100:.2f}%")

    # ---- Improvement 3: Asymmetric HAR Vol ----
    print("\n[4/7] Testing Imp3: Asymmetric HAR Vol...")
    r = run_test(close, dates, strategy_imp3_asym_har_vol, "Imp3: AsymHAR Vol")
    results.append(r)
    print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, MaxDD: {r['MaxDD']*100:.2f}%")

    # ---- Improvement 4: Seasonal Q3/Q4 ----
    print("\n[5/7] Testing Imp4: Seasonal Q3/Q4 Dampening...")
    r = run_test(close, dates, strategy_imp4_seasonal, "Imp4: Seasonal Q3Q4", needs_dates=True)
    results.append(r)
    print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, MaxDD: {r['MaxDD']*100:.2f}%")

    # ---- Improvement 5: CPPI Floor ----
    print("\n[6/7] Testing Imp5: CPPI Dynamic Floor...")
    r = run_test(close, dates, strategy_imp5_cppi_floor, "Imp5: CPPI Floor")
    results.append(r)
    print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, MaxDD: {r['MaxDD']*100:.2f}%")

    # ---- Improvement 6: Hurst Trend Quality ----
    print("\n[7/7] Testing Imp6: Hurst Exponent Trend Quality...")
    r = run_test(close, dates, strategy_imp6_hurst_trend, "Imp6: Hurst Trend")
    results.append(r)
    print(f"  Sharpe: {r['Sharpe']:.4f}, CAGR: {r['CAGR']*100:.2f}%, MaxDD: {r['MaxDD']*100:.2f}%")

    # Print final comparison
    print_results(results)

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'improvement_results.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
