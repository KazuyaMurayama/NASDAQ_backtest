"""
Direction B: Delay-Robust Strategy Design
==========================================

Goal: Design strategies that minimize Sharpe degradation from 5-day execution delay.

Current baseline: Ens2(Slope+TrendTV) Th=20%
  - Sharpe at delay=0: ~1.01
  - Sharpe at delay=5: ~0.86
  - Degradation: -0.15 (-15%)

Core problem: With 5-day delay, any signal computed today won't affect position for
5 business days. By then:
  - In a crash: price may have fallen another 5-10%, DD exit comes too late
  - In recovery: price may have risen, missing early gains
  - Vol may have changed significantly

Design principles:
  1. Soft DD: Gradual leverage reduction (start early, avoid binary cliff)
  2. Anticipatory DD: Predict future drawdown using momentum extrapolation
  3. Predictive Vol: Forward-looking volatility estimate
  4. Signal prediction: Extrapolate signal trend to predict T+5 state
  5. Momentum deceleration: Early warning from trend curvature

Tests:
  Part 1: Delay sensitivity profiling of existing strategies
  Part 2: Soft DD (continuous drawdown-based leverage)
  Part 3: Anticipatory DD (predict future drawdown, delay-adjusted thresholds)
  Part 4: Predictive vol targeting
  Part 5: Momentum deceleration multiplier
  Part 6: Combined delay-robust composite
  Part 7: Full delay robustness comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol,
    calc_vt_leverage, strategy_baseline_bh3x, strategy_baseline_dd_only,
    strategy_baseline_dd_vt
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol,
    strategy_ens2_asym_slope, strategy_ens2_slope_trendtv
)

# =============================================================================
# Constants
# =============================================================================
ANNUAL_COST = 0.015
BASE_LEVERAGE = 3.0
DEFAULT_DELAY = 5
DEFAULT_THRESHOLD = 0.20

# =============================================================================
# Shared utilities
# =============================================================================
def rebalance_threshold(leverage, threshold):
    """Only rebalance when target changes by more than threshold.
    Transitions to/from 0 always trigger immediately."""
    result = pd.Series(0.0, index=leverage.index)
    current = leverage.iloc[0]
    result.iloc[0] = current
    for i in range(1, len(leverage)):
        target = leverage.iloc[i]
        if target == 0.0 and current > 0.0:
            current = 0.0
        elif current == 0.0 and target > 0.0:
            current = target
        elif abs(target - current) > threshold:
            current = target
        result.iloc[i] = current
    return result


def rebalance_threshold_soft(leverage, threshold):
    """Rebalance threshold for soft (continuous) signals.
    No special-case for 0 transition since signal is continuous.
    Force to 0 if target < 0.05 (effective cash)."""
    result = pd.Series(0.0, index=leverage.index)
    current = leverage.iloc[0]
    result.iloc[0] = current
    for i in range(1, len(leverage)):
        target = leverage.iloc[i]
        # Force cash if essentially zero
        if target < 0.05 and current >= 0.05:
            current = 0.0
        elif current < 0.05 and target >= 0.05:
            current = target
        elif abs(target - current) > threshold:
            current = target
        result.iloc[i] = current
    return result


def run_backtest_realistic(close, leverage, exec_delay=DEFAULT_DELAY,
                           annual_cost=ANNUAL_COST):
    """Run backtest with realistic constraints."""
    returns = close.pct_change()
    leveraged_returns = returns * BASE_LEVERAGE
    daily_cost = annual_cost / 252

    delayed_leverage = leverage.shift(exec_delay)
    strategy_returns = delayed_leverage * (leveraged_returns - daily_cost)
    strategy_returns = strategy_returns.fillna(0)

    nav = (1 + strategy_returns).cumprod()
    return nav, strategy_returns


def count_rebalances(leverage):
    changes = leverage.diff().abs()
    return (changes > 0.01).sum()


def test_strategy(close, dates, raw_leverage, dd_signal, name, category,
                  delay=DEFAULT_DELAY, threshold=DEFAULT_THRESHOLD,
                  use_soft_rebal=False):
    """Test a strategy and return metrics dict."""
    if threshold > 0:
        if use_soft_rebal:
            lev = rebalance_threshold_soft(raw_leverage, threshold)
        else:
            lev = rebalance_threshold(raw_leverage, threshold)
    else:
        lev = raw_leverage

    nav, strat_ret = run_backtest_realistic(close, lev, delay)
    metrics = calc_metrics(nav, strat_ret, dd_signal, dates)
    rebal = count_rebalances(lev)
    metrics.update({
        'Strategy': name,
        'Category': category,
        'Delay': delay,
        'Rebalances': rebal,
        'Rebal/Year': rebal / (len(close) / 252),
    })
    return metrics


# =============================================================================
# NEW COMPONENTS: Delay-Robust Building Blocks
# =============================================================================

def calc_soft_dd(close, lookback=200, lower=0.75, upper=0.95):
    """Soft DD: Continuous drawdown-based leverage multiplier.

    Instead of binary 0/1, returns a continuous value 0.0 to 1.0:
      ratio >= upper  -> 1.0 (full leverage)
      ratio <= lower  -> 0.0 (full cash)
      in between      -> linear interpolation

    Key advantage for delay: leverage reduction starts EARLY (at upper threshold),
    so even with 5-day delay, partial reduction is already in effect.
    """
    peak = close.rolling(lookback, min_periods=1).max()
    ratio = close / peak
    dd_mult = (ratio - lower) / (upper - lower)
    dd_mult = dd_mult.clip(0.0, 1.0)
    return dd_mult


def calc_soft_dd_with_floor(close, lookback=200, lower=0.80, upper=0.95, floor_ratio=0.78):
    """Soft DD with hard floor: gradual reduction + forced exit at severe DD.

    Combines benefits of:
      - Soft DD: early partial reduction (delay-friendly)
      - Hard floor: complete protection at extreme drawdowns
    """
    peak = close.rolling(lookback, min_periods=1).max()
    ratio = close / peak
    dd_mult = (ratio - lower) / (upper - lower)
    dd_mult = dd_mult.clip(0.0, 1.0)

    # Hard floor: force to 0 at extreme drawdowns (state machine for hysteresis)
    result = dd_mult.copy()
    in_floor = False
    floor_reentry = floor_ratio + 0.12  # reentry = floor + 12%

    for i in range(len(result)):
        if not in_floor and ratio.iloc[i] <= floor_ratio:
            in_floor = True
        elif in_floor and ratio.iloc[i] >= floor_reentry:
            in_floor = False

        if in_floor:
            result.iloc[i] = 0.0

    return result


def calc_anticipatory_dd(close, lookback=200, exit_th=0.82, reentry_th=0.92,
                         forward=5):
    """Anticipatory DD: Use momentum extrapolation to predict future drawdown.

    Instead of using current ratio for DD decision, predict what ratio will be
    in `forward` days based on recent price trend. This compensates for delay
    by acting on predicted future state rather than current state.
    """
    peak = close.rolling(lookback, min_periods=1).max()
    ratio = close / peak

    # Predict future ratio: extrapolate recent trend
    # Use log returns for better extrapolation
    daily_log_ret = np.log(close / close.shift(1))
    avg_daily_ret = daily_log_ret.rolling(forward).mean()

    # Predicted ratio in `forward` days
    predicted_change = np.exp(avg_daily_ret * forward)
    ratio_pred = ratio * predicted_change
    ratio_pred = ratio_pred.clip(0, 1.5).fillna(ratio)

    # Standard DD state machine, but using PREDICTED ratio
    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'
    for i in range(len(close)):
        r = ratio_pred.iloc[i]
        if state == 'HOLD' and r <= exit_th:
            state = 'CASH'
        elif state == 'CASH' and r >= reentry_th:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    return position


def calc_delay_adjusted_dd(close, lookback=200, base_exit=0.82, base_reentry=0.92,
                           exit_shift=0.05, reentry_shift=0.03):
    """Delay-adjusted DD: Shift thresholds to compensate for execution delay.

    Tighter exit (higher ratio) = exit earlier, before further decline during delay.
    Stricter reentry (higher ratio) = wait for stronger confirmation.
    """
    adjusted_exit = base_exit + exit_shift      # e.g., 0.87 instead of 0.82
    adjusted_reentry = base_reentry + reentry_shift  # e.g., 0.95 instead of 0.92
    return calc_dd_signal(close, adjusted_exit, adjusted_reentry, lookback)


def calc_predictive_vol(returns, span=20, forward=5):
    """Predictive vol: Extrapolate volatility trend to estimate future vol.

    If vol is rising (e.g., entering crisis), predict higher vol in `forward` days.
    This causes leverage to be reduced preemptively.
    """
    ewma_vol = calc_ewma_vol(returns, span)

    # Vol trend: ratio of current vol to vol N days ago
    vol_ratio = ewma_vol / ewma_vol.shift(forward)
    vol_ratio = vol_ratio.clip(0.5, 2.0).fillna(1.0)

    # Predicted future vol = current vol × trend ratio
    predicted_vol = ewma_vol * vol_ratio
    predicted_vol = predicted_vol.clip(lower=0.05)

    return predicted_vol


def calc_predictive_asym_vol(returns, span_up=20, span_dn=5, forward=5):
    """Predictive asymmetric vol: Same extrapolation with AsymEWMA base."""
    asym_vol = calc_asym_ewma_vol(returns, span_up, span_dn)

    vol_ratio = asym_vol / asym_vol.shift(forward)
    vol_ratio = vol_ratio.clip(0.5, 2.0).fillna(1.0)

    predicted_vol = asym_vol * vol_ratio
    predicted_vol = predicted_vol.clip(lower=0.05)

    return predicted_vol


def calc_persistent_vol(returns, span=20, filter_window=5):
    """Persistent vol: Median-filtered vol to ignore transient spikes.

    Transient spikes are already stale by execution time, so using median
    over a window reduces noise without losing real regime changes.
    """
    ewma_vol = calc_ewma_vol(returns, span)
    persistent = ewma_vol.rolling(filter_window).median()
    return persistent.fillna(ewma_vol)


def calc_momentum_decel_mult(close, short=20, long=60, sensitivity=0.3,
                             min_mult=0.5, max_mult=1.3):
    """Momentum deceleration multiplier: detect trend weakening early.

    When short-term momentum < long-term momentum, trend is decelerating.
    This provides early warning before DD triggers.

    Returns a multiplier (0.5 to 1.3):
      - Trend accelerating: mult > 1.0 (boost)
      - Trend decelerating: mult < 1.0 (reduce)
    """
    mom_short = close.pct_change(short)
    mom_long = close.pct_change(long)

    # Normalize long momentum to same period
    mom_long_norm = mom_long * (short / long)

    # Deceleration signal (positive = accelerating, negative = decelerating)
    decel = mom_short - mom_long_norm

    # Z-score normalize
    decel_mean = decel.rolling(120).mean()
    decel_std = decel.rolling(120).std().replace(0, 0.001)
    decel_z = (decel - decel_mean) / decel_std

    mult = (1.0 + sensitivity * decel_z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


def predict_leverage_forward(leverage, forward=5, smooth_span=10):
    """Generic signal predictor: extrapolate leverage trend forward.

    Can be applied to ANY strategy's raw leverage signal.
    Uses EMA smoothing + linear extrapolation.
    """
    smoothed = leverage.ewm(span=smooth_span).mean()
    trend = smoothed - smoothed.shift(forward)
    predicted = smoothed + trend
    return predicted.clip(0.0, 1.5).fillna(leverage)


# =============================================================================
# STRATEGY BUILDERS: Combine components into full strategies
# =============================================================================

def strategy_soft_dd_ens2_st(close, returns, lower=0.75, upper=0.95):
    """Soft DD + Ens2(Slope+TrendTV) VT layer."""
    soft_dd = calc_soft_dd(close, 200, lower, upper)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    leverage = soft_dd * vt_lev * slope_mult
    return leverage.clip(0, 1.0).fillna(0)


def strategy_soft_dd_floor_ens2_st(close, returns, lower=0.80, upper=0.95,
                                    floor_ratio=0.78):
    """Soft DD with hard floor + Ens2(Slope+TrendTV) VT layer."""
    soft_dd = calc_soft_dd_with_floor(close, 200, lower, upper, floor_ratio)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    leverage = soft_dd * vt_lev * slope_mult
    return leverage.clip(0, 1.0).fillna(0)


def strategy_anticipatory_dd_ens2_st(close, returns, forward=5):
    """Anticipatory DD + Ens2(Slope+TrendTV) VT layer."""
    dd_signal = calc_anticipatory_dd(close, 200, 0.82, 0.92, forward)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    leverage = dd_signal * vt_lev * slope_mult
    return leverage.clip(0, 1.0).fillna(0), dd_signal


def strategy_delay_adjusted_dd_ens2_st(close, returns, exit_shift=0.05,
                                        reentry_shift=0.03):
    """Delay-adjusted DD thresholds + Ens2(Slope+TrendTV) VT layer."""
    dd_signal = calc_delay_adjusted_dd(close, 200, 0.82, 0.92,
                                       exit_shift, reentry_shift)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    leverage = dd_signal * vt_lev * slope_mult
    return leverage.clip(0, 1.0).fillna(0), dd_signal


def strategy_predictive_vol_ens2(close, returns, forward=5):
    """Standard DD + Predictive AsymVol + TrendTV + SlopeMult."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92, 200)
    trend_tv = calc_trend_target_vol(close)
    pred_vol = calc_predictive_asym_vol(returns, 20, 5, forward)
    vt_lev = (trend_tv / pred_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    leverage = dd_signal * vt_lev * slope_mult
    return leverage.clip(0, 1.0).fillna(0), dd_signal


def strategy_persistent_vol_ens2(close, returns, filter_window=5):
    """Standard DD + Persistent (median-filtered) Vol + TrendTV + SlopeMult."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92, 200)
    trend_tv = calc_trend_target_vol(close)
    pers_vol = calc_persistent_vol(returns, 20, filter_window)
    # Annualize if needed (calc_persistent_vol already returns annualized via calc_ewma_vol)
    vt_lev = (trend_tv / pers_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    leverage = dd_signal * vt_lev * slope_mult
    return leverage.clip(0, 1.0).fillna(0), dd_signal


def strategy_momentum_decel_ens2_st(close, returns, short=20, long=60,
                                     sensitivity=0.3):
    """Standard Ens2(S+T) + Momentum Deceleration multiplier."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92, 200)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    decel_mult = calc_momentum_decel_mult(close, short, long, sensitivity)
    leverage = dd_signal * vt_lev * slope_mult * decel_mult
    return leverage.clip(0, 1.0).fillna(0), dd_signal


def strategy_predicted_signal_ens2_st(close, returns, forward=5):
    """Ens2(S+T) with signal prediction: extrapolate leverage trend forward."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92, 200)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    raw_leverage = dd_signal * vt_lev * slope_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)

    # Predict forward
    predicted = predict_leverage_forward(raw_leverage, forward)
    return predicted.clip(0, 1.0), dd_signal


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 130)
    print("DIRECTION B: DELAY-ROBUST STRATEGY DESIGN")
    print("=" * 130)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'NASDAQ_Dairy_since1973.csv')
    df = load_data(data_path)
    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']
    n_years = len(df) / 252

    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total: {len(df)} days (~{n_years:.0f} years)")
    print(f"Conditions: delay=5d, cost=1.5%, rebalance threshold=20%\n")

    all_results = []

    def run_test(raw_lev, dd_sig, name, cat, delay=DEFAULT_DELAY,
                 th=DEFAULT_THRESHOLD, soft_rebal=False):
        m = test_strategy(close, dates, raw_lev, dd_sig, name, cat,
                          delay, th, soft_rebal)
        all_results.append(m)
        return m

    def print_result(m):
        print(f"  {m['Strategy']:<65s} Sharpe={m['Sharpe']:.3f}  CAGR={m['CAGR']*100:.1f}%  "
              f"MaxDD={m['MaxDD']*100:.1f}%  W5Y={m['Worst5Y']*100:.1f}%  "
              f"Rebal={m['Rebalances']} ({m['Rebal/Year']:.1f}/yr)")

    # Pre-compute baseline strategies
    lev_st, dd_st = strategy_ens2_slope_trendtv(close, returns, 0.82, 0.92, 20, 5, 1.0)
    lev_as, dd_as = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    lev_dd, dd_dd = strategy_baseline_dd_only(close, returns, 0.82, 0.92)
    lev_dv, dd_dv = strategy_baseline_dd_vt(close, returns, 0.82, 0.92, 0.25, 10)

    # =====================================================================
    # PART 1: DELAY SENSITIVITY PROFILING
    # =====================================================================
    print("=" * 130)
    print("PART 1: DELAY SENSITIVITY PROFILING")
    print("  How much does each strategy degrade as delay increases?")
    print("=" * 130)

    delays_to_test = [0, 1, 3, 5, 7, 10]
    strategies_p1 = {
        'Ens2(S+T) Th20%': (lev_st, dd_st, False),
        'Ens2(A+S) Th20%': (lev_as, dd_as, False),
        'DD+VT(25%) Th20%': (lev_dv, dd_dv, False),
        'DD-Only': (lev_dd, dd_dd, False),
    }

    delay_profile = {}
    for name, (lev, dd, _) in strategies_p1.items():
        print(f"\n  [{name}]")
        delay_profile[name] = {}
        for d in delays_to_test:
            m = run_test(lev, dd, f'{name} delay={d}', 'P1-Profile', delay=d)
            delay_profile[name][d] = m['Sharpe']
            print(f"    delay={d:>2d}d: Sharpe={m['Sharpe']:.4f}  CAGR={m['CAGR']*100:.1f}%  MaxDD={m['MaxDD']*100:.1f}%")

    # Summary table
    print(f"\n  {'Strategy':<25s}", end="")
    for d in delays_to_test:
        print(f"  d={d:<3d}", end="")
    print("  d5/d0   d5-d0")
    print("  " + "-" * 100)

    for name in strategies_p1:
        print(f"  {name:<25s}", end="")
        for d in delays_to_test:
            print(f"  {delay_profile[name][d]:.3f}", end="")
        ratio = delay_profile[name][5] / delay_profile[name][0] if delay_profile[name][0] > 0 else 0
        diff = delay_profile[name][5] - delay_profile[name][0]
        print(f"  {ratio:.3f}  {diff:+.3f}")

    # =====================================================================
    # PART 2: SOFT DD (Continuous Drawdown Leverage)
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 2: SOFT DD - Continuous Drawdown-Based Leverage")
    print("  Replace binary DD (0/1) with continuous multiplier (0.0-1.0)")
    print("  Key: leverage reduction starts EARLIER, smoothing delay impact")
    print("=" * 130)

    # Baseline
    print("\n  [Baselines at delay=5]")
    m = run_test(lev_st, dd_st, 'BASELINE: Ens2(S+T) Th20%', 'Baseline')
    print_result(m)

    # Soft DD parameter sweep
    soft_dd_params = [
        (0.75, 0.95, "Wide(75-95)"),
        (0.75, 1.00, "VeryWide(75-100)"),
        (0.80, 0.95, "Medium(80-95)"),
        (0.80, 0.98, "EarlyStart(80-98)"),
        (0.80, 1.00, "FullRange(80-100)"),
        (0.82, 0.95, "Narrow(82-95)"),
        (0.85, 0.98, "Tight(85-98)"),
        (0.70, 0.95, "UltraWide(70-95)"),
    ]

    print(f"\n  [Soft DD + Ens2(S+T) VT layer]")
    for lower, upper, label in soft_dd_params:
        lev = strategy_soft_dd_ens2_st(close, returns, lower, upper)
        m = run_test(lev, None, f'SoftDD {label} + S+T', 'P2-SoftDD',
                     soft_rebal=True)
        print_result(m)

    # Soft DD with hard floor
    print(f"\n  [Soft DD with Hard Floor + Ens2(S+T) VT layer]")
    floor_params = [
        (0.80, 0.95, 0.75, "Med(80-95) Floor75"),
        (0.80, 0.95, 0.78, "Med(80-95) Floor78"),
        (0.80, 0.98, 0.75, "Early(80-98) Floor75"),
        (0.80, 0.98, 0.78, "Early(80-98) Floor78"),
        (0.80, 1.00, 0.75, "Full(80-100) Floor75"),
        (0.80, 1.00, 0.78, "Full(80-100) Floor78"),
        (0.75, 0.95, 0.72, "Wide(75-95) Floor72"),
    ]

    for lower, upper, floor_r, label in floor_params:
        lev = strategy_soft_dd_floor_ens2_st(close, returns, lower, upper, floor_r)
        m = run_test(lev, None, f'SoftDD+Floor {label} + S+T', 'P2-SoftFloor',
                     soft_rebal=True)
        print_result(m)

    # =====================================================================
    # PART 3: ANTICIPATORY DD
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 3: ANTICIPATORY DD - Predict Future Drawdown")
    print("  A: Momentum-extrapolated DD (predict ratio 5d ahead)")
    print("  B: Delay-adjusted thresholds (tighter exit, stricter reentry)")
    print("=" * 130)

    # A: Anticipatory DD with momentum extrapolation
    print(f"\n  [A: Anticipatory DD (momentum extrapolation)]")
    for fwd in [3, 5, 7, 10]:
        lev, dd = strategy_anticipatory_dd_ens2_st(close, returns, fwd)
        m = run_test(lev, dd, f'AnticDD fwd={fwd}d + S+T', 'P3-AnticDD')
        print_result(m)

    # B: Delay-adjusted thresholds
    print(f"\n  [B: Delay-adjusted DD thresholds]")
    threshold_shifts = [
        (0.03, 0.02, "Exit+3%/Re+2%"),
        (0.05, 0.03, "Exit+5%/Re+3%"),
        (0.05, 0.05, "Exit+5%/Re+5%"),
        (0.07, 0.03, "Exit+7%/Re+3%"),
        (0.07, 0.05, "Exit+7%/Re+5%"),
        (0.10, 0.05, "Exit+10%/Re+5%"),
        (0.03, 0.00, "Exit+3%/Re+0%"),
        (0.05, 0.00, "Exit+5%/Re+0%"),
    ]

    for e_shift, r_shift, label in threshold_shifts:
        lev, dd = strategy_delay_adjusted_dd_ens2_st(close, returns, e_shift, r_shift)
        m = run_test(lev, dd, f'DelayAdj {label} + S+T', 'P3-DelayAdj')
        print_result(m)

    # =====================================================================
    # PART 4: PREDICTIVE VOL
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 4: PREDICTIVE VOLATILITY TARGETING")
    print("  Extrapolate vol trend to target future vol (not current)")
    print("=" * 130)

    # Predictive AsymVol
    print(f"\n  [Predictive AsymVol + DD + TrendTV + SlopeMult]")
    for fwd in [3, 5, 7, 10]:
        lev, dd = strategy_predictive_vol_ens2(close, returns, fwd)
        m = run_test(lev, dd, f'PredVol fwd={fwd}d + Ens2', 'P4-PredVol')
        print_result(m)

    # Persistent Vol (median filter)
    print(f"\n  [Persistent Vol (median filter) + DD + TrendTV + SlopeMult]")
    for win in [3, 5, 7, 10]:
        lev, dd = strategy_persistent_vol_ens2(close, returns, win)
        m = run_test(lev, dd, f'PersistVol win={win}d + Ens2', 'P4-PersistVol')
        print_result(m)

    # =====================================================================
    # PART 5: MOMENTUM DECELERATION
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 5: MOMENTUM DECELERATION MULTIPLIER")
    print("  Detect trend weakening early (before DD triggers)")
    print("=" * 130)

    print(f"\n  [MomDecel + Ens2(S+T)]")
    decel_params = [
        (20, 60, 0.2, "20/60 s=0.2"),
        (20, 60, 0.3, "20/60 s=0.3"),
        (20, 60, 0.5, "20/60 s=0.5"),
        (10, 40, 0.3, "10/40 s=0.3"),
        (20, 120, 0.3, "20/120 s=0.3"),
        (30, 90, 0.3, "30/90 s=0.3"),
    ]

    for short, long, sens, label in decel_params:
        lev, dd = strategy_momentum_decel_ens2_st(close, returns, short, long, sens)
        m = run_test(lev, dd, f'MomDecel {label} + Ens2(S+T)', 'P5-MomDecel')
        print_result(m)

    # Signal prediction
    print(f"\n  [Signal Prediction (extrapolate leverage trend)]")
    for fwd in [3, 5, 7]:
        lev, dd = strategy_predicted_signal_ens2_st(close, returns, fwd)
        m = run_test(lev, dd, f'SigPred fwd={fwd}d + Ens2(S+T)', 'P5-SigPred')
        print_result(m)

    # =====================================================================
    # PART 6: COMBINED DELAY-ROBUST STRATEGIES
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 6: COMBINED DELAY-ROBUST STRATEGIES")
    print("  Mix best components from Parts 2-5")
    print("=" * 130)

    # Find best from each part
    p2_results = [r for r in all_results if r['Category'].startswith('P2')]
    p3_results = [r for r in all_results if r['Category'].startswith('P3')]
    p4_results = [r for r in all_results if r['Category'].startswith('P4')]
    p5_results = [r for r in all_results if r['Category'].startswith('P5')]

    for label, res_list in [("Part2-SoftDD", p2_results), ("Part3-AnticDD", p3_results),
                            ("Part4-PredVol", p4_results), ("Part5-MomDecel", p5_results)]:
        if res_list:
            best = max(res_list, key=lambda x: x['Sharpe'])
            print(f"  Best {label}: {best['Strategy']} (Sharpe={best['Sharpe']:.4f})")

    # Combination A: Soft DD + Predictive Vol + SlopeMult
    print(f"\n  [Combo A: Soft DD + Predictive Vol + TrendTV + SlopeMult]")
    for lower, upper in [(0.75, 0.95), (0.80, 0.95), (0.80, 1.00)]:
        soft_dd = calc_soft_dd(close, 200, lower, upper)
        trend_tv = calc_trend_target_vol(close)
        pred_vol = calc_predictive_asym_vol(returns, 20, 5, 5)
        vt_lev = (trend_tv / pred_vol).clip(0, 1.0)
        slope_mult = calc_slope_multiplier(close)
        lev = (soft_dd * vt_lev * slope_mult).clip(0, 1.0).fillna(0)
        m = run_test(lev, None, f'SoftDD({lower}-{upper})+PredVol+S+T', 'P6-ComboA',
                     soft_rebal=True)
        print_result(m)

    # Combination B: Soft DD with floor + Predictive Vol
    print(f"\n  [Combo B: Soft DD + Floor + Predictive Vol + SlopeMult]")
    for lower, upper, floor_r in [(0.80, 0.95, 0.78), (0.80, 1.00, 0.78),
                                   (0.80, 0.95, 0.75)]:
        soft_dd = calc_soft_dd_with_floor(close, 200, lower, upper, floor_r)
        trend_tv = calc_trend_target_vol(close)
        pred_vol = calc_predictive_asym_vol(returns, 20, 5, 5)
        vt_lev = (trend_tv / pred_vol).clip(0, 1.0)
        slope_mult = calc_slope_multiplier(close)
        lev = (soft_dd * vt_lev * slope_mult).clip(0, 1.0).fillna(0)
        m = run_test(lev, None,
                     f'SoftDD({lower}-{upper})F{floor_r}+PredVol+S+T', 'P6-ComboB',
                     soft_rebal=True)
        print_result(m)

    # Combination C: Anticipatory DD + Predictive Vol
    print(f"\n  [Combo C: Anticipatory DD + Predictive Vol + TrendTV + SlopeMult]")
    for fwd_dd, fwd_vol in [(5, 5), (5, 3), (7, 5), (3, 5)]:
        dd_sig = calc_anticipatory_dd(close, 200, 0.82, 0.92, fwd_dd)
        trend_tv = calc_trend_target_vol(close)
        pred_vol = calc_predictive_asym_vol(returns, 20, 5, fwd_vol)
        vt_lev = (trend_tv / pred_vol).clip(0, 1.0)
        slope_mult = calc_slope_multiplier(close)
        lev = (dd_sig * vt_lev * slope_mult).clip(0, 1.0).fillna(0)
        m = run_test(lev, dd_sig,
                     f'AnticDD(fwd={fwd_dd})+PredVol(fwd={fwd_vol})+S+T', 'P6-ComboC')
        print_result(m)

    # Combination D: Delay-adjusted DD + MomDecel
    print(f"\n  [Combo D: Delay-adjusted DD + MomDecel + TrendTV + SlopeMult]")
    for e_shift, r_shift in [(0.05, 0.03), (0.03, 0.02), (0.07, 0.03)]:
        dd_sig = calc_delay_adjusted_dd(close, 200, 0.82, 0.92, e_shift, r_shift)
        trend_tv = calc_trend_target_vol(close)
        asym_vol = calc_asym_ewma_vol(returns, 20, 5)
        vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
        slope_mult = calc_slope_multiplier(close)
        decel_mult = calc_momentum_decel_mult(close, 20, 60, 0.3)
        lev = (dd_sig * vt_lev * slope_mult * decel_mult).clip(0, 1.0).fillna(0)
        m = run_test(lev, dd_sig,
                     f'DelAdj(e+{e_shift}/r+{r_shift})+MomDec+S+T', 'P6-ComboD')
        print_result(m)

    # Combination E: Soft DD + MomDecel + Predictive Vol (triple delay-robust)
    print(f"\n  [Combo E: Soft DD + MomDecel + PredVol + TrendTV + SlopeMult (Triple)]")
    for lower, upper in [(0.80, 0.95), (0.80, 1.00), (0.75, 0.95)]:
        soft_dd = calc_soft_dd(close, 200, lower, upper)
        trend_tv = calc_trend_target_vol(close)
        pred_vol = calc_predictive_asym_vol(returns, 20, 5, 5)
        vt_lev = (trend_tv / pred_vol).clip(0, 1.0)
        slope_mult = calc_slope_multiplier(close)
        decel_mult = calc_momentum_decel_mult(close, 20, 60, 0.3)
        lev = (soft_dd * vt_lev * slope_mult * decel_mult).clip(0, 1.0).fillna(0)
        m = run_test(lev, None,
                     f'SoftDD({lower}-{upper})+MomDec+PredVol+S+T', 'P6-ComboE',
                     soft_rebal=True)
        print_result(m)

    # Combination F: Signal prediction applied to best baseline
    print(f"\n  [Combo F: Signal prediction on Ens2(S+T)]")
    for fwd in [3, 5, 7]:
        raw_lev = lev_st.copy()
        predicted = predict_leverage_forward(raw_lev, fwd)
        m = run_test(predicted.clip(0, 1.0), dd_st,
                     f'SigPred(fwd={fwd}) on Ens2(S+T)', 'P6-ComboF')
        print_result(m)

    # =====================================================================
    # PART 7: FULL DELAY ROBUSTNESS COMPARISON
    # =====================================================================
    print("\n" + "=" * 130)
    print("PART 7: FULL DELAY ROBUSTNESS COMPARISON")
    print("  Test top strategies from each part at delays 0, 1, 3, 5, 7, 10")
    print("=" * 130)

    # Select top strategies for robustness test
    # (recompute raw leverage for each)
    robustness_strategies = {}

    # Baseline
    robustness_strategies['Ens2(S+T) [Baseline]'] = (lev_st, dd_st, False)
    robustness_strategies['DD-Only [Baseline]'] = (lev_dd, dd_dd, False)

    # Best Soft DD candidates
    best_soft = strategy_soft_dd_ens2_st(close, returns, 0.80, 0.95)
    robustness_strategies['SoftDD(80-95)+S+T'] = (best_soft, None, True)

    best_floor = strategy_soft_dd_floor_ens2_st(close, returns, 0.80, 0.95, 0.78)
    robustness_strategies['SoftDD(80-95)F78+S+T'] = (best_floor, None, True)

    # Best Anticipatory DD
    antic_lev, antic_dd = strategy_anticipatory_dd_ens2_st(close, returns, 5)
    robustness_strategies['AnticDD(5d)+S+T'] = (antic_lev, antic_dd, False)

    # Best Delay-adjusted DD
    dadj_lev, dadj_dd = strategy_delay_adjusted_dd_ens2_st(close, returns, 0.05, 0.03)
    robustness_strategies['DelayAdj(+5/+3)+S+T'] = (dadj_lev, dadj_dd, False)

    # Best Predictive Vol
    pvol_lev, pvol_dd = strategy_predictive_vol_ens2(close, returns, 5)
    robustness_strategies['PredVol(5d)+Ens2'] = (pvol_lev, pvol_dd, False)

    # Best MomDecel
    mdecel_lev, mdecel_dd = strategy_momentum_decel_ens2_st(close, returns, 20, 60, 0.3)
    robustness_strategies['MomDecel(20/60)+Ens2(S+T)'] = (mdecel_lev, mdecel_dd, False)

    # Best combo: Soft DD + PredVol
    soft_dd_c = calc_soft_dd(close, 200, 0.80, 0.95)
    trend_tv_c = calc_trend_target_vol(close)
    pred_vol_c = calc_predictive_asym_vol(returns, 20, 5, 5)
    vt_lev_c = (trend_tv_c / pred_vol_c).clip(0, 1.0)
    slope_mult_c = calc_slope_multiplier(close)
    combo_lev = (soft_dd_c * vt_lev_c * slope_mult_c).clip(0, 1.0).fillna(0)
    robustness_strategies['SoftDD(80-95)+PredVol+S+T'] = (combo_lev, None, True)

    # Best combo: Soft DD + MomDecel + PredVol
    decel_mult_c = calc_momentum_decel_mult(close, 20, 60, 0.3)
    combo_triple = (soft_dd_c * vt_lev_c * slope_mult_c * decel_mult_c).clip(0, 1.0).fillna(0)
    robustness_strategies['Triple: SoftDD+MomDec+PredVol'] = (combo_triple, None, True)

    delays_robustness = [0, 1, 3, 5, 7, 10]
    robustness_data = {}

    for name, (lev, dd, soft) in robustness_strategies.items():
        robustness_data[name] = {}
        for d in delays_robustness:
            m = test_strategy(close, dates, lev, dd, f'{name} d={d}', 'P7',
                              delay=d, threshold=DEFAULT_THRESHOLD,
                              use_soft_rebal=soft)
            robustness_data[name][d] = m
            all_results.append(m)

    # Print robustness table
    print(f"\n  {'Strategy':<40s}", end="")
    for d in delays_robustness:
        print(f"  d={d:<3d}", end="")
    print("  d5/d0   d5-d0   d10-d0")
    print("  " + "-" * 120)

    for name in robustness_strategies:
        print(f"  {name:<40s}", end="")
        for d in delays_robustness:
            print(f"  {robustness_data[name][d]['Sharpe']:.3f}", end="")
        s0 = robustness_data[name][0]['Sharpe']
        s5 = robustness_data[name][5]['Sharpe']
        s10 = robustness_data[name][10]['Sharpe']
        ratio = s5 / s0 if s0 > 0 else 0
        print(f"  {ratio:.3f}  {s5 - s0:+.3f}  {s10 - s0:+.3f}")

    # MaxDD robustness
    print(f"\n  MaxDD at each delay:")
    print(f"  {'Strategy':<40s}", end="")
    for d in delays_robustness:
        print(f"  d={d:<3d}", end="")
    print()
    print("  " + "-" * 100)

    for name in robustness_strategies:
        print(f"  {name:<40s}", end="")
        for d in delays_robustness:
            print(f"  {robustness_data[name][d]['MaxDD']*100:.1f}%", end="")
        print()

    # Worst5Y robustness
    print(f"\n  Worst5Y at each delay:")
    print(f"  {'Strategy':<40s}", end="")
    for d in delays_robustness:
        print(f"  d={d:<3d}", end="")
    print()
    print("  " + "-" * 100)

    for name in robustness_strategies:
        print(f"  {name:<40s}", end="")
        for d in delays_robustness:
            w5y = robustness_data[name][d]['Worst5Y']
            print(f"  {w5y*100:+.1f}%", end="")
        print()

    # =====================================================================
    # FINAL RANKING (at delay=5)
    # =====================================================================
    print("\n" + "=" * 130)
    print("FINAL RANKING: ALL STRATEGIES AT DELAY=5 (sorted by Sharpe)")
    print("=" * 130)

    # Collect all delay=5 results
    d5_results = [r for r in all_results if r.get('Delay') == 5]
    # Remove duplicates (keep last occurrence of each strategy name)
    seen = {}
    for r in d5_results:
        seen[r['Strategy']] = r
    unique_d5 = sorted(seen.values(), key=lambda x: -x['Sharpe'])

    print(f"\n  {'#':<4s} {'Strategy':<65s} {'Sharpe':>7s} {'CAGR':>7s} "
          f"{'MaxDD':>7s} {'W5Y':>7s} {'Rebal':>6s}")
    print("  " + "-" * 110)

    for i, r in enumerate(unique_d5[:30], 1):
        print(f"  {i:<4d} {r['Strategy']:<65s} {r['Sharpe']:.3f} "
              f"{r['CAGR']*100:+.1f}% {r['MaxDD']*100:.1f}% "
              f"{r['Worst5Y']*100:+.1f}% {r['Rebalances']:>5d}")

    # =====================================================================
    # KEY FINDINGS
    # =====================================================================
    print("\n" + "=" * 130)
    print("KEY FINDINGS")
    print("=" * 130)

    baseline_sharpe_5 = robustness_data['Ens2(S+T) [Baseline]'][5]['Sharpe']
    baseline_sharpe_0 = robustness_data['Ens2(S+T) [Baseline]'][0]['Sharpe']

    print(f"\n  Baseline Ens2(S+T) at delay=0: Sharpe = {baseline_sharpe_0:.4f}")
    print(f"  Baseline Ens2(S+T) at delay=5: Sharpe = {baseline_sharpe_5:.4f}")
    print(f"  Degradation: {baseline_sharpe_5 - baseline_sharpe_0:+.4f} "
          f"({(baseline_sharpe_5/baseline_sharpe_0 - 1)*100:+.1f}%)")

    # Best at delay=5
    best_d5 = unique_d5[0]
    print(f"\n  Best strategy at delay=5: {best_d5['Strategy']}")
    print(f"    Sharpe = {best_d5['Sharpe']:.4f}  "
          f"(vs baseline {baseline_sharpe_5:.4f}, diff = {best_d5['Sharpe'] - baseline_sharpe_5:+.4f})")

    # Most delay-robust
    best_robust = None
    best_ratio = 0
    for name, data in robustness_data.items():
        if data[0]['Sharpe'] > 0.5:  # minimum quality filter
            ratio = data[5]['Sharpe'] / data[0]['Sharpe']
            if ratio > best_ratio:
                best_ratio = ratio
                best_robust = name

    if best_robust:
        rd = robustness_data[best_robust]
        print(f"\n  Most delay-robust: {best_robust}")
        print(f"    d0={rd[0]['Sharpe']:.4f}, d5={rd[5]['Sharpe']:.4f}, "
              f"ratio={rd[5]['Sharpe']/rd[0]['Sharpe']:.4f}")

    # Save results
    output_path = os.path.join(script_dir, '..', 'delay_robust_results.csv')
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
