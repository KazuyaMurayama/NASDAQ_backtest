"""
3x Leveraged NASDAQ Backtest Engine - Round 4
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Callable
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Data Loading
# =============================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """Load NASDAQ daily data"""
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Close'] = df['Close'].astype(float)
    return df

# =============================================================================
# Indicator Calculations
# =============================================================================
def calc_ewma_vol(returns: pd.Series, span: int = 10) -> pd.Series:
    """EWMA Volatility (annualized)"""
    ewma_var = returns.ewm(span=span).var()
    return np.sqrt(ewma_var * 252)

def calc_simple_vol(returns: pd.Series, lookback: int = 20) -> pd.Series:
    """Simple rolling volatility (annualized)"""
    return returns.rolling(lookback).std() * np.sqrt(252)

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(lookback).mean()

def calc_bollinger_bands(close: pd.Series, lookback: int = 200, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    ma = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

def calc_vol_of_vol(returns: pd.Series, vol_lookback: int = 20, vov_lookback: int = 60) -> pd.Series:
    """Volatility of Volatility"""
    vol = calc_simple_vol(returns, vol_lookback)
    vol_changes = vol.pct_change()
    return vol_changes.rolling(vov_lookback).std()

def calc_rolling_sharpe(returns: pd.Series, lookback: int = 60) -> pd.Series:
    """Rolling Sharpe Ratio (annualized)"""
    roll_mean = returns.rolling(lookback).mean() * 252
    roll_std = returns.rolling(lookback).std() * np.sqrt(252)
    return roll_mean / roll_std.replace(0, np.nan)

def calc_momentum_score(close: pd.Series, lookback: int = 120) -> pd.Series:
    """Momentum Score (returns over lookback period)"""
    return close.pct_change(lookback)

# =============================================================================
# Signal Generation - DD Control
# =============================================================================
def calc_dd_signal(close: pd.Series, exit_threshold: float = 0.82,
                   reentry_threshold: float = 0.92, lookback: int = 200) -> pd.Series:
    """Standard DD Control Signal"""
    peak = close.rolling(lookback, min_periods=1).max()
    ratio = close / peak

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'

    for i in range(len(close)):
        if state == 'HOLD' and ratio.iloc[i] <= exit_threshold:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= reentry_threshold:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    return position

def calc_dual_dd_signal(close: pd.Series, short_lb: int = 100, long_lb: int = 200,
                        short_exit: float = 0.87, long_exit: float = 0.85,
                        reentry: float = 0.92) -> pd.Series:
    """Dual Timeframe DD Control"""
    peak_short = close.rolling(short_lb, min_periods=1).max()
    peak_long = close.rolling(long_lb, min_periods=1).max()
    ratio_short = close / peak_short
    ratio_long = close / peak_long

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'

    for i in range(len(close)):
        if state == 'HOLD':
            if ratio_short.iloc[i] <= short_exit or ratio_long.iloc[i] <= long_exit:
                state = 'CASH'
        elif state == 'CASH':
            if ratio_short.iloc[i] >= reentry and ratio_long.iloc[i] >= reentry:
                state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    return position

def calc_triple_dd_signal(close: pd.Series, lb1: int = 50, lb2: int = 100, lb3: int = 200,
                          exit1: float = 0.90, exit2: float = 0.87, exit3: float = 0.85,
                          reentry: float = 0.92) -> pd.Series:
    """Triple Timeframe DD Control"""
    peak1 = close.rolling(lb1, min_periods=1).max()
    peak2 = close.rolling(lb2, min_periods=1).max()
    peak3 = close.rolling(lb3, min_periods=1).max()
    ratio1 = close / peak1
    ratio2 = close / peak2
    ratio3 = close / peak3

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'

    for i in range(len(close)):
        if state == 'HOLD':
            if ratio1.iloc[i] <= exit1 or ratio2.iloc[i] <= exit2 or ratio3.iloc[i] <= exit3:
                state = 'CASH'
        elif state == 'CASH':
            if ratio1.iloc[i] >= reentry and ratio2.iloc[i] >= reentry and ratio3.iloc[i] >= reentry:
                state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    return position

# =============================================================================
# Volatility Targeting
# =============================================================================
def calc_vt_leverage(vol: pd.Series, target_vol: float = 0.25, max_lev: float = 1.0) -> pd.Series:
    """Volatility Targeting Leverage

    Returns a leverage factor between 0 and max_lev (default 1.0).
    This factor is applied to the 3x base leverage.
    - leverage=1.0 means 100% position in 3x product (3x effective)
    - leverage=0.5 means 50% position in 3x product (1.5x effective)
    - leverage=0.0 means 0% position (cash)
    """
    leverage = (target_vol / vol).clip(0, max_lev)
    return leverage.fillna(1.0)

# =============================================================================
# Backtest Execution
# =============================================================================
def run_backtest(close: pd.Series, leverage: pd.Series,
                 base_leverage: float = 3.0, annual_cost: float = 0.009) -> Dict:
    """Run backtest with given leverage series

    Cost is only charged when invested (leverage > 0).
    When in cash (leverage = 0), return is 0%.
    """
    returns = close.pct_change()
    leveraged_returns = returns * base_leverage
    daily_cost = annual_cost / 252

    # Apply leverage adjustment - cost only when invested
    # leverage * (3x_returns - cost) = leveraged return minus proportional cost
    strategy_returns = leverage.shift(1) * (leveraged_returns - daily_cost)
    strategy_returns = strategy_returns.fillna(0)

    # Calculate NAV
    nav = (1 + strategy_returns).cumprod()

    return nav, strategy_returns

def calc_metrics(nav: pd.Series, returns: pd.Series, position: pd.Series = None, dates: pd.Series = None) -> Dict:
    """Calculate performance metrics"""
    total_days = len(nav)
    total_years = total_days / 252

    # CAGR
    cagr = (nav.iloc[-1] ** (1 / total_years)) - 1

    # MaxDD
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Sharpe
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    # Sortino
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = annual_ret / downside_vol if downside_vol > 0 else 0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade count (if position provided)
    trades = 0
    if position is not None:
        changes = position.diff().abs()
        trades = (changes > 0.5).sum()

    # Worst 5-year average return
    worst_5y = calc_worst_5year_return(nav)

    # Win rate (annual) - calculate manually without resample
    if dates is not None:
        nav_df = pd.DataFrame({'nav': nav.values, 'date': dates.values})
        nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
        yearly_nav = nav_df.groupby('year')['nav'].last()
        annual_returns = yearly_nav.pct_change().dropna()
        win_rate = (annual_returns > 0).mean() if len(annual_returns) > 0 else 0
    else:
        # Fallback: calculate based on 252-day periods
        annual_nav = nav.iloc[::252]
        annual_returns = annual_nav.pct_change().dropna()
        win_rate = (annual_returns > 0).mean() if len(annual_returns) > 0 else 0

    return {
        'CAGR': cagr,
        'MaxDD': max_dd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Trades': trades,
        'Worst5Y': worst_5y,
        'WinRate': win_rate
    }

def calc_worst_5year_return(nav: pd.Series) -> float:
    """Calculate worst 5-year rolling CAGR"""
    if len(nav) < 252 * 5:
        return np.nan

    # Calculate 5-year rolling returns
    nav_5y_ago = nav.shift(252 * 5)
    rolling_5y_return = (nav / nav_5y_ago) ** (1/5) - 1

    return rolling_5y_return.min()

def count_trades(position: pd.Series) -> int:
    """Count number of trades (state transitions)"""
    changes = position.diff().abs()
    return (changes > 0.5).sum()

# =============================================================================
# Crisis Year Analysis
# =============================================================================
def calc_crisis_returns(nav: pd.Series, dates: pd.Series) -> Dict:
    """Calculate returns for crisis years"""
    df = pd.DataFrame({'nav': nav, 'date': dates})
    df['year'] = df['date'].dt.year

    crisis_years = [1974, 1987, 2000, 2001, 2002, 2008, 2011, 2020]
    crisis_returns = {}

    for year in crisis_years:
        year_data = df[df['year'] == year]
        if len(year_data) > 0:
            start_nav = year_data['nav'].iloc[0]
            end_nav = year_data['nav'].iloc[-1]
            crisis_returns[year] = (end_nav / start_nav - 1) * 100
        else:
            crisis_returns[year] = np.nan

    return crisis_returns

# =============================================================================
# Strategy Implementations - Category B: Vol-of-Vol
# =============================================================================
def strategy_dd_vt_vov(close: pd.Series, returns: pd.Series,
                       exit_th: float = 0.82, reentry_th: float = 0.92,
                       target_vol: float = 0.25, ewma_span: int = 10,
                       vov_mult: float = 1.5, lev_cap: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """DD + VT + Vol-of-Vol Control"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    # Vol-of-Vol
    vov = calc_vol_of_vol(returns)
    vov_median = vov.rolling(500, min_periods=100).median()
    high_vov = vov > vov_median * vov_mult

    # Apply cap when high VoV
    final_lev = vt_lev.copy()
    final_lev[high_vov] = final_lev[high_vov].clip(upper=lev_cap)

    leverage = dd_signal * final_lev
    return leverage, dd_signal

def strategy_dd_vt_volspike(close: pd.Series, returns: pd.Series,
                            exit_th: float = 0.82, reentry_th: float = 0.92,
                            target_vol: float = 0.25, ewma_span: int = 10,
                            spike_mult: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    """DD + VT + Vol Spike Reduction"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    # Vol spike detection
    vol_ratio = ewma_vol / ewma_vol.shift(1)
    spike = vol_ratio > spike_mult

    # Reduce leverage on spike
    final_lev = vt_lev.copy()
    final_lev[spike] = final_lev[spike] * 0.5

    leverage = dd_signal * final_lev
    return leverage, dd_signal

# =============================================================================
# Strategy Implementations - Category D: Adaptive Reentry
# =============================================================================
def strategy_dd_adaptive_reentry(close: pd.Series, returns: pd.Series,
                                  exit_th: float = 0.82,
                                  target_vol: float = 0.25, ewma_span: int = 10,
                                  low_vol_re: float = 0.88, high_vol_re: float = 0.95) -> Tuple[pd.Series, pd.Series]:
    """DD with Adaptive Reentry based on Vol"""
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vol_median = ewma_vol.rolling(500, min_periods=100).median()

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'

    for i in range(len(close)):
        current_vol = ewma_vol.iloc[i] if not pd.isna(ewma_vol.iloc[i]) else 0.2
        median_vol = vol_median.iloc[i] if not pd.isna(vol_median.iloc[i]) else 0.2

        # Dynamic reentry threshold
        if current_vol < median_vol:
            reentry = low_vol_re
        else:
            reentry = high_vol_re

        if state == 'HOLD' and ratio.iloc[i] <= exit_th:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= reentry:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    vt_lev = calc_vt_leverage(ewma_vol, target_vol)
    leverage = position * vt_lev
    return leverage, position

def strategy_dd_trend_confirm_reentry(close: pd.Series, returns: pd.Series,
                                       exit_th: float = 0.82, reentry_th: float = 0.92,
                                       target_vol: float = 0.25, ewma_span: int = 10,
                                       confirm_days: int = 3) -> Tuple[pd.Series, pd.Series]:
    """DD with Trend Confirmation on Reentry"""
    ewma_vol = calc_ewma_vol(returns, ewma_span)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak

    # Count consecutive up days
    up_days = (returns > 0).astype(int)
    consec_up = up_days.rolling(confirm_days).sum()

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'

    for i in range(len(close)):
        if state == 'HOLD' and ratio.iloc[i] <= exit_th:
            state = 'CASH'
        elif state == 'CASH':
            # Need both ratio recovery AND consecutive up days
            if ratio.iloc[i] >= reentry_th and consec_up.iloc[i] >= confirm_days:
                state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    vt_lev = calc_vt_leverage(ewma_vol, target_vol)
    leverage = position * vt_lev
    return leverage, position

def strategy_dd_staged_reentry(close: pd.Series, returns: pd.Series,
                               exit_th: float = 0.82,
                               target_vol: float = 0.25, ewma_span: int = 10,
                               stage1_th: float = 0.88, stage2_th: float = 0.92) -> Tuple[pd.Series, pd.Series]:
    """DD with Staged Reentry"""
    ewma_vol = calc_ewma_vol(returns, ewma_span)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'

    for i in range(len(close)):
        if state == 'HOLD' and ratio.iloc[i] <= exit_th:
            state = 'CASH'
        elif state == 'CASH':
            if ratio.iloc[i] >= stage2_th:
                state = 'HOLD'
            elif ratio.iloc[i] >= stage1_th:
                state = 'HALF'
        elif state == 'HALF':
            if ratio.iloc[i] >= stage2_th:
                state = 'HOLD'
            elif ratio.iloc[i] <= exit_th:
                state = 'CASH'

        if state == 'HOLD':
            position.iloc[i] = 1.0
        elif state == 'HALF':
            position.iloc[i] = 0.5
        else:
            position.iloc[i] = 0.0

    vt_lev = calc_vt_leverage(ewma_vol, target_vol)
    leverage = position * vt_lev
    return leverage, position

# =============================================================================
# Strategy Implementations - Category E: Triple DD
# =============================================================================
def strategy_triple_dd_vt(close: pd.Series, returns: pd.Series,
                          lb1: int = 50, lb2: int = 100, lb3: int = 200,
                          exit1: float = 0.90, exit2: float = 0.87, exit3: float = 0.85,
                          reentry: float = 0.92,
                          target_vol: float = 0.25, ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Triple Timeframe DD + VT"""
    dd_signal = calc_triple_dd_signal(close, lb1, lb2, lb3, exit1, exit2, exit3, reentry)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)
    leverage = dd_signal * vt_lev
    return leverage, dd_signal

# =============================================================================
# Strategy Implementations - Category F: Risk-Adjusted Momentum
# =============================================================================
def strategy_dd_rolling_sharpe(close: pd.Series, returns: pd.Series,
                               exit_th: float = 0.82, reentry_th: float = 0.92,
                               target_vol: float = 0.25, ewma_span: int = 10,
                               sharpe_lb: int = 60, sharpe_th: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """DD + VT + Rolling Sharpe Filter"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    rolling_sharpe = calc_rolling_sharpe(returns, sharpe_lb)
    sharpe_filter = (rolling_sharpe > sharpe_th).astype(float)
    sharpe_filter = sharpe_filter.fillna(1.0)

    # Combine signals
    combined_signal = dd_signal * sharpe_filter
    leverage = combined_signal * vt_lev
    return leverage, combined_signal

def strategy_dd_sharpe_based_lev(close: pd.Series, returns: pd.Series,
                                  exit_th: float = 0.82, reentry_th: float = 0.92,
                                  sharpe_lb: int = 60, sharpe_mult: float = 1.5,
                                  max_lev: float = 1.0) -> Tuple[pd.Series, pd.Series]:
    """DD + Sharpe-Based Leverage"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    rolling_sharpe = calc_rolling_sharpe(returns, sharpe_lb)

    # Leverage based on Sharpe (normalize to 0-1 range)
    sharpe_lev = (rolling_sharpe / 2.0 * sharpe_mult).clip(0, max_lev)
    sharpe_lev = sharpe_lev.fillna(0.5)

    leverage = dd_signal * sharpe_lev
    return leverage, dd_signal

# =============================================================================
# Strategy Implementations - Category H: Advanced Ensemble
# =============================================================================
def strategy_weighted_ensemble(close: pd.Series, returns: pd.Series,
                               weights: List[float] = [0.5, 0.3, 0.2],
                               target_vol: float = 0.25, ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Weighted Ensemble of Multiple Strategies"""
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    # Strategy 1: DD-18/92
    dd1 = calc_dd_signal(close, 0.82, 0.92)
    # Strategy 2: DD-15/90
    dd2 = calc_dd_signal(close, 0.85, 0.90)
    # Strategy 3: Dual DD
    dd3 = calc_dual_dd_signal(close)

    # Weighted average
    combined = weights[0] * dd1 + weights[1] * dd2 + weights[2] * dd3
    leverage = combined * vt_lev

    # For trade counting, use majority signal
    majority = ((dd1 + dd2 + dd3) >= 2).astype(float)
    return leverage, majority

def strategy_voting_ensemble(close: pd.Series, returns: pd.Series,
                             min_votes: int = 2,
                             target_vol: float = 0.25, ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Voting Ensemble (majority vote)"""
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    # Multiple DD strategies
    dd1 = calc_dd_signal(close, 0.82, 0.92)  # DD-18/92
    dd2 = calc_dd_signal(close, 0.85, 0.90)  # DD-15/90
    dd3 = calc_dual_dd_signal(close)         # Dual DD

    # Vote
    total_votes = dd1 + dd2 + dd3
    position = (total_votes >= min_votes).astype(float)

    leverage = position * vt_lev
    return leverage, position

def strategy_dynamic_ensemble(close: pd.Series, returns: pd.Series,
                              perf_lb: int = 60,
                              target_vol: float = 0.25, ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Dynamic Weight Ensemble based on recent performance"""
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    # Calculate individual strategy returns
    dd1 = calc_dd_signal(close, 0.82, 0.92)
    dd2 = calc_dd_signal(close, 0.85, 0.90)
    dd3 = calc_dual_dd_signal(close)

    daily_ret = close.pct_change()

    ret1 = (dd1.shift(1) * daily_ret).rolling(perf_lb).mean()
    ret2 = (dd2.shift(1) * daily_ret).rolling(perf_lb).mean()
    ret3 = (dd3.shift(1) * daily_ret).rolling(perf_lb).mean()

    # Normalize weights
    total = ret1.clip(0) + ret2.clip(0) + ret3.clip(0)
    w1 = (ret1.clip(0) / total).fillna(1/3)
    w2 = (ret2.clip(0) / total).fillna(1/3)
    w3 = (ret3.clip(0) / total).fillna(1/3)

    combined = w1 * dd1 + w2 * dd2 + w3 * dd3
    leverage = combined * vt_lev

    majority = ((dd1 + dd2 + dd3) >= 2).astype(float)
    return leverage, majority

# =============================================================================
# Strategy Implementations - Category A: Trend Strength
# =============================================================================
def strategy_dd_divergence_filter(close: pd.Series, returns: pd.Series,
                                   exit_th: float = 0.82, reentry_th: float = 0.92,
                                   target_vol: float = 0.25, ewma_span: int = 10,
                                   ma_lb: int = 200, div_th: float = 0.03) -> Tuple[pd.Series, pd.Series]:
    """DD + VT + MA Divergence Filter"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    # MA divergence
    ma = close.rolling(ma_lb).mean()
    divergence = (close - ma) / ma
    trend_filter = (divergence > div_th).astype(float)
    trend_filter = trend_filter.fillna(1.0)

    combined = dd_signal * trend_filter
    leverage = combined * vt_lev
    return leverage, combined

def strategy_dd_momentum_score(close: pd.Series, returns: pd.Series,
                               exit_th: float = 0.82, reentry_th: float = 0.92,
                               target_vol: float = 0.25, ewma_span: int = 10,
                               mom_lb: int = 120, mom_th: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """DD + VT + Momentum Score Filter"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    momentum = calc_momentum_score(close, mom_lb)
    mom_filter = (momentum > mom_th).astype(float)
    mom_filter = mom_filter.fillna(1.0)

    combined = dd_signal * mom_filter
    leverage = combined * vt_lev
    return leverage, combined

# =============================================================================
# Strategy Implementations - Category C: ATR/Bollinger
# =============================================================================
def strategy_atr_adaptive_dd(close: pd.Series, returns: pd.Series, high: pd.Series, low: pd.Series,
                              base_exit: float = 0.82, atr_sensitivity: float = 0.3,
                              reentry: float = 0.92,
                              target_vol: float = 0.25, ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """ATR-Adaptive DD + VT"""
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    atr = calc_atr(high, low, close, 20)
    atr_pct = atr / close
    atr_median = atr_pct.rolling(500, min_periods=100).median()

    # Dynamic exit threshold
    atr_ratio = atr_pct / atr_median
    dynamic_exit = base_exit + atr_sensitivity * (atr_ratio - 1) * (1 - base_exit)
    dynamic_exit = dynamic_exit.clip(base_exit - 0.05, base_exit + 0.10)

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'

    for i in range(len(close)):
        exit_th = dynamic_exit.iloc[i] if not pd.isna(dynamic_exit.iloc[i]) else base_exit

        if state == 'HOLD' and ratio.iloc[i] <= exit_th:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= reentry:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    leverage = position * vt_lev
    return leverage, position

def strategy_bb_exit(close: pd.Series, returns: pd.Series,
                     exit_th: float = 0.82, reentry_th: float = 0.92,
                     target_vol: float = 0.25, ewma_span: int = 10,
                     bb_lb: int = 200, bb_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """DD + BB Exit Confirmation + VT"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    ma, upper, lower = calc_bollinger_bands(close, bb_lb, bb_std)

    # Additional exit when price below lower band
    bb_exit = (close < lower).astype(float)

    # Combine: exit if DD says exit OR BB says exit
    combined = dd_signal.copy()
    combined[bb_exit == 1] = 0

    leverage = combined * vt_lev
    return leverage, combined

# =============================================================================
# Strategy Implementations - Category I: Asymmetric
# =============================================================================
def strategy_asymmetric_dd(close: pd.Series, returns: pd.Series,
                           up_exit: float = 0.80, down_exit: float = 0.85,
                           reentry: float = 0.92,
                           target_vol: float = 0.25, ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Asymmetric DD (different thresholds for up/down trends)"""
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    ma200 = close.rolling(200).mean()
    uptrend = close > ma200

    peak = close.rolling(200, min_periods=1).max()
    ratio = close / peak

    position = pd.Series(1.0, index=close.index)
    state = 'HOLD'

    for i in range(len(close)):
        exit_th = up_exit if uptrend.iloc[i] else down_exit

        if state == 'HOLD' and ratio.iloc[i] <= exit_th:
            state = 'CASH'
        elif state == 'CASH' and ratio.iloc[i] >= reentry:
            state = 'HOLD'
        position.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    leverage = position * vt_lev
    return leverage, position

def strategy_asymmetric_leverage(close: pd.Series, returns: pd.Series,
                                  exit_th: float = 0.82, reentry_th: float = 0.92,
                                  up_tv: float = 0.30, down_tv: float = 0.20,
                                  ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """DD + Asymmetric Leverage (different target vol for up/down)"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)

    ma200 = close.rolling(200).mean()
    uptrend = close > ma200

    # Dynamic target vol
    target_vol = pd.Series(down_tv, index=close.index)
    target_vol[uptrend] = up_tv

    vt_lev = (target_vol / ewma_vol).clip(0, 1.0).fillna(1.0)
    leverage = dd_signal * vt_lev
    return leverage, dd_signal

# =============================================================================
# DD + Regime (MA200) + VT - Triple Layer Strategy
# =============================================================================
def strategy_dd_regime_vt(close: pd.Series, returns: pd.Series,
                          exit_th: float = 0.82, reentry_th: float = 0.92,
                          ma_lookback: int = 200,
                          target_vol: float = 0.30, ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """
    Triple Layer Strategy:
    Layer 1: DD Control (crash avoidance)
    Layer 2: Regime Filter (Price > MA200 = Bull)
    Layer 3: Volatility Targeting

    Investment only when ALL three layers are GO:
    - Price < MA200 OR DD = CASH → 0x (cash)
    - Price > MA200 AND DD = HOLD → VT-adjusted leverage
    """
    # Layer 1: DD Signal
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)

    # Layer 2: Regime Filter (Price > MA200)
    ma200 = close.rolling(ma_lookback).mean()
    regime_signal = (close > ma200).astype(float)
    regime_signal = regime_signal.fillna(0)

    # Layer 3: Volatility Targeting
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol, max_lev=1.0)

    # Combined: All three must be GO
    # DD=1 AND Regime=1 → apply VT leverage
    # Otherwise → 0
    combined_signal = dd_signal * regime_signal
    leverage = combined_signal * vt_lev

    return leverage, combined_signal

# =============================================================================
# Category G: Seasonality
# =============================================================================
def strategy_dd_quarterly_filter(close: pd.Series, returns: pd.Series, dates: pd.Series,
                                  exit_th: float = 0.82, reentry_th: float = 0.92,
                                  target_vol: float = 0.25, ewma_span: int = 10,
                                  weak_months: List[int] = [7, 8, 9],
                                  weak_mult: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    """DD + VT + Quarterly Seasonality Filter"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)

    months = dates.dt.month
    seasonal_adj = pd.Series(1.0, index=close.index)
    for m in weak_months:
        seasonal_adj[months == m] = weak_mult

    leverage = dd_signal * vt_lev * seasonal_adj
    return leverage, dd_signal

# =============================================================================
# Baseline Strategies
# =============================================================================
def strategy_baseline_bh3x(close: pd.Series, returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Buy & Hold 3x"""
    leverage = pd.Series(1.0, index=close.index)
    position = pd.Series(1.0, index=close.index)
    return leverage, position

def strategy_baseline_dd_only(close: pd.Series, returns: pd.Series,
                              exit_th: float = 0.82, reentry_th: float = 0.92) -> Tuple[pd.Series, pd.Series]:
    """DD Control Only (no VT)"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    return dd_signal, dd_signal

def strategy_baseline_dd_vt(close: pd.Series, returns: pd.Series,
                            exit_th: float = 0.82, reentry_th: float = 0.92,
                            target_vol: float = 0.25, ewma_span: int = 10) -> Tuple[pd.Series, pd.Series]:
    """DD + EWMA VT (R3 Best)"""
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)
    ewma_vol = calc_ewma_vol(returns, ewma_span)
    vt_lev = calc_vt_leverage(ewma_vol, target_vol)
    leverage = dd_signal * vt_lev
    return leverage, dd_signal

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("Backtest Engine loaded successfully")
