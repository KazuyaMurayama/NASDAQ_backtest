"""
Test Ens2 Strategies: Ens2(Slope+TrendTV) and Ens2(Asym+Slope)
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    load_data, calc_dd_signal, run_backtest, calc_metrics,
    strategy_dd_vt_volspike, strategy_baseline_dd_vt
)

# =============================================================================
# Component 1: AsymEWMA - Asymmetric EWMA Volatility
# =============================================================================
def calc_asym_ewma_vol(returns: pd.Series, span_up: int = 20, span_dn: int = 5) -> pd.Series:
    """
    Asymmetric EWMA Volatility
    - Negative returns: fast EWMA (span_dn=5) for quick vol detection
    - Positive returns: slow EWMA (span_up=20) for stable recovery
    """
    # Initialize with simple variance
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001

    for i in range(1, len(returns)):
        ret = returns.iloc[i]
        prev_var = variance.iloc[i-1]

        # Choose span based on return sign
        if ret < 0:
            alpha = 2 / (span_dn + 1)  # Fast reaction for negative returns
        else:
            alpha = 2 / (span_up + 1)  # Slow reaction for positive returns

        # EWMA variance update
        variance.iloc[i] = (1 - alpha) * prev_var + alpha * (ret ** 2)

    # Annualize
    return np.sqrt(variance * 252)


# =============================================================================
# Component 2: SlopeMult - MA200 Slope Multiplier
# =============================================================================
def calc_slope_multiplier(close: pd.Series, ma_lookback: int = 200,
                          norm_window: int = 60, base: float = 0.7,
                          sensitivity: float = 0.3,
                          min_mult: float = 0.3, max_mult: float = 1.5) -> pd.Series:
    """
    MA Slope Multiplier
    - Calculate MA200 daily change rate
    - Normalize with Z-score over 60-day window
    - Convert to multiplier (0.3 to 1.5)
    """
    # MA200
    ma = close.rolling(ma_lookback).mean()

    # Daily slope (percent change of MA)
    slope = ma.pct_change()

    # Z-score normalization over rolling window
    slope_mean = slope.rolling(norm_window).mean()
    slope_std = slope.rolling(norm_window).std()
    z_score = (slope - slope_mean) / slope_std.replace(0, 0.0001)

    # Convert to multiplier
    multiplier = base + sensitivity * z_score
    multiplier = multiplier.clip(min_mult, max_mult)

    return multiplier.fillna(1.0)


# =============================================================================
# Component 3: TrendTV - Trend-Linked Target Vol
# =============================================================================
def calc_trend_target_vol(close: pd.Series, ma_lookback: int = 150,
                          tv_min: float = 0.15, tv_max: float = 0.35,
                          ratio_low: float = 0.85, ratio_high: float = 1.15) -> pd.Series:
    """
    Trend-Linked Target Vol
    - Price/MA150 ratio maps to target vol
    - ratio <= 0.85 -> TV = 15% (defensive)
    - ratio >= 1.15 -> TV = 35% (aggressive)
    - Linear interpolation in between
    """
    ma = close.rolling(ma_lookback).mean()
    ratio = close / ma

    # Linear interpolation
    # slope = (tv_max - tv_min) / (ratio_high - ratio_low)
    # tv = tv_min + slope * (ratio - ratio_low)
    tv = tv_min + (tv_max - tv_min) * (ratio - ratio_low) / (ratio_high - ratio_low)
    tv = tv.clip(tv_min, tv_max)

    return tv.fillna(0.25)


# =============================================================================
# Strategy: Ens2(Asym+Slope) - DD + AsymEWMA VT + SlopeMult
# =============================================================================
def strategy_ens2_asym_slope(close: pd.Series, returns: pd.Series,
                              exit_th: float = 0.82, reentry_th: float = 0.92,
                              target_vol: float = 0.25,
                              span_up: int = 20, span_dn: int = 5,
                              max_lev: float = 3.0) -> tuple:
    """
    Ens2(Asym+Slope):
    - Layer 1: DD Control (-18/92)
    - Layer 2: AsymEWMA Volatility Targeting (25%, up20/dn5)
    - Layer 3: SlopeMult (MA200, N60) multiplier
    """
    # DD Signal
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)

    # AsymEWMA Vol
    asym_vol = calc_asym_ewma_vol(returns, span_up, span_dn)

    # Base VT leverage
    vt_lev = (target_vol / asym_vol).clip(0, max_lev)

    # Slope multiplier
    slope_mult = calc_slope_multiplier(close)

    # Final leverage
    leverage = dd_signal * vt_lev * slope_mult

    # Cap at max_lev
    leverage = leverage.clip(0, max_lev).fillna(0)

    return leverage, dd_signal


# =============================================================================
# Strategy: Ens2(Slope+TrendTV) - DD + TrendTV + SlopeMult
# =============================================================================
def strategy_ens2_slope_trendtv(close: pd.Series, returns: pd.Series,
                                 exit_th: float = 0.82, reentry_th: float = 0.92,
                                 span_up: int = 20, span_dn: int = 5,
                                 max_lev: float = 3.0) -> tuple:
    """
    Ens2(Slope+TrendTV):
    - Layer 1: DD Control (-18/92)
    - Layer 2: TrendTV (15/35%, MA150) with AsymEWMA vol
    - Layer 3: SlopeMult (MA200, N60) multiplier
    """
    # DD Signal
    dd_signal = calc_dd_signal(close, exit_th, reentry_th)

    # Trend-linked target vol
    trend_tv = calc_trend_target_vol(close)

    # AsymEWMA Vol
    asym_vol = calc_asym_ewma_vol(returns, span_up, span_dn)

    # VT leverage with dynamic target vol
    vt_lev = (trend_tv / asym_vol).clip(0, max_lev)

    # Slope multiplier
    slope_mult = calc_slope_multiplier(close)

    # Final leverage
    leverage = dd_signal * vt_lev * slope_mult

    # Cap at max_lev
    leverage = leverage.clip(0, max_lev).fillna(0)

    return leverage, dd_signal


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 100)
    print("Ens2 Strategies Comparison")
    print("=" * 100)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    results = []

    def add_result(lev, pos, name, category):
        nav, strat_ret = run_backtest(close, lev)
        metrics = calc_metrics(nav, strat_ret, pos, dates)
        metrics['Strategy'] = name
        metrics['Category'] = category
        results.append(metrics)
        print(f"{name}: CAGR={metrics['CAGR']*100:.2f}%, Sharpe={metrics['Sharpe']:.3f}, "
              f"MaxDD={metrics['MaxDD']*100:.2f}%, Trades={int(metrics['Trades'])}")
        return nav

    # ==========================================================================
    # Existing Top Strategies
    # ==========================================================================
    print("--- Existing Top Strategies ---")

    # Baseline: DD(-18/92)+VT(25%)
    lev, pos = strategy_baseline_dd_vt(close, returns, 0.82, 0.92, 0.25, 10)
    add_result(lev, pos, 'DD(-18/92)+VT(25%) [Baseline]', 'Baseline')

    # Top 1: DD+VT+VolSpike(1.5x)
    lev, pos = strategy_dd_vt_volspike(close, returns, 0.82, 0.92, 0.25, 10, 1.5)
    add_result(lev, pos, 'DD+VT+VolSpike(1.5x) [R4 Top1]', 'R4-Top')

    # ==========================================================================
    # New Ens2 Strategies
    # ==========================================================================
    print("\n--- New Ens2 Strategies ---")

    # Ens2(Asym+Slope) with max_lev=1.0 (conservative)
    lev, pos = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    add_result(lev, pos, 'Ens2(Asym+Slope) max_lev=1.0', 'Ens2-New')

    # Ens2(Asym+Slope) with max_lev=3.0 (original spec)
    lev, pos = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 3.0)
    add_result(lev, pos, 'Ens2(Asym+Slope) max_lev=3.0', 'Ens2-New')

    # Ens2(Slope+TrendTV) with max_lev=1.0 (conservative)
    lev, pos = strategy_ens2_slope_trendtv(close, returns, 0.82, 0.92, 20, 5, 1.0)
    add_result(lev, pos, 'Ens2(Slope+TrendTV) max_lev=1.0', 'Ens2-New')

    # Ens2(Slope+TrendTV) with max_lev=3.0 (original spec)
    lev, pos = strategy_ens2_slope_trendtv(close, returns, 0.82, 0.92, 20, 5, 3.0)
    add_result(lev, pos, 'Ens2(Slope+TrendTV) max_lev=3.0', 'Ens2-New')

    # ==========================================================================
    # Results Table
    # ==========================================================================
    print("\n" + "=" * 100)
    print("FULL COMPARISON TABLE (Sorted by Sharpe)")
    print("=" * 100)

    results_df = pd.DataFrame(results)
    cols = ['Category', 'Strategy', 'CAGR', 'Worst5Y', 'Sharpe', 'Sortino', 'Calmar', 'MaxDD', 'Trades', 'WinRate']
    results_df = results_df[cols]
    results_df = results_df.sort_values('Sharpe', ascending=False)

    # Format for display
    display_df = results_df.copy()
    for col in ['CAGR', 'Worst5Y', 'MaxDD', 'WinRate']:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    for col in ['Sharpe', 'Sortino', 'Calmar']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    display_df['Trades'] = display_df['Trades'].astype(int)

    print(display_df.to_string(index=False))

    # ==========================================================================
    # Component Analysis
    # ==========================================================================
    print("\n" + "=" * 100)
    print("COMPONENT ANALYSIS")
    print("=" * 100)

    # Test individual components
    print("\n[AsymEWMA Vol Statistics]")
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    print(f"  Mean: {asym_vol.mean()*100:.2f}%")
    print(f"  Median: {asym_vol.median()*100:.2f}%")
    print(f"  Min: {asym_vol.min()*100:.2f}%")
    print(f"  Max: {asym_vol.max()*100:.2f}%")

    print("\n[Slope Multiplier Statistics]")
    slope_mult = calc_slope_multiplier(close)
    print(f"  Mean: {slope_mult.mean():.3f}")
    print(f"  Median: {slope_mult.median():.3f}")
    print(f"  Min: {slope_mult.min():.3f}")
    print(f"  Max: {slope_mult.max():.3f}")
    print(f"  % time > 1.0: {(slope_mult > 1.0).mean()*100:.1f}%")
    print(f"  % time < 1.0: {(slope_mult < 1.0).mean()*100:.1f}%")

    print("\n[TrendTV Statistics]")
    trend_tv = calc_trend_target_vol(close)
    print(f"  Mean: {trend_tv.mean()*100:.2f}%")
    print(f"  Median: {trend_tv.median()*100:.2f}%")
    print(f"  Min: {trend_tv.min()*100:.2f}%")
    print(f"  Max: {trend_tv.max()*100:.2f}%")
    print(f"  % time at min (15%): {(trend_tv <= 0.16).mean()*100:.1f}%")
    print(f"  % time at max (35%): {(trend_tv >= 0.34).mean()*100:.1f}%")

    # ==========================================================================
    # Crisis Year Performance
    # ==========================================================================
    print("\n" + "=" * 100)
    print("CRISIS YEAR PERFORMANCE")
    print("=" * 100)

    crisis_years = [1974, 1987, 2000, 2001, 2002, 2008, 2020]

    strategies_for_crisis = [
        ('DD+VT+VolSpike(1.5x)', strategy_dd_vt_volspike(close, returns, 0.82, 0.92, 0.25, 10, 1.5)),
        ('Ens2(Asym+Slope) 1.0', strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)),
        ('Ens2(Slope+TrendTV) 1.0', strategy_ens2_slope_trendtv(close, returns, 0.82, 0.92, 20, 5, 1.0)),
    ]

    crisis_data = []
    for name, (lev, pos) in strategies_for_crisis:
        nav, _ = run_backtest(close, lev)

        # Calculate yearly returns
        nav_df = pd.DataFrame({'nav': nav.values, 'date': dates.values})
        nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
        yearly_nav = nav_df.groupby('year')['nav'].last()
        yearly_ret = yearly_nav.pct_change()
        yearly_ret.iloc[0] = yearly_nav.iloc[0] - 1

        row = {'Strategy': name}
        for year in crisis_years:
            if year in yearly_ret.index:
                row[str(year)] = yearly_ret[year] * 100
            else:
                row[str(year)] = np.nan
        crisis_data.append(row)

    crisis_df = pd.DataFrame(crisis_data)
    print("\nCrisis Year Returns (%):")
    display_crisis = crisis_df.copy()
    for col in [str(y) for y in crisis_years]:
        display_crisis[col] = display_crisis[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
    print(display_crisis.to_string(index=False))

    # ==========================================================================
    # Save Results
    # ==========================================================================
    results_df.to_csv(r"C:\Users\user\Desktop\nasdaq_backtest\ens2_comparison_results.csv", index=False)
    print(f"\nResults saved to ens2_comparison_results.csv")

    return results_df


if __name__ == "__main__":
    results = main()
