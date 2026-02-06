"""
Debug: Why is Sharpe 0.94 in our implementation but 1.8+ in original R3?
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import load_data, calc_dd_signal

def main():
    print("=" * 80)
    print("DEBUG: Sharpe Calculation Analysis")
    print("=" * 80)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total days: {len(df)}")
    print(f"Total years: {len(df)/252:.1f}\n")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    # DD Signal
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # EWMA Vol (Span=10)
    ewma_vol = returns.ewm(span=10).std() * np.sqrt(252)

    # ===== Test Different Implementations =====
    print("=" * 80)
    print("Testing Different Implementation Methods")
    print("=" * 80)

    # Method 1: Our current implementation (max_lev=3.0)
    target_vol = 0.25
    vt_lev_3 = (target_vol / ewma_vol).clip(0, 3.0).fillna(1.0)
    leverage_3 = dd_signal * vt_lev_3

    # Apply to 3x leveraged returns
    lev_returns = returns * 3.0
    daily_cost = 0.009 / 252
    strat_ret_3 = leverage_3.shift(1) * (lev_returns - daily_cost)
    strat_ret_3 = strat_ret_3.fillna(0)
    nav_3 = (1 + strat_ret_3).cumprod()

    # Metrics for Method 1
    years = len(nav_3) / 252
    cagr_3 = (nav_3.iloc[-1] ** (1/years)) - 1
    arith_mean_3 = strat_ret_3.mean() * 252
    annual_vol_3 = strat_ret_3.std() * np.sqrt(252)
    sharpe_3 = arith_mean_3 / annual_vol_3

    print(f"\n[Method 1: Our Implementation (max_lev=3.0)]")
    print(f"  CAGR: {cagr_3*100:.2f}%")
    print(f"  Arithmetic Mean Return (ann): {arith_mean_3*100:.2f}%")
    print(f"  Annual Volatility: {annual_vol_3*100:.2f}%")
    print(f"  Sharpe (arith/vol): {sharpe_3:.3f}")
    print(f"  Avg Effective Leverage: {(leverage_3 * 3).mean():.2f}x")

    # Method 2: What if original used CAGR for Sharpe calculation?
    sharpe_cagr = cagr_3 / annual_vol_3
    print(f"\n[Method 2: If Sharpe = CAGR / vol]")
    print(f"  Sharpe (cagr/vol): {sharpe_cagr:.3f}")

    # Method 3: What if original used log returns?
    log_returns = np.log(1 + returns)
    lev_log_returns = log_returns * 3.0
    vt_lev_log = (target_vol / ewma_vol).clip(0, 3.0).fillna(1.0)
    leverage_log = dd_signal * vt_lev_log
    strat_log_ret = leverage_log.shift(1) * (lev_log_returns - daily_cost)
    strat_log_ret = strat_log_ret.fillna(0)
    nav_log = np.exp(strat_log_ret.cumsum())

    arith_mean_log = strat_log_ret.mean() * 252
    annual_vol_log = strat_log_ret.std() * np.sqrt(252)
    sharpe_log = arith_mean_log / annual_vol_log

    print(f"\n[Method 3: Log Returns]")
    print(f"  Arithmetic Mean Return (ann): {arith_mean_log*100:.2f}%")
    print(f"  Annual Volatility: {annual_vol_log*100:.2f}%")
    print(f"  Sharpe: {sharpe_log:.3f}")

    # Method 4: What if VT leverage was applied differently?
    # Original might have: strategy_returns = leverage * daily_returns * 3 (not subtracting cost properly)
    strat_ret_4 = leverage_3.shift(1) * lev_returns - daily_cost  # Cost not multiplied by leverage
    strat_ret_4 = strat_ret_4.fillna(0)
    nav_4 = (1 + strat_ret_4).cumprod()

    arith_mean_4 = strat_ret_4.mean() * 252
    annual_vol_4 = strat_ret_4.std() * np.sqrt(252)
    sharpe_4 = arith_mean_4 / annual_vol_4

    print(f"\n[Method 4: Cost subtracted after (not multiplied by leverage)]")
    print(f"  Arithmetic Mean Return (ann): {arith_mean_4*100:.2f}%")
    print(f"  Annual Volatility: {annual_vol_4*100:.2f}%")
    print(f"  Sharpe: {sharpe_4:.3f}")

    # Method 5: No cost at all
    strat_ret_5 = leverage_3.shift(1) * lev_returns
    strat_ret_5 = strat_ret_5.fillna(0)
    nav_5 = (1 + strat_ret_5).cumprod()

    arith_mean_5 = strat_ret_5.mean() * 252
    annual_vol_5 = strat_ret_5.std() * np.sqrt(252)
    sharpe_5 = arith_mean_5 / annual_vol_5

    print(f"\n[Method 5: No Cost]")
    print(f"  Arithmetic Mean Return (ann): {arith_mean_5*100:.2f}%")
    print(f"  Annual Volatility: {annual_vol_5*100:.2f}%")
    print(f"  Sharpe: {sharpe_5:.3f}")

    # Method 6: Check with only "in market" returns for Sharpe
    in_market_mask = leverage_3.shift(1) > 0
    in_market_returns = strat_ret_3[in_market_mask]
    sharpe_in_market = (in_market_returns.mean() * 252) / (in_market_returns.std() * np.sqrt(252))

    print(f"\n[Method 6: Sharpe calculated only from 'in market' periods]")
    print(f"  Time in market: {in_market_mask.mean()*100:.1f}%")
    print(f"  Sharpe (in-market only): {sharpe_in_market:.3f}")

    # Method 7: What does annual return ratio give us?
    # If original used: Sharpe-like = (final/initial)^(1/years) / vol
    nav_df = pd.DataFrame({'nav': nav_3.values, 'date': dates.values})
    nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
    yearly_nav = nav_df.groupby('year')['nav'].last()
    annual_returns = yearly_nav.pct_change().dropna()

    sharpe_annual = annual_returns.mean() / annual_returns.std()
    print(f"\n[Method 7: Using Yearly Returns for Sharpe]")
    print(f"  Mean annual return: {annual_returns.mean()*100:.2f}%")
    print(f"  Std of annual returns: {annual_returns.std()*100:.2f}%")
    print(f"  Sharpe (annual): {sharpe_annual:.3f}")

    # Check Original R3 implied volatility
    print("\n" + "=" * 80)
    print("ANALYSIS: What does Original R3 imply?")
    print("=" * 80)

    orig_cagr = 0.3741  # 37.41%
    orig_sharpe = 1.806
    implied_vol = orig_cagr / orig_sharpe

    print(f"\nOriginal R3 DD(-18/92)+VT_E(25%,S10):")
    print(f"  CAGR: {orig_cagr*100:.2f}%")
    print(f"  Sharpe: {orig_sharpe:.3f}")
    print(f"  Implied annual vol (if Sharpe = CAGR/vol): {implied_vol*100:.2f}%")

    print(f"\nOur Results (max_lev=3.0):")
    print(f"  CAGR: {cagr_3*100:.2f}%")
    print(f"  Annual Vol: {annual_vol_3*100:.2f}%")
    print(f"  Sharpe: {sharpe_3:.3f}")

    print(f"\nConclusion:")
    print(f"  - Original R3 implied ~20% annual volatility")
    print(f"  - Our implementation has ~{annual_vol_3*100:.0f}% annual volatility")
    print(f"  - The difference suggests original may have used different calculation method")
    print(f"  - Most likely: Original used CAGR instead of arithmetic mean for Sharpe")

    # Final verification
    print(f"\n  If we use CAGR for Sharpe: {cagr_3/annual_vol_3:.3f}")
    print(f"  If we use arithmetic mean: {arith_mean_3/annual_vol_3:.3f}")

if __name__ == "__main__":
    main()
