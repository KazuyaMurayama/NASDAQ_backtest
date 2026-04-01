"""
Portfolio Diversification Backtest
===================================
NASDAQ 3x (A2 strategy) + Gold + Long-term Treasury Bond

Data sources:
- NASDAQ: Extended CSV (1974-2026, daily)
- Gold: GitHub datasets (1833+, monthly) + yfinance GC=F (2000+, daily)
- Treasury: yfinance ^TNX (1974+, daily) -> synthetic total return index
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import load_data, calc_dd_signal, calc_metrics, calc_ewma_vol
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    rebalance_threshold, run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult
)
from test_improvements import DATA_PATH, DEFAULT_DELAY, ANNUAL_COST
from test_vix_integration import calc_vix_proxy

DEFAULT_THRESHOLD = 0.20


# =============================================================================
# Data Preparation
# =============================================================================
def prepare_gold_data(nasdaq_dates):
    """Build daily gold price series aligned to NASDAQ dates.
    Uses LBMA daily gold AM fix (1968+) as primary source.
    Falls back to LBMA JSON API if local file not found."""
    import os

    # Try local LBMA daily CSV first
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'data', 'lbma_gold_daily.csv')
    if os.path.exists(local_path):
        gold_df = pd.read_csv(local_path, parse_dates=['Date'])
        gold_df = gold_df.rename(columns={'USD': 'Gold'})
        print(f"  Gold: loaded LBMA daily from {local_path} ({len(gold_df)} rows)")
    else:
        # Download from LBMA API
        import urllib.request, json
        url = 'https://prices.lbma.org.uk/json/gold_am.json'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        items = data if isinstance(data, list) else data.get('items', [])
        rows = []
        for item in items:
            v = item['v']
            if len(v) > 0 and v[0]:
                rows.append({'Date': pd.Timestamp(item['d']), 'Gold': float(v[0])})
        gold_df = pd.DataFrame(rows)
        print(f"  Gold: downloaded LBMA daily ({len(gold_df)} rows)")

    # Align to NASDAQ trading dates
    nasdaq_df = pd.DataFrame({'Date': nasdaq_dates})
    merged = nasdaq_df.merge(gold_df, on='Date', how='left')
    merged['Gold'] = merged['Gold'].ffill().bfill()
    return merged['Gold'].values


def prepare_bond_data(nasdaq_dates):
    """Build synthetic long-term Treasury bond total return index.
    Uses ^TNX (10-year yield) to simulate price changes + coupon."""
    import yfinance as yf

    tnx = yf.Ticker('^TNX')
    tnx_data = tnx.history(start='1974-01-01', end='2026-03-28')
    tnx_data = tnx_data.reset_index()
    tnx_data['Date'] = pd.to_datetime(tnx_data['Date']).dt.tz_localize(None)
    tnx_daily = tnx_data[['Date', 'Close']].rename(columns={'Close': 'Yield_pct'})

    # Align to NASDAQ dates
    nasdaq_df = pd.DataFrame({'Date': nasdaq_dates})
    merged = nasdaq_df.merge(tnx_daily, on='Date', how='left')
    merged['Yield_pct'] = merged['Yield_pct'].ffill().bfill()

    # Synthetic bond total return
    # Price change ≈ -Duration × ΔYield
    # Coupon = Yield / 252 (daily)
    # Using Duration ≈ 7 years for 10-year Treasury
    duration = 7.0
    yields = merged['Yield_pct'].values / 100  # Convert to decimal
    bond_nav = np.ones(len(yields))

    for i in range(1, len(yields)):
        dy = yields[i] - yields[i-1]
        price_return = -duration * dy
        coupon_return = yields[i-1] / 252
        daily_return = price_return + coupon_return
        # Cap extreme returns (data issues)
        daily_return = np.clip(daily_return, -0.05, 0.05)
        bond_nav[i] = bond_nav[i-1] * (1 + daily_return)

    return bond_nav


# =============================================================================
# Portfolio Construction
# =============================================================================
def build_portfolio(nasdaq_nav, gold_prices, bond_nav,
                    w_nasdaq, w_gold, w_bond,
                    rebal_freq=63):
    """Build multi-asset portfolio with periodic rebalancing.

    Args:
        nasdaq_nav: NASDAQ 3x strategy NAV series
        gold_prices: Gold price series
        bond_nav: Bond total return NAV series
        w_nasdaq, w_gold, w_bond: Target weights (sum to 1.0)
        rebal_freq: Rebalance every N trading days (63 = quarterly)
    """
    n = len(nasdaq_nav)

    # Calculate daily returns for each asset
    nasdaq_ret = np.zeros(n)
    gold_ret = np.zeros(n)
    bond_ret = np.zeros(n)

    for i in range(1, n):
        nasdaq_ret[i] = nasdaq_nav[i] / nasdaq_nav[i-1] - 1 if nasdaq_nav[i-1] > 0 else 0
        gold_ret[i] = gold_prices[i] / gold_prices[i-1] - 1 if gold_prices[i-1] > 0 else 0
        bond_ret[i] = bond_nav[i] / bond_nav[i-1] - 1 if bond_nav[i-1] > 0 else 0

    # Portfolio with periodic rebalancing
    portfolio_nav = np.ones(n)
    # Current weights
    cw_n, cw_g, cw_b = w_nasdaq, w_gold, w_bond

    for i in range(1, n):
        # Portfolio return based on current weights
        port_ret = cw_n * nasdaq_ret[i] + cw_g * gold_ret[i] + cw_b * bond_ret[i]
        portfolio_nav[i] = portfolio_nav[i-1] * (1 + port_ret)

        # Update weights based on drift
        total = cw_n * (1 + nasdaq_ret[i]) + cw_g * (1 + gold_ret[i]) + cw_b * (1 + bond_ret[i])
        if total > 0:
            cw_n = cw_n * (1 + nasdaq_ret[i]) / total
            cw_g = cw_g * (1 + gold_ret[i]) / total
            cw_b = cw_b * (1 + bond_ret[i]) / total

        # Rebalance periodically
        if i % rebal_freq == 0:
            cw_n, cw_g, cw_b = w_nasdaq, w_gold, w_bond

    return portfolio_nav


# =============================================================================
# A2 Strategy NAV
# =============================================================================
def get_a2_nav(close, dates):
    """Generate A2 strategy NAV."""
    returns = close.pct_change()
    dd_signal = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    mom_decel = calc_momentum_decel_mult(close, short=60, long=180,
                                          sensitivity=0.3, min_mult=0.5, max_mult=1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.2 * vix_z).clip(0.5, 1.15)

    raw_leverage = dd_signal * vt_lev * slope_mult * mom_decel * vix_mult
    raw_leverage = raw_leverage.clip(0, 1.0).fillna(0)

    lev = rebalance_threshold(raw_leverage, DEFAULT_THRESHOLD)
    nav, strat_ret = run_backtest_realistic(close, lev, DEFAULT_DELAY, ANNUAL_COST)
    return nav.values, strat_ret.values, dd_signal


# =============================================================================
# Metrics for portfolio NAV
# =============================================================================
def calc_portfolio_metrics(nav_array, dates, name):
    """Calculate metrics from a raw NAV array."""
    nav = pd.Series(nav_array, index=dates.index)
    returns = nav.pct_change().fillna(0)

    total_years = len(nav) / 252
    cagr = (nav.iloc[-1] ** (1 / total_years)) - 1
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    # Worst 5Y
    if len(nav) >= 252 * 5:
        nav_5y = nav.shift(252 * 5)
        rolling_5y = (nav / nav_5y) ** (1/5) - 1
        worst_5y = rolling_5y.min()
    else:
        worst_5y = np.nan

    # OOS
    split_idx = dates[dates >= '2021-05-07'].index[0]
    nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
    ret_oos = returns.iloc[split_idx:]
    oos_sharpe = (ret_oos.mean() * 252) / (ret_oos.std() * np.sqrt(252)) if ret_oos.std() > 0 else 0

    # 2022
    mask_2022 = (dates.dt.year == 2022)
    nav_2022 = nav[mask_2022]
    ret_2022 = (nav_2022.iloc[-1] / nav_2022.iloc[0] - 1) * 100 if mask_2022.any() else np.nan

    return {
        'Strategy': name, 'Sharpe': sharpe, 'CAGR': cagr,
        'MaxDD': max_dd, 'Worst5Y': worst_5y,
        'OOS_Sharpe': oos_sharpe, 'Y2022': ret_2022,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    df = load_data(DATA_PATH)
    close = df['Close']
    dates = df['Date']
    print(f"NASDAQ data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} rows)")

    # Get A2 strategy NAV
    print("\nGenerating A2 strategy NAV...")
    a2_nav, a2_ret, dd_signal = get_a2_nav(close, dates)
    print(f"  A2 NAV: start={a2_nav[0]:.2f}, end={a2_nav[-1]:.2f}")

    # Get Gold data
    print("\nFetching Gold data (monthly 1974+ interpolated + daily 2000+)...")
    gold_prices = prepare_gold_data(dates)
    print(f"  Gold: {len(gold_prices)} rows, first={gold_prices[0]:.2f}, last={gold_prices[-1]:.2f}")

    # Get Bond data
    print("\nFetching Treasury bond data (^TNX -> synthetic total return)...")
    bond_nav = prepare_bond_data(dates)
    print(f"  Bond NAV: start={bond_nav[0]:.2f}, end={bond_nav[-1]:.2f}")

    # Sanity check: individual asset returns
    gold_cagr = (gold_prices[-1] / gold_prices[0]) ** (1 / (len(gold_prices)/252)) - 1
    bond_cagr = (bond_nav[-1] / bond_nav[0]) ** (1 / (len(bond_nav)/252)) - 1
    a2_cagr = (a2_nav[-1] / a2_nav[0]) ** (1 / (len(a2_nav)/252)) - 1
    print(f"\n  Individual CAGRs: A2={a2_cagr*100:.1f}%, Gold={gold_cagr*100:.1f}%, Bond={bond_cagr*100:.1f}%")

    # Correlation check
    a2_rets = pd.Series(a2_nav).pct_change().dropna()
    gold_rets = pd.Series(gold_prices).pct_change().dropna()
    bond_rets = pd.Series(bond_nav).pct_change().dropna()
    min_len = min(len(a2_rets), len(gold_rets), len(bond_rets))
    corr_ag = a2_rets.iloc[:min_len].corr(gold_rets.iloc[:min_len])
    corr_ab = a2_rets.iloc[:min_len].corr(bond_rets.iloc[:min_len])
    corr_gb = gold_rets.iloc[:min_len].corr(bond_rets.iloc[:min_len])
    print(f"  Correlations: A2-Gold={corr_ag:.3f}, A2-Bond={corr_ab:.3f}, Gold-Bond={corr_gb:.3f}")

    # =======================================================================
    # Portfolio Tests
    # =======================================================================
    print(f"\n{'=' * 110}")
    print("PORTFOLIO DIVERSIFICATION RESULTS (1974-2026, quarterly rebalance)")
    print("=" * 110)

    portfolios = [
        ("A2 100% (reference)", 1.0, 0.0, 0.0),
        ("80% A2 + 10% Gold + 10% Bond", 0.80, 0.10, 0.10),
        ("70% A2 + 15% Gold + 15% Bond", 0.70, 0.15, 0.15),
        ("60% A2 + 20% Gold + 20% Bond", 0.60, 0.20, 0.20),
        ("80% A2 + 20% Gold", 0.80, 0.20, 0.00),
        ("80% A2 + 20% Bond", 0.80, 0.00, 0.20),
        ("90% A2 + 5% Gold + 5% Bond", 0.90, 0.05, 0.05),
        ("85% A2 + 10% Gold + 5% Bond", 0.85, 0.10, 0.05),
    ]

    results = []
    ref = None
    for name, wn, wg, wb in portfolios:
        port_nav = build_portfolio(a2_nav, gold_prices, bond_nav, wn, wg, wb, rebal_freq=63)
        m = calc_portfolio_metrics(port_nav, dates, name)
        results.append(m)
        if ref is None:
            ref = m

    # Print
    print(f"\n{'Strategy':<38} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} | {'OOS_Sh':>7} {'Y2022':>7}")
    print("-" * 100)
    for r in results:
        flag = ""
        if r != ref:
            better_maxdd = r['MaxDD'] > ref['MaxDD']
            better_w5y = r['Worst5Y'] > ref['Worst5Y']
            better_oos = r['OOS_Sharpe'] > ref['OOS_Sharpe']
            if better_maxdd and better_w5y and better_oos:
                flag = " ★★★"
            elif better_maxdd and better_w5y:
                flag = " ★★"
            elif better_maxdd:
                flag = " ★(MaxDD)"
        print(f"{r['Strategy']:<38} {r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% "
              f"{r['Worst5Y']*100:>6.2f}% | {r['OOS_Sharpe']:>7.4f} {r['Y2022']:>6.1f}%{flag}")

    # Crisis year detail
    print(f"\n{'=' * 110}")
    print("CRISIS YEAR RETURNS (%)")
    print("=" * 110)

    crisis_years = [1974, 1987, 2000, 2008, 2011, 2022]
    print(f"{'Strategy':<38}", end="")
    for y in crisis_years:
        print(f" {y:>7}", end="")
    print()
    print("-" * 80)

    for name, wn, wg, wb in portfolios:
        port_nav = build_portfolio(a2_nav, gold_prices, bond_nav, wn, wg, wb, rebal_freq=63)
        nav_series = pd.Series(port_nav, index=dates.index)
        print(f"{name:<38}", end="")
        for y in crisis_years:
            mask = (dates.dt.year == y)
            if mask.any():
                yr_nav = nav_series[mask]
                yr_ret = (yr_nav.iloc[-1] / yr_nav.iloc[0] - 1) * 100
                print(f" {yr_ret:>6.1f}%", end="")
            else:
                print(f"    N/A", end="")
        print()

    # Save
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.DataFrame(results).to_csv(os.path.join(base_dir, 'portfolio_diversification_results.csv'), index=False)
    print(f"\nSaved to portfolio_diversification_results.csv")


if __name__ == '__main__':
    main()
