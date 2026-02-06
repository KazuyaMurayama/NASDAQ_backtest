"""
Test hypothesis: Original R3 applied VT leverage to UNLEVERED returns
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import load_data, calc_dd_signal

def main():
    print("=" * 80)
    print("HYPOTHESIS TEST: Original applied VT to unlevered returns")
    print("=" * 80)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    # DD Signal
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # EWMA Vol (Span=10)
    ewma_vol = returns.ewm(span=10).std() * np.sqrt(252)

    # VT Leverage (0 to 3.0)
    target_vol = 0.25
    vt_lev = (target_vol / ewma_vol).clip(0, 3.0).fillna(1.0)
    leverage = dd_signal * vt_lev

    daily_cost = 0.009 / 252
    years = len(returns) / 252

    results = []

    # ===== Method A: Our current (leverage applies to 3x returns) =====
    # Effective leverage = vt_lev (0-3) * 3 = 0-9x
    strat_ret_a = leverage.shift(1) * (returns * 3 - daily_cost)
    strat_ret_a = strat_ret_a.fillna(0)
    nav_a = (1 + strat_ret_a).cumprod()
    cagr_a = (nav_a.iloc[-1] ** (1/years)) - 1
    vol_a = strat_ret_a.std() * np.sqrt(252)
    sharpe_a = (strat_ret_a.mean() * 252) / vol_a

    results.append({
        'Method': 'A: leverage * (3x_ret - cost)',
        'Interpretation': 'VT(0-3) * base(3x) = 0-9x effective',
        'CAGR': cagr_a,
        'Vol': vol_a,
        'Sharpe': sharpe_a,
        'AvgLev': leverage.mean() * 3
    })

    # ===== Method B: VT leverage IS the effective leverage (no additional 3x) =====
    # Effective leverage = vt_lev (0-3) directly
    strat_ret_b = leverage.shift(1) * (returns - daily_cost / 3)
    strat_ret_b = strat_ret_b.fillna(0)
    nav_b = (1 + strat_ret_b).cumprod()
    cagr_b = (nav_b.iloc[-1] ** (1/years)) - 1
    vol_b = strat_ret_b.std() * np.sqrt(252)
    sharpe_b = (strat_ret_b.mean() * 252) / vol_b

    results.append({
        'Method': 'B: leverage * (1x_ret - cost/3)',
        'Interpretation': 'VT(0-3) = 0-3x effective (no 3x base)',
        'CAGR': cagr_b,
        'Vol': vol_b,
        'Sharpe': sharpe_b,
        'AvgLev': leverage.mean()
    })

    # ===== Method C: Original formula directly from markdown =====
    # strategy_returns = leverage_series.shift(1) * leveraged_returns - daily_cost
    # where leveraged_returns = returns * 3
    strat_ret_c = leverage.shift(1) * returns * 3 - daily_cost
    strat_ret_c = strat_ret_c.fillna(0)
    nav_c = (1 + strat_ret_c).cumprod()
    cagr_c = (nav_c.iloc[-1] ** (1/years)) - 1
    vol_c = strat_ret_c.std() * np.sqrt(252)
    sharpe_c = (strat_ret_c.mean() * 252) / vol_c

    results.append({
        'Method': 'C: leverage * 3x_ret - cost (exact original)',
        'Interpretation': 'Cost always charged regardless of position',
        'CAGR': cagr_c,
        'Vol': vol_c,
        'Sharpe': sharpe_c,
        'AvgLev': leverage.mean() * 3
    })

    # ===== Method D: What if cost was annual (not daily) in formula? =====
    # Bug: using annual_cost directly instead of daily
    strat_ret_d = leverage.shift(1) * returns * 3 - 0.009  # Wrong: using annual as daily
    strat_ret_d = strat_ret_d.fillna(0)
    nav_d = (1 + strat_ret_d).cumprod()
    # This will give crazy results, just for comparison
    # Skip this, it's obviously wrong

    # ===== Method E: What if they calculated Sharpe using only positive leverage days? =====
    positive_lev = leverage.shift(1) > 0
    strat_ret_e = strat_ret_a[positive_lev]
    vol_e = strat_ret_e.std() * np.sqrt(252)
    sharpe_e = (strat_ret_e.mean() * 252) / vol_e

    results.append({
        'Method': 'E: Same as A but Sharpe from active days only',
        'Interpretation': 'Exclude cash period from Sharpe calc',
        'CAGR': cagr_a,  # Same CAGR
        'Vol': vol_e,
        'Sharpe': sharpe_e,
        'AvgLev': leverage.mean() * 3
    })

    # ===== Print Results =====
    print(f"\nData: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"DD(-18/92)+VT(25%,S10,max_lev=3.0)")
    print(f"Time in market: {(dd_signal > 0).mean()*100:.1f}%\n")

    print("-" * 100)
    print(f"{'Method':<45} {'Eff.Lev':>8} {'CAGR':>10} {'Vol':>10} {'Sharpe':>10}")
    print("-" * 100)

    for r in results:
        print(f"{r['Method']:<45} {r['AvgLev']:>7.2f}x {r['CAGR']*100:>9.2f}% {r['Vol']*100:>9.2f}% {r['Sharpe']:>10.3f}")

    print("-" * 100)

    # Original R3 target
    print(f"\n{'ORIGINAL R3 TARGET':<45} {'?':>8} {'37.41':>9}% {'20.7':>9}% {'1.806':>10}")

    # Check what VT leverage range gives ~20% vol
    print("\n" + "=" * 80)
    print("Checking what max_lev gives ~20% vol and Sharpe ~1.8")
    print("=" * 80)

    for max_lev_test in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        vt_lev_test = (target_vol / ewma_vol).clip(0, max_lev_test).fillna(1.0)
        lev_test = dd_signal * vt_lev_test
        strat_ret_test = lev_test.shift(1) * (returns * 3 - daily_cost)
        strat_ret_test = strat_ret_test.fillna(0)
        nav_test = (1 + strat_ret_test).cumprod()
        cagr_test = (nav_test.iloc[-1] ** (1/years)) - 1
        vol_test = strat_ret_test.std() * np.sqrt(252)
        sharpe_test = (strat_ret_test.mean() * 252) / vol_test if vol_test > 0 else 0

        print(f"  max_lev={max_lev_test:.1f}: CAGR={cagr_test*100:>6.2f}%, Vol={vol_test*100:>5.2f}%, Sharpe={sharpe_test:.3f}, EffLev={lev_test.mean()*3:.2f}x")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The original R3 results (Sharpe 1.8+) cannot be replicated with standard methods.

Possible explanations:
1. Bug in original Sharpe calculation
2. Different data set or period
3. Original used a completely different methodology
4. Survivorship bias or data snooping in original

Our implementation (Method A with max_lev=3.0) gives:
- Higher CAGR (51% vs 37%) but higher volatility (73% vs 21%)
- This results in Sharpe ~0.94, NOT 1.8

For practical purposes, our R4 results (Sharpe ~0.86-0.90) are more realistic.
""")

if __name__ == "__main__":
    main()
