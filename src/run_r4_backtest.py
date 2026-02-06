"""
Round 4 Backtest Execution - All Strategies
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import *

def run_all_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Run all R4 strategies and collect results"""

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']
    high = df['High']
    low = df['Low']

    results = []

    def add_result(lev, pos, name, category):
        nav, strat_ret = run_backtest(close, lev)
        metrics = calc_metrics(nav, strat_ret, pos, dates)
        metrics['Strategy'] = name
        metrics['Category'] = category
        results.append(metrics)

    # ==========================================================================
    # Baselines
    # ==========================================================================
    print("Running Baselines...")

    # BH 3x
    lev, pos = strategy_baseline_bh3x(close, returns)
    add_result(lev, pos, 'BH 3x', 'Baseline')

    # DD-18/92 only
    lev, pos = strategy_baseline_dd_only(close, returns, 0.82, 0.92)
    add_result(lev, pos, 'DD(-18/92) only', 'Baseline')

    # DD-18/92 + VT(25%, S10) - R3 best baseline
    lev, pos = strategy_baseline_dd_vt(close, returns, 0.82, 0.92, 0.25, 10)
    add_result(lev, pos, 'DD(-18/92)+VT_E(25%,S10) [R3]', 'Baseline')

    # DD-15/90 + VT(25%, S10) - R3 top
    lev, pos = strategy_baseline_dd_vt(close, returns, 0.85, 0.90, 0.25, 10)
    add_result(lev, pos, 'DD(-15/90)+VT_E(25%,S10) [R3]', 'Baseline')

    # Dual DD + VT
    dd_signal = calc_dual_dd_signal(close)
    ewma_vol = calc_ewma_vol(returns, 10)
    vt_lev = calc_vt_leverage(ewma_vol, 0.25)
    lev = dd_signal * vt_lev
    add_result(lev, dd_signal, 'DualDD(100/200)+VT_E(25%) [R3]', 'Baseline')

    # ==========================================================================
    # Category B: Vol-of-Vol
    # ==========================================================================
    print("Running Category B: Vol-of-Vol...")

    # B1-1
    lev, pos = strategy_dd_vt_vov(close, returns, 0.82, 0.92, 0.25, 10, 1.5, 2.0)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+VoV(1.5x,cap2)', 'B-VoV')

    # B1-2
    lev, pos = strategy_dd_vt_vov(close, returns, 0.82, 0.92, 0.25, 10, 2.0, 1.5)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+VoV(2.0x,cap1.5)', 'B-VoV')

    # B1-3
    lev, pos = strategy_dd_vt_vov(close, returns, 0.82, 0.92, 0.30, 10, 1.5, 2.0)
    add_result(lev, pos, 'DD(-18/92)+VT(30%)+VoV(1.5x,cap2)', 'B-VoV')

    # B2-4: Vol Spike Reduction
    lev, pos = strategy_dd_vt_volspike(close, returns, 0.82, 0.92, 0.25, 10, 1.5)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+VolSpike(1.5x)', 'B-VoV')

    # B2-5
    lev, pos = strategy_dd_vt_volspike(close, returns, 0.85, 0.90, 0.25, 10, 1.5)
    add_result(lev, pos, 'DD(-15/90)+VT(25%)+VolSpike(1.5x)', 'B-VoV')

    # ==========================================================================
    # Category D: Adaptive Reentry
    # ==========================================================================
    print("Running Category D: Adaptive Reentry...")

    # D1-1
    lev, pos = strategy_dd_adaptive_reentry(close, returns, 0.82, 0.25, 10, 0.88, 0.95)
    add_result(lev, pos, 'DD(-18/Adap)+VT(25%) Re:88-95', 'D-AdaptRe')

    # D1-2
    lev, pos = strategy_dd_adaptive_reentry(close, returns, 0.85, 0.25, 10, 0.88, 0.95)
    add_result(lev, pos, 'DD(-15/Adap)+VT(25%) Re:88-95', 'D-AdaptRe')

    # D1-3
    lev, pos = strategy_dd_adaptive_reentry(close, returns, 0.82, 0.30, 10, 0.90, 0.96)
    add_result(lev, pos, 'DD(-18/Adap)+VT(30%) Re:90-96', 'D-AdaptRe')

    # D2-1: Trend Confirm Reentry
    lev, pos = strategy_dd_trend_confirm_reentry(close, returns, 0.82, 0.92, 0.25, 10, 3)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+TrendConfirm(3d)', 'D-AdaptRe')

    # D2-2
    lev, pos = strategy_dd_trend_confirm_reentry(close, returns, 0.82, 0.92, 0.25, 10, 5)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+TrendConfirm(5d)', 'D-AdaptRe')

    # D3-1: Staged Reentry
    lev, pos = strategy_dd_staged_reentry(close, returns, 0.82, 0.25, 10, 0.88, 0.92)
    add_result(lev, pos, 'DD(-18/Staged)+VT(25%) 88-92', 'D-AdaptRe')

    # D3-2
    lev, pos = strategy_dd_staged_reentry(close, returns, 0.82, 0.25, 10, 0.85, 0.93)
    add_result(lev, pos, 'DD(-18/Staged)+VT(25%) 85-93', 'D-AdaptRe')

    # ==========================================================================
    # Category E: Triple DD
    # ==========================================================================
    print("Running Category E: Triple DD...")

    # E1-1
    lev, pos = strategy_triple_dd_vt(close, returns, 50, 100, 200, 0.90, 0.87, 0.85, 0.92, 0.25, 10)
    add_result(lev, pos, 'TripleDD(50/100/200,90/87/85)+VT(25%)', 'E-TripleDD')

    # E1-2
    lev, pos = strategy_triple_dd_vt(close, returns, 50, 100, 200, 0.88, 0.85, 0.82, 0.92, 0.25, 10)
    add_result(lev, pos, 'TripleDD(50/100/200,88/85/82)+VT(25%)', 'E-TripleDD')

    # E1-3
    lev, pos = strategy_triple_dd_vt(close, returns, 50, 100, 200, 0.90, 0.87, 0.85, 0.92, 0.30, 10)
    add_result(lev, pos, 'TripleDD(50/100/200,90/87/85)+VT(30%)', 'E-TripleDD')

    # E1-4
    lev, pos = strategy_triple_dd_vt(close, returns, 40, 120, 200, 0.92, 0.88, 0.85, 0.93, 0.25, 10)
    add_result(lev, pos, 'TripleDD(40/120/200,92/88/85)+VT(25%)', 'E-TripleDD')

    # E1-5
    lev, pos = strategy_triple_dd_vt(close, returns, 50, 150, 250, 0.90, 0.86, 0.83, 0.92, 0.25, 10)
    add_result(lev, pos, 'TripleDD(50/150/250,90/86/83)+VT(25%)', 'E-TripleDD')

    # ==========================================================================
    # Category F: Risk-Adjusted Momentum
    # ==========================================================================
    print("Running Category F: Risk-Adjusted Momentum...")

    # F1-1
    lev, pos = strategy_dd_rolling_sharpe(close, returns, 0.82, 0.92, 0.25, 10, 60, 0.0)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+RollSharpe(60d,>0)', 'F-RiskAdj')

    # F1-2
    lev, pos = strategy_dd_rolling_sharpe(close, returns, 0.82, 0.92, 0.25, 10, 120, 0.0)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+RollSharpe(120d,>0)', 'F-RiskAdj')

    # F1-3
    lev, pos = strategy_dd_rolling_sharpe(close, returns, 0.82, 0.92, 0.25, 10, 60, 0.5)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+RollSharpe(60d,>0.5)', 'F-RiskAdj')

    # F2-1
    lev, pos = strategy_dd_sharpe_based_lev(close, returns, 0.82, 0.92, 60, 1.5, 3.0)
    add_result(lev, pos, 'DD(-18/92)+SharpeLev(60d,1.5x)', 'F-RiskAdj')

    # F2-2
    lev, pos = strategy_dd_sharpe_based_lev(close, returns, 0.82, 0.92, 60, 2.0, 3.0)
    add_result(lev, pos, 'DD(-18/92)+SharpeLev(60d,2.0x)', 'F-RiskAdj')

    # F2-3
    lev, pos = strategy_dd_sharpe_based_lev(close, returns, 0.85, 0.90, 60, 1.5, 3.0)
    add_result(lev, pos, 'DD(-15/90)+SharpeLev(60d,1.5x)', 'F-RiskAdj')

    # ==========================================================================
    # Category H: Advanced Ensemble
    # ==========================================================================
    print("Running Category H: Advanced Ensemble...")

    # H1-1
    lev, pos = strategy_weighted_ensemble(close, returns, [0.5, 0.3, 0.2], 0.25, 10)
    add_result(lev, pos, 'WtdEns(DD18:0.5,DD15:0.3,Dual:0.2)+VT', 'H-Ensemble')

    # H1-2
    lev, pos = strategy_weighted_ensemble(close, returns, [0.4, 0.4, 0.2], 0.25, 10)
    add_result(lev, pos, 'WtdEns(DD18:0.4,DD15:0.4,Dual:0.2)+VT', 'H-Ensemble')

    # H3-1: Voting 2/3
    lev, pos = strategy_voting_ensemble(close, returns, 2, 0.25, 10)
    add_result(lev, pos, 'VoteEns(DD18,DD15,Dual,2/3)+VT(25%)', 'H-Ensemble')

    # H3-2: Voting 3/3
    lev, pos = strategy_voting_ensemble(close, returns, 3, 0.25, 10)
    add_result(lev, pos, 'VoteEns(DD18,DD15,Dual,3/3)+VT(25%)', 'H-Ensemble')

    # H4-1: Dynamic Ensemble
    lev, pos = strategy_dynamic_ensemble(close, returns, 60, 0.25, 10)
    add_result(lev, pos, 'DynEns(60d)+VT(25%)', 'H-Ensemble')

    # H4-2
    lev, pos = strategy_dynamic_ensemble(close, returns, 120, 0.25, 10)
    add_result(lev, pos, 'DynEns(120d)+VT(25%)', 'H-Ensemble')

    # ==========================================================================
    # Category A: Trend Strength Filter
    # ==========================================================================
    print("Running Category A: Trend Strength...")

    # A1-1
    lev, pos = strategy_dd_divergence_filter(close, returns, 0.82, 0.92, 0.25, 10, 200, 0.05)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+Div(MA200,5%)', 'A-Trend')

    # A1-2
    lev, pos = strategy_dd_divergence_filter(close, returns, 0.82, 0.92, 0.25, 10, 200, 0.03)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+Div(MA200,3%)', 'A-Trend')

    # A1-3
    lev, pos = strategy_dd_divergence_filter(close, returns, 0.82, 0.92, 0.25, 10, 200, 0.00)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+Div(MA200,0%)', 'A-Trend')

    # A3-1: Momentum Score
    lev, pos = strategy_dd_momentum_score(close, returns, 0.82, 0.92, 0.25, 10, 120, 0.0)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+Mom(120d,>0)', 'A-Trend')

    # A3-2
    lev, pos = strategy_dd_momentum_score(close, returns, 0.82, 0.92, 0.25, 10, 200, 0.0)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+Mom(200d,>0)', 'A-Trend')

    # A3-3
    lev, pos = strategy_dd_momentum_score(close, returns, 0.82, 0.92, 0.25, 10, 120, -0.05)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+Mom(120d,>-5%)', 'A-Trend')

    # ==========================================================================
    # Category C: ATR/Bollinger
    # ==========================================================================
    print("Running Category C: ATR/Bollinger...")

    # C1-1
    lev, pos = strategy_atr_adaptive_dd(close, returns, high, low, 0.82, 0.3, 0.92, 0.25, 10)
    add_result(lev, pos, 'ATR-AdaptDD(base-18,sens0.3)+VT(25%)', 'C-ATR/BB')

    # C1-2
    lev, pos = strategy_atr_adaptive_dd(close, returns, high, low, 0.85, 0.5, 0.92, 0.25, 10)
    add_result(lev, pos, 'ATR-AdaptDD(base-15,sens0.5)+VT(25%)', 'C-ATR/BB')

    # C1-3
    lev, pos = strategy_atr_adaptive_dd(close, returns, high, low, 0.82, 0.3, 0.92, 0.30, 10)
    add_result(lev, pos, 'ATR-AdaptDD(base-18,sens0.3)+VT(30%)', 'C-ATR/BB')

    # C2-1: BB Exit
    lev, pos = strategy_bb_exit(close, returns, 0.82, 0.92, 0.25, 10, 200, 2.0)
    add_result(lev, pos, 'DD(-18/92)+BB(200,-2σ)+VT(25%)', 'C-ATR/BB')

    # C2-2
    lev, pos = strategy_bb_exit(close, returns, 0.82, 0.92, 0.25, 10, 200, 3.0)
    add_result(lev, pos, 'DD(-18/92)+BB(200,-3σ)+VT(25%)', 'C-ATR/BB')

    # ==========================================================================
    # Category I: Asymmetric
    # ==========================================================================
    print("Running Category I: Asymmetric...")

    # I1-1
    lev, pos = strategy_asymmetric_dd(close, returns, 0.80, 0.85, 0.92, 0.25, 10)
    add_result(lev, pos, 'AsymDD(up-20,dn-15)+VT(25%)', 'I-Asymm')

    # I1-2
    lev, pos = strategy_asymmetric_dd(close, returns, 0.78, 0.88, 0.92, 0.25, 10)
    add_result(lev, pos, 'AsymDD(up-22,dn-12)+VT(25%)', 'I-Asymm')

    # I2-1
    lev, pos = strategy_asymmetric_leverage(close, returns, 0.82, 0.92, 0.30, 0.20, 10)
    add_result(lev, pos, 'DD(-18/92)+AsymLev(up30,dn20)', 'I-Asymm')

    # I2-2
    lev, pos = strategy_asymmetric_leverage(close, returns, 0.82, 0.92, 0.35, 0.20, 10)
    add_result(lev, pos, 'DD(-18/92)+AsymLev(up35,dn20)', 'I-Asymm')

    # I2-3
    lev, pos = strategy_asymmetric_leverage(close, returns, 0.85, 0.90, 0.30, 0.20, 10)
    add_result(lev, pos, 'DD(-15/90)+AsymLev(up30,dn20)', 'I-Asymm')

    # ==========================================================================
    # Category G: Seasonality
    # ==========================================================================
    print("Running Category G: Seasonality...")

    # G1-1
    lev, pos = strategy_dd_quarterly_filter(close, returns, dates, 0.82, 0.92, 0.25, 10, [7,8,9], 0.8)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+Q3weak(0.8x)', 'G-Season')

    # G1-2
    lev, pos = strategy_dd_quarterly_filter(close, returns, dates, 0.82, 0.92, 0.25, 10, [7,8,9,10], 0.8)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+Q3Q4weak(0.8x)', 'G-Season')

    # G1-3
    lev, pos = strategy_dd_quarterly_filter(close, returns, dates, 0.82, 0.92, 0.25, 10, [9,10], 0.7)
    add_result(lev, pos, 'DD(-18/92)+VT(25%)+SepOct(0.7x)', 'G-Season')

    # ==========================================================================
    # Additional Hybrid Strategies
    # ==========================================================================
    print("Running Hybrid Strategies...")

    # Hybrid: Triple DD + VoV
    lev_triple, pos_triple = strategy_triple_dd_vt(close, returns, 50, 100, 200, 0.90, 0.87, 0.85, 0.92, 0.25, 10)
    vov = calc_vol_of_vol(returns)
    vov_median = vov.rolling(500, min_periods=100).median()
    high_vov = vov > vov_median * 1.5
    lev_triple[high_vov] = lev_triple[high_vov].clip(upper=2.0)
    add_result(lev_triple, pos_triple, 'TripleDD+VT(25%)+VoV(cap2)', 'Hybrid')

    # Hybrid: Adaptive Re + VoV
    lev, pos = strategy_dd_adaptive_reentry(close, returns, 0.82, 0.25, 10, 0.88, 0.95)
    vov = calc_vol_of_vol(returns)
    vov_median = vov.rolling(500, min_periods=100).median()
    high_vov = vov > vov_median * 1.5
    lev[high_vov] = lev[high_vov].clip(upper=2.0)
    add_result(lev, pos, 'DD(-18/AdapRe)+VT(25%)+VoV', 'Hybrid')

    # Hybrid: Ensemble + Adaptive Re
    lev, pos = strategy_voting_ensemble(close, returns, 2, 0.25, 10)
    add_result(lev, pos, 'VoteEns+VT(25%)+AdapRe', 'Hybrid')

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Reorder columns
    cols = ['Category', 'Strategy', 'CAGR', 'Worst5Y', 'Sharpe', 'Sortino', 'Calmar', 'MaxDD', 'Trades', 'WinRate']
    results_df = results_df[cols]

    return results_df


def main():
    print("=" * 80)
    print("Round 4 Backtest - 3x Leveraged NASDAQ Strategies")
    print("=" * 80)

    # Load data
    data_path = r"C:\Users\user\Desktop\nasdaq_backtest\NASDAQ_Dairy_since1973.csv"
    df = load_data(data_path)
    print(f"Data loaded: {len(df)} rows, {df['Date'].min()} to {df['Date'].max()}")

    # Set index for resampling
    df_indexed = df.set_index('Date')

    # Run all strategies
    results = run_all_strategies(df)

    # Filter by trade count <= 100
    results_filtered = results[results['Trades'] <= 100].copy()

    # Sort by Sharpe
    results_sorted = results_filtered.sort_values('Sharpe', ascending=False)

    # Format percentages
    for col in ['CAGR', 'Worst5Y', 'MaxDD', 'WinRate']:
        results_sorted[col] = results_sorted[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")

    for col in ['Sharpe', 'Sortino', 'Calmar']:
        results_sorted[col] = results_sorted[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    results_sorted['Trades'] = results_sorted['Trades'].astype(int)

    # Save results
    output_path = r"C:\Users\user\Desktop\nasdaq_backtest\R4_results.csv"
    results_sorted.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print top 20
    print("\n" + "=" * 80)
    print("TOP 20 STRATEGIES (Trades <= 100, Sorted by Sharpe)")
    print("=" * 80)
    print(results_sorted.head(20).to_string(index=False))

    # Print category summary
    print("\n" + "=" * 80)
    print("CATEGORY BEST PERFORMERS")
    print("=" * 80)

    for cat in results_sorted['Category'].unique():
        cat_best = results_sorted[results_sorted['Category'] == cat].head(1)
        if len(cat_best) > 0:
            print(f"\n{cat}:")
            print(cat_best[['Strategy', 'CAGR', 'Worst5Y', 'Sharpe', 'MaxDD', 'Trades']].to_string(index=False))

    return results_sorted


if __name__ == "__main__":
    results = main()
