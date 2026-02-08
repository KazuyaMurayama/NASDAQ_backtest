"""
Majority-Vote Ensemble Strategy - Phase 2: Ensemble Construction & Testing

Tests:
  A. Subset size comparison (3/5/7/11/13-vote)
  B. Voting threshold comparison (40%-80%)
  C. Homogeneous vs Diverse comparison (H1)
  D. Stepped leverage model (H5)
  E. Rebalance threshold for practical use
  F. Delay sensitivity (H4)
  G. Final comparison with current best

Realistic conditions: 5-day delay, 1.5% annual cost.
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import load_data, calc_metrics
from test_majority_vote_p1 import (
    calc_S1_dd_standard, calc_S2_dd_loose, calc_S3_ma200_regime,
    calc_S4_dual_ma_cross, calc_S5_momentum_120, calc_S6_momentum_250,
    calc_S7_vol_regime, calc_S8_rolling_sharpe, calc_S9_bb_lower,
    calc_S10_vov, calc_S11_ma_slope, calc_S12_seasonal, calc_S13_dd_short,
    run_backtest, count_trades
)
from test_ens2_strategies import strategy_ens2_slope_trendtv
from test_realistic_product import rebalance_threshold

# =============================================================================
# Constants
# =============================================================================
ANNUAL_COST = 0.015
EXEC_DELAY = 5
BASE_LEV = 3.0


# =============================================================================
# Ensemble Functions
# =============================================================================
def majority_vote(signals: dict, threshold: float) -> pd.Series:
    """Binary majority vote.
    threshold: fraction of strategies that must be HOLD (e.g. 0.5 for simple majority)
    Returns 1.0 (HOLD) or 0.0 (CASH)
    """
    vote_sum = sum(signals.values())
    n = len(signals)
    min_votes = np.ceil(n * threshold)
    return (vote_sum >= min_votes).astype(float)


def stepped_leverage(signals: dict) -> pd.Series:
    """Stepped leverage based on vote ratio.
    vote_ratio >= 0.9 -> 1.0  (3x full)
    vote_ratio >= 0.7 -> 0.7  (2.1x)
    vote_ratio >= 0.5 -> 0.4  (1.2x)
    vote_ratio <  0.5 -> 0.0  (CASH)
    """
    vote_sum = sum(signals.values())
    n = len(signals)
    ratio = vote_sum / n

    lev = pd.Series(0.0, index=ratio.index)
    lev[ratio >= 0.5] = 0.4
    lev[ratio >= 0.7] = 0.7
    lev[ratio >= 0.9] = 1.0
    return lev


def run_backtest_lev(close, leverage, delay=EXEC_DELAY, cost=ANNUAL_COST):
    """Backtest with continuous leverage (for stepped model)."""
    returns = close.pct_change()
    lev_returns = returns * BASE_LEV
    daily_cost = cost / 252
    delayed = leverage.shift(delay)
    strat_ret = delayed * (lev_returns - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret


# =============================================================================
# Main
# =============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'NASDAQ_Dairy_since1973.csv')
    df = load_data(data_path)

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']
    n_years = len(df) / 252

    print("=" * 140)
    print("MAJORITY-VOTE ENSEMBLE - Phase 2: Ensemble Construction & Testing")
    print(f"Conditions: {EXEC_DELAY}-day delay, {ANNUAL_COST*100:.1f}% annual cost")
    print("=" * 140)

    # Build all 13 sub-strategies
    all_signals = {
        'S1':  calc_S1_dd_standard(close),
        'S2':  calc_S2_dd_loose(close),
        'S3':  calc_S3_ma200_regime(close),
        'S4':  calc_S4_dual_ma_cross(close),
        'S5':  calc_S5_momentum_120(close),
        'S6':  calc_S6_momentum_250(close),
        'S7':  calc_S7_vol_regime(returns),
        'S8':  calc_S8_rolling_sharpe(returns),
        'S9':  calc_S9_bb_lower(close),
        'S10': calc_S10_vov(returns),
        'S11': calc_S11_ma_slope(close),
        'S12': calc_S12_seasonal(dates),
        'S13': calc_S13_dd_short(close),
    }

    results = []

    def test(lev_or_sig, name, category, is_binary=True, delay=EXEC_DELAY):
        if is_binary:
            nav, strat_ret = run_backtest(close, lev_or_sig, delay=delay)
        else:
            nav, strat_ret = run_backtest_lev(close, lev_or_sig, delay=delay)
        m = calc_metrics(nav, strat_ret, lev_or_sig, dates)
        trades = count_trades(lev_or_sig) if is_binary else (lev_or_sig.diff().abs() > 0.01).sum()
        hold_pct = (lev_or_sig > 0.01).mean() * 100
        m.update({'Strategy': name, 'Category': category,
                  'Trades': trades, 'Trades/Year': trades / n_years, 'HOLD%': hold_pct})
        results.append(m)
        print(f"  {name:<55s} Sharpe={m['Sharpe']:.3f}  CAGR={m['CAGR']*100:>+.1f}%  "
              f"MaxDD={m['MaxDD']*100:.1f}%  W5Y={m['Worst5Y']*100:>+.1f}%  "
              f"Trades={trades}({trades/n_years:.0f}/yr)  HOLD={hold_pct:.0f}%")
        return m

    # =================================================================
    # A. SUBSET SIZE COMPARISON
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST A: SUBSET SIZE COMPARISON (Simple Majority)")
    print("=" * 140)

    subsets = {
        '3v-Diverse(S1,S5,S7)':       {'S1': all_signals['S1'], 'S5': all_signals['S5'], 'S7': all_signals['S7']},
        '3v-AltDiv(S1,S4,S11)':       {'S1': all_signals['S1'], 'S4': all_signals['S4'], 'S11': all_signals['S11']},
        '5v-Core(S1,S4,S5,S7,S11)':   {k: all_signals[k] for k in ['S1','S4','S5','S7','S11']},
        '5v-Alt(S1,S3,S7,S9,S13)':    {k: all_signals[k] for k in ['S1','S3','S7','S9','S13']},
        '7v-Ext(S1,S3-5,S7,S8,S11)':  {k: all_signals[k] for k in ['S1','S3','S4','S5','S7','S8','S11']},
        '7v-Alt(S1,S4-7,S9,S13)':     {k: all_signals[k] for k in ['S1','S4','S5','S6','S7','S9','S13']},
        '9v(S1-3,S4-6,S7-9)':         {k: all_signals[k] for k in ['S1','S2','S3','S4','S5','S6','S7','S8','S9']},
        '11v-Full(S1-S11)':            {k: all_signals[k] for k in ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11']},
        '11v-NoVoV(S1-9,S11,S12)':    {k: all_signals[k] for k in ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S11','S12']},
        '13v-All':                     all_signals,
    }

    for name, sigs in subsets.items():
        sig = majority_vote(sigs, 0.5)
        test(sig, f'MajVote {name}', 'A-Subset')

    # =================================================================
    # B. VOTING THRESHOLD COMPARISON (using 11v-Full)
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST B: VOTING THRESHOLD COMPARISON (11v-Full S1-S11)")
    print("=" * 140)

    sigs_11 = {k: all_signals[k] for k in ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11']}

    for th_pct, th_name in [(0.36, '4/11=36%'), (0.45, '5/11=45%'), (0.50, '6/11=55%'),
                             (0.64, '7/11=64%'), (0.73, '8/11=73%'), (0.82, '9/11=82%')]:
        sig = majority_vote(sigs_11, th_pct)
        test(sig, f'11v Threshold {th_name}', 'B-Threshold')

    # Also test with 7v best subset
    sigs_7 = {k: all_signals[k] for k in ['S1','S3','S4','S5','S7','S8','S11']}
    for th_pct, th_name in [(0.43, '3/7=43%'), (0.50, '4/7=57%'), (0.71, '5/7=71%'), (0.86, '6/7=86%')]:
        sig = majority_vote(sigs_7, th_pct)
        test(sig, f'7v-Ext Threshold {th_name}', 'B-Threshold7')

    # =================================================================
    # C. HOMOGENEOUS VS DIVERSE (H1 test)
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST C: HOMOGENEOUS vs DIVERSE (H1: diversity helps?)")
    print("=" * 140)

    # Homogeneous: 3 DD variants
    homo = {'S1': all_signals['S1'], 'S2': all_signals['S2'], 'S13': all_signals['S13']}
    sig = majority_vote(homo, 0.5)
    test(sig, 'HOMO 3v-DD(S1,S2,S13)', 'C-H1')

    # Diverse: 3 different categories
    div3 = {'S1': all_signals['S1'], 'S5': all_signals['S5'], 'S7': all_signals['S7']}
    sig = majority_vote(div3, 0.5)
    test(sig, 'DIVERSE 3v(S1,S5,S7)', 'C-H1')

    # Homogeneous: 5 price-trend based
    homo5 = {k: all_signals[k] for k in ['S1', 'S2', 'S3', 'S5', 'S13']}
    sig = majority_vote(homo5, 0.5)
    test(sig, 'HOMO 5v-Price(S1,S2,S3,S5,S13)', 'C-H1')

    # Diverse: 5 different categories
    div5 = {k: all_signals[k] for k in ['S1', 'S4', 'S5', 'S7', 'S11']}
    sig = majority_vote(div5, 0.5)
    test(sig, 'DIVERSE 5v(S1,S4,S5,S7,S11)', 'C-H1')

    # =================================================================
    # D. STEPPED LEVERAGE (H5 test)
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST D: STEPPED LEVERAGE vs BINARY")
    print("=" * 140)

    for name, sigs in [('7v-Ext', sigs_7), ('11v-Full', sigs_11), ('13v-All', all_signals)]:
        # Binary majority
        sig_bin = majority_vote(sigs, 0.5)
        test(sig_bin, f'{name} Binary(>50%)', 'D-Binary')

        # Stepped leverage
        lev_step = stepped_leverage(sigs)
        test(lev_step, f'{name} Stepped(4-step)', 'D-Stepped', is_binary=False)

    # =================================================================
    # E. REBALANCE THRESHOLD (practical trade frequency)
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST E: REBALANCE THRESHOLD (practical frequency)")
    print("=" * 140)

    # Take top performers from tests above, apply rebalance threshold
    # Best binary: typically 7v or 11v with supermajority
    best_binary_configs = [
        ('7v-Ext >50%', majority_vote(sigs_7, 0.5)),
        ('7v-Ext >71%', majority_vote(sigs_7, 0.71)),
        ('11v >55%', majority_vote(sigs_11, 0.50)),
        ('11v >64%', majority_vote(sigs_11, 0.64)),
        ('11v >73%', majority_vote(sigs_11, 0.73)),
    ]

    # Best stepped
    best_stepped_configs = [
        ('7v-Ext Stepped', stepped_leverage(sigs_7)),
        ('11v Stepped', stepped_leverage(sigs_11)),
    ]

    print("\n--- Binary with rebalance threshold ---")
    for cfg_name, sig in best_binary_configs:
        # Binary signals don't benefit from threshold (already 0/1)
        # but we can smooth short CASH/HOLD flickers
        test(sig, f'{cfg_name} (raw)', 'E-BinaryRaw')

    print("\n--- Stepped with rebalance threshold ---")
    for cfg_name, lev in best_stepped_configs:
        for th in [0.15, 0.20, 0.25]:
            lev_th = rebalance_threshold(lev, th)
            test(lev_th, f'{cfg_name} Th={th:.0%}', 'E-StepTh', is_binary=False)

    # =================================================================
    # F. DELAY SENSITIVITY (H4 test)
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST F: DELAY SENSITIVITY (H4: ensemble more robust to delay?)")
    print("=" * 140)

    # Compare individual S1 vs 7v ensemble vs 11v ensemble at various delays
    sig_s1 = all_signals['S1']
    sig_7v = majority_vote(sigs_7, 0.5)
    sig_11v = majority_vote(sigs_11, 0.5)
    lev_7v_step = stepped_leverage(sigs_7)

    for delay in [1, 3, 5, 7, 10]:
        print(f"\n  [Delay = {delay} days]")
        test(sig_s1, f'S1 DD-Only delay={delay}d', 'F-Delay', delay=delay)
        test(sig_7v, f'7v-Ext MajVote delay={delay}d', 'F-Delay', delay=delay)
        test(sig_11v, f'11v MajVote delay={delay}d', 'F-Delay', delay=delay)
        test(lev_7v_step, f'7v-Ext Stepped delay={delay}d', 'F-Delay', is_binary=False, delay=delay)

    # =================================================================
    # G. FINAL COMPARISON WITH CURRENT BEST
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST G: FINAL COMPARISON WITH CURRENT BEST STRATEGIES")
    print("=" * 140)

    # Current best: Ens2(S+T) Th20%
    lev_ens2_raw, dd_ens2 = strategy_ens2_slope_trendtv(close, returns, 0.82, 0.92, 20, 5, 1.0)
    lev_ens2 = rebalance_threshold(lev_ens2_raw, 0.20)
    test(lev_ens2, 'Ens2(S+T) Th20% [Current 1st]', 'G-Current', is_binary=False)

    # DD-Only
    test(all_signals['S1'], 'DD-Only [Current 6th]', 'G-Current')

    # Top ensemble configs (select best from above - hardcode after seeing results)
    # We'll include the most promising from each test
    print("\n--- Top Ensemble Candidates ---")
    sig_7v_57 = majority_vote(sigs_7, 0.50)
    sig_7v_71 = majority_vote(sigs_7, 0.71)
    sig_11v_64 = majority_vote(sigs_11, 0.64)
    sig_11v_73 = majority_vote(sigs_11, 0.73)
    lev_7v_s = stepped_leverage(sigs_7)
    lev_11v_s = stepped_leverage(sigs_11)
    lev_7v_s_th20 = rebalance_threshold(lev_7v_s, 0.20)
    lev_11v_s_th20 = rebalance_threshold(lev_11v_s, 0.20)

    test(sig_7v_57, 'MajVote 7v >50%', 'G-Ensemble')
    test(sig_7v_71, 'MajVote 7v >71%', 'G-Ensemble')
    test(sig_11v_64, 'MajVote 11v >64%', 'G-Ensemble')
    test(sig_11v_73, 'MajVote 11v >73%', 'G-Ensemble')
    test(lev_7v_s_th20, 'Stepped 7v Th=20%', 'G-Ensemble', is_binary=False)
    test(lev_11v_s_th20, 'Stepped 11v Th=20%', 'G-Ensemble', is_binary=False)

    # =================================================================
    # SUMMARY TABLE
    # =================================================================
    print("\n" + "=" * 140)
    print("FULL RESULTS RANKED BY SHARPE")
    print("=" * 140)

    results_df = pd.DataFrame(results)
    cols = ['Category', 'Strategy', 'Sharpe', 'CAGR', 'MaxDD', 'Worst5Y', 'Trades', 'Trades/Year', 'HOLD%']
    results_df = results_df[cols].sort_values('Sharpe', ascending=False)

    display = results_df.copy()
    for c in ['CAGR', 'MaxDD', 'Worst5Y']:
        display[c] = display[c].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "N/A")
    display['Sharpe'] = display['Sharpe'].apply(lambda x: f"{x:.3f}")
    display['Trades'] = display['Trades'].astype(int)
    display['Trades/Year'] = display['Trades/Year'].apply(lambda x: f"{x:.0f}")
    display['HOLD%'] = display['HOLD%'].apply(lambda x: f"{x:.0f}%")

    print(display.head(40).to_string(index=False))

    # Save
    out = os.path.join(script_dir, '..', 'majority_vote_p2_results.csv')
    results_df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
