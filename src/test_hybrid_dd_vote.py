"""
Direction A: Hybrid Strategy - MajVote DD + Ens2 VT Layer

Replace Ens2's DD Control with majority-vote signal from Phase 1,
while keeping the continuous VT/SlopeMult/TrendTV layers intact.

Structure:
  Original Ens2:  DD_signal(0/1) × VT × SlopeMult
  Hybrid:         MajVote(0/1)   × VT × SlopeMult

Also test:
  - Union:     min(DD, MajVote) → CASH if EITHER says CASH (conservative)
  - Intersect: max(DD, MajVote) but 0 if both 0 → CASH only if BOTH say CASH (aggressive)
  - DD + MajVote override: DD as base, but MajVote can force early exit

Realistic conditions: 5-day delay, 1.5% annual cost.
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import load_data, calc_metrics, calc_dd_signal
from test_majority_vote_p1 import (
    calc_S1_dd_standard, calc_S2_dd_loose, calc_S3_ma200_regime,
    calc_S4_dual_ma_cross, calc_S5_momentum_120, calc_S6_momentum_250,
    calc_S7_vol_regime, calc_S8_rolling_sharpe, calc_S9_bb_lower,
    calc_S10_vov, calc_S11_ma_slope, calc_S12_seasonal, calc_S13_dd_short,
    count_trades
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_realistic_product import rebalance_threshold

# =============================================================================
ANNUAL_COST = 0.015
EXEC_DELAY = 5
BASE_LEV = 3.0


# =============================================================================
# VT Layer components (extracted from Ens2 strategies)
# =============================================================================
def build_ens2_vt_layer(close, returns, variant='asym_slope', max_lev=1.0):
    """Build only the VT×Mult continuous layer (without DD).
    Returns leverage in [0, max_lev].
    """
    if variant == 'asym_slope':
        asym_vol = calc_asym_ewma_vol(returns, span_up=20, span_dn=5)
        vt_lev = (0.25 / asym_vol).clip(0, max_lev)
        slope_mult = calc_slope_multiplier(close)
        return (vt_lev * slope_mult).clip(0, max_lev).fillna(0)
    elif variant == 'slope_trendtv':
        asym_vol = calc_asym_ewma_vol(returns, span_up=20, span_dn=5)
        trend_tv = calc_trend_target_vol(close)
        vt_lev = (trend_tv / asym_vol).clip(0, max_lev)
        slope_mult = calc_slope_multiplier(close)
        return (vt_lev * slope_mult).clip(0, max_lev).fillna(0)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# =============================================================================
# Combination modes for DD × MajVote
# =============================================================================
def combine_replace(dd_signal, majvote_signal):
    """Replace DD with MajVote entirely."""
    return majvote_signal

def combine_union(dd_signal, majvote_signal):
    """CASH if EITHER says CASH (conservative). HOLD only if BOTH say HOLD."""
    return (dd_signal * majvote_signal)  # Both must be 1

def combine_intersect(dd_signal, majvote_signal):
    """CASH only if BOTH say CASH (aggressive). HOLD if EITHER says HOLD."""
    return ((dd_signal + majvote_signal) > 0.5).astype(float)

def combine_dd_plus_override(dd_signal, majvote_signal, override_threshold=0.3):
    """DD as base, but MajVote can force CASH if vote ratio is very low.
    Uses majvote_signal which is already thresholded, but here we use
    a stricter threshold to override DD's HOLD.
    """
    # DD says CASH → always CASH
    # DD says HOLD but MajVote says CASH → CASH (MajVote overrides)
    # Both say HOLD → HOLD
    return combine_union(dd_signal, majvote_signal)


# =============================================================================
# Backtest
# =============================================================================
def run_backtest(close, leverage, delay=EXEC_DELAY, cost=ANNUAL_COST):
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
    print("DIRECTION A: HYBRID STRATEGY - MajVote DD Layer + Ens2 VT Layer")
    print(f"Conditions: {EXEC_DELAY}-day delay, {ANNUAL_COST*100:.1f}% annual cost")
    print("=" * 140)

    # Build all sub-strategy signals
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

    # Build VT layers
    vt_asym_slope = build_ens2_vt_layer(close, returns, 'asym_slope')
    vt_slope_trendtv = build_ens2_vt_layer(close, returns, 'slope_trendtv')

    # DD signal
    dd_signal = calc_dd_signal(close, 0.82, 0.92)

    # Majority vote signals with various configs
    def majvote(keys, threshold):
        sigs = {k: all_signals[k] for k in keys}
        vote_sum = sum(sigs.values())
        n = len(sigs)
        min_votes = np.ceil(n * threshold)
        return (vote_sum >= min_votes).astype(float)

    results = []

    def test(lev_raw, name, category, th_rebal=None):
        if th_rebal is not None:
            lev = rebalance_threshold(lev_raw, th_rebal)
        else:
            lev = lev_raw
        nav, strat_ret = run_backtest(close, lev)
        m = calc_metrics(nav, strat_ret, lev, dates)
        trades = count_trades(lev) if lev.nunique() <= 2 else (lev.diff().abs() > 0.01).sum()
        m.update({'Strategy': name, 'Category': category,
                  'Trades': trades, 'Trades/Year': trades / n_years})
        results.append(m)
        print(f"  {name:<65s} Sharpe={m['Sharpe']:.3f}  CAGR={m['CAGR']*100:>+.1f}%  "
              f"MaxDD={m['MaxDD']*100:.1f}%  W5Y={m['Worst5Y']*100:>+.1f}%  "
              f"Tr={trades}({trades/n_years:.0f}/yr)")
        return m

    # =================================================================
    # BASELINE: Original Ens2 strategies
    # =================================================================
    print("\n" + "=" * 140)
    print("BASELINE: Original Ens2 Strategies (DD + VT)")
    print("=" * 140)

    lev_orig_as = dd_signal * vt_asym_slope
    lev_orig_st = dd_signal * vt_slope_trendtv

    test(lev_orig_as, 'Ens2(Asym+Slope) Daily', 'Baseline')
    test(lev_orig_as, 'Ens2(Asym+Slope) Th=20%', 'Baseline', th_rebal=0.20)
    test(lev_orig_st, 'Ens2(Slope+TrendTV) Daily', 'Baseline')
    test(lev_orig_st, 'Ens2(Slope+TrendTV) Th=20%', 'Baseline', th_rebal=0.20)

    # =================================================================
    # TEST 1: REPLACE DD with MajVote
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST 1: REPLACE DD with MajVote (MajVote × VT × SlopeMult)")
    print("=" * 140)

    # Key subsets from Phase 2 insights
    vote_configs = [
        ('3v(S1,S5,S7)',    ['S1','S5','S7']),
        ('5v(S1,S3,S7,S9,S13)', ['S1','S3','S7','S9','S13']),
        ('7v(S1,S3-5,S7,S8,S11)', ['S1','S3','S4','S5','S7','S8','S11']),
        ('11v(S1-S11)',     ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11']),
        ('13v(All)',        list(all_signals.keys())),
    ]

    thresholds = [0.50, 0.67, 0.82]

    for vt_name, vt_layer in [('A+S', vt_asym_slope), ('S+T', vt_slope_trendtv)]:
        print(f"\n--- VT Layer: {vt_name} ---")
        for vc_name, vc_keys in vote_configs:
            for th in thresholds:
                mv = majvote(vc_keys, th)
                lev = mv * vt_layer
                n_strats = len(vc_keys)
                th_label = f"{int(np.ceil(n_strats * th))}/{n_strats}={th:.0%}"
                test(lev, f'Replace {vc_name} th={th_label} × {vt_name}', 'T1-Replace', th_rebal=0.20)

    # =================================================================
    # TEST 2: UNION (CASH if EITHER DD or MajVote says CASH)
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST 2: UNION - CASH if EITHER DD or MajVote (conservative)")
    print("=" * 140)

    for vt_name, vt_layer in [('A+S', vt_asym_slope), ('S+T', vt_slope_trendtv)]:
        print(f"\n--- VT Layer: {vt_name} ---")
        for vc_name, vc_keys in vote_configs:
            for th in [0.50, 0.67]:
                mv = majvote(vc_keys, th)
                union_sig = combine_union(dd_signal, mv)
                lev = union_sig * vt_layer
                n_strats = len(vc_keys)
                th_label = f"{int(np.ceil(n_strats * th))}/{n_strats}"
                test(lev, f'Union DD∩MV({vc_name} {th_label}) × {vt_name}', 'T2-Union', th_rebal=0.20)

    # =================================================================
    # TEST 3: INTERSECT (CASH only if BOTH say CASH)
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST 3: INTERSECT - CASH only if BOTH DD and MajVote (aggressive)")
    print("=" * 140)

    for vt_name, vt_layer in [('A+S', vt_asym_slope), ('S+T', vt_slope_trendtv)]:
        print(f"\n--- VT Layer: {vt_name} ---")
        for vc_name, vc_keys in vote_configs[:3]:  # Focus on smaller subsets
            for th in [0.50, 0.67]:
                mv = majvote(vc_keys, th)
                inter_sig = combine_intersect(dd_signal, mv)
                lev = inter_sig * vt_layer
                n_strats = len(vc_keys)
                th_label = f"{int(np.ceil(n_strats * th))}/{n_strats}"
                test(lev, f'Inter DD∪MV({vc_name} {th_label}) × {vt_name}', 'T3-Inter', th_rebal=0.20)

    # =================================================================
    # TEST 4: MajVote SUPERMAJORITY as enhanced DD
    # =================================================================
    print("\n" + "=" * 140)
    print("TEST 4: SUPERMAJORITY VOTE as Enhanced DD (strictest configs)")
    print("=" * 140)

    # Use higher thresholds - only exit when overwhelming agreement
    for vt_name, vt_layer in [('S+T', vt_slope_trendtv)]:
        print(f"\n--- VT Layer: {vt_name} ---")
        for vc_name, vc_keys in [
            ('11v(S1-S11)', ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11']),
            ('13v(All)', list(all_signals.keys())),
        ]:
            n = len(vc_keys)
            # Test from lenient to strict majority-to-hold
            for min_hold in range(3, n - 1):
                th = min_hold / n
                mv = majvote(vc_keys, th)
                # Union with DD
                union_sig = combine_union(dd_signal, mv)
                lev = union_sig * vt_layer
                hold_pct = union_sig.mean() * 100
                if hold_pct < 40 or hold_pct > 90:
                    continue  # Skip extreme configs
                test(lev, f'Union DD∩MV({vc_name} {min_hold}/{n}) × {vt_name}', 'T4-Super', th_rebal=0.20)

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 140)
    print("TOP 20 RESULTS (sorted by Sharpe)")
    print("=" * 140)

    results_df = pd.DataFrame(results)
    cols = ['Category', 'Strategy', 'Sharpe', 'CAGR', 'MaxDD', 'Worst5Y', 'Trades', 'Trades/Year']
    results_df = results_df[cols].sort_values('Sharpe', ascending=False)

    display = results_df.head(20).copy()
    for c in ['CAGR', 'MaxDD', 'Worst5Y']:
        display[c] = display[c].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "N/A")
    display['Sharpe'] = display['Sharpe'].apply(lambda x: f"{x:.3f}")
    display['Trades'] = display['Trades'].astype(int)
    display['Trades/Year'] = display['Trades/Year'].apply(lambda x: f"{x:.0f}")
    print(display.to_string(index=False))

    # Highlight: best hybrid vs baseline
    print("\n" + "=" * 140)
    print("KEY COMPARISON: Best Hybrid vs Baseline")
    print("=" * 140)

    baseline_rows = results_df[results_df['Category'] == 'Baseline']
    hybrid_rows = results_df[results_df['Category'] != 'Baseline']
    best_baseline = baseline_rows.iloc[0] if len(baseline_rows) > 0 else None
    best_hybrid = hybrid_rows.iloc[0] if len(hybrid_rows) > 0 else None

    if best_baseline is not None and best_hybrid is not None:
        print(f"\n  Best Baseline:  {best_baseline['Strategy']}")
        print(f"    Sharpe={best_baseline['Sharpe']:.3f}  CAGR={best_baseline['CAGR']*100:+.1f}%  "
              f"MaxDD={best_baseline['MaxDD']*100:.1f}%  W5Y={best_baseline['Worst5Y']*100:+.1f}%")
        print(f"\n  Best Hybrid:    {best_hybrid['Strategy']}")
        print(f"    Sharpe={best_hybrid['Sharpe']:.3f}  CAGR={best_hybrid['CAGR']*100:+.1f}%  "
              f"MaxDD={best_hybrid['MaxDD']*100:.1f}%  W5Y={best_hybrid['Worst5Y']*100:+.1f}%")
        print(f"\n  Delta: Sharpe {best_hybrid['Sharpe'] - best_baseline['Sharpe']:+.3f}  "
              f"MaxDD {(best_hybrid['MaxDD'] - best_baseline['MaxDD'])*100:+.1f}%  "
              f"W5Y {(best_hybrid['Worst5Y'] - best_baseline['Worst5Y'])*100:+.1f}%")

    # Also show best Union specifically
    union_rows = results_df[results_df['Category'] == 'T2-Union']
    if len(union_rows) > 0:
        best_union = union_rows.iloc[0]
        print(f"\n  Best Union:     {best_union['Strategy']}")
        print(f"    Sharpe={best_union['Sharpe']:.3f}  CAGR={best_union['CAGR']*100:+.1f}%  "
              f"MaxDD={best_union['MaxDD']*100:.1f}%  W5Y={best_union['Worst5Y']*100:+.1f}%")

    # Save
    out = os.path.join(script_dir, '..', 'hybrid_strategy_results.csv')
    results_df.to_csv(out, index=False)
    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
