"""
Direction C: Vote Ratio as Continuous Parameter
================================================

Core Idea:
  In Phase 2, the binary majority-vote ensemble (Sharpe 0.741) underperformed
  Ens2 (Sharpe 0.863) because binary HOLD/CASH decisions are fundamentally limited.

  Direction C reuses the 13 sub-strategy VOTES but treats the vote ratio
  (proportion voting HOLD) as a CONTINUOUS multiplier on the existing Ens2 strategy.
  This preserves continuous leverage adjustment while adding a "consensus confidence"
  overlay.

  Example: 10/13 vote HOLD → vote_mult = f(10/13) → applied to Ens2 leverage

Current bests:
  - Ens2(S+T) Th20%:                  Sharpe 0.863 (baseline)
  - MomDecel(40/120) + Ens2(S+T):     Sharpe 0.892 (Direction B winner)

Tests:
  Part 1: Vote ratio distribution & return correlation analysis
  Part 2: Vote ratio multiplier × Ens2(S+T) baseline
  Part 3: Vote ratio multiplier × MomDecel + Ens2(S+T) (current best)
  Part 4: Vote ratio → dynamic target vol adjustment
  Part 5: Delay robustness + cross-direction comparison (A vs B vs C)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol
)
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol,
    strategy_ens2_slope_trendtv
)
from test_majority_vote_p1 import (
    calc_S1_dd_standard, calc_S2_dd_loose, calc_S3_ma200_regime,
    calc_S4_dual_ma_cross, calc_S5_momentum_120, calc_S6_momentum_250,
    calc_S7_vol_regime, calc_S8_rolling_sharpe, calc_S9_bb_lower,
    calc_S10_vov, calc_S11_ma_slope, calc_S12_seasonal, calc_S13_dd_short
)
from test_delay_robust import (
    rebalance_threshold, rebalance_threshold_soft,
    run_backtest_realistic, count_rebalances,
    calc_momentum_decel_mult
)

# =============================================================================
# Constants
# =============================================================================
ANNUAL_COST = 0.015
DEFAULT_DELAY = 5
DEFAULT_THRESHOLD = 0.20

# =============================================================================
# Vote Ratio Computation
# =============================================================================
def compute_all_signals(close, returns, dates):
    """Compute all 13 sub-strategy signals."""
    signals = {
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
    return signals


def compute_vote_ratio(signals, subset=None):
    """Compute vote ratio (proportion voting HOLD) for given subset.

    Returns Series with values 0.0 to 1.0.
    """
    if subset is None:
        subset = list(signals.keys())
    n = len(subset)
    total_hold = sum(signals[k] for k in subset)
    return total_hold / n


# =============================================================================
# Vote Ratio → Multiplier Mapping Functions
# =============================================================================
def vote_mult_linear(vote_ratio):
    """Linear: vote_mult = vote_ratio (0.0 to 1.0)"""
    return vote_ratio


def vote_mult_centered(vote_ratio, floor=0.5):
    """Centered: floor + (1-floor) × ratio. Range: [floor, 1.0]"""
    return floor + (1 - floor) * vote_ratio


def vote_mult_threshold(vote_ratio, low_th=0.3, high_th=0.7):
    """Threshold: 0 below low_th, 1 above high_th, linear between.
    Strong conviction zone."""
    result = (vote_ratio - low_th) / (high_th - low_th)
    return result.clip(0.0, 1.0)


def vote_mult_power(vote_ratio, alpha=0.5):
    """Power: ratio^alpha. alpha<1 = concave (gentle), alpha>1 = convex (aggressive)"""
    return vote_ratio.clip(0.001) ** alpha


def vote_mult_asymmetric(vote_ratio, bull_scale=1.0, bear_scale=2.0):
    """Asymmetric: stronger reduction on bearish side, weaker boost on bullish.
    Maps 0.5 ratio → 1.0, 0.0 → 0.0, 1.0 → 1.0 (with different slopes).
    """
    mid = 0.5
    result = vote_ratio.copy()
    bullish = vote_ratio >= mid
    bearish = vote_ratio < mid

    # Bearish side: 0→0, 0.5→1.0, with bear_scale curvature
    result[bearish] = (vote_ratio[bearish] / mid) ** bear_scale
    # Bullish side: 0.5→1.0, 1.0→1.0 + boost
    result[bullish] = 1.0 + bull_scale * (vote_ratio[bullish] - mid)

    return result.clip(0.0, 1.5)


# =============================================================================
# Strategy Builders
# =============================================================================
def build_ens2_st_base(close, returns):
    """Build base Ens2(S+T) leverage components (without DD)."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92, 200)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    base_lev = dd_signal * vt_lev * slope_mult
    return base_lev.clip(0, 1.0).fillna(0), dd_signal


def build_momdecel_ens2_st(close, returns, short=40, long=120, sens=0.3):
    """Build MomDecel + Ens2(S+T) leverage."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92, 200)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    decel_mult = calc_momentum_decel_mult(close, short, long, sens)
    base_lev = dd_signal * vt_lev * slope_mult * decel_mult
    return base_lev.clip(0, 1.0).fillna(0), dd_signal


def strategy_with_vote_mult(base_leverage, vote_ratio, mapping_func, **kwargs):
    """Apply vote ratio multiplier to base leverage."""
    vote_mult = mapping_func(vote_ratio, **kwargs)
    adjusted = base_leverage * vote_mult
    return adjusted.clip(0.0, 1.5).fillna(0)


# =============================================================================
# Test harness
# =============================================================================
def test_strategy(close, dates, raw_leverage, dd_signal, name, category,
                  delay=DEFAULT_DELAY, threshold=DEFAULT_THRESHOLD,
                  use_soft_rebal=False):
    """Test and return metrics dict."""
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
        'Strategy': name, 'Category': category,
        'Delay': delay, 'Rebalances': rebal,
        'Rebal/Year': rebal / (len(close) / 252),
    })
    return metrics


def print_result(m):
    print(f"  {m['Strategy']:<72s} Sharpe={m['Sharpe']:.4f}  CAGR={m['CAGR']*100:.1f}%  "
          f"MaxDD={m['MaxDD']*100:.1f}%  W5Y={m['Worst5Y']*100:.1f}%  "
          f"Rebal={m['Rebalances']} ({m['Rebal/Year']:.1f}/yr)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 140)
    print("DIRECTION C: VOTE RATIO AS CONTINUOUS PARAMETER")
    print("=" * 140)

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

    def run_test(lev, dd, name, cat, delay=DEFAULT_DELAY, th=DEFAULT_THRESHOLD,
                 soft=False):
        m = test_strategy(close, dates, lev, dd, name, cat, delay, th, soft)
        all_results.append(m)
        return m

    # Pre-compute
    print("Computing 13 sub-strategy signals...")
    signals = compute_all_signals(close, returns, dates)
    print("Computing base strategies...")
    ens2_st_lev, ens2_st_dd = build_ens2_st_base(close, returns)
    momdecel_lev, momdecel_dd = build_momdecel_ens2_st(close, returns, 40, 120, 0.3)

    # Define sub-strategy subsets
    subsets = {
        'All13': list(signals.keys()),
        'NoS10S12': [k for k in signals if k not in ('S10', 'S12')],  # 11 strats
        'Diverse5': ['S1', 'S5', 'S7', 'S9', 'S12'],     # Low-corr set
        'Core7': ['S1', 'S3', 'S5', 'S7', 'S8', 'S11', 'S13'],  # Diverse mix
        'Strong5': ['S1', 'S3', 'S4', 'S5', 'S11'],       # Best individual Sharpe
        'DD3': ['S1', 'S2', 'S13'],                        # DD family only
        'NoDDfam': [k for k in signals if k not in ('S1', 'S2', 'S13')],  # Exclude DD family
    }

    # =====================================================================
    # PART 1: VOTE RATIO DISTRIBUTION & RETURN CORRELATION
    # =====================================================================
    print("\n" + "=" * 140)
    print("PART 1: VOTE RATIO DISTRIBUTION & PREDICTIVE POWER ANALYSIS")
    print("=" * 140)

    vr_all = compute_vote_ratio(signals, subsets['All13'])

    # Distribution
    print(f"\n  [Vote Ratio Distribution (All 13)]")
    for low, high, label in [(0, 0.2, "0.0-0.2 (strong bear)"),
                              (0.2, 0.4, "0.2-0.4 (bear)"),
                              (0.4, 0.6, "0.4-0.6 (neutral)"),
                              (0.6, 0.8, "0.6-0.8 (bull)"),
                              (0.8, 1.01, "0.8-1.0 (strong bull)")]:
        pct = ((vr_all >= low) & (vr_all < high)).mean() * 100
        print(f"    {label:<25s}: {pct:.1f}%")

    print(f"\n    Mean: {vr_all.mean():.3f}, Median: {vr_all.median():.3f}")
    print(f"    Autocorrelation (1d): {vr_all.autocorr(1):.3f}")
    print(f"    Autocorrelation (5d): {vr_all.autocorr(5):.3f}")
    print(f"    Autocorrelation (20d): {vr_all.autocorr(20):.3f}")

    # Predictive power: future N-day return by vote ratio bucket
    print(f"\n  [Vote Ratio vs Future Returns (All 13)]")
    fwd_returns = {}
    for horizon in [1, 5, 20]:
        fwd_ret = close.pct_change(horizon).shift(-horizon) * 252 / horizon
        fwd_returns[horizon] = fwd_ret

    print(f"    {'VR Bucket':<20s}", end="")
    for h in [1, 5, 20]:
        print(f"  {'Fwd'+str(h)+'d Ann':>14s}", end="")
    print(f"  {'Count':>8s}")
    print("    " + "-" * 70)

    for low, high in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        mask = (vr_all >= low) & (vr_all < high)
        print(f"    [{low:.1f}-{high:.1f}){' '*(12-len(f'[{low:.1f}-{high:.1f})'))}", end="")
        for h in [1, 5, 20]:
            avg = fwd_returns[h][mask].mean()
            print(f"  {avg*100:>+13.1f}%", end="")
        print(f"  {mask.sum():>8d}")

    # Per-subset stats
    print(f"\n  [Vote Ratio Autocorrelation by Subset]")
    for subset_name, subset_keys in subsets.items():
        vr = compute_vote_ratio(signals, subset_keys)
        print(f"    {subset_name:<15s}: mean={vr.mean():.3f}  autocorr(5d)={vr.autocorr(5):.3f}  "
              f"std={vr.std():.3f}")

    # =====================================================================
    # PART 2: VOTE RATIO MULTIPLIER × Ens2(S+T) BASELINE
    # =====================================================================
    print("\n" + "=" * 140)
    print("PART 2: VOTE RATIO MULTIPLIER × Ens2(S+T) BASELINE")
    print("=" * 140)

    # Baselines
    print(f"\n  [Baselines]")
    m = run_test(ens2_st_lev, ens2_st_dd, 'BASELINE: Ens2(S+T) Th20%', 'Baseline')
    print_result(m)
    m = run_test(momdecel_lev, momdecel_dd, 'BEST(DirB): MomDecel(40/120)+Ens2(S+T)', 'DirB-Best')
    print_result(m)

    # Mapping functions × subsets
    mapping_configs = [
        ('Linear',    vote_mult_linear,    {}),
        ('Centered50', vote_mult_centered,  {'floor': 0.5}),
        ('Centered70', vote_mult_centered,  {'floor': 0.7}),
        ('Thresh30-70', vote_mult_threshold, {'low_th': 0.3, 'high_th': 0.7}),
        ('Thresh20-80', vote_mult_threshold, {'low_th': 0.2, 'high_th': 0.8}),
        ('Thresh40-70', vote_mult_threshold, {'low_th': 0.4, 'high_th': 0.7}),
        ('Power0.3',  vote_mult_power,     {'alpha': 0.3}),
        ('Power0.5',  vote_mult_power,     {'alpha': 0.5}),
        ('Power2.0',  vote_mult_power,     {'alpha': 2.0}),
        ('Asym1-2',   vote_mult_asymmetric, {'bull_scale': 1.0, 'bear_scale': 2.0}),
        ('Asym0.5-3', vote_mult_asymmetric, {'bull_scale': 0.5, 'bear_scale': 3.0}),
    ]

    subset_configs = ['All13', 'NoS10S12', 'Core7', 'Strong5', 'NoDDfam']

    # First: sweep all mappings with All13
    print(f"\n  [All Mappings × All13 → applied to Ens2(S+T)]")
    vr = compute_vote_ratio(signals, subsets['All13'])
    for map_name, map_func, map_kwargs in mapping_configs:
        lev = strategy_with_vote_mult(ens2_st_lev, vr, map_func, **map_kwargs)
        m = run_test(lev, ens2_st_dd,
                     f'VR({map_name},All13) × Ens2(S+T)', 'P2-Map',
                     soft=True)
        print_result(m)

    # Second: best mappings × different subsets
    # (select promising mappings from above)
    best_maps = ['Centered70', 'Thresh30-70', 'Power0.5', 'Asym1-2']
    best_map_configs = [(n, f, k) for n, f, k in mapping_configs if n in best_maps]

    print(f"\n  [Best Mappings × Different Subsets → Ens2(S+T)]")
    for subset_name in subset_configs:
        vr = compute_vote_ratio(signals, subsets[subset_name])
        for map_name, map_func, map_kwargs in best_map_configs:
            lev = strategy_with_vote_mult(ens2_st_lev, vr, map_func, **map_kwargs)
            m = run_test(lev, ens2_st_dd,
                         f'VR({map_name},{subset_name}) × Ens2(S+T)',
                         'P2-Subset', soft=True)
            print_result(m)

    # =====================================================================
    # PART 3: VOTE RATIO × MomDecel + Ens2(S+T) (CURRENT BEST)
    # =====================================================================
    print("\n" + "=" * 140)
    print("PART 3: VOTE RATIO MULTIPLIER × MomDecel(40/120) + Ens2(S+T)")
    print("  Can vote ratio add information beyond MomDecel?")
    print("=" * 140)

    print(f"\n  [All Mappings × All13 → applied to MomDecel+Ens2]")
    vr_all = compute_vote_ratio(signals, subsets['All13'])
    for map_name, map_func, map_kwargs in mapping_configs:
        lev = strategy_with_vote_mult(momdecel_lev, vr_all, map_func, **map_kwargs)
        m = run_test(lev, momdecel_dd,
                     f'VR({map_name},All13) × MomDec+Ens2', 'P3-MomDec',
                     soft=True)
        print_result(m)

    # Best mappings × subsets on MomDecel
    print(f"\n  [Best Mappings × Subsets → MomDecel+Ens2]")
    for subset_name in ['All13', 'NoS10S12', 'Core7', 'NoDDfam']:
        vr = compute_vote_ratio(signals, subsets[subset_name])
        for map_name, map_func, map_kwargs in best_map_configs:
            lev = strategy_with_vote_mult(momdecel_lev, vr, map_func, **map_kwargs)
            m = run_test(lev, momdecel_dd,
                         f'VR({map_name},{subset_name}) × MomDec+Ens2',
                         'P3-MomSubset', soft=True)
            print_result(m)

    # =====================================================================
    # PART 4: VOTE RATIO → DYNAMIC TARGET VOL
    # =====================================================================
    print("\n" + "=" * 140)
    print("PART 4: VOTE RATIO → DYNAMIC TARGET VOL ADJUSTMENT")
    print("  Use vote ratio to modulate TrendTV range (not as multiplier)")
    print("=" * 140)

    dd_sig = calc_dd_signal(close, 0.82, 0.92, 200)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    slope_mult = calc_slope_multiplier(close)

    for subset_name in ['All13', 'NoS10S12', 'Core7']:
        vr = compute_vote_ratio(signals, subsets[subset_name])
        print(f"\n  [Subset: {subset_name}]")

        # Vote-adjusted TrendTV: modulate TV range based on consensus
        # High consensus → wider range (more aggressive)
        # Low consensus → narrower range (more conservative)
        configs = [
            # (tv_min_bear, tv_max_bear, tv_min_bull, tv_max_bull, label)
            (0.10, 0.20, 0.15, 0.35, "Bear10-20/Bull15-35"),
            (0.10, 0.15, 0.20, 0.40, "Bear10-15/Bull20-40"),
            (0.12, 0.20, 0.18, 0.38, "Bear12-20/Bull18-38"),
            (0.10, 0.25, 0.15, 0.35, "Bear10-25/Bull15-35"),
        ]

        for tv_min_b, tv_max_b, tv_min_bu, tv_max_bu, label in configs:
            # Interpolate TV parameters based on vote ratio
            tv_min = tv_min_b + (tv_min_bu - tv_min_b) * vr
            tv_max = tv_max_b + (tv_max_bu - tv_max_b) * vr

            # Use standard TrendTV calculation but with dynamic min/max
            trend_tv = calc_trend_target_vol(close)  # base 15-35%
            # Remap: scale trend_tv from [0.15, 0.35] to [tv_min, tv_max]
            tv_norm = (trend_tv - 0.15) / (0.35 - 0.15)  # 0-1 range
            dynamic_tv = tv_min + (tv_max - tv_min) * tv_norm

            vt_lev = (dynamic_tv / asym_vol).clip(0, 1.0)
            lev = (dd_sig * vt_lev * slope_mult).clip(0, 1.0).fillna(0)

            m = run_test(lev, dd_sig,
                         f'VoteDynTV({label},{subset_name})',
                         'P4-DynTV', soft=True)
            print_result(m)

    # Also test dynamic TV with MomDecel
    print(f"\n  [Dynamic TV + MomDecel(40/120)]")
    decel_mult = calc_momentum_decel_mult(close, 40, 120, 0.3)
    vr = compute_vote_ratio(signals, subsets['All13'])

    for tv_min_b, tv_max_b, tv_min_bu, tv_max_bu, label in [
        (0.10, 0.20, 0.15, 0.35, "Bear10-20/Bull15-35"),
        (0.10, 0.15, 0.20, 0.40, "Bear10-15/Bull20-40"),
        (0.12, 0.20, 0.18, 0.38, "Bear12-20/Bull18-38"),
    ]:
        tv_min = tv_min_b + (tv_min_bu - tv_min_b) * vr
        tv_max = tv_max_b + (tv_max_bu - tv_max_b) * vr
        trend_tv = calc_trend_target_vol(close)
        tv_norm = (trend_tv - 0.15) / (0.35 - 0.15)
        dynamic_tv = tv_min + (tv_max - tv_min) * tv_norm
        vt_lev = (dynamic_tv / asym_vol).clip(0, 1.0)
        lev = (dd_sig * vt_lev * slope_mult * decel_mult).clip(0, 1.0).fillna(0)

        m = run_test(lev, dd_sig,
                     f'VoteDynTV({label})+MomDec', 'P4-DynTVMD',
                     soft=True)
        print_result(m)

    # =====================================================================
    # PART 5: DELAY ROBUSTNESS + CROSS-DIRECTION COMPARISON
    # =====================================================================
    print("\n" + "=" * 140)
    print("PART 5: DELAY ROBUSTNESS + CROSS-DIRECTION COMPARISON")
    print("=" * 140)

    # Collect best from each part
    p2_results = [r for r in all_results if r['Category'].startswith('P2')]
    p3_results = [r for r in all_results if r['Category'].startswith('P3')]
    p4_results = [r for r in all_results if r['Category'].startswith('P4')]

    for label, res_list in [("Part2 (VR×Ens2)", p2_results),
                            ("Part3 (VR×MomDec+Ens2)", p3_results),
                            ("Part4 (DynTV)", p4_results)]:
        if res_list:
            best = max(res_list, key=lambda x: x['Sharpe'])
            print(f"  Best {label}: {best['Strategy']}")
            print(f"    Sharpe={best['Sharpe']:.4f}  CAGR={best['CAGR']*100:.1f}%  "
                  f"MaxDD={best['MaxDD']*100:.1f}%  W5Y={best['Worst5Y']*100:.1f}%")

    # Build best candidates for robustness test
    robustness_strats = {}

    # Baselines
    robustness_strats['Ens2(S+T) [Baseline]'] = (ens2_st_lev, ens2_st_dd, False)
    robustness_strats['MomDecel(40/120)+Ens2 [DirB]'] = (momdecel_lev, momdecel_dd, False)

    # Best from Part 2 (select top candidates)
    vr_all13 = compute_vote_ratio(signals, subsets['All13'])
    vr_nos10s12 = compute_vote_ratio(signals, subsets['NoS10S12'])
    vr_core7 = compute_vote_ratio(signals, subsets['Core7'])
    vr_nodd = compute_vote_ratio(signals, subsets['NoDDfam'])

    # Promising P2 candidates: Centered70 and Power0.5 likely best
    for map_name, map_func, map_kwargs, vr, vr_name in [
        ('Centered70', vote_mult_centered, {'floor': 0.7}, vr_all13, 'All13'),
        ('Power0.5', vote_mult_power, {'alpha': 0.5}, vr_all13, 'All13'),
        ('Thresh30-70', vote_mult_threshold, {'low_th': 0.3, 'high_th': 0.7}, vr_all13, 'All13'),
    ]:
        lev = strategy_with_vote_mult(ens2_st_lev, vr, map_func, **map_kwargs)
        robustness_strats[f'VR({map_name},{vr_name})×Ens2'] = (lev, ens2_st_dd, True)

    # Best P3 candidates
    for map_name, map_func, map_kwargs, vr, vr_name in [
        ('Centered70', vote_mult_centered, {'floor': 0.7}, vr_all13, 'All13'),
        ('Power0.5', vote_mult_power, {'alpha': 0.5}, vr_all13, 'All13'),
    ]:
        lev = strategy_with_vote_mult(momdecel_lev, vr, map_func, **map_kwargs)
        robustness_strats[f'VR({map_name},{vr_name})×MomDec+Ens2'] = (lev, momdecel_dd, True)

    # Delay robustness test
    delays_to_test = [0, 1, 3, 5, 7, 10]
    robustness_data = {}

    for name, (lev, dd, soft) in robustness_strats.items():
        robustness_data[name] = {}
        for d in delays_to_test:
            m = test_strategy(close, dates, lev, dd, f'{name} d={d}', 'P5-Robust',
                              delay=d, threshold=DEFAULT_THRESHOLD,
                              use_soft_rebal=soft)
            robustness_data[name][d] = m
            all_results.append(m)

    # Print Sharpe table
    print(f"\n  Sharpe at each delay:")
    print(f"  {'Strategy':<42s}", end="")
    for d in delays_to_test:
        print(f"  d={d:<3d}", end="")
    print(f"  d1→d5  d5-d1")
    print("  " + "-" * 120)

    for name in robustness_strats:
        print(f"  {name:<42s}", end="")
        for d in delays_to_test:
            print(f"  {robustness_data[name][d]['Sharpe']:.3f}", end="")
        s1 = robustness_data[name][1]['Sharpe']
        s5 = robustness_data[name][5]['Sharpe']
        print(f"  {s5/s1:.3f}  {s5-s1:+.3f}")

    # MaxDD table
    print(f"\n  MaxDD at each delay:")
    print(f"  {'Strategy':<42s}", end="")
    for d in delays_to_test:
        print(f"  d={d:<3d}", end="")
    print()
    print("  " + "-" * 100)

    for name in robustness_strats:
        print(f"  {name:<42s}", end="")
        for d in delays_to_test:
            print(f"  {robustness_data[name][d]['MaxDD']*100:.1f}%", end="")
        print()

    # W5Y table
    print(f"\n  Worst5Y at each delay:")
    print(f"  {'Strategy':<42s}", end="")
    for d in delays_to_test:
        print(f"  d={d:<3d}", end="")
    print()
    print("  " + "-" * 100)

    for name in robustness_strats:
        print(f"  {name:<42s}", end="")
        for d in delays_to_test:
            w5y = robustness_data[name][d]['Worst5Y']
            print(f"  {w5y*100:+.1f}%", end="")
        print()

    # =====================================================================
    # FINAL: ALL-DIRECTION RANKING AT DELAY=5
    # =====================================================================
    print("\n" + "=" * 140)
    print("FINAL: ALL-DIRECTION RANKING AT DELAY=5 (sorted by Sharpe)")
    print("=" * 140)

    d5_results = [r for r in all_results if r.get('Delay') == 5]
    seen = {}
    for r in d5_results:
        seen[r['Strategy']] = r
    unique_d5 = sorted(seen.values(), key=lambda x: -x['Sharpe'])

    print(f"\n  {'#':<4s} {'Strategy':<72s} {'Sharpe':>7s} {'CAGR':>7s} "
          f"{'MaxDD':>7s} {'W5Y':>7s} {'Rebal':>6s}")
    print("  " + "-" * 115)

    for i, r in enumerate(unique_d5[:40], 1):
        print(f"  {i:<4d} {r['Strategy']:<72s} {r['Sharpe']:.3f} "
              f"{r['CAGR']*100:+.1f}% {r['MaxDD']*100:.1f}% "
              f"{r['Worst5Y']*100:+.1f}% {r['Rebalances']:>5d}")

    # =====================================================================
    # KEY FINDINGS
    # =====================================================================
    print("\n" + "=" * 140)
    print("KEY FINDINGS: DIRECTION C")
    print("=" * 140)

    baseline_s = robustness_data['Ens2(S+T) [Baseline]'][5]['Sharpe']
    dirb_s = robustness_data['MomDecel(40/120)+Ens2 [DirB]'][5]['Sharpe']

    print(f"\n  Reference points:")
    print(f"    Ens2(S+T) baseline at d=5:       Sharpe = {baseline_s:.4f}")
    print(f"    MomDecel(40/120)+Ens2 at d=5:     Sharpe = {dirb_s:.4f}")

    # Best Dir C result
    dc_results = [r for r in unique_d5
                  if r['Category'].startswith('P2') or r['Category'].startswith('P3')
                  or r['Category'].startswith('P4')]
    if dc_results:
        best_dc = dc_results[0]
        print(f"\n  Best Direction C at d=5: {best_dc['Strategy']}")
        print(f"    Sharpe = {best_dc['Sharpe']:.4f}  "
              f"(vs baseline {baseline_s:.4f}, diff = {best_dc['Sharpe']-baseline_s:+.4f})")
        print(f"    (vs DirB best {dirb_s:.4f}, diff = {best_dc['Sharpe']-dirb_s:+.4f})")

    # Save
    output_path = os.path.join(script_dir, '..', 'vote_ratio_continuous_results.csv')
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
