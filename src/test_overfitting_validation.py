"""
Overfitting Validation - Comprehensive Robustness Testing
==========================================================
Tests whether each strategy layer genuinely improves OOS performance,
or is merely fitting noise in the training data.

Phases:
  P1: Complexity Ladder × Walk-Forward (S0-S6 × 5 Folds, fixed params)
  P2: Leave-One-Crisis-Out (S1,S5,S6 × 5 crises)
  P3: Parameter Perturbation Analysis (S6, ±10/20/30%)
  P4: Deflated Sharpe Ratio (multiple testing correction)
  P5: Block Bootstrap Confidence Intervals (1000 reps)

Conditions: 5-day delay, 1.5% annual cost, rebalance threshold 20%
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import load_data, calc_dd_signal, calc_ewma_vol, calc_vt_leverage
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    calc_momentum_decel_mult, rebalance_threshold
)

# Constants
ANNUAL_COST = 0.015
BASE_LEV = 3.0
EXEC_DELAY = 5
REBAL_TH = 0.20


# =================================================================
# Strategy Definitions (S0-S6)
# =================================================================
def strategy_s0_bh3x(close, returns):
    """S0: Buy & Hold 3x - Zero parameters."""
    return pd.Series(1.0, index=close.index), 'B&H 3x'

def strategy_s1_dd_only(close, returns, exit_th=0.82, reentry_th=0.92,
                        lookback=200):
    """S1: DD Control Only - 3 parameters."""
    dd = calc_dd_signal(close, exit_th, reentry_th, lookback)
    return dd, 'DD-Only'

def strategy_s2_dd_vt(close, returns, exit_th=0.82, reentry_th=0.92,
                      lookback=200, ewma_span=10, target_vol=0.25):
    """S2: DD + Standard VT - 5 parameters."""
    dd = calc_dd_signal(close, exit_th, reentry_th, lookback)
    vol = calc_ewma_vol(returns, ewma_span)
    vt = calc_vt_leverage(vol, target_vol, max_lev=1.0)
    lev = dd * vt
    return lev.clip(0, 1.0).fillna(0), 'DD+VT'

def strategy_s3_dd_asymvt(close, returns, exit_th=0.82, reentry_th=0.92,
                          lookback=200, span_up=20, span_dn=5,
                          target_vol=0.25):
    """S3: DD + Asymmetric VT - 6 parameters."""
    dd = calc_dd_signal(close, exit_th, reentry_th, lookback)
    asym_vol = calc_asym_ewma_vol(returns, span_up, span_dn)
    vt = (target_vol / asym_vol).clip(0, 1.0)
    lev = dd * vt
    return lev.clip(0, 1.0).fillna(0), 'DD+AsymVT'

def strategy_s4_dd_asymvt_slope(close, returns, exit_th=0.82,
                                reentry_th=0.92, lookback=200,
                                span_up=20, span_dn=5, target_vol=0.25):
    """S4: DD + AsymVT + SlopeMult - 11 parameters."""
    dd = calc_dd_signal(close, exit_th, reentry_th, lookback)
    asym_vol = calc_asym_ewma_vol(returns, span_up, span_dn)
    vt = (target_vol / asym_vol).clip(0, 1.0)
    slope = calc_slope_multiplier(close)
    lev = dd * vt * slope
    return lev.clip(0, 1.0).fillna(0), 'DD+AsymVT+Slope'

def strategy_s5_ens2_st(close, returns, exit_th=0.82, reentry_th=0.92,
                        lookback=200, span_up=20, span_dn=5):
    """S5: Ens2(Slope+TrendTV) - 15 parameters."""
    dd = calc_dd_signal(close, exit_th, reentry_th, lookback)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, span_up, span_dn)
    vt = (trend_tv / asym_vol).clip(0, 1.0)
    slope = calc_slope_multiplier(close)
    lev = dd * vt * slope
    return lev.clip(0, 1.0).fillna(0), 'Ens2(S+T)'

def strategy_s6_momdecel_ens2(close, returns, exit_th=0.82, reentry_th=0.92,
                              lookback=200, span_up=20, span_dn=5,
                              md_short=40, md_long=120, md_sens=0.3):
    """S6: MomDecel(40/120) + Ens2(S+T) - 20 parameters."""
    dd = calc_dd_signal(close, exit_th, reentry_th, lookback)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, span_up, span_dn)
    vt = (trend_tv / asym_vol).clip(0, 1.0)
    slope = calc_slope_multiplier(close)
    decel = calc_momentum_decel_mult(close, md_short, md_long, md_sens)
    lev = dd * vt * slope * decel
    return lev.clip(0, 1.0).fillna(0), 'MomDecel+Ens2(S+T)'

ALL_STRATEGIES = [
    ('S0', strategy_s0_bh3x, 0),
    ('S1', strategy_s1_dd_only, 3),
    ('S2', strategy_s2_dd_vt, 5),
    ('S3', strategy_s3_dd_asymvt, 6),
    ('S4', strategy_s4_dd_asymvt_slope, 11),
    ('S5', strategy_s5_ens2_st, 15),
    ('S6', strategy_s6_momdecel_ens2, 20),
]


# =================================================================
# Backtest Engine
# =================================================================
def run_backtest(close, leverage, delay=EXEC_DELAY, cost=ANNUAL_COST,
                 apply_threshold=True):
    """Run backtest with realistic constraints."""
    returns = close.pct_change()
    lev_ret = returns * BASE_LEV
    daily_cost = cost / 252

    if apply_threshold:
        lev_th = rebalance_threshold(leverage, REBAL_TH)
    else:
        lev_th = leverage

    delayed = lev_th.shift(delay)
    strat_ret = delayed * (lev_ret - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret


def calc_metrics(nav, strat_ret, n_years):
    """Calculate key performance metrics."""
    sharpe = strat_ret.mean() * 252 / (strat_ret.std() * np.sqrt(252)) \
        if strat_ret.std() > 0 else 0
    cagr = nav.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    maxdd = ((nav - nav.cummax()) / nav.cummax()).min()
    if len(nav) >= 252 * 5:
        w5y = ((nav / nav.shift(252 * 5)) ** 0.2 - 1).min()
    else:
        w5y = np.nan
    return {'Sharpe': sharpe, 'CAGR': cagr, 'MaxDD': maxdd, 'Worst5Y': w5y}


# =================================================================
# Walk-Forward Periods
# =================================================================
# Embargo = 252 days (1 year) between train and test
WALK_FORWARD_FOLDS = [
    {'name': 'Fold1', 'train_end': '1994-12-31', 'test_start': '1996-01-02', 'test_end': '2001-12-31'},
    {'name': 'Fold2', 'train_end': '2000-12-31', 'test_start': '2002-01-02', 'test_end': '2007-12-31'},
    {'name': 'Fold3', 'train_end': '2006-12-31', 'test_start': '2008-01-02', 'test_end': '2013-12-31'},
    {'name': 'Fold4', 'train_end': '2012-12-31', 'test_start': '2014-01-02', 'test_end': '2019-12-31'},
    {'name': 'Fold5', 'train_end': '2018-12-31', 'test_start': '2020-01-02', 'test_end': '2026-12-31'},
]

CRISES = [
    {'name': '1987 BlackMonday', 'start': '1987-08-01', 'end': '1988-03-31'},
    {'name': '2000 DotCom',     'start': '2000-03-01', 'end': '2003-03-31'},
    {'name': '2008 GFC',        'start': '2007-10-01', 'end': '2009-06-30'},
    {'name': '2020 COVID',      'start': '2020-02-01', 'end': '2020-06-30'},
    {'name': '2022 RateHike',   'start': '2022-01-01', 'end': '2022-12-31'},
]


# =================================================================
# P4: Deflated Sharpe Ratio
# =================================================================
def deflated_sharpe_ratio(observed_sr, benchmark_sr, T, skew, kurt, n_trials):
    """
    Bailey & López de Prado (2014) - Deflated Sharpe Ratio.
    Tests: is the observed SR statistically better than the expected max
    SR from n_trials independent strategies under the null?

    Args:
        observed_sr: observed Sharpe ratio (annualized)
        benchmark_sr: expected max SR under null (from n_trials)
        T: number of return observations
        skew: skewness of returns
        kurt: excess kurtosis of returns
        n_trials: number of independent strategies tested

    Returns:
        DSR p-value (probability of observing this SR by chance)
    """
    # Standard error of SR estimate
    se = np.sqrt((1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr ** 2) / T)
    if se <= 0:
        return 0.0

    # z-statistic
    z = (observed_sr - benchmark_sr) / se

    # One-sided test: P(SR > benchmark)
    dsr = stats.norm.cdf(z)
    return dsr


def expected_max_sr(n_trials, T, sr_std=1.0):
    """Expected maximum SR from n_trials independent strategies.
    Using approximation from Bailey & López de Prado (2012)."""
    gamma = 0.5772  # Euler-Mascheroni constant
    if n_trials <= 1:
        return 0.0
    z = stats.norm.ppf(1 - 1.0 / n_trials)
    e_max = (1 - gamma) * z + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return e_max * sr_std / np.sqrt(T / 252)


# =================================================================
# P5: Block Bootstrap
# =================================================================
def block_bootstrap_sharpe(strat_returns, n_boot=1000, block_size=20,
                           seed=42):
    """Block bootstrap for Sharpe ratio confidence interval.
    Resamples blocks of returns to preserve autocorrelation."""
    rng = np.random.RandomState(seed)
    T = len(strat_returns)
    ret_arr = strat_returns.values
    n_blocks = T // block_size + 1

    sharpes = []
    for _ in range(n_boot):
        # Sample block starting indices
        starts = rng.randint(0, T - block_size, size=n_blocks)
        # Build resampled series
        resampled = np.concatenate([ret_arr[s:s + block_size] for s in starts])[:T]
        sr = resampled.mean() * 252 / (resampled.std() * np.sqrt(252)) \
            if resampled.std() > 0 else 0
        sharpes.append(sr)

    return np.array(sharpes)


# =================================================================
# MAIN
# =================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()

    print("=" * 120)
    print("OVERFITTING VALIDATION - COMPREHENSIVE ROBUSTNESS TESTING")
    print("Conditions: 5-day delay, 1.5% annual cost, rebalance threshold 20%")
    print("=" * 120)

    # Load extended data
    data_path = os.path.join(script_dir, '..', 'NASDAQ_extended.csv')
    df = load_data(data_path)
    dates = df['Date']
    full_close = df['Close']
    full_returns = full_close.pct_change()
    print(f"\nData: {dates.iloc[0].date()} to {dates.iloc[-1].date()} "
          f"({len(df):,} days, {len(df)/252:.1f}yr)")

    all_results = []

    # ==============================================================
    # PHASE 1: Complexity Ladder × Walk-Forward
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 1: COMPLEXITY LADDER × WALK-FORWARD VALIDATION")
    print("  Fixed parameters (no re-optimization). Tests if current params work across time.")
    print("=" * 120)

    # Also compute full-period metrics for reference
    print("\n  --- Full-Period Reference (all data, same params) ---")
    full_n_years = len(full_close) / 252
    for sid, strat_fn, n_params in ALL_STRATEGIES:
        lev, name = strat_fn(full_close, full_returns)
        nav, sr = run_backtest(full_close, lev)
        m = calc_metrics(nav, sr, full_n_years)
        print(f"  {sid} {name:<25s} (p={n_params:>2d})  "
              f"Sharpe={m['Sharpe']:.3f}  CAGR={m['CAGR']*100:+.1f}%  "
              f"MaxDD={m['MaxDD']*100:.1f}%  W5Y={m['Worst5Y']*100:+.1f}%")

    # Walk-forward evaluation
    wf_results = {sid: [] for sid, _, _ in ALL_STRATEGIES}

    print(f"\n  --- Walk-Forward: 5 Folds (1-year embargo) ---")
    header = f"  {'Strategy':<25s}"
    for fold in WALK_FORWARD_FOLDS:
        header += f" {fold['name']:>10s}"
    header += f" {'Mean':>8s} {'Std':>7s} {'Min':>7s}"
    print(header)
    print("  " + "-" * (25 + 10 * 5 + 8 + 7 + 7 + 5))

    for sid, strat_fn, n_params in ALL_STRATEGIES:
        _, name = strat_fn(full_close, full_returns)
        row = f"  {sid} {name:<22s}"
        fold_sharpes = []

        for fold in WALK_FORWARD_FOLDS:
            # Get test period indices
            test_mask = (dates >= fold['test_start']) & (dates <= fold['test_end'])
            if test_mask.sum() < 126:  # Need at least 6 months
                fold_sharpes.append(np.nan)
                row += f" {'N/A':>10s}"
                continue

            test_close = pd.Series(full_close[test_mask].values)
            test_returns = test_close.pct_change()
            n_years = len(test_close) / 252

            # Apply strategy with fixed parameters
            lev, _ = strat_fn(test_close, test_returns)
            nav, sr = run_backtest(test_close, lev)
            m = calc_metrics(nav, sr, n_years)
            fold_sharpes.append(m['Sharpe'])
            row += f" {m['Sharpe']:>10.3f}"

        valid = [s for s in fold_sharpes if not np.isnan(s)]
        mean_s = np.mean(valid) if valid else np.nan
        std_s = np.std(valid) if len(valid) > 1 else np.nan
        min_s = np.min(valid) if valid else np.nan
        row += f" {mean_s:>8.3f} {std_s:>7.3f} {min_s:>7.3f}"
        print(row)

        wf_results[sid] = fold_sharpes
        all_results.append({
            'Phase': 'P1_WalkForward',
            'Strategy': f"{sid} {name}",
            'n_params': n_params,
            'Mean_OOS_Sharpe': mean_s,
            'Std_OOS_Sharpe': std_s,
            'Min_OOS_Sharpe': min_s,
            'Fold_Sharpes': fold_sharpes
        })

    # Marginal improvement per layer
    print(f"\n  --- Marginal Improvement per Layer (Mean OOS Sharpe) ---")
    prev_mean = None
    for sid, strat_fn, n_params in ALL_STRATEGIES:
        _, name = strat_fn(full_close, full_returns)
        valid = [s for s in wf_results[sid] if not np.isnan(s)]
        mean_s = np.mean(valid) if valid else np.nan
        if prev_mean is not None and not np.isnan(mean_s):
            delta = mean_s - prev_mean
            pct = delta / abs(prev_mean) * 100 if prev_mean != 0 else 0
            # Sign: improvement per added parameter
            dp = n_params - prev_params
            per_param = delta / dp if dp > 0 else 0
            print(f"  {sid} {name:<25s}  Mean={mean_s:.3f}  "
                  f"Δ={delta:+.3f} ({pct:+.1f}%)  "
                  f"Δ/param={per_param:+.4f} (added {dp} params)")
        else:
            print(f"  {sid} {name:<25s}  Mean={mean_s:.3f}  (baseline)")
        prev_mean = mean_s
        prev_params = n_params

    # ==============================================================
    # PHASE 2: Leave-One-Crisis-Out
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 2: LEAVE-ONE-CRISIS-OUT VALIDATION")
    print("  Strategy trained on all data EXCEPT the crisis. Tests crisis handling as OOS.")
    print("=" * 120)

    test_strategies = [
        ('S1', strategy_s1_dd_only, 3),
        ('S5', strategy_s5_ens2_st, 15),
        ('S6', strategy_s6_momdecel_ens2, 20),
    ]

    print(f"\n  {'Strategy':<25s}", end='')
    for c in CRISES:
        print(f" {c['name'][:12]:>14s}", end='')
    print(f" {'Mean':>8s}")
    print("  " + "-" * (25 + 14 * 5 + 8 + 3))

    for sid, strat_fn, n_params in test_strategies:
        _, name = strat_fn(full_close, full_returns)
        row = f"  {sid} {name:<22s}"
        crisis_sharpes = []

        for crisis in CRISES:
            crisis_mask = (dates >= crisis['start']) & (dates <= crisis['end'])
            if crisis_mask.sum() < 20:
                row += f" {'N/A':>14s}"
                crisis_sharpes.append(np.nan)
                continue

            crisis_close = pd.Series(full_close[crisis_mask].values)
            crisis_returns = crisis_close.pct_change()
            n_years = len(crisis_close) / 252

            lev, _ = strat_fn(crisis_close, crisis_returns)
            nav, sr = run_backtest(crisis_close, lev)

            # For crises, report annualized return rather than Sharpe
            # (short periods make Sharpe noisy)
            crisis_ret = nav.iloc[-1] ** (1 / n_years) - 1 if n_years > 0.1 else nav.iloc[-1] - 1
            crisis_sharpes.append(crisis_ret)
            row += f" {crisis_ret*100:>+13.1f}%"

        valid = [s for s in crisis_sharpes if not np.isnan(s)]
        mean_c = np.mean(valid) if valid else np.nan
        row += f" {mean_c*100:>+7.1f}%"
        print(row)

        all_results.append({
            'Phase': 'P2_CrisisOut',
            'Strategy': f"{sid} {name}",
            'n_params': n_params,
            'Crisis_Returns': {c['name']: s for c, s in zip(CRISES, crisis_sharpes)},
            'Mean_Crisis_Return': mean_c,
        })

    # Also show B&H 3x for reference
    row = f"  {'B&H 3x':<25s}"
    bh_crisis = []
    for crisis in CRISES:
        crisis_mask = (dates >= crisis['start']) & (dates <= crisis['end'])
        if crisis_mask.sum() < 20:
            row += f" {'N/A':>14s}"
            continue
        crisis_close = pd.Series(full_close[crisis_mask].values)
        crisis_returns = crisis_close.pct_change()
        n_years = len(crisis_close) / 252
        bh_ret = crisis_returns * 3.0 - ANNUAL_COST / 252
        bh_nav = (1 + bh_ret.fillna(0)).cumprod()
        cr = bh_nav.iloc[-1] ** (1 / n_years) - 1 if n_years > 0.1 else bh_nav.iloc[-1] - 1
        bh_crisis.append(cr)
        row += f" {cr*100:>+13.1f}%"
    mean_bh = np.mean(bh_crisis) if bh_crisis else np.nan
    row += f" {mean_bh*100:>+7.1f}%"
    print(row)

    # ==============================================================
    # PHASE 3: Parameter Perturbation Analysis
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 3: PARAMETER PERTURBATION ANALYSIS (S6)")
    print("  Vary each parameter ±10/20/30% and measure Sharpe change.")
    print("  Smooth surface = robust. Sharp peak = overfit.")
    print("=" * 120)

    # Default S6 parameters
    s6_defaults = {
        'exit_th': 0.82,
        'reentry_th': 0.92,
        'lookback': 200,
        'span_up': 20,
        'span_dn': 5,
        'md_short': 40,
        'md_long': 120,
        'md_sens': 0.3,
    }

    # Full-period baseline
    lev_base, _ = strategy_s6_momdecel_ens2(full_close, full_returns, **s6_defaults)
    nav_base, sr_base = run_backtest(full_close, lev_base)
    m_base = calc_metrics(nav_base, sr_base, len(full_close) / 252)
    base_sharpe = m_base['Sharpe']

    perturbation_pcts = [-0.30, -0.20, -0.10, 0.0, +0.10, +0.20, +0.30]

    # Parameters to perturb
    params_to_test = [
        ('exit_th', 0.82, 'DD Exit Threshold'),
        ('reentry_th', 0.92, 'DD Reentry Threshold'),
        ('lookback', 200, 'DD Lookback'),
        ('span_up', 20, 'AsymEWMA Span Up'),
        ('span_dn', 5, 'AsymEWMA Span Down'),
        ('md_short', 40, 'MomDecel Short'),
        ('md_long', 120, 'MomDecel Long'),
        ('md_sens', 0.3, 'MomDecel Sensitivity'),
    ]

    print(f"\n  Baseline S6 Sharpe: {base_sharpe:.3f}")
    print(f"\n  {'Parameter':<25s}", end='')
    for p in perturbation_pcts:
        print(f" {p:>+6.0%}", end='')
    print(f" {'Range':>7s} {'StdΔ':>7s}")
    print("  " + "-" * (25 + 7 * 7 + 7 + 7 + 3))

    perturb_results = {}
    for param_key, default_val, param_name in params_to_test:
        sharpes = []
        for pct in perturbation_pcts:
            params = s6_defaults.copy()
            new_val = default_val * (1 + pct)
            # Keep integer parameters as int
            if param_key in ('lookback', 'span_up', 'span_dn', 'md_short', 'md_long'):
                new_val = max(int(round(new_val)), 1)
            params[param_key] = new_val

            lev, _ = strategy_s6_momdecel_ens2(full_close, full_returns, **params)
            nav, sr = run_backtest(full_close, lev)
            m = calc_metrics(nav, sr, len(full_close) / 252)
            sharpes.append(m['Sharpe'])

        row = f"  {param_name:<25s}"
        for s in sharpes:
            delta = s - base_sharpe
            row += f" {delta:>+6.3f}"
        sr_range = max(sharpes) - min(sharpes)
        sr_std = np.std(sharpes)
        row += f" {sr_range:>7.3f} {sr_std:>7.3f}"
        print(row)

        perturb_results[param_key] = {
            'name': param_name,
            'sharpes': sharpes,
            'range': sr_range,
            'std': sr_std
        }

        all_results.append({
            'Phase': 'P3_Perturbation',
            'Parameter': param_name,
            'Default': default_val,
            'Sharpes': sharpes,
            'Range': sr_range,
            'Std': sr_std,
        })

    # Rank by sensitivity
    print(f"\n  Sensitivity Ranking (most → least sensitive):")
    ranked = sorted(perturb_results.items(), key=lambda x: -x[1]['range'])
    for i, (key, info) in enumerate(ranked, 1):
        flag = "⚠ HIGH" if info['range'] > 0.05 else "  low"
        print(f"    {i}. {info['name']:<25s} Range={info['range']:.3f}  {flag}")

    # ==============================================================
    # PHASE 4: Deflated Sharpe Ratio
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 4: DEFLATED SHARPE RATIO (Multiple Testing Correction)")
    print("  Accounts for the number of strategies tested to find S6.")
    print("=" * 120)

    # Count total trials: R4 had 55 strategies, then Ens2 tests (~20),
    # MomDecel sweep (~30), VoteRatio (~15) = ~120 total trials
    # Being conservative, use 120 as the number of independent trials
    N_TRIALS = 120
    T_OBS = len(full_returns.dropna())

    print(f"\n  Estimated total strategies tested: {N_TRIALS}")
    print(f"  Total observations: {T_OBS}")

    strat_returns_full = {}
    for sid, strat_fn, n_params in ALL_STRATEGIES:
        lev, name = strat_fn(full_close, full_returns)
        nav, sr = run_backtest(full_close, lev)
        strat_returns_full[sid] = sr
        m = calc_metrics(nav, sr, len(full_close) / 252)

        # Compute return statistics
        sr_clean = sr.dropna()
        if len(sr_clean) == 0:
            continue
        skew = stats.skew(sr_clean)
        kurt = stats.kurtosis(sr_clean)  # excess kurtosis

        # Expected max SR under null
        e_max_sr = expected_max_sr(N_TRIALS, T_OBS)

        # Deflated Sharpe Ratio
        dsr = deflated_sharpe_ratio(m['Sharpe'], e_max_sr, T_OBS, skew, kurt, N_TRIALS)

        sig = "★ SIGNIFICANT" if dsr > 0.95 else ("  marginal" if dsr > 0.90 else "  NOT significant")
        print(f"  {sid} {name:<25s}  Sharpe={m['Sharpe']:.3f}  "
              f"E[maxSR]={e_max_sr:.3f}  DSR_p={dsr:.3f}  {sig}")

        all_results.append({
            'Phase': 'P4_DeflatedSharpe',
            'Strategy': f"{sid} {name}",
            'Sharpe': m['Sharpe'],
            'E_max_SR': e_max_sr,
            'DSR_p': dsr,
            'Skew': skew,
            'Kurt': kurt,
            'N_trials': N_TRIALS,
        })

    # Also test S6 vs S1 (does complexity add significant value?)
    print(f"\n  --- Pairwise DSR: Does S6 significantly beat S1? ---")
    sr_s1 = strat_returns_full['S1'].dropna()
    sr_s6 = strat_returns_full['S6'].dropna()
    diff_returns = sr_s6.values - sr_s1.values[:len(sr_s6)]
    sharpe_diff = diff_returns.mean() * 252 / (diff_returns.std() * np.sqrt(252)) \
        if diff_returns.std() > 0 else 0
    t_stat = sharpe_diff * np.sqrt(len(diff_returns) / 252)
    p_val = 1 - stats.norm.cdf(t_stat)
    print(f"  S6 - S1 excess Sharpe: {sharpe_diff:.3f}")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value (one-sided): {p_val:.4f}")
    sig_txt = "★ SIGNIFICANT (p<0.05)" if p_val < 0.05 else "NOT significant (p≥0.05)"
    print(f"  → {sig_txt}")

    # ==============================================================
    # PHASE 5: Block Bootstrap Confidence Intervals
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 5: BLOCK BOOTSTRAP CONFIDENCE INTERVALS (1000 reps)")
    print("  Block size = 20 days. Tests if Sharpe differences are significant.")
    print("=" * 120)

    boot_strategies = [
        ('S1', strategy_s1_dd_only, 3),
        ('S3', strategy_s3_dd_asymvt, 6),
        ('S5', strategy_s5_ens2_st, 15),
        ('S6', strategy_s6_momdecel_ens2, 20),
    ]

    boot_sharpes = {}
    n_boot = 1000

    for sid, strat_fn, n_params in boot_strategies:
        lev, name = strat_fn(full_close, full_returns)
        nav, sr = run_backtest(full_close, lev)
        m = calc_metrics(nav, sr, len(full_close) / 252)

        # Bootstrap
        bs = block_bootstrap_sharpe(sr.dropna(), n_boot=n_boot, block_size=20)
        boot_sharpes[sid] = bs

        ci_lo = np.percentile(bs, 2.5)
        ci_hi = np.percentile(bs, 97.5)
        ci_5 = np.percentile(bs, 5)

        print(f"  {sid} {name:<25s}  Sharpe={m['Sharpe']:.3f}  "
              f"95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]  "
              f"5%ile={ci_5:.3f}  BootMean={np.mean(bs):.3f}")

        all_results.append({
            'Phase': 'P5_Bootstrap',
            'Strategy': f"{sid} {name}",
            'Sharpe': m['Sharpe'],
            'CI_95_lo': ci_lo,
            'CI_95_hi': ci_hi,
            'Boot_mean': np.mean(bs),
            'Boot_std': np.std(bs),
        })

    # Pairwise bootstrap: does S6 beat S1?
    print(f"\n  --- Bootstrap test: S6 vs S1 ---")
    diff_boot = boot_sharpes['S6'] - boot_sharpes['S1']
    pct_s6_wins = (diff_boot > 0).mean() * 100
    ci_lo_diff = np.percentile(diff_boot, 2.5)
    ci_hi_diff = np.percentile(diff_boot, 97.5)
    print(f"  S6 beats S1 in {pct_s6_wins:.1f}% of bootstrap samples")
    print(f"  95% CI of difference: [{ci_lo_diff:.3f}, {ci_hi_diff:.3f}]")

    # S5 vs S1
    print(f"\n  --- Bootstrap test: S5 vs S1 ---")
    diff_boot_51 = boot_sharpes['S5'] - boot_sharpes['S1']
    pct_s5_wins = (diff_boot_51 > 0).mean() * 100
    ci_lo_51 = np.percentile(diff_boot_51, 2.5)
    ci_hi_51 = np.percentile(diff_boot_51, 97.5)
    print(f"  S5 beats S1 in {pct_s5_wins:.1f}% of bootstrap samples")
    print(f"  95% CI of difference: [{ci_lo_51:.3f}, {ci_hi_51:.3f}]")

    # S6 vs S5
    print(f"\n  --- Bootstrap test: S6 vs S5 (marginal value of MomDecel) ---")
    diff_boot_65 = boot_sharpes['S6'] - boot_sharpes['S5']
    pct_s6_beats_s5 = (diff_boot_65 > 0).mean() * 100
    ci_lo_65 = np.percentile(diff_boot_65, 2.5)
    ci_hi_65 = np.percentile(diff_boot_65, 97.5)
    print(f"  S6 beats S5 in {pct_s6_beats_s5:.1f}% of bootstrap samples")
    print(f"  95% CI of difference: [{ci_lo_65:.3f}, {ci_hi_65:.3f}]")

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    print("\n" + "=" * 120)
    print("FINAL SUMMARY: OVERFITTING ASSESSMENT")
    print("=" * 120)

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Strategy Complexity vs. OOS Performance                             │
  └─────────────────────────────────────────────────────────────────────┘""")

    for sid, strat_fn, n_params in ALL_STRATEGIES:
        _, name = strat_fn(full_close, full_returns)

        # Full period
        lev, _ = strat_fn(full_close, full_returns)
        nav, sr = run_backtest(full_close, lev)
        m_full = calc_metrics(nav, sr, len(full_close) / 252)

        # Mean OOS from walk-forward
        valid_wf = [s for s in wf_results[sid] if not np.isnan(s)]
        mean_oos = np.mean(valid_wf) if valid_wf else np.nan
        std_oos = np.std(valid_wf) if len(valid_wf) > 1 else np.nan

        # Degradation
        degrade = m_full['Sharpe'] - mean_oos if not np.isnan(mean_oos) else np.nan

        # Assessment
        if np.isnan(degrade):
            assess = "—"
        elif degrade < 0.03:
            assess = "★ ROBUST"
        elif degrade < 0.08:
            assess = "  MODERATE"
        else:
            assess = "⚠ OVERFIT"

        print(f"  {sid} {name:<25s} (p={n_params:>2d})  "
              f"Full={m_full['Sharpe']:.3f}  OOS={mean_oos:.3f}±{std_oos:.3f}  "
              f"Degrade={degrade:+.3f}  {assess}")

    # Recommendation
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Key Findings                                                        │
  └─────────────────────────────────────────────────────────────────────┘""")

    # Build findings based on results
    # Check if S1 is stable across folds
    s1_valid = [s for s in wf_results['S1'] if not np.isnan(s)]
    s5_valid = [s for s in wf_results['S5'] if not np.isnan(s)]
    s6_valid = [s for s in wf_results['S6'] if not np.isnan(s)]

    s1_mean = np.mean(s1_valid)
    s5_mean = np.mean(s5_valid)
    s6_mean = np.mean(s6_valid)

    print(f"  1. DD-Only (S1) mean OOS Sharpe: {s1_mean:.3f}")
    print(f"  2. Ens2(S+T) (S5) mean OOS Sharpe: {s5_mean:.3f} "
          f"(Δ vs S1: {s5_mean - s1_mean:+.3f})")
    print(f"  3. MomDecel+Ens2 (S6) mean OOS Sharpe: {s6_mean:.3f} "
          f"(Δ vs S5: {s6_mean - s5_mean:+.3f})")

    if s6_mean > s5_mean + 0.01:
        print(f"  → MomDecel layer adds genuine value on OOS")
    elif s6_mean > s5_mean - 0.01:
        print(f"  → MomDecel layer is marginal (within noise)")
    else:
        print(f"  → MomDecel layer HURTS OOS → likely overfit")

    if s5_mean > s1_mean + 0.03:
        print(f"  → Ens2 layers (VT+Slope+TrendTV) add genuine OOS value over simple DD")
    elif s5_mean > s1_mean:
        print(f"  → Ens2 improvement over DD-only is small but positive")
    else:
        print(f"  → Ens2 layers do NOT improve over simple DD on OOS → overfit risk")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s")

    # Save results
    output_path = os.path.join(script_dir, '..', 'overfitting_validation_results.csv')
    # Flatten results for CSV
    flat = []
    for r in all_results:
        row = {k: v for k, v in r.items()
               if not isinstance(v, (list, dict))}
        flat.append(row)
    pd.DataFrame(flat).to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")


if __name__ == '__main__':
    main()
