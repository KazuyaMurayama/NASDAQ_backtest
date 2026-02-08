"""
Majority-Vote Ensemble Strategy - Phase 1: Sub-strategy Validation & Correlation

13 diverse binary timing strategies + correlation analysis.
Realistic conditions: 5-day delay, 1.5% annual cost.
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    load_data, calc_dd_signal, calc_metrics, calc_ewma_vol,
    calc_rolling_sharpe, calc_momentum_score, calc_vol_of_vol,
    calc_bollinger_bands
)
from test_ens2_strategies import calc_slope_multiplier

# =============================================================================
# Constants
# =============================================================================
ANNUAL_COST = 0.015
EXEC_DELAY = 5
BASE_LEV = 3.0

# =============================================================================
# Generic binary signal with hysteresis
# =============================================================================
def binary_signal_hysteresis(indicator: pd.Series,
                             exit_th: float, reentry_th: float,
                             exit_below: bool = True) -> pd.Series:
    """Generic hysteresis-based binary signal.

    exit_below=True:  indicator <= exit_th → CASH, indicator >= reentry_th → HOLD
    exit_below=False: indicator >= exit_th → CASH, indicator <= reentry_th → HOLD
    """
    signal = pd.Series(1.0, index=indicator.index)
    state = 'HOLD'

    for i in range(len(indicator)):
        val = indicator.iloc[i]
        if pd.isna(val):
            signal.iloc[i] = 1.0 if state == 'HOLD' else 0.0
            continue

        if exit_below:
            if state == 'HOLD' and val <= exit_th:
                state = 'CASH'
            elif state == 'CASH' and val >= reentry_th:
                state = 'HOLD'
        else:
            if state == 'HOLD' and val >= exit_th:
                state = 'CASH'
            elif state == 'CASH' and val <= reentry_th:
                state = 'HOLD'

        signal.iloc[i] = 1.0 if state == 'HOLD' else 0.0

    return signal


# =============================================================================
# 13 Sub-Strategies
# =============================================================================

def calc_S1_dd_standard(close):
    """S1: DD Standard (200d, exit=-18%, reentry=92%)"""
    return calc_dd_signal(close, 0.82, 0.92, lookback=200)

def calc_S2_dd_loose(close):
    """S2: DD Loose (200d, exit=-25%, reentry=95%)"""
    return calc_dd_signal(close, 0.75, 0.95, lookback=200)

def calc_S3_ma200_regime(close):
    """S3: MA200 Regime with hysteresis"""
    ma200 = close.rolling(200).mean()
    ratio = close / ma200
    return binary_signal_hysteresis(ratio, exit_th=0.97, reentry_th=1.03)

def calc_S4_dual_ma_cross(close):
    """S4: Dual MA Cross (MA50 vs MA150) with hysteresis"""
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ratio = ma50 / ma150
    return binary_signal_hysteresis(ratio, exit_th=0.98, reentry_th=1.02)

def calc_S5_momentum_120(close):
    """S5: Mid-term Momentum (120-day return)"""
    mom = calc_momentum_score(close, 120)
    return binary_signal_hysteresis(mom, exit_th=-0.08, reentry_th=-0.03)

def calc_S6_momentum_250(close):
    """S6: Long-term Momentum (250-day return)"""
    mom = calc_momentum_score(close, 250)
    return binary_signal_hysteresis(mom, exit_th=-0.15, reentry_th=-0.05)

def calc_S7_vol_regime(returns):
    """S7: EWMA Vol Regime"""
    vol = calc_ewma_vol(returns, span=10)
    # exit_below=False: vol >= 0.38 → CASH
    return binary_signal_hysteresis(vol, exit_th=0.38, reentry_th=0.28, exit_below=False)

def calc_S8_rolling_sharpe(returns):
    """S8: Rolling Sharpe (90-day)"""
    rs = calc_rolling_sharpe(returns, 90)
    return binary_signal_hysteresis(rs, exit_th=-0.8, reentry_th=-0.2)

def calc_S9_bb_lower(close):
    """S9: Bollinger Band Lower (200-day)"""
    ma, upper, lower = calc_bollinger_bands(close, 200, 2.0)
    std = close.rolling(200).std()
    # Normalize: how many std below MA
    z = (close - ma) / std
    # Exit if z < -1.8, reentry if z > -0.5
    return binary_signal_hysteresis(z, exit_th=-1.8, reentry_th=-0.5)

def calc_S10_vov(returns):
    """S10: Vol-of-Vol"""
    vov = calc_vol_of_vol(returns, vol_lookback=20, vov_lookback=60)
    vov_median = vov.rolling(500, min_periods=100).median()
    ratio = vov / vov_median
    # exit_below=False: ratio >= 2.0 → CASH
    return binary_signal_hysteresis(ratio, exit_th=2.0, reentry_th=1.2, exit_below=False)

def calc_S11_ma_slope(close):
    """S11: MA200 Slope Direction (Z-score)"""
    ma = close.rolling(200).mean()
    slope = ma.pct_change()
    slope_mean = slope.rolling(60).mean()
    slope_std = slope.rolling(60).std().replace(0, 0.0001)
    z = (slope - slope_mean) / slope_std
    return binary_signal_hysteresis(z, exit_th=-1.0, reentry_th=-0.3)

def calc_S12_seasonal(dates):
    """S12: Seasonal Filter (Sep-Oct → CASH)"""
    months = pd.to_datetime(dates).dt.month
    signal = pd.Series(1.0, index=dates.index)
    signal[(months == 9) | (months == 10)] = 0.0
    return signal

def calc_S13_dd_short(close):
    """S13: DD Short (100d, exit=-12%, reentry=95%)"""
    peak = close.rolling(100, min_periods=1).max()
    ratio = close / peak
    return binary_signal_hysteresis(ratio, exit_th=0.88, reentry_th=0.95)


# =============================================================================
# Backtest runner
# =============================================================================
def run_backtest(close, leverage, base_lev=BASE_LEV, cost=ANNUAL_COST, delay=EXEC_DELAY):
    returns = close.pct_change()
    lev_returns = returns * base_lev
    daily_cost = cost / 252
    delayed = leverage.shift(delay)
    strat_ret = delayed * (lev_returns - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret


def count_trades(signal):
    return (signal.diff().abs() > 0.5).sum()


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

    print("=" * 140)
    print("MAJORITY-VOTE ENSEMBLE - Phase 1: Sub-Strategy Validation")
    print(f"Conditions: {EXEC_DELAY}-day delay, {ANNUAL_COST*100:.1f}% annual cost")
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()} ({len(df)} days)")
    print("=" * 140)

    # =====================================================================
    # Step 1.1: Build all 13 sub-strategies
    # =====================================================================
    strategies = {
        'S1:  DD Std(200d)':     calc_S1_dd_standard(close),
        'S2:  DD Loose(200d)':   calc_S2_dd_loose(close),
        'S3:  MA200 Regime':     calc_S3_ma200_regime(close),
        'S4:  Dual MA Cross':    calc_S4_dual_ma_cross(close),
        'S5:  Mom 120d':         calc_S5_momentum_120(close),
        'S6:  Mom 250d':         calc_S6_momentum_250(close),
        'S7:  Vol Regime':       calc_S7_vol_regime(returns),
        'S8:  Roll Sharpe':      calc_S8_rolling_sharpe(returns),
        'S9:  BB Lower':         calc_S9_bb_lower(close),
        'S10: VoV':              calc_S10_vov(returns),
        'S11: MA200 Slope':      calc_S11_ma_slope(close),
        'S12: Seasonal':         calc_S12_seasonal(dates),
        'S13: DD Short(100d)':   calc_S13_dd_short(close),
    }

    # =====================================================================
    # Step 1.1: Individual backtests
    # =====================================================================
    print("\n" + "=" * 140)
    print("STEP 1.1: INDIVIDUAL SUB-STRATEGY PERFORMANCE")
    print("=" * 140)

    print(f"\n{'Strategy':<24s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'W5Y':>8s} "
          f"{'Sortino':>8s} {'Trades':>7s} {'HOLD%':>7s} {'Category'}")
    print("-" * 140)

    categories = {
        'S1':  'Drawdown', 'S2':  'Drawdown', 'S3':  'Trend',
        'S4':  'Trend',    'S5':  'Momentum', 'S6':  'Momentum',
        'S7':  'Volatility','S8': 'RiskAdj',  'S9':  'Statistical',
        'S10': 'VolStruct', 'S11': 'TrendStr', 'S12': 'Seasonal',
        'S13': 'Drawdown',
    }

    metrics_all = {}
    for name, sig in strategies.items():
        nav, strat_ret = run_backtest(close, sig)
        m = calc_metrics(nav, strat_ret, sig, dates)
        trades = count_trades(sig)
        hold_pct = sig.mean() * 100
        cat = categories[name[:3].strip().rstrip(':')]
        metrics_all[name] = m

        print(f"{name:<24s} {m['Sharpe']:>7.3f} {m['CAGR']*100:>+7.1f}% "
              f"{m['MaxDD']*100:>7.1f}% {m['Worst5Y']*100:>+7.1f}% "
              f"{m['Sortino']:>8.3f} {trades:>7d} {hold_pct:>6.1f}%  {cat}")

    # =====================================================================
    # Step 1.2: Pairwise Correlation Matrix
    # =====================================================================
    print("\n" + "=" * 140)
    print("STEP 1.2: PAIRWISE SIGNAL CORRELATION MATRIX")
    print("=" * 140)

    sig_df = pd.DataFrame(strategies)
    corr_matrix = sig_df.corr()

    # Short names for display
    short_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13']
    corr_display = corr_matrix.copy()
    corr_display.index = short_names
    corr_display.columns = short_names

    print(f"\n{'':>5s}", end="")
    for sn in short_names:
        print(f" {sn:>5s}", end="")
    print()
    print("-" * (5 + 6 * len(short_names)))

    for i, sn in enumerate(short_names):
        print(f"{sn:>5s}", end="")
        for j in range(len(short_names)):
            val = corr_display.iloc[i, j]
            if i == j:
                print(f"  1.00", end="")
            else:
                print(f" {val:>5.2f}", end="")
        print()

    # Average pairwise correlation per strategy
    print(f"\nAverage correlation with others:")
    for i, sn in enumerate(short_names):
        others = [corr_display.iloc[i, j] for j in range(len(short_names)) if i != j]
        avg = np.mean(others)
        name = list(strategies.keys())[i]
        print(f"  {name:<24s} avg_corr={avg:.3f}")

    # =====================================================================
    # Step 1.3: Low-correlation pairs (best diversity)
    # =====================================================================
    print(f"\nLowest correlation pairs (top 15):")
    pairs = []
    for i in range(len(short_names)):
        for j in range(i+1, len(short_names)):
            pairs.append((short_names[i], short_names[j], corr_display.iloc[i, j]))
    pairs.sort(key=lambda x: x[2])

    for s1, s2, c in pairs[:15]:
        n1 = list(strategies.keys())[short_names.index(s1)]
        n2 = list(strategies.keys())[short_names.index(s2)]
        print(f"  {s1}-{s2}: {c:+.3f}  ({n1} vs {n2})")

    print(f"\nHighest correlation pairs (top 10):")
    for s1, s2, c in pairs[-10:]:
        n1 = list(strategies.keys())[short_names.index(s1)]
        n2 = list(strategies.keys())[short_names.index(s2)]
        print(f"  {s1}-{s2}: {c:+.3f}  ({n1} vs {n2})")

    # =====================================================================
    # Step 1.3: Crisis Period Signal Analysis
    # =====================================================================
    print("\n" + "=" * 140)
    print("STEP 1.3: CRISIS PERIOD SIGNAL ANALYSIS")
    print("(Number of strategies signaling CASH during crisis months)")
    print("=" * 140)

    crisis_periods = [
        ('1987 Black Monday', '1987-09-01', '1987-12-31'),
        ('2000 Dot-com Start', '2000-03-01', '2000-06-30'),
        ('2001 Dot-com Deep', '2001-01-01', '2001-06-30'),
        ('2008 Lehman', '2008-09-01', '2008-12-31'),
        ('2020 Covid', '2020-02-15', '2020-04-30'),
    ]

    for label, start, end in crisis_periods:
        mask = (dates >= start) & (dates <= end)
        if mask.sum() == 0:
            continue

        print(f"\n--- {label} ({start} to {end}) ---")
        print(f"{'Strategy':<24s} {'CASH days':>10s} {'Total days':>11s} {'CASH %':>8s}")

        for name, sig in strategies.items():
            period_sig = sig[mask]
            cash_days = (period_sig < 0.5).sum()
            total = len(period_sig)
            pct = cash_days / total * 100 if total > 0 else 0
            marker = " <<<" if pct > 50 else ""
            print(f"{name:<24s} {cash_days:>10d} {total:>11d} {pct:>7.1f}%{marker}")

        # Vote summary
        vote_sum = sum(sig[mask] for sig in strategies.values())
        avg_hold = vote_sum.mean()
        print(f"{'Average HOLD votes':<24s} {avg_hold:.1f}/13")

    # =====================================================================
    # Step 1.4: Agreement Analysis (how often do strategies agree?)
    # =====================================================================
    print("\n" + "=" * 140)
    print("STEP 1.4: VOTE DISTRIBUTION ANALYSIS")
    print("(How many strategies signal HOLD on each day?)")
    print("=" * 140)

    total_votes = sum(sig for sig in strategies.values())

    print(f"\n{'HOLD votes':>11s} {'Days':>8s} {'%':>7s} {'Cumulative %':>13s}")
    print("-" * 45)
    cumul = 0
    for v in range(14):
        days = (total_votes == v).sum()
        pct = days / len(total_votes) * 100
        cumul += pct
        marker = " <-- majority threshold" if v == 7 else ""
        print(f"{v:>11d} {days:>8d} {pct:>6.1f}% {cumul:>12.1f}%{marker}")

    print(f"\nIf majority = 7/13 (>50%): HOLD {(total_votes >= 7).mean()*100:.1f}% of time")
    print(f"If supermaj = 9/13 (>69%): HOLD {(total_votes >= 9).mean()*100:.1f}% of time")
    print(f"If supermaj = 10/13 (>76%): HOLD {(total_votes >= 10).mean()*100:.1f}% of time")

    # =====================================================================
    # Save signals for Phase 2
    # =====================================================================
    output_path = os.path.join(script_dir, '..', 'majority_vote_signals.csv')
    save_df = sig_df.copy()
    save_df.insert(0, 'Date', dates.values)
    save_df.to_csv(output_path, index=False)
    print(f"\nSignals saved to {output_path}")

    # Save correlation matrix
    corr_path = os.path.join(script_dir, '..', 'signal_correlation_matrix.csv')
    corr_display.to_csv(corr_path)
    print(f"Correlation matrix saved to {corr_path}")


if __name__ == "__main__":
    main()
