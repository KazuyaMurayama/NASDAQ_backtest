"""
Discrete Leverage Backtest: Ens2 strategies with realistic product constraints.

User constraints:
- Available products: 3x, 2x, 1x NASDAQ (or cash = 0x)
- Switching between products takes ~7 business days
- Signal execution starts the next day

Test patterns:
A) Binary: 0x / 3x only (simplest)
B) 3-level: 0x / 1x / 3x
C) 4-level: 0x / 1x / 2x / 3x
D) Binary with 7-day switching delay
E) 4-level with 7-day switching delay
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import load_data, calc_dd_signal, calc_metrics
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier,
    calc_trend_target_vol, strategy_ens2_asym_slope, strategy_ens2_slope_trendtv
)


# =============================================================================
# Discretization Functions
# =============================================================================
def discretize_binary(leverage: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Binary: 0x or 3x (leverage 0.0 or 1.0)
    If continuous leverage > threshold -> 1.0 (= 3x product)
    Else -> 0.0 (= cash)
    """
    return (leverage > threshold).astype(float)


def discretize_3level(leverage: pd.Series) -> pd.Series:
    """3-level: 0x / 1x / 3x (leverage 0.0, 0.333, 1.0)
    Boundaries: <0.17 -> 0x, 0.17~0.67 -> 1x, >0.67 -> 3x
    """
    result = pd.Series(0.0, index=leverage.index)
    result[leverage > 0.17] = 1/3     # 1x product (1/3 of 3x = 1x effective)
    result[leverage > 0.67] = 1.0     # 3x product
    return result


def discretize_4level(leverage: pd.Series) -> pd.Series:
    """4-level: 0x / 1x / 2x / 3x (leverage 0.0, 0.333, 0.667, 1.0)
    Boundaries: <0.17 -> 0x, 0.17~0.5 -> 1x, 0.5~0.83 -> 2x, >0.83 -> 3x
    """
    result = pd.Series(0.0, index=leverage.index)
    result[leverage > 0.17] = 1/3     # 1x
    result[leverage > 0.50] = 2/3     # 2x
    result[leverage > 0.83] = 1.0     # 3x
    return result


def apply_switching_delay(leverage: pd.Series, delay_days: int = 7) -> pd.Series:
    """Simulate switching delay: when target changes, actual position
    transitions after delay_days business days.

    During transition, we stay in the OLD position until the switch completes.
    """
    result = pd.Series(0.0, index=leverage.index)
    current_lev = leverage.iloc[0]
    result.iloc[0] = current_lev
    switch_target = None
    switch_countdown = 0

    for i in range(1, len(leverage)):
        target_lev = leverage.iloc[i]

        if switch_countdown > 0:
            # Currently switching - stay in old position
            switch_countdown -= 1
            if switch_countdown == 0:
                # Switch complete
                current_lev = switch_target
                switch_target = None
            result.iloc[i] = current_lev
        elif target_lev != current_lev:
            # New switch needed
            switch_target = target_lev
            switch_countdown = delay_days - 1  # -1 because next day starts
            result.iloc[i] = current_lev  # Stay in old position during switch
        else:
            result.iloc[i] = current_lev

    return result


# =============================================================================
# Run backtest (same as engine but accepts discrete leverage)
# =============================================================================
def run_backtest_discrete(close: pd.Series, leverage: pd.Series,
                          base_leverage: float = 3.0,
                          annual_cost: float = 0.009) -> tuple:
    """Run backtest with discrete leverage.
    Cost model: different cost rates for different products.
    For simplicity, use same 0.9% annual cost when invested.
    """
    returns = close.pct_change()
    leveraged_returns = returns * base_leverage
    daily_cost = annual_cost / 252

    # For discrete products, the effective return depends on which product
    # leverage=1.0 -> 3x product -> 3x returns
    # leverage=2/3 -> 2x product -> 2x returns
    # leverage=1/3 -> 1x product -> 1x returns
    # leverage=0.0 -> cash -> 0 returns

    # The leverage already represents the fraction of 3x,
    # so leverage * 3x_returns gives the correct effective return
    strategy_returns = leverage.shift(1) * (leveraged_returns - daily_cost)
    strategy_returns = strategy_returns.fillna(0)

    nav = (1 + strategy_returns).cumprod()
    return nav, strategy_returns


# =============================================================================
# Count product switches
# =============================================================================
def count_switches(leverage: pd.Series) -> int:
    """Count how many times the discrete leverage level changes."""
    changes = leverage.diff().abs()
    return (changes > 0.01).sum()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 110)
    print("DISCRETE LEVERAGE BACKTEST - Ens2 Strategies")
    print("=" * 110)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'NASDAQ_Dairy_since1973.csv')
    df = load_data(data_path)
    print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total trading days: {len(df)}\n")

    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    results = []

    def add_result(lev, dd_signal, name, category):
        nav, strat_ret = run_backtest_discrete(close, lev)
        metrics = calc_metrics(nav, strat_ret, dd_signal, dates)
        switches = count_switches(lev)
        metrics['Strategy'] = name
        metrics['Category'] = category
        metrics['Switches'] = switches
        results.append(metrics)
        print(f"  {name}")
        print(f"    CAGR={metrics['CAGR']*100:.2f}%, Sharpe={metrics['Sharpe']:.3f}, "
              f"MaxDD={metrics['MaxDD']*100:.2f}%, Worst5Y={metrics['Worst5Y']*100:.2f}%, "
              f"Switches={switches}")
        return nav

    # =========================================================================
    # Strategy 1: Ens2(Asym+Slope) - Rank 1
    # =========================================================================
    print("=" * 110)
    print("STRATEGY 1: Ens2(Asym+Slope)")
    print("=" * 110)

    # Original continuous
    lev_cont_1, dd_1 = strategy_ens2_asym_slope(
        close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    print("\n[Original - Continuous Leverage]")
    add_result(lev_cont_1, dd_1, 'Ens2(Asym+Slope) Continuous', 'Original')

    # A) Binary 0x/3x
    print("\n[A] Binary: 0x / 3x")
    for th in [0.3, 0.5, 0.7]:
        lev_bin = discretize_binary(lev_cont_1, threshold=th)
        add_result(lev_bin, dd_1, f'Binary(th={th})', f'A-Binary')

    # B) 3-level 0x/1x/3x
    print("\n[B] 3-level: 0x / 1x / 3x")
    lev_3lv = discretize_3level(lev_cont_1)
    add_result(lev_3lv, dd_1, '3-Level(0x/1x/3x)', 'B-3Level')

    # C) 4-level 0x/1x/2x/3x
    print("\n[C] 4-level: 0x / 1x / 2x / 3x")
    lev_4lv = discretize_4level(lev_cont_1)
    add_result(lev_4lv, dd_1, '4-Level(0x/1x/2x/3x)', 'C-4Level')

    # D) Binary 0x/3x + 7-day delay
    print("\n[D] Binary 0x/3x + 7-day switching delay")
    for th in [0.3, 0.5]:
        lev_bin = discretize_binary(lev_cont_1, threshold=th)
        lev_delayed = apply_switching_delay(lev_bin, delay_days=7)
        add_result(lev_delayed, dd_1, f'Binary(th={th})+7day', 'D-Delayed')

    # E) 4-level + 7-day delay
    print("\n[E] 4-level + 7-day switching delay")
    lev_4lv_delayed = apply_switching_delay(lev_4lv, delay_days=7)
    add_result(lev_4lv_delayed, dd_1, '4-Level+7day', 'E-Delayed')

    # =========================================================================
    # Strategy 2: Ens2(Slope+TrendTV) - Rank 2
    # =========================================================================
    print("\n" + "=" * 110)
    print("STRATEGY 2: Ens2(Slope+TrendTV)")
    print("=" * 110)

    # Original continuous
    lev_cont_2, dd_2 = strategy_ens2_slope_trendtv(
        close, returns, 0.82, 0.92, 20, 5, 1.0)
    print("\n[Original - Continuous Leverage]")
    add_result(lev_cont_2, dd_2, 'Ens2(Slope+TrendTV) Continuous', 'Original')

    # A) Binary 0x/3x
    print("\n[A] Binary: 0x / 3x")
    for th in [0.3, 0.5, 0.7]:
        lev_bin = discretize_binary(lev_cont_2, threshold=th)
        add_result(lev_bin, dd_2, f'Binary(th={th})', f'A-Binary')

    # B) 3-level
    print("\n[B] 3-level: 0x / 1x / 3x")
    lev_3lv = discretize_3level(lev_cont_2)
    add_result(lev_3lv, dd_2, '3-Level(0x/1x/3x)', 'B-3Level')

    # C) 4-level
    print("\n[C] 4-level: 0x / 1x / 2x / 3x")
    lev_4lv = discretize_4level(lev_cont_2)
    add_result(lev_4lv, dd_2, '4-Level(0x/1x/2x/3x)', 'C-4Level')

    # D) Binary + 7-day delay
    print("\n[D] Binary 0x/3x + 7-day switching delay")
    lev_bin = discretize_binary(lev_cont_2, threshold=0.5)
    lev_delayed = apply_switching_delay(lev_bin, delay_days=7)
    add_result(lev_delayed, dd_2, 'Binary(th=0.5)+7day', 'D-Delayed')

    # E) 4-level + 7-day delay
    print("\n[E] 4-level + 7-day switching delay")
    lev_4lv_delayed = apply_switching_delay(lev_4lv, delay_days=7)
    add_result(lev_4lv_delayed, dd_2, '4-Level+7day', 'E-Delayed')

    # =========================================================================
    # Leverage Distribution Analysis
    # =========================================================================
    print("\n" + "=" * 110)
    print("LEVERAGE DISTRIBUTION ANALYSIS")
    print("=" * 110)

    for name, lev in [('Ens2(Asym+Slope)', lev_cont_1),
                       ('Ens2(Slope+TrendTV)', lev_cont_2)]:
        print(f"\n[{name}] Continuous leverage distribution:")
        # Only look at HOLD periods (DD=1)
        dd = dd_1 if 'Asym' in name else dd_2
        hold_lev = lev[dd > 0.5]
        print(f"  CASH periods (DD=0): {(dd < 0.5).sum()} days "
              f"({(dd < 0.5).mean()*100:.1f}%)")
        print(f"  HOLD periods (DD=1): {(dd > 0.5).sum()} days "
              f"({(dd > 0.5).mean()*100:.1f}%)")
        print(f"  During HOLD - leverage distribution:")
        print(f"    Mean:   {hold_lev.mean():.3f} (= {hold_lev.mean()*3:.1f}x effective)")
        print(f"    Median: {hold_lev.median():.3f} (= {hold_lev.median()*3:.1f}x effective)")
        print(f"    Std:    {hold_lev.std():.3f}")
        print(f"    Min:    {hold_lev.min():.3f}")
        print(f"    Max:    {hold_lev.max():.3f}")

        # Bucket distribution
        buckets = [(0, 0.17, '0x zone'),
                   (0.17, 0.50, '1x zone'),
                   (0.50, 0.83, '2x zone'),
                   (0.83, 1.01, '3x zone')]
        print(f"    Bucket distribution (HOLD periods):")
        for lo, hi, label in buckets:
            pct = ((hold_lev >= lo) & (hold_lev < hi)).mean() * 100
            print(f"      {label} ({lo:.2f}-{hi:.2f}): {pct:.1f}%")

    # =========================================================================
    # Summary Comparison Table
    # =========================================================================
    print("\n" + "=" * 110)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 110)

    results_df = pd.DataFrame(results)
    cols = ['Category', 'Strategy', 'CAGR', 'Sharpe', 'MaxDD', 'Worst5Y', 'Switches']
    results_df = results_df[cols]

    # Split by strategy
    # First half = Ens2(Asym+Slope), second half = Ens2(Slope+TrendTV)
    mid = len(results_df) // 2 + 1  # +1 because first strategy has more rows

    for strat_name, start, end in [
        ('Ens2(Asym+Slope)', 0, 10),
        ('Ens2(Slope+TrendTV)', 10, 20)
    ]:
        strat_results = results_df.iloc[start:end] if end <= len(results_df) else results_df.iloc[start:]
        if len(strat_results) == 0:
            continue

        print(f"\n--- {strat_name} ---")
        display_df = strat_results.copy()
        for col in ['CAGR', 'MaxDD', 'Worst5Y']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
        display_df['Sharpe'] = display_df['Sharpe'].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        display_df['Switches'] = display_df['Switches'].astype(int)
        print(display_df.to_string(index=False))

    # =========================================================================
    # DD-Only Strategy (simplest feasible baseline)
    # =========================================================================
    print("\n" + "=" * 110)
    print("REFERENCE: DD-ONLY STRATEGY (Simplest feasible - just HOLD 3x or CASH)")
    print("=" * 110)

    dd_only = calc_dd_signal(close, 0.82, 0.92)
    nav_dd, ret_dd = run_backtest_discrete(close, dd_only)
    metrics_dd = calc_metrics(nav_dd, ret_dd, dd_only, dates)
    switches_dd = count_switches(dd_only)
    print(f"  DD-Only (3x or Cash): CAGR={metrics_dd['CAGR']*100:.2f}%, "
          f"Sharpe={metrics_dd['Sharpe']:.3f}, MaxDD={metrics_dd['MaxDD']*100:.2f}%, "
          f"Worst5Y={metrics_dd['Worst5Y']*100:.2f}%, Switches={switches_dd}")

    # DD-Only with 7-day delay
    dd_delayed = apply_switching_delay(dd_only, delay_days=7)
    nav_dd_d, ret_dd_d = run_backtest_discrete(close, dd_delayed)
    metrics_dd_d = calc_metrics(nav_dd_d, ret_dd_d, dd_only, dates)
    switches_dd_d = count_switches(dd_delayed)
    print(f"  DD-Only + 7day delay: CAGR={metrics_dd_d['CAGR']*100:.2f}%, "
          f"Sharpe={metrics_dd_d['Sharpe']:.3f}, MaxDD={metrics_dd_d['MaxDD']*100:.2f}%, "
          f"Worst5Y={metrics_dd_d['Worst5Y']*100:.2f}%, Switches={switches_dd_d}")

    # Save results
    output_path = os.path.join(script_dir, '..', 'discrete_leverage_results.csv')
    full_results = pd.DataFrame(results)
    full_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return results_df


if __name__ == "__main__":
    results = main()
