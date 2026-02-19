"""
External Signal Enhancement Test
=================================
Test whether external data (HY spread, 10Y yield, VIX) can improve
the MomDecel+Ens2(S+T) strategy.

Approach: Add each signal as a 5th multiplier layer (0.5-1.3 range),
keeping the existing 4 layers intact.

Data availability:
  - VIX: 1990-01-02 ~ present (Yahoo Finance)
  - 10Y yield: 1990-01-02 ~ present (FRED)
  - HY spread: 1996-12-31 ~ present (FRED)
  - Backtest common period: 1997-01 ~ present (~29 years)
"""
import pandas as pd
import numpy as np
import urllib.request
import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import load_data, calc_dd_signal
from test_ens2_strategies import (
    calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
)
from test_delay_robust import (
    calc_momentum_decel_mult, rebalance_threshold
)

ANNUAL_COST = 0.015
EXEC_DELAY = 5
BASE_LEV = 3.0


# =================================================================
# Data Fetching
# =================================================================
def fetch_yahoo(symbol, start='1990-01-01'):
    """Fetch daily data from Yahoo Finance."""
    start_ts = int(time.mktime(time.strptime(start, '%Y-%m-%d')))
    end_ts = int(time.time())
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
           f'?period1={start_ts}&period2={end_ts}&interval=1d')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    closes = result['indicators']['quote'][0]['close']
    df = pd.DataFrame({
        'Date': [pd.Timestamp.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
        'Value': closes
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Value']).reset_index(drop=True)
    return df


def fetch_fred_csv(series_id, start='1990-01-01'):
    """Fetch daily data from FRED CSV endpoint."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    text = resp.read().decode('utf-8')
    lines = text.strip().split('\n')

    rows = []
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) == 2 and parts[1] != '.':
            try:
                rows.append({'Date': pd.to_datetime(parts[0]),
                             'Value': float(parts[1])})
            except ValueError:
                continue
    return pd.DataFrame(rows)


# =================================================================
# Signal Multiplier Designs
# =================================================================
def calc_hy_spread_mult(hy_spread, lookback=120, sensitivity=0.3,
                        min_mult=0.3, max_mult=1.2):
    """HY spread multiplier: rising spread → reduce leverage.

    When credit stress increases (spread widens), reduce exposure.
    Uses Z-score of spread CHANGE (not level) to be regime-agnostic.
    """
    # Rate of change of spread (higher = worsening)
    spread_change = hy_spread.pct_change(20)  # 20-day change

    # Z-score normalize
    z_mean = spread_change.rolling(lookback).mean()
    z_std = spread_change.rolling(lookback).std().replace(0, 0.001)
    z = (spread_change - z_mean) / z_std

    # Negative z (spread tightening) → mult > 1 (boost)
    # Positive z (spread widening) → mult < 1 (reduce)
    mult = (1.0 - sensitivity * z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


def calc_yield_change_mult(yield_10y, lookback=120, sensitivity=0.3,
                           min_mult=0.3, max_mult=1.2):
    """10Y yield change multiplier: rapid yield rise → reduce leverage.

    Rapid rate increases hurt growth stocks (NASDAQ).
    Uses Z-score of yield change over 60 days.
    """
    yield_change = yield_10y.diff(60)  # 60-day absolute change in yield

    z_mean = yield_change.rolling(lookback).mean()
    z_std = yield_change.rolling(lookback).std().replace(0, 0.001)
    z = (yield_change - z_mean) / z_std

    # Positive z (rates rising fast) → reduce
    mult = (1.0 - sensitivity * z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


def calc_vix_regime_mult(vix, lookback=120, sensitivity=0.3,
                         min_mult=0.3, max_mult=1.2):
    """VIX regime multiplier: elevated VIX → reduce leverage.

    Uses VIX level relative to its rolling distribution.
    Complementary to AsymEWMA (which uses realized vol).
    """
    z_mean = vix.rolling(lookback).mean()
    z_std = vix.rolling(lookback).std().replace(0, 0.001)
    z = (vix - z_mean) / z_std

    # Positive z (elevated VIX) → reduce
    mult = (1.0 - sensitivity * z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


def calc_yield_curve_mult(t10y2y, lookback=120, sensitivity=0.3,
                          min_mult=0.3, max_mult=1.2):
    """Yield curve multiplier: inversion or flattening → reduce leverage.

    10Y-2Y spread flattening/inversion signals recession risk.
    """
    z_mean = t10y2y.rolling(lookback).mean()
    z_std = t10y2y.rolling(lookback).std().replace(0, 0.001)
    z = (t10y2y - z_mean) / z_std

    # Negative z (flattening/inversion) → reduce
    mult = (1.0 + sensitivity * z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


# =================================================================
# Strategy builder
# =================================================================
def build_base_strategy(close, returns):
    """Build MomDecel+Ens2(S+T) base leverage (without external signals)."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92, 200)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    decel_mult = calc_momentum_decel_mult(close, 40, 120, 0.3)
    leverage = dd_signal * vt_lev * slope_mult * decel_mult
    return leverage.clip(0, 1.0).fillna(0), dd_signal


def run_backtest(close, leverage, base_lev=BASE_LEV, cost=ANNUAL_COST, delay=EXEC_DELAY):
    returns = close.pct_change()
    lev_returns = returns * base_lev
    daily_cost = cost / 252
    delayed = leverage.shift(delay)
    strat_ret = delayed * (lev_returns - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret


def calc_summary(nav, strat_ret, n_years, leverage):
    sharpe = strat_ret.mean() * 252 / (strat_ret.std() * np.sqrt(252))
    cagr = nav.iloc[-1] ** (1 / n_years) - 1
    maxdd = ((nav - nav.cummax()) / nav.cummax()).min()
    w5y = ((nav / nav.shift(252 * 5)) ** 0.2 - 1).min()
    trades = (leverage.diff().abs() > 0.01).sum()
    return {
        'Sharpe': sharpe, 'CAGR': cagr, 'MaxDD': maxdd,
        'Worst5Y': w5y, 'Trades': trades, 'Trades/yr': trades / n_years
    }


# =================================================================
# Main
# =================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 120)
    print("EXTERNAL SIGNAL ENHANCEMENT TEST")
    print("Conditions: 5-day delay, 1.5% annual cost, rebalance threshold 20%")
    print("=" * 120)

    # --- Fetch all external data ---
    print("\n--- Fetching external data ---")
    vix_df = fetch_yahoo('%5EVIX')
    print(f"  VIX: {len(vix_df)} rows ({vix_df['Date'].min().date()} to {vix_df['Date'].max().date()})")

    tnx_df = fetch_yahoo('%5ETNX')
    print(f"  10Y yield (Yahoo): {len(tnx_df)} rows ({tnx_df['Date'].min().date()} to {tnx_df['Date'].max().date()})")

    hy_df = fetch_fred_csv('BAMLH0A0HYM2')
    print(f"  HY spread (FRED): {len(hy_df)} rows ({hy_df['Date'].min().date()} to {hy_df['Date'].max().date()})")

    t10y2y_df = fetch_fred_csv('T10Y2Y')
    print(f"  10Y-2Y curve (FRED): {len(t10y2y_df)} rows ({t10y2y_df['Date'].min().date()} to {t10y2y_df['Date'].max().date()})")

    # --- Load NASDAQ data ---
    data_path = os.path.join(script_dir, '..', 'NASDAQ_extended.csv')
    df = load_data(data_path)
    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    # --- Align all data to NASDAQ dates ---
    master = pd.DataFrame({'Date': dates, 'Close': close.values}).set_index('Date')

    for name, ext_df in [('VIX', vix_df), ('TNX', tnx_df),
                          ('HY', hy_df), ('T10Y2Y', t10y2y_df)]:
        ext = ext_df.set_index('Date')['Value']
        master[name] = ext.reindex(master.index).ffill()

    # Common period: all signals available
    common_start = master[['VIX', 'TNX', 'HY', 'T10Y2Y']].dropna().index.min()
    print(f"\n  Common period start: {common_start.date()}")

    # Use common period for fair comparison
    mask = master.index >= common_start
    close_c = pd.Series(master.loc[mask, 'Close'].values,
                        index=range(mask.sum()))
    dates_c = pd.Series(master.loc[mask].index.values,
                        index=range(mask.sum()))
    returns_c = close_c.pct_change()
    n_years = len(close_c) / 252

    vix_c = pd.Series(master.loc[mask, 'VIX'].values, index=range(mask.sum()))
    tnx_c = pd.Series(master.loc[mask, 'TNX'].values, index=range(mask.sum()))
    hy_c = pd.Series(master.loc[mask, 'HY'].values, index=range(mask.sum()))
    t10y2y_c = pd.Series(master.loc[mask, 'T10Y2Y'].values, index=range(mask.sum()))

    print(f"  Common period: {pd.Timestamp(dates_c.iloc[0]).date()} to "
          f"{pd.Timestamp(dates_c.iloc[-1]).date()} ({n_years:.1f} years)")

    # --- Build base strategy ---
    base_lev_raw, dd_sig = build_base_strategy(close_c, returns_c)

    # =================================================================
    # PART 1: Individual signal tests (parameter sweep)
    # =================================================================
    print("\n" + "=" * 120)
    print("PART 1: INDIVIDUAL SIGNAL TESTS")
    print("=" * 120)

    # Baseline
    base_lev_th = rebalance_threshold(base_lev_raw, 0.20)
    nav_base, sr_base = run_backtest(close_c, base_lev_th)
    m_base = calc_summary(nav_base, sr_base, n_years, base_lev_th)

    print(f"\n  BASELINE: MomDecel+Ens2(S+T)")
    print(f"  Sharpe={m_base['Sharpe']:.3f}  CAGR={m_base['CAGR']*100:+.1f}%  "
          f"MaxDD={m_base['MaxDD']*100:.1f}%  W5Y={m_base['Worst5Y']*100:+.1f}%  "
          f"Trades={m_base['Trades/yr']:.1f}/yr")

    all_results = [{'Strategy': 'BASELINE MomDecel+Ens2(S+T)', **m_base}]

    def test_signal(name, mult_series, params_label):
        """Test adding a multiplier to the base strategy."""
        enhanced_raw = (base_lev_raw * mult_series).clip(0, 1.0).fillna(0)
        enhanced_th = rebalance_threshold(enhanced_raw, 0.20)
        nav, sr = run_backtest(close_c, enhanced_th)
        m = calc_summary(nav, sr, n_years, enhanced_th)
        delta_sharpe = m['Sharpe'] - m_base['Sharpe']
        label = f"{name} ({params_label})"
        print(f"  {label:<55s} Sharpe={m['Sharpe']:.3f} ({delta_sharpe:+.3f})  "
              f"CAGR={m['CAGR']*100:+.1f}%  MaxDD={m['MaxDD']*100:.1f}%  "
              f"W5Y={m['Worst5Y']*100:+.1f}%  Tr={m['Trades/yr']:.1f}/yr")
        all_results.append({'Strategy': label, **m, 'DeltaSharpe': delta_sharpe})
        return m

    # --- 1a: HY Spread ---
    print(f"\n  [HY Spread Multiplier]")
    for lb in [60, 120, 250]:
        for sens in [0.2, 0.3, 0.5]:
            mult = calc_hy_spread_mult(hy_c, lb, sens)
            test_signal('+ HY Spread', mult, f'lb={lb} s={sens}')

    # --- 1b: 10Y Yield Change ---
    print(f"\n  [10Y Yield Change Multiplier]")
    for lb in [60, 120, 250]:
        for sens in [0.2, 0.3, 0.5]:
            mult = calc_yield_change_mult(tnx_c, lb, sens)
            test_signal('+ 10Y Yield Chg', mult, f'lb={lb} s={sens}')

    # --- 1c: VIX Regime ---
    print(f"\n  [VIX Regime Multiplier]")
    for lb in [60, 120, 250]:
        for sens in [0.2, 0.3, 0.5]:
            mult = calc_vix_regime_mult(vix_c, lb, sens)
            test_signal('+ VIX Regime', mult, f'lb={lb} s={sens}')

    # --- 1d: Yield Curve ---
    print(f"\n  [Yield Curve Multiplier]")
    for lb in [60, 120, 250]:
        for sens in [0.2, 0.3, 0.5]:
            mult = calc_yield_curve_mult(t10y2y_c, lb, sens)
            test_signal('+ Yield Curve', mult, f'lb={lb} s={sens}')

    # =================================================================
    # PART 2: Best of each + Combinations
    # =================================================================
    print("\n" + "=" * 120)
    print("PART 2: BEST INDIVIDUAL + COMBINATIONS")
    print("=" * 120)

    # Find best params for each signal
    signal_types = {
        'HY Spread': [],
        '10Y Yield Chg': [],
        'VIX Regime': [],
        'Yield Curve': []
    }
    for r in all_results[1:]:
        for st in signal_types:
            if st in r['Strategy']:
                signal_types[st].append(r)

    best_params = {}
    for st, results in signal_types.items():
        if results:
            best = max(results, key=lambda x: x['Sharpe'])
            print(f"  Best {st}: {best['Strategy']}  "
                  f"Sharpe={best['Sharpe']:.3f} ({best['DeltaSharpe']:+.3f})")
            best_params[st] = best

    # Rebuild best multipliers
    print(f"\n  [Combinations of best signals]")

    # Best HY
    best_hy = calc_hy_spread_mult(hy_c, 120, 0.3)  # will be overridden by actual best
    best_tnx = calc_yield_change_mult(tnx_c, 120, 0.3)
    best_vix = calc_vix_regime_mult(vix_c, 120, 0.3)
    best_yc = calc_yield_curve_mult(t10y2y_c, 120, 0.3)

    # Re-extract best params from results
    for r in all_results[1:]:
        s = r['Strategy']
        if 'HY Spread' in s and r['Sharpe'] == best_params.get('HY Spread', {}).get('Sharpe', -99):
            # Parse params
            params = s.split('(')[1].rstrip(')')
            parts = {p.split('=')[0].strip(): float(p.split('=')[1]) for p in params.split(' ')}
            best_hy = calc_hy_spread_mult(hy_c, int(parts['lb']), parts['s'])
        if '10Y Yield' in s and r['Sharpe'] == best_params.get('10Y Yield Chg', {}).get('Sharpe', -99):
            params = s.split('(')[1].rstrip(')')
            parts = {p.split('=')[0].strip(): float(p.split('=')[1]) for p in params.split(' ')}
            best_tnx = calc_yield_change_mult(tnx_c, int(parts['lb']), parts['s'])
        if 'VIX Regime' in s and r['Sharpe'] == best_params.get('VIX Regime', {}).get('Sharpe', -99):
            params = s.split('(')[1].rstrip(')')
            parts = {p.split('=')[0].strip(): float(p.split('=')[1]) for p in params.split(' ')}
            best_vix = calc_vix_regime_mult(vix_c, int(parts['lb']), parts['s'])
        if 'Yield Curve' in s and r['Sharpe'] == best_params.get('Yield Curve', {}).get('Sharpe', -99):
            params = s.split('(')[1].rstrip(')')
            parts = {p.split('=')[0].strip(): float(p.split('=')[1]) for p in params.split(' ')}
            best_yc = calc_yield_curve_mult(t10y2y_c, int(parts['lb']), parts['s'])

    # Pairwise combinations
    combos = [
        ('HY + 10Y', best_hy * best_tnx),
        ('HY + VIX', best_hy * best_vix),
        ('HY + YieldCurve', best_hy * best_yc),
        ('10Y + VIX', best_tnx * best_vix),
        ('10Y + YieldCurve', best_tnx * best_yc),
        ('VIX + YieldCurve', best_vix * best_yc),
        ('HY + 10Y + VIX', best_hy * best_tnx * best_vix),
        ('HY + 10Y + YieldCurve', best_hy * best_tnx * best_yc),
        ('All 4', best_hy * best_tnx * best_vix * best_yc),
    ]

    for label, combo_mult in combos:
        test_signal('Combo', combo_mult, label)

    # =================================================================
    # PART 3: RANKING
    # =================================================================
    print("\n" + "=" * 120)
    print("PART 3: TOP 15 RANKING (by Sharpe)")
    print("=" * 120)

    ranked = sorted(all_results, key=lambda x: -x['Sharpe'])

    print(f"\n  {'#':<4s} {'Strategy':<60s} {'Sharpe':>7s} {'CAGR':>7s} "
          f"{'MaxDD':>7s} {'W5Y':>7s} {'Tr/yr':>6s}")
    print("  " + "-" * 100)

    for i, r in enumerate(ranked[:15], 1):
        delta = f" ({r.get('DeltaSharpe', 0):+.3f})" if r.get('DeltaSharpe') else ""
        print(f"  {i:<4d} {r['Strategy']:<60s} {r['Sharpe']:>7.3f}{delta}  "
              f"{r['CAGR']*100:>+6.1f}% {r['MaxDD']*100:>6.1f}% "
              f"{r['Worst5Y']*100:>+6.1f}% {r['Trades/yr']:>5.1f}")

    # =================================================================
    # PART 4: BEST vs BASELINE year-by-year
    # =================================================================
    if len(ranked) > 1 and ranked[0]['Strategy'] != 'BASELINE MomDecel+Ens2(S+T)':
        print("\n" + "=" * 120)
        print("PART 4: BEST ENHANCED vs BASELINE - YEARLY COMPARISON")
        print("=" * 120)

        best_strat = ranked[0]
        # Rebuild best strategy
        # Find the combo multiplier
        best_name = best_strat['Strategy']
        print(f"  Best: {best_name}")

        # For yearly comparison, rebuild with the best combo
        # (Use best individual if combo, otherwise single)
        # We'll use a simple approach: rebuild the top-ranked single signal
        best_single = [r for r in ranked if 'Combo' not in r['Strategy']
                       and 'BASELINE' not in r['Strategy']]
        if best_single:
            bs = best_single[0]
            print(f"  Best single signal: {bs['Strategy']} (Sharpe {bs['Sharpe']:.3f})")

    # Save results
    output_path = os.path.join(script_dir, '..', 'external_signal_results.csv')
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\n  Results saved to {output_path}")

    # =================================================================
    # KEY FINDINGS
    # =================================================================
    print("\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)

    baseline_sharpe = m_base['Sharpe']
    best_overall = ranked[0]
    best_single = max([r for r in all_results[1:] if 'Combo' not in r['Strategy']],
                      key=lambda x: x['Sharpe'], default=None)

    print(f"\n  Baseline Sharpe: {baseline_sharpe:.4f}")
    if best_single:
        print(f"  Best single signal: {best_single['Strategy']}")
        print(f"    Sharpe: {best_single['Sharpe']:.4f} ({best_single['DeltaSharpe']:+.4f})")
    print(f"  Best overall: {best_overall['Strategy']}")
    print(f"    Sharpe: {best_overall['Sharpe']:.4f}")


if __name__ == "__main__":
    main()
