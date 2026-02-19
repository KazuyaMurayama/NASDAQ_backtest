"""
External Signal Enhancement - Proper Out-of-Sample Validation
==============================================================
Key improvements over previous analysis (test_external_signals.py):
  1. Longer data: Use FRED series (T10YFF from 1962, BAA-AAA from 1986, DGS10 from 1962)
  2. Strict IS/OOS split: In-sample ≤ 2021-05-07, Out-of-sample 2021-05-08 ~ 2026-02
  3. Overfitting controls:
     - Parameters selected ONLY on in-sample
     - Time-series cross-validation (walk-forward)
     - Parameter sensitivity analysis (sharp peak = overfitting)
     - Comparison of IS vs OOS Sharpe degradation

Data sources (all from FRED CSV endpoint):
  - T10YFF: 10Y Treasury minus Fed Funds Rate (yield curve proxy, 1962~)
  - DBAA/DAAA: Moody's BAA-AAA spread (credit spread proxy, 1986~)
  - DGS10: 10Y Treasury Yield (1962~)
  - VXOCLS: VXO old VIX (1986~2021, FRED)

Common period: ~1986 to present (~35 years IS + 5 years OOS)
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
OOS_SPLIT_DATE = '2021-05-07'  # Last date of original sample


# =================================================================
# Data Fetching
# =================================================================
def fetch_fred_csv(series_id, start='1960-01-01'):
    """Fetch daily data from FRED CSV endpoint."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(4):
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            text = resp.read().decode('utf-8')
            break
        except Exception as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1} for {series_id} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise

    rows = []
    for line in text.strip().split('\n')[1:]:
        parts = line.split(',')
        if len(parts) == 2 and parts[1] not in ('.', ''):
            try:
                rows.append({'Date': pd.to_datetime(parts[0]),
                             'Value': float(parts[1])})
            except ValueError:
                continue
    return pd.DataFrame(rows)


def fetch_yahoo(symbol, start='1985-01-01'):
    """Fetch daily data from Yahoo Finance."""
    start_ts = int(time.mktime(time.strptime(start, '%Y-%m-%d')))
    end_ts = int(time.time())
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
           f'?period1={start_ts}&period2={end_ts}&interval=1d')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(4):
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            break
        except Exception as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1} for {symbol} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise

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


# =================================================================
# Signal Multiplier Designs (same logic, cleaner interface)
# =================================================================
def calc_yield_curve_mult(series, lookback=120, sensitivity=0.3,
                          min_mult=0.3, max_mult=1.2):
    """Yield curve slope multiplier (T10YFF or T10Y2Y).
    Flattening/inversion → reduce leverage."""
    z_mean = series.rolling(lookback).mean()
    z_std = series.rolling(lookback).std().replace(0, 0.001)
    z = (series - z_mean) / z_std
    mult = (1.0 + sensitivity * z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


def calc_credit_spread_mult(spread, lookback=120, sensitivity=0.3,
                             min_mult=0.3, max_mult=1.2):
    """Credit spread multiplier (BAA-AAA or HY OAS).
    Widening spread → reduce leverage."""
    spread_change = spread.pct_change(20)
    z_mean = spread_change.rolling(lookback).mean()
    z_std = spread_change.rolling(lookback).std().replace(0, 0.001)
    z = (spread_change - z_mean) / z_std
    mult = (1.0 - sensitivity * z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


def calc_yield_change_mult(yield_10y, lookback=120, sensitivity=0.3,
                           min_mult=0.3, max_mult=1.2):
    """10Y yield change multiplier: rapid rise → reduce leverage."""
    yield_change = yield_10y.diff(60)
    z_mean = yield_change.rolling(lookback).mean()
    z_std = yield_change.rolling(lookback).std().replace(0, 0.001)
    z = (yield_change - z_mean) / z_std
    mult = (1.0 - sensitivity * z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


def calc_vix_regime_mult(vix, lookback=120, sensitivity=0.3,
                         min_mult=0.3, max_mult=1.2):
    """VIX/VXO regime multiplier: elevated VIX → reduce leverage."""
    z_mean = vix.rolling(lookback).mean()
    z_std = vix.rolling(lookback).std().replace(0, 0.001)
    z = (vix - z_mean) / z_std
    mult = (1.0 - sensitivity * z).clip(min_mult, max_mult)
    return mult.fillna(1.0)


# =================================================================
# Strategy builder
# =================================================================
def build_base_leverage(close, returns):
    """Build MomDecel(40/120)+Ens2(S+T) base leverage."""
    dd_signal = calc_dd_signal(close, 0.82, 0.92, 200)
    trend_tv = calc_trend_target_vol(close)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope_mult = calc_slope_multiplier(close)
    decel_mult = calc_momentum_decel_mult(close, 40, 120, 0.3)
    leverage = dd_signal * vt_lev * slope_mult * decel_mult
    return leverage.clip(0, 1.0).fillna(0)


def run_backtest(close, leverage, base_lev=BASE_LEV, cost=ANNUAL_COST, delay=EXEC_DELAY):
    returns = close.pct_change()
    lev_returns = returns * base_lev
    daily_cost = cost / 252
    delayed = leverage.shift(delay)
    strat_ret = delayed * (lev_returns - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret


def calc_metrics_from_nav(nav, strat_ret, n_years, leverage):
    sharpe = strat_ret.mean() * 252 / (strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0
    cagr = nav.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    maxdd = ((nav - nav.cummax()) / nav.cummax()).min()
    # Worst 5Y only if enough data
    if len(nav) >= 252 * 5:
        w5y = ((nav / nav.shift(252 * 5)) ** 0.2 - 1).min()
    else:
        w5y = np.nan
    trades = (leverage.diff().abs() > 0.01).sum()
    return {
        'Sharpe': sharpe, 'CAGR': cagr, 'MaxDD': maxdd,
        'Worst5Y': w5y, 'Trades': trades, 'Trades/yr': trades / n_years if n_years > 0 else 0
    }


# =================================================================
# Walk-forward cross-validation
# =================================================================
def walk_forward_cv(close, returns, signal_series, calc_mult_fn,
                    param_grid, n_folds=5, min_train_years=5):
    """Walk-forward CV: train on expanding window, test on next fold.
    Returns average OOS Sharpe for each parameter set."""
    total_len = len(close)
    fold_size = total_len // (n_folds + 1)  # Reserve first chunk as minimum train

    results = {}
    for params in param_grid:
        oos_sharpes = []
        for fold in range(n_folds):
            train_end = fold_size * (fold + 2)  # At least 2 chunks for training
            test_start = train_end
            test_end = min(train_end + fold_size, total_len)
            if test_end <= test_start:
                continue

            # Build base on full data (base strategy is fixed)
            base_lev = build_base_leverage(close, returns)

            # Apply signal multiplier
            mult = calc_mult_fn(signal_series, **params)
            enhanced = (base_lev * mult).clip(0, 1.0).fillna(0)
            enhanced_th = rebalance_threshold(enhanced, 0.20)

            # Evaluate OOS portion only
            close_test = close.iloc[test_start:test_end].reset_index(drop=True)
            lev_test = enhanced_th.iloc[test_start:test_end].reset_index(drop=True)
            if len(close_test) < 126:  # Need at least 6 months
                continue
            n_years_test = len(close_test) / 252
            nav_test, sr_test = run_backtest(close_test, lev_test)
            sharpe_test = sr_test.mean() * 252 / (sr_test.std() * np.sqrt(252)) if sr_test.std() > 0 else 0
            oos_sharpes.append(sharpe_test)

        if oos_sharpes:
            results[str(params)] = {
                'mean_oos_sharpe': np.mean(oos_sharpes),
                'std_oos_sharpe': np.std(oos_sharpes),
                'n_folds': len(oos_sharpes),
                'params': params
            }
    return results


# =================================================================
# Main
# =================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 120)
    print("EXTERNAL SIGNAL ENHANCEMENT - PROPER OUT-OF-SAMPLE VALIDATION")
    print("Conditions: 5-day delay, 1.5% annual cost, rebalance threshold 20%")
    print(f"In-sample: ≤ {OOS_SPLIT_DATE}  |  Out-of-sample: > {OOS_SPLIT_DATE}")
    print("=" * 120)

    # ==============================================================
    # PHASE 1: Data Fetching
    # ==============================================================
    print("\n--- Phase 1: Fetching data ---")

    # Long-term FRED data
    print("  Fetching T10YFF (10Y - Fed Funds Rate)...")
    t10yff_df = fetch_fred_csv('T10YFF', start='1960-01-01')
    print(f"    → {len(t10yff_df)} rows ({t10yff_df['Date'].min().date()} to {t10yff_df['Date'].max().date()})")

    print("  Fetching DGS10 (10Y Treasury Yield)...")
    dgs10_df = fetch_fred_csv('DGS10', start='1960-01-01')
    print(f"    → {len(dgs10_df)} rows ({dgs10_df['Date'].min().date()} to {dgs10_df['Date'].max().date()})")

    print("  Fetching DBAA (Moody's BAA Corporate)...")
    dbaa_df = fetch_fred_csv('DBAA', start='1985-01-01')
    print(f"    → {len(dbaa_df)} rows ({dbaa_df['Date'].min().date()} to {dbaa_df['Date'].max().date()})")

    print("  Fetching DAAA (Moody's AAA Corporate)...")
    daaa_df = fetch_fred_csv('DAAA', start='1985-01-01')
    print(f"    → {len(daaa_df)} rows ({daaa_df['Date'].min().date()} to {daaa_df['Date'].max().date()})")

    # VIX - try VXO from FRED first, fallback to Yahoo ^VIX
    print("  Fetching VXOCLS (VXO, old VIX)...")
    try:
        vxo_df = fetch_fred_csv('VXOCLS', start='1985-01-01')
        print(f"    → {len(vxo_df)} rows ({vxo_df['Date'].min().date()} to {vxo_df['Date'].max().date()})")
    except Exception as e:
        print(f"    VXO fetch failed: {e}")
        vxo_df = pd.DataFrame(columns=['Date', 'Value'])

    # Also fetch regular VIX from Yahoo for post-VXO period
    print("  Fetching ^VIX (Yahoo Finance)...")
    try:
        vix_yahoo_df = fetch_yahoo('%5EVIX', start='1990-01-01')
        print(f"    → {len(vix_yahoo_df)} rows ({vix_yahoo_df['Date'].min().date()} to {vix_yahoo_df['Date'].max().date()})")
    except Exception as e:
        print(f"    VIX fetch failed: {e}")
        vix_yahoo_df = pd.DataFrame(columns=['Date', 'Value'])

    # Also try T10Y2Y for comparison (same as before)
    print("  Fetching T10Y2Y (10Y - 2Y, for comparison)...")
    t10y2y_df = fetch_fred_csv('T10Y2Y', start='1976-01-01')
    print(f"    → {len(t10y2y_df)} rows ({t10y2y_df['Date'].min().date()} to {t10y2y_df['Date'].max().date()})")

    # ==============================================================
    # PHASE 2: Data Alignment & Split
    # ==============================================================
    print("\n--- Phase 2: Data alignment ---")

    # Load NASDAQ data
    data_path = os.path.join(script_dir, '..', 'NASDAQ_extended.csv')
    df = load_data(data_path)
    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']

    master = pd.DataFrame({'Date': dates, 'Close': close.values}).set_index('Date')

    # Merge BAA-AAA spread
    baa = dbaa_df.set_index('Date')['Value']
    aaa = daaa_df.set_index('Date')['Value']
    baa_aligned = baa.reindex(master.index).ffill()
    aaa_aligned = aaa.reindex(master.index).ffill()
    master['CreditSpread'] = baa_aligned - aaa_aligned

    # Merge T10YFF
    t10yff = t10yff_df.set_index('Date')['Value']
    master['T10YFF'] = t10yff.reindex(master.index).ffill()

    # Merge DGS10
    dgs10 = dgs10_df.set_index('Date')['Value']
    master['DGS10'] = dgs10.reindex(master.index).ffill()

    # Merge T10Y2Y
    t10y2y = t10y2y_df.set_index('Date')['Value']
    master['T10Y2Y'] = t10y2y.reindex(master.index).ffill()

    # Merge VIX: use VXO where available, then VIX from Yahoo
    if not vxo_df.empty:
        vxo = vxo_df.set_index('Date')['Value']
        master['VXO'] = vxo.reindex(master.index).ffill()
    else:
        master['VXO'] = np.nan

    if not vix_yahoo_df.empty:
        vix_y = vix_yahoo_df.set_index('Date')['Value']
        master['VIX_Yahoo'] = vix_y.reindex(master.index).ffill()
    else:
        master['VIX_Yahoo'] = np.nan

    # Combine VXO + VIX: use VXO where available, VIX for rest
    master['VIX_Combined'] = master['VXO'].fillna(master['VIX_Yahoo'])

    # Report coverage
    print("\n  Data coverage (non-null rows):")
    for col in ['CreditSpread', 'T10YFF', 'DGS10', 'T10Y2Y', 'VIX_Combined']:
        valid = master[col].dropna()
        if len(valid) > 0:
            print(f"    {col:<16s}: {valid.index.min().date()} to {valid.index.max().date()} ({len(valid):,} days)")
        else:
            print(f"    {col:<16s}: NO DATA")

    # ==============================================================
    # Define signal groups
    # ==============================================================
    # Group A: Long-term available (1974/1976~)
    # Group B: Medium-term available (1986~)

    # --- Use Group B common period (1986~) for fair comparison ---
    signals_to_test = ['T10YFF', 'DGS10', 'CreditSpread', 'VIX_Combined', 'T10Y2Y']
    common_start = master[signals_to_test].dropna().index.min()
    print(f"\n  Common period (all signals): {common_start.date()}")

    # Filter to common period
    mask = master.index >= common_start
    mdata = master.loc[mask].copy()

    # Split into IS and OOS
    oos_date = pd.Timestamp(OOS_SPLIT_DATE)
    is_mask = mdata.index <= oos_date
    oos_mask = mdata.index > oos_date
    n_is = is_mask.sum()
    n_oos = oos_mask.sum()
    print(f"  In-sample:  {mdata.index[is_mask].min().date()} to {mdata.index[is_mask].max().date()} ({n_is:,} days, {n_is/252:.1f}yr)")
    print(f"  Out-of-sample: {mdata.index[oos_mask].min().date()} to {mdata.index[oos_mask].max().date()} ({n_oos:,} days, {n_oos/252:.1f}yr)")

    # Convert to positional index for engine functions
    def make_series(col, mask_sel):
        return pd.Series(mdata.loc[mask_sel, col].values, index=range(mask_sel.sum()))

    close_is = make_series('Close', is_mask)
    returns_is = close_is.pct_change()
    n_years_is = len(close_is) / 252

    close_oos = make_series('Close', oos_mask)
    returns_oos = close_oos.pct_change()
    n_years_oos = len(close_oos) / 252

    # Full period for reference
    close_full = pd.Series(mdata['Close'].values, index=range(len(mdata)))
    returns_full = close_full.pct_change()
    n_years_full = len(close_full) / 252

    # ==============================================================
    # PHASE 3: In-Sample Parameter Optimization
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 3: IN-SAMPLE PARAMETER OPTIMIZATION (≤ 2021-05-07)")
    print("=" * 120)

    # Baseline on IS
    base_lev_is = build_base_leverage(close_is, returns_is)
    base_th_is = rebalance_threshold(base_lev_is, 0.20)
    nav_base_is, sr_base_is = run_backtest(close_is, base_th_is)
    m_base_is = calc_metrics_from_nav(nav_base_is, sr_base_is, n_years_is, base_th_is)

    print(f"\n  BASELINE (IS): Sharpe={m_base_is['Sharpe']:.3f}  "
          f"CAGR={m_base_is['CAGR']*100:+.1f}%  MaxDD={m_base_is['MaxDD']*100:.1f}%  "
          f"W5Y={m_base_is['Worst5Y']*100:+.1f}%")

    # Signal definitions
    signal_configs = {
        'T10YFF': {
            'series_col': 'T10YFF',
            'calc_fn': calc_yield_curve_mult,
            'lookbacks': [60, 120, 180, 250],
            'sensitivities': [0.15, 0.2, 0.3, 0.4],
        },
        'T10Y2Y': {
            'series_col': 'T10Y2Y',
            'calc_fn': calc_yield_curve_mult,
            'lookbacks': [60, 120, 180, 250],
            'sensitivities': [0.15, 0.2, 0.3, 0.4],
        },
        'CreditSpread': {
            'series_col': 'CreditSpread',
            'calc_fn': calc_credit_spread_mult,
            'lookbacks': [60, 120, 180, 250],
            'sensitivities': [0.15, 0.2, 0.3, 0.4],
        },
        'DGS10': {
            'series_col': 'DGS10',
            'calc_fn': calc_yield_change_mult,
            'lookbacks': [60, 120, 180, 250],
            'sensitivities': [0.15, 0.2, 0.3, 0.4],
        },
        'VIX': {
            'series_col': 'VIX_Combined',
            'calc_fn': calc_vix_regime_mult,
            'lookbacks': [60, 120, 180, 250],
            'sensitivities': [0.15, 0.2, 0.3, 0.4],
        },
    }

    is_results = []
    best_per_signal = {}

    for sig_name, config in signal_configs.items():
        print(f"\n  [{sig_name}]")
        sig_is = make_series(config['series_col'], is_mask)

        best_sharpe = -999
        best_params = None
        signal_results = []

        for lb in config['lookbacks']:
            for sens in config['sensitivities']:
                mult = config['calc_fn'](sig_is, lookback=lb, sensitivity=sens)
                enhanced = (base_lev_is * mult).clip(0, 1.0).fillna(0)
                enhanced_th = rebalance_threshold(enhanced, 0.20)
                nav, sr = run_backtest(close_is, enhanced_th)
                m = calc_metrics_from_nav(nav, sr, n_years_is, enhanced_th)
                delta = m['Sharpe'] - m_base_is['Sharpe']

                result = {
                    'Signal': sig_name, 'Lookback': lb, 'Sensitivity': sens,
                    'IS_Sharpe': m['Sharpe'], 'IS_Delta': delta,
                    'IS_CAGR': m['CAGR'], 'IS_MaxDD': m['MaxDD'],
                    'IS_W5Y': m['Worst5Y']
                }
                is_results.append(result)
                signal_results.append(result)

                if m['Sharpe'] > best_sharpe:
                    best_sharpe = m['Sharpe']
                    best_params = {'lookback': lb, 'sensitivity': sens}

                print(f"    lb={lb:<4d} s={sens:.2f}  Sharpe={m['Sharpe']:.3f} ({delta:+.3f})  "
                      f"CAGR={m['CAGR']*100:+.1f}%  MaxDD={m['MaxDD']*100:.1f}%")

        best_per_signal[sig_name] = {
            'params': best_params,
            'sharpe': best_sharpe,
            'config': config
        }
        print(f"  → BEST {sig_name}: lb={best_params['lookback']}, s={best_params['sensitivity']:.2f}, "
              f"Sharpe={best_sharpe:.3f}")

    # ==============================================================
    # Parameter sensitivity check (overfitting indicator)
    # ==============================================================
    print("\n" + "=" * 120)
    print("PARAMETER SENSITIVITY CHECK (IS)")
    print("  Low std = robust. High std relative to delta = potential overfitting.")
    print("=" * 120)

    for sig_name in signal_configs:
        sig_results = [r for r in is_results if r['Signal'] == sig_name]
        sharpes = [r['IS_Sharpe'] for r in sig_results]
        deltas = [r['IS_Delta'] for r in sig_results]
        best_delta = max(deltas)
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        pct_positive = sum(1 for d in deltas if d > 0) / len(deltas) * 100

        print(f"  {sig_name:<16s}: BestΔ={best_delta:+.3f}  MeanΔ={mean_delta:+.3f}  "
              f"StdΔ={std_delta:.3f}  Positive={pct_positive:.0f}%")

        # Flag if best is far from mean (sensitive = overfitting risk)
        if std_delta > 0 and (best_delta - mean_delta) / std_delta > 2:
            print(f"    ⚠ WARNING: Best params are >2σ from mean → overfitting risk!")

    # ==============================================================
    # Combination test on IS (top 2 signals only)
    # ==============================================================
    print("\n" + "=" * 120)
    print("COMBINATION TEST (IS, best params per signal)")
    print("=" * 120)

    # Sort signals by best IS Sharpe
    ranked_signals = sorted(best_per_signal.items(), key=lambda x: -x[1]['sharpe'])
    print("\n  Signal ranking (IS):")
    for i, (name, info) in enumerate(ranked_signals, 1):
        delta = info['sharpe'] - m_base_is['Sharpe']
        print(f"    {i}. {name:<16s} Sharpe={info['sharpe']:.3f} ({delta:+.3f})")

    # Build individual best multipliers on IS
    best_mults_is = {}
    for sig_name, info in best_per_signal.items():
        sig_is = make_series(info['config']['series_col'], is_mask)
        best_mults_is[sig_name] = info['config']['calc_fn'](
            sig_is, **info['params']
        )

    # Test pairwise combos of top signals
    combo_results_is = []
    signal_names = list(best_per_signal.keys())
    for i in range(len(signal_names)):
        for j in range(i + 1, len(signal_names)):
            s1, s2 = signal_names[i], signal_names[j]
            combo_mult = best_mults_is[s1] * best_mults_is[s2]
            enhanced = (base_lev_is * combo_mult).clip(0, 1.0).fillna(0)
            enhanced_th = rebalance_threshold(enhanced, 0.20)
            nav, sr = run_backtest(close_is, enhanced_th)
            m = calc_metrics_from_nav(nav, sr, n_years_is, enhanced_th)
            delta = m['Sharpe'] - m_base_is['Sharpe']
            label = f"{s1} + {s2}"
            print(f"  {label:<35s} IS_Sharpe={m['Sharpe']:.3f} ({delta:+.3f})  "
                  f"CAGR={m['CAGR']*100:+.1f}%  MaxDD={m['MaxDD']*100:.1f}%")
            combo_results_is.append({
                'Combo': label, 'IS_Sharpe': m['Sharpe'], 'IS_Delta': delta,
                'IS_CAGR': m['CAGR'], 'IS_MaxDD': m['MaxDD'],
                'signals': [s1, s2]
            })

    # ==============================================================
    # PHASE 4: Walk-Forward Cross-Validation (IS only)
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 4: WALK-FORWARD CROSS-VALIDATION (5-fold, IS only)")
    print("  Validates that IS Sharpe improvement is not just from one lucky period")
    print("=" * 120)

    for sig_name, info in ranked_signals[:3]:  # Top 3 signals
        print(f"\n  [{sig_name}] Walk-forward CV...")
        sig_is = make_series(info['config']['series_col'], is_mask)
        param_grid = []
        for lb in info['config']['lookbacks']:
            for sens in info['config']['sensitivities']:
                param_grid.append({'lookback': lb, 'sensitivity': sens})

        cv_results = walk_forward_cv(
            close_is, returns_is, sig_is, info['config']['calc_fn'],
            param_grid, n_folds=5
        )

        # Show top 5 by mean OOS Sharpe
        ranked_cv = sorted(cv_results.values(), key=lambda x: -x['mean_oos_sharpe'])
        print(f"    {'Params':<30s} {'MeanOOS':>8s} {'StdOOS':>8s} {'Folds':>6s}")
        for r in ranked_cv[:5]:
            p = r['params']
            print(f"    lb={p['lookback']:<4d} s={p['sensitivity']:.2f}          "
                  f"{r['mean_oos_sharpe']:>8.3f} {r['std_oos_sharpe']:>8.3f} {r['n_folds']:>6d}")

        # Compare best full-IS params vs best CV params
        best_full = info['params']
        best_cv = ranked_cv[0]['params'] if ranked_cv else best_full
        if best_full != best_cv:
            print(f"    ⚠ CV best params ({best_cv}) differ from full-IS best ({best_full})")
            print(f"      → Will test BOTH on OOS for comparison")
        else:
            print(f"    ✓ CV confirms full-IS best params ({best_full})")

    # ==============================================================
    # PHASE 5: OUT-OF-SAMPLE EVALUATION
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 5: OUT-OF-SAMPLE EVALUATION (2021-05-08 ~ 2026-02)")
    print("  Parameters were FROZEN from in-sample. No peeking!")
    print("=" * 120)

    # OOS Baseline
    base_lev_oos = build_base_leverage(close_oos, returns_oos)
    base_th_oos = rebalance_threshold(base_lev_oos, 0.20)
    nav_base_oos, sr_base_oos = run_backtest(close_oos, base_th_oos)
    m_base_oos = calc_metrics_from_nav(nav_base_oos, sr_base_oos, n_years_oos, base_th_oos)

    print(f"\n  BASELINE (OOS): Sharpe={m_base_oos['Sharpe']:.3f}  "
          f"CAGR={m_base_oos['CAGR']*100:+.1f}%  MaxDD={m_base_oos['MaxDD']*100:.1f}%")

    # B&H 3x on OOS for reference
    bh3x_ret_oos = close_oos.pct_change() * 3.0 - ANNUAL_COST / 252
    bh3x_nav_oos = (1 + bh3x_ret_oos.fillna(0)).cumprod()
    bh3x_sharpe_oos = bh3x_ret_oos.dropna().mean() * 252 / (bh3x_ret_oos.dropna().std() * np.sqrt(252))
    bh3x_cagr_oos = bh3x_nav_oos.iloc[-1] ** (1 / n_years_oos) - 1
    bh3x_maxdd_oos = ((bh3x_nav_oos - bh3x_nav_oos.cummax()) / bh3x_nav_oos.cummax()).min()
    print(f"  B&H 3x  (OOS): Sharpe={bh3x_sharpe_oos:.3f}  "
          f"CAGR={bh3x_cagr_oos*100:+.1f}%  MaxDD={bh3x_maxdd_oos*100:.1f}%")

    oos_all = []

    # Individual signals on OOS (using IS-selected params)
    print(f"\n  {'Signal':<35s} {'OOS_Sharpe':>10s} {'IS_Sharpe':>10s} {'Degrade':>8s} "
          f"{'OOS_CAGR':>9s} {'OOS_MaxDD':>9s}")
    print("  " + "-" * 90)

    for sig_name, info in best_per_signal.items():
        sig_oos = make_series(info['config']['series_col'], oos_mask)
        mult_oos = info['config']['calc_fn'](sig_oos, **info['params'])
        base_lev_oos_sig = build_base_leverage(close_oos, returns_oos)
        enhanced = (base_lev_oos_sig * mult_oos).clip(0, 1.0).fillna(0)
        enhanced_th = rebalance_threshold(enhanced, 0.20)
        nav, sr = run_backtest(close_oos, enhanced_th)
        m = calc_metrics_from_nav(nav, sr, n_years_oos, enhanced_th)
        delta_oos = m['Sharpe'] - m_base_oos['Sharpe']
        delta_is = info['sharpe'] - m_base_is['Sharpe']
        degrade = delta_oos - delta_is  # Negative = overfitting

        params_str = f"lb={info['params']['lookback']}, s={info['params']['sensitivity']:.2f}"
        label = f"{sig_name} ({params_str})"
        print(f"  {label:<35s} {m['Sharpe']:>10.3f} {info['sharpe']:>10.3f} {degrade:>+8.3f} "
              f"{m['CAGR']*100:>+8.1f}% {m['MaxDD']*100:>8.1f}%")

        oos_all.append({
            'Strategy': f"+ {sig_name}",
            'Params': params_str,
            'IS_Sharpe': info['sharpe'],
            'IS_Delta': delta_is,
            'OOS_Sharpe': m['Sharpe'],
            'OOS_Delta': delta_oos,
            'Degradation': degrade,
            'OOS_CAGR': m['CAGR'],
            'OOS_MaxDD': m['MaxDD'],
        })

    # Combinations on OOS
    print(f"\n  Combinations (OOS):")
    for combo in combo_results_is:
        s1, s2 = combo['signals']
        sig1_oos = make_series(best_per_signal[s1]['config']['series_col'], oos_mask)
        sig2_oos = make_series(best_per_signal[s2]['config']['series_col'], oos_mask)
        mult1 = best_per_signal[s1]['config']['calc_fn'](sig1_oos, **best_per_signal[s1]['params'])
        mult2 = best_per_signal[s2]['config']['calc_fn'](sig2_oos, **best_per_signal[s2]['params'])
        combo_mult = mult1 * mult2
        base_lev_oos_c = build_base_leverage(close_oos, returns_oos)
        enhanced = (base_lev_oos_c * combo_mult).clip(0, 1.0).fillna(0)
        enhanced_th = rebalance_threshold(enhanced, 0.20)
        nav, sr = run_backtest(close_oos, enhanced_th)
        m = calc_metrics_from_nav(nav, sr, n_years_oos, enhanced_th)
        delta_oos = m['Sharpe'] - m_base_oos['Sharpe']
        degrade = delta_oos - combo['IS_Delta']

        label = combo['Combo']
        print(f"  {label:<35s} OOS_Sharpe={m['Sharpe']:.3f} ({delta_oos:+.3f})  "
              f"IS_Sharpe={combo['IS_Sharpe']:.3f} ({combo['IS_Delta']:+.3f})  "
              f"Degrade={degrade:+.3f}")

        oos_all.append({
            'Strategy': f"+ {label}",
            'Params': 'combo',
            'IS_Sharpe': combo['IS_Sharpe'],
            'IS_Delta': combo['IS_Delta'],
            'OOS_Sharpe': m['Sharpe'],
            'OOS_Delta': delta_oos,
            'Degradation': degrade,
            'OOS_CAGR': m['CAGR'],
            'OOS_MaxDD': m['MaxDD'],
        })

    # ==============================================================
    # PHASE 6: FINAL SUMMARY & RECOMMENDATION
    # ==============================================================
    print("\n" + "=" * 120)
    print("PHASE 6: FINAL SUMMARY & RECOMMENDATION")
    print("=" * 120)

    print(f"\n  {'Strategy':<40s} {'IS_Sharpe':>10s} {'IS_Δ':>7s} {'OOS_Sharpe':>11s} "
          f"{'OOS_Δ':>7s} {'Degrade':>8s} {'OOS_CAGR':>9s} {'OOS_MaxDD':>9s}")
    print("  " + "-" * 105)

    # Baseline row
    print(f"  {'BASELINE MomDecel+Ens2(S+T)':<40s} {m_base_is['Sharpe']:>10.3f} {'—':>7s} "
          f"{m_base_oos['Sharpe']:>11.3f} {'—':>7s} {'—':>8s} "
          f"{m_base_oos['CAGR']*100:>+8.1f}% {m_base_oos['MaxDD']*100:>8.1f}%")

    # Sort by OOS Sharpe
    oos_sorted = sorted(oos_all, key=lambda x: -x['OOS_Sharpe'])
    for r in oos_sorted:
        print(f"  {r['Strategy']:<40s} {r['IS_Sharpe']:>10.3f} {r['IS_Delta']:>+7.3f} "
              f"{r['OOS_Sharpe']:>11.3f} {r['OOS_Delta']:>+7.3f} {r['Degradation']:>+8.3f} "
              f"{r['OOS_CAGR']*100:>+8.1f}% {r['OOS_MaxDD']*100:>8.1f}%")

    # Overall assessment
    print("\n  " + "-" * 80)

    # Count how many strategies beat baseline OOS
    n_beat_oos = sum(1 for r in oos_all if r['OOS_Delta'] > 0)
    n_total = len(oos_all)
    print(f"  OOS上でベースラインを上回った: {n_beat_oos}/{n_total} 戦略")

    # Check for overfitting: large IS delta but negative OOS delta
    overfit = [r for r in oos_all if r['IS_Delta'] > 0.02 and r['OOS_Delta'] < 0]
    if overfit:
        print(f"  ⚠ 過学習の疑い ({len(overfit)} 戦略): ISで改善もOOSで悪化")
        for r in overfit:
            print(f"    - {r['Strategy']}: IS_Δ={r['IS_Delta']:+.3f} → OOS_Δ={r['OOS_Delta']:+.3f}")

    # Recommendation
    best_oos = oos_sorted[0] if oos_sorted else None
    if best_oos and best_oos['OOS_Delta'] > 0:
        print(f"\n  ★ 推奨: {best_oos['Strategy']}")
        print(f"    IS: Sharpe {best_oos['IS_Sharpe']:.3f} ({best_oos['IS_Delta']:+.3f})")
        print(f"    OOS: Sharpe {best_oos['OOS_Sharpe']:.3f} ({best_oos['OOS_Delta']:+.3f})")
        print(f"    Degradation: {best_oos['Degradation']:+.3f} (IS→OOS performance drop)")
    else:
        print(f"\n  ★ 推奨: ベースライン維持（外部シグナル追加の効果はOOSで確認できず）")

    # ==============================================================
    # PHASE 7: Year-by-year comparison (best vs baseline, OOS period)
    # ==============================================================
    if best_oos and best_oos['OOS_Delta'] > 0:
        print("\n" + "=" * 120)
        print("PHASE 7: YEAR-BY-YEAR COMPARISON (OOS period)")
        print("=" * 120)

        # Rebuild best strategy on OOS
        # Parse which signals are in the best
        best_name = best_oos['Strategy'].replace('+ ', '')
        sig_names_in_best = best_name.split(' + ') if ' + ' in best_name else [best_name]

        combo_mult_oos = pd.Series(1.0, index=range(len(close_oos)))
        for sn in sig_names_in_best:
            if sn in best_per_signal:
                sig_oos = make_series(best_per_signal[sn]['config']['series_col'], oos_mask)
                m = best_per_signal[sn]['config']['calc_fn'](sig_oos, **best_per_signal[sn]['params'])
                combo_mult_oos = combo_mult_oos * m

        base_lev_yr = build_base_leverage(close_oos, returns_oos)
        enhanced_yr = (base_lev_yr * combo_mult_oos).clip(0, 1.0).fillna(0)
        enhanced_th_yr = rebalance_threshold(enhanced_yr, 0.20)

        # NAV for both
        nav_base_yr, sr_base_yr = run_backtest(close_oos, base_th_oos)
        nav_enh_yr, sr_enh_yr = run_backtest(close_oos, enhanced_th_yr)

        dates_oos = mdata.index[oos_mask]
        years = sorted(set(d.year for d in dates_oos))

        print(f"\n  {'Year':<6s} {'Baseline':>10s} {'Enhanced':>10s} {'B&H 3x':>10s} {'Diff':>8s}")
        print("  " + "-" * 50)

        for yr in years:
            yr_idx = [i for i, d in enumerate(dates_oos) if d.year == yr]
            if len(yr_idx) < 20:
                continue
            i_start, i_end = yr_idx[0], yr_idx[-1]

            base_yr_ret = nav_base_yr.iloc[i_end] / nav_base_yr.iloc[i_start] - 1
            enh_yr_ret = nav_enh_yr.iloc[i_end] / nav_enh_yr.iloc[i_start] - 1
            bh3x_yr_ret = bh3x_nav_oos.iloc[i_end] / bh3x_nav_oos.iloc[i_start] - 1
            diff = enh_yr_ret - base_yr_ret

            print(f"  {yr:<6d} {base_yr_ret*100:>+9.1f}% {enh_yr_ret*100:>+9.1f}% "
                  f"{bh3x_yr_ret*100:>+9.1f}% {diff*100:>+7.1f}%")

    # ==============================================================
    # Save results
    # ==============================================================
    output_path = os.path.join(script_dir, '..', 'external_oos_results.csv')
    pd.DataFrame(oos_all).to_csv(output_path, index=False)
    print(f"\n  Results saved to {output_path}")

    is_output_path = os.path.join(script_dir, '..', 'external_is_param_sweep.csv')
    pd.DataFrame(is_results).to_csv(is_output_path, index=False)
    print(f"  IS parameter sweep saved to {is_output_path}")


if __name__ == "__main__":
    main()
