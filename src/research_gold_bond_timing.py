"""
Gold/Bond Timing Signal Research — Step 1: Predictive Power Analysis
Checks whether Gold and Bond have exploitable timing signals.
Run each part independently via: python research_gold_bond_timing.py [part]
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from scipy import stats as sp_stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')
DELAY = 2
GOLD_2X_COST = 0.005
BOND_3X_COST = 0.0091
OOS_DATE = '2021-05-07'
GOLD_START = '1975-01-01'  # Exclude pre-free-market period

# =============================================================================
# Part 0: Data Loading
# =============================================================================
def load_all_data():
    """Load and align NASDAQ, Gold, Bond data."""
    from backtest_engine import load_data
    from test_portfolio_diversification import prepare_gold_data, prepare_bond_data

    df = load_data(DATA_PATH)
    close = df['Close']; dates = df['Date']

    gold_prices = prepare_gold_data(dates)
    bond_nav = prepare_bond_data(dates)

    # Get bond yield separately for macro indicators
    import yfinance as yf
    tnx = yf.Ticker('^TNX')
    tnx_data = tnx.history(start='1974-01-01', end='2026-04-01')
    tnx_data = tnx_data.reset_index()
    tnx_data['Date'] = pd.to_datetime(tnx_data['Date']).dt.tz_localize(None)
    tnx_daily = tnx_data[['Date', 'Close']].rename(columns={'Close': 'Yield_pct'})
    ndf = pd.DataFrame({'Date': dates})
    merged = ndf.merge(tnx_daily, on='Date', how='left')
    merged['Yield_pct'] = merged['Yield_pct'].ffill().bfill()
    bond_yield = merged['Yield_pct'].values / 100  # decimal

    # Build unified DataFrame
    out = pd.DataFrame({
        'date': dates.values,
        'nasdaq_close': close.values,
        'gold_price': gold_prices,
        'bond_nav': bond_nav,
        'bond_yield': bond_yield,
    })
    out['nasdaq_ret'] = out['nasdaq_close'].pct_change()
    out['gold_ret'] = out['gold_price'].pct_change()
    out['bond_ret'] = out['bond_nav'].pct_change()

    # Gold 2x and Bond 3x forward returns (for evaluation)
    out['gold_2x_ret'] = out['gold_ret'] * 2 - GOLD_2X_COST / 252
    out['bond_3x_ret'] = out['bond_ret'] * 3 - BOND_3X_COST / 252

    out = out.iloc[1:].reset_index(drop=True)  # drop first NaN row
    print(f"Data loaded: {len(out)} rows, {out['date'].iloc[0]} to {out['date'].iloc[-1]}")
    return out


# =============================================================================
# Part 1: Indicator Library
# =============================================================================

# --- Trend-Following ---
def ind_price_vs_ma(price, window):
    """TF2: price / MA - 1"""
    ma = price.rolling(window).mean()
    return (price / ma - 1).fillna(0)

def ind_momentum_roc(price, period):
    """TF3: Rate of Change"""
    return price.pct_change(period).fillna(0)

def ind_dual_momentum(price, short_p, long_p):
    """TF4: short ROC - long ROC (normalized)"""
    sr = price.pct_change(short_p).fillna(0)
    lr = price.pct_change(long_p).fillna(0)
    return (sr - lr).fillna(0)

def ind_macd(price, fast_span, slow_span, signal_span):
    """TF5: MACD histogram"""
    ema_fast = price.ewm(span=fast_span).mean()
    ema_slow = price.ewm(span=slow_span).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span).mean()
    hist = macd_line - signal_line
    return (hist / price).fillna(0)  # normalize

def ind_ma_slope_z(price, ma_window, z_window):
    """TF6: Z-score of MA slope"""
    ma = price.rolling(ma_window).mean()
    slope = ma.pct_change()
    sm = slope.rolling(z_window).mean()
    ss = slope.rolling(z_window).std().replace(0, 0.0001)
    return ((slope - sm) / ss).fillna(0)

# --- Volatility-Based ---
def ind_vol_level(returns, window):
    """VL1: Annualized realized volatility"""
    return (returns.rolling(window).std() * np.sqrt(252)).fillna(0)

def ind_vol_zscore(returns, vol_win, z_win):
    """VL2: Z-score of realized vol"""
    vol = returns.rolling(vol_win).std() * np.sqrt(252)
    vm = vol.rolling(z_win).mean()
    vs = vol.rolling(z_win).std().replace(0, 0.001)
    return ((vol - vm) / vs).fillna(0)

def ind_asym_ewma_vol(returns, span_up, span_down):
    """VL3: Asymmetric EWMA volatility"""
    var = pd.Series(index=returns.index, dtype=float)
    v0 = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    var.iloc[0] = v0
    for i in range(1, len(returns)):
        r = returns.iloc[i]
        a = 2 / (span_down + 1) if r < 0 else 2 / (span_up + 1)
        var.iloc[i] = (1 - a) * var.iloc[i - 1] + a * (r ** 2)
    return np.sqrt(var * 252)

def ind_vol_trend(returns, vol_win, ma_win):
    """VL4: Vol / Vol_MA - 1 (vol rising or falling)"""
    vol = returns.rolling(vol_win).std() * np.sqrt(252)
    vol_ma = vol.rolling(ma_win).mean()
    return ((vol / vol_ma - 1)).fillna(0)

def ind_vol_of_vol(returns, vol_win, vov_win):
    """VL5: Vol-of-Vol"""
    vol = returns.rolling(vol_win).std() * np.sqrt(252)
    return vol.rolling(vov_win).std().fillna(0)

# --- Macro/Cross-Asset ---
def ind_real_rate(bond_yield, gold_ret, inflation_win):
    """MA1: Real rate proxy = nominal yield - rolling gold return (inflation proxy)"""
    inflation_proxy = gold_ret.rolling(inflation_win).mean() * 252
    return (pd.Series(bond_yield) - inflation_proxy).fillna(0)

def ind_yield_momentum(bond_yield, period):
    """MA2: Yield change direction (rising = bearish bonds)"""
    y = pd.Series(bond_yield)
    return y.diff(period).fillna(0)

def ind_yield_regime(bond_yield, window):
    """MA3: Z-score of yield level"""
    y = pd.Series(bond_yield)
    ym = y.rolling(window).mean()
    ys = y.rolling(window).std().replace(0, 0.001)
    return ((y - ym) / ys).fillna(0)

def ind_rolling_corr(ret1, ret2, window):
    """MA4/MA5: Rolling correlation"""
    return ret1.rolling(window).corr(ret2).fillna(0)

def ind_relative_strength(price1, price2, window):
    """MA6: Relative momentum"""
    r1 = price1.pct_change(window).fillna(0)
    r2 = price2.pct_change(window).fillna(0)
    return (r1 - r2).fillna(0)

def ind_yield_curve_proxy(bond_yield, window):
    """MA8: Yield vs its long-term average (crude term structure proxy)"""
    y = pd.Series(bond_yield)
    return (y - y.rolling(window).mean()).fillna(0)


# =============================================================================
# Part 2: Evaluation Framework
# =============================================================================

def compute_forward_returns(ret_series, horizons=[5, 21, 63]):
    """Compute forward returns at multiple horizons with DELAY."""
    fwd = {}
    for h in horizons:
        # Forward return from t+DELAY to t+DELAY+h
        cum = (1 + ret_series).rolling(h).apply(lambda x: x.prod() - 1, raw=True)
        fwd[h] = cum.shift(-(DELAY + h))
    return fwd

def rank_ic(signal, forward_ret):
    """Spearman rank correlation between signal and forward returns."""
    valid = signal.notna() & forward_ret.notna()
    s = signal[valid]; f = forward_ret[valid]
    if len(s) < 50:
        return np.nan, np.nan
    ic, pval = sp_stats.spearmanr(s, f)
    return ic, pval

def hit_rate(signal, forward_ret):
    """When signal > median, what fraction of forward returns are positive?"""
    valid = signal.notna() & forward_ret.notna()
    s = signal[valid]; f = forward_ret[valid]
    if len(s) < 50:
        return np.nan, np.nan
    med = s.median()
    high = f[s > med]; low = f[s <= med]
    hr_high = (high > 0).mean() if len(high) > 0 else np.nan
    hr_low = (low > 0).mean() if len(low) > 0 else np.nan
    return hr_high, hr_low

def long_short_spread(signal, forward_ret):
    """Top quintile vs bottom quintile annualized return spread."""
    valid = signal.notna() & forward_ret.notna()
    s = signal[valid]; f = forward_ret[valid]
    if len(s) < 100:
        return np.nan, np.nan
    q20 = s.quantile(0.2); q80 = s.quantile(0.8)
    long_r = f[s >= q80]; short_r = f[s <= q20]
    if len(long_r) < 20 or len(short_r) < 20:
        return np.nan, np.nan
    spread = long_r.mean() - short_r.mean()
    se = np.sqrt(long_r.var() / len(long_r) + short_r.var() / len(short_r))
    t_stat = spread / se if se > 0 else 0
    return spread, t_stat

WF_WINDOWS = [
    ('1975-01-01', '1990-01-01', '1990-01-01', '1995-01-01'),
    ('1975-01-01', '1995-01-01', '1995-01-01', '2000-01-01'),
    ('1975-01-01', '2000-01-01', '2000-01-01', '2005-01-01'),
    ('1975-01-01', '2005-01-01', '2005-01-01', '2010-01-01'),
    ('1975-01-01', '2010-01-01', '2010-01-01', '2015-01-01'),
    ('1975-01-01', '2015-01-01', '2015-01-01', '2021-01-01'),
    ('1975-01-01', '2021-01-01', '2021-01-01', '2027-01-01'),
]

def walk_forward_ic(signal, forward_ret, dates):
    """Compute IC on each walk-forward test window."""
    ics = []
    for _, _, test_start, test_end in WF_WINDOWS:
        mask = (dates >= test_start) & (dates < test_end)
        s = signal[mask]; f = forward_ret[mask]
        ic, _ = rank_ic(s, f)
        ics.append(ic)
    return ics

def evaluate_one_signal(name, signal, fwd_rets, dates, horizons=[5, 21, 63]):
    """Full evaluation of a single signal."""
    results = []
    for h in horizons:
        fr = fwd_rets[h]
        ic, pval = rank_ic(signal, fr)
        hr_h, hr_l = hit_rate(signal, fr)
        spr, t_stat = long_short_spread(signal, fr)
        wf_ics = walk_forward_ic(signal, fr, dates)
        wf_avg = np.nanmean(wf_ics)
        wf_stable = sum(1 for x in wf_ics if not np.isnan(x) and x > 0) if ic > 0 else \
                     sum(1 for x in wf_ics if not np.isnan(x) and x < 0)

        results.append({
            'indicator': name, 'horizon': h,
            'IC': ic, 'IC_pval': pval, 'IC_abs': abs(ic) if not np.isnan(ic) else 0,
            'hit_rate_high': hr_h, 'hit_rate_low': hr_l,
            'spread': spr, 'spread_t': t_stat,
            'WF_avg_IC': wf_avg,
            'WF_stable': wf_stable,
            'WF_IC_1': wf_ics[0], 'WF_IC_2': wf_ics[1], 'WF_IC_3': wf_ics[2],
            'WF_IC_4': wf_ics[3], 'WF_IC_5': wf_ics[4], 'WF_IC_6': wf_ics[5],
            'WF_IC_7': wf_ics[6],
        })
    return results


# =============================================================================
# Part 3: Generate All Signals
# =============================================================================
def generate_gold_signals(df):
    """Generate all gold timing signals."""
    price = df['gold_price']; ret = df['gold_ret']
    bond_yield = df['bond_yield']; nasdaq_ret = df['nasdaq_ret']
    gold_price_s = pd.Series(price.values, index=range(len(price)))
    ret_s = pd.Series(ret.values, index=range(len(ret)))

    signals = {}

    # TF2: Price vs MA
    for w in [50, 100, 150, 200, 252]:
        signals[f'TF2_PvMA_{w}'] = ind_price_vs_ma(gold_price_s, w)

    # TF3: Momentum ROC
    for p in [21, 42, 63, 126, 189, 252]:
        signals[f'TF3_MOM_{p}'] = ind_momentum_roc(gold_price_s, p)

    # TF4: Dual Momentum
    for sp in [21, 42, 63]:
        for lp in [126, 189, 252]:
            signals[f'TF4_DualM_{sp}_{lp}'] = ind_dual_momentum(gold_price_s, sp, lp)

    # TF5: MACD
    for fs, ss, sig in [(12, 26, 9), (12, 52, 9), (26, 52, 18)]:
        signals[f'TF5_MACD_{fs}_{ss}_{sig}'] = ind_macd(gold_price_s, fs, ss, sig)

    # TF6: MA Slope Z
    for mw in [100, 200]:
        for zw in [60, 120]:
            signals[f'TF6_SlopeZ_{mw}_{zw}'] = ind_ma_slope_z(gold_price_s, mw, zw)

    # VL1: Vol Level (negative = low vol bullish)
    for w in [10, 21, 42, 63]:
        signals[f'VL1_Vol_{w}'] = -ind_vol_level(ret_s, w)

    # VL2: Vol Z-Score (negative = low vol bullish)
    for vw in [21, 42]:
        for zw in [126, 252]:
            signals[f'VL2_VolZ_{vw}_{zw}'] = -ind_vol_zscore(ret_s, vw, zw)

    # VL3: Asym EWMA (negative)
    for su, sd in [(20, 5), (30, 10)]:
        signals[f'VL3_AsymVol_{su}_{sd}'] = -ind_asym_ewma_vol(ret_s, su, sd)

    # VL4: Vol Trend (negative = vol declining = bullish)
    for vw, mw in [(21, 63), (21, 126)]:
        signals[f'VL4_VolTrend_{vw}_{mw}'] = -ind_vol_trend(ret_s, vw, mw)

    # VL5: Vol of Vol (negative)
    for vw, vvw in [(21, 42), (21, 63)]:
        signals[f'VL5_VoV_{vw}_{vvw}'] = -ind_vol_of_vol(ret_s, vw, vvw)

    # MA1: Real rate (negative = low real rate = bullish gold)
    bond_yield_s = pd.Series(bond_yield.values, index=range(len(bond_yield)))
    gold_ret_s = pd.Series(ret.values, index=range(len(ret)))
    for iw in [63, 126, 252]:
        signals[f'MA1_RealRate_{iw}'] = -ind_real_rate(bond_yield_s, gold_ret_s, iw)

    # MA4: NASDAQ-Gold correlation (negative = more hedge = bullish gold)
    nasdaq_ret_s = pd.Series(nasdaq_ret.values, index=range(len(nasdaq_ret)))
    for w in [63, 126, 252]:
        signals[f'MA4_NQ_Gold_Corr_{w}'] = -ind_rolling_corr(nasdaq_ret_s, gold_ret_s, w)

    # MA6: Relative strength Gold vs Bond
    bond_nav_s = pd.Series(df['bond_nav'].values, index=range(len(df)))
    for w in [63, 126, 252]:
        signals[f'MA6_GoldVsBond_{w}'] = ind_relative_strength(gold_price_s, bond_nav_s, w)

    print(f"  Gold signals: {len(signals)} indicators")
    return signals

def generate_bond_signals(df):
    """Generate all bond timing signals."""
    price = pd.Series(df['bond_nav'].values, index=range(len(df)))
    ret = pd.Series(df['bond_ret'].values, index=range(len(df)))
    bond_yield = df['bond_yield']
    nasdaq_ret = pd.Series(df['nasdaq_ret'].values, index=range(len(df)))

    signals = {}

    # TF2: Price vs MA
    for w in [50, 100, 150, 200, 252]:
        signals[f'TF2_PvMA_{w}'] = ind_price_vs_ma(price, w)

    # TF3: Momentum ROC
    for p in [21, 42, 63, 126, 189, 252]:
        signals[f'TF3_MOM_{p}'] = ind_momentum_roc(price, p)

    # TF4: Dual Momentum
    for sp in [21, 42, 63]:
        for lp in [126, 189, 252]:
            signals[f'TF4_DualM_{sp}_{lp}'] = ind_dual_momentum(price, sp, lp)

    # TF5: MACD
    for fs, ss, sig in [(12, 26, 9), (12, 52, 9), (26, 52, 18)]:
        signals[f'TF5_MACD_{fs}_{ss}_{sig}'] = ind_macd(price, fs, ss, sig)

    # TF6: MA Slope Z
    for mw in [100, 200]:
        for zw in [60, 120]:
            signals[f'TF6_SlopeZ_{mw}_{zw}'] = ind_ma_slope_z(price, mw, zw)

    # VL1-5 (same as gold but for bond returns)
    for w in [10, 21, 42, 63]:
        signals[f'VL1_Vol_{w}'] = -ind_vol_level(ret, w)
    for vw in [21, 42]:
        for zw in [126, 252]:
            signals[f'VL2_VolZ_{vw}_{zw}'] = -ind_vol_zscore(ret, vw, zw)
    for su, sd in [(20, 5), (30, 10)]:
        signals[f'VL3_AsymVol_{su}_{sd}'] = -ind_asym_ewma_vol(ret, su, sd)
    for vw, mw in [(21, 63), (21, 126)]:
        signals[f'VL4_VolTrend_{vw}_{mw}'] = -ind_vol_trend(ret, vw, mw)
    for vw, vvw in [(21, 42), (21, 63)]:
        signals[f'VL5_VoV_{vw}_{vvw}'] = -ind_vol_of_vol(ret, vw, vvw)

    # MA2: Yield momentum (negative = falling yields = bullish bonds)
    bond_yield_s = pd.Series(bond_yield.values, index=range(len(bond_yield)))
    for p in [21, 42, 63, 126]:
        signals[f'MA2_YieldMom_{p}'] = -ind_yield_momentum(bond_yield_s, p)

    # MA3: Yield regime (negative = low yields = potentially bearish due to less room)
    for w in [252, 504]:
        signals[f'MA3_YieldRegime_{w}'] = -ind_yield_regime(bond_yield_s, w)

    # MA5: NASDAQ-Bond correlation
    bond_ret_s = pd.Series(df['bond_ret'].values, index=range(len(df)))
    for w in [63, 126, 252]:
        signals[f'MA5_NQ_Bond_Corr_{w}'] = -ind_rolling_corr(nasdaq_ret, bond_ret_s, w)

    # MA8: Yield curve proxy
    for w in [252]:
        signals[f'MA8_YieldCurve_{w}'] = -ind_yield_curve_proxy(bond_yield_s, w)

    print(f"  Bond signals: {len(signals)} indicators")
    return signals


# =============================================================================
# Part 4: Run Evaluation
# =============================================================================
def run_evaluation(asset_name, signals, fwd_rets, dates):
    """Evaluate all signals for one asset."""
    all_results = []
    total = len(signals)
    for i, (name, sig) in enumerate(signals.items()):
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{total}] {name}")
        results = evaluate_one_signal(name, sig, fwd_rets, dates)
        for r in results:
            r['asset'] = asset_name
        all_results.extend(results)
    return pd.DataFrame(all_results)


# =============================================================================
# Part 5: Cross-Asset Correlation Analysis
# =============================================================================
def run_correlation_analysis(df):
    """Analyze time-varying correlations between 3 assets."""
    nasdaq_ret = pd.Series(df['nasdaq_ret'].values, index=range(len(df)))
    gold_ret = pd.Series(df['gold_ret'].values, index=range(len(df)))
    bond_ret = pd.Series(df['bond_ret'].values, index=range(len(df)))
    dates = pd.to_datetime(df['date'].values)

    results = []
    for window in [63, 126, 252]:
        corr_ng = nasdaq_ret.rolling(window).corr(gold_ret)
        corr_nb = nasdaq_ret.rolling(window).corr(bond_ret)
        corr_gb = gold_ret.rolling(window).corr(bond_ret)

        results.append({
            'window': window,
            'NQ_Gold_mean': corr_ng.mean(), 'NQ_Gold_std': corr_ng.std(),
            'NQ_Bond_mean': corr_nb.mean(), 'NQ_Bond_std': corr_nb.std(),
            'Gold_Bond_mean': corr_gb.mean(), 'Gold_Bond_std': corr_gb.std(),
        })

    corr_df = pd.DataFrame(results)

    # Regime analysis: NASDAQ drawdown vs rally
    from backtest_engine import calc_dd_signal
    nasdaq_close = pd.Series(df['nasdaq_close'].values)
    dd_sig = calc_dd_signal(nasdaq_close, 0.82, 0.92)

    regimes = []
    for regime_name, mask_val in [('DD (CASH)', 0), ('Rally (HOLD)', 1)]:
        mask = dd_sig == mask_val
        n_days = mask.sum()
        if n_days < 10:
            continue
        g_ret = gold_ret[mask.values].mean() * 252
        b_ret = bond_ret[mask.values].mean() * 252
        nq_ret = nasdaq_ret[mask.values].mean() * 252
        regimes.append({
            'regime': regime_name, 'days': int(n_days),
            'pct_days': n_days / len(df) * 100,
            'nasdaq_ann_ret': nq_ret, 'gold_ann_ret': g_ret, 'bond_ann_ret': b_ret,
        })

    regime_df = pd.DataFrame(regimes)
    return corr_df, regime_df


# =============================================================================
# Part 6: Multiple Testing Correction
# =============================================================================
def apply_corrections(results_df):
    """Apply Bonferroni-Holm and BH FDR corrections."""
    out = results_df.copy()

    for h in [5, 21, 63]:
        mask = out['horizon'] == h
        pvals = out.loc[mask, 'IC_pval'].values
        n = len(pvals)
        if n == 0:
            continue

        # Bonferroni-Holm
        sorted_idx = np.argsort(pvals)
        bh_adj = np.ones(n)
        for rank, idx in enumerate(sorted_idx):
            bh_adj[idx] = min(pvals[idx] * (n - rank), 1.0)
        out.loc[mask, 'BH_pval'] = bh_adj

        # Benjamini-Hochberg FDR
        fdr_adj = np.ones(n)
        for rank, idx in enumerate(sorted_idx):
            threshold = (rank + 1) / n * 0.10
            fdr_adj[idx] = 1 if pvals[idx] > threshold else 0
        out.loc[mask, 'FDR_sig'] = (fdr_adj == 0).astype(int)

    return out


# =============================================================================
# Part 7: Summary
# =============================================================================
def generate_summary(gold_results, bond_results):
    """Generate final summary with filters."""
    all_r = pd.concat([gold_results, bond_results], ignore_index=True)
    all_r = apply_corrections(all_r)

    # Practical filters
    # 1. |IC| > 0.03
    # 2. WF stable >= 5/7
    # 3. FDR significant
    filt = all_r[
        (all_r['IC_abs'] > 0.03) &
        (all_r['WF_stable'] >= 5) &
        (all_r['FDR_sig'] == 1)
    ].sort_values('IC_abs', ascending=False)

    return all_r, filt


# =============================================================================
# Main Runner
# =============================================================================
def run_part0():
    """Data loading only."""
    df = load_all_data()
    cache = os.path.join(BASE_DIR, 'research_data_cache.pkl')
    df.to_pickle(cache)
    print(f"Saved cache: {cache}")
    return df

def run_part3(df):
    """Gold signal evaluation."""
    # Filter to Gold start date
    mask = pd.to_datetime(df['date']) >= GOLD_START
    gdf = df[mask].reset_index(drop=True)
    dates = pd.to_datetime(gdf['date'])

    print("Generating gold signals...")
    signals = generate_gold_signals(gdf)

    print("Computing gold 2x forward returns...")
    gold_2x_ret = pd.Series(gdf['gold_2x_ret'].values, index=range(len(gdf)))
    fwd = compute_forward_returns(gold_2x_ret)

    print("Evaluating gold signals...")
    results = run_evaluation('Gold', signals, fwd, dates)

    out = os.path.join(BASE_DIR, 'research_gold_signals.csv')
    results.to_csv(out, index=False)
    print(f"Saved: {out} ({len(results)} rows)")
    return results

def run_part4(df):
    """Bond signal evaluation."""
    dates = pd.to_datetime(df['date'])

    print("Generating bond signals...")
    signals = generate_bond_signals(df)

    print("Computing bond 3x forward returns...")
    bond_3x_ret = pd.Series(df['bond_3x_ret'].values, index=range(len(df)))
    fwd = compute_forward_returns(bond_3x_ret)

    print("Evaluating bond signals...")
    results = run_evaluation('Bond', signals, fwd, dates)

    out = os.path.join(BASE_DIR, 'research_bond_signals.csv')
    results.to_csv(out, index=False)
    print(f"Saved: {out} ({len(results)} rows)")
    return results

def run_part5(df):
    """Cross-asset correlation analysis."""
    print("Running correlation analysis...")
    corr_df, regime_df = run_correlation_analysis(df)

    out1 = os.path.join(BASE_DIR, 'research_correlation_regimes.csv')
    corr_df.to_csv(out1, index=False)
    print(f"Saved: {out1}")

    out2 = os.path.join(BASE_DIR, 'research_regime_returns.csv')
    regime_df.to_csv(out2, index=False)
    print(f"Saved: {out2}")

    print("\nCorrelation Summary:")
    print(corr_df.to_string(index=False))
    print("\nRegime Returns:")
    print(regime_df.to_string(index=False))
    return corr_df, regime_df

def run_part7(gold_results=None, bond_results=None):
    """Multiple testing correction and summary."""
    if gold_results is None:
        gold_results = pd.read_csv(os.path.join(BASE_DIR, 'research_gold_signals.csv'))
    if bond_results is None:
        bond_results = pd.read_csv(os.path.join(BASE_DIR, 'research_bond_signals.csv'))

    all_r, filt = generate_summary(gold_results, bond_results)

    out1 = os.path.join(BASE_DIR, 'research_all_corrected.csv')
    all_r.to_csv(out1, index=False)
    print(f"Saved: {out1} ({len(all_r)} rows)")

    out2 = os.path.join(BASE_DIR, 'research_summary.csv')
    filt.to_csv(out2, index=False)
    print(f"Saved: {out2} ({len(filt)} rows passing all filters)")

    # Print top results
    print(f"\n{'='*100}")
    print("SIGNALS PASSING ALL FILTERS (|IC|>0.03, WF>=5/7, FDR sig)")
    print(f"{'='*100}")

    for asset in ['Gold', 'Bond']:
        af = filt[filt['asset'] == asset]
        if len(af) == 0:
            print(f"\n{asset}: No signals passed all filters")
            continue
        print(f"\n{asset} — {len(af)} signal-horizon combinations passed:")
        print(f"{'Indicator':<25} {'H':>3} {'IC':>7} {'WF_avg':>7} {'WF_stab':>7} {'Spread':>8} {'HitRate':>8}")
        print("-" * 75)
        for _, r in af.head(20).iterrows():
            print(f"{r['indicator']:<25} {r['horizon']:>3} {r['IC']:>+7.4f} "
                  f"{r['WF_avg_IC']:>+7.4f} {int(r['WF_stable']):>5}/7  "
                  f"{r.get('spread', 0):>+7.4f} {r.get('hit_rate_high', 0):>7.1%}")

    return all_r, filt


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if part in ['0', 'data', 'all']:
        df = run_part0()
    else:
        cache = os.path.join(BASE_DIR, 'research_data_cache.pkl')
        if os.path.exists(cache):
            df = pd.read_pickle(cache)
            print(f"Loaded cache: {len(df)} rows")
        else:
            df = run_part0()

    if part in ['3', 'gold', 'all']:
        gold_r = run_part3(df)

    if part in ['4', 'bond', 'all']:
        bond_r = run_part4(df)

    if part in ['5', 'corr', 'all']:
        run_part5(df)

    if part in ['7', 'summary', 'all']:
        gr = gold_r if 'gold_r' in dir() else None
        br = bond_r if 'bond_r' in dir() else None
        run_part7(gr, br)

    print("\n=== DONE ===")
