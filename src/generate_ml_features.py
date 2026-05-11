"""
Generate ML feature table by combining:
  - base_dataset.csv (price series)
  - macro_features.csv (macro/calendar features)
  - leverage_daily_detail.csv (existing DH Dyn signals)
  - NASDAQ OHLC technical features

Output: data/ml_features.csv
Columns:
  - date (index)
  - all features (no lookahead bias)
  - target_ret_21d  : log return of NASDAQ 3x over next 21 days
  - target_ret_5d   : log return of NASDAQ 3x over next 5 days
  - target_sharpe21 : 21-day realized Sharpe of NASDAQ 3x returns
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, win: int = 14) -> pd.Series:
    delta = close.diff()
    up   = delta.clip(lower=0).ewm(com=win - 1, adjust=False).mean()
    down = (-delta).clip(lower=0).ewm(com=win - 1, adjust=False).mean()
    rs   = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f'rsi_{win}')


def bollinger_pct(close: pd.Series, win: int = 20, n_std: float = 2.0) -> pd.Series:
    """Position within Bollinger Band: 0=lower, 1=upper."""
    ma  = close.rolling(win).mean()
    std = close.rolling(win).std()
    return ((close - (ma - n_std * std)) / (2 * n_std * std)).rename(f'bb_pct_{win}')


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    ema_f   = close.ewm(span=fast, adjust=False).mean()
    ema_s   = close.ewm(span=slow, adjust=False).mean()
    macd    = ema_f - ema_s
    signal  = macd.ewm(span=sig, adjust=False).mean()
    return (macd - signal).rename(f'macd_hist_{fast}_{slow}')


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, win: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return (tr.rolling(win).mean() / close).rename(f'atr_pct_{win}')


def donchian_pct(close: pd.Series, win: int = 20) -> pd.Series:
    """Position within Donchian channel: 0=lower, 1=upper."""
    hi = close.rolling(win).max()
    lo = close.rolling(win).min()
    rng = (hi - lo).replace(0, np.nan)
    return ((close - lo) / rng).rename(f'donchian_pct_{win}')


def hurst_approx(ret: pd.Series, win: int = 100) -> pd.Series:
    """
    Simplified Hurst exponent via R/S analysis on rolling window.
    H > 0.5 = trending, H < 0.5 = mean-reverting.
    """
    def _rs(x):
        x = x.dropna()
        if len(x) < 10:
            return np.nan
        z = np.cumsum(x - x.mean())
        r = z.max() - z.min()
        s = x.std(ddof=1)
        return r / s if s > 0 else np.nan

    rs = ret.rolling(win).apply(_rs, raw=False)
    n  = win
    return (np.log(rs) / np.log(n)).rename(f'hurst_{win}')


def rolling_sharpe(ret: pd.Series, win: int) -> pd.Series:
    mu  = ret.rolling(win).mean() * 252
    sig = ret.rolling(win).std()  * np.sqrt(252)
    return (mu / sig.replace(0, np.nan)).rename(f'sharpe_{win}')


def parkinson_vol(high: pd.Series, low: pd.Series, win: int = 21) -> pd.Series:
    log_hl = np.log(high / low)
    return np.sqrt(
        (log_hl ** 2 / (4 * np.log(2))).rolling(win).mean() * 252
    ).rename(f'park_vol_{win}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load sources ---
    base  = pd.read_csv(os.path.join(DATA_DIR, 'base_dataset.csv'),
                        parse_dates=['date'], index_col='date').sort_index()
    macro = pd.read_csv(os.path.join(DATA_DIR, 'macro_features.csv'),
                        parse_dates=['date'], index_col='date').sort_index()

    # Existing DH Dyn signals from leverage_daily_detail.csv
    dh_path = os.path.join(BASE_DIR, 'leverage_daily_detail.csv')
    dh = pd.read_csv(dh_path, parse_dates=['date'], index_col='date').sort_index()
    # Keep only signal columns (exclude fwd_1y_return which is forward-looking)
    dh_cols = ['asym_vol', 'trend_tv', 'vt', 'slope_mult', 'mom_decel', 'raw_leverage', 'dd']
    dh = dh[[c for c in dh_cols if c in dh.columns]]

    # NASDAQ OHLC
    nas_raw = pd.read_csv(os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv'),
                          parse_dates=['Date'], index_col='Date').sort_index()
    nas_raw.index.name = 'date'

    idx = base.index
    out = pd.DataFrame(index=idx)
    out.index.name = 'date'

    # --- NASDAQ price-derived technical features ---
    c = base['nasdaq_close']
    r = base['nasdaq_ret']          # log returns

    o = nas_raw['Open'].reindex(idx, method='ffill')
    h = nas_raw['High'].reindex(idx, method='ffill')
    l = nas_raw['Low'].reindex(idx, method='ffill')

    # Momentum (price ratios)
    for w in [5, 10, 21, 42, 63, 126, 252]:
        out[f'nas_ret{w}'] = np.log(c / c.shift(w))

    # Distance from moving averages (normalized by current price)
    for w in [21, 50, 100, 200]:
        ma = c.rolling(w).mean()
        out[f'nas_ma_dist{w}'] = (c - ma) / ma

    # RSI
    for w in [7, 14, 28]:
        out[f'nas_{rsi(c, w).name}'] = rsi(c, w)

    # Bollinger Band
    for w in [20, 60]:
        out[f'nas_{bollinger_pct(c, w).name}'] = bollinger_pct(c, w)

    # MACD
    out['nas_macd_hist'] = macd_hist(c)

    # ATR (% of price)
    out['nas_atr_pct14'] = atr_pct(h, l, c, 14)

    # Donchian channel
    for w in [20, 60]:
        out[f'nas_{donchian_pct(c, w).name}'] = donchian_pct(c, w)

    # Realized volatility (annualised %)
    for w in [5, 10, 21, 63]:
        out[f'nas_vol{w}'] = r.rolling(w).std() * np.sqrt(252)

    # Parkinson volatility
    out['nas_park_vol21'] = parkinson_vol(h, l, 21)

    # Rolling Sharpe
    for w in [21, 63, 126]:
        out[f'nas_{rolling_sharpe(r, w).name}'] = rolling_sharpe(r, w)

    # Return skewness and kurtosis
    for w in [63, 126]:
        out[f'nas_skew{w}']  = r.rolling(w).skew()
        out[f'nas_kurt{w}']  = r.rolling(w).kurt()

    # Drawdown from rolling peak
    peak = c.expanding().max()
    out['nas_dd_ratio'] = c / peak

    # Volatility ratio (short/long) — vol regime indicator
    out['nas_vol_ratio_5_63'] = (
        r.rolling(5).std() / r.rolling(63).std().replace(0, np.nan)
    )

    # Hurst exponent (trending vs mean-reverting)
    out['nas_hurst100'] = hurst_approx(r, 100)

    # --- Gold technical features ---
    g  = base['gold_usd']
    gr = base['gold_ret']

    for w in [21, 63, 126, 252]:
        out[f'gold_ret{w}'] = np.log(g / g.shift(w))

    for w in [21, 63, 200]:
        ma = g.rolling(w).mean()
        out[f'gold_ma_dist{w}'] = (g - ma) / ma

    out['gold_rsi14']   = rsi(g, 14)
    out['gold_vol21']   = gr.rolling(21).std() * np.sqrt(252)
    out['gold_vol63']   = gr.rolling(63).std() * np.sqrt(252)

    # --- Bond technical features ---
    b  = base['bond_price']
    br = base['bond_ret']

    for w in [21, 63, 126]:
        out[f'bond_ret{w}'] = np.log(b / b.shift(w))

    for w in [21, 63]:
        ma = b.rolling(w).mean()
        out[f'bond_ma_dist{w}'] = (b - ma) / ma

    out['bond_vol21'] = br.rolling(21).std() * np.sqrt(252)
    out['bond_vol63'] = br.rolling(63).std() * np.sqrt(252)

    # --- Rolling cross-asset correlations ---
    for w in [21, 63]:
        out[f'corr_nas_gold_{w}']  = r.rolling(w).corr(gr)
        out[f'corr_nas_bond_{w}']  = r.rolling(w).corr(br)
        out[f'corr_gold_bond_{w}'] = gr.rolling(w).corr(br)

    # --- VIX features ---
    vix = base['vix']
    out['vix'] = vix
    # Vol of vol
    out['vix_vov21'] = vix.rolling(21).std()

    # --- Existing DH Dyn signals ---
    for col in dh.columns:
        out[f'dh_{col}'] = dh[col].reindex(idx, method='ffill')

    # --- Merge macro features ---
    out = out.join(macro, how='left')

    # --- Target variables (FORWARD-LOOKING — only for training, not prediction) ---
    # 3x NASDAQ simulated return (simplified: 3x daily log return, no decay)
    r3x = r * 3.0   # simplification; actual strategy uses dynamic leverage

    out['target_ret_5d']   = r3x.shift(-5).rolling(5).sum()
    out['target_ret_21d']  = r3x.shift(-21).rolling(21).sum()
    # Risk-adjusted: 21-day Sharpe of 3x returns (forward)
    out['target_sharpe21'] = (
        r3x.shift(-21).rolling(21).mean() * 252
    ) / (r3x.shift(-21).rolling(21).std() * np.sqrt(252)).replace(0, np.nan)

    # Meta-labeling target: does DH raw_leverage direction agree with actual 21d outcome?
    # 1 = DH signal correct (invest AND market up, OR cash AND market down)
    # 0 = DH signal wrong
    # NaN = raw_leverage unavailable
    if 'dh_raw_leverage' in out.columns:
        dh_invest = (out['dh_raw_leverage'] > 0.5).astype(float)
        fwd_pos   = (out['target_ret_21d'] > 0).astype(float)
        # correct = both agree in direction
        out['target_meta_21d'] = np.where(
            out['dh_raw_leverage'].isna() | out['target_ret_21d'].isna(),
            np.nan,
            (dh_invest == fwd_pos).astype(float)
        )

    # --- Save ---
    out_path = os.path.join(DATA_DIR, 'ml_features.csv')
    out.to_csv(out_path, float_format='%.6f')

    n_feat = len(out.columns)
    nan_pct = out.drop(columns=[c for c in out.columns if c.startswith('target_')],
                       errors='ignore').isna().mean().mean() * 100
    print(f"Saved: {out_path}")
    print(f"  Rows: {len(out):,}  Total columns: {n_feat}")
    print(f"  Feature avg missing (excl targets): {nan_pct:.2f}%")
    print(f"  Period: {out.index[0].date()} to {out.index[-1].date()}")

    # Column summary by category
    cats = {
        'NASDAQ technical': [c for c in out.columns if c.startswith('nas_') and not c.startswith('nas_gold')],
        'Gold technical':   [c for c in out.columns if c.startswith('gold_')],
        'Bond technical':   [c for c in out.columns if c.startswith('bond_')],
        'Cross-asset corr': [c for c in out.columns if c.startswith('corr_')],
        'DH Dyn signals':   [c for c in out.columns if c.startswith('dh_')],
        'Macro / calendar': [c for c in macro.columns],
        'Targets':          [c for c in out.columns if c.startswith('target_')],
    }
    for cat, cols in cats.items():
        print(f"  {cat}: {len(cols)} features")

    return out


if __name__ == '__main__':
    main()
