"""
Build master dataset: NASDAQ / Gold / Bond (synthetic 1974-2009 + IEF 2009+) / VIX.
Output: data/base_dataset.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_nasdaq() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv'),
                     parse_dates=['Date'], index_col='Date').sort_index()
    return df


def load_gold() -> pd.Series:
    df = pd.read_csv(os.path.join(DATA_DIR, 'lbma_gold_daily.csv'),
                     parse_dates=['Date'], index_col='Date').sort_index()
    df.columns = ['gold_usd']
    return df['gold_usd']


def load_ief() -> pd.Series:
    df = pd.read_csv(os.path.join(DATA_DIR, 'ief_daily.csv'),
                     parse_dates=['Date'], index_col='Date').sort_index()
    df.columns = ['close']
    return df['close']


def load_dgs10() -> pd.Series:
    df = pd.read_csv(os.path.join(DATA_DIR, 'dgs10_daily.csv'),
                     parse_dates=['Date'], index_col='Date').sort_index()
    df.columns = ['yield_pct']
    s = pd.to_numeric(df['yield_pct'], errors='coerce')
    s = s.ffill(limit=3)
    return s


def load_vixcls() -> pd.Series:
    df = pd.read_csv(os.path.join(DATA_DIR, 'vixcls_daily.csv'),
                     parse_dates=['Date'], index_col='Date').sort_index()
    df.columns = ['vix']
    s = pd.to_numeric(df['vix'], errors='coerce')
    s = s.ffill(limit=3)
    return s


# ---------------------------------------------------------------------------
# Bond series
# ---------------------------------------------------------------------------

def synth_bond_ret(yield_pct: pd.Series, d_mod: float = 8.5, conv: float = 80.0) -> pd.Series:
    """Daily bond return from yield change via modified duration formula."""
    dy = yield_pct.diff() / 100.0   # percent → decimal
    return -d_mod * dy + 0.5 * conv * dy ** 2


def build_bond_series(dgs10: pd.Series, ief: pd.Series,
                      calib_start: str = '2009-07-01',
                      calib_end: str = '2024-12-31',
                      splice_date: str = '2009-07-01') -> tuple:
    """
    Returns:
        bond_ret  – daily price returns (pd.Series)
        bond_src  – 'synth' or 'ief' per date (pd.Series)
        d_mod_cal – calibrated modified duration
    """
    synth_ret = synth_bond_ret(dgs10, d_mod=8.5)
    ief_ret = ief.pct_change()

    # Calibrate d_mod: match return std to IEF in overlap period
    s, e = pd.Timestamp(calib_start), pd.Timestamp(calib_end)
    common = (synth_ret.loc[s:e].dropna().index
              .intersection(ief_ret.loc[s:e].dropna().index))
    scale = ief_ret.loc[common].std() / synth_ret.loc[common].std()
    d_mod_cal = 8.5 * scale
    print(f"  Bond calibration: vol_scale={scale:.4f} → d_mod={d_mod_cal:.4f}")

    # Recalculate synthetic with calibrated d_mod
    synth_ret_cal = synth_bond_ret(dgs10, d_mod=d_mod_cal)

    # Splice: synth everywhere, override with IEF at/after splice_date
    splice = pd.Timestamp(splice_date)
    bond_ret = synth_ret_cal.copy()
    ief_post = ief_ret[ief_ret.index >= splice].dropna()
    bond_ret[ief_post.index] = ief_post

    bond_src = pd.Series('synth', index=bond_ret.index, dtype=str)
    bond_src[bond_ret.index >= splice] = 'ief'

    return bond_ret, bond_src, d_mod_cal


# ---------------------------------------------------------------------------
# VIX series
# ---------------------------------------------------------------------------

def yang_zhang_vol(ohlc: pd.DataFrame, win: int = 21) -> pd.Series:
    """Yang-Zhang volatility estimator (annualised, in %)."""
    o = ohlc['Open'].replace(0, np.nan)
    h = ohlc['High'].replace(0, np.nan)
    l = ohlc['Low'].replace(0, np.nan)
    c = ohlc['Close'].replace(0, np.nan)

    log_oc = np.log(o / c.shift(1))   # overnight return
    log_co = np.log(c / o)             # intraday return
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_hc = np.log(h / c)
    log_lc = np.log(l / c)

    k = 0.34
    sigma_o  = log_oc.rolling(win).var()
    sigma_c  = log_co.rolling(win).var()
    sigma_rs = (log_ho * log_hc + log_lo * log_lc).rolling(win).mean()  # Rogers-Satchell

    yz = np.sqrt(sigma_o + k * sigma_c + (1.0 - k) * sigma_rs)
    return (yz * np.sqrt(252) * 100.0).rename('vix_proxy')


def build_vix_proxy(nasdaq: pd.DataFrame, vixcls: pd.Series,
                    calib_start: str = '1990-01-02',
                    calib_end: str = '2000-12-31') -> pd.Series:
    """
    Calibrate YZ proxy to VIXCLS scale.
    Returns proxy_scaled indexed on NASDAQ dates.
    """
    proxy = yang_zhang_vol(nasdaq)

    s, e = pd.Timestamp(calib_start), pd.Timestamp(calib_end)
    common = (proxy.loc[s:e].dropna().index
              .intersection(vixcls.loc[s:e].dropna().index))
    slope, intercept, r, _, _ = stats.linregress(
        proxy.loc[common].values, vixcls.loc[common].values
    )
    print(f"  VIX calibration: slope={slope:.4f} intercept={intercept:.4f} R2={r**2:.4f}")

    proxy_scaled = (proxy * slope + intercept).clip(lower=3.0)
    return proxy_scaled


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def quality_check(df: pd.DataFrame):
    print("\n=== Quality Check ===")
    for col in ['nasdaq_close', 'gold_usd', 'bond_price', 'vix']:
        pct = df[col].isna().mean() * 100
        flag = '!' if pct > 0.1 else 'OK'
        print(f"  {flag} {col} missing: {pct:.3f}%")

    if 'bond_ret' in df.columns:
        br_max = df['bond_ret'].abs().max()
        flag = '!' if br_max > 0.1 else 'OK'
        print(f"  {flag} bond_ret max_abs: {br_max:.4f}")

    if 'vix' in df.columns:
        pct_ok = ((df['vix'] >= 5) & (df['vix'] <= 90)).mean() * 100
        flag = 'OK' if pct_ok > 99 else '!'
        print(f"  {flag} vix in [5,90]: {pct_ok:.2f}%")

    gaps = (df.index[1:] - df.index[:-1]).days
    max_gap = int(gaps.max())
    flag = 'OK' if max_gap <= 5 else '!'
    print(f"  {flag} max_date_gap_days: {max_gap}")

    src = df.get('bond_source', pd.Series(dtype=str))
    if len(src):
        n_synth = (src == 'synth').sum()
        n_ief   = (src == 'ief').sum()
        print(f"  OK bond_source: synth={n_synth}, ief={n_ief}")

    src_v = df.get('vix_source', pd.Series(dtype=str))
    if len(src_v):
        n_prx  = (src_v == 'proxy').sum()
        n_act  = (src_v == 'actual').sum()
        print(f"  OK vix_source: proxy={n_prx}, actual={n_act}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading sources...")
    nasdaq  = load_nasdaq()
    gold    = load_gold()
    ief     = load_ief()
    dgs10   = load_dgs10()
    vixcls  = load_vixcls()

    print("Building bond series...")
    bond_ret, bond_src, d_mod_cal = build_bond_series(dgs10, ief)

    print("Building VIX proxy...")
    vix_proxy = build_vix_proxy(nasdaq, vixcls)

    print("Aligning on NASDAQ business days...")
    idx = nasdaq.index   # master calendar (NYSE business days)
    df  = pd.DataFrame(index=idx)
    df.index.name = 'date'

    df['nasdaq_close'] = nasdaq['Close']
    df['nasdaq_ret']   = np.log(df['nasdaq_close'] / df['nasdaq_close'].shift(1))

    df['gold_usd'] = gold.reindex(idx, method='ffill')
    df['gold_ret'] = np.log(df['gold_usd'] / df['gold_usd'].shift(1))

    # Bond: align returns, reconstruct price from cumulative returns
    br = bond_ret.reindex(idx)
    df['bond_price']  = (1.0 + br.fillna(0.0)).cumprod() * 100.0
    df['bond_ret']    = br
    df['bond_source'] = bond_src.reindex(idx, method='ffill').fillna('synth')

    # VIX: merge proxy (on NASDAQ idx) with actual VIXCLS (different cal)
    splice_vix = pd.Timestamp('1990-01-02')
    vix_proxy_aligned  = vix_proxy.reindex(idx, method='ffill')
    vixcls_aligned     = vixcls.reindex(idx, method='ffill')
    vix_combined       = vix_proxy_aligned.copy()
    use_actual         = (idx >= splice_vix) & vixcls_aligned.notna()
    vix_combined[use_actual] = vixcls_aligned[use_actual]
    df['vix']        = vix_combined
    df['vix_source'] = pd.Series(
        np.where(use_actual, 'actual', 'proxy'), index=idx, dtype=str
    )

    out_path = os.path.join(DATA_DIR, 'base_dataset.csv')
    df.to_csv(out_path)
    print(f"\nSaved: {out_path}")
    print(f"  Rows: {len(df):,}  Period: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Columns: {list(df.columns)}")

    quality_check(df)
    return df


if __name__ == '__main__':
    main()
