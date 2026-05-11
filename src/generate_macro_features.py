"""
Generate macro-derived features for ML pipeline.
Inputs:  data/base_dataset.csv + FRED CSV files in data/
Output:  data/macro_features.csv
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

PUBLICATION_LAG = {
    # series_id: business-day lag before data is publicly available
    'DTB3': 1,
    'DGS2': 1,
    'DGS10': 1,
    'DGS30': 1,
    'BAA10Y': 1,
    'NFCI': 5,   # released Friday, covers week ending prior Friday
}


def load_fred(fname: str, col: str, lag_days: int = 1) -> pd.Series:
    """Load FRED CSV, forward-fill, apply publication lag."""
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    s = s.ffill(limit=5)
    s.name = col
    # Shift to account for data not available until next business day
    s.index = s.index + pd.tseries.offsets.BDay(lag_days)
    return s


def rolling_zscore(s: pd.Series, win: int) -> pd.Series:
    mu  = s.rolling(win).mean()
    sig = s.rolling(win).std()
    return ((s - mu) / sig).rename(s.name + f'_z{win}')


def rolling_mom(s: pd.Series, win: int) -> pd.Series:
    return s.diff(win).rename(s.name + f'_mom{win}')


def main():
    # -----------------------------------------------------------------------
    # Load master index (NASDAQ business days)
    # -----------------------------------------------------------------------
    base = pd.read_csv(os.path.join(DATA_DIR, 'base_dataset.csv'),
                       parse_dates=['date'], index_col='date').sort_index()
    idx = base.index

    out = pd.DataFrame(index=idx)
    out.index.name = 'date'

    # -----------------------------------------------------------------------
    # Yield curve
    # -----------------------------------------------------------------------
    dgs10  = load_fred('dgs10_daily.csv',  'dgs10',  lag_days=1)
    dtb3   = load_fred('dtb3_daily.csv',   'dtb3',   lag_days=1)
    dgs2   = load_fred('dgs2_daily.csv',   'dgs2',   lag_days=1)
    dgs30  = load_fred('dgs30_daily.csv',  'dgs30',  lag_days=1)

    dgs10_a  = dgs10.reindex(idx, method='ffill')
    dtb3_a   = dtb3.reindex(idx, method='ffill')
    dgs2_a   = dgs2.reindex(idx, method='ffill')
    dgs30_a  = dgs30.reindex(idx, method='ffill')

    # Yield curve slopes (basis points)
    out['yc_3m10y']  = (dgs10_a - dtb3_a) * 100      # most reliable recession predictor
    out['yc_2s10s']  = (dgs10_a - dgs2_a) * 100
    out['yc_10y30y'] = (dgs30_a - dgs10_a) * 100

    # Level of 10-year yield + Z-scores
    out['dgs10_level'] = dgs10_a
    out['dgs10_z252']  = rolling_zscore(dgs10_a, 252)
    out['dgs10_mom63'] = rolling_mom(dgs10_a, 63)    # 3M change in yield

    # Inversion flag (Estrella-Mishkin recession predictor)
    out['yc_inverted'] = (out['yc_3m10y'] < 0).astype(float)

    # Slope Z-score
    out['yc_3m10y_z252'] = rolling_zscore(out['yc_3m10y'], 252)
    out['yc_3m10y_z63']  = rolling_zscore(out['yc_3m10y'], 63)

    # -----------------------------------------------------------------------
    # Credit spread (BAA10Y = Moody's Baa corporate - 10y Treasury)
    # Available from 1986; for pre-1986 leave NaN
    # -----------------------------------------------------------------------
    baa10y   = load_fred('baa10y_daily.csv', 'baa10y', lag_days=1)
    baa10y_a = baa10y.reindex(idx, method='ffill')

    out['credit_spread']      = baa10y_a
    out['credit_spread_z252'] = rolling_zscore(baa10y_a, 252)
    out['credit_spread_mom63'] = rolling_mom(baa10y_a, 63)

    # -----------------------------------------------------------------------
    # Dollar index (DTWEXBGS, 2006+; NaN before)
    # -----------------------------------------------------------------------
    dxy   = load_fred('dxy_daily.csv', 'dxy', lag_days=1)
    dxy_a = dxy.reindex(idx, method='ffill')

    out['dxy']        = dxy_a
    out['dxy_ret20']  = np.log(dxy_a / dxy_a.shift(20))   # 20-day momentum
    out['dxy_ret63']  = np.log(dxy_a / dxy_a.shift(63))
    out['dxy_z252']   = rolling_zscore(dxy_a, 252)

    # -----------------------------------------------------------------------
    # Chicago Fed National Financial Conditions Index (weekly, 1971+)
    # -----------------------------------------------------------------------
    nfci   = load_fred('nfci_weekly.csv', 'nfci', lag_days=5)
    nfci_a = nfci.reindex(idx, method='ffill')

    out['nfci']        = nfci_a
    out['nfci_z52w']   = rolling_zscore(nfci_a, 52)    # 52-week Z-score (~ 260 bdays)
    out['nfci_chg13w'] = nfci_a.diff(65)               # 13-week change

    # -----------------------------------------------------------------------
    # Calendar / seasonality features (label-encoded + cyclical encoding)
    # -----------------------------------------------------------------------
    out['month']        = idx.month.astype(float)
    out['month_sin']    = np.sin(2 * np.pi * idx.month / 12)
    out['month_cos']    = np.cos(2 * np.pi * idx.month / 12)
    out['dow']          = idx.dayofweek.astype(float)         # 0=Mon
    out['turn_of_month'] = (idx.day <= 3).astype(float)        # first 3 days
    out['pre_month_end'] = (idx.day >= 26).astype(float)       # last ~3 days

    # "Sell in May" seasonal regime (May–Oct = 0, Nov–Apr = 1)
    out['halloween_regime'] = ((idx.month >= 11) | (idx.month <= 4)).astype(float)

    # Presidential cycle (year mod 4: 0=election yr, 1=post, 2=mid, 3=pre)
    out['pres_cycle'] = (idx.year % 4).astype(float)

    # -----------------------------------------------------------------------
    # Cross-asset momentum (on base_dataset series)
    # -----------------------------------------------------------------------
    nasdaq  = base['nasdaq_close']
    gold    = base['gold_usd']
    bond    = base['bond_price']

    for name, s in [('nasdaq', nasdaq), ('gold', gold), ('bond', bond)]:
        log_p = np.log(s)
        for w in [21, 63, 126, 252]:
            out[f'{name}_mom{w}'] = log_p - log_p.shift(w)

    # NASDAQ vs Gold / Bond relative momentum (63-day)
    out['nas_gold_rel63']  = out['nasdaq_mom63']  - out['gold_mom63']
    out['nas_bond_rel63']  = out['nasdaq_mom63']  - out['bond_mom63']
    out['gold_bond_rel63'] = out['gold_mom63']    - out['bond_mom63']

    # -----------------------------------------------------------------------
    # VIX-derived macro features
    # -----------------------------------------------------------------------
    vix = base['vix']
    out['vix_level']   = vix
    out['vix_z252']    = rolling_zscore(vix, 252)
    out['vix_z63']     = rolling_zscore(vix, 63)
    out['vix_mom21']   = vix.diff(21)
    out['vix_ret_5d']  = np.log(vix / vix.shift(5))

    # VIX regime: high vol (>25) / low vol (<15)
    out['vix_high_regime'] = (vix > 25).astype(float)
    out['vix_low_regime']  = (vix < 15).astype(float)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = os.path.join(DATA_DIR, 'macro_features.csv')
    out.to_csv(out_path, float_format='%.6f')

    n_feat = len(out.columns)
    n_rows = len(out)
    nan_pct = out.isna().mean().mean() * 100
    print(f"Saved: {out_path}")
    print(f"  Rows: {n_rows:,}  Features: {n_feat}  Avg missing: {nan_pct:.2f}%")
    print(f"  Period: {out.index[0].date()} to {out.index[-1].date()}")
    print(f"  Columns: {list(out.columns)}")
    return out


if __name__ == '__main__':
    main()
