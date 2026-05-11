"""
Fetch FRED data for bond yields (DGS10) and VIX (VIXCLS).
Saves to data/ folder as CSV.
"""
import os
import pandas as pd
import pandas_datareader.data as web

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def fetch_series(series_id: str, start: str, end: str, col_name: str, out_file: str) -> pd.DataFrame:
    df = web.DataReader(series_id, 'fred', start, end)
    df.columns = [col_name]
    df.index.name = 'Date'
    path = os.path.join(DATA_DIR, out_file)
    df.to_csv(path)
    valid = df[col_name].notna().sum()
    print(f"  {series_id}: {len(df)} rows ({valid} valid), "
          f"{df.index[0].date()} to {df.index[-1].date()} → {out_file}")
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    series_list = [
        # Already fetched but included for completeness
        ('DGS10',      '1962-01-02', '2026-05-09', 'yield_pct', 'dgs10_daily.csv'),
        ('VIXCLS',     '1990-01-02', '2026-05-09', 'vix',       'vixcls_daily.csv'),
        # Yield curve components
        ('DTB3',       '1954-01-04', '2026-05-09', 'yield_pct', 'dtb3_daily.csv'),
        ('DGS2',       '1976-06-01', '2026-05-09', 'yield_pct', 'dgs2_daily.csv'),
        ('DGS30',      '1977-02-15', '2026-05-09', 'yield_pct', 'dgs30_daily.csv'),
        # Credit spreads
        ('BAA10Y',     '1986-01-02', '2026-05-09', 'spread',    'baa10y_daily.csv'),
        ('BAMLH0A0HYM2','1996-12-31','2026-05-09', 'spread',    'hy_spread_daily.csv'),
        # Dollar index
        ('DTWEXBGS',   '2006-01-02', '2026-05-09', 'dxy',       'dxy_daily.csv'),
        # Financial conditions
        ('NFCI',       '1971-01-08', '2026-05-09', 'nfci',      'nfci_weekly.csv'),
    ]

    for args in series_list:
        try:
            fetch_series(*args)
        except Exception as exc:
            print(f"  WARN: {args[0]} failed: {exc}")

    print("Done.")


if __name__ == '__main__':
    main()
