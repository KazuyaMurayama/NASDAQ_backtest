import os
import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import requests

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
NASDAQ_FILE = os.path.join(BASE_DIR, "NASDAQ_extended_to_2026.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "timing_signals_raw.csv")

FETCH_START = "1974-01-01"
FETCH_END = "2026-03-26"
CACHE_TTL_DAYS = 7

# CPI publication lag: BLS releases CPIAUCSL for month M around the 10th-15th
# business day of month M+1. We shift each monthly observation forward by this
# many business days to mark when the figure is actually available in a backtest.
PUBLICATION_LAG_BDAYS = 15

FRED_SERIES = {
    "T10Y2Y":       (os.path.join(DATA_DIR, "t10y2y_daily.csv"),    "1976-06-01"),
    "DFF":          (os.path.join(DATA_DIR, "dff_daily.csv"),       "1954-07-01"),
    "CPIAUCSL":     (os.path.join(DATA_DIR, "cpiaucsl_monthly.csv"),"1947-01-01"),
    # HY spread splice components
    "BAMLH0A0HYM2": (os.path.join(DATA_DIR, "hy_spread_daily.csv"), "1996-12-31"),
    "BAA10Y":       (os.path.join(DATA_DIR, "baa10y_daily.csv"),    "1986-01-02"),
    # Monthly BAA/AAA for pre-1986 coverage (go back to 1919)
    "BAA":          (os.path.join(DATA_DIR, "baa_monthly.csv"),     "1919-01-01"),
    "AAA":          (os.path.join(DATA_DIR, "aaa_monthly.csv"),     "1919-01-01"),
}

KEY_DATES = ["1974-01-02", "1981-06-01", "1994-03-01", "2000-03-01",
             "2008-09-15", "2022-01-03", "2026-03-26"]


def _is_fresh(path):
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(days=CACHE_TTL_DAYS)


def _fetch_csv_endpoint(series_id, start, end):
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={series_id}&cosd={start}&coed={end}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.set_index(date_col)[val_col].dropna()
    s.name = series_id
    s.index.name = "DATE"
    return s


def fetch_series(series_id, output_path, start, end):
    if _is_fresh(output_path):
        s = pd.read_csv(output_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = series_id
        return s

    last_err = None
    for attempt in range(3):
        try:
            s = _fetch_csv_endpoint(series_id, start, end)
            if s.empty:
                raise RuntimeError(f"empty result for {series_id}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            s.to_frame(series_id).to_csv(output_path)
            return s
        except Exception as e:
            last_err = e
            wait = 2 ** (attempt + 1)
            print(f"  [{series_id}] attempt {attempt+1} failed: {e}; sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {series_id} after 3 attempts: {last_err}")


def load_nasdaq_index():
    df = pd.read_csv(NASDAQ_FILE)
    date_col = next(c for c in df.columns if c.lower() in ("date", "datetime"))
    idx = pd.to_datetime(df[date_col]).drop_duplicates().sort_values()
    return pd.DatetimeIndex(idx)


def splice_hy_spread(hy_oas, baa10y, baa_monthly, aaa_monthly, master_idx):
    # Layer 1: BAMLH0A0HYM2 (true HY OAS) from whenever it actually starts
    hy = hy_oas.reindex(master_idx).ffill()
    # Layer 2: BAA10Y daily (investment grade proxy, level-shifted to match HY at anchor1)
    baa = baa10y.reindex(master_idx).ffill()
    # Layer 3: BAA-AAA monthly spread (forward-filled daily, level-shifted to match baa_adj at anchor2)
    baa_aaa_monthly = (baa_monthly - aaa_monthly).dropna()
    # Forward-fill monthly to daily on master_idx
    baa_aaa = (baa_aaa_monthly
               .reindex(master_idx.union(baa_aaa_monthly.index))
               .ffill()
               .reindex(master_idx))

    anchor1 = hy_oas.first_valid_index()
    anchor2 = baa10y.first_valid_index()
    anchor1 = master_idx[master_idx >= anchor1][0]
    anchor2 = master_idx[master_idx >= anchor2][0]

    # Level-shift BAA10Y to match HY OAS at anchor1
    offset1 = hy.loc[anchor1] - baa.loc[anchor1]
    baa_adj = baa + offset1

    # Level-shift BAA-AAA monthly to match adjusted BAA10Y at anchor2
    offset2 = baa_adj.loc[anchor2] - baa_aaa.loc[anchor2]
    baa_aaa_adj = baa_aaa + offset2

    out = pd.Series(index=master_idx, dtype=float)
    out[master_idx >= anchor1]                                    = hy[master_idx >= anchor1]
    out[(master_idx >= anchor2) & (master_idx < anchor1)]        = baa_adj[(master_idx >= anchor2) & (master_idx < anchor1)]
    out[master_idx < anchor2]                                     = baa_aaa_adj[master_idx < anchor2]

    print(f"  splice: anchor1={anchor1.date()} (HY OAS start) offset1={offset1:.3f}")
    print(f"          anchor2={anchor2.date()} (BAA10Y start) offset2={offset2:.3f}")
    print(f"          pre-anchor2: BAA-AAA monthly proxy")
    return out


def build_cpi_features(cpi_monthly, master_idx):
    cpi = cpi_monthly.copy()
    cpi.index = pd.to_datetime(cpi.index).normalize()
    yoy = cpi.pct_change(12) * 100.0

    shifted_idx = yoy.index + BDay(PUBLICATION_LAG_BDAYS)
    yoy_published = pd.Series(yoy.values, index=shifted_idx).dropna()
    yoy_published = yoy_published[~yoy_published.index.duplicated(keep="last")]

    cpi_yoy_daily = yoy_published.reindex(
        master_idx.union(yoy_published.index)
    ).ffill().reindex(master_idx)

    cpi_accel = cpi_yoy_daily - cpi_yoy_daily.shift(63)
    return cpi_yoy_daily, cpi_accel


def build_combined(master_idx, raw):
    df = pd.DataFrame(index=master_idx)

    df["hy_spread"] = splice_hy_spread(
        raw["BAMLH0A0HYM2"], raw["BAA10Y"],
        raw["BAA"], raw["AAA"], master_idx
    )
    df["t10y2y"] = raw["T10Y2Y"].reindex(master_idx).ffill()
    df["dff"] = raw["DFF"].reindex(master_idx).ffill()

    cpi_yoy, cpi_accel = build_cpi_features(raw["CPIAUCSL"], master_idx)
    df["cpi_yoy"] = cpi_yoy
    df["cpi_accel"] = cpi_accel

    df.index.name = "DATE"
    return df


def print_summary(raw, combined):
    print("\n=== Raw series ===")
    for sid, s in raw.items():
        print(f"  {sid:14s} {s.index.min().date()} -> {s.index.max().date()}  rows={len(s)}")

    print("\n=== Combined NaN % ===")
    for col in combined.columns:
        pct = combined[col].isna().mean() * 100
        print(f"  {col:12s} NaN={pct:5.2f}%  min={combined[col].min():.3f}  "
              f"max={combined[col].max():.3f}  mean={combined[col].mean():.3f}")

    print("\n=== Key-date samples (iloc) ===")
    print(f"  {'date':12s} " + " ".join(f"{c:>10s}" for c in combined.columns))
    for d in KEY_DATES:
        ts = pd.Timestamp(d)
        if ts < combined.index[0] or ts > combined.index[-1]:
            continue
        # Use iloc-based nearest lookup to avoid asof NaN issues
        pos = combined.index.searchsorted(ts)
        pos = min(pos, len(combined) - 1)
        row = combined.iloc[pos]
        vals = " ".join(f"{row[c]:10.3f}" if pd.notna(row[c]) else f"{'NaN':>10s}"
                        for c in combined.columns)
        print(f"  {d:12s} {vals}")

    print("\n=== Cross-check: hy_spread peak in 2022-06 ===")
    seg = combined.loc["2022-06", "hy_spread"]
    peak = seg.max()
    print(f"  max hy_spread in 2022-06 = {peak:.3f}  (expected ~5-6 for true HY OAS; BAA10Y proxy ~4.8)")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    raw = {}
    print("=== Fetching FRED series ===")
    for sid, (path, start) in FRED_SERIES.items():
        print(f"fetching {sid} from {start}")
        raw[sid] = fetch_series(sid, path, start, FETCH_END)

    print("=== Loading NASDAQ business-day index ===")
    master_idx = load_nasdaq_index()
    print(f"  {master_idx[0].date()} -> {master_idx[-1].date()}  rows={len(master_idx)}")

    print("=== Building combined frame ===")
    combined = build_combined(master_idx, raw)
    combined.to_csv(OUTPUT_FILE)
    print(f"  wrote {OUTPUT_FILE}  shape={combined.shape}")

    print_summary(raw, combined)


if __name__ == "__main__":
    main()
