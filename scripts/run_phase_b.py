"""End-to-end Phase B runner: 7 available signals x 3 assets x 3 horizons.

Steps:
  1. Load forward returns (NDX/IEF/GLD from 2009)
  2. Build 7 signal series from repo CSV data files
  3. Apply quantile_cut (4 levels) + daily publication_lag
  4. For each (signal, asset, horizon in {5,20,60}), evaluate_triple
  5. Apply BH-FDR + PASS/FAIL judgment
  6. Write scorecard CSV + report MD + Phase C selection CSV

Output (under data/signals/):
  - scorecard_<date>.csv
  - screening_report_<date>.md
  - phase_b_selection_<date>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from signals.forward_returns import build_forward_returns, load_default_prices
from signals.quantize import quantile_cut
from signals.timing import apply_publication_lag
from signals.screening import (
    batch_evaluate,
    apply_fdr_and_judgment,
    generate_report_markdown,
)


RUN_DATE = '20260604'
RUN_DATE_DASH = '2026-06-04'


def _read_csv_with_date(path: Path) -> pd.DataFrame:
    """Load a CSV that uses either 'Date' or 'DATE' as the index column."""
    df = pd.read_csv(path)
    date_col = None
    for c in df.columns:
        if c.lower() == 'date':
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = 'Date'
    return df


def load_signal_series() -> dict:
    """Load 7 available signals from repo CSV files.

    Returns dict[signal_id] = (signal_name, raw_series).
    """
    data_dir = ROOT / 'data'
    out: dict = {}

    # #6 VIX level
    df = _read_csv_with_date(data_dir / 'vixcls_daily.csv')
    out[6] = ('VIX level', df.iloc[:, 0].astype(float).dropna())

    # #21 ICE BofA HY OAS (BAMLH0A0HYM2)
    df = _read_csv_with_date(data_dir / 'hy_spread_daily.csv')
    out[21] = ('ICE BofA HY OAS', df.iloc[:, 0].astype(float).dropna())

    # #23 BAA10Y credit spread (HY-IG proxy)
    try:
        df = _read_csv_with_date(data_dir / 'baa10y_daily.csv')
        out[23] = ('BAA-10Y credit spread (HY-IG proxy)', df.iloc[:, 0].astype(float).dropna())
    except Exception as e:
        print(f"  (skip #23: {e})")

    # #26 2s10s spread = DGS10 - DGS2
    try:
        dgs10 = _read_csv_with_date(data_dir / 'dgs10_daily.csv').iloc[:, 0].astype(float)
        dgs2 = _read_csv_with_date(data_dir / 'dgs2_daily.csv').iloc[:, 0].astype(float)
        spread = (dgs10 - dgs2).dropna()
        out[26] = ('2s10s spread (DGS10-DGS2)', spread)
    except Exception as e:
        print(f"  (skip #26: {e})")

    # #27 3M10Y spread = DGS10 - DTB3
    try:
        dgs10 = _read_csv_with_date(data_dir / 'dgs10_daily.csv').iloc[:, 0].astype(float)
        dtb3 = _read_csv_with_date(data_dir / 'dtb3_daily.csv').iloc[:, 0].astype(float)
        spread = (dgs10 - dtb3).dropna()
        out[27] = ('3M10Y spread (DGS10-DTB3)', spread)
    except Exception as e:
        print(f"  (skip #27: {e})")

    # #41 DXY
    try:
        dxy = _read_csv_with_date(data_dir / 'dxy_daily.csv').iloc[:, 0].astype(float).dropna()
        out[41] = ('DXY', dxy)
    except Exception as e:
        print(f"  (skip #41: {e})")

    # #28 10Y real yield (DGS10 - CPI YoY%, CPI forward-filled to business days)
    try:
        dgs10 = _read_csv_with_date(data_dir / 'dgs10_daily.csv').iloc[:, 0].astype(float)
        cpi = _read_csv_with_date(data_dir / 'cpiaucsl_monthly.csv').iloc[:, 0].astype(float)
        cpi_yoy = (cpi / cpi.shift(12) - 1.0) * 100.0
        cpi_daily = cpi_yoy.resample('B').ffill()
        real = (dgs10 - cpi_daily).dropna()
        out[28] = ('10Y real yield (DGS10 - CPI YoY)', real)
    except Exception as e:
        print(f"  (skip #28: {e})")

    return out


def main():
    print("[Phase B] Loading forward returns (NDX/IEF/GLD)...")
    prices = load_default_prices()
    print(f"  prices: {len(prices)} rows, {prices.index[0].date()} -> {prices.index[-1].date()}")
    fr = build_forward_returns(prices, horizons=[5, 20, 60])
    print(f"  fr shape: {fr.shape}")

    print("[Phase B] Loading signal series from repo CSVs...")
    signals = load_signal_series()
    print(f"  loaded {len(signals)} signals: ids={sorted(signals.keys())}")
    for sid, (name, s) in signals.items():
        print(f"    #{sid:3d} {name:50s} n={len(s):6d} {s.index[0].date()} -> {s.index[-1].date()}")

    print("[Phase B] Quantizing (4-level) + daily publication lag...")
    quantized: dict = {}
    for sid, (name, raw) in signals.items():
        q = quantile_cut(raw, levels=4)
        q_lagged = apply_publication_lag(q, lag_type='daily')
        # Drop duplicate index entries from BusinessDay shift (rare).
        q_lagged = q_lagged[~q_lagged.index.duplicated(keep='first')]
        quantized[sid] = (name, q_lagged)
        print(f"    #{sid:3d} quantized: nonnull={int(q_lagged.notna().sum())}")

    print("[Phase B] Building triples (signal x asset x horizon)...")
    triples = []
    for sid, (name, sig) in quantized.items():
        for asset in ['NDX', 'IEF', 'GLD']:
            for h in [5, 20, 60]:
                triples.append((sid, name, sig, asset, h, fr[(asset, h)]))
    print(f"  {len(triples)} triples to evaluate")

    print("[Phase B] Computing IC + hit rate + stability per triple...")
    scorecard = batch_evaluate(triples, ic_window=252)
    print(f"  scorecard shape: {scorecard.shape}")

    print("[Phase B] Applying BH-FDR + PASS/FAIL judgment (alpha=0.10)...")
    scorecard = apply_fdr_and_judgment(scorecard, alpha=0.10)

    print("[Phase B] Writing outputs...")
    out_dir = ROOT / 'data' / 'signals'
    out_dir.mkdir(parents=True, exist_ok=True)

    scorecard_csv = out_dir / f'scorecard_{RUN_DATE}.csv'
    scorecard.to_csv(scorecard_csv, index=False)

    report_md = out_dir / f'screening_report_{RUN_DATE}.md'
    generate_report_markdown(
        scorecard, str(report_md),
        title="Phase B Screening Report",
        created_date=RUN_DATE_DASH,
    )

    selection_csv = out_dir / f'phase_b_selection_{RUN_DATE}.csv'
    pass_rows = scorecard[scorecard['pass_flag'] == True].copy()
    pass_rows.to_csv(selection_csv, index=False)

    n_pass = int((scorecard['pass_flag'] == True).sum())
    n_total = len(scorecard)
    print(f"[Phase B] Done. {n_pass}/{n_total} triples PASS.")
    print(f"  scorecard: {scorecard_csv}")
    print(f"  report:    {report_md}")
    print(f"  selection: {selection_csv}")


if __name__ == '__main__':
    main()
