"""Generate calendar-year annual returns (Jan-Dec) for SIGNAL_EXPANSION_FINAL_DECISION_20260607.md.

Evaluation Standard: v1.1
Outputs 5 strategies: S1_F10, S2_D5, S3_DH-W1, E4_Active, S3+nasdaq_mom63_M6_def
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pickle
import pandas as pd
import numpy as np


def annual_calendar_returns(nav: pd.Series, label: str) -> pd.Series:
    """Resample to year-end NAV then pct_change for calendar-year return."""
    yearly_nav = nav.resample('YE').last()
    rets = yearly_nav.pct_change()
    rets.name = label
    return rets


def main():
    bp = pd.read_parquet(ROOT / 'data' / 'signals' / 'integration' / 'baseline_navs_20260605.parquet')
    s1_nav = bp['S1'].dropna()
    s2_nav = bp['S2'].dropna()
    s3_nav = bp['S3'].dropna()

    c = pickle.load(open(ROOT / 'audit_results' / '_cache' / 'e4_nav_cache.pkl', 'rb'))
    dates_e4 = pd.to_datetime(c['dates'])
    nav_e4 = pd.Series(np.asarray(c['nav_e4']), index=dates_e4).dropna()

    from integration.build_strategy_with_signal import build_candidate_nav  # noqa: E402

    macro = pd.read_csv(
        ROOT / 'data' / 'macro_features.csv',
        parse_dates=['date'], index_col='date',
    ).sort_index()
    signal_raw = macro['nasdaq_mom63'].dropna()
    cand_s3 = build_candidate_nav('S3', signal_raw, 'M6', 'defensive').dropna()

    df = pd.concat([
        annual_calendar_returns(s1_nav, 'S1_F10'),
        annual_calendar_returns(s2_nav, 'S2_D5'),
        annual_calendar_returns(s3_nav, 'S3_DH-W1'),
        annual_calendar_returns(nav_e4, 'E4_Active'),
        annual_calendar_returns(cand_s3, 'S3+nasdaq_mom63_M6_def'),
    ], axis=1)

    df.index = df.index.year
    df.index.name = 'Year'

    out_csv = ROOT / 'data' / 'signals' / 'expansion' / 'decision_annual_returns_20260607.csv'
    df.to_csv(out_csv, float_format='%.6f')
    print(f'Wrote {out_csv}')

    pd.set_option('display.float_format', lambda x: f'{x*100:+7.2f}%' if not np.isnan(x) else '   N/A ')
    pd.set_option('display.width', 200)
    print('\n--- Last 15 calendar years ---')
    print(df.tail(15).to_string())
    print('\n--- First 5 calendar years ---')
    print(df.head(5).to_string())
    print(f'\nTotal rows: {len(df)}')


if __name__ == '__main__':
    main()
