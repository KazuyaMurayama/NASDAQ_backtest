"""[SUPERSEDED by compute_aftertax_cagr_v2_20260607.py — 2026-06-07]

Bug: this v1 script normalizes by `years = len(annual_returns)`, which treats
2026 YTD (Jan-Mar, ~60 business days) as a full year and drags after-tax CAGR
down by 3-9pp across all strategies.  V2 normalizes by *actual elapsed days /
365.25* (NAV-endpoint formula), correctly handling the 2026 partial year.

All reports (SIGNAL_EXPANSION_FINAL_DECISION_20260607.md §3.6,
STRATEGY_REGISTRY.md, CURRENT_BEST_STRATEGY.md, S3_OVERLAY_TUNING_REPORT_20260607.md §11,
LESSONS_LEARNED_20260607.md §6) use the V2 numbers.  Kept for reproducibility
of the original (incorrect) figures referenced in commit 93a763a / c76e244.

------ original docstring below ------
Compute after-tax CAGR for all reporting strategies (2026-06-07).

Convention (per CURRENT_BEST_STRATEGY.md and existing
g21f_dh_t4_yearly_returns_aftertax.csv):
  after-tax annual return = annual_return * 0.8273    (regardless of sign)
  after-tax CAGR = compound(after-tax annual returns) ^ (1/years) - 1

This is the simplified uniform-drag convention used throughout the repo
(applies the 20.315% tax factor 1-0.20315=0.79685 ~ rounded to the
0.8273 multiplier that incorporates a partial-realization assumption).

Outputs:
  data/signals/expansion/aftertax_cagr_20260607.csv
  - 6 strategies (S1, S2, S3, E4, S3+V0, S3+V7) x IS/OOS/FULL pretax/aftertax
  - V7 NAV reconstructed on the fly from build_candidate_nav

Usage:
  python scripts/compute_aftertax_cagr_20260607.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd

from integration.build_strategy_with_signal import build_candidate_nav  # noqa: E402


TAX_MULT = 0.8273  # 1 - 0.1727 (existing repo convention)

# Calendar-year IS / OOS boundary (matches the decision_annual_returns CSV).
# canonical daily split is 2021-05-08; for calendar-year analysis we treat
# 2021 as the first full OOS year.
IS_END_YEAR = 2020   # IS: years <= 2020
OOS_START_YEAR = 2021  # OOS: years >= 2021


def annual_returns_from_nav(nav: pd.Series) -> pd.Series:
    """Return calendar-year returns indexed by year-end timestamp."""
    nav = nav.dropna()
    ye = nav.resample('YE').last()
    return ye.pct_change().dropna()


def _filter_period(r: pd.Series, period: str) -> pd.Series:
    """Slice annual returns to IS / OOS / FULL by calendar year."""
    years = r.index.year if hasattr(r.index, 'year') else pd.Index(r.index)
    if period == 'IS':
        mask = years <= IS_END_YEAR
    elif period == 'OOS':
        mask = years >= OOS_START_YEAR
    elif period == 'FULL':
        mask = np.ones(len(r), dtype=bool)
    else:
        raise ValueError(period)
    return r.loc[mask]


def _cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 1:
        return float('nan')
    total = float((1.0 + r).prod())
    years = len(r)
    return total ** (1.0 / years) - 1.0


def pretax_cagr(annual_rets: pd.Series, period: str) -> float:
    return _cagr(_filter_period(annual_rets, period))


def aftertax_cagr(annual_rets: pd.Series, period: str,
                  tax_mult: float = TAX_MULT) -> float:
    """Apply tax_mult to each annual return then compound."""
    r = _filter_period(annual_rets, period)
    after = r * tax_mult
    return _cagr(after)


def metric_set(annual_rets: pd.Series, label: str) -> dict:
    return {
        'label': label,
        'n_IS':  int((annual_rets.index.year <= IS_END_YEAR).sum()),
        'n_OOS': int((annual_rets.index.year >= OOS_START_YEAR).sum()),
        'CAGR_IS_pretax':    pretax_cagr(annual_rets, 'IS'),
        'CAGR_OOS_pretax':   pretax_cagr(annual_rets, 'OOS'),
        'CAGR_FULL_pretax':  pretax_cagr(annual_rets, 'FULL'),
        'CAGR_IS_aftertax':   aftertax_cagr(annual_rets, 'IS'),
        'CAGR_OOS_aftertax':  aftertax_cagr(annual_rets, 'OOS'),
        'CAGR_FULL_aftertax': aftertax_cagr(annual_rets, 'FULL'),
    }


def main() -> pd.DataFrame:
    src_csv = ROOT / 'data' / 'signals' / 'expansion' / 'decision_annual_returns_20260607.csv'
    df = pd.read_csv(src_csv)
    df['Year'] = pd.to_datetime(df['Year'], format='%Y') + pd.offsets.YearEnd(0)
    df = df.set_index('Year').sort_index()

    print(f'=== Loaded annual returns: {src_csv.name} ===')
    print(f'  Years: {df.index.year.min()} - {df.index.year.max()} (n={len(df)})')
    print(f'  Strategies (columns): {list(df.columns)}')
    print(f'  IS years: <={IS_END_YEAR}   OOS years: >={OOS_START_YEAR}')

    # --- Build V7 NAV ---
    print('\n=== Building V7 (pure_boost) NAV via build_candidate_nav ===')
    macro = pd.read_csv(
        ROOT / 'data' / 'macro_features.csv',
        parse_dates=['date'], index_col='date',
    ).sort_index()
    signal_raw = macro['nasdaq_mom63'].dropna()
    print(f'  nasdaq_mom63: {len(signal_raw)} obs  [{signal_raw.index.min().date()} -> {signal_raw.index.max().date()}]')

    v7_nav = build_candidate_nav(
        'S3', signal_raw, 'M6', 'defensive',
        mapping={0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00},
    ).dropna()
    v7_annual = annual_returns_from_nav(v7_nav)
    print(f'  V7 annual returns: {len(v7_annual)} years  [{v7_annual.index.year.min()} - {v7_annual.index.year.max()}]')

    # --- Compute metric set for each existing column + V7 ---
    rows: list[dict] = []
    label_map = {
        'S1_F10':                       'S1_F10_CFD',
        'S2_D5':                        'S2_D5_CFD',
        'S3_DH-W1':                     'S3_DH-W1_ETF',
        'E4_Active':                    'E4_Active_CFD',
        'S3+nasdaq_mom63_M6_def':       'S3+overlay_V0_defensive_ETF',
    }
    for col in df.columns:
        rets = df[col].dropna()
        rows.append(metric_set(rets, label_map.get(col, col)))
    rows.append(metric_set(v7_annual, 'S3+overlay_V7_pure_boost_ETF'))

    out = pd.DataFrame(rows)
    out_csv = ROOT / 'data' / 'signals' / 'expansion' / 'aftertax_cagr_20260607.csv'
    out.to_csv(out_csv, index=False, float_format='%.6f')

    # --- Pretty print ---
    print('\n=== After-tax CAGR table (TAX_MULT=0.8273 per-year compound) ===')
    disp = out.copy()
    for c in disp.columns:
        if c.startswith('CAGR'):
            disp[c] = (disp[c] * 100).round(2)
    print(disp.to_string(index=False))

    print(f'\nWrote {out_csv}')
    return out


if __name__ == '__main__':
    main()
