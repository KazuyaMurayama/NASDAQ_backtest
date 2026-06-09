"""V3 (final): After-tax CAGR using canonical split (IS_END=2021-05-07).

Supersedes V1 (calendar-len bug) and V2 (calendar split discrepancy with other strategies).
This V3 uses the SAME canonical split as vz=0.65+l5+F10ε, vz=0.65+l7+F10ε, DH-W1, E4 entries
in STRATEGY_REGISTRY.md / CURRENT_BEST_STRATEGY.md.

Tax convention: ×0.8273 applied to each periodized return (between canonical boundaries),
then compounded with actual elapsed time.

Periodization for after-tax NAV
-------------------------------
Each yearly period boundary is one tax-realization event. Boundaries are:
  - Series start
  - End of each full calendar year (last trading day on/before YYYY-12-31)
  - End of IS (last trading day on/before 2021-05-07)  <-- canonical split
  - Series end (last trading day, e.g. 2026-03-26 for YTD)
For each period (prev_d -> curr_d):
  period_return = NAV(curr_d)/NAV(prev_d) - 1
  after_return  = period_return * TAX_MULT          (0.8273)
  AT_NAV(curr_d) = AT_NAV(prev_d) * (1 + after_return)
CAGR uses actual elapsed days/365.25 (no rounding to full years).

Single output table per strategy x environment (NISA / CFD-taxed). No calendar/canonical split duality.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import pickle

from integration.build_strategy_with_signal import build_candidate_nav  # noqa: E402


TAX_MULT = 0.8273
IS_END = pd.Timestamp('2021-05-07')  # canonical IS end (OOS starts 2021-05-08)


def build_aftertax_nav(daily_nav: pd.Series, tax_mult: float = TAX_MULT,
                        is_end_date: pd.Timestamp = IS_END) -> pd.Series:
    """Build after-tax NAV at canonical-aware yearly boundaries."""
    nav = daily_nav.dropna().sort_index()
    boundaries: set = set()
    boundaries.add(nav.index[0])

    start_year = nav.index[0].year
    end_year = nav.index[-1].year
    for y in range(start_year, end_year + 1):
        ye = pd.Timestamp(f'{y}-12-31')
        # Insert IS boundary inside its year (before the year-end anchor).
        if is_end_date.year == y:
            cand = nav.index[nav.index <= is_end_date]
            if len(cand) > 0:
                boundaries.add(cand[-1])
        ye_cand = nav.index[nav.index <= ye]
        if len(ye_cand) > 0:
            boundaries.add(ye_cand[-1])
    boundaries.add(nav.index[-1])

    boundaries_sorted = sorted(boundaries)

    at_nav = pd.Series(index=pd.DatetimeIndex(boundaries_sorted), dtype=float)
    at_nav.iloc[0] = float(nav.loc[boundaries_sorted[0]])
    for i in range(1, len(boundaries_sorted)):
        prev_d = boundaries_sorted[i - 1]
        curr_d = boundaries_sorted[i]
        period_return = nav.loc[curr_d] / nav.loc[prev_d] - 1.0
        at_nav.iloc[i] = at_nav.iloc[i - 1] * (1.0 + period_return * tax_mult)
    return at_nav


def _cagr(start_d, end_d, v_start, v_end) -> float:
    if v_start is None or v_start <= 0:
        return float('nan')
    years = (end_d - start_d).days / 365.25
    if years <= 0:
        return float('nan')
    return (v_end / v_start) ** (1.0 / years) - 1.0


def cagr_canonical(daily_nav: pd.Series, is_end_date: pd.Timestamp = IS_END) -> dict:
    """Pretax CAGR (canonical split, anchor = trading day on/before IS_END)."""
    nav = daily_nav.dropna().sort_index()
    is_end_actual = nav.index[nav.index <= is_end_date][-1]
    start = nav.index[0]
    last = nav.index[-1]
    return {
        'CAGR_IS_pretax':   _cagr(start,           is_end_actual, nav.loc[start],         nav.loc[is_end_actual]),
        'CAGR_OOS_pretax':  _cagr(is_end_actual,   last,          nav.loc[is_end_actual], nav.loc[last]),
        'CAGR_FULL_pretax': _cagr(start,           last,          nav.loc[start],         nav.loc[last]),
        'date_start':       start.date().isoformat(),
        'date_is_end':      is_end_actual.date().isoformat(),
        'date_end':         last.date().isoformat(),
        'years_IS':         round((is_end_actual - start).days / 365.25, 3),
        'years_OOS':        round((last - is_end_actual).days / 365.25, 3),
        'years_FULL':       round((last - start).days / 365.25, 3),
    }


def aftertax_cagr_canonical(daily_nav: pd.Series, tax_mult: float = TAX_MULT,
                             is_end_date: pd.Timestamp = IS_END) -> dict:
    at = build_aftertax_nav(daily_nav, tax_mult, is_end_date)
    is_end_actual = at.index[at.index <= is_end_date][-1]
    start = at.index[0]
    last = at.index[-1]
    return {
        'CAGR_IS_aftertax':   _cagr(start,         is_end_actual, at.loc[start],         at.loc[is_end_actual]),
        'CAGR_OOS_aftertax':  _cagr(is_end_actual, last,          at.loc[is_end_actual], at.loc[last]),
        'CAGR_FULL_aftertax': _cagr(start,         last,          at.loc[start],         at.loc[last]),
    }


def _load_e4_nav() -> pd.Series:
    cache = pickle.load(open(ROOT / 'audit_results' / '_cache' / 'e4_nav_cache.pkl', 'rb'))
    dates = pd.to_datetime(cache['dates'])
    nav = pd.Series(np.asarray(cache['nav_e4']), index=pd.DatetimeIndex(dates)).dropna()
    return nav


def main():
    print('=== Loading baselines + caches ===')
    bp = pd.read_parquet(ROOT / 'data' / 'signals' / 'integration' / 'baseline_navs_20260605.parquet')
    s1_nav = bp['S1'].dropna()
    s2_nav = bp['S2'].dropna()
    s3_nav = bp['S3'].dropna()
    print(f'  S1/S2/S3 NAV: {len(s1_nav)} obs  [{s1_nav.index[0].date()} -> {s1_nav.index[-1].date()}]')

    nav_e4 = _load_e4_nav()
    print(f'  E4 NAV: {len(nav_e4)} obs  [{nav_e4.index[0].date()} -> {nav_e4.index[-1].date()}]')

    print('\n=== Rebuilding V0 / V7 overlay NAVs ===')
    macro = pd.read_csv(
        ROOT / 'data' / 'macro_features.csv',
        parse_dates=['date'], index_col='date',
    ).sort_index()
    signal_raw = macro['nasdaq_mom63'].dropna()
    print(f'  nasdaq_mom63: {len(signal_raw)} obs  [{signal_raw.index.min().date()} -> {signal_raw.index.max().date()}]')

    nav_v0 = build_candidate_nav(
        'S3', signal_raw, 'M6', 'defensive',
        mapping={0: 1.10, 1: 1.00, 2: 0.90, 3: 0.80},
    ).dropna()
    print(f'  V0 (def {{1.1,1.0,0.9,0.8}}) NAV: {len(nav_v0)} obs')

    nav_v7 = build_candidate_nav(
        'S3', signal_raw, 'M6', 'defensive',
        mapping={0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00},
    ).dropna()
    print(f'  V7 pure_boost NAV: {len(nav_v7)} obs')

    rows: list[dict] = []

    def add(label: str, nav: pd.Series, env: str):
        pretax = cagr_canonical(nav)
        # CFD: taxed (apply tax)
        # ETF env produces TWO rows: NISA (= pretax) and ETF_taxed (= aftertax)
        if env == 'CFD':
            cases = [('CFD_taxed', True)]
        elif env == 'ETF':
            cases = [('ETF_NISA', False), ('ETF_taxed', True)]
        else:
            raise ValueError(env)
        for case, apply_tax in cases:
            row = {'label': label, 'env': env, 'case': case}
            row.update(pretax)
            if apply_tax:
                row.update(aftertax_cagr_canonical(nav))
            else:
                row['CAGR_IS_aftertax']   = pretax['CAGR_IS_pretax']
                row['CAGR_OOS_aftertax']  = pretax['CAGR_OOS_pretax']
                row['CAGR_FULL_aftertax'] = pretax['CAGR_FULL_pretax']
            rows.append(row)

    # CFD-only strategies (taxed; no NISA case)
    add('S1_F10',     s1_nav, 'CFD')
    add('S2_D5',      s2_nav, 'CFD')
    add('E4_Active',  nav_e4, 'CFD')
    # ETF strategies: NISA + taxed
    add('S3_DH-W1_baseline', s3_nav, 'ETF')
    add('S3+V0_def',         nav_v0, 'ETF')
    add('S3+V7_pure_boost',  nav_v7, 'ETF')

    out = pd.DataFrame(rows)
    out_csv = ROOT / 'data' / 'signals' / 'expansion' / 'aftertax_cagr_canonical_20260607.csv'
    out.to_csv(out_csv, index=False, float_format='%.6f')

    print(f'\n=== V3 After-tax CAGR (canonical split, IS_END={IS_END.date()}) ===')
    disp = out.copy()
    for c in disp.columns:
        if c.startswith('CAGR'):
            disp[c] = (disp[c] * 100).round(2)
    cols_show = ['label', 'env', 'case',
                 'CAGR_IS_pretax', 'CAGR_OOS_pretax', 'CAGR_FULL_pretax',
                 'CAGR_IS_aftertax', 'CAGR_OOS_aftertax', 'CAGR_FULL_aftertax',
                 'date_start', 'date_is_end', 'date_end']
    print(disp[cols_show].to_string(index=False))
    print(f'\nWrote {out_csv}')

    # ---- V7 NISA min CAGR vs 18% target ----
    v7n = out[(out['label'] == 'S3+V7_pure_boost') & (out['case'] == 'ETF_NISA')].iloc[0]
    v7_min = min(v7n['CAGR_IS_aftertax'], v7n['CAGR_OOS_aftertax'])
    print(f"\nV7 NISA min(IS, OOS) aftertax CAGR = {v7_min*100:.2f}%  vs target 18.00%  "
          f"=> {'PASS' if v7_min >= 0.18 else 'MISS'} (gap {((v7_min-0.18)*100):+.2f} pp)")


if __name__ == '__main__':
    main()
