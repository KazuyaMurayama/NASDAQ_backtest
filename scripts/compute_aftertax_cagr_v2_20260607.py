"""V2: After-tax CAGR with proper partial-year (2026 YTD) handling.

V1 (compute_aftertax_cagr_20260607.py) had a bug:
  years = len(annual_returns)
counted 2026 YTD (only Jan-Mar, ~60 business days) as a full year, which
dragged after-tax CAGR down by 3-9pp across all strategies.

V2 fix
------
For each strategy:
  1. Get daily NAV (from baseline_navs_20260605.parquet or rebuild via
     build_candidate_nav for V0/V7 overlays, or load cache for E4).
  2. Resample to calendar year-end NAV (last NAV per year; 2026 -> last
     available NAV, e.g. 2026-03-26).
  3. Compute annual return: r_y = NAV_y / NAV_{y-1} - 1.
  4. Apply tax_mult (=0.8273) to each annual return.
  5. Build after-tax cumulative NAV: NAV_at[y] = NAV_at[y-1] * (1 + r_y*tax_mult).
  6. Compute CAGR with ACTUAL elapsed days / 365.25 (NOT len()):
       CAGR = (NAV_at[end] / NAV_at[start]) ** (1/elapsed_years) - 1
     for IS (first YE -> 2020-12-31), OOS (2020-12-31 -> last YE),
     FULL (first YE -> last YE).

IS/OOS split
------------
Uses calendar-year split (IS end = 2020-12-31, OOS start = 2021-01-01)
to match the v1 convention. This differs from the canonical daily split
(2021-05-08) by ~4 months; the resulting <0.3pp discrepancy with the
§3.1 canonical daily CAGRs is documented in §3.6.

Convention
----------
TAX_MULT = 0.8273 (1 - 0.1727 effective drag, repo-wide constant).
Applied uniformly to each annual return regardless of sign. NISA accounts
skip the tax adjustment (aftertax == pretax).
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
IS_END = pd.Timestamp('2020-12-31')


def _cagr_from_endpoints(v_start: float, v_end: float, d_start: pd.Timestamp, d_end: pd.Timestamp) -> float:
    if v_start is None or v_end is None or v_start <= 0:
        return float('nan')
    years = (d_end - d_start).days / 365.25
    if years <= 0:
        return float('nan')
    return (v_end / v_start) ** (1.0 / years) - 1.0


def compute_aftertax_metrics(nav: pd.Series, label: str, env: str,
                             tax_mult: float = TAX_MULT) -> dict:
    """Compute IS/OOS/FULL pretax & aftertax CAGR for a daily NAV series.

    `env` controls tax treatment: ETF_NISA -> aftertax == pretax (no tax);
    CFD_taxed / ETF_taxed -> apply tax_mult per calendar year.
    """
    nav = nav.dropna().sort_index()

    # ---- Pretax CAGRs from daily NAV (canonical NAV formula) ----
    is_nav = nav.loc[:IS_END]
    oos_nav = nav.loc[IS_END:]  # include 2020-12-31 as OOS anchor

    pretax_is = _cagr_from_endpoints(
        is_nav.iloc[0], is_nav.iloc[-1],
        is_nav.index[0], is_nav.index[-1],
    ) if len(is_nav) >= 2 else float('nan')

    pretax_oos = _cagr_from_endpoints(
        oos_nav.iloc[0], oos_nav.iloc[-1],
        oos_nav.index[0], oos_nav.index[-1],
    ) if len(oos_nav) >= 2 else float('nan')

    pretax_full = _cagr_from_endpoints(
        nav.iloc[0], nav.iloc[-1],
        nav.index[0], nav.index[-1],
    )

    nisa = (env == 'ETF_NISA')

    if nisa:
        aftertax_is = pretax_is
        aftertax_oos = pretax_oos
        aftertax_full = pretax_full
    else:
        # ---- Build after-tax NAV via calendar-year tax application ----
        # Use group-by-year and capture BOTH the value AND the actual last
        # trading date in each year (critical for 2026 YTD: the resample
        # label is 2026-12-31 but the actual last NAV is 2026-03-26).
        years = nav.index.year
        ye_value: list[float] = []
        ye_actual_date: list[pd.Timestamp] = []
        for y in sorted(set(years)):
            seg = nav[years == y]
            ye_value.append(float(seg.iloc[-1]))
            ye_actual_date.append(seg.index[-1])
        ye = pd.Series(ye_value, index=pd.DatetimeIndex(ye_actual_date))

        ye_returns = ye.pct_change().dropna()  # year-over-year returns
        after_returns = ye_returns * tax_mult

        # Anchor: first year-end NAV (pretax). Build cumulative aftertax NAV
        # indexed by ACTUAL last-trading-day of each calendar year.
        at_nav = pd.Series(index=ye.index, dtype=float)
        at_nav.iloc[0] = ye.iloc[0]
        for i in range(1, len(ye)):
            at_nav.iloc[i] = at_nav.iloc[i - 1] * (1.0 + after_returns.iloc[i - 1])

        # Locate IS end label (last actual year-end date on/before 2020-12-31)
        is_mask = at_nav.index <= IS_END
        is_end_label = at_nav.index[is_mask][-1] if is_mask.any() else None
        first_label = at_nav.index[0]
        last_label = at_nav.index[-1]  # e.g. 2026-03-26 for YTD year

        if is_end_label is None:
            aftertax_is = float('nan')
            aftertax_oos = float('nan')
        else:
            aftertax_is = _cagr_from_endpoints(
                at_nav.loc[first_label], at_nav.loc[is_end_label],
                first_label, is_end_label,
            )
            aftertax_oos = _cagr_from_endpoints(
                at_nav.loc[is_end_label], at_nav.loc[last_label],
                is_end_label, last_label,
            )
        aftertax_full = _cagr_from_endpoints(
            at_nav.loc[first_label], at_nav.loc[last_label],
            first_label, last_label,
        )

    return {
        'label': label,
        'env': env,
        'date_start': nav.index[0].date().isoformat(),
        'date_end': nav.index[-1].date().isoformat(),
        'years_IS_actual': round((min(IS_END, nav.index[-1]) - nav.index[0]).days / 365.25, 3),
        'years_OOS_actual': round(max(0.0, (nav.index[-1] - max(IS_END, nav.index[0])).days / 365.25), 3),
        'years_FULL_actual': round((nav.index[-1] - nav.index[0]).days / 365.25, 3),
        'CAGR_IS_pretax': pretax_is,
        'CAGR_OOS_pretax': pretax_oos,
        'CAGR_FULL_pretax': pretax_full,
        'CAGR_IS_aftertax': aftertax_is,
        'CAGR_OOS_aftertax': aftertax_oos,
        'CAGR_FULL_aftertax': aftertax_full,
    }


def _load_e4_nav() -> pd.Series:
    cache = pickle.load(open(ROOT / 'audit_results' / '_cache' / 'e4_nav_cache.pkl', 'rb'))
    dates = pd.to_datetime(cache['dates'])
    nav = pd.Series(np.asarray(cache['nav_e4']), index=dates).dropna()
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
    print(f'  V0 (def {{1.1,1.0,0.9,0.8}}) NAV: {len(nav_v0)} obs  [{nav_v0.index[0].date()} -> {nav_v0.index[-1].date()}]')

    nav_v7 = build_candidate_nav(
        'S3', signal_raw, 'M6', 'defensive',
        mapping={0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00},
    ).dropna()
    print(f'  V7 pure_boost NAV: {len(nav_v7)} obs  [{nav_v7.index[0].date()} -> {nav_v7.index[-1].date()}]')

    rows: list[dict] = []
    # CFD-only strategies (tax applies, no NISA case)
    rows.append(compute_aftertax_metrics(s1_nav, 'S1_F10', env='CFD_taxed'))
    rows.append(compute_aftertax_metrics(s2_nav, 'S2_D5', env='CFD_taxed'))
    rows.append(compute_aftertax_metrics(nav_e4, 'E4_Active', env='CFD_taxed'))

    # ETF strategies: both NISA (tax-free) and taxed cases
    for label, nv in [
        ('S3_DH-W1_baseline', s3_nav),
        ('S3+V0_def', nav_v0),
        ('S3+V7_pure_boost', nav_v7),
    ]:
        rows.append(compute_aftertax_metrics(nv, label, env='ETF_NISA'))
        rows.append(compute_aftertax_metrics(nv, label, env='ETF_taxed'))

    out = pd.DataFrame(rows)
    out_csv = ROOT / 'data' / 'signals' / 'expansion' / 'aftertax_cagr_v2_20260607.csv'
    out.to_csv(out_csv, index=False, float_format='%.6f')

    print('\n=== V2 After-tax CAGR (actual elapsed days / 365.25; 2026 YTD handled) ===')
    disp = out.copy()
    for c in disp.columns:
        if c.startswith('CAGR'):
            disp[c] = (disp[c] * 100).round(2)
    print(disp.to_string(index=False))

    # ---- V1 vs V2 diff (for the 6 v1 rows + V7 NISA case) ----
    print('\n=== V1 vs V2 diff (pp, ETF rows use ETF_taxed for like-for-like) ===')
    v1 = pd.read_csv(ROOT / 'data' / 'signals' / 'expansion' / 'aftertax_cagr_20260607.csv')
    v1_map = {
        'S1_F10_CFD': ('S1_F10', 'CFD_taxed'),
        'S2_D5_CFD': ('S2_D5', 'CFD_taxed'),
        'E4_Active_CFD': ('E4_Active', 'CFD_taxed'),
        'S3_DH-W1_ETF': ('S3_DH-W1_baseline', 'ETF_taxed'),
        'S3+overlay_V0_defensive_ETF': ('S3+V0_def', 'ETF_taxed'),
        'S3+overlay_V7_pure_boost_ETF': ('S3+V7_pure_boost', 'ETF_taxed'),
    }
    diff_rows = []
    for v1_label, (v2_label, v2_env) in v1_map.items():
        v1_row = v1[v1['label'] == v1_label]
        v2_row = out[(out['label'] == v2_label) & (out['env'] == v2_env)]
        if len(v1_row) == 0 or len(v2_row) == 0:
            continue
        v1r = v1_row.iloc[0]
        v2r = v2_row.iloc[0]
        diff_rows.append({
            'label': v1_label,
            'IS_v1_pct':  round(v1r['CAGR_IS_aftertax'] * 100, 2),
            'IS_v2_pct':  round(v2r['CAGR_IS_aftertax'] * 100, 2),
            'IS_delta_pp': round((v2r['CAGR_IS_aftertax'] - v1r['CAGR_IS_aftertax']) * 100, 2),
            'OOS_v1_pct':  round(v1r['CAGR_OOS_aftertax'] * 100, 2),
            'OOS_v2_pct':  round(v2r['CAGR_OOS_aftertax'] * 100, 2),
            'OOS_delta_pp': round((v2r['CAGR_OOS_aftertax'] - v1r['CAGR_OOS_aftertax']) * 100, 2),
            'FULL_v1_pct':  round(v1r['CAGR_FULL_aftertax'] * 100, 2),
            'FULL_v2_pct':  round(v2r['CAGR_FULL_aftertax'] * 100, 2),
            'FULL_delta_pp': round((v2r['CAGR_FULL_aftertax'] - v1r['CAGR_FULL_aftertax']) * 100, 2),
        })
    diff_df = pd.DataFrame(diff_rows)
    print(diff_df.to_string(index=False))
    diff_csv = ROOT / 'data' / 'signals' / 'expansion' / 'aftertax_cagr_v1_v2_diff_20260607.csv'
    diff_df.to_csv(diff_csv, index=False)

    print(f'\nWrote {out_csv}')
    print(f'Wrote {diff_csv}')


if __name__ == '__main__':
    main()
