"""Tune S3 (DH-W1) + nasdaq_mom63 overlay mapping variants (2026-06-07).

Goal: find mapping(s) for which min(CAGR_IS, CAGR_OOS) > 18% while
preserving MaxDD/Worst10Y improvements vs the S3 DH-W1 baseline and the
current ADOPT V0 mapping {q0=1.1, q1=1.0, q2=0.9, q3=0.8}.

Tests 10 variants (V0-V9) using the existing native injector
(src/integration/build_strategy_with_signal.build_candidate_nav) extended
with a `mapping=` override parameter.

Outputs:
  data/signals/expansion/s3_overlay_tuning_20260607.csv
  - 9+1 standard metrics per variant (canonical split 2021-05-08)
  - includes S3 DH-W1 baseline row for reference
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd

# Reuse the 10-metric eval primitives already validated for the §1 decision
from recompute_9metrics_for_decision import (  # noqa: E402
    all_metrics,
    load_baseline_navs,
)
from integration.build_strategy_with_signal import build_candidate_nav  # noqa: E402


# ---------------------------------------------------------------------------
# Variant catalogue (10 total)
# ---------------------------------------------------------------------------
VARIANTS: dict[str, tuple[str, str, dict[int, float]]] = {
    'V0_M6_def_baseline':     ('M6', 'defensive',  {0: 1.10, 1: 1.00, 2: 0.90, 3: 0.80}),
    'V1_M6_def_mild':         ('M6', 'defensive',  {0: 1.05, 1: 1.00, 2: 0.95, 3: 0.85}),
    'V2_M6_def_extreme_only': ('M6', 'defensive',  {0: 1.00, 1: 1.00, 2: 0.95, 3: 0.75}),
    'V3_M6_def_boost_low':    ('M6', 'defensive',  {0: 1.20, 1: 1.05, 2: 0.95, 3: 0.85}),
    'V4_M6_def_strong_asym':  ('M6', 'defensive',  {0: 1.25, 1: 1.10, 2: 0.95, 3: 0.85}),
    'V5_M6_def_narrow_band':  ('M6', 'defensive',  {0: 1.05, 1: 1.00, 2: 1.00, 3: 0.85}),
    'V6_M6_def_boost_heavy':  ('M6', 'defensive',  {0: 1.15, 1: 1.10, 2: 1.00, 3: 0.95}),
    'V7_M6_def_pure_boost':   ('M6', 'defensive',  {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00}),
    'V8_M2_def_sharp':        ('M2', 'defensive',  {0: 1.20, 1: 1.00, 2: 0.70, 3: 0.30}),
    'V9_M2_proc':             ('M2', 'procyclical',{0: 0.70, 1: 0.90, 2: 1.10, 3: 1.30}),
}


def main() -> pd.DataFrame:
    # -------- Load signal once --------
    print('=== Loading nasdaq_mom63 signal ===')
    macro = pd.read_csv(
        ROOT / 'data' / 'macro_features.csv',
        parse_dates=['date'], index_col='date',
    ).sort_index()
    signal_raw = macro['nasdaq_mom63'].dropna()
    print(f'  signal_raw: {len(signal_raw)} obs  [{signal_raw.index.min().date()} -> {signal_raw.index.max().date()}]')

    # -------- S3 baseline NAV (for trades_per_yr_navproxy reference) --------
    print('=== Loading S3 (DH-W1) baseline NAV ===')
    bn = load_baseline_navs()
    s3_nav = bn['S3'].dropna()
    print(f'  S3 baseline: {len(s3_nav)} obs  [{s3_nav.index.min().date()} -> {s3_nav.index.max().date()}]')

    # -------- Compute baseline row --------
    rows: list[dict] = []
    print('\n=== Baseline metrics (S3 DH-W1, no overlay) ===')
    rows.append(all_metrics(s3_nav, 'S3_DH-W1_baseline'))

    # -------- Build each variant --------
    print('\n=== Building 10 variants ===')
    for label, (method, direction, mapping) in VARIANTS.items():
        t0 = time.time()
        print(f'  [{label}]  method={method}  direction={direction}  mapping={mapping}')
        cand_nav = build_candidate_nav(
            'S3', signal_raw, method, direction, mapping=mapping,
        ).dropna()
        elapsed = time.time() - t0
        print(f'    -> {len(cand_nav)} obs  [{cand_nav.index.min().date()} -> {cand_nav.index.max().date()}]  ({elapsed:.1f}s)')
        rows.append(all_metrics(cand_nav, label, baseline_nav_for_trades=s3_nav))

    df = pd.DataFrame(rows)
    out_csv = ROOT / 'data' / 'signals' / 'expansion' / 's3_overlay_tuning_20260607.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format='%.6f')
    print(f'\nWrote {out_csv}')

    # -------- Pretty print summary --------
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)

    df_disp = df[[
        'label', 'CAGR_IS', 'CAGR_OOS', 'CAGR_FULL', 'Sharpe_OOS',
        'MaxDD_FULL', 'Worst10Y_calendar', 'P10_5Y', 'IS_OOS_gap_pp',
        'WFA_CI95_lo_annual', 'WFA_WFE_calendar',
    ]].copy()
    for col in ['CAGR_IS', 'CAGR_OOS', 'CAGR_FULL', 'MaxDD_FULL',
                'Worst10Y_calendar', 'P10_5Y', 'WFA_CI95_lo_annual']:
        df_disp[col] = (df_disp[col] * 100).round(2)
    for col in ['Sharpe_OOS', 'IS_OOS_gap_pp', 'WFA_WFE_calendar']:
        df_disp[col] = df_disp[col].round(3)
    print('\n=== Full 10-metric summary (canonical split 2021-05-08) ===')
    print(df_disp.to_string(index=False))

    # -------- Filter: min(CAGR_IS, CAGR_OOS) > 18% --------
    df['min_CAGR'] = df[['CAGR_IS', 'CAGR_OOS']].min(axis=1)
    qualified = df[df['min_CAGR'] > 0.18].copy()
    print(f'\n=== Variants with min(CAGR_IS, CAGR_OOS) > 18%:  {len(qualified)} ===')
    if len(qualified):
        q_disp = qualified[[
            'label', 'min_CAGR', 'CAGR_IS', 'CAGR_OOS',
            'MaxDD_FULL', 'Worst10Y_calendar', 'P10_5Y', 'Sharpe_OOS',
        ]].copy()
        for col in ['min_CAGR', 'CAGR_IS', 'CAGR_OOS',
                    'MaxDD_FULL', 'Worst10Y_calendar', 'P10_5Y']:
            q_disp[col] = (q_disp[col] * 100).round(2)
        q_disp['Sharpe_OOS'] = q_disp['Sharpe_OOS'].round(3)
        print(q_disp.to_string(index=False))

    # -------- Further filter: also MaxDD < -32% (improvement preserved) --------
    qualified_dd = qualified[qualified['MaxDD_FULL'] > -0.32].copy()
    print(f'\n=== AND MaxDD_FULL > -32% (improvement preserved):  {len(qualified_dd)} ===')
    if len(qualified_dd):
        print(qualified_dd[['label', 'min_CAGR', 'MaxDD_FULL', 'Worst10Y_calendar']]
              .assign(min_CAGR=lambda x: (x['min_CAGR']*100).round(2),
                      MaxDD_FULL=lambda x: (x['MaxDD_FULL']*100).round(2),
                      Worst10Y_calendar=lambda x: (x['Worst10Y_calendar']*100).round(2))
              .to_string(index=False))

    return df


if __name__ == '__main__':
    main()
