"""G3 Native Integration Tier (Session 3, 2026-06-05).

Top 5 signals × 3 strategies × 5 methods × 2 directions = 150 patterns.

Signals (Top 5 from G2 IC screening):
  - macro_nasdaq_mom21       (universal winner, t-stat +17)
  - macro_nfci_z52w          (Chicago Fed NFCI, negative IC)
  - macro_nas_gold_rel63     (NDX/Gold 63d relative)
  - macro_nasdaq_mom63       (NDX 63d momentum)
  - macro_vix_mom21          (VIX 21d momentum)

Strategies:
  - S1 = F10 (CFD, lev_mod_e4 injection point)
  - S2 = vz065lmax5 (CFD, lev_mod_065 injection point)
  - S3 = DH-W1 (DH, lev_raw injection point, hysteresis mask)

Methods × directions (10 combos):
  M1 binary       × defensive / procyclical
  M2 continuous   × defensive / procyclical
  M4 vol-target   × vol_adj   / reverse
  M5 entry/exit   × stop_only / filter_entry
  M6 threshold    × defensive / procyclical   (proxy via scaled-M2)
"""
from __future__ import annotations
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd  # noqa: E402

from integration.build_strategy_with_signal import build_candidate_nav  # noqa: E402
from integration.nine_metric_eval import evaluate, judge_improvement_full  # noqa: E402


DATA = ROOT / 'data'
EXP_DIR = DATA / 'signals' / 'expansion'
INT_DIR = DATA / 'signals' / 'integration'


def load_top5_signals() -> dict[str, pd.Series]:
    df = pd.read_csv(DATA / 'macro_features.csv', parse_dates=['date'])
    df = df.set_index('date').sort_index()
    cols = ['nasdaq_mom21', 'nfci_z52w', 'nas_gold_rel63',
            'nasdaq_mom63', 'vix_mom21']
    out = {}
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            print(f'  WARN signal {c} is empty')
            continue
        out[c] = s
        print(f'  {c}: {len(s):,} obs, {s.index.min().date()}..{s.index.max().date()}')
    return out


def load_baselines() -> dict[str, pd.Series]:
    df = pd.read_parquet(INT_DIR / 'baseline_navs_20260605.parquet')
    out = {}
    for s in ('S1', 'S2', 'S3'):
        if s in df.columns:
            out[s] = df[s].dropna()
            print(f'  {s}: {len(out[s]):,} obs, final={float(out[s].iloc[-1]):,.2f}')
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('G3 Native Integration — Top 5 signals × 3 strategies × 10 method-dirs')
    print('=' * 80)

    print('\n[Load] Signals...')
    signals = load_top5_signals()
    print('\n[Load] Baselines...')
    baselines = load_baselines()

    methods_dirs = [
        ('M1', 'defensive'),  ('M1', 'procyclical'),
        ('M2', 'defensive'),  ('M2', 'procyclical'),
        ('M4', 'vol_adj'),    ('M4', 'reverse'),
        ('M5', 'stop_only'),  ('M5', 'filter_entry'),
        ('M6', 'defensive'),  ('M6', 'procyclical'),
    ]
    n_total = len(signals) * len(baselines) * len(methods_dirs)
    print(f'\n[G3] Total patterns to run: {n_total}')

    rows = []
    i = 0
    t0 = time.time()
    for sname, sig_raw in signals.items():
        for strat, base_nav in baselines.items():
            for method, direction in methods_dirs:
                i += 1
                pattern_id = f'G3_{sname}_{strat}_{method}_{direction}'
                row = {
                    'pattern_id': pattern_id,
                    'signal': sname,
                    'strategy': strat,
                    'method': method,
                    'direction': direction,
                }
                try:
                    cand_nav = build_candidate_nav(strat, sig_raw, method, direction)
                    if cand_nav is None or len(cand_nav) < 252:
                        row['judgment_full'] = 'ERROR: nav too short'
                    else:
                        metrics = evaluate(cand_nav, base_nav, split_date='2018-01-01')
                        judgment = judge_improvement_full(metrics)
                        row.update({k: v for k, v in metrics.items()})
                        row.update(judgment)
                except Exception as e:
                    row['judgment_full'] = f'ERROR: {type(e).__name__}: {str(e)[:140]}'
                rows.append(row)
                if i % 10 == 0 or i == n_total:
                    elapsed = time.time() - t0
                    eta = elapsed / i * (n_total - i) if i else 0
                    print(f'  [{i:3}/{n_total}] {pattern_id} '
                          f'judge={row.get("judgment_full", "?")[:30]} '
                          f'({elapsed:.0f}s elapsed, ETA {eta:.0f}s)')

    df = pd.DataFrame(rows)
    out_csv = EXP_DIR / 'g3_native_top5_results_20260605.csv'
    df.to_csv(out_csv, index=False)
    print(f'\n[G3] Wrote {out_csv}  ({len(df)} rows)')

    # Summary
    if 'judgment_full' in df.columns:
        print('\n[G3] Judgment counts (overall):')
        print(df['judgment_full'].value_counts().to_string())
        print('\n[G3] Judgment counts per strategy:')
        for strat in ('S1', 'S2', 'S3'):
            sub = df[df['strategy'] == strat]
            print(f'  {strat}: ' + ', '.join(
                f'{k}={v}' for k, v in sub['judgment_full'].value_counts().items()
            ))

        pass_df = df[df['judgment_full'].isin(
            ['STRONG_PASS_FULL', 'STANDARD_PASS_FULL']
        )]
        if not pass_df.empty:
            print(f'\n[G3] PASS patterns ({len(pass_df)}):')
            for _, r in pass_df.iterrows():
                print(
                    f"  {r['pattern_id']}: "
                    f"imp={r.get('improved_axes_full', '?')} "
                    f"deg={r.get('degraded_axes_full', '?')}"
                )

    print('\n[G3] DONE.')


if __name__ == '__main__':
    main()
