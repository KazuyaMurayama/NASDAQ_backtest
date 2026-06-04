"""Focused Tier 2 + Tier 3 integration runs with full 9+1 metric evaluation.

Session S3 (2026-06-05) per SIGNAL_INTEGRATION_PLAN_20260604.md focused
expansion. Instead of full enumeration (~66 + 36 = 102 patterns), we
restrict to the most promising axes from Tier 1 / IC analysis:

  Focused Tier 2:
    3 signals  : #21 HY OAS, #23 BAA-10Y, #41 DXY
    3 strategies : S1, S2, S3
    3 methods  : M2_procyclical, M4_vol_adj, M5_filter_entry
    1 direction each
    Total      : 27 patterns

  Focused Tier 3:
    3 pairs    : (21,23), (21,41), (23,41)
    2 operators: AND, OR
    3 strategies: S1, S2, S3
    1 method   : M1_procyclical (binary-compatible)
    Total      : 18 patterns

Outputs:
  data/signals/integration/tier2_focused_results_20260605.csv
  data/signals/integration/tier3_focused_results_20260605.csv
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd  # noqa: E402

from integration.injection_methods import METHOD_REGISTRY  # noqa: E402
from integration.nine_metric_eval import (  # noqa: E402
    evaluate,
    judge_improvement,
    judge_improvement_relaxed,
    judge_improvement_full,
)
from signals.combinations import combine_signals  # noqa: E402

from run_integration_tier import load_signals, prepare_signal  # noqa: E402


FOCUSED_T2_SIGNALS = [21, 23, 41]
FOCUSED_T2_STRATEGIES = ['S1', 'S2', 'S3']
FOCUSED_T2_METHODS = [
    ('M2', 'procyclical'),
    ('M4', 'vol_adj'),
    ('M5', 'filter_entry'),
]

FOCUSED_T3_SIGNALS = [21, 23, 41]
FOCUSED_T3_PAIRS = list(combinations(FOCUSED_T3_SIGNALS, 2))  # (21,23),(21,41),(23,41)
FOCUSED_T3_OPS = ['AND', 'OR']
FOCUSED_T3_STRATEGIES = ['S1', 'S2', 'S3']
# Use M1 procyclical (binary-compatible: 0/1 signal mapped to 0.5/1.0 lev_mod)
FOCUSED_T3_METHOD = ('M1', 'procyclical')


def _load_resources(date_suffix: str):
    print(f'[focused] Loading baselines baseline_navs_{date_suffix}.parquet ...')
    baselines = pd.read_parquet(
        ROOT / 'data' / 'signals' / 'integration' / f'baseline_navs_{date_suffix}.parquet'
    )
    print(f'  baselines: {list(baselines.columns)}, {len(baselines):,} obs')

    print('[focused] Loading + quantizing 6 signals ...')
    raw_signals = load_signals()
    signals = {sid: prepare_signal(v) for sid, v in raw_signals.items()}
    for sid, (name, s) in signals.items():
        print(f'    #{sid:3d} {name:25s} nonnull={int(s.notna().sum()):>6,}')
    return baselines, signals


def run_focused_tier2(baselines, signals, date_suffix: str) -> pd.DataFrame:
    print('\n=== Focused Tier 2 ===')
    rows = []
    total = len(FOCUSED_T2_SIGNALS) * len(FOCUSED_T2_STRATEGIES) * len(FOCUSED_T2_METHODS)
    i = 0
    for sid in FOCUSED_T2_SIGNALS:
        for strat in FOCUSED_T2_STRATEGIES:
            for method, direction in FOCUSED_T2_METHODS:
                i += 1
                pid = f"T2F_{method}_{sid}_{strat}_{direction}"
                row = {
                    'pattern_id': pid,
                    'tier': 2,
                    'signal_id': sid,
                    'strategy': strat,
                    'method': method,
                    'direction': direction,
                }
                signal_name, signal_series = signals[sid]
                row['signal_name'] = signal_name
                method_fn = METHOD_REGISTRY.get((method, direction))
                if method_fn is None or strat not in baselines.columns:
                    row['judgment_full'] = 'SKIP'
                    rows.append(row)
                    continue
                try:
                    base_nav = baselines[strat].dropna()
                    cand_nav = method_fn(base_nav, signal_series)
                    m = evaluate(cand_nav, base_nav, split_date='2018-01-01')
                    row.update(m)
                    row.update(judge_improvement(m))
                    row.update(judge_improvement_relaxed(m))
                    row.update(judge_improvement_full(m))
                except Exception as exc:  # noqa: BLE001
                    row['judgment_full'] = f'ERROR: {type(exc).__name__}: {exc}'
                rows.append(row)
                print(f'  [{i:>2}/{total}] {pid:48s} → {row.get("judgment_full","?")}')

    df = pd.DataFrame(rows)
    out = ROOT / 'data' / 'signals' / 'integration' / f'tier2_focused_results_{date_suffix}.csv'
    df.to_csv(out, index=False)
    print(f'[focused-T2] Wrote {out}')
    print('\n[focused-T2] judgment_full counts:')
    print(df['judgment_full'].value_counts().to_string())
    return df


def run_focused_tier3(baselines, signals, date_suffix: str) -> pd.DataFrame:
    print('\n=== Focused Tier 3 ===')
    rows = []
    total = len(FOCUSED_T3_PAIRS) * len(FOCUSED_T3_OPS) * len(FOCUSED_T3_STRATEGIES)
    i = 0
    method, direction = FOCUSED_T3_METHOD
    method_fn = METHOD_REGISTRY.get((method, direction))
    for s1, s2 in FOCUSED_T3_PAIRS:
        for op in FOCUSED_T3_OPS:
            # Build combined binary signal once per (pair, op)
            name1, sig1 = signals[s1]
            name2, sig2 = signals[s2]
            # combine_signals expects quantized; threshold=2 → "high stress" (top half of 4-level)
            combined = combine_signals(sig1, sig2, operator=op, threshold1=2, threshold2=2)
            for strat in FOCUSED_T3_STRATEGIES:
                i += 1
                pid = f"T3F_{method}_{s1}_{op}_{s2}_{strat}_{direction}"
                row = {
                    'pattern_id': pid,
                    'tier': 3,
                    'signal_id': f'{s1}_{op}_{s2}',
                    'signal_name': f'{name1} {op} {name2}',
                    'strategy': strat,
                    'method': method,
                    'direction': direction,
                }
                if strat not in baselines.columns or method_fn is None:
                    row['judgment_full'] = 'SKIP'
                    rows.append(row)
                    continue
                try:
                    base_nav = baselines[strat].dropna()
                    # combined is 0/1 — M1 procyclical handles binary natively
                    cand_nav = method_fn(base_nav, combined)
                    m = evaluate(cand_nav, base_nav, split_date='2018-01-01')
                    row.update(m)
                    row.update(judge_improvement(m))
                    row.update(judge_improvement_relaxed(m))
                    row.update(judge_improvement_full(m))
                except Exception as exc:  # noqa: BLE001
                    row['judgment_full'] = f'ERROR: {type(exc).__name__}: {exc}'
                rows.append(row)
                print(f'  [{i:>2}/{total}] {pid:60s} → {row.get("judgment_full","?")}')

    df = pd.DataFrame(rows)
    out = ROOT / 'data' / 'signals' / 'integration' / f'tier3_focused_results_{date_suffix}.csv'
    df.to_csv(out, index=False)
    print(f'[focused-T3] Wrote {out}')
    print('\n[focused-T3] judgment_full counts:')
    print(df['judgment_full'].value_counts().to_string())
    return df


def main(date_suffix: str = '20260605') -> int:
    baselines, signals = _load_resources(date_suffix)
    run_focused_tier2(baselines, signals, date_suffix)
    run_focused_tier3(baselines, signals, date_suffix)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
