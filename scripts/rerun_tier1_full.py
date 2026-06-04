"""Re-run Tier 1 (72 patterns) under the full 9+1 metric framework.

Session S3 (2026-06-05): re-evaluates the same Tier 1 enumeration as
scripts/run_integration_tier.py --tier 1, but now uses the enhanced
nine_metric_eval.evaluate() (with Trades/yr, WFE, CI95_lo) and the new
judge_improvement_full() function.

Output: data/signals/integration/tier1_results_full_20260605.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd  # noqa: E402

from integration.pattern_enumerator import enumerate_tier1  # noqa: E402
from integration.injection_methods import METHOD_REGISTRY  # noqa: E402
from integration.nine_metric_eval import (  # noqa: E402
    evaluate,
    judge_improvement,
    judge_improvement_relaxed,
    judge_improvement_full,
)

# Reuse signal loader from the existing tier runner
sys.path.insert(0, str(ROOT / 'scripts'))
from run_integration_tier import load_signals, prepare_signal  # noqa: E402


def main(date_suffix: str = '20260605') -> int:
    print(f'[tier1-full] Loading baselines baseline_navs_{date_suffix}.parquet ...')
    baselines = pd.read_parquet(
        ROOT / 'data' / 'signals' / 'integration' / f'baseline_navs_{date_suffix}.parquet'
    )
    print(f'  baselines: {list(baselines.columns)}, {len(baselines):,} obs')

    print('[tier1-full] Loading + quantizing 6 signals ...')
    raw_signals = load_signals()
    signals = {sid: prepare_signal(v) for sid, v in raw_signals.items()}
    for sid, (name, s) in signals.items():
        print(f'    #{sid:3d} {name:25s} nonnull={int(s.notna().sum()):>6,}')

    patterns = enumerate_tier1()
    print(f'[tier1-full] Enumerated {len(patterns)} Tier-1 patterns')

    rows = []
    for i, p in enumerate(patterns):
        if (i + 1) % 12 == 0 or (i + 1) == len(patterns):
            print(f'  [{i+1:>3}/{len(patterns)}] processing ...')

        row: dict = {**p.__dict__}

        if p.strategy not in baselines.columns:
            row['judgment_full'] = 'SKIP_NO_BASELINE'
            rows.append(row)
            continue
        if not isinstance(p.signal_id, int) or p.signal_id not in signals:
            row['judgment_full'] = 'SKIP_NO_SIGNAL'
            rows.append(row)
            continue

        signal_name, signal_series = signals[p.signal_id]
        row['signal_name'] = signal_name

        method_fn = METHOD_REGISTRY.get((p.method, p.direction))
        if method_fn is None:
            row['judgment_full'] = 'SKIP_NO_METHOD'
            rows.append(row)
            continue

        try:
            base_nav = baselines[p.strategy].dropna()
            cand_nav = method_fn(base_nav, signal_series)
            m = evaluate(cand_nav, base_nav, split_date='2018-01-01')
            row.update(m)
            row.update(judge_improvement(m))
            row.update(judge_improvement_relaxed(m))
            row.update(judge_improvement_full(m))
        except Exception as exc:  # noqa: BLE001
            row['judgment_full'] = f'ERROR: {type(exc).__name__}: {exc}'

        rows.append(row)

    df = pd.DataFrame(rows)
    out = ROOT / 'data' / 'signals' / 'integration' / f'tier1_results_full_{date_suffix}.csv'
    df.to_csv(out, index=False)
    print(f'\n[tier1-full] Wrote {out}')

    # Summary
    if 'judgment_full' in df.columns:
        print('\n[tier1-full] judgment_full counts:')
        print(df['judgment_full'].value_counts().to_string())
        print('\n[tier1-full] Per-strategy summary (judgment_full):')
        for strat in ['S1', 'S2', 'S3']:
            sub = df[df['strategy'] == strat]
            sp = int((sub['judgment_full'] == 'STRONG_PASS_FULL').sum())
            st = int((sub['judgment_full'] == 'STANDARD_PASS_FULL').sum())
            mg = int((sub['judgment_full'] == 'MARGINAL_FULL').sum())
            fl = int((sub['judgment_full'] == 'FAIL_FULL').sum())
            print(f'  {strat}: STRONG={sp} STANDARD={st} MARGINAL={mg} FAIL={fl}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
