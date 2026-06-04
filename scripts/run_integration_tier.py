"""Tier-N integration runner.

Loads baseline NAVs (S1/S2/S3), 6 Phase B PASS signals, enumerates the
patterns for the requested tier, and evaluates each pattern via the 9+1
metric framework. Writes a per-pattern results CSV and prints judgment
summaries per strategy.

Session S2 deliverable (Phase 2) per SIGNAL_INTEGRATION_PLAN_20260604.md.

Usage:
    python scripts/run_integration_tier.py --tier 1
    python scripts/run_integration_tier.py --tier 1 --date 20260605
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd  # noqa: E402

from integration.pattern_enumerator import (  # noqa: E402
    enumerate_tier1, enumerate_tier2, enumerate_tier3, enumerate_tier4,
)
from integration.injection_methods import METHOD_REGISTRY  # noqa: E402
from integration.nine_metric_eval import evaluate, judge_improvement  # noqa: E402
from signals.quantize import quantile_cut  # noqa: E402
from signals.timing import apply_publication_lag  # noqa: E402


# ------------------------------------------------------------------
# Signal loading (replicates scripts/run_phase_b.py:load_signal_series)
# ------------------------------------------------------------------

def _read_csv_with_date(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    if date_col is None:
        raise ValueError(f"No date column in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = 'Date'
    return df


def load_signals() -> dict:
    """Load 6 Phase B PASS signals — IDs 6, 21, 23, 26, 28, 41."""
    data_dir = ROOT / 'data'
    sig: dict = {}

    # #6 VIX level
    sig[6] = (
        'VIX level',
        _read_csv_with_date(data_dir / 'vixcls_daily.csv').iloc[:, 0].astype(float).dropna(),
    )
    # #21 ICE BofA HY OAS
    sig[21] = (
        'ICE BofA HY OAS',
        _read_csv_with_date(data_dir / 'hy_spread_daily.csv').iloc[:, 0].astype(float).dropna(),
    )
    # #23 BAA-10Y credit spread
    sig[23] = (
        'BAA-10Y credit spread',
        _read_csv_with_date(data_dir / 'baa10y_daily.csv').iloc[:, 0].astype(float).dropna(),
    )
    # #26 2s10s spread = DGS10 - DGS2
    dgs10 = _read_csv_with_date(data_dir / 'dgs10_daily.csv').iloc[:, 0].astype(float)
    dgs2 = _read_csv_with_date(data_dir / 'dgs2_daily.csv').iloc[:, 0].astype(float)
    sig[26] = ('2s10s spread', (dgs10 - dgs2).dropna())
    # #41 DXY
    sig[41] = (
        'DXY',
        _read_csv_with_date(data_dir / 'dxy_daily.csv').iloc[:, 0].astype(float).dropna(),
    )
    # #28 10Y real yield (DGS10 - CPI YoY%)
    cpi = _read_csv_with_date(data_dir / 'cpiaucsl_monthly.csv').iloc[:, 0].astype(float)
    cpi_yoy = (cpi / cpi.shift(12) - 1.0) * 100.0
    cpi_daily = cpi_yoy.resample('B').ffill()
    sig[28] = ('10Y real yield', (dgs10 - cpi_daily).dropna())
    return sig


def prepare_signal(name_and_raw: tuple) -> tuple:
    """Quantize raw signal to 4-level + apply daily publication lag."""
    name, raw = name_and_raw
    q = quantile_cut(raw, levels=4)
    q_lagged = apply_publication_lag(q, lag_type='daily')
    q_lagged = q_lagged[~q_lagged.index.duplicated(keep='first')]
    return name, q_lagged


# ------------------------------------------------------------------
# Pattern execution
# ------------------------------------------------------------------

def _enumerate_tier(tier: int):
    if tier == 1:
        return enumerate_tier1()
    if tier == 2:
        return enumerate_tier2()
    if tier == 3:
        return enumerate_tier3()
    if tier == 4:
        return enumerate_tier4()
    raise ValueError(f"Unknown tier: {tier}")


def run_tier(tier: int, date_suffix: str) -> pd.DataFrame:
    print(f'[tier-{tier}] Loading baselines (baseline_navs_{date_suffix}.parquet)...')
    baselines = pd.read_parquet(
        ROOT / 'data' / 'signals' / 'integration' / f'baseline_navs_{date_suffix}.parquet'
    )
    print(f'  baselines: {list(baselines.columns)}, {len(baselines):,} obs')

    print(f'[tier-{tier}] Loading + quantizing 6 signals...')
    raw_signals = load_signals()
    signals = {sid: prepare_signal(v) for sid, v in raw_signals.items()}
    for sid, (name, s) in signals.items():
        print(f'    #{sid:3d} {name:25s} nonnull={int(s.notna().sum()):>6,}')

    patterns = _enumerate_tier(tier)
    print(f'[tier-{tier}] Enumerated {len(patterns)} patterns')

    results = []
    for i, p in enumerate(patterns):
        if (i + 1) % 12 == 0 or (i + 1) == len(patterns):
            print(f'  [{i+1:>3}/{len(patterns)}] processing...')

        row: dict = {**p.__dict__}

        # Resolve baseline NAV
        if p.strategy not in baselines.columns:
            row['judgment'] = 'SKIP_NO_BASELINE'
            results.append(row)
            continue

        # Resolve signal (single int only at Tier 1; combos/composites later)
        if not isinstance(p.signal_id, int):
            row['judgment'] = 'SKIP_COMBO_OR_COMPOSITE'
            results.append(row)
            continue
        if p.signal_id not in signals:
            row['judgment'] = 'SKIP_NO_SIGNAL'
            results.append(row)
            continue

        signal_name, signal_series = signals[p.signal_id]
        row['signal_name'] = signal_name

        # Resolve method
        method_fn = METHOD_REGISTRY.get((p.method, p.direction))
        if method_fn is None:
            row['judgment'] = 'SKIP_NO_METHOD'
            results.append(row)
            continue

        try:
            base_nav = baselines[p.strategy].dropna()
            cand_nav = method_fn(base_nav, signal_series)
            metrics = evaluate(cand_nav, base_nav, split_date='2018-01-01')
            judgment = judge_improvement(metrics)
            row.update(metrics)
            row.update(judgment)
        except Exception as exc:  # noqa: BLE001
            row['judgment'] = f'ERROR: {type(exc).__name__}: {exc}'

        results.append(row)

    df = pd.DataFrame(results)
    out = ROOT / 'data' / 'signals' / 'integration' / f'tier{tier}_results_{date_suffix}.csv'
    df.to_csv(out, index=False)
    print(f'\n[tier-{tier}] Wrote {out}')

    # Summary
    if 'judgment' in df.columns:
        print(f'\n[tier-{tier}] Judgment counts (overall):')
        print(df['judgment'].value_counts().to_string())
        if 'strategy' in df.columns:
            print(f'\n[tier-{tier}] Per-strategy summary:')
            for strat in ['S1', 'S2', 'S3']:
                sub = df[df['strategy'] == strat]
                strong = int((sub['judgment'] == 'STRONG_PASS').sum())
                std = int((sub['judgment'] == 'STANDARD_PASS').sum())
                marg = int((sub['judgment'] == 'MARGINAL').sum())
                fail = int((sub['judgment'] == 'FAIL').sum())
                print(f'  {strat}: STRONG={strong} STANDARD={std} MARGINAL={marg} FAIL={fail}')

    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument('--date', type=str, default='20260605',
                    help='Date suffix for baseline parquet and output CSV')
    args = ap.parse_args()
    run_tier(args.tier, args.date)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
