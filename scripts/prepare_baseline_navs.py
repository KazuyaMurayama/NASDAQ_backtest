"""Prepare baseline NAVs for all 3 strategies (S1, S2, S3) and save as parquet.

S1: F10 (NEW CANDIDATE)             — audit_results/_cache/f10_nav_cache.pkl
S2: D5 (vz065lmax5)                 — audit_results/_cache/vz065lmax5_nav_cache.pkl
S3: DH-W1                           — TODO (Session S2): regenerate via build_W1()

Writes: data/signals/integration/baseline_navs_20260604.parquet

Session S1 deliverable per SIGNAL_INTEGRATION_PLAN_20260604.md.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd  # noqa: E402

from integration.baseline_loader import load_all_baselines  # noqa: E402


OUT = ROOT / 'data' / 'signals' / 'integration' / 'baseline_navs_20260605.parquet'


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print('[prepare] Loading baseline NAVs (S1, S2, S3)...')
    df = load_all_baselines()
    print(f'  shape   : {df.shape}')
    print(f'  columns : {list(df.columns)}')
    print(f'  date    : {df.index.min()} to {df.index.max()}')

    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 1:
            years = (s.index[-1] - s.index[0]).days / 365.25
            cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
            print(
                f'  {col}: {len(s):>6,} obs, '
                f'CAGR={cagr:+.2%}, '
                f'NAV {float(s.iloc[0]):.3f} -> {float(s.iloc[-1]):.4e}'
            )

    df.to_parquet(OUT)
    size_kb = OUT.stat().st_size / 1024
    print(f'[prepare] Wrote {OUT} ({size_kb:.1f} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
