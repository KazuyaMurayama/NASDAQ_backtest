"""Regenerate DH-W1 NAV and persist to audit_results/_cache/dh_w1_nav_cache.pkl.

S3 baseline (DH-W1 = asymmetric+hysteresis HOLD/OUT × DH base allocation)
was missing from the integration baseline parquet. This script rebuilds
it by calling build_W1(load_shared_assets()) and dumping a dict-of-Series
matching the S1/S2 cache schema expected by baseline_loader._coerce_nav.

Session S2 deliverable per SIGNAL_INTEGRATION_PLAN_20260604.md (Phase 1).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pickle
import pandas as pd

# Imports must come after sys.path is patched.
from g23a_dh_refinement_variants import build_W1  # noqa: E402
from g14_wfa_sbi_cfd import load_shared_assets  # noqa: E402


def main() -> int:
    print('[regen_dh_w1] Loading shared assets (NDX/Gold/Bond + S2/F10 prereqs)...')
    assets = load_shared_assets()
    print('[regen_dh_w1] Building W1 NAV (asymmetric+hysteresis state machine)...')
    nav, cost, mask, wn, lev_raw = build_W1(assets)

    dates_series = pd.Series(pd.to_datetime(assets['dates'].values)).reset_index(drop=True)
    nav_series = pd.Series(nav)
    cost_series = pd.Series(cost)

    cache = {
        'dates':         dates_series,
        'nav_dh_w1':     nav_series,
        'cost_dh_w1':    cost_series,
        'mask_dh_w1':    pd.Series(mask),
        'wn_dh_w1':      pd.Series(wn),
        'lev_raw_dh_w1': pd.Series(lev_raw),
    }
    out = ROOT / 'audit_results' / '_cache' / 'dh_w1_nav_cache.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(cache, f)

    first_dt = dates_series.iloc[0]
    last_dt = dates_series.iloc[-1]
    print(
        f'[regen_dh_w1] Wrote {out}, '
        f'NAV {first_dt.date()} -> {last_dt.date()}, '
        f'end value {float(nav_series.iloc[-1]):.4e}, '
        f'HOLD mask mean={float(mask.mean()):.4f}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
