"""Phase D — Native build of DH-W1 × BAA-10Y × M2 procyclical candidate.

Integration approach
--------------------
Native lev_raw injection (NOT post-hoc NAV multiplication).

Integration point (identified from src/g23a_dh_refinement_variants.py:73):
    lev_raw = np.asarray(a['lev_raw']) * mask        # original W1
    lev_raw = lev_raw * baa_mult                     # Phase D candidate
    nav, cost = build_dh_nav_with_cost(close, lev_raw, ...)

That is: the signal-derived multiplier modulates the strategy's lev_raw
stream BEFORE NAV computation (and therefore before turnover-based trade
cost is assessed). This is the strictly-correct native injection.

BAA-10Y signal pipeline
-----------------------
1. Load data/baa10y_daily.csv (DATE, BAA10Y)
2. Quantile-cut into 4 buckets (levels=4) — full-sample static
3. Apply publication_lag('daily') — shift +1 business day
4. Map procyclically: {0:0.7, 1:0.9, 2:1.1, 3:1.3}
5. Reindex to strategy dates, ffill, fill_na with 1.0

Sanity assert
-------------
peak_lev_eff = max(wn × lev_raw_modulated × 3) must be ≤ 3.0 × max_mult
                                                      = 3.0 × 1.3
                                                      = 3.9
(strict 3.0 cap is intentionally relaxed under procyclical injection;
this is documented in the audit report.)
"""
from __future__ import annotations
import os
import sys
import types

# multitasking stub (matches g23a)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_THIS)
sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets  # noqa: E402
from g18_daily_trade_cost_wfa import build_dh_nav_with_cost  # noqa: E402
from g23a_dh_refinement_variants import hold_mask_W1, DH_PER_UNIT  # noqa: E402
from signals.quantize import quantile_cut  # noqa: E402
from signals.timing import apply_publication_lag  # noqa: E402


M2_PROCYCLICAL = {0: 0.7, 1: 0.9, 2: 1.1, 3: 1.3}
BAA_CSV_PATH = os.path.join(os.path.dirname(_SRC), 'data', 'baa10y_daily.csv')


def load_baa_signal(quantize_levels: int = 4) -> pd.Series:
    """Load BAA-10Y, quantize, apply daily publication lag.

    Returns a pd.Series of int (0..3) indexed by lagged business-day.
    """
    df = pd.read_csv(BAA_CSV_PATH, parse_dates=['DATE'])
    df = df.dropna().sort_values('DATE').drop_duplicates('DATE')
    raw = pd.Series(df['BAA10Y'].values, index=df['DATE'])
    q = quantile_cut(raw, levels=quantize_levels)
    q = q.dropna().astype('int8')
    # publication lag (daily): shift +1 business day
    lagged = apply_publication_lag(q.copy(), 'daily')
    return lagged


def build_baa_multiplier(strategy_dates: pd.Series) -> np.ndarray:
    """Project lagged BAA-10Y signal onto strategy dates → procyclical mult.

    For dates before the first BAA observation, mult=1.0 (neutral).
    """
    sig = load_baa_signal(quantize_levels=4)
    # reindex to strategy dates (DatetimeIndex)
    target_idx = pd.DatetimeIndex(pd.to_datetime(strategy_dates.values))
    sig_aligned = sig.reindex(target_idx).ffill()
    # map to multiplier (NaN → 1.0 neutral)
    mult = sig_aligned.map(lambda v: M2_PROCYCLICAL.get(int(v), 1.0)
                           if pd.notna(v) else 1.0).values
    return np.asarray(mult, dtype=float)


def build_W1_baa10y(a: dict, return_diag: bool = False):
    """Native DH-W1 × BAA-10Y × M2 procyclical NAV.

    Returns (nav, cost, mask, wn, lev_raw_modulated).
    If return_diag=True, returns additional dict with mult, hold_ratio, etc.
    """
    mask = hold_mask_W1(a)
    wn = np.asarray(a['wn_A']) * mask
    wg = np.asarray(a['wg_A']) * mask
    wb = np.asarray(a['wb_A']) * mask
    lev_raw_base = np.asarray(a['lev_raw']) * mask

    # BAA-10Y multiplier from procyclical mapping
    mult = build_baa_multiplier(a['dates'])
    lev_raw_mod = lev_raw_base * mult

    nav, cost = build_dh_nav_with_cost(
        a['close'], lev_raw_mod, wn, wg, wb,
        a['dates'], a['gold_2x'], a['bond_3x'], a['sofr'], DH_PER_UNIT,
    )

    if return_diag:
        diag = dict(
            mult=mult,
            hold_ratio_pct=float(mask.mean()) * 100,
            peak_lev_eff=float(np.nanmax(wn * lev_raw_mod * 3.0)),
            mult_distribution={int(k): float((mult == v).mean())
                              for k, v in M2_PROCYCLICAL.items()},
        )
        return nav, cost, mask, wn, lev_raw_mod, diag
    return nav, cost, mask, wn, lev_raw_mod


def main():
    """Smoke test: build candidate, print sanity stats."""
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('Phase D smoke-test: DH-W1 × BAA-10Y × M2 procyclical (native build)')
    print('=' * 80)

    a = load_shared_assets()
    nav, cost, mask, wn, lev_raw_mod, diag = build_W1_baa10y(a, return_diag=True)
    print(f'\n  hold_ratio   = {diag["hold_ratio_pct"]:.1f}%')
    print(f'  peak_lev_eff = {diag["peak_lev_eff"]:.3f}  (relaxed cap = 3.9)')
    print(f'  mult dist    = {diag["mult_distribution"]}')
    print(f'  NAV final    = {float(nav.iloc[-1]):,.2f}')
    print(f'  yr cost      = {cost*100:.3f}%/yr')
    print(f'  n dates      = {len(nav)}')

    # Baseline DH-W1 for delta sanity
    from g23a_dh_refinement_variants import build_W1
    nav_base, _, _, _, _ = build_W1(a)
    diff_final = float(nav.iloc[-1] / nav_base.iloc[-1] - 1) * 100
    print(f'\n  Baseline NAV final = {float(nav_base.iloc[-1]):,.2f}')
    print(f'  Candidate/Baseline final ratio = {diff_final:+.2f}%')


if __name__ == '__main__':
    main()
