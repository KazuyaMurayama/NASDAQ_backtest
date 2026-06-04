"""Phase D — Stationary block bootstrap on OOS daily returns.

Compares OOS CAGR of:
  - Baseline:  DH-W1 (build_W1)
  - Candidate: DH-W1 × BAA-10Y × M2 procyclical (build_W1_baa10y)

Method
------
Paired block bootstrap (block_size=60 trading days, n_resamples=10000).
For each resample, we draw the SAME block-start indices for both series
(paired) so paired CAGR diff preserves the date alignment / regime overlap.

OOS window: 2021-05-08 to 2026-03-26 (matches g23e and product OOS).

Reports
-------
- median, [2.5, 97.5] percentile of paired diff (cand - base) OOS CAGR pp
- P(diff > 0)
- median candidate OOS CAGR, [5%, 95%] percentile
- median baseline OOS CAGR, [5%, 95%] percentile

PASS gate for Phase D: P(diff > 0) > 0.90
"""
from __future__ import annotations
import os
import sys
import types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_THIS)
sys.path.insert(0, _SRC)
sys.path.insert(0, _THIS)

import numpy as np
import pandas as pd

from g14_wfa_sbi_cfd import load_shared_assets  # noqa: E402
from g18_daily_trade_cost_wfa import OOS_START_TS, OOS_END_TS  # noqa: E402
from g23a_dh_refinement_variants import build_W1  # noqa: E402
from build_w1_baa import build_W1_baa10y  # noqa: E402
from corrected_strategy_backtest import TRADING_DAYS  # noqa: E402


REPO = os.path.dirname(_SRC)
OUT_CSV = os.path.join(REPO, 'data', 'signals', 'integration',
                       'phase_d_bootstrap_20260605.csv')

N_BOOTSTRAP = 10000
BLOCK_SIZE = 60
RNG_SEED = 42


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('Phase D Bootstrap — DH-W1 vs DH-W1×BAA10Y×M2_procyclical (OOS)')
    print(f'  block_size={BLOCK_SIZE}, n_resamples={N_BOOTSTRAP}, seed={RNG_SEED}')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']

    print('\n  Building baseline ...')
    nav_b, _, _, _, _ = build_W1(a)
    print('  Building candidate ...')
    nav_c, _, _, _, _ = build_W1_baa10y(a)

    oos_mask = (dates >= OOS_START_TS) & (dates <= OOS_END_TS)
    n_oos = int(oos_mask.sum())
    years_oos = n_oos / TRADING_DAYS

    ret_b = nav_b.pct_change().fillna(0).values[oos_mask.values]
    ret_c = nav_c.pct_change().fillna(0).values[oos_mask.values]

    cum_b = float(np.prod(1.0 + ret_b))
    cum_c = float(np.prod(1.0 + ret_c))
    cagr_b_actual = cum_b ** (1.0 / years_oos) - 1.0
    cagr_c_actual = cum_c ** (1.0 / years_oos) - 1.0
    actual_diff_pp = (cagr_c_actual - cagr_b_actual) * 100
    print(f'\n  OOS days={n_oos}, years={years_oos:.2f}')
    print(f'  Actual OOS CAGR: base={cagr_b_actual*100:+.2f}%  '
          f'cand={cagr_c_actual*100:+.2f}%  diff={actual_diff_pp:+.2f}pp')

    rng = np.random.default_rng(RNG_SEED)
    n_blocks = int(np.ceil(n_oos / BLOCK_SIZE))

    cagr_b_arr = np.empty(N_BOOTSTRAP, dtype=float)
    cagr_c_arr = np.empty(N_BOOTSTRAP, dtype=float)
    diff_arr = np.empty(N_BOOTSTRAP, dtype=float)

    print(f'\n  Resampling {N_BOOTSTRAP} block-paired draws ...')
    for i in range(N_BOOTSTRAP):
        # PAIRED block starts
        starts = rng.integers(0, n_oos - BLOCK_SIZE + 1, size=n_blocks)
        s_b = np.concatenate([ret_b[s:s+BLOCK_SIZE] for s in starts])[:n_oos]
        s_c = np.concatenate([ret_c[s:s+BLOCK_SIZE] for s in starts])[:n_oos]
        c_b = float(np.prod(1.0 + s_b))
        c_c = float(np.prod(1.0 + s_c))
        cg_b = c_b ** (1.0 / years_oos) - 1.0 if c_b > 0 else -1.0
        cg_c = c_c ** (1.0 / years_oos) - 1.0 if c_c > 0 else -1.0
        cagr_b_arr[i] = cg_b
        cagr_c_arr[i] = cg_c
        diff_arr[i] = (cg_c - cg_b) * 100
        if (i + 1) % 2500 == 0:
            print(f'    {i+1}/{N_BOOTSTRAP}')

    # distribution stats
    def _ptile(arr, p):
        return float(np.percentile(arr, p))

    median_b = _ptile(cagr_b_arr, 50) * 100
    p5_b = _ptile(cagr_b_arr, 5) * 100
    p95_b = _ptile(cagr_b_arr, 95) * 100
    median_c = _ptile(cagr_c_arr, 50) * 100
    p5_c = _ptile(cagr_c_arr, 5) * 100
    p95_c = _ptile(cagr_c_arr, 95) * 100
    median_d = _ptile(diff_arr, 50)
    ci_lo = _ptile(diff_arr, 2.5)
    ci_hi = _ptile(diff_arr, 97.5)
    p_pos = float((diff_arr > 0).mean())
    p5_d = _ptile(diff_arr, 5)
    p95_d = _ptile(diff_arr, 95)

    print('\n[Bootstrap result]')
    print(f'  base CAGR : median={median_b:+.2f}%  [P5={p5_b:+.2f}%, P95={p95_b:+.2f}%]')
    print(f'  cand CAGR : median={median_c:+.2f}%  [P5={p5_c:+.2f}%, P95={p95_c:+.2f}%]')
    print(f'  diff pp   : median={median_d:+.2f}pp  [P2.5={ci_lo:+.2f}, P97.5={ci_hi:+.2f}]')
    print(f'             [P5={p5_d:+.2f}, P95={p95_d:+.2f}]')
    print(f'  P(diff>0) = {p_pos*100:.2f}%   (PASS gate >90%)')

    out = pd.DataFrame([dict(
        n_oos_days=n_oos,
        years_oos=years_oos,
        actual_cagr_baseline_pct=cagr_b_actual * 100,
        actual_cagr_candidate_pct=cagr_c_actual * 100,
        actual_diff_pp=actual_diff_pp,
        block_size=BLOCK_SIZE,
        n_resamples=N_BOOTSTRAP,
        seed=RNG_SEED,
        bootstrap_median_baseline_cagr_pct=median_b,
        bootstrap_P5_baseline_cagr_pct=p5_b,
        bootstrap_P95_baseline_cagr_pct=p95_b,
        bootstrap_median_candidate_cagr_pct=median_c,
        bootstrap_P5_candidate_cagr_pct=p5_c,
        bootstrap_P95_candidate_cagr_pct=p95_c,
        bootstrap_median_diff_pp=median_d,
        bootstrap_CI95_lo_diff_pp=ci_lo,
        bootstrap_CI95_hi_diff_pp=ci_hi,
        bootstrap_P5_diff_pp=p5_d,
        bootstrap_P95_diff_pp=p95_d,
        P_cand_gt_base=p_pos,
    )])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format='%.6f')
    print(f'\n→ CSV: {OUT_CSV}')

    print('\n[Verdict]')
    if p_pos > 0.90:
        print(f'  ✓ P(cand>base)={p_pos*100:.2f}% > 90% — PASS')
    else:
        print(f'  ✗ P(cand>base)={p_pos*100:.2f}% ≤ 90% — FAIL')


if __name__ == '__main__':
    main()
