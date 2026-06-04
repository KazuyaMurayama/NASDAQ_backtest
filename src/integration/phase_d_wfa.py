"""Phase D — Native WFA on DH-W1 baseline vs DH-W1 × BAA-10Y × M2 procyclical.

Uses g14_wfa_sbi_cfd.generate_windows (50 yearly windows from 1976-2026) and
g14_wfa_sbi_cfd.compute_window_metrics for per-window CAGR / Sharpe.

For each window we record:
  - IS/OOS labels (window ≥ OOS_START_TS=2021-05-08 is OOS-style; before is IS)
  - per-window CAGR and Sharpe for both baseline and candidate
  - candidate_minus_baseline CAGR
  - Trades/yr (from wn/wb/lev_raw turnover)

Summary statistics:
  - mean OOS Sharpe (cand and base)
  - WFE = mean(cand window Sharpe over windows where window_end ≥ OOS_START)
          / cand full-sample Sharpe
  - CI95_lo of cand per-window CAGR (across all windows, t-distribution)

Note: g14's generate_windows produces ~50 yearly windows. For the WFE
denominator we use the "full" Sharpe over the entire 1974-2026 span.
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
from scipy import stats

from g14_wfa_sbi_cfd import load_shared_assets, generate_windows, compute_window_metrics  # noqa: E402
from g18_daily_trade_cost_wfa import OOS_START_TS  # noqa: E402
from g23a_dh_refinement_variants import build_W1  # noqa: E402
from build_w1_baa import build_W1_baa10y  # noqa: E402


REPO = os.path.dirname(_SRC)
OUT_CSV = os.path.join(REPO, 'data', 'signals', 'integration',
                       'phase_d_wfa_50w_20260605.csv')


def _full_sharpe(nav: pd.Series) -> float:
    r = nav.pct_change().fillna(0).values
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return float('nan')
    return float(np.mean(r) / sd * np.sqrt(252))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('Phase D WFA — baseline DH-W1 vs candidate DH-W1×BAA10Y×M2_procyclical')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    windows = generate_windows(dates)
    print(f'\n[Windows: {len(windows)}]')

    # baseline
    print('\n[1/2] Building baseline DH-W1 ...')
    nav_b, _, mask_b, wn_b, lev_b = build_W1(a)
    wb_b = np.asarray(a['wb_A']) * mask_b
    # candidate
    print('[2/2] Building candidate DH-W1 × BAA-10Y × M2_procyclical ...')
    nav_c, _, mask_c, wn_c, lev_c = build_W1_baa10y(a)
    wb_c = np.asarray(a['wb_A']) * mask_c

    # peak lev sanity
    peak_b = float(np.nanmax(wn_b * lev_b * 3.0))
    peak_c = float(np.nanmax(wn_c * lev_c * 3.0))
    print(f'  peak_lev: baseline={peak_b:.3f}x  candidate={peak_c:.3f}x '
          f'(relaxed cap=3.9 for procyclical)')

    # per-window metrics
    rows = []
    for w in windows:
        m_b = compute_window_metrics(nav_b, w, wn=wn_b, wb=wb_b,
                                     lev_arr=lev_b * 3.0)
        m_c = compute_window_metrics(nav_c, w, wn=wn_c, wb=wb_c,
                                     lev_arr=lev_c * 3.0)
        is_oos = 'OOS' if w['end_date'] >= OOS_START_TS else 'IS'
        rows.append(dict(
            window_id=w['window_id'],
            window_kind=is_oos,
            is_start=str(w['start_date'].date()),
            is_end=str(w['end_date'].date()),
            oos_start=str(w['start_date'].date()) if is_oos == 'OOS' else '',
            oos_end=str(w['end_date'].date()) if is_oos == 'OOS' else '',
            n_days=w['n_days'],
            short_flag=w['short_flag'],
            baseline_cagr=m_b['CAGR'],
            candidate_cagr=m_c['CAGR'],
            baseline_sharpe=m_b['Sharpe'],
            candidate_sharpe=m_c['Sharpe'],
            baseline_maxdd=m_b['MaxDD'],
            candidate_maxdd=m_c['MaxDD'],
            candidate_minus_baseline_cagr=m_c['CAGR'] - m_b['CAGR'],
            candidate_minus_baseline_sharpe=m_c['Sharpe'] - m_b['Sharpe'],
            baseline_trades_yr=m_b['Trades_yr'],
            candidate_trades_yr=m_c['Trades_yr'],
        ))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format='%.6f')
    print(f'\n→ per-window CSV: {OUT_CSV}')

    # summary
    valid = df[~df['short_flag']].copy()
    oos_mask = valid['window_kind'] == 'OOS'

    def _ci95(arr):
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n < 2:
            return np.nan, np.nan, np.nan
        mu = float(arr.mean()); sd = float(arr.std(ddof=1))
        se = sd / np.sqrt(n) if n > 1 else np.nan
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        return mu - tcrit * se, mu + tcrit * se, mu

    # full-sample sharpe (for WFE denominator)
    full_sh_b = _full_sharpe(nav_b)
    full_sh_c = _full_sharpe(nav_c)

    mean_sh_b_all = float(valid['baseline_sharpe'].mean())
    mean_sh_c_all = float(valid['candidate_sharpe'].mean())
    wfe_b_all = mean_sh_b_all / full_sh_b if full_sh_b not in (0, float('nan')) else float('nan')
    wfe_c_all = mean_sh_c_all / full_sh_c if full_sh_c not in (0, float('nan')) else float('nan')

    mean_sh_b_oos = float(valid.loc[oos_mask, 'baseline_sharpe'].mean()) if oos_mask.any() else float('nan')
    mean_sh_c_oos = float(valid.loc[oos_mask, 'candidate_sharpe'].mean()) if oos_mask.any() else float('nan')

    ci_lo_b, ci_hi_b, mu_b = _ci95(valid['baseline_cagr'].values)
    ci_lo_c, ci_hi_c, mu_c = _ci95(valid['candidate_cagr'].values)
    ci_lo_d, ci_hi_d, mu_d = _ci95(valid['candidate_minus_baseline_cagr'].values)

    print('\n[WFA summary]')
    print(f'  full Sharpe          : base={full_sh_b:+.3f}  cand={full_sh_c:+.3f}')
    print(f'  mean window Sharpe   : base={mean_sh_b_all:+.3f}  cand={mean_sh_c_all:+.3f}')
    print(f'  WFE (mean / full)    : base={wfe_b_all:.3f}  cand={wfe_c_all:.3f}')
    print(f'  OOS mean Sharpe      : base={mean_sh_b_oos:+.3f}  cand={mean_sh_c_oos:+.3f}')
    print(f'  Mean window CAGR     : base={mu_b*100:+.2f}%  cand={mu_c*100:+.2f}%  diff={mu_d*100:+.2f}%')
    print(f'  CI95 base CAGR       : [{ci_lo_b*100:+.2f}%, {ci_hi_b*100:+.2f}%]')
    print(f'  CI95 cand CAGR       : [{ci_lo_c*100:+.2f}%, {ci_hi_c*100:+.2f}%]')
    print(f'  CI95 diff CAGR       : [{ci_lo_d*100:+.2f}%, {ci_hi_d*100:+.2f}%]')

    # Persist summary stats into a sidecar row
    summary_csv = OUT_CSV.replace('.csv', '_SUMMARY.csv')
    pd.DataFrame([dict(
        n_windows=int(len(valid)),
        n_oos_windows=int(oos_mask.sum()),
        baseline_full_sharpe=full_sh_b,
        candidate_full_sharpe=full_sh_c,
        baseline_wfe_all=wfe_b_all,
        candidate_wfe_all=wfe_c_all,
        baseline_mean_oos_sharpe=mean_sh_b_oos,
        candidate_mean_oos_sharpe=mean_sh_c_oos,
        baseline_mean_window_cagr=mu_b,
        candidate_mean_window_cagr=mu_c,
        diff_mean_window_cagr=mu_d,
        diff_ci95_lo=ci_lo_d,
        diff_ci95_hi=ci_hi_d,
        candidate_ci95_lo_cagr=ci_lo_c,
        candidate_ci95_hi_cagr=ci_hi_c,
        baseline_ci95_lo_cagr=ci_lo_b,
        baseline_ci95_hi_cagr=ci_hi_b,
        peak_lev_baseline=peak_b,
        peak_lev_candidate=peak_c,
    )]).to_csv(summary_csv, index=False, float_format='%.6f')
    print(f'→ summary CSV  : {summary_csv}')


if __name__ == '__main__':
    main()
