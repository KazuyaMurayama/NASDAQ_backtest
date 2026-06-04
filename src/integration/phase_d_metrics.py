"""Phase D — 9+1 metric comparison (audit grade) on natively-built NAVs.

Outputs both baseline (DH-W1) and candidate (DH-W1×BAA10Y×M2_proc) 9+1 metrics:
  CAGR_OOS, IS-OOS_gap, Sharpe_OOS, MaxDD, Worst10Y, P10_5Y,
  Trades/yr (lev_raw-change-based >5%),
  WFE (from phase_d_wfa SUMMARY),
  CI95_lo (from phase_d_wfa SUMMARY of per-window CAGR).
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
from g18_daily_trade_cost_wfa import metrics_from_nav  # noqa: E402
from g23a_dh_refinement_variants import build_W1  # noqa: E402
from build_w1_baa import build_W1_baa10y  # noqa: E402

REPO = os.path.dirname(_SRC)
WFA_SUMMARY = os.path.join(REPO, 'data', 'signals', 'integration',
                           'phase_d_wfa_50w_20260605_SUMMARY.csv')
OUT_CSV = os.path.join(REPO, 'data', 'signals', 'integration',
                       'phase_d_metrics_20260605.csv')


def trades_per_yr_from_levraw(lev_raw: np.ndarray, dates: pd.Series,
                              threshold: float = 0.05) -> float:
    """Count days where |Δlev_raw| > threshold (relative change), divide by years."""
    lr = np.asarray(lev_raw, dtype=float)
    base = np.where(np.abs(lr[:-1]) > 1e-9, np.abs(lr[:-1]), 1.0)
    rel = np.abs(lr[1:] - lr[:-1]) / base
    flips = int((rel > threshold).sum())
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return flips / max(years, 1e-9)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 80)
    print('Phase D — 9+1 metrics (audit grade)')
    print('=' * 80)

    a = load_shared_assets()
    dates = a['dates']
    ret = a['ret']

    nav_b, _, mask_b, wn_b, lev_b = build_W1(a)
    nav_c, _, mask_c, wn_c, lev_c = build_W1_baa10y(a)

    m_b = metrics_from_nav(nav_b, dates, ret)
    m_c = metrics_from_nav(nav_c, dates, ret)

    tr_b = trades_per_yr_from_levraw(lev_b, dates, threshold=0.05)
    tr_c = trades_per_yr_from_levraw(lev_c, dates, threshold=0.05)

    # WFA stats
    wfa = pd.read_csv(WFA_SUMMARY).iloc[0].to_dict()
    wfe_b = float(wfa['baseline_wfe_all'])
    wfe_c = float(wfa['candidate_wfe_all'])
    ci_lo_b = float(wfa['baseline_ci95_lo_cagr'])
    ci_lo_c = float(wfa['candidate_ci95_lo_cagr'])

    rows = [
        dict(metric='CAGR_OOS_pct', baseline=m_b['CAGR_OOS']*100, candidate=m_c['CAGR_OOS']*100,
             diff_pp=(m_c['CAGR_OOS']-m_b['CAGR_OOS'])*100,
             interpretation='positive diff = improvement'),
        dict(metric='IS_OOS_gap_pp', baseline=m_b['IS_OOS_gap']*100, candidate=m_c['IS_OOS_gap']*100,
             diff_pp=(m_c['IS_OOS_gap']-m_b['IS_OOS_gap'])*100,
             interpretation='negative diff = gap shrunk = improvement'),
        dict(metric='Sharpe_OOS', baseline=m_b['Sharpe_OOS'], candidate=m_c['Sharpe_OOS'],
             diff_pp=(m_c['Sharpe_OOS']-m_b['Sharpe_OOS']),
             interpretation='positive diff = improvement'),
        dict(metric='MaxDD_full_pct', baseline=m_b['MaxDD_FULL']*100, candidate=m_c['MaxDD_FULL']*100,
             diff_pp=(m_c['MaxDD_FULL']-m_b['MaxDD_FULL'])*100,
             interpretation='positive diff = less-negative DD = improvement'),
        dict(metric='Worst10Y_pct', baseline=m_b['Worst10Y_star']*100, candidate=m_c['Worst10Y_star']*100,
             diff_pp=(m_c['Worst10Y_star']-m_b['Worst10Y_star'])*100,
             interpretation='positive diff = improvement'),
        dict(metric='P10_5Y_pct', baseline=m_b['P10_5Y']*100, candidate=m_c['P10_5Y']*100,
             diff_pp=(m_c['P10_5Y']-m_b['P10_5Y'])*100,
             interpretation='positive diff = improvement'),
        dict(metric='Trades_per_yr', baseline=tr_b, candidate=tr_c,
             diff_pp=tr_c-tr_b,
             interpretation='hard cap 200/yr (lower / equal preferred)'),
        dict(metric='WFE_full', baseline=wfe_b, candidate=wfe_c,
             diff_pp=wfe_c-wfe_b,
             interpretation='≥ 1.0 strong; 0.95–1.05 OK; <0.95 weak'),
        dict(metric='CI95_lo_window_CAGR_pct', baseline=ci_lo_b*100, candidate=ci_lo_c*100,
             diff_pp=(ci_lo_c-ci_lo_b)*100,
             interpretation='>0 PASS'),
    ]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format='%.4f')

    print('\n  metric                       baseline      candidate     diff')
    for r in rows:
        if 'pct' in r['metric'] or 'pp' in r['metric']:
            fmt_b = f'{r["baseline"]:+8.3f}'
            fmt_c = f'{r["candidate"]:+8.3f}'
            fmt_d = f'{r["diff_pp"]:+8.3f}'
        elif 'Trades' in r['metric']:
            fmt_b = f'{r["baseline"]:8.1f}'
            fmt_c = f'{r["candidate"]:8.1f}'
            fmt_d = f'{r["diff_pp"]:+8.1f}'
        else:
            fmt_b = f'{r["baseline"]:+8.3f}'
            fmt_c = f'{r["candidate"]:+8.3f}'
            fmt_d = f'{r["diff_pp"]:+8.3f}'
        print(f'  {r["metric"]:28s}  {fmt_b}    {fmt_c}    {fmt_d}')
    print(f'\n→ CSV: {OUT_CSV}')


if __name__ == '__main__':
    main()
