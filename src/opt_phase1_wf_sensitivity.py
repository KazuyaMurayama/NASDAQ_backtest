"""
Phase 1 Tier 1: Walk-Forward Validation + Sensitivity Analysis
==============================================================
Candidate: TV 10-30%, MD 60/180, VIX 0.20
Baseline: TV 15-35%, MD 60/180, VIX 0.20
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal, calc_metrics
from test_ens2_strategies import calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy

ANNUAL_COST = 0.0086; DELAY = 2; BASE_LEV = 3.0; THRESHOLD = 0.20
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

def run_bt(close, leverage):
    returns = close.pct_change()
    lev_ret = returns * BASE_LEV
    daily_cost = ANNUAL_COST / 252
    delayed = leverage.shift(DELAY)
    strat_ret = delayed * (lev_ret - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret

def build_a2_lev(close, returns, md_s=60, md_l=180, vc=0.20, tv_lo=0.15, tv_hi=0.35):
    dd = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close, 150, tv_lo, tv_hi, 0.85, 1.15)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope = calc_slope_multiplier(close)
    mom = calc_momentum_decel_mult(close, md_s, md_l, 0.3, 0.5, 1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - vc * vix_z).clip(0.5, 1.15)
    raw = dd * vt_lev * slope * mom * vix_mult
    raw = raw.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)
    return lev, dd

def sharpe_for_period(close, dates, lev, start, end):
    mask = (dates >= start) & (dates < end)
    if mask.sum() < 100:
        return np.nan
    idx = dates[mask].index
    c = close.iloc[idx[0]:idx[-1]+1]
    l = lev.iloc[idx[0]:idx[-1]+1]
    nav, ret = run_bt(c, l)
    if ret.std() == 0:
        return 0
    return (ret.mean() * 252) / (ret.std() * np.sqrt(252))

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()}\n")

    # ===== Walk-Forward Validation =====
    print("=" * 90)
    print("WALK-FORWARD VALIDATION (3 windows)")
    print("=" * 90)

    wf_windows = [
        ('1974-01-01', '2010-01-01', '2010-01-01', '2015-12-31'),
        ('1979-01-01', '2015-01-01', '2015-01-01', '2020-12-31'),
        ('1984-01-01', '2020-01-01', '2020-01-01', '2026-12-31'),
    ]

    # Candidate: TV 10-30%
    # Baseline: TV 15-35%
    # Also test Top 2-5 from grid
    configs = [
        ('Baseline (15-35%)', 60, 180, 0.20, 0.15, 0.35),
        ('Candidate (10-30%)', 60, 180, 0.20, 0.10, 0.30),
        ('Alt1 (10-30% VC=0.25)', 60, 180, 0.25, 0.10, 0.30),
        ('Alt2 (10-30% VC=0.15)', 60, 180, 0.15, 0.10, 0.30),
        ('Alt3 (20-40% VC=0.10)', 60, 180, 0.10, 0.20, 0.40),
        ('Alt4 (15-50% VC=0.15)', 60, 180, 0.15, 0.15, 0.50),
    ]

    wf_results = []
    for name, md_s, md_l, vc, tv_lo, tv_hi in configs:
        lev, dd = build_a2_lev(close, returns, md_s, md_l, vc, tv_lo, tv_hi)
        test_sharpes = []
        for train_s, train_e, test_s, test_e in wf_windows:
            ts = sharpe_for_period(close, dates, lev, test_s, test_e)
            test_sharpes.append(ts)
        wf_avg = np.nanmean(test_sharpes)

        # Full period
        nav, strat_ret = run_bt(close, lev)
        fm = calc_metrics(nav, strat_ret, dd, dates)

        wf_results.append({
            'Config': name, 'Full_Sharpe': fm['Sharpe'],
            'WF1': test_sharpes[0], 'WF2': test_sharpes[1], 'WF3': test_sharpes[2],
            'WF_Avg': wf_avg, 'CAGR': fm['CAGR'], 'MaxDD': fm['MaxDD'],
            'Worst5Y': fm['Worst5Y'],
        })

    print(f"\n{'Config':<26} {'Full':>6} {'WF1':>6} {'WF2':>6} {'WF3':>6} {'WF_Avg':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7}")
    print("-" * 100)
    for r in wf_results:
        print(f"{r['Config']:<26} {r['Full_Sharpe']:.3f} {r['WF1']:.3f} {r['WF2']:.3f} "
              f"{r['WF3']:.3f} {r['WF_Avg']:>7.3f} {r['CAGR']*100:>6.2f}% "
              f"{r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>6.2f}%")

    # ===== Sensitivity Analysis =====
    print(f"\n{'=' * 90}")
    print("SENSITIVITY ANALYSIS: Candidate (TV 10-30%, MD 60/180, VIX 0.20)")
    print("Each param varied ±20% while others fixed")
    print("=" * 90)

    base_params = {'md_s': 60, 'md_l': 180, 'vc': 0.20, 'tv_lo': 0.10, 'tv_hi': 0.30}
    lev_base, dd_base = build_a2_lev(close, returns, **base_params)
    nav_base, ret_base = run_bt(close, lev_base)
    base_sharpe = calc_metrics(nav_base, ret_base, dd_base, dates)['Sharpe']
    print(f"\nBase Sharpe: {base_sharpe:.4f}")

    # Sensitivity for each param
    sens_tests = {
        'tv_lo': [0.08, 0.09, 0.10, 0.11, 0.12],
        'tv_hi': [0.24, 0.27, 0.30, 0.33, 0.36],
        'vc': [0.16, 0.18, 0.20, 0.22, 0.24],
        'md_s': [48, 54, 60, 66, 72],
        'md_l': [144, 162, 180, 198, 216],
    }

    print(f"\n{'Param':<8} {'Values':>40} {'Sharpes':>50} {'MaxDelta':>10}")
    print("-" * 115)

    for param, values in sens_tests.items():
        sharpes = []
        for v in values:
            p = base_params.copy()
            p[param] = v
            lev, dd = build_a2_lev(close, returns, **p)
            nav, ret = run_bt(close, lev)
            s = calc_metrics(nav, ret, dd, dates)['Sharpe']
            sharpes.append(s)
        max_delta = max(abs(s - base_sharpe) / base_sharpe * 100 for s in sharpes)
        vals_str = ', '.join(f'{v}' for v in values)
        sh_str = ', '.join(f'{s:.4f}' for s in sharpes)
        status = "✅ SMOOTH" if max_delta < 5 else "⚠️ SENSITIVE"
        print(f"{param:<8} [{vals_str:>38}] [{sh_str:>48}] {max_delta:>6.1f}% {status}")

    # Save
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'opt_phase1_wf_results.csv')
    pd.DataFrame(wf_results).to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # ===== VERDICT =====
    print(f"\n{'=' * 90}")
    print("TIER 1 VERDICT")
    print("=" * 90)
    base_wf = [r for r in wf_results if 'Baseline' in r['Config']][0]
    cand_wf = [r for r in wf_results if 'Candidate' in r['Config']][0]
    print(f"  Baseline WF_Avg:  {base_wf['WF_Avg']:.4f}")
    print(f"  Candidate WF_Avg: {cand_wf['WF_Avg']:.4f}")
    if cand_wf['WF_Avg'] > base_wf['WF_Avg']:
        print("  → ✅ Candidate PASSES Walk-Forward")
    else:
        print("  → ❌ Candidate FAILS Walk-Forward")

if __name__ == '__main__':
    main()
