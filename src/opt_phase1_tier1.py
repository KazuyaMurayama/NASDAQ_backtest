"""
Phase 1 Tier 1: A2 Grid Search — MomDecel / VIX coeff / Target Vol
===================================================================
60 patterns, saved incrementally to avoid timeout.
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal, calc_metrics
from test_ens2_strategies import calc_asym_ewma_vol, calc_slope_multiplier, calc_trend_target_vol
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy

ANNUAL_COST = 0.0086
DELAY = 2
BASE_LEV = 3.0
THRESHOLD = 0.20
OOS_DATE = '2021-05-07'
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

def run_backtest(close, leverage, delay=DELAY, cost=ANNUAL_COST):
    returns = close.pct_change()
    lev_ret = returns * BASE_LEV
    daily_cost = cost / 252
    delayed = leverage.shift(delay)
    strat_ret = delayed * (lev_ret - daily_cost)
    strat_ret = strat_ret.fillna(0)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret

def build_a2(close, returns, dates,
             md_short=60, md_long=180, md_sens=0.3,
             vix_coeff=0.2, vix_floor=0.5, vix_ceil=1.15,
             tv_min=0.15, tv_max=0.35):
    dd = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_vol(returns, 20, 5)
    trend_tv = calc_trend_target_vol(close, 150, tv_min, tv_max, 0.85, 1.15)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope = calc_slope_multiplier(close)
    mom = calc_momentum_decel_mult(close, md_short, md_long, md_sens, 0.5, 1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - vix_coeff * vix_z).clip(vix_floor, vix_ceil)
    raw = dd * vt_lev * slope * mom * vix_mult
    raw = raw.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)
    nav, strat_ret = run_backtest(close, lev)
    m = calc_metrics(nav, strat_ret, dd, dates)
    # OOS
    split_idx = dates[dates >= OOS_DATE].index[0]
    nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
    ret_oos = strat_ret.iloc[split_idx:]
    oos_yrs = len(nav_oos) / 252
    oos_cagr = (nav_oos.iloc[-1] ** (1/oos_yrs)) - 1 if oos_yrs > 0 else 0
    oos_sh = (ret_oos.mean()*252)/(ret_oos.std()*np.sqrt(252)) if ret_oos.std()>0 else 0
    return {**m, 'OOS_CAGR': oos_cagr, 'OOS_Sharpe': oos_sh}

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()}")

    # Grid
    md_params = [(40, 120), (60, 180), (80, 240)]
    vix_coeffs = [0.10, 0.15, 0.20, 0.25, 0.30]
    tv_ranges = [(0.10, 0.30), (0.15, 0.35), (0.20, 0.40), (0.15, 0.50)]

    results = []
    total = len(md_params) * len(vix_coeffs) * len(tv_ranges)
    i = 0
    for md_s, md_l in md_params:
        for vc in vix_coeffs:
            for tv_lo, tv_hi in tv_ranges:
                i += 1
                m = build_a2(close, returns, dates,
                             md_short=md_s, md_long=md_l,
                             vix_coeff=vc, tv_min=tv_lo, tv_max=tv_hi)
                m['MD'] = f"{md_s}/{md_l}"
                m['VIX_coeff'] = vc
                m['TV_range'] = f"{tv_lo:.0%}-{tv_hi:.0%}"
                results.append(m)
                if i % 10 == 0:
                    print(f"  [{i}/{total}] MD={md_s}/{md_l} VC={vc} TV={tv_lo}-{tv_hi} "
                          f"Sharpe={m['Sharpe']:.4f} OOS={m['OOS_Sharpe']:.4f}")

    # Save
    rdf = pd.DataFrame(results)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'opt_phase1_tier1_results.csv')
    rdf.to_csv(out, index=False)
    print(f"\nSaved {len(results)} results to {out}")

    # Top 10
    rdf_sorted = rdf.sort_values('Sharpe', ascending=False)
    print(f"\nTOP 10 by Full-period Sharpe:")
    print(f"{'MD':<8} {'VIX_c':>5} {'TV':>10} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} {'OOS_Sh':>7}")
    print("-"*70)
    for _, r in rdf_sorted.head(10).iterrows():
        print(f"{r['MD']:<8} {r['VIX_coeff']:>5.2f} {r['TV_range']:>10} "
              f"{r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% "
              f"{r['Worst5Y']*100:>6.2f}% {r['OOS_Sharpe']:>7.4f}")

    # Baseline reference
    baseline = [r for r in results if r['MD']=='60/180' and r['VIX_coeff']==0.2
                and r['TV_range']=='15%-35%']
    if baseline:
        b = baseline[0]
        print(f"\nBaseline: Sharpe={b['Sharpe']:.4f}, OOS={b['OOS_Sharpe']:.4f}")

if __name__ == '__main__':
    main()
