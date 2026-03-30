"""
Phase 1 Tier 3: DD thresholds / VIX MA window / Rebalance threshold
Fixed from T1+T2: TV=10-30%, VIX=0.25, MD=60/180, SB=0.9, SS=0.35, Asym=30/10
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal, calc_metrics
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy

ANNUAL_COST = 0.0086; DELAY = 2; BASE_LEV = 3.0
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

def run_bt(close, leverage):
    returns = close.pct_change()
    lev_ret = returns * BASE_LEV; daily_cost = ANNUAL_COST / 252
    delayed = leverage.shift(DELAY)
    strat_ret = delayed * (lev_ret - daily_cost); strat_ret = strat_ret.fillna(0)
    return (1 + strat_ret).cumprod(), strat_ret

def calc_asym_ewma(returns, span_up=30, span_dn=10):
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.iloc[:20].var() if len(returns)>20 else 0.0001
    for i in range(1, len(returns)):
        ret = returns.iloc[i]
        alpha = 2/(span_dn+1) if ret < 0 else 2/(span_up+1)
        variance.iloc[i] = (1-alpha)*variance.iloc[i-1] + alpha*(ret**2)
    return np.sqrt(variance * 252)

def calc_trend_tv(close, tv_min=0.10, tv_max=0.30):
    ma = close.rolling(150).mean(); ratio = close / ma
    tv = tv_min + (tv_max - tv_min) * (ratio - 0.85)/(1.15-0.85)
    return tv.clip(tv_min, tv_max).fillna(0.20)

def calc_slope(close, base=0.9, sens=0.35):
    ma = close.rolling(200).mean(); slope = ma.pct_change()
    sm = slope.rolling(60).mean(); ss = slope.rolling(60).std().replace(0, 0.0001)
    z = (slope - sm)/ss
    return (base + sens*z).clip(0.3, 1.5).fillna(1.0)

def build_a2(close, returns, dates, dd_exit=0.82, dd_re=0.92, vix_ma_win=252, rebal_th=0.20):
    dd = calc_dd_signal(close, dd_exit, dd_re)
    asym_vol = calc_asym_ewma(returns, 30, 10)
    trend_tv = calc_trend_tv(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope = calc_slope(close)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(vix_ma_win).mean()
    vix_std = vix_proxy.rolling(vix_ma_win).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.25 * vix_z).clip(0.5, 1.15)
    raw = dd * vt_lev * slope * mom * vix_mult
    raw = raw.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw, rebal_th)
    nav, strat_ret = run_bt(close, lev)
    fm = calc_metrics(nav, strat_ret, dd, dates)
    split_idx = dates[dates >= '2021-05-07'].index[0]
    nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
    ret_oos = strat_ret.iloc[split_idx:]
    oos_sh = (ret_oos.mean()*252)/(ret_oos.std()*np.sqrt(252)) if ret_oos.std()>0 else 0
    return {**fm, 'OOS_Sharpe': oos_sh}, lev, dd

def sharpe_period(close, dates, lev, start, end):
    mask = (dates >= start) & (dates < end)
    if mask.sum() < 100: return np.nan
    idx = dates[mask].index
    c = close.iloc[idx[0]:idx[-1]+1]; l = lev.iloc[idx[0]:idx[-1]+1]
    nav, ret = run_bt(c, l)
    if ret.std() == 0: return 0
    return (ret.mean()*252)/(ret.std()*np.sqrt(252))

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']

    dd_params = [(0.80, 0.90), (0.82, 0.92), (0.84, 0.94)]
    vix_wins = [126, 189, 252]
    rebal_ths = [0.15, 0.20, 0.25]

    print("=" * 90)
    print("PHASE 1 TIER 3: DD / VIX_MA / Rebalance (27 patterns)")
    print("Fixed: TV=10-30%, VIX=0.25, MD=60/180, SB=0.9, SS=0.35, Asym=30/10")
    print("=" * 90)

    results = []
    total = len(dd_params) * len(vix_wins) * len(rebal_ths)
    i = 0
    for dd_e, dd_r in dd_params:
        for vw in vix_wins:
            for rt in rebal_ths:
                i += 1
                m, lev, dd = build_a2(close, returns, dates, dd_e, dd_r, vw, rt)
                m['DD'] = f"{dd_e:.2f}/{dd_r:.2f}"
                m['VIX_MA'] = vw; m['Rebal'] = rt
                results.append(m)

    rdf = pd.DataFrame(results).sort_values('Sharpe', ascending=False)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'opt_phase1_tier3_results.csv')
    rdf.to_csv(out, index=False)

    print(f"\nTOP 10 by Sharpe:")
    print(f"{'DD':>10} {'VIX_MA':>7} {'Rebal':>6} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} {'OOS':>7}")
    print("-" * 70)
    for _, r in rdf.head(10).iterrows():
        print(f"{r['DD']:>10} {int(r['VIX_MA']):>7} {r['Rebal']:>6.2f} "
              f"{r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% "
              f"{r['Worst5Y']*100:>6.2f}% {r['OOS_Sharpe']:>7.4f}")

    # Walk-Forward for top 3 + baseline
    print(f"\n{'=' * 90}")
    print("WALK-FORWARD: Top 3 vs Baseline(0.82/0.92, 252, 0.20)")
    print("=" * 90)
    wf_windows = [('2010-01-01','2015-12-31'), ('2015-01-01','2020-12-31'), ('2020-01-01','2026-12-31')]

    configs = [('Baseline', 0.82, 0.92, 252, 0.20)]
    for _, r in rdf.head(3).iterrows():
        de, dr = [float(x) for x in r['DD'].split('/')]
        configs.append((f"({r['DD']}/V{int(r['VIX_MA'])}/R{r['Rebal']})",
                        de, dr, int(r['VIX_MA']), r['Rebal']))

    for name, de, dr, vw, rt in configs:
        m, lev, dd = build_a2(close, returns, dates, de, dr, vw, rt)
        wfs = [sharpe_period(close, dates, lev, s, e) for s, e in wf_windows]
        print(f"  {name:<35} Full={m['Sharpe']:.4f}  WF=[{wfs[0]:.3f}, {wfs[1]:.3f}, {wfs[2]:.3f}]  Avg={np.nanmean(wfs):.4f}")

    print(f"\nSaved to {out}")

if __name__ == '__main__':
    main()
