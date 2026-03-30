"""
Phase 1 Tier 2: SlopeMult sensitivity + base / AsymEWMA ratio
Tier 1 fixed: TV=10-30%, VIX=0.25, MD=60/180
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal, calc_metrics
from test_ens2_strategies import calc_asym_ewma_vol, calc_slope_multiplier
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

def calc_trend_tv_custom(close, ma_lb=150, tv_min=0.10, tv_max=0.30):
    ma = close.rolling(ma_lb).mean()
    ratio = close / ma
    tv = tv_min + (tv_max - tv_min) * (ratio - 0.85) / (1.15 - 0.85)
    return tv.clip(tv_min, tv_max).fillna(0.20)

def calc_slope_custom(close, ma_lb=200, norm_win=60, base=0.7, sens=0.3, mn=0.3, mx=1.5):
    ma = close.rolling(ma_lb).mean()
    slope = ma.pct_change()
    slope_mean = slope.rolling(norm_win).mean()
    slope_std = slope.rolling(norm_win).std().replace(0, 0.0001)
    z = (slope - slope_mean) / slope_std
    mult = base + sens * z
    return mult.clip(mn, mx).fillna(1.0)

def calc_asym_ewma_custom(returns, span_up=20, span_dn=5):
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    for i in range(1, len(returns)):
        ret = returns.iloc[i]
        alpha = 2/(span_dn+1) if ret < 0 else 2/(span_up+1)
        variance.iloc[i] = (1 - alpha) * variance.iloc[i-1] + alpha * (ret ** 2)
    return np.sqrt(variance * 252)

def build_a2(close, returns, dates, slope_base=0.7, slope_sens=0.3,
             asym_up=20, asym_dn=5):
    dd = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma_custom(returns, asym_up, asym_dn)
    trend_tv = calc_trend_tv_custom(close)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    slope = calc_slope_custom(close, base=slope_base, sens=slope_sens)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.25 * vix_z).clip(0.5, 1.15)
    raw = dd * vt_lev * slope * mom * vix_mult
    raw = raw.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)
    nav, strat_ret = run_bt(close, lev)
    fm = calc_metrics(nav, strat_ret, dd, dates)
    # OOS
    split_idx = dates[dates >= '2021-05-07'].index[0]
    nav_oos = nav.iloc[split_idx:] / nav.iloc[split_idx]
    ret_oos = strat_ret.iloc[split_idx:]
    oos_yrs = len(nav_oos)/252
    oos_sh = (ret_oos.mean()*252)/(ret_oos.std()*np.sqrt(252)) if ret_oos.std()>0 else 0
    return {**fm, 'OOS_Sharpe': oos_sh}

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

    # Grid search
    slope_bases = [0.5, 0.7, 0.9]
    slope_senss = [0.15, 0.25, 0.35, 0.45]
    asym_ratios = [(15, 3), (20, 5), (25, 7), (30, 10)]

    print("=" * 90)
    print("PHASE 1 TIER 2: SlopeMult + AsymEWMA Grid (48 patterns)")
    print("Fixed: TV=10-30%, VIX=0.25, MD=60/180")
    print("=" * 90)

    results = []
    total = len(slope_bases) * len(slope_senss) * len(asym_ratios)
    i = 0
    for sb in slope_bases:
        for ss in slope_senss:
            for au, ad in asym_ratios:
                i += 1
                m = build_a2(close, returns, dates, sb, ss, au, ad)
                m['SlopeBase'] = sb; m['SlopeSens'] = ss
                m['AsymEWMA'] = f"{au}/{ad}"
                results.append(m)
                if i % 12 == 0:
                    print(f"  [{i}/{total}] SB={sb} SS={ss} Asym={au}/{ad} "
                          f"Sharpe={m['Sharpe']:.4f} OOS={m['OOS_Sharpe']:.4f}")

    rdf = pd.DataFrame(results).sort_values('Sharpe', ascending=False)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'opt_phase1_tier2_results.csv')
    rdf.to_csv(out, index=False)

    # Top 10
    print(f"\nTOP 10 by Sharpe:")
    print(f"{'SB':>4} {'SS':>5} {'Asym':>6} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} {'OOS':>7}")
    print("-" * 60)
    for _, r in rdf.head(10).iterrows():
        print(f"{r['SlopeBase']:>4.1f} {r['SlopeSens']:>5.2f} {r['AsymEWMA']:>6} "
              f"{r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% "
              f"{r['Worst5Y']*100:>6.2f}% {r['OOS_Sharpe']:>7.4f}")

    # Walk-Forward for top 3 + baseline
    print(f"\n{'=' * 90}")
    print("WALK-FORWARD for Top 3 + Baseline(0.7/0.3/20/5)")
    print("=" * 90)

    wf_windows = [
        ('2010-01-01', '2015-12-31'),
        ('2015-01-01', '2020-12-31'),
        ('2020-01-01', '2026-12-31'),
    ]

    top3 = rdf.head(3)
    configs = [('Baseline(0.7/0.3/20/5)', 0.7, 0.3, 20, 5)]
    for _, r in top3.iterrows():
        au, ad = r['AsymEWMA'].split('/')
        configs.append((f"({r['SlopeBase']}/{r['SlopeSens']}/{r['AsymEWMA']})",
                        r['SlopeBase'], r['SlopeSens'], int(au), int(ad)))

    for name, sb, ss, au, ad in configs:
        dd = calc_dd_signal(close, 0.82, 0.92)
        asym_vol = calc_asym_ewma_custom(returns, au, ad)
        trend_tv = calc_trend_tv_custom(close)
        vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
        slope = calc_slope_custom(close, base=sb, sens=ss)
        mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
        vix_proxy = calc_vix_proxy(returns)
        vix_ma = vix_proxy.rolling(252).mean()
        vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
        vix_z = (vix_proxy - vix_ma) / vix_std
        vix_mult = (1.0 - 0.25 * vix_z).clip(0.5, 1.15)
        raw = dd * vt_lev * slope * mom * vix_mult
        raw = raw.clip(0, 1.0).fillna(0)
        lev = rebalance_threshold(raw, THRESHOLD)

        wfs = [sharpe_period(close, dates, lev, s, e) for s, e in wf_windows]
        nav, ret = run_bt(close, lev)
        full_sh = calc_metrics(nav, ret, dd, dates)['Sharpe']
        print(f"  {name:<30} Full={full_sh:.4f}  WF=[{wfs[0]:.3f}, {wfs[1]:.3f}, {wfs[2]:.3f}]  Avg={np.nanmean(wfs):.4f}")

    print(f"\nSaved {len(results)} results to {out}")

if __name__ == '__main__':
    main()
