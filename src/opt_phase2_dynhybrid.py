"""
Phase 2: Dyn-Hybrid Dynamic allocation optimization
+ Phase 3: Dyn-Hybrid Static grid search
Uses optimized A2 from Phase 1
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
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from test_dynamic_portfolio import build_dynamic_portfolio, build_static_portfolio

ANNUAL_COST = 0.0086; DELAY = 2; BASE_LEV = 3.0; THRESHOLD = 0.20
OOS_DATE = '2021-05-07'
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

def get_optimized_a2_signals(close, dates):
    """Optimized A2 from Phase 1: TV10-30, VIX0.25, MD60/180, SB0.9, SS0.35, Asym30/10"""
    returns = close.pct_change()
    dd = calc_dd_signal(close, 0.82, 0.92)
    asym_vol = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean(); ratio = close/ma150
    trend_tv = (0.10 + (0.30-0.10)*(ratio-0.85)/(1.15-0.85)).clip(0.10, 0.30).fillna(0.20)
    vt_lev = (trend_tv / asym_vol).clip(0, 1.0)
    ma200 = close.rolling(200).mean(); slope = ma200.pct_change()
    sm = slope.rolling(60).mean(); ss = slope.rolling(60).std().replace(0, 0.0001)
    z = (slope - sm)/ss
    slope_mult = (0.9 + 0.35*z).clip(0.3, 1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vix_proxy = calc_vix_proxy(returns)
    vix_ma = vix_proxy.rolling(252).mean()
    vix_std = vix_proxy.rolling(252).std().replace(0, 0.001)
    vix_z = (vix_proxy - vix_ma) / vix_std
    vix_mult = (1.0 - 0.25 * vix_z).clip(0.5, 1.15)
    raw = dd * vt_lev * slope_mult * mom * vix_mult
    raw = raw.clip(0, 1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)
    nav, strat_ret = run_bt(close, lev)
    return {
        'nav': nav.values, 'ret': strat_ret.values,
        'raw_leverage': raw.values, 'dd_signal': dd.values,
        'vix_z': vix_z.fillna(0).values,
    }

def calc_port_metrics(nav_array, dates, name):
    nav = pd.Series(nav_array, index=dates.index)
    ret = nav.pct_change().fillna(0)
    yrs = len(nav)/252
    cagr = (nav.iloc[-1]**(1/yrs))-1
    dd = (nav/nav.cummax()-1).min()
    sh = (ret.mean()*252)/(ret.std()*np.sqrt(252)) if ret.std()>0 else 0
    # Worst5Y
    if len(nav)>=252*5:
        n5 = nav.shift(252*5); r5 = (nav/n5)**(1/5)-1; w5y = r5.min()
    else: w5y = np.nan
    # WinRate
    nav_df = pd.DataFrame({'nav': nav.values, 'date': dates.values})
    nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
    yn = nav_df.groupby('year')['nav'].last()
    ar = yn.pct_change().dropna()
    wr = (ar>0).mean() if len(ar)>0 else 0
    # OOS
    si = dates[dates >= OOS_DATE].index[0]
    no = nav.iloc[si:]/nav.iloc[si]; ro = ret.iloc[si:]
    oy = len(no)/252
    oc = (no.iloc[-1]**(1/oy))-1 if oy>0 else 0
    os_sh = (ro.mean()*252)/(ro.std()*np.sqrt(252)) if ro.std()>0 else 0
    return {'Strategy': name, 'Sharpe': sh, 'CAGR': cagr, 'MaxDD': dd,
            'Worst5Y': w5y, 'WinRate': wr, 'OOS_CAGR': oc, 'OOS_Sharpe': os_sh}

def alloc_dyn_custom(signals, base_w=0.50, lev_coeff=0.30, vix_coeff=0.10,
                     w_min=0.30, w_max=0.90, gold_ratio=0.55):
    n = len(signals['raw_leverage'])
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lev = signals['raw_leverage'][i]
        vz = max(signals['vix_z'][i], 0)
        w = base_w + lev_coeff * lev - vix_coeff * vz
        w = np.clip(w, w_min, w_max)
        wn[i] = w
        wg[i] = (1-w) * gold_ratio
        wb[i] = (1-w) * (1-gold_ratio)
    return wn, wg, wb

def sharpe_port_period(nav_array, dates, start, end):
    nav = pd.Series(nav_array, index=dates.index)
    mask = (dates >= start) & (dates < end)
    if mask.sum() < 100: return np.nan
    idx = dates[mask].index
    n = nav.iloc[idx[0]:idx[-1]+1]
    n = n / n.iloc[0]
    r = n.pct_change().fillna(0)
    if r.std() == 0: return 0
    return (r.mean()*252)/(r.std()*np.sqrt(252))

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()}\n")

    print("Getting optimized A2 signals...")
    signals = get_optimized_a2_signals(close, dates)
    print("Fetching Gold/Bond data...")
    gold = prepare_gold_data(dates)
    bond = prepare_bond_data(dates)
    print(f"  Gold: {gold[0]:.0f} -> {gold[-1]:.0f}, Bond NAV: {bond[0]:.2f} -> {bond[-1]:.2f}\n")

    wf_windows = [('2010-01-01','2015-12-31'), ('2015-01-01','2020-12-31'), ('2020-01-01','2026-12-31')]

    # ===== Phase 2: Dyn-Hybrid Dynamic =====
    print("=" * 90)
    print("PHASE 2: Dyn-Hybrid Dynamic Allocation Grid (64 patterns)")
    print("=" * 90)

    bases = [0.40, 0.50, 0.60, 0.70]
    lev_cs = [0.15, 0.25, 0.35, 0.45]
    vix_cs = [0.05, 0.10, 0.15, 0.20]

    results_dyn = []
    i = 0; total = len(bases)*len(lev_cs)*len(vix_cs)
    for bw in bases:
        for lc in lev_cs:
            for vc in vix_cs:
                i += 1
                wn, wg, wb = alloc_dyn_custom(signals, bw, lc, vc)
                nav = build_dynamic_portfolio(signals['nav'], gold, bond, wn, wg, wb)
                m = calc_port_metrics(nav, dates, f"B{bw}/L{lc}/V{vc}")
                m['Base'] = bw; m['LevC'] = lc; m['VixC'] = vc
                results_dyn.append(m)
                if i % 16 == 0:
                    print(f"  [{i}/{total}] B={bw} L={lc} V={vc} Sh={m['Sharpe']:.4f} OOS={m['OOS_Sharpe']:.4f}")

    rdf_dyn = pd.DataFrame(results_dyn).sort_values('Sharpe', ascending=False)
    out_dyn = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'opt_phase2_results.csv')
    rdf_dyn.to_csv(out_dyn, index=False)

    print(f"\nTOP 10:")
    print(f"{'B':>4} {'LC':>5} {'VC':>5} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} {'OOS':>7}")
    print("-" * 60)
    for _, r in rdf_dyn.head(10).iterrows():
        print(f"{r['Base']:>4.2f} {r['LevC']:>5.2f} {r['VixC']:>5.2f} "
              f"{r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% {r['MaxDD']*100:>7.2f}% "
              f"{r['Worst5Y']*100:>6.2f}% {r['OOS_Sharpe']:>7.4f}")

    # WF for top 3 + baseline
    print(f"\nWALK-FORWARD:")
    wf_configs = [('Baseline(0.50/0.30/0.10)', 0.50, 0.30, 0.10)]
    for _, r in rdf_dyn.head(3).iterrows():
        wf_configs.append((f"({r['Base']:.2f}/{r['LevC']:.2f}/{r['VixC']:.2f})",
                           r['Base'], r['LevC'], r['VixC']))
    for name, bw, lc, vc in wf_configs:
        wn, wg, wb = alloc_dyn_custom(signals, bw, lc, vc)
        nav = build_dynamic_portfolio(signals['nav'], gold, bond, wn, wg, wb)
        wfs = [sharpe_port_period(nav, dates, s, e) for s, e in wf_windows]
        m = calc_port_metrics(nav, dates, name)
        print(f"  {name:<30} Full={m['Sharpe']:.4f} WF=[{wfs[0]:.3f},{wfs[1]:.3f},{wfs[2]:.3f}] Avg={np.nanmean(wfs):.4f}")

    # ===== Phase 3: Dyn-Hybrid Static =====
    print(f"\n{'=' * 90}")
    print("PHASE 3: Dyn-Hybrid Static Grid (5% increments)")
    print("=" * 90)

    results_static = []
    for wn_pct in range(35, 70, 5):
        for wg_pct in range(10, 40, 5):
            wb_pct = 100 - wn_pct - wg_pct
            if wb_pct < 5 or wb_pct > 40: continue
            wn_f = wn_pct/100; wg_f = wg_pct/100; wb_f = wb_pct/100
            nav = build_static_portfolio(signals['nav'], gold, bond, wn_f, wg_f, wb_f)
            m = calc_port_metrics(nav, dates, f"{wn_pct}/{wg_pct}/{wb_pct}")
            m['WN'] = wn_pct; m['WG'] = wg_pct; m['WB'] = wb_pct
            results_static.append(m)

    rdf_static = pd.DataFrame(results_static).sort_values('Sharpe', ascending=False)
    out_static = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'opt_phase3_results.csv')
    rdf_static.to_csv(out_static, index=False)

    print(f"\nTOP 10:")
    print(f"{'Alloc':>10} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'W5Y':>7} {'OOS':>7}")
    print("-" * 55)
    for _, r in rdf_static.head(10).iterrows():
        print(f"{r['Strategy']:>10} {r['Sharpe']:>7.4f} {r['CAGR']*100:>6.2f}% "
              f"{r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>6.2f}% {r['OOS_Sharpe']:>7.4f}")

    # WF for top 3 + baseline
    print(f"\nWALK-FORWARD:")
    wf_cfgs = [('Baseline(50/25/25)', 0.50, 0.25, 0.25)]
    for _, r in rdf_static.head(3).iterrows():
        wf_cfgs.append((f"({int(r['WN'])}/{int(r['WG'])}/{int(r['WB'])})",
                        r['WN']/100, r['WG']/100, r['WB']/100))
    for name, wn_f, wg_f, wb_f in wf_cfgs:
        nav = build_static_portfolio(signals['nav'], gold, bond, wn_f, wg_f, wb_f)
        wfs = [sharpe_port_period(nav, dates, s, e) for s, e in wf_windows]
        m = calc_port_metrics(nav, dates, name)
        print(f"  {name:<20} Full={m['Sharpe']:.4f} WF=[{wfs[0]:.3f},{wfs[1]:.3f},{wfs[2]:.3f}] Avg={np.nanmean(wfs):.4f}")

    print(f"\nSaved: {out_dyn}, {out_static}")

if __name__ == '__main__':
    main()
