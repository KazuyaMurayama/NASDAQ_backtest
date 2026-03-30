"""
Step 2+3: Dyn-Hybrid CAGR 25%+ grid search
+ Step 4: Walk-Forward for top candidates
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from test_dynamic_portfolio import build_dynamic_portfolio, build_static_portfolio

ANNUAL_COST = 0.0086; DELAY = 2; BASE_LEV = 3.0; THRESHOLD = 0.20
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

def run_bt(close, leverage, delay=DELAY):
    returns = close.pct_change()
    lev_ret = returns * BASE_LEV; dc = ANNUAL_COST / 252
    delayed = leverage.shift(delay)
    sr = delayed * (lev_ret - dc); sr = sr.fillna(0)
    return (1 + sr).cumprod(), sr

def calc_asym_ewma(returns, su=30, sd=10):
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var() if len(returns)>20 else 0.0001
    for i in range(1, len(returns)):
        r = returns.iloc[i]; a = 2/(sd+1) if r<0 else 2/(su+1)
        var.iloc[i] = (1-a)*var.iloc[i-1]+a*(r**2)
    return np.sqrt(var * 252)

def worst_ny(nav, years):
    n = 252*years
    if len(nav)<n: return np.nan
    return ((nav/nav.shift(n))**(1/years)-1).min()

def port_metrics(nav_arr, dates, name):
    nav = pd.Series(nav_arr, index=dates.index)
    ret = nav.pct_change().fillna(0)
    yrs = len(nav)/252; cagr = (nav.iloc[-1]**(1/yrs))-1
    dd = (nav/nav.cummax()-1).min()
    sh = (ret.mean()*252)/(ret.std()*np.sqrt(252)) if ret.std()>0 else 0
    w5 = worst_ny(nav,5); w10 = worst_ny(nav,10)
    ndf = pd.DataFrame({'nav':nav.values,'date':dates.values})
    ndf['year'] = pd.to_datetime(ndf['date']).dt.year
    yn = ndf.groupby('year')['nav'].last(); ar = yn.pct_change().dropna()
    wr = (ar>0).mean() if len(ar)>0 else 0
    si = dates[dates>='2021-05-07'].index[0]
    no = nav.iloc[si:]/nav.iloc[si]; ro = ret.iloc[si:]
    oy = len(no)/252; oc = (no.iloc[-1]**(1/oy))-1 if oy>0 else 0
    os_sh = (ro.mean()*252)/(ro.std()*np.sqrt(252)) if ro.std()>0 else 0
    return {'Strategy':name,'CAGR':cagr,'Sharpe':sh,'MaxDD':dd,
            'Worst5Y':w5,'Worst10Y':w10,'WinRate':wr,
            'OOS_CAGR':oc,'OOS_Sharpe':os_sh}

def sharpe_period(nav_arr, dates, start, end):
    nav = pd.Series(nav_arr, index=dates.index)
    mask = (dates>=start)&(dates<end)
    if mask.sum()<100: return np.nan
    idx = dates[mask].index; n = nav.iloc[idx[0]:idx[-1]+1]
    n = n/n.iloc[0]; r = n.pct_change().fillna(0)
    if r.std()==0: return 0
    return (r.mean()*252)/(r.std()*np.sqrt(252))

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()}\n")

    print("Preparing A2 optimized signals + Gold/Bond...")
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean(); ratio = close/ma150
    ttv = (0.10+(0.30-0.10)*(ratio-0.85)/(1.15-0.85)).clip(0.10,0.30).fillna(0.20)
    vt = (ttv/av).clip(0,1.0)
    ma200 = close.rolling(200).mean(); sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0,0.0001)
    z = (sl-sm)/ss; slope = (0.9+0.35*z).clip(0.3,1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0,0.001)
    vz = (vp-vma)/vs; vm = (1.0-0.25*vz).clip(0.5,1.15)
    raw = dd*vt*slope*mom*vm; raw = raw.clip(0,1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)
    nav_a2, ret_a2 = run_bt(close, lev)
    signals = {'nav':nav_a2.values,'ret':ret_a2.values,
               'raw_leverage':raw.values,'dd_signal':dd.values,'vix_z':vz.fillna(0).values}

    gold = prepare_gold_data(dates); bond = prepare_bond_data(dates)
    print("  Done.\n")

    # ===== STEP 2: Static CAGR25%+ =====
    print("=" * 90)
    print("STEP 2: Dyn-Hybrid Static — CAGR 25%+ grid (WN 65-85%)")
    print("=" * 90)

    static_results = []
    for wn in range(65, 90, 5):
        for wg in range(5, min(100-wn-4, 25), 5):
            wb = 100 - wn - wg
            if wb < 5: continue
            nav = build_static_portfolio(signals['nav'], gold, bond, wn/100, wg/100, wb/100)
            m = port_metrics(nav, dates, f"S-{wn}/{wg}/{wb}")
            m['WN']=wn; m['WG']=wg; m['WB']=wb
            static_results.append(m)

    sdf = pd.DataFrame(static_results)
    sdf_25 = sdf[sdf['CAGR']>=0.25].sort_values('Sharpe', ascending=False)
    out_s = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'step2_static_cagr25_results.csv')
    sdf.to_csv(out_s, index=False)

    print(f"\nCAGR>=25% candidates ({len(sdf_25)} found):")
    print(f"{'Alloc':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'W5Y':>8} {'W10Y':>8} {'OOS':>7}")
    print("-" * 65)
    for _, r in sdf_25.head(10).iterrows():
        print(f"{r['Strategy']:>10} {r['CAGR']*100:>6.2f}% {r['Sharpe']:>7.4f} "
              f"{r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>7.2f}% {r['Worst10Y']*100:>7.2f}% {r['OOS_Sharpe']:>7.4f}")

    # ===== STEP 3: Dynamic CAGR25%+ =====
    print(f"\n{'=' * 90}")
    print("STEP 3: Dyn-Hybrid Dynamic — CAGR 25%+ grid")
    print("=" * 90)

    dyn_results = []
    for base in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for lc in [0.25, 0.30, 0.35, 0.40, 0.45]:
            for vc in [0.05, 0.10]:
                n = len(signals['raw_leverage'])
                wn,wg,wb = np.zeros(n),np.zeros(n),np.zeros(n)
                for i in range(n):
                    lv=signals['raw_leverage'][i]; vzv=max(signals['vix_z'][i],0)
                    w = np.clip(base+lc*lv-vc*vzv, 0.30, 0.90)
                    wn[i]=w; wg[i]=(1-w)*0.55; wb[i]=(1-w)*0.45
                nav = build_dynamic_portfolio(signals['nav'], gold, bond, wn, wg, wb)
                m = port_metrics(nav, dates, f"D-B{base}/L{lc}/V{vc}")
                m['Base']=base; m['LevC']=lc; m['VixC']=vc
                dyn_results.append(m)

    ddf = pd.DataFrame(dyn_results)
    ddf_25 = ddf[ddf['CAGR']>=0.25].sort_values('Sharpe', ascending=False)
    out_d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'step3_dynamic_cagr25_results.csv')
    ddf.to_csv(out_d, index=False)

    print(f"\nCAGR>=25% candidates ({len(ddf_25)} found):")
    print(f"{'Config':>20} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'W5Y':>8} {'W10Y':>8} {'OOS':>7}")
    print("-" * 75)
    for _, r in ddf_25.head(10).iterrows():
        print(f"{r['Strategy']:>20} {r['CAGR']*100:>6.2f}% {r['Sharpe']:>7.4f} "
              f"{r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>7.2f}% {r['Worst10Y']*100:>7.2f}% {r['OOS_Sharpe']:>7.4f}")

    # ===== STEP 4: Walk-Forward =====
    print(f"\n{'=' * 90}")
    print("STEP 4: Walk-Forward Validation (CAGR25%+ top candidates)")
    print("=" * 90)

    wf_windows = [('2010-01-01','2015-12-31'),('2015-01-01','2020-12-31'),('2020-01-01','2026-12-31')]

    # Static top 3 CAGR25%+
    print("\n--- Static CAGR25%+ ---")
    for _, r in sdf_25.head(3).iterrows():
        wn_f=r['WN']/100; wg_f=r['WG']/100; wb_f=r['WB']/100
        nav = build_static_portfolio(signals['nav'], gold, bond, wn_f, wg_f, wb_f)
        wfs = [sharpe_period(nav, dates, s, e) for s,e in wf_windows]
        print(f"  {r['Strategy']:<15} Full={r['Sharpe']:.4f} WF=[{wfs[0]:.3f},{wfs[1]:.3f},{wfs[2]:.3f}] Avg={np.nanmean(wfs):.4f}")

    # Dynamic top 5 CAGR25%+
    print("\n--- Dynamic CAGR25%+ ---")
    for _, r in ddf_25.head(5).iterrows():
        base=r['Base']; lc=r['LevC']; vc=r['VixC']
        n = len(signals['raw_leverage'])
        wn,wg,wb = np.zeros(n),np.zeros(n),np.zeros(n)
        for i in range(n):
            lv=signals['raw_leverage'][i]; vzv=max(signals['vix_z'][i],0)
            w = np.clip(base+lc*lv-vc*vzv, 0.30, 0.90)
            wn[i]=w; wg[i]=(1-w)*0.55; wb[i]=(1-w)*0.45
        nav = build_dynamic_portfolio(signals['nav'], gold, bond, wn, wg, wb)
        wfs = [sharpe_period(nav, dates, s, e) for s,e in wf_windows]
        print(f"  {r['Strategy']:<20} Full={r['Sharpe']:.4f} WF=[{wfs[0]:.3f},{wfs[1]:.3f},{wfs[2]:.3f}] Avg={np.nanmean(wfs):.4f}")

    print(f"\nSaved: {out_s}, {out_d}")

if __name__ == '__main__':
    main()
