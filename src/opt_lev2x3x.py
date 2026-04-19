"""
Gold 2x / Bond 3x Dyn-Hybrid Backtest (≤50 patterns)
Static 25 + Dynamic 25 + WF validation
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
GOLD_2X_COST = 0.005; BOND_3X_COST = 0.0091
OOS_DATE = '2021-05-07'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')

def run_bt(close, leverage):
    returns = close.pct_change()
    lr = returns * BASE_LEV; dc = ANNUAL_COST / 252
    dl = leverage.shift(DELAY)
    sr = dl * (lr - dc); sr = sr.fillna(0)
    return (1 + sr).cumprod(), sr

def calc_asym_ewma(returns, su=30, sd=10):
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var() if len(returns)>20 else 0.0001
    for i in range(1, len(returns)):
        r = returns.iloc[i]; a = 2/(sd+1) if r<0 else 2/(su+1)
        var.iloc[i] = (1-a)*var.iloc[i-1]+a*(r**2)
    return np.sqrt(var * 252)

def build_lev_navs(gold_prices, bond_nav_1x):
    """Build Gold 2x and Bond 3x NAV from 1x data."""
    n = len(gold_prices)
    # Gold 2x
    g2_nav = np.ones(n)
    for i in range(1, n):
        gr = gold_prices[i]/gold_prices[i-1]-1 if gold_prices[i-1]>0 else 0
        g2_ret = gr * 2 - GOLD_2X_COST/252
        g2_nav[i] = g2_nav[i-1] * (1 + g2_ret)
    # Bond 3x
    b3_nav = np.ones(n)
    for i in range(1, n):
        br = bond_nav_1x[i]/bond_nav_1x[i-1]-1 if bond_nav_1x[i-1]>0 else 0
        b3_ret = br * 3 - BOND_3X_COST/252
        b3_nav[i] = b3_nav[i-1] * (1 + b3_ret)
    return g2_nav, b3_nav

def port_metrics(nav_arr, dates, name):
    nav = pd.Series(nav_arr, index=dates.index)
    ret = nav.pct_change().fillna(0)
    yrs = len(nav)/252; cagr = (nav.iloc[-1]**(1/yrs))-1
    dd = (nav/nav.cummax()-1).min()
    sh = (ret.mean()*252)/(ret.std()*np.sqrt(252)) if ret.std()>0 else 0
    w5 = np.nan; w10 = np.nan
    if len(nav)>=252*5:
        w5 = ((nav/nav.shift(252*5))**(1/5)-1).min()
    if len(nav)>=252*10:
        w10 = ((nav/nav.shift(252*10))**(1/10)-1).min()
    ndf = pd.DataFrame({'nav':nav.values,'date':dates.values})
    ndf['year'] = pd.to_datetime(ndf['date']).dt.year
    yn = ndf.groupby('year')['nav'].last(); ar = yn.pct_change().dropna()
    wr = (ar>0).mean() if len(ar)>0 else 0
    si = dates[dates>=OOS_DATE].index[0]
    no = nav.iloc[si:]/nav.iloc[si]; ro = ret.iloc[si:]
    oy = len(no)/252; oc = (no.iloc[-1]**(1/oy))-1 if oy>0 else 0
    os_sh = (ro.mean()*252)/(ro.std()*np.sqrt(252)) if ro.std()>0 else 0
    return {'Strategy':name,'CAGR':cagr,'Sharpe':sh,'MaxDD':dd,
            'Worst5Y':w5,'Worst10Y':w10,'WinRate':wr,'OOS_CAGR':oc,'OOS_Sharpe':os_sh}

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
    years = len(df)/252
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()}\n")

    # A2 optimized signals
    print("Building A2 optimized signals...")
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean(); ratio = close/ma150
    ttv = (0.10+(0.20)*(ratio-0.85)/0.30).clip(0.10,0.30).fillna(0.20)
    vt = (ttv/av).clip(0,1.0)
    ma200 = close.rolling(200).mean(); sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0,0.0001)
    z = (sl-sm)/ss; slope = (0.9+0.35*z).clip(0.3,1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0,0.001)
    vz = (vp-vma)/vs; vm = (1.0-0.25*vz).clip(0.5,1.15)
    raw = dd*vt*slope*mom*vm; raw = raw.clip(0,1.0).fillna(0)
    lev_a2 = rebalance_threshold(raw, THRESHOLD)
    nav_a2, _ = run_bt(close, lev_a2)
    signals = {'nav':nav_a2.values,'raw_leverage':raw.values,
               'dd_signal':dd.values,'vix_z':vz.fillna(0).values}

    # Gold/Bond data + leveraged versions
    print("Fetching Gold/Bond + building 2x/3x NAVs...")
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    gold_2x, bond_3x = build_lev_navs(gold_1x, bond_1x)

    g2_cagr = (gold_2x[-1]**(1/years))-1
    b3_cagr = (bond_3x[-1]**(1/years))-1
    print(f"  Gold 2x CAGR: {g2_cagr*100:.1f}%, Bond 3x CAGR: {b3_cagr*100:.1f}%")

    wf_windows = [('2010-01-01','2015-12-31'),('2015-01-01','2020-12-31'),('2020-01-01','2026-12-31')]

    # ===== STATIC GRID (25 patterns) =====
    print(f"\n{'='*90}")
    print("STATIC GRID: NASDAQ(A2) / Gold2x / Bond3x — 25 patterns")
    print("="*90)

    static_results = []
    for wn in range(50, 75, 5):  # 50,55,60,65,70
        for wg in range(5, 30, 5):  # 5,10,15,20,25
            wb = 100 - wn - wg
            if wb < 5 or wb > 40: continue
            nav = build_static_portfolio(signals['nav'], gold_2x, bond_3x, wn/100, wg/100, wb/100)
            m = port_metrics(nav, dates, f"S-{wn}/{wg}/{wb}")
            m['WN']=wn; m['WG']=wg; m['WB']=wb; m['Type']='Static'
            static_results.append(m)

    sdf = pd.DataFrame(static_results).sort_values('Sharpe', ascending=False)
    sdf_27 = sdf[sdf['CAGR']>=0.27].sort_values('Sharpe', ascending=False)
    print(f"\nAll: {len(sdf)}, CAGR≥27%: {len(sdf_27)}")
    print(f"\n{'Alloc':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'W5Y':>8} {'W10Y':>8} {'OOS':>7}")
    print("-"*65)
    for _, r in sdf_27.head(10).iterrows():
        w10s = f"{r['Worst10Y']*100:+.1f}%" if not pd.isna(r['Worst10Y']) else 'N/A'
        print(f"{r['Strategy']:>10} {r['CAGR']*100:>6.2f}% {r['Sharpe']:>7.4f} "
              f"{r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>7.2f}% {w10s:>8} {r['OOS_Sharpe']:>7.4f}")

    # ===== DYNAMIC GRID (25 patterns) =====
    print(f"\n{'='*90}")
    print("DYNAMIC GRID: Gold2x/Bond3x — 25 patterns")
    print("="*90)

    dyn_results = []
    dyn_configs = []
    for base in [0.55, 0.65]:
        for lc in [0.25, 0.35, 0.45]:
            for vc in [0.05, 0.10]:
                for gr in [0.50, 0.60]:
                    dyn_configs.append((base, lc, vc, gr))
    # Add base=0.50/lc=0.45 (high CAGR candidate)
    dyn_configs.append((0.50, 0.45, 0.05, 0.55))

    for base, lc, vc, gr in dyn_configs:
        n = len(signals['raw_leverage'])
        wn,wg,wb = np.zeros(n),np.zeros(n),np.zeros(n)
        for i in range(n):
            lv=signals['raw_leverage'][i]; vzv=max(signals['vix_z'][i],0)
            w = np.clip(base+lc*lv-vc*vzv, 0.30, 0.90)
            wn[i]=w; wg[i]=(1-w)*gr; wb[i]=(1-w)*(1-gr)
        nav = build_dynamic_portfolio(signals['nav'], gold_2x, bond_3x, wn, wg, wb)
        m = port_metrics(nav, dates, f"D-B{base}/L{lc}/V{vc}/G{gr}")
        m['Base']=base; m['LevC']=lc; m['VixC']=vc; m['GoldR']=gr; m['Type']='Dynamic'
        dyn_results.append(m)

    ddf = pd.DataFrame(dyn_results).sort_values('Sharpe', ascending=False)
    ddf_27 = ddf[ddf['CAGR']>=0.27].sort_values('Sharpe', ascending=False)
    print(f"\nAll: {len(ddf)}, CAGR≥27%: {len(ddf_27)}")
    print(f"\n{'Config':>25} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'W5Y':>8} {'W10Y':>8} {'OOS':>7}")
    print("-"*80)
    for _, r in ddf_27.head(10).iterrows():
        w10s = f"{r['Worst10Y']*100:+.1f}%" if not pd.isna(r['Worst10Y']) else 'N/A'
        print(f"{r['Strategy']:>25} {r['CAGR']*100:>6.2f}% {r['Sharpe']:>7.4f} "
              f"{r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>7.2f}% {w10s:>8} {r['OOS_Sharpe']:>7.4f}")

    # ===== WALK-FORWARD =====
    print(f"\n{'='*90}")
    print("WALK-FORWARD: Top candidates vs baselines")
    print("="*90)

    wf_candidates = []

    # Baselines
    # 1x Gold/Bond Dyn-Hybrid CAGR25+ (existing best)
    n = len(signals['raw_leverage'])
    wn0,wg0,wb0 = np.zeros(n),np.zeros(n),np.zeros(n)
    for i in range(n):
        lv=signals['raw_leverage'][i]; vzv=max(signals['vix_z'][i],0)
        w = np.clip(0.50+0.25*lv-0.10*vzv, 0.30, 0.90)
        wn0[i]=w; wg0[i]=(1-w)*0.55; wb0[i]=(1-w)*0.45
    nav_baseline = build_dynamic_portfolio(signals['nav'], gold_1x, bond_1x, wn0, wg0, wb0)
    wf_candidates.append(('Baseline DH-Dyn 1x (B0.5/L0.25/V0.1)', nav_baseline))

    # A2 alone
    wf_candidates.append(('A2 Optimized (single asset)', signals['nav']))

    # Top 3 Static CAGR27+
    for _, r in sdf_27.head(3).iterrows():
        nav = build_static_portfolio(signals['nav'], gold_2x, bond_3x, r['WN']/100, r['WG']/100, r['WB']/100)
        wf_candidates.append((f"Static 2x3x {r['Strategy']}", nav))

    # Top 3 Dynamic CAGR27+
    for _, r in ddf_27.head(3).iterrows():
        base,lc,vc,gr = r['Base'],r['LevC'],r['VixC'],r['GoldR']
        n = len(signals['raw_leverage'])
        wn,wg,wb = np.zeros(n),np.zeros(n),np.zeros(n)
        for i in range(n):
            lv=signals['raw_leverage'][i]; vzv=max(signals['vix_z'][i],0)
            w = np.clip(base+lc*lv-vc*vzv, 0.30, 0.90)
            wn[i]=w; wg[i]=(1-w)*gr; wb[i]=(1-w)*(1-gr)
        nav = build_dynamic_portfolio(signals['nav'], gold_2x, bond_3x, wn, wg, wb)
        wf_candidates.append((f"Dynamic 2x3x {r['Strategy']}", nav))

    print(f"\n{'Strategy':<45} {'Full_Sh':>7} {'CAGR':>7} {'MaxDD':>8} {'WF1':>6} {'WF2':>6} {'WF3':>6} {'WF_Avg':>7}")
    print("-"*100)
    for name, nav in wf_candidates:
        nav_arr = nav if isinstance(nav, np.ndarray) else nav.values if hasattr(nav,'values') else np.array(nav)
        m = port_metrics(nav_arr, dates, name)
        wfs = [sharpe_period(nav_arr, dates, s, e) for s,e in wf_windows]
        wf_avg = np.nanmean(wfs)
        print(f"{name:<45} {m['Sharpe']:>7.4f} {m['CAGR']*100:>6.2f}% {m['MaxDD']*100:>7.2f}% "
              f"{wfs[0]:>6.3f} {wfs[1]:>6.3f} {wfs[2]:>6.3f} {wf_avg:>7.4f}")

    # Save all results
    all_results = pd.concat([sdf, ddf], ignore_index=True)
    out = os.path.join(BASE_DIR, 'lev2x3x_results.csv')
    all_results.to_csv(out, index=False)
    print(f"\nSaved {len(all_results)} results to {out}")

if __name__ == '__main__':
    main()
