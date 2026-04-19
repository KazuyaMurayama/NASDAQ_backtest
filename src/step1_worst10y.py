"""
Step 1: Calculate Worst 10Y for all 7 strategies (original + optimized)
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
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

def run_bt(close, leverage, base_lev=BASE_LEV, cost=ANNUAL_COST, delay=DELAY):
    returns = close.pct_change()
    lev_ret = returns * base_lev; dc = cost / 252
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
    n = 252 * years
    if len(nav) < n: return np.nan
    nav_ago = nav.shift(n)
    rolling = (nav / nav_ago) ** (1/years) - 1
    return rolling.min()

def get_a2_optimized(close, returns):
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean(); ratio = close/ma150
    ttv = (0.10+(0.30-0.10)*(ratio-0.85)/(1.15-0.85)).clip(0.10,0.30).fillna(0.20)
    vt = (ttv / av).clip(0, 1.0)
    ma200 = close.rolling(200).mean(); sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0,0.0001)
    z = (sl-sm)/ss; slope = (0.9+0.35*z).clip(0.3,1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0,0.001)
    vz = (vp-vma)/vs; vm = (1.0-0.25*vz).clip(0.5, 1.15)
    raw = dd*vt*slope*mom*vm; raw = raw.clip(0,1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)
    return lev, dd, raw, vz

def get_a2_original(close, returns):
    from test_ens2_strategies import calc_asym_ewma_vol as aev, calc_slope_multiplier as csm, calc_trend_target_vol as cttv
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = aev(returns, 20, 5); ttv = cttv(close)
    vt = (ttv / av).clip(0, 1.0); slope = csm(close)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0,0.001)
    vz = (vp-vma)/vs; vm = (1.0-0.2*vz).clip(0.5, 1.15)
    raw = dd*vt*slope*mom*vm; raw = raw.clip(0,1.0).fillna(0)
    lev = rebalance_threshold(raw, THRESHOLD)
    return lev, dd

def get_ens2(close, returns):
    from test_ens2_strategies import strategy_ens2_asym_slope
    lev, dd = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    lev = rebalance_threshold(lev, THRESHOLD)
    return lev, dd

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()}\n")

    # Fetch Gold/Bond for portfolio strategies
    print("Fetching Gold/Bond...")
    gold = prepare_gold_data(dates); bond = prepare_bond_data(dates)

    results = []
    OOS = '2021-05-07'

    def add(nav, strat_ret, dd, name):
        m = calc_metrics(nav, strat_ret, dd, dates)
        w5 = worst_ny(nav, 5); w10 = worst_ny(nav, 10)
        si = dates[dates>=OOS].index[0]
        no = nav.iloc[si:]/nav.iloc[si]; ro = strat_ret.iloc[si:]
        oy = len(no)/252
        oc = (no.iloc[-1]**(1/oy))-1 if oy>0 else 0
        os_sh = (ro.mean()*252)/(ro.std()*np.sqrt(252)) if ro.std()>0 else 0
        results.append({'Strategy': name, 'CAGR': m['CAGR'], 'Sharpe': m['Sharpe'],
            'MaxDD': m['MaxDD'], 'Worst5Y': w5, 'Worst10Y': w10,
            'WinRate': m['WinRate'], 'Trades': m['Trades'],
            'OOS_CAGR': oc, 'OOS_Sharpe': os_sh})
        print(f"  {name}: Sharpe={m['Sharpe']:.4f} CAGR={m['CAGR']*100:.2f}% W5Y={w5*100:+.2f}% W10Y={w10*100:+.2f}%")

    def add_port(nav_arr, name):
        nav = pd.Series(nav_arr, index=dates.index)
        ret = nav.pct_change().fillna(0)
        yrs=len(nav)/252; cagr=(nav.iloc[-1]**(1/yrs))-1
        dd_val=(nav/nav.cummax()-1).min()
        sh=(ret.mean()*252)/(ret.std()*np.sqrt(252)) if ret.std()>0 else 0
        w5=worst_ny(nav,5); w10=worst_ny(nav,10)
        ndf=pd.DataFrame({'nav':nav.values,'date':dates.values})
        ndf['year']=pd.to_datetime(ndf['date']).dt.year
        yn=ndf.groupby('year')['nav'].last(); ar=yn.pct_change().dropna()
        wr=(ar>0).mean() if len(ar)>0 else 0
        si=dates[dates>=OOS].index[0]
        no=nav.iloc[si:]/nav.iloc[si]; ro=ret.iloc[si:]
        oy=len(no)/252; oc=(no.iloc[-1]**(1/oy))-1 if oy>0 else 0
        os_sh=(ro.mean()*252)/(ro.std()*np.sqrt(252)) if ro.std()>0 else 0
        results.append({'Strategy':name,'CAGR':cagr,'Sharpe':sh,'MaxDD':dd_val,
            'Worst5Y':w5,'Worst10Y':w10,'WinRate':wr,'Trades':'N/A',
            'OOS_CAGR':oc,'OOS_Sharpe':os_sh})
        print(f"  {name}: Sharpe={sh:.4f} CAGR={cagr*100:.2f}% W5Y={w5*100:+.2f}% W10Y={w10*100:+.2f}%")

    # 1. BH 1x
    print("[1/7] BH 1x")
    nav1 = close/close.iloc[0]; ret1 = close.pct_change().fillna(0)
    dd1 = pd.Series(1.0, index=close.index)
    add(nav1, ret1, dd1, 'Buy & Hold 1x')

    # 2. BH 3x
    print("[2/7] BH 3x")
    lev2 = pd.Series(1.0, index=close.index)
    nav2, ret2 = run_bt(close, lev2, delay=0)
    add(nav2, ret2, dd1, 'Buy & Hold 3x')

    # 3. DD Only
    print("[3/7] DD Only")
    dd3 = calc_dd_signal(close, 0.82, 0.92)
    lev3 = rebalance_threshold(dd3, THRESHOLD)
    nav3, ret3 = run_bt(close, lev3)
    add(nav3, ret3, dd3, 'DD(-18/92) Only')

    # 4. Ens2(Asym+Slope) - original params with delay=2/cost=0.86%
    print("[4/7] Ens2(Asym+Slope)")
    lev4, dd4 = get_ens2(close, returns)
    nav4, ret4 = run_bt(close, lev4)
    add(nav4, ret4, dd4, 'Ens2(Asym+Slope)')

    # 5. A2 optimized
    print("[5/7] A2 Optimized")
    lev5, dd5, raw5, vz5 = get_a2_optimized(close, returns)
    nav5, ret5 = run_bt(close, lev5)
    add(nav5, ret5, dd5, 'A2 Optimized')

    # 6. Dyn-Hybrid Static optimized (35/30/35)
    print("[6/7] Dyn-Hybrid Static (35/30/35)")
    signals = {'nav': nav5.values, 'ret': ret5.values,
               'raw_leverage': raw5.values, 'dd_signal': dd5.values,
               'vix_z': vz5.fillna(0).values}
    snav = build_static_portfolio(signals['nav'], gold, bond, 0.35, 0.30, 0.35)
    add_port(snav, 'Dyn-Hybrid Static (35/30/35) *')

    # 7. Dyn-Hybrid Dynamic optimized (0.40/0.15/0.05)
    print("[7/7] Dyn-Hybrid Dynamic (0.40/0.15/0.05)")
    n = len(signals['raw_leverage'])
    wn,wg,wb = np.zeros(n),np.zeros(n),np.zeros(n)
    for i in range(n):
        lv=signals['raw_leverage'][i]; vz=max(signals['vix_z'][i],0)
        w = np.clip(0.40+0.15*lv-0.05*vz, 0.30, 0.90)
        wn[i]=w; wg[i]=(1-w)*0.55; wb[i]=(1-w)*0.45
    dnav = build_dynamic_portfolio(signals['nav'], gold, bond, wn, wg, wb)
    add_port(dnav, 'Dyn-Hybrid Dynamic (0.40/0.15/0.05) *')

    # Save
    rdf = pd.DataFrame(results)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'step1_worst10y_results.csv')
    rdf.to_csv(out, index=False)

    print(f"\n{'='*120}")
    print(f"{'#':<3} {'Strategy':<38} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'W5Y':>8} {'W10Y':>8} {'OOS_Sh':>7}")
    print("-"*100)
    for i,r in enumerate(results,1):
        w10s = f"{r['Worst10Y']*100:+.2f}%" if not pd.isna(r['Worst10Y']) else 'N/A'
        print(f"{i:<3} {r['Strategy']:<38} {r['CAGR']*100:>6.2f}% {r['Sharpe']:>7.4f} "
              f"{r['MaxDD']*100:>7.2f}% {r['Worst5Y']*100:>7.2f}% {w10s:>8} {r['OOS_Sharpe']:>7.4f}")
    print(f"\nSaved to {out}")

if __name__ == '__main__':
    main()
