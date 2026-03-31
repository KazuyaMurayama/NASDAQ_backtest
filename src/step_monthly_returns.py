"""
Monthly returns (2021-2026 OOS) + CAGR for 7 strategies
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal
from test_ens2_strategies import strategy_ens2_asym_slope
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from test_dynamic_portfolio import build_dynamic_portfolio, build_static_portfolio

ANNUAL_COST = 0.0086; DELAY = 2; BASE_LEV = 3.0; THRESHOLD = 0.20
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'NASDAQ_extended_to_2026.csv')

def run_bt(close, leverage, delay=DELAY, base_lev=BASE_LEV, cost=ANNUAL_COST):
    returns = close.pct_change()
    lr = returns * base_lev; dc = cost / 252
    dl = leverage.shift(delay)
    sr = dl * (lr - dc); sr = sr.fillna(0)
    return (1 + sr).cumprod(), sr

def calc_asym_ewma(returns, su=30, sd=10):
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[:20].var() if len(returns)>20 else 0.0001
    for i in range(1, len(returns)):
        r = returns.iloc[i]; a = 2/(sd+1) if r<0 else 2/(su+1)
        var.iloc[i] = (1-a)*var.iloc[i-1]+a*(r**2)
    return np.sqrt(var * 252)

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']

    navs = {}

    # BH 1x
    navs['BH 1x'] = close / close.iloc[0]
    # BH 3x
    lev = pd.Series(1.0, index=close.index)
    navs['BH 3x'], _ = run_bt(close, lev, delay=0)
    # DD Only
    dd3 = calc_dd_signal(close, 0.82, 0.92)
    lev3 = rebalance_threshold(dd3, THRESHOLD)
    navs['DD Only'], _ = run_bt(close, lev3)
    # Ens2
    lev4, dd4 = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    lev4 = rebalance_threshold(lev4, THRESHOLD)
    navs['Ens2(Asym+Slope)'], _ = run_bt(close, lev4)
    # A2 Optimized
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
    lev5 = rebalance_threshold(raw, THRESHOLD)
    nav5, ret5 = run_bt(close, lev5)
    navs['A2 Optimized'] = nav5

    # Dyn-Hybrid
    gold = prepare_gold_data(dates); bond = prepare_bond_data(dates)
    signals = {'nav': nav5.values, 'raw_leverage': raw.values,
               'dd_signal': dd.values, 'vix_z': vz.fillna(0).values}
    snav = build_static_portfolio(signals['nav'], gold, bond, 0.35, 0.30, 0.35)
    navs['DH Static (35/30/35)'] = pd.Series(snav, index=dates.index)

    n = len(signals['raw_leverage'])
    wn, wg, wb = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        lv = signals['raw_leverage'][i]; vzv = max(signals['vix_z'][i], 0)
        w = np.clip(0.50 + 0.25*lv - 0.10*vzv, 0.30, 0.90)
        wn[i] = w; wg[i] = (1-w)*0.55; wb[i] = (1-w)*0.45
    dnav = build_dynamic_portfolio(signals['nav'], gold, bond, wn, wg, wb)
    navs['DH Dynamic CAGR25+'] = pd.Series(dnav, index=dates.index)

    order = ['DH Static (35/30/35)', 'DH Dynamic CAGR25+', 'A2 Optimized',
             'Ens2(Asym+Slope)', 'DD Only', 'BH 3x', 'BH 1x']

    # CAGR
    years = len(dates) / 252
    print("CAGR:")
    for name in order:
        nav = navs[name]
        v = nav.values if hasattr(nav, 'values') else nav
        cagr = (v[-1] / v[0]) ** (1/years) - 1
        print(f"  {name}: {cagr*100:.2f}%")

    # Monthly returns 2021-2026
    print("\nMonthly returns (2021-2026):")
    monthly_data = []
    for name in order:
        nav = navs[name]
        ndf = pd.DataFrame({'nav': nav.values if hasattr(nav, 'values') else nav,
                            'date': dates.values})
        ndf['date'] = pd.to_datetime(ndf['date'])
        ndf = ndf[ndf['date'] >= '2021-01-01']
        ndf['ym'] = ndf['date'].dt.to_period('M')
        monthly_nav = ndf.groupby('ym')['nav'].agg(['first', 'last'])
        monthly_nav['return'] = (monthly_nav['last'] / monthly_nav['first'] - 1) * 100
        for ym, row in monthly_nav.iterrows():
            monthly_data.append({'YearMonth': str(ym), 'Strategy': name, 'Return': row['return']})

    mdf = pd.DataFrame(monthly_data)
    pivot = mdf.pivot(index='YearMonth', columns='Strategy', values='Return')[order]
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'monthly_returns_oos.csv')
    pivot.to_csv(out)
    print(f"Saved: {out}")
    print(pivot.to_string())

if __name__ == '__main__':
    main()
