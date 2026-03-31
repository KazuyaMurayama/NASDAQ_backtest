"""
Step 1-2: Calculate Dyn 2x3x yearly/monthly returns and replace DH Static in CSVs
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
from test_dynamic_portfolio import build_dynamic_portfolio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')

def run_bt(close, leverage):
    returns = close.pct_change()
    lr = returns * 3.0; dc = 0.0086 / 252
    dl = leverage.shift(2)
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

    # A2 optimized signals
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
    lev = rebalance_threshold(raw, 0.20)
    nav_a2, _ = run_bt(close, lev)

    # Gold/Bond + leveraged
    gold_1x = prepare_gold_data(dates); bond_1x = prepare_bond_data(dates)
    n = len(dates)
    g2 = np.ones(n); b3 = np.ones(n)
    for i in range(1,n):
        gr = gold_1x[i]/gold_1x[i-1]-1 if gold_1x[i-1]>0 else 0
        g2[i] = g2[i-1]*(1+gr*2-0.005/252)
        br = bond_1x[i]/bond_1x[i-1]-1 if bond_1x[i-1]>0 else 0
        b3[i] = b3[i-1]*(1+br*3-0.0091/252)

    # Dyn 2x3x (B0.55/L0.25/V0.1/G0.6)
    signals_raw = raw.values; signals_vz = vz.fillna(0).values
    wn,wg,wb = np.zeros(n),np.zeros(n),np.zeros(n)
    for i in range(n):
        lv = signals_raw[i]; vzv = max(signals_vz[i],0)
        w = np.clip(0.55+0.25*lv-0.10*vzv, 0.30, 0.90)
        wn[i]=w; wg[i]=(1-w)*0.60; wb[i]=(1-w)*0.40
    dyn_nav = build_dynamic_portfolio(nav_a2.values, g2, b3, wn, wg, wb)
    dyn_nav_s = pd.Series(dyn_nav, index=dates.index)

    # Yearly returns (year-end NAV ratio)
    ndf = pd.DataFrame({'nav': dyn_nav, 'date': dates.values})
    ndf['year'] = pd.to_datetime(ndf['date']).dt.year
    year_end = ndf.groupby('year')['nav'].last()
    yearly = year_end.pct_change() * 100
    yearly.iloc[0] = (year_end.iloc[0] / dyn_nav[0] - 1) * 100

    # Monthly returns (month-end NAV ratio, OOS 2021+)
    ndf['ym'] = pd.to_datetime(ndf['date']).dt.to_period('M')
    month_end = ndf.groupby('ym')['nav'].last()
    month_end_oos = month_end[month_end.index >= '2020-12']
    monthly = month_end_oos.pct_change() * 100
    monthly = monthly[monthly.index >= '2021-01']

    # Update yearly CSV
    yr_csv = os.path.join(BASE_DIR, 'yearly_returns_7strategies.csv')
    yr = pd.read_csv(yr_csv, index_col=0)
    # Replace DH Static column with Dyn 2x3x
    yr = yr.drop(columns=['DH Static (35/30/35)'])
    yr.insert(0, 'DH Dyn 2x3x', yearly.values)
    yr.to_csv(yr_csv)
    print(f"Updated {yr_csv}")

    # Update monthly CSV
    mo_csv = os.path.join(BASE_DIR, 'monthly_returns_oos.csv')
    mo = pd.read_csv(mo_csv, index_col=0)
    mo = mo.drop(columns=['DH Static (35/30/35)'])
    mo.insert(0, 'DH Dyn 2x3x', monthly.values)
    mo.to_csv(mo_csv)
    print(f"Updated {mo_csv}")

    # Verify
    print(f"\nDyn 2x3x yearly sample:")
    for y in [2000, 2008, 2022, 2023, 2024, 2025]:
        print(f"  {y}: {yearly.loc[y]:+.1f}%")

    years_total = len(df)/252
    cagr = (dyn_nav[-1]**(1/years_total))-1
    print(f"\nCAGR: {cagr*100:.2f}%")

if __name__ == '__main__':
    main()
