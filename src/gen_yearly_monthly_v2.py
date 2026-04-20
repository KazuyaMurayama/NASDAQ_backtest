"""
Yearly/Monthly Returns Report v2 (2026-04-20)
=============================================
Regenerates YEARLY_RETURNS_REPORT with:
  - Extended data 1974-01-02 to 2026-03-26 (NASDAQ_extended_to_2026.csv)
  - Proper CAGR computed from data (NOT hardcoded as in gen_yearly_md.py)
  - BOTH Approach A (target for live adoption) and Approach B (current GAS) for DH Dyn 2x3x
  - Independent double-check: CAGR(NAV endpoints) == (1+yearly).prod()^(1/n) - 1

Strategies (8):
  1. DH Dyn 2x3x [A]  - sleeve-independent, physically realizable for live trading
  2. DH Dyn 2x3x [B]  - current GAS (for comparison)
  3. DH Dyn CAGR25+   - 1x Gold/Bond variant
  4. A2 Optimized     - single-asset A2 signal
  5. Ens2(Asym+Slope) - legacy best
  6. DD Only          - baseline with DD control
  7. BH 3x            - 3x leveraged buy-and-hold
  8. BH 1x            - 1x buy-and-hold
"""
import sys, os, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np
from backtest_engine import load_data, calc_dd_signal
from test_ens2_strategies import strategy_ens2_asym_slope
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from test_dynamic_portfolio import build_dynamic_portfolio
from opt_lev2x3x import build_lev_navs, calc_asym_ewma

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
ANNUAL_COST = 0.0086; DELAY = 2; BASE_LEV = 3.0; THRESHOLD = 0.20
OOS_SPLIT = '2021-05-07'

def run_bt(close, leverage, delay=DELAY, base_lev=BASE_LEV, cost=ANNUAL_COST):
    r = close.pct_change(); dc = cost/252
    dl = leverage.shift(delay)
    sr = (dl * (r*base_lev - dc)).fillna(0)
    return (1+sr).cumprod()

def yearly_from_nav(nav, dates):
    ndf = pd.DataFrame({'nav': nav.values if hasattr(nav,'values') else nav, 'date': dates.values})
    ndf['year'] = pd.to_datetime(ndf['date']).dt.year
    yn = ndf.groupby('year')['nav'].last()
    yr = yn.pct_change() * 100
    first = ndf['nav'].iloc[0]
    yr.iloc[0] = (yn.iloc[0] / first - 1) * 100
    return yr

def monthly_from_nav(nav, dates, start='2021-01'):
    ndf = pd.DataFrame({'nav': nav.values if hasattr(nav,'values') else nav, 'date': dates.values})
    ndf['ym'] = pd.to_datetime(ndf['date']).dt.to_period('M')
    me = ndf.groupby('ym')['nav'].last()
    me = me[me.index >= '2020-12']
    mo = me.pct_change() * 100
    return mo[mo.index >= start]

def period_metrics(nav, dates, start, end):
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if mask.sum() < 20: return {}
    idx = dates[mask].index
    n = pd.Series(nav[idx[0]:idx[-1]+1] if isinstance(nav, np.ndarray) else nav.iloc[idx[0]:idx[-1]+1].values)
    n = n / n.iloc[0]
    r = n.pct_change().fillna(0)
    yrs = len(n)/252
    cagr = n.iloc[-1]**(1/yrs) - 1 if yrs>0 and n.iloc[-1]>0 else np.nan
    sh = (r.mean()*252)/(r.std()*np.sqrt(252)) if r.std()>0 else 0
    maxdd = (n/n.cummax()-1).min()
    w5 = ((n/n.shift(252*5))**(1/5)-1).min() if len(n)>=252*5 else np.nan
    return {'CAGR':cagr, 'Sharpe':sh, 'MaxDD':maxdd, 'Worst5Y':w5}

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    n = len(df)
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({n} days, {n/252:.1f} years)\n")
    
    navs = {}
    
    # 1. BH 1x
    navs['BH 1x'] = (close / close.iloc[0]).values
    
    # 2. BH 3x
    lev1 = pd.Series(1.0, index=close.index)
    navs['BH 3x'] = run_bt(close, lev1, delay=0).values
    
    # 3. DD Only
    dd_sig = calc_dd_signal(close, 0.82, 0.92)
    lev_dd = rebalance_threshold(dd_sig, THRESHOLD)
    navs['DD Only'] = run_bt(close, lev_dd).values
    
    # 4. Ens2(Asym+Slope)
    lev_ens, _ = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    lev_ens = rebalance_threshold(lev_ens, THRESHOLD)
    navs['Ens2(Asym+Slope)'] = run_bt(close, lev_ens).values
    
    # A2 Optimized signal construction (current GAS production)
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean(); ratio = close/ma150
    ttv = (0.10+0.20*(ratio-0.85)/0.30).clip(0.10,0.30).fillna(0.20)
    vt = (ttv/av).clip(0,1.0)
    ma200 = close.rolling(200).mean(); sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0,0.0001)
    z = (sl-sm)/ss; slope = (0.9+0.35*z).clip(0.3,1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean(); vs = vp.rolling(252).std().replace(0,0.001)
    vz_ser = (vp-vma)/vs; vm = (1.0-0.25*vz_ser).clip(0.5,1.15)
    raw = (dd*vt*slope*mom*vm).clip(0,1.0).fillna(0)
    lev_a2 = rebalance_threshold(raw, THRESHOLD)
    nav_a2 = run_bt(close, lev_a2)
    navs['A2 Optimized'] = nav_a2.values
    
    # Gold/Bond
    gold_1x = prepare_gold_data(dates); bond_1x = prepare_bond_data(dates)
    gold_2x, bond_3x = build_lev_navs(gold_1x, bond_1x)
    vz_v = vz_ser.fillna(0).values
    raw_v = raw.values
    
    # 5. DH Dyn CAGR25+ (1x Gold/Bond, B=0.50, L=0.25, V=0.10, Gold/Bond=55/45)
    wn,wg,wb = np.zeros(n),np.zeros(n),np.zeros(n)
    for i in range(n):
        lv = raw_v[i]; vzv = max(vz_v[i],0)
        w = np.clip(0.50+0.25*lv-0.10*vzv, 0.30, 0.90)
        wn[i]=w; wg[i]=(1-w)*0.55; wb[i]=(1-w)*0.45
    navs['DH Dyn CAGR25+'] = build_dynamic_portfolio(nav_a2.values, gold_1x, bond_1x, wn, wg, wb)
    
    # 6. DH Dyn 2x3x [Approach A] - physically realizable via sleeve-independent allocation
    # NAV is built as: daily_ret = wn*lev*(r_naq*3 - dc) + wg*r_g2 + wb*r_b3
    # with lev_out and w applied per GAS rebalance rule
    wn_A = np.zeros(n); wg_A = np.zeros(n); wb_A = np.zeros(n); lev_A = np.zeros(n)
    cur_lev = raw_v[0]
    cur_wn = np.clip(0.55+0.25*cur_lev-0.10*max(vz_v[0],0), 0.30, 0.90)
    cur_wg = (1-cur_wn)*0.5; cur_wb = (1-cur_wn)*0.5
    lev_A[0]=cur_lev; wn_A[0]=cur_wn; wg_A[0]=cur_wg; wb_A[0]=cur_wb
    for i in range(1, n):
        t = raw_v[i]
        dd_to_0 = (t==0 and cur_lev>0); dd_from_0 = (cur_lev==0 and t>0)
        if dd_to_0 or dd_from_0 or abs(t-cur_lev) > THRESHOLD:
            cur_lev = t
            cur_wn = np.clip(0.55+0.25*cur_lev-0.10*max(vz_v[i],0), 0.30, 0.90)
            cur_wg = (1-cur_wn)*0.5; cur_wb = (1-cur_wn)*0.5
        lev_A[i]=cur_lev; wn_A[i]=cur_wn; wg_A[i]=cur_wg; wb_A[i]=cur_wb
    
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x).pct_change().fillna(0).values
    dc = ANNUAL_COST/252
    lev_s = pd.Series(lev_A, index=dates.index).shift(DELAY).fillna(0).values
    wn_s = pd.Series(wn_A, index=dates.index).shift(DELAY).fillna(0).values
    wg_s = pd.Series(wg_A, index=dates.index).shift(DELAY).fillna(0).values
    wb_s = pd.Series(wb_A, index=dates.index).shift(DELAY).fillna(0).values
    daily_A = wn_s*lev_s*(r_nas*BASE_LEV - dc) + wg_s*r_g2 + wb_s*r_b3
    navs['DH Dyn 2x3x [A]'] = (1+pd.Series(daily_A, index=dates.index)).cumprod().values
    
    # 7. DH Dyn 2x3x [Approach B] - current GAS: actual = lev × (wn, wg, wb)
    daily_B = lev_s * (wn_s*(r_nas*BASE_LEV - dc) + wg_s*r_g2 + wb_s*r_b3)
    navs['DH Dyn 2x3x [B]'] = (1+pd.Series(daily_B, index=dates.index)).cumprod().values
    
    # ===== Build yearly/monthly tables =====
    order = ['DH Dyn 2x3x [A]', 'DH Dyn 2x3x [B]', 'DH Dyn CAGR25+', 'A2 Optimized',
             'Ens2(Asym+Slope)', 'DD Only', 'BH 3x', 'BH 1x']
    
    yr_df = pd.DataFrame({name: yearly_from_nav(navs[name], dates) for name in order})
    mo_df = pd.DataFrame({name: monthly_from_nav(navs[name], dates) for name in order})
    
    # ===== DOUBLE-CHECK =====
    print("=" * 80)
    print("DOUBLE CHECK: CAGR from NAV endpoints vs from compounded yearly returns")
    print("=" * 80)
    check_ok = True
    for name in order:
        nav = navs[name]
        yrs = n/252
        cagr_nav = nav[-1]**(1/yrs) - 1
        yearly = yr_df[name] / 100
        cagr_yearly = ((1+yearly).prod())**(1/yrs) - 1
        diff = abs(cagr_nav - cagr_yearly)
        status = "OK" if diff < 0.001 else f"FAIL (diff={diff*100:.3f}pp)"
        if diff >= 0.001: check_ok = False
        print(f"  {name:<22} NAV:{cagr_nav*100:+6.2f}%  Yearly:{cagr_yearly*100:+6.2f}%  {status}")
    print()
    
    # ===== Period metrics =====
    full_metrics = {}
    is_metrics = {}
    oos_metrics = {}
    for name in order:
        full_metrics[name] = period_metrics(navs[name], dates, '1974-01-02', '2026-03-28')
        is_metrics[name] = period_metrics(navs[name], dates, '1974-01-02', OOS_SPLIT)
        oos_metrics[name] = period_metrics(navs[name], dates, '2021-05-08', '2026-03-28')
    
    print("=" * 80)
    print("Full-period, IS, OOS metrics")
    print("=" * 80)
    print(f"{'Strategy':<22} {'Full CAGR':>10} {'IS CAGR':>9} {'OOS CAGR':>9} {'Sharpe':>7}")
    for name in order:
        fm = full_metrics[name]; im = is_metrics[name]; om = oos_metrics[name]
        print(f"  {name:<20} {fm['CAGR']*100:>+8.2f}% {im['CAGR']*100:>+7.2f}% {om['CAGR']*100:>+7.2f}% {fm['Sharpe']:>7.3f}")
    
    # Save CSVs
    yr_csv = os.path.join(BASE, 'yearly_returns_8strategies_v2.csv')
    mo_csv = os.path.join(BASE, 'monthly_returns_oos_v2.csv')
    yr_df.to_csv(yr_csv)
    mo_df.to_csv(mo_csv)
    print(f"\nSaved: {yr_csv}")
    print(f"Saved: {mo_csv}")
    
    return {'order': order, 'navs': navs, 'dates': dates, 'yr_df': yr_df, 'mo_df': mo_df,
            'full_metrics': full_metrics, 'is_metrics': is_metrics, 'oos_metrics': oos_metrics,
            'check_ok': check_ok}

if __name__ == '__main__':
    main()
