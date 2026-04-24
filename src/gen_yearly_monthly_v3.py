"""
Yearly/Monthly Returns Report v3 (2026-04-24)
=============================================
Extends v2 with:
  - Berkshire Hathaway (BRK-A per-share market value) as 9th strategy column
  - BRK data hardcoded from Warren Buffett 2023 Annual Letter (1974-2024)
  - BRK double-check: 1974-2023 CAGR expected ~20.3% ±0.3pp
  - DH Dyn 2x3x [A] vs BRK quantitative comparison section

Strategies (9):
  1. DH Dyn 2x3x [A]      - sleeve-independent, Approach A
  2. DH Dyn 2x3x [B]      - current GAS (for comparison)
  3. DH Dyn CAGR25+        - 1x Gold/Bond variant
  4. A2 Optimized          - single-asset A2 signal
  5. Ens2(Asym+Slope)      - legacy best
  6. DD Only               - baseline with DD control
  7. BH 3x                 - 3x leveraged buy-and-hold
  8. BH 1x                 - 1x buy-and-hold
  9. Berkshire Hathaway    - BRK-A per-share market value (annual, no daily NAV)
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

# ============================================================
# BRK-A per-share market value annual returns
# Source: Warren Buffett 2023 Annual Letter (official)
# Market value change (not book value), 1974-2024
# 2025/2026: excluded (incomplete data)
# ============================================================
brk_annual_returns = {
    1974: -0.487, 1975: 0.025, 1976: 1.293, 1977: 0.468, 1978: 0.145,
    1979: 1.025, 1980: 0.328, 1981: 0.318, 1982: 0.384, 1983: 0.690,
    1984: -0.027, 1985: 0.937, 1986: 0.142, 1987: 0.046, 1988: 0.593,
    1989: 0.846, 1990: -0.231, 1991: 0.356, 1992: 0.298, 1993: 0.389,
    1994: 0.250, 1995: 0.574, 1996: 0.062, 1997: 0.349, 1998: 0.522,
    1999: -0.199, 2000: 0.266, 2001: 0.065, 2002: -0.038, 2003: 0.158,
    2004: 0.043, 2005: 0.008, 2006: 0.241, 2007: 0.287, 2008: -0.318,
    2009: 0.027, 2010: 0.214, 2011: -0.047, 2012: 0.168, 2013: 0.327,
    2014: 0.270, 2015: -0.125, 2016: 0.234, 2017: 0.219, 2018: 0.028,
    2019: 0.110, 2020: 0.024, 2021: 0.296, 2022: 0.040, 2023: 0.158,
    2024: 0.255,  # 2024 market value return (approximate, confirmed ~25%)
}
# 2025 / 2026-Q1: データ不完全のため除外 (N/A)

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

def calc_brk_cagr(start_year, end_year):
    """Compute compounded CAGR for BRK from brk_annual_returns dict."""
    years = [y for y in range(start_year, end_year+1) if y in brk_annual_returns]
    if not years:
        return np.nan
    prod = 1.0
    for y in years:
        prod *= (1 + brk_annual_returns[y])
    n_years = len(years)
    return prod**(1/n_years) - 1

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

    # ===== Build yearly/monthly tables (8 strategies from NAV) =====
    order_8 = ['DH Dyn 2x3x [A]', 'DH Dyn 2x3x [B]', 'DH Dyn CAGR25+', 'A2 Optimized',
               'Ens2(Asym+Slope)', 'DD Only', 'BH 3x', 'BH 1x']

    yr_df = pd.DataFrame({name: yearly_from_nav(navs[name], dates) for name in order_8})
    mo_df = pd.DataFrame({name: monthly_from_nav(navs[name], dates) for name in order_8})

    # ===== Add BRK column to yearly table =====
    # BRK has data 1974-2024 only; 2025/2026 are N/A
    brk_series = pd.Series({yr: ret*100 for yr, ret in brk_annual_returns.items()},
                           name='Berkshire Hathaway')
    yr_df['Berkshire Hathaway'] = brk_series

    order_9 = order_8 + ['Berkshire Hathaway']

    # ===== DOUBLE-CHECK for 8 NAV strategies =====
    print("=" * 80)
    print("DOUBLE CHECK: CAGR from NAV endpoints vs from compounded yearly returns")
    print("=" * 80)
    check_ok = True
    for name in order_8:
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

    # ===== BRK Double-Check: 1974-2023 CAGR expected ~20.3% ±0.3pp =====
    print("=" * 80)
    print("BRK DOUBLE CHECK: 1974-2023 CAGR (expect ~20.3% ±0.3pp)")
    print("=" * 80)
    brk_cagr_50y = calc_brk_cagr(1974, 2023)
    brk_expected = 0.203
    brk_tolerance = 0.003
    brk_diff = abs(brk_cagr_50y - brk_expected)
    brk_ok = brk_diff <= brk_tolerance
    brk_status = "PASS" if brk_ok else f"FAIL (diff={brk_diff*100:.2f}pp from expected {brk_expected*100:.1f}%)"
    print(f"  BRK 1974-2023 (50y) CAGR: {brk_cagr_50y*100:.2f}%  Expected: ~{brk_expected*100:.1f}%  [{brk_status}]")

    brk_cagr_51y = calc_brk_cagr(1974, 2024)
    print(f"  BRK 1974-2024 (51y) CAGR: {brk_cagr_51y*100:.2f}%  Expected: ~20.4%")
    print()

    # ===== Period metrics =====
    full_metrics = {}
    is_metrics = {}
    oos_metrics = {}
    for name in order_8:
        full_metrics[name] = period_metrics(navs[name], dates, '1974-01-02', '2026-03-28')
        is_metrics[name] = period_metrics(navs[name], dates, '1974-01-02', OOS_SPLIT)
        oos_metrics[name] = period_metrics(navs[name], dates, '2021-05-08', '2026-03-28')

    # BRK period metrics from annual returns
    # IS: 1974-2020 (47 years), OOS: 2021-2024 (4 years)
    brk_is_cagr = calc_brk_cagr(1974, 2020)
    brk_oos_cagr = calc_brk_cagr(2021, 2024)
    brk_full_cagr = calc_brk_cagr(1974, 2024)

    print("=" * 80)
    print("Full-period, IS, OOS metrics")
    print("=" * 80)
    print(f"{'Strategy':<22} {'Full CAGR':>10} {'IS CAGR':>9} {'OOS CAGR':>9} {'Sharpe':>7}")
    for name in order_8:
        fm = full_metrics[name]; im = is_metrics[name]; om = oos_metrics[name]
        print(f"  {name:<20} {fm['CAGR']*100:>+8.2f}% {im['CAGR']*100:>+7.2f}% {om['CAGR']*100:>+7.2f}% {fm['Sharpe']:>7.3f}")
    print(f"  {'Berkshire Hathaway':<20} {brk_full_cagr*100:>+8.2f}% {brk_is_cagr*100:>+7.2f}% {brk_oos_cagr*100:>+7.2f}%    N/A")
    print()

    # ===== DH [A] vs BRK comparison =====
    print("=" * 80)
    print("DH Dyn 2x3x [A] vs Berkshire Hathaway (1974-2024 overlap)")
    print("=" * 80)

    # Years available for both: 1974-2024
    dh_a_yearly = yr_df['DH Dyn 2x3x [A]']
    brk_yearly_pct = brk_series

    common_years = sorted(set(dh_a_yearly.dropna().index) & set(brk_yearly_pct.dropna().index))
    dh_a_common = dh_a_yearly[common_years]
    brk_common = brk_yearly_pct[common_years]

    # CAGR comparison 1974-2024
    dh_prod = ((1 + dh_a_common/100).prod())
    brk_prod = ((1 + brk_common/100).prod())
    n_common = len(common_years)
    dh_cagr_overlap = dh_prod**(1/n_common) - 1
    brk_cagr_overlap = brk_prod**(1/n_common) - 1

    print(f"  Overlap period: {common_years[0]}-{common_years[-1]} ({n_common} years)")
    print(f"  DH [A] CAGR:  {dh_cagr_overlap*100:+.2f}%")
    print(f"  BRK CAGR:     {brk_cagr_overlap*100:+.2f}%")
    print(f"  Excess:        {(dh_cagr_overlap - brk_cagr_overlap)*100:+.2f}pp")

    # Win/loss record
    dh_wins = sum(1 for y in common_years if dh_a_common[y] > brk_common[y])
    brk_wins = sum(1 for y in common_years if brk_common[y] > dh_a_common[y])
    ties = n_common - dh_wins - brk_wins
    print(f"\n  Year-by-year (DH[A] > BRK): {dh_wins}W - {brk_wins}L - {ties}T out of {n_common} years")

    # Cumulative wealth comparison
    dh_cum = dh_prod
    brk_cum = brk_prod
    print(f"\n  Cumulative wealth (start=1):")
    print(f"    DH [A]: {dh_cum:.1f}x")
    print(f"    BRK:    {brk_cum:.1f}x")
    print(f"    Ratio:  {dh_cum/brk_cum:.1f}x (DH/BRK)")

    # Excess returns by year
    excess = dh_a_common - brk_common
    cum_excess_prod = dh_cum / brk_cum
    print(f"\n  Cumulative excess return ratio: {cum_excess_prod:.2f}x")

    # Decadal averages
    print("\n  10-year average return comparison:")
    decades = [
        ('1970s', [y for y in common_years if 1970 <= y <= 1979]),
        ('1980s', [y for y in common_years if 1980 <= y <= 1989]),
        ('1990s', [y for y in common_years if 1990 <= y <= 1999]),
        ('2000s', [y for y in common_years if 2000 <= y <= 2009]),
        ('2010s', [y for y in common_years if 2010 <= y <= 2019]),
        ('2020s', [y for y in common_years if 2020 <= y <= 2029]),
    ]
    print(f"  {'Decade':<10} {'DH [A] avg':>12} {'BRK avg':>12} {'Excess':>10}")
    decade_stats = {}
    for label, yrs in decades:
        if not yrs: continue
        dh_avg = dh_a_common[yrs].mean()
        brk_avg = brk_common[yrs].mean()
        exc = dh_avg - brk_avg
        decade_stats[label] = (dh_avg, brk_avg, exc, yrs)
        print(f"  {label:<10} {dh_avg:>+11.1f}% {brk_avg:>+11.1f}% {exc:>+9.1f}pp")
    print()

    # Save CSVs
    yr_csv = os.path.join(BASE, 'yearly_returns_9strategies_v3.csv')
    mo_csv = os.path.join(BASE, 'monthly_returns_oos_v2.csv')  # unchanged
    yr_df.to_csv(yr_csv)
    print(f"Saved: {yr_csv}")

    return {
        'order': order_9, 'navs': navs, 'dates': dates,
        'yr_df': yr_df, 'mo_df': mo_df,
        'full_metrics': full_metrics, 'is_metrics': is_metrics, 'oos_metrics': oos_metrics,
        'check_ok': check_ok, 'brk_ok': brk_ok,
        'brk_cagr_50y': brk_cagr_50y, 'brk_cagr_51y': brk_cagr_51y,
        'brk_full_cagr': brk_full_cagr, 'brk_is_cagr': brk_is_cagr, 'brk_oos_cagr': brk_oos_cagr,
        'dh_cagr_overlap': dh_cagr_overlap, 'brk_cagr_overlap': brk_cagr_overlap,
        'common_years': common_years, 'dh_a_common': dh_a_common, 'brk_common': brk_common,
        'dh_wins': dh_wins, 'brk_wins': brk_wins, 'ties': ties,
        'dh_cum': dh_cum, 'brk_cum': brk_cum, 'excess': excess,
        'decade_stats': decade_stats,
    }

if __name__ == '__main__':
    main()
