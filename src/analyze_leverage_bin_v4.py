"""
raw_leverage ビン別 × **ベスト戦略の前方NAVリターン** 年率CAGR分析 V4 rev3 (2026-04-20)
======================================================================================
V3からの変更:
  1. A2 Optimized signal: VIX_MeanReversion層追加
  2. 拡張データ (1974-2026)
  3. GAS Code.gs パラメータ完全一致
  4. ⚠️ **前方リターンを「ベスト戦略 DH Dyn 2x3x [A] のNAV変化」で計算**
     (従来: NASDAQ指数の前方リターン → 実運用と乖離)
  5. 年率CAGR換算で表示
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
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from opt_lev2x3x import build_lev_navs, calc_asym_ewma

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
FORWARD_PERIODS = [5, 10, 20, 60, 250]
BIN_LABELS = ['=0%', '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%', '=100%']
TRADING_DAYS = 252
THRESHOLD = 0.20; BASE_LEV = 3.0; ANNUAL_COST = 0.0086; DELAY = 2

def classify_bin(lev):
    if lev == 0: return '=0%'
    if lev == 1: return '=100%'
    return BIN_LABELS[min(int(lev*10) + 1, 11)]

def annualize(r, p):
    if pd.isna(r): return np.nan
    base = 1 + r
    if base <= 0: return -1.0
    return base ** (TRADING_DAYS / p) - 1

def geom_mean_pct(returns_pct):
    r = returns_pct.dropna() / 100
    r = r[r > -1.0]
    if len(r) == 0: return np.nan
    return (np.exp(np.log(1 + r).mean()) - 1) * 100

def build_best_strategy_nav(close, returns, dates):
    """Build DH Dyn 2x3x [A] (best strategy, sleeve-independent approach)."""
    # A2 raw signal
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
    vz_v = vz_ser.fillna(0).values; raw_v = raw.values
    n = len(dates)
    
    # Gold/Bond 2x/3x
    gold_1x = prepare_gold_data(dates); bond_1x = prepare_bond_data(dates)
    gold_2x, bond_3x = build_lev_navs(gold_1x, bond_1x)
    
    # GAS rebalance simulation
    wn = np.zeros(n); wg = np.zeros(n); wb = np.zeros(n); lev = np.zeros(n)
    cur = raw_v[0]
    cur_w = np.clip(0.55+0.25*cur-0.10*max(vz_v[0],0), 0.30, 0.90)
    cwn, cwg, cwb = cur_w, (1-cur_w)*0.5, (1-cur_w)*0.5
    lev[0]=cur; wn[0]=cwn; wg[0]=cwg; wb[0]=cwb
    for i in range(1, n):
        t = raw_v[i]
        if (t==0 and cur>0) or (cur==0 and t>0) or abs(t-cur)>THRESHOLD:
            cur = t
            cur_w = np.clip(0.55+0.25*cur-0.10*max(vz_v[i],0), 0.30, 0.90)
            cwn, cwg, cwb = cur_w, (1-cur_w)*0.5, (1-cur_w)*0.5
        lev[i]=cur; wn[i]=cwn; wg[i]=cwg; wb[i]=cwb
    
    # Approach A portfolio NAV
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x).pct_change().fillna(0).values
    lev_s = pd.Series(lev, index=dates.index).shift(DELAY).fillna(0).values
    wn_s = pd.Series(wn, index=dates.index).shift(DELAY).fillna(0).values
    wg_s = pd.Series(wg, index=dates.index).shift(DELAY).fillna(0).values
    wb_s = pd.Series(wb, index=dates.index).shift(DELAY).fillna(0).values
    dc = ANNUAL_COST/252
    daily = wn_s*lev_s*(r_nas*BASE_LEV - dc) + wg_s*r_g2 + wb_s*r_b3
    nav = (1 + pd.Series(daily, index=dates.index)).cumprod()
    return raw, nav

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} days)")
    
    print("Building DH Dyn 2x3x [A] NAV (best strategy)...")
    raw, nav_strat = build_best_strategy_nav(close, returns, dates)
    n_yrs = len(df)/252
    strat_cagr = nav_strat.iloc[-1]**(1/n_yrs) - 1
    print(f"Strategy full-period CAGR: +{strat_cagr*100:.2f}%")
    
    warmup = 300
    
    # Forward STRATEGY returns (not NASDAQ)
    fwd = {p: (nav_strat.shift(-p) / nav_strat - 1) * 100 for p in FORWARD_PERIODS}
    bin_col = raw.apply(classify_bin)
    
    all_rows = []
    
    for p in FORWARD_PERIODS:
        print(f"\n{'='*95}")
        print(f"Forward {p}d STRATEGY return (DH Dyn 2x3x [A]) by leverage bin — annualized CAGR")
        print(f"{'='*95}")
        print(f"{'Bin':>10} {'N':>6} {'CAGR ann':>10} {'Median ann':>11} {'Std ann':>9} {'Pos%':>7} {'Worst(期間)':>11}")
        
        valid = pd.Series(True, index=close.index)
        valid.iloc[:warmup] = False
        valid.iloc[-p:] = False
        
        f = fwd[p][valid]
        b = bin_col[valid]
        
        for lbl in BIN_LABELS:
            d = f[b == lbl].dropna()
            if len(d) == 0:
                all_rows.append({'forward_days': p, 'bin': lbl, 'N': 0, 'cagr_ann': np.nan,
                                 'median_ann': np.nan, 'std_ann': np.nan, 'pos_rate': np.nan,
                                 'worst_period': np.nan, 'q25_ann': np.nan, 'q75_ann': np.nan})
                continue
            g = geom_mean_pct(d)
            cagr_ann = annualize(g/100, p) * 100
            med_ann = annualize(d.median()/100, p) * 100
            std_ann = d.std() * np.sqrt(TRADING_DAYS / p)
            q25 = annualize(d.quantile(0.25)/100, p) * 100
            q75 = annualize(d.quantile(0.75)/100, p) * 100
            pos = (d > 0).mean() * 100
            worst = d.min()
            all_rows.append({'forward_days': p, 'bin': lbl, 'N': len(d),
                            'cagr_ann': cagr_ann, 'median_ann': med_ann, 'std_ann': std_ann,
                            'pos_rate': pos, 'worst_period': worst, 'q25_ann': q25, 'q75_ann': q75})
            print(f"  {lbl:>8} {len(d):>6d} {cagr_ann:>+9.2f}% {med_ann:>+10.2f}% "
                  f"{std_ann:>8.2f}% {pos:>6.1f}% {worst:>+10.1f}%")
        
        # Overall + double-check
        d_all = f.dropna()
        g_all = geom_mean_pct(d_all)
        cagr_all = annualize(g_all/100, p) * 100
        stats_p = [r for r in all_rows if r['forward_days']==p and r['N']>0]
        tot = sum(s['N'] for s in stats_p)
        wlog = sum(s['N'] * np.log(1 + geom_mean_pct(f[b == s['bin']].dropna())/100) for s in stats_p)
        wg = (np.exp(wlog/tot) - 1) * 100
        wcagr = annualize(wg/100, p) * 100
        diff = abs(wcagr - cagr_all)
        status = "OK" if diff < 0.01 else f"FAIL ({diff:.4f}pp)"
        print(f"  {'Overall':>8} {len(d_all):>6d} {cagr_all:>+9.2f}%")
        print(f"  [Double-check: bin-weighted CAGR = {wcagr:+.4f}%, diff = {diff:.6f}pp — {status}]")
    
    out_csv = os.path.join(BASE, 'leverage_bin_analysis_v4.csv')
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

if __name__ == '__main__':
    main()
