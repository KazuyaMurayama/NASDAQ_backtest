"""
raw_leverage ビン別 × 前方リターン分析 V4 (2026-04-20, 更新版)
==============================================================
V3からの変更:
  1. A2 Optimized signal: VIX_MeanReversion層追加
  2. 拡張データ (1974-2026)
  3. GAS Code.gs パラメータ完全一致
  4. ダブルチェック: 幾何平均の対数加重で整合性確認
  5. **前方リターンは「年率CAGR換算」で表示** (算術平均はミスリードのため)
     - CAGR_ann (幾何平均ベース): 複利換算で最も正直
     - Median_ann: 中央値の年率換算
     - Std_ann: 標準偏差の年率換算
     - 最悪値は生の期間値 (単発観測のため年率化しない)
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
from test_delay_robust import calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from opt_lev2x3x import calc_asym_ewma

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
FORWARD_PERIODS = [5, 10, 20, 60, 250]
BIN_LABELS = ['=0%', '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%', '=100%']
TRADING_DAYS = 252

def classify_bin(lev):
    if lev == 0: return '=0%'
    if lev == 1: return '=100%'
    return BIN_LABELS[min(int(lev*10) + 1, 11)]

def annualize(period_return, period_days):
    """Convert a period return (fractional, e.g. 0.01 = +1%) to annualized CAGR.
    CAGR = (1 + r)^(252/period_days) - 1
    Works for negative returns too."""
    if pd.isna(period_return): return np.nan
    # Guard: 1+r must be > 0 for fractional exponent
    base = 1 + period_return
    if base <= 0: return -1.0  # -100% CAGR proxy for pathological cases
    return base ** (TRADING_DAYS / period_days) - 1

def geom_mean(returns_pct):
    """Geometric mean of returns (returns expressed as %). Returns in %"""
    r = returns_pct.dropna() / 100
    if len(r) == 0: return np.nan
    # Filter out <= -100% (bankruptcy-equivalent; not meaningful for compounding)
    r = r[r > -1.0]
    if len(r) == 0: return np.nan
    log_mean = np.log(1 + r).mean()
    return (np.exp(log_mean) - 1) * 100  # back to %

def build_a2_signal(close, returns):
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
    vz = (vp-vma)/vs; vm = (1.0-0.25*vz).clip(0.5,1.15)
    return (dd*vt*slope*mom*vm).clip(0,1.0).fillna(0)

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} days)")
    raw = build_a2_signal(close, returns)
    warmup = 300
    
    fwd = {p: (close.shift(-p) / close - 1) * 100 for p in FORWARD_PERIODS}
    bin_col = raw.apply(classify_bin)
    
    all_rows = []
    
    for p in FORWARD_PERIODS:
        print(f"\n{'='*95}")
        print(f"Forward {p}d return by leverage bin (annualized CAGR)")
        print(f"{'='*95}")
        print(f"{'Bin':>10} {'N':>6} {'CAGR ann':>10} {'Median ann':>11} {'Std ann':>9} {'Pos%':>7} {'Worst(期間)':>11}")
        
        valid_mask = pd.Series(True, index=close.index)
        valid_mask.iloc[:warmup] = False
        valid_mask.iloc[-p:] = False
        
        f = fwd[p][valid_mask]
        b = bin_col[valid_mask]
        
        bin_stats = []
        for lbl in BIN_LABELS:
            mask = (b == lbl)
            d = f[mask].dropna()
            if len(d) == 0:
                bin_stats.append({'bin': lbl, 'N': 0, 'cagr_ann': np.nan, 'median_ann': np.nan,
                                  'std_ann': np.nan, 'pos_rate': np.nan, 'worst_period': np.nan,
                                  'q25_ann': np.nan, 'q75_ann': np.nan, 'arith_mean_period': np.nan})
                continue
            # Geometric mean of period returns (in %)
            g_mean_pct = geom_mean(d)  # in %
            # Annualize geometric mean  
            cagr_ann = annualize(g_mean_pct/100, p) * 100  # back to %
            median_ann = annualize(d.median()/100, p) * 100
            # Std annualizes by sqrt(252/p)
            std_ann = d.std() * np.sqrt(TRADING_DAYS / p)
            q25_ann = annualize(d.quantile(0.25)/100, p) * 100
            q75_ann = annualize(d.quantile(0.75)/100, p) * 100
            
            stats = {
                'bin': lbl,
                'N': len(d),
                'cagr_ann': cagr_ann,
                'median_ann': median_ann,
                'std_ann': std_ann,
                'pos_rate': (d > 0).mean() * 100,
                'worst_period': d.min(),  # single-period worst (not annualized)
                'q25_ann': q25_ann,
                'q75_ann': q75_ann,
                'arith_mean_period': d.mean(),  # keep for reference
            }
            bin_stats.append(stats)
            print(f"  {lbl:>8} {stats['N']:>6d} {stats['cagr_ann']:>+9.2f}% {stats['median_ann']:>+10.2f}% "
                  f"{stats['std_ann']:>8.2f}% {stats['pos_rate']:>6.1f}% {stats['worst_period']:>+10.1f}%")
            all_rows.append({'forward_days': p, **stats})
        
        # Overall (all bins combined)
        d_all = f.dropna()
        g_all_pct = geom_mean(d_all)
        cagr_overall = annualize(g_all_pct/100, p) * 100
        
        # Double-check: compute overall via each bin's geometric contribution
        # log(1+g_overall) should equal weighted sum of log(1+g_bin) by N
        # Verify:
        total_n = sum(s['N'] for s in bin_stats)
        weighted_log = sum(s['N'] * np.log(1 + geom_mean(f[b == s['bin']].dropna())/100)
                           for s in bin_stats if s['N'] > 0)
        weighted_g = (np.exp(weighted_log/total_n) - 1) * 100
        weighted_cagr = annualize(weighted_g/100, p) * 100
        diff = abs(weighted_cagr - cagr_overall)
        status = "OK" if diff < 0.01 else f"FAIL ({diff:.4f}pp)"
        
        print(f"  {'Overall':>8} {len(d_all):>6d} {cagr_overall:>+9.2f}%")
        print(f"  [Double-check: bin-weighted CAGR = {weighted_cagr:+.4f}%, diff = {diff:.6f}pp — {status}]")
    
    out_csv = os.path.join(BASE, 'leverage_bin_analysis_v4.csv')
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    
    # Distribution (unchanged)
    print(f"\n{'='*80}\nraw_leverage distribution (after warmup)\n{'='*80}")
    mask = pd.Series(True, index=close.index); mask.iloc[:warmup] = False
    dist = bin_col[mask].value_counts().reindex(BIN_LABELS).fillna(0).astype(int)
    total = dist.sum()
    for lbl in BIN_LABELS:
        print(f"  {lbl:>8} {dist[lbl]:>6d} {dist[lbl]/total*100:>5.1f}%")

if __name__ == '__main__':
    main()
