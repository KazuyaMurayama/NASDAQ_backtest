"""
raw_leverage ビン別 × 前方リターン分析 V4 (2026-04-20)
======================================================
V3からの変更:
  - A2 Optimized signal: VIX_MeanReversion層を追加 (DD × VT × Slope × MomDecel × VIX)
  - 拡張データ (1974-01-02 〜 2026-03-26) を使用
  - GAS production (Code.gs) のパラメータと完全一致
  - ダブルチェック: ビン合計 × 期待値 = 全体平均 (許容 0.01pp)
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

def classify_bin(lev):
    if lev == 0: return '=0%'
    if lev == 1: return '=100%'
    return BIN_LABELS[min(int(lev*10) + 1, 11)]

def build_a2_signal(close, returns):
    """A2 Optimized with VIX layer — matches Code.gs production."""
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
    raw = (dd*vt*slope*mom*vm).clip(0,1.0).fillna(0)
    return raw

def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} days)")
    
    raw = build_a2_signal(close, returns)
    
    # Valid range: skip first ~year for signal warmup, allow max 250d forward
    warmup = 300
    valid_start = warmup
    
    # Build per-period forward returns (in %)
    fwd = {}
    for p in FORWARD_PERIODS:
        fwd[p] = (close.shift(-p) / close - 1) * 100
    
    # Classify bins
    bin_col = raw.apply(classify_bin)
    
    # Main analysis
    result = {}
    all_rows = []
    
    for p in FORWARD_PERIODS:
        print(f"\n{'='*80}")
        print(f"Forward {p}d return by leverage bin")
        print(f"{'='*80}")
        print(f"{'Bin':>10} {'N':>6} {'Mean':>8} {'Median':>8} {'Std':>8} {'Pos%':>7} {'Worst':>9}")
        
        valid_mask = pd.Series(True, index=close.index)
        valid_mask.iloc[:valid_start] = False
        valid_mask.iloc[-p:] = False  # remove last p rows (no forward data)
        
        f = fwd[p][valid_mask]
        b = bin_col[valid_mask]
        
        bin_stats = []
        for lbl in BIN_LABELS:
            mask = (b == lbl)
            d = f[mask].dropna()
            if len(d) == 0:
                bin_stats.append({'bin': lbl, 'N': 0, 'mean': np.nan, 'median': np.nan,
                                  'std': np.nan, 'pos_rate': np.nan, 'worst': np.nan,
                                  'q25': np.nan, 'q75': np.nan})
                continue
            stats = {
                'bin': lbl,
                'N': len(d),
                'mean': d.mean(),
                'median': d.median(),
                'std': d.std(),
                'pos_rate': (d > 0).mean() * 100,
                'worst': d.min(),
                'q25': d.quantile(0.25),
                'q75': d.quantile(0.75),
            }
            bin_stats.append(stats)
            print(f"  {lbl:>8} {stats['N']:>6d} {stats['mean']:>+7.2f}% {stats['median']:>+7.2f}% "
                  f"{stats['std']:>7.2f}% {stats['pos_rate']:>6.1f}% {stats['worst']:>+8.1f}%")
            all_rows.append({'forward_days': p, **stats})
        
        # Overall
        d_all = f.dropna()
        overall_mean = d_all.mean()
        overall_n = len(d_all)
        
        # DOUBLE CHECK: bin-weighted sum should equal overall
        bin_n_sum = sum(s['N'] for s in bin_stats)
        bin_mean_weighted = sum(s['N'] * s['mean'] for s in bin_stats if s['N'] > 0) / bin_n_sum
        diff = abs(bin_mean_weighted - overall_mean)
        status = "OK" if diff < 0.01 else f"FAIL (diff={diff:.4f}pp)"
        print(f"  {'Overall':>8} {overall_n:>6d} {overall_mean:>+7.2f}%")
        print(f"  [Double-check: weighted sum = {bin_mean_weighted:+.4f}%, diff = {diff:.6f}pp — {status}]")
        
        result[p] = {'bin_stats': bin_stats, 'overall_mean': overall_mean, 'overall_n': overall_n}
    
    # Save long-form CSV
    out_csv = os.path.join(BASE, 'leverage_bin_analysis_v4.csv')
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    
    # Bin distribution
    print(f"\n{'='*80}")
    print("raw_leverage distribution (after warmup)")
    print(f"{'='*80}")
    mask = pd.Series(True, index=close.index); mask.iloc[:valid_start] = False
    dist = bin_col[mask].value_counts().reindex(BIN_LABELS).fillna(0).astype(int)
    total = dist.sum()
    print(f"{'Bin':>10} {'N':>6} {'%':>6}")
    for lbl in BIN_LABELS:
        pct = dist[lbl] / total * 100
        print(f"  {lbl:>8} {dist[lbl]:>6d} {pct:>5.1f}%")
    
    return result

if __name__ == '__main__':
    main()
