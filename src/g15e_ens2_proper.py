"""
g15e_ens2_proper.py — Ens2(Asym+Slope) 完全正規版（v4 と整合）
=============================================================================
g15b の bug 修正:
  1. calc_dd_signal: expanding().max() (誤) → rolling(200).max() (正)
     → backtest_engine.calc_dd_signal を直接 import
  2. rebalance_threshold(0.20) を適用（v4 line 316 と同じ）
  3. DELAY=2 (v4 と同じ)
  4. cost formula: v4 build_tqqq_only_corrected と完全一致

入力前提:
  - max_lev=1.0 (gen_yearly_returns_v4.py:315 と同じ)
  - TQQQ Scenario D: 2×SOFR + 0.50% swap + 0.86% TER
  - DELAY=2 (EVALUATION_STANDARD §0)
  - rebalance_threshold THRESHOLD=0.20

期待値（v4 報告から）:
  - CAGR_FULL: +17.50%, CAGR_IS: +18.60%, CAGR_OOS: +7.93%
  - Sharpe: 0.707, MaxDD: -52.0%
"""
import os, sys, types
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

# 正規の関数を import
from backtest_engine import calc_dd_signal  # rolling(200) max version
from test_ens2_strategies import strategy_ens2_asym_slope

NDX_CSV = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
DTB3_CSV = os.path.join(BASE, 'data', 'dtb3_daily.csv')

IS_END = pd.Timestamp('2021-05-07')
OOS_START = pd.Timestamp('2021-05-08')
OOS_END = pd.Timestamp('2026-03-26')
TRADING_DAYS = 252

# v4 と同じ定数
TQQQ_TER = 0.0086
SWAP_SPREAD = 0.0050
BASE_LEV = 3.0
THRESHOLD = 0.20
DELAY = 2
JP_TAX_MULT = 0.8273  # §3-A tax


def rebalance_threshold(leverage, threshold):
    """v4 と同じ実装 (test_final_6strategy.py より複製)."""
    result = pd.Series(0.0, index=leverage.index)
    current = leverage.iloc[0]
    result.iloc[0] = current
    for i in range(1, len(leverage)):
        target = leverage.iloc[i]
        if target == 0.0 and current > 0.0:
            current = 0.0
        elif current == 0.0 and target > 0.0:
            current = target
        elif abs(target - current) > threshold:
            current = target
        result.iloc[i] = current
    return result


def load_sofr(dates_idx):
    df = pd.read_csv(DTB3_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    val_col = [c for c in df.columns if c.lower() not in ['date']][0]
    sofr = pd.to_numeric(df[val_col], errors='coerce') / 100.0
    return sofr.reindex(dates_idx).ffill().bfill().fillna(0.04)


def main():
    print('=' * 100)
    print('g15e: Ens2(Asym+Slope) 完全正規実装 (v4 と整合)')
    print('=' * 100)

    df = pd.read_csv(NDX_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close = df['Close']
    returns = close.pct_change().fillna(0)
    sofr = load_sofr(df.index)

    # 1. 正規の strategy_ens2_asym_slope（backtest_engine.calc_dd_signal を内部利用）
    lev_raw, dd_sig = strategy_ens2_asym_slope(close, returns, 0.82, 0.92, 0.25, 20, 5, 1.0)
    print(f'\n[Step 1] 正規 signal generation:')
    print(f'  max_lev = 1.0')
    print(f'  Raw signal: max={lev_raw.max():.3f}, avg={lev_raw.mean():.3f}')
    print(f'  DD signal flips: {int((dd_sig.diff().abs() > 0).sum())} 回')

    # 2. rebalance_threshold 適用
    lev_filtered = rebalance_threshold(lev_raw, THRESHOLD)
    print(f'\n[Step 2] rebalance_threshold(0.20) 適用:')
    print(f'  Filtered avg={lev_filtered.mean():.3f}')
    print(f'  実際の position 変更: {int((lev_filtered.diff().abs() > 0).sum())} 回 ({(lev_filtered.diff().abs() > 0).sum()/52.26:.1f}/yr)')

    # 3. DELAY=2 + cost (v4 完全同等)
    lev_shifted = lev_filtered.shift(DELAY).fillna(0)
    sofr_daily = sofr / TRADING_DAYS
    dc = TQQQ_TER / TRADING_DAYS
    swap_d = SWAP_SPREAD / TRADING_DAYS
    r_nas = returns.values

    n = len(r_nas)
    nav_arr = np.ones(n)
    lev_arr = lev_shifted.values
    strat_ret = np.zeros(n)
    for i in range(1, n):
        lv = lev_arr[i]
        if lv > 0:
            r = lv * (BASE_LEV * r_nas[i] - dc - 2.0 * sofr_daily.iloc[i] - swap_d)
        else:
            r = 0.0
        strat_ret[i] = r
        nav_arr[i] = nav_arr[i-1] * (1 + r)

    nav = pd.Series(nav_arr, index=df.index)
    strat_ret_s = pd.Series(strat_ret, index=df.index)

    # 4. 9指標 計算
    def cagr(n_seg): return (n_seg.iloc[-1] / n_seg.iloc[0])**(TRADING_DAYS/len(n_seg)) - 1
    def sharpe(r_seg):
        r = r_seg[r_seg != 0]  # zero 期間（cash）除外せず計算
        return strat_ret_s[strat_ret_s.index.isin(r_seg.index)].mean() / strat_ret_s[strat_ret_s.index.isin(r_seg.index)].std() * np.sqrt(TRADING_DAYS) if strat_ret_s[strat_ret_s.index.isin(r_seg.index)].std() > 0 else 0
    def sharpe2(idx):
        r = strat_ret_s.loc[idx]
        return r.mean()/r.std()*np.sqrt(TRADING_DAYS) if r.std() > 0 else 0
    def maxdd(n_seg): return (n_seg / n_seg.cummax() - 1).min()

    is_idx = df.index[df.index <= IS_END]
    oos_idx = df.index[(df.index >= OOS_START) & (df.index <= OOS_END)]
    full_idx = df.index

    is_nav = nav[is_idx] / nav[is_idx].iloc[0]
    oos_nav = nav[oos_idx] / nav[oos_idx].iloc[0]

    cagr_full = cagr(nav)
    cagr_is = cagr(is_nav)
    cagr_oos = cagr(oos_nav)
    sh_full = sharpe2(full_idx)
    sh_oos = sharpe2(oos_idx)
    maxdd_full = maxdd(nav)
    maxdd_oos = maxdd(oos_nav)

    is_oos_gap = cagr_is - cagr_oos

    # Worst10Y★ (calendar year)
    yearly = nav.groupby(nav.index.year).last()
    worst10y = min(
        (yearly.iloc[i+10]/yearly.iloc[i])**(1/10) - 1
        for i in range(len(yearly)-10)
    )
    # P10_5Y▷ (daily rolling 252×5)
    window = TRADING_DAYS * 5
    rolling = [(nav.iloc[i]/nav.iloc[i-window])**(1/5)-1 for i in range(window, len(nav))]
    p10_5y = np.percentile(rolling, 10)

    # 税後 (×0.8273 のみ, B&H/TQQQ ETF style)
    cagr_oos_net = cagr_oos * JP_TAX_MULT
    worst10y_net = worst10y * JP_TAX_MULT
    p10_5y_net = p10_5y * JP_TAX_MULT

    print(f'\n[Step 4] 9指標 (Scenario D, max_lev=1.0, rebalance_threshold, DELAY=2):')
    print(f'  CAGR_FULL_raw       : {cagr_full*100:+.2f}%  (v4 報告: +17.50%)')
    print(f'  CAGR_IS_raw         : {cagr_is*100:+.2f}%  (v4 報告: +18.60%)')
    print(f'  CAGR_OOS_raw        : {cagr_oos*100:+.2f}%  (v4 報告: +7.93%)')
    print(f'  CAGR_OOS_net (⓽)    : {cagr_oos_net*100:+.2f}%  (税後 ×0.8273)')
    print(f'  Sharpe_FULL         : {sh_full:+.3f}  (v4 報告: 0.707)')
    print(f'  Sharpe_OOS (ⓒ)      : {sh_oos:+.3f}')
    print(f'  MaxDD_FULL (ⓒ)      : {maxdd_full*100:+.2f}%  (v4 報告: -52.0%)')
    print(f'  Worst10Y★_raw       : {worst10y*100:+.2f}%')
    print(f'  Worst10Y★_net (⓽)   : {worst10y_net*100:+.2f}%')
    print(f'  P10_5Y▷_raw         : {p10_5y*100:+.2f}%')
    print(f'  P10_5Y▷_net (⓽)     : {p10_5y_net*100:+.2f}%')
    print(f'  IS-OOS gap (ⓒ)      : {is_oos_gap*100:+.2f}pp')
    print(f'  Trades/yr           : {(lev_filtered.diff().abs() > 0).sum()/52.26:.2f}')

    # CSV 保存
    result = {
        'Strategy': 'Ens2(Asym+Slope) max_lev=1.0 (v4 正規)',
        'CAGR_OOS_raw': cagr_oos,
        'CAGR_OOS_net': cagr_oos_net,
        'Sharpe_OOS': sh_oos,
        'MaxDD_FULL': maxdd_full,
        'Worst10Y_star_net': worst10y_net,
        'P10_5Y_net': p10_5y_net,
        'IS_OOS_gap': is_oos_gap,
        'Trades_yr': (lev_filtered.diff().abs() > 0).sum()/52.26,
        'WFA_WFE': None,
        'WFA_CI95_lo': None,
    }
    pd.DataFrame([result]).to_csv(os.path.join(BASE, 'g15e_ens2_proper_results.csv'), index=False)
    print('\n→ CSV saved: g15e_ens2_proper_results.csv')


if __name__ == '__main__':
    main()
