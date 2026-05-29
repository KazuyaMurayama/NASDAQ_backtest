"""
g15b_ens2_oos_scenarioD.py — Ens2(Asym+Slope) を正しい Scenario D コストで再実行
=============================================================================
背景:
  test_ens2_strategies.py が呼ぶ backtest_engine.run_backtest() は
  annual_cost=0.9%/yr のフラットコストを使用しており、レバレッジ × SOFR の
  正しい金利コスト計上ができていない。

  → ens2_comparison_results.csv の Ens2(Asym+Slope) 値（CAGR 28.58%/Sharpe 1.031）
    は Scenario D ではなく実コストを過小評価した値。

修正:
  本 script は test_ens2_strategies.py のシグナル生成ロジック (strategy_ens2_asym_slope)
  を再利用しつつ、コストモデルを TQQQ Scenario D に置換:
    - TER 0.86%/yr
    - sofr_multiplier = 2.0（TQQQ 3x の (L-1) × SOFR 構造）
    - swap_spread 0.50%/yr

  さらに IS (1974-2021-05-07) / OOS (2021-05-08-2026-03-26) で分割計算。

出力:
  - g15b_ens2_oos_results.csv — IS/OOS/FULL 各区間の 9指標
  - 標準出力に Ens2(Asym+Slope) max_lev=1.0 と max_lev=3.0 の比較表
"""
import os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

# データ
NDX_CSV = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
DTB3_CSV = os.path.join(BASE, 'data', 'dtb3_daily.csv')

# 期間
IS_END    = pd.Timestamp('2021-05-07')
OOS_START = pd.Timestamp('2021-05-08')
OOS_END   = pd.Timestamp('2026-03-26')
TRADING_DAYS = 252

# TQQQ Scenario D コスト
TQQQ_TER = 0.0086
TQQQ_SOFR_MULT = 2.0
TQQQ_SWAP_SPREAD = 0.0050
BASE_LEV = 3.0  # TQQQ は 3x NDX

# Ens2 strategy logic from test_ens2_strategies.py
def calc_asym_ewma_vol(returns, span_up=20, span_dn=5):
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    for i in range(1, len(returns)):
        ret = returns.iloc[i]
        prev = variance.iloc[i-1]
        if ret < 0:
            alpha = 2 / (span_dn + 1)
        else:
            alpha = 2 / (span_up + 1)
        variance.iloc[i] = (1 - alpha) * prev + alpha * (ret ** 2)
    return np.sqrt(variance * 252)


def calc_slope_multiplier(close, ma_lookback=200, norm_window=60, base=0.7, sensitivity=0.3, min_mult=0.3, max_mult=1.5):
    ma = close.rolling(ma_lookback).mean()
    slope = ma.pct_change()
    slope_mean = slope.rolling(norm_window).mean()
    slope_std = slope.rolling(norm_window).std()
    z = (slope - slope_mean) / slope_std.replace(0, 0.0001)
    multiplier = base + sensitivity * z
    return multiplier.clip(min_mult, max_mult).fillna(1.0)


def calc_dd_signal(close, exit_th=0.82, reentry_th=0.92):
    """DD(-18/92) Control — exit below 82% of high, re-enter at 92%."""
    high_water = close.expanding().max()
    drawdown = close / high_water
    sig = pd.Series(1.0, index=close.index)
    in_position = True
    for i in range(1, len(close)):
        if in_position:
            if drawdown.iloc[i] < exit_th:
                in_position = False
                sig.iloc[i] = 0
            else:
                sig.iloc[i] = 1
        else:
            if drawdown.iloc[i] > reentry_th:
                in_position = True
                sig.iloc[i] = 1
            else:
                sig.iloc[i] = 0
    return sig


def strategy_ens2_asym_slope(close, returns, exit_th=0.82, reentry_th=0.92,
                              target_vol=0.25, span_up=20, span_dn=5, max_lev=1.0):
    dd_sig = calc_dd_signal(close, exit_th, reentry_th)
    asym_vol = calc_asym_ewma_vol(returns, span_up, span_dn)
    vt_lev = (target_vol / asym_vol).clip(0, max_lev)
    slope_mult = calc_slope_multiplier(close)
    leverage = (dd_sig * vt_lev * slope_mult).clip(0, max_lev).fillna(0)
    return leverage, dd_sig


# SOFR loader (DTB3)
def load_sofr(dates_idx):
    df = pd.read_csv(DTB3_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    val_col = [c for c in df.columns if c.lower() not in ['date']][0]
    sofr = df[val_col]
    # parse to numeric (some files have '.' for missing)
    sofr = pd.to_numeric(sofr, errors='coerce') / 100.0  # percent → decimal
    sofr = sofr.reindex(dates_idx).ffill().bfill().fillna(0.04)
    return sofr.values


def run_ens2_scenario_d(max_lev=1.0):
    # Load NDX
    df = pd.read_csv(NDX_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close = df['Close']
    returns = close.pct_change().fillna(0)

    # Load SOFR
    sofr_annual = load_sofr(df.index)
    sofr_daily = sofr_annual / TRADING_DAYS

    # Generate Ens2 signal
    leverage, dd_sig = strategy_ens2_asym_slope(close, returns, max_lev=max_lev)
    lev_shift = leverage.shift(1).fillna(0)

    # Apply TQQQ Scenario D cost
    # daily_return = lev_shift × (3x × NDX_ret - (3-1) × (SOFR_daily + swap_spread/252) - TER/252)
    cost_per_unit = (TQQQ_SOFR_MULT * sofr_daily + TQQQ_SWAP_SPREAD / TRADING_DAYS + TQQQ_TER / TRADING_DAYS)
    leveraged_ndx_ret = returns * BASE_LEV
    strategy_returns = lev_shift * (leveraged_ndx_ret - cost_per_unit)
    strategy_returns = strategy_returns.fillna(0)
    nav = (1 + strategy_returns).cumprod()

    # Compute metrics for FULL / IS / OOS
    def slice_metrics(nav_seg, ret_seg, lev_seg):
        years = len(nav_seg) / TRADING_DAYS
        cagr = (nav_seg.iloc[-1] / nav_seg.iloc[0]) ** (1/years) - 1
        sharpe = ret_seg.mean() / ret_seg.std() * np.sqrt(TRADING_DAYS) if ret_seg.std() > 0 else 0
        maxdd = (nav_seg / nav_seg.cummax() - 1).min()
        # Trades per year (signal change count)
        trades_total = (lev_seg.diff().abs() > 0.5).sum()
        trades_yr = trades_total / years
        return dict(CAGR=cagr, Sharpe=sharpe, MaxDD=maxdd, Trades_yr=trades_yr)

    # FULL
    full_m = slice_metrics(nav, strategy_returns, leverage)
    # IS
    is_mask = df.index <= IS_END
    is_nav = nav[is_mask] / nav[is_mask].iloc[0]
    is_m = slice_metrics(is_nav, strategy_returns[is_mask], leverage[is_mask])
    # OOS
    oos_mask = (df.index >= OOS_START) & (df.index <= OOS_END)
    oos_nav = nav[oos_mask] / nav[oos_mask].iloc[0]
    oos_m = slice_metrics(oos_nav, strategy_returns[oos_mask], leverage[oos_mask])

    # Worst10Y★ (calendar year, FULL period)
    yearly_nav = nav.groupby(nav.index.year).last()
    worst10y = min(
        (yearly_nav.iloc[i+10]/yearly_nav.iloc[i])**(1/10) - 1
        for i in range(len(yearly_nav)-10)
    ) if len(yearly_nav) >= 11 else np.nan

    # P10_5Y▷ (daily rolling 252×5)
    window = TRADING_DAYS * 5
    rolling = [(nav.iloc[i]/nav.iloc[i-window])**(1/5)-1 for i in range(window, len(nav))]
    p10_5y = np.percentile(rolling, 10)

    return {
        'max_lev': max_lev,
        'avg_signal': float(leverage.mean()),
        'avg_eff_L': float(leverage.mean() * BASE_LEV),  # 信号 × TQQQ 3x
        'avg_SOFR_OOS': float(sofr_annual[oos_mask].mean()),
        'CAGR_FULL': full_m['CAGR'],
        'CAGR_IS':   is_m['CAGR'],
        'CAGR_OOS':  oos_m['CAGR'],
        'Sharpe_FULL': full_m['Sharpe'],
        'Sharpe_IS':   is_m['Sharpe'],
        'Sharpe_OOS':  oos_m['Sharpe'],
        'MaxDD_FULL': full_m['MaxDD'],
        'MaxDD_OOS':  oos_m['MaxDD'],
        'Worst10Y_star': worst10y,
        'P10_5Y': p10_5y,
        'IS_OOS_gap': is_m['CAGR'] - oos_m['CAGR'],
        'Trades_yr_FULL': full_m['Trades_yr'],
    }


def main():
    print('=' * 90)
    print('Ens2(Asym+Slope) — Scenario D 再実行 (proper TQQQ cost)')
    print('=' * 90)

    for max_lev in [1.0, 3.0]:
        print(f'\n--- max_lev = {max_lev} ---')
        r = run_ens2_scenario_d(max_lev=max_lev)
        print(f'  Avg signal       : {r["avg_signal"]:.4f}')
        print(f'  Avg effective L  : {r["avg_eff_L"]:.4f}x (signal × TQQQ 3x)')
        print(f'  Avg OOS SOFR     : {r["avg_SOFR_OOS"]*100:.2f}%/yr')
        print(f'  CAGR_FULL        : {r["CAGR_FULL"]*100:+.2f}%')
        print(f'  CAGR_IS          : {r["CAGR_IS"]*100:+.2f}%')
        print(f'  CAGR_OOS         : {r["CAGR_OOS"]*100:+.2f}%')
        print(f'  Sharpe_FULL      : {r["Sharpe_FULL"]:+.3f}')
        print(f'  Sharpe_OOS       : {r["Sharpe_OOS"]:+.3f}')
        print(f'  MaxDD_FULL       : {r["MaxDD_FULL"]*100:+.2f}%')
        print(f'  Worst10Y★        : {r["Worst10Y_star"]*100:+.2f}%')
        print(f'  P10_5Y▷          : {r["P10_5Y"]*100:+.2f}%')
        print(f'  IS-OOS gap       : {r["IS_OOS_gap"]*100:+.2f}pp')
        print(f'  Trades/yr        : {r["Trades_yr_FULL"]:.2f}')
        print(f'  Sharpe比 OOS/FULL: {r["Sharpe_OOS"]/r["Sharpe_FULL"]*100:.1f}%')

    # CSV 保存
    results = []
    for max_lev in [1.0, 3.0]:
        r = run_ens2_scenario_d(max_lev=max_lev)
        r['Strategy'] = f'Ens2(Asym+Slope) max_lev={max_lev}'
        results.append(r)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(BASE, 'g15b_ens2_oos_results.csv'), index=False)
    print(f'\n→ Saved: g15b_ens2_oos_results.csv')


if __name__ == '__main__':
    main()
