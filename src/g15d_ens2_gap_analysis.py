"""
g15d_ens2_gap_analysis.py — Ens2 OOS CAGR ギャップ要因分析
=============================================================================
ユーザー指摘:
  YEARLY_RETURNS_REPORT_2026-04-01.md は Ens2 max_lev=3.0 で
  OOS CAGR ~+10% を示している (税・SOFR 金利コスト未計上)。
  私の g15b は OOS CAGR -11.99% で gap 約 22pp。
  「コスト+税だけでは説明つかない」というユーザー指摘を定量検証する。

分析対象ギャップ:
  Report OOS NAV (1974 から累積) → my g15b OOS NAV
  6 要因に分解:
    [F1] 期間境界の差: 暦年 2021 vs proper OOS 2021-05-08
    [F2] コスト基盤: Scenario B (TER のみ 0.86%) vs Scenario D (TER + 2×SOFR + swap)
    [F3] 信号 × cost 構造: signal-weighted vs flat
    [F4] DELAY: shift(1) vs shift(2)
    [F5] base_leverage 解釈: TQQQ 3x ベース固定 vs 信号×3x
    [F6] 実装差: DD signal / vol / slope の細部
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

NDX_CSV = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')
DTB3_CSV = os.path.join(BASE, 'data', 'dtb3_daily.csv')

TRADING_DAYS = 252

# Ens2 strategy components (test_ens2_strategies.py から複製)
def calc_asym_ewma_vol(returns, span_up=20, span_dn=5):
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.iloc[:20].var() if len(returns) > 20 else 0.0001
    for i in range(1, len(returns)):
        ret = returns.iloc[i]
        prev = variance.iloc[i-1]
        alpha = 2/(span_dn+1) if ret < 0 else 2/(span_up+1)
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


def gen_ens2_signal(close, returns, max_lev=3.0):
    dd_sig = calc_dd_signal(close)
    asym_vol = calc_asym_ewma_vol(returns)
    vt_lev = (0.25 / asym_vol).clip(0, max_lev)
    slope_mult = calc_slope_multiplier(close)
    leverage = (dd_sig * vt_lev * slope_mult).clip(0, max_lev).fillna(0)
    return leverage


def load_sofr(dates_idx):
    df = pd.read_csv(DTB3_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    val_col = [c for c in df.columns if c.lower() not in ['date']][0]
    sofr = pd.to_numeric(df[val_col], errors='coerce') / 100.0
    return sofr.reindex(dates_idx).ffill().bfill().fillna(0.04)


def run_with_config(close, returns, sofr, max_lev, cost_scenario, delay, period_def):
    """様々なコンフィグで Ens2 を実行し OOS CAGR を返す。

    cost_scenario:
      'A': コスト 0
      'B': TER 0.86% only (no SOFR financing) — YEARLY_RETURNS_REPORT 想定
      'D': TQQQ Scenario D — TER + 2×SOFR + 0.50% swap
      'D_proportional': D だが信号 ≤ 1 のみ TQQQ 適用, 信号 > 1 は追加 margin cost を加算
    delay: 1 or 2 (signal shift days)
    period_def:
      'calendar_2021': 2021-01-01 〜 2026-03-26
      'eval_standard': 2021-05-08 〜 2026-03-26
    """
    lev = gen_ens2_signal(close, returns, max_lev=max_lev)
    lev_shifted = lev.shift(delay).fillna(0)

    if cost_scenario == 'A':
        cost = pd.Series(0.0, index=close.index)
    elif cost_scenario == 'B':
        # TER のみ: signal 比例で TER だけドラッグ
        cost = pd.Series(0.0086 / TRADING_DAYS, index=close.index)
    elif cost_scenario == 'D':
        # TQQQ Scenario D: 2.0 × SOFR_daily + 0.50%/252 + 0.86%/252
        sofr_daily = sofr / TRADING_DAYS
        cost = 2.0 * sofr_daily + (0.0050 + 0.0086) / TRADING_DAYS
    elif cost_scenario == 'D_proportional':
        # signal ≤ 1 は TQQQ Scenario D の signal 倍, signal > 1 は (signal-1) × margin cost 追加
        # margin cost = SOFR + 1.5% (broker margin spread)
        sofr_daily = sofr / TRADING_DAYS
        base_cost = 2.0 * sofr_daily + (0.0050 + 0.0086) / TRADING_DAYS  # signal=1 baseline
        margin_excess = (lev_shifted - 1).clip(0)  # 1 超過分 (signal > 1 時のみ)
        margin_cost_daily = margin_excess * (sofr / TRADING_DAYS + 0.015 / TRADING_DAYS)
        cost = base_cost + margin_cost_daily / lev_shifted.replace(0, 1)  # 信号比で割って戻す
    else:
        raise ValueError(cost_scenario)

    leveraged_ret = returns * 3.0  # TQQQ 3x base
    strategy_ret = lev_shifted * (leveraged_ret - cost)
    strategy_ret = strategy_ret.fillna(0)
    nav = (1 + strategy_ret).cumprod()

    # Period
    if period_def == 'calendar_2021':
        oos_mask = (close.index >= pd.Timestamp('2021-01-01')) & (close.index <= pd.Timestamp('2026-03-26'))
    elif period_def == 'eval_standard':
        oos_mask = (close.index >= pd.Timestamp('2021-05-08')) & (close.index <= pd.Timestamp('2026-03-26'))

    oos_nav = nav[oos_mask] / nav[oos_mask].iloc[0]
    years = len(oos_nav) / TRADING_DAYS
    cagr = (oos_nav.iloc[-1] / oos_nav.iloc[0]) ** (1/years) - 1
    daily_ret = oos_nav.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS) if daily_ret.std() > 0 else 0

    return dict(
        avg_signal=float(lev_shifted.mean()),
        oos_years=years,
        cagr_oos=cagr,
        sharpe_oos=sharpe,
        nav_factor=float(oos_nav.iloc[-1]),
    )


def main():
    print('=' * 100)
    print('g15d: Ens2 OOS CAGR ギャップ要因分析')
    print('=' * 100)

    # Load data
    df = pd.read_csv(NDX_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close = df['Close']
    returns = close.pct_change().fillna(0)
    sofr = load_sofr(df.index)

    # ベースライン: YEARLY_RETURNS_REPORT 想定 (max_lev=3.0, cost B, calendar 2021, delay=2)
    print('\n[Step 1] YEARLY_RETURNS_REPORT 想定の再現を試行')
    print('  max_lev=3.0, cost=B (TER 0.86%), period=calendar_2021, delay=2')
    r1 = run_with_config(close, returns, sofr, max_lev=3.0, cost_scenario='B',
                          delay=2, period_def='calendar_2021')
    print(f'    CAGR_OOS = {r1["cagr_oos"]*100:+.2f}%  Sharpe = {r1["sharpe_oos"]:+.3f}  NAV factor = {r1["nav_factor"]:.3f}')
    print(f'    avg_signal = {r1["avg_signal"]:.3f}, OOS years = {r1["oos_years"]:.2f}')
    print(f'    → REPORT 期待値 (yearly 集計から推定): CAGR ≈ +11% (calendar 2021-2026 で 6年カバー)')

    print('\n[Step 2] my g15b 再現 (max_lev=3.0, cost D, eval_standard, delay=1)')
    r2 = run_with_config(close, returns, sofr, max_lev=3.0, cost_scenario='D',
                          delay=1, period_def='eval_standard')
    print(f'    CAGR_OOS = {r2["cagr_oos"]*100:+.2f}%  Sharpe = {r2["sharpe_oos"]:+.3f}  NAV factor = {r2["nav_factor"]:.3f}')
    print(f'    avg_signal = {r2["avg_signal"]:.3f}, OOS years = {r2["oos_years"]:.2f}')

    print('\n[Step 3] ギャップ要因の独立寄与度を測定（段階的変化）')
    print('  各ステップで1要因だけ変える → CAGR_OOS の変化量を測定')

    # 段階的に変える
    configs = [
        ('REPORT想定 (B, cal-2021, d=2)',   3.0, 'B', 'calendar_2021', 2),
        ('+ 期間 cal→eval (d=2)',          3.0, 'B', 'eval_standard', 2),
        ('+ DELAY 2→1',                    3.0, 'B', 'eval_standard', 1),
        ('+ cost B→D (= my g15b)',         3.0, 'D', 'eval_standard', 1),
    ]
    prev_cagr = None
    rows = []
    for label, mlev, cost, period, delay in configs:
        r = run_with_config(close, returns, sofr, max_lev=mlev, cost_scenario=cost, delay=delay, period_def=period)
        delta = r['cagr_oos'] - prev_cagr if prev_cagr is not None else None
        rows.append((label, r['cagr_oos'], delta, r['avg_signal'], r['nav_factor']))
        prev_cagr = r['cagr_oos']

    print(f'\n{"Config":<40s} {"CAGR_OOS":>10s} {"Δ vs prev":>10s} {"avg signal":>12s} {"NAV factor":>12s}')
    print('-' * 88)
    for label, cagr, delta, sig, nav in rows:
        delta_str = f'{delta*100:+6.2f}pp' if delta is not None else '(base)'
        print(f'{label:<40s} {cagr*100:+8.2f}% {delta_str:>10s} {sig:>12.3f} {nav:>12.3f}')

    print('\n[Step 4] max_lev 1.0 でも同様に分析')
    configs_l1 = [
        ('REPORT想定 (B, cal-2021, d=2)',   1.0, 'B', 'calendar_2021', 2),
        ('+ 期間 cal→eval (d=2)',          1.0, 'B', 'eval_standard', 2),
        ('+ DELAY 2→1',                    1.0, 'B', 'eval_standard', 1),
        ('+ cost B→D',                     1.0, 'D', 'eval_standard', 1),
    ]
    prev_cagr = None
    rows = []
    for label, mlev, cost, period, delay in configs_l1:
        r = run_with_config(close, returns, sofr, max_lev=mlev, cost_scenario=cost, delay=delay, period_def=period)
        delta = r['cagr_oos'] - prev_cagr if prev_cagr is not None else None
        rows.append((label, r['cagr_oos'], delta, r['avg_signal'], r['nav_factor']))
        prev_cagr = r['cagr_oos']

    print(f'\n{"Config (max_lev=1.0)":<40s} {"CAGR_OOS":>10s} {"Δ vs prev":>10s} {"avg signal":>12s} {"NAV factor":>12s}')
    print('-' * 88)
    for label, cagr, delta, sig, nav in rows:
        delta_str = f'{delta*100:+6.2f}pp' if delta is not None else '(base)'
        print(f'{label:<40s} {cagr*100:+8.2f}% {delta_str:>10s} {sig:>12.3f} {nav:>12.3f}')

    # Compare to YEARLY_RETURNS_REPORT yearly Ens2 returns to verify
    print('\n[Step 5] YEARLY_RETURNS_REPORT yearly Ens2 値との突合 (max_lev=3.0, Scenario B, delay=2)')
    rep_yearly = {
        2021: +22.5, 2022: -19.4, 2023: +40.4, 2024: +30.3, 2025: +13.2, 2026: -15.2
    }
    # My run with REPORT config: max_lev=3.0, cost B, delay=2, all data
    lev = gen_ens2_signal(close, returns, max_lev=3.0)
    lev_shifted = lev.shift(2).fillna(0)
    cost_b = 0.0086 / TRADING_DAYS
    strategy_ret = lev_shifted * (returns * 3.0 - cost_b)
    strategy_ret = strategy_ret.fillna(0)
    nav = (1 + strategy_ret).cumprod()
    yearly_ret = nav.groupby(nav.index.year).last().pct_change().dropna()
    print(f'\n{"Year":<6s} {"REPORT":>10s} {"My computation":>15s} {"Δ":>10s}')
    for y, rep_val in rep_yearly.items():
        if y in yearly_ret.index:
            my_val = yearly_ret[y] * 100
            print(f'{y:<6d} {rep_val:>+8.1f}% {my_val:>+13.1f}% {my_val - rep_val:>+8.1f}pp')


if __name__ == '__main__':
    main()
