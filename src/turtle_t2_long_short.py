"""T2: Pure Turtle Long/Short バックテスト on TQQQ/SQQQ synthetics 1974-2026."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turtle_data import (
    load_nasdaq, load_dtb3_aligned,
    build_tqqq_synthetic_ohlc, build_sqqq_synthetic_ohlc,
)
from turtle_sim import simulate_turtle_long_short, compute_metrics, yearly_returns


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IS_END  = pd.Timestamp('2021-05-07')
OOS_END = pd.Timestamp('2026-03-26')


def _period_slice(equity, start=None, end=None):
    s = equity
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return s


def main():
    print("[T2] Loading NASDAQ + SOFR…")
    nasdaq = load_nasdaq()
    sofr   = load_dtb3_aligned(nasdaq.index)

    print("[T2] Building TQQQ + SQQQ synthetic OHLC…")
    tqqq_ohlc = build_tqqq_synthetic_ohlc(nasdaq, sofr)
    sqqq_ohlc = build_sqqq_synthetic_ohlc(nasdaq, sofr)
    print(f"[T2] TQQQ close: {tqqq_ohlc['Close'].iloc[0]:.4f} → {tqqq_ohlc['Close'].iloc[-1]:.2f}")
    print(f"[T2] SQQQ close: {sqqq_ohlc['Close'].iloc[0]:.4f} → {sqqq_ohlc['Close'].iloc[-1]:.8f}")

    print("[T2] Running turtle long/short simulation…")
    result = simulate_turtle_long_short(tqqq_ohlc, sqqq_ohlc, nasdaq, sofr)
    eq = result.equity_series

    full = compute_metrics(eq, 'FULL')
    is_eq  = _period_slice(eq, end=IS_END)
    oos_eq = _period_slice(eq, start=IS_END + pd.Timedelta(days=1), end=OOS_END)
    is_m   = compute_metrics(is_eq, 'IS')
    oos_m  = compute_metrics(oos_eq, 'OOS')

    print("\n========== T2 SUMMARY ==========")
    print(f"FULL:  CAGR={full['cagr']*100:+.2f}%  Sharpe={full['sharpe']:.3f}  "
          f"MaxDD={full['maxdd']*100:.2f}%  Worst5Y={full['worst5y']*100:+.2f}%")
    print(f"IS  :  CAGR={is_m['cagr']*100:+.2f}%  Sharpe={is_m['sharpe']:.3f}")
    print(f"OOS :  CAGR={oos_m['cagr']*100:+.2f}%  Sharpe={oos_m['sharpe']:.3f}")
    print(f"Final equity: ${full['final_equity']:,.0f}")

    print("\n--- Trade statistics ---")
    print(f"Total entries          : {result.n_entries}")
    print(f"  ↳ short entries       : {result.n_short_entries}")
    print(f"  ↳ long entries        : {result.n_entries - result.n_short_entries}")
    print(f"2N stop hits           : {result.n_2n_stops}")
    print(f"Donchian exits (10d-S1): {result.n_donchian_exits_s1}")
    print(f"Donchian exits (20d-S2): {result.n_donchian_exits_s2}")
    print(f"S1 skip events         : {result.n_s1_skips}")
    print(f"Pyramid adds (long)    : {result.n_pyramid_adds}")
    print(f"Trades / year (avg)    : {result.n_entries / full['years']:.2f}")

    if not result.trade_log.empty:
        tl = result.trade_log
        winners = tl[tl['gross_pnl'] > 0]
        win_rate = len(winners) / len(tl) if len(tl) > 0 else 0
        print(f"Win rate               : {win_rate*100:.1f}%  ({len(winners)}/{len(tl)})")
        print(f"Avg holding days       : {tl['holding_days'].mean():.1f}")

    yr = yearly_returns(eq)
    print("\n--- Crash year returns (T2) ---")
    for y in [1981, 1988, 1994, 2000, 2008, 2015, 2022]:
        if y in yr.index:
            print(f"  {y}: {yr[y]*100:+.2f}%")

    out_yr = os.path.join(BASE, 't2_yearly_returns.csv')
    yr.to_frame('return').to_csv(out_yr)
    print(f"\nSaved: {out_yr}")

    out_tl = os.path.join(BASE, 't2_trade_log.csv')
    result.trade_log.to_csv(out_tl, index=False)
    print(f"Saved: {out_tl}")

    out_eq = os.path.join(BASE, 't2_equity_curve.csv')
    eq.to_frame('equity').to_csv(out_eq)
    print(f"Saved: {out_eq}")

    return full, is_m, oos_m, yr, result


if __name__ == '__main__':
    main()
