"""
Threshold x Tax Drag Sensitivity Analysis
==========================================
Sweeps THRESHOLD (rebalancing trigger) from 0.05 to 0.60.
For each threshold, computes:
  - Trades/year (from simulate_rebalance_A)
  - Pre-tax CAGR, Sharpe, MaxDD (from Scenario D backtest)
  - After-tax CAGR (analytical model, 3 slip-rate assumptions)

Tax model:
  r_per_trade = (1 + pretax_cagr)^(1/N) - 1
  aftertax_per_trade = r_per_trade * (1 - slip_rate * TAX_RATE)
  aftertax_cagr = (1 + aftertax_per_trade)^N - 1

slip_rate = fraction of rebalances that trigger realized taxable gains.
  0.25 = optimistic (many loss-side rebalances, good tax-loss harvesting)
  0.50 = base case (screenshot formula assumption)
  0.75 = pessimistic (mostly bull market, most rebalances realize gains)
"""

import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from corrected_strategy_backtest import (
    load_data, load_sofr, build_a2_signal,
    simulate_rebalance_A, build_gold_2x,
    build_bond_1x_nav_corrected, build_bond_3x,
    prepare_gold_data, build_nav, calc_metrics,
    DATA_PATH, DATA_DIR, TRADING_DAYS, SWAP_SPREAD, BASE_LEV, PERIODS,
)

TAX_RATE = 0.20315
SLIP_RATES = [0.25, 0.50, 0.75]
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
FULL_START = '1974-01-02'
FULL_END   = '2026-12-31'


def aftertax_cagr(pretax_cagr: float, trades_per_year: float, slip_rate: float) -> float:
    """Analytical after-tax CAGR under periodic realization model."""
    if trades_per_year <= 0 or pretax_cagr <= -1:
        return float('nan')
    r = (1 + pretax_cagr) ** (1.0 / trades_per_year) - 1
    r_at = r * (1.0 - slip_rate * TAX_RATE)
    return (1.0 + r_at) ** trades_per_year - 1


def main():
    print("Loading data...")
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change()
    dates = df['Date']
    total_years = len(df) / TRADING_DAYS
    print(f"  {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({total_years:.1f} yr)")

    print("Building A2 signal (shared)...")
    raw, vz = build_a2_signal(close, ret)

    print("Building Scenario D asset NAVs (shared)...")
    sofr      = load_sofr(dates)
    gold_1x   = prepare_gold_data(dates)
    gold_2x   = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x   = build_bond_1x_nav_corrected(
        dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x   = build_bond_3x(bond_1x, sofr, apply_sofr=True)

    print(f"\nSweeping {len(THRESHOLDS)} thresholds: {THRESHOLDS}\n")

    rows = []
    for thr in THRESHOLDS:
        lev, wn, wg, wb, n_trades = simulate_rebalance_A(raw, vz, threshold=thr)
        trades_yr = n_trades / total_years

        nav = build_nav(close, lev, wn, wg, wb, dates,
                        gold_2x, bond_3x,
                        sofr_daily=sofr, apply_tqqq_sofr=True,
                        swap_spread=SWAP_SPREAD)

        m = calc_metrics(nav, dates, FULL_START, FULL_END)
        if m is None:
            continue

        row = {
            'threshold': thr,
            'trades_total': n_trades,
            'trades_yr':    round(trades_yr, 1),
            'pretax_CAGR':  round(m['CAGR'] * 100, 3),
            'Sharpe':       round(m['Sharpe'], 3),
            'MaxDD_pct':    round(m['MaxDD'] * 100, 2),
            'Worst5Y_pct':  round(m['Worst5Y'] * 100, 2) if m['Worst5Y'] is not None else float('nan'),
            'WinRate_pct':  round(m['WinRate'] * 100, 1),
        }
        for slip in SLIP_RATES:
            at = aftertax_cagr(m['CAGR'], trades_yr, slip)
            key = f'aftertax_slip{int(slip*100):02d}'
            row[key] = round(at * 100, 3)

        rows.append(row)
        print(f"  thr={thr:.2f}  trades/yr={trades_yr:5.1f}  "
              f"pretax={m['CAGR']*100:6.2f}%  "
              f"aftertax(slip50%)={row['aftertax_slip50']:6.2f}%  "
              f"Sharpe={m['Sharpe']:.3f}  MaxDD={m['MaxDD']*100:.2f}%")

    df_out = pd.DataFrame(rows)

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE, 'threshold_tax_sensitivity_results.csv')
    df_out.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved CSV: {csv_path}")

    # --- Summary table ---
    print("\n" + "=" * 90)
    print("THRESHOLD x TAX DRAG SENSITIVITY  (Scenario D, FULL 1974-2026)")
    print("=" * 90)
    hdr = (f"{'thr':>5}  {'tr/yr':>6}  {'pretax':>8}  "
           f"{'AT_s25':>8}  {'AT_s50':>8}  {'AT_s75':>8}  "
           f"{'Sharpe':>7}  {'MaxDD':>8}  {'W5Y':>8}")
    print(hdr)
    print("-" * 90)
    baseline = None
    for r in rows:
        if abs(r['threshold'] - 0.15) < 0.001:
            baseline = r
        marker = " <-- baseline" if abs(r['threshold'] - 0.15) < 0.001 else ""
        print(f"  {r['threshold']:>4.2f}  {r['trades_yr']:>6.1f}  "
              f"{r['pretax_CAGR']:>7.2f}%  "
              f"{r['aftertax_slip25']:>7.2f}%  "
              f"{r['aftertax_slip50']:>7.2f}%  "
              f"{r['aftertax_slip75']:>7.2f}%  "
              f"{r['Sharpe']:>7.3f}  "
              f"{r['MaxDD_pct']:>7.2f}%  "
              f"{r['Worst5Y_pct']:>7.2f}%{marker}")
    print("=" * 90)

    # --- Find optimum for each slip rate ---
    print("\n[Optimal threshold per slip-rate assumption]")
    for slip in SLIP_RATES:
        key = f'aftertax_slip{int(slip*100):02d}'
        best = max(rows, key=lambda r: r[key])
        print(f"  slip={slip:.2f}: optimal thr={best['threshold']:.2f}  "
              f"trades/yr={best['trades_yr']:.1f}  "
              f"pretax={best['pretax_CAGR']:.2f}%  "
              f"aftertax={best[key]:.2f}%")

    if baseline:
        print(f"\n[vs baseline thr=0.15, trades/yr={baseline['trades_yr']:.1f}]")
        for slip in SLIP_RATES:
            key = f'aftertax_slip{int(slip*100):02d}'
            best = max(rows, key=lambda r: r[key])
            delta = best[key] - baseline[key]
            print(f"  slip={slip:.2f}: best aftertax={best[key]:.2f}% vs "
                  f"baseline={baseline[key]:.2f}%  delta={delta:+.2f}%pt")

    return df_out


if __name__ == '__main__':
    main()
