"""
Delay × Product Cost Comparison: TQQQ vs NASDAQ100 3倍ブル
===========================================================
Extends corrected_strategy_backtest.py (Scenario D) to compare:

  TQQQ       : 2×SOFR drag, TER=0.86%, swap=0.50%, DELAY=2 (baseline)
  3倍ブル(投信): 2×BOJ drag,  TER=1.52%, swap=0%,   DELAY=5 or 7

Key scenarios:
  T2: TQQQ, DELAY=2  -- current baseline (should reproduce ~22.50% CAGR)
  T5: TQQQ, DELAY=5  -- pure timing penalty (TQQQ costs, 5-day delay)
  T7: TQQQ, DELAY=7  -- pure timing penalty (TQQQ costs, 7-day delay)
  D5: 3倍ブル, DELAY=5 -- full product comparison (BOJ rate + higher TER)
  D7: 3倍ブル, DELAY=7 -- 3倍ブル worst case

Output: CAGR / Sharpe / MaxDD / Worst5Y for FULL / IS / OOS periods.
"""

import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Import shared helpers from the official corrected backtest
from corrected_strategy_backtest import (
    load_data, load_sofr, load_yield,
    build_a2_signal, simulate_rebalance_A,
    build_gold_2x, build_bond_1x_nav_corrected, build_bond_3x,
    prepare_gold_data, calc_metrics,
    DATA_PATH, DATA_DIR,
    TRADING_DAYS, THRESHOLD, SWAP_SPREAD, BASE_LEV, PERIODS
)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
TQQQ_TER   = 0.0086   # TQQQ TER  (src/product_costs.py)
TQQQ_SWAP  = 0.0050   # TQQQ swap spread
FUND3X_TER = 0.0152   # NASDAQ100 3倍ブル actual trust fee (incl. other costs)
FUND3X_SWAP = 0.0000  # Swap spread absorbed in trust fee

# -----------------------------------------------------------------------
# BOJ overnight rate: piecewise linear approximation (1974-2026)
# Source: BOJ historical call rate data, IMF
# -----------------------------------------------------------------------
_BOJ_BREAKPOINTS = [
    # (year, month, annual_rate_decimal)
    (1974,  1, 0.090),  # ~9% (1974 oil shock era)
    (1980,  1, 0.075),  # ~7.5%
    (1985,  1, 0.060),  # ~6%
    (1988,  1, 0.040),  # pre-bubble
    (1990,  6, 0.062),  # bubble peak
    (1993,  1, 0.030),  # post-bubble decline
    (1995,  7, 0.005),  # ZIRP begins
    (1999,  2, 0.001),  # formal ZIRP
    (2001,  3, 0.0001), # quantitative easing
    (2006,  7, 0.004),  # first normalization
    (2008, 12, 0.001),  # GFC rate cut
    (2010,  1, 0.001),  # maintained low
    (2016,  2, 0.0001), # negative rate policy
    (2024,  3, 0.001),  # first hike (+10bp)
    (2024,  8, 0.002),  # +20bp
    (2025,  1, 0.005),  # +50bp
    (2025,  7, 0.0075), # +75bp
    (2026,  5, 0.0075), # current (Apr 28 MPM hold)
]


def build_boj_daily_rate(dates_series: pd.Series) -> np.ndarray:
    """
    Build daily BOJ overnight rate array aligned to NASDAQ trading dates.
    Uses piecewise linear interpolation between historical breakpoints.
    Returns daily rate (annual_rate / 252).
    """
    boj_df = pd.DataFrame(_BOJ_BREAKPOINTS, columns=['year', 'month', 'rate'])
    boj_df['date'] = pd.to_datetime(
        boj_df['year'].astype(str) + '-' + boj_df['month'].astype(str).str.zfill(2) + '-01'
    )
    boj_ts = boj_df.set_index('date')['rate']

    # Full date index spanning all NASDAQ trading days
    all_dates = pd.to_datetime(dates_series.values)
    full_range = pd.date_range(all_dates.min(), all_dates.max(), freq='D')
    boj_full = boj_ts.reindex(full_range).interpolate('time').ffill().bfill()
    boj_aligned = boj_full.reindex(all_dates).ffill().bfill()
    return (boj_aligned.values / TRADING_DAYS)  # daily rate


# -----------------------------------------------------------------------
# Parametric NAV builder
# -----------------------------------------------------------------------
def build_nav_parametric(close, lev, wn, wg, wb, dates,
                          gold_2x_nav, bond_3x_nav,
                          delay: int,
                          ter: float,
                          sofr_daily: np.ndarray,
                          use_sofr: bool,
                          boj_daily: np.ndarray = None,
                          use_boj: bool = False,
                          swap_spread: float = SWAP_SPREAD) -> pd.Series:
    """
    Parametric NAV builder for delay × product comparison.

    delay     : execution delay in business days
    ter       : annual TER to deduct from NASDAQ sleeve
    sofr_daily: daily SOFR rate array
    use_sofr  : if True, apply 2×(SOFR + swap) financing drag (TQQQ)
    boj_daily : daily BOJ rate array
    use_boj   : if True, apply 2×BOJ financing drag (3倍ブル)
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x_nav).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x_nav).pct_change().fillna(0).values
    dc    = ter / TRADING_DAYS
    swap_d = swap_spread / TRADING_DAYS

    lev_s = pd.Series(lev, index=dates.index).shift(delay).fillna(0).values
    wn_s  = pd.Series(wn,  index=dates.index).shift(delay).fillna(0).values
    wg_s  = pd.Series(wg,  index=dates.index).shift(delay).fillna(0).values
    wb_s  = pd.Series(wb,  index=dates.index).shift(delay).fillna(0).values

    if use_sofr and sofr_daily is not None:
        nas_ret = r_nas * BASE_LEV - 2.0 * (sofr_daily + swap_d) - dc
    elif use_boj and boj_daily is not None:
        nas_ret = r_nas * BASE_LEV - 2.0 * boj_daily - dc
    else:
        nas_ret = r_nas * BASE_LEV - dc

    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3
    return (1 + pd.Series(daily, index=dates.index)).cumprod()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 100)
    print("DELAY × PRODUCT COMPARISON: TQQQ vs NASDAQ100 3倍ブル")
    print("Base: Scenario D (dgs30, D_var, 2×SOFR, 1×SOFR Gold, splice_fix)")
    print("=" * 100)

    # Load data
    df    = load_data(DATA_PATH)
    close = df['Close']; ret = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} → {dates.iloc[-1].date()} ({len(df):,} days)")

    # Signals (shared across all scenarios)
    print("Building A2 signal...")
    raw, vz = build_a2_signal(close, ret)
    lev, wn, wg, wb, n_tr = simulate_rebalance_A(raw, vz, THRESHOLD)
    trades_per_yr = n_tr / (len(dates) / TRADING_DAYS)
    print(f"  Trades: {n_tr:,}  ({trades_per_yr:.1f}/yr)")

    # Asset NAVs (Scenario D: time-varying duration, SOFR on gold)
    print("Loading gold & SOFR...")
    gold_1x  = prepare_gold_data(dates)
    sofr     = load_sofr(dates)
    sofr_ann = np.nanmean(sofr) * TRADING_DAYS * 100
    print(f"  Mean SOFR (52yr): {sofr_ann:.2f}%/yr")

    print("Building BOJ rate series...")
    boj = build_boj_daily_rate(dates)
    boj_ann = np.nanmean(boj) * TRADING_DAYS * 100
    print(f"  Mean BOJ (52yr):  {boj_ann:.2f}%/yr")

    print("Building Scenario D asset NAVs...")
    gold_2x_D = build_gold_2x(gold_1x, sofr_daily=sofr, apply_sofr=True)
    bond_1x_D = build_bond_1x_nav_corrected(dates, use_time_varying_duration=True, bond_maturity=22.0)
    bond_3x_D = build_bond_3x(bond_1x_D, sofr, apply_sofr=True)

    # -----------------------------------------------------------------------
    # Scenario definitions
    # -----------------------------------------------------------------------
    scenarios = [
        # (label, delay, ter, use_sofr, use_boj, swap_spread, description)
        ("T2",  2,  TQQQ_TER,   True,  False, TQQQ_SWAP,  "TQQQ  DELAY=2  [baseline]"),
        ("T5",  5,  TQQQ_TER,   True,  False, TQQQ_SWAP,  "TQQQ  DELAY=5  [pure timing penalty]"),
        ("T6",  6,  TQQQ_TER,   True,  False, TQQQ_SWAP,  "TQQQ  DELAY=6  [pure timing, actual]"),
        ("T7",  7,  TQQQ_TER,   True,  False, TQQQ_SWAP,  "TQQQ  DELAY=7  [pure timing worst]"),
        ("D5",  5,  FUND3X_TER, False, True,  FUND3X_SWAP, "3倍ブル DELAY=5  [full product comparison]"),
        ("D6",  6,  FUND3X_TER, False, True,  FUND3X_SWAP, "3倍ブル DELAY=6  [actual confirmed delay]"),
        ("D7",  7,  FUND3X_TER, False, True,  FUND3X_SWAP, "3倍ブル DELAY=7  [3倍ブル worst case]"),
    ]

    # -----------------------------------------------------------------------
    # Run all scenarios
    # -----------------------------------------------------------------------
    print("\nRunning 5 scenarios...")
    navs = {}
    for label, delay, ter, use_sofr, use_boj, swap, desc in scenarios:
        nav = build_nav_parametric(
            close, lev, wn, wg, wb, dates,
            gold_2x_D, bond_3x_D,
            delay=delay, ter=ter,
            sofr_daily=sofr, use_sofr=use_sofr,
            boj_daily=boj, use_boj=use_boj,
            swap_spread=swap
        )
        navs[label] = nav
        print(f"  [{label}] {desc}  OK")

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("RESULTS: CAGR / Sharpe / MaxDD / Worst5Y")
    print("=" * 100)

    rows = []
    for pname, pstart, pend in PERIODS:
        print(f"\n--- Period: {pname} ({pstart} → {pend}) ---")
        header = f"{'Scenario':<18} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Worst5Y':>9} {'vs T2 CAGR':>11}"
        print(header)
        print("-" * 65)

        m_t2 = calc_metrics(navs["T2"], dates, pstart, pend)
        ref_cagr = m_t2['CAGR'] if m_t2 else 0.0

        for label, delay, ter, use_sofr, use_boj, swap, desc in scenarios:
            m = calc_metrics(navs[label], dates, pstart, pend)
            if m is None:
                print(f"  {label:<16} -- insufficient data --")
                continue
            delta = (m['CAGR'] - ref_cagr) * 100
            w5y_s = f"{m['Worst5Y']*100:>8.2f}%" if m['Worst5Y'] is not None else "     N/A"
            print(f"  {desc:<38} {m['CAGR']*100:>7.2f}%  {m['Sharpe']:>7.3f}  "
                  f"{m['MaxDD']*100:>7.2f}%  {w5y_s}  {delta:>+9.2f}%pt")
            rows.append({
                'period': pname, 'scenario': label, 'description': desc,
                'delay': delay, 'use_sofr': use_sofr, 'use_boj': use_boj,
                'ter': ter,
                'CAGR%': round(m['CAGR']*100, 3),
                'vs_T2_CAGR%': round(delta, 3),
                'Sharpe': round(m['Sharpe'], 4),
                'MaxDD%': round(m['MaxDD']*100, 2),
                'Worst5Y%': round(m['Worst5Y']*100, 2) if m['Worst5Y'] is not None else None,
                'WinRate%': round(m['WinRate']*100, 1),
            })

    # -----------------------------------------------------------------------
    # Decomposition: timing penalty vs SOFR savings
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DECOMPOSITION (FULL period): Timing Penalty vs SOFR Savings")
    print("=" * 100)

    full_rows = {r['scenario']: r for r in rows if r['period'] == 'FULL'}
    if all(k in full_rows for k in ('T2','T5','T6','T7','D5','D6','D7')):
        t2 = full_rows['T2']['CAGR%']
        t5 = full_rows['T5']['CAGR%']
        t7 = full_rows['T7']['CAGR%']
        d5 = full_rows['D5']['CAGR%']
        d7 = full_rows['D7']['CAGR%']

        timing_5d = t5 - t2  # negative = cost of 5-day delay
        timing_7d = t7 - t2

        sofr_saving_5d = d5 - t5      # positive = SOFR savings (at DELAY=5)
        sofr_saving_7d = d7 - t7

        net_5d = d5 - t2              # net benefit of switching to 3倍ブル (5d)
        net_7d = d7 - t2              # net benefit of switching to 3倍ブル (7d)

        mean_sofr  = sofr_ann
        mean_boj   = boj_ann
        sofr_drag_tqqq = 2 * mean_sofr
        boj_drag_fund  = 2 * mean_boj

        print(f"\n  [Rate environment (52yr avg)]")
        print(f"    SOFR(DTB3):  {mean_sofr:.2f}%/yr  -> TQQQ financing drag: {sofr_drag_tqqq:.2f}%/yr")
        print(f"    BOJ overnight:     {mean_boj:.2f}%/yr  ->  3Fund3x financing:  {boj_drag_fund:.2f}%/yr")
        print(f"    Financing gap (TQQQ - 3xBull): {sofr_drag_tqqq - boj_drag_fund:+.2f}%pt/yr")

        print(f"\n  [FULL period CAGR decomposition]")
        print(f"    -------------------------------------------------")
        print(f"    T2  TQQQ DELAY=2 (baseline)      : {t2:>7.2f}%")
        print(f"    -------------------------------------------------")
        print(f"    Timing penalty (DELAY 2->5)         : {timing_5d:>+7.2f}%pt")
        print(f"    T5  TQQQ DELAY=5                    : {t5:>7.2f}%")
        print(f"    -------------------------------------------------")
        print(f"    Timing penalty (DELAY 2->7)         : {timing_7d:>+7.2f}%pt")
        print(f"    T7  TQQQ DELAY=7                    : {t7:>7.2f}%")
        print(f"    -------------------------------------------------")
        print(f"    SOFR->BOJ savings  (DELAY=5)     : {sofr_saving_5d:>+7.2f}%pt")
        print(f"    D5  3倍ブル DELAY=5                  : {d5:>7.2f}%")
        print(f"    -------------------------------------------------")
        print(f"    SOFR->BOJ savings  (DELAY=7)     : {sofr_saving_7d:>+7.2f}%pt")
        print(f"    D7  3倍ブル DELAY=7                  : {d7:>7.2f}%")
        print(f"    -------------------------------------------------")
        print(f"\n  [Net verdict: D5 vs T2]")
        print(f"    D5 (3xBull 5d) - T2 (TQQQ 2日)   : {net_5d:>+7.2f}%pt  ('3xBull WINS' if net_5d > 0 else 'TQQQ WINS')")
        print(f"    D7 (3xBull 7d) - T2 (TQQQ 2日)   : {net_7d:>+7.2f}%pt  ('3xBull WINS' if net_7d > 0 else 'TQQQ WINS')")

    # -----------------------------------------------------------------------
    # OOS period summary (most important for recent performance)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("OOS (2021-2026) Summary -- recent high-rate environment")
    print("=" * 100)
    oos_rows = {r['scenario']: r for r in rows if r['period'] == 'OOS'}
    if oos_rows:
        for label, _, _, _, _, _, desc in scenarios:
            if label in oos_rows:
                r = oos_rows[label]
                print(f"  {desc:<42} CAGR: {r['CAGR%']:>7.2f}%  Sharpe: {r['Sharpe']:>6.3f}  "
                      f"MaxDD: {r['MaxDD%']:>6.2f}%  vs T2: {r['vs_T2_CAGR%']:>+7.2f}%pt")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_csv = os.path.join(base_dir, 'delay_product_comparison_results.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format='%.4f')
    print(f"\nSaved CSV: {out_csv}")

    return pd.DataFrame(rows)


if __name__ == '__main__':
    results = main()
