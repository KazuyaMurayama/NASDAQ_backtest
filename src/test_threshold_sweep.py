"""
Rebalance Threshold Sweep for Dyn 2x3x A2 Optimized.

Sweeps CONFIG.REBALANCE.THRESHOLD ∈ {0.02, 0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30}
on the currently operated Dyn 2x3x A2 Optimized strategy.

Matches GAS production behavior exactly:
  - shouldRebalance = (DD state change) or (|raw_lev - current_lev| > threshold)
  - When rebalancing: both lev AND weights update to target
  - When not: both lev AND weights are locked to previous value

Output CSV: threshold_sweep_results.csv  (columns: period, threshold, CAGR, Sharpe, MaxDD, Worst5Y, Worst10Y, WinRate, Trades)

Periods evaluated:
  - IS:  1974-01-02 to 2021-05-07  (in-sample, selection period)
  - OOS: 2021-05-08 to 2026-03-28  (out-of-sample, post-hoc confirmation only)
  - WF1: 2010-01-01 to 2015-12-31
  - WF2: 2015-01-01 to 2020-12-31
  - WF3: 2020-01-01 to 2026-12-31

Priority order of metrics: CAGR, Worst5Y, Sharpe, MaxDD, Worst10Y, WinRate, Trades
"""

import sys
import os
import types

# multitasking stub (same pattern as opt_lev2x3x.py — defuses yfinance's dependency)
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from backtest_engine import load_data, calc_dd_signal
from test_delay_robust import rebalance_threshold, calc_momentum_decel_mult  # rebalance_threshold kept for reference
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from opt_lev2x3x import build_lev_navs, calc_asym_ewma

# =============================================================================
# Constants (kept aligned with opt_lev2x3x.py — DO NOT DIVERGE)
# =============================================================================
ANNUAL_COST = 0.0086           # NASDAQ 3x annual cost
DELAY = 2                      # 2-day execution delay (matches GAS production)
BASE_LEV = 3.0                 # 3x NASDAQ
GOLD_2X_COST = 0.005           # 2036 (Gold 2x) annual cost
BOND_3X_COST = 0.0091          # TMF (Bond 3x) annual cost

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'NASDAQ_extended_to_2026.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'threshold_sweep_results.csv')

# Sweep grid (9 thresholds)
THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30]

# Dynamic allocation parameters (matches GAS Dyn 2x3x A2 Optimized settings)
#   w_nasdaq = clip(BASE_W + LEV_COEF*lev - VIX_COEF*max(vz,0), W_MIN, W_MAX)
#   w_gold   = (1 - w_nasdaq) * GOLD_RATIO
#   w_bond   = (1 - w_nasdaq) * (1 - GOLD_RATIO)
BASE_W = 0.55
LEV_COEF = 0.25
VIX_COEF = 0.10
W_MIN, W_MAX = 0.30, 0.90
GOLD_RATIO = 0.50

# Evaluation periods
PERIODS = [
    ('IS',  '1974-01-02', '2021-05-07'),
    ('OOS', '2021-05-08', '2026-03-28'),
    ('WF1', '2010-01-01', '2015-12-31'),
    ('WF2', '2015-01-01', '2020-12-31'),
    ('WF3', '2020-01-01', '2026-12-31'),
]


# =============================================================================
# A2 Optimized signal builder (mirrors opt_lev2x3x.main())
# =============================================================================
def build_a2_signals(close, returns):
    """Build the A2 Optimized 5-layer raw_leverage signal and VIX z-score.

    Layers: DD × AsymEWMA+TrendTV(VT) × SlopeMult × MomDecel × VIX_MeanReversion

    Constants kept IDENTICAL to opt_lev2x3x.main() to avoid drift.

    Returns
    -------
    raw : pd.Series  raw target leverage in [0, 1]
    vz  : pd.Series  VIX z-score, NaNs filled with 0
    """
    dd = calc_dd_signal(close, 0.82, 0.92)

    # AsymEWMA vol (slow-up=30, fast-down=10) and Trend-target-vol
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean()
    ratio = close / ma150
    ttv = (0.10 + 0.20 * (ratio - 0.85) / 0.30).clip(0.10, 0.30).fillna(0.20)
    vt = (ttv / av).clip(0, 1.0)

    # Slope multiplier
    ma200 = close.rolling(200).mean()
    sl = ma200.pct_change()
    sm = sl.rolling(60).mean()
    ss = sl.rolling(60).std().replace(0, 0.0001)
    z = (sl - sm) / ss
    slope = (0.9 + 0.35 * z).clip(0.3, 1.5).fillna(1.0)

    # Momentum deceleration
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)

    # VIX mean-reversion multiplier
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean()
    vs = vp.rolling(252).std().replace(0, 0.001)
    vz = (vp - vma) / vs
    vm = (1.0 - 0.25 * vz).clip(0.5, 1.15)

    raw = (dd * vt * slope * mom * vm).clip(0, 1.0).fillna(0)
    return raw, vz.fillna(0)


# =============================================================================
# Allocation helper — matches Code.gs compute_alloc exactly
# =============================================================================
def compute_alloc(lev, vz_val):
    """Return (w_nasdaq, w_gold, w_bond) given raw leverage and VIX z-score.

    w_nasdaq = clip(BASE_W + LEV_COEF*lev - VIX_COEF*max(vz, 0), W_MIN, W_MAX)
    """
    wn = np.clip(BASE_W + LEV_COEF * lev - VIX_COEF * max(vz_val, 0.0), W_MIN, W_MAX)
    rest = 1.0 - wn
    wg = rest * GOLD_RATIO
    wb = rest * (1.0 - GOLD_RATIO)
    return wn, wg, wb


# =============================================================================
# GAS-equivalent rebalance simulation — THE CORE ROUTINE
# =============================================================================
def simulate_gas_rebalance(raw, vz, threshold):
    """Match Code.gs:177-186 exactly.

    shouldRebalance = (DD enters 0) OR (DD leaves 0) OR (|target - current| > threshold)

    When rebalancing: BOTH leverage AND weights jump to target.
    When NOT rebalancing: BOTH are locked to the previously-held values.

    Parameters
    ----------
    raw : pd.Series  target leverage series in [0, 1]
    vz  : pd.Series  VIX z-score aligned to raw
    threshold : float

    Returns
    -------
    lev_out, wn_out, wg_out, wb_out : pd.Series (all same index as raw)
    """
    n = len(raw)
    idx = raw.index
    lev_out = pd.Series(0.0, index=idx)
    wn_out = pd.Series(0.0, index=idx)
    wg_out = pd.Series(0.0, index=idx)
    wb_out = pd.Series(0.0, index=idx)

    raw_v = raw.values
    vz_v = vz.values

    current_lev = raw_v[0]
    current_wn, current_wg, current_wb = compute_alloc(current_lev, vz_v[0])
    lev_out.iloc[0] = current_lev
    wn_out.iloc[0] = current_wn
    wg_out.iloc[0] = current_wg
    wb_out.iloc[0] = current_wb

    for i in range(1, n):
        target = raw_v[i]
        dd_to_zero = (target == 0.0) and (current_lev > 0.0)
        dd_from_zero = (current_lev == 0.0) and (target > 0.0)
        rebalance = dd_to_zero or dd_from_zero or (abs(target - current_lev) > threshold)

        if rebalance:
            current_lev = target
            current_wn, current_wg, current_wb = compute_alloc(target, vz_v[i])

        lev_out.iloc[i] = current_lev
        wn_out.iloc[i] = current_wn
        wg_out.iloc[i] = current_wg
        wb_out.iloc[i] = current_wb

    return lev_out, wn_out, wg_out, wb_out


# =============================================================================
# Portfolio NAV builder (3-asset, DELAY-shifted leverage & weights)
# =============================================================================
def build_portfolio_nav(close, gold_2x_nav, bond_3x_nav, lev, wn, wg, wb):
    """Daily strategy return =
        lev_shift_DELAY * (wn * (r_nasdaq*BASE_LEV - ANNUAL_COST/252)
                         + wg * r_gold_2x
                         + wb * r_bond_3x)

    Gold 2x / Bond 3x NAVs already incorporate their own leverage and cost,
    so we use their simple daily returns here.
    """
    r_nasdaq = close.pct_change().fillna(0).values
    r_gold = pd.Series(gold_2x_nav).pct_change().fillna(0).values
    r_bond = pd.Series(bond_3x_nav).pct_change().fillna(0).values

    dc = ANNUAL_COST / 252

    lev_shift = lev.shift(DELAY).fillna(0).values
    wn_shift = wn.shift(DELAY).fillna(0).values
    wg_shift = wg.shift(DELAY).fillna(0).values
    wb_shift = wb.shift(DELAY).fillna(0).values

    daily_ret = lev_shift * (
        wn_shift * (r_nasdaq * BASE_LEV - dc)
        + wg_shift * r_gold
        + wb_shift * r_bond
    )
    nav = (1.0 + pd.Series(daily_ret, index=close.index)).cumprod()
    return nav, pd.Series(daily_ret, index=close.index)


# =============================================================================
# 7-metric calculator for a (period-sliced) NAV
# =============================================================================
def compute_7metrics(nav, ret, lev_series, dates, period_start, period_end):
    """Compute the 7 metrics for the given sub-period.

    CAGR, Sharpe, MaxDD, Worst5Y, Worst10Y, WinRate, Trades

    Worst5Y / Worst10Y return NaN if the sub-period is shorter than
    5y / 10y respectively.
    """
    ps = pd.Timestamp(period_start)
    pe = pd.Timestamp(period_end)
    mask = (dates >= ps) & (dates <= pe)
    if mask.sum() < 20:
        return {k: np.nan for k in
                ['CAGR', 'Sharpe', 'MaxDD', 'Worst5Y', 'Worst10Y', 'WinRate', 'Trades']}

    idx = dates[mask].index
    sub_nav = nav.iloc[idx[0]:idx[-1] + 1].copy()
    sub_nav = sub_nav / sub_nav.iloc[0]
    sub_ret = sub_nav.pct_change().fillna(0)
    sub_lev = lev_series.iloc[idx[0]:idx[-1] + 1]
    sub_dates = dates.iloc[idx[0]:idx[-1] + 1]

    n_days = len(sub_nav)
    years = n_days / 252.0
    cagr = (sub_nav.iloc[-1] ** (1.0 / years)) - 1 if years > 0 and sub_nav.iloc[-1] > 0 else np.nan
    sharpe = (sub_ret.mean() * 252) / (sub_ret.std() * np.sqrt(252)) if sub_ret.std() > 0 else 0.0
    maxdd = (sub_nav / sub_nav.cummax() - 1).min()

    w5 = np.nan
    w10 = np.nan
    if n_days >= 252 * 5:
        w5 = ((sub_nav / sub_nav.shift(252 * 5)) ** (1.0 / 5) - 1).min()
    if n_days >= 252 * 10:
        w10 = ((sub_nav / sub_nav.shift(252 * 10)) ** (1.0 / 10) - 1).min()

    # Win rate on calendar-year returns
    ydf = pd.DataFrame({'nav': sub_nav.values, 'date': sub_dates.values})
    ydf['year'] = pd.to_datetime(ydf['date']).dt.year
    year_last = ydf.groupby('year')['nav'].last()
    annual = year_last.pct_change().dropna()
    winrate = (annual > 0).mean() if len(annual) > 0 else np.nan

    trades = int((sub_lev.diff().abs() > 0.01).sum())

    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'MaxDD': maxdd,
        'Worst5Y': w5,
        'Worst10Y': w10,
        'WinRate': winrate,
        'Trades': trades,
    }


# =============================================================================
# Output formatters
# =============================================================================
def _fmt_pct(x, width=6, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return f"{'N/A':>{width}}"
    fmt = f"{{:>{width - 1}.1f}}%"
    val = x * 100
    if sign:
        return f"{val:+.1f}%".rjust(width)
    return fmt.format(val)


def print_period_table(period, rows):
    print(f"\n--- {period} ---")
    print(f"{'THRESHOLD':>10} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} "
          f"{'W5Y':>8} {'W10Y':>8} {'WinRate':>8} {'Trades':>7}")
    for r in rows:
        cagr_s = _fmt_pct(r['CAGR'], 7)
        sh_s = f"{r['Sharpe']:>7.3f}" if not np.isnan(r['Sharpe']) else f"{'N/A':>7}"
        dd_s = _fmt_pct(r['MaxDD'], 7)
        w5_s = _fmt_pct(r['Worst5Y'], 7, sign=True)
        w10_s = _fmt_pct(r['Worst10Y'], 7, sign=True)
        wr_s = _fmt_pct(r['WinRate'], 7)
        tr_s = f"{r['Trades']:>7d}"
        print(f"  {r['threshold']:>8.2f} {cagr_s:>7} {sh_s} {dd_s:>7} "
              f"{w5_s:>7} {w10_s:>7} {wr_s:>7} {tr_s}")


# =============================================================================
# Main
# =============================================================================
def main():
    """Run the full threshold sweep across all 5 periods."""
    print("=== Threshold Sweep: Dyn 2x3x A2 Optimized ===")
    df = load_data(DATA_PATH)
    close = df['Close']
    returns = close.pct_change()
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} days)\n")

    # Build A2 signals once (they are threshold-independent)
    print("Building A2 optimized signals...")
    raw, vz = build_a2_signals(close, returns)

    # Build Gold 2x / Bond 3x NAVs once
    print("Fetching Gold/Bond data and building 2x/3x NAVs...")
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    gold_2x, bond_3x = build_lev_navs(gold_1x, bond_1x)

    all_rows = []
    per_period_rows = {name: [] for name, _, _ in PERIODS}

    print(f"\nSweeping {len(THRESHOLDS)} thresholds × {len(PERIODS)} periods "
          f"= {len(THRESHOLDS) * len(PERIODS)} evaluations...")

    for th in THRESHOLDS:
        lev, wn, wg, wb = simulate_gas_rebalance(raw, vz, th)
        nav, ret = build_portfolio_nav(close, gold_2x, bond_3x, lev, wn, wg, wb)

        for name, ps, pe in PERIODS:
            m = compute_7metrics(nav, ret, lev, dates, ps, pe)
            row = {'period': name, 'threshold': th, **m}
            all_rows.append(row)
            per_period_rows[name].append(row)

    # Print per-period tables
    for name, ps, pe in PERIODS:
        label = f"{name} ({ps} to {pe})" if name in ('IS', 'OOS') else name
        print_period_table(label, per_period_rows[name])

    # Save CSV
    out_df = pd.DataFrame(all_rows, columns=[
        'period', 'threshold', 'CAGR', 'Sharpe', 'MaxDD',
        'Worst5Y', 'Worst10Y', 'WinRate', 'Trades'
    ])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} rows to {OUTPUT_CSV}")

    # Best-per-period summary (Sharpe basis)
    print("\nBest per period (Sharpe基準):")
    for name, _, _ in PERIODS:
        rows = per_period_rows[name]
        valid = [r for r in rows if not (isinstance(r['Sharpe'], float) and np.isnan(r['Sharpe']))]
        if not valid:
            print(f"  {name}: N/A")
            continue
        best = max(valid, key=lambda r: r['Sharpe'])
        print(f"  {name}: T={best['threshold']:.2f} (Sharpe={best['Sharpe']:.3f}, "
              f"CAGR={best['CAGR'] * 100:.1f}%)")


if __name__ == '__main__':
    main()
