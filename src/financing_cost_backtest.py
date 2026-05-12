"""
DH Dyn 2x3x [A] - Financing Cost Correction Backtest
=====================================================
Key finding: prepare_bond_data() already includes bond coupons.
Missing piece: 2x SOFR borrowing cost on both TQQQ and TMF sleeves.

TQQQ: 3*NDX_price_ret - 2*(SOFR+spread) - TER (NDX is price-only, no dividends)
TMF:  3*(bond_total_ret) - 2*(SOFR+spread) - TER  (coupon already in bond_total_ret)

Baseline (current): no SOFR drag on either sleeve.
Corrected:         subtract 2*(SOFR+swap_spread)/252 daily from each sleeve.
"""
import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data, calc_dd_signal
from test_delay_robust import calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from opt_lev2x3x import build_lev_navs, calc_asym_ewma

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

ANNUAL_COST   = 0.0086   # TQQQ TER
GOLD_2X_COST  = 0.0050   # Gold 2x TER
BOND_3X_COST  = 0.0091   # TMF TER
DELAY         = 2
BASE_LEV      = 3.0
TRADING_DAYS  = 252
THRESHOLD     = 0.15     # Approach A

# Swap spread: ProShares empirical ~50bps/yr above SOFR
SWAP_SPREAD   = 0.0050

PERIODS = [
    ('FULL', '1974-01-02', '2026-12-31'),
    ('IS',   '1974-01-02', '2021-05-07'),
    ('OOS',  '2021-05-08', '2026-12-31'),
    ('WF3',  '2020-01-01', '2026-12-31'),
]


# ---------------------------------------------------------------------------
# SOFR loader (DTB3 = 3M T-bill as proxy)
# ---------------------------------------------------------------------------

def load_sofr(dates_series: pd.Series) -> np.ndarray:
    """
    Load DTB3 (3M T-bill) as SOFR proxy.
    Returns daily rate as decimal (annual_pct / 100 / 252), aligned to dates_series.
    """
    path = os.path.join(DATA_DIR, 'dtb3_daily.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.columns = ['yield_pct']
    s = pd.to_numeric(df['yield_pct'], errors='coerce').ffill(limit=5).bfill(limit=5)

    # Align to NASDAQ calendar
    dates_ts = pd.Series(dates_series.values, index=range(len(dates_series)))
    dates_ts.index = dates_series.index
    aligned = s.reindex(dates_series.values).ffill(limit=5).bfill(limit=5)
    daily_rate = (aligned.values / 100.0) / TRADING_DAYS
    return daily_rate


# ---------------------------------------------------------------------------
# Signal builder (identical to test_threshold_sweep_A.py)
# ---------------------------------------------------------------------------

def build_a2_signal(close, returns):
    dd = calc_dd_signal(close, 0.82, 0.92)
    av = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean()
    ratio = close / ma150
    ttv = (0.10 + 0.20 * (ratio - 0.85) / 0.30).clip(0.10, 0.30).fillna(0.20)
    vt = (ttv / av).clip(0, 1.0)
    ma200 = close.rolling(200).mean()
    sl = ma200.pct_change()
    sm = sl.rolling(60).mean()
    ss = sl.rolling(60).std().replace(0, 0.0001)
    z = (sl - sm) / ss
    slope = (0.9 + 0.35 * z).clip(0.3, 1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp = calc_vix_proxy(returns)
    vma = vp.rolling(252).mean()
    vs = vp.rolling(252).std().replace(0, 0.001)
    vz = (vp - vma) / vs
    vm = (1.0 - 0.25 * vz).clip(0.5, 1.15)
    raw = (dd * vt * slope * mom * vm).clip(0, 1.0).fillna(0)
    return raw, vz.fillna(0)


# ---------------------------------------------------------------------------
# Approach A threshold rebalancer (identical to test_threshold_sweep_A.py)
# ---------------------------------------------------------------------------

def simulate_rebalance_A(raw, vz, threshold=THRESHOLD):
    n = len(raw)
    raw_v = raw.values; vz_v = vz.values
    lev = np.zeros(n); wn = np.zeros(n)
    wg = np.zeros(n); wb = np.zeros(n)

    cur_lev = raw_v[0]
    cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[0], 0), 0.30, 0.90))
    cur_wg = (1 - cur_wn) * 0.5; cur_wb = (1 - cur_wn) * 0.5
    lev[0] = cur_lev; wn[0] = cur_wn; wg[0] = cur_wg; wb[0] = cur_wb
    n_trades = 0

    for i in range(1, n):
        t = raw_v[i]
        dd_to_0   = (t == 0 and cur_lev > 0)
        dd_from_0 = (cur_lev == 0 and t > 0)
        if dd_to_0 or dd_from_0 or abs(t - cur_lev) > threshold:
            cur_lev = t
            cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[i], 0), 0.30, 0.90))
            cur_wg = (1 - cur_wn) * 0.5; cur_wb = (1 - cur_wn) * 0.5
            n_trades += 1
        lev[i] = cur_lev; wn[i] = cur_wn; wg[i] = cur_wg; wb[i] = cur_wb

    return lev, wn, wg, wb, n_trades


# ---------------------------------------------------------------------------
# Corrected B3x NAV builder
# ---------------------------------------------------------------------------

def build_corrected_b3x(bond_nav_1x: np.ndarray, sofr_daily: np.ndarray,
                         swap_spread: float = SWAP_SPREAD,
                         k_dur: float = 1.0) -> np.ndarray:
    """
    Bond 3x NAV with 2*(SOFR+swap) subtracted and optional duration scaling.

    bond_nav_1x: total return (price + coupon), from prepare_bond_data.
    k_dur: duration multiplier. Default 1.0 = no change.
           k_dur=2.21 calibrated from actual TMF (2009-2026):
           prepare_bond_data uses 10yr duration=7, TMF tracks 20+yr duration~15yr.
    """
    n = len(bond_nav_1x)
    swap_d = swap_spread / TRADING_DAYS
    b3 = np.ones(n)
    for i in range(1, n):
        br = (bond_nav_1x[i] / bond_nav_1x[i-1] - 1
              if bond_nav_1x[i-1] > 0 else 0.0)
        # Apply duration scaling to the 1x total return before 3x leverage
        b3_ret = (br * k_dur) * 3 - 2.0 * (sofr_daily[i] + swap_d) - BOND_3X_COST / TRADING_DAYS
        b3[i] = b3[i-1] * (1 + b3_ret)
    return b3


# ---------------------------------------------------------------------------
# NAV builders
# ---------------------------------------------------------------------------

def build_nav_baseline(close, gold_2x, bond_3x_baseline,
                       lev, wn, wg, wb, dates):
    """Current simulation: no SOFR drag on either sleeve."""
    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x_baseline).pct_change().fillna(0).values
    dc    = ANNUAL_COST / TRADING_DAYS

    lev_s = pd.Series(lev, index=dates.index).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn,  index=dates.index).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg,  index=dates.index).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb,  index=dates.index).shift(DELAY).fillna(0).values

    daily = wn_s * lev_s * (r_nas * BASE_LEV - dc) + wg_s * r_g2 + wb_s * r_b3
    return (1 + pd.Series(daily, index=dates.index)).cumprod()


def build_nav_corrected(close, gold_2x, bond_3x_corrected,
                        lev, wn, wg, wb, dates, sofr_daily,
                        swap_spread: float = SWAP_SPREAD):
    """
    Corrected simulation:
    - TQQQ sleeve: 3*r_nas - 2*(sofr+swap) - TER
    - TMF sleeve: 3*(total_bond_ret) - 2*(sofr+swap) - TER  [already in bond_3x_corrected]
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x_corrected).pct_change().fillna(0).values
    dc    = ANNUAL_COST / TRADING_DAYS
    swap_d = swap_spread / TRADING_DAYS

    lev_s = pd.Series(lev, index=dates.index).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn,  index=dates.index).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg,  index=dates.index).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb,  index=dates.index).shift(DELAY).fillna(0).values

    # TQQQ: additionally subtract 2*(SOFR+swap) per day
    nas_daily = r_nas * BASE_LEV - 2.0 * (sofr_daily + swap_d) - dc

    daily = wn_s * lev_s * nas_daily + wg_s * r_g2 + wb_s * r_b3
    return (1 + pd.Series(daily, index=dates.index)).cumprod()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calc_metrics(nav, dates, start, end):
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if mask.sum() < 100:
        return None
    idx = dates[mask].index
    ns = nav.loc[idx[0]:idx[-1]].copy() / nav.loc[idx[0]]
    r  = ns.pct_change().fillna(0)
    n  = len(ns); yrs = n / TRADING_DAYS
    cagr  = float(ns.iloc[-1]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sh    = (r.mean() * TRADING_DAYS) / (r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    maxdd = (ns / ns.cummax() - 1).min()
    w5    = ((ns / ns.shift(TRADING_DAYS * 5)) ** 0.2 - 1).min() if n >= TRADING_DAYS * 5 else np.nan
    df_y  = pd.DataFrame({'nav': ns.values, 'dt': dates.loc[idx[0]:idx[-1]].values})
    df_y['year'] = pd.to_datetime(df_y['dt']).dt.year
    yn = df_y.groupby('year')['nav'].last()
    wr = (yn.pct_change().dropna() > 0).mean()
    return dict(CAGR=cagr, Sharpe=sh, MaxDD=maxdd, Worst5Y=w5, WinRate=wr, Years=yrs)


# ---------------------------------------------------------------------------
# Summary: SOFR mean for period
# ---------------------------------------------------------------------------

def sofr_mean_annual(sofr_daily, dates, start, end):
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return float(np.nanmean(sofr_daily[mask.values])) * TRADING_DAYS * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change()
    dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)")

    print("Building A2 signal...")
    raw, vz = build_a2_signal(close, ret)

    print("Loading assets...")
    gold_1x    = prepare_gold_data(dates)
    bond_1x    = prepare_bond_data(dates)   # total return (price + coupon)
    gold_2x, bond_3x_base = build_lev_navs(gold_1x, bond_1x)

    print("Loading SOFR (DTB3)...")
    sofr_daily = load_sofr(dates)
    print(f"  SOFR mean (full period): {np.nanmean(sofr_daily)*TRADING_DAYS*100:.2f}%/yr")

    print("Building corrected B3x NAV (2x SOFR on TMF)...")
    bond_3x_corr = build_corrected_b3x(bond_1x, sofr_daily)

    print("Running Approach A (threshold=0.15)...")
    lev, wn, wg, wb, n_trades = simulate_rebalance_A(raw, vz, THRESHOLD)
    tpy = n_trades / (len(dates) / TRADING_DAYS)
    print(f"  Total trades: {n_trades}, {tpy:.1f}/yr")

    print("Building NAVs...")
    nav_base = build_nav_baseline(close, gold_2x, bond_3x_base,
                                  lev, wn, wg, wb, dates)
    nav_corr = build_nav_corrected(close, gold_2x, bond_3x_corr,
                                   lev, wn, wg, wb, dates, sofr_daily)

    # Print comparison table
    print("\n" + "=" * 90)
    print(f"{'DH Dyn 2x3x [A] - Financing Cost Impact  (swap_spread={SWAP_SPREAD*100:.1f}bps/yr)'}")
    print("=" * 90)
    header = (f"{'Period':<8} | {'CAGR_base':>10} {'CAGR_corr':>10} {'dCAGR':>7} | "
              f"{'Sharpe_b':>9} {'Sharpe_c':>9} {'dSharpe':>8} | "
              f"{'MaxDD_b':>8} {'MaxDD_c':>8} | "
              f"{'SOFR_avg':>9}")
    print(header)
    print("-" * 90)

    rows = []
    for pname, pstart, pend in PERIODS:
        mb = calc_metrics(nav_base, dates, pstart, pend)
        mc = calc_metrics(nav_corr, dates, pstart, pend)
        if mb is None or mc is None:
            continue
        sm = sofr_mean_annual(sofr_daily, dates, pstart, pend)
        dc = mc['CAGR'] - mb['CAGR']
        ds = mc['Sharpe'] - mb['Sharpe']
        print(f"{pname:<8} | {mb['CAGR']*100:>9.2f}% {mc['CAGR']*100:>9.2f}% {dc*100:>+6.2f}% | "
              f"{mb['Sharpe']:>9.3f} {mc['Sharpe']:>9.3f} {ds:>+8.3f} | "
              f"{mb['MaxDD']*100:>7.2f}% {mc['MaxDD']*100:>7.2f}% | "
              f"{sm:>8.2f}%")
        rows.append({
            'period': pname,
            'CAGR_base': round(mb['CAGR']*100, 2),
            'CAGR_corr': round(mc['CAGR']*100, 2),
            'dCAGR': round(dc*100, 2),
            'Sharpe_base': round(mb['Sharpe'], 3),
            'Sharpe_corr': round(mc['Sharpe'], 3),
            'dSharpe': round(ds, 3),
            'MaxDD_base': round(mb['MaxDD']*100, 2),
            'MaxDD_corr': round(mc['MaxDD']*100, 2),
            'Worst5Y_base': round(mb['Worst5Y']*100, 2) if not np.isnan(mb['Worst5Y']) else np.nan,
            'Worst5Y_corr': round(mc['Worst5Y']*100, 2) if not np.isnan(mc['Worst5Y']) else np.nan,
            'WinRate_base': round(mb['WinRate']*100, 1),
            'WinRate_corr': round(mc['WinRate']*100, 1),
            'SOFR_avg_pct': round(sm, 2),
        })
    print("=" * 90)

    # Sensitivity sweep for swap_spread
    print("\n=== Swap Spread Sensitivity (FULL period) ===")
    print(f"{'swap_spread':>12} | {'CAGR_corr':>10} | {'Sharpe_corr':>11} | {'dCAGR':>7}")
    print("-" * 50)
    pstart, pend = PERIODS[0][1], PERIODS[0][2]
    for ss in [0.0, 0.0030, 0.0050, 0.0070, 0.0100]:
        b3c = build_corrected_b3x(bond_1x, sofr_daily, swap_spread=ss)
        nc  = build_nav_corrected(close, gold_2x, b3c, lev, wn, wg, wb, dates,
                                   sofr_daily, swap_spread=ss)
        mc  = calc_metrics(nc, dates, pstart, pend)
        mb  = calc_metrics(nav_base, dates, pstart, pend)
        if mc and mb:
            dc = mc['CAGR'] - mb['CAGR']
            print(f"{ss*100:>11.2f}% | {mc['CAGR']*100:>9.2f}% | {mc['Sharpe']:>11.3f} | {dc*100:>+6.2f}%")

    # k_dur scenarios (2009-calibrated value = 2.21)
    print("\n=== Duration Multiplier Scenarios (k_dur, swap_spread=50bps, FULL period) ===")
    print("  Note: k_dur=2.21 calibrated from actual TMF vs 10yr bond model (2009-2026)")
    print(f"  {'k_dur':>6} | {'CAGR':>8} | {'Sharpe':>7} | {'MaxDD':>7} | {'WinRate':>8} | {'dCAGR':>7}")
    print("  " + "-" * 65)
    for kd in [1.0, 1.5, 2.0, 2.21, 2.5]:
        b3c = build_corrected_b3x(bond_1x, sofr_daily, swap_spread=SWAP_SPREAD, k_dur=kd)
        nc  = build_nav_corrected(close, gold_2x, b3c, lev, wn, wg, wb, dates, sofr_daily)
        mc  = calc_metrics(nc, dates, pstart, pend)
        mb_f = calc_metrics(nav_base, dates, pstart, pend)
        if mc and mb_f:
            tag = " <-- calibrated" if kd == 2.21 else ""
            dc  = mc['CAGR'] - mb_f['CAGR']
            print(f"  {kd:>6.2f} | {mc['CAGR']*100:>7.2f}% | {mc['Sharpe']:>7.3f} | "
                  f"{mc['MaxDD']*100:>7.2f}% | {mc['WinRate']*100:>7.1f}% | "
                  f"{dc*100:>+6.2f}%{tag}")

    # Save
    out_path = os.path.join(BASE, 'financing_cost_results.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nSaved: {out_path}")

    return nav_base, nav_corr, pd.DataFrame(rows)


if __name__ == '__main__':
    main()
