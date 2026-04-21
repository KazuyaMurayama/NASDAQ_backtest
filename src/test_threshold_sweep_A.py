"""
Rebalance threshold sweep — Approach A only (2026-04-21)
========================================================
DH Dyn 2x3x [A] (sleeve-independent) 内で閾値を振る。

式: daily = wn*lev*(r_nas*3 - dc) + wg*r_g2 + wb*r_b3
    (Gold2x/Bond3x に lev を掛けない)

Approach B は含めない。Approach A 内比較のみ。
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
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from opt_lev2x3x import build_lev_navs, calc_asym_ewma

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

ANNUAL_COST = 0.0086
DELAY = 2
BASE_LEV = 3.0
TRADING_DAYS = 252

THRESHOLDS = [0.10, 0.15, 0.20, 0.30, 0.40]

PERIODS = [
    ('IS',  '1974-01-02', '2021-05-07'),
    ('OOS', '2021-05-08', '2026-12-31'),
    ('WF1', '2010-01-01', '2014-12-31'),
    ('WF2', '2015-01-01', '2019-12-31'),
    ('WF3', '2020-01-01', '2026-12-31'),
    ('FULL','1974-01-02', '2026-12-31'),
]


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


def simulate_rebalance_A(raw, vz, threshold):
    """Approach A: produce (lev, wn, wg, wb) series with GAS threshold rule.
    Counts rebalance events for Trades/year metric.
    """
    n = len(raw)
    raw_v = raw.values
    vz_v = vz.values
    lev = np.zeros(n)
    wn = np.zeros(n)
    wg = np.zeros(n)
    wb = np.zeros(n)

    cur_lev = raw_v[0]
    cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[0], 0), 0.30, 0.90))
    cur_wg = (1 - cur_wn) * 0.5
    cur_wb = (1 - cur_wn) * 0.5
    lev[0] = cur_lev; wn[0] = cur_wn; wg[0] = cur_wg; wb[0] = cur_wb

    n_trades = 0
    for i in range(1, n):
        t = raw_v[i]
        dd_to_0 = (t == 0 and cur_lev > 0)
        dd_from_0 = (cur_lev == 0 and t > 0)
        if dd_to_0 or dd_from_0 or abs(t - cur_lev) > threshold:
            cur_lev = t
            cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[i], 0), 0.30, 0.90))
            cur_wg = (1 - cur_wn) * 0.5
            cur_wb = (1 - cur_wn) * 0.5
            n_trades += 1
        lev[i] = cur_lev; wn[i] = cur_wn; wg[i] = cur_wg; wb[i] = cur_wb

    return lev, wn, wg, wb, n_trades


def build_nav_A(close, gold_2x, bond_3x, lev, wn, wg, wb, dates):
    r_nas = close.pct_change().fillna(0).values
    r_g2 = pd.Series(gold_2x).pct_change().fillna(0).values
    r_b3 = pd.Series(bond_3x).pct_change().fillna(0).values
    dc = ANNUAL_COST / TRADING_DAYS
    lev_s = pd.Series(lev, index=dates.index).shift(DELAY).fillna(0).values
    wn_s = pd.Series(wn, index=dates.index).shift(DELAY).fillna(0).values
    wg_s = pd.Series(wg, index=dates.index).shift(DELAY).fillna(0).values
    wb_s = pd.Series(wb, index=dates.index).shift(DELAY).fillna(0).values
    # Approach A: gold/bond NOT multiplied by lev
    daily = wn_s * lev_s * (r_nas * BASE_LEV - dc) + wg_s * r_g2 + wb_s * r_b3
    nav = (1 + pd.Series(daily, index=dates.index)).cumprod()
    return nav


def calc_metrics(nav, dates, start, end, n_trades_total, full_start, full_end):
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if mask.sum() < 20:
        return {}
    idx = dates[mask].index
    n_sub = nav.loc[idx[0]:idx[-1]].copy()
    n_sub = n_sub / n_sub.iloc[0]
    r = n_sub.pct_change().fillna(0)
    n = len(n_sub)
    yrs = n / TRADING_DAYS
    if yrs <= 0 or n_sub.iloc[-1] <= 0:
        cagr = np.nan
    else:
        cagr = n_sub.iloc[-1] ** (1 / yrs) - 1
    # Sharpe
    sh = (r.mean() * TRADING_DAYS) / (r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0
    # MaxDD
    maxdd = (n_sub / n_sub.cummax() - 1).min()
    # Worst5Y rolling
    if len(n_sub) >= TRADING_DAYS * 5:
        w5 = ((n_sub / n_sub.shift(TRADING_DAYS * 5)) ** (1 / 5) - 1).min()
    else:
        w5 = np.nan
    # Worst10Y rolling
    if len(n_sub) >= TRADING_DAYS * 10:
        w10 = ((n_sub / n_sub.shift(TRADING_DAYS * 10)) ** (1 / 10) - 1).min()
    else:
        w10 = np.nan
    # WinRate — yearly
    df_y = pd.DataFrame({'nav': n_sub.values, 'date': dates.loc[idx[0]:idx[-1]].values})
    df_y['year'] = pd.to_datetime(df_y['date']).dt.year
    yn = df_y.groupby('year')['nav'].last()
    yr = yn.pct_change()
    yr.iloc[0] = yn.iloc[0] / df_y['nav'].iloc[0] - 1
    win_rate = (yr > 0).mean() if len(yr) > 0 else np.nan
    # Trades/year — scale from full-period count by this period's fraction
    full_mask = (dates >= pd.Timestamp(full_start)) & (dates <= pd.Timestamp(full_end))
    full_n = full_mask.sum()
    full_yrs = full_n / TRADING_DAYS if full_n > 0 else 1
    trades_per_year_full = n_trades_total / full_yrs
    return {
        'CAGR': cagr, 'Sharpe': sh, 'MaxDD': maxdd,
        'Worst5Y': w5, 'Worst10Y': w10,
        'WinRate': win_rate, 'TradesPerYear': trades_per_year_full,
        'Years': yrs,
    }


def main():
    df = load_data(DATA_PATH)
    close = df['Close']; returns = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df)} days, {len(df)/252:.2f} yrs)")

    raw, vz = build_a2_signal(close, returns)
    gold_1x = prepare_gold_data(dates)
    bond_1x = prepare_bond_data(dates)
    gold_2x, bond_3x = build_lev_navs(gold_1x, bond_1x)

    full_start = dates.iloc[0].strftime('%Y-%m-%d')
    full_end = dates.iloc[-1].strftime('%Y-%m-%d')

    rows = []
    navs_by_threshold = {}

    for th in THRESHOLDS:
        lev, wn, wg, wb, n_trades = simulate_rebalance_A(raw, vz, th)
        nav = build_nav_A(close, gold_2x, bond_3x, lev, wn, wg, wb, dates)
        navs_by_threshold[th] = nav

        for pname, pstart, pend in PERIODS:
            m = calc_metrics(nav, dates, pstart, pend, n_trades, full_start, full_end)
            if not m:
                continue
            rows.append({
                'threshold': th, 'period': pname,
                'start': pstart, 'end': pend,
                'years': m['Years'],
                'CAGR': m['CAGR'],
                'Worst5Y': m['Worst5Y'],
                'Sharpe': m['Sharpe'],
                'MaxDD': m['MaxDD'],
                'Worst10Y': m['Worst10Y'],
                'WinRate': m['WinRate'],
                'TradesPerYear': m['TradesPerYear'],
            })

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(BASE, 'threshold_sweep_A_results.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # --- DOUBLE CHECK against YEARLY_RETURNS_REPORT v2 (threshold 0.20) ---
    print("\n" + "=" * 80)
    print("DOUBLE-CHECK (Approach A, threshold=0.20 vs YEARLY_RETURNS_REPORT v2)")
    print("=" * 80)
    expected = {'FULL': 0.3030, 'IS': 0.3089, 'OOS': 0.2515}
    check_rows = df_out[df_out['threshold'] == 0.20]
    all_ok = True
    for pname, exp in expected.items():
        r = check_rows[check_rows['period'] == pname]
        if r.empty:
            print(f"  [{pname}] NOT FOUND — FAIL")
            all_ok = False; continue
        actual = r.iloc[0]['CAGR']
        diff = abs(actual - exp) * 100
        ok = diff < 0.05
        status = 'OK' if ok else f'FAIL diff={diff:.3f}pp'
        print(f"  [{pname}] expected={exp*100:+.2f}%  actual={actual*100:+.2f}%  {status}")
        if not ok: all_ok = False
    print(f"\nOverall double-check: {'PASS' if all_ok else 'FAIL'}")

    # --- Print summary tables ---
    for metric in ['CAGR', 'MaxDD', 'Sharpe', 'Worst5Y', 'Worst10Y', 'WinRate', 'TradesPerYear']:
        print(f"\n=== {metric} ===")
        piv = df_out.pivot(index='threshold', columns='period', values=metric)
        piv = piv[['FULL', 'IS', 'OOS', 'WF1', 'WF2', 'WF3']]
        print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    return df_out, all_ok


if __name__ == '__main__':
    main()
