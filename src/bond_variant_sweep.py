"""
Bond Leverage Variant Sweep — DH Dyn 2x3x [A]
==============================================
Approach A 戦略の Bond スリーブのレバレッジ/構成を変えて比較する。
現行ベスト (V0: G2x*0.5 + B3x*0.5) vs 7バリアント。

検証した問題: Bond 3x の CAGR (1.1%) < Bond 1x の CAGR (1.4%) over 52 years.
"""
import sys, os, types

# multitasking shim
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
from test_delay_robust import calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data, prepare_bond_data
from opt_lev2x3x import calc_asym_ewma

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

ANNUAL_COST   = 0.0086   # NASDAQ 3x
TRADING_DAYS  = 252
DELAY         = 2
BASE_LEV      = 3.0
THRESHOLD_A   = 0.15     # Approach A rebalancing threshold

PERIODS = [
    ('FULL', '1974-01-02', '2026-12-31'),
    ('IS',   '1974-01-02', '2021-05-07'),
    ('OOS',  '2021-05-08', '2026-12-31'),
    ('WF3',  '2020-01-01', '2026-12-31'),
]

# ---- Asset NAV builders -------------------------------------------------------

def build_asset_navs(gold_prices: np.ndarray, bond_nav_1x: np.ndarray) -> dict:
    """Build all asset NAV series needed for variants."""
    n = len(gold_prices)
    navs = {}

    # Gold 1x (minimal ETF cost 0.40%/yr = GLD)
    g1 = np.ones(n)
    for i in range(1, n):
        gr = gold_prices[i] / gold_prices[i-1] - 1 if gold_prices[i-1] > 0 else 0
        g1[i] = g1[i-1] * (1 + gr - 0.0040 / TRADING_DAYS)
    navs['G1x'] = g1

    # Gold 2x (cost 0.50%/yr, same as existing)
    g2 = np.ones(n)
    for i in range(1, n):
        gr = gold_prices[i] / gold_prices[i-1] - 1 if gold_prices[i-1] > 0 else 0
        g2[i] = g2[i-1] * (1 + gr * 2 - 0.0050 / TRADING_DAYS)
    navs['G2x'] = g2

    # Bond 1x (hold the bond NAV directly, minimal cost)
    b1 = np.ones(n)
    for i in range(1, n):
        br = bond_nav_1x[i] / bond_nav_1x[i-1] - 1 if bond_nav_1x[i-1] > 0 else 0
        b1[i] = b1[i-1] * (1 + br - 0.0015 / TRADING_DAYS)   # IEF-like 0.15%/yr
    navs['B1x'] = b1

    # Bond 2x (cost 0.60%/yr interpolated)
    b2 = np.ones(n)
    for i in range(1, n):
        br = bond_nav_1x[i] / bond_nav_1x[i-1] - 1 if bond_nav_1x[i-1] > 0 else 0
        b2[i] = b2[i-1] * (1 + br * 2 - 0.0060 / TRADING_DAYS)
    navs['B2x'] = b2

    # Bond 3x (cost 0.91%/yr — existing baseline)
    b3 = np.ones(n)
    for i in range(1, n):
        br = bond_nav_1x[i] / bond_nav_1x[i-1] - 1 if bond_nav_1x[i-1] > 0 else 0
        b3[i] = b3[i-1] * (1 + br * 3 - 0.0091 / TRADING_DAYS)
    navs['B3x'] = b3

    return navs


# ---- Variant definitions -------------------------------------------------------

VARIANTS = [
    # name, safe1_key, safe1_ratio, safe2_key, safe2_ratio, wn_max
    ('V0_baseline',     'G2x', 0.50, 'B3x', 0.50, 0.90),  # current best
    ('V1_bond1x',       'G2x', 0.50, 'B1x', 0.50, 0.90),  # bond deleveraged
    ('V2_bond2x',       'G2x', 0.50, 'B2x', 0.50, 0.90),  # bond 2x
    ('V3_gold2x_only',  'G2x', 1.00, None,  0.00, 0.90),  # bond removed
    ('V4_gold1x2x',     'G1x', 0.50, 'G2x', 0.50, 0.90),  # mixed gold leverage
    ('V5_nas_tilt',     'G2x', 0.50, 'B1x', 0.50, 0.95),  # NASDAQ tilt (wn_max 0.95)
    ('V6_gold_heavy',   'G1x', 0.30, 'G2x', 0.70, 0.90),  # gold-heavy no bond
    ('V7_gold1x_b1x',   'G1x', 0.50, 'B1x', 0.50, 0.90),  # both 1x (minimal levered)
]


# ---- DH signal ---------------------------------------------------------------

def build_dh_signal(close, returns):
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


# ---- Threshold rebalancing (Approach A) --------------------------------------

def simulate_rebalance_A(raw, vz, wn_max=0.90):
    n = len(raw)
    raw_v = raw.values; vz_v = vz.values
    lev = np.zeros(n); wn = np.zeros(n)

    cur_lev = float(raw_v[0])
    cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[0], 0), 0.30, wn_max))
    lev[0] = cur_lev; wn[0] = cur_wn
    n_trades = 0

    for i in range(1, n):
        t = float(raw_v[i])
        dd_to_0   = (t == 0 and cur_lev > 0)
        dd_from_0 = (cur_lev == 0 and t > 0)
        if dd_to_0 or dd_from_0 or abs(t - cur_lev) > THRESHOLD_A:
            cur_lev = t
            cur_wn = float(np.clip(0.55 + 0.25 * cur_lev - 0.10 * max(vz_v[i], 0), 0.30, wn_max))
            n_trades += 1
        lev[i] = cur_lev; wn[i] = cur_wn

    return lev, wn, n_trades


# ---- NAV construction --------------------------------------------------------

def build_nav(close, lev_arr, wn_arr, s1_nav, s1_ratio, s2_nav, s2_ratio, dates):
    r_nas = close.pct_change().fillna(0).values
    r_s1  = pd.Series(s1_nav).pct_change().fillna(0).values
    r_s2  = pd.Series(s2_nav).pct_change().fillna(0).values if s2_nav is not None else np.zeros(len(r_nas))
    dc    = ANNUAL_COST / TRADING_DAYS

    lev_s = pd.Series(lev_arr, index=dates.index).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn_arr,  index=dates.index).shift(DELAY).fillna(0).values
    ws    = 1.0 - wn_s
    ws1   = ws * s1_ratio
    ws2   = ws * s2_ratio

    # Approach A formula: safe assets NOT multiplied by lev
    daily = wn_s * lev_s * (r_nas * BASE_LEV - dc) + ws1 * r_s1 + ws2 * r_s2
    nav   = (1 + pd.Series(daily, index=dates.index)).cumprod()
    return nav


# ---- Metrics ----------------------------------------------------------------

def calc_metrics(nav, dates, start, end):
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if mask.sum() < 100:
        return None
    nav_s = nav.loc[dates.index[mask]]
    nav_s = nav_s / nav_s.iloc[0]
    r   = nav_s.pct_change().fillna(0)
    n   = len(nav_s)
    yrs = n / TRADING_DAYS
    cagr  = float(nav_s.iloc[-1]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sh    = (r.mean() * TRADING_DAYS) / (r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    maxdd = (nav_s / nav_s.cummax() - 1).min()
    w5    = ((nav_s / nav_s.shift(int(5 * TRADING_DAYS))) ** (1 / 5) - 1).min() if n >= 5 * TRADING_DAYS else np.nan
    calmar = cagr / abs(maxdd) if maxdd != 0 else np.nan
    yr_nav = nav_s.groupby(pd.to_datetime(dates.loc[dates.index[mask]].values).year).last()
    wr = (yr_nav.pct_change().dropna() > 0).mean()
    return dict(CAGR=cagr, Sharpe=sh, MaxDD=maxdd, Worst5Y=w5, Calmar=calmar, WinRate=wr, Years=yrs)


# ---- Main -------------------------------------------------------------------

def main():
    df    = load_data(DATA_PATH)
    close = df['Close']
    ret   = close.pct_change()
    dates = df['Date']

    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} rows)")

    raw, vz = build_dh_signal(close, ret)
    gold_1x_prices = prepare_gold_data(dates)
    bond_nav_1x    = prepare_bond_data(dates)
    navs           = build_asset_navs(gold_1x_prices, bond_nav_1x)

    rows = []

    for (vname, s1_key, s1_r, s2_key, s2_r, wn_max) in VARIANTS:
        lev, wn_arr, n_trades = simulate_rebalance_A(raw, vz, wn_max=wn_max)
        s1_nav = navs[s1_key]
        s2_nav = navs[s2_key] if s2_key else None

        nav = build_nav(close, lev, wn_arr, s1_nav, s1_r, s2_nav, s2_r, dates)

        full_yrs = len(dates) / TRADING_DAYS
        tpy = n_trades / full_yrs

        for (pname, pstart, pend) in PERIODS:
            m = calc_metrics(nav, dates, pstart, pend)
            if m is None:
                continue
            rows.append({
                'variant': vname,
                'period': pname,
                'CAGR%': round(m['CAGR'] * 100, 2),
                'Sharpe': round(m['Sharpe'], 3),
                'MaxDD%': round(m['MaxDD'] * 100, 2),
                'Worst5Y%': round(m['Worst5Y'] * 100, 2) if not np.isnan(m['Worst5Y']) else np.nan,
                'Calmar': round(m['Calmar'], 3),
                'WinRate%': round(m['WinRate'] * 100, 1),
                'TradesPerYear': round(tpy, 1),
                'safe1': s1_key, 'safe2': s2_key or 'None', 'wn_max': wn_max,
            })

    df_out = pd.DataFrame(rows)

    # --- Print FULL period comparison ---
    full = df_out[df_out['period'] == 'FULL'].copy()
    full = full.sort_values('Sharpe', ascending=False)
    print("\n" + "=" * 100)
    print("FULL PERIOD (1974-2026) - sorted by Sharpe")
    print("=" * 100)
    print(full[['variant', 'CAGR%', 'Sharpe', 'MaxDD%', 'Worst5Y%', 'Calmar', 'WinRate%', 'TradesPerYear']].to_string(index=False))

    # --- Print OOS period ---
    oos = df_out[df_out['period'] == 'OOS'].copy()
    oos = oos.sort_values('Sharpe', ascending=False)
    print("\n" + "=" * 100)
    print("OOS PERIOD (2021-05 to 2026)")
    print("=" * 100)
    print(oos[['variant', 'CAGR%', 'Sharpe', 'MaxDD%', 'WinRate%']].to_string(index=False))

    # --- Print WF3 period (2020-2026) ---
    wf3 = df_out[df_out['period'] == 'WF3'].copy()
    wf3 = wf3.sort_values('Sharpe', ascending=False)
    print("\n" + "=" * 100)
    print("WF3 PERIOD (2020-2026, includes Covid+Bond crash)")
    print("=" * 100)
    print(wf3[['variant', 'CAGR%', 'Sharpe', 'MaxDD%', 'WinRate%']].to_string(index=False))

    # --- Save ---
    out_path = os.path.join(BASE, 'bond_variant_results.csv')
    df_out.to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nSaved: {out_path}")

    # --- Winner summary ---
    best = full.iloc[0]
    baseline = full[full['variant'] == 'V0_baseline'].iloc[0]
    print(f"\n{'='*60}")
    print(f"WINNER (by Sharpe): {best['variant']}")
    print(f"  CAGR%={best['CAGR%']:.2f}  Sharpe={best['Sharpe']:.3f}  MaxDD%={best['MaxDD%']:.2f}")
    print(f"\nBASELINE V0: CAGR%={baseline['CAGR%']:.2f}  Sharpe={baseline['Sharpe']:.3f}  MaxDD%={baseline['MaxDD%']:.2f}")
    delta_sharpe = best['Sharpe'] - baseline['Sharpe']
    print(f"Delta Sharpe vs V0: {delta_sharpe:+.3f}")

    return df_out


if __name__ == '__main__':
    main()
