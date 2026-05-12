"""
Corrected DH Dyn 2x3x [A] Backtest
=====================================
Integrates all empirically verified corrections:

1. Bond model correction: dgs30 + duration=17 + coupon_basis=252 (best fit vs actual TMF)
   - Replaces: dgs10 + duration=7 (current simulate, 19.71%/yr MAE)
   - Result:   dgs30 + duration=17 (3.87%/yr MAE vs actual TMF 2009-2026)

2. SOFR financing correction: both TQQQ and TMF pay 2*(SOFR + swap_spread)
   - Empirically confirmed: TQQQ regression beta_SOFR = -2.13 (R2=0.998)
   - TMF regression beta_SOFR = -2.48 (R2=0.91) with best bond proxy

Three scenarios:
  A. Baseline:   current code (dgs10+dur7+coupon252, no SOFR)
  B. SOFR-only:  SOFR added, bond model unchanged
  C. FULL:       SOFR + corrected bond model (dgs30+dur17)

Note: Gold sleeve (2036/ETN) is unchanged. Gold futures price embeds carry via
      futures roll, and 2x ETN cost ~0.50% TER is already included.
"""
import sys, os, types

_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None; _m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f); _m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest_engine import load_data, calc_dd_signal
from test_delay_robust import calc_momentum_decel_mult
from test_vix_integration import calc_vix_proxy
from test_portfolio_diversification import prepare_gold_data
from opt_lev2x3x import calc_asym_ewma

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

ANNUAL_COST   = 0.0086   # TQQQ TER
GOLD_2X_COST  = 0.0050   # 2036 TER
BOND_3X_COST  = 0.0091   # TMF TER (historical)
DELAY         = 2
BASE_LEV      = 3.0
TRADING_DAYS  = 252
THRESHOLD     = 0.15
SWAP_SPREAD   = 0.0050   # swap spread (50bps, empirically supported)

PERIODS = [
    ('FULL', '1974-01-02', '2026-12-31'),
    ('IS',   '1974-01-02', '2021-05-07'),
    ('OOS',  '2021-05-08', '2026-12-31'),
    ('WF3',  '2020-01-01', '2026-12-31'),
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_yield(name: str) -> pd.Series:
    path = os.path.join(DATA_DIR, f'{name}_daily.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.columns = ['yield_pct']
    return pd.to_numeric(df['yield_pct'], errors='coerce').ffill(limit=5)


def load_sofr(dates_series: pd.Series) -> np.ndarray:
    """DTB3 as SOFR proxy. Returns daily rate array aligned to dates_series."""
    s = load_yield('dtb3')
    daily = (s / 100.0 / TRADING_DAYS).ffill(limit=5).bfill(limit=5)
    aligned = daily.reindex(dates_series.values).ffill(limit=5).bfill(limit=5)
    return aligned.values


# ---------------------------------------------------------------------------
# Corrected bond 1x total return NAV
# ---------------------------------------------------------------------------

def bond_modified_duration(yield_pct: float, maturity: float = 22.0) -> float:
    """
    Yield-dependent modified duration for an approximate par coupon bond.
    At high yields (e.g., 15%), Dmod ~6-7; at low yields (e.g., 2%), Dmod ~16-17.
    maturity=22 is representative for ICE 20+ Year Treasury Index avg maturity.
    """
    y = max(float(yield_pct) / 100.0, 0.005)
    mac = (1.0 - (1.0 + y) ** (-maturity)) / y
    return mac / (1.0 + y)


def build_bond_1x_nav_corrected(dates_series: pd.Series,
                                  duration: float = 17.0,
                                  yield_name: str = 'dgs30',
                                  coupon_basis: int = 252,
                                  use_time_varying_duration: bool = False,
                                  bond_maturity: float = 22.0) -> np.ndarray:
    """
    Build 1x bond total return NAV aligned to NASDAQ dates.
    Best verified params: dgs30, duration=17, coupon_basis=252 (MAE=3.87%/yr vs TMF)

    use_time_varying_duration=True: replaces static duration with yield-dependent Dmod.
    This corrects the 1974-1985 high-yield era where D=17 overstates price sensitivity
    by ~2x (actual Dmod ~6-7 at 15% yield for a 20-yr coupon bond).
    """
    y_series = load_yield(yield_name)

    # Fill gaps: dgs30 starts 1977-02-15; use dgs10 for earlier dates
    y_dgs10  = load_yield('dgs10')
    y_full   = y_series.reindex(y_dgs10.index.union(y_series.index))
    mask_no  = y_full.isna()
    y_full[mask_no] = y_dgs10[mask_no]
    y_full   = y_full.ffill(limit=5)

    # Fix splice discontinuity: when dgs30 first appears (1977-02-15),
    # the level jump from dgs10 (7.38%) to dgs30 (7.70%) creates a fake
    # one-day bond loss of -5.4% (D=17). Level-correct pre-splice values
    # so the yield series is continuous at the transition.
    if yield_name == 'dgs30':
        dgs30_available = y_series.dropna()
        if len(dgs30_available) > 0:
            first_dgs30_date = dgs30_available.index[0]
            pre_splice_mask = y_full.index < first_dgs30_date
            if pre_splice_mask.any() and first_dgs30_date in y_full.index:
                last_pre  = y_full[pre_splice_mask].iloc[-1]   # dgs10 level
                first_post = y_full.loc[first_dgs30_date]      # dgs30 level
                jump = first_post - last_pre                    # e.g. +0.32%
                y_full[pre_splice_mask] = y_full[pre_splice_mask] + jump

    # Align to NASDAQ trading calendar
    y_aligned = y_full.reindex(dates_series.values).ffill(limit=5).bfill(limit=5)
    y_dec = (y_aligned.values / 100.0)  # percent → decimal

    n = len(y_dec)
    nav = np.ones(n)
    for i in range(1, n):
        dy  = y_dec[i] - y_dec[i-1]
        dur = (bond_modified_duration(y_dec[i-1] * 100, bond_maturity)
               if use_time_varying_duration else duration)
        price_ret  = -dur * dy
        coupon_ret = y_dec[i-1] / coupon_basis
        daily_ret  = np.clip(price_ret + coupon_ret, -0.20, 0.20)
        nav[i]     = nav[i-1] * (1 + daily_ret)
    return nav


def build_bond_1x_nav_baseline(dates_series: pd.Series) -> np.ndarray:
    """
    Baseline bond 1x NAV: dgs10, duration=7, coupon_basis=252
    (identical to prepare_bond_data() logic, but using our own data)
    """
    return build_bond_1x_nav_corrected(dates_series,
                                        duration=7.0,
                                        yield_name='dgs10',
                                        coupon_basis=252)


# ---------------------------------------------------------------------------
# Asset NAV builders
# ---------------------------------------------------------------------------

def build_gold_2x(gold_1x_prices: np.ndarray) -> np.ndarray:
    n = len(gold_1x_prices)
    g2 = np.ones(n)
    for i in range(1, n):
        gr = (gold_1x_prices[i] / gold_1x_prices[i-1] - 1
              if gold_1x_prices[i-1] > 0 else 0.0)
        g2[i] = g2[i-1] * (1 + gr * 2 - GOLD_2X_COST / TRADING_DAYS)
    return g2


def build_bond_3x(bond_1x_nav: np.ndarray, sofr_daily: np.ndarray,
                   apply_sofr: bool = True,
                   swap_spread: float = SWAP_SPREAD) -> np.ndarray:
    """Build 3x bond NAV with optional SOFR drag."""
    n = len(bond_1x_nav)
    swap_d = swap_spread / TRADING_DAYS
    b3 = np.ones(n)
    for i in range(1, n):
        br = (bond_1x_nav[i] / bond_1x_nav[i-1] - 1
              if bond_1x_nav[i-1] > 0 else 0.0)
        if apply_sofr:
            b3_ret = br * 3 - 2.0 * (sofr_daily[i] + swap_d) - BOND_3X_COST / TRADING_DAYS
        else:
            b3_ret = br * 3 - BOND_3X_COST / TRADING_DAYS
        b3[i] = b3[i-1] * (1 + b3_ret)
    return b3


# ---------------------------------------------------------------------------
# DH A2 signal
# ---------------------------------------------------------------------------

def build_a2_signal(close, returns):
    dd  = calc_dd_signal(close, 0.82, 0.92)
    av  = calc_asym_ewma(returns, 30, 10)
    ma150 = close.rolling(150).mean(); ratio = close / ma150
    ttv = (0.10 + 0.20 * (ratio - 0.85) / 0.30).clip(0.10, 0.30).fillna(0.20)
    vt  = (ttv / av).clip(0, 1.0)
    ma200 = close.rolling(200).mean(); sl = ma200.pct_change()
    sm = sl.rolling(60).mean(); ss = sl.rolling(60).std().replace(0, 0.0001)
    slope = (0.9 + 0.35 * (sl-sm)/ss).clip(0.3, 1.5).fillna(1.0)
    mom = calc_momentum_decel_mult(close, 60, 180, 0.3, 0.5, 1.3)
    vp  = calc_vix_proxy(returns)
    vz  = ((vp - vp.rolling(252).mean()) / vp.rolling(252).std().replace(0,0.001))
    vm  = (1.0 - 0.25 * vz).clip(0.5, 1.15)
    raw = (dd * vt * slope * mom * vm).clip(0, 1.0).fillna(0)
    return raw, vz.fillna(0)


def simulate_rebalance_A(raw, vz, threshold=THRESHOLD):
    n = len(raw); raw_v = raw.values; vz_v = vz.values
    lev = np.zeros(n); wn = np.zeros(n); wg = np.zeros(n); wb = np.zeros(n)
    cur_lev = raw_v[0]
    cur_wn = float(np.clip(0.55 + 0.25*cur_lev - 0.10*max(vz_v[0],0), 0.30, 0.90))
    cur_wg = (1-cur_wn)*0.5; cur_wb = (1-cur_wn)*0.5
    lev[0] = cur_lev; wn[0] = cur_wn; wg[0] = cur_wg; wb[0] = cur_wb
    n_trades = 0
    for i in range(1, n):
        t = raw_v[i]
        if (t==0 and cur_lev>0) or (cur_lev==0 and t>0) or abs(t-cur_lev)>threshold:
            cur_lev = t
            cur_wn = float(np.clip(0.55+0.25*cur_lev-0.10*max(vz_v[i],0), 0.30, 0.90))
            cur_wg = (1-cur_wn)*0.5; cur_wb = (1-cur_wn)*0.5
            n_trades += 1
        lev[i]=cur_lev; wn[i]=cur_wn; wg[i]=cur_wg; wb[i]=cur_wb
    return lev, wn, wg, wb, n_trades


# ---------------------------------------------------------------------------
# NAV builder
# ---------------------------------------------------------------------------

def build_nav(close, lev, wn, wg, wb, dates,
               gold_2x_nav, bond_3x_nav,
               sofr_daily=None, apply_tqqq_sofr=False,
               swap_spread=SWAP_SPREAD):
    """
    Build strategy NAV.
    apply_tqqq_sofr=True: subtract 2*(sofr+swap) from NASDAQ sleeve.
    bond_3x_nav: already has (or doesn't have) SOFR depending on how it was built.
    """
    r_nas = close.pct_change().fillna(0).values
    r_g2  = pd.Series(gold_2x_nav).pct_change().fillna(0).values
    r_b3  = pd.Series(bond_3x_nav).pct_change().fillna(0).values
    dc    = ANNUAL_COST / TRADING_DAYS
    swap_d = swap_spread / TRADING_DAYS

    lev_s = pd.Series(lev, index=dates.index).shift(DELAY).fillna(0).values
    wn_s  = pd.Series(wn,  index=dates.index).shift(DELAY).fillna(0).values
    wg_s  = pd.Series(wg,  index=dates.index).shift(DELAY).fillna(0).values
    wb_s  = pd.Series(wb,  index=dates.index).shift(DELAY).fillna(0).values

    if apply_tqqq_sofr and sofr_daily is not None:
        nas_ret = r_nas * BASE_LEV - 2.0 * (sofr_daily + swap_d) - dc
    else:
        nas_ret = r_nas * BASE_LEV - dc

    daily = wn_s * lev_s * nas_ret + wg_s * r_g2 + wb_s * r_b3
    return (1 + pd.Series(daily, index=dates.index)).cumprod()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calc_metrics(nav, dates, start, end):
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if mask.sum() < 100:
        return None
    idx = dates[mask].index
    ns  = nav.loc[idx[0]:idx[-1]].copy() / nav.loc[idx[0]]
    r   = ns.pct_change().fillna(0)
    n   = len(ns); yrs = n / TRADING_DAYS
    cagr  = float(ns.iloc[-1])**(1/yrs) - 1 if yrs > 0 else np.nan
    sh    = (r.mean()*TRADING_DAYS) / (r.std()*np.sqrt(TRADING_DAYS)) if r.std()>0 else np.nan
    maxdd = (ns/ns.cummax()-1).min()
    w5    = ((ns/ns.shift(TRADING_DAYS*5))**0.2-1).min() if n>=TRADING_DAYS*5 else np.nan
    df_y  = pd.DataFrame({'nav':ns.values,'dt':dates.loc[idx[0]:idx[-1]].values})
    df_y['year'] = pd.to_datetime(df_y['dt']).dt.year
    yn = df_y.groupby('year')['nav'].last()
    wr = (yn.pct_change().dropna()>0).mean()
    return dict(CAGR=cagr, Sharpe=sh, MaxDD=maxdd, Worst5Y=w5, WinRate=wr, Years=yrs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df    = load_data(DATA_PATH)
    close = df['Close']; ret = close.pct_change(); dates = df['Date']
    print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} ({len(df):,} days)")

    print("Building A2 signal...")
    raw, vz = build_a2_signal(close, ret)

    print("Running Approach A (threshold=0.15)...")
    lev, wn, wg, wb, n_tr = simulate_rebalance_A(raw, vz, THRESHOLD)
    print(f"  Trades: {n_tr}, {n_tr/(len(dates)/TRADING_DAYS):.1f}/yr")

    print("Loading gold data...")
    gold_1x = prepare_gold_data(dates)
    gold_2x = build_gold_2x(gold_1x)

    print("Loading SOFR (DTB3)...")
    sofr = load_sofr(dates)
    sofr_mean = np.nanmean(sofr) * TRADING_DAYS * 100
    print(f"  Mean SOFR (52yr): {sofr_mean:.2f}%/yr")

    print("\nBuilding bond models...")

    # Scenario A: Baseline (dgs10, dur=7, no SOFR) - current code
    print("  A. Baseline (dgs10, dur=7, no SOFR)...")
    bond_1x_base  = build_bond_1x_nav_baseline(dates)
    bond_3x_base  = build_bond_3x(bond_1x_base, sofr, apply_sofr=False)
    nav_A         = build_nav(close, lev, wn, wg, wb, dates,
                               gold_2x, bond_3x_base, apply_tqqq_sofr=False)

    # Scenario B: SOFR correction only, bond model unchanged
    print("  B. SOFR only (dgs10, dur=7, +2xSOFR)...")
    bond_3x_B    = build_bond_3x(bond_1x_base, sofr, apply_sofr=True)
    nav_B        = build_nav(close, lev, wn, wg, wb, dates,
                              gold_2x, bond_3x_B, sofr_daily=sofr, apply_tqqq_sofr=True)

    # Scenario C: Full correction (dgs30, dur=17, +2xSOFR) -- splice fixed
    print("  C. Full correction (dgs30, dur=17, +2xSOFR, splice fixed)...")
    bond_1x_corr = build_bond_1x_nav_corrected(dates)
    bond_3x_corr = build_bond_3x(bond_1x_corr, sofr, apply_sofr=True)
    nav_C        = build_nav(close, lev, wn, wg, wb, dates,
                              gold_2x, bond_3x_corr, sofr_daily=sofr, apply_tqqq_sofr=True)

    # Scenario D: Full correction + time-varying duration (most physically accurate)
    # Fixes static D=17 in 1974-1985 high-yield era (actual Dmod ~6-7 at 15% yield)
    print("  D. Best model (dgs30, D_var, +2xSOFR, splice fixed)...")
    bond_1x_D = build_bond_1x_nav_corrected(dates,
                                              use_time_varying_duration=True,
                                              bond_maturity=22.0)
    bond_3x_D = build_bond_3x(bond_1x_D, sofr, apply_sofr=True)
    nav_D     = build_nav(close, lev, wn, wg, wb, dates,
                           gold_2x, bond_3x_D, sofr_daily=sofr, apply_tqqq_sofr=True)

    # Print comparison
    print("\n" + "=" * 120)
    print("DH Dyn 2x3x [A] -- FOUR SCENARIOS COMPARISON")
    print("=" * 120)
    print("Scenario A = Baseline (current code: dgs10+dur7, no SOFR)")
    print("Scenario B = +SOFR only (2*SOFR on both TQQQ+TMF, bond model unchanged)")
    print("Scenario C = dgs30+dur17+2*SOFR+splice_fix (yield source/duration corrected)")
    print("Scenario D = dgs30+D_var+2*SOFR+splice_fix (time-varying duration, most accurate)")
    print("=" * 120)

    header = (f"{'Period':<8} | "
              f"{'CAGR_A':>8} {'CAGR_B':>8} {'CAGR_C':>8} {'CAGR_D':>8} | "
              f"{'Shrp_A':>7} {'Shrp_D':>7} | "
              f"{'MaxDD_A':>8} {'MaxDD_C':>8} {'MaxDD_D':>8} | "
              f"{'W5Y_A':>7} {'W5Y_D':>7}")
    print(header)
    print("-" * 120)

    rows = []
    for pname, pstart, pend in PERIODS:
        mA = calc_metrics(nav_A, dates, pstart, pend)
        mB = calc_metrics(nav_B, dates, pstart, pend)
        mC = calc_metrics(nav_C, dates, pstart, pend)
        mD = calc_metrics(nav_D, dates, pstart, pend)
        if mA is None: continue
        def _w5(m): return f"{m['Worst5Y']*100:>6.2f}%" if m and m['Worst5Y'] is not None else f"{'nan':>7}"
        print(f"{pname:<8} | "
              f"{mA['CAGR']*100:>7.2f}% {mB['CAGR']*100:>7.2f}% {mC['CAGR']*100:>7.2f}% {mD['CAGR']*100:>7.2f}% | "
              f"{mA['Sharpe']:>7.3f} {mD['Sharpe']:>7.3f} | "
              f"{mA['MaxDD']*100:>7.2f}% {mC['MaxDD']*100:>7.2f}% {mD['MaxDD']*100:>7.2f}% | "
              f"{_w5(mA)} {_w5(mD)}")
        rows.append({
            'period': pname,
            'CAGR_A%': round(mA['CAGR']*100, 2),
            'CAGR_B%': round(mB['CAGR']*100, 2) if mB else np.nan,
            'CAGR_C%': round(mC['CAGR']*100, 2),
            'CAGR_D%': round(mD['CAGR']*100, 2),
            'dCAGR_B': round((mB['CAGR']-mA['CAGR'])*100, 2) if mB else np.nan,
            'dCAGR_C': round((mC['CAGR']-mA['CAGR'])*100, 2),
            'dCAGR_D': round((mD['CAGR']-mA['CAGR'])*100, 2),
            'Sharpe_A': round(mA['Sharpe'], 3),
            'Sharpe_B': round(mB['Sharpe'], 3) if mB else np.nan,
            'Sharpe_C': round(mC['Sharpe'], 3),
            'Sharpe_D': round(mD['Sharpe'], 3),
            'MaxDD_A%': round(mA['MaxDD']*100, 2),
            'MaxDD_C%': round(mC['MaxDD']*100, 2),
            'MaxDD_D%': round(mD['MaxDD']*100, 2),
            'Worst5Y_A%': round(mA['Worst5Y']*100, 2) if mA['Worst5Y'] is not None else np.nan,
            'Worst5Y_D%': round(mD['Worst5Y']*100, 2) if mD['Worst5Y'] is not None else np.nan,
            'WinRate_A%': round(mA['WinRate']*100, 1),
            'WinRate_D%': round(mD['WinRate']*100, 1),
        })
    print("=" * 120)

    # Decomposition: SOFR vs Bond model vs Duration fix
    full_B_row = next((r for r in rows if r['period']=='FULL'), None)
    if full_B_row:
        sofr_impact  = full_B_row['dCAGR_B']
        bond_impact  = full_B_row['dCAGR_C'] - full_B_row['dCAGR_B']
        dur_impact   = full_B_row['dCAGR_D'] - full_B_row['dCAGR_C']
        print(f"\n[FULL period decomposition]")
        print(f"  SOFR correction impact     : {sofr_impact:+.2f}% CAGR (TQQQ+TMF 2x financing)")
        print(f"  Bond model+splice fix (C)  : {bond_impact:+.2f}% CAGR (dgs10+dur7 -> dgs30+dur17+splice_fix)")
        print(f"  Time-varying duration (D)  : {dur_impact:+.2f}% CAGR (static D=17 -> yield-dep Dmod)")
        print(f"  Total correction vs A      : {full_B_row['dCAGR_D']:+.2f}% CAGR")
        print(f"")
        print(f"  Best estimate CAGR  (D)    : {full_B_row['CAGR_D%']:.2f}%  vs Baseline {full_B_row['CAGR_A%']:.2f}%")
        print(f"  Best estimate Sharpe(D)    : {full_B_row['Sharpe_D']:.3f}  vs Baseline {full_B_row['Sharpe_A']:.3f}")
        print(f"  Best estimate MaxDD (D)    : {full_B_row['MaxDD_D%']:.2f}%  vs Baseline {full_B_row['MaxDD_A%']:.2f}%")

    # Save
    out = os.path.join(BASE, 'corrected_strategy_results.csv')
    pd.DataFrame(rows).to_csv(out, index=False, float_format='%.4f')
    print(f"\nSaved: {out}")
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
