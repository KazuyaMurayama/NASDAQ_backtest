"""
Bond Model Verification: Grid Search for Optimal Duration/Yield Parameters
===========================================================================
Tests multiple combinations of (duration, yield_source, coupon_basis)
against actual TMF (2009-2026) to find the best-fitting bond model.

Also verifies TMF beta_bond ≈ 3.0 and beta_SOFR ≈ -2.0 via OLS regression.

Grid:
  duration     : [7, 10, 13, 15, 17, 20]
  yield_source : ['dgs10', 'dgs30', 'avg10_30']
  coupon_basis : [252, 365]    (days/year for daily coupon accrual)

Key question:
  What parameters minimize tracking error vs actual TMF?
  Does coupon_basis=252 vs 365 matter significantly?
"""
import os, sys, types

# multitasking shim
_m = types.ModuleType('multitasking')
_m.set_max_threads = lambda x: None
_m.set_engine = lambda x: None
_m.task = lambda *a, **k: (lambda f: f)
_m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = _m

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')

TRADING_DAYS = 252
TMF_EXPENSE  = 0.0106   # TMF TER: 0.91% was previous, current is higher
TMF_EXPENSE_OLD = 0.0091  # Historical TER used in simulation
SOFR_SPREAD  = 0.0050   # Swap spread assumption


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_yield(name: str) -> pd.Series:
    """Load yield from FRED CSV (yield_pct column, annual %)."""
    path = os.path.join(DATA, f'{name}_daily.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.columns = ['yield_pct']
    return pd.to_numeric(df['yield_pct'], errors='coerce').ffill(limit=5)


def load_sofr() -> pd.Series:
    """DTB3 as SOFR proxy. Returns annual decimal / 252 (daily rate)."""
    s = load_yield('dtb3')
    return (s / 100.0 / 252.0).ffill(limit=5)


def load_etf(filename: str, name: str) -> pd.Series:
    """Load ETF prices (handles both plain CSV and yfinance 2-row-header format)."""
    path = os.path.join(DATA, filename)
    # Always use generic loader: skip row 1 (yfinance Ticker row if present)
    raw = open(path).readline().strip()
    if raw.startswith('Price') or raw.startswith('Ticker'):
        # yfinance format
        df = pd.read_csv(path, parse_dates=[0], index_col=0, skiprows=[1])
    else:
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()].sort_index()
    s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    return s.pct_change().rename(name)


def load_tmf() -> pd.Series:
    return load_etf('tmf_daily.csv', 'r_tmf')


def load_tlt() -> pd.Series:
    return load_etf('tlt_daily.csv', 'r_tlt')


# ---------------------------------------------------------------------------
# Bond model builder
# ---------------------------------------------------------------------------

def build_bond_1x_nav(yield_series: pd.Series, duration: float,
                       coupon_basis: int,
                       start: str = '1974-01-01') -> pd.Series:
    """
    Build 1x bond total return NAV from yield series.

    Args:
        yield_series: Annual yield in % (e.g., 4.5 = 4.5%)
        duration: Modified duration (years)
        coupon_basis: Days per year for coupon accrual (252 or 365)
    """
    y = yield_series.dropna()
    y = y[y.index >= pd.Timestamp(start)]

    nav = pd.Series(np.ones(len(y)), index=y.index, dtype=float)
    y_dec = y / 100.0  # convert % to decimal

    for i in range(1, len(y)):
        dy = y_dec.iloc[i] - y_dec.iloc[i-1]
        price_ret   = -duration * dy
        coupon_ret  = y_dec.iloc[i-1] / coupon_basis
        daily_ret   = np.clip(price_ret + coupon_ret, -0.20, 0.20)
        nav.iloc[i] = nav.iloc[i-1] * (1 + daily_ret)

    return nav


def build_bond_3x_nav(nav_1x: pd.Series, sofr_daily: pd.Series,
                       swap_spread: float = SOFR_SPREAD,
                       ter: float = TMF_EXPENSE_OLD) -> pd.Series:
    """
    Build 3x bond NAV with SOFR financing drag.
    3x total_return - 2*(SOFR+swap) - TER
    """
    r_1x = nav_1x.pct_change().fillna(0)
    aligned_sofr = sofr_daily.reindex(r_1x.index).ffill(limit=5).fillna(0)
    swap_d = swap_spread / TRADING_DAYS

    r_3x = r_1x * 3 - 2 * (aligned_sofr + swap_d) - ter / TRADING_DAYS
    nav_3x = (1 + r_3x).cumprod()
    return nav_3x


# ---------------------------------------------------------------------------
# Annual comparison helper
# ---------------------------------------------------------------------------

def annual_returns(nav_series: pd.Series, start: str, end: str) -> pd.Series:
    s = nav_series.loc[start:end]
    if len(s) == 0:
        return pd.Series(dtype=float)
    s = s / s.iloc[0]
    return s.resample('YE').last().pct_change().dropna()


def tracking_error(ann_synth: pd.Series, ann_actual: pd.Series) -> dict:
    common = ann_synth.index.intersection(ann_actual.index)
    if len(common) < 3:
        return {'mae': np.nan, 'rmse': np.nan, 'cum_gap': np.nan, 'n': 0}
    s = ann_synth.loc[common]
    a = ann_actual.loc[common]
    resid = s - a
    cum_gap = (s.mean() - a.mean()) * 100
    return {
        'mae':     float(resid.abs().mean() * 100),
        'rmse':    float(np.sqrt((resid**2).mean()) * 100),
        'cum_gap': float(cum_gap),    # positive = synth too high
        'n':       len(common),
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(tmf_actual: pd.Series, sofr_daily: pd.Series,
                 dgs10: pd.Series, dgs30: pd.Series,
                 start='2009-04-16') -> pd.DataFrame:

    dgs_avg = ((dgs10 + dgs30) / 2.0).rename('avg10_30')
    yield_sources = {
        'dgs10':     dgs10,
        'dgs30':     dgs30,
        'avg10_30':  dgs_avg,
    }
    durations     = [7, 10, 13, 15, 17, 20]
    coupon_bases  = [252, 365]

    ann_actual = annual_returns(
        (1 + tmf_actual.dropna()).cumprod(), start, '2026-12-31'
    )

    rows = []
    for yname, yseries in yield_sources.items():
        for dur in durations:
            for cb in coupon_bases:
                try:
                    nav_1x = build_bond_1x_nav(yseries, duration=dur, coupon_basis=cb)
                    nav_3x = build_bond_3x_nav(nav_1x, sofr_daily)
                    ann_3x = annual_returns(nav_3x, start, '2026-12-31')
                    te = tracking_error(ann_3x, ann_actual)
                    rows.append({
                        'yield_src': yname,
                        'duration': dur,
                        'coupon_basis': cb,
                        'mae_%': round(te['mae'], 2),
                        'rmse_%': round(te['rmse'], 2),
                        'cum_gap_%': round(te['cum_gap'], 2),
                        'n_years': te['n'],
                    })
                except Exception as e:
                    rows.append({'yield_src': yname, 'duration': dur,
                                 'coupon_basis': cb, 'mae_%': np.nan,
                                 'rmse_%': np.nan, 'cum_gap_%': np.nan,
                                 'n_years': 0, 'error': str(e)})

    df = pd.DataFrame(rows).sort_values('mae_%').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# TMF OLS regression: verify beta_bond ≈ 3.0 and beta_SOFR ≈ -2.0
# ---------------------------------------------------------------------------

def tmf_regression(tmf_actual: pd.Series, sofr_daily: pd.Series,
                    dgs30: pd.Series, duration: float = 17.0,
                    coupon_basis: int = 252,
                    start: str = '2009-04-16') -> dict:
    """
    Regress actual TMF daily returns on:
      r_tmf = alpha + beta_bond * r_bond_1x_proxy + beta_SOFR * sofr_d + eps

    Tests whether:
      beta_bond ≈ 3.0  (3x leverage on bond total return)
      beta_SOFR ≈ -2.0 (2x SOFR financing drag)
    """
    nav_1x = build_bond_1x_nav(dgs30, duration=duration, coupon_basis=coupon_basis)
    r_bond = nav_1x.pct_change()

    aligned = pd.DataFrame({
        'r_tmf':   tmf_actual,
        'r_bond':  r_bond,
        'sofr_d':  sofr_daily,
    }).dropna()
    aligned = aligned.loc[start:]

    X = sm.add_constant(aligned[['r_bond', 'sofr_d']])
    model = sm.OLS(aligned['r_tmf'], X).fit()

    beta_bond = model.params.get('r_bond', np.nan)
    beta_sofr = model.params.get('sofr_d', np.nan)
    alpha     = model.params.get('const', np.nan)

    return {
        'model': model,
        'beta_bond': beta_bond,
        'beta_SOFR': beta_sofr,
        'alpha_daily': alpha,
        'alpha_ann_%': alpha * 252 * 100,
        'r_squared': model.rsquared,
        'n': len(aligned),
        'duration_used': duration,
        'coupon_basis': coupon_basis,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    dgs10    = load_yield('dgs10')
    dgs30    = load_yield('dgs30')
    sofr     = load_sofr()
    tmf      = load_tmf()
    tlt      = load_tlt()

    print(f"  TMF: {tmf.dropna().index[0].date()} to {tmf.dropna().index[-1].date()}")
    print(f"  TLT: {tlt.dropna().index[0].date()} to {tlt.dropna().index[-1].date()}")
    print(f"  DGS30: {dgs30.dropna().index[0].date()} to {dgs30.dropna().index[-1].date()}")

    # === Grid search ===
    print("\nRunning bond model grid search...")
    grid = grid_search(tmf, sofr, dgs10, dgs30)

    print("\n" + "=" * 70)
    print("BOND MODEL GRID SEARCH (sorted by MAE%/yr vs actual TMF 2009-2026)")
    print("=" * 70)
    print(grid.to_string(index=True))

    best = grid.iloc[0]
    print(f"\nBEST FIT: yield={best['yield_src']}, duration={best['duration']}, "
          f"coupon_basis={best['coupon_basis']}")
    print(f"  MAE  = {best['mae_%']:.2f}%/yr")
    print(f"  RMSE = {best['rmse_%']:.2f}%/yr")
    print(f"  Cumulative gap (synth - actual) = {best['cum_gap_%']:+.2f}%/yr average")

    baseline_row = grid[(grid['yield_src']=='dgs10') & (grid['duration']==7) &
                         (grid['coupon_basis']==252)]
    if len(baseline_row):
        br = baseline_row.iloc[0]
        print(f"\nCURRENT MODEL (dgs10, dur=7, /252): MAE={br['mae_%']:.2f}%/yr, "
              f"gap={br['cum_gap_%']:+.2f}%/yr")
        print(f"Improvement from best vs current: "
              f"MAE {br['mae_%'] - best['mae_%']:.2f}%/yr reduction")

    # === Coupon basis sensitivity ===
    print("\n=== Coupon Basis Comparison (/252 vs /365) for best yield+duration ===")
    best_yd = grid[(grid['yield_src'] == best['yield_src']) &
                   (grid['duration'] == best['duration'])]
    print(best_yd[['coupon_basis', 'mae_%', 'rmse_%', 'cum_gap_%']].to_string(index=False))

    # === OLS regression on best params ===
    print(f"\n=== OLS Regression (best params: dur={best['duration']}, "
          f"yield={best['yield_src']}, basis={int(best['coupon_basis'])}) ===")
    ys_map = {'dgs10': dgs10, 'dgs30': dgs30,
              'avg10_30': ((dgs10+dgs30)/2).rename('avg')}
    ys = ys_map[best['yield_src']]
    reg = tmf_regression(tmf, sofr, ys,
                          duration=best['duration'],
                          coupon_basis=int(best['coupon_basis']))

    print(f"  beta_bond  = {reg['beta_bond']:.4f}  (target: ~3.0)")
    print(f"  beta_SOFR  = {reg['beta_SOFR']:.4f}  (target: ~-2.0)")
    print(f"  alpha      = {reg['alpha_ann_%']:.2f}%/yr")
    print(f"  R2         = {reg['r_squared']:.4f}")
    print(f"  n_samples  = {reg['n']:,}")

    # Verdicts
    bb = reg['beta_bond']; bs = reg['beta_SOFR']
    v_bond = "CONFIRMED (3x TR)" if 2.8 <= bb <= 3.2 else f"PARTIAL ({bb:.2f})"
    v_sofr = "CONFIRMED (2x SOFR)" if -2.2 <= bs <= -1.8 else f"PARTIAL ({bs:.2f})"
    print(f"\n  => Bond leverage verdict: {v_bond}")
    print(f"  => SOFR drag verdict    : {v_sofr}")

    # === Annual comparison: best model vs actual TMF ===
    print("\n=== Annual Comparison (Best Model vs Actual TMF) ===")
    nav_1x = build_bond_1x_nav(ys, duration=best['duration'],
                                 coupon_basis=int(best['coupon_basis']))
    nav_3x = build_bond_3x_nav(nav_1x, sofr)
    ann_3x   = annual_returns(nav_3x, '2009-04-16', '2026-12-31') * 100

    # Also current model
    nav_1x_curr = build_bond_1x_nav(dgs10, duration=7, coupon_basis=252)
    nav_3x_curr = build_bond_3x_nav(nav_1x_curr, sofr)
    ann_curr = annual_returns(nav_3x_curr, '2009-04-16', '2026-12-31') * 100

    ann_actual = annual_returns(
        (1+tmf.dropna()).cumprod(), '2009-04-16', '2026-12-31') * 100

    ann_tlt  = annual_returns(
        (1+tlt.dropna()).cumprod(), '2009-04-16', '2026-12-31') * 100

    comp = pd.DataFrame({
        'actual_TMF%': ann_actual,
        f'best_model%': ann_3x,
        'curr_model%': ann_curr,
        '3xTLT_approx%': ann_tlt * 3,
    }).dropna().round(1)
    print(comp.to_string())

    # Save results
    out_grid = os.path.join(BASE, 'bond_model_grid_results.csv')
    grid.to_csv(out_grid, index=False, float_format='%.3f')
    print(f"\nSaved: {out_grid}")

    out_ann = os.path.join(BASE, 'bond_model_annual_comparison.csv')
    comp.to_csv(out_ann, float_format='%.2f')
    print(f"Saved: {out_ann}")

    return grid, reg


if __name__ == '__main__':
    main()
