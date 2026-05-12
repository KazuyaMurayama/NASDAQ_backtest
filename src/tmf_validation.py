"""
TMF Validation: Compare corrected bond simulation vs actual TMF prices.
=======================================================================
Validates the financing_cost_backtest.py correction by checking how well
the corrected simulation matches actual TMF daily/annual returns.

Also estimates duration correction factor k_dur via linear regression:
  actual_TMF_ret ~ k_dur * bond_price_component + coupon + financing
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
from scipy.stats import linregress, pearsonr

from backtest_engine import load_data
from test_portfolio_diversification import prepare_bond_data
from financing_cost_backtest import load_sofr, TRADING_DAYS, BOND_3X_COST, SWAP_SPREAD

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
DATA_PATH = os.path.join(BASE, 'NASDAQ_extended_to_2026.csv')

TMF_START = '2009-04-16'


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tmf_actual(dates_series: pd.Series) -> pd.Series:
    """Load actual TMF daily returns aligned to NASDAQ calendar."""
    path = os.path.join(DATA_DIR, 'tmf_daily.csv')
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.columns = ['close']
    s = df['close'].pct_change()
    # Align to NASDAQ dates
    aligned = s.reindex(dates_series.values).ffill(limit=1)
    aligned.index = dates_series.index
    return aligned


def build_bond_sim_series(bond_nav_1x: np.ndarray, sofr_daily: np.ndarray,
                           swap_spread: float) -> dict:
    """
    Build bond simulation series:
    - baseline: 3*total_ret - TER  (no SOFR)
    - corrected: 3*total_ret - 2*(SOFR+swap) - TER

    Returns dict of daily return arrays.
    """
    n = len(bond_nav_1x)
    swap_d = swap_spread / TRADING_DAYS

    b3_base = np.zeros(n); b3_corr = np.zeros(n)
    # Component series for regression
    price_component_3x  = np.zeros(n)  # 3x price return only
    coupon_component_3x = np.zeros(n)  # 3x coupon only
    sofr_component_2x   = np.zeros(n)  # 2x SOFR drag

    for i in range(1, n):
        br = (bond_nav_1x[i] / bond_nav_1x[i-1] - 1
              if bond_nav_1x[i-1] > 0 else 0.0)
        b3_base[i] = br * 3 - BOND_3X_COST / TRADING_DAYS
        b3_corr[i] = br * 3 - 2.0 * (sofr_daily[i] + swap_d) - BOND_3X_COST / TRADING_DAYS

    return {
        'baseline':  b3_base,
        'corrected': b3_corr,
    }


# ---------------------------------------------------------------------------
# Annual comparison
# ---------------------------------------------------------------------------

def annual_comparison(ret_base, ret_corr, ret_actual,
                       dates_series, start=TMF_START):
    """
    Compute cumulative annual returns for each series.
    Uses DatetimeIndex derived from dates_series for resample.
    """
    mask  = (dates_series >= pd.Timestamp(start)).values
    dt_idx = pd.DatetimeIndex(dates_series.values[mask])

    def annual_ret_arr(ret_arr):
        s   = pd.Series(ret_arr[mask], index=dt_idx)
        nav = (1 + s).cumprod()
        return nav.resample('YE').last().pct_change().dropna()

    def annual_ret_series(s_in):
        # s_in has integer index; pick values where mask applies
        s   = pd.Series(s_in.values[mask], index=dt_idx)
        return s.resample('YE').apply(lambda x: (1 + x).prod() - 1).dropna()

    base_ann   = annual_ret_arr(ret_base)
    corr_ann   = annual_ret_arr(ret_corr)
    actual_ann = annual_ret_series(ret_actual)

    df = pd.DataFrame({
        'sim_baseline': base_ann,
        'sim_corrected': corr_ann,
        'actual_TMF': actual_ann,
    }).dropna()
    df['resid_baseline']  = df['actual_TMF'] - df['sim_baseline']
    df['resid_corrected'] = df['actual_TMF'] - df['sim_corrected']
    return df


# ---------------------------------------------------------------------------
# Duration calibration via regression
# ---------------------------------------------------------------------------

def calibrate_k_dur(bond_nav_1x, sofr_daily, tmf_actual_daily,
                     dates_series, swap_spread=SWAP_SPREAD,
                     start=TMF_START) -> dict:
    """
    Regression to find duration multiplier k_dur:
      tmf_actual_ret ~ k_dur * bond_1x_ret_3x + (coupon_3x - 2*fin) + eps

    Since bond_nav_1x already includes coupon, we decompose:
      bond_1x_ret = price_ret_1x + coupon_1x (from prepare_bond_data)

    But we don't have the decomposition. Instead we regress:
      (tmf_actual + 2*fin - TER) / 3  ~  k_dur * bond_1x_ret

    So: implied_1x = (tmf_actual + 2*fin - TER) / 3
        regress implied_1x ~ bond_1x_ret  --> slope = k_dur
    """
    mask  = (dates_series >= pd.Timestamp(start)).values
    idx   = dates_series.index[mask]

    swap_d = swap_spread / TRADING_DAYS
    fin    = sofr_daily + swap_d

    # bond 1x daily total return
    bond_ret_1x = np.zeros(len(bond_nav_1x))
    for i in range(1, len(bond_nav_1x)):
        bond_ret_1x[i] = (bond_nav_1x[i] / bond_nav_1x[i-1] - 1
                          if bond_nav_1x[i-1] > 0 else 0.0)

    ter_d     = BOND_3X_COST / TRADING_DAYS
    x         = bond_ret_1x[mask]
    fin_sub   = 2.0 * fin[mask]

    # Actual TMF daily returns aligned (integer index series)
    tmf_vals  = tmf_actual_daily.values[mask]
    valid     = np.isfinite(x) & np.isfinite(tmf_vals)

    # implied 1x bond return if k_dur=1
    y = (tmf_vals[valid] + fin_sub[valid] + ter_d) / 3.0
    xv = x[valid]

    slope, intercept, r, p, se = linregress(xv, y)
    cor, _ = pearsonr(xv, y)

    return {
        'k_dur': round(float(slope), 4),
        'intercept': round(float(intercept), 6),
        'r_squared': round(float(r**2), 4),
        'pearson_r': round(float(cor), 4),
        'p_value': float(p),
        'n_samples': int(valid.sum()),
        'period': f"{start} to present",
        'note': 'k_dur>1 means actual TMF has higher duration than bond_nav_1x model',
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df    = load_data(DATA_PATH)
    dates = df['Date']
    print(f"NASDAQ dates: {dates.iloc[0].date()} to {dates.iloc[-1].date()}")

    print("Loading bond 1x (total return)...")
    bond_1x = prepare_bond_data(dates)

    print("Loading SOFR...")
    sofr_daily = load_sofr(dates)

    print("Loading actual TMF...")
    tmf_actual = load_tmf_actual(dates)
    tmf_start_idx = (dates >= pd.Timestamp(TMF_START)).idxmax()
    print(f"  TMF data from {TMF_START}, {(dates >= pd.Timestamp(TMF_START)).sum():,} days")

    sims = build_bond_sim_series(bond_1x, sofr_daily, SWAP_SPREAD)

    # Align actual TMF to integer index
    tmf_arr = tmf_actual.values

    # Annual comparison
    print("\n=== Annual Comparison: Simulation vs Actual TMF ===")
    ann = annual_comparison(sims['baseline'], sims['corrected'], tmf_actual,
                             dates, start=TMF_START)
    ann_fmt = (ann * 100).round(1)
    print(ann_fmt[['actual_TMF','sim_baseline','sim_corrected',
                   'resid_baseline','resid_corrected']].to_string())
    print(f"\n  MAE baseline   : {ann['resid_baseline'].abs().mean()*100:.1f}%/yr")
    print(f"  MAE corrected  : {ann['resid_corrected'].abs().mean()*100:.1f}%/yr")
    print(f"  RMSE baseline  : {(ann['resid_baseline']**2).mean()**0.5*100:.1f}%/yr")
    print(f"  RMSE corrected : {(ann['resid_corrected']**2).mean()**0.5*100:.1f}%/yr")

    # Cumulative comparison (2009 to present)
    mask = (dates >= pd.Timestamp(TMF_START)).values
    def cumret(arr):
        return float((1 + pd.Series(arr[mask])).prod() - 1)

    print(f"\n=== Cumulative Return ({TMF_START} to present) ===")
    print(f"  actual TMF       : {cumret(tmf_actual.values)*100:.1f}%")
    print(f"  sim baseline     : {cumret(sims['baseline'])*100:.1f}%")
    print(f"  sim corrected    : {cumret(sims['corrected'])*100:.1f}%")

    # Duration calibration
    print("\n=== Duration Calibration (k_dur regression) ===")
    calib = calibrate_k_dur(bond_1x, sofr_daily, tmf_actual, dates)
    for k, v in calib.items():
        print(f"  {k}: {v}")

    # Interpretation
    k = calib['k_dur']
    if 0.8 <= k <= 1.2:
        interp = f"k_dur={k:.3f} near 1.0 -- bond_nav_1x duration well-calibrated"
    elif k > 1.2:
        interp = f"k_dur={k:.3f} > 1.2 -- actual TMF has HIGHER duration (20+yr vs 10yr model)"
    else:
        interp = f"k_dur={k:.3f} < 0.8 -- actual TMF has LOWER duration than model"
    print(f"\n  Interpretation: {interp}")

    # Save
    out = os.path.join(BASE, 'tmf_validation_results.csv')
    ann.to_csv(out, float_format='%.4f')
    print(f"\nSaved: {out}")
    return calib


if __name__ == '__main__':
    main()
