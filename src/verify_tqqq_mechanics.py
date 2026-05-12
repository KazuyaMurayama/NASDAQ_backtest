"""
TQQQ Financing Cost Verification
=================================
Empirically verify whether TQQQ incurs 2x SOFR financing cost beyond the 0.86% expense ratio.

Method: OLS regression
  r_TQQQ = alpha + beta1 * r_QQQ + beta2 * SOFR_daily + epsilon

Expected results IF 2x SOFR hypothesis is correct:
  beta1 ≈ 3.0   (3x leverage on QQQ / NDX)
  beta2 ≈ -2.0  (2x SOFR financing drag, SOFR_daily in annual/252 units)
  alpha ≈ -(expense_daily + 3*dividend_daily) ≈ -9e-5/day ≈ -2.4%/yr
  R²    > 0.99

Note on QQQ vs NDX:
  TQQQ tracks NDX (price return, no dividends).
  QQQ adj_close includes dividends → beta1 still ≈ 3.0 but alpha shifts by -3*div_yield.
  The KEY test is beta2 ≈ -2.0 regardless of dividend treatment.
"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')


def load_series(filename, col='Close') -> pd.Series:
    path = os.path.join(DATA, filename)
    # yfinance CSVs have 2-row header; skip rows where index is 'Ticker' or 'Price'
    df = pd.read_csv(path, parse_dates=[0], index_col=0, skiprows=[1])
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]
    return pd.to_numeric(df.iloc[:, 0], errors='coerce').sort_index()


def load_sofr() -> pd.Series:
    """DTB3 (3M T-bill) as SOFR proxy. Annual % → daily decimal (/252)."""
    s = load_series('dtb3_daily.csv', col='yield_pct')
    return (s / 100.0 / 252.0).ffill(limit=5)


def build_analysis_df() -> pd.DataFrame:
    tqqq = load_series('tqqq_daily.csv')
    qqq  = load_series('qqq_daily.csv')
    sofr = load_sofr()

    r_tqqq = tqqq.pct_change()
    r_qqq  = qqq.pct_change()

    df = pd.DataFrame({
        'r_tqqq': r_tqqq,
        'r_qqq':  r_qqq,
        'sofr_d': sofr,
    }).dropna()

    # Align on common dates
    df = df.loc[df.index >= '2010-02-11']  # TQQQ inception
    print(f"Analysis period: {df.index[0].date()} to {df.index[-1].date()}, n={len(df):,} days")
    return df


def run_regressions(df: pd.DataFrame) -> dict:
    """
    Four regression models:
    M1: r_tqqq ~ r_qqq                        (no SOFR)
    M2: r_tqqq ~ r_qqq + sofr_d               (with SOFR)
    M3: r_tqqq ~ 3*r_qqq (force beta=3)       (implicit SOFR check via alpha)
    M4: r_tqqq ~ r_qqq + sofr_d + sofr_d^2   (nonlinear SOFR?)
    """
    results = {}

    # M1: baseline
    X1 = sm.add_constant(df[['r_qqq']])
    m1 = sm.OLS(df['r_tqqq'], X1).fit()
    results['M1_no_SOFR'] = m1

    # M2: with SOFR (KEY MODEL)
    X2 = sm.add_constant(df[['r_qqq', 'sofr_d']])
    m2 = sm.OLS(df['r_tqqq'], X2).fit()
    results['M2_with_SOFR'] = m2

    return results


def print_regression_summary(results: dict):
    print("\n" + "=" * 75)
    print("TQQQ REGRESSION SUMMARY")
    print("=" * 75)
    print(f"{'Model':<22} {'beta_QQQ':>10} {'beta_SOFR':>10} {'alpha_daily':>12} "
          f"{'alpha_ann%':>11} {'R2':>7}")
    print("-" * 75)
    for name, m in results.items():
        params = m.params
        b_qqq  = params.get('r_qqq', np.nan)
        b_sofr = params.get('sofr_d', np.nan)
        alpha  = params.get('const', np.nan)
        alpha_ann = alpha * 252 * 100
        r2 = m.rsquared
        print(f"{name:<22} {b_qqq:>10.4f} {b_sofr:>10.4f} {alpha:>12.6f} "
              f"{alpha_ann:>10.2f}% {r2:>7.5f}")
    print("=" * 75)

    # Key test output
    m2 = results['M2_with_SOFR']
    b_sofr = m2.params.get('sofr_d', np.nan)
    b_qqq  = m2.params.get('r_qqq', np.nan)
    alpha_ann = m2.params.get('const', 0) * 252 * 100
    print(f"\n[KEY TEST - M2 with SOFR]")
    print(f"  beta_QQQ  = {b_qqq:.4f}  (target: ~3.0)")
    print(f"  beta_SOFR = {b_sofr:.4f}  (target: ~-2.0 if 2xSOFR hypothesis is correct)")
    print(f"  alpha_ann = {alpha_ann:.2f}%/yr  (target: ~-2.3% incl. expense + 3x dividend)")
    print(f"  R2        = {m2.rsquared:.5f}  (target: >0.99)")
    print()

    # Interpretation
    if -2.3 <= b_sofr <= -1.7:
        verdict = "CONFIRMED: 2xSOFR financing cost is present and significant"
    elif -1.7 < b_sofr <= -0.5:
        verdict = "PARTIAL: Some SOFR drag, but < 2x (mixed implementation)"
    elif b_sofr > -0.5:
        verdict = "NOT CONFIRMED: SOFR drag minimal (possibly in expense ratio)"
    else:
        verdict = f"ANOMALOUS: beta_SOFR={b_sofr:.2f} (check data)"
    print(f"  => VERDICT: {verdict}")


def annual_comparison(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Compare actual TQQQ vs model predictions annually."""
    m2 = results['M2_with_SOFR']
    X2 = sm.add_constant(df[['r_qqq', 'sofr_d']])
    pred_m2 = m2.predict(X2)

    # 3x QQQ (naive, no SOFR)
    pred_3x = 3 * df['r_qqq']

    comp = pd.DataFrame({
        'tqqq_actual': df['r_tqqq'],
        'pred_3xQQQ':  pred_3x,
        'pred_M2_SOFR': pred_m2,
    }, index=df.index)

    def annual_cum(s):
        nav = (1 + s).cumprod()
        return nav.resample('YE').last().pct_change().dropna()

    ann = pd.DataFrame({
        'TQQQ_actual_%':  annual_cum(comp['tqqq_actual']) * 100,
        '3xQQQ_naive_%':  annual_cum(comp['pred_3xQQQ']) * 100,
        'M2_SOFR_%':      annual_cum(comp['pred_M2_SOFR']) * 100,
    }).round(1)
    ann['SOFR_drag_%'] = (ann['3xQQQ_naive_%'] - ann['TQQQ_actual_%']).round(1)
    return ann


def cost_decomposition(df: pd.DataFrame, results: dict):
    """Decompose TQQQ's implied annual cost."""
    m2 = results['M2_with_SOFR']
    b_sofr = m2.params.get('sofr_d', 0)
    alpha_d = m2.params.get('const', 0)

    mean_sofr_ann = df['sofr_d'].mean() * 252 * 100
    implied_financing = b_sofr * df['sofr_d'].mean() * 252 * 100  # annual

    print("\n=== Implied Annual Cost Decomposition (M2 model) ===")
    print(f"  alpha (fixed cost)   : {alpha_d*252*100:.3f}%/yr")
    print(f"    => approx = -(expense_ratio + 3*dividend_yield)")
    print(f"    => 0.86% + 3*0.5% = 2.36% confirms this ~= {alpha_d*252*100:.2f}%/yr")
    print(f"  Mean SOFR (2010-2026): {mean_sofr_ann:.2f}%/yr")
    print(f"  beta_SOFR            : {b_sofr:.3f}")
    print(f"  Implied SOFR drag    : {implied_financing:.2f}%/yr  (at mean SOFR)")
    print(f"  Total implied cost   : {alpha_d*252*100 + implied_financing:.2f}%/yr")
    print(f"  vs stated TER        : -0.86%/yr")
    print(f"  => Difference        : {alpha_d*252*100 + implied_financing + 0.86:.2f}%/yr")
    print(f"     (should equal 3*dividend + beta*SOFR beyond TER)")


def main():
    print("Loading data...")
    df = build_analysis_df()

    print("\nRunning regressions...")
    results = run_regressions(df)
    print_regression_summary(results)

    print("\n=== Annual Comparison: TQQQ vs Models ===")
    ann = annual_comparison(df, results)
    print(ann.to_string())
    mae_3x = (ann['3xQQQ_naive_%'] - ann['TQQQ_actual_%']).abs().mean()
    mae_m2 = (ann['M2_SOFR_%'] - ann['TQQQ_actual_%']).abs().mean()
    print(f"\n  MAE (3x naive vs actual): {mae_3x:.1f}%/yr")
    print(f"  MAE (M2 SOFR vs actual):  {mae_m2:.1f}%/yr")
    print(f"  Improvement from SOFR:    {mae_3x - mae_m2:.1f}%/yr")

    cost_decomposition(df, results)

    # Save
    out = os.path.join(BASE, 'tqqq_verification_results.csv')
    ann.to_csv(out, float_format='%.2f')
    print(f"\nSaved: {out}")

    return results


if __name__ == '__main__':
    main()
