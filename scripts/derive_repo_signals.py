"""Derive REPO signals from existing repo CSVs and save as parquet.

Session 1 of SIGNAL_EXPANSION_PLAN_20260605.md.
Materializes the REPO-1..REPO-9 signals computable directly from repo data.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'data' / 'signals' / 'expansion' / 'raw'


def _load_daily(filename, value_col=None, date_col=None):
    """Load a CSV with auto-detection of date column ('Date' or 'DATE' or 'date')."""
    df = pd.read_csv(DATA_DIR / filename)
    if date_col is None:
        for cand in ('Date', 'DATE', 'date'):
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            raise KeyError(f"No date column found in {filename}; cols={df.columns.tolist()}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    if value_col is None:
        return df.iloc[:, 0].astype(float)
    return df[value_col].astype(float)


def derive_repo_1():
    """AAA-BAA spread (monthly): BAA - AAA, positive = credit stress widening."""
    aaa = _load_daily('aaa_monthly.csv')
    baa = _load_daily('baa_monthly.csv')
    s = (baa - aaa).dropna()
    s.name = 'repo_1_baa_minus_aaa_spread'
    return s


def derive_repo_2():
    """30Y-10Y term premium (daily)."""
    dgs30 = _load_daily('dgs30_daily.csv')
    dgs10 = _load_daily('dgs10_daily.csv')
    s = (dgs30 - dgs10).dropna()
    s.name = 'repo_2_30y_10y_termpremium'
    return s


def derive_repo_3():
    """Fed Funds 1-month (21 trading days) change."""
    dff = _load_daily('dff_daily.csv')
    s = dff.diff(periods=21).dropna()
    s.name = 'repo_3_dff_change_1m'
    return s


def derive_repo_4():
    """Fed Funds minus 10Y (yield curve front-back proxy)."""
    dff = _load_daily('dff_daily.csv')
    dgs10 = _load_daily('dgs10_daily.csv')
    s = (dff - dgs10).dropna()
    s.name = 'repo_4_dff_minus_10y'
    return s


def derive_repo_5():
    """DGP (DB Gold Double Long ETN, 2x gold) — 21d log return as a sentiment proxy."""
    p = DATA_DIR / 'dgp_daily.csv'
    if not p.exists():
        return None
    s = pd.read_csv(p, parse_dates=['Date'], index_col='Date').sort_index()['Close'].astype(float)
    s = np.log(s).diff(21).dropna()
    s.name = 'repo_5_dgp_2xgold_logret_21d'
    return s


def derive_repo_6():
    """DRN (Direxion Daily MSCI Real Estate Bull 3x) — 21d log return as a REIT risk-on proxy."""
    p = DATA_DIR / 'drn_daily.csv'
    if not p.exists():
        return None
    s = pd.read_csv(p, parse_dates=['Date'], index_col='Date').sort_index()['Close'].astype(float)
    s = np.log(s).diff(21).dropna()
    s.name = 'repo_6_drn_3xreit_logret_21d'
    return s


def derive_repo_7():
    """CPI YoY surprise vs 12mo trailing average."""
    cpi = _load_daily('cpiaucsl_monthly.csv')
    cpi_yoy = (cpi / cpi.shift(12) - 1.0) * 100.0
    s = (cpi_yoy - cpi_yoy.rolling(12).mean()).dropna()
    s.name = 'repo_7_cpi_yoy_surprise'
    return s


def derive_repo_8():
    """ML predictions sign (from ml_oos_predictions.csv pred column)."""
    p = DATA_DIR / 'ml_oos_predictions.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['date'], index_col='date').sort_index()
    pred_col = next((c for c in df.columns if 'pred' in c.lower()), df.columns[0])
    s = np.sign(df[pred_col].astype(float))
    s.name = 'repo_8_ml_pred_sign'
    return s.dropna()


def derive_repo_9():
    """ML features PC1 (StandardScaler -> PCA n=1)."""
    p = DATA_DIR / 'ml_features.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['date'], index_col='date').sort_index()
    num = df.select_dtypes(include='number').dropna()
    if num.shape[1] < 2 or len(num) < 100:
        return None
    Xs = StandardScaler().fit_transform(num.values)
    pc = PCA(n_components=1).fit_transform(Xs).ravel()
    s = pd.Series(pc, index=num.index, name='repo_9_ml_pc1')
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    derivations = {
        'repo_1': derive_repo_1,
        'repo_2': derive_repo_2,
        'repo_3': derive_repo_3,
        'repo_4': derive_repo_4,
        'repo_5': derive_repo_5,
        'repo_6': derive_repo_6,
        'repo_7': derive_repo_7,
        'repo_8': derive_repo_8,
        'repo_9': derive_repo_9,
    }
    summary = []
    for key, fn in derivations.items():
        try:
            s = fn()
            if s is None:
                print(f"  {key}: SKIPPED (source missing)")
                summary.append({'signal': key, 'status': 'skipped', 'n_obs': 0,
                                'date_start': '', 'date_end': '', 'path': ''})
                continue
            out = OUT_DIR / f'{key}_{s.name}.parquet'
            s.to_frame().to_parquet(out)
            print(f"  {key}: n={len(s)}, range {s.index[0].date()}->{s.index[-1].date()}, "
                  f"mean={s.mean():.4f}, std={s.std():.4f}")
            summary.append({'signal': key, 'status': 'ok', 'n_obs': len(s),
                            'date_start': str(s.index[0].date()),
                            'date_end': str(s.index[-1].date()),
                            'path': str(out.relative_to(ROOT))})
        except Exception as e:
            print(f"  {key}: FAILED: {e}")
            summary.append({'signal': key, 'status': f'error: {e}', 'n_obs': 0,
                            'date_start': '', 'date_end': '', 'path': ''})

    smry = pd.DataFrame(summary)
    out_csv = OUT_DIR.parent / 'repo_derivation_summary_20260605.csv'
    smry.to_csv(out_csv, index=False)
    print(f"\nWrote summary to {out_csv.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
