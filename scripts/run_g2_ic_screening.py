"""G2: IC screening of expansion signals vs each strategy's daily returns.

Session 2 of SIGNAL_EXPANSION_PLAN_20260605.md.

For every (signal x strategy) combination:
  1) quantile_cut(levels=4)
  2) apply_publication_lag('daily')
  3) rolling Spearman IC vs strategy daily returns (window=252)
  4) Newey-West HAC t-stat -> two-sided p-value
  5) BH-FDR (alpha=0.15) across the full grid
Rank signals by mean |IC| averaged over the 3 baseline strategies.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from signals.quantize import quantile_cut
from signals.timing import apply_publication_lag
from signals.ic import compute_ic, ic_tstat_newey_west
from signals.multiplicity import fdr_bh


DATA_DIR = ROOT / 'data'
EXP_DIR = DATA_DIR / 'signals' / 'expansion'
RAW_DIR = EXP_DIR / 'raw'


def load_baseline_returns() -> dict:
    """Load S1/S2/S3 daily returns from baseline parquet."""
    nav = pd.read_parquet(DATA_DIR / 'signals' / 'integration' / 'baseline_navs_20260605.parquet')
    rets = {}
    for strat in ['S1', 'S2', 'S3']:
        if strat in nav.columns:
            r = nav[strat].pct_change().dropna()
            rets[strat] = r
    return rets


def load_repo_signals() -> dict:
    """Load all REPO-N parquet files. Return {repo_id: (name, series)}."""
    out = {}
    for p in sorted(RAW_DIR.glob('repo_*.parquet')):
        df = pd.read_parquet(p)
        # parquet always single-column for these files
        s = df.iloc[:, 0]
        # filename pattern: 'repo_<N>_repo_<N>_<descr>'
        stem = p.stem
        parts = stem.split('_')
        rid = '_'.join(parts[:2])  # e.g. 'repo_1'
        # 'repo_1_baa_minus_aaa_spread' -> descriptor
        name = '_'.join(parts[2:]) if len(parts) > 2 else stem
        s = s.dropna()
        s.index = pd.to_datetime(s.index)
        out[rid] = (name, s)
    return out


def load_macro_features() -> dict:
    """Load macro_features.csv columns as individual signals."""
    df = pd.read_csv(DATA_DIR / 'macro_features.csv', parse_dates=['date'], index_col='date').sort_index()
    numeric = df.select_dtypes(include='number')
    out = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 252:
            continue
        if s.nunique() < 4:
            continue
        out[f'macro_{col}'] = (col, s)
    return out


def screen_one(signal: pd.Series, strategy_ret: pd.Series, window: int = 252) -> dict:
    """Quantize, lag, rolling IC, NW t-stat, two-sided p-value."""
    sig_q = quantile_cut(signal, levels=4)
    sig_lagged = apply_publication_lag(sig_q, lag_type='daily')
    # publication_lag can produce duplicates if input had non-business-day timestamps
    sig_lagged = sig_lagged[~sig_lagged.index.duplicated(keep='last')]

    ic_series = compute_ic(sig_lagged.astype('float64'), strategy_ret, window=window)
    ic_clean = ic_series.dropna()
    if len(ic_clean) < 100:
        return {
            'mean_ic': float('nan'),
            't_stat': float('nan'),
            'p_value': float('nan'),
            'n_obs': len(ic_clean),
        }

    mean_ic = float(ic_clean.mean())
    t_stat = ic_tstat_newey_west(ic_series)
    if np.isnan(t_stat):
        p_val = float('nan')
    else:
        p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(t_stat))))
    return {
        'mean_ic': mean_ic,
        't_stat': float(t_stat),
        'p_value': p_val,
        'n_obs': int(len(ic_clean)),
    }


def main() -> None:
    print('[G2] Loading baseline returns...')
    rets = load_baseline_returns()
    print(f'  Strategies: {list(rets.keys())}')
    for k, v in rets.items():
        print(f'    {k}: n={len(v)}  range={v.index[0].date()} .. {v.index[-1].date()}')

    print('[G2] Loading REPO signals...')
    repo = load_repo_signals()
    print(f'  REPO: {len(repo)} signals')

    print('[G2] Loading macro_features...')
    macro = load_macro_features()
    print(f'  Macro: {len(macro)} features')

    all_signals = {**repo, **macro}
    print(f'[G2] Total signals to evaluate: {len(all_signals)}')

    n_tests = len(all_signals) * len(rets)
    print(f'[G2] Running {len(all_signals)} x {len(rets)} = {n_tests} tests...')

    rows = []
    for i, (sid, (name, sig)) in enumerate(all_signals.items()):
        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(all_signals)}] {sid}')
        for strat, ret in rets.items():
            try:
                metrics = screen_one(sig, ret)
                rows.append({
                    'signal_id': sid,
                    'signal_name': name,
                    'strategy': strat,
                    **metrics,
                    'error': '',
                })
            except Exception as e:  # noqa: BLE001
                rows.append({
                    'signal_id': sid,
                    'signal_name': name,
                    'strategy': strat,
                    'mean_ic': float('nan'),
                    't_stat': float('nan'),
                    'p_value': float('nan'),
                    'n_obs': 0,
                    'error': str(e),
                })

    df = pd.DataFrame(rows)
    print(f'\n[G2] {len(df)} rows total; {df["mean_ic"].notna().sum()} with valid IC')

    # BH-FDR across the entire grid of valid p-values
    valid = df.dropna(subset=['p_value']).copy()
    df['p_value_bh'] = float('nan')
    df['reject_bh'] = False
    if not valid.empty:
        adj = fdr_bh(valid['p_value'], alpha=0.15)
        df.loc[valid.index, 'p_value_bh'] = adj['p_bh'].values
        df.loc[valid.index, 'reject_bh'] = adj['reject_bh'].values

    # Per-signal summary across strategies
    sig_summary = df.groupby('signal_id').agg(
        signal_name=('signal_name', 'first'),
        mean_abs_ic=('mean_ic', lambda x: float(np.nanmean(np.abs(x)))),
        max_abs_ic=('mean_ic', lambda x: float(np.nanmax(np.abs(x))) if x.notna().any() else float('nan')),
        n_strats_passing=('reject_bh', lambda x: int(x.fillna(False).sum())),
        min_p_bh=('p_value_bh', lambda x: float(np.nanmin(x)) if x.notna().any() else float('nan')),
    ).sort_values('mean_abs_ic', ascending=False)

    # Outputs
    raw_csv = EXP_DIR / 'g2_ic_screening_20260605.csv'
    df.to_csv(raw_csv, index=False)
    print(f'[G2] Wrote {raw_csv}')

    summary_csv = EXP_DIR / 'g2_signal_summary_20260605.csv'
    sig_summary.to_csv(summary_csv)
    print(f'[G2] Wrote {summary_csv}')

    top20 = sig_summary.head(20).index.tolist()
    top20_df = df[df['signal_id'].isin(top20)].copy()
    top20_csv = EXP_DIR / 'g2_top20_for_g3_20260605.csv'
    top20_df.to_csv(top20_csv, index=False)
    print(f'[G2] Wrote {top20_csv}')

    print('\n[G2] Top 20 signals by mean |IC| across strategies:')
    print(sig_summary.head(20).to_string())

    print('\n[G2] Top 10 (signal, strategy) pairs by |IC|:')
    abs_ic = df['mean_ic'].abs()
    top_pairs = df.loc[abs_ic.sort_values(ascending=False).dropna().index].head(10)
    for _, r in top_pairs.iterrows():
        pbh = r['p_value_bh']
        pbh_s = f'{pbh:.4f}' if pd.notna(pbh) else 'NA'
        print(f"  {r['signal_id']:35s} x {r['strategy']}: IC={r['mean_ic']:+.4f}  t={r['t_stat']:+.2f}  p_BH={pbh_s}  reject={r['reject_bh']}")

    n_pass = int(df['reject_bh'].fillna(False).sum())
    print(f'\n[G2] BH-FDR pass count (alpha=0.15): {n_pass} / {len(valid)} valid tests')


if __name__ == '__main__':
    main()
