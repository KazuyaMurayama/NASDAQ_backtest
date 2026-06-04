"""Multiplicity correction for Phase B screening.

Benjamini-Hochberg FDR (primary) and Bonferroni (reference) corrections
for the 31 signal x 3 asset x 3 horizon = 279 simultaneous tests in
Phase B. Without correction, ~28 false positives expected at p<0.10 raw.
"""
from __future__ import annotations
import pandas as pd
from statsmodels.stats.multitest import multipletests


def fdr_bh(pvals: pd.Series, alpha: float = 0.10) -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction.

    Args:
        pvals: Series of raw p-values (index = test identifier).
        alpha: false discovery rate threshold.

    Returns:
        DataFrame indexed same as pvals with columns:
          'p_raw':        original p-value
          'p_bh':         adjusted p-value (BH method)
          'reject_bh':    True if rejected at given alpha
          'rank':         rank of raw p-value (ascending)
    """
    pvals = pvals.dropna()
    if pvals.empty:
        return pd.DataFrame(columns=['p_raw', 'p_bh', 'reject_bh', 'rank'])

    reject, p_adj, _, _ = multipletests(pvals.values, alpha=alpha, method='fdr_bh')
    out = pd.DataFrame({
        'p_raw': pvals.values,
        'p_bh': p_adj,
        'reject_bh': reject,
        'rank': pvals.rank(method='min').astype(int).values,
    }, index=pvals.index)
    return out


def bonferroni(pvals: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Bonferroni correction (reference only).

    Returns DataFrame with 'p_raw', 'p_bonf', 'reject_bonf', 'rank'.
    """
    pvals = pvals.dropna()
    if pvals.empty:
        return pd.DataFrame(columns=['p_raw', 'p_bonf', 'reject_bonf', 'rank'])

    reject, p_adj, _, _ = multipletests(pvals.values, alpha=alpha, method='bonferroni')
    out = pd.DataFrame({
        'p_raw': pvals.values,
        'p_bonf': p_adj,
        'reject_bonf': reject,
        'rank': pvals.rank(method='min').astype(int).values,
    }, index=pvals.index)
    return out


def correction_summary(pvals: pd.Series, alpha: float = 0.10) -> dict:
    """Both corrections side-by-side + descriptive stats.

    Returns:
        {
          'n_tests': int,
          'n_significant_raw': int,    # raw p < alpha
          'n_significant_bh': int,
          'n_significant_bonf': int,
          'fdr_threshold': float,
          'bonf_threshold': float,
          'table': pd.DataFrame,       # both corrections combined
        }
    """
    pvals_clean = pvals.dropna()
    bh = fdr_bh(pvals_clean, alpha=alpha)
    bonf = bonferroni(pvals_clean, alpha=0.05)  # Bonferroni typically at 5%

    table = bh.join(bonf[['p_bonf', 'reject_bonf']])

    return {
        'n_tests': len(pvals_clean),
        'n_significant_raw': int((pvals_clean < alpha).sum()),
        'n_significant_bh': int(bh['reject_bh'].sum()),
        'n_significant_bonf': int(bonf['reject_bonf'].sum()),
        'fdr_threshold': alpha,
        'bonf_threshold': 0.05,
        'table': table,
    }
