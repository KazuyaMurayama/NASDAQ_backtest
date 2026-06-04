import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
import pytest
from signals.multiplicity import fdr_bh, bonferroni, correction_summary


def test_fdr_bh_all_significant_below_threshold():
    pvals = pd.Series([0.001, 0.002, 0.003, 0.01, 0.02])
    out = fdr_bh(pvals, alpha=0.10)
    assert out['reject_bh'].all()


def test_fdr_bh_uniform_distribution_few_rejections():
    """Uniform p-values under H0 -> roughly alpha fraction expected to falsely reject."""
    rng = np.random.default_rng(42)
    pvals = pd.Series(rng.uniform(0, 1, size=100))
    out = fdr_bh(pvals, alpha=0.10)
    # With FDR control at 10%, on uniform pvals (all true H0), should have few rejections
    assert out['reject_bh'].sum() <= 20  # very loose upper bound


def test_fdr_bh_adjusted_p_monotonic_in_rank():
    """BH-adjusted p-values should be non-decreasing when sorted by raw p."""
    pvals = pd.Series([0.001, 0.05, 0.5, 0.8, 0.99])
    out = fdr_bh(pvals)
    out_sorted = out.sort_values('p_raw')
    assert (out_sorted['p_bh'].diff().dropna() >= -1e-9).all()


def test_bonferroni_strict():
    """Bonferroni rejects only when p < alpha/n."""
    pvals = pd.Series([0.001, 0.01, 0.05, 0.1])
    out = bonferroni(pvals, alpha=0.05)
    # alpha/n = 0.05/4 = 0.0125 - only 0.001 and 0.01 reject
    assert out['reject_bonf'].sum() == 2
    assert out.loc[0, 'reject_bonf']
    assert out.loc[1, 'reject_bonf']


def test_correction_summary_returns_all_keys():
    rng = np.random.default_rng(0)
    pvals = pd.Series(np.concatenate([rng.uniform(0, 0.05, 10), rng.uniform(0, 1, 90)]))
    out = correction_summary(pvals, alpha=0.10)
    assert set(out.keys()) == {'n_tests', 'n_significant_raw', 'n_significant_bh',
                                'n_significant_bonf', 'fdr_threshold', 'bonf_threshold',
                                'table'}
    assert out['n_tests'] == 100
    assert isinstance(out['table'], pd.DataFrame)


def test_empty_pvals_returns_empty():
    pvals = pd.Series([], dtype='float64')
    out = fdr_bh(pvals)
    assert out.empty


def test_nan_pvals_dropped():
    pvals = pd.Series([0.01, float('nan'), 0.02], index=['a', 'b', 'c'])
    out = fdr_bh(pvals)
    assert len(out) == 2
    assert 'b' not in out.index
