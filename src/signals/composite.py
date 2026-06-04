"""PCA composite signal builder for Phase B §5.6.

Composes correlated component signals into a single first-principal-component
series. Re-quantized via signals.quantize.quantile_cut to match standard
0/1/2/3 schema for downstream IC/hit-rate evaluation.

Sign is normalized: PCA loadings are flipped if needed so the composite
loads positively on the first component (handles arbitrary eigenvector sign).
"""
from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .quantize import quantile_cut


COMPOSITE_BLOCKS: dict[str, List[int]] = {
    'sentiment':      [12, 13, 14, 15],
    'credit_stress':  [21, 23, 36],
    'macro_nowcast':  [32, 34, 36],
    'yield_curve':    [26, 27, 30],
}


def build_composite(
    signals: pd.DataFrame,
    method: str = 'pca_first',
    standardize: bool = True,
) -> pd.Series:
    """First principal component of (standardized) component signals.

    Parameters
    ----------
    signals : pd.DataFrame
        Component signal columns indexed by date.
    method : str, default 'pca_first'
        Currently only 'pca_first' is supported.
    standardize : bool, default True
        Whether to z-score each component before PCA (recommended for
        components on heterogeneous scales).

    Returns
    -------
    pd.Series
        Continuous composite indexed identically to ``signals``. Rows where
        any component is NaN become NaN in the output. Sign is normalized so
        the composite correlates positively with the unweighted sum of
        components (interpretable direction).
    """
    if method != 'pca_first':
        raise NotImplementedError(
            f"method={method!r} not supported (only 'pca_first')"
        )

    df = signals.dropna()
    if df.empty or df.shape[1] < 2:
        return pd.Series(
            np.nan, index=signals.index, name='composite', dtype='float64'
        )

    X = df.values.astype(float)
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).ravel()

    # Sign normalization: choose sign so the composite positively correlates
    # with the (un-standardized) sum of components — gives interpretable
    # direction independent of eigenvector sign ambiguity.
    summed = df.sum(axis=1).values
    if np.std(pc1) > 0 and np.std(summed) > 0:
        if np.corrcoef(pc1, summed)[0, 1] < 0:
            pc1 = -pc1

    out = pd.Series(pc1, index=df.index, name='composite', dtype='float64')
    # Restore original index alignment (rows with NaN components -> NaN).
    return out.reindex(signals.index)


def quantize_composite(
    composite: pd.Series,
    levels: int = 4,
    window: Optional[int] = None,
) -> pd.Series:
    """Wrap signals.quantize.quantile_cut for the composite.

    Parameters
    ----------
    composite : pd.Series
        Continuous composite (e.g. output of :func:`build_composite`).
    levels : int, default 4
        Quantile bucket count (standard schema: 4 -> 0/1/2/3).
    window : int or None, default None
        Rolling window; ``None`` for full-sample quantiles.
    """
    return quantile_cut(composite, levels=levels, window=window)
