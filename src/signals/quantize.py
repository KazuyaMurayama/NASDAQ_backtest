"""Signal quantization functions.

Three schemas supported (matches metadata.quantize_scheme):
  - binary_threshold: 0/1 by simple threshold + direction
  - quantile_cut:     0..L-1 by L-quantile (optionally rolling)
  - zscore_band:      0/1/2 by z-score lower/upper bands

All return pd.Series aligned to input index; NaN propagates.

Implementation notes
--------------------
- ``quantile_cut`` uses :func:`pandas.qcut` for the static (window=None)
  case to guarantee equal-sized buckets (perfect 25/25/25/25 on 100
  evenly-spaced values). The original plan multiplied ``rank(pct=True)``
  by ``levels`` and astype'd to Int8, but that triggers a TypeError on
  pandas 2.3 (float->Int8 unsafe cast) and skews boundary counts. The
  test ``abs(c - 25) <= 1`` passes either way; qcut is safer and
  matches §4.1 semantics more directly.
- For the rolling case, we use a window of ``window + 1`` so the first
  valid bucket appears at index ``window`` (i.e. the first ``window``
  observations have no historical reference and return NaN). This
  matches the test expectation ``out.iloc[:window].isna().all()``.
"""
from __future__ import annotations
from typing import Optional, Literal
import numpy as np
import pandas as pd


def binary_threshold(
    s: pd.Series,
    threshold: float,
    direction: Literal['above', 'below'] = 'above',
) -> pd.Series:
    """Binary 0/1 by ``s > threshold`` (above) or ``s < threshold`` (below).

    NaN inputs propagate to NaN outputs (Int8 nullable dtype).
    """
    if direction not in ('above', 'below'):
        raise ValueError(
            f"direction must be 'above' or 'below', got {direction!r}"
        )
    mask = s.notna()
    if direction == 'above':
        bits = (s > threshold)
    else:
        bits = (s < threshold)
    out = bits.astype('Int8')
    out = out.where(mask)
    return out


def quantile_cut(
    s: pd.Series,
    levels: int = 4,
    window: Optional[int] = None,
) -> pd.Series:
    """Bucket ``s`` into ``levels`` quantile bins ``0..levels-1``.

    Parameters
    ----------
    s : pd.Series
    levels : int, default 4
        Number of quantile buckets. Must be >= 2.
    window : int or None, default None
        If None, compute quantiles over the full series (in-sample).
        If int, use a rolling window of ``window`` historical
        observations to bucket each subsequent point. The first
        ``window`` outputs are NaN.
    """
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")

    if window is None:
        # pd.qcut with duplicates='drop' for robustness on ties.
        try:
            out = pd.qcut(s, levels, labels=False, duplicates='drop')
        except ValueError:
            # All values identical etc. -> all NaN
            return pd.Series(pd.NA, index=s.index, dtype='Int8')
        return out.astype('Int8').where(s.notna())

    # Rolling: use window+1 so that index `window` is the first
    # value with `window` prior observations to rank against.
    def _bin(x: pd.Series) -> float:
        if x.isna().iloc[-1]:
            return np.nan
        pct = float(x.rank(pct=True).iloc[-1])
        return float(min(int(pct * levels), levels - 1))

    raw = s.rolling(window=window + 1, min_periods=window + 1).apply(
        _bin, raw=False
    )
    # Cast float -> Int8 via int64 intermediate (safe cast).
    out = pd.Series(pd.NA, index=s.index, dtype='Int8')
    valid = raw.notna()
    if valid.any():
        out.loc[valid] = raw.loc[valid].astype('int64').astype('Int8')
    return out


def zscore_band(
    s: pd.Series,
    lower: float = -1.0,
    upper: float = 1.0,
    window: Optional[int] = None,
) -> pd.Series:
    """Three-level signal by z-score: 0 if z<=lower, 2 if z>=upper, else 1.

    Parameters
    ----------
    s : pd.Series
    lower, upper : float
        Lower/upper z-score thresholds (default -1, +1).
    window : int or None, default None
        If None, use full-sample mean/std. Otherwise rolling.
    """
    if window is None:
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            # Degenerate: all values identical -> all middle band
            out = pd.Series(1, index=s.index, dtype='Int8')
            return out.where(s.notna())
        z = (s - mu) / sd
    else:
        mu = s.rolling(window).mean()
        sd = s.rolling(window).std(ddof=0)
        z = (s - mu) / sd

    out = pd.Series(1, index=s.index, dtype='Int8')
    out = out.mask(z <= lower, 0)
    out = out.mask(z >= upper, 2)
    return out.where(s.notna() & z.notna() if window else s.notna())
