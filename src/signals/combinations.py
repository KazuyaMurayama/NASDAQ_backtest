"""Signal combinations (AND/OR) for Phase C top candidate exploration.

Builds combined binary signals from top PASS signals (e.g., 'Sentiment AND Credit').
"""
from __future__ import annotations
from typing import Literal
import pandas as pd


def combine_signals(
    s1: pd.Series,
    s2: pd.Series,
    operator: Literal['AND', 'OR'],
    threshold1: int = 1,
    threshold2: int = 1,
) -> pd.Series:
    """Combine two quantized signals into a binary 0/1 series.

    Args:
        s1, s2: quantized signals (0/1 or 0/1/2/3).
        operator: 'AND' (both signals >= threshold) or 'OR' (either >= threshold).
        threshold1, threshold2: minimum value to consider each signal "on".

    Returns:
        Binary 0/1 series aligned by date (inner join).
    """
    df = pd.concat([s1.rename('s1'), s2.rename('s2')], axis=1, join='inner').dropna()
    b1 = (df['s1'] >= threshold1).astype('int8')
    b2 = (df['s2'] >= threshold2).astype('int8')
    if operator == 'AND':
        return (b1 & b2).rename('combined')
    if operator == 'OR':
        return (b1 | b2).rename('combined')
    raise ValueError(f"operator must be 'AND'/'OR', got {operator}")
