"""Apply publication lag to prevent look-ahead bias.

Lag types (matches metadata.publication_lag):
  daily   : t-1 close known by t open  -> shift +1 business day
  weekly  : Tue publish -> next Tue close apply -> shift +5 business days
  monthly : kohyobi +1 eigyobi -> shift +1 business day
  event   : kohyobi 21:00 ET -> next session -> shift +1 business day
"""
from __future__ import annotations
from typing import Literal
import pandas as pd
from pandas.tseries.offsets import BusinessDay


_VALID = {'daily', 'weekly', 'monthly', 'event'}


def apply_publication_lag(
    s: pd.Series,
    lag_type: Literal['daily', 'weekly', 'monthly', 'event'],
) -> pd.Series:
    if lag_type not in _VALID:
        raise ValueError(f"lag_type invalid: {lag_type}")
    if lag_type == 'daily':
        shifted_idx = s.index + BusinessDay(1)
    elif lag_type == 'weekly':
        shifted_idx = s.index + BusinessDay(5)
    elif lag_type == 'monthly':
        shifted_idx = s.index + BusinessDay(1)
    else:  # event
        shifted_idx = s.index + BusinessDay(1)
    return pd.Series(s.values, index=shifted_idx)
