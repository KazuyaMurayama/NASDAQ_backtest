"""Signal metadata model and registry loader.

CSV schema (single source: data/signals/tier1_selection_<date>.csv):
  signal_id        int unique
  name             human-readable
  category         A_Breadth..J_NLP (spec §4.2 prefix)
  view             Trader | Actuary | HF
  target_assets    pipe-separated subset of {N,G,B}
  quantize_scheme  binary_threshold | quantile_cut | zscore_band
  q_levels         2 or 4
  priority         A (◎) | B (○) | C (△)
  source_module    matches src/data_loaders/signals/*.py stem
  publication_lag  daily | weekly | monthly | event
  earliest_date    YYYY-MM-DD
  cost_tier        free | low_paid | mid_paid
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union
import pandas as pd


_VALID_PRIORITY = {'A', 'B', 'C'}
_VALID_QUANT = {'binary_threshold', 'quantile_cut', 'zscore_band'}
_VALID_LAG = {'daily', 'weekly', 'monthly', 'event'}
_VALID_COST = {'free', 'low_paid', 'mid_paid'}


@dataclass(frozen=True)
class SignalMeta:
    signal_id: int
    name: str
    category: str
    view: str
    target_assets: List[str]
    quantize_scheme: str
    q_levels: int
    priority: str
    source_module: str
    publication_lag: str
    earliest_date: str
    cost_tier: str

    def __post_init__(self):
        if self.priority not in _VALID_PRIORITY:
            raise ValueError(f"priority must be A/B/C, got {self.priority}")
        if self.quantize_scheme not in _VALID_QUANT:
            raise ValueError(f"quantize_scheme invalid: {self.quantize_scheme}")
        if self.q_levels not in (2, 4):
            raise ValueError(f"q_levels must be 2 or 4, got {self.q_levels}")
        if self.publication_lag not in _VALID_LAG:
            raise ValueError(f"publication_lag invalid: {self.publication_lag}")
        if self.cost_tier not in _VALID_COST:
            raise ValueError(f"cost_tier invalid: {self.cost_tier}")
        for a in self.target_assets:
            if a not in ('N', 'G', 'B'):
                raise ValueError(f"target_assets entry invalid: {a}")


def load_registry(path_or_buf: Union[str, Path, "IO"]) -> List[SignalMeta]:
    df = pd.read_csv(path_or_buf)
    out: List[SignalMeta] = []
    for _, row in df.iterrows():
        out.append(SignalMeta(
            signal_id=int(row['signal_id']),
            name=str(row['name']),
            category=str(row['category']),
            view=str(row['view']),
            target_assets=str(row['target_assets']).split('|'),
            quantize_scheme=str(row['quantize_scheme']),
            q_levels=int(row['q_levels']),
            priority=str(row['priority']),
            source_module=str(row['source_module']),
            publication_lag=str(row['publication_lag']),
            earliest_date=str(row['earliest_date']),
            cost_tier=str(row['cost_tier']),
        ))
    return out
