"""SignalLoader abstract base + disk cache (parquet)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import pandas as pd


class SignalLoader(ABC):
    source_name: str = "unknown"

    def __init__(self, cache_dir: Union[str, Path] = "data/signals/_cache"):
        self.cache_dir = Path(cache_dir) / self.source_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, signal_id: int) -> Path:
        return self.cache_dir / f"signal_{signal_id}.parquet"

    def get(self, signal_id: int, force: bool = False) -> pd.Series:
        p = self._cache_path(signal_id)
        if p.exists() and not force:
            df = pd.read_parquet(p)
            return df.iloc[:, 0]
        s = self._fetch(signal_id)
        s.to_frame(name=s.name or f"sig_{signal_id}").to_parquet(p)
        return s

    @abstractmethod
    def _fetch(self, signal_id: int) -> pd.Series:
        ...
