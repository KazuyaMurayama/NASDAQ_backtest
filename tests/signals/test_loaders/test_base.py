import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import pandas as pd
from data_loaders.signals._base import SignalLoader


class FakeLoader(SignalLoader):
    source_name = 'fake'
    call_count = 0

    def _fetch(self, signal_id: int) -> pd.Series:
        FakeLoader.call_count += 1
        idx = pd.date_range('2024-01-01', periods=5, freq='B')
        return pd.Series([1, 2, 3, 4, 5], index=idx, name=f"sig_{signal_id}")


def test_first_call_hits_fetch(tmp_path):
    FakeLoader.call_count = 0
    ldr = FakeLoader(cache_dir=tmp_path)
    s = ldr.get(signal_id=99)
    assert len(s) == 5
    assert FakeLoader.call_count == 1


def test_second_call_hits_cache(tmp_path):
    FakeLoader.call_count = 0
    ldr = FakeLoader(cache_dir=tmp_path)
    ldr.get(signal_id=99)
    ldr.get(signal_id=99)
    assert FakeLoader.call_count == 1


def test_force_refresh_bypasses_cache(tmp_path):
    FakeLoader.call_count = 0
    ldr = FakeLoader(cache_dir=tmp_path)
    ldr.get(signal_id=99)
    ldr.get(signal_id=99, force=True)
    assert FakeLoader.call_count == 2
