import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pandas as pd
from fetch_gbp_data import fetch_gbp, SIGNAL_GBPJPY, SIGNAL_GBPUSD


class _FakeLoader:
    """Records get() calls and returns a deterministic 3-point series."""
    def __init__(self):
        self.calls = []

    def get(self, signal_id, force=False):
        self.calls.append(signal_id)
        idx = pd.date_range('2024-01-02', periods=3, freq='B')
        return pd.Series([180.0, 181.0, 182.0], index=idx,
                         name=f'yahoo_sig{signal_id}')


def test_fetch_gbp_writes_two_csvs(tmp_path):
    ldr = _FakeLoader()
    out = fetch_gbp(loader=ldr, out_dir=tmp_path, write=True)
    assert set(out.keys()) == {'gbpjpy', 'gbpusd'}
    assert (tmp_path / 'gbpjpy_daily.csv').exists()
    assert (tmp_path / 'gbpusd_daily.csv').exists()


def test_fetch_gbp_primary_is_gbpjpy_first(tmp_path):
    ldr = _FakeLoader()
    fetch_gbp(loader=ldr, out_dir=tmp_path, write=False)
    # GBP/JPY (primary) must be fetched before GBP/USD (secondary)
    assert ldr.calls == [SIGNAL_GBPJPY, SIGNAL_GBPUSD]


def test_fetch_gbp_no_write(tmp_path):
    ldr = _FakeLoader()
    out = fetch_gbp(loader=ldr, out_dir=tmp_path, write=False)
    assert len(out['gbpjpy']) == 3
    assert not (tmp_path / 'gbpjpy_daily.csv').exists()
