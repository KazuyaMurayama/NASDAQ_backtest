import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import pandas as pd
import pytest
from unittest.mock import patch
from data_loaders.signals.cftc import CftcLoader, _parse_cot_csv, _zscore_52w


FIXTURE = Path(__file__).parent.parent / 'fixtures' / 'cot_nq_sample.csv'


def test_parse_filters_to_contract():
    text = FIXTURE.read_text()
    net = _parse_cot_csv(text, '209742')
    assert len(net) == 6
    assert net.iloc[0] == 10000  # 50000 - 40000
    assert net.iloc[-1] == 30000  # 60000 - 30000


def test_parse_filters_unknown_returns_empty():
    text = FIXTURE.read_text()
    net = _parse_cot_csv(text, '999999')
    assert net.empty


def test_zscore_52w_returns_nan_before_52():
    s = pd.Series(range(60))
    z = _zscore_52w(s)
    assert z.iloc[:51].isna().all()
    assert not z.iloc[52:].isna().any()


def test_fetch_raises_for_insufficient_data(tmp_path):
    """6-week fixture is too short for 52-week z-score; _zscore returns NaNs and net is empty after filter."""
    with patch('data_loaders.signals.cftc.requests.get') as mock:
        mock.return_value.text = FIXTURE.read_text()
        mock.return_value.raise_for_status = lambda: None
        ldr = CftcLoader(cache_dir=tmp_path)
        s = ldr.get(signal_id=15, force=True)
    # 6 weeks of data, all z-score values NaN, but the series itself is returned
    assert len(s) == 6
    assert s.isna().all()  # all NaN due to insufficient window


def test_unmapped_signal_raises(tmp_path):
    ldr = CftcLoader(cache_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        ldr._fetch(signal_id=999)
