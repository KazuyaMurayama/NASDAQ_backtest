import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import pandas as pd
import pytest
from unittest.mock import patch
from data_loaders.signals.fred import FredLoader


FIXTURE = Path(__file__).parent.parent / 'fixtures' / 'fred_BAMLH0A0HYM2_sample.csv'


def test_fred_parses_csv(tmp_path):
    with patch('data_loaders.signals.fred.requests.get') as mock:
        mock.return_value.text = FIXTURE.read_text()
        mock.return_value.raise_for_status = lambda: None
        ldr = FredLoader(cache_dir=tmp_path)
        s = ldr.get(signal_id=21, force=True)
    assert len(s) == 4
    assert s.iloc[0] == 3.45
    assert s.name == 'fred_BAMLH0A0HYM2'


def test_unmapped_signal_raises(tmp_path):
    ldr = FredLoader(cache_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        ldr._fetch(signal_id=999)
