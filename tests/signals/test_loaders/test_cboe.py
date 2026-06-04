import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from data_loaders.signals.cboe import CboeLoader, _parse_putcall_csv


FIXTURE = Path(__file__).parent.parent / 'fixtures' / 'cboe_putcall_sample.csv'


def test_parse_putcall_csv():
    text = FIXTURE.read_text()
    s = _parse_putcall_csv(text)
    assert len(s) == 4
    assert s.iloc[0] == 0.75
    assert s.name == 'cboe_putcall_equity'


def test_parse_putcall_no_pc_column_raises():
    text = "Date,Total Volume\n2024-01-02,1000\n"
    with pytest.raises(ValueError, match="PutCall column not found"):
        _parse_putcall_csv(text)


def test_putcall_fetch_via_get(tmp_path):
    with patch('data_loaders.signals.cboe.requests.get') as mock:
        mock.return_value.text = FIXTURE.read_text()
        mock.return_value.raise_for_status = lambda: None
        ldr = CboeLoader(cache_dir=tmp_path)
        s = ldr.get(signal_id=12, force=True)
    assert len(s) == 4
    assert s.name == 'cboe_putcall_equity'


def test_vix_term_contango(tmp_path):
    def _hist_df(values):
        idx = pd.date_range('2024-01-02', periods=len(values), freq='B', tz='America/New_York')
        return pd.DataFrame({'Adj Close': values, 'Close': values, 'Open': values, 'High': values, 'Low': values, 'Volume': [0]*len(values)}, index=idx)

    vix_vals    = [15.0, 16.0, 14.5]
    vix3m_vals  = [16.5, 16.2, 17.0]
    vix6m_vals  = [17.0, 17.5, 18.0]

    def side_effect(ticker):
        m = MagicMock()
        if ticker == '^VIX':
            m.history.return_value = _hist_df(vix_vals)
        elif ticker == '^VIX3M':
            m.history.return_value = _hist_df(vix3m_vals)
        else:  # '^VIX6M'
            m.history.return_value = _hist_df(vix6m_vals)
        return m

    with patch('data_loaders.signals.cboe.yf.Ticker', side_effect=side_effect):
        ldr = CboeLoader(cache_dir=tmp_path)
        s = ldr.get(signal_id=8, force=True)
    assert len(s) == 3
    # Row 0: 15 < 16.5 < 17.0 → contango = 1
    # Row 1: 16 < 16.2 < 17.5 → contango = 1
    # Row 2: 14.5 < 17.0 < 18.0 → contango = 1
    assert list(s.values) == [1, 1, 1]


def test_unmapped_signal_raises(tmp_path):
    ldr = CboeLoader(cache_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        ldr._fetch(signal_id=999)
