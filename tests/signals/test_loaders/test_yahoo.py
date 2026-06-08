import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from data_loaders.signals.yahoo import YahooLoader


def _mock_history_df(values):
    idx = pd.date_range('2024-01-02', periods=len(values), freq='B', tz='America/New_York')
    return pd.DataFrame({
        'Open': values, 'High': values, 'Low': values, 'Close': values,
        'Adj Close': values, 'Volume': [0] * len(values),
    }, index=idx)


def test_single_ticker_returns_adj_close(tmp_path):
    df = _mock_history_df([15.5, 16.0, 14.8])
    mock_t = MagicMock()
    mock_t.history.return_value = df
    with patch('data_loaders.signals.yahoo.yf.Ticker', return_value=mock_t):
        ldr = YahooLoader(cache_dir=tmp_path)
        s = ldr.get(signal_id=6, force=True)
    assert list(s.values) == [15.5, 16.0, 14.8]
    assert s.name == 'yahoo_^VIX'
    # tz_localize(None) means index is naive
    assert s.index.tz is None


def test_ratio_signal(tmp_path):
    num_df = _mock_history_df([4.0, 4.4, 4.2])
    den_df = _mock_history_df([2000.0, 2200.0, 2100.0])

    def side_effect(ticker):
        m = MagicMock()
        m.history.return_value = num_df if ticker == 'HG=F' else den_df
        return m

    with patch('data_loaders.signals.yahoo.yf.Ticker', side_effect=side_effect):
        ldr = YahooLoader(cache_dir=tmp_path)
        s = ldr.get(signal_id=42, force=True)  # Copper/Gold
    assert len(s) == 3
    # 4.0/2000 = 0.002, 4.4/2200 = 0.002, etc.
    assert abs(s.iloc[0] - 0.002) < 1e-9
    assert s.name == 'yahoo_HG=F_over_GC=F'


def test_unmapped_signal_raises(tmp_path):
    ldr = YahooLoader(cache_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        ldr._fetch(signal_id=999)


def test_empty_response_raises(tmp_path):
    mock_t = MagicMock()
    mock_t.history.return_value = pd.DataFrame()
    with patch('data_loaders.signals.yahoo.yf.Ticker', return_value=mock_t):
        ldr = YahooLoader(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="empty data"):
            ldr.get(signal_id=6, force=True)
