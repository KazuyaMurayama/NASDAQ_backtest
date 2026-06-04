import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pandas as pd
from signals.timing import apply_publication_lag


def _series(dates, vals):
    return pd.Series(vals, index=pd.to_datetime(dates))


def test_daily_lag_shifts_one_day():
    s = _series(['2024-01-02', '2024-01-03', '2024-01-04'], [1, 2, 3])
    out = apply_publication_lag(s, lag_type='daily')
    assert out.loc['2024-01-03'] == 1
    assert out.loc['2024-01-04'] == 2


def test_weekly_tue_lag_to_next_tue():
    # COT: Tue 公表 (2024-01-02 火) → 翌火 2024-01-09 close 適用
    s = _series(['2024-01-02', '2024-01-09'], [100, 200])
    out = apply_publication_lag(s, lag_type='weekly')
    assert out.loc['2024-01-09'] == 100
    assert ('2024-01-16' not in out.index) or (out.loc['2024-01-16'] == 200)


def test_event_lag_next_open():
    s = _series(['2024-01-31'], [3])
    out = apply_publication_lag(s, lag_type='event')
    assert out.index[0] >= pd.Timestamp('2024-02-01')


def test_invalid_lag_type_raises():
    import pytest
    s = _series(['2024-01-02'], [1])
    with pytest.raises(ValueError):
        apply_publication_lag(s, lag_type='quarterly')
