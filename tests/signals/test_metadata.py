import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from signals.metadata import SignalMeta, load_registry

FIXTURE = Path(__file__).parent / 'fixtures' / 'metadata_sample.csv'


def test_load_registry_returns_list_of_signalmeta():
    metas = load_registry(FIXTURE)
    assert len(metas) == 5
    assert all(isinstance(m, SignalMeta) for m in metas)


def test_signalmeta_field_types():
    metas = load_registry(FIXTURE)
    m = metas[0]
    assert m.signal_id == 1
    assert m.name == 'NDX 200DMA breadth'
    assert m.category == 'A_Breadth'
    assert m.target_assets == ['N']
    assert m.q_levels == 4
    assert m.priority == 'A'
    assert m.cost_tier == 'free'


def test_multi_target_assets_split():
    metas = load_registry(FIXTURE)
    fed = next(m for m in metas if m.signal_id == 50)
    assert fed.target_assets == ['B', 'N']


def test_priority_invalid_raises():
    import pandas as pd
    df = pd.read_csv(FIXTURE)
    df.loc[0, 'priority'] = 'X'
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    import pytest
    with pytest.raises(ValueError, match="priority must be A/B/C"):
        load_registry(buf)
