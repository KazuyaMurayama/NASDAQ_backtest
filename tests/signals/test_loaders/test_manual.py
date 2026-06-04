import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import pandas as pd
import pytest
from data_loaders.signals.manual import ManualLoader


def _write_manual_csv(tmp_path, fname, rows):
    p = tmp_path / fname
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return p


def test_load_aaii_csv(tmp_path, monkeypatch):
    # Override _MANUAL_DIR to use tmp_path
    monkeypatch.setattr(ManualLoader, '_MANUAL_DIR', tmp_path)
    _write_manual_csv(tmp_path, 'aaii_weekly.csv',
                      [{'Date': '2024-01-04', 'value': 0.4},
                       {'Date': '2024-01-11', 'value': 0.5}])
    ldr = ManualLoader(cache_dir=tmp_path / '_cache')
    s = ldr.get(signal_id=13, force=True)
    assert len(s) == 2
    assert s.iloc[0] == 0.4
    assert s.name == 'manual_13'


def test_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ManualLoader, '_MANUAL_DIR', tmp_path)
    ldr = ManualLoader(cache_dir=tmp_path / '_cache')
    with pytest.raises(FileNotFoundError, match="Manual CSV not found"):
        ldr._fetch(signal_id=13)


def test_missing_value_column_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ManualLoader, '_MANUAL_DIR', tmp_path)
    _write_manual_csv(tmp_path, 'aaii_weekly.csv',
                      [{'Date': '2024-01-04', 'level': 0.4}])  # wrong column name
    ldr = ManualLoader(cache_dir=tmp_path / '_cache')
    with pytest.raises(ValueError, match="missing 'value' column"):
        ldr._fetch(signal_id=13)


def test_unmapped_signal_raises(tmp_path):
    ldr = ManualLoader(cache_dir=tmp_path / '_cache')
    with pytest.raises(NotImplementedError):
        ldr._fetch(signal_id=999)


def test_signal_to_file_covers_tier1_manual_ids():
    """All 16 signals marked source_module='manual' in tier1 CSV should have an entry."""
    repo_root = Path(__file__).resolve().parents[3]
    tier1 = pd.read_csv(repo_root / 'data' / 'signals' / 'tier1_selection_20260603.csv')
    manual_ids = set(tier1[tier1['source_module'] == 'manual']['signal_id'])
    mapped_ids = set(ManualLoader._SIGNAL_TO_FILE.keys())
    missing = manual_ids - mapped_ids
    assert not missing, f"Manual signals not mapped: {missing}"
