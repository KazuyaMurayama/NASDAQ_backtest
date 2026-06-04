import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from signals.lineage import generate_lineage_markdown

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'signals' / 'tier1_selection_20260603.csv'


def test_lineage_markdown_contains_all_52():
    md = generate_lineage_markdown(registry_path=CSV)
    for sid in range(1, 53):
        assert f"| {sid} |" in md, f"signal_id {sid} missing from lineage markdown"


def test_lineage_markdown_has_header_columns():
    md = generate_lineage_markdown(registry_path=CSV)
    assert "| ID | Name | Category | Asset | Source | Lag | Earliest | Cost | Priority |" in md


def test_lineage_markdown_priority_distribution():
    md = generate_lineage_markdown(registry_path=CSV)
    a_count = md.count('| A |')
    b_count = md.count('| B |')
    c_count = md.count('| C |')
    assert a_count == 31, f"expected 31 priority-A rows, got {a_count}"
    assert b_count == 16
    assert c_count == 5
