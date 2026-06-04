import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from signals.metadata import load_registry

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'signals' / 'tier1_selection_20260603.csv'


def test_total_52_signals():
    metas = load_registry(CSV)
    assert len(metas) == 52


def test_priority_counts_match_spec():
    metas = load_registry(CSV)
    a = sum(1 for m in metas if m.priority == 'A')
    b = sum(1 for m in metas if m.priority == 'B')
    c = sum(1 for m in metas if m.priority == 'C')
    assert (a, b, c) == (31, 16, 5), f"got A{a}/B{b}/C{c}, expected 31/16/5"


def test_signal_ids_unique_and_sequential():
    metas = load_registry(CSV)
    ids = [m.signal_id for m in metas]
    assert len(set(ids)) == 52
    assert min(ids) == 1 and max(ids) == 52


def test_categories_match_spec():
    metas = load_registry(CSV)
    expected = {'A_Breadth', 'B_Vol', 'C_Sentiment', 'D_Credit',
                'E_YieldCurve', 'F_MacroNowcast', 'G_Earnings',
                'H_CrossAsset', 'I_Calendar', 'J_NLP'}
    got = {m.category for m in metas}
    assert got == expected


def test_priority_A_set_matches_spec_4_3():
    metas = load_registry(CSV)
    a_ids = sorted(m.signal_id for m in metas if m.priority == 'A')
    expected = sorted([1,2,3,6,7,8,9,10,12,13,15,16,17,21,23,26,27,28,29,30,
                       32,34,36,37,38,40,41,42,46,49,50])
    assert a_ids == expected


def test_priority_C_set_matches_spec_4_3():
    metas = load_registry(CSV)
    c_ids = sorted(m.signal_id for m in metas if m.priority == 'C')
    assert c_ids == [5, 20, 25, 45, 48]
