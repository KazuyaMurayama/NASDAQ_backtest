import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from verify_phase_a import verify


def test_verify_passes():
    r = verify()
    assert r['total'] == 52
    assert r['loaders_configured'] == 5  # fred, yahoo, cftc, cboe, manual
    assert r['schemes_mapped'] >= 2  # binary_threshold + quantile_cut at minimum
    assert r['lag_types_covered'] >= 3  # daily, weekly, event used; monthly possibly


def test_deferred_signals_listed():
    r = verify()
    # fedwatch (signal 30), google_trends (signals 49, 52) are deferred
    deferred = {sid for sid, _ in r['deferred_signals']}
    assert 30 in deferred
    assert 49 in deferred
    assert 52 in deferred
