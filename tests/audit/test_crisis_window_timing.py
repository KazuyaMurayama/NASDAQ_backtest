import numpy as np
from src.audit.crisis_window_timing_20260621 import (
    crisis_window_dd_compare, sign_test_brake_beats_twin)


def test_window_dd_compare_brake_shallower_in_crash():
    """In a crash window, a brake that cut exposure there has a shallower
    windowed MaxDD than a twin that did not."""
    n = 200
    r_twin = np.full(n, 0.001)
    r_twin[100:140] = -0.03                       # deep crash in window
    r_brake = r_twin.copy()
    r_brake[100:140] = -0.01                       # brake softened the crash
    stress = {"crashwin": np.zeros(n, dtype=bool)}
    stress["crashwin"][95:145] = True
    rows = crisis_window_dd_compare(r_brake, r_twin, stress)
    assert len(rows) == 1
    row = rows[0]
    assert row["window"] == "crashwin"
    assert row["brake_maxdd"] > row["twin_maxdd"]   # brake less-negative = shallower
    assert row["brake_shallower"] is True


def test_window_dd_compare_skips_empty_window():
    n = 50
    stress = {"empty": np.zeros(n, dtype=bool)}     # no days
    rows = crisis_window_dd_compare(np.zeros(n), np.zeros(n), stress)
    assert rows == []


def test_sign_test_counts_and_binomial():
    rows = [
        {"window": "w1", "brake_maxdd": -0.10, "twin_maxdd": -0.20, "dd_edge_pp": 10.0, "brake_shallower": True},
        {"window": "w2", "brake_maxdd": -0.15, "twin_maxdd": -0.25, "dd_edge_pp": 10.0, "brake_shallower": True},
        {"window": "w3", "brake_maxdd": -0.30, "twin_maxdd": -0.28, "dd_edge_pp": -2.0, "brake_shallower": False},
        {"window": "w4", "brake_maxdd": -0.05, "twin_maxdd": -0.12, "dd_edge_pp": 7.0, "brake_shallower": True},
        {"window": "w5", "brake_maxdd": -0.08, "twin_maxdd": -0.18, "dd_edge_pp": 10.0, "brake_shallower": True},
    ]
    res = sign_test_brake_beats_twin(rows)
    assert res["n_windows"] == 5
    assert res["n_shallower"] == 4
    assert res["n_deeper"] == 1
    assert 0.0 <= res["binom_p_onesided"] <= 1.0
    # one-sided binom P(X>=4 | n=5, 0.5) = (C(5,4)+C(5,5))*0.5^5 = 6/32 = 0.1875
    assert abs(res["binom_p_onesided"] - 0.1875) < 1e-9
    # mean of dd_edge_pp = (10+10-2+7+10)/5 = 7.0
    assert abs(res["mean_dd_edge_pp"] - 7.0) < 1e-9


def test_sign_test_all_shallower_gives_small_p():
    """5/5 shallower -> one-sided binom p = 0.5^5 = 0.03125."""
    rows = [{"window": f"w{i}", "brake_maxdd": -0.1, "twin_maxdd": -0.2,
             "dd_edge_pp": 10.0, "brake_shallower": True} for i in range(5)]
    res = sign_test_brake_beats_twin(rows)
    assert res["n_shallower"] == 5
    assert abs(res["binom_p_onesided"] - 0.03125) < 1e-9
