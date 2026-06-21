"""Crisis-window (intact-path) timing test for DD-reduction brakes.

Block-bootstrap of full-series MaxDD is invalid: block=21 shuffles apart the
multi-year crash sequences that DEFINE MaxDD, so timing_P_maxdd is pinned near
0.5 regardless of true timing skill (the repo's own multimetric docstring warns
this for Worst10Y). This module tests timing the correct way: on INTACT crisis
windows (no resampling), does the brake achieve a shallower windowed MaxDD than
its equal-average-exposure uniform-delever twin? A cross-window sign test then
asks whether the brake systematically lands its cuts inside crises."""
from __future__ import annotations
import numpy as np
from math import comb

from src.audit.run_p09_tqqq_validate_20260611 import _maxdd_from_returns


def crisis_window_dd_compare(r_brake, r_twin, stress):
    """For each stress window, compute windowed MaxDD of brake vs twin on the
    INTACT (non-resampled) path slice. stress = {name: bool_mask}. Returns a list
    of dicts (one per NON-EMPTY window): window, brake_maxdd, twin_maxdd,
    dd_edge_pp (=(brake-twin)*100; >0 = brake shallower), brake_shallower(bool)."""
    r_brake = np.asarray(r_brake, float)
    r_twin = np.asarray(r_twin, float)
    rows = []
    for name, mask in stress.items():
        m = np.asarray(mask, bool)
        if m.sum() == 0:
            continue
        bdd = _maxdd_from_returns(r_brake[m])
        tdd = _maxdd_from_returns(r_twin[m])
        rows.append({
            "window": name,
            "brake_maxdd": bdd,
            "twin_maxdd": tdd,
            "dd_edge_pp": (bdd - tdd) * 100.0,
            "brake_shallower": bool(bdd > tdd),
        })
    return rows


def sign_test_brake_beats_twin(rows):
    """Cross-window sign test. Returns n_windows, n_shallower, n_deeper,
    binom_p_onesided = P(X >= n_shallower | n, 0.5) (one-sided), and
    mean_dd_edge_pp (avg brake-twin DD edge in pp). One-sided binomial computed
    exactly via math.comb (no scipy dependency)."""
    n = len(rows)
    k = sum(1 for r in rows if r["brake_shallower"])
    deeper = n - k
    if n > 0:
        p = sum(comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    else:
        p = 1.0
    mean_edge = float(np.mean([r["dd_edge_pp"] for r in rows])) if n else 0.0
    return {
        "n_windows": n,
        "n_shallower": k,
        "n_deeper": deeper,
        "binom_p_onesided": float(p),
        "mean_dd_edge_pp": mean_edge,
    }
