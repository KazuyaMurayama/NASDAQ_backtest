"""
src/audit/labor_zero_v6_stats_helpers_20260702.py
=================================================
Wilson CI + paired-path comparison helpers for the labor-zero v6 critical
verification campaign (plan: docs/superpowers/plans/
2026-07-02-labor-zero-v6-critical-verification.md).

Rationale: at N=2000 the resolution of P(labor=0) is 0.0005 and headline claims
differ by ~10 paths. Independent CIs cannot settle such differences; comparing
two configs ON THE SAME PATH SET (same seed -> same bootstrap paths) and
counting discordant paths (McNemar) can.

Self-test: run this file directly.  ASCII-only prints.
"""
from __future__ import annotations

from math import sqrt

import numpy as np

try:                                     # exact binomial if scipy present
    from scipy.stats import binomtest

    def _binom_two_sided(k, n):
        return float(binomtest(k, n, 0.5).pvalue)
except Exception:                        # normal approximation fallback
    from math import erf

    def _binom_two_sided(k, n):
        if n == 0:
            return 1.0
        z = abs(k - n * 0.5) / sqrt(n * 0.25)
        return float(max(0.0, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))))))


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (center - half, center + half)


def paired_diff(labors_a, labors_b):
    """Same-seed same-paths comparison of two configs on P(labor==0).
    labors_a / labors_b: int arrays of labor_years per path (identical path sets,
    i.e. both configs were run with the same seed and identical rng consumption).
    Returns dict(n, p_a, p_b, a_only, b_only, mcnemar_p) where a_only = number of
    paths where A achieves labor==0 but B does not."""
    a0 = (np.asarray(labors_a) == 0)
    b0 = (np.asarray(labors_b) == 0)
    if len(a0) != len(b0):
        raise ValueError("paired_diff requires equal-length labor arrays")
    a_only = int(np.sum(a0 & ~b0))
    b_only = int(np.sum(~a0 & b0))
    m = a_only + b_only
    p = 1.0 if m == 0 else _binom_two_sided(min(a_only, b_only), m)
    return dict(n=int(len(a0)), p_a=float(a0.mean()), p_b=float(b0.mean()),
                a_only=a_only, b_only=b_only, mcnemar_p=p)


def _self_test():
    lo, hi = wilson_ci(1789, 2000)                      # p=0.8945
    assert 0.880 < lo < 0.8945 < hi < 0.908, (lo, hi)
    lo1, hi1 = wilson_ci(2000, 2000)                    # p=1 edge
    assert hi1 <= 1.0 + 1e-12 and lo1 > 0.995, (lo1, hi1)
    a = np.zeros(100, int)
    b = np.zeros(100, int)
    b[:20] = 3                                          # A better on 20 paths
    r = paired_diff(a, b)
    assert r["a_only"] == 20 and r["b_only"] == 0 and r["mcnemar_p"] < 0.001, r
    r2 = paired_diff(a, a)
    assert r2["mcnemar_p"] == 1.0 and r2["a_only"] == 0
    print("stats helpers SELF-TEST PASS  (wilson edge ok, mcnemar ok)")


if __name__ == "__main__":
    _self_test()
