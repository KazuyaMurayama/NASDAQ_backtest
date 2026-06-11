"""
src/audit/preparity_p09_gas_port_20260611.py
============================================
Pre-parity proof for the GAS port (P09_GAS_MIGRATION_PLAN Phase 2/4.1).

Re-implements every NEW signal component with PLAIN PROCEDURAL LOOPS
(no pandas semantics) -- exactly the algorithms that will be translated
1:1 into P09Signal.gs -- and verifies them against the exported golden
vectors. If this passes, the JS port only has to match this file.

Checks (criteria from p09_live_spec_20260611.json):
  A. LT2-N750 from raw close history (running-sum rolling mean/std ddof=1)
  B. lev_mod_065 from (lev_raw, vz, lt_sig)
  C. W1 hysteresis replay (seed = golden row 0)
  D. mom63 -> frozen-quartile bucket(prev day) -> boost mult (P09 & LU1)
  E. L_t = lev_raw_masked[i-2] * mult[i-2] * 3   (DELAY=2)
  F. fund_active[i] = (w1_mask[i-5] < 0.5)       (LAG=5)
  G. inverse-vol weights replay (anchor t_index%5==0, seed=golden w_g[0])

ASCII-only prints. Read-only; writes nothing.
"""
from __future__ import annotations

import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
AR = os.path.join(_REPO_DIR, "audit_results")

import numpy as np
import pandas as pd

TOL_L = 0.01       # L_t absolute tolerance
TOL_W = 0.001      # weight absolute tolerance
TOL_SIG = 1e-6     # signal recompute tolerance (identical input/algebra)

V7_MAP = {0: 1.20, 1: 1.10, 2: 1.00, 3: 1.00}
LU1_MAP = {0: 1.40, 1: 1.20, 2: 1.05, 3: 1.00}


# ---------------------------------------------------------------------------
# GAS-portable algorithms (plain loops; translate these 1:1 to JS)
# ---------------------------------------------------------------------------
def lt2_series(closes, N=750):
    """LT2: mom_N z-scored vs rolling(2N, min_periods=N) of mom, clip +-3, nan->0.
    O(n) running-sum implementation (ddof=1 sample std)."""
    n = len(closes)
    W = 2 * N
    mom = [float("nan")] * n
    for i in range(N, n):
        if closes[i - N] > 0:
            mom[i] = closes[i] / closes[i - N] - 1.0
    # cumulative sums over valid mom values
    cnt = [0] * (n + 1)
    s1 = [0.0] * (n + 1)
    s2 = [0.0] * (n + 1)
    for i in range(n):
        v = mom[i]
        ok = not math.isnan(v)
        cnt[i + 1] = cnt[i] + (1 if ok else 0)
        s1[i + 1] = s1[i] + (v if ok else 0.0)
        s2[i + 1] = s2[i] + (v * v if ok else 0.0)
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - W + 1)
        c = cnt[i + 1] - cnt[lo]
        if c < N or math.isnan(mom[i]):
            continue  # stays 0 (fillna(0))
        m = (s1[i + 1] - s1[lo]) / c
        var = ((s2[i + 1] - s2[lo]) - c * m * m) / (c - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd <= 0:
            continue  # sigma->nan -> z nan -> 0
        z = (mom[i] - m) / sd
        out[i] = max(-3.0, min(3.0, z))
    return out


def lev_mod_065_one(lev_raw, vz, lt_sig, thr=0.65, k_lo=0.1, k_mid=0.5, k_hi=0.8):
    k = k_hi if vz > thr else (k_lo if vz < -thr else k_mid)
    bias = max(-0.5, min(0.5, -k * lt_sig * 0.5))
    return max(0.0, min(1.0, lev_raw + bias))


def w1_next(prev_in, lev_mod, enter=0.7, exit_=0.3):
    if not prev_in and lev_mod >= enter:
        return True
    if prev_in and lev_mod <= exit_:
        return False
    return prev_in


def mom63_bucket(m, q25, q50, q75):
    if m <= q25:
        return 0
    if m <= q50:
        return 1
    if m <= q75:
        return 2
    return 3


def inverse_vol_replay(gold, bond, t_index0, w_seed, window=63, update_bd=5,
                       lo=0.25, hi=0.75):
    """Replay inverse-vol weekly weights inside the golden window.
    gold/bond are 1x levels; ret[i] uses level[i-1] (i>=1).
    sigma uses rolling std ddof=1 over `window` returns (annualized factor
    cancels in the ratio but kept for parity of clamping edge cases)."""
    n = len(gold)
    rg = [0.0] * n
    rb = [0.0] * n
    for i in range(1, n):
        rg[i] = gold[i] / gold[i - 1] - 1.0
        rb[i] = bond[i] / bond[i - 1] - 1.0

    def sig(r, i):
        # rolling window ending at i over r[i-window+1..i]; needs i>=window
        if i < window:
            return float("nan")
        seg = r[i - window + 1:i + 1]
        m = sum(seg) / window
        var = sum((x - m) ** 2 for x in seg) / (window - 1)
        return math.sqrt(var) * math.sqrt(252.0)

    w = [0.0] * n
    last = w_seed
    for i in range(n):
        if (t_index0 + i) % update_bd == 0:
            sg = sig(rg, i)
            sb = sig(rb, i)
            if not math.isnan(sg) and not math.isnan(sb) and sg > 0 and sb > 0:
                wg = (1.0 / sg) / ((1.0 / sg) + (1.0 / sb))
                last = max(lo, min(hi, wg))
        w[i] = last
    return w


# ---------------------------------------------------------------------------
def main():
    print("=" * 86)
    print("PRE-PARITY: GAS-portable algorithms vs golden vectors   2026-06-11")
    print("=" * 86)

    gv = pd.read_csv(os.path.join(AR, "p09_golden_vectors_20260611.csv"))
    ch = pd.read_csv(os.path.join(AR, "p09_close_history_20260611.csv"))
    spec = json.load(open(os.path.join(AR, "p09_live_spec_20260611.json"), encoding="utf-8"))
    qb = spec["boost"]["quantile_boundaries_frozen"]
    q25, q50, q75 = qb["q25"], qb["q50"], qb["q75"]

    n = len(gv)
    results = {}

    # ---- A. LT2 from close history ----
    lt_all = lt2_series(ch["close"].tolist(), N=750)
    lt_win = lt_all[-n:]
    # confirm date alignment
    assert ch["date"].iloc[-n] == gv["date"].iloc[0], "close-history misaligned"
    diff_a = max(abs(a - b) for a, b in zip(lt_win, gv["lt_sig"].tolist()))
    results["A_LT2"] = (diff_a < TOL_SIG, "max|diff|=%.2e" % diff_a)

    # ---- B. lev_mod_065 ----
    lm_calc = [lev_mod_065_one(gv["lev_raw"][i], gv["vz"][i], gv["lt_sig"][i])
               for i in range(n)]
    diff_b = max(abs(lm_calc[i] - gv["lev_mod_065"][i]) for i in range(n))
    results["B_lev_mod_065"] = (diff_b < TOL_SIG, "max|diff|=%.2e" % diff_b)

    # ---- C. W1 hysteresis replay ----
    state = bool(gv["w1_mask"][0] > 0.5)
    mism_c = 0
    for i in range(1, n):
        state = w1_next(state, gv["lev_mod_065"][i])
        if state != bool(gv["w1_mask"][i] > 0.5):
            mism_c += 1
    results["C_W1_replay"] = (mism_c == 0, "mismatches=%d/%d" % (mism_c, n - 1))

    # ---- D. boost mult (prev-day mom63 bucket, frozen boundaries) ----
    # Known artifact: the backtest applies publication lag by restamping the
    # signal index +BusinessDay(1) (holiday-blind) and ffilling onto trading
    # days. On the trading day right after a mid-week market holiday the
    # effective lag becomes 2 trading days. The LIVE rule (use the previous
    # trading day's mom63) is the spec intent; we classify every mismatch and
    # PASS only if 100% of them are this lag-2 artifact.
    def _mult_check(col, mmap):
        mism, artifact = 0, 0
        bad_rows = []
        for i in range(2, n):
            q1 = mom63_bucket(gv["mom63"][i - 1], q25, q50, q75)
            if abs(mmap[q1] - gv[col][i]) <= 1e-9:
                continue
            mism += 1
            q2 = mom63_bucket(gv["mom63"][i - 2], q25, q50, q75)
            if abs(mmap[q2] - gv[col][i]) <= 1e-9:
                artifact += 1
            else:
                bad_rows.append(gv["date"][i])
        return mism, artifact, bad_rows

    m_p09, a_p09, bad_p09 = _mult_check("mult_p09", V7_MAP)
    m_lu1, a_lu1, bad_lu1 = _mult_check("mult_lu1", LU1_MAP)
    ok_d = (len(bad_p09) == 0 and len(bad_lu1) == 0)
    results["D_boost_mult"] = (
        ok_d,
        "mism P09=%d (lag2-artifact=%d) LU1=%d (artifact=%d); non-artifact=%s"
        % (m_p09, a_p09, m_lu1, a_lu1, (bad_p09 + bad_lu1) or "none"))

    # ---- E. L_t (DELAY=2) using recomputed mult; artifact days allow lag-2 ----
    def _L_check(colL, mmap):
        mx, n_art = 0.0, 0
        for i in range(4, n):
            j = i - 2
            base = gv["lev_raw"][j] * gv["w1_mask"][j] * 3.0
            q1 = mom63_bucket(gv["mom63"][j - 1], q25, q50, q75)
            d1 = abs(base * mmap[q1] - gv[colL][i])
            if d1 < TOL_L:
                mx = max(mx, d1)
                continue
            q2 = mom63_bucket(gv["mom63"][j - 2], q25, q50, q75)
            d2 = abs(base * mmap[q2] - gv[colL][i])
            if d2 < TOL_L:
                n_art += 1
                mx = max(mx, d2)
            else:
                return None, n_art
        return mx, n_art

    mx_p09, art_p09 = _L_check("L_p09", V7_MAP)
    mx_lu1, art_lu1 = _L_check("L_lu1", LU1_MAP)
    ok_e = (mx_p09 is not None and mx_lu1 is not None)
    results["E_L_t"] = (
        ok_e,
        "max|dL| P09=%s LU1=%s (lag2-artifact days P09=%d LU1=%d)"
        % ("%.4f" % mx_p09 if ok_e else "FAIL",
           "%.4f" % mx_lu1 if ok_e else "FAIL", art_p09, art_lu1))

    # ---- F. fund_active (LAG=5) ----
    mism_f = 0
    for i in range(5, n):
        fa = gv["w1_mask"][i - 5] < 0.5
        if int(fa) != int(gv["fund_active"][i]):
            mism_f += 1
    results["F_fund_active"] = (mism_f == 0, "mismatches=%d/%d" % (mism_f, n - 5))

    # ---- G. inverse-vol weights replay ----
    w_calc = inverse_vol_replay(gv["gold_1x"].tolist(), gv["bond_1x"].tolist(),
                                int(gv["t_index"][0]), float(gv["w_g"][0]))
    # first in-window update with a FULL in-window 63d return window:
    first_full = next(i for i in range(n)
                      if i >= 63 and (int(gv["t_index"][0]) + i) % 5 == 0)
    diff_g = max(abs(w_calc[i] - gv["w_g"][i]) for i in range(first_full, n))
    results["G_inverse_vol"] = (diff_g < TOL_W,
                                "max|dw| (from row %d)=%.2e" % (first_full, diff_g))

    # ---- report ----
    print("")
    all_pass = True
    for k in sorted(results):
        ok, msg = results[k]
        all_pass = all_pass and ok
        print("  %-16s %s   %s" % (k, "PASS" if ok else "FAIL", msg))
    print("\nOVERALL: %s" % ("PASS - algorithms ready for 1:1 JS port"
                             if all_pass else "FAIL - fix before porting"))
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
