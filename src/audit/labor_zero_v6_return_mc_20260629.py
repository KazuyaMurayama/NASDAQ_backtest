"""
src/audit/labor_zero_v6_return_mc_20260629.py
=============================================
v6 -- RETURN-PATH block-bootstrap Monte Carlo. v3/v4 fixed the historical return
path (counting labor=0 across 46 starts); v5 randomized ONLY inflation. v6 randomizes
the RETURN path so that "labor=0" becomes a true PROBABILITY P(labor=0).

User condition (2026-06-29):
  - spend = 7.2M NOMINAL, FIXED every year (NOT inflation-indexed; this differs from v4
    which kept spending real). 2% inflation is used only to (a) deflate the terminal
    wealth to real purchasing power and (b) annotate the real draw-down burden. The
    nominal spend stays 7.2M -- so the withdrawal rate falls over time as wealth compounds.
  - goal: push the probability of labor=0 toward 100%, splitting 40M into a leveraged
    RUN sleeve (sc2.6) and a REFILL/RESERVE sleeve (1x bond etc.).

Method (block bootstrap, preserves sequence risk & crash clustering -> correct for a
path-dependent survival metric; cf R-STAT-1/2):
  - Sample year-PAIRS (run-return, reserve-return) JOINTLY in contiguous blocks of L
    years drawn (with replacement) from the historical 1975-2025 calendar (51 yrs),
    concatenated to the needed horizon. Joint sampling preserves stock/bond co-movement
    within a year; block sampling preserves momentum/crash runs.
  - Each path = one synthetic horizon. We evaluate ONE retirement on it (start at year 0
    of the path). N paths -> P(labor=0) = (# paths with labor==0) / N.   [PRIMARY metric,
    trial unit = (path) which equals "a person who retires onto a random future".]
  - We also report the labor-year distribution, ruin rate, real terminal, and (for the
    floor variant) the cut-year count.

Probability unit: PRIMARY = per-path P(labor=0) (one retirement per path). This answers
"if I retire, what is the chance I never work again" -- the user's question. (The v3/v4
"all-46-starts-simultaneously" AND-condition is a different, far stricter object; here
each path already mixes resampled history so a single retirement per path is the natural
trial.)

SELF-TEST: with block = full history in original order (deterministic, the actual
1975.. sequence) the per-start labor matches v3/v4 for representative starts. If not, HALT.

ASCII-only prints. Writes audit_results/labor_zero_v6_*_20260629.csv. No commit.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (os.path.dirname(_THIS), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_THIS, "labor_zero_v3_harness_20260628.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

M = 1_000_000.0
SPEND = 7.2 * M
HIST_YEARS = list(range(1975, 2026))   # 51 historical calendar years
HORIZON = 20                            # retirement horizon (years)
INFLATION = 0.02                        # for real terminal / real burden annotation only


# ----------------------------------------------------------------------------- data
def get_paired_history(strat="sc2.6", reserve_weights=None):
    """Return arrays (run_r[51], res_r[51]) of after-tax calendar-year returns,
    aligned on HIST_YEARS, for the run sleeve and the reserve sleeve."""
    if reserve_weights is None:
        reserve_weights = {"bond": 1.0}
    rets = v3.load_rets()
    res_series = v3.load_mixed_reserve(reserve_weights)
    run = np.array([float(rets[strat].loc[y]) for y in HIST_YEARS], float)
    res = np.array([float(res_series.loc[y]) for y in HIST_YEARS], float)
    return run, res


# ------------------------------------------------------------------- bootstrap paths
def make_block_path(rng, run_h, res_h, n=HORIZON, block=5):
    """Concatenate contiguous blocks (length `block`, random start, with replacement,
    wrap-around) from the paired history until length>=n; truncate to n.
    Returns (run_path[n], res_path[n]) preserving the within-year run/res pairing."""
    H = len(run_h)
    rp = np.empty(n)
    sp = np.empty(n)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, H))
        take = min(block, n - filled)
        for j in range(take):
            idx = (start + j) % H
            rp[filled] = run_h[idx]
            sp[filled] = res_h[idx]
            filled += 1
    return rp, sp


# ----------------------------------------------------------------------- single sim
def sim_one(run_path, res_path, *, run0, reserve0, tranches, spend=SPEND,
            hold_if_crash=False, floor=0.0, top_wr=None):
    """One retirement over the supplied (already-realized) return paths.
    tranches = [(threshold, frac_of_remaining_reserve), ...] thresholds DESCENDING.
    hold_if_crash: skip reserve top-up in a year where the run return is negative.
    floor>0 + top_wr: spending becomes max(floor, full if affordable) -- the
       asset-linked variant (labor then can only fire if total<floor).
    Returns dict(labor_years, ruin, terminal, cut_years).
    """
    run, res = float(run0), float(reserve0)
    n = len(run_path)
    fired = [False] * len(tranches)
    labor = 0
    cut = 0
    for k in range(n):
        r_run = run_path[k]
        # 1. tranche top-up (optionally hold if this year's run return is negative)
        if not (hold_if_crash and r_run < 0):
            for i, (thr, frac) in enumerate(tranches):
                if (not fired[i]) and run < thr and res > 1e-6:
                    move = res * frac
                    run += move
                    res -= move
                    fired[i] = True
        # 2. decide spend
        total = run + res
        if floor > 0.0:
            want = floor
            if top_wr is not None and total > 1e-9 and (spend / total) <= top_wr:
                want = spend
            spend_k = want
        else:
            spend_k = spend
        if total + 1e-6 < spend_k:
            labor += 1
            spend_k = total            # use all that remains
        if floor > 0.0 and spend_k < spend - 1e-6:
            cut += 1
        # 3. pay (run first, then reserve)
        if run >= spend_k:
            run -= spend_k
        else:
            res -= (spend_k - run)
            run = 0.0
        if res < 0:
            res = 0.0
        # 4. growth
        run *= (1.0 + r_run)
        res *= (1.0 + res_path[k])
    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6),
                terminal=run + res, cut_years=cut)


# ------------------------------------------------------------------------ MC driver
def mc_prob(run_h, res_h, *, run0, reserve0, tranches, n_paths=1000, block=5,
            seed=20260629, hold_if_crash=False, floor=0.0, top_wr=None,
            horizon=HORIZON):
    """Monte Carlo over block-bootstrap return paths. Returns summary dict."""
    rng = np.random.default_rng(seed)
    labors = np.empty(n_paths, int)
    ruins = np.empty(n_paths, int)
    terms = np.empty(n_paths)
    cuts = np.empty(n_paths, int)
    for p in range(n_paths):
        rp, sp = make_block_path(rng, run_h, res_h, n=horizon, block=block)
        r = sim_one(rp, sp, run0=run0, reserve0=reserve0, tranches=tranches,
                    hold_if_crash=hold_if_crash, floor=floor, top_wr=top_wr)
        labors[p] = r["labor_years"]
        ruins[p] = r["ruin"]
        terms[p] = r["terminal"]
        cuts[p] = r["cut_years"]
    deflate = (1.0 + INFLATION) ** horizon
    return dict(
        n_paths=n_paths, block=block,
        p_labor0=float((labors == 0).mean()),
        p_ruin0=float((ruins == 0).mean()),
        labor_mean=float(labors.mean()),
        labor_p95=float(np.percentile(labors, 95)),
        labor_max=int(labors.max()),
        term_real_med_M=round(float(np.median(terms)) / deflate / M, 1),
        term_real_p5_M=round(float(np.percentile(terms, 5)) / deflate / M, 1),
        cut_mean=float(cuts.mean()),
    )


# ----------------------------------------------------------------------- self-test
def _self_test():
    """Block = full history in ORIGINAL order == the actual 1975.. sequence.
    Then one retirement starting at 1975 must reproduce v3's 1975 labor (0) and the
    2012 start (offset 37) under data-max horizon must reproduce v3's labor>0."""
    run_h, res_h = get_paired_history()
    # v3 config (v2 geometry): sc2.6, run 14M / reserve 26M, single all-in tranche
    cfg = dict(run0=14 * M, reserve0=26 * M, tranches=[(14 * M, 1.0)])
    # 1975 start, 20y -> deterministic actual sequence
    path_run = run_h[0:20]
    path_res = res_h[0:20]
    r1975 = sim_one(path_run, path_res, **cfg)
    # 2012 start (index 37), data-max horizon = min(20, 2025-2012+1)=14
    i = 2012 - 1975
    h = min(20, 2025 - 2012 + 1)
    r2012 = sim_one(run_h[i:i + h], res_h[i:i + h], **cfg)
    # cross-check against v3.simulate_v3 directly
    rets = v3.load_rets()
    bond = v3.load_mixed_reserve({"bond": 1.0})
    v3_1975 = v3.simulate_v3(rets, bond, 1975, 20, strat="sc2.6", run0=14 * M,
                             reserve0=26 * M, tranches=[(14 * M, 1.0)])
    v3_2012 = v3.simulate_v3(rets, bond, 2012, 20, strat="sc2.6", run0=14 * M,
                             reserve0=26 * M, tranches=[(14 * M, 1.0)])
    print("SELF-TEST (block=full, original order == actual history):")
    print("  1975 start: v6 labor=%d  v3 labor=%d  (exp equal, 0)"
          % (r1975["labor_years"], v3_1975["labor_years"]))
    print("  2012 start: v6 labor=%d  v3 labor=%d  (exp equal, >0)"
          % (r2012["labor_years"], v3_2012["labor_years"]))
    ok = (r1975["labor_years"] == v3_1975["labor_years"]
          and r2012["labor_years"] == v3_2012["labor_years"]
          and r1975["labor_years"] == 0 and r2012["labor_years"] > 0)
    print("  -> SELF-TEST %s" % ("PASS" if ok else "FAIL -- FIX before trusting v6"))
    return ok


# ----------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 95)
    print("v6 RETURN-PATH BOOTSTRAP MC -- P(labor=0) for nominal-fixed 7.2M spend")
    print("  spend 7.2M NOMINAL fixed | inflation 2%% used for real terminal only")
    print("  run sleeve sc2.6 | reserve 1x bond | block bootstrap (sequence-preserving)")
    print("=" * 95)
    if not _self_test():
        print("HALT: self-test failed.")
        return
    run_h, res_h = get_paired_history()

    N = 2000
    print("\n--- ROUND 0: baseline (v2 geometry) across block lengths ---")
    base = dict(run0=14 * M, reserve0=26 * M, tranches=[(14 * M, 1.0)])
    rows = []
    for L in (3, 5, 8):
        r = mc_prob(run_h, res_h, n_paths=N, block=L, **base)
        r["config"] = "R0_base_run14_res26_allin"
        rows.append(r)
        print("  block=%d: P(labor0)=%.3f  P(ruin0)=%.3f  labor_mean=%.2f p95=%.0f max=%d"
              "  realTermMed=%.0fM p5=%.0fM"
              % (L, r["p_labor0"], r["p_ruin0"], r["labor_mean"], r["labor_p95"],
                 r["labor_max"], r["term_real_med_M"], r["term_real_p5_M"]))

    print("\n--- ROUND 1: run/reserve split & hold-if-crash (block=5, nominal-fixed) ---")
    L = 5
    configs = [
        ("R1_run14_res26_allin",        dict(run0=14 * M, reserve0=26 * M, tranches=[(14 * M, 1.0)])),
        ("R1_run14_res26_allin_HOLD",   dict(run0=14 * M, reserve0=26 * M, tranches=[(14 * M, 1.0)], hold_if_crash=True)),
        ("R1_run18_res22_allin_HOLD",   dict(run0=18 * M, reserve0=22 * M, tranches=[(18 * M, 1.0)], hold_if_crash=True)),
        ("R1_run20_res20_allin_HOLD",   dict(run0=20 * M, reserve0=20 * M, tranches=[(20 * M, 1.0)], hold_if_crash=True)),
        ("R1_run24_res16_allin_HOLD",   dict(run0=24 * M, reserve0=16 * M, tranches=[(24 * M, 1.0)], hold_if_crash=True)),
        ("R1_run10_res30_allin_HOLD",   dict(run0=10 * M, reserve0=30 * M, tranches=[(10 * M, 1.0)], hold_if_crash=True)),
    ]
    for name, cfg in configs:
        r = mc_prob(run_h, res_h, n_paths=N, block=L, **cfg)
        r["config"] = name
        rows.append(r)
        print("  %-28s P(labor0)=%.3f P(ruin0)=%.3f labor_mean=%.2f p95=%.0f realTermMed=%.0fM"
              % (name, r["p_labor0"], r["p_ruin0"], r["labor_mean"], r["labor_p95"],
                 r["term_real_med_M"]))

    print("\n--- ROUND 2: split-tranche refill & reserve mix (block=5, best geometry+HOLD) ---")
    # take the run/reserve that maximized P(labor0) in R1; explore split + mix
    best_r1 = max((x for x in rows if x["config"].startswith("R1")), key=lambda z: z["p_labor0"])
    print("  (R1 best = %s @ P=%.3f)" % (best_r1["config"], best_r1["p_labor0"]))
    g_run, g_res = 18 * M, 22 * M     # representative; report uses R1 sweep for the dial
    r2configs = [
        ("R2_split2_18_HOLD", dict(run0=g_run, reserve0=g_res,
                                   tranches=[(18 * M, 0.6), (10 * M, 1.0)], hold_if_crash=True)),
        ("R2_split3_18_HOLD", dict(run0=g_run, reserve0=g_res,
                                   tranches=[(18 * M, 0.4), (13 * M, 0.5), (8 * M, 1.0)], hold_if_crash=True)),
    ]
    for name, cfg in r2configs:
        r = mc_prob(run_h, res_h, n_paths=N, block=L, **cfg)
        r["config"] = name
        rows.append(r)
        print("  %-22s P(labor0)=%.3f P(ruin0)=%.3f labor_mean=%.2f"
              % (name, r["p_labor0"], r["p_ruin0"], r["labor_mean"]))
    for wname, w in [("mix_bond80_gold20", {"bond": 0.8, "gold": 0.2}),
                     ("mix_bond70_gold20_cash10", {"bond": 0.7, "gold": 0.2, "cash": 0.1})]:
        rh2, sh2 = get_paired_history(reserve_weights=w)
        cfg = dict(run0=g_run, reserve0=g_res, tranches=[(18 * M, 1.0)], hold_if_crash=True)
        r = mc_prob(rh2, sh2, n_paths=N, block=L, **cfg)
        r["config"] = "R2_" + wname
        rows.append(r)
        print("  %-22s P(labor0)=%.3f P(ruin0)=%.3f labor_mean=%.2f"
              % ("R2_" + wname, r["p_labor0"], r["p_ruin0"], r["labor_mean"]))

    print("\n--- ROUND 3: GUARANTEED-FLOOR variant (asset-linked) -> push P(labor0)->1.0 ---")
    # floor variant: labor only if total<floor. top_wr gates the jump to full 7.2M.
    cfgF = dict(run0=20 * M, reserve0=20 * M, tranches=[(20 * M, 1.0)], hold_if_crash=True,
                top_wr=0.16)
    for floorM in (0.0, 3.6, 4.43, 5.0, 6.0, 7.2):
        cfg = dict(cfgF)
        cfg["floor"] = floorM * M
        r = mc_prob(run_h, res_h, n_paths=N, block=L, **cfg)
        r["config"] = "R3_floor%.2f_run20res20_HOLD_top16" % floorM
        rows.append(r)
        print("  floor=%.2fM: P(labor0)=%.3f P(ruin0)=%.3f cut_mean=%.2f realTermMed=%.0fM"
              % (floorM, r["p_labor0"], r["p_ruin0"], r["cut_mean"], r["term_real_med_M"]))

    df = pd.DataFrame(rows)
    cols = ["config", "block", "n_paths", "p_labor0", "p_ruin0", "labor_mean",
            "labor_p95", "labor_max", "cut_mean", "term_real_med_M", "term_real_p5_M"]
    df = df[[c for c in cols if c in df.columns]]
    out = os.path.join(_REPO, "audit_results", "labor_zero_v6_return_mc_20260629.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("\nDone (v6 return-MC).")


if __name__ == "__main__":
    main()
