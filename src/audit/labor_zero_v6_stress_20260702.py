"""
src/audit/labor_zero_v6_stress_20260702.py
===========================================
Task 6 of the v6 critical-verification campaign (plan: docs/superpowers/plans/
2026-07-02-labor-zero-v6-critical-verification.md): REAL-WORLD STRESS (exposing
historical-path dependence).

Motivation (doubt D3): sc2.6 was itself selected on the same 1975-2025 history
that the block-bootstrap resamples. P(labor=0) is therefore a conditional
probability -- "IF future returns resemble the 1975-2025 US sample, what is the
chance of never working again." This script asks how the headline numbers decay
if the future is systematically worse than that sample, and what the worst
CONTIGUOUS 10-year window in the historical record implies if forced to the
front of the retirement path (a sequence-risk floor, not a random draw).

Reuses (no reimplementation):
  - sim_conv / gen_paths / CONFIGS  from labor_zero_v6_lookahead_20260702.py
    (loaded via importlib, which in turn loads v6.get_paired_history / sim_one
    for its own self-test -- this script re-runs that self-test first and HALTs
    on any mismatch).
  - wilson_ci                       from labor_zero_v6_stats_helpers_20260702.py

Conventions measured: BOTH "lookahead" (canonical V6-A/V6-C convention) and
"lagged" (implementable, non-look-ahead convention; see Task 3 finding) for
every scenario, per the plan's requirement that stress numbers be reported
under both conventions.

Configs (plan Task 6, using CONFIGS names from the lookahead module):
  FA1 = F_20_20_thr20   (V6-A:  run20/res20, thr20M, floor3.6M, top_wr0.16)
  FA2 = F_20_20_thr26   (V6-A': run20/res20, thr26M, floor3.6M, top_wr0.16)
  NB0 = N_20_20_thr20_noHOLD  (nominal-fixed, thr20M, no hold-if-crash: the best
        implementable nominal-fixed config under the "lagged" convention per the
        Task-3 finding that hold-if-crash is itself look-ahead-shaped)
  NC1 = N_20_20_thr20   (nominal-fixed, thr20M, hold=True: canonical V6-C)

Scenarios:
  1. Uniform return shift: run -2pp, run -5pp, and run -2pp & bond -1pp jointly.
     Applied to the 51-year historical arrays BEFORE bootstrap resampling (so
     block structure / crash clustering is preserved, only the level shifts).
  2. Floor-decay curve: for each shift scenario, sweep floor in
     {1.2,1.6,2.0,2.4,2.8,3.2,3.6}M (run20/res20, thr20M, top_wr0.16) and find
     the largest floor for which P(labor=0) >= 0.999, under both conventions.
  3. Worst-10-year block forced to the front: find the contiguous 10-year window
     (no wrap) of the historical run series with the lowest cumulative return,
     fix it as years 0-9 of every path, block-bootstrap only years 10-19.
     Report P(labor=0) for FA1/FA2/NB0/NC1.
  4. C5 stress: under scenario (a) run -2pp, compare nominal-fixed at 49M
     (run24.5/res24.5, thr=run0) and 60M (run30/res30, thr=run0) against FA1
     (floor-fixed 40M), under both hold=True/lookahead and noHOLD/lagged.

ASCII-only prints. CSV -> audit_results/labor_zero_v6_stress_20260702.csv.
No commit (per task instructions). Desktop-safe (repo-relative paths only).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec_v6 = importlib.util.spec_from_file_location(
    "v6", os.path.join(_THIS, "labor_zero_v6_return_mc_20260629.py"))
v6 = importlib.util.module_from_spec(_spec_v6)
_spec_v6.loader.exec_module(v6)

_spec_la = importlib.util.spec_from_file_location(
    "lzla", os.path.join(_THIS, "labor_zero_v6_lookahead_20260702.py"))
lzla = importlib.util.module_from_spec(_spec_la)
_spec_la.loader.exec_module(lzla)   # runs its own module-level defs only (main() is guarded)

_spec_st = importlib.util.spec_from_file_location(
    "lzstats", os.path.join(_THIS, "labor_zero_v6_stats_helpers_20260702.py"))
lzstats = importlib.util.module_from_spec(_spec_st)
_spec_st.loader.exec_module(lzstats)

M = 1_000_000.0
SPEND = 7.2 * M
BLOCK = 5
NPATHS = 2000
SEEDS = [20260629 + i for i in range(5)]
INFLATION = 0.02
HORIZON = 20

CONV_PAIR = ("lookahead", "lagged")   # report both per plan Step 6-1..6-3

# -------------------------------------------------------------- stress configs
STRESS_CONFIGS = {
    "FA1_F202020thr20": dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                              floor=3.6 * M, top_wr=0.16),
    "FA2_F202020thr26": dict(run0=20 * M, reserve0=20 * M, thr=26 * M,
                              floor=3.6 * M, top_wr=0.16),
    "NB0_noHOLD_thr20": dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                              hold_if_crash=False),
    "NC1_hold_thr20":   dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                              hold_if_crash=True),
}
# convention actually used to headline each config (its "natural" implementable
# regime per Task 3 finding), but ALL are evaluated under BOTH conventions below.
NATURAL_CONV = {
    "FA1_F202020thr20": "lookahead",
    "FA2_F202020thr26": "lookahead",
    "NB0_noHOLD_thr20": "lagged",
    "NC1_hold_thr20":   "lookahead",
}

C5_CONFIGS = {
    "C5_40M_FA1":  dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                         floor=3.6 * M, top_wr=0.16),
    "C5_49M_hold": dict(run0=24.5 * M, reserve0=24.5 * M, thr=24.5 * M,
                         hold_if_crash=True),
    "C5_49M_noHOLD": dict(run0=24.5 * M, reserve0=24.5 * M, thr=24.5 * M,
                           hold_if_crash=False),
    "C5_60M_hold": dict(run0=30 * M, reserve0=30 * M, thr=30 * M,
                         hold_if_crash=True),
    "C5_60M_noHOLD": dict(run0=30 * M, reserve0=30 * M, thr=30 * M,
                           hold_if_crash=False),
}
C5_CONV = {
    "C5_40M_FA1":    "lookahead",
    "C5_49M_hold":   "lookahead",
    "C5_49M_noHOLD": "lagged",
    "C5_60M_hold":   "lookahead",
    "C5_60M_noHOLD": "lagged",
}

FLOOR_GRID = [1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6]   # x M


# ------------------------------------------------------------- shift scenarios
def shifted_history(run_h, res_h, run_shift, res_shift):
    """Return NEW (run_h, res_h) arrays with an additive shift applied to
    ARITHMETIC-ish annual returns (returns are stored as e.g. 1.5495 for +154.95%
    for the leveraged run sleeve; the shift is applied directly in return units,
    i.e. -0.02 = -2 percentage points on the annual return itself, matching the
    plan's "-2pp / -5pp" wording)."""
    return run_h + run_shift, res_h + res_shift


SHIFTS = {
    "S0_none":        (0.0, 0.0),
    "Sa_run-2pp":      (-0.02, 0.0),
    "Sb_run-5pp":      (-0.05, 0.0),
    "Sc_run-2_bond-1": (-0.02, -0.01),
}


# ---------------------------------------------------------------- path gen/run
def gen_paths_from(run_h, res_h, seed, n_paths=NPATHS, block=BLOCK, horizon=HORIZON):
    rng = np.random.default_rng(seed)
    return [v6.make_block_path(rng, run_h, res_h, n=horizon, block=block)
            for _ in range(n_paths)]


def run_cfg_on_paths(paths, convention, cfg):
    labors = np.empty(len(paths), int)
    for i, (rp, sp) in enumerate(paths):
        r = lzla.sim_conv(rp, sp, convention=convention, spend=SPEND, **cfg)
        labors[i] = r["labor_years"]
    return labors


# ------------------------------------------------------------ worst-10y window
def worst_10y_window(run_h, hist_years):
    """No-wrap contiguous 10-year window of run_h with the lowest cumulative
    (compounded) return. Returns (start_idx, end_idx_excl, start_year, end_year,
    cum_return)."""
    H = len(run_h)
    best = None
    for start in range(0, H - 10 + 1):
        window = run_h[start:start + 10]
        cum = np.prod(1.0 + window) - 1.0
        if best is None or cum < best[0]:
            best = (cum, start)
    cum, start = best
    return start, start + 10, hist_years[start], hist_years[start + 9], cum


def gen_paths_forced_front(run_h, res_h, front_run, front_res, seed,
                            n_paths=NPATHS, block=BLOCK, horizon=HORIZON):
    """First 10 years = fixed front window (deterministic, identical every path).
    Remaining (horizon-10) years = block bootstrap over the FULL history."""
    rng = np.random.default_rng(seed)
    n_tail = horizon - 10
    paths = []
    for _ in range(n_paths):
        tail_rp, tail_sp = v6.make_block_path(rng, run_h, res_h, n=n_tail, block=block)
        rp = np.concatenate([front_run, tail_rp])
        sp = np.concatenate([front_res, tail_sp])
        paths.append((rp, sp))
    return paths


# -------------------------------------------------------------------- self-test
def _self_test(run_h, res_h):
    """(i) shifted_history with zero shift == original history (identity check).
    (ii) gen_paths_from with S0_none reproduces lzla.gen_paths bit-for-bit for a
    fixed seed (confirms this script's path generator has not drifted from the
    Task-3 anchor). (iii) run_cfg_on_paths(FA1, lookahead) on those paths
    reproduces the P(labor0)=0.9995 anchor (seed=20260629, N=2000, block=5)."""
    ok = True
    rh0, sh0 = shifted_history(run_h, res_h, 0.0, 0.0)
    if not (np.array_equal(rh0, run_h) and np.array_equal(sh0, res_h)):
        print("SELF-TEST FAIL: zero-shift identity broken")
        ok = False

    seed0 = SEEDS[0]
    paths_a = gen_paths_from(run_h, res_h, seed0)
    paths_b = lzla.gen_paths(run_h, res_h, seed0)
    same_paths = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                      for a, b in zip(paths_a, paths_b))
    if not same_paths:
        print("SELF-TEST FAIL: gen_paths_from diverges from lzla.gen_paths")
        ok = False

    labors = run_cfg_on_paths(paths_a, "lookahead", STRESS_CONFIGS["FA1_F202020thr20"])
    p = float((labors == 0).mean())
    if abs(p - 0.9995) > 1e-9:
        print("SELF-TEST FAIL: FA1 lookahead anchor mismatch, got P=%.4f want 0.9995" % p)
        ok = False

    if ok:
        print("SELF-TEST PASS: path generator matches Task-3 anchor bit-for-bit; "
              "FA1 lookahead P(labor0)=%.4f == 0.9995 anchor" % p)
    return ok


# ------------------------------------------------------------------------ main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("Task 6: REAL-WORLD STRESS (historical-path dependence exposure)")
    print("  block=%d N=%d seeds=%s conventions=%s" % (BLOCK, NPATHS, SEEDS, CONV_PAIR))
    print("=" * 100)

    run_h, res_h = v6.get_paired_history()

    if not _self_test(run_h, res_h):
        print("HALT: self-test failed.")
        return

    deflate = (1.0 + INFLATION) ** HORIZON
    all_rows = []

    # ---------------------------------------------------------- [1] uniform shift
    print("\n[1] Uniform return shift: P(labor0) by scenario x config x convention")
    print("    (mean over %d seeds [min-max])" % len(SEEDS))
    shift_store = {}   # shift_store[scenario][conv][config][seed] -> labors array
    for sname, (rshift, sshift) in SHIFTS.items():
        rh, sh = shifted_history(run_h, res_h, rshift, sshift)
        shift_store[sname] = {c: {k: {} for k in STRESS_CONFIGS} for c in CONV_PAIR}
        for seed in SEEDS:
            paths = gen_paths_from(rh, sh, seed)
            for conv in CONV_PAIR:
                for cname, cfg in STRESS_CONFIGS.items():
                    labors = run_cfg_on_paths(paths, conv, cfg)
                    shift_store[sname][conv][cname][seed] = labors
                    p = float((labors == 0).mean())
                    all_rows.append(dict(section="1_shift", scenario=sname,
                                          convention=conv, config=cname, seed=seed,
                                          p_labor0=p))
    for sname in SHIFTS:
        print("  scenario %s" % sname)
        for cname in STRESS_CONFIGS:
            line = "    %-20s" % cname
            for conv in CONV_PAIR:
                ps = [float((shift_store[sname][conv][cname][s] == 0).mean()) for s in SEEDS]
                line += "  %-8s %.4f [%.4f-%.4f]" % (conv, np.mean(ps), min(ps), max(ps))
            print(line)

    # ------------------------------------------------------ [2] floor-decay curve
    print("\n[2] Floor-decay curve: max floor (of grid %s M) with P(labor0)>=0.999" % FLOOR_GRID)
    print("    run20/res20, thr20M, top_wr0.16 (V6-A geometry)")
    floor_rows = []
    for sname, (rshift, sshift) in SHIFTS.items():
        rh, sh = shifted_history(run_h, res_h, rshift, sshift)
        for conv in CONV_PAIR:
            max_floor = {seed: None for seed in SEEDS}
            floor_p = {}
            for floor_m in FLOOR_GRID:
                cfg = dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
                           floor=floor_m * M, top_wr=0.16)
                ps = []
                for seed in SEEDS:
                    paths = gen_paths_from(rh, sh, seed)
                    labors = run_cfg_on_paths(paths, conv, cfg)
                    p = float((labors == 0).mean())
                    ps.append(p)
                    floor_rows.append(dict(section="2_floor_decay", scenario=sname,
                                            convention=conv, floor_M=floor_m, seed=seed,
                                            p_labor0=p))
                    if p >= 0.999 and (max_floor[seed] is None or floor_m > max_floor[seed]):
                        max_floor[seed] = floor_m
                print("    %-16s %-9s floor=%.1fM  P=%.4f [%.4f-%.4f]"
                      % (sname, conv, floor_m, np.mean(ps), min(ps), max(ps)))
            mf_vals = [v for v in max_floor.values() if v is not None]
            n_ok = len(mf_vals)
            if n_ok == len(SEEDS):
                print("    -> max floor with P>=0.999 (all %d seeds): %.1fM"
                      % (len(SEEDS), min(mf_vals)))
            elif n_ok == 0:
                print("    -> NO floor in grid reaches P>=0.999 in any seed "
                      "(largest grid value=%.1fM insufficient)" % FLOOR_GRID[-1])
            else:
                print("    -> max floor with P>=0.999 held in %d/%d seeds "
                      "(inconsistent across seeds): %s"
                      % (n_ok, len(SEEDS), sorted(set(mf_vals))))
    all_rows.extend(floor_rows)

    # -------------------------------------------------- [3] worst-10y forced front
    print("\n[3] Worst contiguous 10-year window (run sleeve, cumulative return, no wrap)")
    w_start, w_end, w_y0, w_y1, w_cum = worst_10y_window(run_h, v6.HIST_YEARS)
    print("    window = %d-%d (idx %d:%d)   cumulative run return = %+.4f (x%.3f)"
          % (w_y0, w_y1, w_start, w_end, w_cum, 1.0 + w_cum))
    front_run = run_h[w_start:w_end]
    front_res = res_h[w_start:w_end]
    print("    front_run (year-by-year): %s" % np.round(front_run, 4).tolist())
    print("    front_res (year-by-year): %s" % np.round(front_res, 4).tolist())

    w3_rows = []
    for conv in CONV_PAIR:
        print("  convention=%s" % conv)
        for cname, cfg in STRESS_CONFIGS.items():
            ps, terms = [], []
            for seed in SEEDS:
                paths = gen_paths_forced_front(run_h, res_h, front_run, front_res, seed)
                labors = np.empty(len(paths), int)
                term = np.empty(len(paths))
                for i, (rp, sp) in enumerate(paths):
                    r = lzla.sim_conv(rp, sp, convention=conv, spend=SPEND, **cfg)
                    labors[i] = r["labor_years"]
                    term[i] = r["terminal"]
                p = float((labors == 0).mean())
                ps.append(p)
                terms.append(float(np.median(term)) / deflate / M)
                w3_rows.append(dict(section="3_worst10_front", convention=conv,
                                     config=cname, seed=seed, p_labor0=p,
                                     term_real_med_M=round(terms[-1], 1)))
            print("    %-20s P=%.4f [%.4f-%.4f]   term_real_med(20y)~%.1fM [%.1f-%.1f]"
                  % (cname, np.mean(ps), min(ps), max(ps),
                     np.mean(terms), min(terms), max(terms)))
    all_rows.extend(w3_rows)

    # ------------------------------------------------------------------- [4] C5 stress
    print("\n[4] C5 stress: scenario Sa_run-2pp -- asset-size vs floor-fixed efficiency ranking")
    rh, sh = shifted_history(run_h, res_h, *SHIFTS["Sa_run-2pp"])
    c5_rows = []
    c5_summary = {}
    for cname, cfg in C5_CONFIGS.items():
        conv = C5_CONV[cname]
        ps, terms = [], []
        for seed in SEEDS:
            paths = gen_paths_from(rh, sh, seed)
            labors = np.empty(len(paths), int)
            term = np.empty(len(paths))
            for i, (rp, sp) in enumerate(paths):
                r = lzla.sim_conv(rp, sp, convention=conv, spend=SPEND, **cfg)
                labors[i] = r["labor_years"]
                term[i] = r["terminal"]
            p = float((labors == 0).mean())
            ps.append(p)
            terms.append(float(np.median(term)) / deflate / M)
            c5_rows.append(dict(section="4_c5_stress", convention=conv, config=cname,
                                 seed=seed, p_labor0=p, term_real_med_M=round(terms[-1], 1)))
        c5_summary[cname] = (np.mean(ps), min(ps), max(ps), np.mean(terms))
        print("    %-16s conv=%-9s P=%.4f [%.4f-%.4f]  term_real_med(20y)~%.1fM"
              % (cname, conv, np.mean(ps), min(ps), max(ps), np.mean(terms)))
    all_rows.extend(c5_rows)

    print("\n  ranking check (Sa_run-2pp, P mean): "
          + ", ".join("%s=%.4f" % (k, v[0]) for k, v in
                       sorted(c5_summary.items(), key=lambda kv: -kv[1][0])))

    # paired test: does FA1(40M floor) beat 49M/60M nominal-fixed under stress,
    # same seeds/paths within each convention group
    print("\n  paired tests (same seed/paths within convention, Sa_run-2pp):")
    fa1_paths_by_seed = {seed: gen_paths_from(rh, sh, seed) for seed in SEEDS}
    fa1_labors = {seed: run_cfg_on_paths(fa1_paths_by_seed[seed], "lookahead",
                                          C5_CONFIGS["C5_40M_FA1"]) for seed in SEEDS}
    for other in ("C5_49M_hold", "C5_49M_noHOLD", "C5_60M_hold", "C5_60M_noHOLD"):
        conv = C5_CONV[other]
        other_labors = np.concatenate([
            run_cfg_on_paths(fa1_paths_by_seed[seed], conv, C5_CONFIGS[other])
            for seed in SEEDS])
        fa1_pooled = np.concatenate([fa1_labors[seed] for seed in SEEDS])
        r = lzstats.paired_diff(fa1_pooled, other_labors)
        print("    FA1(lookahead,40M) vs %-16s(%s)  dP=%+.4f  FA1_only=%4d other_only=%4d p=%.2e"
              % (other, conv, r["p_a"] - r["p_b"], r["a_only"], r["b_only"], r["mcnemar_p"]))

    # ------------------------------------------------------------------------- write CSV
    df = pd.DataFrame(all_rows)
    out_dir = os.path.join(_REPO, "audit_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "labor_zero_v6_stress_20260702.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s (%d rows)" % (out, len(df)))
    print("\nDone (Task 6 real-world stress).")


if __name__ == "__main__":
    main()
