"""
labor_zero_v6_indep_20260702.py

Clean-room independent reimplementation of the "labor-zero" retirement
simulation, built ONLY from the Japanese spec handed to the implementer.
No existing labor_zero_v6*/v3*/LABOR_ZERO_*.md files were read while
writing this script (per instructions). The only external dependency is
the black-box data loader in labor_zero_v3_harness_20260628.py, called
via importlib without reading its source.

============================================================
MY FIRST INTERPRETATION (private this is written before coding, per
instructions to state it in the docstring before implementing):
============================================================

Per-year order of operations (variant "A" / the default / "my first
interpretation"):

  1. INJECTION (refill) CHECK, evaluated at the START of the year,
     using the CURRENT year's run-sleeve return (i.e., the return that
     is ABOUT TO be applied this year) as the "hold_if_crash" signal.
     Rationale: the spec says "sono nen no un'yo return ga mainasu no
     toshi wa mikoku suru" (the year in which that year's run return is
     negative, injection is skipped) -- I read "sono nen" (that year) as
     referring to the same year being processed, i.e. a same-year /
     contemporaneous reference, not the prior year's realized return.
     This is a look-ahead in a strict causal sense, but it is what the
     literal Japanese most naturally denotes ("its own year's return"),
     so I adopt it as interpretation A / the primary interpretation.
     If run balance < thr AND reserve not yet used AND not yet injected
     AND (NOT hold_if_crash OR this year's run return >= 0):
         move 100% of remaining reserve into run (one-time, lump sum).
  2. SPENDING DECISION: compute the required withdrawal for the year
     (nominal-fixed or floor-version rule -- see below), using
     pre-growth (start-of-year, post-injection) balances.
  3. WITHDRAWAL: pay first from run, then from reserve, for the
     spending amount decided in step 2. If total assets < required
     spending, pay out everything (both sleeves go to 0) and count a
     labor year.
  4. GROWTH: apply (1 + run_return) to the remaining run balance and
     (1 + res_return) to the remaining reserve balance, for THIS same
     year's returns (the same return values referenced in step 1's
     hold_if_crash check, and used to grow the post-withdrawal
     balances).

This is "injection -> spending -> growth", with the injection's
hold_if_crash test and the year's growth both keyed to the SAME year's
return values (i.e. we know the whole year's return in advance when
deciding whether to inject -- a simplifying, literal reading of "sono
nen no return").

ORDER_VARIANTS implemented (see ORDER_VARIANTS below):
  A = "injection(this-year-return) -> spend -> grow(this-year-return)"
      (primary / first interpretation, described above)
  B = "spend -> injection(this-year-return) -> grow(this-year-return)"
      (spend first, then decide injection, then grow)
  C = "injection(prior-year-return) -> spend -> grow(this-year-return)"
      (injection's hold_if_crash test uses the PREVIOUS year's run
      return -- a causally cleaner reading where "sono nen" is
      reinterpreted as "the year just observed" rather than the
      still-unrealized current year; first year of a path has no
      prior return, so injection cannot be blocked in year 1 under
      variant C -- treated as "not blocked" i.e. allowed to inject in
      year 1 if thr condition holds)

All three variants share the same spending rule, same withdrawal
waterfall (run first, then reserve), and same one-time lump-sum
all-in injection rule; only the ORDER and the return used for the
hold_if_crash gate differ.
"""

import argparse
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
    "v3", os.path.join(_THIS, "labor_zero_v3_harness_20260628.py")
)
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)

YEARS = list(range(1975, 2026))  # 51 years
N_HIST = len(YEARS)

SPEND_NOMINAL = 7_200_000.0  # 720万円/年 fixed nominal spending
INFLATION = 1.02
N_SIM_YEARS = 20
BLOCK = 5
N_PATHS = 2000
SEED = 20260629

# --------------------------------------------------------------------
# Configurations (all block=5, N=2000, seed=20260629, hold_if_crash=True)
# --------------------------------------------------------------------
CONFIGS = [
    # name, run0,        res0,        thr,         floor,      top_wr
    ("CFG1", 20_000_000, 20_000_000, 20_000_000, 0.0, None),
    ("CFG2", 20_000_000, 20_000_000, 26_000_000, 0.0, None),
    ("CFG3", 10_000_000, 30_000_000, 14_000_000, 0.0, None),
    ("CFG4", 20_000_000, 20_000_000, 20_000_000, 3_600_000.0, 0.16),
    ("CFG5", 20_000_000, 20_000_000, 26_000_000, 3_600_000.0, 0.16),
    ("CFG6", 24_500_000, 24_500_000, 24_500_000, 0.0, None),
]

ORDER_VARIANTS = ["A", "B", "C"]


def load_return_arrays():
    """Load run (sc2.6) and reserve (100% bond) annual return arrays,
    indexed 0..N_HIST-1 corresponding to YEARS[0]..YEARS[-1]."""
    rets = v3.load_rets()
    res_series = v3.load_mixed_reserve({"bond": 1.0})
    run_h = np.array([float(rets["sc2.6"].loc[y]) for y in YEARS], dtype=float)
    res_h = np.array([float(res_series.loc[y]) for y in YEARS], dtype=float)
    return run_h, res_h


def gen_path_indices(rng):
    """Generate one bootstrap path of length N_SIM_YEARS using block
    bootstrap with block size BLOCK, drawing a start index in
    [0, N_HIST) uniformly for each block, and using (start+j) % N_HIST
    for j = 0..block_len-1 (block_len = min(BLOCK, years_remaining)).
    Returns a list of N_SIM_YEARS integer indices into the historical
    arrays (0-based, aligned so index i <-> run_h[i], res_h[i])."""
    idx = []
    while len(idx) < N_SIM_YEARS:
        start = int(rng.integers(0, N_HIST))
        remaining = N_SIM_YEARS - len(idx)
        block_len = min(BLOCK, remaining)
        for j in range(block_len):
            idx.append((start + j) % N_HIST)
    return idx


def simulate_path(run_rets, res_rets, run0, res0, thr, floor, top_wr,
                   hold_if_crash=True, order="A", trace=False):
    """Simulate one 20-year path. run_rets/res_rets are length-20 arrays
    of annual returns (paired by year). Returns a dict of results, and
    optionally a trace list of per-year rows.

    Spending rule:
      - Nominal-fixed (floor == 0): spend = SPEND_NOMINAL every year
        (no inflation adjustment to spending). If total assets (run+res,
        pre-spend) < spend, labor += 1 and everything is paid out
        (assets -> 0 for the year).
      - Floor version (floor > 0): guaranteed floor is always spent.
        If SPEND_NOMINAL / total_assets <= top_wr, spend is raised to
        SPEND_NOMINAL (full nominal amount); otherwise spend = floor.
        If total_assets < floor (i.e. cannot even cover the floor),
        labor += 1 and everything is paid out. If the resulting spend
        for the year is < SPEND_NOMINAL (i.e. spend == floor, floor
        case triggered because affordability check failed), cut += 1.
    """
    run = run0
    res = res0
    injected = False
    labor = 0
    ruin = 0  # counted once per path if run+res hits 0 for the path
    cut = 0
    rows = []

    floor_mode = floor > 0.0

    for t in range(N_SIM_YEARS):
        r_run = run_rets[t]
        r_res = res_rets[t]
        # return used for hold_if_crash gating, depends on order variant
        if order == "C":
            gate_ret = run_rets[t - 1] if t > 0 else None  # None => not blocked
        else:
            gate_ret = r_run  # variants A and B use this-year (contemporaneous) return

        def do_injection():
            nonlocal run, res, injected
            if (not injected) and res > 0.0 and run < thr:
                if hold_if_crash and (gate_ret is not None) and (gate_ret < 0.0):
                    return False
                run = run + res
                res = 0.0
                injected = True
                return True
            return False

        def do_spend():
            nonlocal run, res, labor, cut
            total = run + res
            if floor_mode:
                if total < floor:
                    spend_amt = total
                    labor += 1
                else:
                    if total > 0 and (SPEND_NOMINAL / total) <= top_wr:
                        spend_amt = SPEND_NOMINAL
                    else:
                        spend_amt = floor
                        cut += 1
            else:
                if total < SPEND_NOMINAL:
                    spend_amt = total
                    labor += 1
                else:
                    spend_amt = SPEND_NOMINAL
            # withdrawal waterfall: run first, then reserve
            from_run = min(run, spend_amt)
            run -= from_run
            remainder = spend_amt - from_run
            from_res = min(res, remainder)
            res -= from_res
            return spend_amt, from_run, from_res

        injected_this_year = False
        spend_amt = 0.0
        labor_before = labor

        if order == "B":
            spend_amt, _, _ = do_spend()
            injected_this_year = do_injection()
        else:
            # order A and C: injection first, then spend
            injected_this_year = do_injection()
            spend_amt, _, _ = do_spend()

        labor_flag = 1 if labor > labor_before else 0

        # growth (always applies this-year's return to remaining balances)
        run = run * (1.0 + r_run)
        res = res * (1.0 + r_res)

        if trace:
            rows.append({
                "t": t,
                "run_ret": r_run,
                "res_ret": r_res,
                "injected": injected_this_year,
                "spend": spend_amt,
                "labor_this_year": labor_flag,
                "run_end": run,
                "res_end": res,
            })

    terminal = run + res
    if terminal <= 1e-6:
        ruin = 1

    result = {
        "labor": labor,
        "ruin": ruin,
        "cut": cut,
        "terminal": terminal,
    }
    if trace:
        result["rows"] = rows
    return result


def run_config(order, name, run0, res0, thr, floor, top_wr,
                run_h, res_h, seed=SEED, n_paths=N_PATHS):
    rng = np.random.default_rng(seed)
    labors = np.zeros(n_paths, dtype=int)
    ruins = np.zeros(n_paths, dtype=int)
    cuts = np.zeros(n_paths, dtype=int)
    terminals = np.zeros(n_paths, dtype=float)

    for p in range(n_paths):
        idx = gen_path_indices(rng)
        run_rets = run_h[idx]
        res_rets = res_h[idx]
        res = simulate_path(
            run_rets, res_rets, run0, res0, thr, floor,
            top_wr if top_wr is not None else 0.0,
            hold_if_crash=True, order=order, trace=False,
        )
        labors[p] = res["labor"]
        ruins[p] = res["ruin"]
        cuts[p] = res["cut"]
        terminals[p] = res["terminal"]

    p_labor0 = float(np.mean(labors == 0))
    p_ruin0 = float(np.mean(ruins == 0))
    labor_mean = float(np.mean(labors))
    cut_mean = float(np.mean(cuts))
    real_terminal = terminals / (INFLATION ** N_SIM_YEARS)
    real_median = float(np.median(real_terminal))

    return {
        "order": order,
        "config": name,
        "P_labor0": round(p_labor0, 4),
        "P_ruin0": round(p_ruin0, 4),
        "labor_mean": round(labor_mean, 4),
        "cut_mean": round(cut_mean, 4),
        "real_terminal_median": round(real_median, 4),
    }


def find_trace_path(order, run0, res0, thr, floor_zero_only_config,
                     run_h, res_h, seed=SEED, n_paths=N_PATHS):
    """Find the first path (by path index under the given seed) for
    which the NOMINAL-FIXED version (floor=0) produces labor > 0, then
    return that path's return arrays plus the trace for both the
    nominal-fixed run and the floor-version run (same config's thr/run0/res0,
    floor from CFG4/CFG5 pairing is chosen by caller)."""
    name, run0_, res0_, thr_, _, _ = floor_zero_only_config
    rng = np.random.default_rng(seed)
    for p in range(n_paths):
        idx = gen_path_indices(rng)
        run_rets = run_h[idx]
        res_rets = res_h[idx]
        res = simulate_path(run_rets, res_rets, run0_, res0_, thr_, 0.0, 0.0,
                             hold_if_crash=True, order=order, trace=False)
        if res["labor"] > 0:
            return idx, run_rets, res_rets
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    print("Loading return data (this can take tens of seconds)...")
    run_h, res_h = load_return_arrays()
    print(f"run_h len={len(run_h)}  res_h len={len(res_h)}")
    print()

    all_results = []
    for order in ORDER_VARIANTS:
        print(f"===== ORDER VARIANT {order} =====")
        header = f"{'CFG':6s}{'P_labor0':>10s}{'P_ruin0':>10s}{'labor_mean':>12s}{'cut_mean':>10s}{'real_term_med':>20s}"
        print(header)
        for (name, run0, res0, thr, floor, top_wr) in CONFIGS:
            r = run_config(order, name, run0, res0, thr, floor, top_wr, run_h, res_h)
            all_results.append(r)
            print(f"{r['config']:6s}{r['P_labor0']:>10.4f}{r['P_ruin0']:>10.4f}"
                  f"{r['labor_mean']:>12.4f}{r['cut_mean']:>10.4f}{r['real_terminal_median']:>20,.4f}")
        print()

    if args.trace:
        print("===== TRACE (order=A, CFG1 nominal-fixed vs CFG4 floor version) =====")
        cfg1 = [c for c in CONFIGS if c[0] == "CFG1"][0]
        cfg4 = [c for c in CONFIGS if c[0] == "CFG4"][0]
        idx, run_rets, res_rets = find_trace_path("A", None, None, None, cfg1, run_h, res_h)
        if idx is None:
            print("No labor-triggering path found in first N_PATHS paths.")
        else:
            print(f"Path bootstrap indices (0-based into 1975-2025 array): {idx}")
            print(f"Corresponding years: {[YEARS[i] for i in idx]}")
            print()
            _, name1, run0_1, res0_1, thr1, floor1, top_wr1 = ("A",) + cfg1[0:1] + cfg1[1:]
            # nominal-fixed trace (CFG1 params, floor=0)
            res_nom = simulate_path(run_rets, res_rets, cfg1[1], cfg1[2], cfg1[3], 0.0, 0.0,
                                     hold_if_crash=True, order="A", trace=True)
            print(f"--- NOMINAL-FIXED (CFG1 params: run0={cfg1[1]}, res0={cfg1[2]}, thr={cfg1[3]}) ---")
            print(f"{'t':>3s}{'year':>6s}{'run_ret':>10s}{'res_ret':>10s}{'inj':>5s}{'labor':>6s}{'spend':>14s}{'run_end':>16s}{'res_end':>16s}")
            for row, y in zip(res_nom["rows"], [YEARS[i] for i in idx]):
                print(f"{row['t']:>3d}{y:>6d}{row['run_ret']:>10.4f}{row['res_ret']:>10.4f}"
                      f"{str(row['injected']):>5s}{row['labor_this_year']:>6d}{row['spend']:>14,.0f}{row['run_end']:>16,.0f}{row['res_end']:>16,.0f}")
            print(f"labor={res_nom['labor']}  ruin={res_nom['ruin']}  terminal={res_nom['terminal']:,.0f}")
            print()
            # floor version trace (CFG4 params: same run0/res0/thr as CFG1 but with floor)
            res_floor = simulate_path(run_rets, res_rets, cfg4[1], cfg4[2], cfg4[3], cfg4[4], cfg4[5],
                                       hold_if_crash=True, order="A", trace=True)
            print(f"--- FLOOR VERSION (CFG4 params: run0={cfg4[1]}, res0={cfg4[2]}, thr={cfg4[3]}, "
                  f"floor={cfg4[4]}, top_wr={cfg4[5]}) ---")
            print(f"{'t':>3s}{'year':>6s}{'run_ret':>10s}{'res_ret':>10s}{'inj':>5s}{'labor':>6s}{'spend':>14s}{'run_end':>16s}{'res_end':>16s}")
            for row, y in zip(res_floor["rows"], [YEARS[i] for i in idx]):
                print(f"{row['t']:>3d}{y:>6d}{row['run_ret']:>10.4f}{row['res_ret']:>10.4f}"
                      f"{str(row['injected']):>5s}{row['labor_this_year']:>6d}{row['spend']:>14,.0f}{row['run_end']:>16,.0f}{row['res_end']:>16,.0f}")
            print(f"labor={res_floor['labor']}  ruin={res_floor['ruin']}  cut={res_floor['cut']}  terminal={res_floor['terminal']:,.0f}")


if __name__ == "__main__":
    main()
