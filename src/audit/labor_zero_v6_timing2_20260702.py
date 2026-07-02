"""
src/audit/labor_zero_v6_timing2_20260702.py
============================================
Task 4 of the v6 critical-verification campaign (plan: docs/superpowers/plans/
2026-07-02-labor-zero-v6-critical-verification.md): re-search for a refill
TIMING rule that beats the current best IMPLEMENTABLE (non-look-ahead) rule.

Context (Task 3 finding, confirmed in audit_results/labor_zero_v6_lookahead_20260702.csv):
v6's original refill decision is look-ahead (uses the CURRENT year's full-year
return to decide whether to hold off refilling). The only implementable
convention is "lagged": hold_if_crash consults last year's REALIZED run return
(k=0 => treated as non-negative => refill allowed); injected money still earns
the current year's run return from day 1 of that year (you decide on Jan 1
using Dec 31 information, which is legitimate).

Under "lagged", the previously-reported "early injection helps" results
degenerate:
  - Any rule that wants to inject "as early as possible" collapses to
    day-0-all-in (thr >= run0) because day 0 has no prior-year signal to hold
    on. Day-0-all-in = "N_20_20_thr20_noHOLD"-equivalent, i.e. instant deposit
    of the whole reserve at t=0. That is config B1 = 0.8343 (5-seed mean).
  - The lagged, signal-driven single-tranche rule (thr=20M, hold consults
    previous year) is B2 = 0.7833.
  - The best OBSERVED implementable rule so far is B0: NO SIGNAL AT ALL
    (hold_if_crash=False under "lagged" convention, i.e. inject the instant
    run<thr regardless of last year's sign) = 0.8785. B0 is
    "N_20_20_thr20_noHOLD" run under convention="lagged" in the Task-3 script
    (confirmed here again as this file's baseline self-test).

Floor-variant (floor=3.6M, top_wr=0.16) baselines under "lagged":
  - thr=20M, hold consults previous year (signal-driven)      = 0.9922
  - thr=26M => day-0-all-in (no signal possible)               = 0.9944

Question this file answers: within the space of rules that use ONLY
information available at the time of the decision (no look-ahead), is there
ANYTHING that beats B0=0.8785 (nominal-fixed) or 0.9944 (floor variant)?
This is the fair re-test of claim C2 ("timing dials are all dead") that the
2026-07-01 sweep (labor_zero_v6_refill_timing_20260701.py) did NOT do, because
that sweep used the look-ahead convention throughout.

Implementation: a single self-contained sim_conv2() extends
labor_zero_v6_lookahead_20260702.sim_conv() with:
  - noHOLD threshold sweep (thr, alloc)
  - recovery-confirm injection (L consecutive non-negative REALIZED years, lag=1)
  - realized-drawdown trigger on the run sleeve, optionally gated by recovery
  - partial + early combination (two-tranche)
  - reserve-side gate: inject only if last year's RESERVE return was positive
  - staged mechanical drip (25%/yr, no signal)

SELF-TEST (required): sim_conv2 reduced to "lagged, single tranche, thr=20M,
hold consults previous year" must reproduce lookahead.sim_conv(..., convention=
"lagged", hold_if_crash=True) path-by-path over 300 paths (B2 anchor). HALT on
mismatch.

Statistics: lzstats.paired_diff() against B0 (nominal) / floor-B0 (floor),
same seed = same path set (pooled over 5 seeds x 2000 paths = 10000 pairs).

ASCII-only prints. CSV -> audit_results/labor_zero_v6_timing2_20260702.csv.
No commit (per instructions).
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
# lookahead module's __main__ guard prevents auto-run; exec is safe/side-effect-free
_spec_la.loader.exec_module(lzla)

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


# --------------------------------------------------------------------- sim core
def sim_conv2(run_path, res_path, *, run0, reserve0, spend=SPEND,
              floor=0.0, top_wr=None,
              # --- rule selector ---
              rule="single", thr=None,
              # rule="single": classic one-shot tranche, gated by `gate`
              gate="none",       # "none" | "prevyear" | "recover{L}" | "dd{X}" | "res_prevyear"
              # rule="two_tranche": thr1 unconditional partial, thr2 gated remainder
              thr1=None, frac1=0.5, thr2=None, gate2="recover2",
              # rule="drip": mechanical staged transfer, no signal
              drip_frac=0.25, drip_years=4):
    """One retirement, LAGGED information convention throughout (no look-ahead):
    every injection decision at the START of year k may use only realized
    information through year k-1 (last year's returns, drawdown-to-date). The
    injected money earns the CURRENT year k run return (decided on Jan 1 using
    Dec 31 data -- legitimate/implementable).

    rule="single": one all-in transfer of the full remaining reserve once the
      trigger fires (thr breach) AND gate is satisfied. gate:
        "none"        : no signal -- inject as soon as run<thr (= B0 shape)
        "prevyear"     : last year's run return (or last PRE-retirement year at
                         k=0, treated non-negative) must be >=0 (= B2 shape)
        "recover{L}"   : the last L REALIZED run years (k-L..k-1) must all be
                         >=0. At k<L, uses whatever pre-retirement history is
                         available and treats missing years as satisfied
                         (so k=0 always qualifies, matching B2's k=0 rule)
        "dd{X}"        : run's realized drawdown from its running peak (as of
                         end of year k-1) must be >= X (e.g. "dd0.30"), i.e.
                         inject once a real observed loss has already happened.
                         Combined with prevyear-style "not still falling"
                         sub-gate is NOT applied here (X alone; kept simple).
        "dd{X}_prevyear": dd{X} AND last year's run return >=0 (recovery
                         confirmation after a realized drawdown)
        "res_prevyear" : last year's RESERVE return must be >=0

    rule="two_tranche": at any point run<thr1, immediately move frac1 of the
      ORIGINAL reserve0 (partial, unconditional, once). Later, once run<thr2
      (thr2<=thr1) AND gate2 ("recover2" = last 2 realized run years
      non-negative) is satisfied, move the remaining reserve (once).

    rule="drip": mechanically move reserve0*drip_frac at the START of years
      1..drip_years (k=1..drip_years, i.e. NOT at k=0 -- day-0 drip is
      indistinguishable from a smaller day-0 lump and is out of scope here).
      No signal used at all.
    """
    run, res = float(run0), float(reserve0)
    n = len(run_path)
    fired = False
    fired1 = False
    fired2 = False
    labor = 0
    cut = 0
    run_peak = run
    prev_returns = []       # realized run returns seen so far (k-1, k-2, ...)

    for k in range(n):
        r_run = run_path[k]
        run_peak = max(run_peak, run)

        # ---------------------------------------------------------- refill
        if rule == "single":
            if (not fired) and run < thr and res > 1e-6:
                ok = _gate_ok(gate, k, prev_returns, run, run_peak,
                              res_path, k)
                if ok:
                    run += res
                    res = 0.0
                    fired = True
        elif rule == "two_tranche":
            if (not fired1) and run < thr1 and res > 1e-6:
                move = min(res, reserve0 * frac1)
                run += move
                res -= move
                fired1 = True
            if fired1 and (not fired2) and run < thr2 and res > 1e-6:
                ok = _gate_ok(gate2, k, prev_returns, run, run_peak,
                              res_path, k)
                if ok:
                    run += res
                    res = 0.0
                    fired2 = True
        elif rule == "drip":
            if 1 <= k <= drip_years and res > 1e-6:
                move = min(res, reserve0 * drip_frac)
                run += move
                res -= move
        else:
            raise ValueError("unknown rule %r" % rule)

        # ---------------------------------------------------------- spend
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
            spend_k = total
        if floor > 0.0 and spend_k < spend - 1e-6:
            cut += 1

        # ---------------------------------------------------------- pay
        if run >= spend_k:
            run -= spend_k
        else:
            res -= (spend_k - run)
            run = 0.0
        if res < 0:
            res = 0.0

        # ---------------------------------------------------------- growth
        run *= (1.0 + r_run)
        res *= (1.0 + res_path[k])
        run_peak = max(run_peak, run)

        # record realized return for use as "previous year" from k+1 onward
        prev_returns.append(r_run)

    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6),
                terminal=run + res, cut_years=cut)


def _gate_ok(gate, k, prev_returns, run, run_peak, res_path, kk):
    """Evaluate a lagged (no-look-ahead) gate at the START of year k, using
    only prev_returns = realized run returns for years 0..k-1 (list, may be
    shorter than k only at k=0 where it is empty)."""
    if gate == "none":
        return True
    if gate == "prevyear":
        if k == 0:
            return True          # last pre-retirement year treated as >=0
        return prev_returns[k - 1] >= 0.0
    if gate.startswith("recover"):
        L = int(gate[len("recover"):])
        if k < L:
            return True          # insufficient history -> treated as satisfied
        return all(r >= 0.0 for r in prev_returns[k - L:k])
    if gate.startswith("dd"):
        rest = gate[2:]
        if rest.endswith("_prevyear"):
            X = float(rest[:-len("_prevyear")])
            dd = (run_peak - run) / run_peak if run_peak > 0 else 0.0
            pv_ok = True if k == 0 else (prev_returns[k - 1] >= 0.0)
            return (dd >= X) and pv_ok
        X = float(rest)
        dd = (run_peak - run) / run_peak if run_peak > 0 else 0.0
        return dd >= X
    if gate == "res_prevyear":
        if k == 0:
            return True
        return prev_returns[k - 1] >= 0.0    # note: prev_returns holds RUN returns;
        # reserve-gate variant is handled by a dedicated wrapper (see sim_conv2b)
    raise ValueError("unknown gate %r" % gate)


def sim_conv2b_res_gate(run_path, res_path, *, run0, reserve0, thr, spend=SPEND,
                         floor=0.0, top_wr=None):
    """rule="single" with gate on the RESERVE sleeve's previous-year return
    (a separate function because _gate_ok only tracks run returns)."""
    run, res = float(run0), float(reserve0)
    n = len(run_path)
    fired = False
    labor = 0
    cut = 0
    prev_res_returns = []
    for k in range(n):
        r_run = run_path[k]
        if (not fired) and run < thr and res > 1e-6:
            ok = True if k == 0 else (prev_res_returns[k - 1] >= 0.0)
            if ok:
                run += res
                res = 0.0
                fired = True
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
            spend_k = total
        if floor > 0.0 and spend_k < spend - 1e-6:
            cut += 1
        if run >= spend_k:
            run -= spend_k
        else:
            res -= (spend_k - run)
            run = 0.0
        if res < 0:
            res = 0.0
        run *= (1.0 + r_run)
        res *= (1.0 + res_path[k])
        prev_res_returns.append(res_path[k])
    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6),
                terminal=run + res, cut_years=cut)


# ------------------------------------------------------------------- MC driver
def gen_paths(run_h, res_h, seed, n_paths=NPATHS, block=BLOCK, horizon=HORIZON):
    """Identical rng consumption to lzla.gen_paths / v6.mc_prob."""
    rng = np.random.default_rng(seed)
    return [v6.make_block_path(rng, run_h, res_h, n=horizon, block=block)
            for _ in range(n_paths)]


def run_rule(paths, fn, cfg):
    labors = np.empty(len(paths), int)
    terms = np.empty(len(paths))
    cuts = np.empty(len(paths), int)
    for i, (rp, sp) in enumerate(paths):
        r = fn(rp, sp, **cfg)
        labors[i] = r["labor_years"]
        terms[i] = r["terminal"]
        cuts[i] = r["cut_years"]
    return labors, terms, cuts


# --------------------------------------------------------------- rule catalog
def build_nominal_rules():
    """(name, fn, cfg) tuples, nominal-fixed world (no floor)."""
    rules = []
    base = dict(run0=20 * M, reserve0=20 * M)
    base_1030 = dict(run0=10 * M, reserve0=30 * M)

    # [1] noHOLD thr sweep x allocation
    for label, b in (("20:20", base), ("14:26", dict(run0=14 * M, reserve0=26 * M)),
                      ("10:30", base_1030)):
        thrs = [20, 18, 16, 14, 12] if label != "10:30" else [10, 8, 6]
        for t in thrs:
            rules.append((
                "1-noHOLD thr%dM alloc%s" % (t, label),
                sim_conv2, dict(**b, spend=SPEND, rule="single", thr=t * M, gate="none")))

    # [2] recovery-confirm L in {1,2}, thr in {20,26}
    for L in (1, 2):
        for t in (20, 26):
            rules.append((
                "2-recover%d thr%dM alloc20:20" % (L, t),
                sim_conv2, dict(**base, rule="single", thr=t * M, gate="recover%d" % L)))

    # [3] realized-drawdown trigger, X in {20,30,40}%, then (a) noHOLD (b) prevyear
    for X in (0.20, 0.30, 0.40):
        rules.append((
            "3a-dd%d%% immediate alloc20:20" % int(X * 100),
            sim_conv2, dict(**base, rule="single", thr=20 * M, gate="dd%.2f" % X)))
        rules.append((
            "3b-dd%d%%+prevyear alloc20:20" % int(X * 100),
            sim_conv2, dict(**base, rule="single", thr=20 * M, gate="dd%.2f_prevyear" % X)))

    # [4] partial + early combination
    rules.append((
        "4a-two_tranche thr1=26M(50%%)+thr2=20M(recover2)",
        sim_conv2, dict(**base, rule="two_tranche", thr1=26 * M, frac1=0.5,
                        thr2=20 * M, gate2="recover2")))
    rules.append((
        "4b-two_tranche thr1=20M(50%%)+thr2=10M(noHOLD)",
        sim_conv2, dict(**base, rule="two_tranche", thr1=20 * M, frac1=0.5,
                        thr2=10 * M, gate2="none")))

    # [5] reserve-side gate: inject only if last year's reserve return positive
    rules.append((
        "5-res_prevyear thr20M alloc20:20",
        sim_conv2b_res_gate, dict(**base, thr=20 * M)))

    # [6] staged mechanical drip, 25%/yr for 4 years, no signal
    rules.append((
        "6-drip25pct x4yr alloc20:20",
        sim_conv2, dict(**base, rule="drip", drip_frac=0.25, drip_years=4)))

    return rules


def build_floor_rules():
    """Floor-variant re-runs (floor=3.6M, top_wr=0.16, alloc 20:20 fixed)."""
    base = dict(run0=20 * M, reserve0=20 * M, floor=3.6 * M, top_wr=0.16)
    rules = [
        ("F1-noHOLD thr20M", sim_conv2, dict(**base, rule="single", thr=20 * M, gate="none")),
        ("F2-recover2 thr20M", sim_conv2, dict(**base, rule="single", thr=20 * M, gate="recover2")),
        ("F3-dd30%% immediate", sim_conv2, dict(**base, rule="single", thr=20 * M, gate="dd0.30")),
        ("F4-drip25pct x4yr", sim_conv2, dict(**base, rule="drip", drip_frac=0.25, drip_years=4)),
    ]
    return rules


# ------------------------------------------------------------------- self-test
def _self_test(run_h, res_h):
    """sim_conv2(rule='single', thr=20M, gate='prevyear') must reproduce
    lzla.sim_conv(convention='lagged', hold_if_crash=True, thr=20M) path-by-path
    (this is the B2 anchor = 0.7833). Also check gate='none' reproduces
    convention='lagged', hold_if_crash=False (the B0 anchor = 0.8785), and the
    floor variant likewise."""
    paths = gen_paths(run_h, res_h, SEEDS[0], n_paths=300)
    checks = [
        ("B2 single/prevyear vs lagged+hold",
         dict(run0=20 * M, reserve0=20 * M, rule="single", thr=20 * M, gate="prevyear"),
         dict(run0=20 * M, reserve0=20 * M, thr=20 * M, hold_if_crash=True)),
        ("B0 single/none vs lagged+noHOLD",
         dict(run0=20 * M, reserve0=20 * M, rule="single", thr=20 * M, gate="none"),
         dict(run0=20 * M, reserve0=20 * M, thr=20 * M, hold_if_crash=False)),
        ("floor B2 single/prevyear vs lagged+hold (floor)",
         dict(run0=20 * M, reserve0=20 * M, rule="single", thr=20 * M, gate="prevyear",
              floor=3.6 * M, top_wr=0.16),
         dict(run0=20 * M, reserve0=20 * M, thr=20 * M, hold_if_crash=True,
              floor=3.6 * M, top_wr=0.16)),
    ]
    all_ok = True
    for label, cfg2, cfg_la in checks:
        ok = True
        for rp, sp in paths:
            a = sim_conv2(rp, sp, **cfg2)
            b = lzla.sim_conv(rp, sp, convention="lagged", **cfg_la)
            if (a["labor_years"] != b["labor_years"]
                    or abs(a["terminal"] - b["terminal"]) > 1e-3
                    or a["cut_years"] != b["cut_years"]):
                ok = False
                break
        print("  SELF-TEST %-45s -> %s" % (label, "PASS" if ok else "FAIL"))
        all_ok = all_ok and ok
    return all_ok


# ------------------------------------------------------------------------ main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("Task 4: refill TIMING re-search under the LAGGED (implementable) convention")
    print("  block=%d N=%d seeds=%s" % (BLOCK, NPATHS, SEEDS))
    print("=" * 100)
    run_h, res_h = v6.get_paired_history()

    print("\n[self-test]")
    if not _self_test(run_h, res_h):
        print("HALT: self-test failed.")
        return

    nominal_rules = build_nominal_rules()
    floor_rules = build_floor_rules()

    deflate = (1.0 + INFLATION) ** HORIZON
    rows = []
    # store[seed]["nominal"|"floor"][rule_name] = labors array
    store = {s: {"nominal": {}, "floor": {}} for s in SEEDS}
    # baseline arrays too
    base_store = {s: {} for s in SEEDS}

    for seed in SEEDS:
        paths = gen_paths(run_h, res_h, seed)

        # baselines: B0 nominal (single/none thr20M alloc20:20), floor B0-lagged(thr20,recover1)=0.9922, thr26=0.9944
        b0_labors, b0_terms, b0_cuts = run_rule(
            paths, sim_conv2, dict(run0=20 * M, reserve0=20 * M, rule="single",
                                    thr=20 * M, gate="none"))
        base_store[seed]["B0_nominal"] = b0_labors

        fb_thr20_labors, _, _ = run_rule(
            paths, sim_conv2, dict(run0=20 * M, reserve0=20 * M, rule="single",
                                    thr=20 * M, gate="prevyear",
                                    floor=3.6 * M, top_wr=0.16))
        base_store[seed]["Fbase_thr20_lagged"] = fb_thr20_labors

        fb_thr26_labors, _, _ = run_rule(
            paths, sim_conv2, dict(run0=20 * M, reserve0=20 * M, rule="single",
                                    thr=26 * M, gate="none",   # thr>=run0 -> day0 all-in
                                    floor=3.6 * M, top_wr=0.16))
        base_store[seed]["Fbase_thr26_day0"] = fb_thr26_labors

        for name, fn, cfg in nominal_rules:
            labors, terms, cuts = run_rule(paths, fn, cfg)
            store[seed]["nominal"][name] = labors
            rows.append(dict(seed=seed, world="nominal", rule=name,
                             p_labor0=float((labors == 0).mean()),
                             labor_mean=float(labors.mean()),
                             term_real_med_M=round(float(np.median(terms)) / deflate / M, 1)))

        for name, fn, cfg in floor_rules:
            labors, terms, cuts = run_rule(paths, fn, cfg)
            store[seed]["floor"][name] = labors
            rows.append(dict(seed=seed, world="floor", rule=name,
                             p_labor0=float((labors == 0).mean()),
                             labor_mean=float(labors.mean()),
                             cut_mean=float(cuts.mean()),
                             term_real_med_M=round(float(np.median(terms)) / deflate / M, 1)))

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------- report: nominal
    print("\n[1] NOMINAL world -- P(labor0) per rule (mean over %d seeds [min-max])"
          % len(SEEDS))
    print("    baseline B0 (noHOLD thr20M alloc20:20, lagged) = 0.8785 (confirmed below)")
    b0_ps = [base_store[s]["B0_nominal"] for s in SEEDS]
    b0_p_vals = [float((la == 0).mean()) for la in b0_ps]
    print("    B0 reproduced here: %.4f [%.4f-%.4f]"
          % (np.mean(b0_p_vals), min(b0_p_vals), max(b0_p_vals)))
    print()
    print("    %-52s %-22s %-9s %-9s" % ("rule", "P(labor0)", "dP_vs_B0", "mcnemar_p"))
    beat_b0 = []
    for name, _, _ in nominal_rules:
        sub = df[(df.world == "nominal") & (df.rule == name)].p_labor0
        la_pool = np.concatenate([store[s]["nominal"][name] for s in SEEDS])
        b0_pool = np.concatenate([base_store[s]["B0_nominal"] for s in SEEDS])
        r = lzstats.paired_diff(la_pool, b0_pool)
        dp = r["p_a"] - r["p_b"]
        sig_better = (dp > 0) and (r["mcnemar_p"] < 0.05)
        flag = "  <== BEATS B0" if sig_better else ""
        print("    %-52s %.4f [%.4f-%.4f]  %+.4f   %.2e%s"
              % (name, sub.mean(), sub.min(), sub.max(), dp, r["mcnemar_p"], flag))
        if sig_better:
            beat_b0.append((name, dp, r["mcnemar_p"]))

    print("\n    Rules beating B0 at p<0.05: %s" % (beat_b0 if beat_b0 else "NONE"))

    # ---------------------------------------------------------- report: floor
    print("\n[2] FLOOR world (floor=3.6M, top_wr=0.16, alloc20:20) -- P(labor0)")
    fb20_pool_all = np.concatenate([base_store[s]["Fbase_thr20_lagged"] for s in SEEDS])
    fb26_pool_all = np.concatenate([base_store[s]["Fbase_thr26_day0"] for s in SEEDS])
    fb20_ps = [float((base_store[s]["Fbase_thr20_lagged"] == 0).mean()) for s in SEEDS]
    fb26_ps = [float((base_store[s]["Fbase_thr26_day0"] == 0).mean()) for s in SEEDS]
    print("    baseline thr20M(lagged,recover1)  reproduced: %.4f [%.4f-%.4f]  (claimed 0.9922)"
          % (np.mean(fb20_ps), min(fb20_ps), max(fb20_ps)))
    print("    baseline thr26M(day0 all-in)       reproduced: %.4f [%.4f-%.4f]  (claimed 0.9944)"
          % (np.mean(fb26_ps), min(fb26_ps), max(fb26_ps)))
    print()
    print("    %-40s %-22s %-9s(vs thr26) %-9s" % ("rule", "P(labor0)", "dP", "mcnemar_p"))
    beat_floor = []
    for name, _, _ in floor_rules:
        sub = df[(df.world == "floor") & (df.rule == name)].p_labor0
        cut_sub = df[(df.world == "floor") & (df.rule == name)].cut_mean
        la_pool = np.concatenate([store[s]["floor"][name] for s in SEEDS])
        r = lzstats.paired_diff(la_pool, fb26_pool_all)
        dp = r["p_a"] - r["p_b"]
        sig_better = (dp > 0) and (r["mcnemar_p"] < 0.05)
        flag = "  <== BEATS 0.9944" if sig_better else ""
        print("    %-40s %.4f [%.4f-%.4f] cut=%.3f  %+.4f   %.2e%s"
              % (name, sub.mean(), sub.min(), sub.max(), cut_sub.mean(), dp, r["mcnemar_p"], flag))
        if sig_better:
            beat_floor.append((name, dp, r["mcnemar_p"]))
    print("\n    Rules beating 0.9944 at p<0.05: %s" % (beat_floor if beat_floor else "NONE"))

    out = os.path.join(_REPO, "audit_results", "labor_zero_v6_timing2_20260702.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("\nDone (Task 4 timing re-search).")


if __name__ == "__main__":
    main()
