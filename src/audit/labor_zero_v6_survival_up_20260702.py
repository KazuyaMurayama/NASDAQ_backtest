"""
src/audit/labor_zero_v6_survival_up_20260702.py
===============================================
Follow-up to the v6 critical-verification campaign (user request 2026-07-02):
"keep the nominal-fixed 7.2M spend and look for levers that raise P(labor=0)
further" -- under IMPLEMENTABLE conventions only (no look-ahead; cf R-STAT-4).

Baseline (from the campaign): the best implementable nominal-fixed rule is
B0 = run20/res20, noHOLD immediate all-in refill at thr=run0, P=0.8785
(5 seeds x 2000 paths, block=5). The look-ahead 0.917 "ceiling" is not
implementable.

Levers explored here (none of which the v6 canon ever swept under an
implementable convention):
  L1 PAYMENT ORDER : pay annual spending from the RESERVE first (protect the
     leveraged sleeve's compounding), vs the canon's run-first. Includes the
     "no refill at all" variant (reserve pays until exhausted = natural glide).
  L2 RUN STRATEGY  : scale variants sc2.0/2.2/2.4/2.6 and the gold/bond
     IN-leg blends N4/X4 as the run sleeve (same B0 shape). NOTE: these are
     all selected on the same 1975-2025 history -> selection-bias caveat
     applies (exploratory, not a promotion).
  L3 RESERVE MIX   : bond / gold / 50-50 / bond70-gold20-cash10 under the
     realistic convention (the old "reserve mix is irrelevant" finding assumed
     the reserve is fully injected at the crash entrance -- under noHOLD the
     reserve can live for years, so the mix could matter now).
  L4 COMBOS        : best-of-above combinations.

Statistics: same-seed => same resampled YEAR sequence (path generation draws
year indices independently of the return values), so configs are PAIRED even
across different run strategies / reserve mixes. McNemar on discordant paths.

SELF-TEST: run_first + noHOLD variant must equal la.sim_conv(lagged/noHOLD)
path-by-path; res_first with reserve0=0 must equal run_first with reserve0=0.

ASCII-only prints. CSV -> audit_results/labor_zero_v6_survival_up_20260702.csv.
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

_spec = importlib.util.spec_from_file_location(
    "la", os.path.join(_THIS, "labor_zero_v6_lookahead_20260702.py"))
la = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(la)
v6 = la.v6

_spec2 = importlib.util.spec_from_file_location(
    "lzstats", os.path.join(_THIS, "labor_zero_v6_stats_helpers_20260702.py"))
lzstats = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(lzstats)

M = 1_000_000.0
SPEND = 7.2 * M
SEEDS = [20260629 + i for i in range(5)]
NPATHS = 2000
BLOCK = 5
INFLATION = 0.02


def sim_pay(run_path, res_path, *, run0, reserve0, thr, pay_order="run_first",
            refill=True, hold_signal="none", spend=SPEND):
    """Nominal-fixed sim with configurable payment order and implementable
    refill gating. hold_signal: 'none' (inject whenever run<thr) or
    'prevyear' (skip injection in year k if run return of year k-1 < 0;
    year 0 always allowed = free initial allocation)."""
    run, res = float(run0), float(reserve0)
    n = len(run_path)
    fired = False
    labor = 0
    for k in range(n):
        # refill decision (start of year, implementable info only)
        if refill:
            gate_ok = True
            if hold_signal == "prevyear" and k > 0 and run_path[k - 1] < 0:
                gate_ok = False
            if gate_ok and (not fired) and run < thr and res > 1e-6:
                run += res
                res = 0.0
                fired = True
        # spend
        total = run + res
        spend_k = spend
        if total + 1e-6 < spend_k:
            labor += 1
            spend_k = total
        if pay_order == "run_first":
            if run >= spend_k:
                run -= spend_k
            else:
                res -= (spend_k - run)
                run = 0.0
        else:                                        # res_first
            if res >= spend_k:
                res -= spend_k
            else:
                run -= (spend_k - res)
                res = 0.0
        if res < 0:
            res = 0.0
        if run < 0:
            run = 0.0
        # growth
        run *= (1.0 + run_path[k])
        res *= (1.0 + res_path[k])
    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6),
                terminal=run + res)


def run_cfg(paths, cfg):
    labors = np.empty(len(paths), int)
    terms = np.empty(len(paths))
    for i, (rp, sp) in enumerate(paths):
        r = sim_pay(rp, sp, **cfg)
        labors[i] = r["labor_years"]
        terms[i] = r["terminal"]
    return labors, terms


def _self_test():
    run_h, res_h = v6.get_paired_history()
    paths = la.gen_paths(run_h, res_h, SEEDS[0], n_paths=300)
    cfg_b0 = dict(run0=20 * M, reserve0=20 * M, thr=20 * M)
    for rp, sp in paths:
        a = sim_pay(rp, sp, pay_order="run_first", hold_signal="none", **cfg_b0)
        b = la.sim_conv(rp, sp, convention="lagged", hold_if_crash=False,
                        run0=20 * M, reserve0=20 * M, thr=20 * M)
        if a["labor_years"] != b["labor_years"] or abs(a["terminal"] - b["terminal"]) > 1e-3:
            print("SELF-TEST FAIL: run_first/noHOLD != sim_conv(lagged,noHOLD)")
            return False
    for rp, sp in paths[:100]:
        a = sim_pay(rp, sp, run0=40 * M, reserve0=0.0, thr=0.0, pay_order="run_first")
        b = sim_pay(rp, sp, run0=40 * M, reserve0=0.0, thr=0.0, pay_order="res_first")
        if a["labor_years"] != b["labor_years"] or abs(a["terminal"] - b["terminal"]) > 1e-3:
            print("SELF-TEST FAIL: pay-order must not matter with empty reserve")
            return False
    print("SELF-TEST PASS: run_first/noHOLD == sim_conv(lagged,noHOLD) 300 paths; "
          "empty-reserve order-invariance 100 paths")
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("SURVIVAL-UP search: nominal-fixed 7.2M, implementable conventions only")
    print("  block=%d N=%d seeds=%s  baseline B0 (run-first noHOLD 20:20 thr20) = 0.8785"
          % (BLOCK, NPATHS, SEEDS))
    print("=" * 100)
    if not _self_test():
        print("HALT")
        return
    deflate = (1.0 + INFLATION) ** 20
    rows = []
    store = {}          # (label) -> pooled labor array (over seeds, concatenated)

    def run_exp(label, strat, reserve_weights, cfg):
        run_h, res_h = v6.get_paired_history(strat=strat,
                                             reserve_weights=reserve_weights)
        labs_all, ps, terms_all = [], [], []
        for sd in SEEDS:
            paths = la.gen_paths(run_h, res_h, sd, n_paths=NPATHS, block=BLOCK)
            labors, terms = run_cfg(paths, cfg)
            labs_all.append(labors)
            terms_all.append(terms)
            ps.append(float((labors == 0).mean()))
        ps = np.array(ps)
        pooled = np.concatenate(labs_all)
        store[label] = pooled
        tmed = float(np.median(np.concatenate(terms_all))) / deflate / M
        rows.append(dict(label=label, p_mean=ps.mean(), p_min=ps.min(),
                         p_max=ps.max(), term_real_med_M=round(tmed, 1)))
        print("  %-52s P=%.4f [%.4f-%.4f]  termRealMed=%8.0fM"
              % (label, ps.mean(), ps.min(), ps.max(), tmed))
        return label

    B0 = dict(run0=20 * M, reserve0=20 * M, thr=20 * M,
              pay_order="run_first", hold_signal="none")

    print("\n[E0] baseline")
    base = run_exp("B0 sc2.6 bond 20:20 run_first noHOLD thr20", "sc2.6", None, B0)

    print("\n[E1] payment order (sc2.6, bond reserve)")
    run_exp("E1a 20:20 res_first noHOLD thr20", "sc2.6", None,
            dict(B0, pay_order="res_first"))
    run_exp("E1b 20:20 res_first NO-refill (glide)", "sc2.6", None,
            dict(run0=20 * M, reserve0=20 * M, thr=0.0, refill=False,
                 pay_order="res_first"))
    run_exp("E1c 14:26 res_first noHOLD thr14", "sc2.6", None,
            dict(run0=14 * M, reserve0=26 * M, thr=14 * M, pay_order="res_first",
                 hold_signal="none"))
    run_exp("E1d 10:30 res_first noHOLD thr10", "sc2.6", None,
            dict(run0=10 * M, reserve0=30 * M, thr=10 * M, pay_order="res_first",
                 hold_signal="none"))
    run_exp("E1e 24:16 res_first noHOLD thr24", "sc2.6", None,
            dict(run0=24 * M, reserve0=16 * M, thr=24 * M, pay_order="res_first",
                 hold_signal="none"))
    run_exp("E1f 20:20 res_first prevyear-hold thr20", "sc2.6", None,
            dict(B0, pay_order="res_first", hold_signal="prevyear"))
    run_exp("E1g 24:16 run_first noHOLD thr24", "sc2.6", None,
            dict(run0=24 * M, reserve0=16 * M, thr=24 * M, pay_order="run_first",
                 hold_signal="none"))
    run_exp("E1h 30:10 run_first noHOLD thr30", "sc2.6", None,
            dict(run0=30 * M, reserve0=10 * M, thr=30 * M, pay_order="run_first",
                 hold_signal="none"))

    print("\n[E2] run-sleeve strategy (B0 shape; SELECTION-BIAS CAVEAT: all chosen"
          "\n     on the same history; exploratory only)")
    for strat in ("sc2.0", "sc2.2", "sc2.4", "N4", "X4"):
        run_exp("E2 %s bond 20:20 run_first noHOLD thr20" % strat, strat, None, B0)

    print("\n[E3] reserve composition (sc2.6, B0 shape; reserve can now live"
          "\n     for years under noHOLD -> mix could matter)")
    for wname, w in [("gold100", {"gold": 1.0}),
                     ("bond50_gold50", {"bond": 0.5, "gold": 0.5}),
                     ("bond70_gold20_cash10", {"bond": 0.7, "gold": 0.2, "cash": 0.1}),
                     ("cash100", {"cash": 1.0})]:
        run_exp("E3 sc2.6 %s 20:20 run_first noHOLD thr20" % wname, "sc2.6", w, B0)

    print("\n[E4] combos of the best individual levers (chosen a priori as the"
          "\n     top candidates: res_first x strategy x reserve mix)")
    for strat in ("sc2.2", "sc2.6", "N4"):
        for wname, w in [("bond100", None), ("bond50_gold50", {"bond": 0.5, "gold": 0.5})]:
            run_exp("E4 %s %s 20:20 res_first noHOLD thr20" % (strat, wname),
                    strat, w, dict(B0, pay_order="res_first"))

    print("\n[pairwise] every config vs B0 (same seeds -> same resampled year"
          "\n           sequences -> paired McNemar)")
    base_lab = store[base]
    print("    %-52s      dP  A_only B_only  mcnemar_p" % "config (A) vs B0 (B)")
    for lbl, lab in store.items():
        if lbl == base:
            continue
        r = lzstats.paired_diff(lab, base_lab)
        print("    %-52s %+.4f  %5d  %5d   %.2e"
              % (lbl, r["p_a"] - r["p_b"], r["a_only"], r["b_only"], r["mcnemar_p"]))

    df = pd.DataFrame(rows)
    out = os.path.join(_REPO, "audit_results", "labor_zero_v6_survival_up_20260702.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (survival-up search).")


if __name__ == "__main__":
    main()
