"""
src/audit/labor_zero_v6_floorup_20260702.py
===========================================
Addendum to the monthly-granularity harness (user request 2026-07-02):
(1) FAIRNESS FIX -- the monthly rules in labor_zero_v6_monthly_20260702.py used
    thr=20M, which the annual lump spend mechanically breaches in month 1, so
    every monthly rule degenerated to near-day-0 all-in. Here the monthly
    threshold is SWEPT (8..18M) so injection reacts to genuine dips, and the
    dd-trigger depth is swept (20/30/40%).
(2) FLOOR-PLAN IMPROVEMENT -- search implementable combos (refill rule x
    top_wr x floor) for the V6-A world, asking: what is the best (floor, P)
    frontier we can actually implement, and can floor=3.6M reach P>=0.999
    without the look-ahead convention?

All rules implementable (decisions use observed info only). Year-block
sampling identical to v6 (same rng consumption; ST anchors reproduced by the
monthly module's self-tests, re-checked here cheaply).

ASCII-only prints. CSV -> audit_results/labor_zero_v6_floorup_20260702.csv.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, _REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_specm = importlib.util.spec_from_file_location(
    "lm", os.path.join(_THIS, "labor_zero_v6_monthly_20260702.py"))
lm = importlib.util.module_from_spec(_specm)
_specm.loader.exec_module(lm)
la = lm.la
v6 = lm.v6
lzstats = lm.lzstats

M = 1_000_000.0
SPEND = 7.2 * M
SEEDS = [20260629 + i for i in range(5)]
NPATHS = 2000
HORIZON_Y = 20
INFLATION = 0.02


def sim_m2(yidx, run_m, res_m, *, run0, reserve0, thr=0.0, rule="none",
           dd=0.30, spend=SPEND, floor=0.0, top_wr=None):
    """Monthly sim, generalized from lm.sim_monthly: rule in
    {'none','M-noH','M-h1','M-dd'}; dd = drawdown trigger depth for 'M-dd'."""
    run, res = float(run0), float(reserve0)
    fired = False
    labor = 0
    cut = 0
    prev = None
    peak = run
    for km in range(HORIZON_Y * 12):
        y, mm = km // 12, km % 12
        r_run = run_m[yidx[y], mm]
        r_res = res_m[yidx[y], mm]
        if rule != "none" and (not fired) and res > 1e-6:
            if rule == "M-dd":
                trigger = (peak > 0) and (run <= (1.0 - dd) * peak)
                gate = True
            else:
                trigger = run < thr
                gate = True if rule == "M-noH" else (prev is None or prev >= 0.0)
            if trigger and gate:
                run += res
                res = 0.0
                fired = True
        if mm == 0:
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
        res *= (1.0 + r_res)
        prev = r_run
        if run > peak:
            peak = run
    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6),
                terminal=run + res, cut_years=cut)


def _self_test(run_m, res_m):
    rng = np.random.default_rng(SEEDS[0])
    for _ in range(200):
        yidx = lm.sample_year_indices(rng)
        a = lm.sim_monthly(yidx, run_m, res_m, run0=20 * M, reserve0=20 * M,
                           thr=20 * M, rule="M-dd30", floor=3.6 * M, top_wr=0.16)
        b = sim_m2(yidx, run_m, res_m, run0=20 * M, reserve0=20 * M,
                   rule="M-dd", dd=0.30, floor=3.6 * M, top_wr=0.16)
        if a["labor_years"] != b["labor_years"] or abs(a["terminal"] - b["terminal"]) > 1.0:
            print("SELF-TEST FAIL: sim_m2 M-dd30 != lm.sim_monthly M-dd30")
            return False
    print("SELF-TEST PASS: sim_m2 == lm.sim_monthly for M-dd30/floor, 200 paths")
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("FLOOR-UP addendum: fair monthly thr sweep + implementable floor-plan frontier")
    print("  year-block sampling as v6 | N=%d | seeds=%s" % (NPATHS, SEEDS))
    print("=" * 100)
    run_h, res_h = v6.get_paired_history()
    run_m, res_m = lm.build_monthly()
    if not _self_test(run_m, res_m):
        print("HALT")
        return
    deflate = (1.0 + INFLATION) ** HORIZON_Y
    rows = []
    store = {}

    def run_grid(label, world_kw, sim):
        """sim(yidx) -> result dict; runs 5 seeds x NPATHS, stores pooled labor."""
        labs, cuts, terms, ps = [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            lab = np.empty(NPATHS, int)
            cut = np.empty(NPATHS)
            term = np.empty(NPATHS)
            for p in range(NPATHS):
                yidx = lm.sample_year_indices(rng)
                r = sim(yidx)
                lab[p] = r["labor_years"]
                cut[p] = r["cut_years"]
                term[p] = r["terminal"]
            labs.append(lab)
            cuts.append(cut)
            terms.append(term)
            ps.append(float((lab == 0).mean()))
        ps = np.array(ps)
        pooled = np.concatenate(labs)
        store[label] = pooled
        k = int((pooled == 0).sum())
        lo, hi = lzstats.wilson_ci(k, len(pooled))
        tmed = float(np.median(np.concatenate(terms))) / deflate / M
        cm = float(np.concatenate(cuts).mean())
        rows.append(dict(label=label, p_mean=ps.mean(), p_min=ps.min(), p_max=ps.max(),
                         wilson_lo=lo, wilson_hi=hi, cut_mean=cm,
                         term_real_med_M=round(tmed, 1), **world_kw))
        print("  %-46s P=%.4f [%.4f-%.4f] CI[%.4f-%.4f] cut=%.2f term=%8.0fM"
              % (label, ps.mean(), ps.min(), ps.max(), lo, hi, cm, tmed))
        return ps.mean()

    print("\n[1] NOMINAL: fair monthly thr sweep (benchmark A-noH=0.8785)")
    run_grid("A-noH annual lagged noHOLD thr20", dict(world="nominal"),
             lambda yidx: la.sim_conv(*lm.annual_paths_from_idx(yidx, run_h, res_h),
                                      convention="lagged", hold_if_crash=False,
                                      run0=20 * M, reserve0=20 * M, thr=20 * M))
    for thr in (8, 10, 12, 14, 16, 18):
        run_grid("M-noH thr%dM" % thr, dict(world="nominal"),
                 lambda yidx, t=thr: sim_m2(yidx, run_m, res_m, run0=20 * M,
                                            reserve0=20 * M, thr=t * M, rule="M-noH"))
    for dd in (0.20, 0.30, 0.40, 0.50):
        run_grid("M-dd%.0f%%" % (dd * 100), dict(world="nominal"),
                 lambda yidx, d=dd: sim_m2(yidx, run_m, res_m, run0=20 * M,
                                           reserve0=20 * M, rule="M-dd", dd=d))
    ref = store["A-noH annual lagged noHOLD thr20"]
    print("  paired vs A-noH:")
    for lbl in list(store):
        if lbl.startswith("A-noH"):
            continue
        r = lzstats.paired_diff(store[lbl], ref)
        print("    %-40s dP=%+.4f A_only=%4d B_only=%4d p=%.2e"
              % (lbl, r["p_a"] - r["p_b"], r["a_only"], r["b_only"], r["mcnemar_p"]))

    print("\n[2] FLOOR world: implementable combos (rule x top_wr), floor=3.6M")
    combo_store = {}
    for rule_lbl, mk in [
        ("A-noH", lambda tw: (lambda yidx: la.sim_conv(
            *lm.annual_paths_from_idx(yidx, run_h, res_h), convention="lagged",
            hold_if_crash=False, run0=20 * M, reserve0=20 * M, thr=20 * M,
            floor=3.6 * M, top_wr=tw))),
        ("A-lag", lambda tw: (lambda yidx: la.sim_conv(
            *lm.annual_paths_from_idx(yidx, run_h, res_h), convention="lagged",
            hold_if_crash=True, run0=20 * M, reserve0=20 * M, thr=20 * M,
            floor=3.6 * M, top_wr=tw))),
        ("M-dd30", lambda tw: (lambda yidx: sim_m2(
            yidx, run_m, res_m, run0=20 * M, reserve0=20 * M, rule="M-dd",
            dd=0.30, floor=3.6 * M, top_wr=tw))),
        ("M-dd40", lambda tw: (lambda yidx: sim_m2(
            yidx, run_m, res_m, run0=20 * M, reserve0=20 * M, rule="M-dd",
            dd=0.40, floor=3.6 * M, top_wr=tw))),
        ("M-noH12", lambda tw: (lambda yidx: sim_m2(
            yidx, run_m, res_m, run0=20 * M, reserve0=20 * M, thr=12 * M,
            rule="M-noH", floor=3.6 * M, top_wr=tw))),
    ]:
        for tw in (0.12, 0.14, 0.16):
            lbl = "F %s top%.2f" % (rule_lbl, tw)
            pm = run_grid(lbl, dict(world="floor3.6", rule=rule_lbl, top_wr=tw), mk(tw))
            combo_store[(rule_lbl, tw)] = pm

    print("\n[3] Max floor with P>=0.999 (per-seed min criterion) for the top combos")
    top = sorted(combo_store.items(), key=lambda kv: -kv[1])[:3]
    for (rule_lbl, tw), _pm in top:
        mk = dict([("A-noH", 0), ("A-lag", 1), ("M-dd30", 2), ("M-dd40", 3), ("M-noH12", 4)])
        best_floor = None
        for fl in (2.4, 2.8, 3.2, 3.6, 4.0, 4.43):
            ps = []
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                lab = np.empty(NPATHS, int)
                for p in range(NPATHS):
                    yidx = lm.sample_year_indices(rng)
                    if rule_lbl == "A-noH":
                        r = la.sim_conv(*lm.annual_paths_from_idx(yidx, run_h, res_h),
                                        convention="lagged", hold_if_crash=False,
                                        run0=20 * M, reserve0=20 * M, thr=20 * M,
                                        floor=fl * M, top_wr=tw)
                    elif rule_lbl == "A-lag":
                        r = la.sim_conv(*lm.annual_paths_from_idx(yidx, run_h, res_h),
                                        convention="lagged", hold_if_crash=True,
                                        run0=20 * M, reserve0=20 * M, thr=20 * M,
                                        floor=fl * M, top_wr=tw)
                    elif rule_lbl in ("M-dd30", "M-dd40"):
                        r = sim_m2(yidx, run_m, res_m, run0=20 * M, reserve0=20 * M,
                                   rule="M-dd", dd=0.30 if rule_lbl == "M-dd30" else 0.40,
                                   floor=fl * M, top_wr=tw)
                    else:
                        r = sim_m2(yidx, run_m, res_m, run0=20 * M, reserve0=20 * M,
                                   thr=12 * M, rule="M-noH", floor=fl * M, top_wr=tw)
                    lab[p] = r["labor_years"]
                ps.append(float((lab == 0).mean()))
            pmin = min(ps)
            print("    %s top%.2f floor=%.2fM: P=%.4f [minseed %.4f]"
                  % (rule_lbl, tw, fl, float(np.mean(ps)), pmin))
            rows.append(dict(label="maxfloor %s top%.2f fl%.2f" % (rule_lbl, tw, fl),
                             world="floorsearch", p_mean=float(np.mean(ps)), p_min=pmin,
                             p_max=max(ps)))
            if pmin >= 0.999:
                best_floor = fl
        print("    -> %s top%.2f: max floor with all-seed P>=0.999 = %s"
              % (rule_lbl, tw, ("%.2fM" % best_floor) if best_floor else "none<=4.43M"))

    df = pd.DataFrame(rows)
    out = os.path.join(_REPO, "audit_results", "labor_zero_v6_floorup_20260702.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (floor-up addendum).")


if __name__ == "__main__":
    main()
