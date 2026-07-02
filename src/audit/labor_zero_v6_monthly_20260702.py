"""
src/audit/labor_zero_v6_monthly_20260702.py
===========================================
Remaining-hole #1 of the v6 critical-verification campaign (CRITVERIFY s6.1-1):
MONTHLY decision granularity.

The campaign established that the v6 annual sim's refill rule is a look-ahead
(consults the current YEAR's return at the start of the year). Under the
implementable annual-lagged convention P drops (nominal-fixed B0 0.8785,
hold-lagged 0.7833; V6-A floor 0.9922). But a real retiree observes MONTHLY:
the true implementable value lies between the annual-lagged lower bound and
the annual-lookahead upper bound. This harness resolves that gap.

Design (isolates DECISION granularity from SAMPLING granularity):
  - The bootstrap resamples YEAR-blocks exactly like v6.make_block_path
    (identical rng consumption: one integers(0,51) draw per 5-year block,
    4 draws per 20y path) -> the distribution of resampled YEAR sequences is
    identical to the whole campaign's, and same seed => same year sequence =>
    paired McNemar against the annual sims.
  - Each sampled calendar year is expanded into its 12 HISTORICAL months
    (run/res month pairs), preserving the within-year crash shape.
  - Monthly after-tax returns are built from the same daily NAVs used by the
    annual loader, then geometrically adjusted per calendar year so that the
    12-month compound EXACTLY equals the annual after-tax return used by v6
    (removes tax-model mismatch; the annual anchor is the single source of
    truth). Adjustment factor g_Y = (1+R_aftertax)/(1+R_pretax), spread as
    g_Y^(1/12) per month.
  - Spending stays an ANNUAL lump at the start of each retirement year
    (identical to the annual sim) -- only the REFILL decision runs monthly.

Rules (all implementable; month-k decision uses months <= k-1 only):
  A-look  : annual lookahead hold (v6 reference; year-boundary decision)
  A-lag   : annual lagged hold
  A-noH   : annual noHOLD (B0)
  M-noH   : monthly noHOLD (inject any month-start with run<thr)
  M-h1    : monthly hold, gate = last month's run return >= 0
  M-h3    : monthly hold, gate = trailing 3-month compound >= 0
  M-h12   : monthly hold, gate = trailing 12-month compound >= 0
  M-dd30  : inject when run sits >=30% below its running peak (no sign gate)

SELF-TESTS (HALT on failure):
  ST1 monthly 12-month compound == annual after-tax anchor for every year
  ST2 no-refill config: monthly sim == annual sim path-by-path (exact)
  ST3 A-look on sampled paths, seed 20260629, 20:20 thr20 hold -> 0.8745
      and V6-A floor3.6 -> 0.9995 (reproduces the campaign anchors)

ASCII-only prints. CSV -> audit_results/labor_zero_v6_monthly_20260702.csv.
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
YEARS = list(range(1975, 2026))            # 51 calendar years
NY = len(YEARS)
HORIZON_Y = 20
SEEDS = [20260629 + i for i in range(5)]
NPATHS = 2000
BLOCK_Y = 5
INFLATION = 0.02


# ------------------------------------------------------------------ monthly data
def build_monthly():
    """Returns (run_m, res_m): ndarray [51, 12] of after-tax monthly returns,
    per calendar year 1975..2025, geometrically adjusted so each year's
    12-month compound EXACTLY equals the v6 annual after-tax anchor."""
    import src.audit.dd_reduction_harness_20260626 as H
    ctx = H.setup()
    dates = pd.DatetimeIndex(ctx["dates_dt"])
    nav_run, _r, _t, _e = H.build(ctx, scale=2.6)
    nav_run = pd.Series(np.asarray(nav_run, float), index=dates)
    nav_res = pd.Series(np.cumprod(1.0 + np.asarray(ctx["ret_bond"], float)),
                        index=dates)
    anchor_run = v6.get_paired_history()[0]          # after-tax annual, aligned YEARS
    anchor_res = v6.get_paired_history()[1]

    def monthly_matrix(nav, anchor):
        mlast = nav.groupby([nav.index.year, nav.index.month]).last()
        mret = mlast.pct_change()
        out = np.empty((NY, 12))
        for yi, y in enumerate(YEARS):
            m = np.array([float(mret.loc[(y, mm)]) for mm in range(1, 13)])
            pre = float(np.prod(1.0 + m))
            g = (1.0 + float(anchor[yi])) / pre
            out[yi] = (1.0 + m) * g ** (1.0 / 12.0) - 1.0
        return out

    return monthly_matrix(nav_run, anchor_run), monthly_matrix(nav_res, anchor_res)


# ------------------------------------------------------- year-index path sampling
def sample_year_indices(rng, n_years=HORIZON_Y, block=BLOCK_Y):
    """Same rng consumption as v6.make_block_path (one integers(0,51) per block,
    wrap-around), but returns the year INDICES instead of values."""
    idx = np.empty(n_years, int)
    filled = 0
    while filled < n_years:
        start = int(rng.integers(0, NY))
        take = min(block, n_years - filled)
        for j in range(take):
            idx[filled] = (start + j) % NY
            filled += 1
    return idx


# ------------------------------------------------------------------- monthly sim
def sim_monthly(yidx, run_m, res_m, *, run0, reserve0, thr, rule,
                spend=SPEND, floor=0.0, top_wr=None):
    """One retirement over HORIZON_Y years expanded to months.
    rule in {'M-noH','M-h1','M-h3','M-h12','M-dd30','none'}.
    Spending is an annual lump at the start of each retirement year (month 0
    of the year), refill decision at the start of EVERY month using only
    observed (lagged) monthly returns. Returns dict like the annual sims."""
    run, res = float(run0), float(reserve0)
    fired = False
    labor = 0
    cut = 0
    hist = []                              # realized monthly run returns
    peak = run
    n_m = HORIZON_Y * 12
    for km in range(n_m):
        y = km // 12
        mm = km % 12
        r_run = run_m[yidx[y], mm]
        r_res = res_m[yidx[y], mm]
        # --- refill decision (start of month, lagged info only) ---
        if rule != "none" and (not fired) and res > 1e-6:
            trigger = run < thr
            gate = True
            if rule == "M-h1":
                gate = (len(hist) == 0) or (hist[-1] >= 0.0)
            elif rule == "M-h3":
                gate = (len(hist) < 3) or (np.prod(1.0 + np.array(hist[-3:])) >= 1.0)
            elif rule == "M-h12":
                gate = (len(hist) < 12) or (np.prod(1.0 + np.array(hist[-12:])) >= 1.0)
            elif rule == "M-dd30":
                trigger = (peak > 0) and (run <= 0.70 * peak)
            if trigger and gate:
                run += res
                res = 0.0
                fired = True
        # --- annual lump spend at the start of each retirement year ---
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
        # --- growth ---
        run *= (1.0 + r_run)
        res *= (1.0 + r_res)
        hist.append(r_run)
        if run > peak:
            peak = run
    return dict(labor_years=labor, ruin=int((run + res) <= 1e-6),
                terminal=run + res, cut_years=cut)


def annual_paths_from_idx(yidx, run_h, res_h):
    return run_h[yidx], res_h[yidx]


# ------------------------------------------------------------------- self-tests
def _self_tests(run_m, res_m, run_h, res_h):
    ok = True
    # ST1: yearly compound matches annual anchor exactly
    err_run = np.max(np.abs(np.prod(1.0 + run_m, axis=1) - (1.0 + run_h)))
    err_res = np.max(np.abs(np.prod(1.0 + res_m, axis=1) - (1.0 + res_h)))
    print("ST1 monthly->annual compound: max_err run=%.2e res=%.2e -> %s"
          % (err_run, err_res, "PASS" if max(err_run, err_res) < 1e-10 else "FAIL"))
    ok &= max(err_run, err_res) < 1e-10
    # ST2: no-refill equivalence, 300 sampled paths
    rng = np.random.default_rng(SEEDS[0])
    same = True
    for _ in range(300):
        yidx = sample_year_indices(rng)
        rp, sp = annual_paths_from_idx(yidx, run_h, res_h)
        a = la.sim_conv(rp, sp, convention="lagged", hold_if_crash=False,
                        run0=40 * M, reserve0=0.0, thr=0.0)
        b = sim_monthly(yidx, run_m, res_m, run0=40 * M, reserve0=0.0,
                        thr=0.0, rule="none")
        if a["labor_years"] != b["labor_years"] or abs(a["terminal"] - b["terminal"]) > 1.0:
            same = False
            break
    print("ST2 no-refill monthly == annual (300 paths): %s" % ("PASS" if same else "FAIL"))
    ok &= same
    # ST3: annual lookahead on idx-sampled paths reproduces campaign anchors
    rng = np.random.default_rng(SEEDS[0])
    lab_n = np.empty(NPATHS, int)
    lab_f = np.empty(NPATHS, int)
    for p in range(NPATHS):
        yidx = sample_year_indices(rng)
        rp, sp = annual_paths_from_idx(yidx, run_h, res_h)
        lab_n[p] = la.sim_conv(rp, sp, convention="lookahead",
                               run0=20 * M, reserve0=20 * M, thr=20 * M)["labor_years"]
        lab_f[p] = la.sim_conv(rp, sp, convention="lookahead",
                               run0=20 * M, reserve0=20 * M, thr=20 * M,
                               floor=3.6 * M, top_wr=0.16)["labor_years"]
    p_n = float((lab_n == 0).mean())
    p_f = float((lab_f == 0).mean())
    print("ST3 idx-sampler anchors: nominal=%.4f (exp 0.8745)  floor=%.4f (exp 0.9995) -> %s"
          % (p_n, p_f, "PASS" if (abs(p_n - 0.8745) < 1e-9 and abs(p_f - 0.9995) < 1e-9)
             else "FAIL"))
    ok &= abs(p_n - 0.8745) < 1e-9 and abs(p_f - 0.9995) < 1e-9
    return ok


# ------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("MONTHLY decision granularity -- resolving the annual-lagged/lookahead gap")
    print("  year-block sampling identical to v6 (block=%dy) | N=%d | seeds=%s"
          % (BLOCK_Y, NPATHS, SEEDS))
    print("=" * 100)
    run_h, res_h = v6.get_paired_history()
    run_m, res_m = build_monthly()
    if not _self_tests(run_m, res_m, run_h, res_h):
        print("HALT: self-test failed.")
        return

    NOM = dict(run0=20 * M, reserve0=20 * M, thr=20 * M)
    FLR = dict(NOM, floor=3.6 * M, top_wr=0.16)
    annual_rules = [("A-look", dict(convention="lookahead", hold_if_crash=True)),
                    ("A-lag", dict(convention="lagged", hold_if_crash=True)),
                    ("A-noH", dict(convention="lagged", hold_if_crash=False))]
    monthly_rules = ["M-noH", "M-h1", "M-h3", "M-h12", "M-dd30"]

    deflate = (1.0 + INFLATION) ** HORIZON_Y
    rows = []
    store = {}
    for world, base in (("nominal", NOM), ("floor3.6", FLR)):
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            idx_list = [sample_year_indices(rng) for _ in range(NPATHS)]
            for name, kw in annual_rules:
                lab = np.empty(NPATHS, int)
                cut = np.empty(NPATHS)
                term = np.empty(NPATHS)
                for p, yidx in enumerate(idx_list):
                    rp, sp = annual_paths_from_idx(yidx, run_h, res_h)
                    r = la.sim_conv(rp, sp, run0=base["run0"], reserve0=base["reserve0"],
                                    thr=base["thr"], floor=base.get("floor", 0.0),
                                    top_wr=base.get("top_wr"), **kw)
                    lab[p] = r["labor_years"]
                    cut[p] = r["cut_years"]
                    term[p] = r["terminal"]
                store.setdefault((world, name), []).append(lab)
                rows.append(dict(world=world, rule=name, seed=sd,
                                 p_labor0=float((lab == 0).mean()),
                                 cut_mean=float(cut.mean()),
                                 term_real_med_M=round(float(np.median(term)) / deflate / M, 1)))
            for rule in monthly_rules:
                lab = np.empty(NPATHS, int)
                cut = np.empty(NPATHS)
                term = np.empty(NPATHS)
                for p, yidx in enumerate(idx_list):
                    r = sim_monthly(yidx, run_m, res_m, run0=base["run0"],
                                    reserve0=base["reserve0"], thr=base["thr"],
                                    rule=rule, floor=base.get("floor", 0.0),
                                    top_wr=base.get("top_wr"))
                    lab[p] = r["labor_years"]
                    cut[p] = r["cut_years"]
                    term[p] = r["terminal"]
                store.setdefault((world, rule), []).append(lab)
                rows.append(dict(world=world, rule=rule, seed=sd,
                                 p_labor0=float((lab == 0).mean()),
                                 cut_mean=float(cut.mean()),
                                 term_real_med_M=round(float(np.median(term)) / deflate / M, 1)))
    df = pd.DataFrame(rows)

    for world in ("nominal", "floor3.6"):
        print("\n[%s] P(labor0) mean over %d seeds [min-max]  (annual refs then monthly rules)"
              % (world, len(SEEDS)))
        for name in [n for n, _ in annual_rules] + monthly_rules:
            sub = df[(df.world == world) & (df.rule == name)]
            print("  %-8s P=%.4f [%.4f-%.4f]  cut=%.2f  termRealMed=%8.0fM"
                  % (name, sub.p_labor0.mean(), sub.p_labor0.min(), sub.p_labor0.max(),
                     sub.cut_mean.mean(), sub.term_real_med_M.mean()))
        print("  paired vs A-look (pooled %d paths):" % (NPATHS * len(SEEDS)))
        ref = np.concatenate(store[(world, "A-look")])
        for name in ["A-lag", "A-noH"] + monthly_rules:
            lab = np.concatenate(store[(world, name)])
            r = lzstats.paired_diff(lab, ref)
            print("    %-8s dP=%+.4f  A_only=%4d B_only=%4d  p=%.2e"
                  % (name, r["p_a"] - r["p_b"], r["a_only"], r["b_only"], r["mcnemar_p"]))

    out = os.path.join(_REPO, "audit_results", "labor_zero_v6_monthly_20260702.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (monthly granularity).")


if __name__ == "__main__":
    main()
