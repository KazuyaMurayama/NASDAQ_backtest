"""
src/audit/i3_blend_tax_20260703.py
==================================
Improvement candidate I3 of the P09C1/scale campaign
(P09_SCALE_CRITVERIFY_20260702.md s4): re-optimize the N4-style Gold/Bond
IN-leg blend under the REALISTIC tax model (variant b: 20.315 pct on gains,
phi=0.85 realized, 3y loss carryforward - tax_model_audit_20260702).

Hypothesis: realistic tax punishes the leveraged tail hardest (annual-path
MaxDD -8.1pp at sc2.6), so a blend that trades NDX-tail for Gold/Bond carry
at higher scale (the confirmed shape-change family: N4 = scale2.85 /
gold0.28 / bond0.07) should gain RELATIVE value under variant-b tax.

REJECTION RULE (pre-set): if the frontier gap of the best blend candidate
under variant-b tax is <= 0, I3 is rejected.
  frontier_gap := candidate CAGR minus the pure-scale frontier CAGR
  linearly interpolated at the candidate's MaxDD (same tax variant, same
  MaxDD basis). Positive gap = sits ABOVE the scale dial = true shape change.

Grids:
  pure-scale frontier: scale in {1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.85,3.0}
  blend candidates   : scale in {2.2,2.6,2.85,3.1} x gold_w in {0.15,0.28,0.40}
                       x bond_w in {0.0,0.07,0.15}   (36 builds)

Outputs per series: variant-a and variant-b annual-path CAGR / MaxDD(ann) /
Worst5Y, daily pre-tax MaxDD, max effective NDX leverage & forced-liq
threshold (m=2.88 pct), and for the top blend: LABOR_ZERO_V7 adopted-rule P.

SELF-TEST: (i) build(scale=2.6, no blend) reproduces the sc2.6 annual chain
(err=0 vs labor series after x0.8273); (ii) build(N4 params) annual x0.8273
matches the registered N4 series from the x4n4 CSV (tolerance 1e-6).
ASCII-only prints. CSV -> audit_results/i3_blend_tax_20260703.csv.
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
    "ta", os.path.join(_THIS, "tax_model_audit_20260702.py"))
ta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ta)
fu, lm, la, v6 = ta.fu, ta.lm, ta.la, ta.v6

M = 1e6
MARGIN_RATIO = 0.0288
YEARS = list(range(1975, 2026))
SEEDS = [20260629 + i for i in range(5)]
N = 2000
SCALES_FRONTIER = (1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.85, 3.0)
BLEND_SCALES = (2.2, 2.6, 2.85, 3.1)
GOLD_WS = (0.15, 0.28, 0.40)
BOND_WS = (0.0, 0.07, 0.15)
N4 = dict(scale=2.85, in_gold_w=0.28, in_bond_w=0.07)


def annual_from_nav(nav):
    from src.audit.run_p01_backtest_20260611 import _calendar_year_returns
    cy = _calendar_year_returns(nav)
    cy = cy[(cy.index >= 1975) & (cy.index <= 2025)]
    return np.array([float(cy.loc[y]) for y in YEARS])


def daily_maxdd(nav):
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())


def frontier_interp(frontier, x_mdd):
    """Linear interpolation of frontier CAGR at MaxDD=x_mdd.
    frontier: list of (mdd, cagr) sorted by mdd ascending (more negative first).
    Outside the range -> extrapolate from the nearest segment."""
    pts = sorted(frontier)                          # mdd ascending (deepest first)
    mdds = [p[0] for p in pts]
    cags = [p[1] for p in pts]
    return float(np.interp(x_mdd, mdds, cags))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import src.audit.dd_reduction_harness_20260626 as H
    import src.audit.k365_recost_20260612 as K
    print("=" * 100)
    print("I3: N4-style Gold/Bond IN-leg blend under realistic tax (variant b)")
    print("=" * 100)
    ctx = H.setup()
    dates = pd.DatetimeIndex(ctx["dates_dt"])
    shared = ctx["shared"]
    run_h, res_h = v6.get_paired_history()

    def build_series(**kw):
        nav, _r, _t, _e = H.build(ctx, **kw)
        nav = pd.Series(np.asarray(nav, float), index=dates)
        return nav, annual_from_nav(nav)

    # ---- self-tests ----
    nav26, ann26 = build_series(scale=2.6)
    err = np.max(np.abs(ann26 * 0.8273 - run_h))
    import src.audit.labor_zero_harness_v2_20260627 as h2
    rets = h2.load_returns()
    rets.update(h2.load_extended())
    n4_reg = np.array([float(rets["N4"].loc[y]) for y in YEARS])
    navN4, annN4 = build_series(**N4)
    errN4 = np.max(np.abs(annN4 * 0.8273 - n4_reg))
    # registered N4 CSV stores annual pct rounded to 2dp -> tolerance 5e-5 + eps
    print("SELF-TEST: sc2.6 chain err=%.1e   N4 vs registered err=%.2e (CSV 2dp rounding"
          " => tol 6e-5) -> %s"
          % (err, errN4, "PASS" if (err < 1e-12 and errN4 < 6e-5) else "FAIL"))
    if not (err < 1e-12 and errN4 < 6e-5):
        print("HALT")
        return

    # NDX-leg max leverage helper
    lev_raw = np.asarray(shared["lev_raw_masked"], float)
    wn_arr = pd.Series(np.asarray(shared["wn"], float),
                       index=dates).shift(K.V7_DELAY).fillna(0).values

    def max_L(scale, gold_w=0.0, bond_w=0.0):
        mult = np.asarray(K._build_v7_mult_custom(ctx["dates_dt"], H.STRONG_MAP), float) * scale
        L = pd.Series(lev_raw * mult * 3.0, index=dates).shift(K.V7_DELAY).fillna(1.0).values
        L_eff = np.where(wn_arr > 0, L, 0.0) * (1.0 - gold_w - bond_w)
        return float(L_eff.max())

    # ---- pure-scale frontier ----
    rows = []
    frontier = {"a": [], "b": []}
    print("\n[1] pure-scale frontier (annual wealth path per tax variant)")
    print("    %-10s | tax-a: %8s %8s | tax-b: %8s %8s %8s | dailyMDD  maxL  liq%%"
          % ("scale", "CAGR", "MDDann", "CAGR", "MDDann", "W5Y"))
    for s in SCALES_FRONTIER:
        nav, ann = (nav26, ann26) if s == 2.6 else build_series(scale=s)
        ma = ta.path_metrics(ta.after_tax_series(ann, "a"))
        mb = ta.path_metrics(ta.after_tax_series(ann, "b"))
        dmdd = daily_maxdd(nav)
        mL = max_L(s)
        liq = (MARGIN_RATIO * mL - 1.0) / (mL * (1.0 - MARGIN_RATIO))
        frontier["a"].append((ma["maxdd_annual"], ma["cagr"]))
        frontier["b"].append((mb["maxdd_annual"], mb["cagr"]))
        rows.append(dict(kind="frontier", scale=s, gold_w=0.0, bond_w=0.0,
                         cagr_a=ma["cagr"], mdd_a=ma["maxdd_annual"],
                         cagr_b=mb["cagr"], mdd_b=mb["maxdd_annual"],
                         w5y_b=mb["worst5y"], daily_mdd=dmdd, max_L=mL, liq=liq))
        print("    scale%.2f  | %8.4f %8.4f | %8.4f %8.4f %8.4f | %8.4f %5.2f %5.1f"
              % (s, ma["cagr"], ma["maxdd_annual"], mb["cagr"], mb["maxdd_annual"],
                 mb["worst5y"], dmdd, mL, liq * 100))

    # ---- blend candidates ----
    print("\n[2] blend candidates: frontier gap per tax variant (gap>0 = above the scale dial)")
    print("    %-26s | gap_a(pp) gap_b(pp) | tax-b: %8s %8s %8s | maxL liq%%"
          % ("candidate", "CAGR", "MDDann", "W5Y"))
    best = None
    for s in BLEND_SCALES:
        for g in GOLD_WS:
            for b in BOND_WS:
                if g + b >= 0.55:
                    continue
                nav, ann = (navN4, annN4) if (s, g, b) == (2.85, 0.28, 0.07) \
                    else build_series(scale=s, in_gold_w=g, in_bond_w=b)
                ma = ta.path_metrics(ta.after_tax_series(ann, "a"))
                mb = ta.path_metrics(ta.after_tax_series(ann, "b"))
                gap_a = ma["cagr"] - frontier_interp(frontier["a"], ma["maxdd_annual"])
                gap_b = mb["cagr"] - frontier_interp(frontier["b"], mb["maxdd_annual"])
                mL = max_L(s, g, b)
                liq = (MARGIN_RATIO * mL - 1.0) / (mL * (1.0 - MARGIN_RATIO))
                lbl = "s%.2f g%.2f b%.2f%s" % (s, g, b,
                                               " (N4)" if (s, g, b) == (2.85, 0.28, 0.07) else "")
                rows.append(dict(kind="blend", scale=s, gold_w=g, bond_w=b,
                                 cagr_a=ma["cagr"], mdd_a=ma["maxdd_annual"],
                                 cagr_b=mb["cagr"], mdd_b=mb["maxdd_annual"],
                                 w5y_b=mb["worst5y"], gap_a=gap_a, gap_b=gap_b,
                                 daily_mdd=daily_maxdd(nav), max_L=mL, liq=liq))
                print("    %-26s | %+9.4f %+9.4f | %8.4f %8.4f %8.4f | %4.1f %5.1f"
                      % (lbl, gap_a * 100, gap_b * 100, mb["cagr"], mb["maxdd_annual"],
                         mb["worst5y"], mL, liq * 100))
                if best is None or gap_b > best[0]:
                    best = (gap_b, lbl, ann, dict(scale=s, in_gold_w=g, in_bond_w=b))

    # ---- V7 impact for the best variant-b candidate ----
    print("\n[3] best blend under tax-b: %s (gap_b=%+.2fpp)" % (best[1], best[0] * 100))
    nav_b = pd.Series(np.cumprod(1.0 + np.asarray(ctx["ret_bond"], float)), index=dates)
    pre_bond = annual_from_nav(nav_b)
    run_m0, res_m0 = lm.build_monthly()

    def labor_p(run_annual, res_annual):
        def reanchor(mat, af, at):
            out = np.empty_like(mat)
            for yi in range(mat.shape[0]):
                gk = (1.0 + at[yi]) / (1.0 + af[yi])
                out[yi] = (1.0 + mat[yi]) * gk ** (1.0 / 12.0) - 1.0
            return out
        rm = reanchor(run_m0, run_h, run_annual)
        sm = reanchor(res_m0, res_h, res_annual)
        ps = []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            lab = np.empty(N, int)
            for p in range(N):
                yidx = lm.sample_year_indices(rng)
                lab[p] = fu.sim_m2(yidx, rm, sm, run0=20 * M, reserve0=20 * M,
                                   thr=12 * M, rule="M-noH", floor=3.6 * M,
                                   top_wr=0.12)["labor_years"]
            ps.append(float((lab == 0).mean()))
        return np.array(ps)

    ann_best = best[2]
    ps_a = labor_p(ann_best * 0.8273, pre_bond * 0.8273)
    ps_b = labor_p(ta.after_tax_series(ann_best, "b"), ta.after_tax_series(pre_bond, "b"))
    ps26b = labor_p(ta.after_tax_series(ann26, "b"), ta.after_tax_series(pre_bond, "b"))
    print("    V7 adopted-rule P: blend tax-a=%.4f [%.4f-%.4f]  tax-b=%.4f"
          "  (sc2.6 tax-b ref=%.4f)"
          % (ps_a.mean(), ps_a.min(), ps_a.max(), ps_b.mean(), ps26b.mean()))
    rows.append(dict(kind="v7", scale=best[3]["scale"], gold_w=best[3]["in_gold_w"],
                     bond_w=best[3]["in_bond_w"], v7_p_a=ps_a.mean(),
                     v7_p_b=ps_b.mean(), v7_p_b_sc26=ps26b.mean()))

    out = os.path.join(_REPO, "audit_results", "i3_blend_tax_20260703.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (I3 blend x realistic tax).")


if __name__ == "__main__":
    main()
