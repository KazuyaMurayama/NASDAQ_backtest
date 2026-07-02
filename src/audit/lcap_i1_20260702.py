"""
src/audit/lcap_i1_20260702.py
=============================
Improvement candidate I1/I2 of the P09C1/scale campaign
(P09_SCALE_CRITVERIFY_20260702.md s4): DAILY EFFECTIVE-LEVERAGE CAP on the
sc2.6 run sleeve.

Motivation (campaign findings): sc2.6's IN-day effective leverage is mean
7.55 / p95 11.7 / max 12.48. The right tail (a) sets the forced-liquidation
intraday distance to -5.3 pct at max L, and (b) is punished hardest by the
realistic (3y-carryforward) tax path. A cap clips the tail while keeping the
average exposure roughly intact.

Mechanism: build(ctx, scale=2.6, regime_scale=s) with s(t) = min(1, cap/L(t)),
where L(t) is the same effective-leverage path used by the click365 harness.
(The builder modulates the realized IN-leg return around cash: r_mod = cash +
s*(r_in - cash) -- the standard LEVER-C approximation.)

REJECTION RULE (pre-set in the campaign report): the capped variant must beat
the EQUAL-MEAN-LEVERAGE uniform-delever twin (R-STAT-3) on BOTH CAGR and
MaxDD; otherwise I1 is rejected as "just de-lever".
  twin: s_u = mean(min(L,cap)) / mean(L) over IN days, applied uniformly.

Outputs per cap in {6, 7, 8, 8.68 (=margin-headroom 400 pct <=> I2), 10}:
  [1] pre-tax daily metrics: CAGR full, CAGR_IS/OOS (2021-05-08 split),
      Sharpe, MaxDD, worst day  -- capped vs its twin (paired, same dates)
  [2] realistic-tax (variant b) annual-path CAGR / MaxDD / Worst5Y
  [3] forced-liquidation intraday threshold at the new max L (m=2.88 pct)
  [4] LABOR_ZERO_V7 impact: adopted-rule P(labor=0) with the capped sleeve
      (monthly matrices re-anchored; tax variant a for comparability with
      the canon 0.9983, and variant b for the realistic reading)

SELF-TEST: regime_scale=None reproduces sc2.6 anchors (pre-tax CAGR 0.5093,
daily MaxDD -74.80 pct, labor P(a)=0.9983 via unchanged annual series).
ASCII-only prints. CSV -> audit_results/lcap_i1_20260702.csv.
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
CAPS = (6.0, 7.0, 8.0, 8.68, 10.0)
SCALE = 2.6
OOS_SPLIT = pd.Timestamp("2021-05-08")
SEEDS = [20260629 + i for i in range(5)]
N = 2000
YEARS = list(range(1975, 2026))


def daily_metrics(nav):
    r = nav.pct_change().fillna(0.0)
    n_yr = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = float(nav.iloc[-1] / nav.iloc[0]) ** (1.0 / n_yr) - 1.0
    is_nav = nav[nav.index < OOS_SPLIT]
    oos_nav = nav[nav.index >= OOS_SPLIT]
    def _cagr(x):
        yy = (x.index[-1] - x.index[0]).days / 365.25
        return float(x.iloc[-1] / x.iloc[0]) ** (1.0 / yy) - 1.0
    sharpe = float(r.mean() / r.std() * np.sqrt(252.0))
    peak = nav.cummax()
    mdd = float((nav / peak - 1.0).min())
    wday = float(r.min())
    return dict(cagr=cagr, cagr_is=_cagr(is_nav), cagr_oos=_cagr(oos_nav),
                sharpe=sharpe, maxdd=mdd, worst_day=wday)


def annual_from_nav(nav):
    from src.audit.run_p01_backtest_20260611 import _calendar_year_returns
    cy = _calendar_year_returns(nav)
    cy = cy[(cy.index >= 1975) & (cy.index <= 2025)]
    return np.array([float(cy.loc[y]) for y in YEARS])


def labor_p(run_annual, res_annual, run_m0, res_m0, run_h, res_h):
    """Adopted-rule P with monthly matrices re-anchored to run_annual/res_annual."""
    def reanchor(mat, af, at):
        out = np.empty_like(mat)
        for yi in range(mat.shape[0]):
            g = (1.0 + at[yi]) / (1.0 + af[yi])
            out[yi] = (1.0 + mat[yi]) * g ** (1.0 / 12.0) - 1.0
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import src.audit.dd_reduction_harness_20260626 as H
    import src.audit.k365_recost_20260612 as K
    print("=" * 100)
    print("I1/I2: daily effective-leverage cap on sc%.1f  (twin = equal-mean-L uniform delever)" % SCALE)
    print("=" * 100)
    ctx = H.setup()
    dates = pd.DatetimeIndex(ctx["dates_dt"])
    shared = ctx["shared"]

    # effective leverage path (same construction as click365 harness)
    lev_raw = np.asarray(shared["lev_raw_masked"], float)
    mult_v7 = np.asarray(K._build_v7_mult_custom(ctx["dates_dt"], H.STRONG_MAP), float) * SCALE
    L = pd.Series(lev_raw * mult_v7 * 3.0, index=dates).shift(K.V7_DELAY).fillna(1.0).values
    wn = pd.Series(np.asarray(shared["wn"], float), index=dates).shift(K.V7_DELAY).fillna(0).values
    L_eff = np.where(wn > 0, L, 0.0)
    in_days = L_eff > 0
    meanL = L_eff[in_days].mean()

    # baseline sc2.6
    nav0, _r, _t, _e = H.build(ctx, scale=SCALE)
    nav0 = pd.Series(np.asarray(nav0, float), index=dates)
    m0 = daily_metrics(nav0)
    ann0 = annual_from_nav(nav0)
    run_h, res_h = v6.get_paired_history()
    err = np.max(np.abs(ann0 * 0.8273 - run_h))
    cagr_ann = ta.path_metrics(ann0)["cagr"]        # 1975-2025 annual-series CAGR
    print("SELF-TEST: annual-series CAGR=%.4f (exp 0.5093)  daily MaxDD=%.4f (exp -0.7480)"
          "  ann-chain err=%.1e   [daily-NAV CAGR incl. partial 1974/2026 = %.4f]"
          % (cagr_ann, m0["maxdd"], err, m0["cagr"]))
    if not (abs(cagr_ann - 0.5093) < 5e-4 and abs(m0["maxdd"] - (-0.7480)) < 5e-4
            and err < 1e-12):
        print("HALT: self-test failed")
        return

    run_m0, res_m0 = lm.build_monthly()
    pre_bond = None
    # bond annual pre-tax for variant-b labor runs
    nav_b = pd.Series(np.cumprod(1.0 + np.asarray(ctx["ret_bond"], float)), index=dates)
    pre_bond = annual_from_nav(nav_b)

    rows = []
    print("\n%-10s %7s | %8s %8s %8s %8s | twin: %8s %8s | %s" %
          ("variant", "meanL", "CAGR", "OOS", "MaxDD", "worstD", "CAGR", "MaxDD", "verdict"))
    for cap in CAPS:
        s_cap = np.where(L_eff > cap, cap / np.maximum(L_eff, 1e-9), 1.0)
        capped_meanL = float(np.minimum(L_eff, cap)[in_days].mean())
        s_u = capped_meanL / meanL
        nav_c, _r, _t, _e = H.build(ctx, scale=SCALE, regime_scale=s_cap)
        nav_u, _r, _t, _e = H.build(ctx, scale=SCALE,
                                    regime_scale=np.full_like(s_cap, s_u))
        nav_c = pd.Series(np.asarray(nav_c, float), index=dates)
        nav_u = pd.Series(np.asarray(nav_u, float), index=dates)
        mc, mu = daily_metrics(nav_c), daily_metrics(nav_u)
        beats = (mc["cagr"] > mu["cagr"]) and (mc["maxdd"] > mu["maxdd"])
        verdict = "BEATS TWIN (both)" if beats else "fails R-STAT-3"
        print("cap %5.2f  %7.2f | %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f | %s"
              % (cap, capped_meanL, mc["cagr"], mc["cagr_oos"], mc["maxdd"],
                 mc["worst_day"], mu["cagr"], mu["maxdd"], verdict))
        # realistic-tax annual path + labor P
        ann_c = annual_from_nav(nav_c)
        pm_b = ta.path_metrics(ta.after_tax_series(ann_c, "b"))
        r_star = (MARGIN_RATIO * cap - 1.0) / (cap * (1.0 - MARGIN_RATIO))
        ps_a = labor_p(ann_c * 0.8273, pre_bond * 0.8273, run_m0, res_m0, run_h, res_h)
        ps_b = labor_p(ta.after_tax_series(ann_c, "b"), ta.after_tax_series(pre_bond, "b"),
                       run_m0, res_m0, run_h, res_h)
        print("           tax-b: CAGR=%.4f MaxDD(ann)=%.4f W5Y=%.4f | liq-threshold=%.1f%% |"
              " V7 P(a)=%.4f [%.4f-%.4f]  P(b)=%.4f"
              % (pm_b["cagr"], pm_b["maxdd_annual"], pm_b["worst5y"], r_star * 100,
                 ps_a.mean(), ps_a.min(), ps_a.max(), ps_b.mean()))
        rows.append(dict(cap=cap, mean_L=capped_meanL, s_uniform=s_u,
                         cagr=mc["cagr"], cagr_is=mc["cagr_is"], cagr_oos=mc["cagr_oos"],
                         sharpe=mc["sharpe"], maxdd=mc["maxdd"], worst_day=mc["worst_day"],
                         twin_cagr=mu["cagr"], twin_maxdd=mu["maxdd"],
                         twin_oos=mu["cagr_oos"], beats_twin=beats,
                         taxb_cagr=pm_b["cagr"], taxb_maxdd=pm_b["maxdd_annual"],
                         taxb_w5y=pm_b["worst5y"], liq_thresh=r_star,
                         v7_p_a=ps_a.mean(), v7_p_a_min=ps_a.min(), v7_p_a_max=ps_a.max(),
                         v7_p_b=ps_b.mean()))

    # baseline reference row
    pm0b = ta.path_metrics(ta.after_tax_series(ann0, "b"))
    ps0a = labor_p(ann0 * 0.8273, pre_bond * 0.8273, run_m0, res_m0, run_h, res_h)
    r_star0 = (MARGIN_RATIO * L_eff.max() - 1.0) / (L_eff.max() * (1.0 - MARGIN_RATIO))
    print("\nbaseline sc2.6 (no cap): CAGR=%.4f OOS=%.4f MaxDD=%.4f | tax-b MaxDD(ann)=%.4f"
          " | liq=%.1f%% | V7 P(a)=%.4f" % (m0["cagr"], m0["cagr_oos"], m0["maxdd"],
                                            pm0b["maxdd_annual"], r_star0 * 100, ps0a.mean()))
    rows.append(dict(cap=np.inf, mean_L=meanL, s_uniform=1.0, cagr=m0["cagr"],
                     cagr_is=m0["cagr_is"], cagr_oos=m0["cagr_oos"], sharpe=m0["sharpe"],
                     maxdd=m0["maxdd"], worst_day=m0["worst_day"],
                     taxb_cagr=pm0b["cagr"], taxb_maxdd=pm0b["maxdd_annual"],
                     taxb_w5y=pm0b["worst5y"], liq_thresh=r_star0,
                     v7_p_a=ps0a.mean(), v7_p_a_min=ps0a.min(), v7_p_a_max=ps0a.max()))

    out = os.path.join(_REPO, "audit_results", "lcap_i1_20260702.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (I1 leverage cap).")


if __name__ == "__main__":
    main()
