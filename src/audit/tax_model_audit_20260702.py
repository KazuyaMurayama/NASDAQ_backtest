"""
src/audit/tax_model_audit_20260702.py
=====================================
Task 1 of the P09C1/scale critical-verification campaign (plan:
docs/superpowers/plans/2026-07-02-p09c1-scale-critical-verification.md).

FINDINGS THIS HARNESS QUANTIFIES (established by code reading):
  - The repo tax coefficient 0.8273 = 1 - 0.20315 * 0.85 (20.315% capital-gains
    tax on an assumed 85% realized fraction). Documented in
    CASH_SLEEVE_REPORT_20260607.md; NOT in EVALUATION_STANDARD.
  - Convention (a), used by the frontier/labor chain annual tables:
    r_after = r_pre * 0.8273 for ALL years INCLUDING NEGATIVE ones.
    A negative year multiplied by 0.8273 becomes SHALLOWER -- this implicitly
    assumes an instant same-rate tax REBATE on losses. Reality (JP 申告分離):
    losses only offset future gains within 3 years.
  - Convention (metric-level), used by _apply_aftertax in the 10-metric
    tables: CAGR/Worst10Y/P10 * 0.8273 at the METRIC level; MaxDD/Sharpe/
    worst-day left PRE-TAX (the report headers say "all numbers after-tax",
    which is imprecise; pre-tax MaxDD is the conservative side).

TAX VARIANTS COMPARED (annual wealth-path simulation, per sleeve):
  (a) current   : r_at = r_pre * 0.8273 every year (incl. negative)
  (b) realistic : each year, realized gain = phi * max(gain,0) + min(gain,0)
                  (phi = realized fraction of gains, base 0.85);
                  taxable = max(0, realized_gain - loss_carry(3y FIFO));
                  tax = 0.20315 * taxable, paid from the account year-end;
                  losses feed the carryforward (expire after 3 years).
  (c) strict    : (b) with phi = 1.0 (every gain realized annually; upper
                  bound of tax drag under annual realization).

Outputs:
  [1] per scale (1.4 / 1.6 / 2.0 / 2.6): full-period CAGR, MaxDD of the
      ANNUAL after-tax wealth path, Worst10Y, Worst5Y under (a)/(b)/(c),
      plus the (a)-(b) gap  -> "systematic optimism" per scale.
  [2] LABOR_ZERO_V7 impact: P(labor=0) for the ADOPTED rule (floor 3.6M,
      top_wr 0.12, monthly all-in refill below 12M) and the nominal-fixed
      B0 reference, re-anchoring the monthly matrices to each tax variant's
      annual series (same monthly shapes; same construction as the stress
      harness). Self-test: variant (a) must reproduce P=0.9983 / 0.8785.

Assumptions stated: the non-realized fraction (1-phi) is treated as never
taxed inside the horizon (deferral approximated as exemption -- favourable to
variant (b), i.e. the (a)-(b) gap reported here is a LOWER bound).
ASCII-only prints. CSV -> audit_results/tax_model_audit_20260702.csv.
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
    "fu", os.path.join(_THIS, "labor_zero_v6_floorup_20260702.py"))
fu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fu)
lm, la, v6 = fu.lm, fu.la, fu.v6

M = 1e6
TAX = 0.20315
PHI_BASE = 0.85
AFTER_TAX = 0.8273
SEEDS = [20260629 + i for i in range(5)]
N = 2000
YEARS = list(range(1975, 2026))


# ----------------------------------------------------------------- tax variants
def after_tax_series(r_pre, variant, phi=PHI_BASE):
    """Annual after-tax return series from pre-tax annual returns.
    variant in {'a','b','c'}. Wealth-path simulation for b/c."""
    r_pre = np.asarray(r_pre, float)
    if variant == "a":
        return r_pre * AFTER_TAX
    ph = 1.0 if variant == "c" else phi
    w = 1.0
    out = np.empty_like(r_pre)
    carry = []                       # list of (expiry_index, remaining_loss)
    for i, r in enumerate(r_pre):
        gain = w * r
        realized = ph * gain if gain > 0 else gain
        # offset against carryforward (FIFO, 3-year expiry)
        taxable = realized
        if taxable > 0 and carry:
            new_carry = []
            for (exp_i, loss) in carry:
                if exp_i < i:
                    continue                      # expired
                use = min(loss, taxable)
                loss -= use
                taxable -= use
                if loss > 1e-12:
                    new_carry.append((exp_i, loss))
            carry = new_carry
        tax = TAX * taxable if taxable > 0 else 0.0
        if realized < 0:
            carry.append((i + 3, -realized))      # usable in the next 3 years
        w_next = w * (1.0 + r) - tax
        out[i] = w_next / w - 1.0
        w = w_next
    return out


def path_metrics(r):
    """CAGR / MaxDD (annual wealth path) / Worst10Y / Worst5Y from annual rets."""
    r = np.asarray(r, float)
    w = np.cumprod(1.0 + r)
    n = len(r)
    cagr = w[-1] ** (1.0 / n) - 1.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))
    mdd = float(np.min(np.concatenate([[1.0], w]) / peak - 1.0))

    def worst_roll(k):
        vals = [(w[i + k - 1] / (w[i - 1] if i > 0 else 1.0)) ** (1.0 / k) - 1.0
                for i in range(0, n - k + 1)]
        return float(min(vals))
    return dict(cagr=cagr, maxdd_annual=mdd,
                worst10y=worst_roll(10), worst5y=worst_roll(5))


# ------------------------------------------------------------------------ main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("TAX MODEL AUDIT: 0.8273-all-years vs realistic 3y-carryforward")
    print("  0.8273 = 1 - 0.20315*0.85 (documented in CASH_SLEEVE_REPORT_20260607)")
    print("=" * 100)

    # pre-tax annual series from the same builder the labor chain used
    import src.audit.dd_reduction_harness_20260626 as H
    from src.audit.run_p01_backtest_20260611 import _calendar_year_returns
    ctx = H.setup()
    dates = pd.DatetimeIndex(ctx["dates_dt"])
    pre = {}
    for s in (1.4, 1.6, 2.0, 2.6):
        nav, _r, _t, _e = H.build(ctx, scale=s)
        cy = _calendar_year_returns(pd.Series(np.asarray(nav, float), index=dates))
        cy = cy[(cy.index >= 1975) & (cy.index <= 2025)]
        pre["sc%.1f" % s] = np.array([float(cy.loc[y]) for y in YEARS])
    nav_b = pd.Series(np.cumprod(1.0 + np.asarray(ctx["ret_bond"], float)), index=dates)
    cyb = _calendar_year_returns(nav_b)
    cyb = cyb[(cyb.index >= 1975) & (cyb.index <= 2025)]
    pre["bond"] = np.array([float(cyb.loc[y]) for y in YEARS])

    # self-test: variant (a) on sc2.6 must equal the labor chain's series
    run_h, res_h = v6.get_paired_history()
    err = np.max(np.abs(after_tax_series(pre["sc2.6"], "a") - run_h))
    errb = np.max(np.abs(after_tax_series(pre["bond"], "a") - res_h))
    print("SELF-TEST variant(a) == labor-chain series: sc2.6 err=%.2e bond err=%.2e -> %s"
          % (err, errb, "PASS" if max(err, errb) < 1e-12 else "FAIL"))
    if max(err, errb) >= 1e-12:
        print("HALT")
        return

    print("\n[1] per-scale metrics by tax variant (annual wealth path, 1975-2025)")
    rows = []
    for key in ("sc1.4", "sc1.6", "sc2.0", "sc2.6", "bond"):
        print("  %s  (pre-tax CAGR=%.4f)" % (key, path_metrics(pre[key])["cagr"]))
        base = None
        for var, lbl in (("a", "a: x0.8273 all yrs"), ("b", "b: carryfwd phi=0.85"),
                         ("c", "c: carryfwd phi=1.0")):
            m = path_metrics(after_tax_series(pre[key], var))
            rows.append(dict(series=key, variant=var, **m))
            gap = "" if var == "a" else "  dCAGR=%+.4f dMaxDD=%+.4f" % (
                m["cagr"] - base["cagr"], m["maxdd_annual"] - base["maxdd_annual"])
            if var == "a":
                base = m
            print("    %-22s CAGR=%.4f MaxDD(ann)=%.4f W10Y=%.4f W5Y=%.4f%s"
                  % (lbl, m["cagr"], m["maxdd_annual"], m["worst10y"], m["worst5y"], gap))

    print("\n[2] LABOR_ZERO_V7 impact: adopted rule (floor3.6 top0.12 M-noH12) and")
    print("    nominal B0 (noHOLD thr20), monthly matrices re-anchored per variant")
    run_m, res_m = lm.build_monthly()

    def reanchor(mat, ann_from, ann_to):
        out = np.empty_like(mat)
        for yi in range(mat.shape[0]):
            g = (1.0 + ann_to[yi]) / (1.0 + ann_from[yi])
            out[yi] = (1.0 + mat[yi]) * g ** (1.0 / 12.0) - 1.0
        return out

    for var in ("a", "b", "c"):
        rr = after_tax_series(pre["sc2.6"], var)
        bb = after_tax_series(pre["bond"], var)
        rm = reanchor(run_m, run_h, rr)
        sm = reanchor(res_m, res_h, bb)
        # adopted (monthly floor rule)
        ps_a, ps_n = [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            la_ad = np.empty(N, int)
            la_no = np.empty(N, int)
            for p in range(N):
                yidx = lm.sample_year_indices(rng)
                la_ad[p] = fu.sim_m2(yidx, rm, sm, run0=20 * M, reserve0=20 * M,
                                     thr=12 * M, rule="M-noH", floor=3.6 * M,
                                     top_wr=0.12)["labor_years"]
                la_no[p] = la.sim_conv(rr[yidx], bb[yidx], convention="lagged",
                                       hold_if_crash=False, run0=20 * M,
                                       reserve0=20 * M, thr=20 * M)["labor_years"]
            ps_a.append(float((la_ad == 0).mean()))
            ps_n.append(float((la_no == 0).mean()))
        ps_a, ps_n = np.array(ps_a), np.array(ps_n)
        rows.append(dict(series="V7_adopted", variant=var, cagr=np.nan,
                         maxdd_annual=np.nan, worst10y=np.nan, worst5y=np.nan,
                         p_labor0=ps_a.mean(), p_min=ps_a.min(), p_max=ps_a.max()))
        rows.append(dict(series="B0_nominal", variant=var, cagr=np.nan,
                         maxdd_annual=np.nan, worst10y=np.nan, worst5y=np.nan,
                         p_labor0=ps_n.mean(), p_min=ps_n.min(), p_max=ps_n.max()))
        print("  variant %s: adopted P=%.4f [%.4f-%.4f]   nominal-B0 P=%.4f [%.4f-%.4f]"
              % (var, ps_a.mean(), ps_a.min(), ps_a.max(),
                 ps_n.mean(), ps_n.min(), ps_n.max()))

    out = os.path.join(_REPO, "audit_results", "tax_model_audit_20260702.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("\n  wrote %s" % out)
    print("Done (tax model audit).")


if __name__ == "__main__":
    main()
