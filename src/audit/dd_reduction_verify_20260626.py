"""
src/audit/dd_reduction_verify_20260626.py
=========================================
INDEPENDENT re-verification of the workflow-generated DD-reduction candidates
(2026-06-26). The main loop does NOT trust the subagents' reported numbers; this
script re-runs every candidate through the validated harness and, critically,
tests the load-bearing claim:

  "The gold/bond IN-leg blend is a FRONTIER-EFFICIENCY shift, not a de-lever in
   disguise" -- i.e. at the SAME MaxDD, the blended candidate has HIGHER CAGR_OOS
   than a pure-scale uniform twin (no blend).

For each candidate we:
  1. build + metrics10 (standard-10 + WFA + regime).
  2. find the uniform-scale TWIN (no blend) whose MaxDD matches the candidate's
     MaxDD (bisection on scale), and read its CAGR_OOS.
  3. frontier_gap = candidate.CAGR_OOS - twin.CAGR_OOS  (>0 => real shape-change).

R-STAT compliance: MaxDD compared on direct NAV (no block bootstrap). The twin
test is the equal-MaxDD frontier comparison (the correct "timing vs de-lever"
discriminator for a path-dependent extremum).

ASCII-only prints. Writes audit_results/dd_reduction_verify_20260626.csv.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import src.audit.dd_reduction_harness_20260626 as H

# Candidate list from the ideation workflow (wf_702decac-012). kwargs only.
CANDIDATES = [
    ("sc2.0_anchor",            dict(scale=2.0)),
    ("X4_g32b05_s3.0",          dict(scale=3.0, in_gold_w=0.32, in_bond_w=0.05)),
    ("X1_g30b05_s2.95",         dict(scale=2.95, in_gold_w=0.30, in_bond_w=0.05)),
    ("G_g30b06_s3.0",           dict(scale=3.0, in_gold_w=0.30, in_bond_w=0.06)),
    ("N1_g25_s2.6_goldonly",    dict(scale=2.6, in_gold_w=0.25)),
    ("X3_g28b07_s2.85_cf05",    dict(scale=2.85, in_gold_w=0.28, in_bond_w=0.07, cash_floor=0.05)),
    ("N4_g28b07_s2.85",         dict(scale=2.85, in_gold_w=0.28, in_bond_w=0.07)),
    ("X5_g24b05_s2.7",          dict(scale=2.7, in_gold_w=0.24, in_bond_w=0.05)),
    ("N3_g22b06_s2.65_cf05",    dict(scale=2.65, in_gold_w=0.22, in_bond_w=0.06, cash_floor=0.05)),
    ("N2_g22b06_s2.65",         dict(scale=2.65, in_gold_w=0.22, in_bond_w=0.06)),
    ("G_g28b07_s2.9_cf05",      dict(scale=2.9, in_gold_w=0.28, in_bond_w=0.07, cash_floor=0.05)),
    ("N5_g16b06_s2.45_cf05",    dict(scale=2.45, in_gold_w=0.16, in_bond_w=0.06, cash_floor=0.05)),
    ("B_g15b10_s2.6",           dict(scale=2.6, in_gold_w=0.15, in_bond_w=0.10)),
    ("B_g18b07_cf10_s2.55",     dict(scale=2.55, in_gold_w=0.18, in_bond_w=0.07, cash_floor=0.10)),
    ("B_g15b15_s2.7_bal",       dict(scale=2.7, in_gold_w=0.15, in_bond_w=0.15)),
    ("B_g10b10_s2.4_clean",     dict(scale=2.4, in_gold_w=0.10, in_bond_w=0.10)),
    ("B_g10b10_s2.4_cf05",      dict(scale=2.4, in_gold_w=0.10, in_bond_w=0.10, cash_floor=0.05)),
    ("B_g20b10_s2.7",           dict(scale=2.7, in_gold_w=0.20, in_bond_w=0.10)),
    ("G_g26b08_s2.85_cf08",     dict(scale=2.85, in_gold_w=0.26, in_bond_w=0.08, cash_floor=0.08)),
]


def _maxdd_for_scale(ctx, scale):
    """Pure strong-map uniform twin (no blend, no floor) MaxDD + CAGR_OOS at given scale."""
    nav_dt, r, tpy, exc = H.build(ctx, scale=scale)
    m = H.metrics10(ctx, nav_dt, r, tpy, label="twin", with_wfa=False)
    return m["MaxDD"], m["CAGR_OOS"]


def _twin_cagr_at_maxdd(ctx, target_maxdd, lo=1.0, hi=3.5, iters=22):
    """Bisection: find uniform-scale twin whose MaxDD == target_maxdd; return its CAGR_OOS.
    MaxDD is monotone-decreasing (more negative) in scale, so we bisect on scale."""
    dd_lo, _ = _maxdd_for_scale(ctx, lo)   # shallow (less negative)
    dd_hi, _ = _maxdd_for_scale(ctx, hi)   # deep (more negative)
    # target should lie between; clamp
    if target_maxdd >= dd_lo:
        _, c = _maxdd_for_scale(ctx, lo); return c, lo
    if target_maxdd <= dd_hi:
        _, c = _maxdd_for_scale(ctx, hi); return c, hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        dd_mid, _ = _maxdd_for_scale(ctx, mid)
        if dd_mid > target_maxdd:   # twin shallower than target -> need more scale
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    _, c = _maxdd_for_scale(ctx, mid)
    return c, mid


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 110)
    print("DD-REDUCTION CANDIDATES -- INDEPENDENT RE-VERIFICATION + FRONTIER-TWIN TEST  2026-06-26")
    print("Target: CAGR_OOS >= +28.1%% AND MaxDD better than -61.63%%. frontier_gap>0 => real shape-change.")
    print("=" * 110)

    ctx = H.setup()

    # sc2.0 anchor self-check
    nav0, r0, tpy0, _ = H.build(ctx, scale=2.0)
    m0 = H.metrics10(ctx, nav0, r0, tpy0, label="sc2.0", with_wfa=False)
    ok = (abs(m0["CAGR_OOS"] - 0.291102) <= 0.0015 and abs(m0["MaxDD"] + 0.616342) <= 0.0015)
    print("ANCHOR sc2.0: CAGR_OOS %+.4f%% MaxDD %+.4f%% -> %s\n"
          % (m0["CAGR_OOS"]*100, m0["MaxDD"]*100, "MATCH" if ok else "MISMATCH-HALT"))
    if not ok:
        sys.exit(1)

    rows = []
    for name, kw in CANDIDATES:
        nav_dt, r, tpy, exc = H.build(ctx, **kw)
        m = H.metrics10(ctx, nav_dt, r, tpy, label=name, with_wfa=True)
        if name == "sc2.0_anchor":
            twin_cagr, twin_scale, fgap = m["CAGR_OOS"], kw["scale"], 0.0
        else:
            twin_cagr, twin_scale = _twin_cagr_at_maxdd(ctx, m["MaxDD"])
            fgap = m["CAGR_OOS"] - twin_cagr
        row = dict(
            name=name, **{k: kw.get(k) for k in
                          ("scale", "in_gold_w", "in_bond_w", "cash_floor", "bond_gate")},
            CAGR_OOS=round(m["CAGR_OOS"], 6), CAGR_IS=round(m["CAGR_IS"], 6),
            min9=round(m["min9"], 6), IS_OOS_gap_pp=round(m["IS_OOS_gap_pp"], 4),
            Sharpe_FULL=round(m["Sharpe_FULL"], 4), MaxDD=round(m["MaxDD"], 6),
            Worst10Y=round(m["Worst10Y"], 6), Worst5Y=round(m["Worst5Y"], 6),
            P10_5Y=round(m["P10_5Y"], 6), Regime_min=round(m.get("Regime_min", np.nan), 6),
            WFE=round(m.get("WFE", np.nan), 4), CI95_lo=round(m.get("CI95_lo", np.nan), 6),
            cpcv_p10=round(m.get("cpcv_p10", np.nan), 6), Trades_yr=round(m["Trades_yr"], 1),
            twin_scale=round(twin_scale, 4), twin_CAGR_OOS=round(twin_cagr, 6),
            frontier_gap_pp=round(fgap*100, 3),
            PASS=bool(H.passes(m)),
        )
        rows.append(row)
        print("%-24s CAGR_OOS=%+6.2f%% MaxDD=%+7.2f%% Sh=%.3f W10Y=%+5.2f%% Reg=%+6.2f%% "
              "WFE=%.3f gap=%+5.2f Tr=%.0f | twin@s%.2f CAGR=%+6.2f%% FGAP=%+5.2fpp %s"
              % (name, m["CAGR_OOS"]*100, m["MaxDD"]*100, m["Sharpe_FULL"],
                 m["Worst10Y"]*100, m.get("Regime_min", np.nan)*100, m.get("WFE", np.nan),
                 m["IS_OOS_gap_pp"], m["Trades_yr"], twin_scale, twin_cagr*100, fgap*100,
                 "[PASS]" if row["PASS"] else ""))

    df = pd.DataFrame(rows)
    out = os.path.join(_REPO_DIR, "audit_results", "dd_reduction_verify_20260626.csv")
    df.to_csv(out, index=False, float_format="%.6f", encoding="utf-8-sig")
    print("\nSaved: %s" % out)

    # ---- summary: passers with positive frontier gap, ranked ----
    pas = df[(df["PASS"]) & (df["name"] != "sc2.0_anchor")].copy()
    pas_real = pas[pas["frontier_gap_pp"] > 0].sort_values("MaxDD", ascending=False)
    print("\n[PASSERS with positive frontier_gap, ranked by MaxDD (shallowest first)]")
    for _, r in pas_real.iterrows():
        print("  %-24s MaxDD=%+6.2f%% CAGR_OOS=%+6.2f%% FGAP=%+5.2fpp WFE=%.3f Reg=%+6.2f%%"
              % (r["name"], r["MaxDD"]*100, r["CAGR_OOS"]*100, r["frontier_gap_pp"],
                 r["WFE"], r["Regime_min"]*100))
    n_pass = int(df["PASS"].sum()) - 1
    n_real = len(pas_real)
    print("\n[SUMMARY] %d/%d candidates PASS the gate; %d have positive frontier_gap (real shape-change)."
          % (n_pass, len(CANDIDATES) - 1, n_real))
    print("Done.")


if __name__ == "__main__":
    main()
